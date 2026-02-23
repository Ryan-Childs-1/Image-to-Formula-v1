import os, glob, io, json, base64, zlib, math, time, hashlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# Optional local-AI dependencies (standalone, no API)
# ============================================================
# We integrate the Neural LUT editor *without* breaking compression if torch isn't installed.
# If torch is missing, the AI tab will explain how to install it.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================================================
# Why your computer started crashing (and what we fix here)
# ============================================================
# In Streamlit, *every* widget change triggers a full rerun.
# Your previous version was (re)building compressed download bytes on each rerun.
# If the image is large, repeatedly encoding WEBP/JPEG/PNG can spike CPU + RAM and crash.
#
# Fixes implemented:
# 1) "Generate formula" is now explicit (button/form submit). No heavy work on every rerun.
# 2) Download bytes are built ONLY when user clicks "Prepare download".
# 3) Results are stored in st.session_state, so changing UI doesn't recompute everything.
# 4) Hard caps + pixel caps everywhere, plus byte-size guardrails.
# 5) For large images, we downscale for download and show the resulting file size.
# ============================================================


# ============================================================
# Stability defaults / guardrails
# ============================================================

MAX_SIDE_HARD = 2048
MAX_PIXELS_HARD = 6_000_000          # hard cap for working image (quantization/reconstruction)
MAX_DOWNLOAD_PIXELS_HARD = 4_000_000 # hard cap for downloadable encode (save bytes)
MAX_PALETTE_K = 256
NEAREST_CHUNK = 25_000

# Avoid decompression bombs (still downscale ourselves)
Image.MAX_IMAGE_PIXELS = 20_000_000

# Guardrail: refuse to hold absurdly large download blobs in memory
MAX_DOWNLOAD_BYTES_IN_MEMORY = 60 * 1024 * 1024  # 60 MB


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "PALIMG_v3:"
META_BYTES_SEP = b"\n\n--META/COEFF--\n\n"


# ============================================================
# Utilities
# ============================================================

def list_local_images(folder: str = ".") -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(set(files), key=lambda p: os.path.basename(p).lower())
    return files


def to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0 + 0.5).astype(np.uint8)


def human_size(num_bytes: int) -> str:
    num_bytes = int(num_bytes)
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def safe_file_size_bytes_from_uploaded(upload_file) -> Optional[int]:
    if upload_file is None:
        return None
    sz = getattr(upload_file, "size", None)
    if isinstance(sz, int):
        return sz
    try:
        return len(upload_file.getvalue())
    except Exception:
        return None


def safe_file_size_bytes_from_path(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def load_image_from_choice(source_mode: str, chosen_local_path: Optional[str], upload_file) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"source_mode": source_mode}

    if source_mode == "Local file":
        if not chosen_local_path:
            raise ValueError("No local image selected.")
        info["path"] = chosen_local_path
        info["orig_file_bytes"] = safe_file_size_bytes_from_path(chosen_local_path)
        with Image.open(chosen_local_path) as im:
            im = im.convert("RGB")
            info["orig_w"], info["orig_h"] = im.size
            arr = np.array(im)
        return to_float01(arr), info

    if upload_file is None:
        raise ValueError("No image uploaded.")
    info["upload_name"] = getattr(upload_file, "name", "upload")
    info["orig_file_bytes"] = safe_file_size_bytes_from_uploaded(upload_file)
    with Image.open(upload_file) as im:
        im = im.convert("RGB")
        info["orig_w"], info["orig_h"] = im.size
        arr = np.array(im)
    return to_float01(arr), info


def _scale_for_caps(H: int, W: int, max_side: int, max_pixels: int) -> float:
    max_side = int(min(max_side, MAX_SIDE_HARD))
    s1 = 1.0
    if max(H, W) > max_side:
        s1 = max_side / float(max(H, W))
    s2 = 1.0
    if H * W > max_pixels:
        s2 = math.sqrt(max_pixels / float(H * W))
    return min(s1, s2, 1.0)


def cap_resize_keep_aspect(img01_rgb: np.ndarray, max_side: int, max_pixels: int) -> np.ndarray:
    H, W, _ = img01_rgb.shape
    scale = _scale_for_caps(H, W, max_side=max_side, max_pixels=max_pixels)
    if scale >= 0.999:
        return img01_rgb
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    pil = Image.fromarray(to_uint8(img01_rgb), mode="RGB").resize((newW, newH), Image.LANCZOS)
    return to_float01(np.array(pil))


def maybe_downscale_u8(img_u8: np.ndarray, max_side: int, max_pixels: int) -> np.ndarray:
    H, W, _ = img_u8.shape
    scale = _scale_for_caps(H, W, max_side=max_side, max_pixels=max_pixels)
    if scale >= 0.999:
        return img_u8
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    pil = Image.fromarray(img_u8, mode="RGB").resize((newW, newH), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


def pil_save_bytes(img_u8: np.ndarray, fmt: str, **save_kwargs) -> bytes:
    pil = Image.fromarray(img_u8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def stable_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# ============================================================
# Binary formula encoding/decoding
# ============================================================

def pack_formula(meta: Dict[str, Any], coeff_bytes: bytes) -> str:
    meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    blob = meta_json + META_BYTES_SEP + coeff_bytes
    comp = zlib.compress(blob, level=9)
    b64 = base64.urlsafe_b64encode(comp).decode("ascii")
    return APP_FORMULA_PREFIX + b64


def unpack_formula(formula: str) -> Tuple[Dict[str, Any], bytes]:
    s = (formula or "").strip()
    if not s.startswith(APP_FORMULA_PREFIX):
        raise ValueError(f"Formula must start with '{APP_FORMULA_PREFIX}'")
    b64 = s[len(APP_FORMULA_PREFIX):].strip()

    try:
        comp = base64.urlsafe_b64decode(b64.encode("ascii"))
        blob = zlib.decompress(comp)
    except Exception as e:
        raise ValueError(f"Could not decode formula: {e}")

    sep_idx = blob.find(META_BYTES_SEP)
    if sep_idx < 0:
        raise ValueError("Invalid formula payload (missing separator).")

    meta_json = blob[:sep_idx]
    coeff_bytes = blob[sep_idx + len(META_BYTES_SEP):]
    meta = json.loads(meta_json.decode("utf-8"))
    return meta, coeff_bytes


# ============================================================
# Palettes
# ============================================================

def fixed_palette_rgb_u8(name: str) -> np.ndarray:
    if name == "GameBoy (4-color)":
        return np.array([[15, 56, 15],[48, 98, 48],[139, 172, 15],[155, 188, 15]], dtype=np.uint8)
    if name == "CGA 16":
        return np.array([
            [0,0,0],[0,0,170],[0,170,0],[0,170,170],
            [170,0,0],[170,0,170],[170,85,0],[170,170,170],
            [85,85,85],[85,85,255],[85,255,85],[85,255,255],
            [255,85,85],[255,85,255],[255,255,85],[255,255,255],
        ], dtype=np.uint8)
    if name == "Grayscale 16":
        levels = np.linspace(0, 255, 16).round().astype(np.uint8)
        return np.stack([levels, levels, levels], axis=1)
    if name == "Grayscale 32":
        levels = np.linspace(0, 255, 32).round().astype(np.uint8)
        return np.stack([levels, levels, levels], axis=1)
    if name == "Web-safe 216":
        levels = np.array([0, 51, 102, 153, 204, 255], dtype=np.uint8)
        return np.array([[r,g,b] for r in levels for g in levels for b in levels], dtype=np.uint8)
    return fixed_palette_rgb_u8("CGA 16")


# ============================================================
# Bitpacking indices
# ============================================================

def bits_needed(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return int(math.ceil(math.log2(k)))


def pack_indices(indices: np.ndarray, bpp: int) -> bytes:
    bpp = int(bpp)
    idx = indices.astype(np.uint32, copy=False).ravel()
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bpp) - 1

    for v in idx:
        acc |= (int(v) & mask) << acc_bits
        acc_bits += bpp
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

    if acc_bits > 0:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_indices(packed: bytes, n: int, bpp: int) -> np.ndarray:
    bpp = int(bpp)
    data = np.frombuffer(packed, dtype=np.uint8)
    out = np.zeros((n,), dtype=np.uint32)

    acc = 0
    acc_bits = 0
    i = 0
    for byte in data:
        acc |= int(byte) << acc_bits
        acc_bits += 8
        while acc_bits >= bpp and i < n:
            out[i] = acc & ((1 << bpp) - 1)
            acc >>= bpp
            acc_bits -= bpp
            i += 1
        if i >= n:
            break
    if i < n:
        raise ValueError("Packed stream ended early (corrupt formula).")
    return out.astype(np.int32)


# ============================================================
# Quantization (safer)
# ============================================================

def pil_adaptive_quantize(rgb_u8: np.ndarray, colors: int, dither: bool, method: str) -> Tuple[np.ndarray, np.ndarray]:
    pil = Image.fromarray(rgb_u8, mode="RGB")
    method_map = {"Median-cut": 0, "Fast Octree": 2, "Max Coverage": 1}
    m = method_map.get(method, 2)
    dith = Image.FLOYDSTEINBERG if dither else Image.NONE
    q = pil.quantize(colors=int(colors), method=m, dither=dith)
    pal = np.array(q.getpalette(), dtype=np.uint8).reshape(-1, 3)  # 256x3
    idx = np.array(q, dtype=np.uint8)
    pal_eff = pal[:int(colors)].copy()
    return pal_eff, idx


def nearest_palette_indices(rgb_u8: np.ndarray, palette_u8: np.ndarray) -> np.ndarray:
    H, W, _ = rgb_u8.shape
    X = rgb_u8.reshape(-1, 3).astype(np.int16, copy=False)
    P = palette_u8.astype(np.int16, copy=False)
    n = X.shape[0]
    out = np.zeros((n,), dtype=np.int32)
    chunk = int(NEAREST_CHUNK)

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        Xi = X[i0:i1]
        d2 = ((Xi[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        out[i0:i1] = d2.argmin(axis=1)
    return out.reshape(H, W).astype(np.int32)


# ============================================================
# Codec: palette-indexed image
# ============================================================

def encode_palette_image(
    img01_rgb: np.ndarray,
    max_side: int,
    max_pixels: int,
    palette_mode: str,
    palette_param: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], np.ndarray]:
    """
    Returns formula, meta, recon_u8 (reconstructed from encoded data).
    """
    t0 = time.time()

    # Critical stability step: downscale BEFORE doing anything heavy
    img = cap_resize_keep_aspect(img01_rgb, max_side=int(max_side), max_pixels=int(max_pixels))
    rgb_u8 = to_uint8(img)
    H, W, _ = rgb_u8.shape

    if palette_mode == "Adaptive (Pillow)":
        k = int(palette_param.get("k", 32))
        k = max(2, min(k, MAX_PALETTE_K))
        method = str(palette_param.get("method", "Fast Octree"))
        dither = bool(palette_param.get("dither", True))

        palette_u8, idx_u8 = pil_adaptive_quantize(rgb_u8, colors=k, dither=dither, method=method)
        indices = idx_u8.astype(np.int32, copy=False)

        palette_name = None
        palette_len = int(palette_u8.shape[0])
        indices = np.clip(indices, 0, palette_len - 1)

        palette_bytes = palette_u8.tobytes(order="C")
        k_eff = palette_len

    elif palette_mode == "Fixed Palette":
        palette_name = str(palette_param.get("name", "CGA 16"))
        palette_u8 = fixed_palette_rgb_u8(palette_name)
        k_eff = int(palette_u8.shape[0])
        palette_len = k_eff
        palette_bytes = b""
        indices = nearest_palette_indices(rgb_u8, palette_u8)

    else:
        raise ValueError("Unknown palette_mode")

    bpp = bits_needed(k_eff)
    packed_idx = pack_indices(indices.reshape(-1), bpp=bpp)

    meta = {
        "type": "palette_indexed_v1",
        "version": 3,
        "h": int(H),
        "w": int(W),
        "k": int(k_eff),
        "bpp": int(bpp),
        "palette_mode": palette_mode,
        "palette_name": palette_name,
        "palette_len": int(palette_len),
        "palette_bytes_len": int(len(palette_bytes)),
        "params": palette_param,
        "resized_pixels": int(H * W),
        "encode_seconds": float(time.time() - t0),
        "notes": "Stable palette-indexed image + bit-packed indices + zlib. Heavy work only on button press.",
    }

    coeff_bytes = palette_bytes + packed_idx
    formula = pack_formula(meta, coeff_bytes)

    # recon from encoded data
    recon_u8 = palette_u8[np.clip(indices, 0, k_eff - 1)].reshape(H, W, 3).astype(np.uint8)
    return formula, meta, recon_u8


def decode_palette_image(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    if meta.get("type") != "palette_indexed_v1":
        raise ValueError(f"Unsupported type: {meta.get('type')}")
    H = int(meta["h"]); W = int(meta["w"])
    k = int(meta["k"]); bpp = int(meta["bpp"])
    palette_mode = meta.get("palette_mode", "Adaptive (Pillow)")
    palette_name = meta.get("palette_name", None)
    pal_bytes_len = int(meta.get("palette_bytes_len", 0))

    if palette_mode == "Adaptive (Pillow)":
        palette_bytes = coeff_bytes[:pal_bytes_len]
        packed_idx = coeff_bytes[pal_bytes_len:]
        pal_len = int(meta.get("palette_len", k))
        palette_u8 = np.frombuffer(palette_bytes, dtype=np.uint8)
        if palette_u8.size != pal_len * 3:
            raise ValueError("Palette bytes mismatch (corrupt formula).")
        palette_u8 = palette_u8.reshape(pal_len, 3)
        k = pal_len
    elif palette_mode == "Fixed Palette":
        palette_u8 = fixed_palette_rgb_u8(palette_name or "CGA 16")
        packed_idx = coeff_bytes
        k = int(palette_u8.shape[0])
        bpp = bits_needed(k)
    else:
        raise ValueError("Unknown palette_mode in formula.")

    n = H * W
    idx = unpack_indices(packed_idx, n=n, bpp=bpp)
    idx = np.clip(idx, 0, k - 1)
    return palette_u8[idx].reshape(H, W, 3).astype(np.uint8)


def latex_palette_model() -> str:
    return (
        r"We store the image using a simplified color spectrum (a palette) and a per-pixel index map."
        "\n"
        r"\[ I(x,y)\approx \mathcal{P}\big[\,M(x,y)\,\big] \]"
        "\n"
        r"where $\mathcal{P}\in\mathbb{R}^{K\times 3}$ is a palette of $K$ RGB colors and "
        r"$M(x,y)\in\{0,\dots,K-1\}$ is an index map. "
        r"The index map is bit-packed (using $\lceil\log_2 K\rceil$ bits per pixel) and then compressed."
    )


# ============================================================
# Session-state helpers (prevents recompute + prevents crashes)
# ============================================================

def ss_init():
    st.session_state.setdefault("last_formula", None)
    st.session_state.setdefault("last_meta", None)
    st.session_state.setdefault("last_recon_u8", None)
    st.session_state.setdefault("last_input_info", None)
    st.session_state.setdefault("last_formula_bytes", None)
    st.session_state.setdefault("last_download_blob", None)
    st.session_state.setdefault("last_download_name", None)
    st.session_state.setdefault("last_download_mime", None)
    st.session_state.setdefault("last_download_sig", None)

    # Local AI Editor state
    st.session_state.setdefault("ai_base_u8", None)          # selected base image for AI editor (uint8)
    st.session_state.setdefault("ai_ref_u8", None)           # reference look image (uint8)
    st.session_state.setdefault("ai_model", None)            # trained NeuralLUT (torch)
    st.session_state.setdefault("ai_loss_trace", None)       # list of floats
    st.session_state.setdefault("ai_edited_u8", None)        # edited output (uint8)


def ss_clear_download():
    st.session_state["last_download_blob"] = None
    st.session_state["last_download_name"] = None
    st.session_state["last_download_mime"] = None
    st.session_state["last_download_sig"] = None


ss_init()


# ============================================================
# Local AI Editor: Neural LUT (standalone, train-on-the-fly)
# ============================================================

if TORCH_AVAILABLE:
    class NeuralLUT(nn.Module):
        """
        Tiny MLP mapping RGB -> RGB (acts like a learned smooth 3D LUT).
        Great for "style/grade transfer" (mood, palette shifts), not object edits.
        """
        def __init__(self, width: int = 64, depth: int = 4):
            super().__init__()
            layers = []
            in_dim = 3
            for _ in range(int(depth)):
                layers.append(nn.Linear(in_dim, int(width)))
                layers.append(nn.SiLU())
                in_dim = int(width)
            layers.append(nn.Linear(in_dim, 3))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            y = self.net(x)
            # residual + tanh stabilizes and preserves structure
            y = x + 0.5 * torch.tanh(y)
            return torch.clamp(y, 0.0, 1.0)


    def _resize_keep_aspect_u8(img_u8: np.ndarray, max_side: int) -> np.ndarray:
        H, W, _ = img_u8.shape
        if max(H, W) <= int(max_side):
            return img_u8
        scale = int(max_side) / float(max(H, W))
        newW = max(1, int(round(W * scale)))
        newH = max(1, int(round(H * scale)))
        im = Image.fromarray(img_u8, mode="RGB").resize((newW, newH), Image.LANCZOS)
        return np.array(im, dtype=np.uint8)


    def _resize_to_match_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ha, Wa, _ = a_u8.shape
        imB = Image.fromarray(b_u8, mode="RGB").resize((Wa, Ha), Image.LANCZOS)
        return a_u8, np.array(imB, dtype=np.uint8)


    def _sample_pixels(rgb01_a: np.ndarray, rgb01_b: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        H, W, _ = rgb01_a.shape
        total = H * W
        n = min(int(n), total)
        idx = rng.choice(total, size=n, replace=False)
        xa = rgb01_a.reshape(-1, 3)[idx]
        xb = rgb01_b.reshape(-1, 3)[idx]
        return xa, xb


    def train_neural_lut(
        x_in: np.ndarray,
        y_tgt: np.ndarray,
        width: int,
        depth: int,
        steps: int,
        batch: int,
        lr: float,
        weight_decay: float,
        device: str,
    ) -> Tuple[NeuralLUT, List[float]]:
        torch.manual_seed(0)
        model = NeuralLUT(width=int(width), depth=int(depth)).to(device)

        X = torch.from_numpy(x_in.astype(np.float32, copy=False)).to(device)
        Y = torch.from_numpy(y_tgt.astype(np.float32, copy=False)).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        n = X.shape[0]
        batch = max(64, int(batch))
        loss_trace: List[float] = []

        model.train()
        for t in range(int(steps)):
            idx = torch.randint(0, n, (batch,), device=device)
            xb = X[idx]
            yb = Y[idx]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (t % 25) == 0 or (t == int(steps) - 1):
                loss_trace.append(float(loss.detach().cpu().item()))

        model.eval()
        return model, loss_trace


    @torch.no_grad()
    def apply_lut_chunked(model: NeuralLUT, img01: np.ndarray, device: str, chunk: int) -> np.ndarray:
        H, W, _ = img01.shape
        flat = img01.reshape(-1, 3).astype(np.float32, copy=False)
        out = np.empty_like(flat)

        chunk = int(chunk)
        chunk = max(10_000, min(chunk, 1_000_000))  # safety
        for i0 in range(0, flat.shape[0], chunk):
            i1 = min(flat.shape[0], i0 + chunk)
            xb = torch.from_numpy(flat[i0:i1]).to(device)
            yb = model(xb).detach().cpu().numpy()
            out[i0:i1] = yb

        return out.reshape(H, W, 3)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula + Local AI Editor", layout="wide")
st.title("Image ⇄ Formula — Stable Downloads + Local AI Editor (No API)")

st.markdown(
    "Key change: heavy work only happens when you click buttons.\n\n"
    "- **Generate formula** (quantization + formula creation)\n"
    "- **Prepare download** (builds WEBP/JPEG/PNG bytes once)\n"
    "- **Local AI Editor** (optional): trains a tiny model to learn a color/lighting transform\n"
)

# Use tabs so we don't lose existing functionality
tab1, tab2, tab3 = st.tabs(["Image → Formula", "Formula → Image", "AI Editor (Local, no API)"])


# ============================================================
# TAB 1: IMAGE → FORMULA
# ============================================================
with tab1:
    with st.sidebar:
        st.header("Image input")
        local_images = list_local_images(".")
        source_mode = st.radio("Source", ["Local file", "Upload"], index=0, key="t1_source")

        chosen_local = None
        upload = None
        if source_mode == "Local file":
            if not local_images:
                st.warning("No images found next to app.py. Add .png/.jpg files or switch to Upload.")
            else:
                name_to_path = {os.path.basename(p): p for p in local_images}
                chosen_name = st.selectbox("Choose local image", list(name_to_path.keys()), key="t1_local_name")
                chosen_local = name_to_path[chosen_name]
        else:
            upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="t1_upload")

        st.divider()
        st.header("Safety caps (lower if your machine is struggling)")
        max_pixels = st.select_slider(
            "Max working pixels (H×W cap)",
            options=[750_000, 1_500_000, 3_000_000, 6_000_000],
            value=3_000_000,
            key="t1_maxpix"
        )
        max_pixels = int(min(max_pixels, MAX_PIXELS_HARD))

        max_side = st.select_slider(
            "Max side (preserve aspect)",
            options=[256, 384, 512, 768, 1024, 1280, 1536, 2048],
            value=1024,
            key="t1_maxside"
        )
        max_side = int(min(max_side, MAX_SIDE_HARD))

        st.divider()
        st.header("Simplified color spectrum")
        palette_mode = st.radio("Spectrum type", ["Adaptive (Pillow)", "Fixed Palette"], index=0, key="t1_pal_mode")

        palette_param: Dict[str, Any] = {}
        if palette_mode == "Adaptive (Pillow)":
            k = st.select_slider("Number of colors (K)", options=[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256], value=32, key="t1_k")
            method = st.selectbox("Palette method", ["Fast Octree", "Median-cut", "Max Coverage"], index=0, key="t1_method")
            dither = st.checkbox("Dither (Floyd–Steinberg)", value=True, key="t1_dither")
            palette_param = {"k": int(k), "method": method, "dither": bool(dither)}
        else:
            name = st.selectbox(
                "Fixed palette",
                ["CGA 16", "GameBoy (4-color)", "Grayscale 16", "Grayscale 32", "Web-safe 216"],
                index=0,
                key="t1_fixed_name"
            )
            palette_param = {"name": name}

        st.divider()
        st.header("Download settings (built only on button click)")
        dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0, key="t1_dl_fmt")
        dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768, key="t1_dl_side")
        dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1, key="t1_dl_q")
        dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True, key="t1_dl_png")
        dl_lossless_webp = st.checkbox("Lossless WEBP", value=False, key="t1_dl_webp_lossless")

        dl_max_side = int(min(dl_max_side, MAX_SIDE_HARD))

        st.caption(
            "Tip: WEBP at 70–85 quality is usually the best size/quality. "
            "PNG can be huge; optimize helps but may be slow."
        )

    # Load input (cheap)
    try:
        img01, info = load_image_from_choice(source_mode, chosen_local, upload)

        st.subheader("Input overview")
        c1, c2, c3 = st.columns(3)
        orig_file_bytes = info.get("orig_file_bytes", None)
        if isinstance(orig_file_bytes, int):
            c1.metric("Original file size", human_size(orig_file_bytes))
        else:
            c1.metric("Original file size", "Unknown")
        c2.metric("Original dimensions", f"{info.get('orig_w','?')}×{info.get('orig_h','?')}")
        c3.metric("Source", info.get("upload_name") or os.path.basename(info.get("path", "")) or info.get("source_mode"))

        st.image(to_uint8(cap_resize_keep_aspect(img01, max_side=1024, max_pixels=1_500_000)), use_container_width=True)

        # Heavy step only on submit:
        with st.form("encode_form", clear_on_submit=False):
            st.write("### Generate formula (heavy step)")
            submitted = st.form_submit_button("Generate formula", type="primary")

        if submitted:
            ss_clear_download()
            with st.spinner("Generating formula (bounded + safe)…"):
                formula, meta, recon_u8 = encode_palette_image(
                    img01_rgb=img01,
                    max_side=max_side,
                    max_pixels=max_pixels,
                    palette_mode=palette_mode,
                    palette_param=palette_param,
                )

            st.session_state["last_formula"] = formula
            st.session_state["last_meta"] = meta
            st.session_state["last_recon_u8"] = recon_u8
            st.session_state["last_input_info"] = info
            st.session_state["last_formula_bytes"] = len(formula.encode("utf-8"))

            # Also make it immediately available as AI base
            st.session_state["ai_base_u8"] = recon_u8
            st.session_state["ai_edited_u8"] = None
            st.session_state["ai_model"] = None
            st.session_state["ai_loss_trace"] = None

        # If we have a last result, show it (no recompute)
        if st.session_state["last_formula"] is not None:
            formula = st.session_state["last_formula"]
            meta = st.session_state["last_meta"]
            recon_u8 = st.session_state["last_recon_u8"]
            info0 = st.session_state["last_input_info"]
            formula_bytes = st.session_state["last_formula_bytes"]
            orig_file_bytes0 = info0.get("orig_file_bytes", None)

            st.divider()
            st.subheader("Compression summary (last generated)")
            a, b, c, d = st.columns(4)
            if isinstance(orig_file_bytes0, int):
                a.metric("Original file size", human_size(orig_file_bytes0))
            else:
                a.metric("Original file size", "Unknown")
            b.metric("Working resolution", f"{meta['w']}×{meta['h']}")
            c.metric("Formula size", human_size(formula_bytes))
            d.metric("Encode time", f"{meta.get('encode_seconds', 0.0):.2f}s")

            r1, r2 = st.columns(2)
            with r1:
                if isinstance(orig_file_bytes0, int) and orig_file_bytes0 > 0:
                    r1.metric("Orig / Formula ratio", f"{orig_file_bytes0/float(formula_bytes):.2f}×")
                else:
                    r1.metric("Orig / Formula ratio", "—")
            with r2:
                base_webp = pil_save_bytes(recon_u8, "WEBP", quality=80, method=6)
                if isinstance(orig_file_bytes0, int) and orig_file_bytes0 > 0:
                    r2.metric("Orig / WEBP(80) ratio", f"{orig_file_bytes0/float(len(base_webp)):.2f}×")
                else:
                    r2.metric("WEBP(80) size", human_size(len(base_webp)))

            st.subheader("Single formula string")
            st.text_area("Copy/paste this formula:", value=formula, height=220)

            st.download_button(
                "Download formula as .txt",
                data=formula.encode("utf-8"),
                file_name="image_formula.txt",
                mime="text/plain",
            )

            st.subheader("Reconstructed preview (palette result)")
            st.image(maybe_downscale_u8(recon_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)

            st.subheader("Prepare downloadable compressed image (built only when you click)")
            sig_src = f"{meta['w']}x{meta['h']}|{dl_format}|{dl_max_side}|{dl_quality}|{dl_opt_png}|{dl_lossless_webp}"
            sig = stable_hash_bytes(sig_src.encode("utf-8"))

            colp1, colp2 = st.columns([1, 1])
            with colp1:
                prep = st.button("Prepare download", type="secondary", key="t1_prep_dl")
            with colp2:
                st.caption("This step compresses the image file (WEBP/JPEG/PNG). It can take a moment for large images.")

            if prep:
                ss_clear_download()
                with st.spinner("Building compressed download…"):
                    dl_u8 = maybe_downscale_u8(
                        recon_u8,
                        max_side=int(dl_max_side),
                        max_pixels=min(int(max_pixels), MAX_DOWNLOAD_PIXELS_HARD)
                    )

                    fmt = dl_format.split()[0]
                    if fmt == "WEBP":
                        blob = pil_save_bytes(
                            dl_u8, "WEBP",
                            quality=int(dl_quality),
                            lossless=bool(dl_lossless_webp),
                            method=6
                        )
                        name = "modified_compressed.webp"
                        mime = "image/webp"
                    elif fmt == "JPEG":
                        blob = pil_save_bytes(
                            dl_u8, "JPEG",
                            quality=int(dl_quality),
                            optimize=True,
                            progressive=True
                        )
                        name = "modified_compressed.jpg"
                        mime = "image/jpeg"
                    else:
                        blob = pil_save_bytes(
                            dl_u8, "PNG",
                            optimize=bool(dl_opt_png),
                            compress_level=9
                        )
                        name = "modified_compressed.png"
                        mime = "image/png"

                    if len(blob) > MAX_DOWNLOAD_BYTES_IN_MEMORY:
                        raise RuntimeError(
                            f"Download file is too large to hold in memory ({human_size(len(blob))}). "
                            f"Lower download max side or choose WEBP/JPEG."
                        )

                    st.session_state["last_download_blob"] = blob
                    st.session_state["last_download_name"] = name
                    st.session_state["last_download_mime"] = mime
                    st.session_state["last_download_sig"] = sig

            if st.session_state["last_download_blob"] is not None and st.session_state["last_download_sig"] == sig:
                blob = st.session_state["last_download_blob"]
                name = st.session_state["last_download_name"]
                mime = st.session_state["last_download_mime"]

                d1, d2, d3 = st.columns(3)
                d1.metric("Prepared download size", human_size(len(blob)))
                d2.metric("Download format", dl_format.split()[0])
                if isinstance(orig_file_bytes0, int) and orig_file_bytes0 > 0:
                    d3.metric("Orig / Download ratio", f"{orig_file_bytes0/float(len(blob)):.2f}×")
                else:
                    d3.metric("Orig / Download ratio", "—")

                st.download_button(
                    "Download compressed modified image",
                    data=blob,
                    file_name=name,
                    mime=mime,
                )
            else:
                st.info("Click **Prepare download** to generate the compressed file, then the download button will appear.")

            st.subheader("Mathematical depiction")
            st.latex(latex_palette_model())

            with st.expander("Show meta + input info"):
                st.json({"input_info": info0, "meta": meta})

    except Exception as e:
        st.error(f"Image → Formula failed: {e}")


# ============================================================
# TAB 2: FORMULA → IMAGE
# ============================================================
with tab2:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=220, placeholder=f"{APP_FORMULA_PREFIX}...", key="t2_formula")

    st.markdown("### Download settings (built only on button click)")
    dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0, key="t2_fmt")
    dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768, key="t2_side")
    dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1, key="t2_q")
    dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True, key="t2_png")
    dl_lossless_webp = st.checkbox("Lossless WEBP", value=False, key="t2_webp_lossless")

    colA, colB = st.columns([1, 2])
    with colA:
        do_recon = st.button("Reconstruct image", type="primary", key="t2_recon")
    with colB:
        st.caption("Reconstruction is usually fast. Download compression happens only when you click Prepare download.")

    if do_recon:
        try:
            ss_clear_download()
            with st.spinner("Decoding formula…"):
                meta, coeff = unpack_formula(formula_in)
                img_u8 = decode_palette_image(meta, coeff)

            st.session_state["last_recon_u8"] = img_u8
            st.session_state["last_meta"] = meta

            # Also make it available to AI editor
            st.session_state["ai_base_u8"] = img_u8
            st.session_state["ai_edited_u8"] = None
            st.session_state["ai_model"] = None
            st.session_state["ai_loss_trace"] = None

            st.success(f"Reconstructed {meta['w']}×{meta['h']} | K={meta['k']} | bpp={meta['bpp']} | {meta['palette_mode']}")
            st.image(maybe_downscale_u8(img_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)

            png_blob = pil_save_bytes(img_u8, "PNG", optimize=True, compress_level=9)
            if len(png_blob) <= MAX_DOWNLOAD_BYTES_IN_MEMORY:
                st.download_button("Download reconstructed PNG (lossless)", data=png_blob, file_name="reconstructed.png", mime="image/png")
            else:
                st.warning(f"Lossless PNG is too large to offer directly ({human_size(len(png_blob))}). Use WEBP/JPEG download below.")

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")

    if st.session_state.get("last_recon_u8") is not None and st.session_state.get("last_meta") is not None:
        img_u8 = st.session_state["last_recon_u8"]
        meta = st.session_state["last_meta"]

        st.divider()
        st.subheader("Prepare compressed download")

        sig_src = f"{meta['w']}x{meta['h']}|{dl_format}|{dl_max_side}|{dl_quality}|{dl_opt_png}|{dl_lossless_webp}"
        sig = stable_hash_bytes(sig_src.encode("utf-8"))

        prep = st.button("Prepare download", key="t2_prep")
        if prep:
            ss_clear_download()
            try:
                with st.spinner("Building compressed download…"):
                    dl_u8 = maybe_downscale_u8(img_u8, max_side=int(dl_max_side), max_pixels=MAX_DOWNLOAD_PIXELS_HARD)
                    fmt = dl_format.split()[0]
                    if fmt == "WEBP":
                        blob = pil_save_bytes(dl_u8, "WEBP", quality=int(dl_quality), lossless=bool(dl_lossless_webp), method=6)
                        name = "compressed.webp"
                        mime = "image/webp"
                    elif fmt == "JPEG":
                        blob = pil_save_bytes(dl_u8, "JPEG", quality=int(dl_quality), optimize=True, progressive=True)
                        name = "compressed.jpg"
                        mime = "image/jpeg"
                    else:
                        blob = pil_save_bytes(dl_u8, "PNG", optimize=bool(dl_opt_png), compress_level=9)
                        name = "compressed.png"
                        mime = "image/png"

                    if len(blob) > MAX_DOWNLOAD_BYTES_IN_MEMORY:
                        raise RuntimeError(
                            f"Download file is too large to hold in memory ({human_size(len(blob))}). "
                            f"Lower download max side or choose WEBP/JPEG."
                        )

                    st.session_state["last_download_blob"] = blob
                    st.session_state["last_download_name"] = name
                    st.session_state["last_download_mime"] = mime
                    st.session_state["last_download_sig"] = sig
            except Exception as e:
                st.error(f"Prepare download failed: {e}")

        if st.session_state.get("last_download_blob") is not None and st.session_state.get("last_download_sig") == sig:
            blob = st.session_state["last_download_blob"]
            name = st.session_state["last_download_name"]
            mime = st.session_state["last_download_mime"]

            d1, d2 = st.columns(2)
            d1.metric("Prepared download size", human_size(len(blob)))
            d2.metric("Format", dl_format.split()[0])

            st.download_button("Download compressed image", data=blob, file_name=name, mime=mime)
        else:
            st.info("Click **Prepare download** to generate the compressed file, then the download button will appear.")

        st.subheader("Mathematical depiction")
        st.latex(latex_palette_model())

        with st.expander("Show decoded meta"):
            st.json(meta)


# ============================================================
# TAB 3: AI EDITOR (LOCAL, NO API) — integrated, safe, button-driven
# ============================================================
with tab3:
    st.subheader("AI Editor (Local, no API) — Neural LUT (Color/Lighting Transform)")

    if not TORCH_AVAILABLE:
        st.error(
            "PyTorch is not installed in this environment, so the local AI editor can't run.\n\n"
            "Install it in your environment, then rerun Streamlit:\n"
            "  pip install torch\n"
            "On some systems you may want the CPU-only build from PyTorch's install selector."
        )
        st.stop()

    st.markdown(
        "This editor trains a tiny neural network that learns a smooth mapping:\n\n"
        r"\[(r,g,b)\;\mapsto\;(r',g',b')\]\n\n"
        "- Works great for **mood / palette / lighting** changes (like a learned color grade).\n"
        "- Not meant for adding/removing objects.\n\n"
        "**Integration:** the edited image can be pushed back into the compressor (session_state['last_recon_u8'])."
    )

    # Base selection
    colL, colR = st.columns([1, 1])
    with colL:
        base_source = st.radio(
            "Base image source",
            ["Use last image from compressor", "Upload base image"],
            index=0,
            key="ai_base_source"
        )
        base_u8: Optional[np.ndarray] = None

        if base_source == "Use last image from compressor":
            base_u8 = st.session_state.get("last_recon_u8", None)
            if base_u8 is None:
                st.warning("No last image found. Go to Image→Formula or Formula→Image and generate/reconstruct first.")
            else:
                base_u8 = np.array(base_u8, dtype=np.uint8)
        else:
            up_base = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="ai_up_base")
            if up_base is not None:
                with Image.open(up_base) as im:
                    im = im.convert("RGB")
                    base_u8 = np.array(im, dtype=np.uint8)

    with colR:
        ref_u8: Optional[np.ndarray] = None
        up_ref = st.file_uploader("Upload reference look image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="ai_up_ref")
        if up_ref is not None:
            with Image.open(up_ref) as im:
                im = im.convert("RGB")
                ref_u8 = np.array(im, dtype=np.uint8)

    if base_u8 is None:
        st.stop()
    st.session_state["ai_base_u8"] = base_u8

    st.divider()
    st.subheader("Preview")
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Base")
        st.image(maybe_downscale_u8(base_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)
    with p2:
        st.caption("Reference look (optional but recommended)")
        if ref_u8 is None:
            st.info("Upload a reference look image to train the AI.")
        else:
            st.image(maybe_downscale_u8(ref_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)

    if ref_u8 is None:
        st.stop()

    st.session_state["ai_ref_u8"] = ref_u8

    st.divider()
    st.subheader("Training settings (safe defaults)")

    c1, c2, c3, c4 = st.columns(4)
    train_max_side = c1.selectbox("Train max side", [256, 384, 512, 768, 1024], index=2, key="ai_train_side")
    sample_n = c2.selectbox("Pixel samples", [20_000, 50_000, 100_000, 200_000, 400_000], index=2, key="ai_samples")
    steps = c3.selectbox("Steps", [200, 400, 800, 1200, 2000], index=2, key="ai_steps")
    batch = c4.selectbox("Batch", [256, 512, 1024, 2048, 4096], index=3, key="ai_batch")

    c5, c6, c7, c8 = st.columns(4)
    width = c5.selectbox("Width", [32, 48, 64, 96, 128], index=2, key="ai_width")
    depth = c6.selectbox("Depth", [2, 3, 4, 5, 6], index=2, key="ai_depth")
    lr = c7.selectbox("LR", [1e-4, 2e-4, 5e-4, 1e-3, 2e-3], index=3, key="ai_lr")
    weight_decay = c8.selectbox("Weight decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3], index=3, key="ai_wd")

    apply_chunk = st.selectbox("Apply chunk (RAM safety)", [50_000, 100_000, 200_000, 400_000], index=2, key="ai_chunk")

    use_cuda = st.checkbox("Use GPU if available", value=True, key="ai_use_cuda")
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

    # Buttons
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    train_btn = b1.button("Train AI LUT", type="primary", key="ai_train_btn")
    apply_btn = b2.button("Apply LUT to base", key="ai_apply_btn")
    push_btn = b3.button("Send edited image to compressor", key="ai_push_btn")
    clear_btn = b4.button("Clear AI state", key="ai_clear_btn")

    if clear_btn:
        st.session_state["ai_model"] = None
        st.session_state["ai_loss_trace"] = None
        st.session_state["ai_edited_u8"] = None
        st.success("Cleared AI state.")

    if train_btn:
        try:
            with st.spinner("Preparing training data…"):
                # downscale both for speed
                base_train = _resize_keep_aspect_u8(base_u8, max_side=int(train_max_side))
                ref_train = _resize_keep_aspect_u8(ref_u8, max_side=int(train_max_side))
                base_train, ref_train = _resize_to_match_u8(base_train, ref_train)

                a01 = to_float01(base_train)
                b01 = to_float01(ref_train)
                x_in, y_tgt = _sample_pixels(a01, b01, n=int(sample_n), seed=0)

            with st.spinner("Training Neural LUT…"):
                model, loss_trace = train_neural_lut(
                    x_in=x_in,
                    y_tgt=y_tgt,
                    width=int(width),
                    depth=int(depth),
                    steps=int(steps),
                    batch=int(batch),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    device=device
                )

            st.session_state["ai_model"] = model
            st.session_state["ai_loss_trace"] = loss_trace
            st.session_state["ai_edited_u8"] = None
            st.success(f"Trained. Last loss: {loss_trace[-1]:.6f}")

        except Exception as e:
            st.error(f"Training failed: {e}")

    loss_trace = st.session_state.get("ai_loss_trace", None)
    if isinstance(loss_trace, list) and len(loss_trace) > 1:
        st.line_chart(loss_trace)

    if apply_btn:
        model = st.session_state.get("ai_model", None)
        if model is None:
            st.error("Train the LUT first.")
        else:
            try:
                with st.spinner("Applying LUT to full base image…"):
                    base01 = to_float01(base_u8)
                    out01 = apply_lut_chunked(model, base01, device=device, chunk=int(apply_chunk))
                    out_u8 = to_uint8(out01)

                st.session_state["ai_edited_u8"] = out_u8
                st.success("Applied LUT.")
            except Exception as e:
                st.error(f"Apply failed: {e}")

    edited_u8 = st.session_state.get("ai_edited_u8", None)
    if edited_u8 is not None:
        st.divider()
        st.subheader("Edited output")
        st.image(maybe_downscale_u8(edited_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)

        # Downloads (computed once here, but small + safe; you can still hit memory caps)
        png_blob = pil_save_bytes(edited_u8, "PNG", optimize=True, compress_level=9)
        webp_blob = pil_save_bytes(edited_u8, "WEBP", quality=85, method=6)

        d1, d2, d3 = st.columns(3)
        d1.metric("Edited PNG (lossless)", human_size(len(png_blob)))
        d2.metric("Edited WEBP (85)", human_size(len(webp_blob)))
        d3.metric("Resolution", f"{edited_u8.shape[1]}×{edited_u8.shape[0]}")

        x1, x2 = st.columns(2)
        x1.download_button("Download edited PNG", data=png_blob, file_name="ai_edited.png", mime="image/png")
        x2.download_button("Download edited WEBP", data=webp_blob, file_name="ai_edited.webp", mime="image/webp")

        if push_btn:
            # Push edited image back into the compressor pipeline
            st.session_state["last_recon_u8"] = edited_u8
            st.session_state["last_meta"] = {
                "type": "ai_local_neural_lut",
                "notes": "Edited in local AI tab. Use Image→Formula to re-compress.",
                "h": int(edited_u8.shape[0]),
                "w": int(edited_u8.shape[1]),
            }
            ss_clear_download()
            st.success("Sent edited image to compressor as session_state['last_recon_u8']. Now go to Image→Formula and Generate formula.")
    else:
        if push_btn:
            st.warning("No edited image yet. Train and apply first.")
