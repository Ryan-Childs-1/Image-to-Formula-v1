import os, glob, io, json, base64, zlib, math, time, hashlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


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
    st.session_state.setdefault("last_download_sig", None)  # identifies which settings produced blob


def ss_clear_download():
    st.session_state["last_download_blob"] = None
    st.session_state["last_download_name"] = None
    st.session_state["last_download_mime"] = None
    st.session_state["last_download_sig"] = None


ss_init()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula (Stable Downloads)", layout="wide")
st.title("Image ⇄ Formula — Stable + Safe Download Compression (No More Crashes)")

st.markdown(
    "Key change: heavy work only happens when you click buttons.\n\n"
    "- **Generate formula** (does the quantization + formula creation)\n"
    "- **Prepare download** (builds WEBP/JPEG/PNG bytes once, then you can download)\n"
)


mode = st.radio("Choose conversion", ["Image → Formula", "Formula → Image"], horizontal=True)
st.divider()


# ============================================================
# IMAGE → FORMULA
# ============================================================

if mode == "Image → Formula":
    with st.sidebar:
        st.header("Image input")
        local_images = list_local_images(".")
        source_mode = st.radio("Source", ["Local file", "Upload"], index=0)

        chosen_local = None
        upload = None
        if source_mode == "Local file":
            if not local_images:
                st.warning("No images found next to app.py. Add .png/.jpg files or switch to Upload.")
            else:
                name_to_path = {os.path.basename(p): p for p in local_images}
                chosen_name = st.selectbox("Choose local image", list(name_to_path.keys()))
                chosen_local = name_to_path[chosen_name]
        else:
            upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp", "bmp"])

        st.divider()
        st.header("Safety caps (lower if your machine is struggling)")
        max_pixels = st.select_slider(
            "Max working pixels (H×W cap)",
            options=[750_000, 1_500_000, 3_000_000, 6_000_000],
            value=3_000_000
        )
        max_pixels = int(min(max_pixels, MAX_PIXELS_HARD))

        max_side = st.select_slider(
            "Max side (preserve aspect)",
            options=[256, 384, 512, 768, 1024, 1280, 1536, 2048],
            value=1024
        )
        max_side = int(min(max_side, MAX_SIDE_HARD))

        st.divider()
        st.header("Simplified color spectrum")
        palette_mode = st.radio("Spectrum type", ["Adaptive (Pillow)", "Fixed Palette"], index=0)

        palette_param: Dict[str, Any] = {}
        if palette_mode == "Adaptive (Pillow)":
            k = st.select_slider("Number of colors (K)", options=[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256], value=32)
            method = st.selectbox("Palette method", ["Fast Octree", "Median-cut", "Max Coverage"], index=0)
            dither = st.checkbox("Dither (Floyd–Steinberg)", value=True)
            palette_param = {"k": int(k), "method": method, "dither": bool(dither)}
        else:
            name = st.selectbox(
                "Fixed palette",
                ["CGA 16", "GameBoy (4-color)", "Grayscale 16", "Grayscale 32", "Web-safe 216"],
                index=0
            )
            palette_param = {"name": name}

        st.divider()
        st.header("Download settings (built only on button click)")
        dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0)
        dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768)
        dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1)
        dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True)
        dl_lossless_webp = st.checkbox("Lossless WEBP", value=False)

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
                # A realistic baseline: WEBP(80) for the *same palette image* (not the original)
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
            # Signature identifies whether download settings changed since blob built
            sig_src = f"{meta['w']}x{meta['h']}|{dl_format}|{dl_max_side}|{dl_quality}|{dl_opt_png}|{dl_lossless_webp}"
            sig = stable_hash_bytes(sig_src.encode("utf-8"))

            colp1, colp2 = st.columns([1, 1])
            with colp1:
                prep = st.button("Prepare download", type="secondary")
            with colp2:
                st.caption("This step compresses the image file (WEBP/JPEG/PNG). It can take a moment for large images.")

            if prep:
                ss_clear_download()
                with st.spinner("Building compressed download…"):
                    # Downscale for download with additional hard cap
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
                        # PNG can be huge; keep, but guardrail the memory
                        blob = pil_save_bytes(
                            dl_u8, "PNG",
                            optimize=bool(dl_opt_png),
                            compress_level=9
                        )
                        name = "modified_compressed.png"
                        mime = "image/png"

                    if len(blob) > MAX_DOWNLOAD_BYTES_IN_MEMORY:
                        # Prevent memory blowups / browser issues
                        raise RuntimeError(
                            f"Download file is too large to hold in memory ({human_size(len(blob))}). "
                            f"Lower download max side or choose WEBP/JPEG."
                        )

                    st.session_state["last_download_blob"] = blob
                    st.session_state["last_download_name"] = name
                    st.session_state["last_download_mime"] = mime
                    st.session_state["last_download_sig"] = sig

            # Show download button only if prepared and settings match
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
# FORMULA → IMAGE
# ============================================================

else:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=220, placeholder=f"{APP_FORMULA_PREFIX}...")

    st.markdown("### Download settings (built only on button click)")
    dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0, key="f2_fmt")
    dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768, key="f2_side")
    dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1, key="f2_q")
    dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True, key="f2_png")
    dl_lossless_webp = st.checkbox("Lossless WEBP", value=False, key="f2_webp_lossless")

    colA, colB = st.columns([1, 2])
    with colA:
        do_recon = st.button("Reconstruct image", type="primary")
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

            st.success(f"Reconstructed {meta['w']}×{meta['h']} | K={meta['k']} | bpp={meta['bpp']} | {meta['palette_mode']}")
            st.image(maybe_downscale_u8(img_u8, max_side=1024, max_pixels=1_500_000), use_container_width=True)

            # Always provide lossless PNG download (small enough usually; still can be big)
            png_blob = pil_save_bytes(img_u8, "PNG", optimize=True, compress_level=9)
            if len(png_blob) <= MAX_DOWNLOAD_BYTES_IN_MEMORY:
                st.download_button("Download reconstructed PNG (lossless)", data=png_blob, file_name="reconstructed.png", mime="image/png")
            else:
                st.warning(f"Lossless PNG is too large to offer directly ({human_size(len(png_blob))}). Use WEBP/JPEG download below.")

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")

    # If we have a reconstructed image in session, allow download prep
    if st.session_state.get("last_recon_u8") is not None and st.session_state.get("last_meta") is not None:
        img_u8 = st.session_state["last_recon_u8"]
        meta = st.session_state["last_meta"]

        st.divider()
        st.subheader("Prepare compressed download")

        sig_src = f"{meta['w']}x{meta['h']}|{dl_format}|{dl_max_side}|{dl_quality}|{dl_opt_png}|{dl_lossless_webp}"
        sig = stable_hash_bytes(sig_src.encode("utf-8"))

        prep = st.button("Prepare download", key="f2_prep")
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
