import os, glob, io, json, base64, zlib, math
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "PALIMG_v1:"
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
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0 + 0.5).astype(np.uint8)


def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def load_image_from_choice(source_mode: str, chosen_local_path: Optional[str], upload_file) -> np.ndarray:
    if source_mode == "Local file":
        if not chosen_local_path:
            raise ValueError("No local image selected.")
        img = Image.open(chosen_local_path).convert("RGB")
        return to_float01(np.array(img))
    else:
        if upload_file is None:
            raise ValueError("No image uploaded.")
        img = Image.open(upload_file).convert("RGB")
        return to_float01(np.array(img))


def resize_keep_aspect(img01_rgb: np.ndarray, max_side: int) -> np.ndarray:
    H, W, _ = img01_rgb.shape
    max_side = int(max_side)
    if max(H, W) > max_side:
        scale = max_side / float(max(H, W))
        newW = max(1, int(round(W * scale)))
        newH = max(1, int(round(H * scale)))
    else:
        newH, newW = H, W

    pil = Image.fromarray(to_uint8(img01_rgb), mode="RGB")
    pil = pil.resize((newW, newH), Image.LANCZOS)
    return to_float01(np.array(pil))


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
# Palette definitions (fixed)
# ============================================================

def fixed_palette_rgb_u8(name: str) -> np.ndarray:
    """
    Returns palette as uint8 array shape (K,3)
    """
    if name == "GameBoy (4-color)":
        return np.array([
            [15, 56, 15],
            [48, 98, 48],
            [139, 172, 15],
            [155, 188, 15],
        ], dtype=np.uint8)

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
        pal = np.array([[r, g, b] for r in levels for g in levels for b in levels], dtype=np.uint8)
        return pal

    # fallback
    return fixed_palette_rgb_u8("CGA 16")


# ============================================================
# Bit-packing indices (major size win for small K)
# ============================================================

def bits_needed(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return int(math.ceil(math.log2(k)))


def pack_indices(indices: np.ndarray, bpp: int) -> bytes:
    """
    Pack uint indices into a bitstream with bpp bits per pixel.
    indices: flat array of ints
    """
    bpp = int(bpp)
    idx = indices.astype(np.uint32).ravel()
    n = idx.size

    out = bytearray()
    acc = 0
    acc_bits = 0

    mask = (1 << bpp) - 1
    for v in idx:
        v = int(v) & mask
        acc |= (v << acc_bits)
        acc_bits += bpp
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

    if acc_bits > 0:
        out.append(acc & 0xFF)

    return bytes(out)


def unpack_indices(packed: bytes, n: int, bpp: int) -> np.ndarray:
    """
    Unpack n indices of bpp bits from packed bytes.
    """
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
        raise ValueError("Packed index stream ended early (corrupt formula).")

    return out.astype(np.int32)


# ============================================================
# Quantization engines
#   - Best practical compression here = palette-indexed + zlib
#   - Use Pillow's quantize (median-cut / octree) + optional dithering
# ============================================================

def pil_adaptive_quantize(rgb_u8: np.ndarray, colors: int, dither: bool, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgb_u8: (H,W,3) uint8
    returns:
      palette_u8: (K,3)
      indices: (H,W) uint8 (0..K-1), where K == colors (or less, but usually colors)
    """
    pil = Image.fromarray(rgb_u8, mode="RGB")

    # Pillow methods: 0=MEDIANCUT, 1=MAXCOVERAGE, 2=FASTOCTREE, 3=LIBIMAGEQUANT (may not exist)
    method_map = {
        "Median-cut": 0,
        "Fast Octree": 2,
        "Max Coverage": 1,
    }
    m = method_map.get(method, 0)
    dith = Image.FLOYDSTEINBERG if dither else Image.NONE

    q = pil.quantize(colors=int(colors), method=m, dither=dith)
    pal = np.array(q.getpalette(), dtype=np.uint8).reshape(-1, 3)  # 256x3
    idx = np.array(q, dtype=np.uint8)  # (H,W)

    # effective palette entries are first `colors` (Pillow may still use <=colors)
    pal_eff = pal[:int(colors)].copy()
    return pal_eff, idx


def nearest_palette_indices(rgb_u8: np.ndarray, palette_u8: np.ndarray) -> np.ndarray:
    """
    Vectorized nearest palette assignment in RGB (squared euclidean).
    """
    H, W, _ = rgb_u8.shape
    X = rgb_u8.reshape(-1, 3).astype(np.int16)
    P = palette_u8.astype(np.int16)

    # chunk to keep memory bounded
    n = X.shape[0]
    out = np.zeros((n,), dtype=np.int32)
    chunk = 50000

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        Xi = X[i0:i1]
        d2 = ((Xi[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
        out[i0:i1] = d2.argmin(axis=1)

    return out.reshape(H, W).astype(np.int32)


# ============================================================
# Codec: palette-indexed image
#   - stores (H,W,K,bpp, palette_mode) + packed indices
#   - palette bytes stored ONLY when needed
# ============================================================

def encode_palette_image(
    img01_rgb: np.ndarray,
    max_side: int,
    palette_mode: str,
    palette_param: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    palette_mode:
      - "Adaptive (Pillow)"  -> store palette bytes + indices
      - "Fixed Palette"      -> store only indices; palette derived by name
    """
    img = resize_keep_aspect(img01_rgb, max_side=int(max_side))
    rgb_u8 = to_uint8(img)
    H, W, _ = rgb_u8.shape

    if palette_mode == "Adaptive (Pillow)":
        k = int(palette_param.get("k", 16))
        dither = bool(palette_param.get("dither", True))
        method = str(palette_param.get("method", "Median-cut"))
        palette_u8, idx_u8 = pil_adaptive_quantize(rgb_u8, colors=k, dither=dither, method=method)
        indices = idx_u8.astype(np.int32)
        palette_name = None
        palette_bytes = palette_u8.tobytes(order="C")
        palette_len = palette_u8.shape[0]

        # NOTE: idx may contain values up to 255 even if k<256 (rare but can happen with Pillow palette semantics)
        # Clamp safely:
        indices = np.clip(indices, 0, palette_len - 1)

        k_eff = palette_len

    elif palette_mode == "Fixed Palette":
        palette_name = str(palette_param.get("name", "CGA 16"))
        palette_u8 = fixed_palette_rgb_u8(palette_name)
        k_eff = int(palette_u8.shape[0])
        palette_bytes = b""  # not stored
        palette_len = k_eff

        indices = nearest_palette_indices(rgb_u8, palette_u8)

    else:
        raise ValueError("Unknown palette_mode")

    bpp = bits_needed(k_eff)
    packed_idx = pack_indices(indices.reshape(-1), bpp=bpp)

    meta = {
        "type": "palette_indexed_v1",
        "version": 1,
        "h": int(H),
        "w": int(W),
        "k": int(k_eff),
        "bpp": int(bpp),
        "palette_mode": palette_mode,
        "palette_name": palette_name,
        "palette_len": int(palette_len),
        "palette_bytes_len": int(len(palette_bytes)),
        "params": palette_param,
        "notes": "Best practical formula compression here: palette-indexed image + bit-packed indices + zlib.",
    }

    coeff_bytes = palette_bytes + packed_idx
    formula = pack_formula(meta, coeff_bytes)
    return formula, meta


def decode_palette_image(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    if meta.get("type") != "palette_indexed_v1":
        raise ValueError(f"Unsupported type: {meta.get('type')}")

    H = int(meta["h"])
    W = int(meta["w"])
    k = int(meta["k"])
    bpp = int(meta["bpp"])
    palette_mode = meta.get("palette_mode", "Adaptive (Pillow)")
    palette_name = meta.get("palette_name", None)
    pal_bytes_len = int(meta.get("palette_bytes_len", 0))

    if palette_mode == "Adaptive (Pillow)":
        palette_bytes = coeff_bytes[:pal_bytes_len]
        packed_idx = coeff_bytes[pal_bytes_len:]
        palette_u8 = np.frombuffer(palette_bytes, dtype=np.uint8)
        if palette_u8.size != k * 3:
            # tolerate if k differs from palette_len in meta, use palette_len
            pal_len = int(meta.get("palette_len", k))
            if palette_u8.size != pal_len * 3:
                raise ValueError("Palette bytes mismatch (corrupt formula).")
            k = pal_len
            palette_u8 = palette_u8.reshape(pal_len, 3)
        else:
            palette_u8 = palette_u8.reshape(k, 3)

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

    rgb = palette_u8[idx].reshape(H, W, 3).astype(np.uint8)
    return rgb


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
# Streamlit App
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula (Best Palette Compression)", layout="wide")
st.title("Image ⇄ Formula — Best Practical Compression with Simplified Color Spectrums")

st.markdown(
    "This app uses a **best-practical formula compression** approach (given pure Python + NumPy + Pillow):\n\n"
    "- Convert image to a **simplified color spectrum** (palette)\n"
    "- Store only the **palette** + **bit-packed pixel indices**\n"
    "- Compress with **zlib**, encode as a **single formula string**\n\n"
    "You can choose multiple spectrum types:\n"
    "- **Adaptive palette (Pillow)**: best quality per size, optional dithering\n"
    "- **Fixed palettes**: consistent stylization + very small formula (palette not stored)\n"
)

mode = st.radio("Choose conversion", ["Image → Formula", "Formula → Image"], horizontal=True)
st.divider()


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
        st.header("Output size / resolution")
        max_side = st.select_slider("Max side (preserve aspect)", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=1024)

        st.divider()
        st.header("Simplified color spectrum")
        palette_mode = st.radio("Spectrum type", ["Adaptive (Pillow)", "Fixed Palette"], index=0)

        palette_param: Dict[str, Any] = {}

        if palette_mode == "Adaptive (Pillow)":
            k = st.select_slider("Number of colors (K)", options=[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256], value=32)
            method = st.selectbox("Palette method", ["Median-cut", "Fast Octree", "Max Coverage"], index=1)
            dither = st.checkbox("Dither (Floyd–Steinberg)", value=True)
            palette_param = {"k": int(k), "method": method, "dither": bool(dither)}
            st.caption("Adaptive palettes usually deliver the best quality per byte. Dithering improves gradients.")

        else:
            name = st.selectbox(
                "Fixed palette",
                ["CGA 16", "GameBoy (4-color)", "Grayscale 16", "Grayscale 32", "Web-safe 216"],
                index=0
            )
            palette_param = {"name": name}
            st.caption("Fixed palettes are tiny (palette not stored) and give a strong stylized look.")

    try:
        img01 = load_image_from_choice(source_mode, chosen_local, upload)

        st.subheader("Selected image")
        st.image(to_uint8(img01), use_container_width=True)

        formula, meta = encode_palette_image(
            img01_rgb=img01,
            max_side=int(max_side),
            palette_mode=palette_mode,
            palette_param=palette_param
        )

        size_bytes = len(formula.encode("utf-8"))
        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=220)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Formula size", human_size(size_bytes))
        c2.metric("Resolution", f"{meta['w']}×{meta['h']}")
        c3.metric("Palette K", str(meta["k"]))
        c4.metric("Bits/pixel", str(meta["bpp"]))

        st.subheader("Mathematical depiction")
        st.latex(latex_palette_model())

        st.subheader("Reconstructed preview (from formula)")
        meta2, coeff2 = unpack_formula(formula)
        recon_u8 = decode_palette_image(meta2, coeff2)
        st.image(recon_u8, use_container_width=True)

        st.download_button(
            "Download formula as .txt",
            data=formula.encode("utf-8"),
            file_name="image_formula.txt",
            mime="text/plain",
        )

        with st.expander("Show decoded meta"):
            st.json(meta)

    except Exception as e:
        st.error(f"Image → Formula failed: {e}")


else:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=220, placeholder=f"{APP_FORMULA_PREFIX}...")

    if st.button("Reconstruct image", type="primary"):
        try:
            meta, coeff = unpack_formula(formula_in)
            img_u8 = decode_palette_image(meta, coeff)

            st.success(f"Reconstructed {meta['w']}×{meta['h']} | K={meta['k']} | bpp={meta['bpp']} | {meta['palette_mode']}")
            st.image(img_u8, use_container_width=True)

            out_pil = Image.fromarray(img_u8, mode="RGB")
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "Download reconstructed PNG",
                data=buf.getvalue(),
                file_name="reconstructed.png",
                mime="image/png",
            )

            st.subheader("Mathematical depiction")
            st.latex(latex_palette_model())

            with st.expander("Show decoded meta"):
                st.json(meta)

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")
