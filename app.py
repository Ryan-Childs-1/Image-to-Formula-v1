import os, glob, io, json, base64, zlib, math, time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Stability defaults (prevents crashes)
# ============================================================

# Hard limits to prevent runaway memory/CPU
MAX_PIXELS_HARD = 6_000_000         # ~6 MP cap (e.g., 3000x2000). Adjust if needed.
MAX_SIDE_HARD   = 2048              # cap resize max side in UI
MAX_PALETTE_K   = 256               # Pillow limit
NEAREST_CHUNK   = 25_000            # smaller chunk = lower peak RAM for fixed-palette nearest search

# PIL safety (protect against decompression bombs)
Image.MAX_IMAGE_PIXELS = 20_000_000  # allow up to 20MP, but we will still downscale/cap ourselves


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "PALIMG_v2:"    # bumped version (safe changes)
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
    """
    Streamlit UploadedFile has .size sometimes; otherwise we can read buffer length.
    """
    if upload_file is None:
        return None
    sz = getattr(upload_file, "size", None)
    if isinstance(sz, int):
        return sz
    try:
        # UploadedFile implements getvalue()
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
    """
    Returns:
      img01_rgb: float32 [0,1] (H,W,3)
      info: dict with original file size and original pixel dims
    """
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

    else:
        if upload_file is None:
            raise ValueError("No image uploaded.")
        info["upload_name"] = getattr(upload_file, "name", "upload")
        info["orig_file_bytes"] = safe_file_size_bytes_from_uploaded(upload_file)
        # Use file-like directly; do not .getvalue() unless needed
        with Image.open(upload_file) as im:
            im = im.convert("RGB")
            info["orig_w"], info["orig_h"] = im.size
            arr = np.array(im)
        return to_float01(arr), info


def cap_resize_keep_aspect(img01_rgb: np.ndarray, max_side: int, max_pixels: int) -> np.ndarray:
    """
    Aggressive safety resize:
      - First ensures max_side <= MAX_SIDE_HARD
      - Ensures max(H,W) <= max_side
      - Ensures H*W <= max_pixels
    """
    max_side = int(min(max_side, MAX_SIDE_HARD))
    H, W, _ = img01_rgb.shape

    # Compute required scale based on side cap
    s1 = 1.0
    if max(H, W) > max_side:
        s1 = max_side / float(max(H, W))

    # Compute required scale based on pixel cap
    s2 = 1.0
    if H * W > max_pixels:
        s2 = math.sqrt(max_pixels / float(H * W))

    scale = min(s1, s2, 1.0)
    if scale >= 0.999:
        return img01_rgb

    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))

    pil = Image.fromarray(to_uint8(img01_rgb), mode="RGB")
    pil = pil.resize((newW, newH), Image.LANCZOS)
    return to_float01(np.array(pil))


def maybe_downscale_u8(img_u8: np.ndarray, max_side: int, max_pixels: int) -> np.ndarray:
    """
    Downscale uint8 for download / preview with pixel cap.
    """
    max_side = int(min(max_side, MAX_SIDE_HARD))
    H, W, _ = img_u8.shape

    s1 = 1.0
    if max(H, W) > max_side:
        s1 = max_side / float(max(H, W))

    s2 = 1.0
    if H * W > max_pixels:
        s2 = math.sqrt(max_pixels / float(H * W))

    scale = min(s1, s2, 1.0)
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


# ============================================================
# Binary formula encoding/decoding
# ============================================================

def pack_formula(meta: Dict[str, Any], coeff_bytes: bytes) -> str:
    """
    Single formula string = PREFIX + base64url(zlib(meta_json + SEP + coeff_bytes))
    """
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
    bpp = int(bpp)
    idx = indices.astype(np.uint32, copy=False).ravel()

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
# Quantization engines (bounded for safety)
# ============================================================

def pil_adaptive_quantize(rgb_u8: np.ndarray, colors: int, dither: bool, method: str) -> Tuple[np.ndarray, np.ndarray]:
    pil = Image.fromarray(rgb_u8, mode="RGB")

    method_map = {"Median-cut": 0, "Fast Octree": 2, "Max Coverage": 1}
    m = method_map.get(method, 2)
    dith = Image.FLOYDSTEINBERG if dither else Image.NONE

    # Pillow quantize returns "P" mode
    q = pil.quantize(colors=int(colors), method=m, dither=dith)
    pal = np.array(q.getpalette(), dtype=np.uint8).reshape(-1, 3)
    idx = np.array(q, dtype=np.uint8)

    pal_eff = pal[:int(colors)].copy()
    return pal_eff, idx


def nearest_palette_indices(rgb_u8: np.ndarray, palette_u8: np.ndarray) -> np.ndarray:
    """
    Bounded-memory nearest assignment.
    Avoids huge intermediate arrays by chunking.
    """
    H, W, _ = rgb_u8.shape
    X = rgb_u8.reshape(-1, 3).astype(np.int16, copy=False)
    P = palette_u8.astype(np.int16, copy=False)

    n = X.shape[0]
    out = np.zeros((n,), dtype=np.int32)

    chunk = int(NEAREST_CHUNK)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        Xi = X[i0:i1]
        # (chunk, K, 3) can still be heavy if K is large; but fixed palettes are <=216 here
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
    Returns:
      formula, meta, recon_u8 (reconstructed from the encoded data)
    We reconstruct immediately to provide accurate compression stats + download.
    """
    t0 = time.time()

    # Safety resize BEFORE quantization (big crash fix)
    img = cap_resize_keep_aspect(img01_rgb, max_side=int(max_side), max_pixels=int(max_pixels))
    rgb_u8 = to_uint8(img)
    H, W, _ = rgb_u8.shape

    if palette_mode == "Adaptive (Pillow)":
        k = int(palette_param.get("k", 32))
        k = max(2, min(k, MAX_PALETTE_K))
        dither = bool(palette_param.get("dither", True))
        method = str(palette_param.get("method", "Fast Octree"))
        palette_u8, idx_u8 = pil_adaptive_quantize(rgb_u8, colors=k, dither=dither, method=method)
        indices = idx_u8.astype(np.int32, copy=False)

        palette_name = None
        palette_bytes = palette_u8.tobytes(order="C")
        palette_len = int(palette_u8.shape[0])

        # clamp indices safely
        indices = np.clip(indices, 0, palette_len - 1)
        k_eff = palette_len

    elif palette_mode == "Fixed Palette":
        palette_name = str(palette_param.get("name", "CGA 16"))
        palette_u8 = fixed_palette_rgb_u8(palette_name)
        k_eff = int(palette_u8.shape[0])
        palette_bytes = b""
        palette_len = k_eff

        # Fixed palette assignment can be expensive; we're chunked for safety
        indices = nearest_palette_indices(rgb_u8, palette_u8)

    else:
        raise ValueError("Unknown palette_mode")

    bpp = bits_needed(k_eff)
    packed_idx = pack_indices(indices.reshape(-1), bpp=bpp)

    meta = {
        "type": "palette_indexed_v1",
        "version": 2,
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
        "notes": "Stability-hardened palette-indexed image + bit-packed indices + zlib.",
    }

    coeff_bytes = palette_bytes + packed_idx
    formula = pack_formula(meta, coeff_bytes)

    # Reconstruct for preview/download/stats (no need to decompress formula again)
    recon_u8 = palette_u8[np.clip(indices, 0, k_eff - 1)].reshape(H, W, 3).astype(np.uint8)

    return formula, meta, recon_u8


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
        pal_len = int(meta.get("palette_len", k))
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

st.set_page_config(page_title="Image ⇄ Formula (Stable)", layout="wide")
st.title("Image ⇄ Formula — Stable, Crash-Resistant, With Compression Stats")

st.markdown(
    "This version is hardened to avoid crashing:\n\n"
    "- **Hard caps** on pixels and side-length (prevents runaway memory)\n"
    "- **Chunked** nearest-palette computation\n"
    "- Shows **original file size** + **compressed outputs** so you can see compression ratios\n"
    "- Lets you download a **smaller compressed image** after modification\n"
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
        st.header("Safety caps (prevents crashes)")
        # Users can LOWER caps if their machine is struggling
        max_pixels = st.select_slider(
            "Max working pixels (H×W cap)",
            options=[750_000, 1_500_000, 3_000_000, 6_000_000, 10_000_000],
            value=min(MAX_PIXELS_HARD, 3_000_000)
        )
        st.caption("Lower this if your computer struggles. The app will downscale automatically.")

        st.divider()
        st.header("Output resolution")
        max_side = st.select_slider(
            "Max side (preserve aspect)",
            options=[256, 384, 512, 768, 1024, 1280, 1536, 2048],
            value=1024
        )

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
        st.header("Download smaller image")
        dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0)
        dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768)
        dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1)
        dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True)
        dl_lossless_webp = st.checkbox("Lossless WEBP", value=False)

    try:
        img01, info = load_image_from_choice(source_mode, chosen_local, upload)

        # Display original (but do NOT render gigantic images full-res)
        st.subheader("Original image (input)")
        st.image(to_uint8(cap_resize_keep_aspect(img01, max_side=1024, max_pixels=1_500_000)), use_container_width=True)

        # Compression stats: original file size + original dims
        orig_file_bytes = info.get("orig_file_bytes", None)
        orig_w = info.get("orig_w", None)
        orig_h = info.get("orig_h", None)

        # Encode safely (this is the heavy step)
        with st.spinner("Encoding image into formula (bounded + crash-resistant)…"):
            formula, meta, recon_u8 = encode_palette_image(
                img01_rgb=img01,
                max_side=int(max_side),
                max_pixels=int(max_pixels),
                palette_mode=palette_mode,
                palette_param=palette_param,
            )

        # Sizes
        formula_bytes = len(formula.encode("utf-8"))

        # Also compute a real compressed-image file size for apples-to-apples comparison
        # (e.g., "what if I just saved this image as webp/jpeg?")
        # We'll produce a default "comparison" export using WEBP quality 80 and same size as recon.
        comp_webp = pil_save_bytes(recon_u8, "WEBP", quality=80, method=6)
        comp_webp_bytes = len(comp_webp)

        st.subheader("Compression summary")
        a, b, c, d = st.columns(4)

        if isinstance(orig_file_bytes, int):
            a.metric("Original file size", human_size(orig_file_bytes))
        else:
            a.metric("Original file size", "Unknown")

        b.metric("Working resolution", f"{meta['w']}×{meta['h']}")
        c.metric("Formula size", human_size(formula_bytes))
        d.metric("Encode time", f"{meta.get('encode_seconds', 0.0):.2f}s")

        # Ratios
        r1, r2 = st.columns(2)
        with r1:
            if isinstance(orig_file_bytes, int) and orig_file_bytes > 0:
                ratio = orig_file_bytes / float(formula_bytes)
                st.metric("Orig / Formula ratio", f"{ratio:.2f}×")
            else:
                st.metric("Orig / Formula ratio", "—")
        with r2:
            if isinstance(orig_file_bytes, int) and orig_file_bytes > 0:
                ratio2 = orig_file_bytes / float(comp_webp_bytes)
                st.metric("Orig / WEBP(80) ratio", f"{ratio2:.2f}×")
            else:
                st.metric("Orig / WEBP(80) ratio", "—")

        st.caption(
            "Note: compression ratios depend on the original file format/quality. "
            "A high-quality PNG may be much larger than an equivalently-sized WEBP/JPEG."
        )

        # Formula output
        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=220)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Palette K", str(meta["k"]))
        c2.metric("Bits/pixel", str(meta["bpp"]))
        c3.metric("Palette mode", meta["palette_mode"])
        c4.metric("Pixel cap used", f"{int(max_pixels):,}")

        st.subheader("Mathematical depiction")
        st.latex(latex_palette_model())

        # Preview reconstruction (bounded display)
        st.subheader("Reconstructed preview (from encoded data)")
        preview_u8 = maybe_downscale_u8(recon_u8, max_side=1024, max_pixels=1_500_000)
        st.image(preview_u8, use_container_width=True)

        st.download_button(
            "Download formula as .txt",
            data=formula.encode("utf-8"),
            file_name="image_formula.txt",
            mime="text/plain",
        )

        # NEW: Download compressed modified image with user-selected options
        st.subheader("Download a smaller compressed image (modified)")

        dl_u8 = maybe_downscale_u8(recon_u8, max_side=int(dl_max_side), max_pixels=int(max_pixels))

        fmt = dl_format.split()[0]  # WEBP/JPEG/PNG
        if fmt == "WEBP":
            data = pil_save_bytes(
                dl_u8,
                "WEBP",
                quality=int(dl_quality),
                lossless=bool(dl_lossless_webp),
                method=6
            )
            out_name = "modified_compressed.webp"
            mime = "image/webp"
        elif fmt == "JPEG":
            data = pil_save_bytes(
                dl_u8,
                "JPEG",
                quality=int(dl_quality),
                optimize=True,
                progressive=True
            )
            out_name = "modified_compressed.jpg"
            mime = "image/jpeg"
        else:
            data = pil_save_bytes(
                dl_u8,
                "PNG",
                optimize=bool(dl_opt_png),
                compress_level=9
            )
            out_name = "modified_compressed.png"
            mime = "image/png"

        e1, e2, e3 = st.columns(3)
        e1.metric("Download image size", f"{dl_u8.shape[1]}×{dl_u8.shape[0]}")
        e2.metric("Download file size", human_size(len(data)))
        if isinstance(orig_file_bytes, int) and orig_file_bytes > 0:
            e3.metric("Orig / Download ratio", f"{orig_file_bytes/float(len(data)):.2f}×")
        else:
            e3.metric("Orig / Download ratio", "—")

        st.download_button(
            "Download compressed modified image",
            data=data,
            file_name=out_name,
            mime=mime,
        )

        with st.expander("Show details (meta + input info)"):
            st.json({"input_info": info, "meta": meta})

        if isinstance(orig_w, int) and isinstance(orig_h, int):
            st.caption(f"Original pixel dimensions: {orig_w}×{orig_h}")

    except Exception as e:
        st.error(f"Image → Formula failed: {e}")


# ============================================================
# FORMULA → IMAGE
# ============================================================

else:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=220, placeholder=f"{APP_FORMULA_PREFIX}...")

    st.markdown("### Download options (after reconstruction)")
    dl_format = st.selectbox("File format", ["WEBP", "JPEG", "PNG (optimized)"], index=0, key="dl_fmt_2")
    dl_max_side = st.select_slider("Download max side", options=[256, 384, 512, 768, 1024, 1280, 1536, 2048], value=768, key="dl_side_2")
    dl_quality = st.slider("Quality (WEBP/JPEG)", min_value=10, max_value=95, value=80, step=1, key="dl_q_2")
    dl_opt_png = st.checkbox("Optimize PNG (slower)", value=True, key="dl_png_2")
    dl_lossless_webp = st.checkbox("Lossless WEBP", value=False, key="dl_webp_lossless_2")

    if st.button("Reconstruct image", type="primary"):
        try:
            with st.spinner("Decoding formula…"):
                meta, coeff = unpack_formula(formula_in)
                img_u8 = decode_palette_image(meta, coeff)

            st.success(f"Reconstructed {meta['w']}×{meta['h']} | K={meta['k']} | bpp={meta['bpp']} | {meta['palette_mode']}")

            preview_u8 = maybe_downscale_u8(img_u8, max_side=1024, max_pixels=1_500_000)
            st.image(preview_u8, use_container_width=True)

            # Original reconstructed PNG download (lossless)
            out_pil = Image.fromarray(img_u8, mode="RGB")
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "Download reconstructed PNG (lossless)",
                data=buf.getvalue(),
                file_name="reconstructed.png",
                mime="image/png",
            )

            # NEW: compressed download
            st.subheader("Download a smaller compressed image")
            dl_u8 = maybe_downscale_u8(img_u8, max_side=int(dl_max_side), max_pixels=MAX_PIXELS_HARD)

            fmt = dl_format.split()[0]
            if fmt == "WEBP":
                data = pil_save_bytes(
                    dl_u8,
                    "WEBP",
                    quality=int(dl_quality),
                    lossless=bool(dl_lossless_webp),
                    method=6
                )
                out_name = "compressed.webp"
                mime = "image/webp"
            elif fmt == "JPEG":
                data = pil_save_bytes(
                    dl_u8,
                    "JPEG",
                    quality=int(dl_quality),
                    optimize=True,
                    progressive=True
                )
                out_name = "compressed.jpg"
                mime = "image/jpeg"
            else:
                data = pil_save_bytes(
                    dl_u8,
                    "PNG",
                    optimize=bool(dl_opt_png),
                    compress_level=9
                )
                out_name = "compressed.png"
                mime = "image/png"

            d1, d2 = st.columns(2)
            d1.metric("Download image size", f"{dl_u8.shape[1]}×{dl_u8.shape[0]}")
            d2.metric("Download file size", human_size(len(data)))

            st.download_button(
                "Download compressed image",
                data=data,
                file_name=out_name,
                mime=mime,
            )

            st.subheader("Mathematical depiction")
            st.latex(latex_palette_model())

            with st.expander("Show decoded meta"):
                st.json(meta)

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")
