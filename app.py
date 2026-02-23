import os, glob, io, json, base64, zlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "IMGFORM_v3:"  # version
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


def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


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
# Color space helpers (YCbCr) for smaller base
# ============================================================

def rgb_to_ycbcr(rgb01: np.ndarray) -> np.ndarray:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
    cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(ycc01: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc01[..., 0], ycc01[..., 1] - 0.5, ycc01[..., 2] - 0.5
    r = y + 1.402*cr
    g = y - 0.344136*cb - 0.714136*cr
    b = y + 1.772*cb
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def downsample_mean(img: np.ndarray, factor: int) -> np.ndarray:
    H, W, C = img.shape
    f = int(factor)
    H2 = (H // f) * f
    W2 = (W // f) * f
    img = img[:H2, :W2, :]
    img = img.reshape(H2//f, f, W2//f, f, C).mean(axis=(1,3))
    return img


def upsample_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img), mode="RGB")
    pil = pil.resize((out_w, out_h), Image.BILINEAR)
    return to_float01(np.array(pil))


# ============================================================
# Simplified color spectrum (quantization)
# ============================================================

def quantize_bits(rgb01: np.ndarray, bits_per_channel: int) -> np.ndarray:
    """
    Very fast: reduces each channel to N levels.
    bits_per_channel=3 -> 8 levels; 4 -> 16; 5 -> 32.
    """
    b = int(bits_per_channel)
    levels = (1 << b) - 1
    q = np.round(rgb01 * levels) / levels
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def kmeans_palette(rgb01: np.ndarray, k: int, iters: int = 12, sample: int = 50000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight k-means in RGB space.
    Returns: (palette [k,3], labels [H*W])
    """
    rng = np.random.default_rng(seed)
    H, W, _ = rgb01.shape
    X = rgb01.reshape(-1, 3).astype(np.float32)

    n = X.shape[0]
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    # init centers by random samples
    k = int(k)
    centers = Xs[rng.choice(Xs.shape[0], size=k, replace=False)].copy()

    for _ in range(int(iters)):
        # assign
        d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)

        # update
        new_centers = centers.copy()
        for j in range(k):
            mask = (lab == j)
            if mask.any():
                new_centers[j] = Xs[mask].mean(axis=0)
        # stop if stable
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    # final assign for all pixels
    d2_all = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2_all.argmin(axis=1).astype(np.int32)
    palette = np.clip(centers, 0.0, 1.0).astype(np.float32)
    return palette, labels


def apply_palette(rgb01: np.ndarray, palette: np.ndarray, labels: np.ndarray) -> np.ndarray:
    H, W, _ = rgb01.shape
    out = palette[labels].reshape(H, W, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fixed_palette(name: str) -> np.ndarray:
    """
    Small fixed palettes in float01.
    """
    if name == "GameBoy (4-color)":
        # classic greenish
        pal = np.array([
            [15, 56, 15],
            [48, 98, 48],
            [139, 172, 15],
            [155, 188, 15],
        ], dtype=np.float32) / 255.0
    elif name == "CGA 16":
        pal = np.array([
            [0,0,0],[0,0,170],[0,170,0],[0,170,170],
            [170,0,0],[170,0,170],[170,85,0],[170,170,170],
            [85,85,85],[85,85,255],[85,255,85],[85,255,255],
            [255,85,85],[255,85,255],[255,255,85],[255,255,255],
        ], dtype=np.float32) / 255.0
    else:
        # "Web-safe 216"
        levels = np.array([0, 51, 102, 153, 204, 255], dtype=np.float32) / 255.0
        pal = np.array([[r, g, b] for r in levels for g in levels for b in levels], dtype=np.float32)
    return pal


def apply_fixed_palette(rgb01: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """
    Assign each pixel to nearest palette color (vectorized).
    """
    H, W, _ = rgb01.shape
    X = rgb01.reshape(-1, 3).astype(np.float32)
    # compute nearest palette entry
    d2 = ((X[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    out = pal[idx].reshape(H, W, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def simplify_color_spectrum(rgb01: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """
    spec:
      {"mode":"kmeans","k":16}
      {"mode":"fixed","name":"CGA 16"}
      {"mode":"bits","bpc":4}
    """
    mode = spec.get("mode", "kmeans")
    if mode == "bits":
        return quantize_bits(rgb01, int(spec.get("bpc", 4)))
    if mode == "fixed":
        pal = fixed_palette(spec.get("name", "CGA 16"))
        return apply_fixed_palette(rgb01, pal)
    # kmeans default
    k = int(spec.get("k", 16))
    palette, labels = kmeans_palette(rgb01, k=k, iters=int(spec.get("iters", 12)), sample=int(spec.get("sample", 50000)))
    return apply_palette(rgb01, palette, labels)


# ============================================================
# EBF (Edge + Brushstroke Field) core (unchanged)
# ============================================================

def sobel_edges(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)

    def conv2(a, k):
        H, W = a.shape
        ap = np.pad(a, ((1,1),(1,1)), mode="edge")
        out = np.zeros((H,W), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += k[i,j] * ap[i:i+H, j:j+W]
        return out

    gx = conv2(gray, kx)
    gy = conv2(gray, ky)
    mag = np.sqrt(gx*gx + gy*gy)
    return gx, gy, mag


def pick_sparse_points(mag: np.ndarray, K: int, min_dist: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = mag.shape
    flat_idx = np.argsort(mag.reshape(-1))[::-1]
    selected = []
    blocked = np.zeros((H,W), dtype=bool)
    r = int(min_dist)

    for idx in flat_idx:
        if len(selected) >= K:
            break
        y = int(idx // W)
        x = int(idx %  W)
        if blocked[y, x]:
            continue
        if mag[y, x] <= 1e-6:
            break
        selected.append((y, x))

        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        blocked[y0:y1, x0:x1] = True

    rng.shuffle(selected)
    return np.array(selected, dtype=np.int32)


def render_brushstrokes(base_ycc: np.ndarray, strokes: np.ndarray, sharpen: float) -> np.ndarray:
    ycc = base_ycc.copy()
    H, W, _ = ycc.shape

    yy = (np.arange(H, dtype=np.float32) + 0.5) / H
    xx = (np.arange(W, dtype=np.float32) + 0.5) / W
    Y, X = np.meshgrid(yy, xx, indexing="ij")

    for s in strokes:
        x0, y0, th, aY, aCb, aCr, sp, sn = s
        dx = X - x0
        dy = Y - y0
        c = np.cos(th); si = np.sin(th)
        u =  c*dx + si*dy
        v = -si*dx + c*dy

        sp = max(float(sp), 1e-4)
        sn = max(float(sn), 1e-4)
        g = np.exp(-0.5*((u/sp)**2 + (v/sn)**2)).astype(np.float32)
        stroke = (-v/(sn*sn)) * g

        ycc[..., 0] = np.clip(ycc[..., 0] + aY  * stroke, 0.0, 1.0)
        ycc[..., 1] = np.clip(ycc[..., 1] + aCb * stroke, 0.0, 1.0)
        ycc[..., 2] = np.clip(ycc[..., 2] + aCr * stroke, 0.0, 1.0)

    if sharpen > 0:
        y = ycc[..., 0]
        yp = np.pad(y, ((1,1),(1,1)), mode="edge")
        blur = (
            yp[0:H,0:W] + yp[0:H,1:W+1] + yp[0:H,2:W+2] +
            yp[1:H+1,0:W] + yp[1:H+1,1:W+1] + yp[1:H+1,2:W+2] +
            yp[2:H+2,0:W] + yp[2:H+2,1:W+1] + yp[2:H+2,2:W+2]
        ) / 9.0
        y = np.clip(y + float(sharpen) * (y - blur), 0.0, 1.0)
        ycc[..., 0] = y

    return ycc


def encode_edge_brush(
    img01_rgb: np.ndarray,
    max_side: int,
    base_factor: int,
    K: int,
    min_dist_px: int,
    quant_bits: int,
    sharpen: float,
    spectrum_spec: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    SAME EBF encoding, but meta now includes the simplified color spectrum spec.
    """
    img = resize_keep_aspect(img01_rgb, max_side=max_side)
    H, W, _ = img.shape

    ycc = rgb_to_ycbcr(img)
    y = ycc[..., 0]

    base = downsample_mean(ycc, factor=int(base_factor))
    Hb, Wb, _ = base.shape

    gx, gy, mag = sobel_edges(y)
    pts = pick_sparse_points(mag, K=int(K), min_dist=int(min_dist_px))

    # upsample base to compute residual
    base_rgb_low = ycbcr_to_rgb(base)
    base_rgb_full = upsample_bilinear(base_rgb_low, out_h=H, out_w=W)
    base_ycc_full = rgb_to_ycbcr(base_rgb_full)

    stroke_arr = np.zeros((len(pts), 8), dtype=np.float32)
    for i, (py, px) in enumerate(pts):
        ang = np.arctan2(gy[py, px], gx[py, px])
        theta = ang + np.pi/2

        amp = np.tanh(float(mag[py, px]) * 0.75)

        x0 = (px + 0.5) / W
        y0 = (py + 0.5) / H

        res = (ycc[py, px, :] - base_ycc_full[py, px, :]).astype(np.float32)
        aY  = float(np.clip(res[0] * 2.0 + amp * 0.20, -1.0, 1.0))
        aCb = float(np.clip(res[1] * 1.5, -1.0, 1.0))
        aCr = float(np.clip(res[2] * 1.5, -1.0, 1.0))

        sig_par  = float(np.clip(3.5 / max(W, H), 1e-4, 0.05))
        sig_perp = float(np.clip(1.2 / max(W, H), 1e-4, 0.03))

        stroke_arr[i] = [x0, y0, theta, aY, aCb, aCr, sig_par, sig_perp]

    # quantize base + strokes
    base_u8 = (np.clip(base, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    qb = int(quant_bits)
    qb = 12 if qb < 8 or qb > 14 else qb
    Q = (1 << qb) - 1

    def q01(x):   return np.clip(np.round(x * Q), 0, Q).astype(np.int32)
    def qpm1(x):  return np.clip(np.round((x * 0.5 + 0.5) * Q), 0, Q).astype(np.int32)
    def qang(x):
        x = (x + np.pi) % (2*np.pi) - np.pi
        return np.clip(np.round((x / (2*np.pi) + 0.5) * Q), 0, Q).astype(np.int32)
    def qs(x):    return np.clip(np.round((x / 0.1) * Q), 0, Q).astype(np.int32)

    sx = q01(stroke_arr[:, 0])
    sy = q01(stroke_arr[:, 1])
    stt = qang(stroke_arr[:, 2])
    aY = qpm1(stroke_arr[:, 3])
    aCb = qpm1(stroke_arr[:, 4])
    aCr = qpm1(stroke_arr[:, 5])
    sp = qs(stroke_arr[:, 6])
    sn = qs(stroke_arr[:, 7])

    stroke_q = np.stack([sx, sy, stt, aY, aCb, aCr, sp, sn], axis=1).astype(np.uint16)

    coeff_bytes = base_u8.tobytes(order="C") + stroke_q.tobytes(order="C")

    meta = {
        "type": "edge_brush_v1_simplified_spectrum",
        "version": 3,
        "max_side": int(max_side),
        "res_h": int(H),
        "res_w": int(W),
        "base_factor": int(base_factor),
        "base_h": int(Hb),
        "base_w": int(Wb),
        "K": int(len(pts)),
        "min_dist_px": int(min_dist_px),
        "quant_bits": int(qb),
        "sharpen": float(sharpen),
        "spectrum": spectrum_spec,  # << stored here
        "notes": "EBF decode + simplified color spectrum applied after reconstruction.",
    }

    return pack_formula(meta, coeff_bytes), meta


def decode_edge_brush(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    H = int(meta["res_h"]); W = int(meta["res_w"])
    Hb = int(meta["base_h"]); Wb = int(meta["base_w"])
    K = int(meta["K"])
    qb = int(meta["quant_bits"])
    Q = (1 << qb) - 1
    sharpen = float(meta.get("sharpen", 0.0))
    spectrum_spec = meta.get("spectrum", {"mode": "kmeans", "k": 16})

    # split
    base_n = Hb * Wb * 3
    base_u8 = np.frombuffer(coeff_bytes[:base_n], dtype=np.uint8).reshape(Hb, Wb, 3)
    stroke_q = np.frombuffer(coeff_bytes[base_n:], dtype=np.uint16)
    if stroke_q.size != K * 8:
        raise ValueError("Stroke payload mismatch (corrupt formula).")
    stroke_q = stroke_q.reshape(K, 8).astype(np.int32)

    base = base_u8.astype(np.float32) / 255.0

    base_rgb_low = ycbcr_to_rgb(base)
    base_rgb_full = upsample_bilinear(base_rgb_low, out_h=H, out_w=W)
    base_ycc_full = rgb_to_ycbcr(base_rgb_full)

    def uq01(q):  return (q / float(Q)).astype(np.float32)
    def uqpm1(q): return ((q / float(Q)) * 2.0 - 1.0).astype(np.float32)
    def uqang(q): return (((q / float(Q)) - 0.5) * (2*np.pi)).astype(np.float32)
    def uqs(q):   return ((q / float(Q)) * 0.1).astype(np.float32)

    x0 = uq01(stroke_q[:, 0])
    y0 = uq01(stroke_q[:, 1])
    th = uqang(stroke_q[:, 2])
    aY = uqpm1(stroke_q[:, 3])
    aCb = uqpm1(stroke_q[:, 4])
    aCr = uqpm1(stroke_q[:, 5])
    sp = uqs(stroke_q[:, 6])
    sn = uqs(stroke_q[:, 7])

    strokes = np.stack([x0, y0, th, aY, aCb, aCr, sp, sn], axis=1).astype(np.float32)

    ycc_out = render_brushstrokes(base_ycc_full, strokes, sharpen=sharpen)
    rgb01 = ycbcr_to_rgb(ycc_out)

    # ============================================================
    # APPLY SIMPLIFIED COLOR SPECTRUM HERE
    # ============================================================
    rgb01_simplified = simplify_color_spectrum(rgb01, spectrum_spec)

    return to_uint8(rgb01_simplified)


def latex_simplified_spectrum() -> str:
    return (
        r"We reconstruct an image via EBF and then project it onto a simplified color spectrum."
        "\n"
        r"\[ \hat{I}(x,y)=\Pi_{\mathcal{P}}(I_{\mathrm{EBF}}(x,y)) \]"
        "\n"
        r"where $\Pi_{\mathcal{P}}$ maps each pixel to its nearest color in a small palette $\mathcal{P}$ "
        r"(adaptive k-means palette, fixed palette, or per-channel bit-depth)."
    )


# ============================================================
# Streamlit app
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula (Simplified Spectrum)", layout="wide")
st.title("Image ⇄ Formula — EBF Reconstruction with Simplified Color Spectrum")

st.markdown(
    "- **Image → Formula** stores EBF parameters AND the chosen simplified color spectrum.\n"
    "- **Formula → Image** reconstructs via EBF and then quantizes colors to the simplified spectrum.\n"
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
        st.header("EBF settings")
        max_side = st.select_slider("Max side (preserve aspect)", options=[512, 768, 1024, 1280, 1536, 2048], value=1536)
        base_factor = st.select_slider("Base downsample factor", options=[4, 6, 8, 10, 12, 16], value=10)
        K = st.slider("Number of brushstrokes (K)", min_value=100, max_value=4000, value=900, step=50)
        min_dist = st.slider("Min distance between strokes (px)", min_value=2, max_value=20, value=6, step=1)
        quant_bits = st.select_slider("Quantization bits (smaller=smaller formula)", options=[9, 10, 11, 12, 13], value=11)
        sharpen = st.slider("Deterministic sharpening", 0.0, 1.5, 0.6, 0.05)

        st.divider()
        st.header("Simplified color spectrum")
        spec_mode = st.radio("Spectrum mode", ["Adaptive palette (k-means)", "Fixed palette", "Bits per channel"], index=0)

        if spec_mode == "Adaptive palette (k-means)":
            k = st.select_slider("Number of colors (K)", options=[4, 6, 8, 12, 16, 24, 32, 48, 64], value=16)
            spectrum_spec = {"mode": "kmeans", "k": int(k), "iters": 12, "sample": 50000}
        elif spec_mode == "Fixed palette":
            name = st.selectbox("Palette", ["CGA 16", "GameBoy (4-color)", "Web-safe 216"], index=0)
            spectrum_spec = {"mode": "fixed", "name": name}
        else:
            bpc = st.select_slider("Bits per channel", options=[2, 3, 4, 5, 6], value=4)
            spectrum_spec = {"mode": "bits", "bpc": int(bpc)}

        st.caption("Lower colors/bits = stronger simplification. Adaptive palette usually looks best for small palettes.")

    try:
        img01 = load_image_from_choice(source_mode, chosen_local, upload)
        st.subheader("Selected image")
        st.image(to_uint8(img01), use_container_width=True)

        formula, meta = encode_edge_brush(
            img01_rgb=img01,
            max_side=int(max_side),
            base_factor=int(base_factor),
            K=int(K),
            min_dist_px=int(min_dist),
            quant_bits=int(quant_bits),
            sharpen=float(sharpen),
            spectrum_spec=spectrum_spec,
        )

        size_bytes = len(formula.encode("utf-8"))

        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=220)

        a, b, c = st.columns(3)
        a.metric("Formula size", human_size(size_bytes))
        b.metric("Reconstruction size", f"{meta['res_w']}×{meta['res_h']}")
        c.metric("Spectrum", meta["spectrum"].get("mode", "unknown"))

        st.subheader("Mathematical depiction")
        st.latex(latex_simplified_spectrum())

        st.subheader("Reconstructed preview (from formula)")
        meta2, coeff2 = unpack_formula(formula)
        recon = decode_edge_brush(meta2, coeff2)
        st.image(recon, use_container_width=True)

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
            if meta.get("type") != "edge_brush_v1_simplified_spectrum":
                raise ValueError(
                    f"Unsupported type: {meta.get('type')} (expected edge_brush_v1_simplified_spectrum)."
                )

            img = decode_edge_brush(meta, coeff)
            st.success(f"Reconstructed {meta['res_w']}×{meta['res_h']} with simplified spectrum.")
            st.image(img, use_container_width=True)

            out_pil = Image.fromarray(img, mode="RGB")
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "Download reconstructed PNG",
                data=buf.getvalue(),
                file_name="reconstructed.png",
                mime="image/png",
            )

            st.subheader("Mathematical depiction")
            st.latex(latex_simplified_spectrum())

            with st.expander("Show decoded meta"):
                st.json(meta)

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")
