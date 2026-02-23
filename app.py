import os, glob, io, json, base64, zlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "IMGFORM_v3:"  # NEW version
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
    # rgb01: float [0,1]
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
    # img: (H,W,C)
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
# Novel representation: Edge + Brushstroke Field (EBF)
# ============================================================

def sobel_edges(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    gray: (H,W) float
    Returns: gx, gy, mag
    """
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
    """
    Greedy non-maximum selection on gradient magnitude.
    Returns array of (y,x) points length <=K.
    """
    rng = np.random.default_rng(seed)
    H, W = mag.shape
    flat_idx = np.argsort(mag.reshape(-1))[::-1]  # descending
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

        y0 = max(0, y - r)
        y1 = min(H, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(W, x + r + 1)
        blocked[y0:y1, x0:x1] = True

    # slight shuffle to avoid visible ordering artifacts when rendering
    rng.shuffle(selected)
    return np.array(selected, dtype=np.int32)


def render_brushstrokes(
    base_ycc: np.ndarray,
    strokes: np.ndarray,
    out_h: int,
    out_w: int,
    sharpen: float
) -> np.ndarray:
    """
    base_ycc: (H,W,3) float base already upsampled to out_h/out_w in YCbCr.
    strokes: (K, 8) float32: [x, y, theta, ampY, ampCb, ampCr, sig_par, sig_perp]
      x,y in [0,1], theta in radians, amps in [-1,1] approx, sig in normalized units.
    Adds anisotropic gaussian-derivative-like strokes.
    """
    ycc = base_ycc.copy()
    H, W, _ = ycc.shape

    # precompute coordinate grid in [0,1]
    yy = (np.arange(H, dtype=np.float32) + 0.5) / H
    xx = (np.arange(W, dtype=np.float32) + 0.5) / W
    Y, X = np.meshgrid(yy, xx, indexing="ij")  # (H,W)

    # add strokes
    for s in strokes:
        x0, y0, th, aY, aCb, aCr, sp, sn = s
        # rotate coordinates around (x0,y0)
        dx = X - x0
        dy = Y - y0
        c = np.cos(th); si = np.sin(th)
        u =  c*dx + si*dy
        v = -si*dx + c*dy

        # anisotropic gaussian envelope
        sp = max(float(sp), 1e-4)
        sn = max(float(sn), 1e-4)
        g = np.exp(-0.5*((u/sp)**2 + (v/sn)**2)).astype(np.float32)

        # "edge-like" stroke: derivative along v (perp) gives crisp line
        # d/dv gaussian ~ -(v/sn^2)*g
        stroke = (-v/(sn*sn)) * g

        ycc[..., 0] = np.clip(ycc[..., 0] + aY  * stroke, 0.0, 1.0)
        ycc[..., 1] = np.clip(ycc[..., 1] + aCb * stroke, 0.0, 1.0)
        ycc[..., 2] = np.clip(ycc[..., 2] + aCr * stroke, 0.0, 1.0)

    # optional deterministic sharpening on luminance only (no bytes)
    if sharpen > 0:
        y = ycc[..., 0]
        # unsharp mask: y + k*(y - blur(y))
        # cheap blur:
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
) -> Tuple[str, Dict[str, Any]]:
    """
    Encode:
      - Resize image to max_side
      - Convert to YCbCr
      - Base: downsample by base_factor, store as uint8 (or uint6/uint5 packed) in bytes
      - Strokes: pick K edge points on luminance gradient, store params in quantized int16
    """
    img = resize_keep_aspect(img01_rgb, max_side=max_side)
    H, W, _ = img.shape

    ycc = rgb_to_ycbcr(img)
    y = ycc[..., 0]

    # base low-res field (YCbCr then stored compactly)
    base_factor = int(base_factor)
    base = downsample_mean(ycc, factor=base_factor)  # (Hb,Wb,3)
    Hb, Wb, _ = base.shape

    # edge selection on luminance
    gx, gy, mag = sobel_edges(y)
    pts = pick_sparse_points(mag, K=int(K), min_dist=int(min_dist_px))

    # build stroke parameters
    # normalize coordinates to [0,1]
    strokes = []
    for (py, px) in pts:
        # angle of edge normal/perp; we want stroke aligned along edge tangent
        ang = np.arctan2(gy[py, px], gx[py, px])  # gradient direction
        theta = ang + np.pi/2  # tangent direction

        # amplitude based on local contrast: sample difference across gradient direction
        # simple: scale by mag
        amp = float(mag[py, px])
        # normalize amp into manageable range
        amp = np.tanh(amp * 0.75)

        # color residual at that point (difference from upsampled base)
        # upsample base to full to estimate residual
        # (do once outside loop for speed)
        strokes.append((py, px, theta, amp))

    # upsample base to compute residual colors
    base_rgb_low = ycbcr_to_rgb(base)
    base_rgb_full = upsample_bilinear(base_rgb_low, out_h=H, out_w=W)
    base_ycc_full = rgb_to_ycbcr(base_rgb_full)

    stroke_arr = np.zeros((len(pts), 8), dtype=np.float32)
    for i, (py, px, theta, amp) in enumerate(strokes):
        x0 = (px + 0.5) / W
        y0 = (py + 0.5) / H

        # residual at point in YCbCr
        res = (ycc[py, px, :] - base_ycc_full[py, px, :]).astype(np.float32)
        # stroke amplitudes: aY from amp plus residual; chroma from residual only (scaled)
        aY  = float(np.clip(res[0] * 2.0 + amp * 0.20, -1.0, 1.0))
        aCb = float(np.clip(res[1] * 1.5, -1.0, 1.0))
        aCr = float(np.clip(res[2] * 1.5, -1.0, 1.0))

        # stroke sizes (normalized): tie to base_factor / resolution
        # parallel larger, perpendicular smaller -> edge-like
        sig_par  = float(np.clip(3.5 / max(W, H), 1e-4, 0.05))
        sig_perp = float(np.clip(1.2 / max(W, H), 1e-4, 0.03))

        stroke_arr[i] = [x0, y0, theta, aY, aCb, aCr, sig_par, sig_perp]

    # quantize base + strokes
    # Base as uint8 YCbCr (already 0..1)
    base_u8 = (np.clip(base, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    # Strokes quantized
    # pack to int16 using quant_bits (e.g., 10-12 bits effective)
    qb = int(quant_bits)
    if qb < 8 or qb > 14:
        qb = 12
    Q = (1 << qb) - 1

    def q01(x):  # [0,1] -> [0,Q]
        return np.clip(np.round(x * Q), 0, Q).astype(np.int32)

    def qpm1(x):  # [-1,1] -> [0,Q]
        return np.clip(np.round((x * 0.5 + 0.5) * Q), 0, Q).astype(np.int32)

    def qang(x):  # [-pi,pi] -> [0,Q]
        # wrap to [-pi,pi]
        x = (x + np.pi) % (2*np.pi) - np.pi
        return np.clip(np.round((x / (2*np.pi) + 0.5) * Q), 0, Q).astype(np.int32)

    def qs(x):  # [0,0.1] roughly -> [0,Q]
        return np.clip(np.round((x / 0.1) * Q), 0, Q).astype(np.int32)

    sx = q01(stroke_arr[:, 0])
    sy = q01(stroke_arr[:, 1])
    st = qang(stroke_arr[:, 2])
    aY = qpm1(stroke_arr[:, 3])
    aCb = qpm1(stroke_arr[:, 4])
    aCr = qpm1(stroke_arr[:, 5])
    sp = qs(stroke_arr[:, 6])
    sn = qs(stroke_arr[:, 7])

    stroke_q = np.stack([sx, sy, st, aY, aCb, aCr, sp, sn], axis=1).astype(np.uint16)
    stroke_bytes = stroke_q.tobytes(order="C")

    # Build coeff bytes blob:
    # [base_u8 bytes][stroke bytes]
    base_bytes = base_u8.tobytes(order="C")
    coeff_bytes = base_bytes + stroke_bytes

    meta = {
        "type": "edge_brush_v1",
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
        "base_dtype": "uint8",
        "stroke_dtype": "uint16",
        "stroke_shape": [int(len(pts)), 8],
        "notes": "Novel: low-res YCbCr base + sparse anisotropic brushstroke atoms along edges.",
    }

    formula = pack_formula(meta, coeff_bytes)
    return formula, meta


def decode_edge_brush(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    H = int(meta["res_h"]); W = int(meta["res_w"])
    Hb = int(meta["base_h"]); Wb = int(meta["base_w"])
    K = int(meta["K"])
    qb = int(meta["quant_bits"])
    Q = (1 << qb) - 1
    sharpen = float(meta.get("sharpen", 0.0))

    # split bytes
    base_n = Hb * Wb * 3  # uint8
    base_bytes = coeff_bytes[:base_n]
    stroke_bytes = coeff_bytes[base_n:]

    base_u8 = np.frombuffer(base_bytes, dtype=np.uint8).reshape(Hb, Wb, 3)
    base = base_u8.astype(np.float32) / 255.0

    # upsample base to target size (in RGB then back to YCbCr for stroke add)
    base_rgb_low = ycbcr_to_rgb(base)
    base_rgb_full = upsample_bilinear(base_rgb_low, out_h=H, out_w=W)
    base_ycc_full = rgb_to_ycbcr(base_rgb_full)

    stroke_q = np.frombuffer(stroke_bytes, dtype=np.uint16)
    if stroke_q.size != K * 8:
        raise ValueError("Stroke payload mismatch (corrupt formula).")
    stroke_q = stroke_q.reshape(K, 8).astype(np.int32)

    def uq01(q):  # [0,Q] -> [0,1]
        return (q / float(Q)).astype(np.float32)

    def uqpm1(q):  # [0,Q] -> [-1,1]
        return ((q / float(Q)) * 2.0 - 1.0).astype(np.float32)

    def uqang(q):  # [0,Q] -> [-pi,pi]
        return (((q / float(Q)) - 0.5) * (2*np.pi)).astype(np.float32)

    def uqs(q):  # [0,Q] -> [0,0.1]
        return ((q / float(Q)) * 0.1).astype(np.float32)

    x0 = uq01(stroke_q[:, 0])
    y0 = uq01(stroke_q[:, 1])
    th = uqang(stroke_q[:, 2])
    aY  = uqpm1(stroke_q[:, 3])
    aCb = uqpm1(stroke_q[:, 4])
    aCr = uqpm1(stroke_q[:, 5])
    sp = uqs(stroke_q[:, 6])
    sn = uqs(stroke_q[:, 7])

    strokes = np.stack([x0, y0, th, aY, aCb, aCr, sp, sn], axis=1).astype(np.float32)

    ycc_out = render_brushstrokes(base_ycc_full, strokes, out_h=H, out_w=W, sharpen=sharpen)
    rgb = ycbcr_to_rgb(ycc_out)
    return to_uint8(rgb)


def latex_edge_brush() -> str:
    return (
        r"Novel depiction: Edge + Brushstroke Field (EBF)."
        "\n"
        r"We store a low-frequency base color field $B(x,y)$ and add sparse oriented atoms:"
        "\n"
        r"\[ I(x,y)=U(B(x,y)) + \sum_{k=1}^{K}\alpha_k\;\phi\!\left(R_{\theta_k}\begin{bmatrix}x-x_k\\y-y_k\end{bmatrix};"
        r"\sigma_{k,\parallel},\sigma_{k,\perp}\right) \]"
        "\n"
        r"$U$ is a deterministic upsampler (no extra bytes), and each $\phi$ is an anisotropic edge-like brushstroke."
    )


# ============================================================
# Streamlit app
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula (New Compression)", layout="wide")
st.title("Image ⇄ Formula — Higher Resolution, Smaller Formula (Edge + Brushstroke Field)")

st.markdown(
    "This version introduces a **new approach** designed to **increase resolution while decreasing formula size**:\n\n"
    "- **EBF** = low-res YCbCr base + **sparse edge brushstrokes** (math atoms)\n"
    "- Great for large images because edges are sparse.\n\n"
    "Modes:\n"
    "- Image → Formula\n"
    "- Formula → Image"
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
        sharpen = st.slider("Deterministic sharpening (no extra bytes)", 0.0, 1.5, 0.6, 0.05)

        st.caption("Try: Max side 1536–2048 + base_factor 10–16 + K 600–1200 for strong compression.")

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
        )

        size_bytes = len(formula.encode("utf-8"))

        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=220)

        a, b, c = st.columns(3)
        a.metric("Formula size", human_size(size_bytes))
        b.metric("Reconstruction size", f"{meta['res_w']}×{meta['res_h']}")
        c.metric("Brushstrokes K", str(meta["K"]))

        st.subheader("Mathematical depiction")
        st.latex(latex_edge_brush())

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
            if meta.get("type") != "edge_brush_v1":
                raise ValueError(f"Unsupported type: {meta.get('type')} (expected edge_brush_v1).")

            img = decode_edge_brush(meta, coeff)
            st.success(f"Reconstructed {meta['res_w']}×{meta['res_h']} (EBF).")
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
            st.latex(latex_edge_brush())

            with st.expander("Show decoded meta"):
                st.json(meta)

        except Exception as e:
            st.error(f"Formula → Image failed: {e}")
