import os
import glob
import io
import json
import base64
import zlib
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Formula container
# ============================================================

APP_FORMULA_PREFIX = "IMGFORM_v2:"  # single-string formula prefix
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


def resize_keep_aspect_and_pad(img01_rgb: np.ndarray, max_side: int, pad_multiple: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Resize so max(H,W)=max_side while preserving aspect ratio.
    Then pad to multiples of pad_multiple (for tiling).
    Returns padded image and meta describing original/padded shapes.
    """
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
    resized = to_float01(np.array(pil))

    rH, rW, _ = resized.shape

    # pad to multiple
    def up_to_mult(n, m):
        return ((n + m - 1) // m) * m

    pH = up_to_mult(rH, pad_multiple)
    pW = up_to_mult(rW, pad_multiple)

    pad_bottom = pH - rH
    pad_right = pW - rW

    padded = np.pad(
        resized,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="edge"
    )

    meta = {
        "orig_h": int(H),
        "orig_w": int(W),
        "res_h": int(rH),
        "res_w": int(rW),
        "pad_h": int(pH),
        "pad_w": int(pW),
    }
    return padded, meta


def crop_to_resized(img01_rgb: np.ndarray, res_h: int, res_w: int) -> np.ndarray:
    return img01_rgb[:res_h, :res_w, :]


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
    """
    Single formula string = PREFIX + base64url(zlib( meta_json + SEP + coeff_bytes ))
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

    try:
        meta = json.loads(meta_json.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid meta JSON: {e}")

    return meta, coeff_bytes


# ============================================================
# Global Fourier representation (improved)
# ============================================================

def fft2_truncate_center(chan01: np.ndarray, keep: int) -> np.ndarray:
    """
    Returns centered keep×keep complex block from fftshift(fft2(chan)).
    """
    N0, N1 = chan01.shape
    if N0 != N1:
        raise ValueError("Global FFT mode requires square input internally.")
    N = N0
    keep = int(keep)
    if keep < 1 or keep > N:
        raise ValueError("keep must be between 1 and N.")

    F = np.fft.fft2(chan01)
    Fs = np.fft.fftshift(F)

    c = N // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep
    return Fs[r0:r1, r0:r1].astype(np.complex64)


def ifft2_from_center_block(N: int, keep: int, block: np.ndarray) -> np.ndarray:
    """
    Reconstruct approx channel from keep×keep centered frequency block.
    """
    N = int(N)
    keep = int(keep)
    if block.shape != (keep, keep):
        raise ValueError(f"Block shape {block.shape} != ({keep},{keep})")

    Fs = np.zeros((N, N), dtype=np.complex64)
    c = N // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep
    Fs[r0:r1, r0:r1] = block

    F = np.fft.ifftshift(Fs)
    out = np.fft.ifft2(F).real.astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def image_to_formula_global_fft(img01_rgb: np.ndarray, max_side: int, keep: int) -> Tuple[str, Dict[str, Any]]:
    """
    Convert image -> formula (global FFT).
    Internally we use a square canvas to keep FFT definition simple:
    - resize keep-aspect to max_side, then pad to square (max of dims).
    - store crop info.
    """
    # Make pad_multiple = 1 for global
    padded, shape_meta = resize_keep_aspect_and_pad(img01_rgb, max_side=max_side, pad_multiple=1)

    # pad to square
    H, W, _ = padded.shape
    N = max(H, W)
    if H != W:
        pad_bottom = N - H
        pad_right = N - W
        padded = np.pad(padded, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    else:
        pad_bottom = 0
        pad_right = 0

    keep = int(keep)
    if keep > N:
        keep = N

    blocks = []
    for ch in range(3):
        block = fft2_truncate_center(padded[:, :, ch], keep=keep)
        # store as float16 (real, imag)
        bi = np.stack([block.real, block.imag], axis=-1).astype(np.float16)  # (keep, keep, 2)
        blocks.append(bi)

    coeff = np.stack(blocks, axis=0)  # (3, keep, keep, 2)
    coeff_bytes = coeff.tobytes(order="C")

    meta = {
        "type": "global_fft2_center",
        "version": 2,
        "max_side": int(max_side),
        "keep": int(keep),
        "square_N": int(N),
        "res_h": shape_meta["res_h"],
        "res_w": shape_meta["res_w"],
        "pad_to_square_bottom": int(pad_bottom),
        "pad_to_square_right": int(pad_right),
        "dtype": "float16",
        "coeff_shape": [3, int(keep), int(keep), 2],
        "notes": "Global truncated 2D Fourier center-block coefficients; coefficients stored as float16 real/imag.",
    }

    formula = pack_formula(meta, coeff_bytes)
    return formula, meta


def formula_to_image_global_fft(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    keep = int(meta["keep"])
    N = int(meta["square_N"])
    shape = tuple(meta["coeff_shape"])  # (3, keep, keep, 2)

    coeff = np.frombuffer(coeff_bytes, dtype=np.float16)
    if coeff.size != int(np.prod(shape)):
        raise ValueError("Coefficient payload size mismatch (corrupt formula or wrong type).")
    coeff = coeff.reshape(shape)

    # unpack complex blocks
    out_ch = []
    for ch in range(3):
        real = coeff[ch, :, :, 0].astype(np.float32)
        imag = coeff[ch, :, :, 1].astype(np.float32)
        block = (real + 1j * imag).astype(np.complex64)
        chan = ifft2_from_center_block(N=N, keep=keep, block=block)
        out_ch.append(chan)

    rgb01 = np.stack(out_ch, axis=-1)  # (N,N,3)

    # Crop back to resized (pre-square padding) size
    res_h = int(meta["res_h"])
    res_w = int(meta["res_w"])
    rgb01 = rgb01[:res_h, :res_w, :]

    return to_uint8(rgb01)


# ============================================================
# Novel representation: Spectral Mosaic (tiled windowed Fourier)
# ============================================================

def hann2d(tile: int) -> np.ndarray:
    """
    2D Hann window for overlap-add.
    """
    h = np.hanning(tile).astype(np.float32)
    w2d = np.outer(h, h)
    # avoid near-zero division issues in normalization:
    return np.clip(w2d, 1e-6, 1.0)


def tiled_spectral_mosaic_encode(img01_rgb: np.ndarray, max_side: int, tile: int, overlap: int, keep: int) -> Tuple[str, Dict[str, Any]]:
    """
    Encode image as a sum of local windowed Fourier series:
    - resize keep-aspect to max_side
    - pad to tile grid with overlap-friendly stepping
    - for each tile: multiply by Hann window, take FFT2, store centered keep×keep block
    """
    tile = int(tile)
    overlap = int(overlap)
    keep = int(keep)
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must be in [0, tile-1].")
    step = tile - overlap
    if step <= 0:
        raise ValueError("Invalid overlap; step must be > 0.")

    # Pad to multiples of step and tile for clean tiling
    pad_multiple = step
    padded, shape_meta = resize_keep_aspect_and_pad(img01_rgb, max_side=max_side, pad_multiple=pad_multiple)

    H, W, _ = padded.shape

    # Ensure we can tile to cover fully with tiles of size tile and step stride
    # Add extra padding so last tile fits:
    extra_bottom = (tile - (H - tile) % step) % step if H >= tile else (tile - H)
    extra_right  = (tile - (W - tile) % step) % step if W >= tile else (tile - W)

    padded = np.pad(padded, ((0, extra_bottom), (0, extra_right), (0, 0)), mode="edge")
    Hp, Wp, _ = padded.shape

    if keep > tile:
        keep = tile
    if keep < 1:
        raise ValueError("keep must be >= 1")

    # tile grid
    ys = list(range(0, Hp - tile + 1, step))
    xs = list(range(0, Wp - tile + 1, step))
    Ty, Tx = len(ys), len(xs)

    win = hann2d(tile)

    # coeff storage: (3, Ty, Tx, keep, keep, 2) float16
    coeff = np.zeros((3, Ty, Tx, keep, keep, 2), dtype=np.float16)

    c = tile // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep

    for ch in range(3):
        for iy, y0 in enumerate(ys):
            for ix, x0 in enumerate(xs):
                patch = padded[y0:y0 + tile, x0:x0 + tile, ch].astype(np.float32)
                patch_w = patch * win

                F = np.fft.fft2(patch_w)
                Fs = np.fft.fftshift(F)
                block = Fs[r0:r1, r0:r1].astype(np.complex64)

                coeff[ch, iy, ix, :, :, 0] = block.real.astype(np.float16)
                coeff[ch, iy, ix, :, :, 1] = block.imag.astype(np.float16)

    coeff_bytes = coeff.tobytes(order="C")

    meta = {
        "type": "spectral_mosaic_v1",
        "version": 2,
        "max_side": int(max_side),
        "tile": int(tile),
        "overlap": int(overlap),
        "step": int(step),
        "keep": int(keep),
        "Ty": int(Ty),
        "Tx": int(Tx),
        "pad_h": int(Hp),
        "pad_w": int(Wp),
        "res_h": int(shape_meta["res_h"]),
        "res_w": int(shape_meta["res_w"]),
        "extra_bottom": int(extra_bottom),
        "extra_right": int(extra_right),
        "dtype": "float16",
        "coeff_shape": [3, int(Ty), int(Tx), int(keep), int(keep), 2],
        "notes": "Novel windowed local Fourier expansion (STFT-like). Tiles overlap-add with Hann window.",
    }

    formula = pack_formula(meta, coeff_bytes)
    return formula, meta


def tiled_spectral_mosaic_decode(meta: Dict[str, Any], coeff_bytes: bytes) -> np.ndarray:
    tile = int(meta["tile"])
    overlap = int(meta["overlap"])
    step = int(meta["step"])
    keep = int(meta["keep"])
    Ty = int(meta["Ty"])
    Tx = int(meta["Tx"])
    Hp = int(meta["pad_h"])
    Wp = int(meta["pad_w"])
    res_h = int(meta["res_h"])
    res_w = int(meta["res_w"])

    shape = tuple(meta["coeff_shape"])  # (3, Ty, Tx, keep, keep, 2)
    coeff = np.frombuffer(coeff_bytes, dtype=np.float16)
    if coeff.size != int(np.prod(shape)):
        raise ValueError("Coefficient payload size mismatch (corrupt formula or wrong type).")
    coeff = coeff.reshape(shape)

    ys = list(range(0, Hp - tile + 1, step))
    xs = list(range(0, Wp - tile + 1, step))
    if len(ys) != Ty or len(xs) != Tx:
        raise ValueError("Tile grid mismatch (corrupt meta).")

    win = hann2d(tile)
    win_sum = np.zeros((Hp, Wp), dtype=np.float32)

    # outputs
    out = np.zeros((Hp, Wp, 3), dtype=np.float32)

    c = tile // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep

    for ch in range(3):
        for iy, y0 in enumerate(ys):
            for ix, x0 in enumerate(xs):
                real = coeff[ch, iy, ix, :, :, 0].astype(np.float32)
                imag = coeff[ch, iy, ix, :, :, 1].astype(np.float32)
                block = (real + 1j * imag).astype(np.complex64)

                Fs = np.zeros((tile, tile), dtype=np.complex64)
                Fs[r0:r1, r0:r1] = block
                F = np.fft.ifftshift(Fs)
                patch = np.fft.ifft2(F).real.astype(np.float32)
                patch = np.clip(patch, 0.0, 1.0)

                out[y0:y0 + tile, x0:x0 + tile, ch] += patch * win
                # accumulate weights once (independent of channel)
                if ch == 0:
                    win_sum[y0:y0 + tile, x0:x0 + tile] += win

    # normalize overlap-add
    win_sum = np.clip(win_sum, 1e-6, None)
    out[:, :, 0] /= win_sum
    out[:, :, 1] /= win_sum
    out[:, :, 2] /= win_sum

    # crop to resized shape
    out = out[:res_h, :res_w, :]
    return to_uint8(out)


# ============================================================
# LaTeX explanations (clean + novel)
# ============================================================

def latex_global_fft(keep: int) -> str:
    m = int(keep) // 2
    return (
        r"Global Fourier depiction:"
        "\n"
        rf"\[ f(x,y)\approx \sum_{{u=-{m}}}^{{{m}}}\sum_{{v=-{m}}}^{{{m}}} "
        r"C_{u,v}\,e^{i2\pi(ux+vy)} \]"
        "\n"
        r"The single formula string encodes the complex coefficients $C_{u,v}$ (for each RGB channel)."
    )


def latex_spectral_mosaic(tile: int, keep: int) -> str:
    m = int(keep) // 2
    return (
        r"Novel depiction: **Spectral Mosaic** (windowed local Fourier expansion)."
        "\n"
        r"We tile the image into overlapping patches and express each patch as a truncated Fourier series:"
        "\n"
        rf"\[ f(x,y)\approx \sum_{{p,q}} w_{{p,q}}(x,y)\;"
        rf"\sum_{{u=-{m}}}^{{{m}}}\sum_{{v=-{m}}}^{{{m}}} C_{{p,q,u,v}}\;e^{{i2\pi(u x_{{p,q}} + v y_{{p,q}})}} \]"
        "\n"
        rf"Here $w_{{p,q}}$ is a 2D Hann window over a tile of size {tile}×{tile}, "
        r"and $(x_{p,q},y_{p,q})$ are local tile coordinates."
        "\n"
        r"This representation behaves like a **mathematical stained-glass**: local spectra stitched by overlap-add."
    )


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula (Novel)", layout="wide")
st.title("Image ⇄ Formula — High-Resolution + Novel Spectral Mosaic")

st.markdown(
    "- **Image → Formula**: pick an image → you get ONE formula string.\n"
    "- **Formula → Image**: paste the formula → you get the image.\n\n"
    "**Two math depictions**:\n"
    "1) Global Fourier (classic)\n"
    "2) **Spectral Mosaic** (novel): tiled, windowed local Fourier expansions stitched by overlap-add.\n"
)

mode = st.radio("Choose conversion", ["Image → Formula", "Formula → Image"], horizontal=True)
st.divider()


# ----------------------------
# IMAGE -> FORMULA
# ----------------------------
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
        st.header("Representation")

        rep = st.radio(
            "Math depiction",
            ["Spectral Mosaic (Novel)", "Global Fourier (Classic)"],
            index=0
        )

        st.divider()
        st.header("Quality preset")

        preset = st.selectbox(
            "Preset",
            [
                "Balanced (recommended)",
                "High Fidelity",
                "Extreme (big formula)",
                "Custom"
            ],
            index=0
        )

        # Higher resolution choices
        max_side_options = [256, 384, 512, 768, 1024, 1280, 1536]
        if preset == "Balanced (recommended)":
            max_side = 768
            tile = 96
            overlap = 48
            keep = 21
        elif preset == "High Fidelity":
            max_side = 1024
            tile = 128
            overlap = 64
            keep = 29
        elif preset == "Extreme (big formula)":
            max_side = 1536
            tile = 160
            overlap = 80
            keep = 33
        else:
            max_side = st.select_slider("Max side (preserve aspect)", options=max_side_options, value=768)

            if rep.startswith("Spectral"):
                tile = st.select_slider("Tile size", options=[64, 80, 96, 112, 128, 160, 192, 224, 256], value=96)
                overlap = st.select_slider("Overlap", options=[0, 16, 24, 32, 48, 64, 80, 96, 112, 128], value=48)
                # keep must be <= tile, odd recommended
                keep = st.slider("Frequencies kept per tile (keep×keep)", min_value=5, max_value=min(65, int(tile)), value=21, step=2)
            else:
                # global FFT uses a square canvas N; keep <= N; keep odd recommended
                keep = st.slider("Frequencies kept (keep×keep)", min_value=9, max_value=129, value=33, step=2)
                tile = 0
                overlap = 0

        st.caption("Tip: Spectral Mosaic generally gives better perceptual reconstructions per byte than global FFT.")

    try:
        img01 = load_image_from_choice(source_mode, chosen_local, upload)

        st.subheader("Selected image")
        st.image(to_uint8(img01), use_container_width=True)

        if rep.startswith("Spectral"):
            formula, meta = tiled_spectral_mosaic_encode(
                img01_rgb=img01,
                max_side=int(max_side),
                tile=int(tile),
                overlap=int(overlap),
                keep=int(keep),
            )
        else:
            formula, meta = image_to_formula_global_fft(
                img01_rgb=img01,
                max_side=int(max_side),
                keep=int(keep),
            )

        size_bytes = len(formula.encode("utf-8"))

        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=220)

        a, b, c = st.columns(3)
        a.metric("Formula size", human_size(size_bytes))
        b.metric("Resized image", f"{meta['res_w']}×{meta['res_h']}")
        c.metric("Representation", meta["type"])

        st.subheader("Math formula depiction")
        if meta["type"] == "spectral_mosaic_v1":
            st.latex(latex_spectral_mosaic(tile=int(meta["tile"]), keep=int(meta["keep"])))
        else:
            st.latex(latex_global_fft(keep=int(meta["keep"])))

        # Preview reconstruction
        st.subheader("Reconstructed preview (from the formula)")
        meta2, coeff2 = unpack_formula(formula)
        if meta2["type"] == "spectral_mosaic_v1":
            recon = tiled_spectral_mosaic_decode(meta2, coeff2)
        elif meta2["type"] == "global_fft2_center":
            recon = formula_to_image_global_fft(meta2, coeff2)
        else:
            raise ValueError("Unknown representation in formula.")

        st.image(recon, use_container_width=True)

        st.download_button(
            "Download formula as .txt",
            data=formula.encode("utf-8"),
            file_name="image_formula.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Image → Formula failed: {e}")


# ----------------------------
# FORMULA -> IMAGE
# ----------------------------
else:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=220, placeholder=f"{APP_FORMULA_PREFIX}...")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Reconstruct image", type="primary"):
            try:
                meta, coeff = unpack_formula(formula_in)

                if meta["type"] == "spectral_mosaic_v1":
                    img = tiled_spectral_mosaic_decode(meta, coeff)
                elif meta["type"] == "global_fft2_center":
                    img = formula_to_image_global_fft(meta, coeff)
                else:
                    raise ValueError(f"Unknown representation type: {meta.get('type')}")

                st.success(
                    f"Reconstructed {meta['res_w']}×{meta['res_h']} "
                    f"({meta['type']}, keep={meta['keep']})."
                )
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

                st.subheader("Math formula depiction")
                if meta["type"] == "spectral_mosaic_v1":
                    st.latex(latex_spectral_mosaic(tile=int(meta["tile"]), keep=int(meta["keep"])))
                else:
                    st.latex(latex_global_fft(keep=int(meta["keep"])))

                with st.expander("Show decoded meta"):
                    st.json(meta)

            except Exception as e:
                st.error(f"Formula → Image failed: {e}")

    with colB:
        st.markdown(
            "**Expected format**:\n\n"
            f"- Must start with `{APP_FORMULA_PREFIX}`\n"
            "- Contains compressed meta + coefficient bytes.\n\n"
            "**Tip**: If you want the most novel depiction + best quality per size, use **Spectral Mosaic**."
        )
