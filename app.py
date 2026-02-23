import os
import glob
import io
import json
import base64
import zlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Utilities
# ============================================================

APP_FORMULA_PREFIX = "FFTIMG_v1:"  # single-string formula prefix


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


def resize_to_square_rgb(arr01: np.ndarray, N: int) -> np.ndarray:
    """
    Convert to RGB and resize to NxN.
    """
    if arr01.ndim == 2:
        img = Image.fromarray(to_uint8(arr01), mode="L").convert("RGB")
    else:
        img = Image.fromarray(to_uint8(arr01), mode="RGB")

    img = img.resize((N, N), Image.LANCZOS)
    return to_float01(np.array(img))


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


# ============================================================
# Fourier representation (truncated 2D FFT) <-> "single formula"
# ============================================================

def fft_truncate_channel(chan01: np.ndarray, keep: int) -> Tuple[int, int, np.ndarray]:
    """
    chan01: (N,N) float in [0,1]
    keep: keep x keep centered low-freq block in fftshifted domain
    Returns: (N, keep, block_complex keepxkeep)
    """
    N = int(chan01.shape[0])
    keep = int(keep)
    if keep < 1 or keep > N:
        raise ValueError("keep must be between 1 and N.")
    # FFT
    F = np.fft.fft2(chan01)
    Fs = np.fft.fftshift(F)

    c = N // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep
    block = Fs[r0:r1, r0:r1].astype(np.complex64)
    return N, keep, block


def ifft_from_truncated(N: int, keep: int, block: np.ndarray) -> np.ndarray:
    """
    Reconstruct channel from truncated block.
    """
    N = int(N)
    keep = int(keep)
    if block.shape != (keep, keep):
        raise ValueError(f"Coefficient block shape {block.shape} != ({keep},{keep}).")

    Fs = np.zeros((N, N), dtype=np.complex64)
    c = N // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep
    Fs[r0:r1, r0:r1] = block

    F = np.fft.ifftshift(Fs)
    chan = np.fft.ifft2(F).real.astype(np.float32)
    return np.clip(chan, 0.0, 1.0)


def image_to_payload(img01_rgb: np.ndarray, N: int, keep: int) -> Dict[str, Any]:
    """
    Build a compact payload for RGB image resized to NxN.
    """
    imgN = resize_to_square_rgb(img01_rgb, N)

    payload: Dict[str, Any] = {
        "type": "fft2_truncated_rgb",
        "N": int(N),
        "keep": int(keep),
        "R": None,
        "G": None,
        "B": None,
    }

    # Compute blocks for each channel
    _, _, bR = fft_truncate_channel(imgN[:, :, 0], keep)
    _, _, bG = fft_truncate_channel(imgN[:, :, 1], keep)
    _, _, bB = fft_truncate_channel(imgN[:, :, 2], keep)

    # Store as real/imag float32 lists (JSON serializable)
    payload["R"] = {"real": bR.real.astype(np.float32).tolist(), "imag": bR.imag.astype(np.float32).tolist()}
    payload["G"] = {"real": bG.real.astype(np.float32).tolist(), "imag": bG.imag.astype(np.float32).tolist()}
    payload["B"] = {"real": bB.real.astype(np.float32).tolist(), "imag": bB.imag.astype(np.float32).tolist()}

    return payload


def payload_to_image(payload: Dict[str, Any]) -> np.ndarray:
    """
    Return uint8 RGB image (N,N,3) reconstructed.
    """
    if payload.get("type") != "fft2_truncated_rgb":
        raise ValueError("Unsupported formula payload type.")

    N = int(payload["N"])
    keep = int(payload["keep"])

    def unpack(chan_key: str) -> np.ndarray:
        part = payload[chan_key]
        real = np.array(part["real"], dtype=np.float32)
        imag = np.array(part["imag"], dtype=np.float32)
        block = (real + 1j * imag).astype(np.complex64)
        return block

    bR = unpack("R")
    bG = unpack("G")
    bB = unpack("B")

    R = ifft_from_truncated(N, keep, bR)
    G = ifft_from_truncated(N, keep, bG)
    B = ifft_from_truncated(N, keep, bB)

    rgb01 = np.stack([R, G, B], axis=-1)
    return to_uint8(rgb01)


def payload_to_formula(payload: Dict[str, Any]) -> str:
    """
    Single pasteable "formula string" = PREFIX + base64(zlib(json_bytes))
    """
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    b64 = base64.urlsafe_b64encode(comp).decode("ascii")
    return APP_FORMULA_PREFIX + b64


def formula_to_payload(formula: str) -> Dict[str, Any]:
    """
    Parse the single formula string back into payload.
    """
    s = (formula or "").strip()
    if not s.startswith(APP_FORMULA_PREFIX):
        raise ValueError(f"Formula must start with '{APP_FORMULA_PREFIX}'")

    b64 = s[len(APP_FORMULA_PREFIX):].strip()
    try:
        comp = base64.urlsafe_b64decode(b64.encode("ascii"))
        raw = zlib.decompress(comp)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Could not decode formula: {e}")

    return payload


def formula_math_latex(keep: int) -> str:
    """
    A clean, Streamlit-safe LaTeX string describing the single formula.
    (We do not print coefficients; they are embedded in the formula string.)
    """
    m = int(keep) // 2
    # Use raw string without leading indentation (Streamlit latex is picky)
    return (
        r"Let each color channel be a function $f(x,y)$ on $[0,1)\times[0,1)$."
        r" We represent it with a truncated 2D Fourier series:"
        "\n"
        rf"\[ f(x,y)\approx \sum_{{u=-{m}}}^{{{m}}}\sum_{{v=-{m}}}^{{{m}}} "
        r"\Big(a_{u,v}\cos(2\pi(ux+vy)) + b_{u,v}\sin(2\pi(ux+vy))\Big) \]"
        "\n"
        r"The coefficients $(a_{u,v}, b_{u,v})$ are embedded inside the single formula string."
    )


def human_size(num_bytes: int) -> str:
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    return f"{mb:.2f} MB"


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Image ⇄ Formula", layout="wide")
st.title("Image ⇄ Formula (Single-String Fourier Formula)")

st.markdown(
    "This app does a **simple formula conversion**:\n\n"
    "- **Image → Formula**: select an image → get ONE formula string\n"
    "- **Formula → Image**: paste the formula string → reconstruct the image\n\n"
    "The formula is a single encoded string that contains the truncated Fourier coefficients."
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
        st.header("Formula settings")

        N = st.select_slider("Working resolution (N×N)", options=[64, 96, 128, 192, 256, 320, 384, 512], value=256)
        keep_max = min(129, N if N % 2 == 1 else N - 1)
        keep = st.slider("Frequencies kept (keep×keep)", min_value=3, max_value=keep_max, value=min(33, keep_max), step=2)
        st.caption("Higher keep = more detail, larger formula string.")

    try:
        img01 = load_image_from_choice(source_mode, chosen_local, upload)
        st.subheader("Selected image")
        st.image(to_uint8(img01), use_container_width=True)

        payload = image_to_payload(img01, N=int(N), keep=int(keep))
        formula = payload_to_formula(payload)

        # Compute displayed formula size
        size_bytes = len(formula.encode("utf-8"))

        st.subheader("Single formula string")
        st.text_area("Copy/paste this formula:", value=formula, height=200)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Formula size", human_size(size_bytes))
        with c2:
            st.metric("Reconstruction resolution", f"{payload['N']}×{payload['N']}")

        st.subheader("Mathematical meaning (single formula)")
        st.latex(formula_math_latex(int(keep)))

        # Quick sanity check: reconstruct and show
        st.subheader("Reconstructed preview (from the formula)")
        payload2 = formula_to_payload(formula)
        recon = payload_to_image(payload2)
        st.image(recon, use_container_width=True)

        st.download_button(
            "Download formula as .txt",
            data=formula.encode("utf-8"),
            file_name="image_formula.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Image → Formula failed: {e}")

else:
    st.subheader("Paste a formula string to reconstruct the image")
    formula_in = st.text_area("Formula", height=200, placeholder=f"{APP_FORMULA_PREFIX}...")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Reconstruct image", type="primary"):
            try:
                payload = formula_to_payload(formula_in)
                img = payload_to_image(payload)

                st.success(f"Reconstructed {payload['N']}×{payload['N']} image (keep={payload['keep']}).")
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

                st.subheader("Mathematical meaning (single formula)")
                st.latex(formula_math_latex(int(payload["keep"])))

            except Exception as e:
                st.error(f"Formula → Image failed: {e}")

    with colB:
        st.markdown(
            "**Expected format**:\n\n"
            f"- Must start with `{APP_FORMULA_PREFIX}`\n"
            "- Everything after that is encoded coefficient data.\n\n"
            "Tip: If you change **keep** or **N**, you need to regenerate a new formula from the image."
        )
