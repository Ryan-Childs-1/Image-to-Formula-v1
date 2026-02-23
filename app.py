import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import streamlit as st
from PIL import Image


# -----------------------------
# Core math: Image <-> Fourier "function"
# -----------------------------
def _to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_uint8(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0 + 0.5).astype(np.uint8)


def list_local_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    # Show just base names in UI; keep full path internally
    files = sorted(set(files), key=lambda p: os.path.basename(p).lower())
    return files


def load_image_any(source_choice: str, upload_file) -> Tuple[np.ndarray, str]:
    """
    Returns (img_float01, mode_label)
    img_float01: (H,W) or (H,W,3)
    """
    if source_choice == "Upload":
        if upload_file is None:
            raise ValueError("No file uploaded.")
        img = Image.open(upload_file).convert("RGB")
        arr = np.array(img)
        return _to_float01(arr), "RGB"
    else:
        # local path
        img = Image.open(source_choice)
        # preserve alpha? for simplicity convert to RGB
        img = img.convert("RGB")
        arr = np.array(img)
        return _to_float01(arr), "RGB"


def resize_image(arr01: np.ndarray, out_size: int) -> np.ndarray:
    """
    Resizes to (out_size, out_size) using PIL for nice results.
    """
    if arr01.ndim == 2:
        img = Image.fromarray(_to_uint8(arr01), mode="L")
        img = img.resize((out_size, out_size), Image.LANCZOS)
        return _to_float01(np.array(img))
    else:
        img = Image.fromarray(_to_uint8(arr01), mode="RGB")
        img = img.resize((out_size, out_size), Image.LANCZOS)
        return _to_float01(np.array(img))


def fft_truncate_channel(chan01: np.ndarray, keep: int) -> Dict[str, Any]:
    """
    chan01: (N,N) float in [0,1]
    keep: number of low-freq bins to keep along each axis (square keep x keep, centered)
    Returns dict with truncated coefficients packed as real/imag arrays.
    """
    N = chan01.shape[0]
    F = np.fft.fft2(chan01)              # (N,N) complex
    Fs = np.fft.fftshift(F)              # center low freq at middle

    # keep a centered square of size keep x keep
    keep = int(keep)
    if keep < 1 or keep > N:
        raise ValueError("keep must be between 1 and N.")

    c = N // 2
    half = keep // 2
    # if keep is even, we take [c-half : c-half+keep]
    r0 = c - half
    r1 = r0 + keep
    block = Fs[r0:r1, r0:r1]

    return {
        "N": int(N),
        "keep": int(keep),
        "real": block.real.astype(np.float32).tolist(),
        "imag": block.imag.astype(np.float32).tolist(),
    }


def ifft_from_truncated_channel(payload: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct (approximately) a channel from truncated coefficients.
    payload contains N, keep, real, imag.
    Returns (N,N) float in [0,1] (clipped).
    """
    N = int(payload["N"])
    keep = int(payload["keep"])
    real = np.array(payload["real"], dtype=np.float32)
    imag = np.array(payload["imag"], dtype=np.float32)
    block = real + 1j * imag

    if block.shape != (keep, keep):
        raise ValueError(f"Coefficient block shape {block.shape} != ({keep},{keep}).")

    Fs = np.zeros((N, N), dtype=np.complex64)
    c = N // 2
    half = keep // 2
    r0 = c - half
    r1 = r0 + keep
    Fs[r0:r1, r0:r1] = block

    F = np.fft.ifftshift(Fs)
    chan = np.fft.ifft2(F).real  # reconstructed real signal
    # Some ringing/overshoot can happen; clip
    return np.clip(chan.astype(np.float32), 0.0, 1.0)


def image_to_function_payload(img01: np.ndarray, N: int, keep: int) -> Dict[str, Any]:
    """
    img01 is RGB float [0,1] of any size; resize to NxN then compute truncated FFT per channel.
    Returns JSON-serializable payload.
    """
    imgN = resize_image(img01, N)

    if imgN.ndim == 2:
        payload = {
            "type": "fft2_truncated",
            "mode": "L",
            "channels": {
                "L": fft_truncate_channel(imgN, keep),
            },
        }
    else:
        payload = {
            "type": "fft2_truncated",
            "mode": "RGB",
            "channels": {
                "R": fft_truncate_channel(imgN[:, :, 0], keep),
                "G": fft_truncate_channel(imgN[:, :, 1], keep),
                "B": fft_truncate_channel(imgN[:, :, 2], keep),
            },
        }

    payload["notes"] = {
        "meaning": (
            "This is a truncated 2D Fourier-series representation. "
            "The mathematical function is reconstructed by summing these Fourier basis terms. "
            "Truncation keeps only low frequencies for compression/smoothing; "
            "larger keep => sharper reconstruction but larger JSON."
        ),
        "range_xy": "x,y in [0,1)",
    }
    return payload


def function_payload_to_image(payload: Dict[str, Any]) -> np.ndarray:
    """
    Returns uint8 RGB image (N,N,3).
    """
    if payload.get("type") != "fft2_truncated":
        raise ValueError("Unsupported payload type. Expected type='fft2_truncated'.")

    mode = payload.get("mode", "RGB")
    ch = payload.get("channels", {})

    if mode == "L":
        L = ifft_from_truncated_channel(ch["L"])
        rgb = np.stack([L, L, L], axis=-1)
    else:
        R = ifft_from_truncated_channel(ch["R"])
        G = ifft_from_truncated_channel(ch["G"])
        B = ifft_from_truncated_channel(ch["B"])
        rgb = np.stack([R, G, B], axis=-1)

    return _to_uint8(rgb)


def latex_function_description(keep: int) -> str:
    """
    A readable math description (not the full expanded sum with all coefficients).
    """
    m = keep // 2
    # We describe it as a truncated Fourier series on [0,1)x[0,1)
    return rf"""
Let the image channel be a function \(f(x,y)\) on \([0,1)\times[0,1)\).
We approximate it by a truncated 2D Fourier series:

\[
f(x,y)\;\approx\;\sum_{{u=-{m}}}^{{{m}}}\sum_{{v=-{m}}}^{{{m}}}
c_{{u,v}}\;e^{{i2\pi(ux+vy)}}
\]

where \(c_{{u,v}}\) are complex Fourier coefficients (stored in the JSON),
and the reconstructed image is the real part (then clipped to \([0,1]\)).

Increasing **keep** increases the number of retained frequencies, improving detail.
"""


def payload_size_kb(payload: Dict[str, Any]) -> float:
    s = json.dumps(payload)
    return len(s.encode("utf-8")) / 1024.0


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Image ↔ Mathematical Function (2D Fourier)", layout="wide")
st.title("Image ↔ Mathematical Function (Truncated 2D Fourier Series)")

st.markdown(
    """
This app converts an image into a **mathematical function representation** using a truncated **2D Fourier series**,
and can reconstruct an image back from that function.

- **Image → Function**: resize → FFT → keep low-frequency coefficients → export JSON  
- **Function → Image**: read JSON → place coefficients in frequency grid → inverse FFT → display image

All files can live in the **same GitHub folder** as `app.py`. The app will auto-detect images there.
"""
)

with st.sidebar:
    st.header("Inputs")
    folder = "."
    local_images = list_local_images(folder)

    source_mode = st.radio("Image source", ["Local file in repo folder", "Upload"], index=0)

    if source_mode == "Local file in repo folder":
        if not local_images:
            st.warning("No images found in this folder. Add .png/.jpg next to app.py, or switch to Upload.")
            chosen_local = None
        else:
            label_map = {os.path.basename(p): p for p in local_images}
            chosen_name = st.selectbox("Choose a local image", list(label_map.keys()))
            chosen_local = label_map[chosen_name]
        upload = None
        source_choice = chosen_local if chosen_local else "Upload"  # fallback
    else:
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        source_choice = "Upload"

    st.divider()
    st.header("Representation settings")

    N = st.select_slider("Working resolution (N×N)", options=[64, 96, 128, 192, 256, 320, 384, 512], value=256)
    keep = st.slider("Frequencies to keep (keep×keep)", min_value=3, max_value=min(129, N), value=min(33, N), step=2)
    st.caption("Tip: higher **keep** = more detail, bigger JSON. Keep is forced odd by step=2.")

    st.divider()
    st.header("Function JSON input")
    json_mode = st.radio("How to provide a function?", ["Paste JSON", "Upload JSON"], index=0)
    uploaded_json = None
    pasted_json = None
    if json_mode == "Paste JSON":
        pasted_json = st.text_area("Paste function JSON here", height=200, placeholder='{"type":"fft2_truncated", ... }')
    else:
        uploaded_json = st.file_uploader("Upload .json", type=["json"])


colA, colB = st.columns(2, gap="large")

# -----------------------------
# Left: Image -> Function
# -----------------------------
with colA:
    st.subheader("1) Image → Mathematical Function")

    try:
        if source_choice != "Upload":
            img01, mode = load_image_any(source_choice, None)
        else:
            if upload is None:
                st.info("Choose a local image or upload one to generate the function.")
                img01 = None
            else:
                img01, mode = load_image_any("Upload", upload)

        if img01 is not None:
            st.markdown("**Original image**")
            st.image(_to_uint8(img01), use_container_width=True)

            payload = image_to_function_payload(img01, N=N, keep=keep)

            st.markdown("**Mathematical description**")
            st.latex(latex_function_description(keep))

            size_kb = payload_size_kb(payload)
            st.metric("Function JSON size", f"{size_kb:.1f} KB")

            st.markdown("**Preview: reconstruct from the function (sanity check)**")
            recon = function_payload_to_image(payload)
            st.image(recon, use_container_width=True, caption="Reconstruction from truncated Fourier coefficients")

            st.markdown("**Export function JSON**")
            payload_str = json.dumps(payload, indent=2)
            st.download_button(
                "Download function JSON",
                data=payload_str.encode("utf-8"),
                file_name="image_function_fft2_truncated.json",
                mime="application/json",
            )

            with st.expander("Show JSON (truncated display)"):
                st.code(payload_str[:8000] + ("\n...\n" if len(payload_str) > 8000 else ""), language="json")

    except Exception as e:
        st.error(f"Image → Function failed: {e}")


# -----------------------------
# Right: Function -> Image
# -----------------------------
with colB:
    st.subheader("2) Mathematical Function → Image")

    def read_payload_from_ui() -> Optional[Dict[str, Any]]:
        if json_mode == "Paste JSON":
            if not pasted_json or not pasted_json.strip():
                return None
            return json.loads(pasted_json)
        else:
            if uploaded_json is None:
                return None
            raw = uploaded_json.read().decode("utf-8")
            return json.loads(raw)

    payload_in = None
    try:
        payload_in = read_payload_from_ui()
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")

    if payload_in is None:
        st.info("Provide a function JSON (paste or upload) to reconstruct an image.")
    else:
        try:
            st.markdown("**Parsed payload summary**")
            st.json(
                {
                    "type": payload_in.get("type"),
                    "mode": payload_in.get("mode"),
                    "channels": list((payload_in.get("channels") or {}).keys()),
                    "N": next(iter((payload_in.get("channels") or {}).values())).get("N")
                    if payload_in.get("channels")
                    else None,
                    "keep": next(iter((payload_in.get("channels") or {}).values())).get("keep")
                    if payload_in.get("channels")
                    else None,
                }
            )

            img_out = function_payload_to_image(payload_in)
            st.markdown("**Reconstructed image**")
            st.image(img_out, use_container_width=True)

            # Download
            out_pil = Image.fromarray(img_out, mode="RGB")
            import io
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "Download reconstructed PNG",
                data=buf.getvalue(),
                file_name="reconstructed.png",
                mime="image/png",
            )

            st.markdown("**Math description**")
            keep_in = next(iter(payload_in["channels"].values()))["keep"]
            st.latex(latex_function_description(int(keep_in)))

        except Exception as e:
            st.error(f"Function → Image failed: {e}")


st.divider()
st.caption(
    "Notes: This uses a truncated Fourier representation. It’s a true mathematical function basis and is reversible "
    "up to truncation. If you want an *exact* representation, set keep=N (but the JSON becomes huge)."
)
