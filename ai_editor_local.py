import os, io, time, math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# Utilities
# ============================================================

def to_uint8(arr01: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0 + 0.5).astype(np.uint8)

def to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)

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

def pil_bytes(img_u8: np.ndarray, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode="RGB").save(buf, format=fmt, **kw)
    return buf.getvalue()

def resize_keep_aspect_u8(img_u8: np.ndarray, max_side: int) -> np.ndarray:
    H, W, _ = img_u8.shape
    if max(H, W) <= max_side:
        return img_u8
    scale = max_side / float(max(H, W))
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    im = Image.fromarray(img_u8, mode="RGB").resize((newW, newH), Image.LANCZOS)
    return np.array(im, dtype=np.uint8)

def resize_to_match_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Ha, Wa, _ = a_u8.shape
    imB = Image.fromarray(b_u8, mode="RGB").resize((Wa, Ha), Image.LANCZOS)
    return a_u8, np.array(imB, dtype=np.uint8)

def sample_pixels(rgb01_a: np.ndarray, rgb01_b: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H, W, _ = rgb01_a.shape
    total = H * W
    n = min(int(n), total)
    idx = rng.choice(total, size=n, replace=False)
    xa = rgb01_a.reshape(-1, 3)[idx]
    xb = rgb01_b.reshape(-1, 3)[idx]
    return xa.astype(np.float32), xb.astype(np.float32)


# ============================================================
# NumPy "Neural LUT" (tiny MLP) — manual backprop + Adam
# ============================================================

def silu(x: np.ndarray) -> np.ndarray:
    # SiLU(x) = x * sigmoid(x)
    return x / (1.0 + np.exp(-x))

def dsilu_from_x(x: np.ndarray) -> np.ndarray:
    # derivative of SiLU wrt x: sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    s = 1.0 / (1.0 + np.exp(-x))
    return s + x * s * (1.0 - s)

class NumpyNeuralLUT:
    """
    Tiny MLP mapping RGB -> RGB in [0,1].
    Output: y = clamp(x + 0.5 * tanh(net(x)), 0, 1)
    """

    def __init__(self, width: int = 64, depth: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.width = int(width)
        self.depth = int(depth)

        # layers: [ (W,b), ... ] where first in_dim=3, hidden=width, last out=3
        dims = [3] + [self.width] * self.depth + [3]
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            # He-ish init for SiLU
            w = rng.standard_normal((fan_in, fan_out), dtype=np.float32) * np.sqrt(2.0 / max(1, fan_in))
            b = np.zeros((fan_out,), dtype=np.float32)
            self.W.append(w.astype(np.float32))
            self.b.append(b.astype(np.float32))

        # Adam moments
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        x: (B,3) float32 in [0,1]
        Returns y: (B,3), cache for backprop
        """
        x = x.astype(np.float32, copy=False)
        a = x
        z_list = []
        a_list = [a]

        # hidden layers with SiLU
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = silu(z)
            z_list.append(z)
            a_list.append(a)

        # final linear
        z_out = a @ self.W[-1] + self.b[-1]  # (B,3)
        # residual + bounded
        y = x + 0.5 * np.tanh(z_out)
        y = np.clip(y, 0.0, 1.0)

        cache = {
            "x": x,
            "z_list": z_list,
            "a_list": a_list,
            "z_out": z_out,
        }
        return y, cache

    def backward(self, cache: Dict[str, Any], y_pred: np.ndarray, y_tgt: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Manual backprop for MSE loss.
        Returns grads for W,b and scalar loss.
        """
        x = cache["x"]
        z_list = cache["z_list"]
        a_list = cache["a_list"]
        z_out = cache["z_out"]

        B = float(y_pred.shape[0])

        # MSE
        diff = (y_pred - y_tgt).astype(np.float32)
        loss = float(np.mean(diff * diff))

        # dL/dy
        d_y = (2.0 / (B * 3.0)) * diff  # (B,3)

        # y = clip(x + 0.5*tanh(z_out),0,1)
        # Approx backprop through clip by zeroing gradients where saturated
        unclipped = x + 0.5 * np.tanh(z_out)
        sat = (unclipped <= 0.0) | (unclipped >= 1.0)
        d_y = np.where(sat, 0.0, d_y).astype(np.float32)

        # dy/dz_out = 0.5 * (1 - tanh(z_out)^2)
        tnh = np.tanh(z_out).astype(np.float32)
        d_zout = d_y * (0.5 * (1.0 - tnh * tnh))  # (B,3)

        # grads for final layer
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        a_last = a_list[-1]  # (B,width)
        grads_W[-1] = (a_last.T @ d_zout).astype(np.float32)  # (width,3)
        grads_b[-1] = np.sum(d_zout, axis=0).astype(np.float32)

        # backprop into hidden
        d_a = d_zout @ self.W[-1].T  # (B,width)

        for i in reversed(range(len(self.W) - 1)):
            z = z_list[i]
            a_prev = a_list[i]  # input to this layer

            d_z = d_a * dsilu_from_x(z).astype(np.float32)  # (B,width)
            grads_W[i] = (a_prev.T @ d_z).astype(np.float32)
            grads_b[i] = np.sum(d_z, axis=0).astype(np.float32)

            d_a = d_z @ self.W[i].T  # propagate

        return grads_W, grads_b, loss

    def adam_step(self, grads_W: List[np.ndarray], grads_b: List[np.ndarray], lr: float, weight_decay: float):
        """
        AdamW-like update: decay directly on weights.
        """
        self.t += 1
        t = self.t
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = float(lr)
        wd = float(weight_decay)

        for i in range(len(self.W)):
            gW = grads_W[i]
            gb = grads_b[i]

            # Adam moments
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gW * gW)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb * gb)

            # bias correction
            mW_hat = self.mW[i] / (1 - beta1**t)
            vW_hat = self.vW[i] / (1 - beta2**t)
            mb_hat = self.mb[i] / (1 - beta1**t)
            vb_hat = self.vb[i] / (1 - beta2**t)

            # decoupled weight decay
            self.W[i] = (1.0 - lr * wd) * self.W[i] - lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] = self.b[i] - lr * mb_hat / (np.sqrt(vb_hat) + eps)


def train_neural_lut_numpy(
    x_in: np.ndarray,
    y_tgt: np.ndarray,
    width: int,
    depth: int,
    steps: int,
    batch: int,
    lr: float,
    l2: float,
    seed: int = 0,
) -> Tuple[NumpyNeuralLUT, List[float]]:
    rng = np.random.default_rng(seed)
    model = NumpyNeuralLUT(width=int(width), depth=int(depth), seed=seed)

    X = x_in.astype(np.float32, copy=False)
    Y = y_tgt.astype(np.float32, copy=False)

    n = int(X.shape[0])
    batch = max(64, int(batch))
    steps = int(steps)

    losses: List[float] = []
    for t in range(steps):
        idx = rng.integers(0, n, size=batch, endpoint=False)
        xb = X[idx]
        yb = Y[idx]

        pred, cache = model.forward(xb)
        gW, gb, loss = model.backward(cache, pred, yb)
        model.adam_step(gW, gb, lr=float(lr), weight_decay=float(l2))

        if (t % 25) == 0 or t == steps - 1:
            losses.append(loss)

    return model, losses


def apply_lut_numpy(model: NumpyNeuralLUT, img01: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """
    Apply LUT to full image in chunks to avoid RAM spikes.
    """
    H, W, _ = img01.shape
    flat = img01.reshape(-1, 3).astype(np.float32, copy=False)
    out = np.empty_like(flat)

    chunk = max(10_000, int(chunk))
    for i0 in range(0, flat.shape[0], chunk):
        i1 = min(flat.shape[0], i0 + chunk)
        y, _ = model.forward(flat[i0:i1])
        out[i0:i1] = y

    return out.reshape(H, W, 3)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Standalone AI Editor (Neural LUT, NumPy)", layout="wide")
st.title("Standalone AI Editor (Train a Tiny Neural LUT) — NumPy Only (No PyTorch, No API)")

st.markdown(
    "This is a **standalone AI editor** that trains a tiny neural network (implemented from scratch in NumPy) "
    "to learn a **color/lighting transform**.\n\n"
    "**Workflow**:\n"
    "1) Load a base image (from your compressor session or upload)\n"
    "2) Upload a reference-look image\n"
    "3) Train the Neural LUT (CPU; seconds to minutes depending on steps)\n"
    "4) Apply it and optionally send back to compressor session\n\n"
    "Best for: film looks, palette shifts, contrast/lighting mood.\n"
)

st.session_state.setdefault("ai_local_edited_u8", None)

with st.sidebar:
    st.header("Base image")
    base_mode = st.radio("Source", ["From compressor session", "Upload"], index=0)

    base_u8: Optional[np.ndarray] = None
    if base_mode == "From compressor session":
        sess_img = st.session_state.get("last_recon_u8", None)
        if sess_img is None:
            st.warning("No image in session_state['last_recon_u8']. Run your compressor app first.")
        else:
            base_u8 = np.array(sess_img, dtype=np.uint8)
    else:
        up = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg", "webp", "bmp"])
        if up is not None:
            with Image.open(up) as im:
                im = im.convert("RGB")
                base_u8 = np.array(im, dtype=np.uint8)

    st.divider()
    st.header("Reference look")
    ref_up = st.file_uploader("Upload reference look image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    ref_u8: Optional[np.ndarray] = None
    if ref_up is not None:
        with Image.open(ref_up) as im:
            im = im.convert("RGB")
            ref_u8 = np.array(im, dtype=np.uint8)

    st.divider()
    st.header("Training settings (CPU-safe)")
    train_max_side = st.select_slider("Train resolution max side", options=[256, 384, 512, 768, 1024], value=512)
    sample_n = st.select_slider("Training samples (pixels)", options=[20_000, 50_000, 100_000, 200_000, 400_000], value=100_000)
    steps = st.select_slider("Steps", options=[200, 400, 800, 1200, 2000, 4000], value=800)
    batch = st.select_slider("Batch", options=[256, 512, 1024, 2048, 4096], value=2048)
    lr = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    width = st.select_slider("Model width", options=[24, 32, 48, 64, 96, 128], value=64)
    depth = st.select_slider("Model depth", options=[2, 3, 4, 5, 6], value=4)
    l2 = st.select_slider("Weight decay (stability)", options=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)

    st.divider()
    st.header("Apply settings")
    apply_chunk = st.select_slider("Apply chunk size (RAM safety)", options=[50_000, 100_000, 200_000, 400_000], value=200_000)

# Main display
if base_u8 is None:
    st.stop()

st.subheader("Base image")
st.image(base_u8, use_container_width=True)

if ref_u8 is None:
    st.info("Upload a reference-look image to train the AI.")
    st.stop()

st.subheader("Reference look (will be resized to base size for learning)")
st.image(ref_u8, use_container_width=True)

colA, colB, colC = st.columns([1, 1, 1])
train_btn = colA.button("Train AI LUT", type="primary")
apply_btn = colB.button("Apply LUT to full image")
push_btn = colC.button("Send edited image to compressor")

model_key = "ai_local_lut_model_numpy"
st.session_state.setdefault(model_key, None)

if train_btn:
    try:
        with st.spinner("Preparing training data…"):
            base_train = resize_keep_aspect_u8(base_u8, max_side=int(train_max_side))
            ref_train  = resize_keep_aspect_u8(ref_u8,  max_side=int(train_max_side))
            base_train, ref_train = resize_to_match_u8(base_train, ref_train)

            a01 = to_float01(base_train)
            b01 = to_float01(ref_train)

            x_in, y_tgt = sample_pixels(a01, b01, n=int(sample_n), seed=0)

        with st.spinner("Training Neural LUT (NumPy)…"):
            t0 = time.time()
            model, loss_trace = train_neural_lut_numpy(
                x_in=x_in,
                y_tgt=y_tgt,
                width=int(width),
                depth=int(depth),
                steps=int(steps),
                batch=int(batch),
                lr=float(lr),
                l2=float(l2),
                seed=0,
            )
            dt = time.time() - t0

        st.session_state[model_key] = model
        st.success(f"Trained in {dt:.2f}s. Last loss: {loss_trace[-1]:.6f}")
        st.line_chart(loss_trace)

    except Exception as e:
        st.error(f"Training failed: {e}")

if apply_btn:
    model = st.session_state.get(model_key, None)
    if model is None:
        st.error("Train the LUT first.")
    else:
        try:
            with st.spinner("Applying LUT to full image…"):
                base01_full = to_float01(base_u8)
                out01 = apply_lut_numpy(model, base01_full, chunk=int(apply_chunk))
                out_u8 = to_uint8(out01)

            st.session_state["ai_local_edited_u8"] = out_u8
            st.success("Applied LUT.")
        except Exception as e:
            st.error(f"Apply failed: {e}")

edited_u8 = st.session_state.get("ai_local_edited_u8", None)
if edited_u8 is not None:
    st.divider()
    st.subheader("Edited output")
    st.image(edited_u8, use_container_width=True)

    # Size comparisons (approx)
    base_png = pil_bytes(base_u8, "PNG", optimize=True)
    edited_png = pil_bytes(edited_u8, "PNG", optimize=True)
    edited_webp = pil_bytes(edited_u8, "WEBP", quality=85, method=6)

    c1, c2, c3 = st.columns(3)
    c1.metric("Base PNG (approx)", human_size(len(base_png)))
    c2.metric("Edited PNG", human_size(len(edited_png)))
    c3.metric("Edited WEBP (85)", human_size(len(edited_webp)))

    d1, d2 = st.columns(2)
    d1.download_button("Download edited PNG", data=edited_png, file_name="ai_local_edited.png", mime="image/png")
    d2.download_button("Download edited WEBP", data=edited_webp, file_name="ai_local_edited.webp", mime="image/webp")

if push_btn:
    if edited_u8 is None:
        st.error("No edited image yet. Apply the LUT first.")
    else:
        # This is the key your compressor uses
        st.session_state["last_recon_u8"] = edited_u8
        st.success("Sent edited image to compressor session_state['last_recon_u8']. Now open the compressor and generate a new formula.")
