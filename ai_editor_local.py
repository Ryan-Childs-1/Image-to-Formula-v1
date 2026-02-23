import os, io, time, math
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# Standalone "basic AI" training
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

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
    # Resize reference to base dimensions (keeps style learning stable)
    Ha, Wa, _ = a_u8.shape
    imB = Image.fromarray(b_u8, mode="RGB").resize((Wa, Ha), Image.LANCZOS)
    return a_u8, np.array(imB, dtype=np.uint8)

def sample_pixels(rgb01_a: np.ndarray, rgb01_b: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # sample corresponding pixels from A and B (same H,W)
    rng = np.random.default_rng(seed)
    H, W, _ = rgb01_a.shape
    total = H * W
    n = min(int(n), total)
    idx = rng.choice(total, size=n, replace=False)
    xa = rgb01_a.reshape(-1, 3)[idx]
    xb = rgb01_b.reshape(-1, 3)[idx]
    return xa, xb


# -----------------------------
# Tiny "Neural LUT" model
# -----------------------------
class NeuralLUT(nn.Module):
    """
    A tiny MLP that maps RGB -> RGB.
    This acts like a learned smooth 3D LUT (color grading).
    """
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = 3
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x in [0,1]
        y = self.net(x)
        # Residual helps preserve structure + stabilizes training
        y = x + 0.5 * torch.tanh(y)
        return torch.clamp(y, 0.0, 1.0)


def train_neural_lut(
    x_in: np.ndarray,
    y_tgt: np.ndarray,
    width: int,
    depth: int,
    steps: int,
    batch: int,
    lr: float,
    l2: float,
    device: str,
) -> Tuple[NeuralLUT, list]:
    torch.manual_seed(0)
    model = NeuralLUT(width=width, depth=depth).to(device)

    X = torch.from_numpy(x_in.astype(np.float32)).to(device)
    Y = torch.from_numpy(y_tgt.astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    losses = []

    n = X.shape[0]
    batch = max(64, int(batch))

    model.train()
    for t in range(int(steps)):
        # random mini-batch
        idx = torch.randint(0, n, (batch,), device=device)
        xb = X[idx]
        yb = Y[idx]

        pred = model(xb)
        loss = F.mse_loss(pred, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (t % 25) == 0 or t == steps - 1:
            losses.append(float(loss.detach().cpu().item()))
    model.eval()
    return model, losses


def apply_lut(model: NeuralLUT, img01: np.ndarray, device: str, chunk: int = 200_000) -> np.ndarray:
    """
    Apply LUT to full image in chunks to avoid RAM spikes.
    """
    H, W, _ = img01.shape
    flat = img01.reshape(-1, 3).astype(np.float32)
    out = np.empty_like(flat)

    with torch.no_grad():
        for i0 in range(0, flat.shape[0], chunk):
            i1 = min(flat.shape[0], i0 + chunk)
            xb = torch.from_numpy(flat[i0:i1]).to(device)
            yb = model(xb).detach().cpu().numpy()
            out[i0:i1] = yb
    return out.reshape(H, W, 3)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Standalone AI Editor (Neural LUT)", layout="wide")
st.title("Standalone AI Editor (Train a Tiny Neural LUT) — No API, No Big Models")

st.markdown(
    "This is a **standalone AI editor** that trains a tiny neural network to learn a **color/lighting transform**.\n\n"
    "**Workflow**:\n"
    "1) Load a base image (from your compressor session or upload)\n"
    "2) Upload a reference-look image\n"
    "3) Train the Neural LUT (seconds)\n"
    "4) Apply it and send back to compressor\n\n"
    "This is best for: film looks, palette shifts, contrast/lighting mood.\n"
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
    st.header("Training settings (safe)")
    train_max_side = st.select_slider("Train resolution max side", options=[256, 384, 512, 768, 1024], value=512)
    sample_n = st.select_slider("Training samples (pixels)", options=[20_000, 50_000, 100_000, 200_000, 400_000], value=100_000)
    steps = st.select_slider("Steps", options=[200, 400, 800, 1200, 2000], value=800)
    batch = st.select_slider("Batch", options=[256, 512, 1024, 2048, 4096], value=2048)
    lr = st.select_slider("Learning rate", options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    width = st.select_slider("Model width", options=[32, 48, 64, 96, 128], value=64)
    depth = st.select_slider("Model depth", options=[2, 3, 4, 5, 6], value=4)
    l2 = st.select_slider("Weight decay (stability)", options=[0.0, 1e-6, 1e-5, 1e-4, 1e-3], value=1e-4)

    st.divider()
    st.header("Apply settings")
    apply_chunk = st.select_slider("Apply chunk size (RAM safety)", options=[50_000, 100_000, 200_000, 400_000], value=200_000)

    st.divider()
    st.header("Device")
    use_cuda = st.checkbox("Use GPU if available", value=True)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    st.caption(f"Using: {device}")

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

# Train button
colA, colB, colC = st.columns([1, 1, 1])
train_btn = colA.button("Train AI LUT", type="primary")
apply_btn = colB.button("Apply LUT to full image")
push_btn = colC.button("Send edited image to compressor")

model_key = "ai_local_lut_model"
st.session_state.setdefault(model_key, None)

if train_btn:
    try:
        with st.spinner("Preparing training data…"):
            # downscale both for training stability/speed
            base_train = resize_keep_aspect_u8(base_u8, max_side=int(train_max_side))
            ref_train  = resize_keep_aspect_u8(ref_u8,  max_side=int(train_max_side))
            base_train, ref_train = resize_to_match_u8(base_train, ref_train)

            a01 = to_float01(base_train)
            b01 = to_float01(ref_train)

            x_in, y_tgt = sample_pixels(a01, b01, n=int(sample_n), seed=0)

        with st.spinner("Training Neural LUT…"):
            model, loss_trace = train_neural_lut(
                x_in=x_in,
                y_tgt=y_tgt,
                width=int(width),
                depth=int(depth),
                steps=int(steps),
                batch=int(batch),
                lr=float(lr),
                l2=float(l2),
                device=device,
            )

        st.session_state[model_key] = model
        st.success(f"Trained. Last loss: {loss_trace[-1]:.6f}")

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
                out01 = apply_lut(model, base01_full, device=device, chunk=int(apply_chunk))
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

    # Show size comparisons
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
