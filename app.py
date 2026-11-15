"""
Step 4: Streamlit UI for the UHI U-Net.

Loads best_model.pth (on the Apple MPS backend when available), lets the user
upload a sample patch, runs inference, and shows the optical RGB image beside the
predicted land-surface-temperature heatmap ('inferno') with a Celsius colorbar.

Supported uploads:
  * .npz  -> a patch produced by src/preprocess.py (key 'x', 4 x H x W, [0,1])
  * .tif  -> a 4-band GeoTIFF of raw Landsat DN (R, G, B, NIR) -> auto-scaled
  * .png / .jpg -> RGB fallback; a pseudo-NIR 4th channel is synthesised

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

from src import config as C
from src.model import UNet

BEST_MODEL_PATH = C.PROJECT_ROOT / "best_model.pth"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    device = C.get_device()
    if not BEST_MODEL_PATH.exists():
        return None, device, None, None

    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model = UNet(
        in_channels=ckpt.get("in_channels", 4),
        out_channels=1,
        final_activation=ckpt.get("final_activation", "sigmoid"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, ckpt["temp_min"], ckpt["temp_max"]


# --------------------------------------------------------------------------- #
# Upload parsing -> 4-channel float32 tensor in [0, 1]  (R, G, B, NIR)
# --------------------------------------------------------------------------- #
def parse_upload(uploaded) -> np.ndarray:
    name = uploaded.name.lower()
    raw = uploaded.read()

    if name.endswith(".npz"):
        with np.load(io.BytesIO(raw)) as data:
            key = "x" if "x" in data else data.files[0]
            x = data[key].astype(np.float32)
        if x.ndim != 3 or x.shape[0] != 4:
            raise ValueError(f"Expected a [4, H, W] array in the .npz, got {x.shape}.")
        return np.clip(x, 0.0, 1.0)

    if name.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(io.BytesIO(raw)) as src:
            arr = src.read().astype(np.float32)  # [bands, H, W]
        if arr.shape[0] < 4:
            raise ValueError(f"GeoTIFF needs >= 4 bands (R,G,B,NIR), got {arr.shape[0]}.")
        arr = arr[:4]
        # If values look like raw Landsat DN, apply the SR scaling to reach [0,1].
        if arr.max() > 1.5:
            arr = np.clip(arr * C.SR_SCALE + C.SR_OFFSET, 0.0, 1.0)
        return arr.astype(np.float32)

    # RGB fallback (png/jpg): synthesise a pseudo-NIR channel.
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    nir = r  # crude placeholder; true NIR is unavailable from an RGB photo
    x = np.stack([r, g, b, nir], axis=0)
    return np.clip(x, 0.0, 1.0)


def _pad_to_multiple(x: np.ndarray, m: int = 16):
    """Pad [C,H,W] so H and W are multiples of m (U-Net needs this). Returns
    padded array plus the original (H, W) for cropping the output back."""
    _, h, w = x.shape
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return x, (h, w)


def predict(model, device, x: np.ndarray, temp_min, temp_max) -> np.ndarray:
    x_pad, (h, w) = _pad_to_multiple(x, 16)
    tensor = torch.from_numpy(x_pad).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()  # normalised [0,1]
    pred = pred[:h, :w]
    return pred * (temp_max - temp_min) + temp_min     # -> Celsius


def rgb_for_display(x: np.ndarray) -> np.ndarray:
    """Build a brightness-stretched RGB image from channels R,G,B (0,1,2)."""
    rgb = np.transpose(x[:3], (1, 2, 0))               # H, W, 3
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if hi > lo:
        rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return rgb


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="UHI Temperature Prediction", layout="wide")
st.title("🌡️ Urban Heat Island — Land Surface Temperature Prediction")
st.caption("U-Net regression: 4-channel optical (R, G, B, NIR) → surface temperature map")

model, device, temp_min, temp_max = load_model()

with st.sidebar:
    st.header("Input")
    st.write(f"**Device:** `{device}`")
    if model is None:
        st.error("best_model.pth not found. Train the model first (python -m src.train).")
    else:
        st.success(f"Model loaded. Train range: {temp_min:.1f}–{temp_max:.1f} °C")
    uploaded = st.file_uploader(
        "Upload a sample patch", type=["npz", "tif", "tiff", "png", "jpg", "jpeg"]
    )
    st.caption("Best results with a .npz patch or a 4-band GeoTIFF. RGB images use a pseudo-NIR fallback.")

if model is None:
    st.stop()

if uploaded is None:
    st.info("⬅️ Upload a sample image in the sidebar to run inference.")
    st.stop()

try:
    x = parse_upload(uploaded)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the upload: {e}")
    st.stop()

pred_c = predict(model, device, x, temp_min, temp_max)
rgb = rgb_for_display(x)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Optical image (RGB)")
    fig1, ax1 = plt.subplots()
    ax1.imshow(rgb)
    ax1.axis("off")
    st.pyplot(fig1)

with col2:
    st.subheader("Predicted surface temperature")
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(pred_c, cmap="inferno")
    ax2.axis("off")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")
    st.pyplot(fig2)

c1, c2, c3 = st.columns(3)
c1.metric("Min temp", f"{pred_c.min():.1f} °C")
c2.metric("Mean temp", f"{pred_c.mean():.1f} °C")
c3.metric("Max temp", f"{pred_c.max():.1f} °C")
