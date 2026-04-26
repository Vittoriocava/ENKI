import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import threading
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR  = Path("modello")
THRESHOLD  = 0.4

_model      = None
_model_lock = threading.Lock()

CHANNEL_META = [
    ("Altitude (m)",      "altitude",       "terrain",   None, None),
    ("Slope (°)",         "slope",          "YlOrBr",    None, None),
    ("Impermeability",    "impermeability", "RdYlGn_r",  None, None),
    ("Vegetation",        "vegetation",     "YlGn",      None, None),
    ("Water Sources (m)", "water",          "Blues_r",   None, None),
    ("Rain -2d (mm)",     "rain_2d",        "Blues",     None, None),
    ("Rain -1d (mm)",     "rain_1d",        "Blues",     None, None),
    ("Rain Today (mm)",   "rain_today",     "Blues",     None, None),
]


# ── model loading ──────────────────────────────────────────────────────────────

def _stub(*a, **kw):
    pass


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import keras
        with open(MODEL_DIR / "config.json") as f:
            cfg = json.load(f)
        m = keras.models.model_from_json(
            json.dumps(cfg),
            custom_objects={"bce_dice_loss": _stub, "dice_coef": _stub},
        )
        m.load_weights(str(MODEL_DIR / "model.weights.h5"))
        _model = m
    return _model


# ── inference ─────────────────────────────────────────────────────────────────

def run_prediction(tensor):
    """tensor: (512,512,8) float32  →  (512,512) probability map [0,1]."""
    import torch
    model = get_model()
    x = tensor[np.newaxis, ...]      # (1, 512, 512, 8)
    with torch.no_grad():
        y = model(x, training=False)
        y_np = y[0, :, :, 0].detach().cpu().numpy()
    return y_np.astype(np.float32)


# ── visualisation helpers ─────────────────────────────────────────────────────

def _arr_to_b64(arr, cmap, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=80)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── full pipeline ──────────────────────────────────────────────────────────────

def run_full_pipeline(date_str: str | None = None):
    from datetime import date as _date
    from live_retriever import fetch_input_tensor

    if date_str is None:
        date_str = _date.today().isoformat()

    tensor, raw = fetch_input_tensor(date_str)
    y_raw    = run_prediction(tensor)
    y_thresh = np.where(y_raw >= THRESHOLD, y_raw, 0.0).astype(np.float32)

    channel_imgs = []
    for title, key, cmap, vmin, vmax in CHANNEL_META:
        channel_imgs.append({
            "name": title,
            "img":  _arr_to_b64(raw[key], cmap, vmin, vmax, title),
        })

    raw_img    = _arr_to_b64(y_raw,    "hot_r", 0.0, 1.0, "Raw output (sigmoid)")
    thresh_img = _arr_to_b64(y_thresh, "hot_r", 0.0, 1.0, f"Thresholded (≥ {THRESHOLD})")

    return {
        "date":       date_str,
        "threshold":  THRESHOLD,
        "channels":   channel_imgs,
        "raw_img":    raw_img,
        "thresh_img": thresh_img,
        "heatmap":    y_thresh.round(3).tolist(),
        "stats": {
            "max_prob":    round(float(y_raw.max()),  3),
            "mean_prob":   round(float(y_raw.mean()), 4),
            "flooded_pct": round(float((y_thresh > 0).mean() * 100), 2),
        },
    }
