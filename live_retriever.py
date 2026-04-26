"""
live_retriever.py — fetches all 8 input channels for the ResUNet model.

Static channels  (cached to data/static_cache/):
  0  altitude       — Copernicus GLO-30 DEM (dem-stitcher)
  1  slope          — derived from DEM
  2  impermeability — ESA WorldCover Built-up class → inverted permeability
  3  vegetation     — ESA WorldCover land-cover classes
  4  water_sources  — OSM waterways → distance transform

Dynamic channels (fetched per date from Open-Meteo archive):
  5  rain_2d        — precipitation 2 days before
  6  rain_1d        — precipitation 1 day before
  7  rain_today     — precipitation on the event date

Usage:
    from live_retriever import fetch_input_tensor, CHANNEL_NAMES
    tensor, raw = fetch_input_tensor("2021-11-08")   # returns (512,512,8), raw dict
"""

import sys
from pathlib import Path

# make feat/dataset modules importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import numpy as np
from datetime import datetime

from config import STATIC_DIR, GRID_SIZE
from download.dem import download_and_process_dem
from download.imperviousness import download_and_process_permeability
from download.landcover import download_and_process_vegetation
from download.water_features import download_and_process_water_distance
from download.precipitation import create_precipitation_channels
from processing.normalize import ChannelNormalizer

CACHE_DIR = Path("data/static_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_NAMES = [
    "Altitude (m)",
    "Slope (°)",
    "Impermeability",
    "Vegetation",
    "Water Sources",
    "Rain -2d (mm)",
    "Rain -1d (mm)",
    "Rain Today (mm)",
]

# ── static layers ─────────────────────────────────────────────────────────────

def _get_static_layers() -> dict[str, np.ndarray]:
    """
    Download (once) and cache the 5 static terrain layers.
    Subsequent calls load from data/static_cache/*.npy.
    """
    cache_files = {
        "altitude":       CACHE_DIR / "altitude.npy",
        "slope":          CACHE_DIR / "slope.npy",
        "impermeability": CACHE_DIR / "impermeability.npy",
        "vegetation":     CACHE_DIR / "vegetation.npy",
        "water":          CACHE_DIR / "water.npy",
    }

    if all(p.exists() for p in cache_files.values()):
        print("[LIVE] Loading static layers from cache...")
        return {k: np.load(v) for k, v in cache_files.items()}

    print("[LIVE] Static cache missing — downloading from APIs...")

    # 1. DEM → altitude + slope
    altitude, slope = download_and_process_dem(STATIC_DIR)

    # 2. WorldCover → impermeability (= 100 − permeability)
    permeability = download_and_process_permeability(STATIC_DIR)
    impermeability = (100.0 - permeability).astype(np.float32)

    # 3. WorldCover → vegetation (land-cover class values)
    vegetation = download_and_process_vegetation(STATIC_DIR).astype(np.float32)

    # 4. OSM waterways → distance transform
    water_dist = download_and_process_water_distance(STATIC_DIR).astype(np.float32)

    layers = {
        "altitude":       altitude.astype(np.float32),
        "slope":          slope.astype(np.float32),
        "impermeability": impermeability,
        "vegetation":     vegetation,
        "water":          water_dist,
    }

    for k, v in layers.items():
        np.save(cache_files[k], v)
        print(f"[LIVE]   cached {k}: shape={v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")

    return layers


# ── dynamic layers (precipitation) ────────────────────────────────────────────

def _get_precipitation(date_str: str) -> dict[str, np.ndarray]:
    """
    Fetch precipitation from Open-Meteo for the given date.
    Historical dates (> 1 day ago) are cached; today is always re-fetched.
    """
    from datetime import date as _date
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date = date.replace(hour=12)

    is_today = date_str == _date.today().isoformat()
    if is_today:
        # always fetch fresh — today's data is still accumulating
        output_dir = None
    else:
        output_dir = CACHE_DIR / f"precip_{date_str.replace('-', '')}"
        output_dir.mkdir(exist_ok=True)

    channels = create_precipitation_channels(date, output_dir)

    return {
        "rain_2d":    channels["precip_day_before"],
        "rain_1d":    channels["precip_yesterday"],
        "rain_today": channels["precip_today"],
    }


# ── normalisation (matches feat/dataset training pipeline) ───────────────────

def _normalize(raw: dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply per-channel normalisation matching processing/normalize.py.
    Returns (512, 512, 8) float32 tensor.
    """
    norm = ChannelNormalizer()

    def _fit_norm(arr, name):
        norm.fit(arr, name)
        return norm.normalize(arr, name)

    # permeability normaliser expects 0-100 range
    permeability = (100.0 - raw["impermeability"]).astype(np.float32)

    channels = np.stack([
        _fit_norm(raw["altitude"],       "altitude"),
        _fit_norm(raw["slope"],          "slope"),
        _fit_norm(permeability,          "permeability"),
        _fit_norm(raw["vegetation"],     "vegetation"),
        _fit_norm(raw["water"],          "water_distance"),
        _fit_norm(raw["rain_2d"],        "precip_day_before"),
        _fit_norm(raw["rain_1d"],        "precip_yesterday"),
        _fit_norm(raw["rain_today"],     "precip_today"),
    ], axis=-1)  # (512, 512, 8)

    return channels.astype(np.float32)


# ── public API ─────────────────────────────────────────────────────────────────

def fetch_input_tensor(date_str: str | None = None):
    """
    Fetch all 8 channels for the given date and return normalised tensor.

    Returns
    -------
    tensor : np.ndarray (512, 512, 8)  ready for model inference
    raw    : dict  un-normalised arrays for plotting
    """
    from datetime import date as _date
    if date_str is None:
        date_str = _date.today().isoformat()

    static  = _get_static_layers()
    dynamic = _get_precipitation(date_str)

    raw = {**static, **dynamic}

    tensor = _normalize(raw)
    return tensor, raw


if __name__ == "__main__":
    print("=== live_retriever test run ===")
    tensor, raw = fetch_input_tensor("2021-11-08")
    print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    for i, name in enumerate(CHANNEL_NAMES):
        ch = tensor[:, :, i]
        print(f"  ch{i} {name:<22} min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}")
