"""
Download e processing del DEM Copernicus GLO-30.
Produce: altitude.tif e slope.tif sulla griglia 512×512 @ 40m.
"""
import numpy as np
import rasterio
from pathlib import Path
from dem_stitcher import stitch_dem

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STATIC_DIR, PIXEL_SIZE, GRID_SIZE
from utils.geo import (
    compute_bbox_wgs84, compute_bbox_utm,
    resample_array_to_grid, get_target_profile
)


def download_dem(output_dir: Path = STATIC_DIR) -> Path:
    """
    Scarica il DEM Copernicus GLO-30 per l'area di Roma.

    Returns:
        Path al file DEM raw scaricato
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "dem_raw.tif"

    if raw_path.exists():
        print(f"[DEM] File già esistente: {raw_path}")
        return raw_path

    # Bounding box in WGS84 per dem-stitcher: [xmin, ymin, xmax, ymax] = [lon_min, lat_min, lon_max, lat_max]
    bbox_wgs = compute_bbox_wgs84()
    bounds = [bbox_wgs[0], bbox_wgs[1], bbox_wgs[2], bbox_wgs[3]]

    print(f"[DEM] Scaricamento Copernicus GLO-30 per bounds: {bounds}")
    X, profile = stitch_dem(
        bounds,
        dem_name="glo_30",
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point"
    )

    # Salva il DEM raw
    with rasterio.open(raw_path, "w", **profile) as dst:
        dst.write(X, 1)

    print(f"[DEM] Salvato DEM raw: {raw_path} — shape={X.shape}")
    return raw_path


def process_altitude(
    dem_raw_path: Path = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Ricampiona il DEM alla griglia target 512×512 @ 40m.

    Returns:
        np.ndarray (512, 512) — altitudine in metri
    """
    if dem_raw_path is None:
        dem_raw_path = output_dir / "dem_raw.tif"

    output_path = output_dir / "altitude.tif"

    if output_path.exists():
        print(f"[DEM] Altitude già processata: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    print("[DEM] Resampling altitudine a 512×512 @ 40m (bilineare)...")
    with rasterio.open(dem_raw_path) as src:
        src_array = src.read(1)
        src_transform = src.transform
        src_crs = str(src.crs)

    altitude = resample_array_to_grid(
        src_array, src_transform, src_crs, method="bilinear"
    )

    # Salva
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(altitude, 1)

    print(f"[DEM] Altitudine salvata: {output_path} — range=[{altitude.min():.1f}, {altitude.max():.1f}]m")
    return altitude


def compute_slope(
    altitude: np.ndarray = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Calcola la pendenza (slope) in gradi dal DEM risamplato.

    La pendenza viene calcolata usando np.gradient sulla griglia 40m.

    Returns:
        np.ndarray (512, 512) — pendenza in gradi
    """
    output_path = output_dir / "slope.tif"

    if output_path.exists():
        print(f"[DEM] Slope già calcolata: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if altitude is None:
        alt_path = output_dir / "altitude.tif"
        with rasterio.open(alt_path) as src:
            altitude = src.read(1)

    print("[DEM] Calcolo pendenza su griglia 40m...")

    # Gradiente in direzione y (Nord-Sud) e x (Est-Ovest)
    dy, dx = np.gradient(altitude, PIXEL_SIZE)

    # Pendenza in gradi
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Salva
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope_deg.astype(np.float32), 1)

    print(f"[DEM] Slope salvata: {output_path} — range=[{slope_deg.min():.2f}, {slope_deg.max():.2f}]°")
    return slope_deg


def download_and_process_dem(output_dir: Path = STATIC_DIR) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completa: download DEM → altitudine → pendenza.

    Returns:
        (altitude, slope) — entrambi np.ndarray (512, 512)
    """
    raw_path = download_dem(output_dir)
    altitude = process_altitude(raw_path, output_dir)
    slope = compute_slope(altitude, output_dir)
    return altitude, slope


if __name__ == "__main__":
    altitude, slope = download_and_process_dem()
    print(f"\nAltitude: shape={altitude.shape}, dtype={altitude.dtype}")
    print(f"Slope:    shape={slope.shape}, dtype={slope.dtype}")
