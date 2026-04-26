"""
Download e processing del CLMS Imperviousness Density Layer.
Produce: permeability.tif sulla griglia 512×512 @ 40m.

Il layer Imperviousness (0-100) viene invertito: permeability = 100 - imperviousness.
Fonte: Copernicus Land Monitoring Service via WMS o file locale.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path
import requests
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STATIC_DIR, GRID_SIZE, PIXEL_SIZE
from utils.geo import (
    compute_bbox_wgs84, compute_bbox_utm,
    get_target_profile, resample_array_to_grid
)


# ─────────────────────────────────────────────
# WMS Endpoint del CLMS (Imperviousness Density 2018)
# ─────────────────────────────────────────────
CLMS_WMS_URL = "https://image.discomap.eea.europa.eu/arcgis/services/GioLandPublic/HRL_ImperviousDensity_2018/MapServer/WMSServer"


def download_imperviousness_wms(
    output_dir: Path = STATIC_DIR,
    width: int = 1024,
    height: int = 1024
) -> Path:
    """
    Sostituito WMS con l'utilizzo del layer ESA WorldCover STAC.
    Il server WMS EEA per il 2018 non è più disponibile (404/400).
    Estraiamo l'impermeabilità traducendo il "Built-up" (class 50).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "worldcover_raw.tif"

    if raw_path.exists():
        print(f"[IMPERV] File landcover di base esistente: {raw_path}")
        return raw_path

    from download.landcover import download_worldcover_stac
    print("[IMPERV] Scaricamento sostitutivo via STAC (ESA WorldCover)...")
    return download_worldcover_stac(output_dir)


def process_permeability(
    imperv_path: Path = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Processa ESA WorldCover in permeabilità (invertito) e ricampiona a 512×512.

    WorldCover Built-up (50) -> 90% Imperviousness -> 10% Permeability.
    WorldCover remaining -> 10% Imperviousness -> 90% Permeability.

    Returns:
        np.ndarray (512, 512) — permeabilità [0, 100]
    """
    output_path = output_dir / "permeability.tif"

    if output_path.exists():
        print(f"[IMPERV] Permeabilità già processata: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if imperv_path is None:
        imperv_path = output_dir / "worldcover_raw.tif"

    print("[IMPERV] Processing permeabilità da WorldCover...")

    with rasterio.open(imperv_path) as src:
        worldcover_data = src.read(1)
        # Map class 50 (Built-up) to high imperviousness (85%), others low (15%)
        imperv_data = np.where(worldcover_data == 50, 85.0, 15.0).astype(np.float32)
        src_transform = src.transform
        src_crs = str(src.crs)
        src_shape = imperv_data.shape

    # Se già alla risoluzione target, usa direttamente
    if src_shape == (GRID_SIZE, GRID_SIZE):
        permeability = 100.0 - imperv_data
    else:
        # Resample con metodo Average (dati continui)
        resampled = resample_array_to_grid(
            imperv_data, src_transform, src_crs, method="average"
        )
        permeability = 100.0 - resampled

    permeability = np.clip(permeability, 0, 100).astype(np.float32)

    # Salva
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(permeability, 1)

    print(f"[IMPERV] Permeabilità salvata: {output_path} — range=[{permeability.min():.1f}, {permeability.max():.1f}]")
    return permeability


def download_and_process_permeability(output_dir: Path = STATIC_DIR) -> np.ndarray:
    """
    Pipeline completa: download Imperviousness → calcolo Permeabilità.

    Returns:
        np.ndarray (512, 512) — permeabilità [0, 100]
    """
    raw_path = download_imperviousness_wms(output_dir)
    return process_permeability(raw_path, output_dir)


if __name__ == "__main__":
    perm = download_and_process_permeability()
    print(f"\nPermeabilità: shape={perm.shape}, dtype={perm.dtype}")
    print(f"Range: [{perm.min():.1f}, {perm.max():.1f}]")
