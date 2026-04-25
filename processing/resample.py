"""
Resampling di raster sorgente alla griglia target 512×512 @ 40m UTM.
Wrapper semplificato per le funzioni in utils/geo.py.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRID_SIZE, CRS_UTM
from utils.geo import (
    compute_bbox_utm, get_affine_transform, get_target_profile,
    resample_raster_to_grid, resample_array_to_grid
)


def resample_to_grid(
    input_path: str,
    output_path: str,
    method: str = "bilinear",
    band: int = 1
) -> np.ndarray:
    """
    Ricampiona un file raster alla griglia target.

    Args:
        input_path: percorso al raster sorgente
        output_path: percorso per il raster di output
        method: "bilinear", "nearest", "average", "cubic"
        band: banda da leggere

    Returns:
        np.ndarray (512, 512)
    """
    return resample_raster_to_grid(
        input_path=input_path,
        output_path=output_path,
        method=method,
        band=band
    )


def resample_array(
    data: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str,
    method: str = "bilinear"
) -> np.ndarray:
    """
    Ricampiona un array numpy alla griglia target.

    Args:
        data: array 2D sorgente
        transform: affine transform sorgente
        crs: CRS sorgente
        method: metodo di resampling

    Returns:
        np.ndarray (512, 512)
    """
    return resample_array_to_grid(data, transform, crs, method)


def validate_grid(data: np.ndarray, name: str = "data") -> bool:
    """
    Valida che un array abbia la shape corretta.

    Returns:
        True se valido
    """
    if data.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(
            f"{name}: shape attesa ({GRID_SIZE}, {GRID_SIZE}), "
            f"ottenuta {data.shape}"
        )
    if not np.isfinite(data).all():
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        print(f"[WARN] {name}: {nan_count} NaN, {inf_count} Inf trovati")
    return True
