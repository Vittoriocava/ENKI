"""
Download e processing ESA WorldCover v200 (land cover / vegetazione).
Produce: vegetation.tif sulla griglia 512×512 @ 40m.

Fonte: ESA WorldCover via Microsoft Planetary Computer STAC.
Resampling: Nearest Neighbor (dati categorici).
"""
import numpy as np
import rasterio
from rasterio.crs import CRS
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STATIC_DIR, GRID_SIZE
from utils.geo import (
    compute_bbox_wgs84, resample_array_to_grid, get_target_profile
)

# ─────────────────────────────────────────────
# Classi ESA WorldCover v200
# ─────────────────────────────────────────────
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def download_worldcover_stac(output_dir: Path = STATIC_DIR) -> Path:
    """
    Scarica ESA WorldCover via Microsoft Planetary Computer STAC.

    Returns:
        Path al file scaricato
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "worldcover_raw.tif"

    if raw_path.exists():
        print(f"[LANDCOVER] File già esistente: {raw_path}")
        return raw_path

    try:
        import planetary_computer
        from pystac_client import Client
        import rioxarray

        print("[LANDCOVER] Connessione a Planetary Computer STAC...")
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        bbox_wgs = compute_bbox_wgs84()
        bbox = [bbox_wgs[0], bbox_wgs[1], bbox_wgs[2], bbox_wgs[3]]

        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        )

        items = list(search.items())
        if not items:
            print("[LANDCOVER] Nessun item trovato, fallback a dati sintetici")
            return _generate_synthetic_landcover(output_dir)

        print(f"[LANDCOVER] Trovati {len(items)} tile(s)")

        # Carica il primo item con rioxarray
        item = items[0]
        signed_item = planetary_computer.sign(item)
        asset_href = signed_item.assets["map"].href

        import xarray as xr
        ds = rioxarray.open_rasterio(asset_href)

        # Clip al bounding box
        ds_clipped = ds.rio.clip_box(
            minx=bbox[0], miny=bbox[1],
            maxx=bbox[2], maxy=bbox[3]
        )

        # Salva come GeoTIFF
        ds_clipped.rio.to_raster(str(raw_path))
        print(f"[LANDCOVER] Salvato: {raw_path}")
        return raw_path

    except ImportError as e:
        print(f"[LANDCOVER] Librerie STAC non disponibili: {e}")
        print("[LANDCOVER] Generazione dati sintetici come fallback...")
        return _generate_synthetic_landcover(output_dir)
    except Exception as e:
        print(f"[LANDCOVER] Errore download STAC: {e}")
        print("[LANDCOVER] Generazione dati sintetici come fallback...")
        return _generate_synthetic_landcover(output_dir)


def _generate_synthetic_landcover(output_dir: Path) -> Path:
    """
    Genera dati sintetici di land cover realistici per Roma.
    - Centro: Built-up (50)
    - Media distanza: Cropland (40), Grassland (30)
    - Periferia: Tree cover (10), Shrubland (20)
    - Tevere: Water (80)
    """
    raw_path = output_dir / "worldcover_raw.tif"

    y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    norm_dist = dist / max_dist

    lc = np.full((GRID_SIZE, GRID_SIZE), 40, dtype=np.uint8)  # Default: Cropland

    # Centro urbano
    lc[norm_dist < 0.25] = 50  # Built-up
    # Transizione
    mask_trans = (norm_dist >= 0.25) & (norm_dist < 0.45)
    lc[mask_trans] = np.where(
        np.random.random(mask_trans.sum()) > 0.4, 50, 30
    ).astype(np.uint8)
    # Periferia verde
    lc[norm_dist >= 0.6] = np.where(
        np.random.random((norm_dist >= 0.6).sum()) > 0.5, 10, 20
    ).astype(np.uint8)

    # Tevere (linea diagonale)
    river_mask = np.abs(x - y - 30) < 4
    lc[river_mask] = 80  # Water

    profile = get_target_profile()
    profile["dtype"] = "uint8"
    with rasterio.open(raw_path, "w", **profile) as dst:
        dst.write(lc, 1)

    print(f"[LANDCOVER] Generati dati sintetici: {raw_path}")
    return raw_path


def process_vegetation(
    raw_path: Path = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Ricampiona ESA WorldCover alla griglia 512×512 @ 40m (Nearest Neighbor).

    Returns:
        np.ndarray (512, 512) — classi land cover (uint8)
    """
    output_path = output_dir / "vegetation.tif"

    if output_path.exists():
        print(f"[LANDCOVER] Vegetazione già processata: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if raw_path is None:
        raw_path = output_dir / "worldcover_raw.tif"

    print("[LANDCOVER] Resampling vegetazione a 512×512 @ 40m (nearest)...")

    with rasterio.open(raw_path) as src:
        src_array = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = str(src.crs)
        src_shape = src_array.shape

    if src_shape == (GRID_SIZE, GRID_SIZE):
        vegetation = src_array
    else:
        vegetation = resample_array_to_grid(
            src_array, src_transform, src_crs, method="nearest"
        )

    vegetation = vegetation.astype(np.float32)

    # Salva
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(vegetation, 1)

    unique_classes = np.unique(vegetation[vegetation > 0])
    print(f"[LANDCOVER] Vegetazione salvata: {output_path}")
    print(f"[LANDCOVER] Classi presenti: {unique_classes}")
    return vegetation


def download_and_process_vegetation(output_dir: Path = STATIC_DIR) -> np.ndarray:
    """Pipeline completa: download → resampling."""
    raw_path = download_worldcover_stac(output_dir)
    return process_vegetation(raw_path, output_dir)


if __name__ == "__main__":
    veg = download_and_process_vegetation()
    print(f"\nVegetazione: shape={veg.shape}, dtype={veg.dtype}")
    print(f"Classi: {np.unique(veg)}")
