"""
Utility geospaziali: bounding box, trasformazioni CRS, affine transforms.
"""
import numpy as np
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import rasterio
from rasterio.warp import reproject, Resampling

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ROME_LAT, ROME_LON, PIXEL_SIZE, GRID_SIZE, AREA_SIZE,
    CRS_WGS84, CRS_UTM33N
)


def latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """Converte coordinate WGS84 (lat, lon) in UTM 33N (easting, northing)."""
    transformer = Transformer.from_crs(CRS_WGS84, CRS_UTM33N, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def utm_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    """Converte coordinate UTM 33N in WGS84 (lat, lon)."""
    transformer = Transformer.from_crs(CRS_UTM33N, CRS_WGS84, always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def compute_bbox_utm(
    lat: float = ROME_LAT,
    lon: float = ROME_LON,
    area_size: float = AREA_SIZE
) -> tuple[float, float, float, float]:
    """
    Calcola bounding box in UTM 33N centrato su (lat, lon).

    Returns:
        (xmin, ymin, xmax, ymax) in metri UTM 33N
    """
    cx, cy = latlon_to_utm(lat, lon)
    half = area_size / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def compute_bbox_wgs84(
    lat: float = ROME_LAT,
    lon: float = ROME_LON,
    area_size: float = AREA_SIZE
) -> tuple[float, float, float, float]:
    """
    Calcola bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max).

    Returns:
        (lon_min, lat_min, lon_max, lat_max) in gradi
    """
    xmin, ymin, xmax, ymax = compute_bbox_utm(lat, lon, area_size)
    lat_min, lon_min = utm_to_latlon(xmin, ymin)
    lat_max, lon_max = utm_to_latlon(xmax, ymax)
    # Nota: lon_min, lat_min è l'angolo SW; lon_max, lat_max è l'angolo NE
    return (
        min(lon_min, lon_max),
        min(lat_min, lat_max),
        max(lon_min, lon_max),
        max(lat_min, lat_max)
    )


def get_affine_transform(
    bbox_utm: tuple[float, float, float, float] = None,
    grid_size: int = GRID_SIZE
) -> rasterio.transform.Affine:
    """
    Crea trasformazione affine per la griglia target.

    Args:
        bbox_utm: (xmin, ymin, xmax, ymax) in UTM
        grid_size: numero di pixel per lato

    Returns:
        rasterio Affine transform
    """
    if bbox_utm is None:
        bbox_utm = compute_bbox_utm()
    xmin, ymin, xmax, ymax = bbox_utm
    return from_bounds(xmin, ymin, xmax, ymax, grid_size, grid_size)


def get_target_profile(
    bbox_utm: tuple[float, float, float, float] = None,
    grid_size: int = GRID_SIZE,
    dtype: str = "float32",
    count: int = 1
) -> dict:
    """
    Genera un profilo rasterio completo per la griglia target.

    Returns:
        dict compatibile con rasterio.open(**profile)
    """
    if bbox_utm is None:
        bbox_utm = compute_bbox_utm()

    return {
        "driver": "GTiff",
        "dtype": dtype,
        "width": grid_size,
        "height": grid_size,
        "count": count,
        "crs": CRS.from_epsg(32633),
        "transform": get_affine_transform(bbox_utm, grid_size),
        "nodata": None,
    }


def _parse_crs(crs_input) -> CRS:
    """
    Converte un input CRS in un oggetto rasterio CRS in modo robusto.
    Gestisce CRS objects, stringhe EPSG, e WKT.
    """
    if isinstance(crs_input, CRS):
        return crs_input
    if hasattr(crs_input, 'to_epsg'):
        # È un oggetto CRS-like (pyproj, rasterio)
        epsg = crs_input.to_epsg()
        if epsg:
            return CRS.from_epsg(epsg)
        # Fallback: prova a costruire dal WKT
        try:
            return CRS.from_wkt(crs_input.to_wkt())
        except Exception:
            return CRS(crs_input)
    if isinstance(crs_input, str):
        # Prova prima come EPSG string (es. "EPSG:4326")
        crs_input = crs_input.strip()
        if crs_input.upper().startswith("EPSG:"):
            try:
                epsg_code = int(crs_input.split(":")[1])
                return CRS.from_epsg(epsg_code)
            except (ValueError, IndexError):
                pass
        # Fallback generico
        try:
            return CRS.from_user_input(crs_input)
        except Exception:
            # Ultimo tentativo: interpreto come WKT
            return CRS.from_wkt(crs_input)
    # Tipo sconosciuto — prova comunque
    return CRS.from_user_input(crs_input)


def resample_array_to_grid(
    src_array: np.ndarray,
    src_transform: rasterio.transform.Affine,
    src_crs,
    method: str = "bilinear",
    bbox_utm: tuple = None,
    grid_size: int = GRID_SIZE
) -> np.ndarray:
    """
    Reproietta e ricampiona un array alla griglia target 512x512 @ 40m UTM33N.

    Args:
        src_array: array 2D sorgente
        src_transform: trasformazione affine sorgente
        src_crs: CRS sorgente (str, rasterio.crs.CRS, o pyproj.CRS)
        method: "bilinear", "nearest", "average"
        bbox_utm: bounding box target in UTM
        grid_size: dimensione griglia target

    Returns:
        np.ndarray di shape (grid_size, grid_size)
    """
    resampling_methods = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "average": Resampling.average,
        "cubic": Resampling.cubic,
    }

    if bbox_utm is None:
        bbox_utm = compute_bbox_utm()

    dst_transform = get_affine_transform(bbox_utm, grid_size)
    dst_crs = CRS.from_epsg(32633)

    dst_array = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Assicura che src_array sia 2D
    if src_array.ndim == 1:
        raise ValueError("src_array deve essere almeno 2D")

    parsed_src_crs = _parse_crs(src_crs)

    reproject(
        source=src_array.astype(np.float32),
        destination=dst_array,
        src_transform=src_transform,
        src_crs=parsed_src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling_methods.get(method, Resampling.bilinear),
    )

    return dst_array


def resample_raster_to_grid(
    input_path: str,
    output_path: str = None,
    method: str = "bilinear",
    band: int = 1,
    bbox_utm: tuple = None
) -> np.ndarray:
    """
    Legge un raster GeoTIFF e lo ricampiona alla griglia target.

    Args:
        input_path: percorso al raster sorgente
        output_path: se fornito, salva il risultato come GeoTIFF
        method: metodo di resampling
        band: banda da leggere
        bbox_utm: bounding box target

    Returns:
        np.ndarray (GRID_SIZE, GRID_SIZE)
    """
    with rasterio.open(input_path) as src:
        src_array = src.read(band)
        src_transform = src.transform
        src_crs = str(src.crs)

    result = resample_array_to_grid(
        src_array, src_transform, src_crs, method, bbox_utm
    )

    if output_path:
        profile = get_target_profile(bbox_utm)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(result, 1)

    return result


# ─────────────────────────────────────────────
# Info di debug
# ─────────────────────────────────────────────
if __name__ == "__main__":
    bbox_utm = compute_bbox_utm()
    bbox_wgs = compute_bbox_wgs84()
    print(f"Centro Roma UTM33N: {latlon_to_utm(ROME_LAT, ROME_LON)}")
    print(f"BBox UTM33N: {bbox_utm}")
    print(f"BBox WGS84:  {bbox_wgs}")
    print(f"Area: {AREA_SIZE}m × {AREA_SIZE}m = {AREA_SIZE/1000:.1f}km × {AREA_SIZE/1000:.1f}km")
    print(f"Griglia: {GRID_SIZE}×{GRID_SIZE} @ {PIXEL_SIZE}m/pixel")
    print(f"Affine Transform:\n{get_affine_transform(bbox_utm)}")
