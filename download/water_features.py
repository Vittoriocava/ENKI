"""
Download features idriche da OpenStreetMap e calcolo Distance Transform.
Produce: water_distance.tif sulla griglia 512×512 @ 40m.

Workflow:
1. Query Overpass API via osmnx per water/waterway features
2. Rasterizzazione su griglia 512×512
3. Distance Transform (distanza euclidea da ogni pixel alla fonte idrica più vicina)
"""
import numpy as np
import rasterio
from rasterio import features as rfeatures
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from shapely.geometry import box

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STATIC_DIR, GRID_SIZE, PIXEL_SIZE
from utils.geo import (
    compute_bbox_wgs84, compute_bbox_utm, get_target_profile,
    get_affine_transform
)


def download_water_features(output_dir: Path = STATIC_DIR) -> Path:
    """
    Scarica features idriche da OpenStreetMap per l'area di Roma.
    Usa direttamente l'Overpass API per evitare le limitazioni di osmnx
    sulle aree grandi.

    Tags cercati:
    - natural=water (laghi, bacini)
    - waterway=* (fiumi, canali, torrenti)
    - natural=wetland

    Returns:
        Path al file GeoJSON salvato
    """
    import requests as req
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "water_features.geojson"

    if geojson_path.exists():
        print(f"[WATER] File già esistente: {geojson_path}")
        return geojson_path

    bbox_wgs = compute_bbox_wgs84()  # (lon_min, lat_min, lon_max, lat_max)
    south, west = bbox_wgs[1], bbox_wgs[0]
    north, east = bbox_wgs[3], bbox_wgs[2]

    print(f"[WATER] Scaricamento features idriche da Overpass API...")
    print(f"[WATER] BBox: S={south:.4f}, W={west:.4f}, N={north:.4f}, E={east:.4f}")

    # Query Overpass diretta — una singola richiesta per tutti i tipi
    bbox_str = f"{south},{west},{north},{east}"
    query = (
        f'[out:json][timeout:120];'
        f'(way["natural"="water"]({bbox_str});'
        f'relation["natural"="water"]({bbox_str});'
        f'way["waterway"]({bbox_str});'
        f'way["natural"="wetland"]({bbox_str});'
        f'relation["natural"="wetland"]({bbox_str}););'
        f'out geom;'
    )

    headers = {
        "User-Agent": "EnkiSwarm-FloodDataset/1.0",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # Prova più endpoint Overpass (mirror)
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    try:
        print("[WATER] Invio query Overpass (può richiedere ~30s)...")
        response = None
        for url in overpass_urls:
            try:
                # POST con data URL-encoded
                response = req.post(
                    url, data=f"data={query}",
                    headers=headers, timeout=180
                )
                if response.status_code == 200:
                    break
                print(f"[WATER]   {url}: status {response.status_code}, provo il prossimo...")
            except req.RequestException:
                continue

        if response is None or response.status_code != 200:
            raise req.RequestException(f"Tutti gli endpoint Overpass hanno fallito")

        data = response.json()

        elements = data.get("elements", [])
        print(f"[WATER] Ricevuti {len(elements)} elementi da Overpass")

        if not elements:
            print("[WATER] Nessun elemento trovato, generazione sintetica...")
            return _generate_synthetic_water(output_dir)

        # Converti Overpass JSON → GeoJSON
        features = []
        for elem in elements:
            geom = _overpass_element_to_geojson(elem)
            if geom is not None:
                features.append({
                    "type": "Feature",
                    "properties": {},
                    "geometry": geom,
                })

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        with open(geojson_path, "w") as f:
            json.dump(geojson_data, f)

        print(f"[WATER] Salvate {len(features)} features: {geojson_path}")
        return geojson_path

    except req.RequestException as e:
        print(f"[WATER] Errore Overpass API: {e}")
        print("[WATER] Generazione sintetica come fallback...")
        return _generate_synthetic_water(output_dir)
    except Exception as e:
        print(f"[WATER] Errore generico: {e}")
        return _generate_synthetic_water(output_dir)


def _overpass_element_to_geojson(element: dict):
    """Converte un elemento Overpass in geometria GeoJSON."""
    etype = element.get("type")

    if etype == "way" and "geometry" in element:
        coords = [(n["lon"], n["lat"]) for n in element["geometry"]]
        if len(coords) < 2:
            return None
        # Se il primo e l'ultimo punto coincidono → Polygon
        if coords[0] == coords[-1] and len(coords) >= 4:
            return {"type": "Polygon", "coordinates": [coords]}
        else:
            return {"type": "LineString", "coordinates": coords}

    elif etype == "relation" and "members" in element:
        # Per le relation, estrai i way outer
        rings = []
        for member in element["members"]:
            if member.get("role") == "outer" and "geometry" in member:
                coords = [(n["lon"], n["lat"]) for n in member["geometry"]]
                if len(coords) >= 4:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    rings.append(coords)
        if rings:
            return {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}

    return None


def _generate_synthetic_water(output_dir: Path) -> Path:
    """
    Genera features idriche sintetiche (simulazione Tevere + laghi).
    Salva direttamente il raster binarizzato.
    """
    geojson_path = output_dir / "water_features.geojson"

    # Crea raster binario direttamente
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    # Simula il Tevere come linea curva
    for row in range(GRID_SIZE):
        # Curva sinusoidale per il Tevere
        col = int(GRID_SIZE * 0.4 + 30 * np.sin(row * 2 * np.pi / GRID_SIZE))
        col = max(0, min(GRID_SIZE - 1, col))
        for dc in range(-3, 4):  # Larghezza ~7 pixels = ~280m
            c = col + dc
            if 0 <= c < GRID_SIZE:
                mask[row, c] = 1

    # Aggiungi qualche lago/bacino
    rng = np.random.RandomState(42)
    for _ in range(5):
        cy, cx = rng.randint(50, GRID_SIZE - 50, 2)
        ry, rx = rng.randint(5, 15, 2)
        yy, xx = np.ogrid[-cy:GRID_SIZE-cy, -cx:GRID_SIZE-cx]
        ellipse = (yy**2 / ry**2 + xx**2 / rx**2) <= 1.0
        mask[ellipse] = 1

    # Salva come GeoTIFF (il geojson non è necessario per dati sintetici)
    raster_path = output_dir / "water_mask.tif"
    profile = get_target_profile()
    profile["dtype"] = "uint8"
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"[WATER] Generato water mask sintetico: {raster_path}")
    # Creo anche un geojson vuoto come placeholder
    import json
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": []}, f)

    return geojson_path


def rasterize_water_features(
    geojson_path: Path = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Rasterizza le features idriche su griglia 512×512.

    Returns:
        np.ndarray (512, 512) — maschera binaria (1=acqua, 0=terra)
    """
    mask_path = output_dir / "water_mask.tif"

    if mask_path.exists():
        print(f"[WATER] Maschera già esistente: {mask_path}")
        with rasterio.open(mask_path) as src:
            return src.read(1)

    if geojson_path is None:
        geojson_path = output_dir / "water_features.geojson"

    print("[WATER] Rasterizzazione features idriche...")

    import geopandas as gpd
    from pyproj import Transformer

    gdf = gpd.read_file(geojson_path)

    if len(gdf) == 0:
        print("[WATER] GeoJSON vuoto, maschera tutta a zero")
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    else:
        # Riproietta in UTM 33N
        gdf = gdf.to_crs("EPSG:32633")

        bbox_utm = compute_bbox_utm()
        transform = get_affine_transform(bbox_utm)

        # Rasterizza
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        mask = rfeatures.rasterize(
            shapes=shapes,
            out_shape=(GRID_SIZE, GRID_SIZE),
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

    # Salva
    profile = get_target_profile()
    profile["dtype"] = "uint8"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)

    water_pct = 100 * mask.sum() / mask.size
    print(f"[WATER] Maschera salvata: {mask_path} — {water_pct:.2f}% acqua")
    return mask


def compute_water_distance(
    water_mask: np.ndarray = None,
    output_dir: Path = STATIC_DIR
) -> np.ndarray:
    """
    Calcola la distanza euclidea di ogni pixel dalla fonte idrica più vicina.

    Returns:
        np.ndarray (512, 512) — distanza in metri
    """
    output_path = output_dir / "water_distance.tif"

    if output_path.exists():
        print(f"[WATER] Distance transform già calcolata: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if water_mask is None:
        mask_path = output_dir / "water_mask.tif"
        with rasterio.open(mask_path) as src:
            water_mask = src.read(1)

    print("[WATER] Calcolo Distance Transform...")

    # distance_transform_edt: calcola distanza da pixel=0 al pixel=1 più vicino
    # Noi vogliamo distanza da non-acqua ad acqua, quindi invertiamo
    # Se mask: 1=acqua, 0=terra → distanza dei pixel terra da acqua
    distance = distance_transform_edt(1 - water_mask) * PIXEL_SIZE  # in metri

    distance = distance.astype(np.float32)

    # Salva
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(distance, 1)

    print(f"[WATER] Distance Transform salvata: {output_path}")
    print(f"[WATER] Range: [{distance.min():.0f}, {distance.max():.0f}]m")
    return distance


def download_and_process_water_distance(output_dir: Path = STATIC_DIR) -> np.ndarray:
    """Pipeline completa: download OSM → rasterize → distance transform."""
    geojson_path = download_water_features(output_dir)
    water_mask = rasterize_water_features(geojson_path, output_dir)
    return compute_water_distance(water_mask, output_dir)


if __name__ == "__main__":
    dist = download_and_process_water_distance()
    print(f"\nDistanza acqua: shape={dist.shape}, dtype={dist.dtype}")
    print(f"Range: [{dist.min():.0f}, {dist.max():.0f}]m")
