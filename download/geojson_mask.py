import json
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRID_SIZE, PIXEL_SIZE, EVENTS_DIR
from utils.geo import latlon_to_utm, compute_bbox_utm, get_affine_transform, get_target_profile

def get_radius_from_description(desc: str) -> float:
    """
    Sceglie il raggio (in metri) in base ai termini presenti nella descrizione dell'evento storicizzato.
    """
    desc = desc.lower()
    # Parole chiave per allagamenti molto estesi
    if any(w in desc for w in ["fiume", "esondati", "allagamenti diffusi", "quartiere", "sott'acqua", "fiumi", "sommersa", "lago", "inondato"]):
        return 5000.0  # 5 km
    # Parole chiave per strade intere / assi viari principali
    elif any(w in desc for w in ["strada", "strade", "via", "viale", "incrocio", "rotatoria", "allagata"]):
        return 2000.0   # 2 km
    # Punti specifici circoscritti e piccole aree
    elif any(w in desc for w in ["sottopasso", "tunnel", "stazione", "metro", "cantina", "tetto", "scuola", "asilo", "ospedale", "canile"]):
        return 1000.0   # 1 km
    # Disagi minori come rami e alberi caduti senza grandi inondazioni
    elif any(w in desc for w in ["albero", "rami", "voragine", "cedimento"]):
        return 500.0   # 500 m
    
    return 500.0       # Radius di default


def generate_flood_mask_from_geojson(
    event_date: datetime,
    output_dir: Path = None,
) -> np.ndarray:
    """
    Genera la maschera alluvione (target) disegnando dei cerchi intorno 
    ai punti di rischio presenti nel file GeoJSON, basando il raggio 
    sulla gravità descritta nel commento.
    """
    if output_dir is None:
        event_id = event_date.strftime("%Y%m%d")
        output_dir = EVENTS_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "flood_mask.tif"
    
    if output_path.exists():
        print(f"[GEOJSON] Flood mask già esistente: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1).astype(np.float32)

    # 1. Carica GeoJSON
    geojson_path = Path(__file__).resolve().parent.parent / "historical_events_rome.geojson"
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            fc = json.load(f)
    except FileNotFoundError:
        print(f"[GEOJSON] File non trovato: {geojson_path}")
        fc = {"features": []}
        
    # 2. Setup griglia e trasformazione per coordinate UTM
    bbox_utm = compute_bbox_utm()
    transform = get_affine_transform(bbox_utm, GRID_SIZE)
    inv_transform = ~transform
    
    flood_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    # 3. Identifica gli eventi storici che corrispondono al giorno di 'event_date'
    event_day_str = event_date.strftime("%Y-%m-%d")
    
    features_found = 0
    for feature in fc.get("features", []):
        props = feature.get("properties", {})
        ms_timestamp = props.get("data_evento")
        
        if ms_timestamp:
            feat_date = datetime.fromtimestamp(ms_timestamp / 1000.0, tz=timezone.utc)
            if feat_date.strftime("%Y-%m-%d") == event_day_str:
                geom = feature.get("geometry", {})
                if geom.get("type") == "Point":
                    lon, lat = geom.get("coordinates", [0, 0])
                    desc_str = props.get("descrizione", "")
                    
                    radius_m = get_radius_from_description(desc_str)
                    radius_px = radius_m / PIXEL_SIZE
                    
                    # Converti WGS84 -> UTM -> (col, row) pixel
                    easting, northing = latlon_to_utm(lat, lon)
                    col_f, row_f = inv_transform * (easting, northing)
                    col, row = int(round(col_f)), int(round(row_f))
                    
                    # Disegna un cerchio centrato in (row, col) con raggio 'radius_px'
                    # Usiamo ogrid per calcolare rapidamente la distanza di ogni pixel
                    y, x = np.ogrid[-row:GRID_SIZE-row, -col:GRID_SIZE-col]
                    dist_sq = x**2 + y**2
                    
                    # Maschera booleana dei pixel coperti
                    circle_mask = dist_sq <= radius_px**2
                    
                    # Valvola di sicurezza: assicuriamoci di restare nel bounding della griglia 512x512
                    valid_mask = (row + y >= 0) & (row + y < GRID_SIZE) & \
                                 (col + x >= 0) & (col + x < GRID_SIZE) & circle_mask
                    
                    flood_mask[valid_mask] = 1.0
                    features_found += 1

    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(flood_mask, 1)

    flood_pct = (flood_mask == 1).sum() / flood_mask.size * 100
    print(f"[GEOJSON] Generata mask per {event_date.strftime('%Y-%m-%d')}: "
          f"{features_found} zone trovate, {flood_pct:.2f}% allagato")
    
    return flood_mask
