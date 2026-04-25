"""
Download dati precipitazione da Open-Meteo API.
Produce 3 canali + 1 canale umidità suolo per ogni evento,
interpolati su una griglia 512x512 partendo da un insieme di punti.

Canali:
  - precip_today:       precipitazione giornaliera del giorno (mm)
  - precip_yesterday:   precipitazione giornaliera del giorno prima (mm)
  - precip_day_before:  precipitazione di 2 giorni prima (mm)
  - soil_moisture:      umidità suolo 0-7cm (m³/m³)
"""
import numpy as np
import requests
import math
import scipy.ndimage as ndi
from pathlib import Path
from datetime import datetime, timedelta
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ROME_LAT, ROME_LON, EVENTS_DIR, GRID_SIZE, PIXEL_SIZE,
    OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL
)

GRID_POINTS = 4  # Si usano 4x4 coordinate per l'interpolazione

def _build_coordinates_grid() -> tuple[list, list]:
    """Crea una griglia 4x4 di lat/lon che copre l'area."""
    half_m = (GRID_SIZE * PIXEL_SIZE) / 2.0
    half_lat = half_m / 111_000.0
    half_lon = half_m / (111_000.0 * math.cos(math.radians(ROME_LAT)))

    # Ordine Nord -> Sud per righe [0, 511]
    lats = np.linspace(ROME_LAT + half_lat, ROME_LAT - half_lat, GRID_POINTS)
    # Ordine Ovest -> Est per colonne [0, 511]
    lons = np.linspace(ROME_LON - half_lon, ROME_LON + half_lon, GRID_POINTS)
    
    return lats, lons


def fetch_spatial_precipitation(
    date: datetime,
) -> dict:
    """Scarica dati di precipitazione ed umidità per l'intera area."""
    now = datetime.utcnow()
    
    start_date = (date - timedelta(days=2)).strftime("%Y-%m-%d")
    end_date = date.strftime("%Y-%m-%d")
    
    # Se il dato è > 7 giorni fa si usa archive-api 
    url = OPEN_METEO_ARCHIVE_URL if date < now - timedelta(days=7) else OPEN_METEO_FORECAST_URL
    
    lats_arr, lons_arr = _build_coordinates_grid()
    flat_lats, flat_lons = [], []
    for lat in lats_arr:
        for lon in lons_arr:
            flat_lats.append(round(lat, 5))
            flat_lons.append(round(lon, 5))

    params = {
        "latitude": ",".join(map(str, flat_lats)),
        "longitude": ",".join(map(str, flat_lons)),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "hourly": "soil_moisture_0_to_7cm",
        "timezone": "UTC",
    }
        
    print(f"[PRECIP] Fetching ({GRID_POINTS}x{GRID_POINTS}) precipitazione [{start_date} -> {end_date}]...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data_list = response.json()
        
        if not isinstance(data_list, list):
            data_list = [data_list] # Fallback in caso eccezionale

        matrices = {
            "precip_today": np.zeros((GRID_POINTS, GRID_POINTS)),
            "precip_yesterday": np.zeros((GRID_POINTS, GRID_POINTS)),
            "precip_day_before": np.zeros((GRID_POINTS, GRID_POINTS)),
            "soil_moisture": np.zeros((GRID_POINTS, GRID_POINTS)),
        }
        
        target_date_today = date.strftime("%Y-%m-%d")
        target_date_yday = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        target_date_2day = (date - timedelta(days=2)).strftime("%Y-%m-%d")
        target_hour = date.strftime("%Y-%m-%dT%H:00")

        # Processa ciascun punto restituito
        for i in range(GRID_POINTS):
            for j in range(GRID_POINTS):
                idx = i * GRID_POINTS + j
                point_data = data_list[idx] if idx < len(data_list) else {}
                
                daily = point_data.get("daily", {})
                d_times = daily.get("time", [])
                p_sums = daily.get("precipitation_sum", [])
                
                hourly = point_data.get("hourly", {})
                h_times = hourly.get("time", [])
                sm_vals = hourly.get("soil_moisture_0_to_7cm", [])

                def get_precip(d_str):
                    if d_str in d_times:
                        k = d_times.index(d_str)
                        v = p_sums[k]
                        return float(v) if v is not None else 0.0
                    return 0.0

                matrices["precip_today"][i, j] = get_precip(target_date_today)
                matrices["precip_yesterday"][i, j] = get_precip(target_date_yday)
                matrices["precip_day_before"][i, j] = get_precip(target_date_2day)
                
                # Soil moisture
                sm = 0.3
                if target_hour in h_times:
                    k = h_times.index(target_hour)
                    v = sm_vals[k]
                    if v is not None: sm = float(v)
                else:
                    valid_sm = [v for v in sm_vals if v is not None]
                    if valid_sm: sm = float(np.mean(valid_sm))
                matrices["soil_moisture"][i, j] = sm

        return matrices

    except Exception as e:
        print(f"[PRECIP] Errore API: {e}")
        return _synthetic_matrices(date)


def _synthetic_matrices(date: datetime) -> dict:
    """Genera gradienti sintetici (es. da ovest verso est) come fallback."""
    rng = np.random.RandomState(int(date.timestamp()) % 2**31)
    
    mats = {}
    base_today = float(rng.exponential(4.0))
    base_yesterday = float(rng.exponential(3.0))
    
    # Crea un falso gradiente (+ rumore)
    yy, xx = np.mgrid[0:GRID_POINTS, 0:GRID_POINTS]
    gradient = (xx / float(GRID_POINTS)) * rng.uniform(0.5, 1.5)
    
    mats["precip_today"] = (np.ones((GRID_POINTS, GRID_POINTS)) * base_today) + gradient * base_today
    mats["precip_yesterday"] = (np.ones((GRID_POINTS, GRID_POINTS)) * base_yesterday) + gradient * base_yesterday
    mats["precip_day_before"] = np.ones((GRID_POINTS, GRID_POINTS)) * rng.exponential(2.0)
    mats["soil_moisture"] = np.clip(np.ones((GRID_POINTS, GRID_POINTS)) * 0.2 + (gradient * 0.1), 0.0, 1.0)
    return mats


def create_precipitation_channels(
    event_date: datetime,
    output_dir: Path = None
) -> dict[str, np.ndarray]:
    """Crea tutti i 4 canali meteo interpolati a 512×512 pixel."""
    if output_dir is None:
        event_id = event_date.strftime("%Y%m%d_%H")
        output_dir = EVENTS_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = {}
    channel_names = [
        "precip_today", "precip_yesterday", "precip_day_before",
        "soil_moisture"
    ]
    all_exist = all((output_dir / f"{name}.npy").exists() for name in channel_names)

    if all_exist:
        for name in channel_names:
            channels[name] = np.load(output_dir / f"{name}.npy")
        print(f"[PRECIP] Tutti i canali già esistenti in {output_dir}")
        return channels

    # 1. Recupera le matrici sparse (4x4)
    coarse_matrices = fetch_spatial_precipitation(event_date)
    
    # 2. Interpola le feature alla risoluzione finale (512x512)
    zoom_factor = GRID_SIZE / GRID_POINTS
    
    avg_logs = {}
    for name in channel_names:
        coarse = coarse_matrices[name]
        
        # Interpola da XxX a 512x512 (order=3: cubic, smooth gradient)
        fine = ndi.zoom(coarse, zoom_factor, order=3)
        
        # Evita che l'interpolazione cubica generi precipitazioni negative
        fine = np.clip(fine, 0.0, None)
        
        channels[name] = fine.astype(np.float32)
        np.save(output_dir / f"{name}.npy", fine)
        
        avg_logs[name] = float(np.mean(fine))

    # Salva anche metadata per riferimento
    meta = {
        "event_date": event_date.isoformat(),
        "lat_center": ROME_LAT,
        "lon_center": ROME_LON,
        "average_values": avg_logs
    }
    with open(output_dir / "precipitation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[PRECIP] Interpolazione completata e salvata ({avg_logs['precip_today']:.1f} avg mm today).")
    return channels


if __name__ == "__main__":
    event = datetime.utcnow()
    output_test = Path("test_out")
    chans = create_precipitation_channels(event, output_test)
    print(f"Dimensione canali: {chans['precip_today'].shape}")
