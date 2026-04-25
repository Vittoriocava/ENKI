"""
Download dati precipitazione da Open-Meteo API.
Produce 5 canali + 1 canale umidità suolo per ogni evento.

Canali:
  - precip_24h:       somma precipitazioni ultime 24h (mm)
  - precip_1h:        precipitazione ultima ora (mm)
  - precip_forecast_1h: precipitazione prevista +1h (mm)
  - precip_forecast_2h: precipitazione prevista +2h (mm)
  - precip_forecast_3h: precipitazione prevista +3h (mm)
  - soil_moisture:    umidità suolo 0-7cm (m³/m³)

Nota: per i canali meteo, il valore puntuale (centro area) viene
replicato su tutta la matrice 512×512 come da specifica.
"""
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ROME_LAT, ROME_LON, EVENTS_DIR, GRID_SIZE,
    OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL
)


def fetch_historical_precipitation(
    date: datetime,
    lat: float = ROME_LAT,
    lon: float = ROME_LON
) -> dict:
    """
    Scarica dati storici di precipitazione da Open-Meteo Archive API.

    Args:
        date: data dell'evento
        lat: latitudine
        lon: longitudine

    Returns:
        dict con chiavi: precip_24h, precip_1h, soil_moisture
    """
    # Range: giorno precedente + giorno corrente
    start_date = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = date.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,soil_moisture_0_to_7cm",
        "timezone": "UTC",
    }

    print(f"[PRECIP] Fetching dati storici {start_date} → {end_date}...")

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip_values = hourly.get("precipitation", [])
        soil_moist_values = hourly.get("soil_moisture_0_to_7cm", [])

        # Trova l'indice dell'ora dell'evento
        target_hour = date.strftime("%Y-%m-%dT%H:00")
        if target_hour in times:
            idx = times.index(target_hour)
        else:
            # Usa l'ultima ora disponibile del giorno
            idx = len(times) - 1

        # Precipitazione ultima ora
        precip_1h = precip_values[idx] if idx < len(precip_values) else 0.0
        precip_1h = precip_1h if precip_1h is not None else 0.0

        # Somma precipitazioni ultime 24h
        start_idx = max(0, idx - 23)
        precip_24h_values = precip_values[start_idx:idx + 1]
        precip_24h = sum(v for v in precip_24h_values if v is not None)

        # Umidità suolo
        soil_moisture = 0.3  # default
        if soil_moist_values and idx < len(soil_moist_values):
            val = soil_moist_values[idx]
            if val is not None:
                soil_moisture = val

        result = {
            "precip_24h": float(precip_24h),
            "precip_1h": float(precip_1h),
            "soil_moisture": float(soil_moisture),
        }
        print(f"[PRECIP] Storico: 24h={result['precip_24h']:.1f}mm, "
              f"1h={result['precip_1h']:.1f}mm, "
              f"soil_moist={result['soil_moisture']:.3f}")
        return result

    except requests.RequestException as e:
        print(f"[PRECIP] Errore API storico: {e}")
        return _synthetic_historical(date)


def fetch_forecast_precipitation(
    date: datetime,
    lat: float = ROME_LAT,
    lon: float = ROME_LON
) -> dict:
    """
    Scarica previsioni di precipitazione da Open-Meteo.

    Per date storiche usa l'Historical Forecast API.
    Per date future usa la Forecast API standard.

    Args:
        date: data/ora di partenza
        lat: latitudine
        lon: longitudine

    Returns:
        dict con chiavi: precip_forecast_1h, precip_forecast_2h, precip_forecast_3h
    """
    # Determina quale API usare
    now = datetime.utcnow()
    if date < now - timedelta(days=7):
        # Usa Historical Forecast API
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        date_str = date.strftime("%Y-%m-%d")
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "precipitation",
            "timezone": "UTC",
        }
    else:
        # Usa Forecast API standard
        url = OPEN_METEO_FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "forecast_days": 2,
            "timezone": "UTC",
        }

    print(f"[PRECIP] Fetching previsioni per {date.strftime('%Y-%m-%d %H:00')}...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip_values = hourly.get("precipitation", [])

        # Trova l'indice dell'ora dell'evento
        target_hour = date.strftime("%Y-%m-%dT%H:00")
        if target_hour in times:
            idx = times.index(target_hour)
        else:
            idx = 0

        # Previsioni +1h, +2h, +3h
        def safe_get(vals, i):
            if i < len(vals) and vals[i] is not None:
                return float(vals[i])
            return 0.0

        result = {
            "precip_forecast_1h": safe_get(precip_values, idx + 1),
            "precip_forecast_2h": safe_get(precip_values, idx + 2),
            "precip_forecast_3h": safe_get(precip_values, idx + 3),
        }

        print(f"[PRECIP] Previsioni: +1h={result['precip_forecast_1h']:.1f}mm, "
              f"+2h={result['precip_forecast_2h']:.1f}mm, "
              f"+3h={result['precip_forecast_3h']:.1f}mm")
        return result

    except requests.RequestException as e:
        print(f"[PRECIP] Errore API forecast: {e}")
        return _synthetic_forecast(date)


def _synthetic_historical(date: datetime) -> dict:
    """Genera dati storici sintetici."""
    rng = np.random.RandomState(int(date.timestamp()) % 2**31)
    return {
        "precip_24h": float(rng.exponential(8.0)),
        "precip_1h": float(rng.exponential(3.0)),
        "soil_moisture": float(0.2 + rng.random() * 0.3),
    }


def _synthetic_forecast(date: datetime) -> dict:
    """Genera previsioni sintetiche."""
    rng = np.random.RandomState(int(date.timestamp()) % 2**31 + 1)
    return {
        "precip_forecast_1h": float(rng.exponential(4.0)),
        "precip_forecast_2h": float(rng.exponential(5.0)),
        "precip_forecast_3h": float(rng.exponential(6.0)),
    }


def create_precipitation_channels(
    event_date: datetime,
    output_dir: Path = None
) -> dict[str, np.ndarray]:
    """
    Crea tutti i 6 canali meteo (5 precipitazione + 1 umidità suolo).
    Ogni canale è una matrice 512×512 con valore costante.

    Args:
        event_date: data/ora dell'evento
        output_dir: directory di output

    Returns:
        dict[str, np.ndarray] con 6 canali, ciascuno (512, 512)
    """
    if output_dir is None:
        event_id = event_date.strftime("%Y%m%d_%H")
        output_dir = EVENTS_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Controlla se già generati
    channels = {}
    all_exist = True
    channel_names = [
        "precip_24h", "precip_1h",
        "precip_forecast_1h", "precip_forecast_2h", "precip_forecast_3h",
        "soil_moisture"
    ]

    for name in channel_names:
        path = output_dir / f"{name}.npy"
        if path.exists():
            channels[name] = np.load(path)
        else:
            all_exist = False

    if all_exist:
        print(f"[PRECIP] Tutti i canali già esistenti in {output_dir}")
        return channels

    # Fetch dati
    historical = fetch_historical_precipitation(event_date)
    forecast = fetch_forecast_precipitation(event_date)

    # Unisci
    all_values = {**historical, **forecast}

    # Crea matrici costanti 512×512
    for name in channel_names:
        value = all_values.get(name, 0.0)
        channel = np.full((GRID_SIZE, GRID_SIZE), value, dtype=np.float32)
        channels[name] = channel

        # Salva
        np.save(output_dir / f"{name}.npy", channel)

    # Salva anche un JSON con i valori puntuali per riferimento
    meta = {
        "event_date": event_date.isoformat(),
        "lat": ROME_LAT,
        "lon": ROME_LON,
        "values": {k: float(v) for k, v in all_values.items()}
    }
    with open(output_dir / "precipitation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[PRECIP] Tutti i canali salvati in {output_dir}")
    return channels


if __name__ == "__main__":
    event = datetime(2023, 11, 15, 12, 0)
    channels = create_precipitation_channels(event)
    for name, arr in channels.items():
        print(f"  {name}: shape={arr.shape}, value={arr[0, 0]:.3f}")
