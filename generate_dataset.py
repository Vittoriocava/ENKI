"""
Script orchestratore per la generazione del dataset completo.

Workflow:
1. Scarica tutti i dati statici (DEM, slope, permeability, vegetation, water distance)
2. Per ogni evento/timestamp selezionato:
   a. Scarica/genera dati SAR (soil state + flood mask)
   b. Scarica precipitazioni (5 canali + umidità suolo)
   c. Assembla tensore 12-canali
   d. Genera maschera target
3. Salva il dataset completo pronto per il training

Uso: python generate_dataset.py [--synthetic] [--n-events N] [--use-stac]
"""
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ROME_LAT, ROME_LON, STATIC_DIR, EVENTS_DIR,
    TENSOR_DIR, GRID_SIZE, NUM_INPUT_CHANNELS
)

# Download modules
from download.dem import download_and_process_dem
from download.imperviousness import download_and_process_permeability
from download.landcover import download_and_process_vegetation
from download.water_features import download_and_process_water_distance
from download.sentinel1 import download_sar_flood_mask
from download.geojson_mask import generate_flood_mask_from_geojson
from download.precipitation import create_precipitation_channels

# Processing
from processing.tensor_builder import (
    load_static_channels, build_and_save_sample, build_dataset_index
)


# ─────────────────────────────────────────────
# Date di eventi storici (Roma e dintorni / generale)
# ─────────────────────────────────────────────
HISTORICAL_FLOOD_EVENTS = [
    datetime(2017, 5, 19, 12, 0),
    datetime(2017, 9, 3, 12, 0),
    datetime(2017, 11, 5, 12, 0),
    datetime(2017, 12, 27, 12, 0),
    datetime(2018, 3, 6, 12, 0),
    datetime(2018, 4, 8, 12, 0),
    datetime(2018, 7, 23, 12, 0),
    datetime(2018, 10, 9, 12, 0),
    datetime(2018, 10, 21, 12, 0),
    datetime(2018, 10, 22, 12, 0),
    datetime(2018, 11, 20, 12, 0),
    datetime(2019, 5, 12, 12, 0),
    datetime(2019, 5, 30, 12, 0),
    datetime(2019, 7, 27, 12, 0),
    datetime(2019, 8, 25, 12, 0),
    datetime(2019, 9, 2, 12, 0),
    datetime(2019, 10, 2, 12, 0),
    datetime(2019, 11, 11, 12, 0),
    datetime(2019, 12, 2, 12, 0),
    datetime(2020, 9, 23, 12, 0),
    datetime(2020, 10, 7, 12, 0),
    datetime(2020, 10, 15, 12, 0),
    datetime(2021, 1, 3, 12, 0),
    datetime(2021, 1, 23, 12, 0),
    datetime(2021, 1, 24, 12, 0),
    datetime(2021, 4, 19, 12, 0),
    datetime(2021, 6, 8, 12, 0),
    datetime(2021, 11, 8, 12, 0),
    datetime(2021, 12, 2, 12, 0),
    datetime(2022, 4, 22, 12, 0),
    datetime(2022, 8, 6, 12, 0),
    datetime(2022, 8, 9, 12, 0),
    datetime(2022, 10, 11, 12, 0),
    datetime(2022, 12, 3, 12, 0),
    datetime(2022, 12, 13, 12, 0),
    datetime(2023, 4, 15, 12, 0),
    datetime(2023, 6, 11, 12, 0),
    datetime(2023, 6, 13, 12, 0),
    datetime(2023, 6, 14, 12, 0),
    datetime(2023, 10, 16, 12, 0),
    datetime(2023, 10, 24, 12, 0),
    datetime(2023, 12, 5, 12, 0),
    datetime(2024, 9, 3, 12, 0),
    datetime(2024, 9, 13, 12, 0),
    datetime(2024, 9, 25, 12, 0),
    datetime(2024, 10, 5, 12, 0),
    datetime(2024, 10, 24, 12, 0),
    datetime(2025, 5, 6, 12, 0),
    datetime(2025, 7, 13, 12, 0),
    datetime(2025, 9, 10, 12, 0),
    datetime(2026, 1, 6, 12, 0),
    datetime(2026, 1, 28, 12, 0),
    datetime(2026, 3, 12, 12, 0),
]

# Date "normali" (non-alluvione) per bilanciare il dataset
NORMAL_EVENTS = [
    # datetime(2023, 7, 15, 12, 0),    # Estate (secco)
    # datetime(2023, 3, 10, 10, 0),    # Primavera
    # datetime(2022, 8, 20, 14, 0),    # Estate
    # datetime(2022, 4, 5, 9, 0),      # Primavera
    # datetime(2021, 7, 22, 11, 0),    # Estate
    # datetime(2021, 3, 15, 8, 0),     # Primavera
    # datetime(2020, 6, 18, 13, 0),    # Estate
    # datetime(2020, 4, 10, 10, 0),    # Primavera
    # datetime(2019, 8, 5, 15, 0),     # Estate
    # datetime(2019, 3, 22, 9, 0),     # Primavera
    # datetime(2018, 7, 30, 12, 0),    # Estate
    # datetime(2018, 5, 15, 11, 0),    # Primavera
    # datetime(2017, 8, 10, 14, 0),    # Estate
    # datetime(2017, 4, 20, 10, 0),    # Primavera
    # datetime(2016, 7, 5, 13, 0),     # Estate
]


def generate_static_data():
    """Scarica e processa tutti i dati statici."""
    print("=" * 60)
    print("FASE 1: Download dati statici")
    print("=" * 60)

    print("\n[1/4] DEM + Slope...")
    try:
        altitude, slope = download_and_process_dem()
    except Exception as e:
        print(f"  ⚠ Errore DEM: {e}")
        print("  → Generazione dati sintetici...")
        altitude = np.random.uniform(0, 200, (GRID_SIZE, GRID_SIZE)).astype(np.float32)
        slope = np.random.uniform(0, 30, (GRID_SIZE, GRID_SIZE)).astype(np.float32)
        _save_synthetic_static("altitude", altitude)
        _save_synthetic_static("slope", slope)

    print("\n[2/4] Permeabilità...")
    try:
        permeability = download_and_process_permeability()
    except Exception as e:
        print(f"  ⚠ Errore Permeabilità: {e}")
        permeability = np.random.uniform(10, 90, (GRID_SIZE, GRID_SIZE)).astype(np.float32)
        _save_synthetic_static("permeability", permeability)

    print("\n[3/4] Vegetazione...")
    try:
        vegetation = download_and_process_vegetation()
    except Exception as e:
        print(f"  ⚠ Errore Vegetazione: {e}")
        vegetation = np.random.choice([10, 20, 30, 40, 50, 80], (GRID_SIZE, GRID_SIZE)).astype(np.float32)
        _save_synthetic_static("vegetation", vegetation)

    print("\n[4/4] Distanza fonti idriche...")
    try:
        water_dist = download_and_process_water_distance()
    except Exception as e:
        print(f"  ⚠ Errore Water Distance: {e}")
        y, x = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]
        water_dist = (np.abs(x - GRID_SIZE*0.4) * 40).astype(np.float32)
        _save_synthetic_static("water_distance", water_dist)

    print("\n✓ Tutti i dati statici pronti!")
    return {"altitude": altitude, "slope": slope,
            "permeability": permeability, "vegetation": vegetation,
            "water_distance": water_dist}


def _save_synthetic_static(name: str, data: np.ndarray):
    """Salva un dato sintetico statico come GeoTIFF."""
    import rasterio
    from utils.geo import get_target_profile
    output_path = STATIC_DIR / f"{name}.tif"
    profile = get_target_profile()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"  Salvato: {output_path}")


def generate_event_data(
    event_date: datetime,
    is_flood: bool = True,
    use_stac: bool = False,
    use_sar: bool = False,
    use_geojson: bool = False
):
    """
    Genera i dati variabili per un singolo evento.

    Args:
        event_date: data dell'evento
        is_flood: se True è un evento di alluvione
        use_stac: se True usa STAC (Planetary Computer) per Sentinel-1 SAR
    """
    event_id = event_date.strftime("%Y%m%d")
    event_dir = EVENTS_DIR / event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    # SAR flood mask, baseline e event (scaricati insieme) oppure target da geojson
    if use_geojson:
        try:
            flood_mask = generate_flood_mask_from_geojson(event_date, event_dir)
        except Exception as e:
            print(f"    ⚠ Errore generazione mask da GeoJSON: {e}")
            import shutil
            shutil.rmtree(event_dir, ignore_errors=True)
            return False
    else:
        try:
            flood_mask = download_sar_flood_mask(event_date, event_dir, use_stac=use_stac)
        except ValueError as e:
            print(f"    ⚠ SAR scartato: {e}")
            import shutil
            shutil.rmtree(event_dir, ignore_errors=True)
            return False
        except Exception as e:
            print(f"    ⚠ SAR flood mask: {e}")

    # Forza a zero la mask per gli eventi "normali" per evitare falsi positivi da rumore SAR/GeoJSON vuoto
    if not is_flood:
        flood_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        import rasterio
        from utils.geo import get_target_profile
        output_path = event_dir / "flood_mask.tif"
        profile = get_target_profile()
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(flood_mask, 1)

    # Precipitazioni
    try:
        precip = create_precipitation_channels(event_date, event_dir)
    except Exception as e:
        print(f"    ⚠ Precipitazioni: {e}")

    # Salva metadata
    meta = {
        "event_date": event_date.isoformat(),
        "is_flood": is_flood,
        "lat": ROME_LAT,
        "lon": ROME_LON,
    }
    with open(event_dir / "event_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def assemble_tensors(
    static_channels: dict,
    events: list[tuple[datetime, bool]],
):
    """
    Assembla tutti i tensori del dataset.
    """
    print("\n" + "=" * 60)
    print("FASE 3: Assemblaggio tensori")
    print("=" * 60)

    for i, (event_date, is_flood) in enumerate(tqdm(events, desc="Assemblaggio")):
        try:
            build_and_save_sample(
                event_date=event_date,
                static_channels=static_channels,
                output_dir=TENSOR_DIR,
                normalize=True,
                sample_id=i
            )
        except Exception as e:
            print(f"\n  ⚠ Errore assemblaggio sample {i} ({event_date}): {e}")

    # Crea indice
    index = build_dataset_index(TENSOR_DIR)
    print(f"\n✓ Dataset assemblato: {len(index)} sample")


def main():
    parser = argparse.ArgumentParser(
        description="Generazione dataset per training UNet alluvioni"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Usa dati sintetici (no download reali)"
    )
    parser.add_argument(
        "--use-stac", action="store_true",
        help="Usa STAC (Planetary Computer) per Sentinel-1 SAR"
    )
    parser.add_argument(
        "--use-sar", action="store_true",
        help="Avvia estrazione target (flood mask) tramite Sentinel-1 SAR (Regola dello Specchio)"
    )
    parser.add_argument(
        "--use-geojson", action="store_true",
        help="Avvia estrazione target (flood mask) da dati storicizzati GeoJSON"
    )
    parser.add_argument(
        "--n-events", type=int, default=30,
        help="Numero totale di eventi da generare"
    )
    parser.add_argument(
        "--flood-ratio", type=float, default=0.5,
        help="Rapporto eventi alluvione vs normali (default: 0.5)"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ENKI SWARM — Generazione Dataset Alluvioni            ║")
    print("║  UNet Flood Prediction Pipeline                        ║")
    print(f"║  Centro: Roma ({ROME_LAT:.4f}°N, {ROME_LON:.4f}°E)         ║")
    print(f"║  Griglia: {GRID_SIZE}×{GRID_SIZE} @ 40m/pixel                    ║")
    print(f"║  Canali: {NUM_INPUT_CHANNELS}                                        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── FASE 1: Dati statici ──
    static_channels = generate_static_data()

    # ── FASE 2: Dati per evento ──
    print("\n" + "=" * 60)
    print("FASE 2: Download dati per evento")
    print("=" * 60)

    # Seleziona eventi
    n_flood = int(args.n_events * args.flood_ratio)
    n_normal = args.n_events - n_flood

    flood_events = HISTORICAL_FLOOD_EVENTS[:n_flood]
    normal_events = NORMAL_EVENTS[:n_normal]

    all_events = [(d, True) for d in flood_events] + [(d, False) for d in normal_events]
    valid_events = []

    for event_date, is_flood in tqdm(all_events, desc="Eventi"):
        try:
            success = generate_event_data(
                event_date, 
                is_flood, 
                use_stac=args.use_stac,
                use_sar=args.use_sar,
                use_geojson=args.use_geojson
            )
            if success is not False:
                valid_events.append((event_date, is_flood))
        except Exception as e:
            print(f"\n  ⚠ Errore evento {event_date}: {e}")

    # ── FASE 3: Assemblaggio ──
    assemble_tensors(static_channels, valid_events)

    # ── Riepilogo ──
    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"  Dati statici:   {STATIC_DIR}")
    print(f"  Dati eventi:    {EVENTS_DIR}")
    print(f"  Tensori:        {TENSOR_DIR}")
    print(f"  Totale sample:  {len(list(TENSOR_DIR.glob('input_*.npy')))}")
    print(f"\n  Per lanciare il training:")
    print(f"    python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
