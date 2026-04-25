"""
Configurazione globale per il pipeline di generazione dataset alluvioni.
Coordinate centrate su Roma, griglia 512x512 @ 40m/pixel.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────
# Coordinate centro area di interesse
# ─────────────────────────────────────────────
ROME_LAT = 41.8931
ROME_LON = 12.4828

# ─────────────────────────────────────────────
# Griglia di output
# ─────────────────────────────────────────────
PIXEL_SIZE = 40          # metri per pixel
GRID_SIZE = 512          # pixel per lato
AREA_SIZE = PIXEL_SIZE * GRID_SIZE  # 20480m = 20.48 km

# ─────────────────────────────────────────────
# CRS
# ─────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32633"   # UTM Zone 33N (Roma)

# ─────────────────────────────────────────────
# Canali input del tensore
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
NUM_INPUT_CHANNELS = 11
CHANNEL_NAMES = [
    "altitude",              # 0 - DEM Copernicus GLO-30
    "slope",                 # 1 - Pendenza derivata dal DEM
    "permeability",          # 2 - CLMS Imperviousness (invertito)
    "vegetation",            # 3 - ESA WorldCover
    "water_distance",        # 4 - OSM Distance Transform
    "sar_baseline",          # 5 - Sentinel-1 SAR Mosaico di riferimento
    "sar_event",             # 6 - Sentinel-1 SAR Giorno evento
    "precip_today",          # 7 - Precipitazioni del giorno
    "precip_yesterday",      # 8 - Precipitazioni del giorno prima
    "precip_day_before",     # 9 - Precipitazioni di 2 giorni prima
    "soil_moisture",         # 10 - Umidità suolo Open-Meteo
]

# ─────────────────────────────────────────────
# Sentinel-1 SAR
# ─────────────────────────────────────────────
SAR_POLARIZATION = "VV"
SAR_INSTRUMENT_MODE = "IW"
SAR_FLOOD_THRESHOLD_DB = 3.0   # differenza dB per classificare alluvione

# ─────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────
UNET_BASE_FILTERS = 64
UNET_DEPTH = 4

# Loss
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
DICE_SMOOTH = 1.0

# Training
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
TRAIN_VAL_SPLIT = 0.8

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TENSOR_DIR = DATA_DIR / "tensors"
STATIC_DIR = RAW_DIR / "static"      # Dati statici (DEM, landcover, ecc.)
EVENTS_DIR = RAW_DIR / "events"      # Dati per evento (SAR, meteo)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Crea directory se non esistono
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TENSOR_DIR,
          STATIC_DIR, EVENTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Open-Meteo API
# ─────────────────────────────────────────────
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
