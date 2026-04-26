"""
Assemblaggio del tensore finale di training.

Combina i 12 canali di input e la maschera target in file .pt (PyTorch)
pronti per il DataLoader.

Shape input:  (12, 512, 512) — float32
Shape target: (1, 512, 512) — float32 {0.0, 1.0}
"""
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GRID_SIZE, NUM_INPUT_CHANNELS, CHANNEL_NAMES,
    STATIC_DIR, EVENTS_DIR, TENSOR_DIR, PROCESSED_DIR
)
from processing.normalize import ChannelNormalizer


def load_static_channels(static_dir: Path = STATIC_DIR) -> dict[str, np.ndarray]:
    """
    Carica tutti i canali statici (invarianti nel tempo).

    Returns:
        dict con chiavi: altitude, slope, permeability, vegetation, water_distance
    """
    channels = {}
    static_files = {
        "altitude": "altitude.tif",
        "slope": "slope.tif",
        "permeability": "permeability.tif",
        "vegetation": "vegetation.tif",
        "water_distance": "water_distance.tif",
    }

    for name, filename in static_files.items():
        path = static_dir / filename
        if path.exists():
            with rasterio.open(path) as src:
                channels[name] = src.read(1).astype(np.float32)
            print(f"[TENSOR] Caricato {name}: shape={channels[name].shape}, "
                  f"range=[{channels[name].min():.2f}, {channels[name].max():.2f}]")
        else:
            print(f"[TENSOR] ATTENZIONE: {path} non trovato, usando zeri")
            channels[name] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    return channels


def load_event_channels(
    event_date: datetime,
    events_dir: Path = EVENTS_DIR
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Carica i canali variabili nel tempo per un evento specifico.

    Returns:
        (event_channels, flood_mask) dove:
        - event_channels: dict con chiavi: sar_baseline, sar_event, precip_*, soil_moisture
        - flood_mask: np.ndarray (512, 512) binario
    """
    # Cerca la directory dell'evento (formato YYYYMMDD o YYYYMMDD_HH)
    event_id = event_date.strftime("%Y%m%d")
    event_id_h = event_date.strftime("%Y%m%d_%H")

    event_dir = None
    for eid in [event_id_h, event_id]:
        candidate = events_dir / eid
        if candidate.exists():
            event_dir = candidate
            break

    if event_dir is None:
        raise FileNotFoundError(
            f"Directory evento non trovata per {event_date} in {events_dir}"
        )

    channels = {}

    # SAR baseline & event
    for name in ["sar_baseline", "sar_event"]:
        path = event_dir / f"{name}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                channels[name] = src.read(1).astype(np.float32)
        else:
            channels[name] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Precipitazioni (3 canali + umidità)
    precip_names = [
        "precip_today", "precip_yesterday", "precip_day_before",
        "soil_moisture"
    ]
    for name in precip_names:
        npy_path = event_dir / f"{name}.npy"
        if npy_path.exists():
            channels[name] = np.load(npy_path).astype(np.float32)
        else:
            channels[name] = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Flood mask (target)
    mask_path = event_dir / "flood_mask.tif"
    if mask_path.exists():
        with rasterio.open(mask_path) as src:
            flood_mask = src.read(1).astype(np.float32)
    else:
        flood_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    return channels, flood_mask


def build_tensor(
    static_channels: dict[str, np.ndarray],
    event_channels: dict[str, np.ndarray],
    normalize: bool = True
) -> np.ndarray:
    """
    Assembla il tensore di input dai canali statici e variabili.

    Args:
        static_channels: dict dei canali statici
        event_channels: dict dei canali variabili
        normalize: se True, applica normalizzazione per canale

    Returns:
        np.ndarray (12, 512, 512) float32
    """
    # Ordine dei canali come da CHANNEL_NAMES
    all_channels = {**static_channels, **event_channels}

    tensor = np.zeros((NUM_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for i, name in enumerate(CHANNEL_NAMES):
        if name in all_channels:
            channel = all_channels[name].astype(np.float32)
            # Valida shape
            if channel.shape != (GRID_SIZE, GRID_SIZE):
                raise ValueError(
                    f"Canale {name}: shape attesa ({GRID_SIZE}, {GRID_SIZE}), "
                    f"ottenuta {channel.shape}"
                )
            tensor[i] = channel
        else:
            print(f"[TENSOR] Canale {name} non trovato, usando zeri")

    # Normalizzazione
    if normalize:
        normalizer = ChannelNormalizer()
        tensor = normalizer.normalize_tensor(tensor)

    return tensor


def build_and_save_sample(
    event_date: datetime,
    static_channels: dict[str, np.ndarray],
    output_dir: Path = TENSOR_DIR,
    normalize: bool = True,
    sample_id: Optional[int] = None
) -> tuple[Path, Path]:
    """
    Costruisce e salva un sample completo (input + target).

    Args:
        event_date: data dell'evento
        static_channels: canali statici pre-caricati
        output_dir: directory per salvare i tensori
        normalize: se normalizzare
        sample_id: ID numerico del sample

    Returns:
        (input_path, target_path) — percorsi ai file salvati
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_id is None:
        sample_id = int(event_date.timestamp())

    input_path = output_dir / f"input_{sample_id:06d}.npy"
    target_path = output_dir / f"target_{sample_id:06d}.npy"

    if input_path.exists() and target_path.exists():
        print(f"[TENSOR] Sample {sample_id} già esistente")
        return input_path, target_path

    # Carica canali evento
    event_channels, flood_mask = load_event_channels(event_date)

    # Assembla tensore input
    input_tensor = build_tensor(static_channels, event_channels, normalize)

    # Target: (1, H, W)
    target_tensor = flood_mask.reshape(1, GRID_SIZE, GRID_SIZE).astype(np.float32)

    # Salva
    np.save(input_path, input_tensor)
    np.save(target_path, target_tensor)

    flood_pct = 100 * flood_mask.sum() / flood_mask.size
    print(f"[TENSOR] Sample {sample_id} salvato:")
    print(f"  Input:  {input_path} — shape={input_tensor.shape}")
    print(f"  Target: {target_path} — shape={target_tensor.shape} ({flood_pct:.2f}% flood)")

    return input_path, target_path


def build_dataset_index(
    tensor_dir: Path = TENSOR_DIR
) -> list[dict]:
    """
    Crea un indice di tutti i sample nel dataset.

    Returns:
        list di dict con chiavi: sample_id, input_path, target_path
    """
    index = []
    input_files = sorted(tensor_dir.glob("input_*.npy"))

    for inp_path in input_files:
        sample_id = inp_path.stem.split("_")[1]
        tgt_path = tensor_dir / f"target_{sample_id}.npy"
        if tgt_path.exists():
            index.append({
                "sample_id": sample_id,
                "input_path": str(inp_path),
                "target_path": str(tgt_path),
            })

    # Salva indice
    index_path = tensor_dir / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[TENSOR] Indice dataset: {len(index)} sample trovati")
    return index


if __name__ == "__main__":
    static = load_static_channels()
    print(f"\nCanali statici caricati: {list(static.keys())}")
