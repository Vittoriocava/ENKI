"""
Normalizzazione per canale del tensore di input.

Ogni canale ha una strategia di normalizzazione specifica:
- DEM/Slope: Min-Max [0, 1]
- Permeabilità: /100
- Vegetazione: /255 (classi categoriche)
- Distanza acqua: Log-transform + Min-Max
- SAR backscatter: normalizzazione dB con clipping
- Precipitazioni: Min-Max con clipping al percentile 99
- Umidità suolo: già in [0, 1] circa
"""
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHANNEL_NAMES


class ChannelNormalizer:
    """
    Normalizza i canali del tensore input per il training.
    Supporta fit/transform per mantenere le statistiche.
    """

    def __init__(self):
        self.stats = {}  # {channel_name: {min, max, mean, std}}

    def fit(self, channel_data: np.ndarray, channel_name: str):
        """
        Calcola le statistiche di normalizzazione per un canale.

        Args:
            channel_data: array (H, W) del canale
            channel_name: nome del canale
        """
        valid = channel_data[np.isfinite(channel_data)]
        self.stats[channel_name] = {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "p01": float(np.percentile(valid, 1)),
            "p99": float(np.percentile(valid, 99)),
        }

    def normalize(self, channel_data: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Normalizza un canale secondo la strategia appropriata.

        Args:
            channel_data: array (H, W)
            channel_name: nome del canale

        Returns:
            np.ndarray normalizzato in [0, 1] (approssimativamente)
        """
        data = channel_data.copy().astype(np.float32)

        # Sostituisci NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if channel_name == "altitude":
            return self._minmax(data, channel_name)

        elif channel_name == "slope":
            return self._minmax(data, channel_name)

        elif channel_name == "permeability":
            return data / 100.0

        elif channel_name == "vegetation":
            # Classi categoriche: normalizza a [0, 1]
            return data / 100.0  # Le classi ESA WorldCover vanno da 10 a 100

        elif channel_name == "water_distance":
            return self._log_minmax(data, channel_name)

        elif channel_name == "soil_state":
            return self._sar_normalize(data)

        elif channel_name == "soil_moisture":
            # Già in [0, 1] circa (m³/m³)
            return np.clip(data, 0, 1)

        elif channel_name.startswith("precip"):
            return self._precip_normalize(data, channel_name)

        else:
            # Fallback: Min-Max
            return self._minmax(data, channel_name)

    def _minmax(self, data: np.ndarray, name: str) -> np.ndarray:
        """Normalizzazione Min-Max [0, 1]."""
        if name in self.stats:
            vmin = self.stats[name]["min"]
            vmax = self.stats[name]["max"]
        else:
            vmin = data.min()
            vmax = data.max()

        if vmax - vmin < 1e-8:
            return np.zeros_like(data)

        return (data - vmin) / (vmax - vmin)

    def _log_minmax(self, data: np.ndarray, name: str) -> np.ndarray:
        """Log-transform + Min-Max per distribuzioni long-tail (es. distanza acqua)."""
        # Log(1 + x) per evitare log(0)
        log_data = np.log1p(data)
        return self._minmax(log_data, name + "_log")

    def _sar_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizza backscatter SAR.
        Range tipico VV: [-25, 0] dB → mappiamo a [0, 1]
        """
        # Clip al range tipico
        clipped = np.clip(data, -25.0, 0.0)
        return (clipped + 25.0) / 25.0

    def _precip_normalize(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Normalizza precipitazioni con clipping al percentile 99.
        """
        if name in self.stats:
            vmax = max(self.stats[name]["p99"], 1.0)  # almeno 1mm
        else:
            vmax = max(data.max(), 1.0)

        clipped = np.clip(data, 0, vmax)
        return clipped / vmax

    def normalize_tensor(self, tensor: np.ndarray, channel_names: list = None) -> np.ndarray:
        """
        Normalizza un intero tensore (C, H, W).

        Args:
            tensor: shape (C, H, W)
            channel_names: lista dei nomi canali

        Returns:
            np.ndarray normalizzato (C, H, W)
        """
        if channel_names is None:
            channel_names = CHANNEL_NAMES

        result = np.zeros_like(tensor, dtype=np.float32)
        for i, name in enumerate(channel_names):
            if i < tensor.shape[0]:
                self.fit(tensor[i], name)
                result[i] = self.normalize(tensor[i], name)

        return result

    def get_stats(self) -> dict:
        """Restituisce le statistiche calcolate."""
        return self.stats

    def save_stats(self, path: str):
        """Salva le statistiche su file JSON."""
        import json
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def load_stats(self, path: str):
        """Carica statistiche da file JSON."""
        import json
        with open(path) as f:
            self.stats = json.load(f)
