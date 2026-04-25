from django.http import JsonResponse
from django.shortcuts import render

import math
import random
import threading
import numpy as np

ROME_LAT = 41.8931
ROME_LON = 12.4828
PIXEL_M  = 40
SIZE     = 256
REFRESH_MS = 1000

def _matrix_bounds():
    half_m   = (SIZE * PIXEL_M) / 2
    half_lat = half_m / 111_000
    half_lon = half_m / (111_000 * math.cos(math.radians(ROME_LAT)))
    return {
        "south": ROME_LAT - half_lat,
        "north": ROME_LAT + half_lat,
        "west":  ROME_LON - half_lon,
        "east":  ROME_LON + half_lon,
    }


def map_view(request):
    coverage_km2 = (SIZE * PIXEL_M / 1000) ** 2
    return render(request, "enki_dashboard/map.html", {
        "center_lat": ROME_LAT,
        "center_lon": ROME_LON,
        "zoom": 13,
        "coverage_km2": round(coverage_km2, 1),
        "pixel_m": PIXEL_M,
        "refresh_ms": REFRESH_MS,
    })

_blobs = []
_lock  = threading.Lock()

SIGMA_MIN, SIGMA_MAX = 3.0, 9.0
PEAK_MIN,  PEAK_MAX  = 0.05, 1.0


def _spawn_blob():
    sigma  = random.uniform(SIGMA_MIN, SIGMA_MAX)
    margin = int(3 * sigma) + 1
    return {
        "cy":    random.randint(margin, SIZE - margin),
        "cx":    random.randint(margin, SIZE - margin),
        "sigma": sigma,
        "peak":  random.uniform(0.4, 1.0),
    }


def _clamp_in_bounds(b):
    margin = int(3 * b["sigma"]) + 1
    b["cy"] = max(margin, min(SIZE - margin, b["cy"]))
    b["cx"] = max(margin, min(SIZE - margin, b["cx"]))


def _evolve_blob(b):
    # RANDOM TOKENS
    action = random.choices(
        ["stay", "move", "grow", "shrink", "intensify", "weaken", "disappear"],
        weights=[78,    10,    3,      3,         2,          2,        2],
    )[0]

    if action == "stay":
        return True

    if action == "move":
        b["cy"] += random.randint(-4, 4)
        b["cx"] += random.randint(-4, 4)
        _clamp_in_bounds(b)
        return True

    if action == "grow":
        b["sigma"] = min(SIGMA_MAX, b["sigma"] + random.uniform(0.3, 0.8))
        _clamp_in_bounds(b)
        return True

    if action == "shrink":
        b["sigma"] = max(SIGMA_MIN, b["sigma"] - random.uniform(0.3, 0.8))
        return True

    if action == "intensify":
        b["peak"] = min(PEAK_MAX, b["peak"] + random.uniform(0.05, 0.15))
        return True

    if action == "weaken":
        b["peak"] -= random.uniform(0.05, 0.15)
        return b["peak"] >= PEAK_MIN

    return False


def _step_state():
    global _blobs
    with _lock:
        if not _blobs:
            _blobs = [_spawn_blob() for _ in range(random.randint(2, 5))]
            return list(_blobs)

        _blobs = [b for b in _blobs if _evolve_blob(b)]

        if random.random() < 0.20:
            _blobs.append(_spawn_blob())

        if not _blobs:
            _blobs.append(_spawn_blob())

        return list(_blobs)


def flood_data(request):
    blobs = _step_state()

    yy, xx = np.ogrid[:SIZE, :SIZE]
    matrix = np.zeros((SIZE, SIZE), dtype=np.float32)
    for b in blobs:
        d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
        matrix = np.maximum(matrix, b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2)))

    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix.round(3).tolist(),
    })
