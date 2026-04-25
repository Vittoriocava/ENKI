import math
import random
import numpy as np

from django.http import JsonResponse
from django.shortcuts import render

ROME_LAT = 41.8931
ROME_LON = 12.4828
PIXEL_M  = 40
SIZE     = 256

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
    return render(request, "enki_dashboard/map.html", {
        "center_lat": ROME_LAT,
        "center_lon": ROME_LON,
        "zoom": 13,
    })

def flood_data(request):
    yy, xx = np.ogrid[:SIZE, :SIZE]
    matrix = np.zeros((SIZE, SIZE), dtype=np.float32)

    n_blobs = random.randint(2, 5)
    for _ in range(n_blobs):
        sigma  = random.uniform(3, 9)
        margin = int(3 * sigma) + 1
        cy = random.randint(margin, SIZE - margin)
        cx = random.randint(margin, SIZE - margin)
        peak = random.uniform(0.4, 1.0)

        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = peak * np.exp(-d2 / (2 * sigma ** 2))
        matrix = np.maximum(matrix, blob)

    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix.round(3).tolist(),
    })

    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix,
    })
