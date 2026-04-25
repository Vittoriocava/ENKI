from pathlib import Path
import math
import random
import threading
import time
import json

import numpy as np
import osmnx as ox
import networkx as nx
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

# --- Config ---
ROME_LAT   = 41.8931
ROME_LON   = 12.4828
PIXEL_M    = 40
SIZE       = 256
REFRESH_MS = 100
SIM_TICK_S = 1.0

GRAPH_PATH = Path("data/rome_graph.graphml")

_GRAPH = None
def _load_graph():
    global _GRAPH
    if _GRAPH is None:
        if not GRAPH_PATH.exists():
            raise RuntimeError(
                f"Grafo non trovato in {GRAPH_PATH}. "
                f"Esegui: uv run manage.py build_graph"
            )
        _GRAPH = ox.load_graphml(GRAPH_PATH)
    return _GRAPH


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
        "center_lat":   ROME_LAT,
        "center_lon":   ROME_LON,
        "zoom":         13,
        "coverage_km2": round(coverage_km2, 1),
        "pixel_m":      PIXEL_M,
        "refresh_ms":   REFRESH_MS,
    })


# ============================================================
# MOCKUP FLOOD STATE (uguale a prima)
# ============================================================
_blobs = []
_last_tick_at = None
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
    action = random.choices(
        ["stay", "move", "grow", "shrink", "intensify", "weaken", "disappear"],
        weights=[78,    10,    3,      3,         2,          2,        2],
    )[0]
    if action == "stay":      return True
    if action == "move":
        b["cy"] += random.randint(-4, 4)
        b["cx"] += random.randint(-4, 4)
        _clamp_in_bounds(b); return True
    if action == "grow":
        b["sigma"] = min(SIGMA_MAX, b["sigma"] + random.uniform(0.3, 0.8))
        _clamp_in_bounds(b); return True
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
    global _blobs, _last_tick_at
    with _lock:
        now = time.monotonic()

        if not _blobs:
            _blobs = [_spawn_blob() for _ in range(random.randint(2, 5))]
            _last_tick_at = now
            return list(_blobs)

        elapsed = now - (_last_tick_at or now)
        n_ticks = int(elapsed // SIM_TICK_S)
        if n_ticks <= 0:
            return list(_blobs)

        _last_tick_at = (_last_tick_at or now) + n_ticks * SIM_TICK_S
        n_ticks = min(n_ticks, 60)

        for _ in range(n_ticks):
            _blobs = [b for b in _blobs if _evolve_blob(b)]
            if random.random() < 0.20:
                _blobs.append(_spawn_blob())
            if not _blobs:
                _blobs.append(_spawn_blob())

        return list(_blobs)

def _build_matrix(blobs):
    yy, xx = np.ogrid[:SIZE, :SIZE]
    matrix = np.zeros((SIZE, SIZE), dtype=np.float32)
    for b in blobs:
        d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
        matrix = np.maximum(matrix, b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2)))
    return matrix


def _current_matrix():
    with _lock:
        blobs_snapshot = list(_blobs)
    return _build_matrix(blobs_snapshot)


def flood_data(request):
    blobs = _step_state()
    matrix = _build_matrix(blobs)
    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix.round(3).tolist(),
    })

SPAWN_DEFAULT_SIGMA = 6.0
SPAWN_DEFAULT_PEAK  = 0.9

@csrf_exempt
@require_http_methods(["POST"])
def spawn_flood(request):
    """Inserisce manualmente un blob alla posizione (lat, lon) indicata.

    Body JSON: { "lat": ..., "lon": ..., "sigma": optional, "peak": optional }
    """
    try:
        body = json.loads(request.body)
        lat = float(body["lat"])
        lon = float(body["lon"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "missing or invalid lat/lon"}, status=400)

    sigma = float(body.get("sigma", SPAWN_DEFAULT_SIGMA))
    peak  = float(body.get("peak",  SPAWN_DEFAULT_PEAK))
    sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    peak  = max(PEAK_MIN,  min(PEAK_MAX,  peak))

    # Lat/lon -> coordinate griglia
    bounds = _matrix_bounds()
    if not (bounds["south"] <= lat <= bounds["north"] and
            bounds["west"]  <= lon <= bounds["east"]):
        return JsonResponse({"error": "point outside monitored area"}, status=400)

    ny = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    nx = (lon - bounds["west"])  / (bounds["east"]  - bounds["west"])
    cy = max(0, min(SIZE - 1, int(ny * SIZE)))
    cx = max(0, min(SIZE - 1, int(nx * SIZE)))

    blob = {"cy": cy, "cx": cx, "sigma": sigma, "peak": peak}
    _clamp_in_bounds(blob)

    with _lock:
        _blobs.append(blob)

    return JsonResponse({"ok": True, "blob": blob})

FLOOD_PENALTY   = 80
BLOCK_THRESHOLD = 0.85

def _flood_at(matrix, lat, lon, bounds):
    if not (bounds["south"] <= lat <= bounds["north"] and
            bounds["west"]  <= lon <= bounds["east"]):
        return 0.0
    ny = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    nx = (lon - bounds["west"])  / (bounds["east"]  - bounds["west"])
    y = max(0, min(SIZE - 1, int(ny * SIZE)))
    x = max(0, min(SIZE - 1, int(nx * SIZE)))
    return float(matrix[y, x])


def _edge_flood_cost(G, u, v, k, matrix, bounds):
    edge = G.edges[u, v, k]
    length = edge.get("length", 1.0)

    lat_u, lon_u = G.nodes[u]["y"], G.nodes[u]["x"]
    lat_v, lon_v = G.nodes[v]["y"], G.nodes[v]["x"]
    mid_lat = (lat_u + lat_v) / 2
    mid_lon = (lon_u + lon_v) / 2
    f = _flood_at(matrix, mid_lat, mid_lon, bounds)

    if f >= BLOCK_THRESHOLD:
        return float("inf")
    return length * (1 + f * FLOOD_PENALTY)


@require_http_methods(["GET"])
def route(request):
    try:
        a_lat = float(request.GET["a_lat"])
        a_lon = float(request.GET["a_lon"])
        b_lat = float(request.GET["b_lat"])
        b_lon = float(request.GET["b_lon"])
    except (KeyError, ValueError):
        return JsonResponse({"error": "missing or invalid coordinates"}, status=400)

    G = _load_graph()
    bounds = _matrix_bounds()
    matrix = _current_matrix()

    src_node = ox.distance.nearest_nodes(G, X=a_lon, Y=a_lat)
    dst_node = ox.distance.nearest_nodes(G, X=b_lon, Y=b_lat)

    def heuristic(n1, n2):
        y1, x1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
        y2, x2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
        # ~ metri per gradi alla latitudine di Roma
        dy = (y1 - y2) * 111_000
        dx = (x1 - x2) * 111_000 * math.cos(math.radians(ROME_LAT))
        return math.hypot(dx, dy)

    def weight(u, v, edge_dict):
        return min(
            _edge_flood_cost(G, u, v, k, matrix, bounds)
            for k in edge_dict
        )

    try:
        node_path = nx.astar_path(G, src_node, dst_node,
                                  heuristic=heuristic, weight=weight)
    except nx.NetworkXNoPath:
        return JsonResponse({"error": "no path"}, status=404)

    coords = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = min(G.get_edge_data(u, v).values(),
                        key=lambda d: d.get("length", float("inf")))
        if "geometry" in edge_data:
            coords.extend([(lat, lon) for lon, lat in edge_data["geometry"].coords])
        else:
            coords.append((G.nodes[u]["y"], G.nodes[u]["x"]))
            coords.append((G.nodes[v]["y"], G.nodes[v]["x"]))

    return JsonResponse({"path": coords})
