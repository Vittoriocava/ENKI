from pathlib import Path
import math
import json
import threading

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
SIZE       = 512
REFRESH_MS = 100

GRAPH_PATH      = Path("data/rome_graph.graphml")
HISTORICAL_PATH = Path("data/historical_events_rome.geojson")

# Parametri visivi degli eventi storici sulla heatmap
HISTORICAL_SIGMA = 5.0   # raggio del blob in pixel (~200 m con PIXEL_M=40)
HISTORICAL_PEAK  = 0.85  # intensita' "danger" ma non al massimo

_lock = threading.Lock()
_extra_blobs = []   # blob aggiunti manualmente con Shift+Click

SIGMA_MIN, SIGMA_MAX = 3.0, 9.0
PEAK_MIN,  PEAK_MAX  = 0.05, 1.0

# --- Carico il grafo una volta sola all'avvio ---
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
# HISTORICAL EVENTS -> matrice di flood
# ============================================================
# Carica una volta sola, calcola la matrice una volta sola: e' statica.

_historical_blobs = None  # lista di {cy, cx, sigma, peak} in coordinate griglia
_historical_matrix = None  # np.ndarray (SIZE, SIZE) gia' renderizzata

def _latlon_to_grid(lat, lon, bounds):
    """Converte (lat, lon) in (riga, colonna) della griglia. None se fuori."""
    if not (bounds["south"] <= lat <= bounds["north"] and
            bounds["west"]  <= lon <= bounds["east"]):
        return None
    ny = (bounds["north"] - lat) / (bounds["north"] - bounds["south"])
    nx = (lon - bounds["west"])  / (bounds["east"]  - bounds["west"])
    cy = max(0, min(SIZE - 1, int(ny * SIZE)))
    cx = max(0, min(SIZE - 1, int(nx * SIZE)))
    return cy, cx


def _load_historical_blobs():
    """Legge il GeoJSON e converte ogni feature in un blob della griglia."""
    global _historical_blobs
    if _historical_blobs is not None:
        return _historical_blobs

    if not HISTORICAL_PATH.exists():
        raise RuntimeError(f"Manca {HISTORICAL_PATH}")

    with open(HISTORICAL_PATH, encoding="utf-8") as f:
        geojson = json.load(f)

    bounds = _matrix_bounds()
    blobs = []
    for feat in geojson["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        pos = _latlon_to_grid(lat, lon, bounds)
        if pos is None:
            continue   # eventi fuori dalla griglia (es. Civitavecchia, Bracciano)
        cy, cx = pos
        blobs.append({
            "cy": cy, "cx": cx,
            "sigma": HISTORICAL_SIGMA,
            "peak":  HISTORICAL_PEAK,
        })
    _historical_blobs = blobs
    return blobs


def _build_historical_matrix():
    """Renderizza una sola volta la matrice cumulativa degli eventi storici."""
    global _historical_matrix
    if _historical_matrix is not None:
        return _historical_matrix

    blobs = _load_historical_blobs()
    yy, xx = np.ogrid[:SIZE, :SIZE]
    matrix = np.zeros((SIZE, SIZE), dtype=np.float32)
    for b in blobs:
        d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
        matrix = np.maximum(matrix, b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2)))
    _historical_matrix = matrix
    return matrix


# ============================================================
# MOCKUP DI SIMULAZIONE EVOLUTIVA - DISATTIVATO
# ============================================================
# La generazione di blob casuali e la loro evoluzione (move/grow/shrink/...)
# e' stata sostituita con la lettura statica degli eventi storici di Roma
# (CittaClima.it). Lascio il codice qui sotto commentato come riferimento per
# riattivarlo se serve una simulazione dinamica.
#
# import random
# import threading
# import time
#
# _blobs = []
# _lock  = threading.Lock()
# _last_tick_at = None
# SIM_TICK_S = 1.0
# SIGMA_MIN, SIGMA_MAX = 3.0, 9.0
# PEAK_MIN,  PEAK_MAX  = 0.05, 1.0
#
# def _spawn_blob():
#     sigma  = random.uniform(SIGMA_MIN, SIGMA_MAX)
#     margin = int(3 * sigma) + 1
#     return {
#         "cy":    random.randint(margin, SIZE - margin),
#         "cx":    random.randint(margin, SIZE - margin),
#         "sigma": sigma,
#         "peak":  random.uniform(0.4, 1.0),
#     }
#
# def _clamp_in_bounds(b):
#     margin = int(3 * b["sigma"]) + 1
#     b["cy"] = max(margin, min(SIZE - margin, b["cy"]))
#     b["cx"] = max(margin, min(SIZE - margin, b["cx"]))
#
# def _evolve_blob(b):
#     action = random.choices(
#         ["stay", "move", "grow", "shrink", "intensify", "weaken", "disappear"],
#         weights=[78,    10,    3,      3,         2,          2,        2],
#     )[0]
#     if action == "stay":      return True
#     if action == "move":
#         b["cy"] += random.randint(-4, 4)
#         b["cx"] += random.randint(-4, 4)
#         _clamp_in_bounds(b); return True
#     if action == "grow":
#         b["sigma"] = min(SIGMA_MAX, b["sigma"] + random.uniform(0.3, 0.8))
#         _clamp_in_bounds(b); return True
#     if action == "shrink":
#         b["sigma"] = max(SIGMA_MIN, b["sigma"] - random.uniform(0.3, 0.8))
#         return True
#     if action == "intensify":
#         b["peak"] = min(PEAK_MAX, b["peak"] + random.uniform(0.05, 0.15))
#         return True
#     if action == "weaken":
#         b["peak"] -= random.uniform(0.05, 0.15)
#         return b["peak"] >= PEAK_MIN
#     return False
#
# def _step_state():
#     global _blobs, _last_tick_at
#     with _lock:
#         now = time.monotonic()
#         if not _blobs:
#             _blobs = [_spawn_blob() for _ in range(random.randint(2, 5))]
#             _last_tick_at = now
#             return list(_blobs)
#         elapsed = now - (_last_tick_at or now)
#         n_ticks = int(elapsed // SIM_TICK_S)
#         if n_ticks <= 0:
#             return list(_blobs)
#         _last_tick_at = (_last_tick_at or now) + n_ticks * SIM_TICK_S
#         n_ticks = min(n_ticks, 60)
#         for _ in range(n_ticks):
#             _blobs = [b for b in _blobs if _evolve_blob(b)]
#             if random.random() < 0.20: _blobs.append(_spawn_blob())
#             if not _blobs:             _blobs.append(_spawn_blob())
#         return list(_blobs)


def _current_matrix():
    """Matrice corrente: storica + eventuali blob spawnati manualmente."""
    matrix = _build_historical_matrix().copy()

    with _lock:
        extras = list(_extra_blobs)

    if extras:
        yy, xx = np.ogrid[:SIZE, :SIZE]
        for b in extras:
            d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
            blob = b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2))
            matrix = np.maximum(matrix, blob)

    return matrix


def flood_data(request):
    matrix = _current_matrix()
    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix.round(3).tolist(),
    })

SPAWN_DEFAULT_SIGMA = 6.0
SPAWN_DEFAULT_PEAK  = 0.9


def _clamp_blob_in_bounds(b):
    margin = int(3 * b["sigma"]) + 1
    b["cy"] = max(margin, min(SIZE - margin, b["cy"]))
    b["cx"] = max(margin, min(SIZE - margin, b["cx"]))


@csrf_exempt
@require_http_methods(["POST"])
def spawn_flood(request):
    """Aggiunge un blob alla posizione (lat, lon)."""
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

    bounds = _matrix_bounds()
    pos = _latlon_to_grid(lat, lon, bounds)
    if pos is None:
        return JsonResponse({"error": "point outside monitored area"}, status=400)

    cy, cx = pos
    blob = {"cy": cy, "cx": cx, "sigma": sigma, "peak": peak}
    _clamp_blob_in_bounds(blob)

    with _lock:
        _extra_blobs.append(blob)

    return JsonResponse({"ok": True, "blob": blob})

# ============================================================
# SPAWN MANUALE - mantiene la possibilita' di aggiungere blob al volo
# ============================================================
# Per ora disabilitato: la matrice e' statica. Se vuoi riattivarlo, dovrai
# rimettere _blobs e _lock dalla sezione commentata sopra e fare overlay.
#
# @csrf_exempt
# @require_http_methods(["POST"])
# def spawn_flood(request):
#     return JsonResponse({"error": "disabled in historical mode"}, status=400)


# ============================================================
# ROUTING (A* sul grafo, costo = lunghezza * (1 + flood * penalty))
# ============================================================
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


def _nearest_node(G, lat, lon):
    cos_lat = math.cos(math.radians(lat))
    best_node, best_d2 = None, float("inf")
    for n, d in G.nodes(data=True):
        dy = (d["y"] - lat) * 111_000
        dx = (d["x"] - lon) * 111_000 * cos_lat
        d2 = dy * dy + dx * dx
        if d2 < best_d2:
            best_d2, best_node = d2, n
    return best_node


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

    src_node = _nearest_node(G, a_lat, a_lon)
    dst_node = _nearest_node(G, b_lat, b_lon)

    def heuristic(n1, n2):
        y1, x1 = G.nodes[n1]["y"], G.nodes[n1]["x"]
        y2, x2 = G.nodes[n2]["y"], G.nodes[n2]["x"]
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
