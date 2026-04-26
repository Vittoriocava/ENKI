from pathlib import Path
import math
import json
import threading
from datetime import datetime, timezone

import numpy as np
import osmnx as ox
import networkx as nx
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt


# ============================================================
# CONFIG
# ============================================================
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

# Bound per i blob spawnati manualmente
SIGMA_MIN, SIGMA_MAX = 3.0, 9.0
PEAK_MIN,  PEAK_MAX  = 0.05, 1.0

# Parametri default dello spawn manuale
SPAWN_DEFAULT_SIGMA = 6.0
SPAWN_DEFAULT_PEAK  = 0.9

# Costi del routing
FLOOD_PENALTY   = 80     # quanto un edge "allagato" costa di piu'
BLOCK_THRESHOLD = 0.85   # sopra questa intensita' l'edge e' impassable

# Date degli eventi alluvione validati (da generate_dataset.py, feat/dataset)
FLOOD_EVENT_DATES = [
    (2017,  5, 19), (2017,  9,  3), (2017, 11,  5), (2017, 12, 27),
    (2018,  3,  6), (2018,  4,  8), (2018,  7, 23), (2018, 10,  9),
    (2018, 10, 21), (2018, 10, 22), (2018, 11, 20),
    (2019,  5, 12), (2019,  5, 30), (2019,  7, 27), (2019,  8, 25),
    (2019,  9,  2), (2019, 10,  2), (2019, 11, 11), (2019, 12,  2),
    (2020,  9, 23), (2020, 10,  7), (2020, 10, 15),
    (2021,  1,  3), (2021,  1, 23), (2021,  1, 24), (2021,  4, 19),
    (2021,  6,  8), (2021, 11,  8), (2021, 12,  2),
    (2022,  4, 22), (2022,  8,  6), (2022,  8,  9), (2022, 10, 11),
    (2022, 12,  3), (2022, 12, 13),
    (2023,  4, 15), (2023,  6, 11), (2023,  6, 13), (2023,  6, 14),
    (2023, 10, 16), (2023, 10, 24), (2023, 12,  5),
    (2024,  9,  3), (2024,  9, 13), (2024,  9, 25), (2024, 10,  5),
    (2024, 10, 24),
    (2025,  5,  6),# (2025,  7, 13), (2025,  9, 10),
    # (2026,  1,  6), (2026,  1, 28), (2026,  3, 12),
]

# Stato condiviso
_lock         = threading.Lock()
_extra_blobs  = []     # blob aggiunti manualmente con Shift+Click
_GRAPH        = None   # grafo stradale, lazy
_historical_blobs = None  # eventi storici parsati dal GeoJSON


# ============================================================
# UTILITY GEOGRAFICHE
# ============================================================
def _matrix_bounds():
    """Bounding box geografico della griglia."""
    half_m   = (SIZE * PIXEL_M) / 2
    half_lat = half_m / 111_000
    half_lon = half_m / (111_000 * math.cos(math.radians(ROME_LAT)))
    return {
        "south": ROME_LAT - half_lat,
        "north": ROME_LAT + half_lat,
        "west":  ROME_LON - half_lon,
        "east":  ROME_LON + half_lon,
    }


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


# ============================================================
# GRAFO STRADALE (per il routing)
# ============================================================
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


# ============================================================
# EVENTI STORICI -> matrice di flood
# ============================================================
def _load_historical_blobs():
    """Legge il GeoJSON una volta sola e converte ogni feature in un blob."""
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
        props = feat["properties"]
        raggio_m = props.get("raggio_m")
        sigma = raggio_m / PIXEL_M if raggio_m else HISTORICAL_SIGMA
        blobs.append({
            "cy": cy, "cx": cx,
            "sigma": sigma,
            "peak":  HISTORICAL_PEAK,
            "ts_ms": props.get("data_evento"),
        })
    _historical_blobs = blobs
    return blobs


def _build_historical_matrix(year=None, month=None, day=None):
    """Renderizza la matrice degli eventi storici filtrati per data."""
    blobs = _load_historical_blobs()

    if year is not None or month is not None or day is not None:
        def keep(b):
            ts = b.get("ts_ms")
            if ts is None:
                return False
            d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            if year  is not None and d.year  != year:  return False
            if month is not None and d.month != month: return False
            if day   is not None and d.day   != day:   return False
            return True
        blobs = [b for b in blobs if keep(b)]

    yy, xx = np.ogrid[:SIZE, :SIZE]
    matrix = np.zeros((SIZE, SIZE), dtype=np.float32)
    for b in blobs:
        d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
        matrix = np.maximum(matrix, b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2)))
    return matrix


def _current_matrix(year=None, month=None, day=None):
    """Matrice corrente: storica filtrata + blob spawnati manualmente."""
    matrix = _build_historical_matrix(year=year, month=month, day=day).copy()

    with _lock:
        extras = list(_extra_blobs)

    if extras:
        yy, xx = np.ogrid[:SIZE, :SIZE]
        for b in extras:
            d2 = (yy - b["cy"]) ** 2 + (xx - b["cx"]) ** 2
            blob = b["peak"] * np.exp(-d2 / (2 * b["sigma"] ** 2))
            matrix = np.maximum(matrix, blob)

    return matrix


# ============================================================
# HELPER: querystring -> filtro periodo
# ============================================================
def _parse_period(request):
    """Estrae year/month/day dalla querystring. Tutti opzionali."""
    def _opt_int(name):
        v = request.GET.get(name)
        return int(v) if v and v.isdigit() else None
    return _opt_int("year"), _opt_int("month"), _opt_int("day")


# ============================================================
# VIEWS
# ============================================================
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


def flood_data(request):
    year, month, day = _parse_period(request)
    matrix = _current_matrix(year=year, month=month, day=day)
    return JsonResponse({
        "bounds": _matrix_bounds(),
        "size":   SIZE,
        "data":   matrix.round(3).tolist(),
    })


@require_http_methods(["GET"])
def historical_periods(request):
    """Date degli eventi alluvione validati, per il picker giornaliero."""
    return JsonResponse({
        "days": sorted(
            [{"year": y, "month": m, "day": d} for (y, m, d) in FLOOD_EVENT_DATES],
            key=lambda x: (x["year"], x["month"], x["day"]),
            reverse=True,
        ),
    })


# ============================================================
# SPAWN MANUALE (Shift+Click sulla mappa)
# ============================================================
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


@csrf_exempt
@require_http_methods(["POST"])
def clear_spawns(request):
    """Rimuove tutti i blob aggiunti manualmente."""
    with _lock:
        _extra_blobs.clear()
    return JsonResponse({"ok": True})


# ============================================================
# ROUTING (A* sul grafo stradale, costo = lunghezza * (1 + flood * penalty))
# ============================================================
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

    year, month, day = _parse_period(request)

    G = _load_graph()
    bounds = _matrix_bounds()
    matrix = _current_matrix(year=year, month=month, day=day)

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
