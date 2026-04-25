"""Estrae il grafo stradale di Roma dal PBF e lo salva come graphml.

Uso:
    uv run manage.py build_graph

Esegui una volta sola dopo aver scaricato il PBF in data/centro-latest.osm.pbf.
"""
import math
from pathlib import Path

import osmnx as ox
from django.core.management.base import BaseCommand

# Stessi valori di views.py: il grafo deve coprire la stessa area della griglia flood
ROME_LAT = 41.8931
ROME_LON = 12.4828
PIXEL_M  = 40
SIZE     = 256

PBF_PATH    = Path("data/centro-latest.osm.pbf")
OUTPUT_PATH = Path("data/rome_graph.graphml")


def matrix_bbox():
    """Bounding box (south, west, north, east) della griglia flood."""
    half_m   = (SIZE * PIXEL_M) / 2
    half_lat = half_m / 111_000
    half_lon = half_m / (111_000 * math.cos(math.radians(ROME_LAT)))
    return (
        ROME_LAT - half_lat,  # south
        ROME_LON - half_lon,  # west
        ROME_LAT + half_lat,  # north
        ROME_LON + half_lon,  # east
    )


class Command(BaseCommand):
    help = "Estrae il grafo stradale di Roma dal PBF Geofabrik."

    def handle(self, *args, **options):
        if not PBF_PATH.exists():
            self.stderr.write(self.style.ERROR(
                f"Manca {PBF_PATH}. Scaricalo da Geofabrik:\n"
                f"  wget -P data https://download.geofabrik.de/europe/italy/centro-latest.osm.pbf"
            ))
            return

        south, west, north, east = matrix_bbox()
        self.stdout.write(f"Bounding box: S={south:.4f} W={west:.4f} N={north:.4f} E={east:.4f}")
        self.stdout.write("Scarico il grafo (la prima volta puo' impiegare 1-2 minuti)...")

        # 'drive' = solo strade percorribili da auto. Cambia in 'walk' per pedonali.
        # OSMnx scarica via Overpass: in alternativa si potrebbe parsare il PBF
        # direttamente con pyrosm, ma per un'area di 100 km^2 Overpass va bene.
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type="drive",
            simplify=True,
        )

        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        ox.save_graphml(G, OUTPUT_PATH)
        self.stdout.write(self.style.SUCCESS(
            f"Grafo salvato in {OUTPUT_PATH} ({len(G.nodes)} nodi, {len(G.edges)} archi)"
        ))
