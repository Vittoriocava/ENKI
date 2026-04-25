import math
import numpy as np
from config import ROME_LAT, ROME_LON, PIXEL_SIZE, GRID_SIZE
from scipy.interpolate import RegularGridInterpolator

grid_points = 5

half_m = (GRID_SIZE * PIXEL_SIZE) / 2.0
half_lat = half_m / 111_000.0
half_lon = half_m / (111_000.0 * math.cos(math.radians(ROME_LAT)))

# Row 0 is North, Row 511 is South
lats = np.linspace(ROME_LAT + half_lat, ROME_LAT - half_lat, grid_points)
lons = np.linspace(ROME_LON - half_lon, ROME_LON + half_lon, grid_points)

print(lats)
print(lons)
