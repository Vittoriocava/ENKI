"""Quick test: load one S1 scene with the fixed windowed approach."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pystac_client
import planetary_computer
import rasterio
import rasterio.windows
import numpy as np
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from utils.geo import compute_bbox_wgs84, resample_array_to_grid

bbox_wgs = compute_bbox_wgs84()
print(f"AOI bbox WGS84: {bbox_wgs}")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-1-grd"],
    bbox=list(bbox_wgs),
    datetime="2023-11-01/2023-12-01",
)

items = list(search.items())
print(f"Trovate {len(items)} scene")

# Test with first item
item = items[0]
print(f"\nItem: {item.id}")
print(f"  item.bbox: {item.bbox}")

vv = item.assets["vv"]
geo_crs = RioCRS.from_epsg(4326)

with rasterio.open(vv.href) as src:
    print(f"  COG size: {src.width}x{src.height}")
    
    # Build geo transform from item.bbox
    lon_min, lat_min, lon_max, lat_max = item.bbox
    geo_transform = transform_from_bounds(
        lon_min, lat_min, lon_max, lat_max, src.width, src.height
    )
    print(f"  Geo transform: {geo_transform}")
    
    # Compute read window for AOI
    aoi_lon_min, aoi_lat_min, aoi_lon_max, aoi_lat_max = bbox_wgs
    window = window_from_bounds(
        aoi_lon_min, aoi_lat_min, aoi_lon_max, aoi_lat_max, geo_transform
    )
    print(f"  Raw window: {window}")
    
    # Clip to raster bounds
    window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    print(f"  Clipped window: {window}")
    print(f"  Window size: {int(window.width)}x{int(window.height)} pixels")
    
    # Read windowed data
    data = src.read(1, window=window).astype(np.float32)
    print(f"  Data shape: {data.shape}")
    print(f"  Data range: [{data.min():.1f}, {data.max():.1f}]")
    
    # Compute window transform
    win_transform = rasterio.windows.transform(window, geo_transform)
    print(f"  Win transform: {win_transform}")

# Resample to target grid
print("\nResampling to 512x512 UTM grid...")
resampled = resample_array_to_grid(data, win_transform, geo_crs, method="bilinear")
print(f"  Resampled shape: {resampled.shape}")
print(f"  Resampled range: [{resampled.min():.1f}, {resampled.max():.1f}]")

# Convert to dB
resampled_db = np.where(resampled > 0, 10.0 * np.log10(resampled), -30.0)
print(f"  dB range: [{resampled_db.min():.1f}, {resampled_db.max():.1f}]")

print("\n✓ SUCCESS!")
