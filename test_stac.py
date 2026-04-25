import pystac_client
import planetary_computer
import rasterio

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-1-grd"],
    bbox=[12.35, 41.75, 12.65, 42.05],
    limit=1
)
item = list(search.items())[0]
print(item.id)
print(item.assets["vv"].href)
with rasterio.open(item.assets["vv"].href) as src:
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Width/Height:", src.width, src.height)
