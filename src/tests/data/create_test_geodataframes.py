import os
import geopandas as gpd
from shapely import Polygon, affinity


def write_shape_file(gdf, path):
    if os.path.exists(path):
        files = os.listdir(path)
        for file in files:
            os.remove(os.path.join(path, file))
        os.removedirs(path)
    gdf.to_file(path)


CATEGORY_COLUMN = "land_use"
CATEGORIES = ["forest", "fields", "built-up", "plantations"]
custom_crs = "+proj=tmerc +lat_0=0 +lon_0=0 +k_0=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

x_size = 1000  # metres
y_size = 1000  # metres

polygon = Polygon(
    [
        (-1, -1),
        (-1, +1),
        (+1, +1),
        (+1, -1),
        (-1, -1),
    ]
)

polygon = affinity.scale(polygon, x_size / 2, y_size / 2)
polygon = affinity.translate(polygon, xoff=x_size / 2, yoff=y_size / 2)

geometries = [
    affinity.translate(polygon, xoff=ii * x_size)
    for ii in range(len(CATEGORIES))
]

gdf_original = gpd.GeoDataFrame(
    data={CATEGORY_COLUMN: CATEGORIES},
    geometry=geometries,
    crs=custom_crs,
)
write_shape_file(gdf_original, "gis/side_by_side")

for category in CATEGORIES:
    gdf_copy = gdf_original.copy()
    gdf_copy = gdf_copy[gdf_copy[CATEGORY_COLUMN] == category]
    # gdf_copy = gdf_copy.dissolve()
    write_shape_file(gdf_copy, f"gis/{category}")
