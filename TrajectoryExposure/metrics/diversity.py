import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import pandas as pd
import shapely

from .base import Metric


class DiversityIndex(Metric):
    metric_title = "diversity_index"

    def __init__(self, radius: int | float):
        super().__init__()
        self.radius = radius
        self.name = self.get_name(self.radius)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return (self.radius,)

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
        gdf_input["geometry"] = gdf_input.geometry.buffer(0)  # quick fix for invalid geometries
        return pd.Series(
            [sidi_patch(gdf_input, polygon, self.radius) for polygon in gdf_raster["geometry"]],
        )


def sidi_patch(gdf: gpd.GeoDataFrame, patch: shapely.Polygon, r: float) -> float:
    patch_center = patch.centroid
    circle = patch_center.buffer(r)
    mask = gpd.GeoDataFrame(geometry=[circle], crs=gdf.crs)
    clipped = gdf.clip(mask)

    if len(clipped) == 0:
        return 0.0

    return 1 - ((clipped.area / clipped.area.sum()) ** 2).sum()
