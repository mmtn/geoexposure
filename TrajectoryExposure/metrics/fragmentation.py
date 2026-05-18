"""Fragmentation."""
import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import pandas as pd
from shapely.geometry.polygon import Polygon

from .base import Metric


class Fragmentation(Metric):
    """Fragmentation."""

    metric_title = "fragmentation"

    def __init__(self, column: str, value: str | float | int, radius: int | float) -> None:
        """Initialise a Fragmentation metric for the selected geometry."""
        super().__init__()
        self.column = column
        self.value = value
        self.radius = radius
        self.name = self.get_name(self.column, self.value, self.radius)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.column, self.value, self.radius

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
        gdf_input["geometry"] = gdf_input.geometry.buffer(0)  # quick fix for invalid geometries
        gdf_to = gdf_input[gdf_input[self.column] == self.value]
        return pd.Series(
            [patch_density(gdf_to, polygon, self.radius) for polygon in gdf_raster["geometry"]],
        )


def patch_density(
        gdf: gpd.GeoDataFrame,
        patch: Polygon,
        r: float,
        tolerance: float = 1e-6,
) -> float:
    """Compute patch density in a circle of radius ``r`` placed at the centroid of ``patch``."""
    patch_center = patch.centroid
    circle = patch_center.buffer(r)
    window_area = circle.area
    mask = gpd.GeoDataFrame(geometry=[circle], crs=gdf.crs)
    clipped = gdf.clip(mask)

    # Proportion of the window covered by the chosen subset
    # i.e. where gdf_input[self.column] == self.value
    focal_union = clipped.union_all()
    coverage = focal_union.area / window_area

    # Fragmentation is zero if area has ZERO or ONLY polygons of the chosen type
    if coverage < tolerance or coverage > 1.0 - tolerance:
        return 0.0

    num_patches = len(clipped.explode(index_parts=False))
    return num_patches / window_area
