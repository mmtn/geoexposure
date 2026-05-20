"""Land cover fragmentation metric.

Computes the patch density of a specified land cover category within a
circular neighbourhood of each raster cell. Higher values indicate greater
fragmentation — more numerous, smaller patches of the target category within
the neighbourhood.
"""

import logging
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry.polygon import Polygon

from .base import Metric

logger = logging.getLogger(__name__)


class Fragmentation(Metric):
    r"""Spatial metric computing land cover patch density within a fixed radius.

    For each raster cell, counts the number of distinct non-contiguous patches
    of a specified land cover category within a circular neighbourhood and
    normalises by the neighbourhood area:

    .. math::

        \\Phi_k(\\mathbf{x}, r) = \\frac{m_k}{\\pi r^2}

    where :math:`m_k` is the number of distinct patches of category :math:`k`
    within radius :math:`r` of the cell centroid.

    Returns ``0.0`` for cells where the target category covers none or
    effectively all of the neighbourhood, as neither extreme represents
    meaningful fragmentation.

    Attributes:
        column: Name of the column in the input GeoDataFrame used to filter
            land cover categories.
        value: The category value within ``column`` to compute fragmentation for.
        radius: Neighbourhood radius in CRS units.
        name: Metric name derived from the title, column, value, and radius.
    """

    metric_title = "fragmentation"

    def __init__(self, column: str, value: str | float | int, radius: int | float) -> None:
        """Initialise a Fragmentation metric for a specified land cover category.

        Args:
            column: Column in the input GeoDataFrame identifying land cover categories.
            value: The category value to compute fragmentation for.
            radius: Neighbourhood radius in CRS units over which fragmentation
                is computed for each raster cell.
        """
        super().__init__()
        self.column = column
        self.value = value
        self.radius = radius
        self.name = self.get_name(self.value, self.radius)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.column, self.value, self.radius

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Compute fragmentation for each cell in the raster grid.

        Args:
            gdf_input: Land cover polygons containing the column specified
                at initialisation.
            gdf_raster: Raster grid defining the evaluation cells.

        Returns:
            Series of fragmentation values, one per row in ``gdf_raster``.
            Values are non-negative, with higher values indicating greater
            patch density of the target category within the neighbourhood.
        """
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
    """Compute the patch density of a land cover category within a circular neighbourhood.

    Clips the input land cover data to a circle of radius ``r`` centred on
    the centroid of ``patch``, counts the number of distinct non-contiguous
    polygons of the target category, and normalises by the neighbourhood area.

    Returns ``0.0`` when the target category covers none or effectively all of
    the neighbourhood, as fragmentation is not meaningful in either extreme.

    Args:
        gdf: Land cover polygons pre-filtered to the target category.
        patch: Raster cell polygon whose centroid defines the neighbourhood centre.
        r: Neighbourhood radius in CRS units.
        tolerance: Coverage threshold below which the neighbourhood is treated
            as having zero or full coverage. Defaults to ``1e-6``.

    Returns:
        Patch density in units of patches per square CRS unit. Returns ``0.0``
        if coverage is below ``tolerance`` or above ``1 - tolerance``.
    """
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
