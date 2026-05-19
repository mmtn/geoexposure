"""Simpson Diversity Index metric for land cover heterogeneity.

Computes the Simpson Diversity Index (SIDI) for each cell in a raster grid,
measuring the probability that two randomly selected points within a
neighbourhood belong to different land cover categories. Higher values
indicate greater landscape diversity.
"""

import logging

import geopandas as gpd
import pandas as pd
import shapely

from .base import Metric

logger = logging.getLogger(__name__)

class DiversityIndex(Metric):
    r"""Spatial metric computing the Simpson Diversity Index within a fixed radius.

    For each raster cell, the SIDI is calculated over all land cover patches
    within ``radius`` of the cell centroid:

    .. math::

        \\Psi = 1 - \\sum_k \\left( \\frac{A_k}{A_{\\mathrm{total}}} \\right)^2

    where :math:`A_k` is the area of land cover category :math:`k` within the
    neighbourhood and :math:`A_{\\mathrm{total}}` is the total area of all
    categories present.

    Attributes:
        radius: Neighbourhood radius in CRS units used to compute diversity.
        name: Metric name derived from the title and radius.
    """
    metric_title = "diversity_index"

    def __init__(self, radius: int | float) -> None:
        """Initialise a DiversityIndex metric.

        Args:
            radius: Radius in CRS units defining the neighbourhood over which
                diversity is computed for each raster cell.
        """
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
        """Compute the SIDI for each cell in the raster grid.

        Args:
            gdf_input: Land cover polygons used to compute diversity.
            gdf_raster: Raster grid defining the evaluation cells.

        Returns:
            Series of SIDI values, one per row in ``gdf_raster``, in the
            range ``[0, 1)``. A value of ``0`` indicates a single land cover
            type within the neighbourhood; values approaching ``1`` indicate
            high diversity.
        """
        gdf_input["geometry"] = gdf_input.geometry.buffer(0)  # quick fix for invalid geometries
        return pd.Series(
            [sidi_patch(gdf_input, polygon, self.radius) for polygon in gdf_raster["geometry"]],
        )


def sidi_patch(gdf: gpd.GeoDataFrame, patch: shapely.Polygon, r: float) -> float:
    """Compute the Simpson Diversity Index for a single raster cell.

    Clips the input land cover data to a circular neighbourhood of radius
    ``r`` centred on the centroid of ``patch`` and computes the SIDI from
    the relative areas of each land cover category present.

    Args:
        gdf: Land cover polygons with a geometry column and CRS set.
        patch: Raster cell polygon whose centroid defines the neighbourhood centre.
        r: Neighbourhood radius in CRS units.

    Returns:
        SIDI value in the range ``[0, 1)``. Returns ``0.0`` if no land cover
        data is present within the neighbourhood.
    """
    patch_center = patch.centroid
    circle = patch_center.buffer(r)
    mask = gpd.GeoDataFrame(geometry=[circle], crs=gdf.crs)
    clipped = gdf.clip(mask)

    if len(clipped) == 0:
        return 0.0

    return 1 - ((clipped.area / clipped.area.sum()) ** 2).sum()
