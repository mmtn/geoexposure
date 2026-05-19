"""Proximity metric computing distances from raster cell centroids to target geometries.

For each cell in a raster grid, the proximity metric returns the distance from
the cell centroid to the nearest edge of a specified set of input geometries.
An optional column and value filter can be used to restrict the target
geometries to a particular land cover category.
"""

import logging

import geopandas as gpd
import pandas as pd

from ..core.utils import get_gdf_centroids
from .base import Metric

logger = logging.getLogger(__name__)

class Proximity(Metric):
    """Spatial metric computing the distance from each raster cell to target geometries.

    Distances are measured from raster cell centroids to the nearest edge of the
    target geometry. If ``column`` and ``value`` are provided, only geometries
    matching that category are used as the target; otherwise all input geometries
    are used.

    Attributes:
        column: Column in the input GeoDataFrame used to filter target geometries.
            ``None`` if all geometries are used.
        value: Category value within ``column`` identifying the target geometries.
            ``None`` if all geometries are used.
        name: Metric name derived from the title, column, and value.
    """
    metric_title = "proximity"

    def __init__(
            self,
            column: str | None = None,
            value: int | float | None = None,
    ) -> None:
        """Initialise a Proximity metric.

        Args:
            column: Column in the input GeoDataFrame identifying land cover categories.
                Must be set together with ``value``, or both must be ``None``.
            value: Category value within ``column`` identifying the target geometries.
                Must be set together with ``column``, or both must be ``None``.

        Raises:
            ValueError: If exactly one of ``column`` and ``value`` is ``None``.
        """
        super().__init__()
        if (column is None) ^ (value is None):
            raise ValueError("Both 'column' and 'value' must be set, or both must be None.")
        self.column = column
        self.value = value
        self.name = self.get_name(self.column, self.value)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.column, self.value

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Calculate the distance from rasterised centroids to the filtered input geometry.

        Applies a filter to the input geometries if ``column`` and ``value`` are defined in this
        instance. Distances are calculated from each centroid to the nearest edge of the input
        geometry.

        Args:
            gdf_input: GeoDataFrame with input geometry
            gdf_raster: rasterised GeoDataFrame

        Returns:
            pd.Series: list of distances
        """
        if self.column is not None:
            if self.column not in gdf_input.columns:
                raise KeyError(f"Column {self.column!r} not found in input GeoDataFrame.")
            if self.value not in gdf_input[self.column].values:
                raise ValueError(f"value '{self.value!r}' not found in '{self.column!r}' column")
            gdf_to = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_to = gdf_input

        gdf_from = gdf_raster
        proximity = calculate_gdf_proximity(gdf_from, gdf_to)

        self.data = pd.Series(proximity)
        return self.data


def calculate_gdf_proximity(gdf_from: gpd.GeoDataFrame, gdf_to: gpd.GeoDataFrame) -> list[float]:
    """Compute the nearest distance from each geometry in ``gdf_from`` to ``gdf_to``.

    Polygon and MultiPolygon geometries in ``gdf_from`` are converted to their
    centroids before distance calculation. Where multiple geometries in ``gdf_to``
    are equidistant from a source point, the minimum distance is used.

    Args:
        gdf_from: GeoDataFrame of source geometries. Polygons are reduced to
            centroids; Point geometries are used directly.
        gdf_to: GeoDataFrame of target geometries to measure distance to.

    Returns:
        List of nearest distances, one per row in ``gdf_from``, in CRS units.

    Raises:
        TypeError: If either input is not a GeoDataFrame with a geometry column.
        ValueError: If ``gdf_from`` geometries are not Points or Polygons.
    """
    if not isinstance(gdf_from, gpd.GeoDataFrame) or not isinstance(gdf_to, gpd.GeoDataFrame):
        raise TypeError("Inputs must be GeoDataFrames with a 'geometry' column")

    geoms_from = gdf_from.geometry

    if geoms_from.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        geoms_from, _ = get_gdf_centroids(gdf_from)
        geoms_from = gpd.GeoSeries(geoms_from, name="geometry")

    if not geoms_from.geom_type.eq("Point").all():
        raise ValueError("'from' geometries must be Points or Polygons convertible to centroids")

    gdf_points = gpd.GeoDataFrame(geometry=geoms_from, crs=gdf_from.crs)

    # Spatial join to find the nearest geometry in gdf_to for each geometry in gdf_from
    joined = gpd.sjoin_nearest(
        gdf_points,
        gdf_to[["geometry"]],
        how="left",
        distance_col="distance",
    )

    # One distance per source row: take minima if there are ties
    distance_minima = (
        joined["distance"]
        .groupby(joined.index)  # group by left index
        .min()
        .reindex(gdf_points.index)  # ensure order & length match gdf_from
    )

    return distance_minima.to_list()
