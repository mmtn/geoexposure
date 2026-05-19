"""Geospatial data representation and metric computation for exposure modelling.

:class:`SpatialData` wraps a vector dataset loaded from disk and associates it
with a set of :class:`~metrics.base.Metric` objects that are evaluated on a
raster grid. Instances can be interpolated between two states to support
temporally varying spatial data within :class:`~data.temporal.TemporalData`.
"""

import logging
from collections.abc import Mapping
from copy import copy as shallow_copy
from copy import deepcopy
from os import PathLike
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..metrics import Metric

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

class SpatialData:
    """Class docstring: representation of geospatial environmental data including metrics."""

    def __init__(
            self,
            filename: str,
            epsg: int | None = None,
            metrics: Mapping["Metric", int | float] | None = None,
    ) -> None:
        """Initialise a SpatialData instance from a file.

        Args:
            filename: Path to a vector dataset readable by GeoPandas (e.g. Shapefile, GeoJSON).
            epsg: EPSG code to reproject the geometry. If None, use the CRS from the input file.
            metrics: Mapping of metric objects to their weights. If None, no metrics are attached.

        Raises:
            TypeError: If `metrics` is not a dict or None.
            OSError: If the file cannot be read by GeoPandas.
            ValueError: If the EPSG code is invalid or unsupported.
        """
        if not isinstance(metrics, Mapping) and metrics is not None:
            raise TypeError("metrics must be a mapping (e.g. dict) or None")

        self.gdf: gpd.GeoDataFrame = None
        self.gdf_metrics: gpd.GeoDataFrame = None
        self._calculated: bool = False

        self._set_data(filename, epsg)
        self.metrics = metrics if metrics is not None else {}
        self._metrics_list: list[Metric] = list(self.metrics.keys())
        self._metric_weights: list[float] = list(self.metrics.values())

    def __str__(self) -> str:
        """Return a string representation of the SpatialData."""
        df = pd.DataFrame(
            data={
                "metric": self._metrics_list,
                "weight": self._metric_weights,
            },
        )
        if df.empty:
            return "No metrics"
        return df.to_string(index=False)

    @classmethod
    def from_interpolation(cls, a: "SpatialData", b: "SpatialData", loc: float) -> "SpatialData":
        """New instance interpolated between a and b."""
        return a.interpolate(b, loc)

    def copy(self) -> "SpatialData":
        """Return a copy of the SpatialData instance."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.gdf = None if self.gdf is None else self.gdf.copy()
        new.gdf_metrics = None if self.gdf_metrics is None else self.gdf_metrics.copy()
        new.metrics = shallow_copy(self.metrics)
        new._metrics_list = list(new.metrics.keys())
        new._metric_weights = list(new.metrics.values())
        new._calculated = self._calculated
        return new

    def calculate(self, gdf_raster: gpd.GeoDataFrame) -> None:
        """Compute all defined metrics."""
        self.gdf_metrics = gdf_raster.copy()
        for metric in self._metrics_list:
            self.gdf_metrics[metric.name] = metric.calculate(self.gdf, gdf_raster)
        self._calculated = True

    def _set_data(self, filename: str | PathLike[str], epsg: int | None = None) -> None:
        """Read geospatial data and optionally transform coordinates."""
        self.gdf = gpd.GeoDataFrame.from_file(filename)
        if epsg is not None:
            self.gdf = self.gdf.to_crs(epsg=epsg)

    def interpolate(self, other: "SpatialData", loc: float) -> "SpatialData":
        """Interpolate metrics between two instances."""
        if loc < 0 or loc > 1:
            raise ValueError(f"'loc' must between 0 and 1: got {loc}")

        new_gdf_metrics = self.gdf_metrics[["geometry"]].copy()
        new_metrics = {}

        self_scale = 1.0 - loc
        other_scale = loc
        _add_interpolated_metrics(self, new_gdf_metrics, new_metrics, self_scale)
        _add_interpolated_metrics(other, new_gdf_metrics, new_metrics, other_scale)

        new = self.__class__.__new__(self.__class__)
        new.gdf = None if self.gdf is None else self.gdf.copy()
        new.gdf_metrics = new_gdf_metrics
        new._calculated = True
        new.metrics = new_metrics
        new._metrics_list = list(new.metrics.keys())
        new._metric_weights = list(new.metrics.values())
        return new

    def metric_sum(self) -> float:
        """Return the weighted sum of all metrics."""
        m_sum = pd.Series(0.0, index=range(len(self.gdf_metrics)))
        for metric, weight in self.metrics.items():
            m_sum += weight * self.gdf_metrics[metric.name]
        return m_sum

    def set_weights(self, weights: list) -> None:
        """Assign new weightings to the metrics for this SpatialData."""
        self.metrics = dict(zip(self.metrics.keys(), weights, strict=True))
        self._metrics_list = list(self.metrics.keys())
        self._metric_weights = list(self.metrics.values())


def _add_interpolated_metrics(
        src: "SpatialData",
        gdf: gpd.GeoDataFrame,
        new_metrics: dict[Any, Any],
        scale: float,
) -> None:
    """Add new metrics to interpolated instance scaled by relative linear position."""
    for metric, weight in src.metrics.items():
        new_name = f"interpolated_{metric.name}"
        gdf[new_name] = src.gdf_metrics[metric.name] * scale
        metric_copy = deepcopy(metric)
        metric_copy.name = new_name
        if new_name in new_metrics.keys():
            raise RuntimeError("overwriting existing data during interpolation")
        new_metrics.update({metric_copy: weight})
