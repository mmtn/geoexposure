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
    """Geospatial vector data associated with a set of spatial exposure metrics.

    Wraps a vector dataset loaded from disk and associates it with a mapping of
    :class:`~metrics.base.Metric` objects and their weights. Metrics are evaluated
    on a raster grid via :meth:`calculate` and the weighted sum is accessible via
    :meth:`metric_sum`. Instances can be interpolated between two states to support
    temporally varying spatial data within :class:`~data.temporal.TemporalData`.

    Attributes:
        gdf: The loaded vector GeoDataFrame.
        metrics: Mapping of :class:`~metrics.base.Metric` instances to their weights.
        gdf_metrics: GeoDataFrame of computed metric values, populated after
            :meth:`calculate` is called.
    """

    def __init__(
            self,
            gdf: gpd.GeoDataFrame,
            metrics: Mapping["Metric", int | float] | None = None,
    ) -> None:
        """Initialise a SpatialData instance from a GeoDataFrame.

        Args:
            gdf: GeoDataFrame containing the geospatial data. Must have a
                geometry column and CRS set.
            metrics: Mapping of metric objects to their weights. If ``None``,
                no metrics are attached.

        Raises:
            TypeError: If ``metrics`` is not a dict or ``None``.
            ValueError: If ``gdf`` has no CRS set.
        """
        if not isinstance(metrics, Mapping) and metrics is not None:
            raise TypeError("metrics must be a mapping (e.g. dict) or None")

        self.gdf = gdf.copy()
        self.metrics = self._validate_metrics(metrics)
        self._metrics_list: list[Metric] = list(self.metrics.keys())
        self._metric_weights: list[float] = list(self.metrics.values())

        self.gdf_metrics: gpd.GeoDataFrame = None
        self._calculated: bool = False

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
    def from_file(
            cls,
            filename: str | PathLike[str],
            epsg: int | None = None,
            metrics: Mapping["Metric", int | float] | None = None,
    ) -> "SpatialData":
        """Initialise a SpatialData instance from a file.

        Args:
            filename: Path to a vector dataset readable by GeoPandas
                (e.g. Shapefile, GeoJSON).
            epsg: EPSG code to reproject the geometry. If ``None``, the CRS
                from the input file is used.
            metrics: Mapping of metric objects to their weights. If ``None``,
                no metrics are attached.

        Raises:
            OSError: If the file cannot be read by GeoPandas.
            ValueError: If the EPSG code is invalid or unsupported.
        """
        instance = cls.__new__(cls)
        instance._set_data(filename, epsg)
        instance.metrics = cls._validate_metrics(metrics)
        instance.gdf_metrics = None
        return instance

    @classmethod
    def from_interpolation(cls, a: "SpatialData", b: "SpatialData", loc: float) -> "SpatialData":
        """New instance interpolated between a and b."""
        return a.interpolate(b, loc)

    @staticmethod
    def _validate_metrics(
            metrics: Mapping["Metric", int | float] | None,
    ) -> Mapping["Metric", int | float]:
        """Validate and return the metrics mapping.

        Args:
            metrics: Mapping of metric objects to weights, or ``None``.

        Returns:
            The validated mapping, or an empty dict if ``None``.

        Raises:
            TypeError: If ``metrics`` is not a dict or ``None``.
        """
        if metrics is None:
            return {}
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict or None")
        return metrics

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
