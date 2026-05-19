"""Base class and interface for spatial metrics evaluated on a raster grid.

A :class:`Metric` transforms geospatial input data into a
:class:`~pandas.Series` of values aligned to a raster grid, with optional
disk caching via :class:`~TrajectoryExposure.core.cachable.Cachable`.

Concrete subclasses must implement :meth:`Metric._calculate_metric` and
should set :attr:`Metric.metric_title` to a descriptive identifier used in
cache filenames and metric naming.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from geopandas import GeoDataFrame

from ..core.cachable import Cachable

logger = logging.getLogger(__name__)


class Metric(Cachable, ABC):
    """Abstract base class for spatial metrics evaluated on a raster grid.

    Subclasses must implement :meth:`_calculate_metric`, which receives the
    input and raster :class:`~geopandas.GeoDataFrame` instances and returns a
    :class:`~pandas.Series` of metric values aligned to the raster grid.

    Results are cached via the inherited :class:`~TrajectoryExposure.core.cachable.Cachable`
    mixin using a hash derived from the input data and any metric-specific
    parameters returned by :meth:`_hash_params`.

    Attributes:
        metric_title: Short identifier used in cache filenames and metric names.
        cache_dir: Directory used for caching computed metric results.
        name: Set after calling :meth:`calculate`; identifies the metric instance.
        data: Set after calling :meth:`calculate`; holds the computed metric values.
    """

    metric_title = "metric"
    cache_dir = ".cache/metrics"

    def __init__(self) -> None:
        """Abstract base class constructor for Metric."""
        self.name = None
        self.data = None

    def get_name(self, *args: object) -> str:
        """Return a string identifying this metric from its type and arguments.

        Args:
            *args: Values to append to the metric title, separated by underscores.
                ``None`` values are filtered out.

        Returns:
            A string of the form ``"{metric_title}_{arg1}_{arg2}..."``, or
            just ``"{metric_title}"`` if no non-``None`` arguments are provided.
        """
        joining_str = "_"
        filtered = [f"{arg}" for arg in args if arg is not None]
        arg_string = joining_str.join(filtered) if filtered else None

        if arg_string is None:
            return f"{self.metric_title}"
        return f"{self.metric_title}_{arg_string}"

    def calculate(
            self,
            gdf_input: GeoDataFrame,
            gdf_raster: GeoDataFrame,
    ) -> pd.Series:
        """Compute the metric, returning a cached result if available.

        Results are stored to and retrieved from disk using a hash of the
        input GeoDataFrames and any metric-specific parameters from
        :meth:`_hash_params`.

        Args:
            gdf_input: Geospatial input data (e.g. land use polygons, waterways).
            gdf_raster: Raster grid over which the metric is evaluated.

        Returns:
            Series of metric values aligned to the rows of ``gdf_raster``.
        """
        self.data = self._get_or_compute(
            fn=self._calculate_metric,
            args=(gdf_input, gdf_raster),
            hash_args=self._hash_params(),
            label=self.metric_title,
        )
        return self.data

    def _hash_params(self) -> tuple:
        """Return metric-specific parameters to include in the cache hash key.

        Subclasses should override this method to include any parameter values
        that affect the computed result, ensuring that cached results are
        invalidated when parameters change.

        Returns:
            Tuple of hashable parameter values. Returns an empty tuple by default.
        """
        return ()

    @abstractmethod
    def _calculate_metric(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame) -> pd.Series:
        """Compute the metric values over the raster grid.

        Subclasses must implement this method to define the metric calculation.
        It is called by :meth:`calculate` and should not be called directly.

        Args:
            gdf_input: Geospatial input data used to compute the metric.
            gdf_raster: Raster grid defining the evaluation points.

        Returns:
            Series of metric values with one entry per row in ``gdf_raster``.
        """
        ...
