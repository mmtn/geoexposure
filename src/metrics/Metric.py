from typing import Any

import pandas as pd
from geopandas import GeoDataFrame

from ..Caching import Caching


class Metric(Caching):
    """Base class for spatial metrics evaluated on a raster grid."""

    metric_title = "metric"
    cache_dir = ".cache/metrics"

    def __init__(self):
        self.data = None

    def _hash_params(self) -> tuple:
        """Return metric-specific parameters for cache hashing.

        Subclasses can override to include parameter values in the cache key
        (in addition to the input GeoDataFrames).

        Returns:
            Tuple of hashable parameters.
        """
        return ()

    def get_name(self, *args: Any) -> str:
        joining_str = "_"
        filtered = [f"{arg}" for arg in args if arg is not None]
        arg_string = joining_str.join(filtered) if filtered else None

        if arg_string is None:
            return f"{self.metric_title}"
        else:
            return f"{self.metric_title}_{arg_string}"

    def calculate(
        self,
        gdf_input: GeoDataFrame,
        gdf_raster: GeoDataFrame,
    ) -> pd.Series:
        self.data = self._get_or_compute(
            fn=self._calculate_metric,
            args=(gdf_input, gdf_raster),
            hash_args=self._hash_params(),
            label=self.metric_title,
        )
        return self.data

    def _calculate_metric(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame):
        # Intended to be overridden by subclasses.
        raise NotImplementedError("subclasses must implement _calculate_metric()")
