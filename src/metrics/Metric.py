from collections import abc
from typing import Any

import pandas as pd
from geopandas import GeoDataFrame


class Metric:
    metric_title = "metric"

    def __init__(self):
        self._calculated = False
        self.data = None

    def get_name(self, *args: Any) -> str:
        joining_str = "_"
        filtered = [f"{arg}" for arg in args if arg is not None]
        arg_string = joining_str.join(filtered) if filtered else None

        if arg_string is None:
            return f"{self.metric_title}"
        else:
            return f"{self.metric_title}_{arg_string}"

    def calculate(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame) -> pd.Series:
        if not self._calculated:
            self._calculate_metric(gdf_input, gdf_raster)
        return self.data

    def _calculate_metric(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame):
        # Intended to be overridden by subclasses.
        raise NotImplementedError("subclasses must implement _calculate_metric()")
