import pandas as pd
from geopandas import GeoDataFrame


class Metric:
    def __init__(self):
        self._calculated = False
        self.data = None

    def calculate(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame) -> pd.Series:
        if not self._calculated:
            self._calculate_metric(gdf_input, gdf_raster)
        return self.data

    def _calculate_metric(self, gdf_input: GeoDataFrame, gdf_raster: GeoDataFrame):
        # Intended to be overridden by subclasses.
        raise NotImplementedError("subclasses must implement _calculate_metric()")
