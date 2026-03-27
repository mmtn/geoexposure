from typing import Any

from .Metric import Metric

import geopandas as gpd


class Fragmentation(Metric):
    metric_title = "fragmentation"

    def __init__(self, column: str | None = None, value: Any | None = None):
        super().__init__()
        self.column = column
        self.value = value
        self.name = self.get_name(self.column, self.value)

    def _hash_params(self) -> tuple:
        return self.column, self.value

    def _calculate_metric(
        self, gdf_input: gpd.GeoDataFrame, gdf_raster: gpd.GeoDataFrame
    ):
        # TODO: implement Fragmentation.calculate()
        raise NotImplemented()
