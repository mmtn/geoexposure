from typing import Any

from Metric import Metric
from src.utils import metric_name

import geopandas as gpd


class Fragmentation(Metric):
    def __init__(self, column: str | None = None, value: Any | None = None):
        super().__init__()
        self.column = column
        self.value = value
        self.name = metric_name("fragmentation", (self.column, self.value))

    def _calculate_metric(
        self, gdf_input: gpd.GeoDataFrame, gdf_raster: gpd.GeoDataFrame
    ):
        # TODO: implement Fragmentation.calculate()
        raise NotImplemented()
