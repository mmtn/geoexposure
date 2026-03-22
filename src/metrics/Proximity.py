import pandas as pd

from src.metrics.Metric import Metric
from src.utils import calculate_gdf_proximity, metric_name, proximity_to_risk


class Proximity(Metric):
    NAME = "proximity"

    def __init__(self, column=None, value=None):
        super().__init__()
        self.column = column
        self.value = value
        self.name = metric_name(self.NAME, (self.column, self.value))

    def _calculate_metric(self, gdf_input, gdf_raster):
        if self.column is not None and self.value is not None:
            gdf_to = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_to = gdf_input

        gdf_from = gdf_raster
        proximity = calculate_gdf_proximity(gdf_from, gdf_to)

        self.data = pd.Series(proximity)
        self._calculated = True


class ProximityRisk(Metric):
    NAME = "proximity_risk"

    def __init__(self, column=None, value=None, threshold=None):
        super().__init__()
        self.column = column
        self.value = value
        self.threshold = threshold
        self.name = metric_name(self.NAME, (self.column, self.value))

    def _calculate_metric(self, gdf_input, gdf_raster):
        if self.column is not None and self.value is not None:
            gdf_to = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_to = gdf_input

        gdf_from = gdf_raster
        proximity = calculate_gdf_proximity(gdf_from, gdf_to)
        risk = proximity_to_risk(proximity, self.threshold)

        self.data = pd.Series(risk)
        self._calculated = True
