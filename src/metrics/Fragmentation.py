from src.metrics.Metric import Metric
from src.utils import metric_name


class Fragmentation(Metric):
    def __init__(self, column=None, value=None):
        super().__init__()
        self.column = column
        self.value = value
        self.name = metric_name("fragmentation", (self.column, self.value))

    def calculate(self, spatial_data, gdf_raster):
        # TODO: implement Fragmentation.calculate()
        raise NotImplemented()
