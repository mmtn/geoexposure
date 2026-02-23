import numpy as np

from src.data.Spatial import SpatialData
from src.utils import raster


class Environment:
    def __init__(
            self,
            spatial_resolution,
            temporal_resolution,
            spatial_data: dict = None,
            temporal_data: dict = None,
            primary_spatial_data=None
    ):
        self.spatial_data = spatial_data
        self.temporal_data = temporal_data
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.primary_spatial_data = primary_spatial_data

        self.gdf_raster = self._calculate_raster()
        self.calculated = False

    def metric_names(self):
        all_metrics = list()
        for name, data in self.spatial_data.items():
            [all_metrics.append(f"{name}_{metric.name}") for metric in data.metrics]
        return all_metrics

    def save(self, filename):
        # TODO: implement Environment.save()
        raise NotImplemented()

    def load(self, filename):
        # TODO: implement Environment.load()
        raise NotImplemented()

    def calculate(self):
        # Populate DataFrame with unique land/metric ids
        for name, data in self.spatial_data.items():
            self._add_metrics_to_gdf(data, name)

        # TODO: test this loop
        # TODO: check metric names are unique
        for name, data in self.temporal_data.items():
            if data.data_type is SpatialData:
                self._add_metrics_to_gdf(data, name)

        self.calculated = True

    def sample(self, timestamp):
        # TODO: handle temporally placed spatial layers
        # TODO: handle end time where window spans multiple temporal data objects
        if not self.calculated:
            print("run calculate() method before sample()")

        if self.temporal_data is None:
            scaling = 1.0
        else:
            scale_factors = [
                data.sample(timestamp)
                for data in self.temporal_data.values()
                if data.temporal_type == float
            ]
            scaling = np.prod(scale_factors)

        # TODO: replace magic 'exposure' string
        gdf_sample = self.gdf_raster.copy()
        gdf_sample["exposure"] = 0.0

        # TODO: access metrics at specific times
        for metric in self.metric_names():
            gdf_sample["exposure"] += self.gdf_raster[metric]

        gdf_sample["exposure"] *= scaling

        return gdf_sample[["geometry", "exposure"]]

    def _add_metrics_to_gdf(self, data, name):
        for metric in data.metrics:
            result = metric.calculate(data.gdf, self.gdf_raster)
            column_name = f"{name}_{metric.name}"
            self.gdf_raster[column_name] = result

    def _calculate_raster(self):
        if self.primary_spatial_data not in self.spatial_data.keys():
            raise ValueError()
        return raster(
            self.spatial_data[self.primary_spatial_data].gdf,
            self.spatial_resolution
        )
