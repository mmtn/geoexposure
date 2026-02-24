import numpy as np

from src.data.SpatialData import SpatialData
from src.utils import get_cyclic_timestamp, raster


class Environment:
    EXPOSURE_COLUMN = "exposure"

    def __init__(
            self,
            spatial_resolution,
            spatial_data: dict = None,
            temporal_data: dict = None,
            spatial_data_ref=None
    ):
        self.spatial_data = spatial_data
        self.temporal_data = temporal_data
        self.spatial_resolution = spatial_resolution
        self.spatial_data_ref = spatial_data_ref
        self.gdf_raster = self._calculate_raster()
        self._calculated = False
        self._set_temporal_resolution()

    def save(self, filename):
        # TODO: implement Environment.save()
        raise NotImplemented()

    def load(self, filename):
        # TODO: implement Environment.load()
        raise NotImplemented()

    def calculate(self):
        # Populate DataFrame with unique land/metric ids
        for name, data in self.spatial_data.items():
            data.calculate(self.gdf_raster)

        for name, temporal in self.temporal_data.items():
            if temporal.data_type is not SpatialData:
                continue
            for ii, spatial in enumerate(temporal.data):
                # TODO: set this as an object property to avoid duplicating this logic
                spatial.calculate(self.gdf_raster)
                # prefix = f"{name}_{ii:03d}"
                # self._add_metrics_to_gdf(spatial, prefix)

        self._calculated = True

    def sample(self, timestamp, to="nearest"):
        """

        :param timestamp:
        :param to:
        :return:
        """
        if not self._calculated:
            print("run 'calculate()' before 'sample()'")

        # Create new GeoDataFrame with zero exposure
        gdf_sample = self.gdf_raster.copy()
        gdf_sample[self.EXPOSURE_COLUMN] = 0.0

        # Add all metrics from spatial only metrics
        for key, spatial in self.spatial_data.items():
            for metric in spatial.metrics:
                gdf_sample[self.EXPOSURE_COLUMN] += spatial.gdf_metrics[metric.name]

        # Add spatial metrics at given timestamp
        for key, temporal in self.temporal_data.items():
            if temporal.data_type is not SpatialData:
                continue
            spatial = temporal.sample(timestamp, to=to)
            for metric in spatial.metrics:
                gdf_sample[self.EXPOSURE_COLUMN] += spatial.gdf_metrics[metric.name]

        # Scale by any time dependent factors
        gdf_sample[self.EXPOSURE_COLUMN] *= self._scaling_factors(timestamp, to=to)

        return gdf_sample[["geometry", self.EXPOSURE_COLUMN]]

    def _scaling_factors(self, timestamp, to="nearest") -> float:
        if self.temporal_data is None:
            return 1.0
        elif all(d.data_type is SpatialData for d in self.temporal_data.values()):
            return 1.0
        else:
            scaling_factors = [
                temporal.sample(timestamp, to=to)
                for temporal in self.temporal_data.values()
                if temporal.data_type == float
            ]
            return np.prod(scaling_factors)

    def _add_metrics_to_gdf(self, data, name):
        for metric in data.metrics:
            result = metric.calculate(data.gdf, self.gdf_raster)
            column_name = f"{name}_{metric.name}"
            if column_name in self.gdf_raster.columns:
                raise ValueError(f"column {column_name} already present")
            self.gdf_raster[column_name] = result

    def _calculate_raster(self):
        return raster(
            self.spatial_data_ref.gdf,
            self.spatial_resolution
        )

    def _set_temporal_resolution(self):
        if self.temporal_data is None:
            self.temporal_resolution = None
        else:
            self.temporal_resolution = np.min(
                [
                    data.temporal_resolution
                    for data in self.temporal_data.values()
                ]
            )
