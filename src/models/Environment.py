import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from src.data.SpatialData import SpatialData
from src.utils import get_cyclic_timestamp, raster


class Environment:
    EXPOSURE_COLUMN = "exposure"

    def __init__(
            self,
            spatial_resolution,
            spatial_data: dict = None,
            temporal_data: dict = None,
            spatial_reference_data=None
    ):
        self.spatial_data = spatial_data if spatial_data is not None else None
        self.temporal_data = temporal_data if temporal_data is not None else None
        self.spatial_resolution = spatial_resolution
        self.spatial_reference_data = spatial_reference_data
        self.gdf_raster = self._calculate_raster()
        self._calculated = False
        self._set_temporal_resolution()

    def __str__(self):
        spatial = ""
        for name, data in self.spatial_data.items():
            spatial += f"{name}\n"
            spatial += f"{data}"

        temporal = ""
        for name, data in self.temporal_data.items():
            temporal += f"{name}\n"
            for k, v in data._input_dict.items():
                if data.data_type is SpatialData:
                    temporal += f"{k}\n{v}\n\n"
                if data.data_type is float:
                    temporal += f"{k}: {v}\n"
            temporal += "\n"
        return f"Spatial:\n{spatial}\n\nTemporal:\n{temporal}"

    def save(self, filename):
        # TODO: implement Environment.save()
        # Use a hash over grid, metrics, and other attributes
        raise NotImplemented()

    def load(self, filename):
        # TODO: implement Environment.load()
        raise NotImplemented()

    def calculate(self):
        for name, data in self.spatial_data.items():
            data.calculate(self.gdf_raster.copy())

        if self.temporal_data is None:
            self._calculated = True
            return

        for name, temporal in self.temporal_data.items():
            if temporal.data_type is not SpatialData:
                continue
            for spatial in temporal.data:
                spatial.calculate(self.gdf_raster.copy())

        self._calculated = True

    def get_spatial_exposure(self) -> GeoDataFrame:
        spatial_total = self.gdf_raster.copy()
        for key, spatial in self.spatial_data.items():
            for metric, weight in spatial.metrics.items():
                col = f"{key}_{metric.name}"
                spatial_total[col] = spatial.gdf_metrics[metric.name]
        return spatial_total.drop(columns=["geometry"])

    def get_temporal_exposure(self, timestamp, method="nearest"):
        temporal_total = self.gdf_raster.copy()
        temporal_data = self.temporal_data if self.temporal_data is not None else {}

        for key, temporal in temporal_data.items():
            col = f"temporal_{key}"
            data_at_timestamp = temporal.sample(timestamp, method=method)
            if issubclass(temporal.data_type, (float, np.floating)):
                continue  # floats for scaling are used separately in Exposure.py
            elif temporal.data_type is SpatialData:
                temporal_total[col] = data_at_timestamp.metric_sum()
            else:
                raise TypeError("unknown data type")

        return temporal_total.drop(columns=["geometry"])

    def sample(self, timestamp, method="nearest"):
        if not self._calculated:
            print("run 'calculate()' before 'sample()'")

        gdf_sample = self.gdf_raster.copy()
        spatial = self.get_spatial_exposure()
        temporal = self.get_temporal_exposure(timestamp, method=method)
        merged = gpd.GeoDataFrame(
            pd.concat([gdf_sample, spatial, temporal], axis=1),
            geometry="geometry",
            crs=gdf_sample.crs
        )

        return merged

    def _scaling_factors(self, timestamp, method="nearest") -> float:
        if self.temporal_data is None:
            return 1.0
        elif all(d.data_type is SpatialData for d in self.temporal_data.values()):
            return 1.0
        else:
            scaling_factors = [
                temporal.sample(timestamp, method=method)
                for temporal in self.temporal_data.values()
                if issubclass(temporal.data_type, (float, np.floating))
            ]
            return np.prod(scaling_factors)

    def _calculate_raster(self):
        return raster(
            self.spatial_reference_data.gdf,
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

    def plot(self, datetime=None, method="nearest", **kwargs):
        exposure = self.get_spatial_exposure()
        if datetime is None:
            title = "Spatial data only - no temporal variation shown"
        else:
            exposure += self.get_temporal_exposure(datetime, method)
            title = str(datetime)

        if exposure.empty:
            warnings.warn("No exposure sources to plot")
            return None

        gdf_plot = self.gdf_raster.copy()
        gdf_plot[self.EXPOSURE_COLUMN] = exposure
        ax = gdf_plot.plot(column=self.EXPOSURE_COLUMN, **kwargs)
        ax.set_title(title)
        return ax

    def plot_reference(self, **kwargs):
        ax = self.spatial_reference_data.plot(**kwargs)
        return ax
