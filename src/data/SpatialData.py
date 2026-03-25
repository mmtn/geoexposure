from copy import copy as shallow_copy, deepcopy

import pandas as pd
from geopandas import GeoDataFrame


class SpatialData:
    def __init__(self, file, crs=None, metrics=None):
        self.gdf = None
        self.gdf_metrics = None
        self._calculated = False

        self._set_data(file, crs)
        self.metrics = metrics if metrics is not None else dict()

        assert type(self.metrics) is dict, "metrics must be a 'Metric: weight' dict"

        self._metrics_list = list(self.metrics.keys())
        self._metric_weights = list(self.metrics.values())

    def __str__(self):
        df = pd.DataFrame(
            data={
                "metric": [metric.name for metric in self._metrics_list],
                "weight": [weight for weight in self._metric_weights],
                "calculated": [metric._calculated for metric in self._metrics_list],
            },
        )
        if df.empty:
            return "No metrics"
        else:
            return df.to_string(index=False)


    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.gdf = None if self.gdf is None else self.gdf.copy()
        new.gdf_metrics = None if self.gdf_metrics is None else self.gdf_metrics.copy()
        new.metrics = shallow_copy(self.metrics)
        new._metrics_list = list(new.metrics.keys())
        new._metric_weights = list(new.metrics.values())
        new._calculated = self._calculated
        return new

    def calculate(self, gdf_raster):
        self.gdf_metrics = gdf_raster.copy()
        for metric in self._metrics_list:
            self.gdf_metrics[metric.name] = metric.calculate(self.gdf, gdf_raster)
        self._calculated = True

    def _set_data(self, file, crs):
        if not isinstance(file, str):
            raise TypeError("'data' must be 'path/to/shape/file'")
        self.gdf = GeoDataFrame.from_file(file)
        if crs is not None:
            self.gdf = self.gdf.to_crs(epsg=crs)

    def interpolate(self, other, loc):
        if loc < 0 or loc > 1:
            raise ValueError(f"'loc' must between 0 and 1: got {loc}")

        self_scale = 1.0 - loc
        other_scale = loc

        new_gdf_metrics = self.gdf_metrics[["geometry"]].copy()
        new_metrics = dict()

        for metric, weight in self.metrics.items():
            new_name = f"interpolated_{metric.name}"
            new_gdf_metrics[new_name] = self.gdf_metrics[metric.name] * self_scale
            metric_copy = deepcopy(metric)
            metric_copy.name = new_name
            if new_name in new_metrics.keys():
                raise RuntimeError("overwriting existing data during interpolation")
            new_metrics.update({metric_copy: weight})

        for metric, weight in other.metrics.items():
            new_name = f"interpolated_{metric.name}"
            new_gdf_metrics[new_name] = other.gdf_metrics[metric.name] * other_scale
            metric_copy = deepcopy(metric)
            metric_copy.name = new_name
            if new_name in new_metrics.keys():
                raise RuntimeError("overwriting existing data during interpolation")
            new_metrics.update({metric_copy: weight})

        new = self.__class__.__new__(self.__class__)
        new.gdf = None if self.gdf is None else self.gdf.copy()
        new.gdf_metrics = new_gdf_metrics
        new._calculated = True
        new.metrics = new_metrics
        new._metrics_list = list(new.metrics.keys())
        new._metric_weights = list(new.metrics.values())
        return new

    def metric_sum(self):
        m_sum = pd.Series(0.0, index=range(len(self.gdf_metrics)))
        for metric, weight in self.metrics.items():
            m_sum += weight * self.gdf_metrics[metric.name]
        return m_sum

    def set_weights(self, weights: list):
        self.metrics = {
            metric: weight
            for metric, weight in zip(self.metrics.keys(), weights)
        }
        self._metrics_list = list(self.metrics.keys())
        self._metric_weights = list(self.metrics.values())
