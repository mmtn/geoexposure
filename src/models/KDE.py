import geopandas as gpd
import numpy as np
from shapely import Point
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

from src.models.Mobility import Mobility
from src import utils

MAX_NUM_GRID_POINTS = 1e6


class KDE(Mobility):
    def __init__(self, kernel, bandwidth):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _get_estimator(self, coordinates, weights):
        x, y = coordinates
        if weights is None: weights = np.ones_like(x)
        values = np.vstack([x, y])
        estimator = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        estimator.fit(values.transpose(), sample_weight=weights)
        return estimator

    def distribution(
            self,
            trajectory,
            gdf_geometry,
            weights=None,
            bounds=None
    ):
        latitude = trajectory["latitude"]
        longitude = trajectory["longitude"]
        weights = trajectory["dwell_time_seconds"]
        coordinates = np.array([latitude, longitude])

        evaluation_coordinates = utils.get_gdf_centroids(gdf_geometry, bounds)
        x, y = evaluation_coordinates.transpose()
        xy_pairs = list(zip(x, y))
        geometry = [Point(xy) for xy in xy_pairs]

        if len(evaluation_coordinates) > MAX_NUM_GRID_POINTS:
            raise ValueError(f"Too many coordinates (max = {MAX_NUM_GRID_POINTS}")

        if coordinates.shape[1] == 0:
            return gpd.GeoDataFrame(
                data={"density": np.zeros_like(x)},
                geometry=geometry,
                crs=gdf_geometry.crs
            )

        estimator = self._get_estimator(coordinates, weights)
        np.seterr(divide="ignore")
        log_scores = estimator.score_samples(evaluation_coordinates)
        np.seterr(divide="warn")
        scores = np.exp(log_scores)

        return gpd.GeoDataFrame(
            data={"density": scores},
            geometry=geometry,
            crs=gdf_geometry.crs
        )
