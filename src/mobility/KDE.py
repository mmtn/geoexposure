import geopandas as gpd
import numpy as np
from sklearn.neighbors import KernelDensity

from src import Trajectory, Environment, Mobility


class KDE(Mobility):
    def __init__(self, kernel: str, bandwidth: float):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _get_estimator(
        self, coordinates: np.ndarray, weights: np.ndarray = None
    ) -> KernelDensity:
        x, y = coordinates
        if weights is None:
            weights = np.ones_like(x)
        values = np.vstack([x, y])
        estimator = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        estimator.fit(values.transpose(), sample_weight=weights)
        return estimator

    def distribution(
        self,
        trajectory: Trajectory,
        environment: Environment,
        bounds=None,
    ) -> gpd.GeoDataFrame:

        standard_deviations = 3
        buffer = self.bandwidth * standard_deviations
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        x, y, t, dt = data.x, data.y, data.t, data.dt
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf
        eval_coords = data.eval_coords
        density = data.zero_density_gdf.density.to_numpy().copy()

        #

        coordinates = np.array([x, y])
        estimator = self._get_estimator(coordinates, weights=dt)

        np.seterr(divide="ignore")
        log_scores = estimator.score_samples(eval_coords)
        np.seterr(divide="warn")

        density[mask] = np.exp(log_scores)

        return gpd.GeoDataFrame(
            data={
                "density": density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.geometry_polygons,
            crs=environment.crs,
        )
