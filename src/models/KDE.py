import geopandas as gpd
import numpy as np
from shapely import Point
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from src.data.Trajectory import Trajectory, X, Y
from src.models.Mobility import MAX_NUM_GRID_POINTS, Mobility
from src import utils


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
            trajectory: Trajectory,
            gdf_geometry: gpd.GeoDataFrame,
            bounds=None,
    ) -> gpd.GeoDataFrame:

        standard_deviations = 3
        buffer = self.bandwidth * standard_deviations
        data = self._prepare_mobility_data(trajectory, gdf_geometry, bounds, buffer)

        x, y, t, dt = data.x, data.y, data.t, data.dt
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf

        eval_centroids = data.eval_centroids
        points = data.points
        density = data.zero_density_gdf.density.to_numpy().copy()

        #

        print(f"Evaluating KDE at {len(eval_centroids)} points...")

        coordinates = np.array([x, y])
        estimator = self._get_estimator(coordinates, weights=dt)

        np.seterr(divide="ignore")
        batch_size = 500
        batches = np.array_split(
            eval_centroids,
            max(1, len(eval_centroids) // batch_size)
            )
        log_scores = np.concatenate(
            [
                estimator.score_samples(batch)
                for batch in tqdm(batches, desc=f"KDE (batches of {batch_size} points)")
            ]
        )
        np.seterr(divide="warn")

        density[mask] = np.exp(log_scores)

        return gpd.GeoDataFrame(
            data={
                "density": density,
                "point_geometry": points,
            },
            geometry=gdf_geometry.geometry,
            crs=gdf_geometry.crs,
        )
