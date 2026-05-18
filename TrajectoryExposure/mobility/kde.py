from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.neighbors import KernelDensity

import geopandas as gpd
import numpy as np

from TrajectoryExposure.core.cachable import Cachable
from ..data import Trajectory
from TrajectoryExposure.core.environment import Environment
from ..mobility import Mobility
logger = logging.getLogger(__name__)


class KDE(Mobility, Cachable):
    cache_dir = ".cache/kde"
    MAX_BUFFER_METRES = 1000

    def __init__(self, kernel: str, bandwidth: float):
        super().__init__()
        if bandwidth <= 0.0:
            raise ValueError('KDE bandwidth must be >= 0')
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.kernel, self.bandwidth

    def __repr__(self) -> str:
        return f"KDE(kernel={self.kernel!r}, bandwidth={self.bandwidth!r})"

    def _get_estimator(self, coordinates: np.ndarray, weights: np.ndarray = None) -> "KernelDensity":
        from sklearn.neighbors import KernelDensity  # heavy import deferred until it's needed
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
        buffer = min(self.bandwidth * standard_deviations, self.MAX_BUFFER_METRES)
        buffer = 2000
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        if len(data.x) == 0 or not np.any(data.mask) or data.dt.sum() == 0:
            return data.zero_density_gdf
        density = data.zero_density_gdf.density.to_numpy().copy()

        coordinates = np.array([data.x, data.y])
        estimator = self._get_estimator(coordinates, weights=data.dt)

        np.seterr(divide="ignore")
        log_scores = estimator.score_samples(data.eval_coords)
        # log_scores = self._get_or_compute(
        #     fn=estimator.score_samples,
        #     args=(data.eval_coords,),
        #     hash_args=(*self._hash_params(), coordinates, data.dt),
        #     label="kde",
        #     verbose=True,
        # )
        np.seterr(divide="warn")

        density[data.mask] = np.exp(log_scores)

        return gpd.GeoDataFrame(
            data={
                "density": density / density.sum(),
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.gdf_raster.geometry,
            crs=environment.crs,
        )
