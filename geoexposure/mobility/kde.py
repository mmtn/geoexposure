"""Kernel density estimation mobility model.

:class:`KDE` estimates the spatial occupancy distribution of a trajectory by
placing a kernel at each recorded position, weighted by dwell time, and
evaluating the resulting density on the environment raster grid. Results are
cached to disk via the :class:`~core.cachable.Cachable` mixin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np

from ..core.cachable import Cachable
from ..mobility import Mobility

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.neighbors import KernelDensity

    from ..core.environment import Environment
    from ..data import Trajectory

logger = logging.getLogger(__name__)


class KDE(Mobility, Cachable):
    """Concrete Mobility class using kernel density estimation to compute occupancy density."""
    MAX_BUFFER_METRES = 1000

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory for KDEs, nested under the base cache directory."""
        return super().cache_dir / "kde"

    def __init__(self, kernel: str, bandwidth: float) -> None:
        """Initialise KDE mobility instance."""
        super().__init__()
        if bandwidth <= 0.0:
            raise ValueError('KDE bandwidth must be >= 0')
        self.kernel = kernel
        self.bandwidth = bandwidth

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.kernel, self.bandwidth

    def __repr__(self) -> str:
        """Human readable representation of KDE object."""
        return f"KDE(kernel={self.kernel!r}, bandwidth={self.bandwidth!r})"

    def _get_estimator(
            self,
            coordinates: np.ndarray,
            weights: np.ndarray | None = None
    ) -> KernelDensity:
        """Create KDE estimator from trajectory coordinates and weights.

        Args:
            coordinates: x and y positions from trajectory
            weights: weight associated with each position (e.g. dwell time)

        Returns:
            KernelDensity object which can be evaluated at arbitrary coordinates.
        """
        # heavy import deferred until it's needed
        from sklearn.neighbors import KernelDensity  # noqa: PLC0415
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
            bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """Compute density from KDE evaluation on rasterised grid from environment.

        This method uses Cachable._get_or_compute() to avoid unnecessary recalculation.

        Args:
            trajectory: input trajectory to create estimator.
            environment: spatial information defines the evaluation grid.
            bounds: optional limits for spatial evaluation.

        Returns:
            GeoDataFrame with `density` column computed from KDE
        """
        # TODO: fix definition of buffer
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
        log_scores = self._get_or_compute(
            fn=estimator.score_samples,
            args=(data.eval_coords,),
            hash_args=(*self._hash_params(), coordinates, data.dt),
            label="kde",
        )
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
