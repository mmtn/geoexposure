import datetime as dt
import logging

import geopandas as gpd
import numpy as np

from TrajectoryExposure.core.environment import Environment
from TrajectoryExposure.data.trajectory import Trajectory
from TrajectoryExposure.mobility.base import Mobility

logger = logging.getLogger(__name__)


class AdaptiveUncertainty(Mobility):
    """Time-integrated 2D Gaussian density field from observed positions."""

    def __init__(
            self,
            sigma0: float,
            v: float,
            k: float,
            timestep: dt.timedelta,
            eps: float = 1e-6,
            sigma_min: float = 4.0,
    ) -> None:
        """Initialises the Gaussian density model with uncertainty parameters.

        Args:
            sigma0: Base positional uncertainty.
            v: Speed parameter for uncertainty growth.
            k: Scaling factor for uncertainty growth.
            timestep: Time step size for trapezoidal integration.
            eps: Numerical safety floor for variance.
            sigma_min: Minimum standard deviation for visualisation.
        """
        super().__init__()
        self.sigma0 = sigma0
        self.v = v
        self.k = k
        self.timestep = timestep
        self.eps = eps
        self.sigma_min = sigma_min

    @staticmethod
    def _mean_position(
            t: dt.datetime,
            x_values: np.ndarray,
            y_values: np.ndarray,
            time_values: np.ndarray,
    ) -> tuple[float, float]:
        """Linear interpolation of position along the observed path at time t.
        Outside the observed range, returns the nearest endpoint.

        Args:
            t:
            x_values:
            y_values:
            time_values:

        Returns:
            tuple(x, y): mean x and y coordinates
        """
        if t <= time_values[0]:
            return x_values[0], y_values[0]
        if t >= time_values[-1]:
            return x_values[-1], y_values[-1]

        i = np.searchsorted(time_values, t, side="right") - 1
        t0, t1 = time_values[i], time_values[i + 1]

        seconds_since_t0 = (t - t0) / np.timedelta64(1, "s")
        seconds_total = (t1 - t0) / np.timedelta64(1, "s")
        alpha = seconds_since_t0 / seconds_total

        x = x_values[i] + alpha * (x_values[i + 1] - x_values[i])
        y = y_values[i] + alpha * (y_values[i + 1] - y_values[i])
        return x, y

    @staticmethod
    def _delta_t_eff(
            t: dt.datetime,
            time_values: np.ndarray,
    ) -> float:
        """Compute effective time to the nearest observation in seconds.

        This is the minimum of time since the previous observation and time until
        the next observation. It is defined as 0 outside the observed time range.
        """
        if t <= time_values[0] or t >= time_values[-1]:
            return 0.0

        i = np.searchsorted(time_values, t, side="right") - 1
        dt_prev = (t - time_values[i]).total_seconds()
        dt_next = (time_values[i + 1] - t).total_seconds()
        return min(dt_prev, dt_next)

    def _instantaneous_density(
            self,
            t: dt.datetime,
            x_values: np.ndarray,
            y_values: np.ndarray,
            time_values: np.ndarray,
            eval_x: np.ndarray,
            eval_y: np.ndarray,
    ) -> np.ndarray:
        """Evaluate instantaneous 2D density at the provided coordinates.

        Args:
            t: Time at which to evaluate the density.
            x_values: Observed x coordinates.
            y_values: Observed y coordinates.
            time_values: Observation times aligned with x/y arrays.
            eval_x: 1D x coordinates of evaluation points.
            eval_y: 1D y coordinates of evaluation points.

        Returns:
            1D array of density values, one per evaluation point.
        """
        mu_x, mu_y = self._mean_position(t, x_values, y_values, time_values)
        dt_eff = self._delta_t_eff(t, time_values)

        sigma2 = self.sigma0 ** 2 + (self.v * dt_eff / self.k) ** 2
        sigma2 = max(sigma2, self.eps)
        sigma2 = max(sigma2, self.sigma_min ** 2)

        r2 = (eval_x - mu_x) ** 2 + (eval_y - mu_y) ** 2
        norm = 1.0 / (2.0 * np.pi * sigma2)
        return norm * np.exp(-0.5 * r2 / sigma2)

    def distribution(
            self,
            trajectory: Trajectory,
            environment: Environment,
            bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """Compute time-integrated density over the trajectory time range.

        Args:
            trajectory: Observed positions and times.
            environment: Evaluation grid and CRS.
            bounds: Optional spatial bounds to restrict evaluation.

        Returns:
            GeoDataFrame containing a ``density`` column aligned to the
            environment raster geometry.
        """
        standard_deviations = 3
        buffer = 1000  # self.sigma0 * standard_deviations
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        x, y, t, dt_seconds = data.x, data.y, data.t, data.dt
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf
        eval_centroids = data.eval_coords
        density = data.zero_density_gdf.density.to_numpy().copy()

        eval_x, eval_y = eval_centroids.transpose()

        t = list(t)
        t_start, t_end = t[0], t[-1]
        duration = (t_end - t_start) / np.timedelta64(1, "s")
        dt_seconds = self.timestep.total_seconds()
        num_steps = max(2, int(np.ceil(duration / dt_seconds)) + 1)
        times = [t_start + self.timestep * i for i in range(num_steps)]

        # Ensure the final point is exactly t_end
        if times[-1] != t_end:
            times.append(t_end)

        for step in range(num_steps - 1):
            t0, t1 = times[step], times[step + 1]
            dt_seconds = (t1 - t0) / np.timedelta64(1, "s")
            z0 = self._instantaneous_density(t0, x, y, t, eval_x, eval_y)
            z1 = self._instantaneous_density(t1, x, y, t, eval_x, eval_y)
            density[mask] += 0.5 * (z0 + z1) * dt_seconds  # trapezoidal integration

        return gpd.GeoDataFrame(
            data={
                "density"       : density / density.sum(),
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.gdf_raster.geometry,
            crs=environment.crs,
        )
