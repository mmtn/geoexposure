"""Adaptive uncertainty mobility model using time-integrated 2D Gaussian densities.

:class:`AdaptiveUncertainty` models positional uncertainty as a Gaussian that
grows with the time elapsed since the nearest observation, reflecting the
increasing probability of deviation from the recorded path. The density field
is integrated over the trajectory time range using trapezoidal quadrature.
"""

import datetime as dt
import logging
import typing

import geopandas as gpd
import numpy as np

from geoexposure.core.environment import Environment
from geoexposure.data.trajectory import Trajectory
from geoexposure.mobility.base import Mobility, MobilityData

logger = logging.getLogger(__name__)


class AdaptiveUncertainty(Mobility):
    """Time-integrated 2D Gaussian density field from observed positions."""

    @typing.override
    def __init__(self, sigma0: float, v: float, timestep: dt.timedelta, eps: float = 1e-6) -> None:
        """Initialises the Gaussian density model with uncertainty parameters.

        Args:
            sigma0: Base positional uncertainty.
            v: Speed parameter for uncertainty growth.
            timestep: Time step size for trapezoidal integration.
            eps: Numerical safety floor for variance.
        """
        super().__init__()
        self.sigma0 = sigma0
        self.v = v
        self.timestep = timestep
        self.eps = eps

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
            t: current evaluation time
            x_values: x-position observations in trajectory
            y_values: y-position observations in trajectory
            time_values: time observations in trajectory

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

    @staticmethod
    def _delta_x_eff(
            mu_x,
            mu_y,
            t: dt.datetime,
            data: MobilityData,
    ) -> float:
        """Compute effective time to the nearest observation in seconds.

        This is the minimum of time since the previous observation and time until
        the next observation. It is defined as 0 outside the observed time range.
        """
        if t <= data.t[0] or t >= data.t[-1]:
            return 0.0

        i = np.searchsorted(data.t, t, side="right") - 1
        x_prev, x_next = data.x[i], data.x[i + 1]
        y_prev, y_next = data.y[i], data.y[i + 1]
        distance_prev = np.sqrt((mu_x - x_prev)**2 + (mu_y - y_prev)**2)
        distance_next = np.sqrt((mu_x - x_next)**2 + (mu_y - y_next)**2)
        return min(distance_prev, distance_next)

    def _instantaneous_density(
            self,
            t: dt.datetime,
            eval_x: np.ndarray,
            eval_y: np.ndarray,
            data: MobilityData
            ) -> np.ndarray:
        """Evaluate instantaneous 2D density at the provided coordinates.

        Args:
            t: Time at which to evaluate the density.
            eval_x: 1D x coordinates of evaluation points.
            eval_y: 1D y coordinates of evaluation points.
            data: observed data in Trajectory

        Returns:
            1D array of density values, one per evaluation point.
        """
        mu_x, mu_y = self._mean_position(t, data.x, data.y, data.t)
        dt_eff = self._delta_t_eff(t, data.t)
        dx_eff = self._delta_x_eff(mu_x, mu_y, t, data)

        intrinsic = self.sigma0
        adaptive = max(0.0, (self.v * dt_eff) - dx_eff)

        sigma2 = intrinsic**2 + adaptive**2
        sigma2 = max(sigma2, self.eps)

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
        # TODO: fix definition of buffer
        buffer = 1000  # self.sigma0 * standard_deviations
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        x, t, dt_seconds = data.x, data.t, data.dt
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
            z0 = self._instantaneous_density(t0, eval_x, eval_y, data)
            z1 = self._instantaneous_density(t1, eval_x, eval_y, data)
            density[mask] += 0.5 * (z0 + z1) * dt_seconds  # trapezoidal integration

        return gpd.GeoDataFrame(
            data={
                "density"       : density / density.sum(),
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.gdf_raster.geometry,
            crs=environment.crs,
        )
