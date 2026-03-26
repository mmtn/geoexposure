import datetime as dt

import geopandas as gpd
import numpy as np
from tqdm import tqdm

from .. import Environment, Mobility
from ..data import Trajectory


class DensityModel(Mobility):
    """
    Computes a time-integrated 2D Gaussian density field from a sequence of
    observed (x, y, t) positions, returned as a GeoDataFrame
    """

    def __init__(
        self,
        sigma0: float,
        v: float,
        k: float,
        timestep: dt.timedelta,
        eps: float = 1e-6,
        sigma_min: float = 4.0,
    ):
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

    # Helper functions
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
        """
        Effective time to nearest observation in seconds:
        minimum of time since previous observation and time until next.
        Zero outside the observed time range.
        """
        if t <= time_values[0] or t >= time_values[-1]:
            return 0.0

        i = np.searchsorted(time_values, t, side="right") - 1
        dt_prev = (t - time_values[i]).total_seconds()
        dt_next = (time_values[i + 1] - t).total_seconds()
        return min(dt_prev, dt_next)

    def _instantaneous_density(
        self,
        t: float,
        x_values: np.ndarray,
        y_values: np.ndarray,
        time_values: np.ndarray,
        eval_x: np.ndarray,
        eval_y: np.ndarray,
    ) -> np.ndarray:
        """
        Instantaneous 2D density evaluated at the provided coordinates

        Parameters
        ----------
        t : float
            Time at which to evaluate the density.
        x_values, y_values, time_values : np.ndarray
            Arrays of observed data.
        eval_x, eval_y : np.ndarray
            1D arrays of evaluation point coordinates.

        Returns
        -------
        np.ndarray
            1D array of density values, one per evaluation point.
        """
        mu_x, mu_y = self._mean_position(t, x_values, y_values, time_values)
        dt_eff = self._delta_t_eff(t, time_values)

        sigma2 = self.sigma0**2 + (self.v * dt_eff / self.k) ** 2
        sigma2 = max(sigma2, self.eps)
        sigma2 = max(sigma2, self.sigma_min**2)

        r2 = (eval_x - mu_x) ** 2 + (eval_y - mu_y) ** 2
        norm = 1.0 / (2.0 * np.pi * sigma2)
        return norm * np.exp(-0.5 * r2 / sigma2)

    def distribution(
        self,
        trajectory: Trajectory,
        environment: Environment,
        bounds=None,
    ) -> gpd.GeoDataFrame:
        """
        Time-integrated density over the trajectory's time range.

        Parameters
        ----------
        trajectory : Trajectory
            Observed positions and times.
        bounds : optional
            Passed to utils.get_gdf_centroids to spatially restrict evaluation.

        Returns
        -------
        gpd.GeoDataFrame
            Contains 'density' column from evaluation, 'point_geometry', and the CRS
            and geometry of gdf_geometry.
        """
        standard_deviations = 3
        buffer = self.sigma0 * standard_deviations
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        x, y, t, dt_seconds = data.x, data.y, data.t, data.dt
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf
        eval_centroids = data.eval_coords
        density = data.zero_density_gdf.density.to_numpy().copy()

        #

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

        for step in tqdm(range(num_steps - 1), desc="Density calculation"):
            t0, t1 = times[step], times[step + 1]
            dt_seconds = (t1 - t0) / np.timedelta64(1, "s")
            z0 = self._instantaneous_density(t0, x, y, t, eval_x, eval_y)
            z1 = self._instantaneous_density(t1, x, y, t, eval_x, eval_y)
            density[mask] += 0.5 * (z0 + z1) * dt_seconds  # trapezoidal integration

        return gpd.GeoDataFrame(
            data={
                "density": density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.geometry_polygons,
            crs=environment.crs,
        )
