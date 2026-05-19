"""Abstract base class and shared data structures for mobility models.

:class:`Mobility` defines the interface that all concrete mobility models must
implement. The static helper :meth:`~Mobility._get_mobility_data` prepares the
trajectory arrays and raster evaluation coordinates shared by all
implementations. :class:`MobilityData` is an immutable data carrier holding
these prepared arrays.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import attrs
import geopandas as gpd
import numpy as np
from shapely import Point

from ..data.columns import DATETIME, DWELL_TIME_SECONDS, X, Y

if TYPE_CHECKING:
    from ..core.environment import Environment
    from ..data.trajectory import Trajectory

logger = logging.getLogger(__name__)
MAX_NUM_GRID_POINTS = 1e6


@attrs.frozen
class MobilityData:
    """Arrays and evaluation coordinates for mobility model computation."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    dt: np.ndarray
    eval_coords: np.ndarray
    points: list[Point]
    mask: np.ndarray
    zero_density_gdf: gpd.GeoDataFrame


class Mobility(ABC):
    """Abstract base class for spatial mobility models.

    Subclasses must implement :meth:`distribution`, which transforms a
    :class:`~..data.trajectory.Trajectory` into a normalised spatial density distribution over the
    grid defined by an  :class:`~..core.environment.Environment`.

    The static helper :meth:`_get_mobility_data` prepares the arrays and evaluation coordinates
    shared by all concrete implementations.
    """
    # TODO: testing - are there issues with few data points at high temporal resolutions?

    @abstractmethod
    def distribution(
            self,
            trajectory: "Trajectory",
            environment: "Environment",
            bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """Compute a normalised spatial density distribution for a trajectory.

        Subclasses must return a :class:`~geopandas.GeoDataFrame` with the
        same geometry and CRS as ``environment.gdf_raster``, containing at
        minimum a ``density`` column whose values sum to 1.

        Args:
            trajectory: Input trajectory providing observed positions and times.
            environment: Spatiotemporal environment defining the evaluation grid.
            bounds: Optional ``(x_min, y_min, x_max, y_max)`` to restrict the
                region of the grid that is evaluated. If ``None``, bounds are
                inferred from the trajectory extent.

        Returns:
            GeoDataFrame with a normalised ``density`` column aligned to the
            environment raster grid.
        """
        ...

    @staticmethod
    def _get_mobility_data(
            trajectory: "Trajectory",
            environment: "Environment",
            bounds: tuple[float, float, float, float] | None,
            buffer: float,
    ) -> MobilityData:
        """Prepare cached arrays and evaluation coordinates for mobility models.

        Args:
            trajectory: Input trajectory.
            environment: Environment providing the evaluation grid.
            bounds: Optional (x_min, y_min, x_max, y_max) bounds to limit evaluation.
                If not provided, computed from the trajectory extent plus buffer.
            buffer: Padding in coordinate units applied when inferring bounds from the trajectory.

        Returns:
            A :class:`MobilityData` instance containing trajectory arrays, the
            masked subset of grid coordinates to evaluate, and a zero-density
            GeoDataFrame as a default return value.

        Raises:
            ValueError: If the number of evaluation coordinates exceeds
                ``MAX_NUM_GRID_POINTS``.
        """
        # Default to zero density everywhere
        all_x, all_y = environment.centroids_np.transpose()
        zero_density = np.zeros(len(all_x))
        zero_density_gdf = gpd.GeoDataFrame(
            data={
                "density"       : zero_density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.gdf_raster.geometry,
            crs=environment.crs,
        )

        x = trajectory.df[X].to_numpy(dtype=float)
        y = trajectory.df[Y].to_numpy(dtype=float)
        t = trajectory.df[DATETIME].to_numpy(dtype=np.datetime64)
        dt = trajectory.df[DWELL_TIME_SECONDS].to_numpy(dtype=float)

        # Compute bounds from trajectory extent if not provided
        if bounds is None:
            bounds = (x.min() - buffer, y.min() - buffer, x.max() + buffer, y.max() + buffer)

        # Use bounds to get the subset of centroids to evaluate
        x_min, y_min, x_max, y_max = bounds
        mask = (all_x >= x_min) & (all_x <= x_max) & (all_y >= y_min) & (all_y <= y_max)
        eval_coords = environment.centroids_np[mask]

        if len(eval_coords) > MAX_NUM_GRID_POINTS:
            raise ValueError(f"Too many coordinates (max = {MAX_NUM_GRID_POINTS})")

        return MobilityData(
            x=x,
            y=y,
            t=t,
            dt=dt,
            eval_coords=eval_coords,
            points=environment.geometry_points,
            mask=mask,
            zero_density_gdf=zero_density_gdf,
        )
