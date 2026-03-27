from abc import ABC, abstractmethod
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import Point

from . import Environment
from .data import Trajectory

MAX_NUM_GRID_POINTS = 1e6


@dataclass
class MobilityData:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    dt: np.ndarray
    eval_coords: np.ndarray
    points: list[Point]
    mask: np.ndarray
    zero_density_gdf: gpd.GeoDataFrame


class Mobility(ABC):
    # TODO: subclass for new method
    # TODO: precompute and store spatial distribution(s)
    # TODO: are there issues with few data points at high temporal resolutions?
    # TODO: normalisation options
    # TODO: decide on possible methods for Mobility model

    @abstractmethod
    def distribution(
        self,
        trajectory: Trajectory,
        environment: Environment,
        bounds: tuple[float, float, float, float] | None = None,
    ):
        """Compute a spatial density distribution for a trajectory."""
        raise NotImplementedError

    @staticmethod
    def _get_mobility_data(
        trajectory: Trajectory,
        environment: Environment,
        bounds: tuple[float, float, float, float] | None,
        buffer: float,
    ) -> MobilityData | None:
        """Prepare cached arrays and evaluation coordinates for mobility models.

        Args:
            trajectory: Input trajectory.
            environment: Environment providing the evaluation grid.
            bounds: Optional (x_min, y_min, x_max, y_max) bounds to limit evaluation.
                If not provided, computed from the trajectory extent plus buffer.
            buffer: Padding applied when inferring bounds from the trajectory.

        Returns:
            Prepared mobility data, or None if the trajectory has no points.
        """

        # Default to zero density everywhere
        all_x, all_y = environment.centroids_np.transpose()
        zero_density = np.zeros(len(all_x))
        zero_density_gdf = gpd.GeoDataFrame(
            data={
                "density": zero_density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.geometry_polygons,
            crs=environment.crs,
        )

        x, y, t, dt = trajectory.get_data_arrays()
        if len(x) == 0:
            return MobilityData(
                x=x,
                y=y,
                t=t,
                dt=dt,
                eval_coords=np.empty((0, 2), dtype=float),
                points=environment.geometry_points,
                mask=np.zeros(len(all_x), dtype=bool),
                zero_density_gdf=zero_density_gdf,
            )

        # Compute bounds from trajectory extent if not provided
        if bounds is None:
            bounds = (
                x.min() - buffer,
                y.min() - buffer,
                x.max() + buffer,
                y.max() + buffer,
            )

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
