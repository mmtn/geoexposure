from abc import ABC, abstractmethod
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import Point

from src import Trajectory, Environment

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
        bounds: tuple = None,
    ):
        pass

    @staticmethod
    def _get_mobility_data(
        trajectory: Trajectory, environment: Environment, bounds: tuple, buffer: float
    ) -> MobilityData:
        """

        Args:
            trajectory:
            gdf_geometry:
            bounds:
            buffer:

        Returns:
            MobilityData
        """
        x, y, t, dt = trajectory.get_data_arrays()
        all_x, all_y = environment.centroids_np.transpose()

        # Default to zero density everywhere
        zero_density = np.zeros(len(all_x))
        zero_density_gdf = gpd.GeoDataFrame(
            data={
                "density": zero_density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.geometry_polygons,
            crs=environment.crs,
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
