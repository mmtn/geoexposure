from abc import ABC, abstractmethod
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from shapely import Point

from src import utils

MAX_NUM_GRID_POINTS = 1e6


@dataclass
class MobilityData:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    dt: np.ndarray
    eval_centroids: np.ndarray
    points: list[Point]
    mask: np.ndarray
    zero_density_gdf: GeoDataFrame


class Mobility(ABC):
    # TODO: subclass for new method
    # TODO: precompute and store spatial distribution(s)
    # TODO: are there issues with few data points at high temporal resolutions?
    # TODO: normalisation options
    # TODO: decide on possible methods for Mobility model

    @abstractmethod
    def distribution(self, trajectory, gdf_geometry):
        # This method is designed to be overridden by subclass implementations.
        pass

    def _prepare_mobility_data(self, trajectory, gdf_geometry, bounds, buffer):
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

        all_centroids = utils.get_gdf_centroids(gdf_geometry, as_numpy=True)
        all_x, all_y = all_centroids.transpose()
        points = [Point(xy) for xy in zip(all_x, all_y)]

        # Default to zero density everywhere
        zero_density = np.zeros(len(all_centroids))
        zero_density_gdf = gpd.GeoDataFrame(
            data={
                "density": zero_density,
                "point_geometry": points,
            },
            geometry=gdf_geometry.geometry,
            crs=gdf_geometry.crs
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
        eval_centroids = all_centroids[mask]

        if len(eval_centroids) > MAX_NUM_GRID_POINTS:
            raise ValueError(f"Too many coordinates (max = {MAX_NUM_GRID_POINTS})")

        return MobilityData(
            x=x,
            y=y,
            t=t,
            dt=dt,
            eval_centroids=eval_centroids,
            points=points,
            mask=mask,
            zero_density_gdf=zero_density_gdf,
        )
