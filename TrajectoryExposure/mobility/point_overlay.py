"""Point overlay mobility model.

:class:`PointOverlay` estimates the spatial occupancy distribution of a
trajectory by counting the number of recorded positions falling within each
cell of the environment raster grid. It is the simplest available mobility
model and carries no assumptions about movement between observations.
"""

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np

from ..mobility import Mobility

if TYPE_CHECKING:
    from ..core.environment import Environment
    from ..data import Trajectory

logger = logging.getLogger(__name__)


class PointOverlay(Mobility):
    """Mobility model counting points falling within each grid polygon."""

    def __init__(self, buffer: float) -> None:
        """Initialise PointOverlay instance."""
        super().__init__()
        self.buffer = buffer

    def distribution(
            self,
            trajectory: "Trajectory",
            environment: "Environment",
            bounds: tuple[float, float, float, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """Computes density by counting trajectory points within each polygon.

        Args:
            trajectory: Source of observed positions and times.
            environment: Spatiotemporal exposure environment.
            bounds: Optional spatial bounds to restrict evaluation.

        Returns:
            GeoDataFrame with columns 'density' and 'point_geometry', and the
            CRS and geometry of the environment raster. Density values represent
            the count of trajectory points falling within each polygon.
        """
        data = self._get_mobility_data(trajectory, environment, bounds, self.buffer)

        x, y = data.x, data.y
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf

        density = data.zero_density_gdf.density.to_numpy().copy()

        # Build shapely points from trajectory coordinates
        trajectory_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x, y),
            crs=environment.crs,
        )

        # Spatial join — each trajectory point is matched to the polygon it falls within
        joined = gpd.sjoin(
            trajectory_points,
            environment.gdf_raster.geometry[mask]
            .reset_index()
            .rename(columns={"index": "geometry_index"}),
            how="inner",
            predicate="intersects",
        )

        # Count how many trajectory points landed in each polygon
        counts = joined.groupby("geometry_index").size()

        # Write counts back into the correct positions of the full density array
        density[counts.index] = counts.values

        return gpd.GeoDataFrame(
            data={
                "density"       : density / density.sum(),
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.gdf_raster.geometry,
            crs=environment.crs,
        )
