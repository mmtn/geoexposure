import geopandas as gpd
import numpy as np

from .. import Environment, Mobility
from ..data import Trajectory

class PointOverlay(Mobility):

    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def distribution(
        self,
        trajectory: Trajectory,
        environment: Environment,
        bounds=None,
    ) -> gpd.GeoDataFrame:
        """Computes density by counting trajectory points within each polygon.

        Args:
            trajectory: Source of observed positions and times.
            gdf_geometry: GeoDataFrame whose geometry defines the evaluation
                locations.
            bounds: Optional spatial bounds to restrict evaluation.

        Returns:
            GeoDataFrame with columns 'density' and 'point_geometry', and the
            CRS and geometry of gdf_geometry. Density values represent the
            count of trajectory points falling within each polygon.
        """
        buffer = self.buffer
        data = self._get_mobility_data(trajectory, environment, bounds, buffer)

        x, y, t, dt = data.x, data.y, data.t, data.dt
        mask = data.mask
        if len(x) == 0 or not np.any(mask):
            return data.zero_density_gdf

        density = data.zero_density_gdf.density.to_numpy().copy()

        #

        # Build shapely points from trajectory coordinates
        trajectory_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x, y),
            crs=environment.crs,
        )

        # Spatial join — each trajectory point is matched to the polygon it
        # falls within
        joined = gpd.sjoin(
            trajectory_points,
            environment.geometry_polygons[mask].reset_index(names="geometry_index"),
            how="inner",
            predicate="within",
        )

        # Count how many trajectory points landed in each polygon
        counts = joined.groupby("geometry_index").size()

        # Write counts back into the correct positions of the full density array
        density[counts.index] = counts.values

        return gpd.GeoDataFrame(
            data={
                "density": density,
                "point_geometry": environment.geometry_points,
            },
            geometry=environment.geometry_polygons,
            crs=environment.crs,
        )
