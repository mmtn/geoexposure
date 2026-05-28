"""spatial_utils.py contains functions to perform common tasks related to spatial operations.

Classes and functions in this file handle:
- RasterGrid data
- rasterising input GeoDataFrames on arbitrary grids
- calculating the centroids of GeoDataFrame geometries
"""

import logging

import attrs
import geopandas as gpd
import numpy as np
from shapely import Point, Polygon

logger = logging.getLogger(__name__)


@attrs.frozen
class RasterGrid:
    """Parameters defining a regular raster grid.

    Attributes:
        pixel_size: Edge length of each cell in CRS units.
        x_min: X coordinate of the leftmost cell centroid in CRS units.
        y_min: Y coordinate of the bottommost cell centroid in CRS units.
        n_rows: Number of rows.
        n_cols: Number of columns.
    """
    pixel_size: float
    x_min: float
    y_min: float
    n_rows: int
    n_cols: int

    def to_polygon_gdf(self, crs: str) -> gpd.GeoDataFrame:
        """Construct a regular grid GeoDataFrame from raster parameters.

        Builds a grid of square polygon cells starting from using values from `grid`. Cell corner
        coordinates are derived from the centroid origin. Centroid coordinates are stored in ``cx``
        and ``cy`` columns.

        Args:
            self: :class:`RasterGrid` object with spatial information.
            crs: Coordinate reference system for the output GeoDataFrame.

        Returns:
            GeoDataFrame of square polygon cells with ``cx`` and ``cy`` centroid columns.
        """
        px = self.pixel_size
        x_min = self.x_min - px * 0.5  # convert centroid origin to corner origin
        y_min = self.y_min - px * 0.5
        n_cols = self.n_cols
        n_rows = self.n_rows

        logger.info(
            f"Constructing {n_cols}x{n_rows} raster ({px}m res,  {n_cols * n_rows} points)",
        )

        polys, cx_list, cy_list = [], [], []
        for col in range(n_cols):
            for row in range(n_rows):
                x0 = x_min + px * col
                y0 = y_min + px * row
                x1 = x0 + px
                y1 = y0 + px
                polys.append(Polygon(((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))))
                cx_list.append(x0)
                cy_list.append(y0)

        return gpd.GeoDataFrame({"cx": cx_list, "cy": cy_list}, geometry=polys, crs=crs)

    def to_point_gdf(self, crs: str) -> gpd.GeoDataFrame:
        """Construct a GeoDataFrame of centroid points covering the grid.

        Args:
            crs: Coordinate reference system for the output GeoDataFrame.

        Returns:
            GeoDataFrame of :class:`~shapely.geometry.Point` geometries
            with ``cx`` and ``cy`` centroid columns.
        """
        px = self.pixel_size
        x_min = self.x_min
        y_min = self.y_min
        n_cols = self.n_cols
        n_rows = self.n_rows

        logger.info(
            f"Constructing {n_cols}x{n_rows} raster ({px}m res,  {n_cols * n_rows} points)",
        )

        polys, cx_list, cy_list = [], [], []
        for col in range(n_cols):
            for row in range(n_rows):
                x = x_min + px * col
                y = y_min + px * row
                polys.append(Point(x, y))
                cx_list.append(x + px * 0.5)
                cy_list.append(y + px * 0.5)

        return gpd.GeoDataFrame({"cx": cx_list, "cy": cy_list}, geometry=polys, crs=crs)


def get_gdf_centroids(
        gdf: gpd.GeoDataFrame,
        bounds: tuple[float, float, float, float] | None = None,
        *,
        as_numpy: bool = False,
) -> list[Point] | np.ndarray:
    """Return the centroids of all geometries as list and numpy array."""
    if bounds is not None:
        gdf = gdf.clip(bounds)
    centroids = gdf.geometry.centroid  # GeoSeries[Point]
    if as_numpy:
        return np.column_stack((centroids.x.to_numpy(), centroids.y.to_numpy()))
    return centroids


def rasterise(gdf: gpd.GeoDataFrame, pixel_size_metres: int | float) -> gpd.GeoDataFrame:
    """Return a rasterised version of the input GeoDataFrame at the given resolution.

    Computes a regular grid of square cells covering the bounding box of ``gdf``,
    with cell edges aligned to multiples of ``pixel_size_metres``.

    Args:
        gdf: Input GeoDataFrame whose bounding box defines the raster extent.
        pixel_size_metres: Edge length of each raster cell in CRS units.

    Returns:
        GeoDataFrame of square polygon cells covering the input extent, with
        ``cx`` and ``cy`` columns for cell centroid coordinates.
    """

    def round_down(value: float, precision: float) -> float:
        return np.floor(value / precision) * precision

    def round_up(value: float, precision: float) -> float:
        return np.ceil(value / precision) * precision

    px_m = pixel_size_metres
    gdf_x_min, gdf_y_min, gdf_x_max, gdf_y_max = gdf.total_bounds
    crs = gdf.crs

    x_min = round_down(gdf_x_min, px_m)
    y_min = round_down(gdf_y_min, px_m)
    x_size = round_up(gdf_x_max - gdf_x_min, px_m)
    y_size = round_up(gdf_y_max - gdf_y_min, px_m)
    n_cols = int(x_size / px_m)
    n_rows = int(y_size / px_m)
    grid = RasterGrid(
        pixel_size=px_m,
        x_min=x_min,
        y_min=y_min,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return grid.to_polygon_gdf(crs)


def infer_raster_grid(coordinates: np.ndarray) -> RasterGrid:
    """Infer raster grid parameters from centroid coordinates.

    Args:
        coordinates: Array of shape (n, 2) with columns [x, y].

    Returns:
        RasterGrid inferred from the centroid coordinates.
    """
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    pixel_size = float(np.min(np.diff(unique_x)))
    x_min = float(unique_x.min()) - pixel_size * 0.5
    y_min = float(unique_y.min()) - pixel_size * 0.5
    return RasterGrid(
        pixel_size=pixel_size,
        x_min=x_min,
        y_min=y_min,
        n_rows=len(unique_y),
        n_cols=len(unique_x),
    )


