"""Land cover exposure metric with Gaussian distance decay.

Computes a continuous exposure value for each raster cell based on its
proximity to a specified land cover type. Cells within the target land cover
receive a maximum exposure value; cells outside decay smoothly with distance
using a Gaussian profile, reaching zero beyond a specified radius.
"""
import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GeometryTypeError

from .base import Metric
from .proximity import Proximity

logger = logging.getLogger(__name__)

class LandCover(Metric):
    """Spatial metric computing proximity-weighted exposure to a land cover type.

    For each raster cell, exposure is ``1.0`` inside the target land cover and
    decays smoothly with distance outside it, reaching ``0.0`` beyond ``radius``.
    The decay profile is Gaussian with the radius corresponding to approximately
    three standard deviations.

    If ``column`` and ``value`` are both ``None``, all geometries in the input
    are treated as the target land cover.

    Attributes:
        radius: Influence radius in CRS units beyond which exposure is zero.
        column: Column in the input GeoDataFrame used to filter land cover categories.
            ``None`` if all geometries are used.
        value: Category value within ``column`` identifying the target land cover.
            ``None`` if all geometries are used.
        min_inside: Minimum exposure value assigned to cells within the target
            land cover. Defaults to ``1.0``.
        name: Metric name derived from the title, column, value, and radius.
    """
    metric_title = "land_cover"

    def __init__(
            self,
            radius: float,
            column: str | None = None,
            value: str | float | int | None = None,
            min_inside: float = 1.0,
    ) -> None:
        """Initialise a LandCover metric.

        Args:
            radius: Influence radius in CRS units. Exposure decays to zero at this
                distance from the target land cover boundary.
            column: Column in the input GeoDataFrame identifying land cover categories.
                Must be set together with ``value``, or both must be ``None``.
            value: Category value within ``column`` identifying the target land cover.
                Must be set together with ``column``, or both must be ``None``.
            min_inside: Minimum exposure value for cells within the target land cover.
                Defaults to ``1.0``.

        Raises:
            ValueError: If ``radius`` is negative.
            ValueError: If exactly one of ``column`` and ``value`` is ``None``.
        """
        super().__init__()
        if radius < 0.0:
            raise ValueError("radius must be >= 0")
        if (column is None) ^ (value is None):
            raise ValueError("Both 'column' and 'value' must be set, or both must be None.")
        self.radius = radius
        self.column = column
        self.value = value
        self.min_inside = min_inside
        self.name = self.get_name(self.value, self.radius)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.column, self.value, self.radius

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
        """Compute proximity-weighted land cover exposure for each raster cell.

        Determines which raster cells lie within the target land cover geometry,
        then applies a Gaussian distance decay to cells outside it via
        :func:`add_distance_decay`.

        Polygon and MultiPolygon geometries use exact containment; LineString and
        MultiLineString geometries treat cells within half a cell diagonal as inside.

        Args:
            gdf_input: Land cover polygons or lines used to determine the target area.
            gdf_raster: Raster grid defining the evaluation cells. Must contain
                ``cx`` and ``cy`` columns for cell centroids.

        Returns:
            Series of exposure values in the range ``[0.0, 1.0]``, one per row
            in ``gdf_raster``.

        Raises:
            GeometryTypeError: If the target geometry type is not polygon- or
                line-like.
        """
        proximity_metric = Proximity(self.column, self.value)
        distances = proximity_metric.calculate(gdf_input, gdf_raster)

        if self.column is not None:
            gdf_sel = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_sel = gdf_input

        geom_types = set(gdf_sel.geometry.geom_type.unique())

        cx = gdf_raster["cx"].to_numpy()
        cy = gdf_raster["cy"].to_numpy()
        dx = np.diff(np.sort(np.unique(cx))).mean()
        dy = np.diff(np.sort(np.unique(cy))).mean()
        cell_size = float((dx + dy) / 2)

        if geom_types <= {"Polygon", "MultiPolygon"}:
            within = np.where(np.isclose(distances, 0.0), 1.0, 0.0)

        # Line-like: treat cells as "within" if centroid is close enough to the line
        elif geom_types <= {"LineString", "MultiLineString"}:
            tol = cell_size / np.sqrt(2.0)  # half-diagonal of a cell
            within = np.where(distances <= tol, 1.0, 0.0)

        else:
            raise GeometryTypeError(f"don't know how to handle {geom_types}")

        gdf_within = gdf_raster.copy()
        gdf_within["within"] = within
        blurred = add_distance_decay(
            gdf_within,
            column="within",
            radius=self.radius,
            min_inside=self.min_inside,
        )
        new_column_name = f"within_blurred_{self.radius}"
        exposure = blurred[new_column_name]
        self.data = pd.Series(exposure.values, index=gdf_raster.index, name=self.name)
        return self.data


def add_distance_decay(
        gdf: gpd.GeoDataFrame,
        column: str,
        radius: float,
        min_inside: float = 1.0,
) -> gpd.GeoDataFrame:
    """Create a distance-decay 'risk' field from a regular grid GeoDataFrame.

    Semantics:
    - Cells where `gdf[column] >= min_inside` are treated as the 'risk area'.
    - Inside that area, the resulting value is at least `min_inside` (never
      reduced below it).
    - Outside the area, values decay smoothly with distance from the area
      using a Gaussian profile, and are set to 0 beyond `radius`.
    - The transition at the boundary is smooth on the outside.

    Args:
        gdf: Regular grid GeoDataFrame (polygons).
        column: Name of the column defining the risk area (e.g. 1 inside, 0 outside).
        radius: Effective influence distance in CRS units (e.g. metres). Risk is
            essentially 0 at this distance.
        min_inside: Minimum value allowed inside the risk area (default 1.0).

    Returns:
        A copy of `gdf` with a new column `<column>_blurred_<radius>`.
    """
    # import deferred for speedup
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    gdf = gdf.copy()
    if "cx" not in gdf.columns or "cy" not in gdf.columns:
        gdf["cx"] = gdf.geometry.centroid.x
        gdf["cy"] = gdf.geometry.centroid.y
    table = gdf.pivot(index="cy", columns="cx", values=column)
    arr = table.to_numpy(dtype=float)  # shape: (ny, nx)
    x_vals = table.columns.to_numpy()
    y_vals = table.index.to_numpy()

    core_mask = arr >= min_inside

    if radius == 0.0:
        result = np.zeros_like(arr, dtype=float)
        result[core_mask] = np.maximum(arr[core_mask], min_inside)
    else:
        dx = np.diff(np.sort(x_vals)).mean()
        dy = np.diff(np.sort(y_vals)).mean()
        cell_size = float((dx + dy) / 2)

        # Distance transform: distance to nearest core cell (outside only)
        base = np.where(core_mask, 0, 1)
        dist_pixels = distance_transform_edt(base)
        dist_real = dist_pixels * cell_size  # in CRS units

        # 5. Gaussian distance-decay outside the core
        # radius is the distance at which we want risk ~ 0.
        # Use radius ≈ 3 * sigma_real.
        sigma_real = radius / 3.0

        # Avoid division warnings at distance 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            decay = np.exp(-0.5 * (dist_real / sigma_real) ** 2)

        decay[dist_real > radius] = 0.0
        result = decay.copy()
        result[core_mask] = np.maximum(arr[core_mask], min_inside)

    # Map result back to the GeoDataFrame
    val_map = {
        (x_vals[ix], y_vals[iy]): result[iy, ix]
        for iy in range(result.shape[0])
        for ix in range(result.shape[1])
    }

    new_col = f"{column}_blurred_{radius}"
    gdf[new_col] = gdf.apply(lambda r: val_map[(r["cx"], r["cy"])], axis=1)

    return gdf.drop(columns=["cx", "cy"])

