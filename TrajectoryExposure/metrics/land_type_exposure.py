"""LandTypeExposure calculates the exposure level between 0.0 and 1.0 to the given land type."""
import logging

logger = logging.getLogger(__name__)

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import GeometryTypeError

from .base import Metric
from .proximity import Proximity


class LandTypeExposure(Metric):
    """LandTypeExposure calculates the exposure level to the given land type."""
    metric_title = "land_type_exposure"

    def __init__(
            self,
            radius: float,
            column: str | None = None,
            value: str | float | int | None = None,
            min_inside: float = 1.0,
    ) -> None:
        """Initialise a LandTypeExposure metric with a proximity effect."""
        super().__init__()
        if radius < 0.0:
            raise ValueError("radius must be >= 0")
        if (column is None) ^ (value is None):
            raise ValueError("Both 'column' and 'value' must be set, or both must be None.")
        self.radius = radius
        self.column = column
        self.value = value
        self.min_inside = min_inside
        self.name = self.get_name(self.column, self.value, self.radius)

    def _hash_params(self) -> tuple:
        """Additional hashing parameters for :class:`Cachable` method _make_hash()."""
        return self.column, self.value, self.radius

    def _calculate_metric(
            self,
            gdf_input: gpd.GeoDataFrame,
            gdf_raster: gpd.GeoDataFrame,
    ) -> pd.Series:
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
    from scipy.ndimage import distance_transform_edt  # import deferred for speedup

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

