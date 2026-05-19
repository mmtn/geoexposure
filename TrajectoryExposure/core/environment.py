"""Environment class combining spatial and temporal exposure data on a raster grid."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ..data.temporal import TemporalData

import geopandas as gpd
import numpy as np
import pandas as pd

from ..data.spatial import SpatialData
from .cachable import Cachable
from .enums import SamplingMethod
from .utils import get_gdf_centroids, rasterise

logger = logging.getLogger(__name__)

class Environment(Cachable):
    """Spatial/temporal exposure sources sampled on a fixed raster."""

    EXPOSURE_COLUMN = "exposure"
    cache_dir = ".cache/environment"

    def __init__(
            self,
            spatial_resolution: int | float,
            spatial_data: dict[str, SpatialData],
            spatial_reference_data: str,
            temporal_data: dict[str, TemporalData] | None = None
        ) -> None:
        """Initialise an Environment from spatial and/or temporal data sources.

        Args:
            spatial_resolution: Edge length of each raster cell in CRS units (metres).
            spatial_data: Named spatial datasets to include in the environment.
            spatial_reference_data: Key from spatial data dict used to select the spatial
                raster extent and to test whether trajectories lie within the domain.
            temporal_data: Optional named temporal datasets to include in the environment.
        """
        self.spatial_resolution = spatial_resolution
        self.spatial_data = spatial_data
        self.spatial_reference_data = spatial_reference_data
        self.temporal_data = temporal_data if temporal_data is not None else None

        self.gdf_raster = self._calculate_raster()
        self.geometry_points, self.centroids_np = get_gdf_centroids(self.gdf_raster)
        self.crs = self.gdf_raster.crs
        self.calculated = False

        self.temporal_resolution = (
            None if self.temporal_data is None
            else np.min(data.temporal_resolution for data in self.temporal_data.values())
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the Environment."""
        spatial = ""
        for name, data in self.spatial_data.items():
            spatial += f"{name}\n"
            spatial += f"{data}\n"

        if self.temporal_data is None:
            return f"Spatial:\n{spatial}"

        temporal = ""
        for name, data in self.temporal_data.items():
            temporal += f"{name}\n"
            for k, v in data._input_dict.items():
                if data.data_type is SpatialData:
                    temporal += f"{k}\n{v}\n\n"
                if data.data_type is float:
                    temporal += f"{k}: {v}\n"
            temporal += "\n"

        return f"Spatial:\n{spatial}\n\nTemporal:\n{temporal}"

    def calculate(self) -> None:
        """Compute all spatial/temporal layers on the raster grid."""
        if self.calculated:
            return

        for spatial in self.spatial_data.values():
            spatial.calculate(self.gdf_raster.copy())

        if self.temporal_data is None:
            self.calculated = True
            return

        for temporal in self.temporal_data.values():
            if temporal.data_type is not SpatialData:
                continue
            for spatial in temporal.data:
                spatial.calculate(self.gdf_raster.copy())

        self.calculated = True

    @property
    def columns(self) -> Sequence[str]:
        """Return column names produced by `sample()` (excluding geometry)."""
        temporal_data = self.temporal_data if self.temporal_data is not None else {}
        columns: list[str] = []

        columns.extend(
            f"{key}_{metric.name}"
            for key, spatial in self.spatial_data.items()
            for metric in spatial.metrics.keys()
        )

        columns.extend(
            f"temporal_{key}"
            for key in temporal_data.keys()
        )

        return columns

    def get_spatial_exposure(self) -> gpd.GeoDataFrame:
        """Return the raster GeoDataFrame with static spatial exposure columns."""
        spatial_total = self.gdf_raster.copy()
        for key, spatial in self.spatial_data.items():
            for metric in spatial.metrics.keys():
                col = f"{key}_{metric.name}"
                spatial_total[col] = spatial.gdf_metrics[metric.name]
        return spatial_total

    def get_temporal_exposure(
        self, timestamp: dt.datetime, method: SamplingMethod = SamplingMethod.NEAREST
    ) -> gpd.GeoDataFrame:
        """Return the raster GeoDataFrame with temporal exposure columns at the given time.

        Args:
            timestamp: The datetime at which to sample temporal data.
            method: Sampling strategy; see :class:`~core.enums.SamplingMethod`.

        Returns:
            GeoDataFrame with one row per raster cell and one column per temporal layer.
        """
        temporal_total = self.gdf_raster.copy()
        temporal_data = self.temporal_data if self.temporal_data is not None else {}

        for key, temporal in temporal_data.items():
            col = f"temporal_{key}"
            data_at_timestamp = temporal.sample(timestamp, method=method)
            if issubclass(temporal.data_type, (float, np.floating)):
                continue  # floats for scaling are used separately in exposure.py
            if temporal.data_type is SpatialData:
                temporal_total[col] = data_at_timestamp.metric_sum()
            else:
                raise TypeError("unknown data type")

        return temporal_total

    def sample(
        self, timestamp: dt.datetime, method:  SamplingMethod = SamplingMethod.NEAREST
    ) -> gpd.GeoDataFrame:
        """Return the full raster GeoDataFrame with all exposure columns at the given time.

        Combines static spatial columns with temporal columns sampled at ``timestamp``.

        Args:
            timestamp: The datetime at which to sample temporal data.
            method: Sampling strategy; see :class:`~core.enums.SamplingMethod`.

        Returns:
            GeoDataFrame with one row per raster cell and columns for every layer.
        """
        if not self.calculated:
            print("run 'calculate()' before 'sample()'")

        gdf_sample = self.gdf_raster.copy()
        spatial = self.get_spatial_exposure()
        temporal = self.get_temporal_exposure(timestamp, method=method)

        # Drop columns to avoid errors in concatenation
        repeated_columns = ("cx", "cy", "geometry")
        spatial.drop(columns=repeated_columns, inplace=True, errors="ignore")
        temporal.drop(columns=repeated_columns, inplace=True, errors="ignore")

        return gpd.GeoDataFrame(
            pd.concat([gdf_sample, spatial, temporal], axis=1),
            geometry="geometry",
            crs=gdf_sample.crs,
        )

    def scaling_at_timestamp(
        self, timestamp: dt.datetime, method: SamplingMethod = SamplingMethod.NEAREST
    ) -> float:
        """Return the scalar temporal scaling factor at the given datetime.

        Args:
            timestamp: The datetime at which to evaluate the scaling.
            method: Sampling strategy; see :class:`~core.enums.SamplingMethod`.

        Returns:
            Scalar multiplier to apply to exposure values at ``timestamp``.
        """
        if self.temporal_data is None:
            return 1.0
        if all(d.data_type is SpatialData for d in self.temporal_data.values()):
            return 1.0
        scaling_factors = [
            temporal.sample(timestamp, method=method)
            for temporal in self.temporal_data.values()
            if issubclass(temporal.data_type, (float, np.floating))
        ]
        return np.prod(scaling_factors)

    def _calculate_raster(self) -> gpd.GeoDataFrame:
        """Build the internal raster GeoDataFrame from the spatial reference data."""
        return self._get_or_compute(
            fn=rasterise,
            args=(self.spatial_reference_data.gdf, self.spatial_resolution),
            label="raster",
        )

    def plot_exposure(
        self,
        timestamp: dt.datetime | None = None,
        *,
        method: SamplingMethod = SamplingMethod.NEAREST,
        **kwargs,
    ) -> Axes:
        """Plot the exposure raster, optionally at a specific datetime.

        Args:
            timestamp: If provided, temporal layers are sampled at this time before
                plotting. If None, only static spatial layers are shown.
            method: Sampling strategy used when ``datetime`` is provided.
            **kwargs: Additional keyword arguments forwarded to the GeoDataFrame
                plot method.
        """
        if not self.calculated:
            logger.warning("run 'calculate()' before 'plot_exposure()'")

        exposure = self.get_spatial_exposure()
        if timestamp is None:
            title = "Spatial data only - no temporal variation shown"
        else:
            exposure += self.get_temporal_exposure(timestamp, method)
            title = str(timestamp)

        if exposure.empty:
            logger.warning("No exposure sources to plot")
            return None

        gdf_plot = self.gdf_raster.copy()
        gdf_plot[self.EXPOSURE_COLUMN] = exposure.sum(axis=1)
        ax = gdf_plot.plot(column=self.EXPOSURE_COLUMN, **kwargs)
        ax.set_title(title)
        return ax

    def plot_reference(self, **kwargs) -> Axes:
        """Wrapper for GeoDataFrame plot method."""
        return self.spatial_reference_data.gdf.plot(**kwargs)
