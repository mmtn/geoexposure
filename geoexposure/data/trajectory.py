"""Trajectory class and utilities for time-ordered GPS point observations.

A :class:`Trajectory` is a time-ordered sequence of spatial observations
(x, y, datetime) representing the recorded path of a single participant.
This module provides tools for loading, resampling, and computing dwell
times from trajectory data, with several strategies for handling gaps in
the record via :class:`GapMethod`.
"""


import datetime as dt
import logging
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ..core.enums import GapMethod
from .columns import (
    DATETIME,
    DWELL_BACKWARD,
    DWELL_FORWARD,
    DWELL_TIME_SECONDS,
    REQUIRED_COLS_LIST,
    REQUIRED_COLUMNS,
    X,
    Y,
)
from .resampling import SamplingMethod, get_resampler
from .spatial import SpatialData
from .temporal import TimeDeltaLike

logger = logging.getLogger(__name__)


class Trajectory:
    """Time-ordered point observations."""

    def __init__(
            self,
            df: pd.DataFrame,
            source_id: str | None = None,
            pd_datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initialise a new Trajectory from a DataFrame."""
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            raise ValueError(f"DataFrame must have columns {', '.join(REQUIRED_COLUMNS)!r}")

        df = df.copy()
        df[DATETIME] = pd.to_datetime(df[DATETIME], format=pd_datetime_format, errors="raise")
        self.df = df
        self.source_id = source_id
        self.gap_method = None

    def __len__(self) -> int:
        """Return the length (number of points) in the Trajectory."""
        return len(self.df)

    def copy(self, *, deep: bool = True) -> "Trajectory":
        """Return a (deep) copy of this Trajectory."""
        new = self.__class__.__new__(self.__class__)
        new.df = self.df.copy(deep=deep)
        new.source_id = self.source_id
        new.gap_method = self.gap_method
        return new

    @classmethod
    def from_csv(
            cls,
            filename: str | Path,
            source_id: str | None = None,
    ) -> "Trajectory":
        """Initialise a new Trajectory directly from a CSV file with the required columns."""
        path = Path(filename)
        df = pd.read_csv(filename)
        source_id = path.name if source_id is None else source_id
        return cls(df, source_id=source_id)

    @classmethod
    def _empty_trajectory(cls, df: pd.DataFrame, source_id: str | None = None) -> "Trajectory":
        full = cls(df, source_id)
        empty = full.copy()
        empty.df = pd.DataFrame({}, columns=full.df.columns)
        return empty

    @property
    def start_time(self) -> pd.Timestamp:
        """Return the first listed time in this Trajectory."""
        return self.df[DATETIME].min()

    @property
    def end_time(self) -> pd.Timestamp:
        """Return the last listed time in this Trajectory."""
        return self.df[DATETIME].max()

    @property
    def duration_in_seconds(self) -> float:
        """Return the total duration of the Trajectory in seconds."""
        time_difference = self.df[DATETIME].max() - self.df[DATETIME].min()
        return time_difference.total_seconds()

    @property
    def coordinates(self) -> np.ndarray:
        """Return the x and y coordinates as a numpy array."""
        return np.array([self.df[X], self.df[Y]])

    @property
    def bounds(self) -> tuple:
        """Return the absolute bounds of the Trajectory."""
        return self.df[X].min(), self.df[X].max(), self.df[Y].min(), self.df[Y].max()

    @property
    def extent(self) -> tuple:
        """Return the spatial extent of the Trajectory in x and y."""
        return self.df[X].max() - self.df[X].min(), self.df[Y].max() - self.df[Y].min()

    def summary(self) -> str:
        """Return a human-readable summary of this Trajectory."""
        return (
            f"Trajectory\n"
            f"  source_id  : {self.source_id}\n"
            f"  points     : {len(self)}\n"
            f"  start      : {self.start_time}\n"
            f"  end        : {self.end_time}\n"
            f"  duration   : {self.end_time - self.start_time}\n"
            f"  x range    : [{self.bounds[0]:.1f}, {self.bounds[1]:.1f}]\n"
            f"  y range    : [{self.bounds[2]:.1f}, {self.bounds[3]:.1f}]\n"
            f"  dwell times: {'True' if self.has_dwell_times() else 'False'}\n"
        )

    def has_dwell_times(self) -> bool:
        """Check if dwell times have been added to this Trajectory yet."""
        missing = [
            col
            for col in [DWELL_TIME_SECONDS, DWELL_BACKWARD, DWELL_FORWARD]
            if col not in self.df.columns
        ]
        return False if missing else True

    def is_within(self, spatial_reference: SpatialData | gpd.GeoDataFrame) -> bool:
        """Return True if this trajectory's spatial extent is entirely within reference bounds.

        Args:
            spatial_reference: Either SpatialData or GeoDataFrame defining the reference boundary.
        """
        gdf = (
            spatial_reference.gdf if isinstance(spatial_reference, SpatialData)
            else spatial_reference
        )

        (ref_x_min, ref_y_min, ref_x_max, ref_y_max) = gdf.total_bounds
        (traj_x_min, traj_x_max, traj_y_min, traj_y_max) = self.bounds

        traj_mins = np.array((traj_x_min, traj_y_min))
        ref_mins = np.array((ref_x_min, ref_y_min))
        traj_maxs = np.array((traj_x_max, traj_y_max))
        ref_maxs = np.array((ref_x_max, ref_y_max))

        return all((traj_mins > ref_mins) & (traj_maxs < ref_maxs))

    def with_dwell_times(
            self,
            gap_method: GapMethod,
            resolution: TimeDeltaLike | None = None,
    ) -> "Trajectory":
        """Apply the selected gap-filling strategy and return the result."""
        from .gap_methods import get_gap_filler  # noqa: PLC0415

        if gap_method is not GapMethod.VORONOI and resolution is None:
            raise ValueError(
                "time resolution must be provided for gap methods other than GapMethod.VORONOI",
            )

        filler = get_gap_filler(gap_method)
        new = filler.fill(self, resolution)
        new.gap_method = gap_method
        return new

    def window(
            self,
            start: dt.datetime,
            end: dt.datetime,
            *,
            include_first: bool = False,
            include_last: bool = False,
    ) -> "Trajectory":
        """Return a new Trajectory containing only points within the time window.

        If dwell times are present, they are clipped to the window boundaries so
        that exposure is not double-counted across adjacent windows. If dwell times
        are not present, a simple time filter is applied.

        Args:
            start: Start of the window (inclusive).
            end: End of the window (inclusive).
            include_first: If ``True`` and a gap method is set, always include the
                first point of the trajectory regardless of its dwell interval.
            include_last: If ``True`` and a gap method is set, always include the
                last point of the trajectory regardless of its dwell interval.

        Raises:
            ValueError: If ``start`` is after ``end``.
        """
        if start > end:
            raise ValueError("start of window must be before end")

        df = self.df.copy()

        if not self.has_dwell_times():
            mask = (df[DATETIME] >= start) & (df[DATETIME] <= end)
            return Trajectory(df[mask].reset_index(drop=True), source_id=self.source_id)

        # Dwell-aware windowing: clip dwell intervals to window boundaries
        datetimes = df[DATETIME]
        dt_backwards = pd.to_timedelta(df[DWELL_BACKWARD], unit="s")
        dt_forwards = pd.to_timedelta(df[DWELL_FORWARD], unit="s")

        interval_start = datetimes - dt_backwards
        interval_end = datetimes + dt_forwards

        clipped_start = interval_start.clip(lower=start)
        clipped_end = interval_end.clip(upper=end)
        clipped_duration = (
            (clipped_end - clipped_start).dt.total_seconds().fillna(0.0).clip(lower=0.0)
        )

        boundary_indices = set()
        if include_first and self.gap_method != GapMethod.IGNORE:
            boundary_indices.add(df.index[0])
        if include_last and self.gap_method != GapMethod.IGNORE:
            boundary_indices.add(df.index[-1])

        mask = (clipped_duration > 0) | df.index.isin(boundary_indices)

        df_mask = df[mask]

        if len(df_mask) == 0:
            return self._empty_trajectory(df, self.source_id)

        window = Trajectory(df=df_mask[REQUIRED_COLS_LIST], source_id=self.source_id)
        window.df[DWELL_TIME_SECONDS] = clipped_duration[mask].values
        window.df[DWELL_BACKWARD] = df_mask[DWELL_BACKWARD].values
        window.df[DWELL_FORWARD] = df_mask[DWELL_FORWARD].values
        return window

    def resample(self, times: list[dt.datetime], method: SamplingMethod) -> "Trajectory":
        """Resample the trajectory at the given times.

        Times outside the trajectory's observed range are dropped with a warning.
        The returned trajectory may therefore contain fewer rows than ``times``

        Args:
            times: Datetimes to sample at.
            method: Resampling strategy; see :class:`~core.enums.SamplingMethod`.

        Returns:
            A new trajectory containing the resampled observations.
        """
        times = pd.Series(times)

        # Warn and drop times outside the trajectory bounds
        t_min = self.df[DATETIME].min()
        t_max = self.df[DATETIME].max()
        out_of_bounds = (times < t_min) | (times > t_max)
        if out_of_bounds.any():
            logger.warning(
                f"{out_of_bounds.sum()} time(s) are outside the trajectory bounds "
                f"[{t_min}, {t_max}] and will be ignored.",
            )
            for t in times[out_of_bounds]:
                logger.debug("  [out of bounds] %s", t)
            times = times[~out_of_bounds].reset_index(drop=True)


        resampler = get_resampler(method)
        df = resampler.resample(self, times)
        return Trajectory(df, source_id=self.source_id)


def read_csv_directory(
        data_directory: str | Path,
        max_files: int | float | None = None,
) -> Sequence[Trajectory]:
    """Read CSV files in a directory into `Trajectory` objects.

    Args:
        data_directory: Directory containing CSV files with at least
            ``datetime``, ``x``, and ``y`` columns.
        max_files: Maximum number of files to read.

    Returns:
        list of Trajectory objects created from each CSV.
    """
    data_directory = Path(data_directory)
    logger.info(f"Searching for CSV files in {data_directory}")
    csv_files = list(data_directory.glob("*.csv"))
    logger.info(f"Loading {len(csv_files)} trajectories")
    return [Trajectory.from_csv(csv) for csv in islice(csv_files, max_files)]
