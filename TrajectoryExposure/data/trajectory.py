"""The Trajectory module."""
# TODO: extend documentation for Trajectory

import datetime as dt
import logging
from enum import StrEnum, auto
from itertools import islice
from pathlib import Path
from typing import Literal, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from .temporal import TimeDeltaLike
from .spatial import SpatialData
from ..core import utils
from ..core.enums import SamplingMethod

logger = logging.getLogger(__name__)

# Input columns
DATETIME = "datetime"
X = "x"
Y = "y"

# Computed columns
DWELL_TIME_SECONDS = "dwell_time_seconds"
DWELL_FORWARD = "_dwell_forward"
DWELL_BACKWARD = "_dwell_backward"

REQUIRED_COLUMNS = (DATETIME, X, Y)
REQUIRED_COLS_LIST = list(REQUIRED_COLUMNS)
DWELL_COLUMNS = (DWELL_TIME_SECONDS, DWELL_FORWARD, DWELL_BACKWARD)


class GapMethod(StrEnum):
    IGNORE = auto()
    INTERPOLATE = auto()
    RECENT = auto()
    VORONOI = auto()


class Trajectory:
    """Time-ordered point observations."""

    def __init__(
            self,
            df: pd.DataFrame,
            source_id: str | None = None,
            gap_method: GapMethod | None = None,
            pd_datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initialise a new Trajectory from a DataFrame."""
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            raise ValueError(f"DataFrame must have columns {', '.join(REQUIRED_COLUMNS)!r}")

        df = df.copy()
        df[DATETIME] = pd.to_datetime(df[DATETIME], format=pd_datetime_format, errors="raise")
        self.df = df
        self.source_id = source_id
        self.gap_method = gap_method

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
            gap_method: GapMethod | None = None,
            source_id: str | None = None,
    ) -> "Trajectory":
        """Initialise a new Trajectory directly from a CSV file with the required columns."""
        path = Path(filename)
        df = pd.read_csv(filename)
        source_id = path.name if source_id is None else source_id
        return cls(df, source_id=source_id, gap_method=gap_method)

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
        if len(self) == 0:
            return None, None, None, None
        return self.df[X].min(), self.df[X].max(), self.df[Y].min(), self.df[Y].max()

    @property
    def extent(self) -> tuple:
        """Return the spatial extent of the Trajectory in x and y."""
        if len(self) == 0:
            return 0.0, 0.0
        return self.df[X].max() - self.df[X].min(), self.df[Y].max() - self.df[Y].min()

    def has_dwell_times(self) -> bool:
        """Check if dwell times have been added to this Trajectory yet."""
        missing = [
            col
            for col in [DWELL_TIME_SECONDS, DWELL_BACKWARD, DWELL_FORWARD]
            if col not in self.df.columns
        ]
        return False if missing else True

    def is_within(self, spatial_reference: SpatialData | gpd.GeoDataFrame) -> bool:
        """Return True if this trajectory's spatial extent lies entirely within the spatial reference bounds.

        Args:
            spatial_reference: Either a SpatialData instance or a GeoDataFrame
                defining the reference boundary.
        """
        if isinstance(spatial_reference, SpatialData):
            gdf = spatial_reference.gdf
        else:
            gdf = spatial_reference
        (sxmin, symin, sxmax, symax) = gdf.total_bounds
        (txmin, txmax, tymin, tymax) = self.bounds
        tmins = np.array((txmin, tymin))
        smins = np.array((sxmin, symin))
        tmaxs = np.array((txmax, tymax))
        smaxs = np.array((sxmax, symax))
        return all((tmins > smins) & (tmaxs < smaxs))

    def add_dwells(
            self,
            gap_method: GapMethod | None = None,
            resolution: TimeDeltaLike | None = None,
    ) -> "Trajectory":
        # TODO: option to do this 'in place' instead of returning a new object
        """Use one of the methods to add dwell times to this Trajectory."""
        if gap_method is None:
            gap_method = self.gap_method
        if gap_method is None and self.gap_method is None:
            raise ValueError("no GapMethod set or provided")
        if gap_method is not GapMethod.VORONOI and resolution is None:
            raise ValueError(
                "time resolution must be provided for gap methods other than GapMethod.VORONOI",
            )

        if gap_method is GapMethod.VORONOI:
            return self.with_voronoi_dwells()
        if gap_method is GapMethod.INTERPOLATE:
            return self.with_interpolated_gaps()
        if gap_method is GapMethod.RECENT:
            return self.with_recent_fill()
        if gap_method is GapMethod.IGNORE:
            return self.with_ignored_gaps()

        raise ValueError(f"unknown GapMethod {gap_method}")

    def with_voronoi_dwells(self) -> "Trajectory":
        """Return a trajectory with dwell times computed by 1D voronoi calculation.

        Dwell at point i is equal to half the time since point i - 1 plus half the time until
        point i + 1.

                  t_{i} - t_{i-1}   t_{i+1} - t_{i-1}
        dwell_i = --------------- + -----------------
                         2                  2

        Returns:
            The same Trajectory with dwell times computed as described.
        """
        new = Trajectory(self.df.loc[:, REQUIRED_COLS_LIST].copy(), source_id=self.source_id)
        datetimes = new.df[DATETIME]

        backward = (datetimes - datetimes.shift(1)).dt.total_seconds().fillna(0) / 2
        forward = (datetimes.shift(-1) - datetimes).dt.total_seconds().fillna(0) / 2

        # First and last points have no full context so contribute nothing
        backward.iloc[[0, -1]] = 0.0
        forward.iloc[[0, -1]] = 0.0

        new.df[DWELL_TIME_SECONDS] = backward + forward
        new.df[DWELL_BACKWARD] = backward
        new.df[DWELL_FORWARD] = forward
        new.gap_method = GapMethod.VORONOI
        return new

    def with_interpolated_gaps(
            self,
            resolution: dt.timedelta,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
    ) -> "Trajectory":
        """Return a trajectory with gaps filled by linear interpolation.

        Only interpolates across gaps larger than the resolution. Original
        observations are preserved unchanged.

        Args:
            resolution: Minimum gap size to interpolate across. Also sets
                the time step between synthesised observations within gaps.
            start_time: Start of the interpolation range. Defaults to the
                trajectory start time.
            end_time: End of the interpolation range. Defaults to the
                trajectory end time.

        Returns:
            A new Trajectory with gap-filling interpolated observations.
        """
        start = start_time or self.start_time.to_pydatetime()
        end = end_time or self.end_time.to_pydatetime()

        datetimes = pd.to_datetime(self.df[DATETIME])

        # Collect synthetic timestamps only within gaps larger than resolution
        synthetic_times = []
        for ii in range(len(datetimes) - 1):
            t_curr = datetimes.iloc[ii].to_pydatetime()
            t_next = datetimes.iloc[ii + 1].to_pydatetime()
            gap = t_next - t_curr

            if gap > resolution:
                # Fill the gap at regular intervals, excluding endpoints
                # (original points are kept as-is)
                current = t_curr + resolution
                while current < t_next:
                    if start <= current <= end:
                        synthetic_times.append(current)
                    current += resolution

        if not synthetic_times:
            return self

        # Resample only at synthetic times and merge with originals
        synthesised = self.resample(synthetic_times, method=SamplingMethod.INTERP)

        combined_data = (
            pd.concat([self.df[REQUIRED_COLS_LIST], synthesised.df[REQUIRED_COLS_LIST]])
            .sort_values(DATETIME)
            .reset_index(drop=True)
        )

        combined = Trajectory(combined_data, source_id=self.source_id)
        new = combined.with_voronoi_dwells()
        new.gap_method = GapMethod.INTERPOLATE
        return new

    def with_recent_fill(
            self,
            resolution: dt.timedelta,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
    ) -> "Trajectory":
        """Return a trajectory with gaps filled by carrying the last known position forward.

        Original observations are preserved. Synthesised points are inserted
        only in sections where no observation exists for longer than
        ``resolution``.

        Args:
            resolution: Maximum allowable gap between observations. Gaps longer
                than this will be filled at this interval.
            start_time: Start of the fill range. Defaults to the trajectory
                start time.
            end_time: End of the fill range. Defaults to the trajectory end time.

        Returns:
            A new Trajectory with original observations preserved and gaps
            filled by carrying the last known position forward.
        """
        start = start_time or self.start_time.to_pydatetime()
        end = end_time or self.end_time.to_pydatetime()

        data = self.df[(self.df[DATETIME] >= start) & (self.df[DATETIME] <= end)].copy()

        datetimes = data[DATETIME].sort_values()
        fill_times = []

        for t_start, t_end in zip(datetimes.iloc[:-1], datetimes.iloc[1:], strict=True):
            gap = t_end - t_start
            if gap > resolution:
                current = t_start + resolution
                while current < t_end:
                    fill_times.append(current)
                    current += resolution

        if fill_times:
            filled = self.resample(fill_times, method=SamplingMethod.MOST_RECENT)
            combined = (
                pd.concat([data, filled.df], ignore_index=True)
                .sort_values(DATETIME)
                .reset_index(drop=True)
            )
        else:
            combined = data.reset_index(drop=True)

        new = Trajectory(combined).with_voronoi_dwells()
        new.gap_method = GapMethod.RECENT
        return new

    def with_ignored_gaps(
            self,
            resolution: dt.timedelta,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
    ) -> "Trajectory":
        """Return a copy where dwell times respect data gaps.

        Points whose nearest neighbour is more than one window length away
        have their dwell time bounded by the window boundary rather than
        extending across the gap.

        Args:
            resolution: Window length used to determine whether a gap is
                significant and to compute window boundaries.
            start_time: Start of the window range. Defaults to the trajectory
                start time.
            end_time: End of the window range. Defaults to the trajectory
                end time.

        Returns:
            A new Trajectory with gap-aware dwell times.
        """
        start = start_time or self.start_time.to_pydatetime()
        end = end_time or self.end_time.to_pydatetime()
        windows = utils.get_time_windows(start, end, resolution)
        window_length = windows[0][1] - windows[0][0]

        new = Trajectory(self.df[REQUIRED_COLS_LIST].copy(), source_id=self.source_id)
        datetimes = new.df[DATETIME]

        def find_window_bound(
                t: pd.Timestamp,
                bound: Literal["start", "end"],
        ) -> pd.Timestamp:
            for w_start, w_end in windows:
                if w_start <= t < w_end:
                    return w_start if bound == "start" else w_end
            return t

        backward_list = []
        forward_list = []

        for ii, t in enumerate(datetimes):
            if ii == 0 or ii == len(datetimes) - 1:
                forward = 0.0
                backward = 0.0

            else:
                # --- Backward component ---
                t_prev = datetimes.iloc[ii - 1]
                gap_back = (t - t_prev).total_seconds()
                if gap_back > window_length.total_seconds():
                    # Cap at distance to window start, not full window
                    backward = min(
                        (t - find_window_bound(t, "start")).total_seconds(),
                        window_length.total_seconds() / 2,
                    )
                else:
                    backward = gap_back / 2

                # --- Forward component ---
                t_next = datetimes.iloc[ii + 1]
                gap_forward = (t_next - t).total_seconds()
                if gap_forward > window_length.total_seconds():
                    # Cap at distance to window end, not full window
                    forward = min(
                        (find_window_bound(t, "end") - t).total_seconds(),
                        window_length.total_seconds() / 2,
                    )
                else:
                    forward = gap_forward / 2

            backward_list.append(backward)
            forward_list.append(forward)

        new.df[DWELL_BACKWARD] = backward_list
        new.df[DWELL_FORWARD] = forward_list
        new.df[DWELL_TIME_SECONDS] = new.df[DWELL_BACKWARD] + new.df[DWELL_FORWARD]
        new.gap_method = GapMethod.IGNORE
        return new

    def data_in_window(
            self,
            start: dt.datetime,
            end: dt.datetime,
            *,
            include_first: bool = False,
            include_last: bool = False,
    ) -> "Trajectory":
        """Returns a new Trajectory with a subset of data between the start and end times."""
        if not self.has_dwell_times():
            raise ValueError(
                f"Trajectory is missing dwell time columns: {DWELL_COLUMNS}. "
                "Call a with_*() method before windowing.",
            )

        df = self.df.copy()
        datetimes = df[DATETIME]
        dt_backwards = pd.to_timedelta(df[DWELL_BACKWARD], unit="s")
        dt_forwards = pd.to_timedelta(df[DWELL_FORWARD], unit="s")

        interval_start = datetimes - dt_backwards
        interval_end = datetimes + dt_forwards

        # Clip intervals to window bounds and compute overlap duration
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
        window = Trajectory(df=df_mask[REQUIRED_COLS_LIST])
        window.df[DWELL_TIME_SECONDS] = clipped_duration[mask]
        return window

    def resample(self, times: list[dt.datetime], method: SamplingMethod) -> "Trajectory":
        """Resample the trajectory at the given times.

        Args:
            times: Datetimes to sample at.
            method: Either ``"nearest"`` or ``"interp"``.

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
            for ii, t in enumerate(out_of_bounds):
                if t:
                    logger.warning(f"[out of bounds {ii + 1}] {times[ii]}")
            times = times[~out_of_bounds].reset_index(drop=True)

        if method == SamplingMethod.NEAREST:
            resampled = self._resample_nearest(times)
        elif method == SamplingMethod.MOST_RECENT:
            resampled = self._resample_previous(times)
        elif method == SamplingMethod.INTERP:
            resampled = self._resample_interp(times)
        else:
            raise ValueError(f"unknown resampling method: {method}")

        return Trajectory(resampled)

    def _resample_previous(self, times: pd.Series) -> pd.DataFrame:
        """Select the most recent recorded point before or at each requested time."""
        indices = [self.df[self.df[DATETIME] <= t][DATETIME].argmax() for t in times]
        resampled = self.df.loc[indices, REQUIRED_COLS_LIST].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)

    def _resample_nearest(self, times: pd.Series) -> pd.DataFrame:
        """Select the closest recorded point for each requested time."""
        indices = [(self.df[DATETIME] - t).abs().argmin() for t in times]
        resampled = self.df.loc[indices, REQUIRED_COLS_LIST].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)

    def _resample_interp(self, times: pd.Series) -> pd.DataFrame:
        """Linearly interpolate x/y between adjacent timestamps."""
        rows = []
        for t in times:
            before = self.df[self.df[DATETIME] <= t].iloc[-1]
            after = self.df[self.df[DATETIME] >= t].iloc[0]

            if before[DATETIME] == after[DATETIME]:
                x = before[X]
                y = before[Y]
            else:
                weight = (t - before[DATETIME]).total_seconds() / (
                        after[DATETIME] - before[DATETIME]
                ).total_seconds()
                x = before[X] + weight * (after[X] - before[X])
                y = before[Y] + weight * (after[Y] - before[Y])

            rows.append({DATETIME: t, X: x, Y: y})

        return pd.DataFrame(rows)

    def get_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract sorted numpy arrays from a trajectory."""
        x = self.df[X].to_numpy(dtype=float)
        y = self.df[Y].to_numpy(dtype=float)
        t = self.df[DATETIME].to_numpy(dtype=np.datetime64)
        order = np.argsort(t)

        if DWELL_TIME_SECONDS in self.df.columns:
            dt = self.df[DWELL_TIME_SECONDS].to_numpy(dtype=float)
        else:
            logger.warning(
                "Trajectory has no dwell times — using uniform weights. "
                "Call a with_*() method to assign dwell times explicitly.",
            )
            dt = np.ones(len(t))  # Equal weighting if no dwell times

        return x[order], y[order], t[order], dt[order]


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
