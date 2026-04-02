import datetime as dt
import warnings
import logging
from typing import Literal

import numpy as np
import pandas as pd

from .. import utils
from ..enums import SamplingMethod


DATETIME = "datetime"
X = "x"
Y = "y"

REQUIRED_COLUMNS = [
    DATETIME,
    X,
    Y,
]

# Computed columns
DWELL_TIME_SECONDS = "dwell_time_seconds"
DWELL_FORWARD = "_dwell_forward"
DWELL_BACKWARD = "_dwell_backward"

class Trajectory:
    """Time-ordered point observations with derived dwell times."""

    def __init__(self, df: pd.DataFrame, source_id: str | None = None):
        assert sorted(df.columns) == sorted(
            REQUIRED_COLUMNS
        ), "DataFrame must have columns 'datetime', 'x', 'y'"
        df[DATETIME] = pd.to_datetime(df[DATETIME], format="%Y-%m-%d %H:%M:%S")
        self.data = df
        self.source_id = source_id

    def __len__(self) -> int:
        return len(self.data)

    @property
    def start_time(self) -> pd.Timestamp:
        return self.data[DATETIME].min()

    @property
    def end_time(self) -> pd.Timestamp:
        return self.data[DATETIME].max()

    @property
    def duration_in_seconds(self) -> float:
        time_difference = self.data[DATETIME].max() - self.data[DATETIME].min()
        return time_difference.total_seconds()

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([self.data[X], self.data[Y]])

    @property
    def bounds(self) -> tuple:
        if len(self) == 0:
            return None, None, None, None
        x = self.data[X]
        y = self.data[Y]
        return x.min(), x.max(), y.min(), y.max()

    @property
    def extent(self) -> tuple:
        if len(self) == 0:
            return 0.0, 0.0
        x = self.data[X]
        y = self.data[Y]
        return x.max() - x.min(), y.max() - y.min()

    def with_voronoi_dwells(self) -> "Trajectory":
        new = Trajectory(self.data[REQUIRED_COLUMNS].copy(), source_id=self.source_id)
        datetimes = new.data[DATETIME]

        backward = (datetimes - datetimes.shift(1)).dt.total_seconds().fillna(0) / 2
        forward = (datetimes.shift(-1) - datetimes).dt.total_seconds().fillna(0) / 2

        # First and last points have no full context so contribute nothing
        backward.iloc[[0, -1]] = 0.0
        forward.iloc[[0, -1]] = 0.0

        new.data[DWELL_TIME_SECONDS] = backward + forward
        new.data[DWELL_BACKWARD] = backward
        new.data[DWELL_FORWARD] = forward
        return new

    def with_interpolated_gaps(
            self,
            resolution: dt.timedelta,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
    ) -> "Trajectory":
        """Return a trajectory with gaps filled by linear interpolation.

        Synthesises new observations at regular intervals across the full
        trajectory duration, interpolating x/y positions between known points.
        Useful when movement between observations is expected to be continuous.

        Args:
            resolution: Time step between synthesised observations.
            start_time: Start of the interpolation range. Defaults to the
                trajectory start time.
            end_time: End of the interpolation range. Defaults to the
                trajectory end time.
        Returns:
            A new Trajectory with interpolated observations at regular intervals.
        """
        start = start_time or self.start_time.to_pydatetime()
        end = end_time or self.end_time.to_pydatetime()

        times = []
        current = start
        # Go one window past the end
        while current <= end:
            times.append(current)
            current += resolution

        return self.resample(times, method="interp").with_voronoi_dwells()

    def with_recent_fill(
            self,
            resolution: dt.timedelta,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
    ) -> "Trajectory":
        """Return a trajectory with gaps filled by carrying the last known position forward.

        At each time step, the most recent observed position is used. This is
        equivalent to a zero-order hold and is appropriate when the subject is
        likely stationary between observations.

        Args:
            resolution: Time step between synthesised observations.
            start_time: Start of the fill range. Defaults to the trajectory
                start time.
            end_time: End of the fill range. Defaults to the trajectory end time.
        Returns:
            A new Trajectory with positions carried forward at regular intervals.
        """
        start = start_time or self.start_time.to_pydatetime()
        end = end_time or self.end_time.to_pydatetime()

        times = []
        current = start
        while current <= end:
            times.append(current)
            current += resolution

        return self.resample(times, method="nearest").with_voronoi_dwells()

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

        new = Trajectory(self.data[REQUIRED_COLUMNS].copy(), source_id=self.source_id)
        datetimes = new.data[DATETIME]

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
            if ii == 0 or  ii == len(datetimes) - 1:
                forward = 0.0
                backward = 0.0

            else:
                # --- Backward component ---
                t_prev = datetimes.iloc[ii - 1]
                gap_back = (t - t_prev).total_seconds()
                if gap_back > window_length.total_seconds():
                    backward = (t - find_window_bound(t, "start")).total_seconds()
                else:
                    backward = gap_back / 2

                # --- Forward component ---
                t_next = datetimes.iloc[ii + 1]
                gap_forward = (t_next - t).total_seconds()
                if gap_forward > window_length.total_seconds():
                    forward = (find_window_bound(t, "end") - t).total_seconds()
                else:
                    forward = gap_forward / 2

            backward_list.append(backward)
            forward_list.append(forward)

        new.data[DWELL_BACKWARD] = backward_list
        new.data[DWELL_FORWARD] = forward_list
        new.data[DWELL_TIME_SECONDS] = \
            new.data[DWELL_BACKWARD] + new.data[DWELL_FORWARD]
        return new

    def data_in_window(
            self,
            start: dt.datetime,
            end: dt.datetime,
            include_first: bool = False,
            include_last: bool = False,
    ) -> "Trajectory":
        missing = [
            col for col in [DWELL_TIME_SECONDS, DWELL_BACKWARD, DWELL_FORWARD]
            if col not in self.data.columns
        ]
        if missing:
            raise ValueError(
                f"Trajectory is missing columns: {missing}. "
                "Call a with_*() method before windowing."
            )

        df = self.data.copy()
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
        if include_first:
            boundary_indices.add(df.index[0])
        if include_last:
            boundary_indices.add(df.index[-1])

        mask = (clipped_duration > 0) | df.index.isin(boundary_indices)

        df_mask = df[mask]
        window = Trajectory(df=df_mask[REQUIRED_COLUMNS])
        window.data[DWELL_TIME_SECONDS] = clipped_duration[mask]
        return window

    def resample(self, times: list[dt.datetime], method: str) -> "Trajectory":
        """Resample the trajectory at the given times.

        Args:
            times: Datetimes to sample at.
            method: Either ``"nearest"`` or ``"interp"``.

        Returns:
            A new trajectory containing the resampled observations.
        """
        times = pd.Series(times)

        # Warn and drop times outside the trajectory bounds
        t_min = self.data[DATETIME].min()
        t_max = self.data[DATETIME].max()
        out_of_bounds = (times < t_min) | (times > t_max)
        if out_of_bounds.any():
            warnings.warn(
                f"{out_of_bounds.sum()} time(s) are outside the trajectory bounds "
                f"[{t_min}, {t_max}] and will be ignored."
            )
            for ii, t in enumerate(out_of_bounds):
                if t:
                    warnings.warn(f"[out of bounds {ii + 1}] {times[ii]}")
            times = times[~out_of_bounds].reset_index(drop=True)

        if method == SamplingMethod.NEAREST:
            resampled = self._resample_nearest(times)
        elif method == SamplingMethod.INTERP:
            resampled = self._resample_interp(times)
        else:
            raise ValueError(f"unknown resampling method: {method}")

        return Trajectory(resampled)

    def _resample_nearest(self, times: pd.Series) -> pd.DataFrame:
        """Select the closest recorded point for each requested time."""
        indices = [(self.data[DATETIME] - t).abs().argmin() for t in times]
        resampled = self.data.iloc[indices][REQUIRED_COLUMNS].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)

    def _resample_interp(self, times: pd.Series) -> pd.DataFrame:
        """Linearly interpolate x/y between adjacent timestamps."""
        rows = []
        for t in times:
            before = self.data[self.data[DATETIME] <= t].iloc[-1]
            after = self.data[self.data[DATETIME] >= t].iloc[0]

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
        x = self.data[X].to_numpy(dtype=float)
        y = self.data[Y].to_numpy(dtype=float)
        t = self.data[DATETIME].to_numpy(dtype=np.datetime64)
        order = np.argsort(t)

        if DWELL_TIME_SECONDS in self.data.columns:
            dt = self.data[DWELL_TIME_SECONDS].to_numpy(dtype=float)
        else:
            logging.warning(
                "Trajectory has no dwell times — using uniform weights. "
                "Call a with_*() method to assign dwell times explicitly."
            )
            dt = np.ones(len(t))  # Equal weighting if no dwell times

        return x[order], y[order], t[order], dt[order]
