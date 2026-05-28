"""Gap-filling strategies for Trajectory objects."""

import datetime as dt
from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd

from ..core import datetime_utils
from ..core.enums import GapMethod
from .columns import (
    DATETIME,
    DWELL_BACKWARD,
    DWELL_FORWARD,
    DWELL_TIME_SECONDS,
    REQUIRED_COLS_LIST,
)
from .resampling import InterpResampler, PreviousResampler
from .temporal import DateTimeLike, TimeDeltaLike
from .trajectory import Trajectory


class GapFiller(ABC):
    """Abstract base class for gap-filling strategies applied to a Trajectory."""

    @abstractmethod
    def fill(
            self,
            trajectory: Trajectory,
            resolution: TimeDeltaLike | None = None,
            start_time: DateTimeLike | None = None,
            end_time: DateTimeLike | None = None,
    ) -> Trajectory:
        """Return a new Trajectory with gaps handled according to this strategy.

        Args:
            trajectory: The source trajectory to process.
            resolution: Time step used to detect and fill gaps.
            start_time: Start of the fill range. Defaults to trajectory start.
            end_time: End of the fill range. Defaults to trajectory end.

        Returns:
            A new Trajectory instance with gaps handled.
        """


class VoronoiDwells(GapFiller):
    """Assigns dwell times using a 1D Voronoi (midpoint) rule."""

    def fill(
            self,
            trajectory: Trajectory,
            _resolution: TimeDeltaLike | None = None,
            _start_time: DateTimeLike | None = None,
            _end_time: DateTimeLike | None = None,
    ) -> Trajectory:
        """Return a trajectory with dwell times computed by 1D voronoi calculation.

        Dwell at point i is equal to half the time since point i - 1 plus half the time until
        point i + 1.

                  t_{i} - t_{i-1}   t_{i+1} - t_{i}
        dwell_i = --------------- + ---------------
                         2                  2

        The first and last points receive a dwell time of zero because they lack
        a preceding or following neighbour respectively.

        Args:
            trajectory: The source trajectory to process.
            _resolution: Unused by this strategy; accepted for interface compatibility.
            _start_time: Unused by this strategy; accepted for interface compatibility.
            _end_time: Unused by this strategy; accepted for interface compatibility.

        Returns:
            A new Trajectory with dwell times computed as described.
        """
        new = Trajectory(
            trajectory.df.loc[:, REQUIRED_COLS_LIST].copy(),
            source_id=trajectory.source_id,
        )
        datetimes = new.df[DATETIME]

        backward = (datetimes - datetimes.shift(1)).dt.total_seconds().fillna(0) / 2
        forward = (datetimes.shift(-1) - datetimes).dt.total_seconds().fillna(0) / 2

        # First and last points have no full context so contribute nothing
        backward.iloc[[0, -1]] = 0.0
        forward.iloc[[0, -1]] = 0.0

        new.df[DWELL_TIME_SECONDS] = backward + forward
        new.df[DWELL_BACKWARD] = backward
        new.df[DWELL_FORWARD] = forward
        return new


class InterpolatedGaps(GapFiller):
    """Fills gaps by linear interpolation at a fixed time step."""

    def fill(
            self,
            trajectory: Trajectory,
            resolution: TimeDeltaLike | None = None,
            start_time: DateTimeLike | None = None,
            end_time: DateTimeLike | None = None,
    ) -> Trajectory:
        """Return a trajectory with gaps filled by linear interpolation.

        Only interpolates across gaps larger than the resolution. Original
        observations are preserved unchanged.

        Args:
            trajectory: The source trajectory to process.
            resolution: Minimum gap size to interpolate across. Also sets
                the time step between synthesised observations within gaps.
            start_time: Start of the interpolation range. Defaults to the
                trajectory start time.
            end_time: End of the interpolation range. Defaults to the
                trajectory end time.

        Returns:
            A new Trajectory with gap-filling interpolated observations.

        Raises:
            ValueError: If ``resolution`` is not provided.
        """
        start = start_time or trajectory.start_time.to_pydatetime()
        end = end_time or trajectory.end_time.to_pydatetime()

        datetimes = pd.to_datetime(trajectory.df[DATETIME])

        # Collect synthetic timestamps only within gaps larger than resolution
        synthetic_times = self._synthetic_times(datetimes, resolution, start, end)

        if not synthetic_times:
            return trajectory

        # Resample only at synthetic times and merge with originals
        # Bypass Trajectory.resample() to avoid redundant bounds checking.
        resampler = InterpResampler()
        synthesised_df = resampler.resample(trajectory, pd.Series(synthetic_times))

        combined_data = (
            pd.concat([trajectory.df[REQUIRED_COLS_LIST], synthesised_df[REQUIRED_COLS_LIST]])
            .sort_values(DATETIME)
            .reset_index(drop=True)
        )
        combined = Trajectory(combined_data, source_id=trajectory.source_id)

        # Use voronoi dwells on interpolated data
        return VoronoiDwells().fill(combined)

    def _synthetic_times(
            self,
            datetimes: pd.Series,
            resolution: dt.timedelta,
            start: dt.datetime,
            end: dt.datetime,
    ) -> list[dt.datetime]:
        """Collect interpolation timestamps for all gaps larger than resolution.

        Args:
            datetimes: Sorted series of observed datetimes.
            resolution: Minimum gap size and interpolation step.
            start: Start of the valid interpolation range.
            end: End of the valid interpolation range.

        Returns:
            List of synthetic datetimes falling within gaps and within [start, end].
        """
        times = []
        for ii in range(len(datetimes) - 1):
            t_curr = datetimes.iloc[ii].to_pydatetime()
            t_next = datetimes.iloc[ii + 1].to_pydatetime()
            if t_next - t_curr > resolution:
                current = t_curr + resolution
                while current < t_next:
                    if start <= current <= end:
                        times.append(current)
                    current += resolution
        return times


class RecentFill(GapFiller):
    """Fills gaps by carrying the last known position forward."""

    def fill(
            self,
            trajectory: Trajectory,
            resolution: TimeDeltaLike | None = None,
            start_time: DateTimeLike | None = None,
            end_time: DateTimeLike | None = None,
    ) -> Trajectory:
        """Return a trajectory with gaps filled by carrying the last known position forward.

        Original observations are preserved. Synthesised points are inserted
        only in sections where no observation exists for longer than
        ``resolution``.

        Args:
            trajectory: The source trajectory to process.
            resolution: Maximum allowable gap between observations. Gaps longer
                than this will be filled at this interval.
            start_time: Start of the fill range. Defaults to the trajectory
                start time.
            end_time: End of the fill range. Defaults to the trajectory end time.

        Returns:
            A new Trajectory with original observations preserved, gaps filled
            by carrying the last known position forward, and Voronoi dwell times.
        """
        start = start_time or trajectory.start_time.to_pydatetime()
        end = end_time or trajectory.end_time.to_pydatetime()

        in_bounds = (trajectory.df[DATETIME] >= start) & (trajectory.df[DATETIME] <= end)
        data = trajectory.df[in_bounds].copy()

        datetimes = data[DATETIME].sort_values()
        fill_times = []

        for t_start, t_end in zip(datetimes.iloc[:-1], datetimes.iloc[1:], strict=True):
            if t_end - t_start > resolution:
                current = t_start + resolution
                while current < t_end:
                    fill_times.append(current)
                    current += resolution

        if fill_times:
            resampler = PreviousResampler()
            filled_df = resampler.resample(trajectory, pd.Series(fill_times))
            combined = (
                pd.concat([data, filled_df], ignore_index=True)
                .sort_values(DATETIME)
                .reset_index(drop=True)
            )
        else:
            combined = data.reset_index(drop=True)

        combined_traj = Trajectory(combined, source_id=trajectory.source_id)
        return VoronoiDwells().fill(combined_traj)


class IgnoredGaps(GapFiller):
    """Assigns gap-aware dwell times without synthesising new observations."""

    def fill(
            self,
            trajectory: Trajectory,
            resolution: TimeDeltaLike | None = None,
            start_time: DateTimeLike | None = None,
            end_time: DateTimeLike | None = None,
    ) -> Trajectory:
        """Return a copy where dwell times respect data gaps.

        Points whose nearest neighbour is more than one window length away
        have their dwell time bounded by the window boundary rather than
        extending across the gap.

        Args:
            trajectory: The source trajectory to process.
            resolution: Window length used to determine whether a gap is
                significant and to compute window boundaries.
            start_time: Start of the window range. Defaults to the trajectory
                start time.
            end_time: End of the window range. Defaults to the trajectory
                end time.

        Returns:
            A new Trajectory with gap-aware dwell times.
        """
        start = start_time or trajectory.start_time.to_pydatetime()
        end = end_time or trajectory.end_time.to_pydatetime()
        windows = datetime_utils.get_time_windows(start, end, resolution)
        window_length = windows[0][1] - windows[0][0]

        new = Trajectory(trajectory.df[REQUIRED_COLS_LIST].copy(), source_id=trajectory.source_id)
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


_GAP_METHOD_MAPPING: dict[GapMethod, GapFiller] = {
    GapMethod.VORONOI    : VoronoiDwells(),
    GapMethod.INTERPOLATE: InterpolatedGaps(),
    GapMethod.RECENT     : RecentFill(),
    GapMethod.IGNORE     : IgnoredGaps(),
}


def get_gap_filler(method: GapMethod) -> GapFiller:
    """Return the GapFiller instance corresponding to the given GapMethod.

    Args:
        method: The selected gap-filling strategy.

    Returns:
        A GapFiller instance ready to call.

    Raises:
        KeyError: If ``method`` is not registered.
    """
    return _GAP_METHOD_MAPPING[method]
