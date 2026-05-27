"""Time-indexed values sampled with nearest or interpolation."""

import datetime as dt
import logging
from collections.abc import Mapping
from typing import SupportsFloat

import numpy as np
import pandas as pd

from ..core.enums import SamplingMethod, TemporalType
from ..core.utils import check_iter_types, get_cyclic_timestamp, match_datetime_in_list
from .spatial import SpatialData

logger = logging.getLogger(__name__)

type DateTimeLike = dt.time | dt.date | dt.datetime | pd.Timestamp
type TimeDeltaLike = dt.timedelta | pd.Timedelta


class TemporalData:
    """A time-indexed series of spatial or scalar values sampled at arbitrary times.

    Maps a set of timestamps to either :class:`~data.spatial.SpatialData` instances
    — representing the spatial exposure environment at different points in time —
    or scalar ``float`` values representing a time-varying scaling factor.

    Supports two temporal modes via :class:`~core.enums.TemporalType`:

    - ``CYCLIC``: timestamps are interpreted as offsets within a repeating cycle
      of length ``cycle_duration`` (e.g. time of day or season).
    - ``DATED``: timestamps are absolute :class:`~datetime.datetime` values with
      no repeating structure.

    Values can be retrieved at arbitrary times using nearest, floor, ceil, or
    linear interpolation strategies via :meth:`sample`.

    Attributes:
        temporal_type: Whether the data is cyclic or dated.
        temporal_resolution: The minimum time step between entries, inferred
            from the data if not provided.
        cycle_duration: Length of the repeating cycle for ``CYCLIC`` data.
            Must be set if ``temporal_type`` is ``CYCLIC``.
    """

    VALID_TYPES = (float, SpatialData)

    def __init__(
            self,
            time_data_dict: Mapping[DateTimeLike, SpatialData | SupportsFloat],
            temporal_type: TemporalType,
            temporal_resolution: TimeDeltaLike | None = None,
            cycle_duration: TimeDeltaLike | None = None,
    ) -> None:
        """Initialise a TemporalData instance.

        Args:
            time_data_dict: mapping between datetimes and values. Values can be ``SpatialData``
                instances representing the environment at different times, or ``float`` to
                include time dependent scaling
            temporal_resolution: the temporal scale at which data is changing
            temporal_type: whether the temporal changes are absolute or cyclical
            cycle_duration: the length of the cycle when ``temporal_type`` is cyclical
        """
        # Set from arguments
        self.cycle_duration = cycle_duration
        self.temporal_resolution = temporal_resolution
        self.temporal_type = temporal_type
        self._input_dict = time_data_dict
        self._create_sorted_dict(time_data_dict)

        # Initialise variables
        self.timestamp_type: type | None = None
        self.data_type: type | None = None

        self._set_timestamp_type()
        self._set_data_type()
        self._set_temporal_resolution()
        self._arg_check()

    @property
    def min_time(self) -> DateTimeLike:
        """Return the earliest time in the internal dict."""
        return np.array(list(self._dict.keys())).min()

    @property
    def max_time(self) -> DateTimeLike:
        """Return the latest time in the internal dict."""
        return np.array(list(self._dict.keys())).max()

    def sample(
            self, timestamp: DateTimeLike, method: SamplingMethod = SamplingMethod.NEAREST,
    ) -> SpatialData | float:
        """Return a SpatialData instance from the available listed datetimes.

        Args:
            timestamp: Target time at which to sample the data. Can be a ``datetime``,
                ``pd.Timestamp`` or other object compatible with ``DateTimeLike``.
            method: Strategy used to select or compute the value at ``timestamp``.
                Supported values are:

                * ``SamplingMethod.NEAREST``: select the value at the closest available time.
                * ``SamplingMethod.FLOOR``: select the latest value at or before ``timestamp``.
                * ``SamplingMethod.CEIL``: select the earliest value at or after ``timestamp``.
                * ``SamplingMethod.INTERP``: interpolate between neighbouring timestamps.

                Defaults to ``SamplingMethod.NEAREST``.

        Returns:
            SpatialData | float: The sampled value. For spatial series this is
            typically a :class:`SpatialData` instance; for scalar series this may
            be a float.

        Raises:
            ValueError: If ``method`` is not a member of :class:`SamplingMethod`.
            ValueError: If the instance is cyclic (``TemporalType.CYCLIC``) but ``cycle_duration``
                is not set.
        """
        if not isinstance(method, SamplingMethod):
            raise ValueError(f"unknown method for sampling temporal data: {method}")

        is_cyclic = self.temporal_type is TemporalType.CYCLIC
        if is_cyclic and self.cycle_duration is None:
            raise ValueError("cycle_duration must be set for TemporalType.CYCLIC")
        if is_cyclic:
            timestamp = get_cyclic_timestamp(timestamp, self.cycle_duration)

        # Interpolated gets a representation at the exact timestamp
        if method is SamplingMethod.INTERP:
            return self._get_value_interpolated(timestamp)

        # All other methods get a matched existing timestamp by method (floor, ceil, nearest)
        match = match_datetime_in_list(timestamp, self._datetime, self.cycle_duration, to=method)
        return self._get_value_for_timestamp(match)

    def _arg_check(self) -> None:
        if len(self.data) <= 1:
            raise ValueError("multiple time points must be provided for TemporalData")

        if self.temporal_type is TemporalType.CYCLIC:
            temp = [get_cyclic_timestamp(ts, dt.timedelta(days=366)) for ts in self.timestamps]
            if self.cycle_duration is None:
                raise ValueError("cycle duration must be defined for TemporalType.CYCLIC")
            if self.cycle_duration < (max(temp) - min(temp)):
                raise ValueError("cycle duration must be longer than time between points")

        if self.temporal_type is TemporalType.DATED and self.timestamp_type is not dt.datetime:
            raise ValueError("'time_data_dict' keys must be dt.datetime for TemporalType.DATED")

    def _get_value_for_timestamp(self, timestamp: DateTimeLike) -> SpatialData | float:
        try:
            return self._dict[timestamp]
        except KeyError as e:
            raise ValueError(f"no data at timestamp {timestamp}") from e

    def _set_timestamp_type(self) -> None:
        first_timestamp = self.timestamps[0]
        first_type = type(first_timestamp)
        if not check_iter_types(self.timestamps, first_type):
            raise ValueError("all keys of 'time_data_dict' must have same type")
        self.timestamp_type = first_type

    def _set_data_type(self) -> None:
        first_value = self.data[0]
        if not isinstance(first_value, self.VALID_TYPES):
            raise TypeError("values in 'time_data_dict' must have type 'float' or 'SpatialData'")
        if not check_iter_types(self.data, type(first_value)):
            raise ValueError("all values of 'time_data_dict' must have same type")
        self.data_type = type(first_value)

    def _set_temporal_resolution(self) -> None:
        if self.temporal_resolution is not None:
            return
        temporal_resolution = np.min(np.diff(self._datetime))
        if self.temporal_type is TemporalType.CYCLIC and self.cycle_duration is not None:
            time_after_cycle = self._datetime[0] + self.cycle_duration
            difference_to_end = time_after_cycle - self._datetime[-1]
            temporal_resolution = min(temporal_resolution, difference_to_end)
        self.temporal_resolution = temporal_resolution

    def _timestamps_to_datetime(self) -> None:
        if self.temporal_type == TemporalType.CYCLIC:
            if self.cycle_duration is None:
                raise ValueError("cycle_duration must be set for TemporalType.CYCLIC")
            self._datetime = [
                get_cyclic_timestamp(ts, self.cycle_duration) for ts in self.timestamps
            ]
        elif self.temporal_type == TemporalType.DATED:
            self._datetime = self.timestamps
        else:
            raise ValueError(f"unknown TemporalType: {self.temporal_type}")

    def _create_sorted_dict(self, time_data_dict: dict) -> None:
        timestamps_in = list(time_data_dict.keys())
        data_in = list(time_data_dict.values())
        sorting = np.argsort(timestamps_in)

        self.timestamps = [timestamps_in[ii] for ii in sorting]
        self.data = [data_in[ii] for ii in sorting]
        self._timestamps_to_datetime()
        self._dict = dict(zip(self._datetime, self.data, strict=True))

    def _get_value_interpolated(self, datetime: DateTimeLike) -> SpatialData | float:
        dt_previous = match_datetime_in_list(
            datetime, self._datetime, self.cycle_duration, to="floor",
        )
        dt_next = match_datetime_in_list(datetime, self._datetime, self.cycle_duration, to="ceil")
        prev_value = self._get_value_for_timestamp(dt_previous)
        next_value = self._get_value_for_timestamp(dt_next)

        if prev_value == next_value:
            return prev_value

        to_previous = abs(datetime - dt_previous)
        if dt_next > dt_previous:
            time_diff = abs(dt_next - dt_previous)
        else:
            time_diff = abs(dt_next + self.cycle_duration - dt_previous)
        loc = to_previous / time_diff

        if issubclass(self.data_type, (float, np.floating)):
            total_diff = next_value - prev_value
            interpolated = prev_value + (loc * total_diff)
        elif self.data_type is SpatialData:
            interpolated = prev_value.interpolate(next_value, loc=loc)
        else:
            raise TypeError("unknown data type")

        return interpolated
