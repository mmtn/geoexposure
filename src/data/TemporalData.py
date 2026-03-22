import datetime as dt
import numpy as np

from src.Enums import TemporalType
from src.data.SpatialData import SpatialData

from src.utils import (check_iter_types, get_cyclic_timestamp,
                       match_datetime_in_list, round_datetime)


class TemporalData:
    VALID_TYPES = (
        float,
        SpatialData
    )

    def __init__(
            self,
            time_data_dict,
            cycle_duration=None,
            temporal_resolution=None,
            temporal_type=None
    ):
        # Set from arguments
        self.cycle_duration = cycle_duration
        self.temporal_resolution = temporal_resolution
        self.temporal_type = temporal_type
        self._create_dict(time_data_dict)

        # Initialise variables
        self.timestamp_type = None
        self.data_type = None

        self._set_timestamp_type()
        self._set_data_type()
        self._set_temporal_resolution()
        self._arg_check()

    @property
    def min_time(self):
        return np.array(list(self._dict.keys())).min()

    @property
    def max_time(self):
        return np.array(list(self._dict.keys())).max()

    def _arg_check(self):
        if len(self.data) <= 1:
            raise ValueError("multiple time points must be provided for TemporalData")

        if self.temporal_type is TemporalType.CYCLIC and self.cycle_duration is None:
            raise ValueError("cycle duration must be defined for TemporalType.CYCLIC")

        if self.temporal_type is TemporalType.DATED and self.timestamp_type is dt.datetime:
            raise ValueError(
                "'time_data_dict' keys must be dt.datetime for TemporalType.DATED"
            )

    def sample(self, timestamp, method="nearest"):
        if self.temporal_type is TemporalType.CYCLIC:
            timestamp = get_cyclic_timestamp(timestamp, self.cycle_duration)

        if method == "nearest":
            dt_nearest = match_datetime_in_list(
                timestamp,
                self._datetime,
                self.cycle_duration,
                to=method
            )
            return self._get_value_by_key(dt_nearest)
        elif method == "interp":
            return self._get_value_interpolated(timestamp)
        else:
            raise ValueError(f"unknown sample method: {method}")

    def _get_value_by_key(self, timestamp):
        try:
            return self._dict[timestamp]
        except:
            raise ValueError(f"no data at timestamp {timestamp}")

    def _set_timestamp_type(self):
        first_timestamp = self.timestamps[0]
        first_type = type(first_timestamp)
        if not check_iter_types(self.timestamps, first_type):
            raise ValueError("all keys of 'time_data_dict' must have same type")
        self.timestamp_type = first_type

    def _set_data_type(self):
        first_value = self.data[0]
        if not isinstance(first_value, self.VALID_TYPES):
            raise TypeError(
                "values in 'time_data_dict' must have type 'float' or 'SpatialData'"
            )
        if not check_iter_types(self.data, type(first_value)):
            raise ValueError("all values of 'time_data_dict' must have same type")
        self.data_type = type(first_value)

    def _set_temporal_resolution(self):
        if self.temporal_resolution is not None:
            return
        temporal_resolution = np.min(np.diff(self._datetime))
        time_after_cycle = self._datetime[0] + self.cycle_duration
        difference_to_end = time_after_cycle - self._datetime[-1]
        if difference_to_end < temporal_resolution:
            temporal_resolution = difference_to_end
        self.temporal_resolution = temporal_resolution

    def _timestamps_to_datetime(self):
        if self.temporal_type == TemporalType.CYCLIC:
            self._datetime = [
                get_cyclic_timestamp(ts, self.cycle_duration)
                for ts in self.timestamps
            ]
        elif self.temporal_type == TemporalType.DATED:
            self._datetime = self.timestamps
        else:
            raise ValueError(f"unknown TemporalType: {self.temporal_type}")

    def _create_dict(self, time_data_dict):
        timestamps_in = list(time_data_dict.keys())
        data_in = list(time_data_dict.values())
        sorting = np.argsort(timestamps_in)

        self.timestamps = [timestamps_in[ii] for ii in sorting]
        self.data = [data_in[ii] for ii in sorting]
        self._timestamps_to_datetime()
        self._dict = {
            key: value
            for key, value in zip(self._datetime, self.data)
        }

    def _get_value_interpolated(self, datetime):
        dt_previous = match_datetime_in_list(
            datetime,
            self._datetime,
            self.cycle_duration,
            to="floor"
        )
        dt_next = match_datetime_in_list(
            datetime,
            self._datetime,
            self.cycle_duration,
            to="ceil"
        )
        prev_value = self._get_value_by_key(dt_previous)
        next_value = self._get_value_by_key(dt_next)

        if prev_value == next_value:
            return prev_value

        to_previous = abs(datetime - dt_previous)
        time_diff = abs(dt_next - dt_previous)
        loc = to_previous / time_diff

        if self.data_type is float:
            total_diff = next_value - prev_value
            interpolated = prev_value + (loc * total_diff)

        if self.data_type is SpatialData:
            interpolated = prev_value.interpolate(next_value, loc=loc)

        return interpolated
