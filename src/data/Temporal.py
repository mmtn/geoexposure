import datetime as dt

from src.Enums import TemporalType
from src.data.Spatial import SpatialData
from src.utils import REFERENCE_TIME, check_iter_types, round_datetime


class TemporalData:
    def __init__(
            self,
            time_data_dict,
            cycle_duration=None,
            temporal_resolution=None,
            temporal_type=None
    ):
        # Initial default settings
        self.timestamps = time_data_dict.keys()
        self.data = time_data_dict.values()
        self.timestamp_type = None
        self.data_type = None
        self._set_timestamp_type()
        self._set_data_type()
        self._map = time_data_dict

        # Set from arguments
        self.cycle_duration = cycle_duration
        self.temporal_resolution = temporal_resolution
        self.temporal_type = temporal_type

    def sample(self, timestamp, to="nearest"):
        # TODO: method to interpolate between elements
        timestamp_rounded = round_datetime(timestamp, self.temporal_resolution, to=to)
        if self.temporal_type == TemporalType.CYCLIC:
            timestamp_cyclic = self.get_cyclic_timestamp(timestamp_rounded)
            return self._get_value_by_key(timestamp_cyclic)
        elif self.temporal_type == TemporalType.DATED:
            return self._get_value_by_key(timestamp_rounded)
        else:
            raise ValueError(f"unknown TemporalType: {self.temporal_type}")

    def get_cyclic_timestamp(self, timestamp):
        # TODO: test TemporalData.get_cyclic_timestamp()
        # TODO: match/strip some elements of datetime before match
        if self.timestamp_type == dt.time:
            timestamp_cyclic = timestamp.time()
        elif self.timestamp_type == dt.date:
            timestamp_cyclic = timestamp.date()
            timestamp_cyclic.replace(year=REFERENCE_TIME.year)
        else:
            raise ValueError(f"unknown timestamp type: {self.timestamp_type}")
        return timestamp_cyclic

    def _get_value_by_key(self, timestamp):
        try:
            return self._map[timestamp]
        except:
            raise ValueError(f"No data at timestamp {timestamp}")

    def _set_data_type(self):
        data_type = type(self.data[0])
        if not data_type == float and not data_type == SpatialData:
            raise TypeError(
                "values in 'time_data_dict' must have type 'float' or 'SpatialData'"
            )
        if not check_iter_types(self.time_data_dict, data_type):
            raise ValueError("all values of 'time_data_dict' must have same type")
        self.data_type = data_type

    def _set_timestamp_type(self):
        data_type = type(self.timestamps[0])
        if not check_iter_types(self.timestamps, data_type):
            raise ValueError("all keys of 'time_data_dict' must have same type")
        self.timestamp_type = data_type
