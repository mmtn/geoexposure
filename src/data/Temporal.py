import datetime as dt

from src.Enums import TemporalType
from src.utils import REFERENCE_TIME, check_iter_types, round_datetime


class TemporalData:
    def __init__(
            self,
            data,
            data_type,
            timestamps,
            cycle_duration=None,
            temporal_resolution=None,
            temporal_type=None
    ):
        # Initial default settings
        self._map = dict()
        self.data_type = None
        self.timestamp_type = None

        # Set from arguments
        self.data = data
        self.timestamps = timestamps
        self.cycle_duration = cycle_duration
        self.temporal_resolution = temporal_resolution
        self.temporal_type = temporal_type

        # Process inputs
        self._set_data_type()
        self._set_timestamp_type()
        self._build_map()

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

    def _build_map(self):
        # TODO: make TemporalData.__init__() argument a dict so this isn't necessary
        if len(self.data) != len(self.timestamps):
            raise ValueError("'data' and 'timestamps' must have equal lengths")

        self._map = {
            t: d
            for t, d in zip(self.timestamps, self.data)
        }

    def _get_value_by_key(self, timestamp):
        try:
            return self._map[timestamp]
        except:
            raise ValueError(f"No data at datetime {timestamp}")

    def _set_data_type(self):
        data_type = type(self.data[0])
        if not check_iter_types(self.data, data_type):
            raise ValueError("all elements of 'data' must have same type")
        self.data_type = data_type

    def _set_timestamp_type(self):
        data_type = type(self.timestamps[0])
        if not check_iter_types(self.timestamps, data_type):
            raise ValueError("all elements of 'timestamps' must have same type")
        self.timestamp_type = data_type
