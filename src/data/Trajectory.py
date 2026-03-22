import warnings

import numpy as np
import pandas as pd

DATETIME = "datetime"
LATITUDE = "latitude"
LONGITUDE = "longitude"
DWELL_TIME_SECONDS = "dwell_time_seconds"

REQUIRED_COLUMNS = [
    DATETIME,
    LATITUDE,
    LONGITUDE,
]

MIN_ROWS = 3


class Trajectory:
    def __init__(self, df):
        assert (sorted(df.columns) == sorted(REQUIRED_COLUMNS)), \
            "DataFrame must have columns 'datetime', 'latitude', 'longitude'"
        assert (len(df) >= MIN_ROWS), \
            f"DataFrame for a trajectory must have at least {MIN_ROWS} rows"

        df[DATETIME] = pd.to_datetime(df[DATETIME], format="%Y-%m-%d %H:%M:%S")
        self.data = df
        self._add_dwell_times()

    @property
    def start_time(self):
        return self.data[DATETIME].min()

    @property
    def end_time(self):
        return self.data[DATETIME].max()

    @property
    def coordinates(self):
        return np.array([self.data[LATITUDE], self.data[LONGITUDE]])

    def data_in_window(self, start, end):
        df = self.data.copy()
        datetimes = df[DATETIME]

        interval_start = datetimes - (datetimes - datetimes.shift(1)) / 2
        interval_end = datetimes + (datetimes.shift(-1) - datetimes) / 2

        clipped_start = interval_start.clip(lower=start)
        clipped_end = interval_end.clip(upper=end)
        clipped_duration = clipped_end - clipped_start
        clipped_duration = clipped_duration.dt.total_seconds().clip(lower=0.0)

        mask = clipped_duration > 0
        df = df[mask]
        df[DWELL_TIME_SECONDS] = clipped_duration[mask]

        return df.reset_index(drop=True)

    def _add_dwell_times(self):
        datetimes = self.data[DATETIME]
        dwell_times = (datetimes.shift(-1) - datetimes.shift(1)) / 2
        dwell_times_seconds = dwell_times.dt.total_seconds()
        dwell_times_seconds = dwell_times_seconds.fillna(0)
        self.data[DWELL_TIME_SECONDS] = dwell_times_seconds

    def resample(self, times, method):
        """
        Resample the trajectory at the given times.

        :param times: a list of datetimes
        :param method: "nearest" or "interp"
        :return: new Trajectory object with resampled data
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

        if method == "nearest":
            resampled = self._resample_nearest(times)

        elif method == "interp":
            resampled = self._resample_interp(times)

        else:
            raise ValueError(f"unknown resampling method: {method}")

        return Trajectory(resampled)

    def _resample_nearest(self, times):
        """
        For each requested time, return the row with the closest timestamp.
        """
        indices = [
            (self.data[DATETIME] - t).abs().argmin()
            for t in times
        ]
        resampled = self.data.iloc[indices][REQUIRED_COLUMNS].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)

    def _resample_interp(self, times):
        """
        For each requested time, linearly interpolate latitude and longitude
        between the two adjacent timestamps.
        """
        rows = []
        for t in times:
            before = self.data[self.data[DATETIME] <= t].iloc[-1]
            after = self.data[self.data[DATETIME] >= t].iloc[0]

            if before[DATETIME] == after[DATETIME]:
                lat = before[LATITUDE]
                lon = before[LONGITUDE]
            else:
                weight = (
                        (t - before[DATETIME]).total_seconds() /
                        (after[DATETIME] - before[DATETIME]).total_seconds()
                )
                lat = before[LATITUDE] + weight * (after[LATITUDE] - before[LATITUDE])
                lon = before[LONGITUDE] + weight * (
                            after[LONGITUDE] - before[LONGITUDE])

            rows.append({DATETIME: t, LATITUDE: lat, LONGITUDE: lon})

        return pd.DataFrame(rows)
