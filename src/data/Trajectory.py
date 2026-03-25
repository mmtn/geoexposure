import datetime as dt
import warnings

import numpy as np
import pandas as pd

DATETIME = "datetime"
X = "x"
Y = "y"
DWELL_TIME_SECONDS = "dwell_time_seconds"

REQUIRED_COLUMNS = [
    DATETIME,
    X,
    Y,
]


class Trajectory:
    def __init__(self, df: pd.DataFrame):
        assert sorted(df.columns) == sorted(
            REQUIRED_COLUMNS
        ), "DataFrame must have columns 'datetime', 'x', 'y'"
        df[DATETIME] = pd.to_datetime(df[DATETIME], format="%Y-%m-%d %H:%M:%S")
        self.data = df
        self._add_dwell_times()

    def __len__(self):
        return len(self.data)

    @property
    def start_time(self) -> dt.datetime:
        return self.data[DATETIME].min()

    @property
    def end_time(self) -> dt.datetime:
        return self.data[DATETIME].max()

    @property
    def duration_in_seconds(self):
        time_difference = self.data[DATETIME].max() - self.data[DATETIME].min()
        return time_difference.total_seconds()

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([self.data[X], self.data[Y]])

    def data_in_window(self, start: dt.datetime, end: dt.datetime) -> "Trajectory":
        df = self.data.copy()
        datetimes = df[DATETIME]

        interval_start = datetimes - (datetimes - datetimes.shift(1)) / 2
        interval_end = datetimes + (datetimes.shift(-1) - datetimes) / 2

        clipped_start = interval_start.clip(lower=start)
        clipped_end = interval_end.clip(upper=end)
        clipped_duration = clipped_end - clipped_start
        clipped_duration = (
            clipped_duration.dt.total_seconds().fillna(0.0).clip(lower=0.0)
        )

        mask = clipped_duration > 0
        df_mask = df[mask]
        window = Trajectory(df=df_mask[REQUIRED_COLUMNS])
        window.data[DWELL_TIME_SECONDS] = clipped_duration[mask]
        return window

    def _add_dwell_times(self):
        datetimes = self.data[DATETIME]
        dwell_times = (datetimes.shift(-1) - datetimes.shift(1)) / 2
        dwell_times_seconds = dwell_times.dt.total_seconds()
        dwell_times_seconds = dwell_times_seconds.fillna(0)
        self.data[DWELL_TIME_SECONDS] = dwell_times_seconds

    def resample(self, times: list, method: str):
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

    def _resample_nearest(self, times: pd.Series):
        """
        For each requested time, return the row with the closest timestamp.
        """
        indices = [(self.data[DATETIME] - t).abs().argmin() for t in times]
        resampled = self.data.iloc[indices][REQUIRED_COLUMNS].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)

    def _resample_interp(self, times: pd.Series):
        """
        For each requested time, linearly interpolate latitude and longitude
        between the two adjacent timestamps.
        """
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

    def get_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract sorted numpy arrays from a Trajectory.
        """
        x_values = self.data[X].to_numpy(dtype=float)
        y_values = self.data[Y].to_numpy(dtype=float)
        t_values = self.data[DATETIME].to_numpy(
            dtype=np.datetime64
        )  # array of datetime objects
        dt_values = self.data[DWELL_TIME_SECONDS].to_numpy(dtype=float)
        order = np.argsort(t_values)
        return x_values[order], y_values[order], t_values[order], dt_values[order]
