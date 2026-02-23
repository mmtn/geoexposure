import pandas as pd

DATA_COLUMNS = [
    "datetime",
    "latitude",
    "longitude",
]


class Trajectory:
    def __init__(self, df):
        assert (sorted(df.columns) == sorted(DATA_COLUMNS)), \
            "DataFrame must have columns 'datetime', 'latitude', 'longitude'"
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
        self.data = df
