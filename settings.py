import datetime as dt

import pandas as pd

from src.Enums import SpatialMetric, TemporalType
from src.data.Temporal import TemporalData
from src.data.Spatial import SpatialData
from src.metrics.Proximity import Proximity
from src.utils import HOURLY, MONTHLY


class Settings:
    interpolation = {
        "method": "linear",
    }

    mobility_data_path = "data/gps/preprocessed"

    mobility_filters = [
        # (Metric.TOTAL_HOURS, operator.gt, 24),
        # etc.
    ]

    spatial_data_land = SpatialData(
        file="data/gis/Pepey_Land/Pepey_Land.shp",
        crs=32648,
        metrics=[
            Proximity(column="LU", value="forest")
            # SpatialMetric.FRAGMENTATION: ("forest", "LU"),
        ]
    )
    spatial_data_water = SpatialData(
        file="data/gis/Pepey_Water/Pepey_Water.shp",
        crs=32648,
        metrics=[
            Proximity()
            # SpatialMetric.FRAGMENTATION: "water"
        ]
    )

    spatial_data_dict = {
        "land": spatial_data_land,
        "water": spatial_data_water,
    }

    spatial_data_ref = spatial_data_land

    hourly_biting_data = pd.read_csv("data/boyer_biting_hours/average.csv")["rate"].values
    temporal_data = {
        "hourly": TemporalData(
            time_data_dict={
                timestamp: value
                for timestamp, value in zip(HOURLY, hourly_biting_data)
            },
            cycle_duration=dt.timedelta(days=1),
            temporal_resolution=dt.timedelta(hours=1),
            temporal_type=TemporalType.CYCLIC
            ),
        "test": TemporalData(
            time_data_dict={
                dt.time(hour=0): spatial_data_land,
                dt.time(hour=12): spatial_data_water,
            },
            cycle_duration=dt.timedelta(days=1),
            temporal_resolution=dt.timedelta(hours=12),
            temporal_type=TemporalType.CYCLIC,
        )
    }
