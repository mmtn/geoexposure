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

    spatial_data = {
        "land": SpatialData(
            file="data/gis/Pepey_Land/Pepey_Land.shp",
            crs=32648,
            metrics=[
                Proximity(column="LU", value="forest")
                # SpatialMetric.FRAGMENTATION: ("forest", "LU"),
            ]
        ),
        "water": SpatialData(
            file="data/gis/Pepey_Water/Pepey_Water.shp",
            crs=32648,
            metrics=[
                Proximity()
                # SpatialMetric.FRAGMENTATION: "water"
            ]
        )
    }

    primary_spatial_data = "land"

    temporal_data = {
        "hourly": TemporalData(
            data=pd.read_csv("data/boyer_biting_hours/average.csv")["rate"].values,
            data_type=float,
            timestamps=HOURLY,
            cycle_duration=dt.timedelta(days=1),
            temporal_resolution=dt.timedelta(hours=1),
            temporal_type=TemporalType.CYCLIC
            ),
        "test": TemporalData(
            data=[
                SpatialData(
                    file="data/gis/Pepey_Land/Pepey_Land.shp",
                    crs=32648,
                    metrics=[
                        Proximity(column="LU", value="forest")
                        # SpatialMetric.FRAGMENTATION: ("forest", "LU"),
                    ]
                ),
                SpatialData(
                    file="data/gis/Pepey_Water/Pepey_Water.shp",
                    crs=32648,
                    metrics=[
                        Proximity()
                        # SpatialMetric.FRAGMENTATION: "water"
                    ]
                )
            ],
            data_type=SpatialData,
            timestamps=[
                dt.time(hour=0),
                dt.time(hour=12),
            ],
            cycle_duration=dt.timedelta(days=1),
            temporal_resolution=dt.timedelta(hours=12),
            temporal_type=TemporalType.CYCLIC,
        )
    }
