import datetime as dt
import pandas as pd

from src import (
    Environment, Mobility, Proximity, SpatialData, TemporalData, calculate_exposure,
    utils
)
from src.Enums import TemporalType
from src.utils import HOURLY

#
# Settings
#
mobility_data_path = "data/gps/preprocessed"

spatial_data_land = SpatialData(
    file="data/gis/Pepey_Land/Pepey_Land.shp",
    crs=32648,
    metrics=[
        Proximity(column="LU", value="forest")
    ]
)

spatial_data_water = SpatialData(
    file="data/gis/Pepey_Water/Pepey_Water.shp",
    crs=32648,
    metrics=[
        Proximity()
    ]
)

spatial_data_dict = {
    "land": spatial_data_land,
    "water": spatial_data_water,
}

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
    )
}

#
# Calculation
#
trajectories = utils.read_csv_directory(mobility_data_path)

#

environment = Environment(
    spatial_reference_data=spatial_data_land,
    spatial_data=spatial_data_dict,
    spatial_resolution=500,  # metres
    temporal_data=temporal_data,
)
environment.calculate()

#

mobility = Mobility(
    method="KDE",
    normalisation=True,
)

#

exposures = [
    calculate_exposure(trajectory, mobility, environment)
    for trajectory in trajectories
]
