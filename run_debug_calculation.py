import datetime as dt

import matplotlib.pyplot as plt

from src import Environment, Mobility, calculate_exposure, utils, SpatialData, Proximity, TemporalData
from src.Enums import TemporalType
from src.models.KDE import KDE

#
# Generated data
mobility_data_path = "src/tests/data/trajectories"

# Generated geometries
side_by_side = SpatialData(
    "src/tests/data/gis/side_by_side/side_by_side.shp",
    # metrics=[Proximity("land_use", "forest")]
)
forest = SpatialData(
    "src/tests/data/gis/forest/forest.shp",
    metrics=[Proximity("land_use", "forest")]
)
fields = SpatialData(
    "src/tests/data/gis/fields/fields.shp",
    metrics=[Proximity("land_use", "fields")]
)
plantations = SpatialData(
    "src/tests/data/gis/plantations/plantations.shp",
    metrics=[Proximity("land_use", "plantations")]
)
built_up = SpatialData(
    "src/tests/data/gis/built-up/built-up.shp",
    metrics=[Proximity("land_use", "built-up")]
)

spatial_data = {
    "side_by_side": side_by_side,
    # "forest": forest,
}

temporal_data = {
    "numbers": TemporalData(
        time_data_dict={
            dt.time(hour=2, minute=0): 3.0,
            dt.time(hour=0, minute=0): 1.0,
            dt.time(hour=1, minute=0): 2.0,
            dt.time(hour=3, minute=0): 4.0,
        },
        cycle_duration=dt.timedelta(hours=4, minutes=00),
        temporal_type=TemporalType.CYCLIC,
    ),
    "high_frequency": TemporalData(
        time_data_dict={
            dt.time(hour=0, minute=0): forest,
            dt.time(hour=0, minute=30): fields,
            dt.time(hour=1, minute=0): built_up,
            dt.time(hour=1, minute=30): plantations,
        },
        cycle_duration=dt.timedelta(hours=2, minutes=00),
        temporal_resolution=dt.timedelta(minutes=30),
        temporal_type=TemporalType.CYCLIC,
    )
}


trajectories = utils.read_csv_directory(mobility_data_path)

#

environment = Environment(
    spatial_data=spatial_data,
    spatial_data_ref=side_by_side,
    spatial_resolution=50,  # metres
    temporal_data=temporal_data,
)
environment.calculate()

#

mobility = KDE(kernel="gaussian", bandwidth=200)

#

exposures = [
    calculate_exposure(trajectory, mobility, environment)
    for trajectory in trajectories
]
