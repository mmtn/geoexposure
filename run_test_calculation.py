from settings import Settings
from src.data import mobility
from src.models.Mobility import Mobility
from src.models.Environment import Environment
from src.models.Exposure import exposure

#

trajectories = mobility.load(Settings.mobility_data_path)
trajectories = mobility.apply_filters(trajectories, filters=Settings.mobility_filters)
trajectories = mobility.resample(trajectories, settings=Settings.interpolation)

#

environment = Environment(
    primary_spatial_data="land",
    spatial_data=Settings.spatial_data_dict,
    spatial_resolution=500,  # metres
    temporal_data=Settings.temporal_data,
    temporal_resolution=10,  # minutes
)
environment.calculate()

#

mobility = Mobility(
    method="KDE",
    normalisation=True,
)

#

trajectories = trajectories[0:2]

# TODO: create trajectory data specifically for testing

exposures = [
    exposure(
        trajectory,
        mobility,
        environment,
        temporal_resolution=environment.temporal_resolution
    )
    for trajectory in trajectories
]
