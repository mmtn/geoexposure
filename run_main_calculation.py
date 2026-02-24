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
    spatial_data_ref=Settings.spatial_data_ref,
    spatial_data=Settings.spatial_data_dict,
    spatial_resolution=500,  # metres
    temporal_data=Settings.temporal_data,
)
environment.calculate()

#

mobility = Mobility(
    method="KDE",
    normalisation=True,
)

#

exposures = [
    exposure(
        trajectory,
        mobility,
        environment,
        temporal_resolution=environment.temporal_resolution
    )
    for trajectory in trajectories
]
