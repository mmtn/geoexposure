from settings import Settings
from src import calculate_exposure, Mobility, Environment, utils

#

trajectories = utils.read_csv_directory(Settings.mobility_data_path)

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
    calculate_exposure(
        trajectory,
        mobility,
        environment,
        temporal_resolution=environment.temporal_resolution
    )
    for trajectory in trajectories
]
