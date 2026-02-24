import datetime as dt

import matplotlib.pyplot as plt

from settings_debug import Settings
from src import Environment, Mobility, calculate_exposure, utils

#

trajectories = utils.read_csv_directory(Settings.mobility_data_path)

#

environment = Environment(
    spatial_data=Settings.spatial_data,
    spatial_data_ref=Settings.spatial_data_ref,
    spatial_resolution=50,  # metres
    temporal_data=Settings.temporal_data,
)
environment.calculate()

#

gdf = environment.sample(dt.time(hour=0, minute=0))
gdf.plot("exposure")
plt.show()

gdf = environment.sample(dt.time(hour=0, minute=30))
gdf.plot("exposure")
plt.show()

gdf = environment.sample(dt.time(hour=1, minute=0))
gdf.plot("exposure")
plt.show()

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
