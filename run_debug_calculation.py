import datetime as dt

from settings_debug import Settings
from src.data import mobility
from src.models.Mobility import Mobility
from src.models.Environment import Environment
from src.models.Exposure import exposure

#

trajectories = mobility.load(Settings.mobility_data_path)

#

environment = Environment(
    spatial_data=Settings.spatial_data,
    spatial_data_ref=Settings.spatial_data_ref,
    spatial_resolution=50,  # metres
    temporal_data=Settings.temporal_data,
)
environment.calculate()

#

import matplotlib.pyplot as plt

gdf = environment.sample(dt.time(hour=0, minute=0))
gdf.plot("exposure")
plt.show()

gdf = environment.sample(dt.time(hour=0, minute=30))
gdf.plot("exposure")
plt.show()

gdf = environment.sample(dt.time(hour=1, minute=0))
gdf.plot("exposure")
plt.show()

# #
#
# mobility = Mobility(
#     method="KDE",
#     normalisation=True,
# )
#
# #
#
# trajectories = trajectories[0:2]
#
# # TODO: create trajectory data specifically for testing
#
# exposures = [
#     exposure(
#         trajectory,
#         mobility,
#         environment,
#         temporal_resolution=environment.temporal_resolution
#     )
#     for trajectory in trajectories
# ]
