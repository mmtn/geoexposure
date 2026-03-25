import datetime as dt

import pandas as pd
from matplotlib import pyplot as plt

from src import (Environment, SpatialData, TemporalData, calculate_exposure, utils)
from src.Enums import TemporalType
from src.metrics.Proximity import ProximityRisk
from src.models.KDE import KDE
from src.utils import HOURLY

#
# Settings
#
mobility_data_path = "data/gps/preprocessed"

spatial_data_land = SpatialData(
    file="data/gis/Pepey_Land/Pepey_Land.shp",
    crs=32648,
    metrics={
        ProximityRisk("LU", "forest", 200): 1.0,
        ProximityRisk("LU", "fields", 200): 1.0,
        ProximityRisk("LU", "built-up", 200): 1.0,
        ProximityRisk("LU", "plantations", 200): 1.0,
    }
)

spatial_data_water = SpatialData(
    file="data/gis/Pepey_Water/Pepey_Water.shp",
    crs=32648,
    metrics={
        ProximityRisk(threshold=200): 1.0
    }
)

spatial_data = {
    "land": spatial_data_land,
    "water": spatial_data_water,
}

hourly_biting_data = pd.read_csv("data/boyer_biting_hours/average_FIXED.csv")[
    "rate"].values
temporal_data = {
    "hourly": TemporalData(
        time_data_dict={
            timestamp: value
            for timestamp, value in zip(HOURLY, hourly_biting_data)
        },
        cycle_duration=dt.timedelta(days=1),
        temporal_type=TemporalType.CYCLIC
    )
}

#
# Calculation
#
trajectories = utils.read_csv_directory(mobility_data_path, max_files=3)

#

environment = Environment(
    spatial_data=spatial_data,
    spatial_reference_data=spatial_data_land,
    temporal_data=temporal_data,
    spatial_resolution=500,  # metres
)
environment.calculate()

#

mobility = KDE(kernel="gaussian", bandwidth=200)

#

traj = trajectories[2]

start_time = traj.datetime.min()  # dt.datetime(2019, 10, 22, 0, 0, 0)
end_time = traj.datetime.max()  # dt.datetime(2019, 10, 25, 0, 0, 0)
exposure_df = calculate_exposure(
    traj,
    mobility,
    environment,
    temporal_resolution=dt.timedelta(hours=4),
    return_snapshots=True
    # start_time=start_time,
    # end_time=end_time,
)

df_exposure_only = exposure_df.loc[
    :,
    ~exposure_df.columns.isin(["geometry", "scaling"]) &
    ~exposure_df.columns.str.startswith("window")
]
exposure_sum = df_exposure_only.sum(axis=1)
# exposure_df["exposure_total"] = exposure_sum

#

fig, ax = plt.subplots()

for col in exposure_df.columns:
    if col.startswith("window") or col.startswith("scaling"):
        continue

    ax.plot(
        exposure_df["window_centre"],
        exposure_df[col] * exposure_df["scaling"],
        '-',
        label=col
    )

# fig.legend(loc="upper center")
plt.xticks(rotation=90)
fig.tight_layout()
plt.show()

#

traj_subset = traj.data[
    (traj.data.datetime >= start_time) & (traj.data.datetime <= end_time)
    ]

lu_colour_map = {
    "forest": "#4daf4a",
    "fields": "#ffff99",
    "built-up areas": "#e41a1c",
    "plantations": "#ff7f00",
}
land = spatial_data_land.gdf
land["colour"] = land["LU"].map(lu_colour_map)

ax = land.plot(
    color=land["colour"],
    edgecolor="black",
    linewidth=0.1,
    alpha=0.4,
)

c = (traj_subset.datetime - start_time).dt.total_seconds() / 3600
# ax = environment.plot_reference()
sc = ax.scatter(
    traj_subset["x"],
    traj_subset["y"],
    c=c,
    s=20,
    cmap=plt.cm.hsv
)
sc.set_clim(0.0, 72.0)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(f"Time since {start_time} (h)")  # optional label

buffer = 1000
x_min = traj_subset["x"].min()
x_max = traj_subset["x"].max()
y_min = traj_subset["y"].min()
y_max = traj_subset["y"].max()
x_extent = x_max - x_min
y_extent = y_max - y_min
xy_diff = (x_extent - y_extent) / 2.0

if xy_diff < 0:
    x_buffer = buffer + xy_diff
    y_buffer = buffer
else:
    x_buffer = buffer
    y_buffer = buffer + xy_diff

ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
plt.show()

#
# #
#
# fig, ax = plt.subplots()
#
# ax.plot(
#     exposure_df["window_centre"],
#     exposure_df["exposure"],
#     '.-'
# )
#
# plt.show()

# exposures = [
#     calculate_exposure(trajectory, mobility_kde, environment)
#     for trajectory in trajectories
# ]

print("End of script")
