import datetime as dt

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from plot_utils import fix_axis_limits
from src import Environment, SpatialData, TemporalData, calculate_exposure, utils
from src.Enums import TemporalType
from src.metrics.Proximity import ProximityRisk
from src.models.DensityModel import DensityModel
from src.models.KDE import KDE
from src.models.PointOverlay import PointOverlay
from src.utils import get_times

#
# 1 Read data
#
mobility_data_path = "data/gps/preprocessed"
trajectories = utils.read_csv_directory(mobility_data_path)


#
# 2 Load geometries
#
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


#
# 3 Build environment
#
environment = Environment(
    spatial_data=spatial_data,
    spatial_reference_data=spatial_data_land,
    spatial_resolution=200,  # metres
)
environment.calculate()

#
# 4 Mobility model
#
mobility_kde = KDE(kernel="gaussian", bandwidth=200)
mobility_density = DensityModel(
    sigma0=200.0,
    v=2.5,
    k=3.0,
    timestep=dt.timedelta(minutes=15)
)
mobility_point_overlay = PointOverlay(buffer=1000)

mobility_1 = mobility_density
mobility_2 = mobility_kde

#
# 5 Select trajectory
#

# Top 5 longest gaps
gaps = np.array(
    [
        t.data.datetime.diff().max().total_seconds()
        for t in trajectories
    ]
)

n = 5
idx = np.argsort(gaps)

traj = trajectories[idx[0]]
times = get_times(traj.start_time, traj.end_time, dt.timedelta(minutes=10))
resampled = traj.resample(times, "interp")
traj = resampled
print(f"{len(traj)} points in trajectory")

#
# 6 Compute densities
#
rho1 = mobility_1.distribution(traj, environment.gdf_raster)
rho2 = mobility_2.distribution(traj, environment.gdf_raster)

rho1["normalised"] = rho1.density / rho1.density.sum()
rho2["normalised"] = rho2.density / rho2.density.sum()

dens_max = max(rho1["normalised"].max(), rho2["normalised"].max())

ax = rho1.plot("normalised", vmin=0.0, vmax=dens_max)
ax.scatter(traj.data.x, traj.data.y, color="red", s=10, alpha=0.5)
fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
plt.show()

ax = rho2.plot("normalised")#, vmin=0.0, vmax=dens_max)
ax.scatter(traj.data.x, traj.data.y, color="red", s=10, alpha=0.5)
fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
plt.show()

fig, ax = plt.subplots()
rho1["normdiff"] = rho2["normalised"] - rho1["normalised"]
max_val = max(abs(rho1["normdiff"]))
vmin = -max_val
vmax = max_val
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap_diverging = plt.cm.managua
scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_diverging)
ax = rho1.plot(
    column="normdiff",
    cmap=cmap_diverging,
    vmin=vmin,
    vmax=vmax,
    ax=ax,
    legend=False
)
plt.colorbar(scalar_mappable, ax=ax, label="Difference")
# ax.scatter(traj.data.x, traj.data.y, color="red", s=10, alpha=0.5)
fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
ax.set_title("Difference")
plt.show()
