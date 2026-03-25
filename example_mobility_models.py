import datetime as dt

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from plot_utils import fix_axis_limits
from src import Environment, SpatialData, TemporalData, calculate_exposure, utils
from src.Enums import TemporalType
from src.metrics.Proximity import ProximityRisk
from src.models.DensityModel import DensityModel
from src.models.KDE import KDE
from src.utils import get_times

#
# 1 Read data
#
mobility_data_path = "src/tests/data/trajectories"
trajectories = utils.read_csv_directory(mobility_data_path)

#
# 2 Load geometries
#
side_by_side = SpatialData(
    "src/tests/data/gis/side_by_side/side_by_side.shp",
)

land_types = SpatialData(
    "src/tests/data/gis/side_by_side/side_by_side.shp",
    metrics={
        ProximityRisk("land_use", "forest"): 1.0,
        ProximityRisk("land_use", "fields"): 2.0,
        ProximityRisk("land_use", "built-up"): 3.0,
        ProximityRisk("land_use", "plantations"): 4.0,
    }
)

forest = SpatialData(
    "src/tests/data/gis/forest/forest.shp",
    metrics={
        ProximityRisk("land_use", "forest", 200): 1.0,
        # ProximityRisk("land_use", "forest", 50): 1.0,
    }
)
fields = SpatialData(
    "src/tests/data/gis/fields/fields.shp",
    metrics={
        ProximityRisk("land_use", "fields", 200): 1.0
    }
)
plantations = SpatialData(
    "src/tests/data/gis/plantations/plantations.shp",
    metrics={
        ProximityRisk("land_use", "plantations", 200): 1.0
    }
)
built_up = SpatialData(
    "src/tests/data/gis/built-up/built-up.shp",
    metrics={
        ProximityRisk("land_use", "built-up", 200): 1.0
    }
)

temporal_data = {
    "scaling": TemporalData(
        time_data_dict={
            dt.time(hour=0, minute=0): 1.0,
            dt.time(hour=1, minute=0): 1.5,
            dt.time(hour=2, minute=0): 2.0,
            dt.time(hour=3, minute=0): 1.5,
        },
        cycle_duration=dt.timedelta(hours=4, minutes=00),
        temporal_type=TemporalType.CYCLIC,
    ),
    # "spatial": TemporalData(
    #     time_data_dict={
    #         dt.time(hour=0, minute=0): forest,
    #         dt.time(hour=3, minute=0): fields,
    #         dt.time(hour=6, minute=0): built_up,
    #         dt.time(hour=9, minute=0): plantations,
    #     },
    #     cycle_duration=dt.timedelta(hours=12, minutes=00),
    #     temporal_resolution=dt.timedelta(minutes=30),
    #     temporal_type=TemporalType.CYCLIC,
    # )
}

#
# 3 Build environment
#
environment = Environment(
    spatial_data={"land_types": land_types},
    spatial_reference_data=side_by_side,
    spatial_resolution=50,  # metres
    temporal_data=temporal_data,
)
environment.calculate()

#
# 4 Mobility model
#
mobility_kde = KDE(kernel="gaussian", bandwidth=200)
mobility_density = DensityModel(
    sigma0=200.0,
    v=10.0,
    k=3.0,
    timestep=dt.timedelta(minutes=15)
)

#
# 5 Select trajectory
#
traj = trajectories[0]
times = get_times(traj.start_time, traj.end_time, dt.timedelta(minutes=30))
resampled = traj.resample(times, "nearest")

traj = resampled

#
# 6 Compute densities
#
rho1 = mobility_kde.distribution(traj, environment.gdf_raster)
rho2 = mobility_density.distribution(traj, environment.gdf_raster)

#
# 7 Plot outputs
#
rho1["normalised"] = rho1.density / rho1.density.sum()
rho2["normalised"] = rho2.density / rho2.density.sum()

dens_max = max(rho1["normalised"].max(), rho2["normalised"].max())

ax = rho1.plot("normalised")#, vmin=0.0, vmax=dens_max)
ax.scatter(traj.data.x, traj.data.y, color="red", s=10, alpha=0.5)
# fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
ax.set_title("KDE")
plt.show()

ax = rho2.plot("normalised", vmin=0.0, vmax=dens_max)
ax.scatter(traj.data.x, traj.data.y, color="red", s=10, alpha=0.5)
# fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
ax.set_title("Gaussian integration")
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
# fix_axis_limits(ax, buffer=1000, x=traj.data.x, y=traj.data.y)
ax.set_title("Difference")
plt.show()
