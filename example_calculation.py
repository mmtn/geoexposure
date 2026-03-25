import datetime as dt

from matplotlib import pyplot as plt

from src import Environment, SpatialData, TemporalData, calculate_exposure, utils
from src.Enums import TemporalType
from src.metrics.Proximity import ProximityRisk
from src.models.KDE import KDE

#
# Load generated geometries
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
# Read generated data
#
mobility_data_path = "src/tests/data/trajectories"
trajectories = utils.read_csv_directory(mobility_data_path)

# for t in trajectories:
#     times = get_times(t.start_time, t.end_time, dt.timedelta(minutes=30))
#     resampled = t.resample(times, "nearest")

#

environment = Environment(
    spatial_data={"land_types": land_types},
    spatial_reference_data=side_by_side,
    spatial_resolution=50,  # metres
    temporal_data=temporal_data,
)
environment.calculate()


mobility = KDE(kernel="gaussian", bandwidth=200)

#

exp, snp = calculate_exposure(
    trajectories[0],
    mobility,
    environment,
    temporal_resolution=dt.timedelta(minutes=15),
    return_snapshots=True,
    env_sampling_method="interp"
)

df_exposure_only = exp.loc[
    :,
    ~exp.columns.isin(["geometry", "scaling"]) &
    ~exp.columns.str.startswith("window")
]
exposure_sum = df_exposure_only.sum(axis=1)
exp["exposure_total"] = exposure_sum


# fig, ax = plt.subplots()
#
# for col in exp.columns:
#     if col.startswith("window") or col.startswith("scaling"):
#         continue
#
#     ax.plot(
#         exp["window_centre"],
#         exp[col] * exp["scaling"],
#         '-',
#         label=col
#     )
#
# plt.show()

