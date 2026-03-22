import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from src import utils


def calculate_exposure(
        trajectory,
        mobility,
        environment,
        start_time=None,
        end_time=None,
        temporal_resolution=None,
        env_sampling_method="interp",
):
    if start_time is None:
        start_time = trajectory.data["datetime"].min()

    if end_time is None:
        end_time = trajectory.data["datetime"].max()

    # Sample at highest frequency provided
    if temporal_resolution is None:
        temporal_resolution = environment.temporal_resolution
    else:
        temporal_resolution = min(temporal_resolution, environment.temporal_resolution)

    print(f"Computing exposure between {start_time} and {end_time}")
    print(f"Temporal resolution: {temporal_resolution}")

    windows = utils.get_time_windows(start_time, end_time, temporal_resolution)
    exposures = np.zeros(len(windows))
    durations = np.zeros(len(windows))

    for ii, (start, end) in enumerate(windows):
        window = trajectory.data_in_window(start, end)
        length = end - start
        durations[ii] = length.total_seconds()

        center = start + length / 2
        env = environment.sample(center, method=env_sampling_method)
        rho = mobility.distribution(window, environment.gdf_raster)
        exposure = env.exposure * rho.density / rho.density.sum()
        exposures[ii] = exposure.sum()

    print("Complete")
    return np.array(exposures * durations)
