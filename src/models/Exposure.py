import datetime as dt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import utils
from src.data.Trajectory import Trajectory
from src.models.Environment import Environment
from src.models.Mobility import Mobility


def calculate_exposure(
        trajectory: Trajectory,
        mobility: Mobility,
        environment: Environment,
        start_time: dt.datetime = None,
        end_time: dt.datetime = None,
        temporal_resolution: dt.timedelta = None,
        env_sampling_method: str = "interp",
        return_snapshots: bool = False
):
    if start_time is None:
        start_time = trajectory.data["datetime"].min()

    if end_time is None:
        end_time = trajectory.data["datetime"].max()

    # Sample at highest frequency provided
    resolutions = (
        temporal_resolution,
        environment.temporal_resolution
    )

    temporal_resolution = min(res for res in resolutions if res is not None)

    if temporal_resolution is None:
        raise ValueError(
            "temporal_resolution must be defined by argument or from environment"
        )

    windows = utils.get_time_windows(start_time, end_time, temporal_resolution)
    num_windows = len(windows)
    scaling = np.zeros(num_windows)
    durations = np.zeros(num_windows)
    centres = list()
    snapshots = list()
    snapshot_sums = list()

    print(f"Computing exposure between {start_time} and {end_time}")
    print(f"Temporal resolution: {temporal_resolution}")
    print(f"{num_windows} windows")

    for ii, (start, end) in enumerate(windows):
        window = trajectory.data_in_window(start, end)
        length = end - start
        durations[ii] = length.total_seconds()
        centres.append(start + length / 2)

        rho = mobility.distribution(window, environment.gdf_raster)
        exposure_sources = environment.sample(centres[ii], method=env_sampling_method)
        scaling[ii] = environment._scaling_factors(centres[ii], method=env_sampling_method)
        normalised_density = rho.density / rho.density.sum()
        snapshot_ii = exposure_sources.drop(
            columns=["geometry"]
        ).mul(
            normalised_density,
            axis=0
        )
        sums_ii = snapshot_ii.sum()
        snapshot_sums.append(sums_ii)

        snapshot_ii["geometry"] = exposure_sources["geometry"]
        snapshots.append(snapshot_ii)

        print(f"Window {ii + 1:<5d}|   {start} - {end}   |   ({len(window)} points)")

    summary_df = pd.DataFrame(snapshot_sums)
    summary_df["scaling"] = scaling
    summary_df["window_start"] = [start for start, end in windows]
    summary_df["window_centre"] = centres
    summary_df["window_end"] = [end for start, end in windows]
    summary_df["window_length_seconds"] = durations

    print("Complete")

    if return_snapshots:
        return summary_df, snapshots
    else:
        return summary_df
