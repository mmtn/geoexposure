import datetime as dt
import warnings

import numpy as np
import pandas as pd

from . import Mobility, Environment, utils
from .data import Trajectory


def calculate_exposure(
    trajectory: Trajectory,
    mobility: Mobility,
    environment: Environment,
    start_time: dt.datetime = None,
    end_time: dt.datetime = None,
    temporal_resolution: dt.timedelta = None,
    env_sampling_method: str = "interp",
    return_snapshots: bool = False,
) -> pd.DataFrame:
    if start_time is None:
        start_time = trajectory.data["datetime"].min()

    if end_time is None:
        end_time = trajectory.data["datetime"].max()

    # Sample at highest frequency provided
    resolutions = (temporal_resolution, environment.temporal_resolution)

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

        if start < trajectory.start_time:
            start = trajectory.start_time.to_pydatetime()

        if end > trajectory.end_time:
            end = trajectory.end_time.to_pydatetime()

        window = trajectory.data_in_window(
            start=start,
            end=end,
            include_first=(ii == 0),
            include_last=(ii == len(windows) - 1),
        )
        length = end - start
        durations[ii] = length.total_seconds()
        centres.append(start + length / 2)

        scaling[ii] = environment._scaling_factors(
            centres[ii], method=env_sampling_method
        )

        if len(window) == 0:
            print(f"Window {ii + 1:<5d}|   {start} - {end}   |   NO DATA [WARNING]")

        rho = mobility.distribution(window, environment)
        exposure_sources = environment.sample(centres[ii], method=env_sampling_method)
        normalised_density = rho.density / rho.density.sum()
        snapshot_ii = (
            exposure_sources.drop(columns=["geometry"]).mul(normalised_density, axis=0)
            * durations[ii]
        )
        # Multiply by window duration to get seconds of exposure to each environment
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


def exposure_sums(
    trajectories: list[Trajectory],
    mobility: Mobility,
    environment: Environment,
    timestep: dt.datetime = None,
    per_second: bool = False,
) -> pd.DataFrame:

    results = list()

    for trajectory in trajectories:

        exposure_df = calculate_exposure(
            trajectory,
            mobility,
            environment,
            temporal_resolution=timestep,
            return_snapshots=False,
        )
        df_exposure_only = exposure_df.loc[
            :,
            ~exposure_df.columns.isin(["geometry", "scaling"])
            & ~exposure_df.columns.str.startswith("window"),
        ]
        exposure_sum = df_exposure_only.sum()
        if per_second:
            exposure_sum = exposure_sum / trajectory.duration_in_seconds
        exposure_sum["filename"] = trajectory.csv_file
        results.append(exposure_sum)

    df = pd.DataFrame(results)
    return df
