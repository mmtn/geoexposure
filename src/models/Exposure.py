import datetime as dt

import numpy as np
from matplotlib import pyplot as plt

from src import utils


def exposure(
        trajectory,
        mobility,
        environment,
        start_time=None,
        end_time=None,
        temporal_resolution=None
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

    delta = dt.timedelta(minutes=temporal_resolution)
    times = utils.get_times(start_time, end_time, delta)

    exposures = list()
    for time in times:
        env = environment.sample(time)
        env.plot("exposure")
        print(env)
        break

        # aggregate = trajectory.aggregate(time, time + delta)
        # rho = mobility(aggregate)
        # exposure_step = rho * env
        # exposures.append(exposure_step)

    plt.show()

    return np.array(exposures)
