"""
# Test cases

Temporal data
- float
- SpatialData

Temporal scales
- frequency of exposure calculation (windowing trajectory data)
- frequency of changes to temporal data

SpatialData


NOTE: exposure calculation should always use the smallest window from both Exposure
and Environment
"""

import datetime as dt
import numpy as np
import pandas as pd

from src.data.Trajectory import DATETIME, X, Y


def get_xyt(
        start_time,
        end_time,
        frequency,
        movement_speed,
        max_x_value,
        max_entries=1e6
):
    x, y, t = list(), list(), list()
    x_movement = movement_speed * frequency.seconds
    y_movement = 0.0
    ii = 0
    current_time = start_time

    while True:
        x_ii = (ii * x_movement) + x_start
        y_ii = (ii * y_movement) + y_start
        current_time += frequency
        if (current_time > end_time) or (ii > max_entries) or (x_ii > max_x_value):
            break
        x.append(x_ii)
        y.append(y_ii)
        t.append(current_time)
        ii += 1

    return pd.DataFrame(data={DATETIME: t, X: x, Y: y})


#

ANCHOR_TIME = dt.datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)
MAX_DURATION = dt.timedelta(days=14)

x_start = 1.0  # metres
y_start = 500.0  # metres
x_max = 4001.0

low_frequency = dt.timedelta(hours=2)
high_frequency = dt.timedelta(minutes=1)
speed = 2 * 0.016666666667  # metres per second

#

# 1. Frequent GPS data
high_freq_df = get_xyt(
    ANCHOR_TIME,
    ANCHOR_TIME + MAX_DURATION,
    high_frequency,
    speed,
    x_max,
)

# 2. Sparse GPS data
low_freq_df = get_xyt(
    ANCHOR_TIME,
    ANCHOR_TIME + MAX_DURATION,
    low_frequency,
    speed,
    x_max,
)

# 3. High frequency with random missingness
total_entries = len(high_freq_df)
proportion_to_drop = 0.5
removal_indices = np.random.choice(
    range(total_entries),
    int(np.ceil(total_entries * proportion_to_drop)),
    replace=False
)
inconsistent_frequency_df = high_freq_df.copy().drop(removal_indices)

inconsistent_frequency_df.to_csv("trajectories/01_inconsistent_frequency.csv", index=False)
high_freq_df.to_csv("trajectories/02_high_frequency.csv", index=False)
low_freq_df.to_csv("trajectories/03_low_frequency.csv", index=False)
