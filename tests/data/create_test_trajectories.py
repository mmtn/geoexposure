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
import pandas as pd


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
        y_ii = (ii * y_movement) + x_start
        current_time += frequency
        if (current_time > end_time) or (ii > max_entries) or (x_ii > max_x_value):
            break
        x.append(x_ii)
        y.append(y_ii)
        t.append(current_time)
        ii += 1

    return pd.DataFrame(data={"datetime": t, "latitude": x, "longitude": y})


#

ANCHOR_TIME = dt.datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)
MAX_DURATION = dt.timedelta(days=14)

x_start = 0.0  # metres
y_start = 500.0  # metres
x_max = 4000.0

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

high_freq_df.to_csv("data/trajectories/high_frequency.csv", index=False)
low_freq_df.to_csv("data/trajectories/low_frequency.csv", index=False)
