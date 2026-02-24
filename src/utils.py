import bisect
import datetime as dt
import os
from collections import abc

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon

from src.data.Trajectory import Trajectory

#

REFERENCE_TIME = dt.datetime(
    year=2020,
    month=1,
    day=1,
    hour=0,
    minute=0,
    second=0,
    microsecond=0
)

HOURLY = [
    dt.time(hour=h)
    for h in range(24)
]

DAILY = [
    dt.date(year=REFERENCE_TIME.year, month=1, day=1) + ii * dt.timedelta(days=1)
    for ii in range(366)
]

MONTHLY = [
    dt.date(year=REFERENCE_TIME.year, month=(m + 1), day=1)
    for m in range(12)
]


def metric_name(metric, args, join_str="_"):
    if args is None:
        arg_string = None
    elif isinstance(args, str):
        arg_string = args
    elif isinstance(args, abc.Iterable):
        arg_string = join_str.join([f"{arg}" for arg in args])
    else:
        arg_string = f"{args}"

    if arg_string is None:
        return f"{metric}"
    else:
        return f"{metric}_{arg_string}"


def nearest_window(timestamp, window_starts):
    pass


def nearest_window_cyclic(timestamp, window_starts, cycle_duration):
    pass


def round_datetime(timestamp, delta, to="nearest"):
    """
    Round a datetime to a multiple of a timedelta.

    :param timestamp: the datetime to round
    :param delta: increment to round to (e.g. dt.timedelta(minutes=15))
    :param to: 'nearest', 'floor', or 'ceil'
    :return: rounded datetime.
    """
    time_seconds = (timestamp - REFERENCE_TIME).total_seconds()
    delta_seconds = delta.total_seconds()

    if to == "nearest":
        rounded = round(time_seconds / delta_seconds) * delta_seconds
    elif to == "floor":
        rounded = (time_seconds // delta_seconds) * delta_seconds
    elif to == "ceil":
        rounded = -(-time_seconds // delta_seconds) * delta_seconds
    else:
        raise ValueError("'to' must be 'nearest', 'floor', or 'ceil'")

    return REFERENCE_TIME + dt.timedelta(seconds=rounded)


def get_times(start_time, end_time, delta):
    start_new = round_datetime(start_time, delta, to="floor")
    end_new = round_datetime(end_time, delta, to="ceil")
    num_times = int((end_new - start_new) / delta)
    return [start_new + (ii * delta) for ii in range(num_times)]


def check_iter_types(iterable, data_type):
    return all(isinstance(item, data_type) for item in iterable)


def raster(gdf, pixel_size_metres):
    def round_down(value, precision):
        return np.floor(value / precision) * precision

    def round_up(value, precision):
        return np.ceil(value / precision) * precision

    px_m = pixel_size_metres
    gdf_x_min, gdf_y_min, gdf_x_max, gdf_y_max = gdf.total_bounds
    gdf_x_size, gdf_y_size = (gdf_x_max - gdf_x_min), (gdf_y_max - gdf_y_min)

    x_min = round_down(gdf_x_min, px_m)
    y_min = round_down(gdf_y_min, px_m)
    x_size = round_up(gdf_x_size, px_m)
    y_size = round_up(gdf_y_size, px_m)
    ii_max = x_size / px_m
    jj_max = y_size / px_m
    total_points = ii_max * jj_max

    print(
        f"Rasterising {px_m}m resolution "
        f"({ii_max:.0f} x {jj_max:.0f} = {total_points:.0f})"
    )

    raster_list = [
        Polygon(
            (
                (x_min + (px_m * ii), y_min + (px_m * jj)),
                (x_min + (px_m * ii), y_min + (px_m * (jj + 1))),
                (x_min + (px_m * (ii + 1)), y_min + (px_m * (jj + 1))),
                (x_min + (px_m * (ii + 1)), y_min + (px_m * jj)),
                (x_min + (px_m * ii), y_min + (px_m * jj)),
            )
        )
        for jj in range(int(jj_max))
        for ii in range(int(ii_max))
    ]

    return gpd.GeoDataFrame(geometry=raster_list, crs=gdf.crs)


def get_cyclic_timestamp(dt_object):
    """
    dt.time objects assume daily cycle
    dt.date objects assume yearly cycle
    dt.datetime objects: assume yearly cycle

    :param dt_object:
    :return:
    """
    # TODO: test TemporalData.get_cyclic_timestamp()
    # TODO: match/strip some elements of datetime before match
    if isinstance(dt_object, dt.time):
        ts = REFERENCE_TIME.replace(
            hour=dt_object.hour,
            minute=dt_object.minute,
            second=dt_object.second
        )
    elif isinstance(dt_object, dt.date):
        ts = REFERENCE_TIME.replace(
            year=REFERENCE_TIME.year,
            month=dt_object.month,
            day=dt_object.day
        )
    elif isinstance(dt_object, dt.datetime):
        ts = REFERENCE_TIME.replace(
            year=REFERENCE_TIME.year,
            month=dt_object.month,
            day=dt_object.day,
            hour=dt_object.hour,
            minute=dt_object.minute,
            second=dt_object.second
        )
    else:
        raise ValueError(f"unknown timestamp type: {type(dt_object)}")
    return ts


def match_datetime_in_list(target, sorted_datetimes, cycle=None, to="nearest"):
    first = sorted_datetimes[0]
    last = sorted_datetimes[-1]

    if cycle is not None:
        dt_cycle = first + cycle
        sorted_datetimes.append(dt_cycle)

    i = bisect.bisect_left(sorted_datetimes, target)

    if to == "floor":
        if target < first:
            raise ValueError(f"no datetime <= target: {target}")
        elif i < len(sorted_datetimes) and target == sorted_datetimes[i]:
            result = sorted_datetimes[i]
        else:
            result = sorted_datetimes[i - 1]

    elif to == "ceil":
        if target > last:
            raise ValueError(f"no datetime >= target: {target}")
        else:
            result = sorted_datetimes[i]

    elif to == "nearest":
        if i == 0:
            result = first
        elif i == len(sorted_datetimes):
            result = last
        else:
            before = sorted_datetimes[i - 1]
            after = sorted_datetimes[i]
            if target - before <= after - target:
                result = before
            else:
                result = after

    else:
        ValueError(f"unknown rounding method: {to}")

    if cycle is not None and result == dt_cycle:
        result = first

    return result


def read_csv_directory(data_directory):
    """
    Reads data from CSV files in given directory to Trajectory objects

    :param data_directory: contains CSV files with datetime, latitude, longitude
    :return: list of Trajectory objects
    """
    csv_files = [
        os.path.join(data_directory, file)
        for file in os.listdir(data_directory)
        if file.endswith("csv")
    ]
    return [
        Trajectory(pd.read_csv(csv))
        for csv in csv_files
    ]
