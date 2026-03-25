import bisect
import calendar
import datetime as dt
import os
from collections import abc

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon, Point
from tqdm import tqdm

from src import Trajectory

#

REFERENCE_TIME = dt.datetime(
    year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
)

HOURLY = [dt.time(hour=h) for h in range(24)]

DAILY = [
    dt.date(year=REFERENCE_TIME.year, month=1, day=1) + ii * dt.timedelta(days=1)
    for ii in range(366)
]

MONTHLY = [dt.date(year=REFERENCE_TIME.year, month=(m + 1), day=1) for m in range(12)]


def metric_name(
    metric: str, args: abc.Iterable | str | None, join_str: str = "_"
) -> str:
    if args is None:
        arg_string = None
    elif isinstance(args, str):
        arg_string = args
    elif isinstance(args, abc.Iterable):
        arg_string = join_str.join([f"{arg}" for arg in args if arg is not None])
    else:
        arg_string = f"{args}"

    if arg_string is None:
        return f"{metric}"
    else:
        return f"{metric}_{arg_string}"


def round_datetime(
    timestamp: dt.datetime, delta: dt.timedelta, to: str = "nearest"
) -> dt.datetime:
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


def get_times(
    start_time: dt.datetime, end_time: dt.datetime, delta: dt.timedelta
) -> list:
    start_new = round_datetime(start_time, delta, to="floor")
    end_new = round_datetime(end_time, delta, to="ceil")
    num_times = int((end_new - start_new) / delta)
    return [start_new + (ii * delta) for ii in range(num_times)]


def get_time_windows(
    start_time: dt.datetime, end_time: dt.datetime, delta: dt.timedelta
) -> list[tuple[dt.datetime, dt.datetime]]:
    start_new = round_datetime(start_time, delta, to="floor")
    end_new = round_datetime(end_time, delta, to="ceil")
    num_times = int((end_new - start_new) / delta)
    return [
        (start_new + (ii * delta), start_new + ((ii + 1) * delta))
        for ii in range(num_times)
    ]


def check_iter_types(iterable: abc.Iterable, data_type: type) -> bool:
    return all(isinstance(item, data_type) for item in iterable)


def raster(gdf: gpd.GeoDataFrame, pixel_size_metres: int | float) -> gpd.GeoDataFrame:
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
        f"Calculating raster at {px_m}m resolution "
        f"({ii_max:.0f} x {jj_max:.0f} = {total_points:.0f})"
    )

    total = int(ii_max) * int(jj_max)
    raster_list = []

    with tqdm(total=total, desc="Rasterising") as pbar:
        for jj in range(int(jj_max)):
            for ii in range(int(ii_max)):
                raster_list.append(
                    Polygon(
                        (
                            (x_min + (px_m * ii), y_min + (px_m * jj)),
                            (x_min + (px_m * ii), y_min + (px_m * (jj + 1))),
                            (x_min + (px_m * (ii + 1)), y_min + (px_m * (jj + 1))),
                            (x_min + (px_m * (ii + 1)), y_min + (px_m * jj)),
                            (x_min + (px_m * ii), y_min + (px_m * jj)),
                        )
                    )
                )
                pbar.update(1)

    return gpd.GeoDataFrame(geometry=raster_list, crs=gdf.crs)


def get_cyclic_timestamp(
    dt_input: dt.time | dt.date | dt.datetime, cycle_duration: dt.timedelta
) -> dt.datetime:
    """
    dt.time objects assume daily cycle
    dt.date objects assume yearly cycle
    dt.datetime objects: assume yearly cycle

    :param dt_input:
    :return:
    """
    if cycle_duration.total_seconds() <= 0:
        raise ValueError("cycle duration must be greater than 0")

    if isinstance(dt_input, dt.time):
        ts = REFERENCE_TIME.replace(
            hour=dt_input.hour, minute=dt_input.minute, second=dt_input.second
        )

    elif isinstance(dt_input, dt.date) and not isinstance(dt_input, dt.datetime):
        day = dt_input.day

        # Clamp Feb 29 to Feb 28 if the reference year is not a leap year
        if (
            dt_input.month == 2
            and day == 29
            and not calendar.isleap(REFERENCE_TIME.year)
        ):
            day = 28

        ts = REFERENCE_TIME.replace(month=dt_input.month, day=day)

    elif isinstance(dt_input, dt.datetime):
        ts = REFERENCE_TIME.replace(
            month=dt_input.month,
            day=dt_input.day,
            hour=dt_input.hour,
            minute=dt_input.minute,
            second=dt_input.second,
        )

    else:
        raise ValueError(f"unknown timestamp type: {type(dt_input)}")

    while ts < REFERENCE_TIME:
        ts += cycle_duration

    while ts >= (REFERENCE_TIME + cycle_duration):
        ts -= cycle_duration

    return ts


def match_datetime_in_list(
    target: dt.datetime,
    datetime_list: list[dt.datetime],
    cycle: dt.timedelta | None = None,
    to: str = "nearest",
) -> dt.datetime:
    sorted_list = sorted(datetime_list)
    if sorted_list != datetime_list:
        raise RuntimeError("'datetime_list' must be sorted")

    first = sorted_list[0]
    last = sorted_list[-1]

    if cycle is None and (target < first or target > last):
        raise ValueError(f"target datetime outside listed values: {target} {cycle}")

    if cycle is not None:
        last = first + cycle
        sorted_list.append(last)
        # Move target into appropriate range of values
        while target < first:
            target += cycle
        while target > last:
            target -= cycle

    index = bisect.bisect_left(sorted_list, target)

    if to == "floor":
        if target < first:
            raise ValueError(f"no datetime <= target: {target}")
        elif index < len(sorted_list) and target == sorted_list[index]:
            match = sorted_list[index]
        else:
            match = sorted_list[index - 1]

    elif to == "ceil":
        if target > last:
            raise ValueError(f"no datetime >= target: {target}")
        else:
            match = sorted_list[index]

    elif to == "nearest":
        if index == 0:
            match = first
        elif index == len(sorted_list):
            match = last
        else:
            before = sorted_list[index - 1]
            after = sorted_list[index]
            if target - before <= after - target:
                match = before
            else:
                match = after

    else:
        ValueError(f"unknown rounding method: {to}")

    if cycle is not None and match == last:
        match = first

    return match


def read_csv_directory(data_directory: str, max_files: int | float = np.inf):
    """
    Reads data from CSV files in given directory to Trajectory objects

    Args:
        data_directory: contains CSV files with datetime, x, y
        max_files:

    Returns:
        list of Trajectory objects
    """
    csv_files = [
        os.path.join(data_directory, file)
        for file in os.listdir(data_directory)
        if file.endswith("csv")
    ]
    return [
        Trajectory(pd.read_csv(csv))
        for file_num, csv in enumerate(csv_files)
        if file_num < max_files
    ]


def get_gdf_centroids(
    gdf: gpd.GeoDataFrame, bounds: tuple | None = None
) -> tuple[list[Point], np.ndarray]:
    if bounds is not None:
        gdf = gdf.clip(bounds)
    points = [geom.centroid for geom in gdf.geometry]
    points_np = np.array([[point.x, point.y] for point in points])
    return points, points_np
