import bisect
import datetime as dt
from collections import abc

import geopandas as gpd
import numpy as np
from shapely import Polygon

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
    Args:
        timestamp (dt.datetime): The datetime to round (aware or naive).
        delta (dt.timedelta): The increment to round to (e.g. timedelta(minutes=15)).
        to (str): 'nearest', 'floor', or 'ceil'.
    Returns:
        datetime: Rounded datetime (same tzinfo as input).
    """

    # Time since "anchor" (midnight) in seconds
    anchor = timestamp.replace(
        year=2020,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )
    seconds = (timestamp - anchor).total_seconds()
    step = delta.total_seconds()

    if to == "nearest":
        rounded = round(seconds / step) * step
    elif to == "floor":
        rounded = (seconds // step) * step
    elif to == "ceil":
        rounded = -(-seconds // step) * step  # ceiling division
    else:
        raise ValueError("to must be 'nearest', 'floor', or 'ceil'")

    return anchor + dt.timedelta(seconds=rounded)


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
