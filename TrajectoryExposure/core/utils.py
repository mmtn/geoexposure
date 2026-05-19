"""utils.py contains several helpful functions to perform common tasks.

Functions in this file handle:
- management of datetimes (rounding, evenly spaces time between bounds, etc).
- handling spatial operations on GeoDataFrames (rasterise, centroid calculation)
"""

import bisect
import calendar
import datetime as dt
import logging
from collections.abc import Sequence
from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely import Point, Polygon

logger = logging.getLogger(__name__)

REFERENCE_TIME = dt.datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
HOURLY = [dt.time(hour=h) for h in range(24)]
DAILY = [
    dt.date(year=REFERENCE_TIME.year, month=1, day=1) + ii * dt.timedelta(days=1)
    for ii in range(366)
]
MONTHLY = [dt.date(year=REFERENCE_TIME.year, month=(m + 1), day=1) for m in range(12)]


def round_datetime(
        timestamp: dt.datetime, delta: dt.timedelta, to: str = "nearest",
) -> dt.datetime:
    """Round a datetime to a multiple of a timedelta.

    Args:
        timestamp: The datetime to round.
        delta: Increment to round to (e.g. ``dt.timedelta(minutes=15)``).
        to: Rounding method: ``"nearest"``, ``"floor"``, or ``"ceil"``.

    Returns:
        The rounded datetime.
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
        start_time: dt.datetime, end_time: dt.datetime, delta: dt.timedelta,
) -> list[dt.datetime]:
    """Return datetimes evenly spaced at the given interval."""
    start_new = round_datetime(start_time, delta, to="floor")
    end_new = round_datetime(end_time, delta, to="ceil")
    num_times = int((end_new - start_new) / delta)
    return [start_new + (ii * delta) for ii in range(num_times)]


def get_time_windows(
        start_time: dt.datetime, end_time: dt.datetime, delta: dt.timedelta,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Return start and end times for windows of the specified duration."""
    if delta is None:
        return [(start_time, end_time)]

    start_rounded_up = round_datetime(start_time, delta, to="ceil")
    end_rounded_down = round_datetime(end_time, delta, to="floor")
    num_times = int((end_rounded_down - start_rounded_up) / delta)

    windows = [
        (start_rounded_up + (ii * delta), start_rounded_up + ((ii + 1) * delta))
        for ii in range(num_times)
    ]

    # Prepend partial window at the start
    if start_rounded_up > start_time:
        windows.insert(0, (start_time, start_rounded_up))

    # Append partial window at the end
    if end_rounded_down < end_time:
        windows.append((end_rounded_down, end_time))

    return windows


def check_iter_types(iterable: Sequence, data_type: type) -> bool:
    """Check if all items in an iterable have the same data type."""
    return all(isinstance(item, data_type) for item in iterable)


def get_cyclic_timestamp(
        dt_input: dt.time | dt.date | dt.datetime, cycle_duration: dt.timedelta,
) -> dt.datetime:
    """Map a date/time into a repeating cycle anchored at `REFERENCE_TIME`.

    Notes:
        - `dt.time` assumes a daily cycle.
        - `dt.date` assumes a yearly cycle.
        - `dt.datetime` assumes a yearly cycle (month/day/time retained).

    Args:
        dt_input: Input timestamp.
        cycle_duration: Cycle length. Must be positive.

    Returns:
        A datetime in the half-open range
        [`REFERENCE_TIME`, `REFERENCE_TIME` + cycle_duration).
    """
    if cycle_duration.total_seconds() <= 0:
        raise ValueError("cycle duration must be greater than 0")

    if isinstance(dt_input, dt.time):
        ts = REFERENCE_TIME.replace(
            hour=dt_input.hour, minute=dt_input.minute, second=dt_input.second,
        )

    elif isinstance(dt_input, dt.date) and not isinstance(dt_input, dt.datetime):
        day = dt_input.day

        # Clamp Feb 29 to Feb 28 if the reference year is not a leap year
        feb_leap_year = 29
        february = 2
        is_leap_year = calendar.isleap(REFERENCE_TIME.year, )
        if dt_input.month == february and day == feb_leap_year and not is_leap_year:
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
    """Return a datetime from the input list using chosen matching logic (floor, ceil, nearest)."""
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
        match = _match_datetime_floor(first, index, sorted_list, target)
    elif to == "ceil":
        match = _match_datetime_ceil(index, last, sorted_list, target)
    elif to == "nearest":
        match = _match_datetime_nearest(first, index, last, sorted_list, target)
    else:
        raise ValueError(f"unknown rounding method: {to}")

    if cycle is not None and match == last:
        match = first

    return match


def _match_datetime_nearest(
        first: datetime,
        index: int,
        last: datetime,
        sorted_list: list[datetime],
        target: datetime
        ) -> datetime:
    """Return nearest datetime (helper for ``match_datetime_in_list``)."""
    if index == 0:
        match = first
    elif index == len(sorted_list):
        match = last
    else:
        before = sorted_list[index - 1]
        after = sorted_list[index]
        # "less than or equal" here returns earliest of two equidistant values
        if target - before <= after - target:
            match = before
        else:
            match = after
    return match


def _match_datetime_ceil(
        index: int,
        last: datetime,
        sorted_list: list[datetime],
        target: datetime
        ) -> datetime:
    """Return closest later datetime (helper for ``match_datetime_in_list``)."""
    if target > last:
        raise ValueError(f"no datetime >= target: {target}")
    return sorted_list[index]


def _match_datetime_floor(
        first: datetime,
        index: int,
        sorted_list: list[datetime],
        target: datetime
        ) -> datetime:
    """Return closest earlier datetime (helper for ``match_datetime_in_list``)."""
    if target < first:
        raise ValueError(f"no datetime <= target: {target}")
    if index < len(sorted_list) and target == sorted_list[index]:
        match = sorted_list[index]
    else:
        match = sorted_list[index - 1]
    return match


def rasterise(gdf: gpd.GeoDataFrame, pixel_size_metres: int | float) -> gpd.GeoDataFrame:
    """Return a rasterised version of the input GeoDataFrame at the given resolution."""

    def round_down(value: float, precision: float) -> float:
        return np.floor(value / precision) * precision

    def round_up(value: float, precision: float) -> float:
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
    total = int(ii_max) * int(jj_max)

    logger.info(
        f"Calculating raster at {px_m}m resolution "
        f"({ii_max:.0f} x {jj_max:.0f} = {total:.0f})",
    )

    polys: list[Polygon] = []
    cx_list: list[float] = []
    cy_list: list[float] = []

    for jj in range(int(jj_max)):
        for ii in range(int(ii_max)):
            x0 = x_min + (px_m * ii)
            x1 = x_min + (px_m * (ii + 1))
            y0 = y_min + (px_m * jj)
            y1 = y_min + (px_m * (jj + 1))
            cx = x0 + px_m * 0.5
            cy = y0 + px_m * 0.5
            points_list = ((x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0))
            polys.append(Polygon(points_list))
            cx_list.append(cx)
            cy_list.append(cy)

    return gpd.GeoDataFrame(
        {
            "cx": cx_list,
            "cy": cy_list
        },
        geometry=polys,
        crs=gdf.crs,
    )


def get_gdf_centroids(
        gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[Point], np.ndarray]:
    """Return the centroids of all geometries as list and numpy array."""
    if bounds is not None:
        gdf = gdf.clip(bounds)
    centroids = gdf.geometry.centroid  # GeoSeries[Point]
    centroids_np = np.column_stack((centroids.x.to_numpy(), centroids.y.to_numpy()))
    return centroids, centroids_np


