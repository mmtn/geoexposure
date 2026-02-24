import datetime as dt

import numpy as np
import pytest

from src.utils import REFERENCE_TIME, match_datetime_in_list

# Input values
dt_test = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=5),  # 00:05
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=25),  # 00:25
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=25),  # 01:25
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=45),  # 01:45
]

# Expected outputs
cycle_True_nearest = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:45 -> 00:00
]

cycle_True_floor = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:45 -> 01:00
]

cycle_True_ceil = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:05 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:45 -> 00:00
]

cycle_False_nearest = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:45 -> 01:00
]

cycle_False_floor = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:45 -> 01:00
]

cycle_False_ceil = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:05 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    ValueError,  # 01:25 -> error
    ValueError,  # 01:45 -> error
]


@pytest.mark.parametrize(
    "cycle_arg, to, expected",
    [
        (dt.timedelta(hours=2), "nearest", cycle_True_nearest),
        (dt.timedelta(hours=2), "floor", cycle_True_floor),
        (dt.timedelta(hours=2), "ceil", cycle_True_ceil),
        (None, "nearest", cycle_False_nearest),
        (None, "floor", cycle_False_floor),
        (None, "ceil", cycle_False_ceil),
    ],
)
def test_match_dt_in_list(cycle_arg, to, expected):
    dt_list = [
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),
        REFERENCE_TIME + dt.timedelta(hours=1, minutes=0)
    ]
    for dt_input, exp in zip(dt_test, expected):
        if isinstance(exp, type) and issubclass(exp, Exception):
            with pytest.raises(exp):  # expect an exception
                match_datetime_in_list(dt_input, dt_list, cycle=cycle_arg, to=to)
        else:
            result = match_datetime_in_list(dt_input, dt_list, cycle=cycle_arg, to=to)
            assert result == exp
