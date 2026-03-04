import datetime as dt

import pytest

from src.utils import REFERENCE_TIME, match_datetime_in_list

#

#
# Inputs
#
# NOTE: list of datetimes to match to is in the function below


#
# Expected outputs
#
cycle_True_nearest = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # -00:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:45 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 02:55 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 03:59 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 04:01 -> 00:00
]

cycle_True_floor = [
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # -00:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:45 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 02:55 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 03:59 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 04:01 -> 00:00
]

cycle_True_ceil = [
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # -00:25 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:05 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 01:45 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 02:55 -> 01:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 03:59 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 04:01 -> 00:00
]

cycle_False_nearest = [
    ValueError,  # -00:25 -> outside cycle -> error
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    ValueError,  # 01:25 -> outside cycle -> error
    ValueError,  # 01:45 -> outside cycle -> error
    ValueError,  # 02:55 -> outside cycle -> error
    ValueError,  # 03:59 -> outside cycle -> error
    ValueError,  # 04:01 -> outside cycle -> error
]

cycle_False_floor = [
    ValueError,  # -00:25 -> outside cycle -> error
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:05 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:25 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    ValueError,  # 01:25 -> outside cycle -> error
    ValueError,  # 01:45 -> outside cycle -> error
    ValueError,  # 02:55 -> outside cycle -> error
    ValueError,  # 03:59 -> outside cycle -> error
    ValueError,  # 04:01 -> outside cycle -> error
]

cycle_False_ceil = [
    ValueError,  # -00:25 -> outside cycle -> error
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00 -> 00:00
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:05 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),  # 00:25 -> 00:30
    REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00 -> 01:00
    ValueError,  # 01:25 -> no value above -> error
    ValueError,  # 01:45 -> no value above -> error
    ValueError,  # 02:55 -> outside cycle -> error
    ValueError,  # 03:59 -> outside cycle -> error
    ValueError,  # 04:01 -> outside cycle -> error
]


@pytest.mark.parametrize(
    "cycle_duration_hours, to, expected",
    [
        (2, "nearest", cycle_True_nearest),
        (2, "floor", cycle_True_floor),
        (2, "ceil", cycle_True_ceil),
        (None, "nearest", cycle_False_nearest),
        (None, "floor", cycle_False_floor),
        (None, "ceil", cycle_False_ceil),
    ],
)
def test_match_dt_in_list(cycle_duration_hours, to, expected):
    if cycle_duration_hours is None:
        cycle = None
    else:
        cycle = dt.timedelta(hours=cycle_duration_hours)

    DATA_DATETIMES = [
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=30),
        REFERENCE_TIME + dt.timedelta(hours=1, minutes=0)
    ]

    TEST_DATETIMES = [
        REFERENCE_TIME - dt.timedelta(hours=0, minutes=25),  # -00:25
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=0),  # 00:00
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=5),  # 00:05
        REFERENCE_TIME + dt.timedelta(hours=0, minutes=25),  # 00:25
        REFERENCE_TIME + dt.timedelta(hours=1, minutes=0),  # 01:00
        REFERENCE_TIME + dt.timedelta(hours=1, minutes=25),  # 01:25
        REFERENCE_TIME + dt.timedelta(hours=1, minutes=45),  # 01:45
        REFERENCE_TIME + dt.timedelta(hours=2, minutes=55),  # 02:55
        REFERENCE_TIME + dt.timedelta(hours=3, minutes=59),  # 03:59
        REFERENCE_TIME + dt.timedelta(hours=4, minutes=1),  # 04:01
    ]

    for test_value, exp in zip(TEST_DATETIMES, expected):
        if isinstance(exp, type) and issubclass(exp, Exception):
            with pytest.raises(exp):  # expect an exception
                match_datetime_in_list(
                    test_value,
                    DATA_DATETIMES,
                    cycle=cycle,
                    to=to
                )
        else:
            result = match_datetime_in_list(
                test_value,
                DATA_DATETIMES,
                cycle=cycle,
                to=to
            )
            assert result == exp
