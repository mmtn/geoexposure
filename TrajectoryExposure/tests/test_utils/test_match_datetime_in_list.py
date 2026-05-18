import datetime as dt

import pytest

from TrajectoryExposure.utils import REFERENCE_TIME, match_datetime_in_list

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rt(hours, minutes):
    """Return REFERENCE_TIME offset by the given hours and minutes."""
    return REFERENCE_TIME + dt.timedelta(hours=hours, minutes=minutes)


def _run_test(to, cycle, expected_list):
    """Iterate over TEST_DATETIMES and assert each result matches the
    corresponding entry in expected_list. Entries that are exception
    types are asserted to raise rather than return.
    """
    for test_value, expected in zip(TEST_DATETIMES, expected_list):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                match_datetime_in_list(
                    test_value,
                    DATA_DATETIMES,
                    cycle=cycle,
                    to=to,
                )
        else:
            result = match_datetime_in_list(
                test_value,
                DATA_DATETIMES,
                cycle=cycle,
                to=to,
            )
            assert result == expected


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Timestamps within cycle with assigned data
DATA_DATETIMES = [
    rt(0, 0),  # 00:00
    rt(0, 30),  # 00:30
    rt(1, 0),  # 01:00
]

# Timestamps to input to tests
TEST_DATETIMES = [
    REFERENCE_TIME - dt.timedelta(minutes=25),  # -00:25
    rt(0, 0),  # 00:00
    rt(0, 5),  # 00:05
    rt(0, 25),  # 00:25
    rt(1, 0),  # 01:00
    rt(1, 25),  # 01:25
    rt(1, 45),  # 01:45
    rt(2, 55),  # 02:55
    rt(3, 59),  # 03:59
    rt(4, 1),  # 04:01
]

CYCLE = dt.timedelta(hours=2)


# ===========================================================================
# cycle=CYCLE (cyclic matching enabled)
# ===========================================================================


class TestMatchDatetimeInListWithCycle:
    def test_nearest(self):
        expected = [
            rt(0, 0),  # -00:25 -> 00:00
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 0),  # 00:05  -> 00:00
            rt(0, 30),  # 00:25  -> 00:30
            rt(1, 0),  # 01:00  -> 01:00
            rt(1, 0),  # 01:25  -> 01:00
            rt(0, 0),  # 01:45  -> 00:00
            rt(1, 0),  # 02:55  -> 01:00
            rt(0, 0),  # 03:59  -> 00:00
            rt(0, 0),  # 04:01  -> 00:00
        ]
        _run_test(to="nearest", cycle=CYCLE, expected_list=expected)

    def test_floor(self):
        expected = [
            rt(1, 0),  # -00:25 -> 01:00
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 0),  # 00:05  -> 00:00
            rt(0, 0),  # 00:25  -> 00:00
            rt(1, 0),  # 01:00  -> 01:00
            rt(1, 0),  # 01:25  -> 01:00
            rt(1, 0),  # 01:45  -> 01:00
            rt(0, 30),  # 02:55  -> 00:30
            rt(1, 0),  # 03:59  -> 01:00
            rt(0, 0),  # 04:01  -> 00:00
        ]
        _run_test(to="floor", cycle=CYCLE, expected_list=expected)

    def test_ceil(self):
        expected = [
            rt(0, 0),  # -00:25 -> 00:00
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 30),  # 00:05  -> 00:30
            rt(0, 30),  # 00:25  -> 00:30
            rt(1, 0),  # 01:00  -> 01:00
            rt(0, 0),  # 01:25  -> 00:00
            rt(0, 0),  # 01:45  -> 00:00
            rt(1, 0),  # 02:55  -> 01:00
            rt(0, 0),  # 03:59  -> 00:00
            rt(0, 30),  # 04:01  -> 00:30
        ]
        _run_test(to="ceil", cycle=CYCLE, expected_list=expected)


# ===========================================================================
# cycle=None (no cyclic matching, out-of-range inputs raise ValueError)
# ===========================================================================


class TestMatchDatetimeInListWithoutCycle:
    def test_nearest(self):
        expected = [
            ValueError,  # -00:25 -> outside cycle -> error
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 0),  # 00:05  -> 00:00
            rt(0, 30),  # 00:25  -> 00:30
            rt(1, 0),  # 01:00  -> 01:00
            ValueError,  # 01:25  -> outside cycle -> error
            ValueError,  # 01:45  -> outside cycle -> error
            ValueError,  # 02:55  -> outside cycle -> error
            ValueError,  # 03:59  -> outside cycle -> error
            ValueError,  # 04:01  -> outside cycle -> error
        ]
        _run_test(to="nearest", cycle=None, expected_list=expected)

    def test_floor(self):
        expected = [
            ValueError,  # -00:25 -> outside cycle -> error
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 0),  # 00:05  -> 00:00
            rt(0, 0),  # 00:25  -> 00:00
            rt(1, 0),  # 01:00  -> 01:00
            ValueError,  # 01:25  -> outside cycle -> error
            ValueError,  # 01:45  -> outside cycle -> error
            ValueError,  # 02:55  -> outside cycle -> error
            ValueError,  # 03:59  -> outside cycle -> error
            ValueError,  # 04:01  -> outside cycle -> error
        ]
        _run_test(to="floor", cycle=None, expected_list=expected)

    def test_ceil(self):
        expected = [
            ValueError,  # -00:25 -> outside cycle -> error
            rt(0, 0),  # 00:00  -> 00:00
            rt(0, 30),  # 00:05  -> 00:30
            rt(0, 30),  # 00:25  -> 00:30
            rt(1, 0),  # 01:00  -> 01:00
            ValueError,  # 01:25  -> no value above -> error
            ValueError,  # 01:45  -> no value above -> error
            ValueError,  # 02:55  -> outside cycle -> error
            ValueError,  # 03:59  -> outside cycle -> error
            ValueError,  # 04:01  -> outside cycle -> error
            ValueError,  # 04:01  -> outside cycle -> error
        ]
        _run_test(to="ceil", cycle=None, expected_list=expected)
