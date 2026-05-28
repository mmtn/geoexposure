import calendar
import datetime as dt

import pytest

from geoexposure.utils import REFERENCE_TIME, get_cyclic_timestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ONE_DAY = dt.timedelta(days=1)
ONE_HOUR = dt.timedelta(hours=1)
days_in_ref_year = 366 if calendar.isleap(REFERENCE_TIME.year) else 365
ONE_YEAR = dt.timedelta(days=days_in_ref_year)


def assert_in_cycle(result, cycle_duration):
    """Assert result sits within [REFERENCE_TIME, REFERENCE_TIME + cycle_duration]."""
    assert REFERENCE_TIME <= result <= REFERENCE_TIME + cycle_duration


# ===========================================================================
# dt.time inputs
# ===========================================================================


class TestTimeInput:
    def test_midnight_returns_reference_time(self):
        """dt.time(0,0,0) maps exactly onto REFERENCE_TIME."""
        result = get_cyclic_timestamp(dt.time(0, 0, 0), ONE_DAY)
        assert result == REFERENCE_TIME

    def test_midday(self):
        result = get_cyclic_timestamp(dt.time(12, 0, 0), ONE_DAY)
        expected = REFERENCE_TIME.replace(hour=12, minute=0, second=0)
        assert result == expected

    def test_end_of_day(self):
        result = get_cyclic_timestamp(dt.time(23, 59, 59), ONE_DAY)
        expected = REFERENCE_TIME.replace(hour=23, minute=59, second=59)
        assert result == expected

    def test_arbitrary_time(self):
        result = get_cyclic_timestamp(dt.time(8, 30, 15), ONE_DAY)
        expected = REFERENCE_TIME.replace(hour=8, minute=30, second=15)
        assert result == expected

    def test_result_is_datetime(self):
        """Return type should always be dt.datetime."""
        result = get_cyclic_timestamp(dt.time(6, 0, 0), ONE_DAY)
        assert isinstance(result, dt.datetime)

    def test_result_within_cycle(self):
        result = get_cyclic_timestamp(dt.time(18, 0, 0), ONE_DAY)
        assert_in_cycle(result, ONE_DAY)


# ===========================================================================
# dt.date inputs
# ===========================================================================


class TestDateInput:
    def test_first_day_of_year(self):
        """Jan 1 maps onto REFERENCE_TIME itself."""
        result = get_cyclic_timestamp(dt.date(2025, 1, 1), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=1, day=1)
        assert result == expected

    def test_mid_year_date(self):
        result = get_cyclic_timestamp(dt.date(2025, 6, 15), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=6, day=15)
        assert result == expected

    def test_end_of_year(self):
        result = get_cyclic_timestamp(dt.date(2025, 12, 31), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=12, day=31)
        assert result == expected

    def test_time_components_are_zero(self):
        """A plain dt.date input must not populate hour/minute/second."""
        result = get_cyclic_timestamp(dt.date(2025, 7, 4), ONE_YEAR)
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_result_within_cycle(self):
        result = get_cyclic_timestamp(dt.date(2025, 3, 20), ONE_YEAR)
        assert_in_cycle(result, ONE_YEAR)

    def test_leap_day_in_leap_reference_year(self):
        """Feb 29 is valid because REFERENCE_TIME.year (2020) is a leap year."""
        result = get_cyclic_timestamp(dt.date(2020, 2, 29), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=2, day=29)
        assert result == expected

    def test_leap_day_clamped_in_non_leap_reference_year(self):
        """When REFERENCE_TIME falls in a non-leap year, Feb 29 should
        be clamped to Feb 28. We temporarily monkey-patch REFERENCE_TIME
        to a non-leap year to exercise this branch.
        """
        from geoexposure import datetime_utils

        original = utils.REFERENCE_TIME
        try:
            datetime_utils.REFERENCE_TIME = original.replace(year=2019)
            result = get_cyclic_timestamp(dt.date(2020, 2, 29), ONE_YEAR)
            expected = datetime_utils.REFERENCE_TIME.replace(month=2, day=28)
            assert result == expected
        finally:
            datetime_utils.REFERENCE_TIME = original


# ===========================================================================
# dt.datetime inputs
# ===========================================================================


class TestDatetimeInput:
    def test_reference_time_itself(self):
        """Passing REFERENCE_TIME as input should return REFERENCE_TIME."""
        result = get_cyclic_timestamp(REFERENCE_TIME, ONE_YEAR)
        assert result == REFERENCE_TIME

    def test_mid_year_datetime(self):
        result = get_cyclic_timestamp(dt.datetime(2025, 6, 15, 14, 30, 0), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=6, day=15, hour=14, minute=30, second=0)
        assert result == expected

    def test_year_is_normalised_to_reference_year(self):
        """The year of the input is discarded; only month/day/time are kept."""
        result_2020 = get_cyclic_timestamp(dt.datetime(2020, 5, 10, 9, 0, 0), ONE_YEAR)
        result_2099 = get_cyclic_timestamp(dt.datetime(2099, 5, 10, 9, 0, 0), ONE_YEAR)
        assert result_2020 == result_2099

    def test_end_of_year_datetime(self):
        result = get_cyclic_timestamp(dt.datetime(2025, 12, 31, 23, 59, 59), ONE_YEAR)
        expected = REFERENCE_TIME.replace(month=12, day=31, hour=23, minute=59, second=59)
        assert result == expected

    def test_result_within_cycle(self):
        result = get_cyclic_timestamp(dt.datetime(2025, 8, 20, 10, 0, 0), ONE_YEAR)
        assert_in_cycle(result, ONE_YEAR)

    def test_result_is_datetime(self):
        """Return type should always be dt.datetime."""
        result = get_cyclic_timestamp(dt.datetime(2025, 4, 1, 12, 0, 0), ONE_YEAR)
        assert isinstance(result, dt.datetime)


# ===========================================================================
# cycle_duration validation
# ===========================================================================


class TestCycleDurationValidation:
    def test_zero_cycle_duration_raises(self):
        with pytest.raises(ValueError, match="cycle duration must be greater than 0"):
            get_cyclic_timestamp(dt.time(12, 0, 0), dt.timedelta(seconds=0))

    def test_negative_cycle_duration_raises(self):
        with pytest.raises(ValueError, match="cycle duration must be greater than 0"):
            get_cyclic_timestamp(dt.time(12, 0, 0), dt.timedelta(seconds=-1))


# ===========================================================================
# Invalid input types
# ===========================================================================


class TestInvalidInputType:
    def test_string_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown timestamp type"):
            get_cyclic_timestamp("2020-01-01", ONE_DAY)

    def test_integer_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown timestamp type"):
            get_cyclic_timestamp(42, ONE_DAY)

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown timestamp type"):
            get_cyclic_timestamp(None, ONE_DAY)

    def test_float_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown timestamp type"):
            get_cyclic_timestamp(3.14, ONE_DAY)
