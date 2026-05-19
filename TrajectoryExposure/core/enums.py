"""Enumeration types shared across the codebase.

- :class:`SamplingMethod` is used to select a timestamp-matching strategy
"""

from enum import StrEnum, auto


class SamplingMethod(StrEnum):
    """Strategy used to select or compute a value at a requested timestamp.

    Members are lowercase strings (via ``auto()``), so they can be compared
    directly with string arguments accepted by public-facing methods.
    """

    CEIL = auto()
    """Select the earliest available value at or after the requested time."""

    FLOOR = auto()
    """Select the latest available value at or before the requested time."""

    INTERP = auto()
    """Linearly interpolate between the two nearest available values."""

    MOST_RECENT = auto()
    """Carry the last known value forward."""

    NEAREST = auto()
    """Select the value closest to the requested time, select the earliest time if equidistant."""
