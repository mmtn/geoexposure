"""Enumeration types shared across the codebase.

- :class:`GapMethod` is used to select strategy for filling gaps in trajectories.
- :class:`SamplingMethod` is used to select a timestamp-matching strategy.
- :class:`TemporalType` is used to define how timestamps for temporal data are given.
"""

from enum import Enum, StrEnum, auto


class GapMethod(StrEnum):
    """Enumeration of strategies for handling gaps in trajectory data."""

    IGNORE = auto()
    """Exclude gap periods from dwell time calculations.

    Points whose nearest neighbour is more than one window length away have
    their dwell time bounded by the window boundary rather than extending
    across the gap.
    """

    INTERPOLATE = auto()
    """Fill gaps by linear interpolation between known positions.

    Synthetic observations are inserted at regular intervals across gaps
    larger than the specified resolution. Original observations are preserved
    unchanged.
    """

    RECENT = auto()
    """Fill gaps by carrying the last known position forward in time.

    Synthetic observations are inserted at regular intervals across gaps
    larger than the specified resolution, each inheriting the coordinates
    of the most recent recorded point.
    """

    VORONOI = auto()
    """Assign dwell times using a 1D Voronoi partition of the time axis.

    The dwell time at point ``i`` is half the time elapsed since point
    ``i - 1`` plus half the time until point ``i + 1``. No gap filling
    is performed and no resolution argument is required.
    """


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


class TemporalType(Enum):
    """Enumeration of the type of temporally varying data."""

    CYCLIC = 0
    """Data repeats over a fixed cycle duration (e.g. time of day, season).

    Keys in ``time_data_dict`` are interpreted as offsets within the cycle.
    Requires ``cycle_duration`` to be set on the parent ``TemporalData`` instance.
    """

    DATED = 1
    """Data is indexed by absolute datetimes with no repeating structure.

    Keys in ``time_data_dict`` must be ``dt.datetime`` instances.
    """
