"""Data representations for exposure calculations.

This submodule provides classes for loading, storing, and processing the three
principal data types consumed by the exposure calculator:

- :mod:`~data.spatial` — geospatial data and associated metrics,
  represented by :class:`~data.spatial.SpatialData`.
- :mod:`~data.temporal` — time-indexed series of spatial or scalar values,
  represented by :class:`~data.temporal.TemporalData`.
- :mod:`~data.trajectory` — time-ordered point observations of individual movement,
  represented by :class:`~data.trajectory.Trajectory`.
- :mod:`~data.gap_methods` — strategies for filling or handling gaps in
  trajectory data, implementing the :class:`~data.gap_methods.GapFiller`
  interface.
- :mod:`~data.resampling` — strategies for resampling trajectory observations
  at arbitrary times, implementing the :class:`~data.resampling.Resampler`
  interface.
"""

from .spatial import SpatialData
from .temporal import TemporalData
from .trajectory import Trajectory

__all__ = [
    "SpatialData",
    "TemporalData",
    "Trajectory",
]
