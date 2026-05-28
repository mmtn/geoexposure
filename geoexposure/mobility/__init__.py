"""Mobility models for computing spatial occupancy distributions from trajectories.

This submodule provides concrete :class:`~mobility.base.Mobility` implementations
that transform a :class:`~data.trajectory.Trajectory` into a normalised spatial
density distribution over an :class:`~core.environment.Environment` raster grid:

- :mod:`~mobility.kde` - kernel density estimation weighted by dwell time.
- :mod:`~mobility.point_overlay` - simple point counting within raster cells.
- :mod:`~mobility.adaptive_uncertainty` - time-integrated Gaussian density with
  uncertainty growing between observations.
"""

from .adaptive_uncertainty import AdaptiveUncertainty
from .base import Mobility
from .kde import KDE
from .point_overlay import PointOverlay

__all__ = [
    "KDE",
    "AdaptiveUncertainty",
    "Mobility",
    "PointOverlay",
]
