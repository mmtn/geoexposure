"""This module contains classes for types of data required for the Exposure calculation."""

from .spatial import SpatialData
from .temporal import TemporalData, TemporalType
from .trajectory import Trajectory, GapMethod

__all__ = [
    "SpatialData",
    "TemporalData",
    "TemporalType",
    "Trajectory",
    "GapMethod",
]
