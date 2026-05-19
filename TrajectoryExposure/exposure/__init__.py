"""Exposure estimation combining mobility and environment models.

This submodule provides the top-level :class:`~exposure.calculator.Exposure`
class, which integrates a :class:`~mobility.base.Mobility` model with an
:class:`~core.environment.Environment` to compute time-windowed exposure
estimates for one or more trajectories. Results are returned as
:class:`~exposure.results.ExposureSeries` objects.
"""

from .calculator import Exposure
from .results import ExposureSeries

__all__ = [
    "Exposure",
    "ExposureSeries",
]
