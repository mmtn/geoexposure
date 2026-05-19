"""Spatial metrics for computing environmental exposure on raster grids.

This submodule provides concrete :class:`~metrics.base.Metric` implementations
used to transform geospatial input data into per-cell exposure values:

- :mod:`~metrics.proximity` — distance from each raster cell to target geometries.
- :mod:`~metrics.land_type_exposure` — proximity-weighted exposure with Gaussian decay.
- :mod:`~metrics.fragmentation` — patch density of a land cover category.
- :mod:`~metrics.diversity` — Simpson Diversity Index of land cover within a neighbourhood.
"""

from .base import Metric
from .diversity import DiversityIndex
from .fragmentation import Fragmentation
from .land_type_exposure import LandTypeExposure
from .proximity import Proximity

__all__ = [
    "DiversityIndex",
    "Fragmentation",
    "LandTypeExposure",
    "Metric",
    "Proximity",
]
