"""TrajectoryExposure: estimating environmental exposure from GPS trajectory data.

Combines time-ordered mobility data (e.g. GPS trajectories) with geospatial
descriptions of the local environment to compute individual-level estimates of
exposure to different land cover types and environmental features. Exposure is
calculated by integrating a spatial occupancy distribution — derived from the
trajectory — against a raster representation of the environment over a series
of time windows.

The package is designed with infectious disease research in mind, where
exposure estimates can be linked with serological or diagnostic outcomes to
identify environmental and behavioural risk factors. However, the framework
is general and can be applied to any context where understanding the
relationship between human movement and environmental features is of interest,
such as air quality, green space access, or noise exposure.

Submodules
----------
- :mod:`~TrajectoryExposure.data` — trajectory, spatial, and temporal data representations.
- :mod:`~TrajectoryExposure.core` — environment modelling, shared utilities, and caching.
- :mod:`~TrajectoryExposure.mobility` — models for computing occupancy distributions.
- :mod:`~TrajectoryExposure.metrics` — spatial metrics evaluated on raster grids.
- :mod:`~TrajectoryExposure.exposure` — exposure calculation and result handling.
"""

from .core import utils
from .core.enums import GapMethod, SamplingMethod, TemporalType
from .core.environment import Environment
from .data import SpatialData, TemporalData, Trajectory
from .exposure import Exposure, ExposureSeries
from .metrics import DiversityIndex, Fragmentation, LandCover, Proximity
from .mobility import KDE, AdaptiveUncertainty, Mobility, PointOverlay

__all__ = [
    "KDE",
    "AdaptiveUncertainty",
    "DiversityIndex",
    "Environment",
    "Exposure",
    "ExposureSeries",
    "Fragmentation",
    "GapMethod",
    "LandCover",
    "Mobility",
    "PointOverlay",
    "Proximity",
    "SamplingMethod",
    "SpatialData",
    "TemporalData",
    "TemporalType",
    "Trajectory",
    "utils",
]
