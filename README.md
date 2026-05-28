# geoexposure library

A Python library for computing environmental exposure from GPS trajectory data.

Exposure is estimated by combining a **mobility model** (how time is distributed across space) 
with an **environment** (spatially or temporally varying exposure values on a raster grid). 
The library supports multiple mobility models, gap-filling strategies, and temporal scaling, and 
provides tools for running large batches of scenarios in parallel.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mmtn/geoexposure.git
cd geoexposure
pip install -e .
```

## Quick Start

```python
from geoexposure import Environment, Exposure, GapMethod, KDE, SpatialData, Trajectory

# Load a trajectory
trajectory = Trajectory.from_csv("data/gps_0285.csv").with_dwell_times(GapMethod.VORONOI)

# Build an environment
spatial_data = {"land": SpatialData.from_file("data/land_cover.shp", epsg=32754)}
environment = Environment(
    spatial_resolution=50,
    spatial_data=spatial_data,
    spatial_reference_data="land",
)

# Compute exposure
mobility = KDE(kernel="gaussian", bandwidth=20.0)
exposure = Exposure(mobility=mobility, environment=environment)
result = exposure.for_trajectory(trajectory)

print(result.total())
print(result.mean_rate())
```

See `examples/` for further usage including batch processing and visualisation.

## Key Concepts

**Trajectory** - a time-ordered sequence of GPS observations (x, y, datetime) for a single 
participant. Loaded from CSV and optionally gap-filled before exposure calculation.

**Gap method** - Trajectories can contain gaps where no observations were recorded. A `GapMethod` 
determines how dwell time is assigned around these gaps, ranging from ignoring them entirely to 
synthesising new observations by interpolation. The choice of gap method can materially affect 
exposure estimates.

**Environment** - a raster grid of spatial exposure values, built from one or more `SpatialData` 
layers with associated `Metric` objects. Supports optional temporal scaling via `TemporalData`.

**Mobility model** - estimates how a participant distributes their time across the raster grid 
given their trajectory. Available models: `KDE`, `AdaptiveUncertainty`, `PointOverlay`.

**Exposure** - combines a mobility model with an environment to compute time-weighted exposure for 
a trajectory. Returns an `ExposureSeries` with methods for aggregation, rate calculation, and 
temporal scaling.

## Caching

Expensive computations (raster construction, metric calculation, KDE fitting) are cached to disk 
automatically. Set the cache directory via environment variable before importing the library:

```python
import os
os.environ["TRAJECTORY_EXPOSURE_CACHE_DIR"] = "/path/to/project/.cache"
```

If not set, the cache defaults to `.cache/` relative to the working directory, which may result in 
multiple cache locations if scripts are run from different directories.

## Requirements

- Python 3.12+
- geopandas
- numpy
- pandas
- scikit-learn
- shapely
- attrs
- matplotlib
- cmocean
