"""Shared fixtures for geoexposure tests."""

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from geoexposure import Environment
from geoexposure.data.columns import DATETIME, X, Y
from geoexposure.data.spatial import SpatialData
from geoexposure.data.trajectory import Trajectory
from geoexposure.metrics.land_cover import LandCover
from geoexposure.mobility.kde import KDE

# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_trajectory() -> Trajectory:
    """A short trajectory moving diagonally across a 1km grid."""
    n = 50
    start = dt.datetime(2020, 1, 1, 8, 0, 0)
    times = [start + dt.timedelta(minutes=5 * i) for i in range(n)]
    x = np.linspace(100.0, 900.0, n)
    y = np.linspace(100.0, 900.0, n)
    df = pd.DataFrame({DATETIME: times, X: x, Y: y})
    return Trajectory(df, source_id="test_trajectory")


# ---------------------------------------------------------------------------
# Land cover GeoDataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def land_cover_gdf() -> gpd.GeoDataFrame:
    """A simple 2x2 land cover grid covering a 1km x 1km area.

    Top half is forest, bottom half is fields.
    """
    forest = box(0, 500, 1000, 1000)
    fields = box(0, 0, 1000, 500)
    return gpd.GeoDataFrame(
        {"land_type": ["forest", "fields"]},
        geometry=[forest, fields],
        crs="EPSG:32648",
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@pytest.fixture
def spatial_data(land_cover_gdf) -> dict:
    """A minimal SpatialData dict built from the synthetic land cover."""
    sd = SpatialData(
        gdf=land_cover_gdf,
        metrics={
            LandCover(radius=100, column="land_type", value="forest"): 1.0,
        },
    )
    return {"land": sd}


@pytest.fixture
def environment(spatial_data, tmp_path) -> Environment:
    """A minimal Environment built from the synthetic land cover."""
    env = Environment(
        spatial_resolution=100,
        spatial_data=spatial_data,
        spatial_reference_data="land",
    )
    env.calculate()
    return env


# ---------------------------------------------------------------------------
# Mobility
# ---------------------------------------------------------------------------

@pytest.fixture
def kde_model() -> KDE:
    """A KDE mobility model with a small bandwidth."""
    return KDE(kernel="gaussian", bandwidth=50)
