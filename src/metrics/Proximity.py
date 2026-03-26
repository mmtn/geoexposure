from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Point, Polygon

from .Metric import Metric
from ..utils import get_gdf_centroids


class Proximity(Metric):
    metric_title = "proximity"

    def __init__(
            self,
            column: str | None = None,
            value: int | float | None = None,
            ):
        super().__init__()
        self.column = column
        self.value = value
        self.name = self.get_name(self.column, self.value)

    def _calculate_metric(self, gdf_input, gdf_raster):
        if self.column is not None and self.value is not None:
            assert (
                    self.value in gdf_input[self.column].values
            ), f"value '{self.value}' not found in '{self.column}' column"
            gdf_to = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_to = gdf_input

        gdf_from = gdf_raster
        proximity = calculate_gdf_proximity(gdf_from, gdf_to)

        self.data = pd.Series(proximity)
        self._calculated = True


class ProximityRisk(Metric):
    metric_title = "proximity_risk"

    def __init__(
            self,
            column: str | None = None,
            value: Any | None = None,
            threshold: float | None = None,
    ):
        super().__init__()
        self.column = column
        self.value = value
        self.threshold = threshold
        self.name = self.get_name(self.column, self.value, self.threshold)

    def _calculate_metric(
            self, gdf_input: gpd.GeoDataFrame, gdf_raster: gpd.GeoDataFrame
    ):
        # TODO: remove duplication of code
        if self.column is not None and self.value is not None:
            assert (
                    self.value in gdf_input[self.column].values
            ), f"value '{self.value}' not found in '{self.column}' column"
            gdf_to = gdf_input[gdf_input[self.column] == self.value]
        else:
            gdf_to = gdf_input

        gdf_from = gdf_raster
        proximity = calculate_gdf_proximity(gdf_from, gdf_to)
        risk = proximity_to_risk(proximity, self.threshold)

        self.data = pd.Series(risk)
        self._calculated = True


# Helper functions
def calculate_gdf_proximity(
        gdf_from: gpd.GeoDataFrame, gdf_to: gpd.GeoDataFrame
) -> list:
    if not all(isinstance(x, gpd.GeoDataFrame) for x in (gdf_from, gdf_to)):
        raise ValueError("Inputs must be GeoDataFrames with 'geometry' column")

    if all(isinstance(x, Polygon) for x in gdf_from.geometry):
        gdf_from["geometry"], _ = get_gdf_centroids(gdf_from)

    if not all(isinstance(x, Point) for x in gdf_from.geometry):
        raise ValueError("'from' geometries must be Points")

    # Spatial join to find the nearest geometry in gdf_to for each geometry in gdf_from
    joined = gpd.sjoin_nearest(
        gdf_from, gdf_to[["geometry"]], how="left", distance_col="distance"
    )

    return joined["distance"].to_list()


def proximity_to_risk(
        distances: pd.Series, threshold: float, shape: float = 4.0
) -> pd.Series:
    d = np.asarray(distances, dtype=float)

    if threshold == 0 or threshold is None:
        risk = np.zeros_like(d, dtype=float)
        risk[d == 0] = 1.0
        return risk

    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    if shape <= 0:
        raise ValueError("shape must be > 0")

    x = np.clip(d / threshold, 0.0, 1.0)  # Normalise distance to [0, 1]

    # Normalised exponential decay: 1 at x=0, 0 at x=1
    exp_neg_shape = np.exp(-shape)
    risk = (np.exp(-shape * x) - exp_neg_shape) / (1.0 - exp_neg_shape)

    # Explicitly zero out anything beyond the threshold
    risk[d > threshold] = 0.0
    return risk
