import pandas as pd

from src.metrics.Metric import Metric
from src.utils import metric_name


class Proximity(Metric):
    def __init__(self, column=None, value=None):
        super().__init__()
        self.column = column
        self.value = value
        self.name = metric_name("proximity", (self.column, self.value))

    def calculate(self, gdf_spatial, gdf_raster):
        # TODO: turn proximity into something that can be summed with other exposures
        def get_geometry(gdf):
            if isinstance(gdf, pd.DataFrame):
                return gdf["geometry"]
            elif isinstance(gdf, pd.Series):
                return gdf
            else:
                raise ValueError(
                    "Inputs must be Series, or DataFrame with 'geometry' column"
                )

        if self.column is not None and self.value is not None:
            to_gdf = gdf_spatial[gdf_spatial[self.column] == self.value]
        else:
            to_gdf = gdf_spatial

        to_geom = get_geometry(to_gdf)
        from_geom = get_geometry(gdf_raster)

        spatial_index = to_geom.sindex  # R-tree spatial index for fast lookup
        min_distances = []

        for point in from_geom:
            nearest = list(spatial_index.nearest(point, return_all=True))
            distance = min(point.distance(to_geom.iloc[idx[0]]) for idx in nearest)
            min_distances.append(distance)

        return pd.Series(min_distances)
