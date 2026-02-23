from geopandas import GeoDataFrame


class SpatialData:
    def __init__(self, file, crs, metrics=None):
        # Set from arguments
        self._set_data(file, crs)
        self.metrics = metrics
        self.metric_gdf = None

        # Initial default settings
        self._calculated = False

    def _set_data(self, file, crs):
        # Check string first because strings are also Iterables
        if isinstance(file, str):
            self.gdf = GeoDataFrame.from_file(file).to_crs(epsg=crs)
        else:
            raise TypeError("'data' must be 'path/to/shape/file'")

    def calculate(self, gdf_raster):
        if self.metric_gdf is None:
            self.metric_gdf = gdf_raster.copy()

        for metric, weight in self.metrics.items():
            self.metric_gdf[metric.name] = metric.calculate(self.gdf, gdf_raster)

        self._calculated = True
