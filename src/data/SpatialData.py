from geopandas import GeoDataFrame


class SpatialData:
    def __init__(self, file, crs=None, metrics=None):
        self.gdf = None
        self.gdf_metrics = None
        self._calculated = False
        self._set_data(file, crs)
        self.metrics = metrics if metrics is not None else list()

    def _set_data(self, file, crs):
        # Check string first because strings are also Iterables
        if not isinstance(file, str):
            raise TypeError("'data' must be 'path/to/shape/file'")
        self.gdf = GeoDataFrame.from_file(file)
        if crs is not None:
            self.gdf = self.gdf.to_crs(epsg=crs)

    def calculate(self, gdf_raster):
        self.gdf_metrics = gdf_raster.copy()
        for metric in self.metrics:
            self.gdf_metrics[metric.name] = metric.calculate(self.gdf, gdf_raster)
        self._calculated = True
