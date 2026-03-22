class Mobility:
    # TODO: subclass for new method
    # TODO: precompute and store spatial distribution(s)
    # TODO: are there issues with few data points at high temporal resolutions?
    # TODO: normalisation options
    # TODO: decide on possible methods for Mobility model
    def __init__(self):
        pass

    def distribution(self, trajectory, gdf_geometry, weights):
        # Intended to be overridden by subclasses.
        raise NotImplementedError("subclasses must implement distribution()")

