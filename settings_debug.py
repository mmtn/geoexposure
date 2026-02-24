import datetime as dt

from src import Proximity, SpatialData, TemporalData
from src.Enums import TemporalType


class Settings:
    # Generated data
    mobility_data_path = "tests/data/trajectories"

    # Generated geometries
    side_by_side = SpatialData(
        "tests/data/gis/side_by_side/side_by_side.shp",
        # metrics=[Proximity("land_use", "forest")]
    )
    forest = SpatialData(
        "tests/data/gis/forest/forest.shp",
        metrics=[Proximity("land_use", "forest")]
    )
    fields = SpatialData(
        "tests/data/gis/fields/fields.shp",
        metrics=[Proximity("land_use", "fields")]
    )
    plantations = SpatialData(
        "tests/data/gis/plantations/plantations.shp",
        metrics=[Proximity("land_use", "plantations")]
    )
    built_up = SpatialData(
        "tests/data/gis/built-up/built-up.shp",
        metrics=[Proximity("land_use", "built-up")]
    )

    spatial_data_ref = side_by_side

    spatial_data = {
        "side_by_side": side_by_side,
        # "forest": forest,
    }

    temporal_data = {
        # "numbers": TemporalData(
        #     time_data_dict={
        #         dt.time(hour=0, minute=0): 0.0,
        #         dt.time(hour=0, minute=30): 1.0,
        #         dt.time(hour=1, minute=0): 2.0,
        #     },
        #     cycle_duration=dt.timedelta(hours=2, minutes=00),
        #     temporal_type=TemporalType.CYCLIC,
        # ),
        "high_frequency": TemporalData(
            time_data_dict={
                dt.time(hour=0, minute=0): forest,
                dt.time(hour=0, minute=30): fields,
                dt.time(hour=1, minute=0): built_up,
                # dt.time(hour=1, minute=30): built_up,
            },
            cycle_duration=dt.timedelta(hours=2, minutes=00),
            temporal_resolution=dt.timedelta(minutes=30),
            temporal_type=TemporalType.CYCLIC,
        )
    }
