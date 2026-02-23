from enum import Enum, StrEnum


class SpatialMetric(StrEnum):
    PROXIMITY = "proximity"
    FRAGMENTATION = "fragmentation"


class TemporalType(Enum):
    CYCLIC = 0
    DATED = 1
