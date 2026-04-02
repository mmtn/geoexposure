from enum import Enum, StrEnum


class TemporalType(Enum):
    CYCLIC = 0
    DATED = 1


class GapMethod(StrEnum):
    IGNORE = "ignore"
    INTERPOLATE = "interpolate"
    RECENT = "recent"
    VORONOI = "voronoi"


class SamplingMethod(StrEnum):
    CEIL = "ceil"
    FLOOR = "floor"
    INTERP = "interp"
    NEAREST = "nearest"
