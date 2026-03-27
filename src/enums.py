from enum import Enum, StrEnum


class TemporalType(Enum):
    CYCLIC = 0
    DATED = 1


class GapMethod(StrEnum):
    VORONOI = "voronoi"
    INTERPOLATE = "interpolate"
    RECENT = "recent"
    IGNORE = "ignore"


class InterpMethod(StrEnum):
    NEAREST = "nearest"
    INTERP = "interp"
    FLOOR = "floor"
    CEIL = "ceil"
