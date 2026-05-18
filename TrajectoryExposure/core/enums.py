from enum import StrEnum, auto


class SamplingMethod(StrEnum):
    CEIL = auto()
    FLOOR = auto()
    INTERP = auto()
    MOST_RECENT = auto()
    NEAREST = auto()
