import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


def start_logging(level: int | str = logging.INFO, log_str: str | None = None) -> "Logger":
    """Initialise logger with default settings."""
    if log_str is None:
        log_str = "%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s | %(message)s"

    logger = logging.getLogger("TrajectoryExposure")
    formatter = logging.Formatter(log_str)
    handler = logging.StreamHandler()
    logger.setLevel(level)
    logger.propagate = False
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
