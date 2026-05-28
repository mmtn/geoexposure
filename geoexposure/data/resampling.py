"""Resampling strategies for Trajectory objects."""

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..core.enums import SamplingMethod

if TYPE_CHECKING:
    from .trajectory import Trajectory

import pandas as pd

from .columns import DATETIME, REQUIRED_COLS_LIST, X, Y


class Resampler(ABC):
    """Abstract base class for trajectory resampling strategies."""

    @abstractmethod
    def resample(
            self,
            trajectory: "Trajectory",
            times: list[dt.datetime],
    ) -> pd.DataFrame:
        """Return a DataFrame of resampled observations at the requested times.

        Args:
            trajectory: The source trajectory to resample.
            times: Target datetimes at which to produce observations.

        Returns:
            DataFrame with columns matching the trajectory data.
        """


class NearestResampler(Resampler):
    """Selects the closest recorded point for each requested time."""

    def resample(
            self,
            trajectory: "Trajectory",
            times: list[dt.datetime],
    ) -> pd.DataFrame:
        """Select the closest recorded point for each requested time."""
        indices = [(trajectory.df[DATETIME] - t).abs().argmin() for t in times]
        resampled = trajectory.df.loc[indices, REQUIRED_COLS_LIST].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)


class PreviousResampler(Resampler):
    """Selects the most recent recorded point at or before each requested time."""

    def resample(
            self,
            trajectory: "Trajectory",
            times: list[dt.datetime],
    ) -> pd.DataFrame:
        """Select the most recent recorded point before or at each requested time."""
        indices = [trajectory.df[trajectory.df[DATETIME] <= t][DATETIME].argmax() for t in times]
        resampled = trajectory.df.loc[indices, REQUIRED_COLS_LIST].copy()
        resampled[DATETIME] = times.values
        return resampled.reset_index(drop=True)


class InterpResampler(Resampler):
    """Linearly interpolates x/y between adjacent timestamps."""

    def resample(
            self,
            trajectory: "Trajectory",
            times: list[dt.datetime],
    ) -> pd.DataFrame:
        """Linearly interpolate x/y between adjacent timestamps."""
        rows = []
        for t in times:
            before = trajectory.df[trajectory.df[DATETIME] <= t].iloc[-1]
            after = trajectory.df[trajectory.df[DATETIME] >= t].iloc[0]

            if before[DATETIME] == after[DATETIME]:
                x = before[X]
                y = before[Y]
            else:
                weight = (t - before[DATETIME]).total_seconds() / (
                        after[DATETIME] - before[DATETIME]
                ).total_seconds()
                x = before[X] + weight * (after[X] - before[X])
                y = before[Y] + weight * (after[Y] - before[Y])

            rows.append({DATETIME: t, X: x, Y: y})

        return pd.DataFrame(rows)


_RESAMPLER_MAPPING = {
    SamplingMethod.NEAREST    : NearestResampler(),
    SamplingMethod.INTERP     : InterpResampler(),
    SamplingMethod.MOST_RECENT: PreviousResampler(),
}


def get_resampler(method: SamplingMethod) -> Resampler:
    """Return the Resampler instance corresponding to the given SamplingMethod.

    Args:
        method: The selected sampling method.

    Returns:
        A Resampler instance ready to call.

    Raises:
        KeyError: If ``method`` is not registered.
    """
    return _RESAMPLER_MAPPING[method]
