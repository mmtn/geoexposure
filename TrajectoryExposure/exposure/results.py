"""Time-windowed exposure results with aggregation and plotting utilities.

:class:`ExposureSeries` stores the output of an exposure model evaluation as
a pair of aligned DataFrames — one holding raw exposure integrals per window
per metric, and one holding window metadata such as start time, end time, and
temporal scaling. Methods are provided to compute rates, apply scaling,
aggregate to coarser time grids, and plot exposure over time.
"""

import datetime as dt
import pickle
from pathlib import Path

import pandas as pd

from ..core.utils import round_datetime
from ..data.temporal import TimeDeltaLike


class ExposureSeries:
    """Time-windowed exposure results paired with window metadata.

    Stores raw exposure integrals computed over a sequence of time windows,
    alongside metadata describing each window. Provides methods to compute
    exposure rates, apply temporal scaling, aggregate to coarser time steps,
    and visualise results.

    Attributes:
        data: DataFrame of raw exposure integrals, one row per window and one
            column per exposure metric.
        metadata: DataFrame of window metadata aligned row-wise with ``data``.
            May contain any subset of :attr:`META_COLUMNS`.
        source_id: Optional identifier for the trajectory or participant that
            produced this series.
    """

    META_COLUMNS = (
        "window_start",
        "window_centre",
        "window_end",
        "window_length_seconds",
        "scaling",
    )

    def __init__(
            self, data: pd.DataFrame, metadata: pd.DataFrame, source_id: str | None = None,
    ) -> None:
        """Initialise an ExposureSeries from separate data and metadata DataFrames.

        Args:
            data: Raw exposure integrals, one row per window.
            metadata: Window metadata aligned row-wise with ``data``.
            source_id: Optional identifier for the source trajectory or participant.

        Raises:
            ValueError: If ``data`` and ``metadata`` have different numbers of rows.
        """
        if len(data) != len(metadata):
            raise ValueError("data and metadata must have the same number of rows")
        self.data = data.reset_index(drop=True).copy()
        self.metadata = metadata.reset_index(drop=True).copy()
        self.source_id = source_id

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source_id: str | None = None) -> "ExposureSeries":
        """Construct an ExposureSeries from a single combined DataFrame.

        Columns whose names appear in :attr:`META_COLUMNS` are treated as metadata;
        all remaining columns are treated as exposure data.

        Args:
            df: Combined DataFrame containing both metadata and exposure columns.
            source_id: Optional identifier for the source trajectory or participant.

        Returns:
            A new :class:`ExposureSeries` instance.
        """
        meta_cols = [col for col in cls.META_COLUMNS if col in df.columns]
        exposure_cols = [col for col in df.columns if col not in meta_cols]
        metadata = df[meta_cols].copy()
        data = df[exposure_cols].copy()
        return cls(data, metadata, source_id)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return metadata and exposure data combined into a single DataFrame."""
        return pd.concat([self.metadata, self.data], axis=1)

    @property
    def _scaling(self) -> "ExposureSeries":
        """Return the per-window temporal scaling factors from metadata.

        Raises:
            ValueError: If ``metadata`` does not contain a ``scaling`` column.
        """
        if "scaling" not in self.metadata.columns:
            raise ValueError("metadata does not contain 'scaling'")
        return self.metadata["scaling"]

    @property
    def _window_lengths(self) -> pd.Series:
        """Return the duration of each window in seconds from metadata.

        Raises:
            ValueError: If ``metadata`` does not contain a ``window_length_seconds`` column.
        """
        if "window_length_seconds" not in self.metadata.columns:
            raise ValueError("metadata does not contain 'window_length_seconds'")
        return self.metadata["window_length_seconds"]

    def total(self) -> pd.Series:
        """Sum of raw exposure integrals across all windows: [exposure units]."""
        series = self.data.sum()
        series.name = self.source_id
        return series

    def mean_rate(self) -> pd.Series:
        """Total exposure divided by total elapsed time - [exposure units] per second."""
        weights = self._window_lengths / self._window_lengths.sum()
        return self.rate().mul(weights, axis=0).sum()

    def mean_rate_per(self, denominator: TimeDeltaLike) -> pd.Series:
        """Total exposure divided by total elapsed time - [exposure units] per [denominator]."""
        return self.mean_rate() * denominator.total_seconds()

    def rate(self) -> pd.DataFrame:
        """Divides exposure by the window's duration in seconds - [exposure units] per second."""
        return self.data.div(self._window_lengths, axis=0)

    def rate_per(self, denominator: TimeDeltaLike) -> pd.DataFrame:
        """As rate() but rescaled to arbitrary time unit - [exposure units] per [denominator]."""
        return self.rate() * denominator.total_seconds()

    def per_second(self) -> pd.DataFrame:
        """Wrapper to quickly get exposure per second."""
        return self.rate_per(dt.timedelta(seconds=1))

    def per_hour(self) -> pd.DataFrame:
        """Wrapper to quickly get exposure per hour."""
        return self.rate_per(dt.timedelta(hours=1))

    def per_day(self) -> pd.DataFrame:
        """Wrapper to quickly get exposure per day."""
        return self.rate_per(dt.timedelta(days=1))

    def scaled(self) -> pd.DataFrame:
        """Applies temporally varying scaling to result."""
        return self.data.mul(self._scaling, axis=0)

    def mean(self) -> pd.Series:
        """Return the duration-normalised mean exposure per metric.

        Normalises each window by its fractional share of the total elapsed time,
        then returns the mean across windows. This gives the expected exposure per
        unit of total duration rather than a simple row-wise mean.

        Returns:
            Series of normalised mean exposure values, one per metric column.
        """
        window_lengths = self.metadata["window_length_seconds"]
        normalised_windows = window_lengths / window_lengths.sum()
        normalised_exposure = self.data.div(normalised_windows, axis=0)
        return normalised_exposure.mean()

    def aggregate(self, timestep: dt.timedelta) -> "ExposureSeries":
        """Aggregate windows into a coarser time grid by summing raw exposure integrals.

        Raw integrals are summed rather than averaged so that the total
        accumulated exposure is preserved.

        Args:
            timestep: The target window size.
        """
        if not {"window_start", "window_end", "window_length_seconds"}.issubset(
                self.metadata.columns,
        ):
            raise ValueError(
                "metadata must contain 'window_start', 'window_end', and "
                "'window_length_seconds' to resample",
            )

        df = self.dataframe.copy()
        df["_window_bin"] = df["window_start"].apply(
            lambda t: round_datetime(t, timestep, to="floor"),
        )

        exposure_cols = self.data.columns.tolist()

        aggregated = (
            df.groupby("_window_bin")
            .agg(
                **{col: (col, "sum") for col in exposure_cols},
                window_start=("window_start", "min"),
                window_end=("window_end", "max"),
                window_length_seconds=("window_length_seconds", "sum"),
                scaling=("scaling", "mean"),
            )
            .reset_index(drop=True)
        )

        starts = aggregated["window_start"]
        ends = aggregated["window_end"]
        aggregated["window_centre"] = starts + ((ends - starts) / 2)

        return ExposureSeries.from_dataframe(aggregated, source_id=self.source_id)

    def save(self, path: Path) -> None:
        """Serialise this ExposureSeries to disk.

        Args:
            path: Destination file path. Parent directory must exist.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ExposureSeries":
        """Deserialise an ExposureSeries from disk.

        Args:
            path: Path to a previously saved ExposureSeries file.

        Returns:
            The deserialised ExposureSeries instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
