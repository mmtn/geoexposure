import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from TrajectoryExposure.core.utils import round_datetime
from TrajectoryExposure.data.temporal import TimeDeltaLike


class ExposureSeries:
    META_COLUMNS = (
        "window_start",
        "window_centre",
        "window_end",
        "window_length_seconds",
        "scaling",
    )

    def __init__(
            self, data: pd.DataFrame, metadata: pd.DataFrame, source_id: str | None = None,
    ):
        if len(data) != len(metadata):
            raise ValueError("data and metadata must have the same number of rows")
        self.data = data.reset_index(drop=True).copy()
        self.metadata = metadata.reset_index(drop=True).copy()
        self.source_id = source_id

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, source_id: str | None = None) -> "ExposureSeries":
        meta_cols = [col for col in cls.META_COLUMNS if col in df.columns]
        exposure_cols = [col for col in df.columns if col not in meta_cols]
        metadata = df[meta_cols].copy()
        data = df[exposure_cols].copy()
        return cls(data, metadata, source_id)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.concat([self.metadata, self.data], axis=1)

    @property
    def _scaling(self) -> "ExposureSeries":
        if "scaling" not in self.metadata.columns:
            raise ValueError("metadata does not contain 'scaling'")
        return self.metadata["scaling"]

    @property
    def _window_lengths(self) -> pd.Series:
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
        """As rate() but rescaled to an arbitrary time unit - [exposure units] per [denominator].
        """
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
        """Rescales data based on window durations then returns the mean exposure in each
        category per temporal_resolution in the original Exposure evaluation
        """
        window_lengths = self.metadata["window_length_seconds"]
        normalised_windows = window_lengths / window_lengths.sum()
        normalised_exposure = self.data.div(normalised_windows, axis=0)
        return normalised_exposure.mean()

    def aggregate(self, timestep: dt.timedelta) -> "ExposureSeries":
        """Aggregate windows into a coarser time grid by summing raw exposure
        integrals within each new window.

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

    def plot_over_time(
            self,
            ax=None,
            *,
            cumulative: bool = False,
            x_col: str = "window_centre",
            xlim: tuple[Any, Any] | None = None,
            ylim: tuple[float, float] | None = None,
            title: str | None = None,
            apply_scaling: bool = True,
            auto_ylim_pad: float | None = None,
            line_kwargs: dict | None = None,
            legend: bool = True,
            legend_kwargs: dict | None = None,
            xtick_rotation: int = 90,
            show: bool = True,
            with_sum: bool = False,
    ):
        import matplotlib.pyplot as plt

        exposure_df = self.scaled() if apply_scaling else self.data.copy()
        df = pd.concat([self.metadata, exposure_df], axis=1)

        if x_col not in self.metadata.columns:
            raise ValueError(f"metadata must contain x_col={x_col!r}")

        for col in ["window_start", "window_end"]:
            if col not in self.metadata.columns:
                raise ValueError(
                    f"metadata must contain '{col}' for window plot. "
                    "Use plot_over_time with x_col='window_centre' for point plots.",
                )

        if line_kwargs is None:
            line_kwargs = {}

        if auto_ylim_pad is None:
            auto_ylim_pad = 2.0

        if legend_kwargs is None:
            legend_kwargs = {
                "loc"           : "upper left",
                "fontsize"      : "small",
                "bbox_to_anchor": (0.1, 0.9),
                "borderaxespad" : 0,
            }

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        plot_df = df.sort_values(by=x_col)
        window_starts = plot_df["window_start"]
        window_ends = plot_df["window_end"]

        for col in exposure_df.columns:
            y = plot_df[col]
            if cumulative:
                y = np.cumsum(y)

            (ref_line,) = ax.plot([], [], label=col, **line_kwargs)
            colour = ref_line.get_color()

            ax.hlines(
                y=y,
                xmin=plot_df["window_start"],
                xmax=plot_df["window_end"],
                colors=colour,
                **line_kwargs,
            )

            # Vertical connectors between consecutive segments
            ax.vlines(
                x=plot_df["window_end"].iloc[:-1],
                ymin=y.iloc[1:].values,
                ymax=y.iloc[:-1].values,
                colors=colour,
                **line_kwargs,
            )

        if with_sum:
            plot_df["sum"] = plot_df[exposure_df.columns].sum(axis=1)
            y = plot_df["sum"]
            ax.hlines(
                y=y,
                xmin=plot_df["window_start"],
                xmax=plot_df["window_end"],
                colors="black",
            )
            # Vertical connectors between consecutive segments
            ax.vlines(
                x=plot_df["window_end"].iloc[:-1],
                ymin=y.iloc[1:].values,
                ymax=y.iloc[:-1].values,
                colors="black",
            )

        if title is not None:
            ax.set_title(title)

        if ylim is not None:
            ax.set_ylim(ylim)
        elif auto_ylim_pad is not None:
            try:
                y_max = float(np.nanmax(exposure_df.to_numpy(dtype=float)))
                if np.isfinite(y_max):
                    ax.set_ylim((0.0, y_max * auto_ylim_pad))
            except Exception:
                pass

        if xlim is not None:
            ax.set_xlim(xlim)

        if legend:
            fig.legend(**legend_kwargs)

        if xtick_rotation is not None:
            for tick in ax.get_xticklabels():
                tick.set_rotation(xtick_rotation)

        fig.tight_layout()
        if show:
            plt.show()

        return fig, ax
