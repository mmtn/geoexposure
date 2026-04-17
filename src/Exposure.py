import datetime as dt
import logging
from typing import Any

from .utils import round_datetime

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import Environment, Mobility, utils
from .data import Trajectory
from .enums import GapMethod, SamplingMethod


class ExposureSeries:
    META_COLUMNS = (
        "window_start",
        "window_centre",
        "window_end",
        "window_length_seconds",
        "scaling",
    )

    def __init__(
        self,
        exposure_data: pd.DataFrame,
        metadata: pd.DataFrame,
        source_id: str | None = None
    ):
        if len(exposure_data) != len(metadata):
            raise ValueError(
                "exposure_data and metadata must have the same number of rows"
            )
        self.exposure_data = exposure_data.reset_index(drop=True).copy()
        self.metadata = metadata.reset_index(drop=True).copy()
        self.source_id = source_id

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source_id: str | None = None
    ) -> "ExposureSeries":
        meta_cols = [col for col in cls.META_COLUMNS if col in df.columns]
        metadata = df[meta_cols].copy()
        exposure_cols = [col for col in df.columns if col not in meta_cols]
        exposure_data = df[exposure_cols].copy()
        return cls(exposure_data, metadata, source_id)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.concat([self.metadata, self.exposure_data], axis=1)

    def aggregate(self, timestep: dt.timedelta) -> "ExposureSeries":
        """
        Aggregate windows into a coarser time grid by summing raw exposure
        integrals within each new window.

        Raw integrals are summed rather than averaged so that the total
        accumulated exposure is preserved.

        Args:
            timestep: The target window size.
        """
        if not {"window_start", "window_end", "window_length_seconds"}.issubset(
                self.metadata.columns
        ):
            raise ValueError(
                "metadata must contain 'window_start', 'window_end', and "
                "'window_length_seconds' to resample"
            )

        df = self.dataframe.copy()
        df["_window_bin"] = df["window_start"].apply(
            lambda t: round_datetime(t, timestep, to="floor")
        )

        exposure_cols = self.exposure_data.columns.tolist()

        aggregated = df.groupby("_window_bin").agg(
            **{col: (col, "sum") for col in exposure_cols},
            window_start=("window_start", "min"),
            window_end=("window_end", "max"),
            window_length_seconds=("window_length_seconds", "sum"),
            scaling=("scaling", "mean"),
        ).reset_index(drop=True)

        starts = aggregated["window_start"]
        ends = aggregated["window_end"]
        aggregated["window_centre"] = starts + ((ends - starts) / 2)

        return ExposureSeries.from_dataframe(aggregated, source_id=self.source_id)


    def total(self) -> pd.Series:
        """
        Sum of raw exposure integrals across all windows.
        Units: [exposure units]
        """
        return self.exposure_data.sum()

    def mean_rate(self) -> pd.Series:
        """
        Total accumulated exposure divided by total elapsed time.
        Units: [exposure units] per second.
        """
        weights = self._window_lengths / self._window_lengths.sum()
        return self.rate().mul(weights, axis=0).sum()

    def mean_rate_per(self, denominator: dt.timedelta) -> pd.Series:
        """
        As mean_rate() but rescaled to an arbitrary time unit.
        Units: [exposure units] per [denominator].
        """
        return self.mean_rate() * denominator.total_seconds()

    def rate(self) -> pd.DataFrame:
        """
        Divides each exposure by the window's duration in seconds.
        Units: [exposure units] per second.
        """
        return self.exposure_data.div(self._window_lengths, axis=0)

    def rate_per(self, denominator: dt.timedelta) -> pd.DataFrame:
        """
        As rate() but rescaled to an arbitrary time unit.
        Units: [exposure units] per [denominator].
        """
        return self.rate() * denominator.total_seconds()

    def per_second(self) -> pd.DataFrame:
        return self.rate_per(dt.timedelta(seconds=1))

    def per_hour(self) -> pd.DataFrame:
        return self.rate_per(dt.timedelta(hours=1))

    def per_day(self) -> pd.DataFrame:
        return self.rate_per(dt.timedelta(days=1))

    def scaled(self) -> pd.DataFrame:
        return self.exposure_data.mul(self._scaling, axis=0)

    def mean(self) -> pd.Series:
        """
        Rescales data based on window durations then returns the mean exposure in each
        category per temporal_resolution in the original Exposure evaluation
        """
        window_lengths = self.metadata["window_length_seconds"]
        normalised_windows = window_lengths / window_lengths.sum()
        normalised_exposure = self.exposure_data.div(normalised_windows, axis=0)
        return normalised_exposure.mean()

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
    ):
        import matplotlib.pyplot as plt

        exposure_df = self.scaled() if apply_scaling else self.exposure_data.copy()
        df = pd.concat([self.metadata, exposure_df], axis=1)

        if x_col not in self.metadata.columns:
            raise ValueError(f"metadata must contain x_col={x_col!r}")

        for col in ["window_start", "window_end"]:
            if col not in self.metadata.columns:
                raise ValueError(
                    f"metadata must contain '{col}' for window plot. "
                    "Use plot_over_time with x_col='window_centre' for point plots."
                )

        if line_kwargs is None:
            line_kwargs = {}

        if auto_ylim_pad is None:
            auto_ylim_pad = 2.0

        if legend_kwargs is None:
            legend_kwargs = {
                "loc": "upper left",
                "fontsize": "small",
                "bbox_to_anchor": (0.1, 0.9),
                "borderaxespad": 0,
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

class Exposure:
    def __init__(
        self,
        mobility: Mobility,
        environment: Environment,
        *,
        timestep: dt.timedelta | None = None,
        interp_method: SamplingMethod = SamplingMethod.INTERP,
    ):
        """

        Args:
            mobility:
            environment:
            timestep:
            interp_method: "interp", "nearest"
        """
        self.mobility = mobility
        self.environment = environment
        self.timestep = timestep
        self.interp_method = interp_method
        self.environment.calculate()

    def for_trajectory(
        self,
        trajectory: Trajectory,
        *,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        timestep: dt.timedelta | None = None,
    ) -> ExposureSeries:
        effective_resolution = self._resolve_temporal_resolution(timestep)
        summary_df = self._calculate_exposure_dataframe(
            trajectory=trajectory,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution=effective_resolution,
        )
        series = ExposureSeries.from_dataframe(summary_df, source_id=trajectory.source_id)
        if timestep is not None and timestep > effective_resolution:
            series = series.aggregate(timestep)
        return series

    def for_trajectories(
        self,
        trajectories: list[Trajectory],
        *,
        temporal_resolution: dt.timedelta | None = None,
    ) -> list[ExposureSeries]:
        return [
            self.for_trajectory(
                trajectory=trajectory,
                timestep=temporal_resolution,
            )
            for trajectory in trajectories
        ]

    def sums(
        self,
        inputs: list[Trajectory] | list[ExposureSeries],
        *,
        temporal_resolution: dt.timedelta | None = None,
        per_second: bool = True,
        apply_scaling: bool = False,
    ) -> pd.DataFrame:
        """Compute exposure sums from either trajectories or precomputed exposure series."""
        if not inputs:
            return pd.DataFrame()

        first = inputs[0]

        if isinstance(first, Trajectory):
            return self._sums_trajectory_list(
                inputs,
                temporal_resolution=temporal_resolution,
                per_second=per_second,
                apply_scaling=apply_scaling,
            )
        elif isinstance(first, ExposureSeries):
            if temporal_resolution is not None:
                raise ValueError(
                    "`temporal_resolution` is not applicable when passing "
                    "precomputed ExposureSeries objects."
                )
            return self._sums_exposure_series_list(
                inputs,
                per_second=per_second,
                apply_scaling=apply_scaling,
            )
        else:
            raise TypeError(
                f"Unsupported input type '{type(first).__name__}'. "
                "Expected a list of Trajectory or ExposureSeries objects."
            )

    def _sums_trajectory_list(
            self,
            trajectories: list[Trajectory],
            *,
            temporal_resolution: dt.timedelta | None = None,
            per_second: bool = True,
            apply_scaling: bool = False,
    ) -> pd.DataFrame:
        exposure_series_list = self.for_trajectories(
            trajectories,
            temporal_resolution=temporal_resolution
        )
        return self._sums_exposure_series_list(
            exposure_series_list,
            per_second=per_second,
            apply_scaling=apply_scaling,
        )

    def _sums_exposure_series_list(
            self,
            exposure_series_list: list[ExposureSeries],
            *,
            per_second: bool = True,
            apply_scaling: bool = False,
    ) -> pd.DataFrame:
        results = []

        for series in exposure_series_list:
            if per_second:
                series = series.per_second()
            if apply_scaling:
                series = series.scaled()

            exposure_sum = series.mean()

            if series.source_id is not None:
                exposure_sum["filename"] = series.source_id

            results.append(exposure_sum)

        return pd.DataFrame(results)

    def _calculate_exposure_dataframe(
        self,
        trajectory: Trajectory,
        *,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        temporal_resolution: dt.timedelta | None = None,
    ) -> pd.DataFrame:

        if start_time is None:
            start_time = trajectory.df["datetime"].min()
        if end_time is None:
            end_time = trajectory.df["datetime"].max()

        logging.info(f"Computing exposure between {start_time} and {end_time}")

        windows = utils.get_time_windows(start_time, end_time, temporal_resolution)

        ## REFACTOR FROM HERE
        #
        scaling = []
        durations = []
        centres = []
        exposure_data = []

        last_index = len(windows) - 1
        for ii, (start, end) in tqdm(
                enumerate(windows), total=len(windows), desc="Calculating exposure"
        ):
            window = trajectory.data_in_window(
                start=start,
                end=end,
                include_first=(ii == 0), # and self.gap_method != GapMethod.IGNORE,
                include_last=(ii == last_index), # and self.gap_method != GapMethod.IGNORE,
            )

            length = end - start
            duration = length.total_seconds()
            centre = start + (length / 2)
            scaling_ii = self.environment.scaling_at_datetime(centre, self.interp_method)
            rho = self.mobility.distribution(window, self.environment)
            exposure_sources = self.environment.sample(centre, self.interp_method)

            normalised_density = rho.density / rho.density.sum()
            sources_only = exposure_sources.drop(columns=["geometry"])
            exposure_ii = sources_only.mul(normalised_density, axis=0) * duration

            scaling.append(scaling_ii)
            centres.append(centre)
            durations.append(duration)
            exposure_data.append(exposure_ii.sum())

        window_starts, window_ends = zip(*windows)
        summary_df = pd.DataFrame(exposure_data)
        summary_df["scaling"] = scaling
        summary_df["window_start"] = list(window_starts)
        summary_df["window_centre"] = centres
        summary_df["window_end"] = list(window_ends)
        summary_df["window_length_seconds"] = durations
        #
        ## END REFACTOR

        logging.info("Exposure calculation complete")
        return summary_df

    def _resolve_temporal_resolution(
            self,
            timestep: dt.timedelta | None,
    ) -> dt.timedelta:
        """Resolve effective temporal resolution from argument, Exposure, or Environment."""
        candidates = [
            timestep,
            self.timestep,
            self.environment.temporal_resolution,
        ]
        valid = [r for r in candidates if r is not None]
        resolved = min(valid) if valid else None
        logging.info(f"Using temporal resolution of {resolved}")
        return resolved

