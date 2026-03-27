import datetime as dt
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from . import Environment, Mobility, utils
from .data import Trajectory
from .enums import GapMethod, InterpMethod


class ExposureSeries:
    META_COLUMNS = (
        "scaling",
        "window_start",
        "window_centre",
        "window_end",
        "window_length_seconds",
    )

    def __init__(
        self,
        exposure_data: DataFrame,
        metadata: DataFrame,
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
        df: DataFrame,
        source_id: str | None = None
    ) -> "ExposureSeries":
        meta_cols = [col for col in cls.META_COLUMNS if col in df.columns]
        metadata = df[meta_cols].copy()
        exposure_cols = [col for col in df.columns if col not in meta_cols]
        exposure_data = df[exposure_cols].copy()
        return cls(exposure_data=exposure_data, metadata=metadata, source_id=source_id)

    @property
    def dataframe(self) -> DataFrame:
        return pd.concat([self.exposure_data, self.metadata], axis=1)

    def per_second(self) -> "ExposureSeries":
        if "window_length_seconds" not in self.metadata.columns:
            raise ValueError("metadata does not contain 'window_length_seconds'")
        exposure_norm = self.exposure_data.div(
            self.metadata["window_length_seconds"], axis=0
        )
        return ExposureSeries(exposure_data=exposure_norm, metadata=self.metadata)

    def scaled(self) -> "ExposureSeries":
        if "scaling" not in self.metadata.columns:
            raise ValueError("metadata does not contain 'scaling'")
        scaled_exposure = self.exposure_data.mul(self.metadata["scaling"], axis=0)
        return ExposureSeries(exposure_data=scaled_exposure, metadata=self.metadata)

    def mean(self) -> pd.Series:
        return self.exposure_data.mean()

    def plot_over_time(
        self,
        ax=None,
        *,
        cumulative: bool = False,
        x_col: str = "window_centre",
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

        series = self.scaled() if apply_scaling else self

        if x_col not in series.metadata.columns:
            raise ValueError(f"metadata must contain x_col={x_col!r}")

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

        plot_df = series.dataframe.sort_values(by=x_col)

        for col in series.exposure_data.columns:
            y = plot_df[col]
            if cumulative:
                y = np.cumsum(y)

            ax.plot(
                plot_df[x_col],
                y,
                "o-",
                label=col,
                **line_kwargs,
            )

        if title is not None:
            ax.set_title(title)

        if ylim is not None:
            ax.set_ylim(ylim)
        elif auto_ylim_pad is not None:
            try:
                y_max = float(np.nanmax(series.exposure_data.to_numpy(dtype=float)))
                if np.isfinite(y_max):
                    ax.set_ylim((0.0, y_max * auto_ylim_pad))
            except Exception:
                pass

        if legend:
            fig.legend(**legend_kwargs)

        if xtick_rotation is not None:
            for tick in ax.get_xticklabels():
                tick.set_rotation(xtick_rotation)

        fig.tight_layout()
        if show:
            plt.show()

        return fig, ax


class Exposure:
    def __init__(
        self,
        mobility: Mobility,
        environment: Environment,
        *,
        timestep: dt.timedelta | None = None,
        interp_method: InterpMethod = InterpMethod.INTERP,
        gap_method: GapMethod = GapMethod.VORONOI
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
        self.gap_method = gap_method
        self.environment.calculate()

    def for_trajectory(
        self,
        trajectory: Trajectory,
        *,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        temporal_resolution: dt.timedelta | None = None,
    ) -> ExposureSeries:
        summary_df = self._calculate_exposure_dataframe(
            trajectory=trajectory,
            start_time=start_time,
            end_time=end_time,
            temporal_resolution=temporal_resolution,
        )
        return ExposureSeries.from_dataframe(summary_df, source_id=trajectory.source_id)

    def for_trajectories(
        self,
        trajectories: list[Trajectory],
        *,
        temporal_resolution: dt.timedelta | None = None,
    ) -> list[ExposureSeries]:
        return [
            self.for_trajectory(
                trajectory=trajectory,
                temporal_resolution=temporal_resolution,
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
        """Compute exposure sums from either trajectories or precomputed exposure series.

        Dispatches to the appropriate internal method based on the type of the
        first element in the input list.

        Args:
            inputs: Either a list of Trajectory objects or a list of precomputed
                ExposureSeries objects. The type of the first element determines
                which internal method is used.
            temporal_resolution: The temporal resolution to use when computing
                exposure series from trajectories. Not applicable when passing
                precomputed ExposureSeries objects.
            per_second: Whether to normalise exposure values per second before
                computing sums. Defaults to True.
            apply_scaling: Whether to apply scaling to the exposure series before
                computing sums. Defaults to False.

        Returns:
            A DataFrame containing the exposure sums for each input, with one row
            per trajectory or exposure series. Rows will include a ``filename``
            column where a ``source_id`` is present on the ExposureSeries.

        Raises:
            ValueError: If ``timestep`` is provided alongside a list of
                precomputed ExposureSeries objects.
            TypeError: If the input list contains unsupported types.
        """
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
                    "`timestep` is not applicable when passing "
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

    def _prepare_trajectory(
            self,
            trajectory: Trajectory,
            resolution: dt.timedelta,
    ) -> Trajectory:
        """
        Args:
            trajectory:
            resolution:

        Returns:
            a new Trajectory with dwell times in the data
        """
        match self.gap_method:
            case GapMethod.VORONOI:
                return trajectory.with_voronoi_dwells()
            case GapMethod.INTERPOLATE:
                return trajectory.with_interpolated_gaps(resolution)
            case GapMethod.RECENT:
                return trajectory.with_recent_fill(resolution)
            case GapMethod.IGNORE:
                return trajectory.with_ignored_gaps(resolution)

    def _calculate_exposure_dataframe(
        self,
        trajectory: Trajectory,
        *,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        temporal_resolution: dt.timedelta | None = None,
    ) -> pd.DataFrame:

        if start_time is None:
            start_time = trajectory.data["datetime"].min()
        if end_time is None:
            end_time = trajectory.data["datetime"].max()

        effective_resolution = self._resolve_temporal_resolution(temporal_resolution)
        trajectory = self._prepare_trajectory(trajectory, effective_resolution)
        windows = utils.get_time_windows(start_time, end_time, effective_resolution)

        scaling = []
        durations = []
        centres = []
        exposure_data = []

        logging.info(f"Computing exposure between {start_time} and {end_time}")
        logging.info(f"Temporal resolution: {effective_resolution}")

        last_index = len(windows) - 1
        for ii, (window_start, window_end) in tqdm(
                enumerate(windows), total=len(windows), desc="Windows"
        ):
            # Use trajectory bounds where window extends too far
            start = max(window_start, trajectory.start_time.to_pydatetime())
            end = min(window_end, trajectory.end_time.to_pydatetime())

            window = trajectory.data_in_window(
                start=start,
                end=end,
                include_first=(ii == 0) and self.gap_method != GapMethod.IGNORE,
                include_last=(ii == last_index) and self.gap_method != GapMethod.IGNORE,
            )
            length = end - start
            duration = length.total_seconds()
            centre = start + length / 2
            scaling_ii = self.environment._get_scaling(centre, self.interp_method)
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
        logging.info("Exposure calculation complete")
        return summary_df

    def _resolve_temporal_resolution(
            self,
            temporal_resolution: dt.timedelta | None,
    ) -> dt.timedelta:
        """Resolve effective temporal resolution from argument, Exposure, or Environment."""
        candidates = [
            temporal_resolution,
            self.timestep,
            self.environment.temporal_resolution,
        ]
        valid = [r for r in candidates if r is not None]
        if not valid:
            raise ValueError(
                "timestep must be defined by argument, Exposure, or Environment"
            )
        return min(valid)

