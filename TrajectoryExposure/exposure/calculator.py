import datetime as dt
import logging
from typing import Sequence

import pandas as pd

from TrajectoryExposure import Environment, Mobility, SamplingMethod, Trajectory
from TrajectoryExposure.core import utils
from TrajectoryExposure.data.trajectory import DATETIME
from TrajectoryExposure.exposure.results import ExposureSeries

logger = logging.getLogger(__name__)


class Exposure:
    def __init__(
        self,
        mobility: Mobility,
        environment: Environment,
        *,
        timestep: dt.timedelta | None = None,
        interp_method: SamplingMethod = SamplingMethod.INTERP,
    ) -> None:
        """Args:
        mobility:
        environment:
        timestep:
        interp_method: "interp", "nearest"
        """
        self.mobility = mobility
        self.environment = environment
        self.timestep = timestep
        self.interp_method = interp_method

    def for_trajectory(
        self,
        trajectory: Trajectory,
        *,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        timestep: dt.timedelta | None = None,
    ) -> ExposureSeries:
        """

        Args:
            trajectory:
            start_time:
            end_time:
            timestep:

        Returns:

        """
        if not self.environment.calculated:
            self.environment.calculate()
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
        trajectories: Sequence[Trajectory],
        *,
        temporal_resolution: dt.timedelta | None = None,
    ) -> Sequence[ExposureSeries]:
        """

        Args:
            trajectories:
            temporal_resolution:

        Returns:

        """
        # TODO: add progress bar or logging here
        return [
            self.for_trajectory(
                trajectory=trajectory,
                timestep=temporal_resolution,
            )
            for trajectory in trajectories
        ]

    def sums(
        self,
        inputs: Sequence[Trajectory] | Sequence[ExposureSeries],
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

        if isinstance(first, ExposureSeries):
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
        """

        Args:
            trajectories:
            temporal_resolution:
            per_second:
            apply_scaling:

        Returns:

        """
        exposure_series_list = self.for_trajectories(
            trajectories, temporal_resolution=temporal_resolution
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
        """

        Args:
            exposure_series_list:
            per_second:
            apply_scaling:

        Returns:

        """
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
        """

        Args:
            trajectory:
            start_time:
            end_time:
            temporal_resolution:

        Returns:

        """

        if start_time is None:
            start_time = trajectory.df[DATETIME].min()
        if end_time is None:
            end_time = trajectory.df[DATETIME].max()

        windows = utils.get_time_windows(start_time, end_time, temporal_resolution)

        ## REFACTOR FROM HERE
        #
        scaling = []
        durations = []
        centres = []
        exposure_data = []

        last_index = len(windows) - 1
        logger.info(
            f"Computing exposure in {len(windows)} windows between {start_time} and {end_time}"
            f" (temporal resolution: {temporal_resolution})"
        )

        for ii, (start, end) in enumerate(windows):
            window = trajectory.data_in_window(
                start=start,
                end=end,
                include_first=(ii == 0),  # and self.gap_method != GapMethod.IGNORE,
                include_last=(ii == last_index),  # and self.gap_method != GapMethod.IGNORE,
            )

            length = end - start
            duration = length.total_seconds()
            centre = start + (length / 2)

            scaling_ii = self.environment.scaling_at_timestamp(centre, self.interp_method)

            rho = self.mobility.distribution(window, self.environment)
            normalised_density = rho.density / rho.density.sum()

            exposure_sources = self.environment.sample(centre, self.interp_method)
            extra_cols = [
                col
                for col in exposure_sources.columns
                if col not in self.environment.columns()
            ]
            sources_only = exposure_sources.drop(columns=extra_cols)

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
        return resolved
