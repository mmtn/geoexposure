"""Exposure model combining mobility and environment to estimate environmental exposure.

:class:`Exposure` integrates a :class:`~TrajectoryExposure.Mobility` model with an
:class:`~TrajectoryExposure.Environment` to compute time-windowed exposure estimates
for one or more :class:`~TrajectoryExposure.Trajectory` instances. Results are
returned as :class:`~TrajectoryExposure.exposure.results.ExposureSeries` objects
which can be aggregated, scaled, and summarised.
"""

import datetime as dt
import logging
from collections.abc import Sequence

import pandas as pd

from ..core import utils
from ..core.environment import Environment
from ..data.columns import DATETIME
from ..data.resampling import SamplingMethod
from ..data.trajectory import Trajectory
from ..exposure.results import ExposureSeries
from ..mobility.base import Mobility

logger = logging.getLogger(__name__)


class Exposure:
    """Combines a mobility model and environment to compute exposure estimates.

    Exposure is calculated by dividing each trajectory into time windows,
    computing the occupancy distribution within each window using the mobility
    model, and integrating it against the environmental exposure sources.

    Attributes:
        mobility: Mobility model used to compute occupancy distributions.
        environment: Environment defining the spatial and temporal exposure sources.
        timestep: Default time window size. If ``None``, resolved from the
            environment's temporal resolution.
        interp_method: Sampling method used when querying the environment at
            window centre times.
    """

    def __init__(
            self,
            mobility: Mobility,
            environment: Environment,
            *,
            timestep: dt.timedelta | None = None,
            interp_method: SamplingMethod = SamplingMethod.INTERP,
    ) -> None:
        """Initialise an Exposure model.

        Args:
            mobility: Mobility model used to compute occupancy distributions from
                trajectory data.
            environment: Environment defining the spatial and temporal exposure
                sources over the study area.
            timestep: Default time window size for exposure calculations. If
                ``None``, the resolution is inferred from the environment or
                supplied per call.
            interp_method: Method used to sample the environment at each window
                centre time. Defaults to :attr:`~SamplingMethod.INTERP`.
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
        """Compute time-windowed exposure for a single trajectory.

        Divides the trajectory into windows of the effective temporal resolution,
        computes the occupancy distribution and environmental exposure within each
        window, and returns the results as an :class:`ExposureSeries`. If
        ``timestep`` is coarser than the effective resolution, the series is
        aggregated before being returned.

        Args:
            trajectory: Input trajectory providing observed positions and times.
            start_time: Start of the exposure calculation window. Defaults to the
                trajectory start time.
            end_time: End of the exposure calculation window. Defaults to the
                trajectory end time.
            timestep: Time window size for this call. Overrides the instance-level
                ``timestep`` and the environment's temporal resolution if provided.

        Returns:
            An :class:`ExposureSeries` containing one row per time window.
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
        """Compute time-windowed exposure for a sequence of trajectories.

        Calls :meth:`for_trajectory` for each trajectory in the sequence and
        returns the results in the same order.

        Args:
            trajectories: Sequence of trajectories to compute exposure for.
            temporal_resolution: Time window size applied to all trajectories.
                If ``None``, resolved from the instance or environment.

        Returns:
            Sequence of :class:`ExposureSeries` objects, one per trajectory.
        """
        # TODO: replace with parallel calculation over Scenarios...
        return [
            self.for_trajectory(
                trajectory=trajectory,
                timestep=temporal_resolution,
            )
            for trajectory in trajectories
        ]

    def _calculate_exposure_dataframe(
            self,
            trajectory: Trajectory,
            *,
            start_time: dt.datetime | None = None,
            end_time: dt.datetime | None = None,
            temporal_resolution: dt.timedelta | None = None,
    ) -> pd.DataFrame:
        """Compute raw exposure integrals over time windows for a single trajectory.

        Divides the time range into windows of ``temporal_resolution``, computes
        the normalised occupancy distribution within each window using the mobility
        model, and integrates it against the environmental exposure sources sampled
        at the window centre time.

        Args:
            trajectory: Input trajectory providing observed positions and times.
            start_time: Start of the exposure calculation range. Defaults to the
                trajectory start time.
            end_time: End of the exposure calculation range. Defaults to the
                trajectory end time.
            temporal_resolution: Duration of each calculation window.

        Returns:
            DataFrame with one row per window containing raw exposure integrals for
            each environmental source, plus metadata columns for window start,
            centre, end, duration, and scaling.
        """
        if start_time is None:
            start_time = trajectory.df[DATETIME].min()
        if end_time is None:
            end_time = trajectory.df[DATETIME].max()

        windows = utils.get_time_windows(start_time, end_time, temporal_resolution)

        scaling = []
        durations = []
        centres = []
        exposure_data = []

        last_index = len(windows) - 1
        logger.info(
            f"Computing exposure in {len(windows)} windows between {start_time} and {end_time}"
            f" (temporal resolution: {temporal_resolution})",
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
                if col not in self.environment.columns
            ]
            sources_only = exposure_sources.drop(columns=extra_cols)

            exposure_ii = sources_only.mul(normalised_density, axis=0) * duration

            scaling.append(scaling_ii)
            centres.append(centre)
            durations.append(duration)
            exposure_data.append(exposure_ii.sum())

        window_starts, window_ends = zip(*windows, strict=True)
        summary_df = pd.DataFrame(exposure_data)
        summary_df["scaling"] = scaling
        summary_df["window_start"] = list(window_starts)
        summary_df["window_centre"] = centres
        summary_df["window_end"] = list(window_ends)
        summary_df["window_length_seconds"] = durations

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
        return min(valid) if valid else None
