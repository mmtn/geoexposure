"""Scenario execution engine for batched exposure calculations.

Provides :class:`ScenarioCalculator`, which executes all scenario combinations defined in a
:class:`~exposure.scenario.ScenarioBatch`, and the module-level helper :func:`_run_one`, which
evaluates a single :class:`~exposure.scenario.Scenario` and serialises the result to disk.

Supports both sequential and parallel execution via
:class:`~concurrent.futures.ProcessPoolExecutor`. Environments are pre-calculated
before workers are spawned to ensure metric caches are warm and avoid write races
in parallel execution.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .exposure import Exposure
from .scenario import Scenario, ScenarioBatch, ScenarioKey, ScenarioResult

logger = logging.getLogger(__name__)


def _run_one(key: ScenarioKey, scenario: Scenario, output_dir: Path) -> None:
    """Compute exposure for a single scenario and save the result to disk.

    Applies the gap-filling strategy to the trajectory, constructs an
    :class:`~exposure.exposure.Exposure` model, computes the exposure series and
    occupancy distribution, and serialises a :class:`~exposure.scenario.ScenarioResult`
    to the path determined by ``key.to_path(output_dir)``.

    This is a module-level function so that it is pickleable for use with
    :class:`~concurrent.futures.ProcessPoolExecutor`. The code is duplicated in
    :meth:`~exposure.scenario.Scenario.run` for convenience when calling outside
    a parallel context.

    Args:
        key: Identifier for this scenario, used to determine the output path.
        scenario: Fully resolved scenario containing the trajectory, mobility model,
            environment, gap method, and timestep.
        output_dir: Root directory under which the result file is written.
    """
    method, resolution = scenario.gap_method
    trajectory = scenario.trajectory.with_dwell_times(method, resolution=resolution)
    exposure = Exposure(
        mobility=scenario.mobility,
        environment=scenario.environment,
        timestep=scenario.timestep,
    )
    series = exposure.for_trajectory(trajectory)
    occupancy = scenario.mobility.distribution(trajectory, scenario.environment)
    coords = np.column_stack([occupancy.point_geometry.x, occupancy.point_geometry.y], )
    result = ScenarioResult(
        key=key,
        trajectory=trajectory,
        exposure=series,
        occupancy_density=occupancy["density"],
        occupancy_coordinates=coords,
        crs=occupancy.crs
    )
    result.save(result.key.to_path(output_dir))


class ScenarioCalculator:
    """Executes all input combinations in a ScenarioBatch.

    Supports sequential and parallel execution, optional result
    saving to disk, and pre-computation of shared environments
    before workers are spawned.

    Args:
        batch: The ScenarioBatch to execute.
        output_dir: If provided, each ExposureSeries is saved to disk
            under this directory using ScenarioKey.to_path().
        num_workers: Number of parallel workers. If None or 1,
            runs sequentially.
    """

    def __init__(
            self,
            batch: ScenarioBatch,
            output_dir: Path,
            num_workers: int | None = None,
    ) -> None:
        """Initialise a ScenarioCalculator.

        Args:
            batch: The :class:`~exposure.scenario.ScenarioBatch` defining the scenarios to execute.
            output_dir: Directory under which each result is saved.
            num_workers: Number of parallel worker processes, sequential if ``None`` or ``1``.
        """
        self.batch = batch
        self.output_dir = output_dir
        self.num_workers = num_workers

    def run(self, mode: str = "product") -> list[ScenarioKey]:
        """Execute all scenario combinations and return their keys.

        Expands the batch using the specified mode, pre-calculates all environments, then runs each
        scenario either sequentially or in parallel depending on ``num_workers``.

        Args:
            mode: Expansion strategy for the batch. Either ``"product"`` for a full Cartesian
                product of all axes, or ``"zip"`` for paired combinations.

        Returns:
            List of :class:`~exposure.scenario.ScenarioKey` instances, one per scenario executed,
            in expansion order.

        Raises:
            ValueError: If ``mode`` is not ``"product"`` or ``"zip"``.
        """
        self._precalculate_environments()

        if mode == "product":
            scenarios = self.batch.to_product()
        elif mode == "zip":
            scenarios = self.batch.to_zip()
        else:
            raise ValueError(f"unknown mode: {mode}")

        lengths = self.batch.axis_lengths
        for axis in self.batch.axes:
            logger.info(f"- {lengths[axis]:-3d} {axis}")

        if self.num_workers == 1 or self.num_workers is None:
            self._run_sequential(scenarios)
        else:
            self._run_parallel(scenarios)

        logger.info("Calculation complete")
        return [key for key, _ in scenarios]

    def _precalculate_environments(self) -> None:
        """Call calculate() on all environments before parallelising.

        Ensures metric caches are warm so parallel workers do not race
        on cache writes.
        """
        for label, env in self.batch.environments.items():
            logger.info("Pre-calculating environment: %s", label)
            env.calculate()

    def _run_sequential(self, scenarios: list[tuple[ScenarioKey, Scenario]]) -> None:
        """Run combinations sequentially."""
        logger.info("Calculating %s scenarios", len(scenarios))
        for key, scenario in scenarios:
            _run_one(key, scenario, self.output_dir)
            logger.info("Completed: %s", key)

    def _run_parallel(self, scenarios: list[tuple[ScenarioKey, Scenario]]) -> None:
        """Run combinations in parallel using ProcessPoolExecutor."""
        logger.info(
            "Calculating %s scenarios with %s workers",
            len(scenarios),
            self.num_workers,
        )
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(_run_one, key, scenario, self.output_dir): key
                for key, scenario in scenarios
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                    logger.info("Completed: %s", key)
                except Exception:
                    logger.exception("Failed: %s", key)
                    raise
