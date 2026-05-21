import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .exposure import Exposure
from .results import ExposureSeries
from .scenario import Scenario, ScenarioBatch, ScenarioKey, ScenarioResult

logger = logging.getLogger(__name__)


def _run_single(key: ScenarioKey, scenario: Scenario) -> ExposureSeries:
    """Compute exposure for a single scenario combination.

    This is a module-level function so it is pickleable for
    multiprocessing.

    Args:
        scenario: Fully resolved single scenario to evaluate.

    Returns:
        The computed ExposureSeries.
    """
    trajectory = scenario.trajectory.with_dwells(scenario.gap_method)
    exposure = Exposure(
        mobility=scenario.mobility,
        environment=scenario.environment,
        timestep=scenario.timestep,
    )
    series = exposure.for_trajectory(trajectory)
    occupancy = scenario.mobility.distribution(trajectory, scenario.environment)
    return ScenarioResult(
        key=key,
        exposure=series,
        occupancy=occupancy,
        trajectory=trajectory,
    )


class ScenarioCalculator:
    """Executes all input combinations in a ScenarioBatch.

    Supports sequential and parallel execution, optional result
    saving to disk, and pre-computation of shared environments
    before workers are spawned.

    Args:
        batch: The ScenarioBatch to execute.
        output_dir: If provided, each ExposureSeries is saved to disk
            under this directory using ScenarioKey.to_path().
        max_workers: Number of parallel workers. If None or 1,
            runs sequentially.
    """
    def __init__(
            self,
            batch: ScenarioBatch,
            output_dir: Path | None = None,
            max_workers: int | None = None,
    ) -> None:
        self.batch = batch
        self.output_dir = output_dir
        self.max_workers = max_workers

    def run(self) -> dict[ScenarioKey, ExposureSeries]:
        """Run all scenario combinations and return results.

        Environments are pre-calculated before workers are spawned to
        avoid cache write races in parallel execution.

        Returns:
            Mapping from ScenarioKey to ExposureSeries for every combination in the batch.
        """
        self._precalculate_environments()
        scenarios = self.batch.to_scenarios()

        if self.max_workers == 1 or self.max_workers is None:
            logging.info("Calculating %s scenarios", len(scenarios))
            results = self._run_sequential(scenarios)
        else:
            logging.info(
                "Calculating %s scenarios with %s workers",
                len(scenarios),
                self.max_workers,
            )
            results = self._run_parallel(scenarios)

        if self.output_dir is not None:
            self._save(results)

        return results

    def _precalculate_environments(self) -> None:
        """Call calculate() on all environments before parallelising.

        Ensures metric caches are warm so parallel workers do not race
        on cache writes.
        """
        for label, env in self.batch.environments.items():
            logger.info("Pre-calculating environment: %s", label)
            env.calculate()

    def _run_sequential(
            self, scenarios: list[tuple[ScenarioKey, Scenario]],
    ) -> list[ScenarioResult]:
        """Run combinations sequentially."""
        results = []
        for key, scenario in scenarios:
            logger.info("Running scenario:\n>>> %s", key)
            results.append(_run_single(key, scenario))
        return results

    def _run_parallel(
            self, scenarios: list[tuple[ScenarioKey, Scenario]],
    ) -> list[ScenarioResult]:
        """Run combinations in parallel using ProcessPoolExecutor."""
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_single, key, scenario): key
                for key, scenario in scenarios
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results.append(future.result())
                    logger.info("Completed: %s", key)
                except Exception:
                    logger.exception("Failed: %s", key)
        return results

    def _save(self, results: list[ScenarioResult]) -> None:
        """Serialise each ExposureSeries to disk under output_dir."""
        for result in results:
            path = result.key.to_path(self.output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.exposure.save(path)
