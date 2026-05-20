import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .exposure import Exposure
from .results import ExposureSeries
from .scenario import Scenario, ScenarioBatch, ScenarioKey

logger = logging.getLogger(__name__)


def _run_single(
    scenario: Scenario,
    key: ScenarioKey,
) -> tuple[ScenarioKey, ExposureSeries]:
    """Compute exposure for a single scenario combination.

    This is a module-level function so it is pickleable for
    multiprocessing.

    Args:
        scenario: The parent scenario providing shared configuration.
        key: Identifies the specific trajectory/mobility/environment
            combination to evaluate.

    Returns:
        The key paired with its computed ExposureSeries.
    """
    exposure = Exposure(
        mobility=scenario.mobility,
        environment=scenario.environment,
        timestep=scenario.timestep,
    )
    return key, exposure.for_trajectory(scenario.trajectory)



class ScenarioCalculator:
    """Executes all input combinations in a ScenarioBatch.

    Supports sequential and parallel execution, optional result
    caching to disk, and pre-computation of shared environments
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
            Mapping from ScenarioKey to ExposureSeries for every
            combination in the scenario.
        """
        self._precalculate_environments()
        keys = self._make_keys()

        if self.max_workers == 1 or self.max_workers is None:
            results = self._run_sequential(keys)
        else:
            results = self._run_parallel(keys)

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

    def _make_keys(self) -> list[ScenarioKey]:
        """Return all (trajectory, mobility, environment) combinations."""
        return [
            ScenarioKey(
                source_id=t.source_id,
                mobility=mob_label,
                environment=env_label,
            )
            for t in self.batch.trajectories
            for mob_label in self.batch.mobility_models
            for env_label in self.batch.environments
        ]

    def _run_sequential(
        self, keys: list[ScenarioKey]
    ) -> dict[ScenarioKey, ExposureSeries]:
        """Run combinations sequentially."""
        results = {}
        for key in keys:
            logger.info("Running: %s", key)
            _, series = _run_single(self.batch, key)
            results[key] = series
        return results

    def _run_parallel(
        self, keys: list[ScenarioKey]
    ) -> dict[ScenarioKey, ExposureSeries]:
        """Run combinations in parallel using ProcessPoolExecutor."""
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_single, self.batch, key): key
                for key in keys
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    _, series = future.result()
                    results[key] = series
                    logger.info("Completed: %s", key)
                except Exception:
                    logger.exception("Failed: %s", key)
        return results

    def _save(self, results: dict[ScenarioKey, ExposureSeries]) -> None:
        """Serialise each ExposureSeries to disk under output_dir."""
        for key, series in results.items():
            path = key.to_path(self.output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            series.save(path)
