import datetime as dt
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import attrs

from ..core.enums import GapMethod
from ..core.environment import Environment
from ..data.trajectory import Trajectory
from ..mobility.base import Mobility

if TYPE_CHECKING:
    import geopandas as gpd

    from .. import ExposureSeries


def _sanitise_label(label: str) -> str:
    """Replace characters unsafe in folder/file names with underscores."""
    return re.sub(r'[<>:"/\\|?* ]', "_", label)


@attrs.frozen
class Scenario:
    trajectory: Trajectory
    mobility: Mobility
    environment: Environment
    gap_method: GapMethod | None = None
    timestep: dt.timedelta | None = None


@attrs.frozen
class ScenarioKey:
    environment: str
    mobility: str
    gap_method: str
    timestep: str
    source_id: str

    def __attrs_post_init__(self) -> None:
        """Ensure all keys are usable as directory names."""
        object.__setattr__(self, "source_id", _sanitise_label(self.source_id))
        object.__setattr__(self, "mobility", _sanitise_label(self.mobility))
        object.__setattr__(self, "environment", _sanitise_label(self.environment))
        object.__setattr__(self, "timestep", _sanitise_label(self.timestep))
        object.__setattr__(self, "gap_method", _sanitise_label(self.gap_method))

    def to_path(self, base_dir: Path) -> Path:
        return (
                base_dir
                / self.environment
                / f"{self.mobility}__{self.gap_method}__{self.timestep}"
                / f"{self.source_id}.pkl"
        )


@attrs.frozen
class ScenarioResult:
    key: ScenarioKey
    exposure: "ExposureSeries"
    occupancy: "gpd.GeoDataFrame"
    trajectory: Trajectory


@attrs.define
class ScenarioBatch:
    trajectories: Sequence[Trajectory]
    mobility_models: dict[str, Mobility]
    environments: dict[str, Environment]
    gap_methods: Sequence[GapMethod]
    timesteps: Sequence[dt.timedelta]

    def __attrs_post_init__(self) -> None:
        missing = [i for i, t in enumerate(self.trajectories) if t.source_id is None]
        if missing:
            raise ValueError(
                f"All trajectories must have a source_id. "
                f"Missing at indices: {missing}",
            )

    @classmethod
    def create(
            cls,
            trajectories: Trajectory | Sequence[Trajectory],
            mobility_models: Mobility | dict[str, Mobility],
            environments: Environment | dict[str, Environment],
            gap_methods: GapMethod | Sequence[GapMethod],
            timesteps: dt.timedelta | Sequence[dt.timedelta] | None,
    ) -> "ScenarioBatch":
        return cls(
            trajectories=_ensure_sequence(trajectories),
            mobility_models=_ensure_dict(mobility_models, Mobility, "mobility"),
            environments=_ensure_dict(environments, Environment, "environment"),
            gap_methods=_ensure_sequence(gap_methods),
            timesteps=_ensure_sequence(timesteps),
        )

    def to_scenarios(self) -> list[tuple[ScenarioKey, Scenario]]:
        """Expand all combinations into individual Scenario instances."""
        return [
            (
                ScenarioKey(
                    mobility=mob_label,
                    environment=env_label,
                    gap_method=str(gm) if gm is not None else None,
                    timestep=str(int(ts.total_seconds())) if ts is not None else "total",
                    source_id=t.source_id,
                ),
                Scenario(
                    trajectory=t,
                    mobility=mob,
                    environment=env,
                    gap_method=gm,
                    timestep=ts,
                ),
            )
            for t in self.trajectories
            for mob_label, mob in self.mobility_models.items()
            for env_label, env in self.environments.items()
            for gm in self.gap_methods
            for ts in self.timesteps
        ]

    def process(
            self,
            max_workers: int | None = None,
            output_dir: Path | None = None,
    ) -> list[ScenarioResult]:
        from .calculator import ScenarioCalculator  # noqa: PLC0415
        calculator = ScenarioCalculator(self, output_dir, max_workers)
        return calculator.run()


def _ensure_sequence[T](value: T | Sequence[T]) -> Sequence[T]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value]
    return value


def _ensure_dict[T](
        value: T | dict[str, T],
        type_: type[T],
        fallback_key: str = "default",
) -> dict[str, T]:
    if isinstance(value, dict):
        return value
    if isinstance(value, type_):
        return {fallback_key: value}
    raise TypeError(f"Expected {type_.__name__} or dict, got {type(value).__name__}")
