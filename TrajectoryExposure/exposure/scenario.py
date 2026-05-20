import datetime as dt
from pathlib import Path
from typing import Sequence

import attrs

from ..core.enums import GapMethod
from ..core.environment import Environment
from ..data.trajectory import Trajectory
from ..mobility.base import Mobility


@attrs.frozen
class Scenario:
    trajectory: Trajectory
    mobility: Mobility
    environment: Environment
    timestep: dt.timedelta | None = None
    gap_method: GapMethod | None = None


@attrs.frozen
class ScenarioKey:
    source_id: str
    mobility: str
    environment: str

    def to_path(self, base_dir: Path) -> Path:
        return base_dir / self.environment / self.mobility / f"{self.source_id}.pkl"


@attrs.define
class ScenarioBatch:
    trajectories: Sequence[Trajectory]
    mobility_models: dict[str, Mobility]
    environments: dict[str, Environment]
    timestep: dt.timedelta | None = None
    gap_method: GapMethod | None = None

    def __attrs_post_init__(self) -> None:
        missing = [i for i, t in enumerate(self.trajectories) if t.source_id is None]
        if missing:
            raise ValueError(
                f"All trajectories must have a source_id. "
                f"Missing at indices: {missing}"
            )

    def to_scenarios(self) -> list[tuple[ScenarioKey, Scenario]]:
        """Expand all combinations into individual Scenario instances."""
        raise NotImplementedError
