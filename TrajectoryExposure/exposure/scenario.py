import datetime as dt
import pickle
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
    gap_method: tuple[GapMethod, dt.timedelta | None]
    timestep: dt.timedelta | None = None


@attrs.frozen
class ScenarioKey:
    environment: str
    mobility: str
    gap_method: str
    source_id: str
    timestep: str

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
                / f"{self.mobility}"
                / f"{self.gap_method}__{self.source_id}__{self.timestep}.pkl"
        )


@attrs.frozen
class ScenarioResult:
    key: ScenarioKey
    exposure: "ExposureSeries"
    occupancy: "gpd.GeoDataFrame"
    trajectory: Trajectory

    def save(self, path: Path) -> None:
        """Serialise this ScenarioResult to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ScenarioResult":
        """Deserialise a ScenarioResult from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

@attrs.define
class ScenarioBatch:
    trajectories: Sequence[Trajectory]
    mobility_models: dict[str, Mobility]
    environments: dict[str, Environment]
    gap_methods: Sequence[tuple[GapMethod, dt.timedelta | None]]
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
            gap_methods: tuple[GapMethod, dt.timedelta] | Sequence[tuple[GapMethod, dt.timedelta]],
            timesteps: dt.timedelta | Sequence[dt.timedelta] | None,
    ) -> "ScenarioBatch":
        return cls(
            trajectories=_ensure_sequence(trajectories),
            mobility_models=_ensure_dict(mobility_models, Mobility, "mobility"),
            environments=_ensure_dict(environments, Environment, "environment"),
            gap_methods=_ensure_gap_methods(gap_methods),
            timesteps=_ensure_sequence(timesteps),
        )

    def to_product(self) -> list[tuple[ScenarioKey, Scenario]]:
        """Expand all combinations into individual Scenario instances."""
        return [
            (
                ScenarioKey(
                    mobility=mob_label,
                    environment=env_label,
                    gap_method=_gap_method_key(gm),
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

    def to_zip(self) -> list[tuple[ScenarioKey, Scenario]]:
        """Expand paired combinations into individual Scenario instances.

        All non-singular axes must have the same length. Singular axes
        (length 1) are broadcast to match.

        Raises:
            ValueError: If non-singular axes have different lengths.
        """
        lengths = {
            "trajectories"   : len(self.trajectories),
            "mobility_models": len(self.mobility_models),
            "environments"   : len(self.environments),
            "timesteps"      : len(self.timesteps),
            "gap_methods"    : len(self.gap_methods),
        }
        non_singular = {k: v for k, v in lengths.items() if v != 1}
        if len(set(non_singular.values())) > 1:
            raise ValueError(
                f"In zip mode all non-singular axes must have the same length. "
                f"Got: {non_singular}",
            )

        n = max(lengths.values())
        trajectories = _broadcast(self.trajectories, n)
        mob_items = _broadcast(list(self.mobility_models.items()), n)
        env_items = _broadcast(list(self.environments.items()), n)
        timesteps = _broadcast(self.timesteps, n)
        gap_methods = _broadcast(self.gap_methods, n)

        return [
            (
                ScenarioKey(
                    mobility=mob_label,
                    environment=env_label,
                    gap_method=_gap_method_key(gm),
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
            for t, (mob_label, mob), (env_label, env), gm, ts
            in zip(trajectories, mob_items, env_items, gap_methods, timesteps, strict=True)
        ]

    def process(
            self,
            max_workers: int | None = None,
            output_dir: Path | None = None,
            mode: str = "product"
    ) -> list[ScenarioResult]:
        from .calculator import ScenarioCalculator  # noqa: PLC0415
        calculator = ScenarioCalculator(self, output_dir, max_workers)
        return calculator.run(mode=mode)


def _broadcast(seq: Sequence, length: int) -> Sequence:
    """Repeat a length-1 sequence to match length, or return as-is."""
    if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
        return [seq] * length
    if len(seq) == 1:
        return list(seq) * length
    return seq


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


def _ensure_gap_methods(
        value: tuple[GapMethod, dt.timedelta | None] | Sequence[tuple[GapMethod, dt.timedelta | None]],
) -> Sequence[tuple[GapMethod, dt.timedelta | None]]:
    """Wrap a single (GapMethod, resolution) pair in a list if needed."""
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], GapMethod):
        return [value]
    return value


def _gap_method_key(gm: tuple[GapMethod, dt.timedelta | None]) -> str:
    method, resolution = gm
    if resolution is None:
        return str(method)
    return f"{method}_{int(resolution.total_seconds())}s"
