"""Scenario management for batched exposure calculations.

This module provides data structures and utilities for defining, organising,
and executing exposure calculations across multiple combinations of
trajectories, mobility models, environments, gap-filling strategies, and
time steps.

A :class:`Scenario` represents a single fully-resolved combination of inputs.
A :class:`ScenarioBatch` holds collections of each input type and expands them
into individual :class:`Scenario` instances either as a full Cartesian product
or as paired combinations. Results are stored as :class:`ScenarioResult`
objects which can be serialised to and deserialised from disk.
"""

import datetime as dt
import logging
import pickle
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np

from ..core.enums import GapMethod
from ..core.environment import Environment
from ..core.spatial_utils import infer_raster_grid
from ..data.trajectory import Trajectory
from ..mobility.base import Mobility

if TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd

    from .. import ExposureSeries

logger = logging.getLogger(__name__)


def _sanitise_label(label: str) -> str:
    """Replace characters unsafe in folder/file names with underscores."""
    return re.sub(r'[<>:"/\\|?* ]', "_", label)


@attrs.frozen
class Scenario:
    """"A fully resolved set of inputs for a single exposure calculation.

    Attributes:
        trajectory: The Trajectory to evaluate.
        mobility: Mobility model used to compute the occupancy distribution.
        environment: Environment defining spatial and temporal exposure sources.
        gap_method: Tuple of ``(GapMethod, resolution)`` used to fill trajectory gaps.
        timestep: Time window size for the exposure calculation. If ``None``, resolved from the
            environment or mobility model.
    """
    trajectory: Trajectory
    mobility: Mobility
    environment: Environment
    gap_method: tuple[GapMethod, dt.timedelta | None]
    timestep: dt.timedelta | None = None


@attrs.frozen
class ScenarioKey:
    """Immutable identifier for a single scenario combination.

    Used as a key when storing and retrieving :class:`ScenarioResult` objects.
    All fields are sanitised on construction to be safe for use as directory
    and file name components.

    Attributes:
        environment: Label identifying the environment used.
        mobility: Label identifying the mobility model used.
        gap_method: String encoding the gap method and resolution.
        source_id: Sanitised trajectory source identifier.
        timestep: String encoding the timestep in seconds, or ``"total"``.
    """
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
        """Return the output file path for this scenario under a base directory.

        Double-underscores are used as separators to enable components of the key to be split.

        The path is structured as::

            base_dir / environment / mobility / source_id__gap_method__timestep.pkl

        Args:
            base_dir: Root directory under which results are stored.

        Returns:
            Full path to the result file for this scenario.
        """
        return (
                base_dir
                / self.environment
                / f"{self.mobility}"
                / f"{self.source_id}__{self.gap_method}__{self.timestep}.pkl"
        )

    def to_str(self) -> str:
        """Return a single string representation of this scenario key.

        Returns:
            String as ``"{environment}__{mobility}__{source_id}__{gap_method}__{timestep}.pkl"``.
        """
        return (
            f"{self.environment}__"
            f"{self.mobility}__"
            f"{self.source_id}__"
            f"{self.gap_method}__"
            f"{self.timestep}.pkl"
        )


@attrs.frozen
class ScenarioResult:
    """The result of a single scenario evaluation.

    Stores the exposure series, occupancy distribution, and metadata needed to reconstruct and
    inspect the result. Can be save and loaded from disk using :meth:`save` and :meth:`load`.

    Attributes:
        key: The :class:`ScenarioKey` identifying this result.
        trajectory: The :class:`Trajectory` used in the calculation.
        exposure: The computed :class:`~exposure.results.ExposureSeries`.
        occupancy_density: Normalised density values from the mobility model, one per raster cell.
        occupancy_coordinates: Array of raster cell centroid coordinates.
        crs: Coordinate reference system of the raster grid as a WKT string.
    """
    key: ScenarioKey
    trajectory: Trajectory
    exposure: "ExposureSeries"
    occupancy_density: "pd.Series"
    occupancy_coordinates: np.ndarray
    crs: str

    def save(self, path: Path) -> None:
        """Save this ScenarioResult to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ScenarioResult":
        """Load a ScenarioResult from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    def occupancy_gdf(self) -> "gpd.GeoDataFrame":
        """Reconstruct the occupancy distribution as a GeoDataFrame.

        Infers the raster grid parameters from the stored centroid coordinates and constructs a
        GeoDataFrame with a ``density`` column aligned to the raster geometry.

        Returns:
            GeoDataFrame with one row per raster cell and a ``density`` column
            containing the normalised occupancy values.
        """
        grid = infer_raster_grid(self.occupancy_coordinates)
        gdf = grid.to_polygon_gdf(self.crs)
        gdf["density"] = self.occupancy_density.values
        return gdf


@attrs.define
class ScenarioBatch:
    """A collection of input axes to be expanded into individual scenarios.

    Holds sequences of trajectories, mobility models, environments, gap methods,
    and timesteps. These can be expanded into individual :class:`Scenario`
    instances either as a full Cartesian product via :meth:`to_product`, or as
    paired combinations via :meth:`to_zip`.

    All trajectories must have a ``source_id`` set.

    Attributes:
        trajectories: Sequence of trajectories to evaluate.
        mobility_models: Named mapping of mobility models.
        environments: Named mapping of environments.
        gap_methods: Sequence of ``(GapMethod, resolution)`` pairs.
        timesteps: Sequence of time window sizes.
    """
    trajectories: Sequence[Trajectory]
    mobility_models: dict[str, Mobility]
    environments: dict[str, Environment]
    gap_methods: Sequence[tuple[GapMethod, dt.timedelta | None]]
    timesteps: Sequence[dt.timedelta]

    def __attrs_post_init__(self) -> None:
        """Validate that all trajectories have ``source_id`` set.

        Raises:
            ValueError: If any trajectory is missing ``source_id``.
        """
        missing = [i for i, t in enumerate(self.trajectories) if t.source_id is None]
        if missing:
            raise ValueError(
                f"All trajectories must have a source_id. "
                f"Missing at indices: {missing}",
            )

    @property
    def axes(self) -> list[str]:
        """Return the names of the batch axes in expansion order."""
        return self.axis_lengths.keys()

    @property
    def axis_lengths(self) -> dict[str, int]:
        """Return the number of elements along each batch axis."""
        return {
            "trajectories"   : len(self.trajectories),
            "mobility_models": len(self.mobility_models),
            "environments"   : len(self.environments),
            "timesteps"      : len(self.timesteps),
            "gap_methods"    : len(self.gap_methods),
        }

    @property
    def len_product(self) -> int:
        """Return the total number of scenarios in a full Cartesian product expansion."""
        return (
                len(self.trajectories)
                * len(self.mobility_models)
                * len(self.environments)
                * len(self.timesteps)
                * len(self.gap_methods)
        )

    @property
    def len_zip(self) -> int:
        """Return the number of scenarios in a paired zip expansion.

        Raises:
            ValueError: If non-singular axes have different lengths.
        """
        self._validate_zip_lengths()
        return max(
            len(self.trajectories),
            len(self.mobility_models),
            len(self.environments),
            len(self.timesteps),
            len(self.gap_methods),
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
        """Construct a ScenarioBatch from flexible input types.

        Accepts single instances or sequences for each axis, and single instances or dicts for
        mobility models and environments. Single instances are wrapped automatically. Single-valued
        instances for `mobility_models` and `environments` are stored with the dict keys "mobility"
        and "environment" respectively.

        Args:
            trajectories: One or more trajectories to evaluate.
            mobility_models: One or more named mobility models.
            environments: One or more named environments.
            gap_methods: One or more ``(GapMethod, resolution)`` pairs.
            timesteps: One or more time window sizes, or ``None`` for a single ``None`` timestep.

        Returns:
            A new :class:`ScenarioBatch` instance.
        """
        return cls(
            trajectories=_ensure_sequence(trajectories),
            mobility_models=_ensure_dict(mobility_models, Mobility, "mobility"),
            environments=_ensure_dict(environments, Environment, "environment"),
            gap_methods=_ensure_gap_methods(gap_methods),
            timesteps=_ensure_sequence(timesteps),
        )

    def to_product(self) -> list[tuple[ScenarioKey, Scenario]]:
        """Expand all combinations into individual Scenario instances."""
        logger.info("Expanding scenarios in 'product' mode")
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
        logger.info("Expanding scenarios in 'zip' mode")
        max_axis_length = self._validate_zip_lengths()
        trajectories = _broadcast(self.trajectories, max_axis_length)
        mob_items = _broadcast(list(self.mobility_models.items()), max_axis_length)
        env_items = _broadcast(list(self.environments.items()), max_axis_length)
        timesteps = _broadcast(self.timesteps, max_axis_length)
        gap_methods = _broadcast(self.gap_methods, max_axis_length)

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
            mode: str = "product",
    ) -> list[ScenarioKey]:
        """Returns a list of ScenarioResults for all Scenarios in the batch.

        Args:
            max_workers: the maximum number of parallel processing jobs.
            output_dir: where to save results from batch processing.
            mode: method to expand batch, either 'product' or 'zip.

        Returns:
            a list of ScenarioResults

        Raises:
            ValueError when mode is not 'product' or 'zip'
        """
        if mode == "product":
            batch_len = self.len_product
        elif mode == "zip":
            batch_len = self.len_zip
        else:
            raise ValueError(f"unknown batch mode: {mode}")

        num_workers = None if max_workers is None else min(max_workers, batch_len)
        from .calculator import ScenarioCalculator  # noqa: PLC0415
        calculator = ScenarioCalculator(self, output_dir, num_workers)
        return calculator.run(mode=mode)

    def _validate_zip_lengths(self) -> int:
        """Validate zip axis lengths and return the batch size.

        Returns:
            Length of the zip batch.

        Raises:
            ValueError: If non-singular axes have different lengths.
        """
        lengths = self.axis_lengths
        non_singular = {k: v for k, v in lengths.items() if v != 1}
        if len(set(non_singular.values())) > 1:
            raise ValueError(
                f"In zip mode all non-singular axes must have the same length. "
                f"Got: {non_singular}",
            )
        return max(lengths.values())


def _broadcast(seq: Sequence, length: int) -> Sequence:
    """Repeat a length-1 sequence to match length, or return as-is."""
    if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
        return [seq] * length
    if len(seq) == 1:
        return list(seq) * length
    return seq


def _ensure_sequence[T](value: T | Sequence[T]) -> Sequence[T]:
    """Wrap a single value in a list, or return a sequence as-is.

    Args:
        value: A single value or an existing sequence.

    Returns:
        A sequence containing the value, or the original sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return [value]
    return value


def _ensure_dict[T](
        value: T | dict[str, T],
        type_: type[T],
        fallback_key: str = "default",
) -> dict[str, T]:
    """Wrap a single instance in a dict, or return a dict as-is.

    Args:
        value: A single instance of ``type_`` or an existing dict.
        type_: The expected single instance type.
        fallback_key: Key to use when wrapping a single instance.

    Returns:
        A dict mapping string keys to instances of ``type_``.

    Raises:
        TypeError: If ``value`` is neither a ``dict`` nor an instance of ``type_``.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, type_):
        return {fallback_key: value}
    raise TypeError(f"Expected {type_.__name__} or dict, got {type(value).__name__}")


def _ensure_gap_methods(
        value: tuple[GapMethod, dt.timedelta | None] | Sequence[
            tuple[GapMethod, dt.timedelta | None]],
) -> Sequence[tuple[GapMethod, dt.timedelta | None]]:
    """Wrap a single (GapMethod, resolution) pair in a list if needed."""
    expected_len = 2
    if isinstance(value, tuple) and len(value) == expected_len and isinstance(value[0], GapMethod):
        return [value]
    return value


def _gap_method_key(gap_method_tuple: tuple[GapMethod, dt.timedelta | None]) -> str:
    """Return a string key for a ``(GapMethod, resolution)`` pair.

    Args:
        gap_method_tuple: Tuple of gap method and optional resolution timedelta.

    Returns:
        String of the form ``"{method}_{seconds}s"``, or just ``"{method}"``
        if resolution is ``None``.
    """
    method, resolution = gap_method_tuple
    if resolution is None:
        return str(method)
    return f"{method}_{int(resolution.total_seconds())}s"
