# Contributing to geoexposure

This document describes the conventions, tooling, and workflow used in this project.

## Licence

This project is licensed under the
[GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html).
By contributing, you agree that your contributions will be released under the same licence.


## Requirements

- Python 3.12 or later
- Dependencies are listed in pyproject.toml
- Install the package in editable mode: pip install -e ".[dev]"

## Code Style

All code is formatted and linted with Ruff. Configuration is in pyproject.toml.
- Line length: 99 characters
- Docstring convention: Google style, British English
- Use `X | Y` unions, not `Optional[X]`
- Use `from __future__ import annotations` in files with `TYPE_CHECKING` imports
- Third-party annotation-only imports go under `TYPE_CHECKING`

Run ruff check . and ruff format . before committing.

## Docstrings

- All public modules, classes, methods, and functions must have docstrings
- Private methods need at minimum a one-line docstring
- `Attributes:` on the class docstring; `Args:`/`Raises:` on `__init__`
- Abstract methods document the full contract in the base class


## Data Classes

- `@attrs.frozen` for immutable data carriers
- `@attrs.define` for mutable classes with validation
- No `@dataclass` - use attrs throughout
- Mutable defaults use `attrs.Factory`

## Naming

Metric column names use `__` separators:
- Spatial: `spatial__{key}__{metric.name}`
- Temporal: `temporal__{key}`
- Metric names: `{metric_title}__{arg1}__{arg2}...`
Non-alphanumeric characters in values are sanitised via `Metric._sanitise_label()`.

## Design Notes

- Cache directory resolves via: instance `_cache_dir` > `TRAJECTORY_EXPOSURE_CACHE_DIR` env var > `.cache/`
- Subclasses override `_hash_params()` to include parameters in the cache key
- `Trajectory.df` uses `pd.Timestamp`; `get_data_arrays()` returns `dt.datetime`
- do not mix `np.datetime64` with `dt.timedelta`
- Abstract method bodies use `...` not `raise NotImplementedError`
- `TemporalType` is required on `TemporalData`; `CYCLIC` requires `cycle_duration`; `DATED` requires `dt.datetime` keys

## Pull Requests

- Open an issue before starting significant work
- One feature or fix per PR
- Ruff must pass before opening a PR
- Update docstrings for any changed public interface
- Describe what changed and why; reference related issues
