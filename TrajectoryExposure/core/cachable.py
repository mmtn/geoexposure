"""Mixin providing general-purpose disk caching for any callable result."""

import hashlib
import logging
import os
import pickle
from _hashlib import HASH
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame

logger = logging.getLogger(__name__)


class Cachable:
    """Mixin providing general-purpose disk caching for any callable result.

    Classes using this mixin can cache and retrieve any pickleable object
    using a hash-based key derived from arbitrary inputs.

    Note that hash values may not be consistent across systems. In these cases the cache will
    become unreliable.

    Attributes:
        cache_dir: Directory in which cached files are stored.
    """

    _cache_dir: Path | None = None

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory as a Path.

        Resolution order:
        1. Instance-level ``_cache_dir`` if set.
        2. ``TRAJECTORY_EXPOSURE_CACHE_DIR`` environment variable if set.
        3. Defaults to ``".cache"`` in the current working directory.
        """
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        env = os.environ.get("TRAJECTORY_EXPOSURE_CACHE_DIR")
        if env is not None:
            return Path(env)
        return Path(".cache")

    @cache_dir.setter
    def cache_dir(self, value: str | Path) -> None:
        """Set the instance-level cache directory, overriding environment and default."""
        self._cache_dir = Path(value)

    def _make_hash(self, *args) -> str:
        """Computes a deterministic MD5 hash from arbitrary input arguments.

        Handles the following types explicitly:
            - gpd.GeoDataFrame: hashed via WKB geometry, CRS, and columns
            - np.ndarray: hashed via raw bytes
            - float, int, str, bool: hashed via numpy or encoded bytes

        All other types are hashed via pickle serialisation.

        Args:
            *args: Any number of objects to include in the hash.

        Returns:
            Hex digest string uniquely identifying the input combination.
        """
        # MD5 is acceptable for disk caching, not security related
        hasher = hashlib.md5()  # noqa: S324

        for arg in args:
            if isinstance(arg, gpd.GeoDataFrame):
                hasher = self._hash_geodataframe(arg, hasher)
            elif isinstance(arg, np.ndarray):
                hasher.update(arg.tobytes())
            elif isinstance(arg, (float, int)):
                hasher.update(np.array([arg]).tobytes())
            elif isinstance(arg, str):
                hasher.update(arg.encode())
            elif isinstance(arg, bool):
                hasher.update(bytes([arg]))
            else:   # Fallback for any other pickleable type
                hasher.update(pickle.dumps(arg))

        return hasher.hexdigest()

    @staticmethod
    def _hash_geodataframe(gdf: GeoDataFrame, hasher: HASH) -> HASH:
        """Add a GeoDataFrame argument to the hasher.

        Args:
            gdf: GeoDataFrame to add to hash
            hasher: hash object to update

        Returns:
            hasher: Updated hash object
        """
        geom_name = gdf.geometry.name
        geom_wkb = gdf[geom_name].to_wkb().to_numpy()
        geom_wkb_sorted = np.sort(geom_wkb)
        hasher.update(b"".join(geom_wkb_sorted))

        if gdf.crs is not None:
            hasher.update(gdf.crs.to_wkt().encode())

        # --- Non-geometry columns: unchanged semantics ------------------
        for col in sorted(gdf.columns):
            if col == geom_name:
                continue
            arr = gdf[col].to_numpy()
            if arr.dtype == object:
                # Same behaviour: repr per value, concatenated in row order
                hasher.update("".join(repr(v) for v in arr).encode())
            else:
                hasher.update(arr.tobytes())

        return hasher

    def _cache_path(self, key: str, label: str = "cache") -> Path:
        """Constructs the full file path for a cache entry.

        Args:
            key: Hash key identifying the cache entry.
            label: Human-readable label prepended to the filename.

        Returns:
            Full path to the cache file.
        """
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{label}_{key}.pkl"

    def _save_to_cache(self, obj: Any, key: str, label: str = "cache") -> None:
        """Serialises and saves an object to disk.

        Args:
            obj: Any pickleable object to cache.
            key: Hash key identifying the cache entry.
            label: Human-readable label prepended to the filename.
        """
        # TODO: file lock before parallelisation to prevent issues
        path = self._cache_path(key, label)
        logger.debug("Saving to %s", path)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def _load_from_cache(self, key: str, label: str = "cache") -> Any | None:
        """Loads a cached object from disk if it exists.

        Args:
            key: Hash key identifying the cache entry.
            label: Human-readable label prepended to the filename.

        Returns:
            The deserialised object if the cache entry exists, else None.
        """
        path = self._cache_path(key, label)
        logger.debug("Loading from %s", path)
        if not path.exists():
            return None

        try:
            with path.open("rb") as f:
                return pickle.load(f)  # noqa: S301
        except (NotImplementedError, pickle.UnpicklingError, AttributeError, TypeError) as e:
            logger.warning(
                "Cache file '%s' could not be loaded and will be deleted: %s",
                path, e,
            )
            path.unlink()
            return None


    def _get_or_compute(
        self,
        fn: Callable,
        args: tuple,
        hash_args: tuple = (),
        label: str = "cache",
    ) -> Any:
        """Returns a cached result if available, otherwise computes and caches it.

        Args:
            fn: Callable to invoke if no cached result is found.
            args: Tuple of arguments passed both to fn and to the hasher.
            hash_args: Tuple of additional args used to make the hash.
            label: Human-readable label for the cache file.

        Returns:
            The cached or freshly computed result.
        """
        key = self._make_hash(*args, *hash_args)
        result = self._load_from_cache(key, label)

        if result is not None:
            return result

        logger.info(f"Computing {label} ({key})...")
        result = fn(*args)
        self._save_to_cache(result, key, label)
        return result
