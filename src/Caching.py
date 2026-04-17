import logging

logger = logging.getLogger(__name__)

import hashlib
import pickle
import os
from typing import Any, Callable

import geopandas as gpd
import numpy as np


class Caching:
    """Mixin providing general-purpose disk caching for any callable result.

    Classes using this mixin can cache and retrieve any pickleable object
    using a hash-based key derived from arbitrary inputs.

    Attributes:
        cache_dir: Directory in which cached files are stored.
    """

    cache_dir: str = ".cache"

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
        hasher = hashlib.md5()

        for arg in args:
            if isinstance(arg, gpd.GeoDataFrame):
                wkb_sorted = sorted(geom.wkb for geom in arg.geometry)
                for wkb in wkb_sorted:
                    hasher.update(wkb)
                if arg.crs is not None:
                    hasher.update(arg.crs.to_wkt().encode())
                for col in sorted(arg.columns):
                    if col != arg.geometry.name:
                        arr = arg[col].to_numpy()
                        if arr.dtype == object:
                            # Encode via repr for any object array — avoids hashing
                            # memory addresses which change between sessions
                            hasher.update("".join(repr(v) for v in arr).encode())
                        else:
                            # Safe to use raw bytes for concrete numeric dtypes
                            hasher.update(arr.tobytes())

            elif isinstance(arg, np.ndarray):
                hasher.update(arg.tobytes())

            elif isinstance(arg, (float, int)):
                hasher.update(np.array([arg]).tobytes())

            elif isinstance(arg, str):
                hasher.update(arg.encode())

            elif isinstance(arg, bool):
                hasher.update(bytes([arg]))

            else:
                # Fallback for any other pickleable type
                hasher.update(pickle.dumps(arg))

        return hasher.hexdigest()

    def _cache_path(self, key: str, label: str = "cache") -> str:
        """Constructs the full file path for a cache entry.

        Args:
            key: Hash key identifying the cache entry.
            label: Human-readable label prepended to the filename.

        Returns:
            Full path to the cache file.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{label}_{key}.pkl")

    def _save_to_cache(self, obj: Any, key: str, label: str = "cache") -> None:
        """Serialises and saves an object to disk.

        Args:
            obj: Any pickleable object to cache.
            key: Hash key identifying the cache entry.
            label: Human-readable label prepended to the filename.
        """
        path = self._cache_path(key, label)
        with open(path, "wb") as f:
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
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _get_or_compute(
        self,
        fn: Callable,
        args: tuple,
        hash_args: tuple = (),
        label: str = "cache",
        verbose: bool = True,
    ) -> Any:
        """Returns a cached result if available, otherwise computes and caches it.

        Args:
            fn: Callable to invoke if no cached result is found.
            args: Tuple of arguments passed both to fn and to the hasher.
            label: Human-readable label for the cache file.

        Returns:
            The cached or freshly computed result.
        """
        key = self._make_hash(*args, *hash_args)
        result = self._load_from_cache(key, label)

        if result is not None:
            if verbose:
                logger.info(f"Loading cached {label} ({key})...")
            return result

        if verbose:
            logger.info(f"Computing {label} ({key})...")
        result = fn(*args)
        self._save_to_cache(result, key, label)
        return result
