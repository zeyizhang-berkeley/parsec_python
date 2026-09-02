"""Defer the full-grid finite-difference CSR when symmetry caches suffice.

The finite-difference operator is determined completely by the active integer
grid, its lookup table, the stencil order, and the grid spacing.  A SHA-256
fingerprint of those exact inputs can therefore validate a cached reduced
operator without first allocating the much larger full-grid CSR matrix.

This object is deliberately narrow: it exposes only ``shape`` and ``nnz`` for
setup reporting, an exact ``cache_key`` for downstream content-addressed
caches, and :meth:`materialize` for every cache-miss or full-grid fallback.
Materialization still calls the validated C++ builder and checks its shape and
nonzero count, so deferral changes setup order rather than the operator.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid


_KEY_FORMAT = 1
_NNZ_CACHE_FORMAT = 1
_MEMORY_NNZ_CACHE: dict[tuple[str, str], int] = {}


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def _operator_key(grid: RealSpaceGrid) -> str:
    """Hash every discrete input used by the native stencil builder."""

    digest = sha256()
    digest.update(f"native-negative-laplacian-v{_KEY_FORMAT}".encode("ascii"))
    digest.update(np.int64(grid.settings.expansion_order).tobytes())
    digest.update(np.float64(grid.spacing).tobytes())
    _hash_array(digest, "integer_coordinates", grid.integer_coordinates)
    _hash_array(digest, "index_min", grid.index_min)
    _hash_array(digest, "lookup", grid.lookup)
    return digest.hexdigest()


def _operator_nnz(grid: RealSpaceGrid) -> int:
    """Count the exact centered and in-domain axial stencil entries.

    This vectorized count is far cheaper than allocating CSR indices and
    float64 coefficients.  It follows the same signed-shell enumeration as
    ``build_negative_laplacian_buffers`` and is checked again if the matrix is
    eventually materialized.
    """

    coordinates = np.asarray(grid.integer_coordinates, dtype=np.int64)
    local = coordinates - np.asarray(grid.index_min, dtype=np.int64)
    lookup = np.asarray(grid.lookup, dtype=np.int64)
    flat_lookup = lookup.reshape(-1)
    shape = np.asarray(lookup.shape, dtype=np.int64)
    strides = np.asarray(
        (int(shape[1] * shape[2]), int(shape[2]), 1), dtype=np.int64
    )
    base_offsets = (
        local[:, 0] * strides[0]
        + local[:, 1] * strides[1]
        + local[:, 2]
    )
    count = int(coordinates.shape[0])  # centered coefficient in every row
    width = int(grid.settings.expansion_order) // 2
    for axis in range(3):
        axis_local = local[:, axis]
        axis_stride = int(strides[axis])
        axis_extent = int(shape[axis])
        for signed_shell in range(-width, width + 1):
            if signed_shell == 0:
                continue
            displaced = axis_local + signed_shell
            valid = (displaced >= 0) & (displaced < axis_extent)
            if not np.any(valid):
                continue
            offsets = base_offsets[valid] + signed_shell * axis_stride
            count += int(np.count_nonzero(flat_lookup[offsets] >= 0))
    return count


def _nnz_cache_path(
    cache_directory: os.PathLike[str] | str | None,
    cache_key: str,
) -> Path | None:
    if cache_directory is None:
        return None
    return (
        Path(cache_directory)
        / f"negative-laplacian-nnz-v{_NNZ_CACHE_FORMAT}-{cache_key}.json"
    )


def _validated_cached_nnz(
    payload: Any,
    cache_key: str,
    shape: tuple[int, int],
    maximum_nnz: int,
) -> int:
    """Validate a tiny reporting cache before accepting its integer value."""

    if not isinstance(payload, dict):
        raise ValueError("NNZ cache payload is not an object")
    if int(payload.get("format", -1)) != _NNZ_CACHE_FORMAT:
        raise ValueError("NNZ cache format does not match")
    if payload.get("cache_key") != cache_key:
        raise ValueError("NNZ cache key does not match")
    if payload.get("shape") != list(shape):
        raise ValueError("NNZ cache shape does not match")
    value = payload.get("nnz")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("NNZ cache value is not an integer")
    if not shape[0] <= value <= maximum_nnz:
        raise ValueError("NNZ cache value lies outside stencil bounds")
    return value


def _load_or_count_nnz(
    grid: RealSpaceGrid,
    cache_key: str,
    shape: tuple[int, int],
    cache_directory: os.PathLike[str] | str | None,
) -> tuple[int, str, Path | None]:
    """Resolve exact NNZ metadata without rescanning a known full grid."""

    path = _nnz_cache_path(cache_directory, cache_key)
    if path is None:
        return _operator_nnz(grid), "disabled", None

    memory_key = (str(path.resolve()), cache_key)
    remembered = _MEMORY_NNZ_CACHE.get(memory_key)
    if remembered is not None:
        return remembered, "memory-hit", path

    maximum_nnz = shape[0] * (1 + 3 * int(grid.settings.expansion_order))
    invalid_cache = False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = _validated_cached_nnz(
            payload,
            cache_key,
            shape,
            maximum_nnz,
        )
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        invalid_cache = True
    else:
        _MEMORY_NNZ_CACHE[memory_key] = value
        return value, "disk-hit", path

    value = _operator_nnz(grid)
    payload = {
        "format": _NNZ_CACHE_FORMAT,
        "cache_key": cache_key,
        "shape": list(shape),
        "nnz": value,
    }
    status = "invalid-rebuilt" if invalid_cache else "miss-written"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        status = "miss-unwritten"
    _MEMORY_NNZ_CACHE[memory_key] = value
    return value, status, path


class DeferredNativeNegativeLaplacian:
    """Exact lazy proxy for one native full-grid ``-nabla_FD^2`` matrix."""

    def __init__(
        self,
        grid: RealSpaceGrid,
        *,
        cache_directory: os.PathLike[str] | str | None = None,
    ) -> None:
        started = perf_counter()
        self.grid = grid
        self.shape = (int(grid.size), int(grid.size))
        self.cache_key = _operator_key(grid)
        self.hash_seconds = perf_counter() - started
        started = perf_counter()
        self.nnz, self.nnz_cache_status, self.nnz_cache_path = (
            _load_or_count_nnz(
                grid,
                self.cache_key,
                self.shape,
                cache_directory,
            )
        )
        self.nnz_count_seconds = perf_counter() - started
        self.materialization_seconds = 0.0
        self._matrix: sp.csr_matrix | None = None

    @property
    def materialized(self) -> bool:
        return self._matrix is not None

    def materialize(self) -> sp.csr_matrix:
        """Build and memoize the validated native CSR on first demand."""

        if self._matrix is None:
            # Local import avoids importing the optional extension merely to
            # construct an exact cache key.
            from ..backends.native import build_native_negative_laplacian

            started = perf_counter()
            matrix = build_native_negative_laplacian(self.grid)
            self.materialization_seconds += perf_counter() - started
            if matrix.shape != self.shape or int(matrix.nnz) != self.nnz:
                raise RuntimeError(
                    "materialized finite-difference operator does not match "
                    "its exact deferred descriptor"
                )
            self._matrix = matrix
        return self._matrix

    def __repr__(self) -> str:
        state = "materialized" if self.materialized else "deferred"
        return (
            f"DeferredNativeNegativeLaplacian(shape={self.shape}, "
            f"nnz={self.nnz}, state={state!r})"
        )


def materialize_negative_laplacian(operator: Any) -> sp.csr_matrix:
    """Return a canonical CSR from either a matrix or the lazy proxy."""

    if isinstance(operator, DeferredNativeNegativeLaplacian):
        return operator.materialize()
    matrix = sp.csr_matrix(operator, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


__all__ = [
    "DeferredNativeNegativeLaplacian",
    "materialize_negative_laplacian",
]
