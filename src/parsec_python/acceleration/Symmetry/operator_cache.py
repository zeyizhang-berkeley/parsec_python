"""Exact persistent cache for reflection-reduced static operators.

The cached objects are only basis transformations of already validated
float64 operators.  A SHA-256 key covers every canonical sparse buffer and
every representation mapping/phase, so changing the grid, geometry,
pseudopotential projectors, finite-difference coefficients, or symmetry
invalidates the entry automatically.  SCF fields are never cached.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
import zipfile

import numpy as np
import scipy.sparse as sp

from parsec_python.V_ion import NonlocalProjectorOperator

from ..backends.cupy_stencil_major import (
    StencilMajorHostMetadata,
    build_stencil_major_metadata,
)

from .representations import ReflectionRepresentationDecomposition


_CACHE_FORMAT = 4
_MEMORY_BUNDLES: OrderedDict[str, "ReducedOperatorBundle"] = OrderedDict()
_MEMORY_BUNDLES_LOCK = Lock()


@dataclass(frozen=True)
class ReducedOperatorCacheInfo:
    """Provenance and setup timing for one reduced-operator bundle."""

    enabled: bool
    status: str
    key: str
    path: Path | None
    hash_seconds: float
    load_seconds: float
    build_seconds: float
    write_seconds: float


@dataclass(frozen=True)
class ReducedOperatorBundle:
    """GPU-ready finite-difference metadata and reduced KB factors."""

    stencil_metadata: tuple[StencilMajorHostMetadata, ...]
    nonlocal_operators: tuple[NonlocalProjectorOperator, ...]
    cache_info: ReducedOperatorCacheInfo


def _resident_memory_cache_size() -> int:
    if os.environ.get("PARSEC_ACCELERATED_RESIDENT", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return 0
    raw = os.environ.get("PARSEC_RESIDENT_OPERATOR_CACHE_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE must be an integer"
        ) from error
    if value < 0:
        raise ValueError("PARSEC_RESIDENT_OPERATOR_CACHE_SIZE cannot be negative")
    return value


def _remember_bundle(memory_key: str, bundle: ReducedOperatorBundle, capacity: int) -> None:
    if capacity < 1:
        return
    with _MEMORY_BUNDLES_LOCK:
        _MEMORY_BUNDLES[memory_key] = bundle
        _MEMORY_BUNDLES.move_to_end(memory_key)
        while len(_MEMORY_BUNDLES) > capacity:
            _MEMORY_BUNDLES.popitem(last=False)


def _canonical_kinetic(negative_laplacian: object) -> sp.csr_matrix:
    from ..Laplacian import materialize_negative_laplacian

    kinetic = materialize_negative_laplacian(negative_laplacian)
    # Native and reference builders already return canonical CSR.  Preserve
    # their buffers instead of making a several-hundred-megabyte validation
    # copy on every startup; copy only a genuinely noncanonical caller input.
    if not kinetic.has_canonical_format or not kinetic.has_sorted_indices:
        kinetic = kinetic.copy()
        kinetic.sum_duplicates()
        kinetic.sort_indices()
    return kinetic


def _canonical_projectors(
    nonlocal_operator: NonlocalProjectorOperator,
) -> sp.csc_matrix:
    projectors = sp.csc_matrix(
        nonlocal_operator.projectors, dtype=np.float64, copy=False
    )
    if not projectors.has_canonical_format or not projectors.has_sorted_indices:
        projectors = projectors.copy()
        projectors.sum_duplicates()
        projectors.sort_indices()
    return projectors


def _hash_array(digest: "hashlib._Hash", values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))


def _cache_key(
    decomposition: ReflectionRepresentationDecomposition,
    kinetic: sp.csr_matrix | None,
    projectors: sp.csc_matrix,
    signs: np.ndarray,
    *,
    kinetic_key_seed: str | None,
    decomposition_key_seed: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"parsec-reflection-operators-v{_CACHE_FORMAT}".encode("ascii"))
    if kinetic_key_seed is None:
        if kinetic is None:
            raise ValueError("kinetic matrix is required without an exact key seed")
        for values in (kinetic.indptr, kinetic.indices, kinetic.data):
            _hash_array(digest, np.asarray(values))
    else:
        digest.update(b"exact-kinetic-key\0")
        digest.update(kinetic_key_seed.encode("ascii"))
    for values in (
        projectors.indptr,
        projectors.indices,
        projectors.data,
        np.asarray(signs, dtype=np.float64),
    ):
        _hash_array(digest, np.asarray(values))
    if decomposition_key_seed is None:
        for values in (
            decomposition.reduction.representative_rows,
            decomposition.reduction.full_to_wedge,
            decomposition.reduction.signs,
            decomposition.characters,
            decomposition.phases,
            decomposition.orbit_to_sector,
        ):
            _hash_array(digest, np.asarray(values))
    else:
        digest.update(b"exact-decomposition-key\0")
        digest.update(decomposition_key_seed.encode("ascii"))
    return digest.hexdigest()


def _load_bundle(
    path: Path,
    decomposition: ReflectionRepresentationDecomposition,
    nonlocal_operator: NonlocalProjectorOperator,
) -> tuple[
    tuple[StencilMajorHostMetadata, ...],
    tuple[NonlocalProjectorOperator, ...],
]:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["cache_format"][0]) != _CACHE_FORMAT:
            raise ValueError("unsupported reduced-operator cache format")
        count = int(archive["representation_count"][0])
        if count != decomposition.representation_count:
            raise ValueError("cached representation count does not match")
        sector_sizes = tuple(
            int(value) for value in archive["sector_sizes"]
        )
        if sector_sizes != decomposition.sector_sizes:
            raise ValueError("cached sector dimensions do not match")
        neighbors_are_shared = bool(archive["neighbors_are_shared"][0])
        shared_neighbors = (
            np.asarray(archive["shared_neighbors"], dtype=np.int32)
            if neighbors_are_shared
            else None
        )

        metadata: list[StencilMajorHostMetadata] = []
        nonlocals: list[NonlocalProjectorOperator] = []
        for index in range(count):
            stencil = StencilMajorHostMetadata(
                shape=tuple(
                    int(value) for value in archive[f"s{index}_shape"]
                ),
                neighbors=np.asarray(
                    (
                        shared_neighbors
                        if shared_neighbors is not None
                        else archive[f"s{index}_neighbors"]
                    ),
                    dtype=np.int32,
                ),
                coefficient_codes=np.asarray(
                    archive[f"s{index}_codes"], dtype=np.uint8
                ),
                coefficient_palette=np.asarray(
                    archive[f"s{index}_palette"], dtype=np.float64
                ),
            )
            sector_size = decomposition.sector_size(index)
            if stencil.shape != (sector_size, sector_size):
                raise ValueError("cached stencil dimension does not match")
            projector_shape = tuple(
                int(value) for value in archive[f"b{index}_shape"]
            )
            if projector_shape[0] != sector_size:
                raise ValueError("cached projector dimension does not match")
            projector = sp.csc_matrix(
                (
                    np.asarray(archive[f"b{index}_data"], dtype=np.float64),
                    np.asarray(archive[f"b{index}_indices"]),
                    np.asarray(archive[f"b{index}_indptr"]),
                ),
                shape=projector_shape,
            )
            metadata.append(stencil)
            nonlocals.append(
                NonlocalProjectorOperator(
                    projectors=projector,
                    signs=np.asarray(
                        nonlocal_operator.signs, dtype=np.float64
                    ).copy(),
                    labels=tuple(nonlocal_operator.labels),
                )
            )
    return tuple(metadata), tuple(nonlocals)


def _write_bundle(
    path: Path,
    metadata: tuple[StencilMajorHostMetadata, ...],
    nonlocals: tuple[NonlocalProjectorOperator, ...],
    sector_sizes: tuple[int, ...],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "cache_format": np.asarray((_CACHE_FORMAT,), dtype=np.int64),
        "representation_count": np.asarray((len(metadata),), dtype=np.int64),
        "sector_sizes": np.asarray(sector_sizes, dtype=np.int64),
    }
    neighbors_are_shared = bool(metadata) and all(
        np.array_equal(metadata[0].neighbors, item.neighbors)
        for item in metadata[1:]
    )
    arrays["neighbors_are_shared"] = np.asarray(
        (neighbors_are_shared,), dtype=np.uint8
    )
    if neighbors_are_shared:
        arrays["shared_neighbors"] = metadata[0].neighbors
    for index, (stencil, nonlocal_operator) in enumerate(
        zip(metadata, nonlocals, strict=True)
    ):
        projector = nonlocal_operator.projectors.tocsc(copy=False)
        arrays.update(
            {
                f"s{index}_shape": np.asarray(stencil.shape, dtype=np.int64),
                f"s{index}_codes": stencil.coefficient_codes,
                f"s{index}_palette": stencil.coefficient_palette,
                f"b{index}_data": projector.data,
                f"b{index}_indices": projector.indices,
                f"b{index}_indptr": projector.indptr,
                f"b{index}_shape": np.asarray(projector.shape, dtype=np.int64),
            }
        )
        if not neighbors_are_shared:
            arrays[f"s{index}_neighbors"] = stencil.neighbors
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.tmp.npz"
    )
    try:
        # Uncompressed NPZ is intentional.  These arrays are consumed once at
        # startup and SSD bandwidth is more valuable than a smaller cache.
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def load_or_build_reduced_operators(
    decomposition: ReflectionRepresentationDecomposition,
    negative_laplacian: object,
    nonlocal_operator: NonlocalProjectorOperator,
    *,
    cache_directory: Path | None,
    kinetic_key_seed: str | None = None,
    decomposition_key_seed: str | None = None,
) -> ReducedOperatorBundle:
    """Load an exact matching static bundle or build it in one sparse pass.

    Exact upstream content keys may replace re-hashing the full kinetic and
    symmetry buffers.  On a cache hit this also permits a deferred full-grid
    Laplacian to remain unmaterialized.  Omitting either seed retains the
    original byte-for-byte hashing path for independent modular callers.
    """

    hash_started = perf_counter()
    projectors = _canonical_projectors(nonlocal_operator)
    kinetic = (
        _canonical_kinetic(negative_laplacian)
        if kinetic_key_seed is None
        else None
    )
    key = _cache_key(
        decomposition,
        kinetic,
        projectors,
        np.asarray(nonlocal_operator.signs, dtype=np.float64),
        kinetic_key_seed=kinetic_key_seed,
        decomposition_key_seed=decomposition_key_seed,
    )
    hash_seconds = perf_counter() - hash_started
    path = (
        None
        if cache_directory is None
        else Path(cache_directory) / f"reflection-v{_CACHE_FORMAT}-{key}.npz"
    )

    memory_capacity = _resident_memory_cache_size()
    memory_key = (
        f"{path.resolve()}::{key}" if path is not None else f"disabled::{key}"
    )
    memory_started = perf_counter()
    if memory_capacity:
        with _MEMORY_BUNDLES_LOCK:
            remembered = _MEMORY_BUNDLES.get(memory_key)
            if remembered is not None:
                _MEMORY_BUNDLES.move_to_end(memory_key)
        if remembered is not None:
            return ReducedOperatorBundle(
                stencil_metadata=remembered.stencil_metadata,
                nonlocal_operators=remembered.nonlocal_operators,
                cache_info=ReducedOperatorCacheInfo(
                    enabled=path is not None,
                    status="memory-hit",
                    key=key,
                    path=path,
                    hash_seconds=hash_seconds,
                    load_seconds=perf_counter() - memory_started,
                    build_seconds=0.0,
                    write_seconds=0.0,
                ),
            )

    load_seconds = 0.0
    status = "disabled"
    if path is not None and path.is_file():
        load_started = perf_counter()
        try:
            metadata, nonlocals = _load_bundle(
                path, decomposition, nonlocal_operator
            )
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            EOFError,
            zipfile.BadZipFile,
        ):
            status = "invalid-rebuilt"
        else:
            load_seconds = perf_counter() - load_started
            bundle = ReducedOperatorBundle(
                stencil_metadata=metadata,
                nonlocal_operators=nonlocals,
                cache_info=ReducedOperatorCacheInfo(
                    enabled=True,
                    status="hit",
                    key=key,
                    path=path,
                    hash_seconds=hash_seconds,
                    load_seconds=load_seconds,
                    build_seconds=0.0,
                    write_seconds=0.0,
                ),
            )
            _remember_bundle(memory_key, bundle, memory_capacity)
            return bundle
        load_seconds = perf_counter() - load_started

    build_started = perf_counter()
    if kinetic is None:
        kinetic = _canonical_kinetic(negative_laplacian)
    laplacians = decomposition.reduce_operators(kinetic)
    metadata = tuple(
        build_stencil_major_metadata(laplacian)
        for laplacian in laplacians
    )
    canonical_nonlocal = NonlocalProjectorOperator(
        projectors=projectors,
        signs=np.asarray(nonlocal_operator.signs, dtype=np.float64).copy(),
        labels=tuple(nonlocal_operator.labels),
    )
    nonlocals = decomposition.reduce_nonlocal_operators(canonical_nonlocal)
    build_seconds = perf_counter() - build_started

    write_seconds = 0.0
    if path is not None:
        write_started = perf_counter()
        try:
            _write_bundle(path, metadata, nonlocals, decomposition.sector_sizes)
        except OSError:
            status = "write-failed"
        else:
            status = "miss-written" if status == "disabled" else status
        write_seconds = perf_counter() - write_started

    bundle = ReducedOperatorBundle(
        stencil_metadata=metadata,
        nonlocal_operators=nonlocals,
        cache_info=ReducedOperatorCacheInfo(
            enabled=path is not None,
            status=status,
            key=key,
            path=path,
            hash_seconds=hash_seconds,
            load_seconds=load_seconds,
            build_seconds=build_seconds,
            write_seconds=write_seconds,
        ),
    )
    _remember_bundle(memory_key, bundle, memory_capacity)
    return bundle


__all__ = [
    "ReducedOperatorCacheInfo",
    "ReducedOperatorBundle",
    "load_or_build_reduced_operators",
]
