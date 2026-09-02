"""Optional FP32 Chebyshev recurrence with an FP64 DFT outer algorithm.

Consumer NVIDIA GPUs execute FP32 much faster than FP64.  The Chebyshev
filter is a particularly suitable mixed-precision boundary: it constructs a
subspace rather than a reported observable, and the resulting vectors are
immediately converted back to FP64 before Rayleigh--Ritz, density formation,
SCF convergence tests, and energy evaluation.

This module deliberately owns only the recurrence work.  It mirrors the
production stencil-major and separable Kleinman--Bylander projector kernels,
using FP32 values and vectors while preserving their sparse traversal and
PARSEC polynomial.  Automatic selection is restricted to large sectors; the
ordinary FP64 kernels stay the authoritative fallback and can be forced.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from .cupy_compile import compile_cupy_raw
from .cupy_projectors import _SOURCE as _PROJECTOR_SOURCE
from .cupy_stencil_major import _CUDA_SOURCE as _STENCIL_SOURCE


def _float_source(source: str, names: tuple[str, ...]) -> str:
    """Return a separately named FP32 version of one CUDA source string."""

    converted = source
    # Use placeholders because ``..._projection`` is a prefix of
    # ``..._projection_serial``; a direct series of replacements would rename
    # part of the already-renamed longer identifier a second time.
    placeholders: list[tuple[str, str]] = []
    for index, name in enumerate(sorted(names, key=len, reverse=True)):
        placeholder = f"PARSEC_FLOAT_KERNEL_{index}"
        converted = converted.replace(name, placeholder)
        placeholders.append((placeholder, f"{name}_float32"))
    for placeholder, replacement in placeholders:
        converted = converted.replace(placeholder, replacement)
    return converted.replace("double", "float")


_SOURCE = _float_source(
    _STENCIL_SOURCE,
    ("stencil_major_spmm6", "stencil_major_chebyshev6"),
) + _float_source(
    _PROJECTOR_SOURCE,
    (
        "sparse_projector_projection",
        "sparse_projector_projection_serial",
        "sparse_projector_scatter",
    ),
)

_KERNEL_NAMES = (
    "stencil_major_chebyshev6_float32",
    "sparse_projector_projection_float32",
    "sparse_projector_projection_serial_float32",
)
_KERNEL_CACHE: dict[int, tuple[Any, Any, Any]] = {}
_KERNEL_LOCK = Lock()


def _kernels(cp: Any) -> tuple[Any, Any, Any]:
    """Compile the invariant FP32 recurrence kernels once per CUDA device."""

    device_id = int(cp.cuda.Device().id)
    with _KERNEL_LOCK:
        kernels = _KERNEL_CACHE.get(device_id)
        if kernels is None:
            module = cp.RawModule(
                code=_SOURCE,
                options=("--std=c++11",),
                name_expressions=_KERNEL_NAMES,
            )
            compile_cupy_raw(module)
            kernels = tuple(module.get_function(name) for name in _KERNEL_NAMES)
            _KERNEL_CACHE[device_id] = kernels
    return kernels


class CuPyMixedPrecisionRecurrence:
    """FP32 stencil/projector recurrence used only inside Chebyshev filters."""

    chunk_width = 6

    def __init__(
        self,
        cp: Any,
        stencil: Any,
        host_projectors: Any,
        projector_signs: np.ndarray,
        effective_potential: np.ndarray,
    ) -> None:
        host = host_projectors.tocsr(copy=True)
        host.sum_duplicates()
        host.sort_indices()
        host_transpose = host.T.tocsr(copy=True)
        host_transpose.sum_duplicates()
        host_transpose.sort_indices()
        if host.indices.dtype != np.int32 or host.indptr.dtype != np.int32:
            raise ValueError("mixed projector factors require int32 host CSR")
        if (
            host_transpose.indices.dtype != np.int32
            or host_transpose.indptr.dtype != np.int32
        ):
            raise ValueError("mixed projector transpose requires int32 host CSR")

        self.cp = cp
        self.shape = tuple(stencil.shape)
        self.slot_count = int(stencil.slot_count)
        # Integer stencil metadata is immutable and safe to share with FP64.
        self.neighbors = stencil.neighbors
        self.coefficient_codes = stencil.coefficient_codes
        self.coefficient_palette = stencil.coefficient_palette.astype(
            cp.float32
        )
        self.effective_potential = cp.asarray(
            effective_potential, dtype=cp.float32
        )

        self.projector_count = int(host.shape[1])
        self.projector_row_offsets = cp.asarray(host.indptr, dtype=cp.int32)
        self.projector_columns = cp.asarray(host.indices, dtype=cp.int32)
        self.projector_values = cp.asarray(host.data, dtype=cp.float32)
        self.transpose_row_offsets = cp.asarray(
            host_transpose.indptr, dtype=cp.int32
        )
        self.transpose_grid_rows = cp.asarray(
            host_transpose.indices, dtype=cp.int32
        )
        self.transpose_values = cp.asarray(
            host_transpose.data, dtype=cp.float32
        )
        self.projector_signs = cp.asarray(projector_signs, dtype=cp.float32)
        row_lengths = np.diff(host_transpose.indptr)
        self.parallel_projection = bool(
            row_lengths.size and int(row_lengths.max(initial=0)) >= 256
        )

        kernels = _kernels(cp)
        self.recurrence_kernel = kernels[0]
        self.projection_kernel = (
            kernels[1] if self.parallel_projection else kernels[2]
        )

    def update_potential(self, effective_potential: Any) -> None:
        """Refresh the FP32 shadow of the current FP64 SCF local field."""

        cp = self.cp
        potential = cp.asarray(effective_potential, dtype=cp.float32)
        if potential.shape != self.effective_potential.shape:
            raise ValueError("effective_potential does not match mixed operator")
        cp.copyto(self.effective_potential, potential)

    def _columns(self, vectors: Any) -> tuple[Any, bool]:
        cp = self.cp
        block = (
            vectors
            if isinstance(vectors, cp.ndarray)
            and vectors.dtype == cp.dtype(cp.float32)
            else cp.asarray(vectors, dtype=cp.float32)
        )
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        if block.ndim != 2 or block.shape[0] != self.shape[0]:
            raise ValueError("vectors do not match the mixed operator")
        return block, was_vector

    def _signed_projector_coefficients(self, vectors: Any):
        cp = self.cp
        width = int(vectors.shape[1])
        output = cp.empty(
            (self.projector_count, width), dtype=cp.float32, order="F"
        )
        if self.projector_count == 0:
            return output
        threads = 128
        pair_count = self.projector_count * width
        itemsize = int(vectors.dtype.itemsize)
        grid = (
            (pair_count,)
            if self.parallel_projection
            else ((pair_count + threads - 1) // threads,)
        )
        self.projection_kernel(
            grid,
            (threads,),
            (
                np.int32(self.projector_count),
                np.int32(width),
                self.transpose_row_offsets,
                self.transpose_grid_rows,
                self.transpose_values,
                self.projector_signs,
                vectors,
                np.int64(vectors.strides[0] // itemsize),
                np.int64(vectors.strides[1] // itemsize),
                output,
                np.int64(output.strides[0] // itemsize),
                np.int64(output.strides[1] // itemsize),
            ),
        )
        return output

    def __call__(
        self,
        current: Any,
        *,
        center: float,
        scale: float,
        sigma_next: float,
        previous: Any | None = None,
        sigma: float = 0.0,
    ):
        """Execute one normalized PARSEC recurrence in FP32."""

        cp = self.cp
        block, was_vector = self._columns(current)
        if previous is None:
            previous_block = block
            add_previous = 0
        else:
            previous_block, previous_was_vector = self._columns(previous)
            if previous_was_vector != was_vector or previous_block.shape != block.shape:
                raise ValueError("previous and current mixed blocks must match")
            add_previous = 1

        coefficients = self._signed_projector_coefficients(block)
        add_nonlocal = int(self.projector_count > 0)
        output = cp.empty(block.shape, dtype=cp.float32, order="F")
        threads = 256
        grid = ((self.shape[0] + threads - 1) // threads,)
        itemsize = int(block.dtype.itemsize)
        for start in range(0, int(block.shape[1]), self.chunk_width):
            stop = min(start + self.chunk_width, int(block.shape[1]))
            source = block[:, start:stop]
            previous_source = previous_block[:, start:stop]
            coefficient_source = coefficients[:, start:stop]
            target = output[:, start:stop]
            self.recurrence_kernel(
                grid,
                (threads,),
                (
                    np.int64(self.shape[0]),
                    np.int32(self.slot_count),
                    self.neighbors,
                    self.coefficient_codes,
                    self.coefficient_palette,
                    self.effective_potential,
                    self.projector_row_offsets,
                    self.projector_columns,
                    self.projector_values,
                    coefficient_source,
                    np.int64(coefficient_source.strides[0] // itemsize),
                    np.int64(coefficient_source.strides[1] // itemsize),
                    np.int32(add_nonlocal),
                    source,
                    np.int64(source.strides[0] // itemsize),
                    np.int64(source.strides[1] // itemsize),
                    previous_source,
                    np.int64(previous_source.strides[0] // itemsize),
                    np.int64(previous_source.strides[1] // itemsize),
                    np.int32(add_previous),
                    np.int32(stop - start),
                    np.float32(center),
                    np.float32(scale),
                    np.float32(sigma),
                    np.float32(sigma_next),
                    target,
                    np.int64(target.strides[0] // itemsize),
                    np.int64(target.strides[1] // itemsize),
                ),
            )
        return output[:, 0] if was_vector else output

__all__ = ["CuPyMixedPrecisionRecurrence"]
