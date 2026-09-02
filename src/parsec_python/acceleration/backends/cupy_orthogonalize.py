"""Small CUDA kernels used by GPU block orthogonalization.

The CHEBDAV workspace is row-major so that its six appended vectors are
adjacent within each grid row.  cuBLAS can project that block against the
complete contiguous workspace efficiently, but applying only the active
prefix otherwise requires zeroing inactive coefficients and multiplying by
all allocated columns.  The kernel below fuses the active-prefix update

``X <- X - Q (Q.T X)``

after cuBLAS has formed the coefficients.  It neither changes the projection
nor approximates any value; it only specializes the very skinny row-major
matrix product and subtraction used by CHEBDAV.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from .cupy_compile import compile_cupy_raw


_CUDA_SOURCE = r"""
extern "C" __global__
void subtract_active_prefix(
    const long long row_count,
    const int active_columns,
    const int block_width,
    const double* __restrict__ basis,
    const long long basis_row_stride,
    const long long basis_column_stride,
    const double* __restrict__ coefficients,
    const long long coefficient_row_stride,
    const long long coefficient_column_stride,
    double* __restrict__ target,
    const long long target_row_stride,
    const long long target_column_stride
) {
    const long long row =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }

    double update[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int prefix = 0; prefix < active_columns; ++prefix) {
        const double basis_value = basis[
            row * basis_row_stride +
            static_cast<long long>(prefix) * basis_column_stride
        ];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < block_width) {
                update[column] += basis_value * coefficients[
                    static_cast<long long>(prefix) * coefficient_row_stride +
                    static_cast<long long>(column) * coefficient_column_stride
                ];
            }
        }
    }
#pragma unroll
    for (int column = 0; column < 6; ++column) {
        if (column < block_width) {
            target[
                row * target_row_stride +
                static_cast<long long>(column) * target_column_stride
            ] -= update[column];
        }
    }
}
"""


_KERNEL_CACHE: dict[int, Any] = {}
_KERNEL_CACHE_LOCK = Lock()


def _compiled_kernel(cp: Any) -> Any:
    device_id = int(cp.cuda.Device().id)
    with _KERNEL_CACHE_LOCK:
        kernel = _KERNEL_CACHE.get(device_id)
        if kernel is None:
            kernel = cp.RawKernel(
                _CUDA_SOURCE,
                "subtract_active_prefix",
                options=("--std=c++11",),
            )
            compile_cupy_raw(kernel)
            _KERNEL_CACHE[device_id] = kernel
    return kernel


def subtract_active_prefix(
    cp: Any,
    basis: Any,
    coefficients: Any,
    target: Any,
    *,
    active_columns: int,
) -> None:
    """Apply one active-prefix update in place using explicit array strides."""

    rows = int(basis.shape[0])
    width = int(target.shape[1])
    active = int(active_columns)
    if not 0 < active <= int(basis.shape[1]):
        raise ValueError("active_columns is outside the Davidson workspace")
    if coefficients.shape[0] < active or coefficients.shape[1] != width:
        raise ValueError("projection coefficients have an invalid shape")
    itemsize = int(basis.dtype.itemsize)
    threads = 256
    _compiled_kernel(cp)(
        ((rows + threads - 1) // threads,),
        (threads,),
        (
            np.int64(rows),
            np.int32(active),
            np.int32(width),
            basis,
            np.int64(basis.strides[0] // itemsize),
            np.int64(basis.strides[1] // itemsize),
            coefficients,
            np.int64(coefficients.strides[0] // itemsize),
            np.int64(coefficients.strides[1] // itemsize),
            target,
            np.int64(target.strides[0] // itemsize),
            np.int64(target.strides[1] // itemsize),
        ),
    )


__all__ = ["subtract_active_prefix"]
