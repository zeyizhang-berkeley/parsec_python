"""A complex CSR matrix on the device with a block product on row-major complex blocks.

The operators of the sectors of a k-point (:mod:`..SCF.kpoint_sectors`) are complex sparse matrices of no
stencil form: the reduced Bloch projectors and, with ``PARSEC_KPOINT_SECTOR_STENCIL=off`` (or kinetic matrices that
are no axial stencil of the grid; else :mod:`.sector_bloch_stencil`), the reduced kinetic term (``-nabla^2 - 2ik.grad
+ |k|^2`` or the Bloch Laplacian, about as many entries per row as the stencil).  The CuPy product of a complex CSR matrix and
a row-major block (cuSPARSE, with the conversions of the block and the result between the two layouts) ran about
forty times below the stencil kernel per row and column on A100 (diamond 7^3, 2 x 2 x 2 mesh).  Here one thread
forms one entry of the result, ``out[r, c] = sum_p M[r, j_p] x[j_p, c]`` (plus ``d[r] x[r, c]`` where a diagonal
is given), with the threads of a warp along the columns of one row: the row's entries are read once per warp and
the block rows it gathers are read along their columns.  Rows of very many entries (the adjoint of the
projectors, one row per projector over its whole support) split their sum over the threads of one block
(:func:`complex_csr_adjoint_kernel`), a fixed order of summation, so the product is the same in every run.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_SOURCE = r"""
#include <cuComplex.h>
extern "C" __global__ void complex_csr_block_product(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double2* __restrict__ values,
    const double2* __restrict__ block, const long long block_stride,
    double2* __restrict__ output, const long long output_stride,
    const double* __restrict__ diagonal, const int with_diagonal, const int accumulate)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    double re = 0.0, im = 0.0;
    const int end = indptr[r + 1];
    for (int p = indptr[r]; p < end; ++p) {
        const double2 a = values[p];
        const double2 x = block[(long long)indices[p] * block_stride + c];
        re += a.x * x.x - a.y * x.y;
        im += a.x * x.y + a.y * x.x;
    }
    if (with_diagonal) {
        const double d = diagonal[r];
        const double2 x = block[r * block_stride + c];
        re += d * x.x;
        im += d * x.y;
    }
    const long long o = r * output_stride + c;
    if (accumulate) {
        output[o].x += re;
        output[o].y += im;
    } else {
        output[o] = make_double2(re, im);
    }
}

// One block of 256 threads per (row, tile of 8 columns): thread (lane, column) = (threadIdx.x % 32,
// threadIdx.x / 32); the 32 lanes of a column stride the row's entries and their partial sums are added in a
// fixed tree (shared memory), the same order in every run.
extern "C" __global__ void complex_csr_long_rows(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double2* __restrict__ values,
    const double2* __restrict__ block, const long long block_stride,
    double2* __restrict__ output, const long long output_stride, const int accumulate)
{
    __shared__ double2 partial[256];
    const long long r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column_in_tile = threadIdx.x >> 5;
    const long long c = (long long)blockIdx.y * 8 + column_in_tile;
    double re = 0.0, im = 0.0;
    if (r < rows && c < count) {
        const int end = indptr[r + 1];
        for (int p = indptr[r] + lane; p < end; p += 32) {
            const double2 a = values[p];
            const double2 x = block[(long long)indices[p] * block_stride + c];
            re += a.x * x.x - a.y * x.y;
            im += a.x * x.y + a.y * x.x;
        }
    }
    partial[threadIdx.x] = make_double2(re, im);
    __syncthreads();
    for (int width = 16; width > 0; width >>= 1) {
        if (lane < width) {
            partial[threadIdx.x].x += partial[threadIdx.x + width].x;
            partial[threadIdx.x].y += partial[threadIdx.x + width].y;
        }
        __syncthreads();
    }
    if (lane == 0 && r < rows && c < count) {
        const long long o = r * output_stride + c;
        if (accumulate) {
            output[o].x += partial[threadIdx.x].x;
            output[o].y += partial[threadIdx.x].y;
        } else {
            output[o] = partial[threadIdx.x];
        }
    }
}
"""

# The same two products for a real matrix and a real block (a complex block goes through its real view, two real
# columns per complex one).
_REAL_SOURCE = r"""
extern "C" __global__ void real_csr_block_product(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double* __restrict__ values,
    const double* __restrict__ block, const long long block_stride,
    double* __restrict__ output, const long long output_stride,
    const double* __restrict__ diagonal, const int with_diagonal, const int accumulate)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    double sum = 0.0;
    const int end = indptr[r + 1];
    for (int p = indptr[r]; p < end; ++p) {
        sum += values[p] * block[(long long)indices[p] * block_stride + c];
    }
    if (with_diagonal) {
        sum += diagonal[r] * block[r * block_stride + c];
    }
    const long long o = r * output_stride + c;
    if (accumulate) {
        output[o] += sum;
    } else {
        output[o] = sum;
    }
}

extern "C" __global__ void real_csr_long_rows(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double* __restrict__ values,
    const double* __restrict__ block, const long long block_stride,
    double* __restrict__ output, const long long output_stride, const int accumulate)
{
    __shared__ double partial[256];
    const long long r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column_in_tile = threadIdx.x >> 5;
    const long long c = (long long)blockIdx.y * 8 + column_in_tile;
    double sum = 0.0;
    if (r < rows && c < count) {
        const int end = indptr[r + 1];
        for (int p = indptr[r] + lane; p < end; p += 32) {
            sum += values[p] * block[(long long)indices[p] * block_stride + c];
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = 16; width > 0; width >>= 1) {
        if (lane < width) {
            partial[threadIdx.x] += partial[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (lane == 0 && r < rows && c < count) {
        const long long o = r * output_stride + c;
        if (accumulate) {
            output[o] += partial[threadIdx.x];
        } else {
            output[o] = partial[threadIdx.x];
        }
    }
}
"""

_KERNELS: dict[tuple[int, str], object] = {}
# rows of more entries than this on average take the kernel that splits a row over 32 threads
_LONG_ROWS = 256


def _kernel(cp, name: str):
    device = int(cp.cuda.Device().id)
    kernel = _KERNELS.get((device, name))
    if kernel is None:
        source = _REAL_SOURCE if name.startswith("real_") else _SOURCE
        kernel = cp.RawKernel(source, name, options=("--std=c++11", "--fmad=false"))
        kernel.compile()
        _KERNELS[(device, name)] = kernel
    return kernel


ORDERED_SUMS_SWITCH = "PARSEC_ORDERED_SUMS"


def ordered_sums_setting() -> str:
    """``PARSEC_ORDERED_SUMS``: ``auto`` (the default since 2026-11-22) gives the sparse matrices of the device
    operators of the k-point, band and spin-orbit solvers (Laplacian and ``k.grad`` where no stencil kernel takes
    them, the Kleinman-Bylander, LDA+U and spin-orbit projectors), of the overlaps of the k-point nonlocal forces, the
    fields of a periodic meta-GGA, the reduction of a hybrid's exchange to a sector, the boundary term of the device
    wire and slab Hartree solve and the spinor backend products that add their terms in a fixed order
    (:class:`OrderedCSR`, :func:`ordered_device_matrix`), and the k-point workers of one process add their density
    parts in the order of the workers: the same input on the same devices gives the same bits in every run.  ``off``: the CuPy (cuSPARSE) products and the order in which the
    workers finish, as before; cuSPARSE's ``csr @ dense`` differs in the last bits from call to call (A100: all 30
    repeats of a projector product, by up to 1e-13), which a slowly converging SCF can amplify into other iteration
    counts or, on a chaotic one, other states."""

    import os

    value = os.environ.get(ORDERED_SUMS_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{ORDERED_SUMS_SWITCH} must be auto or off, not {value!r}")
    return value


class DeviceComplexCSR:
    """A complex128 CSR matrix on the current device (int32 indices) with :meth:`product` on row-major blocks.

    ``data``, ``indices`` and ``indptr`` are the device arrays (what the memory accounting of the k-point pools
    reads); ``dtype`` is complex128.
    """

    dtype = np.complex128

    def __init__(self, cp, matrix) -> None:
        matrix = sp.csr_matrix(matrix, dtype=np.complex128)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz >= 2**31 or max(matrix.shape) >= 2**31:
            raise ValueError("a complex device CSR matrix needs fewer than 2**31 entries and rows")
        self.cp = cp
        self.shape = tuple(int(value) for value in matrix.shape)
        self.indptr = cp.asarray(matrix.indptr.astype(np.int32))
        self.indices = cp.asarray(matrix.indices.astype(np.int32))
        self.data = cp.asarray(np.ascontiguousarray(matrix.data))
        self.long_rows = matrix.nnz > _LONG_ROWS * max(1, self.shape[0])
        self.nbytes = int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    def product(self, block, output=None, *, diagonal=None, accumulate: bool = False):
        """``output (+)= M block (+ diag(diagonal) block)`` for a complex block of shape ``(columns of M, k)``."""

        cp = self.cp
        if block.ndim != 2 or block.shape[0] != self.shape[1]:
            raise ValueError("the block does not match the matrix")
        block = cp.ascontiguousarray(block, dtype=cp.complex128)
        rows, count = self.shape[0], int(block.shape[1])
        if output is None:
            output = cp.empty((rows, count), dtype=cp.complex128)
            accumulate = False
        elif output.shape != (rows, count) or not output.flags.c_contiguous or output.dtype != cp.complex128:
            raise ValueError("the output must be a row-major complex block of the product's shape")
        if rows * count == 0:
            return output
        if diagonal is not None and self.shape[0] != self.shape[1]:
            raise ValueError("a diagonal term needs a square matrix")
        if self.long_rows and diagonal is None:
            tiles = (count + 7) // 8
            _kernel(cp, "complex_csr_long_rows")(
                (rows, tiles),
                (256,),
                (
                    np.int64(rows), np.int64(count), self.indptr, self.indices, self.data,
                    block, np.int64(block.strides[0] // 16),
                    output, np.int64(output.strides[0] // 16), np.int32(bool(accumulate)),
                ),
            )
            return output
        threads = 256
        total = rows * count
        _kernel(cp, "complex_csr_block_product")(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                np.int64(rows), np.int64(count), self.indptr, self.indices, self.data,
                block, np.int64(block.strides[0] // 16),
                output, np.int64(output.strides[0] // 16),
                self.data if diagonal is None else cp.ascontiguousarray(diagonal, dtype=cp.float64),
                np.int32(diagonal is not None), np.int32(bool(accumulate)),
            ),
        )
        return output


class OrderedCSR:
    """A CuPy CSR matrix (real or complex) whose products with dense device blocks add their terms in a fixed order.

    ``M @ X`` (and :meth:`dot`, :meth:`product`) runs the kernels of this module: one thread per entry of the result
    for rows of few entries, a fixed tree over 32 threads per entry for long rows (the adjoint of the projectors), so a
    product gives the same bits in every run, where the cuSPARSE product of CuPy does not.  A real matrix takes a
    complex block through its real view; a complex matrix takes a real block as complex.  The result is a new
    row-major array (a vector for a vector).  Every other attribute (``data``, ``indices``, ``indptr``, ``nnz``,
    ``dtype``, ``shape``, ...) is that of the CuPy matrix, which keeps the only copy of the entries.
    """

    def __init__(self, matrix) -> None:
        import cupy as cp

        # (the order of the stored entries is the order of the sums: sorted or not, the same in every run)
        self.matrix = matrix
        self.cp = cp
        self.shape = tuple(int(value) for value in matrix.shape)
        self.dtype = matrix.dtype
        self.complex = bool(cp.iscomplexobj(matrix.data))
        if max(self.shape) >= 2**31 or int(matrix.nnz) >= 2**31:
            raise ValueError("an ordered device CSR matrix needs fewer than 2**31 entries and rows")
        self.long_rows = int(matrix.nnz) > _LONG_ROWS * max(1, self.shape[0])

    def __getattr__(self, name):
        # (only reached for attributes this class does not define)
        return getattr(self.__dict__["matrix"], name)

    def __matmul__(self, other):
        return self.dot(other)

    def dot(self, other):
        cp = self.cp
        if not isinstance(other, cp.ndarray):
            return self.matrix @ other
        if other.ndim == 1:
            return self.product(other[:, None]).reshape(-1)
        return self.product(other)

    def product(self, block, output=None, *, diagonal=None, accumulate: bool = False):
        """``output (+)= M block (+ diag(diagonal) block)`` for a dense block ``(columns of M, k)``; the signature of
        :meth:`DeviceComplexCSR.product`, for blocks of either kind."""

        cp = self.cp
        if block.ndim != 2 or block.shape[0] != self.shape[1]:
            raise ValueError("the block does not match the matrix")
        if diagonal is not None and self.shape[0] != self.shape[1]:
            raise ValueError("a diagonal term needs a square matrix")
        rows, count = self.shape[0], int(block.shape[1])
        complex_out = self.complex or block.dtype == cp.complex128
        dtype = cp.complex128 if complex_out else cp.float64
        if output is None:
            output = cp.empty((rows, count), dtype=dtype)
            accumulate = False
        elif output.shape != (rows, count) or not output.flags.c_contiguous or output.dtype != dtype:
            raise ValueError("the output must be a row-major block of the product's shape and kind")
        if rows * count == 0:
            return output
        matrix = self.matrix
        if self.complex:
            source = cp.ascontiguousarray(block, dtype=cp.complex128)
            out, width = output, count
            name, unit, values = "complex_csr", 16, cp.ascontiguousarray(matrix.data, dtype=cp.complex128)
        else:
            if block.dtype == cp.complex128:
                # the real view: two real columns per complex one (a diagonal term applies to both)
                source = cp.ascontiguousarray(block).view(cp.float64)
                out, width = output.view(cp.float64), 2 * count
            else:
                source = cp.ascontiguousarray(block, dtype=cp.float64)
                out, width = output, count
            name, unit, values = "real_csr", 8, cp.ascontiguousarray(matrix.data, dtype=cp.float64)
        indptr = cp.ascontiguousarray(matrix.indptr, dtype=cp.int32)
        indices = cp.ascontiguousarray(matrix.indices, dtype=cp.int32)
        diagonal_values = None if diagonal is None else cp.ascontiguousarray(diagonal, dtype=cp.float64)
        if self.long_rows and diagonal_values is None:
            tiles = (width + 7) // 8
            _kernel(cp, f"{name}_long_rows")(
                (rows, tiles),
                (256,),
                (
                    np.int64(rows), np.int64(width), indptr, indices, values,
                    source, np.int64(source.strides[0] // unit),
                    out, np.int64(out.strides[0] // unit), np.int32(bool(accumulate)),
                ),
            )
            return output
        threads = 256
        total = rows * width
        _kernel(cp, f"{name}_block_product")(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                np.int64(rows), np.int64(width), indptr, indices, values,
                source, np.int64(source.strides[0] // unit),
                out, np.int64(out.strides[0] // unit),
                values if diagonal_values is None else diagonal_values,
                np.int32(diagonal_values is not None), np.int32(bool(accumulate)),
            ),
        )
        return output


def ordered_device_matrix(matrix):
    """A CuPy sparse matrix with products in a fixed order under ``PARSEC_ORDERED_SUMS=auto`` (an
    :class:`OrderedCSR`; a CSC matrix becomes CSR first), the matrix as it is under ``off``."""

    if ordered_sums_setting() != "auto":
        return matrix
    if max(int(value) for value in matrix.shape) >= 2**31 or int(matrix.nnz) >= 2**31:
        return matrix  # beyond the int32 indices of the kernels: CuPy's product
    if getattr(matrix, "format", "csr") != "csr":
        matrix = matrix.tocsr()
    return OrderedCSR(matrix)


__all__ = ["DeviceComplexCSR", "OrderedCSR", "ordered_device_matrix", "ordered_sums_setting"]
