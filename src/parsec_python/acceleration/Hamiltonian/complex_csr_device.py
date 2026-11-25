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

One block per row leaves most of the device idle where a few rows hold all the entries (a small cell: 8 projector
rows of 2 carbon atoms, two blocks of 256 threads each for a block of 9 columns), and a row's sum then takes as long
as its support is large (the projectors filtered by ``PARSEC_LOCAL_FILTER=all`` end at 1.5 times the file's support
radius, 3.4 times the entries).  Under ``PARSEC_LONG_ROW_SPLIT=auto`` (:func:`long_row_split_setting`) a row of more
than :data:`LONG_ROW_CHUNK` entries is summed in chunks of that many entries, one block per chunk, each in the order
of the kernel above, and a second kernel adds the partial sums of a row in the order of its chunks: still a fixed
order (the same bits in every run), another one than a single block's for such rows (the last bits of their sums
differ).  A matrix whose rows all fit one chunk keeps the single-block kernel and its bits.
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

// PARSEC_LONG_ROW_SPLIT=auto: complex_csr_long_rows over the entries [chunk_begin[k], chunk_end[k]) of one chunk of a
// row per block (the same lanes, the same tree), the partial sum of chunk k and column c to partials[k * count + c].
extern "C" __global__ void complex_csr_row_chunks(
    const long long chunks, const long long count,
    const int* __restrict__ chunk_begin, const int* __restrict__ chunk_end,
    const int* __restrict__ indices, const double2* __restrict__ values,
    const double2* __restrict__ block, const long long block_stride, double2* __restrict__ partials)
{
    __shared__ double2 partial[256];
    const long long k = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column_in_tile = threadIdx.x >> 5;
    const long long c = (long long)blockIdx.y * 8 + column_in_tile;
    double re = 0.0, im = 0.0;
    if (k < chunks && c < count) {
        const int end = chunk_end[k];
        for (int p = chunk_begin[k] + lane; p < end; p += 32) {
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
    if (lane == 0 && k < chunks && c < count) {
        partials[k * count + c] = partial[threadIdx.x];
    }
}

// The partial sums of the chunks [row_chunks[r], row_chunks[r + 1]) of row r added in the order of the chunks.
extern "C" __global__ void complex_csr_chunk_sums(
    const long long rows, const long long count, const int* __restrict__ row_chunks,
    const double2* __restrict__ partials, double2* __restrict__ output, const long long output_stride,
    const int accumulate)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    const int first = row_chunks[r];
    const int last = row_chunks[r + 1];
    double2 sum = partials[(long long)first * count + c];
    for (int k = first + 1; k < last; ++k) {
        const double2 x = partials[(long long)k * count + c];
        sum.x += x.x;
        sum.y += x.y;
    }
    const long long o = r * output_stride + c;
    if (accumulate) {
        output[o].x += sum.x;
        output[o].y += sum.y;
    } else {
        output[o] = sum;
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

extern "C" __global__ void real_csr_row_chunks(
    const long long chunks, const long long count,
    const int* __restrict__ chunk_begin, const int* __restrict__ chunk_end,
    const int* __restrict__ indices, const double* __restrict__ values,
    const double* __restrict__ block, const long long block_stride, double* __restrict__ partials)
{
    __shared__ double partial[256];
    const long long k = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int column_in_tile = threadIdx.x >> 5;
    const long long c = (long long)blockIdx.y * 8 + column_in_tile;
    double sum = 0.0;
    if (k < chunks && c < count) {
        const int end = chunk_end[k];
        for (int p = chunk_begin[k] + lane; p < end; p += 32) {
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
    if (lane == 0 && k < chunks && c < count) {
        partials[k * count + c] = partial[threadIdx.x];
    }
}

extern "C" __global__ void real_csr_chunk_sums(
    const long long rows, const long long count, const int* __restrict__ row_chunks,
    const double* __restrict__ partials, double* __restrict__ output, const long long output_stride,
    const int accumulate)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    const int first = row_chunks[r];
    const int last = row_chunks[r + 1];
    double sum = partials[(long long)first * count + c];
    for (int k = first + 1; k < last; ++k) {
        sum += partials[(long long)k * count + c];
    }
    const long long o = r * output_stride + c;
    if (accumulate) {
        output[o] += sum;
    } else {
        output[o] = sum;
    }
}
"""

_KERNELS: dict[tuple[int, str], object] = {}
# rows of more entries than this on average take the kernel that splits a row over 32 threads
_LONG_ROWS = 256
#: ``PARSEC_LONG_ROW_SPLIT=auto``: a long row of more entries than this is summed in chunks of this many entries, one
#: block of threads per chunk (32 entries per lane and pass, 32 passes).
LONG_ROW_CHUNK = 1024
LONG_ROW_SPLIT_SWITCH = "PARSEC_LONG_ROW_SPLIT"


def long_row_split_setting() -> str:
    """``PARSEC_LONG_ROW_SPLIT``: ``auto`` (the default) sums a row of a long-row product (the adjoint of the
    Kleinman-Bylander projectors, one row per projector over its support) of more than :data:`LONG_ROW_CHUNK` entries
    in chunks, one block of threads per chunk, and adds the chunks in their order (the module docstring); ``off``: one
    block per row, the order and the bits of before.  An unknown value is an error."""

    import os

    value = os.environ.get(LONG_ROW_SPLIT_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{LONG_ROW_SPLIT_SWITCH} must be auto or off, not {value!r}")
    return value


def row_chunks(indptr: np.ndarray, chunk: int = LONG_ROW_CHUNK):
    """``(chunk_begin, chunk_end, row_chunks)`` of the rows of a CSR ``indptr`` (host int arrays): every row in
    consecutive chunks of ``chunk`` entries (the last one shorter, an empty row one empty chunk); the chunks of row
    ``r`` are ``row_chunks[r]:row_chunks[r + 1]``.  ``None`` where no row has more than ``chunk`` entries (the
    single-block kernel then sums every row in the same order)."""

    indptr = np.asarray(indptr, dtype=np.int64)
    lengths = np.diff(indptr)
    if lengths.size == 0 or int(lengths.max()) <= chunk:
        return None
    counts = np.maximum(1, -(-lengths // chunk))
    row_starts = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(counts, out=row_starts[1:])
    owner = np.repeat(np.arange(lengths.size), counts)
    position = np.arange(int(row_starts[-1])) - row_starts[owner]
    begin = indptr[owner] + position * chunk
    end = np.minimum(begin + chunk, indptr[owner + 1])
    return begin.astype(np.int32), end.astype(np.int32), row_starts.astype(np.int32)


def _split_long_rows(cp, name: str, chunks, rows: int, width: int, indices, values, source, unit: int, out,
                     accumulate: bool) -> None:
    """The two kernels of ``PARSEC_LONG_ROW_SPLIT=auto`` (``name`` ``complex_csr`` or ``real_csr``) for the device
    chunk arrays ``chunks`` (:func:`row_chunks`)."""

    begin, end, starts = chunks
    count = int(begin.shape[0])
    partials = cp.empty((count, width), dtype=cp.complex128 if name == "complex_csr" else cp.float64)
    _kernel(cp, f"{name}_row_chunks")(
        (count, (width + 7) // 8),
        (256,),
        (
            np.int64(count), np.int64(width), begin, end, indices, values,
            source, np.int64(source.strides[0] // unit), partials,
        ),
    )
    threads = 256
    total = rows * width
    _kernel(cp, f"{name}_chunk_sums")(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            np.int64(rows), np.int64(width), starts, partials,
            out, np.int64(out.strides[0] // unit), np.int32(bool(accumulate)),
        ),
    )


def _device_chunks(cp, indptr):
    """:func:`row_chunks` of a device or host ``indptr`` as device arrays, or ``None``."""

    host = cp.asnumpy(indptr) if isinstance(indptr, cp.ndarray) else np.asarray(indptr)
    found = row_chunks(host)
    return None if found is None else tuple(cp.asarray(array) for array in found)


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
        # PARSEC_LONG_ROW_SPLIT=auto: the chunks of the rows longer than LONG_ROW_CHUNK (None: one block per row)
        self.chunks = (
            _device_chunks(cp, matrix.indptr) if self.long_rows and long_row_split_setting() == "auto" else None
        )
        self.nbytes = int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)
        if self.chunks is not None:
            self.nbytes += int(sum(array.nbytes for array in self.chunks))

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
        if self.long_rows and diagonal is None and self.chunks is not None:
            _split_long_rows(cp, "complex_csr", self.chunks, rows, count, self.indices, self.data, block, 16, output,
                             accumulate)
            return output
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
        # PARSEC_LONG_ROW_SPLIT (read once per matrix): the chunks of the rows longer than LONG_ROW_CHUNK, made on the
        # first long-row product from the indptr it reads (None: one block per row)
        self.split = self.long_rows and long_row_split_setting() == "auto"
        self._chunks = None
        self._chunked_indptr = None

    def _row_chunks(self, indptr):
        """The device chunks of ``indptr`` (the matrix's, as the product reads it), made again if it changed."""

        if self._chunked_indptr is not indptr:
            self._chunks = _device_chunks(self.cp, indptr)
            self._chunked_indptr = indptr
        return self._chunks

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
        if self.long_rows and diagonal_values is None and self.split:
            chunks = self._row_chunks(matrix.indptr)
            if chunks is not None:
                _split_long_rows(cp, name, chunks, rows, width, indices, values, source, unit, out, accumulate)
                return output
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


__all__ = [
    "DeviceComplexCSR",
    "LONG_ROW_CHUNK",
    "OrderedCSR",
    "long_row_split_setting",
    "ordered_device_matrix",
    "ordered_sums_setting",
    "row_chunks",
]
