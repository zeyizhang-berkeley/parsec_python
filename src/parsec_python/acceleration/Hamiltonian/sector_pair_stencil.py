"""The kinetic term of a sector of a k-point in its real form (``PARSEC_KPOINT_SECTOR_REAL``): the gathered stencil
of :mod:`.sector_bloch_stencil` on the real coordinates of :class:`..SCF.kpoint_real_forms.RealForm`.

A real block ``x`` holds the coordinates of the sector vectors ``v = T x``: ``v_w = (x_w + i x_pi(w)) / sqrt(2)``
and ``v_pi(w) = s_w (x_w - i x_pi(w)) / sqrt(2)`` for a pair ``w < pi(w)``, ``v_w = x_w`` or ``i x_w`` for a fixed
row.  ``(T^H (K + v) T x)`` takes the complex row ``y_w`` of ``(K + v) v`` (the slots and weights of the sector's
:class:`.sector_bloch_stencil.SectorStencilTable`, each neighbour rebuilt from its pair) at the first row of every
pair and every fixed row: ``sqrt(2) Re y_w`` and ``sqrt(2) Im y_w`` go to rows ``w`` and ``pi(w)``, ``Re y_w`` or
``Im y_w`` to a fixed row (``A`` keeps ``y``, so ``y_pi(w) = s_w conj(y_w)`` holds the same numbers).  One complex row
per pair: half the rows of the complex sector, and half its bytes (but as many memory transactions in the gathers: the
two coordinates of a neighbour sit on two rows).  The kernel takes the step of the Chebyshev
recurrence as the complex one does (``out = (((K + v) x + added - c x) a - b x_previous) d``).
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Any

import numpy as np

from .sector_bloch_stencil import _PALETTE, _THREADS

_ROOT_TWO = math.sqrt(2.0)


class PairStencilTable:
    """A sector's :class:`.sector_bloch_stencil.SectorStencilTable` with the tables of its real form: per row its
    kind (0 the first of a pair, 1 the second, 2 and 3 a fixed row of sign ``+1``, ``-1``), the other row of its pair,
    the sign of the pair, and the rows the kernel computes (the first of every pair, the fixed rows)."""

    sector_stencil = True

    def __init__(self, table, form) -> None:
        self.table = table
        self.form = form
        self.kinds = np.ascontiguousarray(form.kinds(), dtype=np.int8)
        self.partners = np.ascontiguousarray(form.partners(), dtype=np.int32)
        self.signs = np.ascontiguousarray(form.row_signs(), dtype=np.float64)
        self.targets = np.ascontiguousarray(form.targets(), dtype=np.int32)
        self.shape = table.shape
        # the kernel's tables over the computed rows (slot-major): the neighbour, the other row of its pair, the code
        # of its factor and its mode (0 first of a pair, 1/2 second of s = +1/-1, 3/4 fixed of s = +1/-1)
        columns = np.asarray(table.columns, dtype=np.int64)[:, self.targets]
        kinds, signs = self.kinds[columns], self.signs[columns]
        modes = np.where(
            kinds == 0, 0, np.where(kinds == 1, np.where(signs > 0, 1, 2), np.where(kinds == 2, 3, 4))
        )
        self.neighbours = np.ascontiguousarray(columns, dtype=np.int32)
        self.neighbour_pairs = np.ascontiguousarray(self.partners[columns], dtype=np.int32)
        self.neighbour_codes = np.ascontiguousarray(np.asarray(table.codes)[:, self.targets], dtype=np.uint8)
        self.neighbour_modes = np.ascontiguousarray(modes, dtype=np.uint8)
        target_kinds = self.kinds[self.targets]
        self.target_pairs = np.ascontiguousarray(self.partners[self.targets], dtype=np.int32)
        self.target_modes = np.ascontiguousarray(np.where(target_kinds == 0, 0, np.where(target_kinds == 2, 3, 4)),
                                                 dtype=np.uint8)

    @property
    def nbytes(self) -> int:
        return int(
            self.table.nbytes + self.kinds.nbytes + self.partners.nbytes + self.signs.nbytes + self.targets.nbytes
            + self.neighbours.nbytes + self.neighbour_pairs.nbytes + self.neighbour_codes.nbytes
            + self.neighbour_modes.nbytes + self.target_pairs.nbytes + self.target_modes.nbytes
        )


class _PairStencil:
    sector = True
    real_capable = True

    def __init__(self, table: PairStencilTable) -> None:
        self.pair_table = table
        self.centre = float(table.table.centre)

    def diagonal(self, diagonal):
        """``c_0 + v``: the diagonal the kernel adds."""
        return diagonal + self.centre


class HostPairStencil(_PairStencil):
    """The kernel's arithmetic on NumPy arrays (the NumPy backend of the device code, the tests)."""

    device_bytes = 0

    def apply(self, block, diagonal_with_centre, *, added=None, previous=None, center: float = 0.0, scale: float = 1.0,
              before=None, after=None):
        pairs = self.pair_table
        table, form = pairs.table, pairs.form
        block = np.ascontiguousarray(block)
        if np.iscomplexobj(block):
            raise TypeError("the real form of a sector takes real blocks (complex ones go by their parts)")
        if previous is not None and before is None:
            raise ValueError("a recurrence step with a previous block needs its coefficients")
        vectors = form.to_complex(block)
        rows = np.asarray(form.targets(), dtype=np.int64)
        values = np.asarray(diagonal_with_centre, dtype=np.float64)[rows, None] * vectors[rows]
        for slot in range(table.slots):
            values += table.factors(slot)[rows, None] * vectors[table.columns[slot][rows]]
        result = np.empty(block.shape, dtype=np.float64)
        kinds = pairs.kinds[rows]
        firsts = rows[kinds == 0]
        result[firsts] = _ROOT_TWO * values[kinds == 0].real
        result[pairs.partners[firsts]] = _ROOT_TWO * values[kinds == 0].imag
        result[rows[kinds == 2]] = values[kinds == 2].real
        result[rows[kinds == 3]] = values[kinds == 3].imag
        if added is not None:
            result += added
        result = (result - center * block) * scale
        if previous is not None:
            result -= np.asarray(before, dtype=np.float64).reshape(1, -1) * previous
        if after is not None:
            result *= np.asarray(after, dtype=np.float64).reshape(1, -1)
        return result


_SOURCE = r"""
// A thread per (computed row, column): the rows that compute are the first of every pair and the fixed rows.  Per
// slot and computed row the table holds the neighbour's row j, the other row q of its pair, the code of its factor and
// its mode (0 the first of a pair, 1 and 2 the second with s = +1 and -1, 3 and 4 a fixed row of s = +1 and -1), so
// that the coordinates of the neighbour are read without a lookup of its pair; the slots are added in their order.
#define REBUILD(MODE, XJ, XQ, VR, VI) {                                                                    \
    if (MODE == 0) { VR = XJ * HALF; VI = XQ * HALF; }                                                    \
    else if (MODE == 1) { VR = XQ * HALF; VI = -XJ * HALF; }                                              \
    else if (MODE == 2) { VR = -XQ * HALF; VI = XJ * HALF; }                                              \
    else if (MODE == 3) { VR = XJ; VI = 0.0; }                                                            \
    else { VR = 0.0; VI = XJ; } }

extern "C" __global__ void sector_pair_stencil(
    const double* __restrict__ x,
    double* __restrict__ out,
    const double* __restrict__ diagonal,
    const int* __restrict__ neighbour,
    const int* __restrict__ neighbour_pair,
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ modes,
    const double* __restrict__ palette,
    const double* __restrict__ weights,
    const int* __restrict__ targets,
    const int* __restrict__ target_pair,
    const unsigned char* __restrict__ target_mode,
    const long long computed,
    const long long count,
    const double* __restrict__ nonlocal,
    const int has_nonlocal,
    const double* __restrict__ previous,
    const int has_previous,
    const double center,
    const double scale,
    const double* __restrict__ before,
    const double* __restrict__ after,
    const int has_after,
    const int palette_size)
{
    const double HALF = 0.70710678118654752440;
    const double ROOT_TWO = 1.41421356237309504880;
    __shared__ double shared_palette[PALETTE];
    __shared__ double shared_weights[2 * SLOTS];
    for (int i = threadIdx.x; i < palette_size; i += blockDim.x) shared_palette[i] = palette[i];
    for (int i = threadIdx.x; i < 2 * SLOTS; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= computed * count) return;
    const long long i = t / count;
    const long long c = t - i * count;
    const int w = targets[i];
    const int kw = target_mode[i];
    const long long pw = (long long)w * count + c;
    const long long qw = (long long)target_pair[i] * count + c;
    const double xw = x[pw];
    const double xqw = kw == 0 ? x[qw] : 0.0;
    double vr, vi;
    REBUILD(kw, xw, xqw, vr, vi);
    const double d = diagonal[w];
    double re = d * vr;
    double im = d * vi;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const long long e = (long long)s * computed + i;
        const int m = modes[e];
        const double f = shared_palette[codes[e]];
        const double xj = x[(long long)neighbour[e] * count + c];
        const double xq = m < 3 ? x[(long long)neighbour_pair[e] * count + c] : 0.0;
        double ur, ui;
        REBUILD(m, xj, xq, ur, ui);
        const double wr = shared_weights[2 * s] * f;
        const double wi = shared_weights[2 * s + 1] * f;
        re += wr * ur - wi * ui;
        im += wr * ui + wi * ur;
    }
    long long place[2];
    double value[2];
    int outputs = 1;
    place[0] = pw;
    if (kw == 0) {
        place[1] = qw;
        value[0] = ROOT_TWO * re;
        value[1] = ROOT_TWO * im;
        outputs = 2;
    } else {
        value[0] = kw == 3 ? re : im;
    }
    for (int o = 0; o < outputs; ++o) {
        const long long p = place[o];
        double v = value[o];
        if (has_nonlocal) v += nonlocal[p];
        v = (v - center * x[p]) * scale;
        if (has_previous) v -= before[c] * previous[p];
        if (has_after) v *= after[c];
        out[p] = v;
    }
}
"""

_KERNELS: dict[tuple[int, int], Any] = {}
_LOCK = Lock()


def _kernel(cp, slots: int):
    key = (int(cp.cuda.Device().id), int(slots))
    with _LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            source = _SOURCE.replace("SLOTS", str(int(slots))).replace("PALETTE", str(_PALETTE))
            module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
            kernel = module.get_function("sector_pair_stencil")
            _KERNELS[key] = kernel
        return kernel


class DevicePairStencil(_PairStencil):
    """The tables of a sector's real form on the current CuPy device and its kernel."""

    def __init__(self, cp, table: PairStencilTable) -> None:
        super().__init__(table)
        self.cp = cp
        base = table.table
        self.rows = int(base.shape[0])
        self.palette = cp.asarray(base.palette)
        self.weights = cp.asarray(np.ascontiguousarray(base.weights.view(np.float64)))
        self.targets = cp.asarray(table.targets)
        self.target_pairs = cp.asarray(table.target_pairs)
        self.target_modes = cp.asarray(table.target_modes)
        self.neighbours = cp.asarray(table.neighbours)
        self.neighbour_pairs = cp.asarray(table.neighbour_pairs)
        self.neighbour_codes = cp.asarray(table.neighbour_codes)
        self.neighbour_modes = cp.asarray(table.neighbour_modes)
        self.kernel = _kernel(cp, base.slots)

    @property
    def device_bytes(self) -> int:
        return int(
            self.palette.nbytes + self.weights.nbytes + self.targets.nbytes + self.target_pairs.nbytes
            + self.target_modes.nbytes + self.neighbours.nbytes + self.neighbour_pairs.nbytes
            + self.neighbour_codes.nbytes + self.neighbour_modes.nbytes
        )

    def apply(self, block, diagonal_with_centre, *, added=None, previous=None, center: float = 0.0, scale: float = 1.0,
              before=None, after=None):
        """``T^H (K + v) T block`` of a C-ordered float64 block ``(rows, columns)``, with ``added``, ``previous``,
        ``before`` and ``after`` the step of the recurrence (the contract of
        :meth:`.sector_bloch_stencil.DeviceSectorStencil.apply`)."""
        cp = self.cp
        if block.dtype != cp.float64:
            raise TypeError("the real form of a sector takes float64 blocks (complex ones go by their parts)")
        if not block.flags.c_contiguous:
            block = cp.ascontiguousarray(block)
        if int(block.shape[0]) != self.rows:
            raise ValueError("the block does not match the sector")
        out = cp.empty_like(block)
        count = int(block.shape[1])
        computed = int(self.targets.size)
        total = computed * count
        if total == 0:
            return out
        if previous is not None and before is None:
            raise ValueError("a recurrence step with a previous block needs its coefficients")

        def same_shape(array):
            if array is None:
                return block, np.int32(0)
            array = cp.ascontiguousarray(array, dtype=cp.float64)
            if tuple(array.shape) != tuple(block.shape):
                raise ValueError("the arrays of a recurrence step must have the shape of the block")
            return array, np.int32(1)

        def per_column(values):
            if values is None:
                return self.palette, np.int32(0)
            values = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float64).reshape(-1))
            if values.size != count:
                raise ValueError("one coefficient per column is needed")
            return values, np.int32(1)

        nonlocal_part, has_nonlocal = same_shape(added)
        previous_part, has_previous = same_shape(previous)
        before_part, _ = per_column(before if previous is not None else None)
        after_part, has_after = per_column(after)
        self.kernel(
            ((total + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                block, out, cp.ascontiguousarray(diagonal_with_centre, dtype=cp.float64),
                self.neighbours, self.neighbour_pairs, self.neighbour_codes, self.neighbour_modes,
                self.palette, self.weights, self.targets, self.target_pairs, self.target_modes,
                np.int64(computed), np.int64(count),
                nonlocal_part, has_nonlocal, previous_part, has_previous,
                np.float64(center), np.float64(scale), before_part, after_part, has_after,
                np.int32(self.palette.size),
            ),
        )
        return out


_CSR_SOURCE = r"""
// The real counterparts of the kernels of complex_csr_device: a thread per entry of the result along the columns of
// a row, and rows of many entries (the adjoint of the projectors) split over 32 lanes summed in a fixed tree.
extern "C" __global__ void real_csr_block_product(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double* __restrict__ values,
    const double* __restrict__ block, double* __restrict__ output)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    double value = 0.0;
    const int end = indptr[r + 1];
    for (int p = indptr[r]; p < end; ++p) value += values[p] * block[(long long)indices[p] * count + c];
    output[t] = value;
}

extern "C" __global__ void real_csr_long_rows(
    const long long rows, const long long count,
    const int* __restrict__ indptr, const int* __restrict__ indices, const double* __restrict__ values,
    const double* __restrict__ block, double* __restrict__ output)
{
    __shared__ double partial[256];
    const long long r = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const long long c = (long long)blockIdx.y * 8 + (threadIdx.x >> 5);
    double value = 0.0;
    if (r < rows && c < count) {
        const int end = indptr[r + 1];
        for (int p = indptr[r] + lane; p < end; p += 32) value += values[p] * block[(long long)indices[p] * count + c];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int width = 16; width > 0; width >>= 1) {
        if (lane < width) partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (lane == 0 && r < rows && c < count) output[r * count + c] = partial[threadIdx.x];
}
"""

_CSR_KERNELS: dict[tuple[int, str], Any] = {}
# rows of more entries than this on average take the kernel that splits a row over 32 threads
_LONG_ROWS = 256


def _csr_kernel(cp, name: str):
    key = (int(cp.cuda.Device().id), name)
    with _LOCK:
        kernel = _CSR_KERNELS.get(key)
        if kernel is None:
            kernel = cp.RawKernel(_CSR_SOURCE, name, options=("--std=c++11", "--fmad=false"))
            kernel.compile()
            _CSR_KERNELS[key] = kernel
        return kernel


class DeviceRealCSR:
    """A real CSR matrix on the current device (int32 indices) whose ``@`` takes row-major float64 blocks with the
    kernels above (the real projector parts and the real kinetic CSR of a real form: the cuSPARSE product of a CSR
    matrix and a row-major block converts both layouts)."""

    dtype = np.float64

    def __init__(self, cp, matrix) -> None:
        import scipy.sparse as sp

        matrix = sp.csr_matrix(matrix, dtype=np.float64)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz >= 2**31 or max(matrix.shape) >= 2**31:
            raise ValueError("a real device CSR matrix needs fewer than 2**31 entries and rows")
        self.cp = cp
        self.shape = tuple(int(value) for value in matrix.shape)
        self.indptr = cp.asarray(matrix.indptr.astype(np.int32))
        self.indices = cp.asarray(matrix.indices.astype(np.int32))
        self.data = cp.asarray(np.ascontiguousarray(matrix.data))
        self.long_rows = matrix.nnz > _LONG_ROWS * max(1, self.shape[0])
        self.nbytes = int(self.indptr.nbytes + self.indices.nbytes + self.data.nbytes)

    def __matmul__(self, block):
        cp = self.cp
        if block.ndim != 2 or block.shape[0] != self.shape[1]:
            raise ValueError("the block does not match the matrix")
        if block.dtype != cp.float64:
            raise TypeError("a real device CSR matrix takes float64 blocks")
        block = cp.ascontiguousarray(block)
        rows, count = self.shape[0], int(block.shape[1])
        output = cp.empty((rows, count), dtype=cp.float64)
        if rows * count == 0:
            return output
        if self.long_rows:
            _csr_kernel(cp, "real_csr_long_rows")(
                (rows, (count + 7) // 8), (256,),
                (np.int64(rows), np.int64(count), self.indptr, self.indices, self.data, block, output),
            )
            return output
        total = rows * count
        _csr_kernel(cp, "real_csr_block_product")(
            ((total + _THREADS - 1) // _THREADS,), (_THREADS,),
            (np.int64(rows), np.int64(count), self.indptr, self.indices, self.data, block, output),
        )
        return output


__all__ = ["DevicePairStencil", "DeviceRealCSR", "HostPairStencil", "PairStencilTable"]
