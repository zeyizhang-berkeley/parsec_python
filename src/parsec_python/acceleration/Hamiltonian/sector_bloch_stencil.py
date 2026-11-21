"""The kinetic term of a sector of a k-point as a gathered stencil (``PARSEC_KPOINT_SECTOR_STENCIL``).

A sector of a k-point (:mod:`..SCF.kpoint_sectors`) solves ``U^T H_k U`` on the representative points of the orbits
of its little group ``H_k`` (``U`` real: the characters are ``+-1``).  The kinetic term of ``H_k`` on an orthorhombic
periodic grid is an axial stencil with one complex weight per neighbour, ``w_{a,+s}`` and ``w_{a,-s}`` for the point
``s`` steps along axis ``a`` (``c_{a,s} -+ 2i k_a g_{a,s}`` of ``-nabla^2 - 2ik.grad + |k|^2``, ``c_{a,s} e^{+-ik_a s h_a}``
of the Bloch Laplacian, :mod:`.bloch_stencil`), and the same at every point.  Row ``r`` of a sector (the
representative point ``p_r`` of the orbit ``O_r``) of the reduced operator is

``(U^T A U x)_r = (c_0 + v_r) x_r + sum_{a,+-s} w_{a,+-s} chi(q) sqrt(|O_r| / |O_j(q)|) x_{j(q)}``,

``q = p_r +- s e_a`` (wrapped), ``j(q)`` the sector column of its orbit (no term where the sector admits no state on
that orbit), ``chi(q)`` the phase of ``q`` in the sector basis: the entries of
:meth:`..Symmetry.representations.ReflectionRepresentationDecomposition.reduce_hermitian_operators`, term by term the
same products of the same numbers, only added in another order (round-off).  So a sector keeps per row and neighbour
the column ``j`` (int32) and a code of its real factor ``chi sqrt(|O_r|/|O_j|)`` (uint8, a palette of the few such
values of a 2-group), 5 bytes against the 20 of a complex CSR entry, and in the gradient form no matrix of the whole
grid is built for it (the Bloch form reads its weights off the driver's Bloch Laplacian of the whole grid).

On the device one thread forms one entry of the result, the threads of a warp along the columns of one row (the
layout of :mod:`.complex_csr_device`), and the same kernel takes one step of the Chebyshev recurrence of the filters
(``Y = (((H X - c X) a) - b_j X_previous) d_j`` with the nonlocal part of ``H X`` computed before), the contract of
:meth:`.bloch_stencil.BlochStencil.apply`: the four elementwise passes of the recurrence are not made.  Only the
device code takes the tables (:attr:`..SCF.kpoint_sectors.KPointSectors.device_operators`): on the CPU route the
sectors keep the reduced CSR, whose scipy product runs 3-4 times faster on the host than a loop over the slots.
``PARSEC_KPOINT_SECTOR_STENCIL=off`` keeps the reduced complex CSR of before, bit for bit.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np
import scipy.sparse as sp

from .bloch_stencil import AxialStencils, _axial_stencil, _offsets, grid_shape_of

_SWITCH = "PARSEC_KPOINT_SECTOR_STENCIL"
# rows of the whole grid checked at a time against the stencil (about 50 MB of temporaries for 49 entries per row)
_CHECK_ROWS = 1 << 16
# the most values of the real factors of a sector's entries (uint8 codes)
_PALETTE = 256
_THREADS = 256


def sector_stencil_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_STENCIL``: ``auto`` (default) applies the kinetic term of a sector of a k-point as a
    gathered stencil from the grid (:class:`SectorStencilTable`) where the kinetic matrices of the whole grid are an
    axial stencil, with the Chebyshev recurrence in its kernel; ``off`` keeps the reduced complex CSR of before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


# ------------------------------------------------------------------ the stencils of the whole grid
def _slots(shape, half_width: int):
    """``(axis, shift)`` of every neighbour, in the order of the slots: axis by axis, ``+1, -1, +2, -2, ...``."""

    return [(axis, sign * step) for axis in range(3) for step in range(1, half_width + 1) for sign in (1, -1)]


def _moved(points: np.ndarray, shape, axis: int, shift: int) -> np.ndarray:
    """The grid index of ``points`` moved by ``shift`` steps along ``axis`` (wrapped; index ``(x n1 + y) n2 + z``)."""

    n0, n1, n2 = shape
    strides = (n1 * n2, n2, 1)
    coordinate = (points // strides[axis]) % shape[axis]
    moved = (coordinate + shift) % shape[axis]
    return points + (moved - coordinate) * strides[axis]


def _rows_match(matrix, shape, entries, *, tolerance: float = 0.0) -> bool:
    """Whether every row of ``matrix`` holds exactly the axial ``entries`` (``(axis, shift, value)``, ``axis`` None
    for the diagonal) about its point, checked :data:`_CHECK_ROWS` rows at a time (no matrix of the whole grid is
    rebuilt); values equal (``tolerance`` 0) or within ``tolerance`` times the largest."""

    csr = sp.csr_matrix(matrix)
    size = int(np.prod(shape))
    count = len(entries)
    if csr.shape != (size, size) or csr.nnz != size * count:
        return False
    if not np.array_equal(csr.indptr, np.arange(size + 1, dtype=np.int64) * count):
        return False
    values = np.asarray([value for _, _, value in entries])
    limit = float(tolerance) * float(np.max(np.abs(values))) if values.size else 0.0
    for start in range(0, size, _CHECK_ROWS):
        stop = min(start + _CHECK_ROWS, size)
        points = np.arange(start, stop, dtype=np.int64)
        expected = np.empty((stop - start, count), dtype=np.int64)
        for slot, (axis, shift, _) in enumerate(entries):
            expected[:, slot] = points if axis is None else _moved(points, shape, axis, shift)
        order = np.argsort(expected, axis=1, kind="stable")
        expected = np.take_along_axis(expected, order, axis=1)
        wanted = values[order]
        given_columns = np.asarray(csr.indices[start * count : stop * count], dtype=np.int64).reshape(-1, count)
        given_values = np.asarray(csr.data[start * count : stop * count]).reshape(-1, count)
        given_order = np.argsort(given_columns, axis=1, kind="stable")
        if not np.array_equal(np.take_along_axis(given_columns, given_order, axis=1), expected):
            return False
        given_values = np.take_along_axis(given_values, given_order, axis=1)
        if tolerance == 0.0:
            if not np.array_equal(given_values, wanted):
                return False
        elif float(np.max(np.abs(given_values - wanted))) > limit:
            return False
    return True


def checked_axial_stencils(laplacian, gradient) -> AxialStencils | None:
    """The stencils of ``-nabla^2`` and the three first derivatives of a periodic orthorhombic grid, read off row 0
    as :meth:`.bloch_stencil.AxialStencils.from_matrices` reads them and checked row by row against every row of the
    matrices (in blocks of rows, without the rebuilt matrices of the whole grid); ``None`` where they are not that
    stencil (or wrap onto themselves)."""

    size = int(laplacian.shape[0])
    if len(gradient) != 3:
        return None
    shape = grid_shape_of(tuple(gradient), size)
    if shape is None or min(shape) < 3:
        return None
    lap = sp.csr_matrix(laplacian, dtype=np.float64)
    row = slice(lap.indptr[0], lap.indptr[1])
    moving = _offsets(lap.indices[row], shape)
    half_width = int(np.max(np.abs(moving))) if moving.size else 0
    if half_width < 1 or 2 * half_width >= min(shape):
        return None
    found = _axial_stencil(lap.indices[row], lap.data[row], shape, half_width, symmetric=True)
    if found is None:
        return None
    centre, laplacian_coefficients = found
    gradient_coefficients = np.zeros((3, half_width))
    for axis, matrix in enumerate(gradient):
        g = sp.csr_matrix(matrix, dtype=np.float64)
        grow = slice(g.indptr[0], g.indptr[1])
        stencil = _axial_stencil(g.indices[grow], g.data[grow], shape, half_width, symmetric=False)
        if stencil is None or stencil[0] != 0.0:
            return None
        if np.any(np.delete(stencil[1], axis, axis=0) != 0.0):
            return None
        gradient_coefficients[axis] = stencil[1][axis]
    entries = [(None, 0, float(centre))] if centre != 0.0 else []
    entries += [
        (axis, shift, float(laplacian_coefficients[axis, abs(shift) - 1]))
        for axis, shift in _slots(shape, half_width)
        if laplacian_coefficients[axis, abs(shift) - 1] != 0.0
    ]
    if not _rows_match(lap, shape, entries):
        return None
    for axis, matrix in enumerate(gradient):
        entries = [
            (axis, sign * step, float(sign * gradient_coefficients[axis, step - 1]))
            for step in range(1, half_width + 1)
            for sign in (1, -1)
            if gradient_coefficients[axis, step - 1] != 0.0
        ]
        if not _rows_match(sp.csr_matrix(matrix, dtype=np.float64), shape, entries):
            return None
    return AxialStencils(shape, half_width, float(centre), laplacian_coefficients, gradient_coefficients)


def kinetic_weights(stencils: AxialStencils, k_point, bloch=None) -> tuple[float, np.ndarray] | None:
    """``(centre, weights)`` of the kinetic term of a k-point: the diagonal and the complex weight of every slot
    (:func:`_slots`), ``None`` where ``bloch`` (the Bloch Laplacian of the k-point) is not that stencil.

    The gradient form ``-nabla^2 - 2ik.grad + |k|^2``: ``c_0 + |k|^2`` and ``c_{a,s} -+ i q_{a,s}``,
    ``q = 2 k_a g_{a,s}``, the numbers of the matrix the complex CSR route builds (``2 (k_a g)`` is ``(2 k_a) g``
    exactly).  The Bloch form: the entries of row 0 of ``bloch``, checked against every row to 1e-13 of the largest
    (:meth:`.bloch_stencil.BlochStencil.from_bloch`'s test).
    """

    shape, width = stencils.shape, stencils.half_width
    slots = _slots(shape, width)
    if bloch is None:
        k = np.asarray(k_point, dtype=np.float64).reshape(3)
        q = 2.0 * k[:, None] * stencils.gradient
        weights = np.empty(len(slots), dtype=np.complex128)
        for slot, (axis, shift) in enumerate(slots):
            c = stencils.laplacian[axis, abs(shift) - 1]
            g = q[axis, abs(shift) - 1]
            weights[slot] = complex(c, -g) if shift > 0 else complex(c, g)
        return float(stencils.centre + float(k @ k)), weights
    matrix = sp.csr_matrix(bloch, dtype=np.complex128)
    strides = (shape[1] * shape[2], shape[2], 1)
    row = slice(matrix.indptr[0], matrix.indptr[1])
    values = dict(zip(matrix.indices[row].tolist(), matrix.data[row].tolist()))
    weights = np.empty(len(slots), dtype=np.complex128)
    for slot, (axis, shift) in enumerate(slots):
        value = values.get((shift % shape[axis]) * strides[axis])
        if value is None:
            return None
        weights[slot] = value
    centre = values.get(0)
    if centre is None or centre.imag != 0.0:
        return None
    entries = [(None, 0, complex(centre))] + [(axis, shift, complex(weights[slot])) for slot, (axis, shift) in enumerate(slots)]
    if not _rows_match(matrix, shape, entries, tolerance=1.0e-13):
        return None
    return float(centre.real), weights


# ------------------------------------------------------------------------------ one sector
class SectorStencilTable:
    """The kinetic term of one sector of a k-point: per slot and row the sector column of the neighbour and the code of
    its real factor, the palette of those factors, the complex weight of every slot and the diagonal ``centre``.

    It stands where the reduced kinetic matrix stood (``bloch_laplacian`` of the sector's Hamiltonian): ``shape`` and
    ``table @ block`` (the term with its diagonal, on the host); the device converter gives it its kernel
    (:class:`DeviceSectorStencil`).
    """

    sector_stencil = True
    dtype = np.dtype(np.complex128)

    def __init__(self, columns: np.ndarray, codes: np.ndarray, palette: np.ndarray, weights: np.ndarray, centre: float):
        self.columns = np.ascontiguousarray(columns, dtype=np.int32)
        self.codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self.palette = np.ascontiguousarray(palette, dtype=np.float64)
        self.weights = np.ascontiguousarray(weights, dtype=np.complex128)
        self.centre = float(centre)
        size = int(self.columns.shape[1])
        self.shape = (size, size)
        self._csr: sp.csr_matrix | None = None  # the matrix of the host products (__matmul__), made on first use

    @property
    def slots(self) -> int:
        return int(self.columns.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.columns.nbytes + self.codes.nbytes + self.palette.nbytes + self.weights.nbytes)

    def factors(self, slot: int) -> np.ndarray:
        """The complex coefficient of every row in ``slot`` (its weight times the real factor of the row)."""

        return self.weights[slot] * self.palette[self.codes[slot]]

    def __matmul__(self, block):
        """``table @ block`` on the host: the product of :meth:`to_csr`, made once on the first product and kept (one
        scipy product runs 3-4 times faster than a loop over the slots; the routes on the CPU keep the reduced CSR,
        :attr:`..SCF.kpoint_sectors.KPointSectors.device_operators`, so this serves the checks and stray host
        products)."""
        matrix = self._csr
        if matrix is None:
            matrix = self._csr = self.to_csr()
        return matrix @ np.asarray(block, dtype=np.complex128)

    def to_csr(self) -> sp.csr_matrix:
        """The matrix of the term (its entries added up), for the checks."""

        size = self.shape[0]
        rows = np.tile(np.arange(size, dtype=np.int64), self.slots)
        values = np.concatenate([self.factors(slot) for slot in range(self.slots)])
        matrix = sp.coo_matrix(
            (values, (rows, self.columns.astype(np.int64).reshape(-1))), shape=self.shape
        ).tocsr()
        matrix = matrix + self.centre * sp.identity(size, dtype=np.complex128, format="csr")
        matrix.sum_duplicates()
        return matrix


def build_sector_table(decomposition, sector: int, shape, half_width: int, centre: float, weights) -> SectorStencilTable | None:
    """The :class:`SectorStencilTable` of ``sector`` of ``decomposition`` (a
    :class:`..Symmetry.representations.ReflectionRepresentationDecomposition` of a grid of ``shape``), ``None`` where
    its factors need more than :data:`_PALETTE` values.

    Row ``r``, slot ``(a, s)``: the neighbour ``q`` of the representative point of the row's orbit, its orbit ``w``,
    the sector column ``j`` of ``w`` and the factor ``chi(q) sqrt(m_r / m_w)`` (0, column ``r``, where the sector
    admits no state on ``w``), as :meth:`~..Symmetry.representations.ReflectionRepresentationDecomposition.reduce_hermitian_operators`
    weighs the entry of ``q``.
    """

    reduction = decomposition.reduction
    orbits = decomposition.sector_orbit_indices(int(sector))
    points = np.asarray(reduction.representative_rows, dtype=np.int64)[orbits]
    multiplicities = np.asarray(reduction.multiplicities, dtype=np.float64)
    to_wedge = np.asarray(reduction.full_to_wedge, dtype=np.int64)
    to_sector = np.asarray(decomposition.orbit_to_sector[int(sector)], dtype=np.int64)
    phases = decomposition.phases[int(sector)]
    slots = _slots(shape, half_width)
    rows = np.arange(orbits.size, dtype=np.int64)
    columns = np.empty((len(slots), orbits.size), dtype=np.int32)
    factors = np.empty((len(slots), orbits.size), dtype=np.float64)
    own = multiplicities[orbits]
    for slot, (axis, shift) in enumerate(slots):
        neighbours = _moved(points, shape, axis, shift)
        orbit = to_wedge[neighbours]
        column = to_sector[orbit]
        factor = phases[neighbours].astype(np.float64) * np.sqrt(own / multiplicities[orbit])
        missing = column < 0
        factor[missing] = 0.0
        column[missing] = rows[missing]
        columns[slot] = column
        factors[slot] = factor
    palette, codes = np.unique(factors, return_inverse=True)
    if palette.size > _PALETTE:
        return None
    return SectorStencilTable(columns, codes.reshape(factors.shape), palette, weights, centre)


# ------------------------------------------------------------------------------- the kernels
_SOURCE = r"""
#include <cuComplex.h>
// A thread per (row, column) of the result, the threads of a warp along the columns of one row: the row's columns
// and codes are read once per warp, the gathered rows of the block along their columns.  The slots are added in
// their order, the same in every run.  The palette and the weights sit in shared memory and the loop over the slots
// is unrolled (the same bits as a global-memory loop unrolled by 4; 0.317 -> 0.254 ms for a 78,732-row sector of
// 24 slots and 32 columns on an A100).
extern "C" __global__ void sector_bloch_stencil(
    const double2* __restrict__ u,
    double2* __restrict__ out,
    const double* __restrict__ diagonal,
    const int* __restrict__ columns,
    const unsigned char* __restrict__ codes,
    const double* __restrict__ palette,
    const double* __restrict__ weights,
    const long long rows,
    const long long count,
    const double2* __restrict__ nonlocal,
    const int has_nonlocal,
    const double2* __restrict__ previous,
    const int has_previous,
    const double center,
    const double scale,
    const double* __restrict__ before,
    const double* __restrict__ after,
    const int has_after,
    const int palette_size)
{
    __shared__ double shared_palette[PALETTE];
    __shared__ double shared_weights[2 * SLOTS];
    for (int i = threadIdx.x; i < palette_size; i += blockDim.x) shared_palette[i] = palette[i];
    for (int i = threadIdx.x; i < 2 * SLOTS; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    const double2 centre = u[t];
    const double d = diagonal[r];
    double re = d * centre.x;
    double im = d * centre.y;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const long long e = (long long)s * rows + r;
        const int j = columns[e];
        const double f = shared_palette[codes[e]];
        const double2 x = u[(long long)j * count + c];
        const double wr = shared_weights[2 * s] * f;
        const double wi = shared_weights[2 * s + 1] * f;
        re += wr * x.x - wi * x.y;
        im += wr * x.y + wi * x.x;
    }
    if (has_nonlocal) {
        const double2 extra = nonlocal[t];
        re += extra.x;
        im += extra.y;
    }
    re = (re - center * centre.x) * scale;
    im = (im - center * centre.y) * scale;
    if (has_previous) {
        const double2 p = previous[t];
        const double b = before[c];
        re -= b * p.x;
        im -= b * p.y;
    }
    if (has_after) {
        const double f = after[c];
        re *= f;
        im *= f;
    }
    out[t] = make_double2(re, im);
}
"""

# The same kernel on a real block (the sectors of k = 0, PARSEC_KPOINT_REAL_GAMMA): the weights are real there, so
# every entry is the real part of the complex kernel's, in the same order of the slots.
_REAL_SOURCE = r"""
extern "C" __global__ void sector_bloch_stencil_real(
    const double* __restrict__ u,
    double* __restrict__ out,
    const double* __restrict__ diagonal,
    const int* __restrict__ columns,
    const unsigned char* __restrict__ codes,
    const double* __restrict__ palette,
    const double* __restrict__ weights,
    const long long rows,
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
    __shared__ double shared_palette[PALETTE];
    __shared__ double shared_weights[SLOTS];
    for (int i = threadIdx.x; i < palette_size; i += blockDim.x) shared_palette[i] = palette[i];
    for (int i = threadIdx.x; i < SLOTS; i += blockDim.x) shared_weights[i] = weights[i];
    __syncthreads();
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * count) return;
    const long long r = t / count;
    const long long c = t - r * count;
    const double centre = u[t];
    double value = diagonal[r] * centre;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        const long long e = (long long)s * rows + r;
        const int j = columns[e];
        const double f = shared_palette[codes[e]];
        value += (shared_weights[s] * f) * u[(long long)j * count + c];
    }
    if (has_nonlocal) value += nonlocal[t];
    value = (value - center * centre) * scale;
    if (has_previous) value -= before[c] * previous[t];
    if (has_after) value *= after[c];
    out[t] = value;
}
"""

_KERNELS: dict[tuple[int, int, bool], Any] = {}
_LOCK = Lock()


def _kernel(cp, slots: int, real: bool = False):
    key = (int(cp.cuda.Device().id), int(slots), bool(real))
    with _LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            source = (_REAL_SOURCE if real else _SOURCE).replace("SLOTS", str(int(slots))).replace(
                "PALETTE", str(_PALETTE)
            )
            module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
            kernel = module.get_function("sector_bloch_stencil_real" if real else "sector_bloch_stencil")
            _KERNELS[key] = kernel
        return kernel


class _SectorStencil:
    """What the device operator reads of a sector's stencil (``stencil`` of
    :class:`.device_bloch.DeviceBlochOperator`): :meth:`diagonal` and :meth:`apply`, the contract of
    :class:`.bloch_stencil.BlochStencil`."""

    sector = True

    def __init__(self, table: SectorStencilTable) -> None:
        self.table = table
        self.centre = float(table.centre)
        # real weights (a sector of k = 0): the kernel also takes real blocks (PARSEC_KPOINT_REAL_GAMMA)
        self.real_capable = bool(np.all(np.asarray(table.weights).imag == 0.0))

    def diagonal(self, diagonal):
        """``c_0 + v``: the diagonal the kernel adds (``v`` the potential of the sector's rows)."""
        return diagonal + self.centre


class HostSectorStencil(_SectorStencil):
    """The kernel's arithmetic on NumPy arrays (the NumPy backend of the device code, the tests)."""

    device_bytes = 0

    def apply(self, block, diagonal_with_centre, *, added=None, previous=None, center: float = 0.0, scale: float = 1.0,
              before=None, after=None):
        table = self.table
        real = np.asarray(block).dtype == np.float64 and self.real_capable
        block = np.ascontiguousarray(block, dtype=np.float64 if real else np.complex128)
        if previous is not None and before is None:
            raise ValueError("a recurrence step with a previous block needs its coefficients")
        result = np.asarray(diagonal_with_centre, dtype=np.float64)[:, None] * block
        for slot in range(table.slots):
            factors = table.factors(slot)
            result += (factors.real if real else factors)[:, None] * block[table.columns[slot]]
        if added is not None:
            result += added
        result = (result - center * block) * scale
        if previous is not None:
            result -= np.asarray(before, dtype=np.float64).reshape(1, -1) * previous
        if after is not None:
            result *= np.asarray(after, dtype=np.float64).reshape(1, -1)
        return result


class DeviceSectorStencil(_SectorStencil):
    """The table of one sector on the current CuPy device and its kernel."""

    def __init__(self, cp, table: SectorStencilTable) -> None:
        super().__init__(table)
        self.cp = cp
        self.rows = int(table.shape[0])
        self.columns = cp.asarray(table.columns)
        self.codes = cp.asarray(table.codes)
        self.palette = cp.asarray(table.palette)
        self.weights = cp.asarray(np.ascontiguousarray(table.weights.view(np.float64)))
        self.kernel = _kernel(cp, table.slots)
        # the real weights and kernel of a sector of k = 0 (made at the first real block)
        self._real_weights = None
        self._real_kernel = None

    @property
    def device_bytes(self) -> int:
        return int(self.columns.nbytes + self.codes.nbytes + self.palette.nbytes + self.weights.nbytes)

    def apply(self, block, diagonal_with_centre, *, added=None, previous=None, center: float = 0.0, scale: float = 1.0,
              before=None, after=None):
        """``(K + v) block`` (``K`` the kinetic term of the sector) for a C-ordered complex128 block ``(rows,
        columns)``; with ``added``, ``previous``, ``before`` and ``after`` the step
        ``(((K block + v block + added) - center block) scale - before previous) after`` of the recurrence.  A float64
        block of a sector with real weights (:attr:`real_capable`, a sector of ``k = 0``) takes the real kernel: the
        real part of the complex kernel's result, bit for bit."""
        cp = self.cp
        if block.dtype == cp.float64 and self.real_capable:
            return self._apply_real(block, diagonal_with_centre, added, previous, center, scale, before, after)
        if block.dtype != cp.complex128 or not block.flags.c_contiguous:
            block = cp.ascontiguousarray(block, dtype=cp.complex128)
        if int(block.shape[0]) != self.rows:
            raise ValueError("the block does not match the sector")
        out = cp.empty_like(block)
        count = int(block.shape[1])
        total = self.rows * count
        if total == 0:
            return out
        if previous is not None and before is None:
            raise ValueError("a recurrence step with a previous block needs its coefficients")

        def same_shape(array):
            if array is None:
                return block, np.int32(0)
            array = cp.ascontiguousarray(array, dtype=cp.complex128)
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
                self.columns, self.codes, self.palette, self.weights,
                np.int64(self.rows), np.int64(count),
                nonlocal_part, has_nonlocal, previous_part, has_previous,
                np.float64(center), np.float64(scale), before_part, after_part, has_after,
                np.int32(self.palette.size),
            ),
        )
        return out

    def _apply_real(self, block, diagonal_with_centre, added, previous, center, scale, before, after):
        """:meth:`apply` of a real block with the real kernel."""
        cp = self.cp
        if self._real_kernel is None:
            self._real_weights = cp.asarray(np.ascontiguousarray(np.asarray(self.table.weights).real))
            self._real_kernel = _kernel(cp, self.table.slots, real=True)
        if not block.flags.c_contiguous:
            block = cp.ascontiguousarray(block)
        if int(block.shape[0]) != self.rows:
            raise ValueError("the block does not match the sector")
        out = cp.empty_like(block)
        count = int(block.shape[1])
        total = self.rows * count
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
        self._real_kernel(
            ((total + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                block, out, cp.ascontiguousarray(diagonal_with_centre, dtype=cp.float64),
                self.columns, self.codes, self.palette, self._real_weights,
                np.int64(self.rows), np.int64(count),
                nonlocal_part, has_nonlocal, previous_part, has_previous,
                np.float64(center), np.float64(scale), before_part, after_part, has_after,
                np.int32(self.palette.size),
            ),
        )
        return out


__all__ = [
    "DeviceSectorStencil",
    "HostSectorStencil",
    "SectorStencilTable",
    "build_sector_table",
    "checked_axial_stencils",
    "kinetic_weights",
    "sector_stencil_setting",
]
