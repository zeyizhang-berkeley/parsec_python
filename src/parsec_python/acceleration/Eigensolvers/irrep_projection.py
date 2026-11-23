"""The projector of a unit of irreducible representations onto them, applied to filtered blocks on the GPU.

A unit (``PARSEC_ORBITAL_IRREPS``, :func:`..Symmetry.orbital_groups.irrep_units`) solves the states of one
sector of the subgroup ``H`` that lie in some irreducible representations of the full point group ``G``.  Its
Hamiltonian is that of the sector, which commutes with the projector ``P`` onto them
(:meth:`..Symmetry.representations.ReflectionRepresentationDecomposition.projection_tables`): a basis in the range
of ``P`` stays there, but for rounding.  ``P`` is applied to every filtered block (:func:`project_filtered`),
before the Rayleigh-Ritz step, in every filter route (one device, a group of devices, ranks that share a basis),
and the random start of a first solve is projected by its first filter: the CHEBFF blocks, and every block that
CHEBDAV filters and appends to its Davidson basis (its random start and random candidates included,
``chebdav._filter_in_unit``).  A filter amplifies a part outside the range as much as a level of the sector below
the wanted ones, so the projection after it removes that part to the rounding of the projection.

Two steps between the filters still leave the range, and are repaired here:

* An orthonormalization of a filtered block divides by what remains of nearly dependent columns.  A high first
  filter degree on a coarse grid (Ge5 at 0.7 bohr with degree 90) left 2e-7 to 9e-6 of the states outside the
  range (review of 2026-11-20), and the replacements that PARSEC's selective Gram-Schmidt draws for dependent
  columns (and the columns that a Householder QR completes for them) are vectors of the whole sector.
  :func:`orthonormal_in_unit` measures the share outside the range after every orthonormalization of a unit on one
  device (also the gathered basis of a group of devices) and, where it exceeds :data:`UNIT_BASIS_TOLERANCE` or a
  column was drawn at random, projects the basis and orthonormalizes it again.  Ranks that share a basis
  orthonormalize it by a shifted Cholesky QR (combinations of the filtered columns, nothing drawn) and project their
  columns before the Ritz step (``mpi_shared_basis.MPIShardGroup.orthonormal_ritz``).  CHEBDAV checks each block it
  appends after its orthonormalization the same way (``chebdav._orthonormalize_appended_in_unit``: projected and
  orthonormalized against the columns before it again).
* The generalized Rayleigh-Ritz step whitens the filtered block itself (``X L^-T``), with the same loss.
  :func:`check_unit_basis` after every solve on one device projects the saved states back and orthonormalizes them
  symmetrically where their share exceeds the tolerance, and stops the run only beyond :data:`UNIT_BASIS_LIMIT`:
  rounding stays far below it, and one random column among ``n`` states shows about ``1/sqrt(n)``.

A part outside the range of a basis that devices or ranks share is removed by the projection after the next filter;
until then it changes the energy of the step by the square of its share.

``(P x)[s, j] = sum_t scales[t, s] * x[indices[t, s], j]``: a gather of ``[G:H]`` terms at most per entry, in
column chunks of a bounded temporary, written back into the block.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ..backends.cupy import require_cupy

_KERNEL_SOURCE = r"""
extern "C" __global__ void irrep_project(
    const double* __restrict__ x, double* __restrict__ out,
    const long long* __restrict__ indices, const double* __restrict__ scales,
    const long long rows, const long long columns, const int terms,
    const long long x_row, const long long x_column, const int columns_fastest)
{
    const long long total = rows * columns;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        long long s, j;
        if (columns_fastest) { s = i / columns; j = i - s * columns; }
        else { j = i / rows; s = i - j * rows; }
        double total_value = 0.0;
        for (int t = 0; t < terms; ++t) {
            const long long entry = (long long)t * rows + s;
            total_value += scales[entry] * x[indices[entry] * x_row + j * x_column];
        }
        // out is a dense chunk in the layout of the loop
        if (columns_fastest) out[s * columns + j] = total_value; else out[j * rows + s] = total_value;
    }
}
"""

_KERNELS: dict[int, Any] = {}
_KERNEL_LOCK = threading.Lock()
# The largest temporary of one projection chunk (bytes).
_CHUNK_BYTES = 256 * 1024 * 1024


def _kernel(cp):
    device = int(cp.cuda.Device().id)
    with _KERNEL_LOCK:
        kernel = _KERNELS.get(device)
        if kernel is None:
            kernel = cp.RawKernel(_KERNEL_SOURCE, "irrep_project")
            _KERNELS[device] = kernel
    return kernel


class IrrepProjection:
    """The projector of one unit: host tables and their copies on every device that applies it.

    ``applications`` and ``columns`` count the filtered blocks that a filter route projected (and their columns);
    ``repairs`` the bases projected back into the range after an orthonormalization or a solve; ``largest_share``
    is the largest share outside the range that a measurement saw (:func:`unit_basis_share`).
    """

    def __init__(self, indices: np.ndarray, scales: np.ndarray, representation: int | None = None) -> None:
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        scales = np.ascontiguousarray(scales, dtype=np.float64)
        if indices.ndim != 2 or indices.shape != scales.shape or indices.shape[0] < 1:
            raise ValueError("a projector needs terms x rows indices and scales")
        self.indices = indices
        self.scales = scales
        self.representation = representation
        self.rows = int(indices.shape[1])
        self.terms = int(indices.shape[0])
        self._device_tables: dict[int, tuple[Any, Any]] = {}
        self._lock = threading.Lock()
        self.applications = 0
        self.columns = 0
        self.repairs = 0
        self.largest_share = 0.0

    def _tables(self, cp):
        device = int(cp.cuda.Device().id)
        with self._lock:
            tables = self._device_tables.get(device)
            if tables is None:
                tables = (cp.asarray(self.indices), cp.asarray(self.scales))
                self._device_tables[device] = tables
        return tables

    def device_bytes(self) -> int:
        """The bytes of the tables on one device."""

        return int(self.indices.nbytes + self.scales.nbytes)

    def apply(self, block: Any, *, filtered: bool = True) -> Any:
        """Project ``block`` (rows x columns, or one vector, float64 on the current device) in place; returns it.

        ``filtered`` counts it as the projection of a filtered block (``applications``); a measurement or a repair
        passes ``False``.
        """

        cp, _ = require_cupy()
        if block.dtype != cp.dtype(cp.float64):
            raise ValueError("the projector of a unit applies to float64 blocks")
        matrix = block[:, None] if block.ndim == 1 else block
        if matrix.ndim != 2 or int(matrix.shape[0]) != self.rows:
            raise ValueError("the block does not match the rows of the unit")
        columns = int(matrix.shape[1])
        if columns == 0:
            return block
        with cp.cuda.Device(matrix.device.id):
            indices, scales = self._tables(cp)
            kernel = _kernel(cp)
            itemsize = matrix.itemsize
            row_stride = int(matrix.strides[0]) // itemsize
            column_stride = int(matrix.strides[1]) // itemsize
            columns_fastest = 1 if column_stride == 1 and columns > 1 else 0
            width = max(1, min(columns, _CHUNK_BYTES // max(1, self.rows * itemsize)))
            buffer = cp.empty(self.rows * width, dtype=cp.float64)
            threads = 256
            for start in range(0, columns, width):
                stop = min(columns, start + width)
                count = stop - start
                source = matrix[:, start:stop]
                total = self.rows * count
                blocks = int(min(65535 * 8, max(1, (total + threads - 1) // threads)))
                out = buffer[:total]
                kernel(
                    (blocks,), (threads,),
                    (source, out, indices, scales, np.int64(self.rows), np.int64(count), np.int32(self.terms),
                     np.int64(row_stride), np.int64(column_stride), np.int32(columns_fastest)),
                )
                if columns_fastest:
                    source[...] = out.reshape(self.rows, count)
                else:
                    source[...] = out.reshape(count, self.rows).T
            del buffer
        if filtered:
            with self._lock:
                self.applications += 1
                self.columns += columns
        return block

    def apply_host(self, block: np.ndarray) -> np.ndarray:
        """The same projection of a host block (a copy)."""

        array = np.asarray(block, dtype=np.float64)
        if array.ndim == 1:
            return np.sum(self.scales * array[self.indices], axis=0)
        out = np.zeros_like(array)
        for row_indices, row_scales in zip(self.indices, self.scales):
            out += row_scales[:, None] * array[row_indices]
        return out

    def record(self, share: float | None = None, repair: bool = False) -> None:
        with self._lock:
            if share is not None:
                self.largest_share = max(self.largest_share, float(share))
            if repair:
                self.repairs += 1


#: the share outside the range of its projector below which a basis of a unit is left as it is
UNIT_BASIS_TOLERANCE = 1.0e-7
#: the share beyond which the states of a unit are not rounding off its range: the run stops
UNIT_BASIS_LIMIT = 1.0e-3
# Orthonormalizations of a unit's basis after a projection, at most (a second one has always been enough).
_REPAIR_PASSES = 3
_SHARE_SEED = 20261119


def unit_basis_share(projection: IrrepProjection, vectors: Any, columns: int | None = None) -> float | None:
    """The share of the first ``columns`` states (all without it) outside the range of ``projection``.

    One combination of the states with fixed pseudo-random coefficients, projected: a part of any state outside
    the range shows in it (with probability one), at the cost of one product over the basis and the projection of a
    single vector; nothing is written.  ``None`` (nothing measured) for a basis that devices or ranks share, whose
    columns no device holds together.
    """

    cp, _ = require_cupy()
    if not isinstance(vectors, cp.ndarray) or vectors.ndim != 2:
        return None
    count = int(vectors.shape[1]) if columns is None else min(int(columns), int(vectors.shape[1]))
    if count < 1:
        return None
    with cp.cuda.Device(vectors.device.id):
        coefficients = cp.asarray(
            np.random.default_rng(_SHARE_SEED).standard_normal(count), dtype=cp.float64
        )
        combination = vectors[:, :count] @ coefficients
        projected = projection.apply(combination.copy(), filtered=False)
        norm = float(cp.linalg.norm(combination))
        outside = float(cp.linalg.norm(combination - projected))
    share = outside / norm if norm > 0.0 else 0.0
    projection.record(share)
    return share


def _symmetric_orthonormalization(vectors: Any) -> None:
    """``V <- V (V^T V)^(-1/2)`` in place, in row chunks: the orthonormal basis nearest to ``V`` (Loewdin)."""

    cp, _ = require_cupy()
    with cp.cuda.Device(vectors.device.id):
        gram = vectors.T @ vectors
        gram = 0.5 * (gram + gram.T)
        values, rotations = cp.linalg.eigh(gram)
        smallest = float(values.min())
        if not smallest > 0.5:
            raise RuntimeError(
                f"the projected states of a unit are no longer independent (smallest overlap {smallest:.3e})"
            )
        inverse_root = (rotations * (1.0 / cp.sqrt(values))[None, :]) @ rotations.T
        rows, width = map(int, vectors.shape)
        chunk = max(1, _CHUNK_BYTES // max(1, 8 * width))
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            vectors[start:stop] = vectors[start:stop] @ inverse_root


def check_unit_basis(projection: IrrepProjection, vectors: Any, columns: int) -> float | None:
    """Keep the saved states of a unit in the range of its projector after a solve; the share found.

    The share of the first ``columns`` states outside the range (:func:`unit_basis_share`).  Up to
    :data:`UNIT_BASIS_TOLERANCE` nothing changes.  Up to :data:`UNIT_BASIS_LIMIT` (the rounding of a generalized
    Ritz step on an ill-conditioned block) every column of ``vectors`` is projected in place and the columns are
    orthonormalized symmetrically, which moves them by the square of the share.  Beyond it the states are not
    those of the unit (a column drawn at random that no projection reached): ``RuntimeError``.  ``None`` (nothing
    checked) for a basis that devices or ranks share.
    """

    share = unit_basis_share(projection, vectors, columns)
    if share is None or share <= UNIT_BASIS_TOLERANCE:
        return share
    if share > UNIT_BASIS_LIMIT:
        raise RuntimeError(
            f"the states of unit {projection.representation} left the range of its projector "
            f"({share:.2e} of a combination of them; PARSEC_ORBITAL_IRREPS=off solves the sectors of before)"
        )
    projection.apply(vectors, filtered=False)
    _symmetric_orthonormalization(vectors)
    projection.record(repair=True)
    return share


def orthonormal_in_unit(operator: Any, result: Any, rng: Any) -> Any:
    """The orthonormal basis of ``result`` (an ``orthonormalize_complete_subspace`` result), kept in a unit's range.

    The basis itself where ``operator`` has no unit projector (a sector solved whole: the route of before) or where
    nothing was drawn at random and its share outside the range is at most :data:`UNIT_BASIS_TOLERANCE`.  Otherwise
    the basis is projected and orthonormalized again with ``rng`` (at most three times), which also turns the
    columns drawn for dependent ones into vectors of the unit; ``RuntimeError`` where that does not help.
    """

    projection = getattr(operator, "irrep_projection", None)
    basis = result.basis
    if projection is None:
        return basis
    from .orthogonalize import orthonormalize_complete_subspace

    drawn = int(getattr(result, "random_replacements", 0))
    for _ in range(_REPAIR_PASSES):
        share = unit_basis_share(projection, basis)
        if share is None or (share <= UNIT_BASIS_TOLERANCE and not drawn):
            return basis
        projection.apply(basis, filtered=False)
        projection.record(repair=True)
        result = orthonormalize_complete_subspace(basis, rng=rng)
        basis = result.basis
        drawn = int(getattr(result, "random_replacements", 0))
    share = unit_basis_share(projection, basis)
    if share is not None and (share > UNIT_BASIS_TOLERANCE or drawn):
        raise RuntimeError(
            f"the orthonormalized basis of unit {projection.representation} stays outside the range of its "
            f"projector ({share:.2e}; PARSEC_ORBITAL_IRREPS=off solves the sectors of before)"
        )
    return basis


def project_filtered(operator: Any, block: Any) -> Any:
    """``block`` projected by the unit of ``operator`` (its ``irrep_projection``), in place; the block itself
    where the operator has none (a sector solved whole: the route of before, untouched)."""

    projection = getattr(operator, "irrep_projection", None)
    if projection is None:
        return block
    return projection.apply(block)


__all__ = [
    "IrrepProjection",
    "UNIT_BASIS_LIMIT",
    "UNIT_BASIS_TOLERANCE",
    "check_unit_basis",
    "orthonormal_in_unit",
    "project_filtered",
    "unit_basis_share",
]
