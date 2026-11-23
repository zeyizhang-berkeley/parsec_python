"""Real arithmetic in the sectors of a k-point through an antiunitary operation (``PARSEC_KPOINT_SECTOR_REAL``).

The Bloch Hamiltonian of the u gauge is complex at a general k-point.  Where an operation ``g: r -> R r + t`` of the
cell maps ``k`` onto ``-k`` exactly (the inversion of a centrosymmetric cell, a twofold axis or a mirror normal to
``k``) and the potential is invariant under it, the antiunitary ``A = P_g K`` (``K`` complex conjugation, ``P_g f(r) =
f(g^-1 r)``) commutes with ``H_k``: ``K H_k K = H_{-k}`` for the real potential and the real stencils and projectors
of the u gauge, and ``P_g H_{-k} P_g^-1 = H_{R(-k)} = H_k`` (:mod:`.kpoint_sectors`).  Where ``g`` normalizes the
commuting involutions ``H_k`` of the sectors and fixes the character of a sector (``chi(g^-1 h g) = chi(h)``), ``A``
maps that sector onto itself: on its coefficients ``A v = S conj(v)``, ``S`` the signed permutation of its rows of
:class:`.kpoint_sectors.ClassMember` (row ``w`` to ``pi(w)`` with the sign ``s_w``).  Where ``A^2 = S^2 = 1`` the
vectors that ``A`` keeps form a real structure: the basis

``(e_w + s_w e_pi(w)) / sqrt(2)``, ``i (e_w - s_w e_pi(w)) / sqrt(2)`` for a pair ``w < pi(w)``, and ``e_w``
(``s_w = 1``) or ``i e_w`` (``s_w = -1``) for a row ``pi(w) = w``

is orthonormal and kept by ``A``, so ``T^H H_s T`` (``T`` its columns, unitary) is real symmetric and the sector is
solved on real blocks (:class:`RealForm`; the coordinate of the first member of a pair sits on row ``w``, that of the
second on ``pi(w)``): half the bytes of its subspace, a quarter of the flops of its dense products, one complex row of
``H_s`` per pair in its stencil (:mod:`..Hamiltonian.sector_pair_stencil`).  The spectrum is that of the complex
sector; the orbitals handed out are ``U T x``.  On A100 the filter takes as long as the complex one (its gathers read
two coordinates of a neighbour in two places where the complex kernel reads one complex entry: the same memory
transactions), the orthonormalization and Rayleigh-Ritz steps less, and the memory of the subspaces falls; whole runs
are within a few per cent of the complex ones, which is why the switch stays off.

``PARSEC_KPOINT_SECTOR_REAL``: ``off`` (default) keeps the complex solve; ``auto`` solves such sectors on real blocks
in the device code of an unpolarized or collinear-spin k-point SCF without the exchange of a hybrid (not ``k = 0``,
which has its own real solve, ``PARSEC_KPOINT_REAL_GAMMA``; not the band paths; not the pairs of the time-reversal
invariant points in the Bloch form), with operations the density is symmetrized over.
"""

from __future__ import annotations

import math
import os

import numpy as np
import scipy.sparse as sp

_SWITCH = "PARSEC_KPOINT_SECTOR_REAL"
_HALF = 1.0 / math.sqrt(2.0)


def sector_real_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_REAL``: ``off`` (default) or ``auto`` (see the module text)."""

    value = os.environ.get(_SWITCH, "off").strip().lower() or "off"
    if value not in {"off", "auto"}:
        raise ValueError(f"{_SWITCH} must be off or auto, not {value!r}")
    return value


class RealForm:
    """The real structure of a sector kept by ``A = P_g K`` (``A^2 = 1``): its pairs and fixed rows.

    ``first`` and ``second = pi(first)`` (``first < second``) are the pairs, ``signs`` their ``s_w`` (``s_w =
    s_pi(w)`` where ``A^2 = 1``), ``plus`` and ``minus`` the rows ``pi(w) = w`` of ``s_w = 1`` and ``-1``.  The real
    coordinates of a sector vector sit on its rows: ``x[first]`` and ``x[second]`` the two of a pair, ``x[w]`` that of
    a fixed row.  :meth:`to_complex` is ``T x``, :meth:`to_real` ``T^H y``, both complex-linear (a complex ``x`` is a
    complex combination of real coordinate vectors: an eigenvector of the real operator found in complex arithmetic
    gives an eigenvector of the sector)."""

    def __init__(self, size: int, rows: np.ndarray, signs: np.ndarray, operation: int) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        signs = np.asarray(signs, dtype=np.float64)
        self.size = int(size)
        self.operation = int(operation)
        index = np.arange(self.size, dtype=np.int64)
        fixed = rows == index
        first = np.flatnonzero(~fixed & (rows > index))
        self.first = first
        self.second = rows[first]
        self.signs = signs[first]
        self.plus = np.flatnonzero(fixed & (signs > 0))
        self.minus = np.flatnonzero(fixed & (signs < 0))
        self._device: dict[int, tuple] = {}
        for array in (self.first, self.second, self.signs, self.plus, self.minus):
            array.setflags(write=False)

    @classmethod
    def of_signed_permutation(cls, rows, signs, operation: int) -> "RealForm | None":
        """The form of ``S`` (row ``w`` to ``rows[w]`` with ``signs[w]``) where ``S^2 = 1``, else ``None``."""

        rows = np.asarray(rows, dtype=np.int64)
        signs = np.asarray(signs, dtype=np.float64)
        size = rows.size
        if size == 0 or np.any(rows < 0) or np.any(rows >= size) or np.unique(rows).size != size:
            return None
        if not np.array_equal(rows[rows], np.arange(size)) or not np.all(signs * signs[rows] == 1.0):
            return None
        return cls(size, rows, signs, operation)

    # -------------------------------------------------------------------- tables of the kernel
    def kinds(self) -> np.ndarray:
        """Per row: 0 the first of a pair, 1 the second, 2 a fixed row of ``s = 1``, 3 one of ``s = -1``."""

        kinds = np.empty(self.size, dtype=np.int8)
        kinds[self.first] = 0
        kinds[self.second] = 1
        kinds[self.plus] = 2
        kinds[self.minus] = 3
        return kinds

    def partners(self) -> np.ndarray:
        """Per row the other row of its pair (itself for a fixed row)."""

        partners = np.arange(self.size, dtype=np.int32)
        partners[self.first] = self.second
        partners[self.second] = self.first
        return partners

    def row_signs(self) -> np.ndarray:
        """Per row the sign ``s`` of its pair (1 for a fixed row)."""

        signs = np.ones(self.size, dtype=np.float64)
        signs[self.first] = self.signs
        signs[self.second] = self.signs
        return signs

    def targets(self) -> np.ndarray:
        """The rows whose complex entry of ``H v`` gives the output: the first of every pair and the fixed rows."""

        return np.sort(np.concatenate((self.first, self.plus, self.minus))).astype(np.int32)

    def adjoint_matrix(self) -> sp.csr_matrix:
        """``T^H`` as a sparse matrix (at most two entries per row)."""

        rows, columns, values = [], [], []
        rows += [self.first, self.first, self.second, self.second, self.plus, self.minus]
        columns += [self.first, self.second, self.first, self.second, self.plus, self.minus]
        values += [
            np.full(self.first.size, _HALF, dtype=np.complex128),
            self.signs * _HALF + 0j,
            np.full(self.first.size, -1j * _HALF),
            1j * self.signs * _HALF,
            np.ones(self.plus.size, dtype=np.complex128),
            np.full(self.minus.size, -1j),
        ]
        return sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(self.size, self.size)
        )

    def symmetric_rows(self, values: np.ndarray) -> np.ndarray:
        """A per-row real field (the potential of the sector rows) made equal on the two rows of every pair (their
        mean: the field is invariant under ``g``, to round-off)."""

        values = np.array(values, dtype=np.float64, copy=True)
        mean = 0.5 * (values[self.first] + values[self.second])
        values[self.first] = mean
        values[self.second] = mean
        return values

    # ------------------------------------------------------------------------- the transforms
    def _indices(self, xp, device):
        if xp is np:
            return self.first, self.second, self.signs, self.plus, self.minus
        found = self._device.get(device)
        if found is None:
            with xp.cuda.Device(device):
                found = tuple(xp.asarray(array) for array in (self.first, self.second, self.signs, self.plus, self.minus))
            self._device[device] = found
        return found

    def to_complex(self, block, xp=np):
        """``T x`` of a block of real coordinates (or of complex combinations of them), host or device."""

        device = None if xp is np or isinstance(block, np.ndarray) else int(block.device.id)
        if device is None:
            xp = np
            block = np.asarray(block)
        first, second, signs, plus, minus = self._indices(xp, device)
        squeeze = block.ndim == 1
        if squeeze:
            block = block[:, None]
        out = xp.empty(block.shape, dtype=xp.complex128)
        a, b = block[first], block[second]
        out[first] = (a + 1j * b) * _HALF
        out[second] = signs[:, None] * (a - 1j * b) * _HALF
        out[plus] = block[plus]
        out[minus] = 1j * block[minus]
        return out[:, 0] if squeeze else out

    def to_real(self, block, xp=np, *, real: bool = True):
        """``T^H y`` (complex-linear); ``real``: its real part (``y = H T x`` of real ``x``, whose ``T^H y`` is real)."""

        device = None if xp is np or isinstance(block, np.ndarray) else int(block.device.id)
        if device is None:
            xp = np
            block = np.asarray(block)
        first, second, signs, plus, minus = self._indices(xp, device)
        squeeze = block.ndim == 1
        if squeeze:
            block = block[:, None]
        out = xp.empty(block.shape, dtype=xp.complex128)
        a, b = block[first], signs[:, None] * block[second]
        out[first] = (a + b) * _HALF
        out[second] = -1j * (a - b) * _HALF
        out[plus] = block[plus]
        out[minus] = -1j * block[minus]
        if real:
            out = xp.ascontiguousarray(out.real)
        return out[:, 0] if squeeze else out


def sector_real_forms(cell, elements, decomposition, k_point, invariance, sectors) -> dict[int, RealForm] | None:
    """The :class:`RealForm` of every sector of ``sectors`` that an antiunitary ``P_g K`` maps onto itself with
    ``A^2 = 1``, ``None`` where none does: ``g`` an operation of ``cell`` with ``R k = -k`` exactly, in the density
    group (:meth:`.kpoint_sectors.PotentialInvariance.in_density_group`), that normalizes ``H_k`` (``elements``) and
    fixes the sector's character; the first such operation in the order of the cell group."""

    from .kpoint_sectors import _inverse_elements, exact_operation_test

    k = np.asarray(k_point, dtype=np.float64).reshape(3)
    table = cell.table
    if table is None or not np.any(k != 0.0) or getattr(decomposition, "is_complex", False):
        return None
    inverse = _inverse_elements(table)
    position = {int(element): index for index, element in enumerate(elements)}
    characters = np.ascontiguousarray(decomposition.characters, dtype=np.int8)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(k))))
    keeps_atoms = exact_operation_test(cell)
    candidates: list[tuple[int, np.ndarray]] = []
    for g in range(cell.order):
        matrix = np.asarray(cell.matrices[g], dtype=np.float64)
        if float(np.max(np.abs(matrix @ k + k))) > tolerance or not invariance.in_density_group(g):
            continue
        if not keeps_atoms(g):
            continue
        found = np.empty(len(elements), dtype=np.int64)
        for index, h in enumerate(elements):
            image = position.get(int(table[table[inverse[g], int(h)], g]))
            if image is None:
                break
            found[index] = image
        else:
            candidates.append((g, found))
    if not candidates:
        return None
    to_wedge = np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64)
    representatives = np.asarray(decomposition.reduction.representative_rows, dtype=np.int64)
    forms: dict[int, RealForm] = {}
    for sector in sectors:
        sector = int(sector)
        rows_of_sector = None
        for g, permutation in candidates:
            if not np.array_equal(characters[sector, permutation], characters[sector]):
                continue
            if rows_of_sector is None:
                rows_of_sector = representatives[decomposition.sector_orbit_indices(sector)]
            image = cell.row_map(g, rows=rows_of_sector)
            if image is None:
                continue
            image = np.asarray(image, dtype=np.int64)
            rows = np.asarray(decomposition.orbit_to_sector[sector][to_wedge[image]], dtype=np.int64)
            signs = np.asarray(decomposition.phases[sector][image], dtype=np.float64)
            if np.any(signs == 0.0):
                continue
            form = RealForm.of_signed_permutation(rows, signs, g)
            if form is not None:
                forms[sector] = form
                break
    return forms or None


def real_projector_parts(projectors: sp.spmatrix, signs: np.ndarray, form: RealForm):
    """``(forward, adjoint, signs)`` (real CSR, ``(n, 2P)``, ``(2P, n)``, ``2P``) of the projector term of a real form:
    ``T^H B D B^H T = R D R^T + I D I^T`` with ``B_r = T^H B = R + iI``, as ``[R, -I] diag(D, D) [R, -I]^T``."""

    reduced = (form.adjoint_matrix() @ sp.csr_matrix(projectors, dtype=np.complex128)).tocsr()
    # (index arrays of their own: eliminate_zeros works in place)
    real = sp.csr_matrix(
        (reduced.data.real.copy(), reduced.indices.copy(), reduced.indptr.copy()), shape=reduced.shape
    )
    imaginary = sp.csr_matrix(
        (-reduced.data.imag, reduced.indices.copy(), reduced.indptr.copy()), shape=reduced.shape
    )
    real.eliminate_zeros()
    imaginary.eliminate_zeros()
    forward = sp.hstack([real, imaginary], format="csr")
    adjoint = forward.T.tocsr()
    signs = np.asarray(signs, dtype=np.float64)
    return forward, adjoint, np.concatenate((signs, signs))


def real_kinetic_matrix(kinetic, form: RealForm) -> sp.csr_matrix:
    """``T^H K T`` of a complex kinetic matrix (the reduced CSR of a sector without its gathered stencil) as a real
    CSR; its imaginary part, round-off where ``A`` keeps ``K``, is dropped."""

    adjoint = form.adjoint_matrix()
    product = (adjoint @ sp.csr_matrix(kinetic, dtype=np.complex128) @ adjoint.conj().T).tocsr()
    real = sp.csr_matrix(
        (product.data.real.copy(), product.indices.copy(), product.indptr.copy()), shape=product.shape
    )
    real.eliminate_zeros()
    return real


class RealFormHamiltonian:
    """The Hamiltonian ``T^H H_s T`` of a sector in its real form: the complex sector Hamiltonian ``complex`` (a
    :class:`.kpoint_sectors.SectorBlochHamiltonian`), its :class:`RealForm` and its static real parts ``(stencil
    table or None, real kinetic CSR or None, projector parts)``.  On the host it applies the complex Hamiltonian
    between ``T`` and ``T^H`` (complex blocks: a complex combination of real coordinate vectors); the device code
    takes :meth:`device_operator`: the paired stencil kernel or the real kinetic CSR, the real projector parts and
    the potential of the rows, a real operator on float64 blocks."""

    real_form_sector = True

    def __init__(self, complex_hamiltonian, form: RealForm, parts) -> None:
        self.complex = complex_hamiltonian
        self.form = form
        self.parts = parts
        self.k_point = complex_hamiltonian.k_point
        self.shape = (form.size, form.size)
        self.effective_potential = form.symmetric_rows(complex_hamiltonian.effective_potential)
        self.nonlocal_operator = None  # the real projector parts stand for it (no Bloch projectors of a whole grid)

    def apply(self, block):
        block = np.asarray(block)
        real = not np.iscomplexobj(block)
        return self.form.to_real(self.complex.apply(self.form.to_complex(block)), real=real)

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        def product(block):
            return self.form.to_real(self.complex.apply(self.form.to_complex(np.asarray(block))), real=False)

        return LinearOperator(self.shape, matvec=product, matmat=product, rmatvec=product, dtype=complex)

    def device_operator(self, converter, real=None):
        """The real device operator of this form on ``converter``'s backend (a
        :class:`..Hamiltonian.device_bloch.DeviceBlochOperator` with ``real_arithmetic``)."""

        from ..Hamiltonian.device_bloch import DeviceBlochOperator
        from ..Hamiltonian.sector_pair_stencil import DevicePairStencil, DeviceRealCSR, HostPairStencil

        backend = converter.backend
        backend.use()
        table, kinetic, projectors = self.parts
        size = self.form.size
        cupy = backend.name == "cupy"

        def matrix(host):
            # the CSR kernels of row-major real blocks on CuPy (cuSPARSE converts the layouts), the host's own else
            return DeviceRealCSR(backend.cp, host) if cupy else backend.matrix(host, dtype=np.float64)

        stencil = None
        laplacian = None
        if table is not None:
            if cupy:
                stencil = converter._cached(table, lambda: DevicePairStencil(backend.cp, table))
            else:
                stencil = converter._cached(table, lambda: HostPairStencil(table))
        else:
            laplacian = converter._cached(kinetic, lambda: matrix(kinetic))
        forward = adjoint = signs = None
        if projectors is not None:
            forward, adjoint, signs = converter._cached(
                projectors,
                lambda: (
                    matrix(projectors[0]),
                    matrix(projectors[1]),
                    backend.asarray(np.asarray(projectors[2], dtype=np.float64)),
                ),
            )
        diagonal = backend.asarray(np.asarray(self.effective_potential, dtype=np.float64))
        operator = DeviceBlochOperator(
            backend,
            grid_size=size,
            spinor=False,
            laplacian=laplacian,
            k_gradient=None,
            diagonal=diagonal,
            projector_forward=forward,
            projector_adjoint=adjoint,
            projector_signs=signs,
            soc=None,
            projector_real=True,
            global_grid_size=size,
        )
        if stencil is not None:
            operator.stencil = stencil
            operator.stencil_diagonal = stencil.diagonal(diagonal)
        operator.real_arithmetic = True
        return operator


__all__ = [
    "RealForm",
    "RealFormHamiltonian",
    "real_kinetic_matrix",
    "real_projector_parts",
    "sector_real_forms",
    "sector_real_setting",
]
