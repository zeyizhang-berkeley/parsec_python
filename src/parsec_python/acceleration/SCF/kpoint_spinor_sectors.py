"""The k-points of a periodic spin-orbit SCF in the double-group sectors of their little groups
(``PARSEC_SOC_KPOINT_SECTORS``).

The spinor Bloch Hamiltonian of a k-point, ``H_k = h_k (x) 1 + V_SO,k`` (the u gauge of
:mod:`parsec_python.SCF.kpoints_soc`, one spin-unpolarized potential), commutes with every operation
``g: r -> R r + t`` of the little group (``R k = k``, :mod:`.kpoint_sectors`) acting on spinors as
``G = U(R') (x) P_g``: ``P_g`` the permutation of the grid points of the scalar sectors and ``U(R')`` the
rotation of the spin by the proper part ``R' = det(R) R`` (a mirror turns the spin as the two-fold rotation
about its normal, the inversion not at all).  The spin-orbit term keeps it because the projectors of an atom go
over into those of its image with the rotation matrices of their harmonics, which the ladder algebra of
``L.S`` commutes with.

**The sectors.**  The operations used are the commuting involutions of the little group whose proper part is the
identity or the two-fold rotation about one axis ``a`` (the inversion, the half-cell translations, ``C2a`` and the
mirror ``sigma_a`` with their translations): their spatial maps form the abelian group ``H`` of the scalar sectors,
with real characters ``chi``, and their spin rotations commute.  ``C2a`` acts on spinors as ``-i sigma_a P_g``, so a
spinor ``f_+ |+a> + f_- |-a>`` (``|+-a>`` the spin states along ``a``) is an eigenvector of every ``G`` of ``H`` where
``f_+`` lies in the scalar sector of a character ``chi`` and ``f_-`` in that of ``chi phi``, ``phi(g) = -1`` where
the proper part of ``g`` is ``C2a`` and ``+1`` otherwise (``phi`` is a character of ``H``).  The spinor space is the
direct sum of the ``|H|`` sectors ``(chi, chi phi)``, each holding ``n_chi + n_(chi phi)`` rows of the ``2 n`` of
the whole grid; the Hamiltonian holds each one, and the spectrum of ``H_k`` is the union of theirs.  The axis ``a``
is ``z``, ``x`` or ``y``, whichever gives the least work.  In a sector the scalar part of each component is the
reduced Bloch Hamiltonian of its scalar sector (:meth:`.kpoint_sectors.KPointSectors.hamiltonian`: the gathered
stencil or reduced CSR, the orbit means of the potential, the reduced Kleinman-Bylander projectors) and the
spin-orbit term couples the two through ``B_chi = U_chi^H P_SO`` of each (the projectors of every atom); with
``a`` other than ``z`` the overlaps turn into the up and down overlaps of the ladder algebra through the 2 x 2
frame of ``|+-a>``.

**Kramers pairs.**  Where an operation ``h`` of the cell group with ``R_h k = -k`` (the identity at ``k = 0``, the
inversion, ``C2a`` in the plane perpendicular to ``a``, ``sigma_a`` on the axis) commutes with ``H``, has the
identity or ``C2a`` as its proper part and keeps the potential at every step (the density is symmetrized over it,
or ``h`` is the identity), the antiunitary ``A = G_h T`` (``T = -i sigma_y K`` time reversal) commutes with ``H_k``
and carries the sector ``(chi, chi phi)`` onto ``(chi phi, chi)``: where ``phi`` is not trivial (``H`` holds a
``C2a`` or ``sigma_a``) the sectors pair up, one of each pair is solved, and its partner takes its levels; the
orbitals of the partner are ``A u = U_h (-i sigma_y) conj(u)`` at the rows ``h r`` (a constant phase dropped), its
density the image under ``h`` of the density of those columns, its ``<S_z>`` the opposite.

A sector of ``m`` of the ``2 n`` rows solves ``ceil(c M m / 2n) + 6`` states (``M`` the states of a k-point, ``c``
= ``PARSEC_KPOINT_SECTOR_MARGIN``), at most ``m``; the k-point keeps the lowest ``M`` levels of the union of the
sectors and their partners, and a sector with fewer states than rows whose highest level lies below the ``M``-th one
gets 1.5 times its states and is solved again.  After the first solve of a loop each solved sector keeps the levels
it holds among those kept plus a tenth of them, at least six more (``PARSEC_KPOINT_SECTOR_REFIT``, as the scalar
sectors; its Ritz states cut to their lowest pairs at its next solve).  A k-point stays one task where its sectors would not halve the work
of its eigenproblem (rows times states squared) and, under ``auto``, where its whole-grid basis (``2 n M`` complex
entries) is below :data:`MINIMUM_ENTRIES`; ``auto`` also keeps one task per k-point where that solver would deal the
k-points over more than one device of the process (the many small eigenproblems of the sectors do not scale over the
GPUs of one process, so there they save no time; a single k-point, Gamma, takes them on any number of devices).  The
Hamiltonians of the k-points in sectors are made before their first solve, several at a time on several workers.

The potential is invariant under the operations of ``H`` where the density is (the wedge of a reduced mesh or
``Symmetrize_Charge_Density``), or at self-consistency where the k-points form a set the operation maps onto itself
(:class:`.kpoint_sectors.PotentialInvariance`); the sector Hamiltonians take the potential averaged over the orbits,
as the scalar sectors do.  The nonlocal forces and the spinors of a k-point that a driver keeps are those of the
whole grid: the sector columns unfolded, with the partners, in the order of the levels; the forces are formed on the
device that holds the sectors of the k-point (on the host where they are on several devices or spilled).  The Ritz
states of the sectors go to the loop of the next structure of a relaxation or dynamics run where its sectors are the
same (:meth:`SpinorSectorKPointSolver.export_states`, :meth:`SpinorSectorKPointSolver.import_states`).

The route: the non-magnetic spin-orbit SCF of a bulk orthorhombic cell
(:func:`parsec_python.SCF.kpoints_soc.run_self_consistent_soc_kpoints`) on the device solvers of one process.  The
collinear spin-polarized spin-orbit SCF (:func:`parsec_python.SCF.kpoints_soc.run_self_consistent_soc_kpoints_spin_polarized`,
a magnetization along ``z``) takes it only under ``PARSEC_SOC_SPIN_KPOINT_SECTORS=on`` (opt-in): the operations about
``z`` alone (their spin rotations keep ``sigma_z``), no Kramers pairs (time reversal turns the magnetization over), the
k-points mapped onto themselves without time reversal, the up component with ``V + delta`` and the down one with
``V - delta``; there is no symmetry guard of the magnetization (a state that breaks an operation of the sectors is held
in its symmetric form), which is why it is not the default.  The non-collinear SCF (no symmetry: a magnetic group of
the moments would be needed), the MPI k-point solvers and the CPU route keep one task per k-point.  ``off`` keeps one
task per k-point, bit for bit as before.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ...SCF.kpoint_solver import KPointJob
from ...SCF.orbital_density import weighted_square_sum

SWITCH = "PARSEC_SOC_KPOINT_SECTORS"
_SETTINGS = ("off", "auto", "on")
#: The value where none is named.
DEFAULT = "auto"
#: The smallest whole-grid spinor basis of a k-point (``2 n`` rows times the states, complex entries) that
#: ``auto`` splits (2^24, the floor of the scalar k-point sectors).
MINIMUM_ENTRIES = 1 << 24
#: A k-point splits where the work of its solved sectors (rows times states squared) is at most this share of the
#: whole k-point's (``auto``; ``on``: any saving).
WORK_FRACTION = 0.5
_EXTRA_STATES = 6
_GROWTH = 1.5
# the key of the states of the sectors in what export_states hands to the next loop
_STATES_KEY = "kpoint_spinor_sectors"
_ATTEMPTS = 4
# (up, down) amplitudes of |+a> and |-a> per axis a (as the double-group sectors of a cluster, .soc_sectors)
_ROOT_HALF = 2.0 ** -0.5
_SPIN_FRAMES = {
    2: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    0: np.array([[_ROOT_HALF, _ROOT_HALF], [_ROOT_HALF, -_ROOT_HALF]], dtype=np.complex128),
    1: np.array([[_ROOT_HALF, _ROOT_HALF], [1j * _ROOT_HALF, -1j * _ROOT_HALF]], dtype=np.complex128),
}
_PAULI = {
    0: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    1: np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    2: np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}
_IDENTITY_TYPE = -1


def soc_kpoint_sectors_setting() -> str:
    """``PARSEC_SOC_KPOINT_SECTORS``: ``auto`` (the default: the k-points whose whole-grid spinor basis reaches
    :data:`MINIMUM_ENTRIES` and whose sectors at most halve its work), ``on`` (every k-point whose sectors save work,
    at any size) or ``off`` (one task per k-point, as before).  Another value is an error before the SCF."""

    text = os.environ.get(SWITCH, DEFAULT).strip().lower() or DEFAULT
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be one of {', '.join(_SETTINGS)}, not {text!r}")
    return text


SPIN_SWITCH = "PARSEC_SOC_SPIN_KPOINT_SECTORS"


def soc_spin_kpoint_sectors_setting() -> str:
    """``PARSEC_SOC_SPIN_KPOINT_SECTORS``: ``off`` (the default: the collinear spin-polarized spin-orbit loop keeps one
    task per k-point) or ``on`` (opt-in: it takes the sectors of the operations about ``z`` under the rules of
    :data:`SWITCH`, without Kramers pairs and without a symmetry guard of the magnetization).  Another value is an
    error before the SCF."""

    text = os.environ.get(SPIN_SWITCH, "off").strip().lower() or "off"
    if text not in ("off", "on"):
        raise ValueError(f"{SPIN_SWITCH} must be off or on, not {text!r}")
    return text


class _MagneticInvariance:
    """The operations a collinear magnetization along ``z`` keeps the potentials of: those the densities are
    symmetrized over, the translations of the cell, and those that map the set of k-points onto itself without time
    reversal (which turns the magnetization over), at equal weights."""

    def __init__(self, invariance) -> None:
        self.invariance = invariance

    def in_density_group(self, index: int) -> bool:
        return self.invariance.in_density_group(index)

    def __call__(self, index: int) -> bool:
        invariance = self.invariance
        matrix = invariance.cell.matrices[index]
        if np.array_equal(matrix, np.eye(3, dtype=matrix.dtype)) or invariance.in_density_group(index):
            return True
        rotated = (invariance.k_points @ matrix.astype(np.float64).T) @ invariance.lattice.T / (2.0 * np.pi)
        scale = max(float(np.max(np.abs(invariance.weights), initial=0.0)), 1.0e-300)
        for row, weight in zip(rotated, invariance.weights):
            same = np.abs(invariance.weights - weight) <= 1.0e-10 * scale
            direct = row[None, :] - invariance.fractional
            if not np.any(np.all(np.abs(direct - np.rint(direct)) <= 1.0e-8, axis=1) & same):
                return False
        return True


def _proper_type(matrix) -> int | None:
    """The proper part ``det(R) R`` of a signed permutation: :data:`_IDENTITY_TYPE` for the identity, ``a`` for the
    two-fold rotation about axis ``a``, ``None`` for anything else."""

    rotation = np.asarray(matrix, dtype=np.int64)
    proper = int(round(float(np.linalg.det(rotation)))) * rotation
    if np.array_equal(proper, np.eye(3, dtype=np.int64)):
        return _IDENTITY_TYPE
    for axis in range(3):
        target = -np.eye(3, dtype=np.int64)
        target[axis, axis] = 1
        if np.array_equal(proper, target):
            return axis
    return None


def _spin_rotation(kind: int) -> np.ndarray:
    """``U(R')`` of a proper part of :func:`_proper_type`: the identity, or ``-i sigma_a``."""

    if kind == _IDENTITY_TYPE:
        return np.eye(2, dtype=np.complex128)
    return -1.0j * _PAULI[int(kind)]


def _soc_matrix(soc_projectors, size: int) -> sp.csr_matrix:
    """``[v_ion | v_so]`` of every atom as one ``(n, 16 atoms)`` complex matrix."""

    rows, columns, data = [], [], []
    for index, atom in enumerate(soc_projectors):
        support = np.asarray(atom.support_rows, dtype=np.int64)
        rows.append(np.repeat(support, 16))
        columns.append(np.tile(np.arange(16) + 16 * index, support.size))
        data.append(np.asarray(atom.stacked).reshape(-1))
    if not rows:
        return sp.csr_matrix((size, 0), dtype=np.complex128)
    return sp.coo_matrix(
        (np.concatenate(data).astype(np.complex128), (np.concatenate(rows), np.concatenate(columns))),
        shape=(size, 16 * len(soc_projectors)),
    ).tocsr()


def _soc_signs(soc_projectors) -> np.ndarray:
    """The KB sign of every harmonic of every atom, ``(atoms, 8, 1)``."""

    return np.asarray(
        [np.where(np.asarray([1, 1, 1, 2, 2, 2, 2, 2]) == 1, atom.sign[0], atom.sign[1]) for atom in soc_projectors],
        dtype=np.float64,
    ).reshape(len(soc_projectors), 8, 1)


# ------------------------------------------------------------------------------------------------ the plans
class SpinorKPointPlan:
    """The double-group sectors of one k-point (see the module text).

    ``spatial`` holds the scalar sectors of the group ``H`` (:class:`.kpoint_sectors.KPointSectors`); spinor sector
    ``j`` has its ``|+a>`` component in scalar sector ``j`` and its ``|-a>`` component in ``down[j]``.  ``partner[j]``
    is the sector its Kramers partner lies in (``None`` without the pairs); ``solved`` are the sectors solved (one of
    each pair), ``counts[j]`` their states.
    """

    def __init__(self, k_index: int, k_point, spatial, axis: int, phi: np.ndarray, down: list[int],
                 elements: list[int], antiunitary: int | None, antiunitary_type: int | None, cell, solved, counts):
        self.k_index = int(k_index)
        self.k_point = np.asarray(k_point, dtype=np.float64).reshape(3)
        self.spatial = spatial
        self.axis = int(axis)
        self.frame = _SPIN_FRAMES[self.axis]
        self.phi = np.asarray(phi)
        self.down = [int(value) for value in down]
        self.elements = list(elements)
        self.antiunitary = antiunitary
        self.cell = cell
        self.spatial_sizes = tuple(int(size) for size in spatial.sizes)
        self.order = len(self.spatial_sizes)
        self.sizes = tuple(self.spatial_sizes[j] + self.spatial_sizes[self.down[j]] for j in range(self.order))
        self.partner: list[int | None] = [None] * self.order
        if antiunitary is not None:
            for j in range(self.order):
                if self.down[j] != j:
                    self.partner[j] = self.down[j]
        self.solved = [int(j) for j in solved]
        self.counts = {int(j): int(c) for j, c in zip(self.solved, counts)}
        self._spin_turn = None if antiunitary_type is None else _spin_rotation(antiunitary_type)
        self._lock = threading.Lock()
        self._soc: dict[int, tuple[Any, sp.csr_matrix]] = {}
        self._soc_full: tuple[Any, sp.csr_matrix] | None = None
        self._blocks: dict[int, tuple] = {}  # by spinor sector: (projectors, diag(B_+, B_-), signs)
        self._image_rows: np.ndarray | None = None
        self._device_rows: dict[int, Any] = {}  # by device: the image rows there
        self._fields: tuple | None = None  # (V, delta, V + delta, V - delta) of a magnetization along z
        # the loop of a collinear magnetization along z (set by plan_spinor_sectors): its densities by spin
        self.magnetic = False

    # ------------------------------------------------------------------ operators
    def _reduced_soc(self, sector: int, soc_projectors) -> sp.csr_matrix:
        """``B = U^H [v_ion | v_so]`` of scalar sector ``sector`` (made once per set of projectors)."""

        with self._lock:
            found = self._soc.get(sector)
            if found is not None and found[0] is soc_projectors:
                return found[1]
            if self._soc_full is None or self._soc_full[0] is not soc_projectors:
                self._soc.clear()
                self._soc_full = (soc_projectors, _soc_matrix(soc_projectors, int(self.spatial.decomposition.reduction.full_size)))
            matrix = (self.spatial._gather(sector) @ self._soc_full[1]).tocsr()
            matrix.sort_indices()
            self._soc[sector] = (soc_projectors, matrix)
            return matrix

    def _sector_soc(self, sector: int, soc_projectors) -> tuple[sp.csr_matrix, np.ndarray]:
        """``diag(B_+, B_-)`` of spinor sector ``sector`` (the reduced projectors of its two scalar sectors) and the KB
        signs ``(atoms, 8, 1)``, made once per set of projectors."""

        up = self._reduced_soc(sector, soc_projectors)
        down = self._reduced_soc(self.down[sector], soc_projectors)
        with self._lock:
            found = self._blocks.get(sector)
            if found is not None and found[0] is soc_projectors:
                return found[1], found[2]
            if any(entry[0] is not soc_projectors for entry in self._blocks.values()):
                self._blocks.clear()
            block = sp.block_diag((up, down), format="csr")
            block.sort_indices()
            signs = _soc_signs(soc_projectors)
            self._blocks[sector] = (soc_projectors, block, signs)
            return block, signs

    def _split_potentials(self, scalar, delta):
        """``V + delta`` and ``V - delta`` (the potentials of the up and down components of a collinear magnetization
        along ``z``), made once per pair of fields (the sectors of one step share them)."""

        with self._lock:
            found = self._fields
            if found is not None and found[0] is scalar.effective_potential and found[1] is delta:
                return found[2], found[3]
            potential = np.asarray(scalar.effective_potential, dtype=np.float64)
            field = np.asarray(delta, dtype=np.float64)
            up, down = potential + field, potential - field
            self._fields = (scalar.effective_potential, delta, up, down)
            return up, down

    def hamiltonian(self, sector: int, full) -> "SpinorSectorHamiltonian":
        """The Hamiltonian of spinor sector ``sector`` from the spinor Bloch Hamiltonian ``full`` of the whole grid
        (``scalar_hamiltonian`` and ``soc_projectors``; a collinear field ``xc_delta`` along ``z`` adds ``+delta`` to the
        up and ``-delta`` to the down component, which only sectors about ``z`` without Kramers pairs hold; no
        non-collinear field)."""

        import dataclasses

        if getattr(full, "xc_field", None) is not None:
            raise RuntimeError("the spinor sectors of a k-point take no non-collinear field")
        delta = getattr(full, "xc_delta", None)
        scalar = full.scalar_hamiltonian
        down = self.down[sector]
        if delta is not None:
            if self.axis != 2 or self.antiunitary is not None:
                raise RuntimeError("a magnetization along z needs the sectors about z without Kramers pairs")
            up_potential, down_potential = self._split_potentials(scalar, delta)
            up_hamiltonian = self.spatial.hamiltonian(
                sector, dataclasses.replace(scalar, effective_potential=up_potential)
            )
            down_hamiltonian = self.spatial.hamiltonian(
                down, dataclasses.replace(scalar, effective_potential=down_potential)
            )
        else:
            up_hamiltonian = self.spatial.hamiltonian(sector, scalar)
            down_hamiltonian = up_hamiltonian if down == sector else self.spatial.hamiltonian(down, scalar)
        soc_projectors = full.soc_projectors
        atoms = len(soc_projectors)
        soc = signs = None
        if atoms:
            soc, signs = self._sector_soc(sector, soc_projectors)
        return SpinorSectorHamiltonian(up_hamiltonian, down_hamiltonian, soc, signs, atoms, self.frame)

    # ------------------------------------------------------------------ orbitals and densities
    def unfold(self, sector: int, block, xp=np):
        """The whole-grid spinors ``(2 n, columns)`` (up then down) of columns of ``sector``, on the host or on the
        device of ``block`` (``xp``: the array namespace of ``block``)."""

        if xp is np:
            block = np.asarray(block, dtype=np.complex128)
        n_up = self.spatial_sizes[sector]
        plus = self.spatial.expand(sector, block[:n_up], xp)
        minus = self.spatial.expand(self.down[sector], block[n_up:], xp)
        frame = self.frame
        up = frame[0, 0] * plus + frame[0, 1] * minus
        down = frame[1, 0] * plus + frame[1, 1] * minus
        return xp.concatenate((up, down), axis=0)

    def image_rows(self, xp=np):
        """``map[i]``: the row of ``h r_i`` (``h`` the operation of the Kramers pairs), on the current device of ``xp``
        (made once per device)."""

        with self._lock:
            if self._image_rows is None:
                mapping = self.cell.row_map(int(self.antiunitary))
                if mapping is None:
                    raise RuntimeError("the operation of the Kramers pairs does not map the grid")
                self._image_rows = np.asarray(mapping, dtype=np.int64)
            if xp is np:
                return self._image_rows
            device = int(xp.cuda.runtime.getDevice())
            found = self._device_rows.get(device)
            if found is None:
                found = self._device_rows[device] = xp.asarray(self._image_rows)
            return found

    def partner_spinors(self, whole, xp=np):
        """``A u = U_h (-i sigma_y) conj(u)`` at the rows ``h r`` of whole-grid spinors ``u`` (up then down), on the host
        or on the current device (``xp``)."""

        n = whole.shape[0] // 2
        up, down = whole[:n], whole[n:]
        flipped_up, flipped_down = -xp.conj(down), xp.conj(up)
        turn = self._spin_turn
        new_up = complex(turn[0, 0]) * flipped_up + complex(turn[0, 1]) * flipped_down
        new_down = complex(turn[1, 0]) * flipped_up + complex(turn[1, 1]) * flipped_down
        rows = self.image_rows(xp)
        out = xp.empty_like(whole)
        out[rows] = new_up
        out[n + rows] = new_down
        return out

    def density(self, sector: int, sums_up: np.ndarray, sums_down: np.ndarray) -> np.ndarray:
        """``sum_n f_n |psi_n|^2`` on the whole grid from the orbit sums of the two components."""

        return self.spatial.density(sector, sums_up) + self.spatial.density(self.down[sector], sums_down)

    def image_density(self, density: np.ndarray) -> np.ndarray:
        """The density of the partners: ``rho(h^-1 r)``."""

        rows = self.image_rows()
        out = np.empty_like(density)
        out[rows] = density
        return out

    def moment_factors(self) -> tuple[float, float]:
        """``<S_z>`` per unit norm of the ``|+a>`` and ``|-a>`` components (different scalar sectors are orthogonal:
        no cross terms where ``phi`` is not trivial; where it is, ``a`` is ``z``)."""

        frame = self.frame
        return (
            float(abs(frame[0, 0]) ** 2 - abs(frame[1, 0]) ** 2),
            float(abs(frame[0, 1]) ** 2 - abs(frame[1, 1]) ** 2),
        )

    def work(self) -> float:
        return float(sum(self.sizes[j] * self.counts[j] ** 2 for j in self.solved))


def _sector_counts(sizes, states: int, rows: int, margin: float) -> list[int]:
    return [min(int(size), math.ceil(margin * states * size / rows) + _EXTRA_STATES) for size in sizes]


def antiunitary_candidates(cell, k_point, elements, axis: int, invariance) -> list[tuple[int, int]]:
    """Every ``(index, proper type)`` of an operation ``h`` that can make the Kramers pairs (module text), in the order of
    the cell group: the identity alone at ``k = 0``; elsewhere the operations with ``R k = -k`` whose proper part is the
    identity or ``C2a``, that the density is symmetrized over and that commute with ``elements``.  Each gives the same
    pairs of sectors and the same partner states (up to a phase per state)."""

    k = np.asarray(k_point, dtype=np.float64).reshape(3)
    if not np.any(k != 0.0):
        return [(0, _IDENTITY_TYPE)]
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(k))))
    table = cell.table
    found = []
    for index, matrix in enumerate(cell.matrices):
        if float(np.max(np.abs(matrix.astype(np.float64) @ k + k))) > tolerance:
            continue
        kind = _proper_type(matrix)
        if kind is None or (kind != _IDENTITY_TYPE and kind != axis):
            continue
        if not invariance.in_density_group(index):
            continue
        if any(int(table[index, e]) != int(table[e, index]) for e in elements):
            continue
        found.append((index, kind))
    return found


def _antiunitary(cell, k_point, elements, axis: int, invariance) -> tuple[int | None, int | None]:
    """``(index, proper type)`` of the operation ``h`` of the Kramers pairs: the first of
    :func:`antiunitary_candidates`, or ``(None, None)``."""

    found = antiunitary_candidates(cell, k_point, elements, axis, invariance)
    return found[0] if found else (None, None)


def plan_spinor_sectors(job: KPointJob, notes: list | None = None) -> list[SpinorKPointPlan | None] | None:
    """The double-group sectors of every k-point of ``job`` (``None`` for a k-point that stays one task), ``None``
    for none; ``notes`` receives the line that says what was found or why nothing was searched."""

    from .kpoint_sectors import (
        CellGroup,
        KPointSectors,
        PotentialInvariance,
        _cell_group_text,
        generated_subgroup,
        kpoint_sector_atom_tolerance,
        largest_commuting_involutions,
        little_group_reduction,
        sector_margin,
    )
    from ..Symmetry.axis_reflection import _lattice_period, _periodic_row_major, sector_orbits_exact
    from ..Symmetry.representations import ReflectionRepresentationDecomposition

    def said(text: str, plans=None):
        if notes is not None:
            notes.append(text)
        return plans

    setting = soc_kpoint_sectors_setting()
    geometry = job.geometry
    grid = geometry["grid"]
    try:
        if _lattice_period(grid) is None or not _periodic_row_major(grid):
            return said("skipped: the spinor k-point sectors take a bulk orthorhombic box of row-major points")
    except ValueError:
        return said("skipped: the spinor k-point sectors take a bulk orthorhombic box of row-major points")
    k_points = np.asarray(geometry["k_points"], dtype=np.float64).reshape(-1, 3)
    if k_points.shape[0] != job.n_tasks:
        return said("skipped: the tasks of this job are not one per k-point")
    states, rows = int(job.number_of_states), 2 * int(grid.size)
    if setting == "auto" and rows * states < MINIMUM_ENTRIES:
        return said(
            f"skipped: {rows} spinor rows x {states} states are below the floor of the spinor k-point sectors, "
            f"{MINIMUM_ENTRIES} entries ({SWITCH}=on takes them at any size; one task per k-point)"
        )
    tolerance = kpoint_sector_atom_tolerance(forces=bool(geometry.get("forces")))
    margin = sector_margin()
    cell = CellGroup(grid, geometry["atoms"], atom_tolerance=tolerance)
    if cell.table is None or cell.order < 2:
        return said(f"identity only: no cell group at an atom tolerance of {tolerance:.1e} bohr; one task per k-point")
    group_text = _cell_group_text(cell, tolerance)
    invariance = PotentialInvariance(
        cell, geometry.get("density_symmetry"), k_points, geometry["weights"], geometry["lattice"]
    )
    # a collinear magnetization along z (the spin-polarized loop, PARSEC_SOC_SPIN_KPOINT_SECTORS=on): the operations
    # about z, no time reversal (neither in the Kramers pairs nor in the set of k-points)
    magnetic = bool(geometry.get("magnetic"))
    if magnetic:
        invariance = _MagneticInvariance(invariance)
    axes = (2,) if magnetic else (2, 0, 1)
    kinds = [_proper_type(matrix) for matrix in cell.matrices]
    decompositions: dict[tuple[int, ...], Any] = {}
    plans: list[SpinorKPointPlan | None] = []
    for index, k_point in enumerate(k_points):
        members = generated_subgroup(cell.table, cell.little_group(k_point, invariance))
        best = None
        for axis in axes:
            allowed = [m for m in members if kinds[m] is not None and kinds[m] in (_IDENTITY_TYPE, axis)]
            elements, generators = largest_commuting_involutions(cell.table, allowed)
            if len(elements) < 2:
                continue
            key = tuple(elements)
            if key not in decompositions:
                decomposition = None
                reduction = little_group_reduction(cell, elements, generators)
                if reduction is not None and not sector_orbits_exact(reduction, cell.atoms, grid):
                    import dataclasses

                    reduction = dataclasses.replace(reduction, atom_representatives=None, atom_orbit_sizes=None)
                if reduction is not None:
                    try:
                        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                    except RuntimeError:
                        decomposition = None
                decompositions[key] = decomposition
            decomposition = decompositions[key]
            if decomposition is None:
                continue
            phi = np.asarray([-1 if kinds[e] == axis else 1 for e in elements], dtype=np.int8)
            characters = np.asarray(decomposition.characters, dtype=np.int8)
            lookup = {characters[s].tobytes(): s for s in range(characters.shape[0])}
            down = [lookup.get((characters[s] * phi).astype(np.int8).tobytes()) for s in range(characters.shape[0])]
            if any(value is None for value in down):
                continue
            sizes = [int(s) for s in decomposition.sector_sizes]
            spinor_sizes = [sizes[j] + sizes[down[j]] for j in range(len(sizes))]
            antiunitary, kind = (None, None)
            if np.any(phi < 0) and not magnetic:
                antiunitary, kind = _antiunitary(cell, k_point, elements, axis, invariance)
            if antiunitary is not None:
                solved = [j for j in range(len(sizes)) if j < down[j]]
            else:
                solved = list(range(len(sizes)))
            counts = _sector_counts([spinor_sizes[j] for j in solved], states, rows, margin)
            work = float(sum(spinor_sizes[j] * c * c for j, c in zip(solved, counts)))
            if best is None or work < best[0]:
                best = (work, axis, elements, decomposition, phi, down, antiunitary, kind, solved, counts)
        if best is None:
            plans.append(None)
            continue
        work, axis, elements, decomposition, phi, down, antiunitary, kind, solved, counts = best
        whole = float(rows) * states * states
        if work > (WORK_FRACTION if setting == "auto" else 1.0) * whole:
            plans.append(None)
            continue
        spatial = KPointSectors(index, k_point, decomposition, list(decomposition.sector_sizes))
        plans.append(
            SpinorKPointPlan(index, k_point, spatial, axis, phi, down, elements, antiunitary, kind, cell, solved, counts)
        )
    split = [plan for plan in plans if plan is not None]
    if not split:
        return said(
            f"{group_text}; no k-point of the {k_points.shape[0]} splits into spinor sectors (no commuting involutions "
            "about one axis in a little group the potential keeps, or sectors that would not halve the work)"
        )
    paired = sum(1 for plan in split if plan.antiunitary is not None)
    said(
        f"{group_text}; {len(split)} of {k_points.shape[0]} k-point(s) split into double-group spinor sectors (the "
        f"largest of {max(plan.order for plan in split)} sectors), {paired} of them in Kramers pairs ({SWITCH}"
        + (f", {SPIN_SWITCH}=on: the operations about z of a magnetization along z" if magnetic else "")
        + ")"
    )
    for plan in split:
        plan.magnetic = magnetic
    shared: dict = {}
    for plan in split:
        plan.spatial._shared = shared
    return plans


# ------------------------------------------------------------------------------------------------ operators
class SpinorSectorHamiltonian:
    """The Hamiltonian of one spinor sector of a k-point on the host: the reduced scalar Bloch Hamiltonians of its
    two components (``up``: the ``|+a>`` one, ``down``: the ``|-a>`` one, the same object where they share a scalar
    sector and potential), the reduced spin-orbit projectors of both as one block-diagonal matrix ``diag(B_+, B_-)``
    (rows of the sector by 16 columns per atom for ``|+a>``, then 16 per atom for ``|-a>``) with the KB signs, and the
    frame of ``|+-a>``.  The device converter takes it through :meth:`device_operator` (``spinor_sector``)."""

    spinor_sector = True

    def __init__(self, up, down, soc, soc_signs, atoms: int, frame: np.ndarray) -> None:
        self.up = up
        self.down = down
        self.soc = soc
        self.soc_signs = soc_signs
        self.atoms = int(atoms)
        self.frame = np.asarray(frame, dtype=np.complex128)
        self.up_rows = int(up.effective_potential.shape[0])
        self.down_rows = int(down.effective_potential.shape[0])
        dimension = self.up_rows + self.down_rows
        self.shape = (dimension, dimension)

    @property
    def k_point(self):
        return self.up.k_point

    def device_operator(self, converter, real=None) -> "SpinorSectorOperator":
        backend = converter.backend
        backend.use()
        up = converter.operator(self.up, real=False)
        down = up if self.down is self.up else converter.operator(self.down, real=False)
        soc = None
        if self.atoms:
            matrix = self.soc
            soc = converter._cached(
                matrix,
                lambda: (
                    backend.matrix(matrix, dtype=np.complex128),
                    backend.matrix(matrix.conj().T.tocsr(), dtype=np.complex128),
                ),
            )
        return SpinorSectorOperator(backend, up, down, soc, self.soc_signs, self.atoms, self.frame)

    def as_linear_operator(self) -> LinearOperator:
        from ..Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend

        return self.device_operator(DeviceBlochConverter(NumpyArrayBackend())).as_linear_operator()


def _ladder_algebra(frame: np.ndarray) -> np.ndarray:
    """The ladder algebra of :meth:`..Hamiltonian.device_bloch.DeviceBlochOperator._spin_orbit` as one ``32 x 32``
    matrix on the signed overlaps of one atom with the two components, ``[io_+; so_+; io_-; so_-]`` (``+-`` the spin
    states ``|+-a>`` of ``frame``), giving the coefficients of the two components: the up and down overlaps are
    ``(W (x) 1) [o_+; o_-]`` and the coefficients go back through ``W^H`` (the KB signs act on the harmonics alone)."""

    from ..Hamiltonian.device_bloch import _ladder_and_lz

    ladder_down, ladder_up, lz_half, quarter_ll1, _ = _ladder_and_lz()
    lz = np.diag(np.asarray(lz_half, dtype=np.float64).ravel())
    quarter = np.diag(np.asarray(quarter_ll1, dtype=np.float64).ravel())
    from_down = np.asarray(ladder_down, dtype=np.complex128)
    from_up = np.asarray(ladder_up, dtype=np.complex128)
    algebra = np.zeros((32, 32), dtype=np.complex128)
    io_up, so_up, io_dn, so_dn = slice(0, 8), slice(8, 16), slice(16, 24), slice(24, 32)
    algebra[0:8, so_up] = lz
    algebra[0:8, so_dn] = from_down
    algebra[8:16, io_up] = lz
    algebra[8:16, so_up] = quarter - 0.5 * lz
    algebra[8:16, io_dn] = from_down
    algebra[8:16, so_dn] = -0.5 * from_down
    algebra[16:24, so_dn] = -lz
    algebra[16:24, so_up] = from_up
    algebra[24:32, io_dn] = -lz
    algebra[24:32, so_dn] = quarter + 0.5 * lz
    algebra[24:32, io_up] = from_up
    algebra[24:32, so_up] = -0.5 * from_up
    turn = np.kron(np.asarray(frame, dtype=np.complex128), np.eye(16))
    return turn.conj().T @ algebra @ turn


class SpinorSectorOperator:
    """The device operator of one spinor sector (``shape``, ``apply``): the two scalar sector operators and the
    spin-orbit term: the overlaps of both components with the projectors of every atom in one product with
    ``diag(B_+, B_-)^H``, the ladder algebra of all atoms in one product with :func:`_ladder_algebra` (the frame of
    ``|+-a>`` folded in), and one product with ``diag(B_+, B_-)`` back to the rows of the sector."""

    real_arithmetic = False

    def __init__(self, backend, up, down, soc, signs, atoms: int, frame: np.ndarray) -> None:
        self.backend = backend
        self.xp = backend.xp
        self.up = up
        self.down = down
        self.soc = soc
        self.atoms = int(atoms)
        self.up_rows = int(up.local_dimension)
        self.down_rows = int(down.local_dimension)
        dimension = self.up_rows + self.down_rows
        self.shape = (dimension, dimension)
        self.local_dimension = dimension
        if soc is not None:
            xp = self.xp
            self._algebra = xp.asarray(_ladder_algebra(frame))
            sign = np.asarray(signs, dtype=np.float64).reshape(self.atoms, 8, 1)
            self._signs = xp.asarray(np.concatenate((sign, sign, sign, sign), axis=1))  # (atoms, 32, 1)

    def as_eigensolver_operator(self) -> "SpinorSectorOperator":
        return self

    def _spin_orbit(self, block):
        """The spin-orbit term of a block of the sector (rows of both components)."""

        xp = self.xp
        atoms = self.atoms
        vectors = int(block.shape[1])
        forward, adjoint = self.soc
        overlaps = xp.ascontiguousarray(adjoint @ block)  # (2 x 16 atoms, vectors): those of |+a>, then of |-a>
        stacked = xp.ascontiguousarray(overlaps.reshape(2, atoms, 16, vectors).transpose(1, 0, 2, 3))
        stacked = stacked.reshape(atoms, 32, vectors)
        stacked *= self._signs
        coefficients = xp.matmul(self._algebra, stacked)  # (atoms, 32, vectors)
        coefficients = xp.ascontiguousarray(coefficients.reshape(atoms, 2, 16, vectors).transpose(1, 0, 2, 3))
        return xp.ascontiguousarray(forward @ coefficients.reshape(32 * atoms, vectors))

    def apply(self, block):
        xp = self.xp
        n_up = self.up_rows
        vectors = int(block.shape[1])
        plus, minus = block[:n_up], block[n_up:]
        result = self._spin_orbit(block) if self.soc is not None else xp.zeros_like(block)
        if self.down is self.up:
            # one scalar operator for both components: one product of twice the columns
            both = self.up.apply(xp.concatenate((plus, minus), axis=1))
            result[:n_up] += both[:, :vectors]
            result[n_up:] += both[:, vectors:]
        else:
            result[:n_up] += self.up.apply(xp.ascontiguousarray(plus))
            result[n_up:] += self.down.apply(xp.ascontiguousarray(minus))
        return result

    def matmat(self, host_block: np.ndarray) -> np.ndarray:
        host_block = np.asarray(host_block, dtype=complex)
        single = host_block.ndim == 1
        if single:
            host_block = host_block[:, None]
        result = self.backend.asnumpy(self.apply(self.backend.asarray(np.ascontiguousarray(host_block))))
        return result[:, 0] if single else result

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(self.shape, matvec=self.matmat, matmat=self.matmat, rmatvec=self.matmat, dtype=complex)


# ------------------------------------------------------------------------------------------------ the solver
def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    return np.asarray(get() if get is not None and not isinstance(value, np.ndarray) else value)


class SpinorSectorKPointSolver:
    """The solver of a spinor k-point job whose k-points are split into double-group sectors.

    The tasks of the inner solver (made by ``make_solver``, the solver the job would have had) are the solved
    sectors ``(k-point, sector)`` and ``(k-point, None)`` for a k-point that stays whole, the largest work first.
    The driver sees one task per k-point: the lowest levels of the union of its sectors and their Kramers partners,
    their occupations sent back to the columns that hold them.
    """

    backend_name = "kpoint-spinor-sectors"

    def __init__(self, job: KPointJob, plans: list, make_solver: Callable[..., Any], options) -> None:
        self.job = job
        self._plans = plans
        self._make_solver = make_solver
        self._options = dict(options)
        self._grid_size = int(job.geometry["grid"].size)
        items: list[tuple[int, int | None]] = []
        for k_index in range(job.n_tasks):
            plan = plans[k_index]
            if plan is None:
                items.append((k_index, None))
            else:
                items.extend((k_index, sector) for sector in plan.solved)
        self._items = sorted(items, key=lambda item: -self._rows(item) * self._states(item) ** 2)
        self._position = {item: position for position, item in enumerate(self._items)}
        self._k_locks = [threading.Lock() for _ in range(job.n_tasks)]
        # by k-point: the (task, column, partner) of each of its levels in the order of the union
        self._placement: dict[int, list[tuple[int, int, bool]]] = {}
        # by task: the occupations of the partners of its columns (accumulate)
        self._partner_occupations: dict[int, np.ndarray] = {}
        self._regrowths = 0
        self._regrown_tasks = 0
        # PARSEC_KPOINT_SECTOR_REFIT: the counts cut after the first solve (_refit), once per solver
        from .kpoint_sectors import sector_refit_setting

        self._refit_wanted = sector_refit_setting() == "auto"
        self._refitted = False
        self._refit_states = 0
        self._prebuilt = False
        self._inner = None
        self._inner = self._new_inner()

    def _rows(self, item) -> int:
        k_index, sector = item
        return 2 * self._grid_size if sector is None else self._plans[k_index].sizes[sector]

    def _states(self, item) -> int:
        k_index, sector = item
        return int(self.job.number_of_states) if sector is None else self._plans[k_index].counts[sector]

    def _new_inner(self):
        counts = tuple(self._states(item) for item in self._items)
        inner_job = KPointJob(
            len(self._items),
            max(counts),
            self.job.eigval_settings,
            self._make_hamiltonian,
            self._accumulate,
            extras=None if self.job.extras is None else self._inner_extras,
            states_per_task=counts,
            rows_per_task=tuple(self._rows(item) for item in self._items),
            grid_points=self._grid_size,
        )
        options = dict(self._options)
        options["problem_size"] = max(self._rows(item) for item in self._items)
        return self._make_solver(inner_job, **options)

    @property
    def n_workers(self) -> int:
        return int(getattr(self._inner, "n_workers", 1))

    @property
    def backend(self):
        return getattr(self._inner, "backend", "cpu")

    # ------------------------------------------------------------------ the inner job
    def _make_hamiltonian(self, task: int, potentials: dict):
        k_index, sector = self._items[task]
        with self._k_locks[k_index]:  # the projectors of a k-point are built once (the driver's cache)
            full = self.job.make_hamiltonian(k_index, potentials)
        if sector is None:
            return full
        return self._plans[k_index].hamiltonian(sector, full)

    def _accumulate(self, task: int, vectors, occupations, xp=np) -> dict:
        k_index, sector = self._items[task]
        if sector is None:
            return self.job.accumulate(k_index, vectors, occupations, xp=xp)
        plan = self._plans[k_index]
        n_up = plan.spatial_sizes[sector]
        sums_up = _host(weighted_square_sum(vectors[:n_up], occupations, xp))
        sums_down = _host(weighted_square_sum(vectors[n_up:], occupations, xp))
        part = self.job.geometry.get("density_part", self.job.accumulate)
        if plan.magnetic:
            # a magnetization along z (the frame of z, no partners): the |+z> component is the up density
            up = plan.spatial.density(sector, sums_up)
            down = plan.spatial.density(plan.down[sector], sums_down)
            column = np.concatenate((np.sqrt(np.maximum(up, 0.0)), np.sqrt(np.maximum(down, 0.0))))[:, None]
            return part(k_index, column, np.ones(1))
        density = plan.density(sector, sums_up, sums_down)
        partner = self._partner_occupations.get(task)
        if partner is not None and np.any(partner):
            images = plan.density(
                sector,
                _host(weighted_square_sum(vectors[:n_up], partner, xp)),
                _host(weighted_square_sum(vectors[n_up:], partner, xp)),
            )
            density = density + plan.image_density(images)
        column = np.concatenate((np.sqrt(np.maximum(density, 0.0)), np.zeros(self._grid_size)))[:, None]
        return part(k_index, column, np.ones(1))

    def _inner_extras(self, task: int, vectors, xp=np):
        k_index, sector = self._items[task]
        if sector is None:
            return self.job.extras(k_index, vectors, xp=xp)
        return None

    # ------------------------------------------------------------------ steps
    def _union(self, values) -> list[np.ndarray]:
        states = int(self.job.number_of_states)
        result = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                result.append(np.asarray(values[self._position[(k_index, None)]], dtype=np.float64))
                continue
            entries = []
            for sector in plan.solved:
                task = self._position[(k_index, sector)]
                levels = np.asarray(values[task], dtype=np.float64)
                for column, value in enumerate(levels):
                    entries.append((float(value), sector, column, False, task))
                    if plan.partner[sector] is not None:
                        entries.append((float(value), sector, column, True, task))
            entries.sort(key=lambda entry: (entry[0], entry[1], entry[3], entry[2]))
            kept = entries[:states]
            self._placement[k_index] = [(task, column, partner) for _, _, column, partner, task in kept]
            result.append(np.asarray([entry[0] for entry in kept], dtype=np.float64))
        return result

    def _short_sectors(self, values) -> list[tuple[int, int]]:
        states = int(self.job.number_of_states)
        short = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                continue
            parts = {}
            for sector in plan.solved:
                part = np.asarray(values[self._position[(k_index, sector)]], dtype=np.float64)
                parts[sector] = part
            union = np.sort(
                np.concatenate(
                    [part for sector, part in parts.items()]
                    + [part for sector, part in parts.items() if plan.partner[sector] is not None]
                )
            )
            if union.size < states:
                short.extend((k_index, s) for s in plan.solved if plan.counts[s] < plan.sizes[s])
                continue
            cut = union[states - 1]
            for sector, part in parts.items():
                if plan.counts[sector] < plan.sizes[sector] and (part.size == 0 or float(np.max(part)) < cut):
                    short.append((k_index, sector))
        return short

    def _grow(self, short, potentials: dict, filter_degree: int, tolerance) -> list:
        counts = {}
        for k_index, sector in short:
            plan = self._plans[k_index]
            count = plan.counts[sector]
            grown = min(plan.sizes[sector], max(math.ceil(_GROWTH * count), count + _EXTRA_STATES))
            plan.counts[sector] = grown
            counts[self._position[(k_index, sector)]] = grown
        self._regrowths += 1
        resize = getattr(self._inner, "resize_tasks", None)
        if resize is not None and resize(counts):
            self._regrown_tasks += len(counts)
            return self._inner.solve(potentials, filter_degree, tolerance, tasks=sorted(counts))
        self._inner.close()
        self._inner = None
        self._inner = self._new_inner()
        return self._inner.solve(potentials, filter_degree, tolerance)

    def _refit(self) -> None:
        """Cut the states of every solved sector to the levels it holds among those its k-point keeps plus a buffer
        (:func:`.kpoint_sectors.refitted_count`), after the first solve of the loop, as the scalar sectors do
        (``PARSEC_KPOINT_SECTOR_REFIT``): the inner solver keeps the lowest Ritz pairs at the next solves."""

        from .kpoint_sectors import refitted_count

        self._refitted = True
        trim = getattr(self._inner, "trim_tasks", None)
        if trim is None:
            return
        counts: dict[int, int] = {}
        for k_index, plan in enumerate(self._plans):
            if plan is None or k_index not in self._placement:
                continue
            held: dict[int, int] = {}
            for task, _, partner in self._placement[k_index]:
                if not partner:
                    held[task] = held.get(task, 0) + 1
            for sector in plan.solved:
                task = self._position[(k_index, sector)]
                count = refitted_count(held.get(task, 0), plan.sizes[sector], plan.counts[sector])
                if count < plan.counts[sector]:
                    counts[task] = count
        if not counts or not trim(counts):
            return
        for task, count in counts.items():
            k_index, sector = self._items[task]
            self._refit_states += self._plans[k_index].counts[sector] - count
            self._plans[k_index].counts[sector] = count

    def _prebuild(self, potentials: dict) -> None:
        """The Hamiltonians of the k-points in sectors made once before their first solve, several at a time where the
        inner solver has several workers: the projectors of a k-point (the driver's cache) and the static operators of
        its sectors, which the first sector of each k-point would otherwise make while the workers of its other sectors
        wait for it."""

        self._prebuilt = True
        workers = int(getattr(self._inner, "n_workers", 1))
        split = [k for k, plan in enumerate(self._plans) if plan is not None]
        if workers < 2 or len(split) < 2:
            return
        from concurrent.futures import ThreadPoolExecutor

        def build(k_index: int) -> None:
            plan = self._plans[k_index]
            with self._k_locks[k_index]:
                full = self.job.make_hamiltonian(k_index, potentials)
            plan.hamiltonian(plan.solved[0], full)

        with ThreadPoolExecutor(max_workers=min(workers, len(split))) as pool:
            list(pool.map(build, split))

    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None) -> list[np.ndarray]:
        """The lowest ``number_of_states`` levels of every k-point (task order)."""

        if not self._prebuilt:
            self._prebuild(potentials)
        values = self._inner.solve(potentials, filter_degree, tolerance)
        for _ in range(_ATTEMPTS):
            short = self._short_sectors(values)
            if not short:
                union = self._union(values)
                if self._refit_wanted and not self._refitted:
                    self._refit()
                return union
            values = self._grow(short, potentials, filter_degree, tolerance)
        raise RuntimeError(
            f"the spinor sectors of the k-points did not hold the lowest levels after {_ATTEMPTS} attempts; "
            f"{SWITCH}=off solves each k-point whole"
        )

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        inner = [np.zeros(self._states(item), dtype=np.float64) for item in self._items]
        partners: dict[int, np.ndarray] = {}
        for k_index, plan in enumerate(self._plans):
            values = np.asarray(occupations[k_index], dtype=np.float64)
            if plan is None:
                inner[self._position[(k_index, None)]] = values
                continue
            for place, (task, column, partner) in enumerate(self._placement[k_index]):
                if partner:
                    target = partners.setdefault(task, np.zeros(self._states(self._items[task]), dtype=np.float64))
                    target[column] = values[place]
                else:
                    inner[task][column] = values[place]
        self._partner_occupations = partners
        return self._inner.accumulate(inner)

    def _sector_vectors(self, task: int) -> np.ndarray:
        return np.asarray(_host(self._inner.vectors(task)), dtype=np.complex128)

    def _assemble(self, k_index: int, blocks: dict, xp=np, columns: int | None = None):
        """The whole-grid spinors of k-point ``k_index`` in the order of its levels from the sector columns ``blocks``
        (inner task -> columns, all on the host or all on the current device of ``xp``)."""

        plan = self._plans[k_index]
        placement = self._placement[k_index]
        if columns is not None:
            placement = placement[: int(columns)]
        out = xp.empty((2 * self._grid_size, len(placement)), dtype=xp.complex128)
        for inner_task in sorted({entry[0] for entry in placement}):
            sector = self._items[inner_task][1]
            places = [place for place, entry in enumerate(placement) if entry[0] == inner_task]
            wanted = sorted({placement[place][1] for place in places})
            where = {column: position for position, column in enumerate(wanted)}
            block = blocks[inner_task][:, xp.asarray(np.asarray(wanted, dtype=np.int64))]
            whole = plan.unfold(sector, xp.ascontiguousarray(block), xp)
            for partner in (False, True):
                chosen = [place for place in places if placement[place][2] == partner]
                if not chosen:
                    continue
                source = plan.partner_spinors(whole, xp) if partner else whole
                targets = xp.asarray(np.asarray(chosen, dtype=np.int64))
                picks = xp.asarray(np.asarray([where[placement[place][1]] for place in chosen], dtype=np.int64))
                out[:, targets] = source[:, picks]
                del source
            del whole
        return out

    def vectors(self, task: int, columns: int | None = None) -> np.ndarray:
        """The whole-grid spinors of k-point ``task`` (``2 n`` rows) in the order of its levels, on the host."""

        plan = self._plans[task]
        if plan is None:
            return self._inner.vectors(self._position[(task, None)], columns)
        placement = self._placement[task] if columns is None else self._placement[task][: int(columns)]
        blocks = {inner: self._sector_vectors(inner) for inner in sorted({entry[0] for entry in placement})}
        return self._assemble(task, blocks, np, columns)

    def _device_extras(self, k_index: int):
        """The extras (moments and nonlocal forces) of k-point ``k_index`` from its whole-grid spinors assembled on the
        device that holds the columns of all its sectors (as one task per k-point forms them on its worker's device),
        or ``None`` where they are not on one device (spilled to the host, or on several devices)."""

        device_vectors = getattr(self._inner, "device_vectors", None)
        if device_vectors is None:
            return None
        tasks = sorted({entry[0] for entry in self._placement[k_index]})
        blocks = {task: device_vectors(task) for task in tasks}
        if any(isinstance(block, np.ndarray) or not hasattr(block, "device") for block in blocks.values()):
            return None
        devices = {int(block.device.id) for block in blocks.values()}
        if len(devices) != 1:
            return None
        import cupy as cp

        with cp.cuda.Device(devices.pop()):
            whole = self._assemble(k_index, blocks, cp)
            found = self.job.extras(k_index, whole, xp=cp)
            del whole
        return found

    def extras(self) -> list[Any]:
        inner = self._inner.extras() if self.job.extras is not None else [None] * len(self._items)
        result = []
        forces = bool(getattr(self.job.extras, "compute_forces", True))
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                result.append(inner[self._position[(k_index, None)]])
                continue
            if self.job.extras is None:
                result.append(None)
                continue
            if forces:
                # on the device of the sectors where they all are (the forces of one task per k-point are formed on
                # its device too), else on the host
                found = self._device_extras(k_index)
                result.append(found if found is not None else self.job.extras(k_index, self.vectors(k_index)))
                continue
            plus, minus = plan.moment_factors()
            moments = np.empty(len(self._placement[k_index]))
            norms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for place, (task, column, partner) in enumerate(self._placement[k_index]):
                if task not in norms:
                    block = self._sector_vectors(task)
                    n_up = plan.spatial_sizes[self._items[task][1]]
                    norms[task] = (
                        np.sum(np.abs(block[:n_up]) ** 2, axis=0),
                        np.sum(np.abs(block[n_up:]) ** 2, axis=0),
                    )
                value = plus * norms[task][0][column] + minus * norms[task][1][column]
                moments[place] = -value if partner else value
            result.append(moments)
        return result

    def _layout(self) -> list[tuple]:
        """What the states of the inner tasks belong to: per item its k-point, sector (-1 whole), rows, and for a
        sector the axis, the operations of its group and the scalar sector of its ``|-a>`` component."""

        layout = []
        for k_index, sector in self._items:
            plan = self._plans[k_index]
            entry = (int(k_index), -1 if sector is None else int(sector), self._rows((k_index, sector)))
            if sector is not None:
                entry += (plan.axis, tuple(int(e) for e in plan.elements), int(plan.down[sector]))
            layout.append(entry)
        return layout

    def export_states(self) -> dict:
        """The Ritz states of the sectors with their layout and numbers of states (for the next loop of the same or a
        moved structure: :meth:`import_states`), as the scalar sectors hand them."""

        counts = [self._states(item) for item in self._items]
        return {_STATES_KEY: {"layout": self._layout(), "counts": counts, "states": self._inner.export_states()}}

    def import_states(self, states: dict) -> None:
        """Start from the states of :meth:`export_states` where they belong to sectors of the same layout, with their
        numbers of states; anything else (the states of one task per k-point, or of other sectors) is not taken, and
        the sectors start afresh."""

        carried = states.get(_STATES_KEY) if isinstance(states, dict) else None
        if not isinstance(carried, dict):
            return
        layout = [tuple(tuple(part) if isinstance(part, (list, tuple)) else part for part in entry)
                  for entry in carried.get("layout", ())]
        if layout != self._layout():
            return
        counts = [int(count) for count in carried["counts"]]
        if counts != [self._states(item) for item in self._items]:
            for (k_index, sector), count in zip(self._items, counts):
                if sector is not None:
                    self._plans[k_index].counts[sector] = count
            self._inner.close()
            self._inner = None
            self._inner = self._new_inner()
        importer = getattr(self._inner, "import_states", None)
        if importer is not None:
            importer(carried["states"])
            # the counts were cut by the loop that made them
            self._refitted = True

    def residuals(self) -> list[float]:
        inner = self._inner.residuals()
        result = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                result.append(inner[self._position[(k_index, None)]])
                continue
            values = [inner[self._position[(k_index, s)]] for s in plan.solved]
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            result.append(max(finite) if finite else float("nan"))
        return result

    def stats(self) -> dict[str, float]:
        stats = getattr(self._inner, "stats", None)
        total = dict(stats()) if stats is not None else {}
        split = [plan for plan in self._plans if plan is not None]
        total["spinor_sector_kpoints"] = float(len(split))
        total["spinor_sector_tasks"] = float(len(self._items))
        total["spinor_sector_regrowths"] = float(self._regrowths)
        total["spinor_sector_regrown_tasks"] = float(self._regrown_tasks)
        total["spinor_sector_refit_states"] = float(self._refit_states)
        return total

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()

    def __enter__(self) -> "SpinorSectorKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def spinor_sector_solver(job: KPointJob, make_solver: Callable[..., Any], *, notes: list | None = None, **options):
    """A :class:`SpinorSectorKPointSolver` for a spinor job (``geometry["spinor"]``) whose k-points split, else
    ``None`` with the reason in ``notes``."""

    from ..Parallel import mpi
    from .kpoint_sectors import sector_operators_on_device

    def refused(text: str):
        if notes is not None:
            notes.append(text)
        return None

    setting = soc_kpoint_sectors_setting()
    if setting == "off":
        return refused(f"disabled by option ({SWITCH}=off; one task per k-point)")
    if job.states_per_task is not None:
        return refused("skipped: the k-point job holds its own count of states per task")
    if mpi.context() is not None:
        return refused("skipped: the spinor k-point sectors run in one process (the MPI solvers split the rows)")
    if not sector_operators_on_device(options.get("backend")):
        return refused("skipped: the spinor k-point sectors run on the device solver (not the CPU route)")
    if setting == "auto":
        from .kpoint_sectors import _solver_devices

        devices = min(int(_solver_devices(options.get("backend"))), int(job.n_tasks))
        if devices > 1:
            return refused(
                f"skipped: one task per k-point deals the {job.n_tasks} k-points over {devices} devices of this "
                f"process, where the spinor sectors save no time ({SWITCH}=on takes them; one task per k-point)"
            )
    plans = plan_spinor_sectors(job, notes)
    if plans is None:
        return None
    return SpinorSectorKPointSolver(job, plans, make_solver, options)


__all__ = [
    "MINIMUM_ENTRIES",
    "SPIN_SWITCH",
    "SWITCH",
    "SpinorKPointPlan",
    "SpinorSectorHamiltonian",
    "SpinorSectorKPointSolver",
    "SpinorSectorOperator",
    "plan_spinor_sectors",
    "soc_kpoint_sectors_setting",
    "soc_spin_kpoint_sectors_setting",
    "spinor_sector_solver",
]
