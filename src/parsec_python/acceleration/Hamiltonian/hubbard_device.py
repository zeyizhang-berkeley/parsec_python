"""LDA+U of ``upot.F90``, ``plusu.f90z`` and ``calcden.f90z`` on accelerated backends.

The physics is that of :mod:`parsec_python.Hamiltonian.hubbard`; this module
supplies what an accelerated calculation holds differently.

The term of spin channel ``s``,

``H_U = sum_a sum_(m,m') |phi_a,m> D^(a,s)_(m,m') <phi_a,m'|``,

is a separable operator like the Kleinman--Bylander term, whose device
kernels apply ``B diag(s) B.T``: the projection, the scatter fused into the
stencil, the recorded filter graphs, the operators of the symmetry sectors,
the replicas of a sector basis that several devices share and the opt-in
FP32 filter all read the coefficients ``s`` from one device array per
operator.  With ``J = 0`` the matrices ``D`` are symmetric, and a symmetric
``5 x 5`` matrix is a combination of fifteen fixed rank-one terms,

``D = sum_i (D_ii - sum_(j != i) D_ij) e_i e_i^T
      + sum_(i<j) D_ij (e_i + e_j)(e_i + e_j)^T``

(:func:`expansion_coefficients`).  The columns ``phi_a,i`` and
``phi_a,i + phi_a,j`` of every LDA+U atom are therefore appended to the KB
factor once per structure (:func:`augmented_nonlocal_operator`), with
coefficient zero, and every SCF iteration writes only their fifteen
coefficients per atom into the device arrays of the signs
(:meth:`DeviceHubbard.bind`).  No kernel changes.  In a symmetry sector the
reduced factor ``U_G^T [B | P E]`` gives ``U_G^T H_U U_G``, and ``H_U``
commutes with the operations of the group: the occupation matrices of the
orbitals of all sectors transform with the d harmonics of the atoms, so
``D`` does too, and the sector blocks are exact.

PARSEC's ``J`` term contracts the tensors of complex harmonics with the real
occupation matrices; its ``D`` is not symmetric (see the reference module),
so it has no such form, and :class:`DeviceHubbard` refuses ``J != 0``.

The occupation matrices ``N`` and the LDA+U forces need the overlaps of the
orbitals with the five projectors of every atom (and their gradients).  They
are formed where the orbitals are: a full-grid basis on its device, the
basis of every symmetry sector, on its device or spilled to the host, a
sector basis that several devices share, and a basis that holds the
triangular Ritz hand-over; only ``(atoms, 5, 5)`` arrays come back
(:meth:`DeviceHubbard.occupation_matrices`).  ``B_G.T c = B.T psi`` for an
orbital ``psi`` of sector ``G`` with sector vector ``c``, so the reduced
projectors give the overlaps of the full-grid orbitals.

Projector orbits (``PARSEC_LDAU_PROJECTOR_ORBITS``, since 2026-11-21).  An
operation ``g: r -> R r + t`` of the abelian sectors that carries atom ``a``
onto atom ``b`` carries its projectors with the d harmonics turned,
``P_g B_a = B_b D(R)`` (:func:`parsec_python.Hamiltonian.hubbard.d_harmonic_rotation`,
``Y(R^-1 v) = Y(v) D(R)``, orthogonal).  Every orbital of a sector of a real
one-dimensional character is an eigenvector of ``P_g``, so the density matrix
of the solved states commutes with it and the occupation matrices of an orbit
follow from that of one atom, ``N_b = D N_a D^T``; so do the potential
matrices, ``D_b = D(R) D_a D(R)^T``, hence ``B_b D_b B_b^T = P_g B_a D_a
B_a^T P_g^T`` and every atom of an orbit gives a sector the same block.
Under ``auto`` (the default) the sectors of a reduction that knows the orbits
of its atoms therefore carry the KB and the fifteen LDA+U columns of one atom
per orbit, scaled by the square root of the orbit size
(:func:`parsec_python.acceleration.Symmetry.axis_reflection.compress_projectors_for_sectors`,
as for the KB columns alone), the coefficients of the representative are
written there (:class:`HubbardOrbits`), the occupation matrices are formed for
the representatives alone and turned onto the other atoms of their orbits.
The forces read the projectors of every atom, as before.  The images of the
atoms are matched within ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` (1e-4 bohr at
least, :func:`_image_tolerance`), the tolerance the group was detected at;
the driver gives a structure that keeps the group only to more than 1e-6
bohr the columns of every atom (:func:`ldau_orbits_exact`).
``off`` gives the sectors of an LDA+U run the columns of every atom
(``driver._without_atom_orbits``), bit for bit as before that date in the
abelian sectors; alone it keeps the weighted sectors below, so the former
bits of an LDA+U run take ``PARSEC_LDAU_WEIGHTED_SECTORS=off`` too, for a cell
``PARSEC_GAMMA_LDAU_SECTORS=off`` (its default before that date) and for a
spin run ``PARSEC_LDAU_SPIN_SECTORS=off`` (the default again since the evening
of that date).

Weighted sectors (``PARSEC_LDAU_WEIGHTED_SECTORS``, since 2026-11-21).  The
one-dimensional sectors of a subgroup ``H`` of the point group ``G`` with
weights solve one partner of some degenerate levels; the occupation matrices
of their weighted states are averaged over ``G``, as their density is, which
gives those of every partner, and the potential matrices of those are
covariant under ``G`` (:class:`HubbardGroupAverage`; the LDA+U forces are
averaged alike).  A result lists every partner with its own occupation; its
occupations are summed back per solved state where the orbitals carry the
weights (:meth:`DeviceHubbard._channel_blocks`).

Units: Ry, bohr; projectors and orbitals carry PARSEC's ``sqrt(h^3)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import scipy.sparse as sp

from parsec_python.Hamiltonian.hubbard import (
    HUBBARD_ORBITALS,
    HubbardOperator,
    HubbardProjectors,
    HubbardTerms,
    d_harmonic_rotation,
    hubbard_terms,
    symmetrize_occupations,
)
from parsec_python.V_ion import NonlocalProjectorOperator


#: The switch of the projector orbits of LDA+U in the symmetry sectors (see the module text).
LDAU_PROJECTOR_ORBITS_SWITCH = "PARSEC_LDAU_PROJECTOR_ORBITS"

#: Distance (bohr) within which an image of an LDA+U atom names the atom of its orbit, at least.  The orbits
#: themselves were found by the detection of the group at its atom tolerance; this only tells which atom an
#: operation reaches (:func:`_image_tolerance`).
_ORBIT_IMAGE_TOLERANCE = 1.0e-4


def _image_tolerance() -> float:
    """The distance (bohr) within which an image of an LDA+U atom names an atom: the larger of
    :data:`_ORBIT_IMAGE_TOLERANCE` and ``PARSEC_SYMMETRY_ATOM_TOLERANCE``, the tolerance at which the group of the
    sectors was detected (at most: the hybrid and k-point routes take a smaller one).  A structure accepted within a
    loose tolerance (1e-3 bohr, say) has atoms that far off their images, which a match at 1e-4 bohr alone refused
    (an error at the preparation of every such LDA+U run in the sectors before 2026-11-21 evening)."""

    from ..Symmetry.axis_reflection import symmetry_atom_tolerance

    return max(_ORBIT_IMAGE_TOLERANCE, float(symmetry_atom_tolerance()))


#: Distance (bohr) within which the atoms must keep the operations of the sectors for an LDA+U run to take the
#: columns of one atom per orbit (that of the forces and moves in the sectors, ``Forces/sector_symmetry.MOVE_TOLERANCE``).
LDAU_EXACT_ORBIT_TOLERANCE = 1.0e-6


def ldau_orbits_exact(reduction, atoms, grid) -> bool:
    """Whether the atoms keep the operations of the sectors of ``reduction`` within
    :data:`LDAU_EXACT_ORBIT_TOLERANCE`, where the reduction knows the orbits of its atoms.

    The columns of one atom per orbit give the sector blocks of every atom only for a structure that keeps the group;
    one accepted within a loose ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` (1e-3 bohr, say) with atoms off it by more gives
    every atom of an orbit the offset of its representative.  The driver then gives an LDA+U run the columns of
    every atom, as before 2026-11-21 (``driver._without_atom_orbits``): fcc Ag108 with one atom 5e-4 bohr off,
    ``PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-3``, A100: the energy of the full grid of that structure to 1.3e-3 Ry with
    the orbits (as without LDA+U, 1.2e-3 Ry) and 7.7e-5 Ry with every atom's columns (the abelian route of before
    alike), the run 9.3 against 10.7 s.  ``True`` at once at a detection tolerance of at most that distance (every
    operation detected maps the atoms within it) and for a reduction without orbits.
    """

    from ..Symmetry.axis_reflection import (
        abelian_sector_operations,
        atom_image_table,
        sector_orbits_exact_setting,
        symmetry_atom_tolerance,
    )

    if reduction is None or getattr(reduction, "atom_representatives", None) is None or not atoms:
        return True
    if float(symmetry_atom_tolerance()) <= LDAU_EXACT_ORBIT_TOLERANCE:
        return True
    translations = getattr(reduction, "translations", None)
    images = atom_image_table(
        tuple(atoms),
        abelian_sector_operations(reduction),
        None if translations is None else np.asarray(translations, dtype=np.int64).reshape(-1, 3),
        grid,
        LDAU_EXACT_ORBIT_TOLERANCE,
        # the distance itself; under PARSEC_SECTOR_ORBITS_EXACT=off the search of before, up to twice it
        within_tolerance=sector_orbits_exact_setting() != "off",
    )
    return images is not None


def _lda_u_atom_images(atoms, operations, translations, grid, what: str) -> np.ndarray:
    """``images[g, k]`` of :func:`..Symmetry.axis_reflection.atom_image_table` for the LDA+U atoms ``atoms`` at
    :func:`_image_tolerance`; every row a permutation, else ``RuntimeError`` naming ``what`` (the operations)."""

    from ..Symmetry.axis_reflection import _permutation_rows, atom_image_table

    images = atom_image_table(atoms, operations, translations, grid, _image_tolerance())
    if images is None or not _permutation_rows(images):
        raise RuntimeError(f"an operation of {what} does not map the LDA+U atoms onto themselves")
    return images


def ldau_projector_orbits_setting() -> str:
    """``PARSEC_LDAU_PROJECTOR_ORBITS``: ``auto`` (default) or ``off`` (``0``, ``false``, ``no`` mean ``off``)."""

    value = os.environ.get(LDAU_PROJECTOR_ORBITS_SWITCH, "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "off"):
        raise ValueError(f"{LDAU_PROJECTOR_ORBITS_SWITCH} must be auto or off, not {value!r}")
    return value


@dataclass(frozen=True)
class HubbardOrbits:
    """What the sector operators of a reduction with atom orbits hold of the LDA+U term (see the module text).

    ``kept``: the positions ``k`` (in ``HubbardProjectors.atom_indices``, ascending) of the LDA+U atoms whose
    fifteen columns the sector operators keep, the representatives of their orbits; ``sources[k]``: the position
    of the representative of the orbit of atom ``k``; ``rotations[k]``: ``D(R)`` of an operation of the sectors
    that carries that representative onto atom ``k`` (the identity for a representative); ``kb_columns``: the KB
    columns in front of the kept LDA+U columns in the sector operators.
    """

    kept: np.ndarray
    sources: np.ndarray
    rotations: np.ndarray
    kb_columns: int

    def expand(self, occupations: np.ndarray) -> np.ndarray:
        """``N_k = D_k N_(sources[k]) D_k^T`` of every LDA+U atom from ``occupations`` of the kept ones."""

        kept = np.asarray(occupations, dtype=np.float64)
        if kept.shape != (self.kept.size, HUBBARD_ORBITALS, HUBBARD_ORBITALS):
            raise ValueError("the occupations of the representatives have the wrong shape")
        slot = np.full(int(self.sources.size), -1, dtype=np.int64)
        slot[self.kept] = np.arange(self.kept.size)
        return np.einsum(
            "amk,akl,anl->amn", self.rotations, kept[slot[self.sources]], self.rotations
        )


#: The switch of LDA+U in the weighted sectors of a subgroup of the point group (see :class:`HubbardGroupAverage`).
LDAU_WEIGHTED_SECTORS_SWITCH = "PARSEC_LDAU_WEIGHTED_SECTORS"
LDAU_WEIGHTED_SECTORS_DEFAULT = "auto"


def ldau_weighted_sectors_setting() -> str:
    """``PARSEC_LDAU_WEIGHTED_SECTORS``: ``auto`` or ``off`` (``0``, ``false``, ``no`` mean ``off``).

    ``auto``: an LDA+U run takes the one-dimensional sectors of a subgroup ``H`` of the point group with weights
    where the plan of the group does (``driver._orbital_group_refusal``, ``Symmetry/orbital_groups.py``), with the
    occupation matrices averaged over the full group (:class:`HubbardGroupAverage`); ``off``: the abelian group, as
    before.
    """

    value = os.environ.get(LDAU_WEIGHTED_SECTORS_SWITCH, LDAU_WEIGHTED_SECTORS_DEFAULT).strip().lower()
    value = value or LDAU_WEIGHTED_SECTORS_DEFAULT
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "off"):
        raise ValueError(f"{LDAU_WEIGHTED_SECTORS_SWITCH} must be auto or off, not {value!r}")
    return value


@dataclass(frozen=True)
class HubbardGroupAverage:
    """The average over the full point group ``G`` of the occupation matrices and LDA+U forces of weighted sectors.

    The weighted sectors (:class:`..Symmetry.PointGroupReduction`) solve one partner of some degenerate levels of
    ``G`` in the one-dimensional sectors of a subgroup ``H`` and count it with the weight of its sector: the
    density of the weighted states is invariant under ``H`` only, and its average over ``G`` is that of all the
    partners.  The occupation matrices are linear in the density matrix, so the matrices of all the partners are the
    average ``N_b = (1/|G|) sum_g D(R_g) N_(g^-1 b) D(R_g)^T``
    (:func:`parsec_python.Hamiltonian.hubbard.symmetrize_occupations`), and the potential matrices of those are
    covariant under ``G``: ``H_U`` commutes with every operation and its blocks in the sectors of ``H`` are exact.
    The LDA+U forces of the weighted states at those potential matrices are averaged alike,
    ``F_b = (1/|G|) sum_g R_g F_(g^-1 b)``.  ``operations``: ``(D(R_g), image)`` for every ``g``, ``image[k]`` the
    LDA+U atom (position in ``HubbardProjectors.atom_indices``) that ``g`` carries atom ``k`` onto; ``rotations``:
    the Cartesian ``R_g``.
    """

    operations: tuple
    rotations: np.ndarray

    def occupations(self, matrices: np.ndarray) -> np.ndarray:
        return symmetrize_occupations(np.asarray(matrices, dtype=np.float64), self.operations)

    def forces(self, forces: np.ndarray) -> np.ndarray:
        """``(atoms, 3)`` forces of the LDA+U atoms averaged over ``G``."""

        forces = np.asarray(forces, dtype=np.float64)
        total = np.zeros_like(forces)
        for (_harmonics, image), rotation in zip(self.operations, self.rotations):
            total[image] += forces @ rotation.T
        return total / len(self.operations)


def sector_group_average(projectors: HubbardProjectors, reduction, atoms, grid) -> HubbardGroupAverage | None:
    """The :class:`HubbardGroupAverage` of weighted sectors (a reduction with ``full_operations``), ``None`` for
    the abelian groups, whose sectors solve every partner."""

    full = None if reduction is None else getattr(reduction, "full_operations", None)
    if full is None:
        return None
    operations = np.asarray(full, dtype=np.float64).reshape(-1, 3, 3)
    translations = getattr(reduction, "full_translations", None)
    indices = np.asarray(projectors.atom_indices, dtype=np.int64)
    images = _lda_u_atom_images(
        [atoms[int(atom)] for atom in indices],
        operations,
        None if translations is None else np.asarray(translations, dtype=np.int64).reshape(-1, 3),
        grid,
        "the point group",
    )
    turned: dict[bytes, np.ndarray] = {}
    pairs = []
    for g, rotation in enumerate(operations):
        key = np.rint(rotation).astype(np.int64).tobytes()
        if key not in turned:
            turned[key] = d_harmonic_rotation(rotation)
        pairs.append((turned[key], np.ascontiguousarray(images[g], dtype=np.int64)))
    return HubbardGroupAverage(operations=tuple(pairs), rotations=np.ascontiguousarray(operations))


def sector_hubbard_orbits(projectors: HubbardProjectors, kb_labels, reduction, atoms, grid) -> HubbardOrbits | None:
    """The :class:`HubbardOrbits` of the sector operators of ``reduction``, or ``None`` where they hold every column.

    The columns kept are those of
    :func:`parsec_python.acceleration.Symmetry.axis_reflection.compress_projectors_for_sectors` applied to the KB
    factor of ``kb_labels`` with the LDA+U columns appended (:func:`augmented_nonlocal_operator`): ``None`` where
    that function leaves the factor as it is (no reduction, no orbits, ``PARSEC_SECTOR_PROJECTOR_ORBITS=off``, a
    label outside the atoms, or every column kept).  The operation of each atom is found among the operations of
    the abelian sectors (:func:`..Symmetry.axis_reflection.abelian_sector_operations`, with the translations of a
    cell) by the images of the LDA+U atoms (:func:`..Symmetry.axis_reflection.atom_image_table`).
    """

    from ..Symmetry.axis_reflection import abelian_sector_operations, sector_projector_orbits_setting

    representatives = None if reduction is None else getattr(reduction, "atom_representatives", None)
    sizes = None if reduction is None else getattr(reduction, "atom_orbit_sizes", None)
    if representatives is None or sizes is None or sector_projector_orbits_setting() == "off":
        return None
    representatives = np.asarray(representatives, dtype=np.int64)
    indices = np.asarray(projectors.atom_indices, dtype=np.int64)
    kb_atoms = np.fromiter((int(label[0]) for label in kb_labels), dtype=np.int64)
    columns = np.concatenate((kb_atoms, np.repeat(indices, EXPANSION_COLUMNS)))
    if not columns.size or columns.min() < 0 or columns.max() >= representatives.size:
        return None
    if np.all(representatives[columns] == columns):
        return None
    kept = np.flatnonzero(representatives[indices] == indices)
    position = {int(atom): k for k, atom in enumerate(indices)}
    try:
        sources = np.array([position[int(representatives[atom])] for atom in indices], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError("an orbit of LDA+U atoms has a representative without LDA+U") from error
    operations = abelian_sector_operations(reduction)
    translations = getattr(reduction, "translations", None)
    images = _lda_u_atom_images(
        [atoms[int(atom)] for atom in indices],
        operations,
        None if translations is None else np.asarray(translations, dtype=np.int64).reshape(-1, 3),
        grid,
        "the sectors",
    )
    rotations = np.empty((indices.size, HUBBARD_ORBITALS, HUBBARD_ORBITALS), dtype=np.float64)
    turned: dict[int, np.ndarray] = {}
    for k in range(indices.size):
        if sources[k] == k:
            rotations[k] = np.eye(HUBBARD_ORBITALS)
            continue
        found = np.flatnonzero(images[:, sources[k]] == k)
        if not found.size:
            raise RuntimeError("no operation of the sectors carries an LDA+U atom onto the atoms of its orbit")
        g = int(found[0])
        if g not in turned:
            turned[g] = d_harmonic_rotation(np.asarray(operations[g], dtype=np.float64))
        rotations[k] = turned[g]
    for array in (kept, sources, rotations):
        array.setflags(write=False)
    return HubbardOrbits(
        kept=kept,
        sources=sources,
        rotations=rotations,
        kb_columns=int(np.count_nonzero(representatives[kb_atoms] == kb_atoms)),
    )


#: Angular label of the appended LDA+U columns: ``(atom, -2, k)``.  No KB
#: channel has a negative angular momentum.
HUBBARD_COLUMN_CHANNEL = -2

#: The index pairs ``i < j`` of the ten off-diagonal terms, in this order.
_PAIRS = tuple(
    (first, second)
    for first in range(HUBBARD_ORBITALS)
    for second in range(first + 1, HUBBARD_ORBITALS)
)

#: Columns per LDA+U atom: the five projectors and the ten sums of pairs.
EXPANSION_COLUMNS = HUBBARD_ORBITALS + len(_PAIRS)

#: Device bytes of one column chunk of the overlaps.
_CHUNK_BYTES = 256 * 1024 * 1024


def expansion_vectors() -> np.ndarray:
    """``(15, 5)``: the rows ``e_i`` and then ``e_i + e_j`` for ``i < j``."""

    vectors = np.zeros((EXPANSION_COLUMNS, HUBBARD_ORBITALS), dtype=np.float64)
    vectors[:HUBBARD_ORBITALS] = np.eye(HUBBARD_ORBITALS)
    for index, (first, second) in enumerate(_PAIRS):
        vectors[HUBBARD_ORBITALS + index, (first, second)] = 1.0
    return vectors


def expansion_coefficients(potentials: np.ndarray) -> np.ndarray:
    """Coefficients of the fifteen columns of every atom for ``D`` (``(atoms, 5, 5)``).

    ``sum_k c_k v_k v_k^T = (D + D^T)/2`` for the rows ``v_k`` of
    :func:`expansion_vectors`; atom after atom, as the columns are appended.
    """

    matrices = np.asarray(potentials, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (HUBBARD_ORBITALS, HUBBARD_ORBITALS):
        raise ValueError("potentials must have shape (atoms, 5, 5)")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    coefficients = np.empty((matrices.shape[0], EXPANSION_COLUMNS), dtype=np.float64)
    off_diagonal = symmetric.copy()
    off_diagonal[:, np.arange(HUBBARD_ORBITALS), np.arange(HUBBARD_ORBITALS)] = 0.0
    coefficients[:, :HUBBARD_ORBITALS] = (
        np.einsum("amm->am", symmetric) - np.sum(off_diagonal, axis=2)
    )
    for index, (first, second) in enumerate(_PAIRS):
        coefficients[:, HUBBARD_ORBITALS + index] = symmetric[:, first, second]
    return coefficients.reshape(-1)


def require_symmetric_term(projectors: HubbardProjectors) -> None:
    """Refuse ``ldaplusu_j != 0``, whose ``D`` the device operators cannot hold."""

    if np.any(np.asarray(projectors.hubbard_j) != 0.0):
        raise NotImplementedError(
            "LDA+U with ldaplusu_j != 0 is not implemented on the accelerated path: "
            "PARSEC's J term (the P3/P4 tensors on the real d harmonics) gives a "
            "potential matrix that is not symmetric, and the device eigensolvers "
            "apply symmetric separable terms only; run it with the reference solver"
        )


def expansion_columns(projectors: sp.spmatrix, atom_count: int) -> sp.csc_matrix:
    """``P E`` for ``P`` with five columns per atom: fifteen columns per atom."""

    vectors = expansion_vectors()
    block = sp.block_diag([sp.csr_matrix(vectors.T)] * int(atom_count), format="csr")
    columns = sp.csc_matrix(sp.csc_matrix(projectors, dtype=np.float64) @ block)
    columns.sum_duplicates()
    columns.sort_indices()
    return columns


def augmented_nonlocal_operator(
    nonlocal_operator: NonlocalProjectorOperator, projectors: HubbardProjectors
) -> NonlocalProjectorOperator:
    """The KB factor with the fifteen LDA+U columns of every atom appended.

    The appended columns have coefficient zero (``D = 0``, as in the first
    iteration of every SCF loop) and the labels ``(atom, -2, k)``.
    """

    require_symmetric_term(projectors)
    hubbard = expansion_columns(projectors.projectors, projectors.atom_count)
    kb = sp.csc_matrix(nonlocal_operator.projectors, dtype=np.float64)
    if kb.shape[0] != hubbard.shape[0]:
        raise ValueError("the LDA+U projectors and the KB factor have different grids")
    combined = sp.hstack([kb, hubbard], format="csc")
    combined.sort_indices()
    signs = np.concatenate(
        (
            np.asarray(nonlocal_operator.signs, dtype=np.float64),
            np.zeros(hubbard.shape[1], dtype=np.float64),
        )
    )
    labels = tuple(nonlocal_operator.labels) + tuple(
        (int(atom), HUBBARD_COLUMN_CHANNEL, column)
        for atom in projectors.atom_indices
        for column in range(EXPANSION_COLUMNS)
    )
    return NonlocalProjectorOperator(projectors=combined, signs=signs, labels=labels)


def orbital_nonlocal_operator(reference, selected_backend: str):
    """The factor the device Hamiltonians of ``reference`` are built with.

    The KB factor with the LDA+U columns appended where the backend is CuPy
    and the system has LDA+U atoms; ``None`` otherwise, which leaves the KB
    factor of ``reference`` (the CPU backends apply the term on the host,
    :class:`HostHubbardHamiltonian`).
    """

    projectors = getattr(reference, "hubbard", None)
    if projectors is None or selected_backend != "cupy":
        return None
    return augmented_nonlocal_operator(reference.nonlocal_operator, projectors)


def _transpose(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Host CSR of ``M.T`` with int32 indices: the layout the projection kernels read."""

    transpose = sp.csr_matrix(matrix, dtype=np.float64).T.tocsr(copy=True)
    transpose.sum_duplicates()
    transpose.sort_indices()
    if transpose.indices.dtype != np.int32 or transpose.indptr.dtype != np.int32:
        transpose = sp.csr_matrix(
            (
                transpose.data,
                transpose.indices.astype(np.int32),
                transpose.indptr.astype(np.int32),
            ),
            shape=transpose.shape,
        )
    return transpose


class _DeviceColumns:
    """``M.T X`` for one host CSR transpose, on the current device."""

    def __init__(self, transpose: sp.csr_matrix) -> None:
        from ..backends.cupy import require_cupy
        from ..backends.cupy_projectors import CuPySparseProjectorProjection

        cp, _ = require_cupy()
        self.count = int(transpose.shape[0])
        lengths = np.diff(transpose.indptr)
        self.projection = CuPySparseProjectorProjection(
            cp,
            (
                cp.asarray(transpose.indptr, dtype=cp.int32),
                cp.asarray(transpose.indices, dtype=cp.int32),
                cp.asarray(transpose.data, dtype=cp.float64),
                transpose.shape,
            ),
            cp.ones(self.count, dtype=cp.float64),
            max_row_entries=int(lengths.max(initial=0)),
        )

    def __call__(self, vectors):
        return self.projection(vectors)


class HostHubbardHamiltonian:
    """A bound CPU-backend Hamiltonian plus the host LDA+U term of one channel.

    The native and SciPy backends keep their KB factor inside their own
    storage; the term (:class:`parsec_python.Hamiltonian.hubbard.HubbardOperator`)
    is added to every application here, as
    :class:`parsec_python.Hamiltonian.operator.KohnShamHamiltonian` adds it.
    """

    def __init__(self, bound, term: HubbardOperator) -> None:
        self.bound = bound
        self.term = term

    @property
    def shape(self):
        return self.bound.shape

    def apply(self, vectors):
        return self.bound.apply(vectors) + self.term.apply(np.asarray(vectors, dtype=np.float64))

    def apply_kinetic(self, vectors):
        return self.bound.apply_kinetic(vectors)

    def apply_local(self, vectors):
        return self.bound.apply_local(vectors)

    def apply_nonlocal(self, vectors):
        return self.bound.apply_nonlocal(vectors) + self.term.apply(
            np.asarray(vectors, dtype=np.float64)
        )

    def as_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.apply,
            matmat=self.apply,
            rmatvec=self.apply,
            dtype=np.float64,
        )

    def as_eigensolver_operator(self):
        return self.as_linear_operator()

    def as_sparse(self) -> sp.csr_matrix:
        return (self.bound.as_sparse() + self.term.as_sparse()).tocsr()

    def __matmul__(self, vectors):
        return self.apply(vectors)


class DeviceHubbard:
    """The LDA+U term of a CuPy calculation (see the module text).

    ``projectors`` are the LDA+U projectors of the reference system,
    ``kb_count`` the number of KB columns in front of the appended ones,
    ``backend`` the :class:`CuPyHamiltonianBackend` and ``sectors`` its
    symmetry eigensolver, ``None`` on the full grid.  ``orbits``: what the
    sector operators hold where they carry one atom per orbit
    (:func:`sector_hubbard_orbits`), ``None`` where they hold every column.
    """

    def __init__(
        self,
        projectors: HubbardProjectors,
        kb_count: int,
        backend: Any,
        sectors: Any | None = None,
        orbits: HubbardOrbits | None = None,
        average: HubbardGroupAverage | None = None,
    ) -> None:
        require_symmetric_term(projectors)
        self.backend = backend
        self.sectors = sectors
        self.kb_count = int(kb_count)
        self.orbits = orbits
        # The average over the full point group of weighted sectors (None: the sectors solve every partner).
        self.average = average
        self._lock = Lock()
        self._set_projectors(projectors)
        # Seconds of the overlaps and of the coefficient uploads, for reports.
        self.occupation_seconds = 0.0
        self.bind_seconds = 0.0

    # ----- geometry -----------------------------------------------------

    def _set_projectors(self, projectors: HubbardProjectors) -> None:
        self.projectors = projectors
        self.atom_count = int(projectors.atom_count)
        self.column_count = EXPANSION_COLUMNS * self.atom_count
        with self._lock:
            # Host transposes per (sector or None, kind) and device
            # projections per (sector or None, kind, device).
            self._host: dict[tuple, sp.csr_matrix] = {}
            self._device: dict[tuple, _DeviceColumns] = {}

    def move(self, moved) -> NonlocalProjectorOperator:
        """Take the projectors of moved atoms; return the factor for the devices.

        ``moved`` is the reference system of the new structure.  The device
        operators install the returned factor with coefficient zero, which
        is ``D`` of the first iteration of the next SCF loop.  The sectors
        keep their group, so the orbits are those of the moved atoms under it.
        """

        projectors = moved.hubbard
        if projectors is None or tuple(projectors.atom_indices) != tuple(
            self.projectors.atom_indices
        ):
            raise ValueError("moved atoms must keep their LDA+U atoms")
        # The KB columns in front of the appended ones, of the new structure.
        self.kb_count = int(len(moved.nonlocal_operator.signs))
        if self.orbits is not None:
            self.orbits = sector_hubbard_orbits(
                projectors,
                moved.nonlocal_operator.labels,
                self.sectors.decomposition.reduction,
                moved.atoms,
                moved.grid,
            )
        if self.average is not None:
            self.average = sector_group_average(
                projectors, self.sectors.decomposition.reduction, moved.atoms, moved.grid
            )
        self._set_projectors(projectors)
        return augmented_nonlocal_operator(moved.nonlocal_operator, projectors)

    # ----- the term in the Hamiltonian ------------------------------------

    def _operators(self) -> Iterator[Any]:
        """Every device operator that holds the appended columns."""

        operator = getattr(self.backend, "device_operator", None)
        if operator is not None:
            yield operator
        if self.sectors is not None:
            for operator in self.sectors._operators:
                if operator is None:
                    continue
                yield operator
                worker = getattr(operator, "_distributed_filter", None)
                if worker is not None:
                    yield from worker.replicas.values()

    def coefficient_layouts(self, potentials: np.ndarray | None) -> dict[int, tuple[int, np.ndarray]]:
        """``{columns of an operator: (first LDA+U column, coefficients)}`` of ``D`` (``None``: zero).

        The factor of every atom (the full grid, and the sectors under ``PARSEC_LDAU_PROJECTOR_ORBITS=off``):
        the coefficients of every LDA+U atom behind the ``kb_count`` KB columns.  The sector operators of one atom
        per orbit (:attr:`orbits`): those of the representatives behind the KB columns they keep (their columns
        carry the square root of the orbit size).  The second has fewer columns, so the two never share a count.
        """

        coefficients = (
            np.zeros(self.column_count, dtype=np.float64)
            if potentials is None
            else expansion_coefficients(potentials)
        )
        layouts = {self.kb_count + self.column_count: (self.kb_count, coefficients)}
        orbits = self.orbits
        if orbits is not None:
            first = int(orbits.kb_columns)
            kept = (
                np.zeros(EXPANSION_COLUMNS * orbits.kept.size, dtype=np.float64)
                if potentials is None
                else expansion_coefficients(np.asarray(potentials, dtype=np.float64)[orbits.kept])
            )
            layouts.setdefault(first + kept.size, (first, kept))
        return layouts

    def bind(self, potentials: np.ndarray | None) -> None:
        """Write the coefficients of ``D`` (``(atoms, 5, 5)``, ``None``: zero) on the devices."""

        from time import perf_counter

        from ..backends.cupy import require_cupy

        started = perf_counter()
        cp, _ = require_cupy()
        layouts = self.coefficient_layouts(potentials)
        start, stop = self.kb_count, self.kb_count + self.column_count
        coefficients = layouts[stop][1]
        for host in (
            getattr(self.backend, "host_signs", None),
            getattr(getattr(self.backend, "host_nonlocal_operator", None), "signs", None),
        ):
            if host is not None and host.shape[0] == stop:
                host[start:stop] = coefficients
        for operator in self._operators():
            signs = operator.projector_signs
            layout = layouts.get(int(signs.shape[0]))
            if layout is None:
                raise RuntimeError("a device operator does not hold the LDA+U columns")
            first, values = layout
            last = first + values.size
            with cp.cuda.Device(int(signs.device.id)):
                signs[first:last] = cp.asarray(values)
                mixed = getattr(operator, "mixed_precision_recurrence", None)
                if mixed is not None:
                    mixed.projector_signs[first:last] = cp.asarray(
                        values, dtype=cp.float32
                    )
                # The sector solves run on streams of their own.
                cp.cuda.get_current_stream().synchronize()
        self.bind_seconds += perf_counter() - started

    def hamiltonian(self, backend, effective_potential, term: HubbardOperator | None):
        """Bind ``D`` of ``term`` (``None``: zero) and the local potential."""

        self.bind(None if term is None else term.potentials)
        return backend.bind(effective_potential)

    # ----- overlaps ---------------------------------------------------------

    def _left_atoms(self, kind: str) -> int:
        """The atoms of the projector columns ``kind``: every LDA+U atom (``P``), the representatives (``R``)."""

        return self.atom_count if kind == "P" else int(self.orbits.kept.size)

    def _host_columns(self, key, kind: str, gradients=None) -> sp.csr_matrix:
        """The CSR transpose of the columns ``kind`` (``P``, ``R``: the representatives of the orbits, or
        ``G0..G2``) of sector ``key``."""

        with self._lock:
            found = self._host.get((key, kind))
        if found is not None:
            return found
        if kind == "P":
            matrix = self.projectors.projectors
        elif kind == "R":
            columns = (
                HUBBARD_ORBITALS * np.asarray(self.orbits.kept, dtype=np.int64)[:, None]
                + np.arange(HUBBARD_ORBITALS)[None, :]
            ).reshape(-1)
            matrix = sp.csc_matrix(self.projectors.projectors)[:, columns]
        else:
            if gradients is None:
                raise RuntimeError("the projector gradients were not built")
            matrix = gradients[int(kind[1])]
        if key is not None:
            reduced = self.sectors.decomposition.reduce_nonlocal_operators(
                SimpleNamespace(
                    projectors=matrix,
                    signs=np.ones(matrix.shape[1], dtype=np.float64),
                    labels=tuple(range(matrix.shape[1])),
                ),
                (int(key),),
            )
            matrix = reduced[int(key)].projectors
        transpose = _transpose(matrix)
        with self._lock:
            self._host[(key, kind)] = transpose
        return transpose

    def _projection(self, key, kind: str, device: int, gradients=None) -> _DeviceColumns:
        with self._lock:
            found = self._device.get((key, kind, device))
        if found is not None:
            return found
        transpose = self._host_columns(key, kind, gradients)
        projection = _DeviceColumns(transpose)
        with self._lock:
            self._device[(key, kind, device)] = projection
        return projection

    def _drop_gradients(self) -> None:
        with self._lock:
            for table in (self._host, self._device):
                for name in [name for name in table if name[1] not in ("P", "R")]:
                    del table[name]

    def _block_products(self, key, device: int, block, weights, gradients=None, left_kind: str = "P"):
        """``sum_n f_n c c^T`` and, with ``gradients``, ``sum_n f_n g_c c^T`` of one block.

        ``block`` holds the columns of the weights ``weights`` on ``device``
        or on the host; the result is a host ``(atoms, 5, 5)`` array and,
        with gradients, a ``(3, atoms, 5, 5)`` one.  ``left_kind`` ``R``: the
        occupations of the representatives of the orbits alone (no gradients).
        """

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        atoms = self._left_atoms(left_kind)
        weights = np.asarray(weights, dtype=np.float64)
        with cp.cuda.Device(int(device)):
            left = self._projection(key, left_kind, int(device))
            rights = (
                ()
                if gradients is None
                else tuple(
                    self._projection(key, f"G{axis}", int(device), gradients)
                    for axis in range(3)
                )
            )
            occupation = cp.zeros((atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS), dtype=cp.float64)
            slopes = cp.zeros(
                (3, atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS), dtype=cp.float64
            )
            width = int(block.shape[1])
            rows = max(1, int(block.shape[0]))
            on_device = isinstance(block, cp.ndarray)
            per_column = 8 * (
                (rows if not on_device else 0)
                + (1 + len(rights)) * HUBBARD_ORBITALS * max(atoms, 1) * 2
            )
            chunk = max(1, int(_CHUNK_BYTES // max(per_column, 1)))
            for start in range(0, width, chunk):
                stop = min(width, start + chunk)
                factors = weights[start:stop]
                if not np.any(factors != 0.0):
                    continue
                columns = block[:, start:stop]
                if not on_device:
                    columns = cp.asarray(columns, dtype=cp.float64, order="F")
                count = stop - start
                overlaps = left(columns).reshape(atoms, HUBBARD_ORBITALS, count)
                weighted = overlaps * cp.asarray(factors, dtype=cp.float64)[None, None, :]
                transposed = cp.swapaxes(overlaps, 1, 2)
                occupation += cp.matmul(weighted, transposed)
                for axis, right in enumerate(rights):
                    slope = right(columns).reshape(atoms, HUBBARD_ORBITALS, count)
                    slopes[axis] += cp.matmul(
                        slope * cp.asarray(factors, dtype=cp.float64)[None, None, :],
                        transposed,
                    )
            occupation_host = np.asarray(cp.asnumpy(occupation), dtype=np.float64)
            slopes_host = (
                None
                if gradients is None
                else np.asarray(cp.asnumpy(slopes), dtype=np.float64)
            )
        return occupation_host, slopes_host

    def _sector_device(self, key, vectors) -> int:
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        if isinstance(vectors, cp.ndarray):
            return int(vectors.device.id)
        if key is not None and self.sectors is not None:
            return int(self.sectors._sector_device_ids[int(key)])
        return int(cp.cuda.Device().id)

    def _products(self, key, vectors, weights, record=None, gradients=None, left_kind: str = "P"):
        """The products of :meth:`_block_products` for one basis of any storage."""

        weights = np.asarray(weights, dtype=np.float64)
        atoms = self._left_atoms(left_kind)
        occupation = np.zeros((atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS))
        slopes = np.zeros((3, atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS))
        occupied = np.flatnonzero(weights != 0.0)
        if record is not None and not record.rotated:
            # A hand-over basis: all its columns with weight one, less the
            # Ritz vectors whose occupation is not one, weighted 1 - f
            # (handover.correction_set), as the density and the forces do.
            from ..Eigensolvers.handover import (
                CORRECTION_BYTES,
                correction_set,
                ritz_columns,
            )

            width = int(vectors.shape[1])
            if weights.size > width:
                raise RuntimeError("a sector no longer holds the orbitals of the last SCF iteration")
            _full, correction, corrections = correction_set(
                weights, np.arange(weights.size), width
            )
            total = self._products(key, vectors, np.ones(width), None, gradients, left_kind)
            occupation, slopes = total[0], total[1]
            step = max(1, CORRECTION_BYTES // (8 * max(1, int(vectors.shape[0]))))
            for start in range(0, correction.size, step):
                chunk = slice(start, start + step)
                ritz = ritz_columns(vectors, record, correction[chunk])
                part = self._products(key, ritz, -corrections[chunk], None, gradients, left_kind)
                occupation = occupation + part[0]
                slopes = slopes + part[1]
                del ritz
            return occupation, slopes
        if not occupied.size:
            return occupation, slopes
        count = int(occupied[-1]) + 1
        if count > int(vectors.shape[1]):
            raise RuntimeError("more occupations than stored orbital columns")
        weights = weights[:count]
        group = getattr(vectors, "group", None)
        if group is not None:
            selected = vectors[:, :count]

            def partial(index, device):
                block = selected.blocks[index]
                if not block.shape[1]:
                    return None
                held = [
                    weights[column : column + width]
                    for column, _offset, width in selected.layout.spans(index)
                ]
                return self._block_products(
                    key,
                    device,
                    block,
                    held[0] if len(held) == 1 else np.concatenate(held),
                    gradients,
                    left_kind,
                )

            # Fixed device order keeps the host sum deterministic.
            for part in group.map(partial):
                if part is not None:
                    occupation = occupation + part[0]
                    if part[1] is not None:
                        slopes = slopes + part[1]
            return occupation, slopes
        device = self._sector_device(key, vectors)
        part = self._block_products(key, device, vectors[:, :count], weights, gradients, left_kind)
        occupation = occupation + part[0]
        if part[1] is not None:
            slopes = slopes + part[1]
        return occupation, slopes

    def _channel_blocks(self, wavefunctions, occupations):
        """``(sector or None, basis, weights, hand-over)`` of the orbitals of one channel."""

        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

        occupations = np.asarray(occupations, dtype=np.float64)
        if isinstance(wavefunctions, CuPySymmetryOrbitals):
            repeats = getattr(wavefunctions, "state_weights", None)
            if repeats is not None:
                # The orbitals of a result of weighted sectors (SCF.single_point._public_states): its occupations
                # list every partner, f each; the sums over the partners of a solved state are the occupations
                # times the weights that the SCF loop gives the density and these matrices.
                repeats = np.asarray(repeats, dtype=np.int64)
                solved = getattr(wavefunctions, "state_occupations", None)
                if solved is not None and occupations.size in {int(repeats.sum()), repeats.size}:
                    # a level at the end of the window shares the states left: its listed partners are not those
                    # of single solved states, whose occupations the orbitals keep
                    occupations = np.asarray(solved, dtype=np.float64)
                elif occupations.size == int(repeats.sum()) and occupations.size != repeats.size:
                    starts = np.concatenate(([0], np.cumsum(repeats)[:-1]))
                    occupations = np.add.reduceat(occupations, starts)
            if wavefunctions.sector_vectors is None or wavefunctions.representation_columns is None:
                raise RuntimeError(
                    "the symmetry orbitals of this iteration were released before "
                    "their LDA+U occupation matrices were formed"
                )
            if any(item is not None for item in (getattr(wavefunctions, "sector_expansions", None) or ())):
                raise NotImplementedError(
                    "LDA+U reads the sector orbitals on the rows of their sectors; units of irreducible "
                    "representations on partner rows (PARSEC_ORBITAL_IRREP_ROWS=partner) are not supported with it "
                    "(PARSEC_ORBITAL_IRREP_ROWS=sector)"
                )
            handovers = wavefunctions.sector_handovers or ()
            for representation, vectors in enumerate(wavefunctions.sector_vectors):
                selected = np.flatnonzero(wavefunctions.representations == representation)
                if not selected.size:
                    continue
                columns = wavefunctions.representation_columns[selected]
                if not np.array_equal(columns, np.arange(columns.size)):
                    raise RuntimeError(
                        "the selected states of a sector are not its leading columns"
                    )
                record = handovers[representation] if representation < len(handovers) else None
                yield representation, vectors, occupations[selected], record
            return
        if getattr(wavefunctions, "ndim", 2) != 2:
            raise ValueError("orbitals must be a (grid, states) block")
        yield None, wavefunctions, occupations, None

    def occupation_matrices(self, wavefunctions, occupations, gradients=None):
        """``N`` of one channel (``(atoms, 5, 5)``, ``N[a, m', m]`` of the reference).

        With ``gradients`` (the three matrices of
        :func:`parsec_python.Forces.hubbard_force.build_hubbard_projector_gradients`)
        also ``A_c[a, m, k] = sum_n f_n <d_c phi_a,m|psi_n> <phi_a,k|psi_n>``.
        Sector orbitals whose operators carry one atom per orbit (``orbits``)
        give the matrices of the representatives, turned onto the other atoms
        of their orbits (:meth:`HubbardOrbits.expand`); the products of the
        forces read every atom.
        """

        from time import perf_counter

        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

        started = perf_counter()
        left_kind = (
            "R"
            if gradients is None and self.orbits is not None and isinstance(wavefunctions, CuPySymmetryOrbitals)
            else "P"
        )
        atoms = self._left_atoms(left_kind)
        occupation = np.zeros((atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS))
        slopes = np.zeros((3, atoms, HUBBARD_ORBITALS, HUBBARD_ORBITALS))
        for key, vectors, weights, record in self._channel_blocks(wavefunctions, occupations):
            if vectors is None:
                raise RuntimeError("a sector of these orbitals holds no basis")
            part = self._products(key, vectors, weights, record, gradients, left_kind)
            occupation = occupation + part[0]
            slopes = slopes + part[1]
        if left_kind == "R":
            occupation = self.orbits.expand(occupation)
        if self.average is not None and isinstance(wavefunctions, CuPySymmetryOrbitals):
            # weighted sectors: the matrices of every partner (HubbardGroupAverage)
            occupation = self.average.occupations(occupation)
        self.occupation_seconds += perf_counter() - started
        if gradients is None:
            return occupation
        return occupation, slopes

    def terms(
        self, wavefunctions: Sequence[Any], occupations: Sequence[np.ndarray]
    ) -> HubbardTerms:
        """``calcden.f90z`` of the orbitals of each channel, up first."""

        return hubbard_terms(
            self.projectors,
            [
                self.occupation_matrices(vectors, weights)
                for vectors, weights in zip(wavefunctions, occupations)
            ],
        )

    # ----- forces -------------------------------------------------------------

    def forces(self, system, result) -> np.ndarray:
        """The LDA+U forces (``(atoms, 3)``, Ry/bohr) of a converged accelerated result.

        :func:`parsec_python.Forces.hubbard_force.hubbard_forces` with the
        products formed on the devices: ``D`` of the final orbitals and

        ``F_a = (2/nspin) sum_s sum_(m,k) D^s_(m,k) (A^s_c[a,m,k] + A^s_c[a,k,m])``.
        """

        from parsec_python.Forces.hubbard_force import build_hubbard_projector_gradients
        from parsec_python.Hamiltonian.hubbard import hubbard_replica_lattice

        require_symmetric_term(self.projectors)
        channels = getattr(result, "spin", None)
        orbitals = (
            [(result.wavefunctions, result.occupations)]
            if channels is None
            else [(channel.wavefunctions, channel.occupations) for channel in channels]
        )
        # The images of a periodic cell, as the projectors were built.
        gradients = build_hubbard_projector_gradients(
            system.grid,
            self.projectors,
            system.atoms,
            system.pseudopotentials,
            hubbard_replica_lattice(system),
        )
        try:
            products = [
                self.occupation_matrices(vectors, weights, gradients)
                for vectors, weights in orbitals
            ]
        finally:
            self._drop_gradients()
        terms = hubbard_terms(self.projectors, [item[0] for item in products])
        factor = 2.0 / len(orbitals)
        forces = np.zeros((len(system.atoms), 3), dtype=np.float64)
        indices = list(self.projectors.atom_indices)
        hubbard = np.zeros((len(indices), 3), dtype=np.float64)
        for spin, (_occupation, slopes) in enumerate(products):
            matrix = terms.potentials[spin]
            for axis in range(3):
                value = np.einsum("amk,amk->a", matrix, slopes[axis]) + np.einsum(
                    "amk,akm->a", matrix, slopes[axis]
                )
                hubbard[:, axis] += factor * value
        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

        if self.average is not None and any(isinstance(vectors, CuPySymmetryOrbitals) for vectors, _ in orbitals):
            # weighted sectors: the forces of every partner (HubbardGroupAverage)
            hubbard = self.average.forces(hubbard)
        forces[indices] += hubbard
        return forces


def device_hubbard_for(reference, backend, sectors, selected_backend: str):
    """The :class:`DeviceHubbard` of a prepared accelerated system, or ``None``.

    ``None`` without LDA+U and on the CPU backends, which apply the term on
    the host.  Sectors whose reduction knows the orbits of its atoms carry one
    atom per orbit (:func:`sector_hubbard_orbits`; the driver drops the orbits
    under ``PARSEC_LDAU_PROJECTOR_ORBITS=off``).
    """

    projectors = getattr(reference, "hubbard", None)
    if projectors is None or selected_backend != "cupy":
        return None
    decomposition = getattr(sectors, "decomposition", None)
    orbits = (
        None
        if decomposition is None or ldau_projector_orbits_setting() == "off"
        else sector_hubbard_orbits(
            projectors,
            reference.nonlocal_operator.labels,
            decomposition.reduction,
            reference.atoms,
            reference.grid,
        )
    )
    average = (
        None
        if decomposition is None
        else sector_group_average(projectors, decomposition.reduction, reference.atoms, reference.grid)
    )
    return DeviceHubbard(
        projectors,
        len(reference.nonlocal_operator.signs),
        backend,
        sectors,
        orbits,
        average,
    )


__all__ = [
    "DeviceHubbard",
    "EXPANSION_COLUMNS",
    "HUBBARD_COLUMN_CHANNEL",
    "HostHubbardHamiltonian",
    "HubbardGroupAverage",
    "HubbardOrbits",
    "LDAU_PROJECTOR_ORBITS_SWITCH",
    "LDAU_WEIGHTED_SECTORS_DEFAULT",
    "LDAU_WEIGHTED_SECTORS_SWITCH",
    "ldau_orbits_exact",
    "ldau_projector_orbits_setting",
    "ldau_weighted_sectors_setting",
    "sector_group_average",
    "sector_hubbard_orbits",
    "augmented_nonlocal_operator",
    "device_hubbard_for",
    "expansion_coefficients",
    "expansion_vectors",
    "orbital_nonlocal_operator",
    "require_symmetric_term",
]
