"""Space group of a periodic cell, as Fortran PARSEC finds it.

Translation of ``symmetries.f90`` (lines 52-238) and ``symgen.f90``
(``symgen``, ``atftmt``, ``pgl``, ``rot``, ``rlv``, ``symm_ident``).  The
search tests a *fixed* set of candidate rotations: the 24 operations of the
hexagonal holohedry with the six-fold axis along z (``rot`` with ``nr = 24``)
or, when fewer than twelve of those leave the lattice invariant, the 48
operations of the cubic holohedry with the axes along x, y and z
(``nr = 48``).  A candidate is a lattice operation when it maps every lattice
vector onto a lattice vector (``pgl``, tolerance ``10 * 1e-8``); it is a
crystal operation when, with one fractional translation, it maps every atom
onto an atom of the same type (``atftmt``).  Because the candidates are fixed
and axis aligned, only cells in PARSEC's conventional orientation reveal their
full symmetry; spglib would find more for a rotated cell.

A crystal operation ``T`` acts on lattice (fractional) coordinates ``f`` as

``T(f) = M (f - t)``

with the integer matrix ``M`` (Fortran ``symm%rmtrx``) and the fractional
translation ``t`` (``symm%tnp``); in Cartesian coordinates ``T(r) = U (r - A t)``
with the orthogonal ``U = A M inv(A)`` (``symm%trans``) and the lattice vectors
``A`` as columns.  ``T`` is the inverse of the operation ``atftmt`` tested
(``x -> R x - v`` with ``t = -v``), so the set is the same group.  In
reciprocal-lattice coordinates a k-point rotates with
``G = transpose(M)`` (``symm%gmtrx``; ``B G k = R B k`` in Cartesian
coordinates).

The arithmetic follows the Fortran statement by statement (sums in Fortran's
order, each started from ``+0.0`` as ``dot_product`` does, ``nint``/``anint``
rounding halves away from zero): the fractional translations then carry the
same signs, including those of zero, that PARSEC prints.

Not ported: ``symchk`` (Fortran calls it only with an Abelian subgroup, which
the k-point runs do not build), the Abelian subgroup itself (``abelian``: this
port keeps every field on the full grid) and the multiplication table that
``atftmt`` prints at ``Output_Level >= 2``.

The search of :func:`find_space_group` is :func:`fast_crystal_operations` (a
neighbour tree; the operations and vectors of the port :func:`_crystal_operations`),
once per structure.  :func:`find_grid_space_group` keeps the operations that
also map a given grid onto itself (the irreducible k-points of the default
routines), and :func:`periodic_grid_symmetry` builds the density table from the
integer affine maps of the operations.

Units: bohr.  This module does not print; :mod:`parsec_python.Output.kpoint_output`
formats the report.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace

import numpy as np

from .point_group import GridPointGroup

ROOT3 = 1.73205080756887729352744634151
"""``root3`` of ``const.f90``."""

SYMMETRY_TOLERANCE = 1.0e-8
"""``eps`` of ``pgl`` and ``atftmt``."""

CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "cubic",
    "trigonal",
    "hexagonal",
)
"""``cst`` of ``atftmt``, indexed by the holohedral group number ``ihg - 1``."""

_AXES = {-2: "C2", -1: "C3", 0: "C4", 1: "C6", 2: "E "}


def fortran_anint(value: float) -> float:
    """Fortran ``anint``: nearest whole number, halves away from zero (exact)."""

    value = float(value)
    whole = math.trunc(value)
    fraction = value - whole
    if fraction >= 0.5:
        whole += 1
    elif fraction <= -0.5:
        whole -= 1
    # trunc keeps the sign of zero for values in (-1, 1): -0.3 -> -0.0.
    return math.copysign(float(whole), value) if whole == 0 else float(whole)


def fortran_nint(value: float) -> int:
    """Fortran ``nint``."""

    return int(fortran_anint(value))


def _matvec(matrix, vector) -> list[float]:
    """``matvec3('N', mat, vec)``: ``dot_product(mat(i,:), vec)`` from ``+0.0``."""

    return [
        ((0.0 + matrix[i][0] * vector[0]) + matrix[i][1] * vector[1])
        + matrix[i][2] * vector[2]
        for i in range(3)
    ]


def _matvec_transposed(matrix, vector) -> list[float]:
    """``matvec3('T', mat, vec)``: ``dot_product(mat(:,i), vec)`` from ``+0.0``."""

    return [
        ((0.0 + matrix[0][i] * vector[0]) + matrix[1][i] * vector[1])
        + matrix[2][i] * vector[2]
        for i in range(3)
    ]


def _mtrxin(m) -> list[list[float]]:
    """``mtrxin`` of ``symgen.f90``: inverse by cofactors (``stop 501`` if singular)."""

    a = [[0.0] * 3 for _ in range(3)]
    a[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    a[1][0] = -m[1][0] * m[2][2] + m[1][2] * m[2][0]
    a[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    a[0][1] = -m[0][1] * m[2][2] + m[0][2] * m[2][1]
    a[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    a[2][1] = -m[0][0] * m[2][1] + m[0][1] * m[2][0]
    a[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    a[1][2] = -m[0][0] * m[1][2] + m[0][2] * m[1][0]
    a[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    determinant = m[0][0] * a[0][0] + m[0][1] * a[1][0] + m[0][2] * a[2][0]
    if abs(determinant) < 1.0e-5:
        raise ValueError("the lattice vectors are (nearly) linearly dependent (PARSEC stop 501)")
    return [[a[i][j] / determinant for j in range(3)] for i in range(3)]


def lattice_columns(lattice_vectors) -> list[list[float]]:
    """``alatt`` (Fortran ``latt_vec``): the lattice vectors as columns."""

    rows = np.asarray(lattice_vectors, dtype=float)
    if rows.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    return [[float(rows[column][row]) for column in range(3)] for row in range(3)]


def fortran_reciprocal_lattice(lattice_vectors) -> np.ndarray:
    """Reciprocal vectors as rows, computed as ``inipbc.f90:61-71`` does.

    ``bvec = 2 pi inv(transpose(latt_vec))`` by ``mtrxin``; row ``i`` of the
    result is Fortran's column ``bvec(:,i)``.
    """

    columns = lattice_columns(lattice_vectors)
    transposed = [[columns[j][i] for j in range(3)] for i in range(3)]
    inverse = _mtrxin(transposed)
    twopi = 2.0 * math.pi
    bvec = [[twopi * inverse[i][j] for j in range(3)] for i in range(3)]
    return np.array([[bvec[row][column] for row in range(3)] for column in range(3)])


def candidate_rotations(count: int) -> list[list[list[float]]]:
    """``rot``: the 24 hexagonal (``count <= 24``) or 48 cubic rotations.

    Cartesian matrices, ``R[n][i][j]`` = Fortran ``r(n+1, i+1, j+1)``,
    generated with Fortran's products so that the hexagonal entries carry
    the same bits.
    """

    size = 24 if count <= 24 else 48
    r = [[[0.0] * 3 for _ in range(3)] for _ in range(size + 1)]

    def element(n, i, j):  # 1-based, as in the Fortran
        return r[n][i - 1][j - 1]

    def set_element(n, i, j, value):
        r[n][i - 1][j - 1] = value

    def accumulate(target, left, right, last):
        for i in range(1, last + 1):
            for j in range(1, last + 1):
                for k in range(1, last + 1):
                    set_element(
                        target, i, j,
                        element(target, i, j) + element(left, i, k) * element(right, k, j),
                    )

    if size == 24:
        f = ROOT3 * 0.5
        set_element(2, 1, 1, 0.5)
        set_element(2, 1, 2, -f)
        set_element(2, 2, 1, f)
        set_element(2, 2, 2, 0.5)
        set_element(7, 1, 1, -0.5)
        set_element(7, 1, 2, -f)
        set_element(7, 2, 1, -f)
        set_element(7, 2, 2, 0.5)
        for n in range(1, 7):
            set_element(n, 3, 3, 1.0)
            set_element(n + 18, 3, 3, 1.0)
            set_element(n + 6, 3, 3, -1.0)
            set_element(n + 12, 3, 3, -1.0)
        for i in range(1, 3):
            set_element(1, i, i, 1.0)
            for j in range(1, 3):
                set_element(6, i, j, element(2, j, i))
                for k in range(1, 3):
                    set_element(3, i, j, element(3, i, j) + element(2, i, k) * element(2, k, j))
                    set_element(8, i, j, element(8, i, j) + element(2, i, k) * element(7, k, j))
                    set_element(12, i, j, element(12, i, j) + element(7, i, k) * element(2, k, j))
        for i in range(1, 3):
            for j in range(1, 3):
                set_element(5, i, j, element(3, j, i))
                for k in range(1, 3):
                    set_element(4, i, j, element(4, i, j) + element(2, i, k) * element(3, k, j))
                    set_element(9, i, j, element(9, i, j) + element(2, i, k) * element(8, k, j))
                    set_element(10, i, j, element(10, i, j) + element(12, i, k) * element(3, k, j))
                    set_element(11, i, j, element(11, i, j) + element(12, i, k) * element(2, k, j))
        for n in range(1, 13):
            for i in range(1, 3):
                for j in range(1, 3):
                    set_element(n + 12, i, j, -element(n, i, j))
    else:
        set_element(9, 1, 3, 1.0)
        set_element(9, 2, 1, 1.0)
        set_element(9, 3, 2, 1.0)
        set_element(19, 1, 1, 1.0)
        set_element(19, 2, 3, -1.0)
        set_element(19, 3, 2, 1.0)
        for i in range(1, 4):
            set_element(1, i, i, 1.0)
            for j in range(1, 4):
                set_element(20, i, j, element(19, j, i))
                set_element(5, i, j, element(9, j, i))
                for k in range(1, 4):
                    set_element(2, i, j, element(2, i, j) + element(19, i, k) * element(19, k, j))
                    set_element(16, i, j, element(16, i, j) + element(9, i, k) * element(19, k, j))
                    set_element(23, i, j, element(23, i, j) + element(19, i, k) * element(9, k, j))
        for target, left, right in (
            (6, 2, 5), (7, 16, 23), (8, 5, 2), (10, 2, 9), (11, 9, 2),
            (12, 23, 16), (14, 16, 2), (15, 2, 16), (22, 23, 2), (24, 2, 23),
        ):
            accumulate(target, left, right, 3)
        for target, left, right in (
            (3, 5, 12), (4, 5, 10), (13, 23, 11), (17, 16, 12), (18, 16, 10), (21, 12, 15),
        ):
            accumulate(target, left, right, 3)
        for n in range(1, 25):
            for i in range(1, 4):
                for j in range(1, 4):
                    set_element(n + 24, i, j, -element(n, i, j))
    return [r[n] for n in range(1, size + 1)]


def _remove_lattice_vector(p, g) -> tuple[list[float], int]:
    """``rlv``: ``y = p g`` minus its nearest integer vector; ``l`` flags a removal."""

    y = _matvec(p, g)
    removed = sum(fortran_nint(abs(value)) for value in y)
    return [value - 1.0 * fortran_nint(value) for value in y], removed


def lattice_point_group(alatt, inverse) -> tuple[list, list[int], int, int]:
    """``pgl``: lattice operations among the candidate rotations.

    Returns ``(rotations, indices, ihg, ihc)``: the candidate set that was
    kept (hexagonal or cubic), the 1-based indices of the operations that map
    the lattice onto itself, the holohedral group number ``ihg`` (1..7, see
    :data:`CRYSTAL_SYSTEMS`) and ``ihc`` (0 hexagonal, 1 cubic candidates).
    """

    eps = SYMMETRY_TOLERANCE
    hexagonal_trial = True
    while True:
        rotations = candidate_rotations(24 if hexagonal_trial else 48)
        indices = []
        for number, rotation in enumerate(rotations, start=1):
            total = 0.0
            for column in range(3):
                xa = [
                    ((0.0 + rotation[i][0] * alatt[0][column]) + rotation[i][1] * alatt[1][column])
                    + rotation[i][2] * alatt[2][column]
                    for i in range(3)
                ]
                remainder, _ = _remove_lattice_vector(inverse, xa)
                for value in remainder:
                    total = total + abs(value)
            if total <= 10.0 * eps:
                indices.append(number)
        count = len(indices)
        if hexagonal_trial:
            if count == 12:
                return rotations, indices, 6, 0
            if count > 12:
                return rotations, indices, 7, 0
            hexagonal_trial = False
            continue
        if count == 16:
            return rotations, indices, 4, 1
        if count > 16:
            return rotations, indices, 5, 1
        if count == 4:
            return rotations, indices, 2, 1
        if count > 4:
            return rotations, indices, 3, 1
        return rotations, indices, 1, 1


def _fortran_mod(values: np.ndarray) -> np.ndarray:
    """Fortran ``mod(a, 1.0)`` (``fmod``: the sign of ``a``), elementwise."""

    return np.fmod(values, 1.0)


def _crystal_operations(inverse, positions, types, rotations, lattice_indices):
    """``atftmt``: the lattice operations that map the atoms onto themselves.

    ``positions`` is ``(3, atoms)`` Cartesian (Fortran ``x``), ``types`` the
    integer type of every atom.  Returns ``(indices, translations)``: the
    1-based candidate indices of the crystal operations, in the order tested,
    and for each the vector ``v`` (lattice coordinates) of the pair of atoms
    that matched first (Fortran ``v(:,l) = vr``; the fractional translation is
    ``tnp = -v``).
    """

    eps = SYMMETRY_TOLERANCE
    x = np.asarray(positions, dtype=np.float64)
    types = np.asarray(types)
    atoms = x.shape[1]
    same_type = types[:, None] == types[None, :]
    ai = [[float(value) for value in row] for row in inverse]
    found: list[int] = []
    translations: list[list[float]] = []
    for number in lattice_indices:
        rotation = rotations[number - 1]
        rx = np.empty_like(x)
        for i in range(3):
            rx[i] = ((0.0 + rotation[i][0] * x[0]) + rotation[i][1] * x[1]) + rotation[i][2] * x[2]
        # rdiff[k1, k2] = rlv(ai, R x_k1 - x_k2)  (Fortran rdiff(:, k2, k1)).
        xb = rx[:, :, None] - x[:, None, :]
        y = [
            ((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2]
            for i in range(3)
        ]
        rdiff = np.stack(
            [component - 1.0 * _nint_array(component) for component in y], axis=-1
        )
        match_vector = None
        give_up = False
        for k1 in range(atoms):
            for k2 in range(atoms):
                if types[k1] != types[k2]:
                    continue
                vr = rdiff[k1, k2]
                dif = np.zeros((atoms, atoms))
                for i in range(3):
                    dif = dif + _fortran_mod(np.abs(vr[i] - rdiff[:, :, i]) + eps)
                matched = (dif <= 10.0 * eps) & same_type
                has_partner = np.any(matched, axis=1)
                if has_partner.all():
                    match_vector = [float(value) for value in vr]
                    break
                if not has_partner[0]:
                    # The first atom found no partner: ks == 0, the
                    # operation is given up (atftmt's shortcut).
                    give_up = True
                    break
            if match_vector is not None or give_up:
                break
        if match_vector is None:
            continue
        found.append(number)
        translations.append(match_vector)
    return found, translations


def _nint_array(values: np.ndarray) -> np.ndarray:
    """Elementwise Fortran ``nint`` as floats (halves away from zero)."""

    whole = np.trunc(values)
    fraction = values - whole
    return whole + (fraction >= 0.5) - (fraction <= -0.5)


TRANSLATION_DENOMINATORS = 48
"""Largest denominator ``q`` of the fractions ``p/q`` a translation is snapped to."""

TRANSLATION_SNAP_TOLERANCE = 1.0e-8
"""A translation within this of ``p/q`` is that fraction (``eps`` of ``atftmt``)."""

TRANSLATION_ROUND_OFF = 1.0e-14
"""Deviations up to this are round-off of an exact input: the value is kept bit for bit."""


def snapped_translation(value: float) -> float:
    """``value`` as the simplest fraction ``p/q`` (``q <= 48``) within ``1e-8``.

    A crystal read from rounded coordinates has its fractional translations
    off a simple fraction by the rounding (``0.25 - 3.7e-12`` with ten
    decimals), which ``adjust_step`` cannot meet: its trial ``x = tau/nn``
    and the fitted step ``L/n`` never agree within ``1e-15``.  The smallest
    ``q`` with ``|value - p/q| <= 1e-8`` decides; where the deviation is
    round-off (at most ``1e-14``, an exact input) ``value`` itself is
    returned, so such inputs keep PARSEC's bits.  A value near no such
    fraction is returned unchanged.
    """

    value = float(value)
    for denominator in range(1, TRANSLATION_DENOMINATORS + 1):
        numerator = math.floor(value * denominator + 0.5)
        fraction = numerator / denominator
        deviation = abs(value - fraction)
        if deviation <= TRANSLATION_SNAP_TOLERANCE:
            return value if deviation <= TRANSLATION_ROUND_OFF else fraction
    return value


def _labels(rotations) -> tuple[str, ...]:
    """``symm_ident``: E, C2, C3, C4, C6 and their products with the inversion."""

    labels = []
    for matrix in rotations:
        a = [[float(value) for value in row] for row in matrix]
        det = (
            a[0][0] * a[1][1] * a[2][2]
            + a[1][0] * a[2][1] * a[0][2]
            + a[0][1] * a[1][2] * a[2][0]
            - a[2][0] * a[1][1] * a[0][2]
            - a[1][0] * a[0][1] * a[2][2]
            - a[0][0] * a[2][1] * a[1][2]
        )
        trace = a[0][0] + a[1][1] + a[2][2]
        if fortran_nint(det) == 1:
            labels.append(_AXES[fortran_nint(trace - 1.0)].ljust(3))
        else:
            labels.append(("I" + _AXES[fortran_nint(-trace - 1.0)]).ljust(3)[:3])
    return tuple(labels)


@dataclass(frozen=True)
class SpaceGroup:
    """The symmetry operations of a periodic cell (Fortran ``symm``).

    Every field is a tuple so the object is hashable and compares by value
    (it may sit inside a frozen :class:`~parsec_python.models.SinglePointInput`).

    ``rotations`` are the integer matrices ``M`` (lattice coordinates,
    ``rmtrx``) and ``translations`` the fractional translations ``t``
    (``tnp``) of ``T(f) = M (f - t)``; ``cartesian_rotations`` are ``U``
    (``symm%trans``).  ``labels`` are ``symm_ident``'s names (three
    characters, e.g. ``"IC2"``).  ``crystal_system`` is the holohedry of the
    lattice, ``full_holohedry`` says whether the crystal has all of its
    operations; ``lattice_operations`` are the 1-based candidate numbers of
    the lattice operations (what PARSEC prints when the group is not the full
    holohedry), ``candidate_numbers`` those of the crystal operations.
    ``symmorphic`` is PARSEC's verdict, which looks at the translation of the
    *last* operation only (``atftmt``: ``vs`` is overwritten in its loop).
    ``ignored`` marks the identity group of ``Ignore_Symmetry``.
    """

    rotations: tuple[tuple[tuple[int, int, int], ...], ...]
    translations: tuple[tuple[float, float, float], ...]
    cartesian_rotations: tuple[tuple[tuple[float, float, float], ...], ...]
    labels: tuple[str, ...]
    crystal_system: str = "triclinic"
    full_holohedry: bool = False
    lattice_operations: tuple[int, ...] = (1,)
    candidate_numbers: tuple[int, ...] = (1,)
    symmorphic: bool = True
    ignored: bool = False

    @property
    def order(self) -> int:
        return len(self.rotations)

    @property
    def rotation_matrices(self) -> np.ndarray:
        """``(order, 3, 3)`` integer ``M`` (``rmtrx``)."""

        return np.asarray(self.rotations, dtype=np.int64).reshape(-1, 3, 3)

    @property
    def reciprocal_rotations(self) -> np.ndarray:
        """``(order, 3, 3)`` integer ``G = transpose(M)`` (``gmtrx``)."""

        return np.transpose(self.rotation_matrices, (0, 2, 1)).copy()

    @property
    def translation_vectors(self) -> np.ndarray:
        """``(order, 3)`` fractional translations ``t`` (``tnp``)."""

        return np.asarray(self.translations, dtype=np.float64).reshape(-1, 3)

    def commensurate_translations(self, directions: int = 3, *, tolerance: float = 1.0e-8):
        """The translations ``adjust_step`` makes the grid commensurate with.

        ``inipbc.f90:381-391``: for every operation the components of
        ``|M t|`` larger than ``tolerance``, per lattice direction, in the
        order of the operations, each snapped to a simple fraction
        (:func:`snapped_translation`): coordinates written with ten decimals
        give ``0.2499999999963`` for diamond's quarter, for which no step
        settles (``adjust_step`` alternates between ``x L`` and ``L/n``).
        """

        lists: list[list[float]] = [[] for _ in range(directions)]
        for matrix, translation in zip(self.rotations, self.translations):
            mt = _matvec([[float(v) for v in row] for row in matrix], translation)
            for direction in range(directions):
                value = mt[direction]
                if value < 0.0:
                    value = -value
                if value > tolerance:
                    lists[direction].append(snapped_translation(float(value)))
        return tuple(tuple(values) for values in lists)


def identity_space_group(*, ignored: bool = True) -> SpaceGroup:
    """The group of the identity alone (``symmetries.f90`` with ``info = 2``)."""

    return SpaceGroup(
        rotations=(((1, 0, 0), (0, 1, 0), (0, 0, 1)),),
        translations=((0.0, 0.0, 0.0),),
        cartesian_rotations=(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),),
        labels=("E  ",),
        ignored=ignored,
    )


def skipped_space_group() -> SpaceGroup:
    """The identity of a run whose symmetry PARSEC does not search.

    ``symmetries.f90`` with ``info = 2`` without ``Ignore_Symmetry`` (a slab,
    molecular dynamics, Polarizability, manual movements: ``parsec.F90:470-486``):
    "skipping non-trivial symmetry operations".  Marked by an empty
    ``crystal_system`` (no lattice was examined).
    """

    return replace(identity_space_group(ignored=False), crystal_system="")


@dataclass(frozen=True)
class _CellSearch:
    """What ``symgen`` searches with: ``alatt``, its inverse, the atoms (``(3, atoms)`` Cartesian, Fortran ``x``,
    and their integer types) and the lattice operations among the candidate rotations (:func:`lattice_point_group`)."""

    alatt: list
    inverse: list
    positions: np.ndarray
    types: list
    rotations: list
    lattice_indices: list
    holohedral: int


def _cell_search(lattice_vectors, atoms) -> _CellSearch:
    """``symmetries.f90:120-170``: the atoms in lattice coordinates and back, the lattice operations.

    Atom types are the species symbols in order of first appearance (PARSEC's ``Atom_Type`` blocks), atoms of one
    type in input order (``coorat``).
    """

    alatt = lattice_columns(lattice_vectors)
    # symmetries.f90: invlat = inv(transpose(alatt)); coorat = invlat^T r.
    invlat = _mtrxin([[alatt[j][i] for j in range(3)] for i in range(3)])
    order: list[str] = []
    grouped: dict[str, list] = {}
    for atom in atoms:
        if atom.symbol not in grouped:
            order.append(atom.symbol)
            grouped[atom.symbol] = []
        grouped[atom.symbol].append(atom)
    columns = []
    types = []
    for type_index, symbol in enumerate(order):
        for atom in grouped[symbol]:
            position = [float(value) for value in np.asarray(atom.position, dtype=float)]
            fractional = _matvec_transposed(invlat, position)
            # symgen: x = a * coorat, summed over the lattice vectors.
            cartesian = [
                ((0.0 + alatt[k][0] * fractional[0]) + alatt[k][1] * fractional[1])
                + alatt[k][2] * fractional[2]
                for k in range(3)
            ]
            columns.append(cartesian)
            types.append(type_index)
    positions = np.array(columns, dtype=np.float64).T
    inverse = _mtrxin(alatt)
    rotations, lattice_indices, holohedral, _hexagonal_flag = lattice_point_group(alatt, inverse)
    return _CellSearch(alatt, inverse, positions, types, rotations, lattice_indices, holohedral)


def _lattice_matrix(rotation, alatt, inverse) -> tuple[tuple[int, int, int], ...]:
    """``rmtrx`` of a Cartesian rotation: ``r1 = transpose(R) a``, ``rlat = b r1``, ``nint(rlat)`` (``symgen``)."""

    r1 = [
        [
            ((0.0 + rotation[0][j] * alatt[0][k]) + rotation[1][j] * alatt[1][k])
            + rotation[2][j] * alatt[2][k]
            for k in range(3)
        ]
        for j in range(3)
    ]
    rlat = [
        [
            ((0.0 + inverse[j][0] * r1[0][k]) + inverse[j][1] * r1[1][k])
            + inverse[j][2] * r1[2][k]
            for k in range(3)
        ]
        for j in range(3)
    ]
    return tuple(tuple(fortran_nint(value) for value in row) for row in rlat)


def _assembled_space_group(search: _CellSearch, numbers, vectors) -> SpaceGroup:
    """``symgen.f90:111-221``: the :class:`SpaceGroup` of the crystal operations ``numbers`` and their vectors."""

    if not numbers or numbers[0] != 1:
        raise ValueError(
            "the symmetry search did not find the identity (overlapping atoms?); "
            "PARSEC stops with 'SYMMETRY GROUP DOES NOT HAVE IDENTITY'"
        )
    matrices = []
    cartesian = []
    translations = []
    for number, vector in zip(numbers, vectors):
        rotation = search.rotations[number - 1]
        matrices.append(_lattice_matrix(rotation, search.alatt, search.inverse))
        # symm%trans = transpose(trans), trans = R.
        cartesian.append(tuple(tuple(float(rotation[j][i]) for j in range(3)) for i in range(3)))
        translations.append(tuple(-value for value in vector))
    holohedral = search.holohedral
    full = (holohedral == 7 and len(numbers) == 24) or (holohedral == 5 and len(numbers) == 48)
    last = vectors[-1]
    symmorphic = not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE)
    return SpaceGroup(
        rotations=tuple(matrices),
        translations=tuple(translations),
        cartesian_rotations=tuple(cartesian),
        labels=_labels(matrices),
        crystal_system=CRYSTAL_SYSTEMS[holohedral - 1],
        full_holohedry=full,
        lattice_operations=tuple(search.lattice_indices),
        candidate_numbers=tuple(numbers),
        symmorphic=symmorphic,
    )


def space_group_with_search(lattice_vectors, atoms, search=None) -> SpaceGroup:
    """:func:`find_space_group` with the search ``search`` (default :func:`fast_crystal_operations`), not cached.

    ``search=_crystal_operations`` runs the statement-by-statement port of ``atftmt``, the reference the fast
    search is tested against.
    """

    cell = _cell_search(lattice_vectors, atoms)
    numbers, vectors = (search or fast_crystal_operations)(
        cell.inverse, cell.positions, cell.types, cell.rotations, cell.lattice_indices
    )
    return _assembled_space_group(cell, numbers, vectors)


_SPACE_GROUP_CACHE: dict = {}
_SPACE_GROUP_CACHE_SIZE = 8


def _structure_key(lattice_vectors, atoms) -> tuple:
    """The lattice and the atoms (symbols, positions) bit for bit, a key of the caches of this module."""

    lattice = np.ascontiguousarray(np.asarray(lattice_vectors, dtype=np.float64))
    positions = np.ascontiguousarray(
        np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms]).reshape(-1, 3)
    )
    return (lattice.tobytes(), tuple(atom.symbol for atom in atoms), positions.tobytes())


def _cached(key, compute):
    """``compute()`` once per ``key`` (the last :data:`_SPACE_GROUP_CACHE_SIZE` keys; a group is immutable)."""

    if key in _SPACE_GROUP_CACHE:
        return _SPACE_GROUP_CACHE[key]
    value = compute()
    if len(_SPACE_GROUP_CACHE) >= _SPACE_GROUP_CACHE_SIZE:
        _SPACE_GROUP_CACHE.pop(next(iter(_SPACE_GROUP_CACHE)))
    _SPACE_GROUP_CACHE[key] = value
    return value


def find_space_group(lattice_vectors, atoms) -> SpaceGroup:
    """Space group of the atoms in a three-dimensional periodic cell.

    ``lattice_vectors`` are the rows ``a1, a2, a3`` in bohr and ``atoms`` the
    :class:`~parsec_python.models.Atom` objects at their input coordinates.
    Atom types are the species symbols in order of first appearance (PARSEC's
    ``Atom_Type`` blocks), atoms of one type in input order (``coorat``).
    Translation of ``symmetries.f90:120-170`` with ``nops = -1`` (fractional
    translations included) followed by ``symgen.f90:111-221``.

    The crystal operations come from :func:`fast_crystal_operations`, the
    search of ``atftmt`` with a neighbour tree (the same operations and
    vectors as the port :func:`_crystal_operations`, in ``O(N log N)`` per
    trial instead of ``(N, N)`` arrays: 1 s and a few MB for 4,096 atoms).
    The group of a structure is computed once (the parser, the choice of the
    routines and the report ask for it): the result is cached by the lattice
    and the atoms, bit for bit.
    """

    return _cached(
        ("space group",) + _structure_key(lattice_vectors, atoms),
        lambda: space_group_with_search(lattice_vectors, atoms),
    )


# ----------------------------------------------------------------------------
# The crystal operations, fast (atftmt with a neighbour tree)
# ----------------------------------------------------------------------------

# Atoms of the sample test of _maps_all_atoms.
_SAMPLE_ATOMS = 64


def fast_crystal_operations(inverse, positions, types, rotations, lattice_indices, *, admissible=None):
    """``atftmt``'s search with a neighbour tree: same operations, same vectors.

    Arguments and result as :func:`_crystal_operations`.  For an
    operation ``R`` the Fortran loop takes the first atom pair ``(k1, k2)``
    whose vector ``v = rlv(R x_k1 - x_k2)`` maps every atom onto an atom of
    its type (within ``sum_i mod(|dv_i| + eps, 1) <= 10 eps``).  With
    ``k1 = 1`` some ``k2`` succeeds whenever the operation is a symmetry
    (``k2`` the image of the first atom), and a pair with ``k1 > 1`` maps the
    first atom as well, so the operation and its vector are those of the
    first successful ``k2`` of the first atom.  The test of a vector looks
    up every image in a periodic k-d tree of the fractional coordinates
    (L1 metric) and then applies Fortran's criterion to the hits.

    ``admissible(number, vectors)`` (optional) returns a boolean mask of the
    candidate vectors ``vectors`` ``(k2, 3)`` of candidate ``number`` that
    may be taken (:func:`find_grid_space_group`: those that map a grid onto
    itself); the first admissible vector that maps the atoms is kept.
    """

    from scipy.spatial import cKDTree

    eps = SYMMETRY_TOLERANCE
    x = np.asarray(positions, dtype=np.float64)
    types = np.asarray(types)
    ai = [[float(value) for value in row] for row in inverse]

    def fractional(vectors):
        return np.stack(
            [((0.0 + ai[i][0] * vectors[0]) + ai[i][1] * vectors[1]) + ai[i][2] * vectors[2] for i in range(3)],
            axis=-1,
        )

    fx = fractional(x)
    folded = fx - np.floor(fx)
    folded[folded >= 1.0] = 0.0
    trees = {}
    members = {}
    for kind in np.unique(types):
        index = np.flatnonzero(types == kind)
        members[kind] = index
        trees[kind] = cKDTree(folded[index], boxsize=1.0)
    first_type = types[0]
    candidates_k2 = np.flatnonzero(types == first_type)
    # Look-up radius in L1: Fortran accepts sum |dv| <= 7 eps (plus wrap).
    radius = 1.0e-6

    found: list[int] = []
    translations: list[list[float]] = []
    for number in lattice_indices:
        rotation = rotations[number - 1]
        rx = np.empty_like(x)
        for i in range(3):
            rx[i] = ((0.0 + rotation[i][0] * x[0]) + rotation[i][1] * x[1]) + rotation[i][2] * x[2]
        frx = fractional(rx)
        match_vector = None
        # rdiff(:, k2, 1) = rlv(ai, R x_1 - x_k2) of every candidate k2, with
        # Fortran's operations in its order (elementwise, so the same bits).
        xb = rx[:, :1] - x[:, candidates_k2]
        y = [((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2] for i in range(3)]
        vectors = np.stack([value - 1.0 * _nint_array(value) for value in y], axis=-1)
        # The first test of _maps_all_atoms (the image of the last atom) for
        # all candidates in one tree query: a large cluster has thousands of
        # candidates per operation that is not a symmetry, and one query per
        # candidate dominated the search.  The survivors are tested in full,
        # in Fortran's order, so the first match is the same.
        last = x.shape[1] - 1
        targets = frx[last][None, :] - vectors
        targets = targets - np.floor(targets)
        targets[targets >= 1.0] = 0.0
        probe, _ = trees[types[last]].query(targets, k=1, p=1, distance_upper_bound=radius)
        survivors = np.isfinite(probe)
        if admissible is not None:
            survivors &= np.asarray(admissible(number, vectors), dtype=bool)
        for candidate in np.flatnonzero(survivors):
            vr = vectors[candidate]
            if _maps_all_atoms(vr, frx, fx, types, trees, members, radius, eps, rx, x, ai):
                match_vector = [float(value) for value in vr]
                break
        if match_vector is None:
            continue
        found.append(number)
        translations.append(match_vector)
    return found, translations


def _maps_all_atoms(vr, frx, fx, types, trees, members, radius, eps, rx, x, ai) -> bool:
    """Whether every atom ``k3`` has a partner ``k4`` with ``rdiff(k3, k4) ~ vr``.

    The nearest stored atom (periodic L1 distance of the fractional
    coordinates) of every image is the only candidate that can pass
    Fortran's criterion, which is then applied to it.  The last atom is
    tested first: a wrong vector almost always fails there.
    """

    targets = frx - vr[None, :]
    folded = targets - np.floor(targets)
    folded[folded >= 1.0] = 0.0
    last = folded.shape[0] - 1
    probe_kind = types[last]
    distance, _ = trees[probe_kind].query(folded[last], k=1, p=1, distance_upper_bound=radius)
    if not np.isfinite(distance):
        return False
    count = folded.shape[0]
    if count > _SAMPLE_ATOMS:
        # A spread sample first: a lattice translation of a large crystallite
        # maps the last atom and most inner atoms onto atoms, and the full
        # test of each such candidate would cost a query of every atom.  The
        # sample's test is part of the full one, so the answer is the same.
        sample = np.unique(np.linspace(0, count - 1, _SAMPLE_ATOMS).astype(np.int64))
        if not _atoms_map(sample, vr, folded, types, trees, members, radius, eps, rx, x, ai):
            return False
    return _atoms_map(np.arange(count), vr, folded, types, trees, members, radius, eps, rx, x, ai)


def _atoms_map(selection, vr, folded, types, trees, members, radius, eps, rx, x, ai) -> bool:
    """Whether the atoms ``selection`` have partners ``k4`` with ``rdiff(k3, k4) ~ vr``.

    ``folded`` holds the folded fractional images of every atom.
    """

    for kind, tree in trees.items():
        own = selection[types[selection] == kind]
        if own.size == 0:
            continue
        distances, local = tree.query(folded[own], k=1, p=1, distance_upper_bound=radius)
        if not np.all(np.isfinite(distances)):
            return False
        partners = members[kind][local]
        # rdiff(:, k4, k3) and Fortran's comparison with vr, elementwise.
        xb = rx[:, own] - x[:, partners]
        y = [((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2] for i in range(3)]
        dif = np.zeros(own.size)
        for i in range(3):
            vt = y[i] - 1.0 * _nint_array(y[i])
            da = np.abs(vr[i] - vt) + eps
            dif = dif + np.fmod(da, 1.0)
        if not np.all(dif <= 10.0 * eps):
            return False
    return True


GRID_MAP_TOLERANCE = 1.0e-6
"""An operation maps a grid point onto a point when its image is a whole number of steps within this."""


def find_grid_space_group(lattice_vectors, atoms, counts, shift) -> SpaceGroup:
    """The operations of the crystal that map the atoms AND a periodic grid onto themselves.

    ``counts`` are the points along the three lattice vectors and ``shift`` the offset of the first point (grid
    steps): point ``i`` has the fractional coordinates ``f = (i + s)/n``.  An operation ``T(f) = M (f - t)`` maps the
    grid onto itself when ``M (f - t) n - s`` is a whole vector for every ``f``
    (:func:`grid_commensurability_problem`).  A rotation of the crystal comes with several translations (``t`` plus
    any lattice translation of the crystal smaller than the cell, as the centring vectors of a conventional cell or
    the cells of a supercell), so the search of :func:`fast_crystal_operations` keeps for every lattice rotation the
    first image of the first atom whose translation maps the grid onto itself and every atom onto an atom of its
    type.  Every operation of the crystal that maps the grid onto itself maps the first atom onto some atom, so a
    rotation is missed only when it has no such translation; and products of operations that map the grid onto
    itself do so again: the rotations found form a group, which with time reversal reduces a k-point mesh and
    symmetrizes the density on that grid (the default routines, which keep the grid of ``Grid_Spacing``; the
    Fortran-derived routines change the grid instead, ``adjust_step``).

    Example: diamond in the conventional cube with 18 points per edge, an atom at the origin and the points half a
    step off it (Andrew Grieder's grid).  The 24 operations with the translation ``a/4 (1, 1, 1)`` (4.5 steps) map the
    grid onto the unshifted one; the 24 of ``Td`` about the atom do not move the grid.  ``Td`` with time reversal
    reduces a Monkhorst-Pack mesh as ``Oh`` does.  The result is cached as :func:`find_space_group`'s.
    """

    counts = tuple(int(value) for value in np.asarray(counts).reshape(-1))
    shift = tuple(float(value) for value in np.asarray(shift, dtype=np.float64).reshape(-1))
    if len(counts) != 3 or len(shift) != 3 or min(counts) < 1:
        raise ValueError("a grid of three lattice directions: three counts and three shifts")
    return _cached(
        ("grid space group", counts, shift) + _structure_key(lattice_vectors, atoms),
        lambda: _grid_space_group(lattice_vectors, atoms, counts, shift),
    )


def _grid_space_group(lattice_vectors, atoms, counts, shift) -> SpaceGroup:
    cell = _cell_search(lattice_vectors, atoms)
    n = np.asarray(counts, dtype=np.float64)
    s = np.asarray(shift, dtype=np.float64)
    corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64)
    fractional_corners = (corners + s) / n

    def admissible(number, vectors):
        matrix = np.asarray(
            _lattice_matrix(cell.rotations[number - 1], cell.alatt, cell.inverse), dtype=np.float64
        )
        # The operation is x -> R x - A vr, so t = -vr: M (f - t) = M (f + vr).
        mapped = (fractional_corners[None, :, :] + vectors[:, None, :]) @ matrix.T
        target = mapped * n - s
        return np.max(np.abs(target - np.rint(target)), axis=(1, 2)) <= GRID_MAP_TOLERANCE

    numbers, vectors = fast_crystal_operations(
        cell.inverse, cell.positions, cell.types, cell.rotations, cell.lattice_indices, admissible=admissible
    )
    return _assembled_space_group(cell, numbers, vectors)


ATOM_IMAGE_TOLERANCE = 1.0e-6
"""``tol`` of ``symm_forces.f90``: an atom and its image coincide within it
(a length in lattice coordinates)."""


def space_group_atom_images(
    group: SpaceGroup,
    lattice_vectors,
    atoms,
    *,
    tolerance: float = ATOM_IMAGE_TOLERANCE,
) -> np.ndarray:
    """``(order, atoms)`` index of the atom each operation maps every atom onto.

    ``symm_forces.f90:62-91``: operation ``T`` takes atom ``a`` to
    ``cro = U (R_a - A t)`` (``symop``) and its image is the first atom ``b``
    of the same type with ``inv(A) (cro - R_b)`` a lattice vector within
    ``tolerance`` (lattice coordinates, ``nint`` removed).  ``lattice_vectors``
    are the rows ``a1, a2, a3`` in bohr; positions may lie outside the cell.
    Raises ``ValueError`` (Fortran error 411) when an atom has no image, or
    when an operation does not permute the atoms (two atoms within
    ``tolerance`` of each other).
    """

    from scipy.spatial import cKDTree

    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    columns = vectors.T
    inverse = np.linalg.inv(columns)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms]).reshape(-1, 3)
    symbols = [atom.symbol for atom in atoms]
    count = len(symbols)
    images = np.empty((group.order, count), dtype=np.int64)
    rotations = np.asarray(group.cartesian_rotations, dtype=np.float64).reshape(-1, 3, 3)
    # The atoms of each species in a periodic k-d tree of their lattice coordinates (the box of one cell): the
    # distance of the tree is the norm of the difference with nint removed from each component, Fortran's test,
    # for all atoms at once instead of a loop over pairs (1,728 atoms and 24 operations: 100 s in that loop).
    species: dict[str, np.ndarray] = {}
    for index, symbol in enumerate(symbols):
        species.setdefault(symbol, []).append(index)
    trees = {}
    for symbol, members in species.items():
        members = np.asarray(members, dtype=np.int64)
        species[symbol] = members
        trees[symbol] = cKDTree(_unit_box(positions[members] @ inverse.T), boxsize=1.0)
    for number, (rotation, translation) in enumerate(
        zip(rotations, group.translation_vectors)
    ):
        shift = columns @ translation
        mapped = _unit_box((positions - shift) @ rotation.T @ inverse.T)
        for symbol, members in species.items():
            for index, near in zip(members, trees[symbol].query_ball_point(mapped[members], r=tolerance)):
                if not near:
                    raise ValueError(
                        f"symmetry operation {number + 1} maps atom {index + 1} at "
                        f"{positions[index].tolist()} onto no atom (PARSEC error 411: "
                        "unable to find equivalent of atom)"
                    )
                images[number, index] = members[min(near)]  # the first atom of the species, as the loop takes it
        if np.unique(images[number]).size != count:
            raise ValueError(
                f"symmetry operation {number + 1} does not permute the atoms"
            )
    return images


FIXED_SPECIES_SUFFIX = "|fixed"
"""What :func:`atoms_with_fixed_species` appends to the symbol of a fixed atom (the label of the accelerated
driver's detection, ``acceleration.driver.FIXED_ATOM_LABEL``)."""


def atoms_with_fixed_species(atoms, movable) -> tuple:
    """``atoms`` with those ``movable`` marks ``False`` as a species of their own, for a symmetry search.

    A search of the space group of these atoms (:func:`find_space_group`, :func:`find_grid_space_group`) finds the
    operations that map the fixed atoms onto fixed atoms, each rotation with a translation that does so where one
    exists: the subgroup a relaxation whose ``Move_Flag`` fixes atoms keeps (its forces and moved positions averaged
    over it stay invariant under it).  The group of the atoms themselves holds one translation per rotation, which
    need not fix them (in a supercell any lattice vector of the smaller cell may be added), so its operations alone
    do not give that subgroup.  The labels are read by the search only.
    """

    flags = [bool(flag) for flag in movable]
    if len(flags) != len(atoms):
        raise ValueError("one movable flag per atom is required")
    return tuple(
        atom if moves else replace(atom, symbol=atom.symbol + FIXED_SPECIES_SUFFIX)
        for atom, moves in zip(atoms, flags)
    )


def respects_flags(group: SpaceGroup, lattice_vectors, atoms, movable) -> bool | None:
    """Whether every operation of ``group`` maps the atoms ``movable`` marks onto atoms it marks alike.

    ``None`` where an operation maps an atom onto no atom (:func:`space_group_atom_images` raises).
    """

    mask = np.asarray([bool(flag) for flag in movable], dtype=bool)
    if mask.size != len(atoms):
        raise ValueError("one movable flag per atom is required")
    try:
        images = space_group_atom_images(group, lattice_vectors, atoms)
    except ValueError:
        return None
    return all(np.array_equal(mask[row], mask) for row in images)


def _unit_box(fractions: np.ndarray) -> np.ndarray:
    """Lattice coordinates in ``[0, 1)`` for a periodic k-d tree of box 1 (``mod`` can round up to 1)."""

    wrapped = np.mod(fractions, 1.0)
    wrapped[wrapped >= 1.0] = 0.0
    return wrapped


@dataclass(frozen=True)
class PeriodicGridSymmetry(GridPointGroup):
    """Space-group operations as permutations of the points of a periodic grid.

    :func:`~parsec_python.Grid.point_group.symmetrize_scalar` averages a field
    over them (``symm_scalar`` of ``newrho.F90``).  ``reduced_order`` is one:
    the periodic grid of this port is never reduced to a wedge, as PARSEC's
    grid of a k-point run is not (its Abelian subgroup is the identity), so the
    printed deviation is relative to the sum over the whole cell.

    ``images`` is the table ``(operations, points)`` of the image rows, or, where every operation is an integer
    affine map of a row-major grid (``affine_maps``) and ``PARSEC_PERIODIC_IMAGE_ROWS`` takes them
    (:func:`periodic_grid_symmetry`), an :class:`AffineImageRows` that computes a row where it is read and holds only
    the rows of its oblique operations (48 operations of 10^7 points are 1.9 GB of int32 as a table).  The readers take
    the rows through :meth:`image_row`, :meth:`image_rows`, :meth:`gather` and :meth:`accumulate`, which give the
    integers and values of the table either way.
    """

    affine_maps: tuple | None = field(default=None, kw_only=True, compare=False, repr=False)
    """``((M, c), ...)`` of every operation where all are integer affine maps of the grid indices: the point
    ``i`` goes to ``(M i + c) mod n`` (row-major order of the points), as ``images`` holds them.  A device
    version can apply them without the table (:mod:`parsec_python.acceleration.SCF.periodic_symmetry`).
    ``None`` where the table was built otherwise."""

    counts: tuple | None = field(default=None, kw_only=True, compare=False, repr=False)
    """The points per lattice direction of the grid of ``affine_maps``."""

    @property
    def reduced_order(self) -> int:
        return 1

    @property
    def size(self) -> int:
        """The points of the grid (the length of a row of ``images``)."""

        return int(self.images.shape[1])

    @property
    def holds_table(self) -> bool:
        """Whether ``images`` is a table in memory (``False`` for :class:`AffineImageRows`)."""

        return isinstance(self.images, np.ndarray)

    @property
    def table_bytes(self) -> int:
        """The bytes of the table ``images`` (held, or computed row by row where :attr:`holds_table` is false)."""

        operations, points = self.images.shape
        return int(operations) * int(points) * int(np.dtype(self.images.dtype).itemsize)

    @property
    def held_bytes(self) -> int:
        """The bytes of the images held now: the table, or what an :class:`AffineImageRows` keeps (its index vectors
        and the rows of its oblique operations, :attr:`AffineImageRows.held_bytes`; never more than the table)."""

        if isinstance(self.images, AffineImageRows):
            return self.images.held_bytes
        return int(np.asarray(self.images).nbytes)

    def image_row(self, operation: int) -> np.ndarray:
        """``images[operation]``: the row of the image of every point under the operation."""

        if isinstance(self.images, AffineImageRows):
            return self.images.row(operation)
        return self.images[operation]

    def image_rows(self, operation: int, rows) -> np.ndarray:
        """``images[operation, rows]`` for an index array ``rows``, without the whole row where none is held."""

        if isinstance(self.images, AffineImageRows):
            return self.images.at(operation, rows)
        return self.images[operation][np.asarray(rows)]

    def gather(self, values, operation: int) -> np.ndarray:
        """``values[..., images[operation]]`` of an array ``(..., points)``: a new array (with the points along the last axis
        innermost in memory, as the gather of a row of the table lays it out), the same values."""

        if isinstance(self.images, AffineImageRows):
            return self.images.gather(values, operation)
        return np.asarray(values)[..., self.images[operation]]

    def accumulate(self, total: np.ndarray, values: np.ndarray, operation: int) -> None:
        """``total += values[images[operation]]`` (float arrays of the points), the same sum element by element."""

        if isinstance(self.images, AffineImageRows):
            self.images.accumulate(total, values, operation)
            return
        total += values[self.images[operation]]

    def image_sum(self, values) -> np.ndarray:
        """``sum_g values[images[g]]`` over the operations in order, from a zero field (the sum of
        :func:`~parsec_python.Grid.point_group.symmetrize_scalar` before its division by the order)."""

        values = np.asarray(values, dtype=np.float64)
        if isinstance(self.images, AffineImageRows):
            return self.images.image_sum(values)
        total = np.zeros_like(values)
        for images in self.images:
            total += values[images]
        return total


def grid_fractional_points(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(integer points, steps along each lattice direction, shift)`` of a grid.

    The fractional coordinate of point ``i`` along direction ``d`` is
    ``(i_d + shift_d) / n_d``, the ``rw`` of ``symm_scalar``, with the grid's
    own shift: Fortran's (0.5 unless ``Ignore_Symmetry``, or the ``Grid_Shift``
    block) on every grid of the Fortran-derived routines, the orthorhombic one
    included since PARSEC 1.5's parity round (it started at the origin before).
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(getattr(grid, "shift", (0.0, 0.0, 0.0)), dtype=np.float64)
    return points, counts, shift


def grid_commensurability_problem(group: SpaceGroup, counts, shift) -> str | None:
    """Why ``group`` does not map a periodic grid onto itself, or ``None``.

    ``counts`` are the points along the three lattice directions and
    ``shift`` the grid shift (units of the steps).  An operation maps the
    point ``f = (i + s)/n`` to ``M (f - t)``, which must again be of that form:
    with ``M`` mixing only directions of equal ``n`` and ``s``, this needs
    every ``(M t)_d n_d`` to be a whole number (``adjust_step``).
    """

    counts = np.asarray(counts, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    # Test on the eight corners of one grid cell: enough for an affine map.
    corners = np.array(
        [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64
    )
    fractional = (corners + shift) / counts
    for number, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors), start=1
    ):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = mapped * counts - shift
        if np.max(np.abs(target - np.rint(target))) > 1.0e-6:
            return (
                f"operation {number} ({group.labels[number - 1].strip()}, fractional "
                "translation "
                + " ".join(f"{value:.5f}" for value in translation)
                + ") does not map the grid "
                + "x".join(str(int(value)) for value in counts)
                + " onto itself"
            )
    return None


def periodic_grid_symmetry(grid, group: SpaceGroup) -> PeriodicGridSymmetry:
    """Map the operations of ``group`` onto the points of a periodic grid.

    ``symm_scalar`` (``newrho.F90:262-283``): the point of fractional
    coordinates ``rw = (i + s)/n`` goes to ``rrw = M (rw - t)`` and to the
    integer point ``nint(rrw n - s)``, folded into the cell.  Raises
    ``ValueError`` where an operation does not map the grid onto itself.

    Where every operation is an integer affine map of a row-major grid, ``images`` is an :class:`AffineImageRows` that
    computes the rows where they are read: under ``PARSEC_PERIODIC_IMAGE_ROWS=auto`` (the default since 2026-11-21)
    for a grid of at least :data:`_IMAGE_ROWS_POINTS` points whose averages run on two threads or more (the periodic
    setup's, ``PARSEC_PERIODIC_SETUP_THREADS`` / ``OMP_NUM_THREADS``); a smaller grid (a table of at most 24 MiB for 48
    operations) and a run on one thread keep the table, whose gathers are faster there; ``on`` takes the rows at
    every size and thread count and keeps no row of an oblique operation either (the least memory, since
    2026-11-22: :class:`_ObliqueFactor`); ``off`` builds the table of operations x points as before.  The rows are
    the same integers.  A slab whose points fill a box takes them as well (:func:`confined_grid_symmetry`).
    """

    rows_setting = periodic_image_rows_setting()
    points, counts, shift = grid_fractional_points(grid)
    problem = grid_commensurability_problem(group, counts, shift)
    if problem is not None:
        raise ValueError(
            "Symmetrize_Charge_Density: " + problem + "; PARSEC's adjust_step makes "
            "the grid spacing commensurate with the fractional translations, which "
            "this cell's grid is not"
        )
    # One row of point indices per operation (int32: half the memory of the
    # table, operations x points, for any grid below 2**31 points).
    index_type = np.int32 if points.shape[0] < 2**31 - 1 else np.int64
    maps = [
        grid_affine_map(matrix, translation, counts, shift)
        for matrix, translation in zip(group.rotation_matrices, group.translation_vectors)
    ]
    row_major = all(entry is not None for entry in maps) and _row_major_points(grid, points, counts)
    if row_major and _image_rows_taken(rows_setting, points.shape[0]):
        # no table: the rows (M i + c) mod n of every operation are formed where they are read, the same integers
        return PeriodicGridSymmetry(
            matrices=group.rotation_matrices.astype(np.float64),
            images=AffineImageRows(maps, counts, index_type, lean=rows_setting == "on"),
            affine_maps=tuple(maps),
            counts=tuple(int(value) for value in counts),
        )
    images = np.empty((group.order, points.shape[0]), dtype=index_type)
    fractional = None
    for index, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors)
    ):
        if row_major:
            # nint(M (f - t) n - s) = M i + c exactly (the operation maps the grid onto itself), and with
            # det M = +-1 and M mixing only directions of equal n the map is a permutation of the points
            images[index] = affine_grid_images(*maps[index], counts, index_type)
            continue
        if fractional is None:
            fractional = (points.astype(np.float64) + shift) / counts
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = _nint_array(mapped * counts - shift).astype(np.int64)
        rows = grid.rows_for_integer_coordinates(target % counts)
        if np.bincount(rows, minlength=rows.size).max(initial=0) > 1:
            raise ValueError(
                f"Symmetrize_Charge_Density: operation {index + 1} does not permute "
                "the grid points"
            )
        images[index] = rows
    return PeriodicGridSymmetry(
        matrices=group.rotation_matrices.astype(np.float64),
        images=images,
        affine_maps=tuple(maps) if row_major else None,
        counts=tuple(int(value) for value in counts) if row_major else None,
    )


def grid_affine_map(matrix, translation, counts, shift, *, tolerance: float = GRID_MAP_TOLERANCE):
    """``(M, c)`` with ``nint(M ((i + s)/n - t) n - s) = M i + c`` for every grid index ``i``, or ``None``.

    ``M`` (integer, lattice coordinates) may mix only lattice directions of equal ``n`` (then the factors
    ``n_d/n_e`` are one), and ``c = M s - s - n (M t)`` must be whole within ``tolerance`` (the operation maps the
    grid onto itself).  With ``det M = +-1`` the map ``i -> (M i + c) mod n`` is then a permutation of the points;
    ``None`` where any of this fails (the caller falls back to rounding the images point by point).
    """

    matrix = np.asarray(matrix, dtype=np.int64).reshape(3, 3)
    counts = np.asarray(counts, dtype=np.int64).reshape(3)
    shift = np.asarray(shift, dtype=np.float64).reshape(3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    for row in range(3):
        for column in range(3):
            if matrix[row, column] != 0 and counts[row] != counts[column]:
                return None
    if abs(round(float(np.linalg.det(matrix.astype(np.float64))))) != 1:
        return None
    real = matrix.astype(np.float64)
    constant = real @ shift - shift - counts.astype(np.float64) * (real @ translation)
    whole = np.rint(constant)
    if np.max(np.abs(constant - whole)) > tolerance:
        return None
    return matrix, whole.astype(np.int64)


def affine_grid_indices(matrix, constant, counts, *, xp=np) -> list:
    """The three index arrays ``j_d = (M i + c)_d mod n_d`` of the image of every point ``i`` of a grid of ``counts``
    points, each of a shape that broadcasts to ``counts`` (a length along the axes its row of ``M`` reads, one along
    the others): ``values.reshape(counts)[j_0, j_1, j_2]`` is the field at the images, without a row of the points."""

    n = [int(value) for value in counts]
    axes = [
        xp.arange(n[axis], dtype=xp.int64).reshape([n[axis] if other == axis else 1 for other in range(3)])
        for axis in range(3)
    ]
    indices = []
    for row in range(3):
        component = xp.full((1, 1, 1), int(constant[row]), dtype=xp.int64)
        for column in range(3):
            if matrix[row][column]:
                component = component + int(matrix[row][column]) * axes[column]
        indices.append(component % n[row])
    return indices


def affine_grid_images(matrix, constant, counts, index_type=np.int64, *, xp=np):
    """Row of the image ``(M i + c) mod n`` of every point ``i`` of a row-major grid of ``counts`` points.

    Built from one-dimensional index vectors broadcast along the axes: a signed permutation of the axes costs one
    sum of three broadcast vectors per point, without the coordinates of the points.  ``xp`` is the array module
    (NumPy, or CuPy for the indices on a device).
    """

    n = [int(value) for value in counts]
    strides = (n[1] * n[2], n[2], 1)
    terms = [index * stride for index, stride in zip(affine_grid_indices(matrix, constant, n, xp=xp), strides)]
    size = n[0] * n[1] * n[2]
    if xp is not np or size < _THREADED_IMAGES:
        flat = terms[0] + terms[1] + terms[2]
        return xp.broadcast_to(flat, tuple(n)).reshape(-1).astype(index_type)
    # A large grid: the sum of the three broadcast terms written once into the result, by slabs of the first axis on
    # threads (NumPy leaves the lock in these sums); the same integers.
    result = np.empty(tuple(n), dtype=index_type)

    def part(term, first, last):
        return term[first:last] if term.shape[0] > 1 else term

    def fill(first: int, last: int) -> None:
        slab = result[first:last]
        np.add(part(terms[0], first, last), part(terms[1], first, last), out=slab, casting="unsafe")
        np.add(slab, part(terms[2], first, last), out=slab, casting="unsafe")

    _run_by_slabs(fill, n[0])
    return result.reshape(-1)


_THREADED_IMAGES = 1 << 22
"""Points of a grid from which :func:`affine_grid_images` fills its result by slabs on threads."""

_THREADED_SUMS = 1 << 17
"""Points of a grid from which :meth:`AffineImageRows.image_sum` and :meth:`AffineImageRows.gather` work by slabs on
threads (a laptop, 4 threads, 48 operations: 262,144 points 39 ms with the table, 57 ms unthreaded, 17 ms threaded;
46,656 points 4, 8 and 7 ms)."""

_IMAGE_ROWS_POINTS = 1 << 17
"""Points of a grid from which :func:`periodic_grid_symmetry` keeps no table under ``PARSEC_PERIODIC_IMAGE_ROWS=auto``
(below, the gathers of the table are faster than those of the index vectors: 48 operations on a laptop, 46,656
points 3.9 against 6.8 ms per average, 262,144 points 43 against 13 ms with 4 threads; the table of 48 operations of
2^17 points is 24 MiB)."""


def _image_rows_taken(setting: str, points: int) -> bool:
    """Whether a row-major affine grid group of ``points`` points takes :class:`AffineImageRows` under ``setting``
    (``PARSEC_PERIODIC_IMAGE_ROWS``): ``on`` always; ``auto`` from :data:`_IMAGE_ROWS_POINTS` points where the averages
    run on two threads or more (on one thread the gathers of a row of the table are faster than those of the index
    vectors: an A100 node, the diamond cube on 2.4 M points, 24 operations, a scalar average 0.206 s with the table
    and 0.305 s with the rows, a field of three components 1.07 and 1.40 s; on 16 threads 0.048 and 0.605 s with the
    rows); ``off`` never.  The rows are the same integers either way, so the ranks of an MPI run may differ in it
    without any change of their numbers."""

    if setting == "off":
        return False
    if setting == "on":
        return True
    from ..Laplacian.finite_difference import _build_threads

    return points >= _IMAGE_ROWS_POINTS and _build_threads() >= 2


def _run_by_slabs(work, count: int) -> None:
    """``work(first, last)`` over slabs of ``range(count)`` on the threads of the periodic setup."""

    from ..Laplacian.finite_difference import _build_threads

    threads = max(1, min(_build_threads(), count))
    if threads == 1 or threading.current_thread().name.startswith(_SLAB_THREAD_PREFIX):
        # (one thread, or a call from a slab of the kept pool itself, which must not wait on its own workers)
        work(0, count)
        return
    bounds = np.linspace(0, count, threads + 1).astype(int)
    pool = _slab_pool(threads)
    for future in [pool.submit(work, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]:
        future.result()


_SLAB_POOLS: dict = {}
"""The thread pools of :func:`_run_by_slabs` by ``(process id, threads)``: kept, so that the gathers of an average
(one call per operation for a vector field) do not start their threads each time; a forked process makes its own."""

_SLAB_POOL_LOCK = threading.Lock()

_SLAB_THREAD_PREFIX = "parsec-slabs"


def _slab_pool(threads: int):
    """The kept thread pool of ``threads`` workers of this process (:data:`_SLAB_POOLS`)."""

    import os
    from concurrent.futures import ThreadPoolExecutor

    key = (os.getpid(), int(threads))
    pool = _SLAB_POOLS.get(key)
    if pool is None:
        with _SLAB_POOL_LOCK:
            pool = _SLAB_POOLS.get(key)
            if pool is None:
                for stale in [entry for entry in _SLAB_POOLS if entry[0] != key[0]]:
                    del _SLAB_POOLS[stale]  # the pools of a parent process: their threads are not in this one
                pool = _SLAB_POOLS[key] = ThreadPoolExecutor(
                    max_workers=key[1], thread_name_prefix=_SLAB_THREAD_PREFIX
                )
    return pool


def _row_major_points(grid, points, counts) -> bool:
    """Whether the points of ``grid`` are its integer points in row-major order with the identity lookup.

    Tested by blocks of :data:`_ROW_MAJOR_BLOCK` points (the same answer as one test of the whole grid, without its
    arrays of every point: 10^7 points took 0.3 GB of them)."""

    size = int(np.prod(counts))
    if points.shape != (size, 3):
        return False
    lookup = np.asarray(getattr(grid, "lookup", None))
    if lookup.shape != tuple(int(value) for value in counts):
        return False
    flat_lookup = lookup.reshape(-1)
    n1, n2 = int(counts[1]), int(counts[2])
    for first in range(0, size, _ROW_MAJOR_BLOCK):
        last = min(size, first + _ROW_MAJOR_BLOCK)
        order = np.arange(first, last, dtype=np.int64)
        if not (
            np.array_equal(flat_lookup[first:last], order)
            and np.array_equal(points[first:last, 0], order // (n1 * n2))
            and np.array_equal(points[first:last, 1], (order // n2) % n1)
            and np.array_equal(points[first:last, 2], order % n2)
        ):
            return False
    return True


_ROW_MAJOR_BLOCK = 1 << 20
"""Points per block of :func:`_row_major_points`."""


_IMAGE_ROWS_SWITCH = "PARSEC_PERIODIC_IMAGE_ROWS"


def periodic_image_rows_setting() -> str:
    """``PARSEC_PERIODIC_IMAGE_ROWS``: ``auto`` (default, since 2026-11-21) keeps no table of the images of a
    :class:`PeriodicGridSymmetry` whose operations are integer affine maps of a row-major grid of at least
    :data:`_IMAGE_ROWS_POINTS` points, where its averages run on two threads or more (:class:`AffineImageRows`: the
    rows are formed where they are read, the density average gathers the field at the images by broadcast index
    vectors, or from the row of an oblique operation, by slabs on threads; the same integers and the same sums bit
    for bit); a smaller grid or a run on one thread keeps its table; ``on`` takes the rows at every size and thread
    count and holds no row of an oblique operation (its gathers form the indices of a few planes at a time,
    :class:`_ObliqueFactor`, since 2026-11-22); ``off`` builds the table of operations x points (int32) at the
    preparation for every grid, as before.  Slabs whose points fill a box follow the same rule since 2026-11-22
    (:func:`confined_grid_symmetry`); wires keep their table."""

    import os

    value = os.environ.get(_IMAGE_ROWS_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{_IMAGE_ROWS_SWITCH} must be auto, on or off, not {value!r}")
    return value


_FACTOR_BLOCK_POINTS = 1 << 16
"""Points of the planes whose indices :meth:`_ObliqueFactor.gather` forms at a time (a few hundred KiB of integers)."""


class _ObliqueFactor:
    """The gather indices of an oblique operation ``i -> (M i + c) mod n`` of a row-major grid, formed a few planes of
    the first axis at a time instead of held as a row (``lean`` of :class:`AffineImageRows`).

    The component ``a`` of the image of ``(i0, i1, i2)`` is ``(P_a[i1, i2] + (M_a0 i0 mod n_a)) mod n_a`` with the
    plane ``P_a = (M_a1 i1 + M_a2 i2 + c_a) mod n_a`` (values in ``[0, n_a)``), so the flat index of a plane is the
    sum over ``a`` of ``L_a[P_a + t_a]`` with the table ``L_a[v] = (v mod n_a) * stride_a`` of ``2 n_a`` entries and
    the scalar ``t_a = M_a0 i0 mod n_a``: the integers of the row, from three planes of the components (int32 below
    2^31 points).
    """

    __slots__ = ("counts", "planes", "steps", "tables", "dtype")

    def __init__(self, matrix, constant, counts, index_type) -> None:
        n = [int(value) for value in counts]
        self.counts = tuple(n)
        self.dtype = np.dtype(index_type)
        strides = (n[1] * n[2], n[2], 1)
        first = np.arange(n[1], dtype=np.int64).reshape(n[1], 1)
        second = np.arange(n[2], dtype=np.int64).reshape(1, n[2])
        planes, steps, tables = [], [], []
        for row in range(3):
            plane = np.full((1, 1), int(constant[row]), dtype=np.int64)
            if int(matrix[row][1]):
                plane = plane + int(matrix[row][1]) * first
            if int(matrix[row][2]):
                plane = plane + int(matrix[row][2]) * second
            planes.append(np.ascontiguousarray(plane % n[row]).astype(self.dtype)[None])
            steps.append(int(matrix[row][0]))
            values = np.arange(2 * n[row], dtype=np.int64)
            tables.append(((values % n[row]) * strides[row]).astype(self.dtype))
        self.planes, self.steps, self.tables = tuple(planes), tuple(steps), tuple(tables)

    @property
    def nbytes(self) -> int:
        return sum(int(part.nbytes) for part in self.planes) + sum(int(part.nbytes) for part in self.tables)

    def indices(self, first: int, last: int) -> np.ndarray:
        """The flat image indices of the points of the planes ``first:last`` of the first axis, ``(last - first, n1,
        n2)``, the integers of the row of the table there."""

        n0, n1, n2 = self.counts
        planes = np.arange(first, last, dtype=np.int64).reshape(-1, 1, 1)
        result = np.zeros((last - first, n1, n2), dtype=self.dtype)
        for row in range(3):
            shift = (self.steps[row] * planes) % self.counts[row]
            np.add(result, self.tables[row][self.planes[row] + shift.astype(self.dtype)], out=result)
        return result

    def gather(self, source: np.ndarray, first: int, last: int) -> np.ndarray:
        """``source[row[planes first:last]]``: the field at the images of those planes' points, row-major, formed a
        block of planes at a time (:data:`_FACTOR_BLOCK_POINTS`)."""

        plane = self.counts[1] * self.counts[2]
        result = np.empty((last - first) * plane, dtype=source.dtype)
        block = max(1, _FACTOR_BLOCK_POINTS // max(1, plane))
        for start in range(first, last, block):
            stop = min(last, start + block)
            result[(start - first) * plane:(stop - first) * plane] = source[self.indices(start, stop).reshape(-1)]
        return result


class AffineImageRows:
    """The table ``images[g, i] = (M_g i + c_g) mod n`` of a :class:`PeriodicGridSymmetry` on a row-major grid,
    formed where it is read instead of held (``PARSEC_PERIODIC_IMAGE_ROWS``, :func:`periodic_grid_symmetry`).

    It reads as the table does where its readers take it: ``shape``, ``dtype``, ``len``, iteration over the rows,
    ``rows[g]`` and ``rows[g, index]``; ``numpy.asarray`` builds the whole table (tests and old readers).
    :meth:`row` is :func:`affine_grid_images` of the operation, the row the table held; :meth:`at` the entries of
    some points of a row; :meth:`gather` and :meth:`image_sum` the field at the images, by the broadcast index
    vectors of :func:`affine_grid_indices` (no row of indices), the values and sums of the gathers of the table.

    What it keeps after the first average (:attr:`held_bytes`, never more than the table): for an operation whose
    rows of ``M`` read at most two axes each (a signed permutation of a box; the hexagonal, bcc and rhombohedral
    cells) its broadcast index vectors, lengths of an axis or planes, kilobytes; for an oblique one (:meth:`oblique`:
    a row of ``M`` that reads all three axes, as the mirrors of a cubic crystal in the primitive fcc cell, rows such
    as ``(1, 1, 1)``), whose index vector would be an int64 array of every point, the row of the table itself (int32,
    4 bytes a point).  Diamond in the primitive fcc cell keeps 36 rows of its 48.  With ``lean`` (``on`` of
    ``PARSEC_PERIODIC_IMAGE_ROWS``, since 2026-11-22) an oblique operation keeps no row either: its gathers form the
    indices of a few planes at a time from one plane of each index component (:class:`_ObliqueFactor`), the same
    integers and values.
    """

    __slots__ = ("maps", "counts", "dtype", "lean", "_held")

    ndim = 2

    def __init__(self, maps, counts, index_type, *, lean: bool = False) -> None:
        self.maps = tuple(
            (np.asarray(matrix, dtype=np.int64).reshape(3, 3), np.asarray(constant, dtype=np.int64).reshape(3))
            for matrix, constant in maps
        )
        self.counts = tuple(int(value) for value in counts)
        self.dtype = np.dtype(index_type)
        self.lean = bool(lean)
        # what each operation's gathers read, once formed: its broadcast index vectors (a tuple of three arrays), or,
        # where those would hold more than the row of every point (an oblique operation), that row (a 1-D array), or
        # under ``lean`` the planes of its index components (an _ObliqueFactor)
        self._held = {}

    def __getstate__(self):
        return {"maps": self.maps, "counts": self.counts, "dtype": self.dtype, "lean": self.lean}

    def __setstate__(self, state) -> None:
        for name in ("maps", "counts", "dtype"):
            object.__setattr__(self, name, state[name])
        object.__setattr__(self, "lean", bool(state.get("lean", False)))
        object.__setattr__(self, "_held", {})

    @property
    def held_bytes(self) -> int:
        """The bytes this object keeps now: the index vectors, rows and planes formed by the averages so far."""

        total = 0
        for held in self._held.values():
            if isinstance(held, (np.ndarray, _ObliqueFactor)):
                total += int(held.nbytes)
            else:
                total += sum(int(part.nbytes) for part in held)
        return total

    def oblique(self, operation: int) -> bool:
        """Whether the broadcast index vectors of the operation would hold at least the bytes of its row (a row of
        ``M`` that reads all three axes: an int64 array of every point), so its gathers read the row instead."""

        matrix, _ = self.maps[self._operation(operation)]
        points = int(np.prod(self.counts))
        vector_bytes = sum(
            int(np.prod([self.counts[column] for column in range(3) if matrix[row, column]], dtype=np.int64)) * 8
            for row in range(3)
        )
        return vector_bytes >= points * self.dtype.itemsize

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.maps), int(np.prod(self.counts)))

    @property
    def size(self) -> int:
        operations, points = self.shape
        return operations * points

    def __len__(self) -> int:
        return len(self.maps)

    def __iter__(self):
        for operation in range(len(self.maps)):
            yield self.row(operation)

    def _operation(self, operation) -> int:
        index = int(operation)
        if index < 0:
            index += len(self.maps)
        if not 0 <= index < len(self.maps):
            raise IndexError(f"operation {operation} of {len(self.maps)}")
        return index

    def row(self, operation: int) -> np.ndarray:
        """``images[operation]``: the image row of every point (the row of the table, the same integers; read-only
        where it is the row an oblique operation keeps)."""

        index = self._operation(operation)
        held = self._held.get(index)
        if isinstance(held, np.ndarray):
            return held
        matrix, constant = self.maps[index]
        return affine_grid_images(matrix, constant, self.counts, self.dtype)

    def at(self, operation: int, rows) -> np.ndarray:
        """``images[operation, rows]`` for an index array ``rows`` (row-major points), the integers of the row."""

        matrix, constant = self.maps[self._operation(operation)]
        rows = np.asarray(rows)
        flat = rows.astype(np.int64, copy=False).reshape(-1)
        n0, n1, n2 = self.counts
        if flat.size and (int(flat.min()) < -(n0 * n1 * n2) or int(flat.max()) >= n0 * n1 * n2):
            raise IndexError("a row outside the grid")
        flat = np.where(flat < 0, flat + n0 * n1 * n2, flat)
        coordinates = (flat // (n1 * n2), (flat // n2) % n1, flat % n2)
        strides = (n1 * n2, n2, 1)
        result = np.zeros(flat.shape, dtype=np.int64)
        for row in range(3):
            component = np.full(flat.shape, int(constant[row]), dtype=np.int64)
            for column in range(3):
                if matrix[row, column]:
                    component = component + int(matrix[row, column]) * coordinates[column]
            result += (component % self.counts[row]) * strides[row]
        return result.astype(self.dtype).reshape(rows.shape)

    def _gatherer(self, operation: int):
        """What the gathers of the operation read (formed once, kept, read-only): its broadcast index vectors of
        :func:`affine_grid_indices` (a tuple of three arrays), or the row of an :meth:`oblique` operation."""

        index = self._operation(operation)
        found = self._held.get(index)
        if found is None:
            matrix, constant = self.maps[index]
            if self.oblique(index) and self.lean:
                found = _ObliqueFactor(matrix, constant, self.counts, self.dtype)
            elif self.oblique(index):
                found = affine_grid_images(matrix, constant, self.counts, self.dtype)
                found.flags.writeable = False
            else:
                found = tuple(affine_grid_indices(matrix, constant, self.counts))
                for part in found:
                    part.flags.writeable = False
            self._held[index] = found
        return found

    def _slab(self, held, first: int, last: int):
        """The piece of ``held`` (:meth:`_gatherer`) for the planes ``first:last`` of the first axis."""

        if isinstance(held, np.ndarray):
            plane = self.counts[1] * self.counts[2]
            return held[first * plane:last * plane]
        if isinstance(held, _ObliqueFactor):
            return (held, first, last)
        return tuple(index[first:last] if index.shape[0] > 1 else index for index in held)

    def _gathered_slab(self, source: np.ndarray, piece):
        """``source[row]`` (a 1-D field of the points) at the planes of ``piece`` (:meth:`_slab`): the field at the
        images there, a new array of their points in row-major order (the values of the gather of the table)."""

        if isinstance(piece, np.ndarray):
            return source[piece]
        if isinstance(piece[0], _ObliqueFactor):
            return piece[0].gather(source, piece[1], piece[2])
        return source.reshape(self.counts)[piece]

    def _run(self, work) -> None:
        """``work(first, last)`` over the planes of the first axis: by slabs on the threads of the periodic setup for a
        grid of ``_THREADED_SUMS`` points or more, else at once."""

        if self.shape[1] < _THREADED_SUMS:
            work(0, self.counts[0])
        else:
            _run_by_slabs(work, self.counts[0])

    def gather(self, values, operation: int) -> np.ndarray:
        """``values[..., images[operation]]`` of an array ``(..., points)``: a new array laid out as NumPy lays out the
        gather of a row of the table (the points outermost in memory, the leading axes inside them in C order: strides
        ``(8, 24)`` for a field ``(3, points)``), the same values; each component of the leading axes in turn, by
        slabs on threads (:meth:`_run`)."""

        values = np.asarray(values)
        points = self.shape[1]
        if values.ndim < 1 or values.shape[-1] != points:
            raise ValueError(f"a field of {points} grid points is required, got shape {values.shape}")
        held = self._gatherer(operation)
        lead = values.shape[:-1]
        sources = values.reshape(-1, points)
        storage = np.empty((points,) + lead, dtype=values.dtype)
        result = np.moveaxis(storage, 0, -1)
        targets = storage.reshape(points, -1).T
        plane = self.counts[1] * self.counts[2]

        def work(first: int, last: int) -> None:
            piece = self._slab(held, first, last)
            for source, target in zip(sources, targets):
                target[first * plane:last * plane] = self._gathered_slab(source, piece).reshape(-1)

        self._run(work)
        return result

    def image_sum(self, values, operations=None) -> np.ndarray:
        """``sum_g values[images[g]]`` over ``operations`` (all, in order, where ``None``) of a float field of the
        points: the zero field plus each gather in turn, as ``total += values[images[g]]`` adds them, element by
        element; a grid of ``_THREADED_SUMS`` points or more by slabs of the first axis on the threads of the
        periodic setup, each slab taking every operation in turn (the same additions)."""

        values = np.asarray(values)
        if values.ndim != 1 or values.shape[0] != self.shape[1]:
            raise ValueError(f"a field of {self.shape[1]} grid points is required, got shape {values.shape}")
        chosen = range(len(self.maps)) if operations is None else [self._operation(value) for value in operations]
        held = [self._gatherer(operation) for operation in chosen]
        total = np.zeros_like(values)
        plane = self.counts[1] * self.counts[2]

        def work(first: int, last: int) -> None:
            slab = total[first * plane:last * plane]
            for one in held:
                slab += self._gathered_slab(values, self._slab(one, first, last)).reshape(-1)

        self._run(work)
        return total

    def accumulate(self, total: np.ndarray, values: np.ndarray, operation: int) -> None:
        """``total += values[images[operation]]`` for float fields of the points (1-D), the same additions."""

        if total.shape != (self.shape[1],) or not total.flags.c_contiguous:
            raise ValueError("the total must be a contiguous field of the grid points")
        values = np.asarray(values)
        total += self._gathered_slab(values, self._slab(self._gatherer(operation), 0, self.counts[0])).reshape(-1)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.row(key)
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], (int, np.integer)):
            operation, points = key
            if isinstance(points, slice):
                return self.row(operation)[points]
            return self.at(operation, points)
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], slice)
            and key[0] == slice(None)
            and not isinstance(key[1], slice)
        ):
            return np.stack([self.at(operation, key[1]) for operation in range(len(self.maps))])
        return np.asarray(self)[key]

    def __array__(self, dtype=None, copy=None):
        table = np.empty(self.shape, dtype=self.dtype)
        for operation in range(len(self.maps)):
            table[operation] = self.row(operation)
        return table if dtype is None else table.astype(dtype, copy=False)

    def __repr__(self) -> str:
        return f"AffineImageRows({len(self.maps)} operations, counts {self.counts}, {self.dtype})"


def confined_symmetry_lattice(lattice_vectors, periodicity: int, radius: float) -> np.ndarray:
    """``alatt`` of a wire or slab (``symmetries.f90:114-118``), as rows.

    ``symm%alatt = pbc%latt_vec`` with ``4*rmax`` on the diagonal of every
    confined direction (``latt_vec`` holds unit vectors there), as for a
    cluster's artificial cubic cell.
    """

    lattice = np.array(lattice_vectors, dtype=np.float64, copy=True)
    for axis in range(int(periodicity), 3):
        lattice[axis, axis] = 4.0 * float(radius)
    return lattice


def grid_preserving_operations(group: SpaceGroup, periodicity: int, *, tolerance: float = 1.0e-6):
    """The operations of a wire or slab group that map its grid onto itself.

    Returns ``(group, dropped)``.  Fortran's group of a wire or slab
    (:func:`find_wire_slab_space_group`) may hold operations that a
    confined grid does not have: a fractional translation along a confined
    direction (a mirror or inversion about a plane other than ``z = 0``, in
    units of ``4*rmax``) moves the cylinder or slab off itself, and a matrix
    that mixes a periodic with a confined direction (possible only when
    ``4*rmax`` equals a lattice length) maps the grid onto another one.
    ``symm_scalar`` then averages over points that are not images of each
    other (with ``bandstruc`` a slab turns Fortran's symmetry on: graphene
    with one atom lifted by 0.15 bohr ends 0.21 Ry off its own run).  Those
    operations are dropped here (a subgroup remains: block-diagonal matrices
    with no confined translation compose to the same form); every other
    field of ``group`` is kept, ``full_holohedry`` becomes false when an
    operation is dropped.
    """

    periodicity = int(periodicity)
    kept = []
    for index, (matrix, translation) in enumerate(zip(group.rotations, group.translations)):
        matrix = np.asarray(matrix, dtype=np.int64)
        mixes = np.any(matrix[:periodicity, periodicity:] != 0) or np.any(
            matrix[periodicity:, :periodicity] != 0
        )
        shifted = any(
            abs(float(translation[axis])) > tolerance for axis in range(periodicity, 3)
        )
        if not (mixes or shifted):
            kept.append(index)
    dropped = group.order - len(kept)
    if dropped == 0:
        return group, 0
    last = group.translations[kept[-1]]
    return (
        SpaceGroup(
            rotations=tuple(group.rotations[index] for index in kept),
            translations=tuple(group.translations[index] for index in kept),
            cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
            labels=tuple(group.labels[index] for index in kept),
            crystal_system=group.crystal_system,
            full_holohedry=False,
            lattice_operations=group.lattice_operations,
            candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
            symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
            ignored=group.ignored,
        ),
        dropped,
    )


def find_wire_slab_space_group(lattice_vectors, periodicity: int, radius: float, atoms):
    """Space group of a wire or slab: ``symmetries.f90`` with ``pbc%per < 3``.

    ``lattice_vectors`` are the rows of ``latt_vec`` (unit vectors along the
    confined directions), ``radius`` is ``Boundary_Sphere_Radius``.  The
    search of :func:`find_space_group` runs on ``alatt``
    (:func:`confined_symmetry_lattice`); the operations that do not map the
    confined grid onto itself are dropped (:func:`grid_preserving_operations`).
    Returns ``(group, dropped)``.  ``M`` and ``t`` are relative to ``alatt``;
    along the periodic directions they are those of ``latt_vec``, so the
    reciprocal rotations act on Monkhorst-Pack fractions as in ``irrbz``.
    """

    alatt = confined_symmetry_lattice(lattice_vectors, periodicity, radius)
    group = find_space_group(alatt, atoms)
    return grid_preserving_operations(group, periodicity)


SLAB_MESH_SYMMETRY_ENV = "PARSEC_SLAB_MESH_SYMMETRY"


def slab_mesh_symmetry_setting() -> str:
    """``PARSEC_SLAB_MESH_SYMMETRY``: the space group of a slab with a Monkhorst-Pack mesh (Fortran-derived routines).

    PARSEC 1.5 searches no symmetry for a slab (``parsec.F90:470-486``, ``ierr = 2``) unless a band structure,
    spin-orbit coupling or an NSCF run asks for it, so a slab mesh is reduced by time reversal alone (h-BN with
    ``Monkhorst_Pack_Grid 6 6 1``: 20 points).  ``auto`` (the default) gives a slab of scalar or collinear-spin
    orbitals with a native LDA or GGA and a Monkhorst-Pack mesh (a ``Monkhorst_Pack_Shift`` block, or PARSEC 1.5's
    zero shift without it) the operations of its group that map its grid and its mesh onto themselves
    (:func:`slab_mesh_subgroup`) and whose lattice matrix is a signed permutation of the two in-plane directions (and
    ``z``): the finite differences along the lattice directions map onto each other, so the Hamiltonian of every
    k-point follows them to round-off.  The mesh is reduced to their irreducible wedge with time reversal, the
    density symmetrized with them and the forces averaged over them, as for a wire (h-BN 6 x 6 x 1: 13 points; every
    operation of a rectangular or square slab).  Every point of the time-reversal wedge of ``off`` is then exactly
    (not modulo a reciprocal vector) an image of its representative, so the levels are those of ``off`` to round-off
    (:func:`slab_mesh_wedge_exactness`); a four-fold rotation without a diagonal mirror (S4, C4, C4h) on a mesh
    through the zone boundary fails that test and the slab keeps time reversal alone, as does a group that would
    reduce no point.  Where the mesh has a component 1/2, the averaged density of such a point takes images that
    differ from the point of ``off`` by a reciprocal vector: a difference of second order in the error of the finite
    differences.  Operations that reverse ``z`` are dropped where charged sheets are not symmetric about ``z = 0``.
    The grid is that of ``off`` (no operation whose fractional translation would change the counts of
    ``adjust_step`` is kept).

    ``full`` keeps the three- and six-fold rotations of a hexagonal slab (and the mirrors that are no signed
    permutation) as well (h-BN: 7 points).  They map the grid points onto points, but the first derivatives of the
    Bloch term ``-2 i k . nabla`` are differences along the two lattice directions of the plane only, so the
    Hamiltonian of a k-point other than Gamma follows such a rotation only up to the error of those differences
    (PARSEC's wedge of a hexagonal bulk cell takes them the same way): the wedge differs from the whole mesh by an
    amount that vanishes with the grid (A100, order 12: h-BN 6 x 6 x 1 at 0.15 bohr -1.8e-7 Ry, graphene 9 x 9 x 1
    at 0.25 bohr 3.0e-6 Ry, silicene 6 x 6 x 1 at 0.30 bohr -2e-8 Ry; a two-atom h-BN slab 3 x 3 x 1 at 0.45 bohr
    and order 6: 5.1e-4 Ry).  ``off``: no search, as before.
    """

    import os

    value = os.environ.get(SLAB_MESH_SYMMETRY_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "full", "off"):
        raise ValueError(f"{SLAB_MESH_SYMMETRY_ENV} must be auto, full or off, got {value!r}")
    return value


def slab_mesh_subgroup(
    group: SpaceGroup,
    counts,
    shift,
    confined_planes,
    *,
    exact: bool = False,
    reverse_z: bool = True,
    tolerance: float = 1.0e-6,
):
    """The operations of a slab group that map the slab grid onto itself without changing it.

    ``group`` is the group of :func:`find_wire_slab_space_group` (no translation along ``z``, no matrix that mixes
    ``z`` with the plane); ``counts`` and ``shift`` are the points and the shift (grid steps) along the two periodic
    directions of the grid built without fractional translations, ``confined_planes`` the ``z`` of its planes.  An
    operation is kept when its in-plane part maps the points ``f = (i + s)/n`` of the plane onto such points (its
    fractional translation a whole number of steps, :func:`grid_commensurability_problem` restricted to the plane)
    and, where it reverses ``z``, the planes are symmetric about ``z = 0`` and ``reverse_z`` allows it (``False``
    where an external potential such as charged sheets is not symmetric about ``z = 0``); with ``exact`` also only
    where its in-plane matrix is a signed permutation (the finite differences along the lattice directions map onto
    each other).  The operations that map a grid onto itself form a group, and so do the signed permutations and
    the operations that keep ``z`` among them.  Returns ``(subgroup, dropped)``.
    """

    counts = np.asarray(counts, dtype=np.float64).reshape(-1)[:2]
    shift = np.asarray(shift, dtype=np.float64).reshape(-1)[:2]
    planes = np.sort(np.asarray(confined_planes, dtype=np.float64).reshape(-1))
    scale = max(float(np.max(np.abs(planes))) if planes.size else 1.0, 1.0)
    planes_symmetric = bool(
        planes.size and np.max(np.abs(planes + planes[::-1])) <= tolerance * scale
    )
    corners = np.array([[a, b] for a in (0, 1) for b in (0, 1)], dtype=np.float64)
    fractional = (corners + shift) / counts
    kept = []
    for index, (matrix, translation) in enumerate(zip(group.rotation_matrices, group.translation_vectors)):
        if np.any(matrix[:2, 2] != 0) or np.any(matrix[2, :2] != 0) or abs(float(translation[2])) > tolerance:
            continue
        if matrix[2, 2] == -1 and not (planes_symmetric and reverse_z):
            continue
        plane = np.abs(matrix[:2, :2])
        if exact and not (np.all(plane.sum(axis=0) == 1) and np.all(plane.sum(axis=1) == 1)):
            continue
        mapped = (fractional - translation[:2]) @ matrix[:2, :2].T.astype(np.float64)
        target = mapped * counts - shift
        if np.max(np.abs(target - np.rint(target))) <= tolerance:
            kept.append(index)
    dropped = group.order - len(kept)
    if dropped == 0:
        return group, 0
    last = group.translations[kept[-1]]
    return (
        SpaceGroup(
            rotations=tuple(group.rotations[index] for index in kept),
            translations=tuple(group.translations[index] for index in kept),
            cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
            labels=tuple(group.labels[index] for index in kept),
            crystal_system=group.crystal_system,
            full_holohedry=False,
            lattice_operations=group.lattice_operations,
            candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
            symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
            ignored=group.ignored,
        ),
        dropped,
    )


def slab_mesh_wedge_exactness(
    lattice_vectors, dimensions, shift, reciprocal_rotations, *, tolerance: float = 1.0e-8
) -> tuple[bool, int, int]:
    """Whether a group's wedge of a Monkhorst-Pack mesh computes the k-points of the time-reversal wedge exactly.

    Returns ``(exact, wedge_count, pair_count)``: the points of the wedge of ``reciprocal_rotations`` with time
    reversal and of time reversal alone (:func:`parsec_python.Grid.kpoints.irreducible_monkhorst_pack`, which
    represents a star by its first mesh point folded into ``(-1/2, 1/2]``).  The finite-difference Bloch operator is
    not invariant under ``k -> k + G``, so a point of the time-reversal wedge has the levels of its representative in
    the group's wedge to round-off only where it is exactly ``+-G_R k`` of an operation, not modulo a reciprocal
    vector; ``exact`` says that every point is.  Under signed permutations of the lattice directions it fails for a
    four-fold rotation without a diagonal mirror on a mesh through the zone boundary (a component 1/2): its star joins
    two time-reversal pairs, such as ``(1/2, 1/4)`` and ``(1/4, 1/2)``, whose folded points are images of each other
    only modulo a reciprocal vector (A100, a square S4 slab of four carbon atoms at 0.45 bohr and order 6 on the
    unshifted 4 x 4 x 1 mesh: 2.4e-5 Ry).  With a diagonal mirror, or on a mesh without such points, every point is
    an exact image.  Both wedges are those the run takes (cached, :func:`~parsec_python.Grid.kpoints.kpoint_sampling_details`).
    """

    from .kpoints import _cached_irreducible_monkhorst_pack

    lattice = tuple(tuple(float(value) for value in row) for row in np.asarray(lattice_vectors, dtype=float))
    dims = tuple(int(value) for value in dimensions)
    shifts = tuple(float(value) for value in shift)
    rotations = np.asarray(reciprocal_rotations, dtype=np.int64).reshape(-1, 3, 3)
    pairs = _cached_irreducible_monkhorst_pack(lattice, dims, shifts, None, True)
    wedge = _cached_irreducible_monkhorst_pack(
        lattice, dims, shifts, tuple(tuple(tuple(int(v) for v in row) for row in matrix) for matrix in rotations), True
    )
    pair_index = np.asarray(pairs.grid_map).reshape(-1)
    wedge_index = np.asarray(wedge.grid_map).reshape(-1)
    # (wedge points, operations, 3): the exact images G_R k of every representative
    images = np.einsum("rij,kj->kri", rotations.astype(np.float64), np.asarray(wedge.fractional, dtype=np.float64))
    for star in range(wedge.count):
        members = np.asarray(pairs.fractional, dtype=np.float64)[np.unique(pair_index[wedge_index == star])]
        candidates = np.concatenate([images[star], -images[star]])
        distance = np.max(np.abs(members[:, None, :] - candidates[None, :, :]), axis=2)
        if np.any(np.min(distance, axis=1) > tolerance):
            return False, wedge.count, pairs.count
    return True, wedge.count, pairs.count


def sheet_preserving_operations(group: SpaceGroup, sheets, alatt, *, tolerance: float = 1.0e-6):
    """The operations of a slab group that map its charged sheets onto sheets of the same charge.

    ``sheets`` are ``(charge, height)`` pairs (or objects with ``charge`` and ``height``: the ``add_charged_sheets``
    planes ``z = height``, whose potential ``sum_s 4 pi (q_s/A) |z - z_s|`` enters the Hamiltonian of a slab), ``alatt``
    the rows of :func:`confined_symmetry_lattice`.  An operation ``r -> U (r - A t)`` maps the plane ``z = z_s`` onto
    the plane ``z = U_zz z_s - (U A t)_z`` where ``U`` does not mix ``z`` with the plane; it is kept when that maps the
    multiset of ``(charge, height)`` onto itself within ``tolerance`` bohr.  PARSEC 1.5 searches the group of a slab
    from the atoms alone (``symmetries.f90``), so with a band structure, spin-orbit coupling or an NSCF run a
    ``z``-reversing operation symmetrized the density of an asymmetric capacitor; the kept operations form a group
    (the stabilizer of the sheets).  Returns ``(subgroup, dropped)``; every other field of ``group`` is kept,
    ``full_holohedry`` becomes false when an operation is dropped.
    """

    pairs = []
    for sheet in sheets:
        charge, height = (sheet.charge, sheet.height) if hasattr(sheet, "charge") else sheet
        pairs.append((float(charge), float(height)))
    if not pairs or group.order <= 1:
        return group, 0
    charges = np.array([charge for charge, _ in pairs], dtype=np.float64)
    heights = np.array([height for _, height in pairs], dtype=np.float64)
    alatt = np.asarray(alatt, dtype=np.float64).reshape(3, 3)
    kept = []
    for index, (rotation, translation) in enumerate(zip(group.cartesian_rotations, group.translations)):
        matrix = np.asarray(rotation, dtype=np.float64)
        if abs(matrix[2, 0]) > tolerance or abs(matrix[2, 1]) > tolerance:
            continue
        shift = -(matrix @ (np.asarray(translation, dtype=np.float64) @ alatt))[2]
        mapped = matrix[2, 2] * heights + shift
        unmatched = list(range(len(pairs)))
        for charge, height in zip(charges, mapped):
            match = next(
                (
                    position
                    for position, other in enumerate(unmatched)
                    if abs(charges[other] - charge) <= 1.0e-12 * max(1.0, abs(charge))
                    and abs(heights[other] - height) <= tolerance
                ),
                None,
            )
            if match is None:
                break
            del unmatched[match]
        if not unmatched:
            kept.append(index)
    dropped = group.order - len(kept)
    if dropped == 0:
        return group, 0
    last = group.translations[kept[-1]]
    return (
        SpaceGroup(
            rotations=tuple(group.rotations[index] for index in kept),
            translations=tuple(group.translations[index] for index in kept),
            cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
            labels=tuple(group.labels[index] for index in kept),
            crystal_system=group.crystal_system,
            full_holohedry=False,
            lattice_operations=group.lattice_operations,
            candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
            symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
            ignored=group.ignored,
        ),
        dropped,
    )


def _confined_affine_maps(grid, group: SpaceGroup, points, counts, shift):
    """``(maps, box)`` of the operations of ``group`` on a slab grid as integer affine maps of a row-major box, or
    ``None`` (a wire, a slab whose points do not fill a box in that order, an operation without such a map).

    The points of a slab (``Grid.confined``) are those of the whole surface cell on a range of planes, numbered with
    ``i``, then ``j``, then ``k`` from their largest to their smallest value: point ``p`` has the reversed indices
    ``r = hi - x = unravel(p, box)``, ``box = hi - lo + 1``.  An operation maps ``x`` to ``M x + c`` (the map of
    :func:`grid_affine_map` on the index box, folded along the periodic directions), so ``r`` goes to
    ``(M r + hi - M hi - c) mod box``; along the confined direction ``M`` must keep the axis (``+-1`` on it alone)
    and map the range of planes onto itself, where no folding happens.  The rows are those of
    :func:`confined_grid_symmetry`'s table, the same integers.
    """

    periodicity = int(getattr(grid, "periodicity", 3))
    if periodicity != 2 or points.ndim != 2 or points.shape[0] == 0:
        return None
    lowest = points.min(axis=0)
    highest = points.max(axis=0)
    index_min = np.asarray(getattr(grid, "index_min", lowest), dtype=np.int64)
    index_max = np.asarray(getattr(grid, "index_max", highest), dtype=np.int64)
    box = highest - lowest + 1
    if int(np.prod(box)) != points.shape[0]:
        return None
    for axis in range(periodicity):
        if lowest[axis] != index_min[axis] or highest[axis] != index_max[axis] or box[axis] != counts[axis]:
            return None
    # the order of the points: row-major in the reversed indices (by blocks, without arrays of every point)
    planes = int(box[1]) * int(box[2])
    for first in range(0, points.shape[0], _ROW_MAJOR_BLOCK):
        last = min(points.shape[0], first + _ROW_MAJOR_BLOCK)
        order = np.arange(first, last, dtype=np.int64)
        if not (
            np.array_equal(points[first:last, 0], highest[0] - order // planes)
            and np.array_equal(points[first:last, 1], highest[1] - (order // int(box[2])) % int(box[1]))
            and np.array_equal(points[first:last, 2], highest[2] - order % int(box[2]))
        ):
            return None
    maps = []
    for matrix, translation in zip(group.rotation_matrices, group.translation_vectors):
        found = grid_affine_map(matrix, translation, counts, shift)
        if found is None:
            return None
        integer, constant = found
        for row in range(periodicity, 3):
            # the confined direction: kept (no mixing with the periodic ones), its planes onto themselves
            if any(integer[row, column] != 0 for column in range(3) if column != row) or any(
                integer[column, row] != 0 for column in range(3) if column != row
            ):
                return None
        reversed_constant = highest - integer @ highest - constant
        for row in range(periodicity, 3):
            ends = [int(integer[row, row]) * end + int(reversed_constant[row]) for end in (0, int(box[row]) - 1)]
            if min(ends) < 0 or max(ends) >= int(box[row]):
                return None
        maps.append((integer, reversed_constant.astype(np.int64)))
    return maps, tuple(int(value) for value in box)


def confined_grid_symmetry(grid, group: SpaceGroup) -> PeriodicGridSymmetry:
    """The operations of ``group`` as permutations of a wire or slab grid.

    ``symm_scalar`` (``newrho.F90:262-283``) with ``is_pbc``: the point of
    fractional coordinates ``rw = (i + s)/n`` goes to ``rrw = M (rw - t)``
    and to the integer point ``nint(rrw n - s)``.  Fortran folds every
    direction into the index box; here only the periodic ones are folded,
    and a point mapped outside the cylinder or slab is an error (the
    operations of :func:`grid_preserving_operations` never do that).

    A slab under ``PARSEC_PERIODIC_IMAGE_ROWS`` (``auto``: from
    :data:`_IMAGE_ROWS_POINTS` points averaged on two threads or more, as a
    bulk cell, since 2026-11-22; ``on``: always) holds no table: its points
    fill a box, and :class:`AffineImageRows` forms the rows where they are
    read (:func:`_confined_affine_maps`, the same integers).  ``off``, a
    wire (a cylinder of points) and a smaller slab keep the table.
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(grid.shift, dtype=np.float64)
    problem = grid_commensurability_problem(group, counts, shift)
    if problem is not None:
        raise ValueError(
            "Symmetrize_Charge_Density: " + problem + "; PARSEC's adjust_step makes "
            "the grid spacing commensurate with the fractional translations, which "
            "this grid is not"
        )
    index_type = np.int32 if points.shape[0] < 2**31 - 1 else np.int64
    # PARSEC_PERIODIC_IMAGE_ROWS: a slab whose points fill a box (every point of the surface cell, a range of planes)
    # numbered in row-major order of their reversed indices takes the rows formed where they are read, as a bulk
    # cell does (the same integers); a wire (a cylinder of points) keeps its table
    affine = _confined_affine_maps(grid, group, points, counts, shift)
    rows_setting = periodic_image_rows_setting()
    if affine is not None and _image_rows_taken(rows_setting, points.shape[0]):
        maps, box = affine
        return PeriodicGridSymmetry(
            matrices=group.rotation_matrices.astype(np.float64),
            images=AffineImageRows(maps, box, index_type, lean=rows_setting == "on"),
            affine_maps=tuple(maps),
            counts=tuple(int(value) for value in box),
        )
    fractional = (points.astype(np.float64) + shift) / counts
    images = np.empty((group.order, points.shape[0]), dtype=index_type)
    for index, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors)
    ):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = _nint_array(mapped * counts - shift).astype(np.int64)
        rows = grid.rows_for_integer_coordinates(target)
        if np.any(rows < 0) or np.unique(rows).size != rows.size:
            raise ValueError(
                f"Symmetrize_Charge_Density: operation {index + 1} does not permute "
                "the points of the wire or slab grid"
            )
        images[index] = rows
    return PeriodicGridSymmetry(
        matrices=group.rotation_matrices.astype(np.float64), images=images
    )


__all__ = [
    "ATOM_IMAGE_TOLERANCE",
    "CRYSTAL_SYSTEMS",
    "GRID_MAP_TOLERANCE",
    "PeriodicGridSymmetry",
    "SpaceGroup",
    "affine_grid_images",
    "candidate_rotations",
    "confined_grid_symmetry",
    "confined_symmetry_lattice",
    "fast_crystal_operations",
    "find_grid_space_group",
    "find_space_group",
    "FIXED_SPECIES_SUFFIX",
    "atoms_with_fixed_species",
    "find_wire_slab_space_group",
    "grid_affine_map",
    "snapped_translation",
    "space_group_with_search",
    "skipped_space_group",
    "SLAB_MESH_SYMMETRY_ENV",
    "sheet_preserving_operations",
    "slab_mesh_subgroup",
    "slab_mesh_symmetry_setting",
    "slab_mesh_wedge_exactness",
    "grid_preserving_operations",
    "fortran_anint",
    "fortran_nint",
    "fortran_reciprocal_lattice",
    "grid_commensurability_problem",
    "identity_space_group",
    "lattice_point_group",
    "periodic_grid_symmetry",
    "respects_flags",
    "space_group_atom_images",
]
