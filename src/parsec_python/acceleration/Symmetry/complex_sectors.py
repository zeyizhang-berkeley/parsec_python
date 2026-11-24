"""Sectors of the complex one-dimensional characters of an abelian little group of a k-point
(``PARSEC_KPOINT_SECTOR_CHARACTERS=complex``, the default since 2026-11-23; ``real`` keeps the commuting involutions).

The sectors of a k-point (:mod:`..SCF.kpoint_sectors`) use the commuting involutions ``H_k`` of its little group,
whose characters are ``+-1``.  The Bloch Hamiltonian of a k-point other than Gamma is complex anyway, so nothing asks
for real characters there: an abelian subgroup ``A`` of the little group with a threefold or fourfold axis has the
complex characters ``chi(a) = e^{2 pi i m / n}``, and its sectors are as many as its elements.  On the threefold axis
of C3v (a defect cell without the translations of a supercell) ``C3`` gives three sectors of a third of the rows each,
two of which the mirrors carry into each other (``PARSEC_KPOINT_SECTOR_CLASSES``), where ``C_s`` gives two halves.

For an orbit ``O_w`` with stabilizer ``S_w`` a character admits the orbit exactly where it is trivial on ``S_w``, and
the sector basis is ``U[i, w] = chi(g_i) / sqrt(|O_w|)`` for the operation ``g_i`` that carries the representative to
row ``i`` (well defined on an admitted orbit): the rule of :mod:`.representations` with complex ``chi``.  ``U^H H U``
is formed from the representative rows as before (the stencil of :mod:`..Hamiltonian.sector_bloch_stencil` with a
complex palette), the nonlocal projectors ``U^H B`` of one atom per orbit (their phases have modulus one, so
:func:`.axis_reflection.compress_projectors_for_sectors` holds).  The characters are found from the regular
representation of ``A`` (the joint eigenvectors of its permutation matrices) and kept as exponents ``m`` modulo the
exponent ``n`` of the group; the phase of a row as its exponent (int8), as the real sectors keep their signs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

_SWITCH = "PARSEC_KPOINT_SECTOR_CHARACTERS"


def sector_characters_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_CHARACTERS``: ``complex`` (default since 2026-11-23) also takes an abelian subgroup of
    the little group of a k-point other than Gamma with complex characters where its sectors (with their classes)
    cost less than those of the commuting involutions; ``real`` keeps the commuting involutions alone, bit for bit as
    before."""

    value = os.environ.get(_SWITCH, "complex").strip().lower() or "complex"
    if value not in {"real", "complex"}:
        raise ValueError(f"{_SWITCH} must be real or complex, not {value!r}")
    return value


def largest_abelian_subgroup(table: np.ndarray, members) -> list[int]:
    """A largest abelian subgroup of ``members`` (indices of ``table``; a group), ascending with the identity 0 first:
    from the cyclic subgroup of every member, the members that commute with all of the group so far are added one at a
    time, the one whose generated group is largest first; the largest group found is kept (the first on ties)."""

    members = [int(member) for member in members]
    best: list[int] = [0]

    def closure(group: set[int], extra: int) -> set[int]:
        grown = set(group)
        frontier = list(group)
        while frontier:
            fresh = []
            for element in frontier:
                product = int(table[element, extra])
                if product not in grown:
                    grown.add(product)
                    fresh.append(product)
            frontier = fresh
        return grown

    seen: set[frozenset] = set()
    for start in members:
        group = closure({0}, start)
        key = frozenset(group)
        if key in seen:
            continue
        seen.add(key)
        while True:
            candidates = [
                c for c in members
                if c not in group and all(int(table[c, g]) == int(table[g, c]) for g in group)
            ]
            if not candidates:
                break
            grown = max((closure(group, c) for c in candidates), key=len)
            group = grown
        if len(group) > len(best):
            best = sorted(group)
    return best


def character_exponents(table: np.ndarray, elements) -> tuple[np.ndarray, int]:
    """``(exponents, n)`` of the characters of the abelian group ``elements`` (indices of ``table``, the identity
    first): ``chi_c(elements[a]) = exp(2 pi i exponents[c, a] / n)``, ``n`` the exponent of the group (the lcm of the
    orders of its elements), the trivial character first and the others in ascending order of their rows."""

    elements = [int(element) for element in elements]
    order = len(elements)
    where = {element: index for index, element in enumerate(elements)}
    orders = []
    for element in elements:
        power, count = element, 1
        while power != 0:
            power = int(table[power, element])
            count += 1
        orders.append(count)
    modulus = 1
    for value in orders:
        modulus = modulus * value // math.gcd(modulus, value)
    rng = np.random.default_rng(20261122)
    coefficients = rng.normal(size=order) + 1j * rng.normal(size=order)
    matrix = np.zeros((order, order), dtype=np.complex128)
    permutations = []
    for a, element in enumerate(elements):
        permutation = np.array([where[int(table[element, other])] for other in elements], dtype=np.int64)
        permutations.append(permutation)
        matrix[permutation, np.arange(order)] += coefficients[a]
    _, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    exponents = np.empty((order, order), dtype=np.int64)
    for a, permutation in enumerate(permutations):
        moved = np.zeros((order, order), dtype=np.complex128)
        moved[permutation] = vectors
        values = np.einsum("ij,ji->i", inverse, moved)
        exponents[:, a] = np.mod(np.rint(np.angle(values) / (2.0 * np.pi) * modulus), modulus).astype(np.int64)
    # every character once, the trivial one first
    rows = sorted({tuple(row) for row in exponents.tolist()})
    if len(rows) != order:
        raise RuntimeError("the characters of the abelian group were not separated")
    trivial = tuple([0] * order)
    rows.remove(trivial)
    exponents = np.asarray([trivial] + rows, dtype=np.int64)
    # a check: the characters are homomorphisms and orthogonal
    phases = np.exp(2j * np.pi * exponents / modulus)
    gram = phases @ phases.conj().T
    if not np.allclose(gram, order * np.eye(order), atol=1.0e-8):
        raise RuntimeError("the characters of the abelian group are not orthogonal")
    return exponents, int(modulus)


@dataclass(frozen=True)
class ComplexReduction:
    """The orbits of the grid rows under the abelian group ``A`` of a k-point (operations ``r -> R r + t``)."""

    representative_rows: np.ndarray
    full_to_wedge: np.ndarray
    multiplicities: np.ndarray
    operations: np.ndarray
    translations: np.ndarray
    atom_representatives: np.ndarray | None = None
    atom_orbit_sizes: np.ndarray | None = None

    @property
    def wedge_size(self) -> int:
        return int(self.representative_rows.size)

    @property
    def full_size(self) -> int:
        return int(self.full_to_wedge.size)

    @property
    def group_order(self) -> int:
        return int(self.operations.shape[0])


def complex_reduction(cell, elements) -> tuple[ComplexReduction, np.ndarray] | None:
    """``(reduction, images)`` of ``elements`` (indices of the cell group, the identity first; ``images[a, w]`` the
    row of the image of representative ``w`` under element ``a``), ``None`` where an element misses the grid."""

    from .axis_reflection import _RowOrderTable, _atom_orbits, _fast_maps_requested, _orbit_labels

    row_table = _RowOrderTable(cell.grid) if _fast_maps_requested() else None
    smallest = None
    for index in elements:
        mapping = cell.row_map(int(index), table=row_table)
        if mapping is None:
            return None
        if smallest is None:
            smallest = np.array(mapping, dtype=np.int64, copy=True)
        else:
            np.minimum(smallest, mapping, out=smallest)
        del mapping
    representatives, full_to_wedge, multiplicities = _orbit_labels([smallest])
    del smallest
    representatives = np.ascontiguousarray(representatives, dtype=np.int64)
    images = np.empty((len(elements), representatives.size), dtype=np.int64)
    for position, index in enumerate(elements):
        found = cell.row_map(int(index), rows=representatives)
        if found is None:
            return None
        images[position] = found
    operations = np.ascontiguousarray([cell.matrices[int(index)] for index in elements], dtype=np.int8)
    translations = np.ascontiguousarray([cell.shifts[int(index)] for index in elements], dtype=np.int64)
    orbits = _atom_orbits(cell.atoms, list(operations), list(translations), cell.grid, cell.atom_tolerance)
    reduction = ComplexReduction(
        representative_rows=representatives,
        full_to_wedge=np.ascontiguousarray(full_to_wedge, dtype=np.int64),
        multiplicities=np.ascontiguousarray(multiplicities, dtype=np.int64),
        operations=operations,
        translations=translations,
        atom_representatives=None if orbits is None else orbits[0],
        atom_orbit_sizes=None if orbits is None else orbits[1],
    )
    return reduction, images


class _PhaseRows:
    """``phases[s]``: the complex phase of every grid row in sector ``s`` (0 where its orbit is not admitted)."""

    def __init__(self, codes: np.ndarray, modulus: int) -> None:
        self.codes = codes
        self.roots = np.exp(2j * np.pi * np.arange(modulus) / modulus)
        self.roots[np.abs(self.roots.real) < 1.0e-15] = 1j * np.sign(self.roots[np.abs(self.roots.real) < 1.0e-15].imag)
        self.roots[np.abs(self.roots.imag) < 1.0e-15] = np.sign(self.roots[np.abs(self.roots.imag) < 1.0e-15].real)

    def __getitem__(self, sector):
        codes = self.codes[int(sector)]
        out = np.zeros(codes.shape, dtype=np.complex128)
        admitted = codes >= 0
        out[admitted] = self.roots[codes[admitted]]
        return out

    @property
    def shape(self):
        return self.codes.shape


class ComplexCharacterDecomposition:
    """The sectors of the complex characters of an abelian group of a k-point: the interface of
    :class:`.representations.ReflectionRepresentationDecomposition` that :mod:`..SCF.kpoint_sectors` reads
    (``reduction``, ``phases[s]``, ``orbit_to_sector``, ``characters``, the sizes and orbits of the sectors, the orbit
    means of a field and the reduced operators), with ``exponents`` and ``modulus`` of the characters."""

    is_complex = True

    def __init__(self, reduction: ComplexReduction, exponents: np.ndarray, modulus: int, codes: np.ndarray,
                 orbit_to_sector: np.ndarray) -> None:
        self.reduction = reduction
        self.exponents = np.ascontiguousarray(exponents, dtype=np.int64)
        self.modulus = int(modulus)
        self.codes = codes
        self.orbit_to_sector = orbit_to_sector
        self.phases = _PhaseRows(codes, self.modulus)
        self._orbit_lists: dict[int, np.ndarray] = {}

    @classmethod
    def build(cls, reduction: ComplexReduction, images: np.ndarray, exponents: np.ndarray, modulus: int):
        count, order = exponents.shape
        wedge = reduction.wedge_size
        stabilizer = images == reduction.representative_rows[None, :]
        codes = np.full((count, reduction.full_size), -1, dtype=np.int8)
        orbit_to_sector = np.full((count, wedge), -1, dtype=np.int64)
        for character in range(count):
            admitted = np.all(~stabilizer | (exponents[character][:, None] == 0), axis=0)
            columns = np.flatnonzero(admitted)
            if columns.size == 0:
                raise RuntimeError("a character admits no orbit")
            orbit_to_sector[character, columns] = np.arange(columns.size, dtype=np.int64)
            for element in range(order):
                codes[character, images[element, columns]] = exponents[character, element]
        if not np.all(codes[:, reduction.representative_rows] <= 0):
            raise RuntimeError("invalid phase convention of a complex sector")
        return cls(reduction, exponents, modulus, codes, orbit_to_sector)

    @property
    def characters(self) -> np.ndarray:
        return np.exp(2j * np.pi * self.exponents / self.modulus)

    @property
    def representation_count(self) -> int:
        return int(self.exponents.shape[0])

    @property
    def full_size(self) -> int:
        return self.reduction.full_size

    @property
    def wedge_size(self) -> int:
        return self.reduction.wedge_size

    @property
    def sector_sizes(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(row >= 0)) for row in self.orbit_to_sector)

    def sector_size(self, representation: int) -> int:
        return int(np.count_nonzero(self.orbit_to_sector[int(representation)] >= 0))

    def sector_orbit_indices(self, representation: int) -> np.ndarray:
        index = int(representation)
        found = self._orbit_lists.get(index)
        if found is None:
            mapping = self.orbit_to_sector[index]
            admitted = np.flatnonzero(mapping >= 0)
            found = np.ascontiguousarray(admitted[np.argsort(mapping[admitted], kind="stable")], dtype=np.int64)
            found.setflags(write=False)
            self._orbit_lists[index] = found
        return found

    def invariant_wedge_values(self, values: np.ndarray) -> np.ndarray:
        """The mean of a field over each orbit."""

        sums = np.bincount(self.reduction.full_to_wedge, weights=np.asarray(values, dtype=np.float64),
                           minlength=self.wedge_size)
        return sums / self.reduction.multiplicities

    def basis(self, representation: int) -> sp.csr_matrix:
        """``U`` of a sector (rows of the grid by the sector's columns), complex."""

        index = int(representation)
        rows = np.arange(self.full_size, dtype=np.int64)
        orbits = self.reduction.full_to_wedge
        columns = self.orbit_to_sector[index][orbits]
        keep = columns >= 0
        values = self.phases[index][keep] / np.sqrt(self.reduction.multiplicities[orbits[keep]])
        return sp.csr_matrix((values, (rows[keep], columns[keep])), shape=(self.full_size, self.sector_size(index)))

    def reduce_hermitian_operators(self, matrix, representations=None):
        """``U^H M U`` of every sector of ``representations`` (all where ``None``), indexed by sector."""

        wanted = range(self.representation_count) if representations is None else list(representations)
        matrix = sp.csr_matrix(matrix, dtype=np.complex128)
        found: list = [None] * self.representation_count
        for index in wanted:
            basis = self.basis(index)
            found[int(index)] = (basis.conj().T @ matrix @ basis).tocsr()
        return found

    def reduce_nonlocal_operators(self, operator, representations=None):
        """``U^H B`` of the projectors of ``operator`` for every sector of ``representations``, indexed by sector."""

        from parsec_python.V_ion import NonlocalProjectorOperator

        wanted = range(self.representation_count) if representations is None else list(representations)
        projectors = sp.csr_matrix(operator.projectors, dtype=np.complex128)
        found: list = [None] * self.representation_count
        for index in wanted:
            reduced = (self.basis(index).conj().T @ projectors).tocsc()
            reduced.sum_duplicates()
            reduced.eliminate_zeros()
            reduced.sort_indices()
            found[int(index)] = NonlocalProjectorOperator(
                projectors=reduced,
                signs=np.asarray(operator.signs, dtype=np.float64).copy(),
                labels=tuple(operator.labels),
            )
        return found


def complex_decomposition(cell, elements) -> ComplexCharacterDecomposition | None:
    """The :class:`ComplexCharacterDecomposition` of the abelian group ``elements`` of ``cell``, ``None`` where an
    element misses the grid or a character admits no orbit."""

    built = complex_reduction(cell, elements)
    if built is None:
        return None
    reduction, images = built
    exponents, modulus = character_exponents(cell.table, elements)
    try:
        return ComplexCharacterDecomposition.build(reduction, images, exponents, modulus)
    except RuntimeError:
        return None


__all__ = [
    "ComplexCharacterDecomposition",
    "ComplexReduction",
    "character_exponents",
    "complex_decomposition",
    "complex_reduction",
    "largest_abelian_subgroup",
    "sector_characters_setting",
]
