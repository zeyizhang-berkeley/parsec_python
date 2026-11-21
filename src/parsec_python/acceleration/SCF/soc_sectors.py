"""Double-group sectors and Kramers pairs of the self-consistent spin-orbit loop of a cluster.

``PARSEC_SOC_SECTORS`` (``off`` by default; ``kramers``, ``on``).  PARSEC 1.5 drops to the identity with spin-orbit
coupling, and so does the accelerated spinor loop of :mod:`.self_consistent_soc` without this switch: it filters
``N_e + 6`` complex spinors of ``2 n`` rows.  With it, the loop solves the sectors of the double group of the cluster
that hold one state of every Kramers pair:

* **The Kramers sector** (``kramers`` and ``on``).  The point group of the atoms on the grid is the subgroup of the
  reflections and two-fold rotations along the axes
  (:class:`~parsec_python.acceleration.Symmetry.axis_reflection.AxisReflectionReduction`).  A two-fold rotation or a
  mirror ``g`` with its axis along ``a`` acts on spinors as ``-i sigma_a`` times its spatial map, so ``g^2 = -1`` and
  its eigenvalues are ``+/- i``.  Time reversal ``T = -i sigma_y K`` commutes with ``g`` and sends the ``+i`` sector to
  the ``-i`` one, so the ``+i`` sector holds exactly one state of every Kramers pair: half the states on half the rows
  (``n`` complex).  In it the component along the spin state ``|+a>`` is odd under the spatial ``g`` and the one along
  ``|-a>`` even.  ``g`` is taken along ``z`` where the group has one (``C2z`` or ``sigma_z``), else along ``x`` or
  ``y``.
* **Real arithmetic.**  When ``g`` lies along ``z`` and the group also holds an operation ``h`` along ``x`` or ``y``
  (a two-fold rotation or a mirror), ``A = T h`` is antiunitary with ``A^2 = +1`` and maps the sector onto itself
  (``h`` anticommutes with ``g`` in the double group), so the sector has a real basis: the operator restricted to it
  is a real symmetric matrix of ``n`` rows.  The basis is made of the real symmetry functions of the scalar group
  ``<g, h>`` (two characters per spin component) times a phase per block, so the kinetic term, the local potential
  and the Kleinman-Bylander projectors stay block diagonal and as sparse as on the whole grid; only the spin-orbit
  projectors couple the blocks.  Without such an ``h`` the sector is complex with ``n`` rows.
* **Parity** (``on``).  Inversion is spin-blind and commutes with every operation, so a cluster with a centre of
  inversion splits the sector further into its even and odd halves, two independent problems of about ``n / 2``
  rows.  The number of pairs each one solves is first set by its share of the rows (with a margin); when every
  state a parity solved lies among the lowest pairs of both, it may miss some, so its count grows and it is solved
  again.
* Clusters without a two-fold rotation or mirror (``Ci``, ``C1``) keep the whole grid: the run says so in a note
  and takes the former route.

Every reduced state stands for a Kramers pair: its eigenvalue enters the occupations twice, as the two states of
the whole-grid route, and its density counts with the occupations of both (the partner ``T psi`` has the same
density point by point).  The Hartree and exchange-correlation potentials, the mixing and the energies stay those of
the reference loop on the whole grid; the final spinors come back to the host as the ``N`` states of the whole-grid
route (each reduced state and its partner).

The folded matrices are ``S^T K S`` (kinetic), ``S^T P`` (projectors) and ``P_so^H S`` (spin-orbit projectors) of
the sparse symmetry functions ``S`` of every block, built once with SciPy and uploaded once; an SCF iteration
uploads only the local potential of the sector (the orbit averages of the potential).  With
``PARSEC_SYMMETRY_ATOM_TOLERANCE`` above its default the atoms may be off their symmetric places by that much; the
sectors then solve the operator projected on them (the orbit averages of the potential, the projectors of every
atom folded), which differs from that of the input by about that distance.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

SWITCH = "PARSEC_SOC_SECTORS"
_SETTINGS = ("off", "kramers", "on")

# Sign operations (x, y, z) -> (s_x x, s_y y, s_z z) of the axis-reflection group.
_C2Z = (-1, -1, 1)
_MZ = (1, 1, -1)
_INVERSION = (-1, -1, -1)
# Operations with their axis along x or y: C2x, C2y, sigma_x, sigma_y.
_PERPENDICULAR = ((1, -1, -1), (-1, 1, -1), (-1, 1, 1), (1, -1, 1))
# The choices of g in order: along z, then along x, then along y.
_FIRST = (_C2Z, _MZ, (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (1, -1, 1))
_ROOT_HALF = 2.0 ** -0.5
# |+a>, |-a> (eigenstates of sigma_a with +1, -1) as (up, down) amplitudes, per axis a.
_SPIN_STATES = {
    0: ((_ROOT_HALF + 0j, _ROOT_HALF + 0j), (_ROOT_HALF + 0j, -_ROOT_HALF + 0j)),
    1: ((_ROOT_HALF + 0j, 1j * _ROOT_HALF), (_ROOT_HALF + 0j, -1j * _ROOT_HALF)),
    2: ((1.0 + 0j, 0j), (0j, 1.0 + 0j)),
}
_NAMES = {
    (1, 1, 1): "E",
    (-1, -1, 1): "C2z",
    (-1, 1, -1): "C2y",
    (1, -1, -1): "C2x",
    (1, 1, -1): "sigma_z",
    (1, -1, 1): "sigma_y",
    (-1, 1, 1): "sigma_x",
    (-1, -1, -1): "i",
}


def soc_sectors_setting() -> str:
    """``PARSEC_SOC_SECTORS``: ``off`` (default, the whole grid), ``kramers`` (the ``+i`` sector of a two-fold
    operation: one state per Kramers pair) or ``on`` (that sector split by inversion parity where the cluster has a
    centre of inversion).  A cluster without a two-fold rotation or mirror keeps the whole grid with a note.  Another
    value is an error before the SCF."""

    text = os.environ.get(SWITCH, "off").strip().lower() or "off"
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be one of {', '.join(_SETTINGS)}, not {text!r}")
    return text


def _spin_axis(signs: tuple[int, ...]) -> int | None:
    """The axis of the spin rotation of a sign operation: a two-fold rotation turns about its +1 axis, a mirror is
    the inversion (spin-blind) times the rotation about its -1 axis; ``E`` and ``i`` turn no spin."""

    minus = [axis for axis, value in enumerate(signs) if value < 0]
    if len(minus) == 2:
        return 3 - sum(minus)
    if len(minus) == 1:
        return minus[0]
    return None


def _phase(spin: int, character: int, axis: int | None) -> complex:
    """The phase ``c`` of the block (spin, character of ``h``) that makes ``c f`` (``f`` real) real under
    ``A = T h``: ``c^2 = -i chi`` (upper) or ``i chi`` (lower) for ``h`` about ``x``, ``c^2 = -chi`` about ``y``."""

    if axis is None:
        return 1.0 + 0.0j
    if axis == 0:
        square = (-1j if spin == 0 else 1j) * character
    else:
        square = -float(character) + 0.0j
    return complex(np.sqrt(complex(square)))


@dataclasses.dataclass
class SectorBlock:
    """One block of the reduced basis: the real symmetry functions ``basis`` (``n x size``, orthonormal columns) of
    one character of the scalar group times the spin state ``spinor`` (the up and down amplitudes of ``|+a>`` or
    ``|-a>`` along the axis ``a`` of ``g``: ``spin`` 0 or 1) and ``phase``; ``sector`` is its parity sector."""

    spin: int
    spinor: tuple[complex, complex]
    characters: tuple[int, ...]
    phase: complex
    basis: sp.csr_matrix
    sector: int

    @property
    def size(self) -> int:
        return int(self.basis.shape[1])


@dataclasses.dataclass
class SpinorSectorPlan:
    """The reduced bases of the sectors (see the module docstring).

    ``real``: an ``h`` exists and the reduced operators are real symmetric.  ``generators`` names ``g`` (and ``h``,
    and ``i`` with parity sectors), ``group`` the operations the atoms and the grid keep, ``parities`` the inversion
    parity of every sector (``(None,)`` for one sector).
    """

    grid_size: int
    blocks: tuple[SectorBlock, ...]
    real: bool
    generators: tuple[str, ...]
    group: tuple[str, ...]
    parities: tuple[int | None, ...]

    @property
    def sector_count(self) -> int:
        return len(self.parities)

    def sector_blocks(self, sector: int) -> tuple[SectorBlock, ...]:
        return tuple(block for block in self.blocks if block.sector == sector)

    def sector_dimension(self, sector: int) -> int:
        return sum(block.size for block in self.sector_blocks(sector))

    @property
    def dimension(self) -> int:
        return sum(block.size for block in self.blocks)

    def description(self) -> str:
        arithmetic = "real" if self.real else "complex"
        text = f"group {{{', '.join(self.group)}}}; sector +i of {self.generators[0]}"
        if self.real:
            text += f" made real by T {self.generators[1]}"
        if self.sector_count > 1:
            sizes = " and ".join(
                f"{self.sector_dimension(index)} ({'even' if parity > 0 else 'odd'})"
                for index, parity in enumerate(self.parities)
            )
            text += f", split by the parity of i: {sizes} {arithmetic} rows"
        else:
            text += f": {self.dimension} {arithmetic} rows"
        return text + (
            f" for one state of each Kramers pair (whole grid: {2 * self.grid_size} complex rows for both)"
        )


def _character_basis(maps, characters, representatives: np.ndarray, size: int) -> sp.csr_matrix:
    """Orthonormal real functions ``sum_g chi(g) e_(g r)`` of the orbits of ``representatives``; the orbits whose
    stabilizer has a character -1 give zero and are left out."""

    count = int(representatives.size)
    rows = np.concatenate([np.asarray(mapping)[representatives] for mapping in maps])
    columns = np.tile(np.arange(count, dtype=np.int64), len(maps))
    data = np.repeat(np.asarray(characters, dtype=np.float64), count)
    matrix = sp.csr_matrix((data, (rows, columns)), shape=(size, count))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    norms = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
    keep = np.flatnonzero(norms > 0.5)
    matrix = matrix.tocsc()[:, keep] @ sp.diags(1.0 / np.sqrt(norms[keep]))
    matrix = sp.csr_matrix(matrix)
    matrix.sort_indices()
    return matrix


@dataclasses.dataclass(frozen=True)
class _SectorGenerators:
    """The operations of a sector route: ``group`` (names of the operations the atoms and the grid keep), ``g``,
    the real-making ``h`` (or ``None``) and whether the parity of ``i`` splits the sector; ``g`` is ``None`` with
    ``reason`` for a cluster without a two-fold rotation or mirror."""

    group: tuple[str, ...]
    first: tuple[int, ...] | None
    second: tuple[int, ...] | None
    inversion: bool
    reason: str

    def describe(self) -> str:
        """The generators in words (the ``orbital_symmetry`` line of the report)."""
        text = f"sector +i of {_NAMES[self.first]}"
        if self.second is not None:
            text += f" made real by T {_NAMES[self.second]}"
        if self.inversion:
            text += ", split by the parity of i"
        return text + f" in the double group of {{{', '.join(self.group)}}}"


def _sector_generators(grid, atoms, *, parity: bool, atom_tolerance: float | None) -> _SectorGenerators:
    """The group of the atoms on the grid and the operations :func:`detect_sector_plan` builds the sectors of."""

    from ..Symmetry.axis_reflection import AxisReflectionReduction, symmetry_atom_tolerance

    tolerance = symmetry_atom_tolerance() if atom_tolerance is None else float(atom_tolerance)
    reduction = AxisReflectionReduction.detect(grid, atoms, atom_tolerance=tolerance)
    present = [tuple(int(value) for value in row) for row in np.asarray(reduction.signs)]
    group = tuple(_NAMES[signs] for signs in sorted(present, key=lambda s: list(_NAMES).index(s)))
    # g: along z if the group has such an operation (an h along x or y may then make the sector real), else along
    # x or y (then no operation is perpendicular to it: the product would lie along z).
    first = next((signs for signs in _FIRST if signs in present), None)
    if first is None:
        return _SectorGenerators(
            group=group,
            first=None,
            second=None,
            inversion=False,
            reason=f"the group {{{', '.join(group)}}} of the cluster on the grid has no two-fold rotation or mirror",
        )
    second = None
    if _spin_axis(first) == 2:
        second = next((signs for signs in _PERPENDICULAR if signs in present), None)
    return _SectorGenerators(
        group=group, first=first, second=second, inversion=bool(parity and _INVERSION in present), reason=""
    )


def soc_orbital_symmetry(grid, atoms, setting: str, *, atom_tolerance: float | None = None) -> tuple[str, str]:
    """What the report says of the orbitals of a run with ``PARSEC_SOC_SECTORS=setting`` (``kramers`` or ``on``)
    whose input allows symmetry, and the reason the spinors stay on the whole grid (``""`` where the sectors are
    used).  It finds the group as :func:`detect_sector_plan` does, without building the bases."""

    generators = _sector_generators(grid, atoms, parity=setting == "on", atom_tolerance=atom_tolerance)
    if generators.first is None:
        return f"full grid ({SWITCH}={setting}: {generators.reason})", generators.reason
    return (
        f"spin-orbit double-group {generators.describe()}, one state per Kramers pair ({SWITCH}={setting})",
        "",
    )


def detect_sector_plan(
    grid, atoms, *, parity: bool = True, atom_tolerance: float | None = None
) -> tuple[SpinorSectorPlan | None, str]:
    """The reduced bases of a cluster (split by inversion parity with ``parity`` where the cluster has a centre of
    inversion), or ``None`` with the reason the whole grid stays."""

    from ..Symmetry.axis_reflection import _grid_row_mapping, _orbit_labels

    chosen = _sector_generators(grid, atoms, parity=parity, atom_tolerance=atom_tolerance)
    group = chosen.group
    if chosen.first is None:
        return None, chosen.reason
    first, second, inversion = chosen.first, chosen.second, chosen.inversion
    first_axis = _spin_axis(first)
    generators = [first] + ([second] if second is not None else []) + ([_INVERSION] if inversion else [])
    elements: list[tuple[tuple[int, ...], tuple[int, ...]]] = []  # (signs, exponents)
    for exponents in np.ndindex(*(2,) * len(generators)):
        signs = np.ones(3, dtype=np.int64)
        for generator, power in zip(generators, exponents):
            if power:
                signs = signs * np.asarray(generator)
        elements.append((tuple(int(value) for value in signs), tuple(int(value) for value in exponents)))
    maps = []
    for signs, _ in elements:
        mapping = _grid_row_mapping(grid, np.asarray(signs, dtype=np.int8), 1.0e-10)
        if mapping is None:  # the detector accepted it: cannot happen
            return None, f"the grid does not map onto itself under {_NAMES[signs]}"
        maps.append(np.asarray(mapping, dtype=np.int64))
    representatives, _, _ = _orbit_labels(maps)
    size = int(grid.size)
    axis = None if second is None else _spin_axis(second)
    spin_states = _SPIN_STATES[first_axis]
    parities: tuple[int | None, ...] = (1, -1) if inversion else (None,)
    blocks: list[SectorBlock] = []
    for sector, parity_character in enumerate(parities):
        for spin in (0, 1):
            # +i of -i sigma_a g: the component along |+a> is odd under g, the one along |-a> even.
            first_character = -1 if spin == 0 else 1
            for second_character in ((1, -1) if second is not None else (None,)):
                generator_characters = (
                    (first_character,)
                    + ((second_character,) if second_character is not None else ())
                    + ((parity_character,) if parity_character is not None else ())
                )
                characters = [
                    int(np.prod([chi if power else 1 for chi, power in zip(generator_characters, exponents)]))
                    for _, exponents in elements
                ]
                basis = _character_basis(maps, characters, representatives, size)
                phase = _phase(spin, second_character if second_character is not None else 1, axis)
                blocks.append(
                    SectorBlock(
                        spin=spin,
                        spinor=spin_states[spin],
                        characters=generator_characters,
                        phase=phase,
                        basis=basis,
                        sector=sector,
                    )
                )
    plan = SpinorSectorPlan(
        grid_size=size,
        blocks=tuple(blocks),
        real=second is not None,
        generators=tuple(_NAMES[signs] for signs in generators),
        group=group,
        parities=parities,
    )
    if plan.dimension != size:
        raise RuntimeError(f"the spinor sectors have {plan.dimension} rows, not the {size} of half the spinor space")
    if any(plan.sector_dimension(index) == 0 for index in range(plan.sector_count)):
        raise RuntimeError("a parity sector of the spinors is empty")
    return plan, ""


# --------------------------------------------------------------- folded host matrices
def _soc_projector_matrix(soc_projectors, size: int) -> sp.csr_matrix:
    """``[v_ion | v_so]`` of every atom as one ``(n, 16 atoms)`` matrix (the ``forward`` of
    :meth:`~parsec_python.acceleration.Hamiltonian.device_bloch.DeviceBlochConverter._soc_parts`)."""

    rows, columns, data = [], [], []
    for index, atom in enumerate(soc_projectors):
        support = np.asarray(atom.support_rows, dtype=np.int64)
        rows.append(np.repeat(support, 16))
        columns.append(np.tile(np.arange(16) + 16 * index, support.size))
        data.append(np.asarray(atom.stacked).reshape(-1))
    values = np.concatenate(data)
    return sp.coo_matrix(
        (values.astype(np.complex128), (np.concatenate(rows), np.concatenate(columns))),
        shape=(size, 16 * len(soc_projectors)),
    ).tocsr()


def _soc_signs(soc_projectors) -> np.ndarray:
    return np.asarray(
        [
            np.where(np.asarray([1, 1, 1, 2, 2, 2, 2, 2]) == 1, atom.sign[0], atom.sign[1])
            for atom in soc_projectors
        ],
        dtype=np.float64,
    )[:, :, None]


@dataclasses.dataclass
class FoldedParts:
    """The host matrices of the reduced operator of one sector (see :func:`fold_operator`)."""

    kinetic: sp.csr_matrix  # block diagonal S^T K S
    averages: sp.csr_matrix  # (rows, n): the orbit averages of a potential, (S o S)^T
    projectors: sp.csr_matrix | None  # block diagonal S^T P (rows, blocks * projectors)
    projector_signs: np.ndarray | None
    soc: sp.csr_matrix | None  # real: [Re G_up; Im G_up; Re G_dn; Im G_dn]; complex: [G_up; G_dn], G_s = P_so^H S c
    soc_signs: np.ndarray | None
    soc_atoms: int
    unfold: sp.csr_matrix  # real: [Re U_up; Im U_up; Re U_dn; Im U_dn] (4n, rows); complex: [U_up; U_dn] (2n, rows)
    real: bool
    grid_size: int

    @property
    def dimension(self) -> int:
        return int(self.kinetic.shape[0])


def fold_operator(
    plan: SpinorSectorPlan, laplacian, nonlocal_operator, soc_projectors, sector: int = 0
) -> FoldedParts:
    """Fold the static terms of the spinor operator into the reduced basis of one sector of ``plan``."""

    size = plan.grid_size
    blocks = plan.sector_blocks(sector)
    laplacian = sp.csr_matrix(laplacian, dtype=np.float64)
    kinetic_blocks, average_blocks, projector_blocks = [], [], []
    projectors = None
    if nonlocal_operator is not None and nonlocal_operator.projectors.shape[1]:
        projectors = nonlocal_operator.projectors
        if np.iscomplexobj(projectors.data) and np.any(projectors.data.imag != 0):
            raise NotImplementedError("the sectors of the spin-orbit loop take real Kleinman-Bylander projectors")
        projectors = sp.csr_matrix(
            projectors.real if np.iscomplexobj(projectors.data) else projectors, dtype=np.float64
        )
    for block in blocks:
        basis = block.basis
        transposed = basis.T.tocsr()
        kinetic_blocks.append((transposed @ (laplacian @ basis)).tocsr())
        average_blocks.append(transposed.multiply(transposed).tocsr())
        if projectors is not None:
            projector_blocks.append((transposed @ projectors).tocsr())
    kinetic = sp.block_diag(kinetic_blocks, format="csr")
    kinetic.sort_indices()
    averages = sp.vstack(average_blocks, format="csr")
    folded_projectors = None
    projector_signs = None
    if projectors is not None:
        folded_projectors = sp.block_diag(projector_blocks, format="csr")
        projector_signs = np.tile(np.asarray(nonlocal_operator.signs, dtype=np.float64), len(blocks))
    soc = None
    soc_signs = None
    atoms = len(soc_projectors)
    if atoms:
        adjoint = _soc_projector_matrix(soc_projectors, size).conj().T.tocsr()
        folded = [(adjoint @ block.basis).tocsr() for block in blocks]
        per_spin = []
        for spin in (0, 1):
            pieces = []
            for block, matrix in zip(blocks, folded):
                amplitude = block.spinor[spin] * block.phase
                if amplitude != 0:
                    pieces.append(matrix * amplitude)
                else:
                    pieces.append(sp.csr_matrix((16 * atoms, block.size), dtype=np.complex128))
            per_spin.append(sp.hstack(pieces, format="csr"))
        if plan.real:
            stacked = [part for matrix in per_spin for part in (matrix.real, matrix.imag)]
            soc = sp.vstack(stacked, format="csr").astype(np.float64)
        else:
            soc = sp.vstack(per_spin, format="csr")
            if not np.any(soc.data.imag):
                soc = soc.real.tocsr()
        soc.eliminate_zeros()
        soc_signs = _soc_signs(soc_projectors)
    unfold_rows = []
    for spin in (0, 1):
        pieces = []
        for block in blocks:
            amplitude = block.spinor[spin] * block.phase
            if amplitude != 0:
                pieces.append(block.basis * amplitude)
            else:
                pieces.append(sp.csr_matrix((size, block.size), dtype=np.complex128))
        unfold_rows.append(sp.hstack(pieces, format="csr").astype(np.complex128))
    if plan.real:
        unfold = sp.vstack([part for matrix in unfold_rows for part in (matrix.real, matrix.imag)], format="csr")
        unfold = unfold.astype(np.float64)
    else:
        unfold = sp.vstack(unfold_rows, format="csr")
        if not np.any(unfold.data.imag):
            unfold = unfold.real.tocsr()
    unfold.eliminate_zeros()
    return FoldedParts(
        kinetic=kinetic,
        averages=averages,
        projectors=folded_projectors,
        projector_signs=projector_signs,
        soc=soc,
        soc_signs=soc_signs,
        soc_atoms=atoms,
        unfold=unfold,
        real=plan.real,
        grid_size=size,
    )


# ----------------------------------------------------------------------- device side
class DeviceSectorParts:
    """The folded matrices of one sector on one device (uploaded once)."""

    def __init__(self, backend, folded: FoldedParts) -> None:
        from ..Hamiltonian.device_bloch import _ladder_and_lz

        xp = backend.xp
        self.backend = backend
        self.xp = xp
        self.real = folded.real
        self.dimension = folded.dimension
        self.grid_size = folded.grid_size
        self.kinetic = backend.matrix(folded.kinetic, dtype=np.float64)
        self.averages = folded.averages
        if folded.projectors is not None:
            self.projectors = backend.matrix(folded.projectors, dtype=np.float64)
            self.projectors_adjoint = backend.matrix(folded.projectors.T.tocsr(), dtype=np.float64)
            self.projector_signs = backend.asarray(folded.projector_signs)
        else:
            self.projectors = self.projectors_adjoint = self.projector_signs = None
        self.atoms = folded.soc_atoms
        if folded.soc is not None:
            soc_dtype = np.float64 if not np.iscomplexobj(folded.soc.data) else np.complex128
            self.soc_dtype = soc_dtype
            self.soc = backend.matrix(folded.soc, dtype=soc_dtype)
            self.soc_adjoint = backend.matrix(folded.soc.conj().T.tocsr(), dtype=soc_dtype)
            self.soc_signs = backend.asarray(folded.soc_signs)
            ladder_down, ladder_up, lz_half, quarter_ll1, _ = _ladder_and_lz()
            self.ladder_from_down = xp.asarray(ladder_down).astype(xp.complex128)
            self.ladder_from_up = xp.asarray(ladder_up).astype(xp.complex128)
            self.lz_half = xp.asarray(lz_half)[:, None]
            self.quarter_ll1 = xp.asarray(quarter_ll1)[:, None]
        else:
            self.soc = None
        unfold_dtype = np.float64 if not np.iscomplexobj(folded.unfold.data) else np.complex128
        self.unfold = backend.matrix(folded.unfold, dtype=unfold_dtype)
        self.unfold_complex = unfold_dtype == np.complex128

    def potential(self, host_potential: np.ndarray):
        """The local potential of the sector rows: the orbit averages of the whole-grid potential."""
        values = self.averages @ np.asarray(host_potential, dtype=np.float64)
        return self.backend.asarray(np.ascontiguousarray(values))

    # ------------------------------------------------------------- products
    def product(self, matrix, block, matrix_complex: bool = False):
        """``matrix @ block``, C-contiguous; a real matrix on a complex block runs on its real view."""
        xp = self.xp
        if not matrix_complex and block.dtype == xp.complex128:
            return xp.ascontiguousarray(matrix @ block.view(xp.float64)).view(xp.complex128)
        return xp.ascontiguousarray(matrix @ block)

    def spin_orbit(self, block):
        """The spin-orbit term of the reduced block (``DeviceBlochOperator._spin_orbit`` in the sector rows)."""
        xp = self.xp
        atoms = self.atoms
        rows = 16 * atoms
        vectors = int(block.shape[1])
        soc_complex = self.soc_dtype == np.complex128
        if self.real:
            stacked = self.product(self.soc, block)  # (4 rows, vectors) real
            overlaps_up = stacked[:rows] + 1j * stacked[rows : 2 * rows]
            overlaps_dn = stacked[2 * rows : 3 * rows] + 1j * stacked[3 * rows :]
        else:
            if soc_complex and block.dtype != xp.complex128:
                block = block.astype(xp.complex128)
            stacked = self.product(self.soc, block, soc_complex)
            overlaps_up, overlaps_dn = stacked[:rows], stacked[rows:]
        sign = self.soc_signs
        overlaps_up = overlaps_up.reshape(atoms, 16, vectors)
        overlaps_dn = overlaps_dn.reshape(atoms, 16, vectors)
        io_up, so_up = sign * overlaps_up[:, :8], sign * overlaps_up[:, 8:]
        io_dn, so_dn = sign * overlaps_dn[:, :8], sign * overlaps_dn[:, 8:]
        coefficients = xp.empty((2, atoms, 16, vectors), dtype=xp.complex128)
        for spin, (sigma, io_same, so_same, ladder, io_other, so_other) in enumerate(
            (
                (1.0, io_up, so_up, self.ladder_from_down, io_dn, so_dn),
                (-1.0, io_dn, so_dn, self.ladder_from_up, io_up, so_up),
            )
        ):
            lz = sigma * self.lz_half
            coefficients[spin, :, :8] = lz * so_same + xp.matmul(ladder, so_other)
            coefficients[spin, :, 8:] = (
                lz * io_same
                + (self.quarter_ll1 - 0.5 * lz) * so_same
                + xp.matmul(ladder, io_other - 0.5 * so_other)
            )
        coefficients = coefficients.reshape(2 * rows, vectors)
        if self.real:
            # Re(G^H c) = Re(G)^T Re(c) + Im(G)^T Im(c), in the stacked order of the rows of ``soc``
            parts = xp.empty((4 * rows, vectors), dtype=xp.float64)
            parts[:rows] = coefficients[:rows].real
            parts[rows : 2 * rows] = coefficients[:rows].imag
            parts[2 * rows : 3 * rows] = coefficients[rows:].real
            parts[3 * rows :] = coefficients[rows:].imag
            return self.product(self.soc_adjoint, parts)
        return self.product(self.soc_adjoint, coefficients, soc_complex)

    def unfold_block(self, block):
        """The whole-grid spinors ``(2n, vectors)`` complex of reduced columns."""
        xp = self.xp
        n = self.grid_size
        if self.real:
            parts = self.product(self.unfold, block)
            return xp.concatenate((parts[:n] + 1j * parts[n : 2 * n], parts[2 * n : 3 * n] + 1j * parts[3 * n :]))
        if self.unfold_complex and block.dtype != xp.complex128:
            block = block.astype(xp.complex128)
        return self.product(self.unfold, block, self.unfold_complex)

    def density(self, vectors, weights: np.ndarray, chunk: int):
        """``sum_j w_j |psi_j|^2`` (device, grid length) of the unfolded columns of ``vectors``."""
        xp = self.xp
        n = self.grid_size
        total = xp.zeros(n, dtype=xp.float64)
        for c0 in range(0, weights.size, chunk):
            c1 = min(c0 + chunk, weights.size)
            if not np.any(weights[c0:c1]):
                continue
            block = xp.ascontiguousarray(vectors[:, c0:c1])
            if self.real:
                values = self.product(self.unfold, block)  # (4n, columns): Re/Im of up, Re/Im of down
                squares = values * values
            else:
                values = self.unfold_block(block)
                squares = values.real * values.real + values.imag * values.imag
            per_point = squares.reshape(-1, n, c1 - c0).sum(axis=0)
            total += per_point @ xp.asarray(np.ascontiguousarray(weights[c0:c1]))
        return total


class SectorSpinorOperator:
    """The spinor operator of one SCF iteration in the reduced basis of one sector (``shape``, ``apply``)."""

    def __init__(self, parts: DeviceSectorParts, diagonal) -> None:
        self.parts = parts
        self.diagonal = diagonal
        self.shape = (parts.dimension, parts.dimension)
        self.real_arithmetic = parts.real

    def as_eigensolver_operator(self) -> "SectorSpinorOperator":
        return self

    def apply(self, block):
        parts = self.parts
        result = parts.product(parts.kinetic, block)
        result += self.diagonal[:, None] * block
        if parts.projectors is not None:
            coefficients = parts.product(parts.projectors_adjoint, block)
            coefficients *= parts.projector_signs[:, None]
            result += parts.product(parts.projectors, coefficients)
        if parts.soc is not None:
            result += parts.spin_orbit(block)
        return result


class SectorOperators:
    """The operators of every sector of one SCF iteration (what the hooks pass to the eigensolver hook)."""

    def __init__(self, operators: list[SectorSpinorOperator]) -> None:
        self.sectors = operators
        dimension = sum(operator.shape[0] for operator in operators)
        self.shape = (dimension, dimension)

    def as_eigensolver_operator(self) -> "SectorOperators":
        return self


@dataclasses.dataclass
class SectorSpinors:
    """The orbitals of a sector solve: per sector its ``vectors`` (reduced rows x solved pairs) on the device and
    the place of every column among the ``count // 2`` lowest pairs (``-1``: not among them); each column stands for
    one Kramers pair of the ``count`` whole-grid states."""

    parts: list[DeviceSectorParts]
    vectors: list[Any]
    places: list[np.ndarray]
    count: int

    @classmethod
    def single(cls, parts: DeviceSectorParts, vectors, count: int) -> "SectorSpinors":
        columns = int(vectors.shape[1])
        return cls([parts], [vectors], [np.arange(columns, dtype=np.int64)], int(count))

    @property
    def shape(self) -> tuple[int, int]:
        return (2 * self.parts[0].grid_size, self.count)

    def pair_weights(self, occupations: np.ndarray) -> np.ndarray:
        """The occupation of every pair: the sum of those of its two states (the second may lie beyond ``count``)."""
        occupations = np.asarray(occupations, dtype=np.float64)
        pairs = (self.count + 1) // 2
        padded = np.zeros(2 * pairs)
        padded[: occupations.size] = occupations
        return padded[0::2] + padded[1::2]

    def density(self, occupations: np.ndarray, volume_element: float, chunk: int) -> np.ndarray:
        weights = self.pair_weights(occupations)
        total = None
        for parts, vectors, places in zip(self.parts, self.vectors, self.places):
            sector_weights = np.where(places >= 0, weights[np.maximum(places, 0)], 0.0)
            part = parts.density(vectors, sector_weights, chunk)
            total = part if total is None else total + part
        backend = self.parts[0].backend
        return np.asarray(backend.asnumpy(total), dtype=np.float64) / volume_element

    def full_spinors(self, chunk: int) -> np.ndarray:
        """The ``count`` whole-grid spinors on the host: each pair as ``psi`` then ``T psi = (-conj(d), conj(u))``.

        The pairs are taken in the order of their places, ``chunk // 2`` at a time from whichever sector holds
        them, unfolded and interleaved on the device (``psi``, ``T psi``, ...) and brought to the host in one copy
        that fills consecutive columns of the output: every row of the host array is written in one stretch."""
        n = self.parts[0].grid_size
        count = self.count
        pairs = (count + 1) // 2
        backend = self.parts[0].backend
        xp = self.parts[0].xp
        # the sector and the column that hold every place (-1: none, its two columns stay zero)
        owner_sector = np.full(pairs, -1, dtype=np.int64)
        owner_column = np.full(pairs, -1, dtype=np.int64)
        for sector, places in enumerate(self.places):
            columns = np.flatnonzero((places >= 0) & (places < pairs))
            owner_sector[places[columns]] = sector
            owner_column[places[columns]] = columns
        output = np.empty((2 * n, count), dtype=np.complex128)
        step = max(1, int(chunk) // 2)
        for p0 in range(0, pairs, step):
            p1 = min(p0 + step, pairs)
            width = min(2 * p1, count) - 2 * p0
            owners = owner_sector[p0:p1]
            if np.all(owners < 0):
                output[:, 2 * p0 : 2 * p0 + width] = 0.0
                continue
            allocate = xp.zeros if np.any(owners < 0) else xp.empty
            combined = allocate((2 * n, 2 * (p1 - p0)), dtype=xp.complex128)
            for sector, (parts, vectors) in enumerate(zip(self.parts, self.vectors)):
                local = np.flatnonzero(owners == sector)
                if local.size == 0:
                    continue
                columns = owner_column[p0:p1][local]
                if np.all(np.diff(columns) == 1):
                    block = vectors[:, int(columns[0]) : int(columns[-1]) + 1]
                else:
                    block = vectors[:, xp.asarray(columns)]
                spinors = parts.unfold_block(xp.ascontiguousarray(block))
                if local.size == p1 - p0:
                    # every pair of the chunk from this sector, in order (always so for one sector)
                    first, second = slice(0, None, 2), slice(1, None, 2)
                else:
                    first, second = xp.asarray(2 * local), xp.asarray(2 * local + 1)
                combined[:, first] = spinors
                combined[:n, second] = -xp.conj(spinors[n:])
                combined[n:, second] = xp.conj(spinors[:n])
                del spinors
            host = backend.asnumpy(combined)
            del combined
            output[:, 2 * p0 : 2 * p0 + width] = host[:, :width]
        return output


@dataclasses.dataclass
class SectorEigvalResult:
    eigenvalues: np.ndarray
    vectors: SectorSpinors
    state: Any
    stats: dict


@dataclasses.dataclass
class _SectorStates:
    """The eigensolver states of the sectors and the number of pairs each one solves."""

    counts: list[int]
    states: list[Any]


def _add_stats(total: dict, stats: dict) -> None:
    for key, value in stats.items():
        total[key] = total.get(key, 0.0) + value


class SectorRoute:
    """The hooks of :func:`parsec_python.SCF.self_consistent_soc.run_self_consistent_soc` for the sectors."""

    def __init__(self, plan: SpinorSectorPlan, backend, soc_projectors) -> None:
        self.plan = plan
        self.backend = backend
        self.soc_projectors = soc_projectors
        self.parts: list[DeviceSectorParts] | None = None
        self._anchors: tuple = ()
        self.setup_seconds = 0.0
        self.regrown = 0

    def _device_parts(self, laplacian, nonlocal_operator) -> list[DeviceSectorParts]:
        anchors = (laplacian, nonlocal_operator)
        if self.parts is None or any(a is not b for a, b in zip(anchors, self._anchors)):
            start = time.perf_counter()
            self.parts = [
                DeviceSectorParts(
                    self.backend, fold_operator(self.plan, laplacian, nonlocal_operator, self.soc_projectors, sector)
                )
                for sector in range(self.plan.sector_count)
            ]
            self.backend.synchronize()
            self.setup_seconds += time.perf_counter() - start
            self._anchors = anchors
        return self.parts

    def hamiltonian(self, scalar_hamiltonian, soc_projectors):
        from .self_consistent_soc import _host_parts

        if soc_projectors is not self.soc_projectors:
            self.soc_projectors = soc_projectors
            self.parts = None
        laplacian, nonlocal_operator, potential = _host_parts(scalar_hamiltonian)
        parts = self._device_parts(laplacian, nonlocal_operator)
        operators = [SectorSpinorOperator(part, part.potential(potential)) for part in parts]
        return operators[0] if len(operators) == 1 else SectorOperators(operators)

    def _first_counts(self, pairs: int) -> list[int]:
        """Pairs per sector of a first solve: its share of the rows and two more."""
        total = self.plan.dimension
        counts = []
        for index in range(self.plan.sector_count):
            dimension = self.plan.sector_dimension(index)
            share = pairs * dimension / total
            counts.append(int(min(dimension, max(1, math.ceil(share) + 2))))
        return counts

    def eigensolver(self, operator, requested_states: int, *, settings, state=None) -> SectorEigvalResult:
        from ..Eigensolvers.chebff_spinor import solve_spinor_eigval

        requested = int(requested_states)
        pairs = (requested + 1) // 2
        buffer_pairs = max(pairs, (requested + int(settings.safety_buffer) + 1) // 2) - pairs
        xp, sync = self.backend.xp, self.backend.synchronize
        if not isinstance(operator, SectorOperators):
            sector_settings = dataclasses.replace(settings, safety_buffer=buffer_pairs)
            result = solve_spinor_eigval(operator, pairs, settings=sector_settings, state=state, xp=xp, sync=sync)
            eigenvalues = np.repeat(np.asarray(result.eigenvalues, dtype=np.float64), 2)[:requested]
            return SectorEigvalResult(
                eigenvalues=eigenvalues,
                vectors=SectorSpinors.single(operator.parts, result.vectors, requested),
                state=result.state,
                stats=result.stats,
            )
        sectors = operator.sectors
        if state is None:
            state = _SectorStates(counts=self._first_counts(pairs), states=[None] * len(sectors))
        dimensions = [int(sector.shape[0]) for sector in sectors]
        total_dimension = sum(dimensions)
        stats: dict = {}
        results: list = [None] * len(sectors)
        pending = list(range(len(sectors)))
        while True:
            for index in pending:
                share = buffer_pairs * dimensions[index] / total_dimension
                buffer = min(max(2, math.ceil(share)), dimensions[index] - state.counts[index])
                sector_settings = dataclasses.replace(settings, safety_buffer=max(0, buffer))
                result = solve_spinor_eigval(
                    sectors[index],
                    state.counts[index],
                    settings=sector_settings,
                    state=state.states[index],
                    xp=xp,
                    sync=sync,
                )
                _add_stats(stats, result.stats)
                results[index] = result
                state.states[index] = result.state
            merged = sorted(
                (float(value), index, column)
                for index, result in enumerate(results)
                for column, value in enumerate(np.asarray(result.eigenvalues, dtype=np.float64))
            )
            if len(merged) < pairs:
                raise RuntimeError("the parity sectors solved fewer pairs than requested")
            cut = merged[pairs - 1][0]
            # a sector whose every solved pair lies at or below the cut may miss pairs below it
            pending = [
                index
                for index, result in enumerate(results)
                if float(np.max(result.eigenvalues)) <= cut and state.counts[index] < dimensions[index]
            ]
            if not pending:
                break
            for index in pending:
                state.counts[index] = min(
                    dimensions[index], state.counts[index] + max(2, math.ceil(0.25 * state.counts[index]))
                )
                state.states[index] = None  # a first solve with the larger count
                self.regrown += 1
        places = [np.full(int(result.eigenvalues.size), -1, dtype=np.int64) for result in results]
        values = []
        for place, (value, index, column) in enumerate(merged[:pairs]):
            places[index][column] = place
            values.append(value)
        eigenvalues = np.repeat(np.asarray(values, dtype=np.float64), 2)[:requested]
        spinors = SectorSpinors(
            parts=[sector.parts for sector in sectors],
            vectors=[result.vectors for result in results],
            places=places,
            count=requested,
        )
        stats["sector_pairs"] = float(sum(state.counts))
        return SectorEigvalResult(eigenvalues=eigenvalues, vectors=spinors, state=state, stats=stats)

    def density(self, spinors, occupations, volume_element) -> np.ndarray:
        from ..Eigensolvers.chebff_spinor import _filter_columns

        return spinors.density(occupations, volume_element, _filter_columns())

    def to_host(self, spinors) -> np.ndarray:
        from ..Eigensolvers.chebff_spinor import _filter_columns

        if isinstance(spinors, SectorSpinors):
            return spinors.full_spinors(_filter_columns())
        return self.backend.asnumpy(spinors)


__all__ = [
    "SWITCH",
    "DeviceSectorParts",
    "FoldedParts",
    "SectorOperators",
    "SectorRoute",
    "SectorSpinorOperator",
    "SectorSpinors",
    "SpinorSectorPlan",
    "detect_sector_plan",
    "fold_operator",
    "soc_sectors_setting",
]
