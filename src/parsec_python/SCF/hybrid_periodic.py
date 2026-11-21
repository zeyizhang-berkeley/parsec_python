"""Self-consistent hybrid functionals of a periodic cell with k-points: the exchange engine of the Fock loop.

The Fock loop is :func:`parsec_python.SCF.hybrid.run_hybrid_scf`, unchanged:
start from a semilocal SCF, ``V_ACE`` of the orbitals of the loop before,
inner SCF loops with the operator fixed, ``dexx = a_x [2 E_x(P, P0) -
E_x(P0) - E_x(P)]``, the stop rule, the adaptive inner criterion and the
outer DIIS of the cluster loop.  What a periodic cell with k-points changes
lives in :class:`BlochExchange`, the engine the loop asks for its pieces:

* the orbitals are the Bloch factors ``u_nk`` of the k-points solved, the
  representatives of the full Monkhorst-Pack mesh under time reversal
  (:class:`parsec_python.V_xc.bloch_exchange.BlochOrbitalSet`, one per spin
  channel or one unpolarized set), or of a single point's irreducible wedge
  under the grid operations of the space group and time reversal
  (``PARSEC_HYBRID_KPOINT_WEDGE``, :func:`hybrid_mesh_stars`: the other points
  of the mesh are the images of the wedge's factors, their grid points
  permuted);
* ``V_x u`` sums over the full mesh, ``-k`` taking ``conj(u_k)``, with the pair
  potentials of the cell's exchange kernel by FFT
  (:func:`parsec_python.V_xc.bloch_exchange.apply_bloch_exchange`), and the
  operator is the ACE of every (channel, k-point)
  (:class:`PeriodicFockOperator`);
* the SCF loops are the k-point drivers
  :func:`parsec_python.SCF.kpoints.run_scf_kpoints` and
  :func:`~parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized` with the
  operator in every Bloch Hamiltonian, ``E_x(P, P0)`` accumulated where each
  k-point's orbitals are, and the warm start of a Fock loop (the Ritz states
  of every k-point, the carried mixer, the relative criterion, the level
  passes);
* the energies are BZ averages: ``E_x = (s/2) sum_k w_k sum_n f_nk <u|V_x u>``
  and ``E_x(P, P0) = (s/2) sum_k w_k sum_n f_nk <u|V_ACE|u>`` (Ry);
* the outer DIIS keeps one block per (channel, k-point), its overlaps
  Hermitian (:class:`BlochOccupiedBlock`).

At the Gamma point alone (no k-point input) the orbitals are real and the
cluster engine (:class:`parsec_python.SCF.hybrid.HostExchange`) runs with the
FFT pair solver of :func:`periodic_pair_potentials`
(:func:`parsec_python.SCF.hybrid.default_pair_solver`).

Units Ry and bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..V_xc.bloch_exchange import (
    BlochCompressedExchange,
    BlochOrbitalSet,
    GridBox,
    MeshStars,
    PeriodicPairPotentials,
    apply_bloch_exchange,
    bloch_band_energy,
    bloch_orbital_set,
    compress_bloch_exchange,
    mesh_stars,
    periodic_pair_solver,
)
from ..V_xc.exact_exchange import OCCUPATION_THRESHOLD
from .fock_diis import OccupiedBlock


#: Reduced coordinates closer than this are one value (and steps of a mesh this close are equal).
MESH_TOLERANCE = 1.0e-6


def uniform_mesh_counts(fractions) -> tuple[int, int, int]:
    """The counts ``(N_1, N_2, N_3)`` of the uniform mesh that the points ``fractions`` (reduced coordinates, rows) are.

    A Monkhorst-Pack mesh, shifted or not: along every axis the reduced
    coordinates (modulo 1) are ``s_i + j/N_i``, ``j = 0 .. N_i - 1`` (equal
    steps ``1/N_i``, one shift), and the points are every combination of
    them, each once (the full Cartesian product).  Any other set is refused
    with the reason: the exchange sum and the kernel's treatment of the
    Coulomb singularity (``k_mesh``) assume that mesh.
    """

    points = np.atleast_2d(np.asarray(fractions, dtype=np.float64))
    reduced = points - np.floor(points)
    reduced[np.abs(reduced - 1.0) < MESH_TOLERANCE] = 0.0
    counts, indices = [], []
    for axis in range(3):
        values = np.sort(reduced[:, axis])
        distinct = [values[0]]
        for value in values[1:]:
            if value - distinct[-1] > MESH_TOLERANCE:
                distinct.append(value)
        if len(distinct) > 1 and 1.0 - distinct[-1] + distinct[0] < MESH_TOLERANCE:
            distinct.pop()  # 0 and 1 - epsilon are one value modulo 1
        count = len(distinct)
        # s + j/N: every distinct value at an integer number of steps 1/N from the first
        steps = (reduced[:, axis] - distinct[0]) * count
        steps = steps - np.floor(steps / count + MESH_TOLERANCE) * count
        rounded = np.round(steps)
        if np.any(np.abs(steps - rounded) > MESH_TOLERANCE * count) or len(np.unique(rounded % count)) != count:
            raise NotImplementedError(
                "the exchange of a periodic cell needs the k-points of a uniform Monkhorst-Pack mesh: the reduced "
                f"coordinates along b{axis + 1} ({', '.join(f'{v:.6g}' for v in distinct)}) are not equally spaced "
                "by 1/N (Monkhorst_Pack_Grid, or a Kpoint_List that writes out every point of one)"
            )
        counts.append(count)
        indices.append(rounded.astype(np.int64) % count)
    combined = np.ravel_multi_index(tuple(indices), tuple(counts))
    if points.shape[0] != int(np.prod(counts)) or np.unique(combined).size != points.shape[0]:
        raise NotImplementedError(
            "the exchange of a periodic cell needs the k-points of a uniform Monkhorst-Pack mesh: "
            f"{points.shape[0]} points are not every point of the {'x'.join(str(n) for n in counts)} mesh "
            "their coordinates span, each once (Monkhorst_Pack_Grid, or a Kpoint_List that writes out every point)"
        )
    return tuple(counts)


def mesh_counts(problem) -> tuple[int, int, int]:
    """The counts of the full Monkhorst-Pack mesh of ``problem`` (``(1, 1, 1)`` without one).

    ``Monkhorst_Pack_Grid``, or for a ``Kpoint_List`` of equal weights the
    mesh it writes out (:func:`uniform_mesh_counts`); any other list is
    refused.
    """

    dimensions = getattr(problem, "monkhorst_pack_dimensions", None)
    if dimensions is not None:
        return tuple(int(value) for value in dimensions)
    listed = getattr(problem, "kpoints", None)
    if listed is None:
        return (1, 1, 1)
    from ..Grid.cell import reciprocal_lattice

    points = np.atleast_2d(np.asarray(listed, dtype=np.float64))
    weights = np.asarray(problem.kpoint_weights, dtype=np.float64)
    if not np.allclose(weights, weights[0], rtol=1.0e-12, atol=0.0):
        raise NotImplementedError(
            "a hybrid functional of a periodic cell needs a uniform k-point mesh: a Kpoint_List with unequal "
            "weights is not one"
        )
    return uniform_mesh_counts(points @ np.linalg.inv(reciprocal_lattice(problem.periodic_cell.lattice_vectors)))


def full_mesh_counts(k_points, partners, reciprocal) -> tuple[int, int, int]:
    """The counts ``(N_1, N_2, N_3)`` of the uniform mesh whose time-reversal representatives are ``k_points``.

    The full mesh is ``k_points`` and the partners ``-k`` (``partners``),
    checked by :func:`uniform_mesh_counts` along the reciprocal vectors of
    ``reciprocal`` (rows).
    """

    points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    full = np.vstack([points, -points[np.asarray(partners, dtype=bool)]])
    return uniform_mesh_counts(full @ np.linalg.inv(np.asarray(reciprocal, dtype=np.float64)))


def periodic_pair_potentials(system, definition, k_mesh=None) -> PeriodicPairPotentials:
    """The FFT pair potentials of the exchange kernel of ``definition`` on the grid of ``system``.

    ``k_mesh`` defaults to the mesh of the input (``(1, 1, 1)`` at the Gamma
    point), which the kernel's treatment of the Coulomb singularity reads.
    """

    settings = getattr(system.input.scf, "exact_exchange", None)
    return periodic_pair_solver(
        system.grid,
        definition.functional,
        definition.fraction,
        k_mesh=mesh_counts(system.input) if k_mesh is None else k_mesh,
        divergence=getattr(settings, "divergence", None),
        auxiliary_exponent=getattr(settings, "auxiliary_exponent", None),
        slab_coulomb=getattr(settings, "slab_coulomb", "padded"),
    )


# ---------------------------------------------------------------------------
# The operator of one inner loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlochChannelExchange:
    """The compressed exchange operators of one channel, one per k-point solved."""

    operators: tuple

    @property
    def rank(self) -> int:
        return max((operator.rank for operator in self.operators if operator is not None), default=0)

    @property
    def asymmetry(self) -> float:
        return max((operator.asymmetry for operator in self.operators if operator is not None), default=0.0)


@dataclass(frozen=True)
class PeriodicFockOperator:
    """The exact-exchange term of one inner SCF loop of a periodic cell with k-points.

    ``channels`` holds a :class:`BlochChannelExchange` per spin channel (one
    for an unpolarized loop), ``None`` for a channel without electrons;
    ``spin_weight`` is that of the channels (2 or 1).  ``reference_energy``
    is ``E_x(P0)`` of the orbitals the operators were built from (Ry,
    unscaled, all channels).
    """

    channels: tuple
    fraction: float
    reference_energy: float
    spin_weight: float = 2.0

    def operator(self, spin: int, k: int) -> BlochCompressedExchange | None:
        """``a_x V_ACE`` of channel ``spin`` at k-point ``k`` for its Bloch Hamiltonian."""

        channel = self.channels[spin]
        return None if channel is None else channel.operators[k]

    def task_cross(self, spin: int, k: int, vectors, occupations, weight: float, xp=np):
        """``(s/2) w_k sum_n f_nk <u_nk|V_ACE|u_nk>`` of one task (unscaled), as an ``xp`` array of one value.

        Called by the k-point job where the task's orbitals are (``xp`` the
        array module of ``vectors``); a host operator brings device orbitals
        to the host.
        """

        operator = self.operator(spin, k)
        values = np.asarray(occupations, dtype=np.float64)
        occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
        if operator is None or not occupied.size:
            return xp.zeros(1)
        if xp is not np and isinstance(operator.projectors, np.ndarray):
            vectors = xp.asnumpy(vectors)
        expectation = operator.expectation_values(vectors[:, occupied], scaled=False)
        if not isinstance(expectation, np.ndarray):
            expectation = xp.asnumpy(expectation)
        value = 0.5 * self.spin_weight * float(weight) * float(np.dot(values[occupied], expectation))
        return xp.asarray([value])

    def energies_from_cross(self, energies, cross: float):
        """The energies of an inner iteration from its ``E_x(P, P0)`` (:func:`parsec_python.SCF.hybrid.linearized_energies`)."""

        from .hybrid import linearized_energies

        return linearized_energies(energies, float(cross), self.fraction, self.reference_energy)


# ---------------------------------------------------------------------------
# The orbitals of a k-point result.
# ---------------------------------------------------------------------------


def _split(pooled, blocks) -> list[np.ndarray]:
    values = np.asarray(pooled, dtype=np.float64)
    parts, offset = [], 0
    for block in blocks:
        count = int(np.shape(block)[1])
        parts.append(values[offset : offset + count])
        offset += count
    if offset != values.size:
        raise ValueError("the occupations do not match the orbitals of the k-points")
    return parts


def bloch_orbital_sets_of(result, box: GridBox | None = None, stars: MeshStars | None = None) -> tuple[BlochOrbitalSet, ...]:
    """The Bloch orbital sets of a k-point result: one unpolarized set or the two channels.

    ``result`` is one of :func:`parsec_python.SCF.kpoints.run_scf_kpoints`
    (spin weight 2) or :func:`~parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`
    (two channels, spin weight 1), with its orbitals kept.  ``stars``: the
    k-points are an irreducible wedge of the mesh (:func:`hybrid_mesh_stars`).
    """

    if box is None:
        box = GridBox(result.grid)
    vectors = getattr(result, "k_wavefunctions", None)
    if vectors is None:
        raise ValueError("the k-point result holds no orbitals (keep_wavefunctions)")
    k_points = np.asarray(result.k_points, dtype=np.float64)
    weights = np.asarray(result.k_weights, dtype=np.float64)
    if hasattr(result, "occupations_up"):
        return tuple(
            bloch_orbital_set(k_points, weights, blocks, _split(pooled, blocks), 1.0, box, stars)
            for blocks, pooled in zip(vectors, (result.occupations_up, result.occupations_down))
        )
    return (bloch_orbital_set(k_points, weights, vectors, _split(result.occupations, vectors), 2.0, box, stars),)


def hybrid_mesh_stars(system, k_points, k_weights, box: GridBox) -> MeshStars | None:
    """The stars of the irreducible wedge of a hybrid's mesh (``SinglePointInput.hybrid_kpoint_wedge``: set by the
    input reader under ``PARSEC_HYBRID_KPOINT_WEDGE=on``, and by :meth:`BlochExchange.after_start` under ``auto``),
    ``None`` for a mesh reduced by time reversal only.

    The images of the k-points solved under time reversal and the operations of the space group that map the grid
    onto itself (those of the density average, ``system.density_symmetry``), the grid's finite-difference operators
    and the mesh onto themselves make up the Monkhorst-Pack mesh of the input
    (:func:`parsec_python.V_xc.bloch_exchange.mesh_stars`, which leaves out the operations that move the mesh: the
    density average of the Fortran-derived routines takes the whole space group, the wedge only those of
    :func:`parsec_python.Grid.kpoints.hybrid_wedge_group`)."""

    problem = system.input
    if not getattr(problem, "hybrid_kpoint_wedge", False):
        return None
    symmetry = getattr(system, "density_symmetry", None)
    dimensions = getattr(problem, "monkhorst_pack_dimensions", None)
    shift = getattr(problem, "monkhorst_pack_shift", None)
    if symmetry is None or getattr(symmetry, "images", None) is None or dimensions is None or shift is None:
        raise NotImplementedError(
            "the irreducible wedge of a hybrid's k-point mesh needs the Monkhorst-Pack mesh and the operations of the "
            "density average (Symmetrize_Charge_Density); PARSEC_HYBRID_KPOINT_WEDGE=off keeps the time-reversal pairs"
        )
    # the images take only the operations that keep the finite-difference operators of the grid, as the wedge does
    # (Grid.kpoints.hybrid_wedge_operations; the density average of the Fortran-derived routines keeps the whole
    # space group, which on an oblique grid holds others)
    return mesh_stars(
        k_points, k_weights, dimensions, shift, box.reciprocal, symmetry, usable=_stencil_exact(system, symmetry)
    )


def _stencil_exact(system, symmetry) -> list[bool]:
    """Whether each operation of ``symmetry`` keeps the finite-difference operators of the grid of ``system``."""

    from ..Grid.kpoints import grid_stencil_lines, stencil_preserving_matrices

    problem = system.input
    counts, lines = grid_stencil_lines(problem.periodic_cell, problem.grid)
    return stencil_preserving_matrices(np.rint(symmetry.matrices).astype(np.int64), counts, lines)


@dataclass(frozen=True)
class WedgePlan:
    """The wedge an auto hybrid moves its Fock loop to after an insulating semilocal start on the pairs
    (:meth:`BlochExchange.after_start`).

    ``indices[j]`` is the point of the pairs whose label and orbitals representative ``j`` of the wedge takes (every
    star holds a point of the pairs: the wedge's k-points are points of the pairs, with their labels); ``k_points``
    and ``k_weights`` those labels and the weights of the stars; ``system`` the system of the wedge (the input with
    ``hybrid_kpoint_wedge``, the mesh, and the density averaged over the wedge's group where the pairs averaged
    none); ``stars`` its :class:`MeshStars`; ``operations`` the order of the wedge's group."""

    indices: tuple
    k_points: np.ndarray
    k_weights: np.ndarray
    operations: int
    system: object = None
    stars: MeshStars | None = None


def hybrid_wedge_plan(system, deferred, pair_points, reciprocal, *, complete: bool = True) -> WedgePlan:
    """The :class:`WedgePlan` of the deferred wedge ``deferred`` (a
    :class:`parsec_python.Grid.kpoints.DeferredHybridWedge`) of a run whose pairs are ``pair_points`` (Cartesian, the representatives of the time-reversal pairs of the
    mesh), the fractions along ``reciprocal`` (rows).  ``complete=False`` stops after the k-points (what the note of
    a run that keeps the pairs names)."""

    from dataclasses import replace

    pair_points = np.atleast_2d(np.asarray(pair_points, dtype=np.float64))
    dimensions = np.asarray([int(value) for value in deferred.dimensions], dtype=np.int64)
    shift = np.asarray([float(value) for value in deferred.shift], dtype=np.float64)
    size = int(np.prod(dimensions))
    inverse = np.linalg.inv(np.asarray(reciprocal, dtype=np.float64))

    def index_of(fraction):
        steps = np.asarray(fraction, dtype=np.float64) * dimensions - shift
        rounded = np.rint(steps)
        if np.max(np.abs(steps - rounded)) > 1.0e-6:
            raise NotImplementedError("a k-point of the pairs is not a point of the deferred wedge's mesh")
        return int(np.ravel_multi_index(tuple(int(v) % int(n) for v, n in zip(rounded, dimensions)), tuple(dimensions)))

    rotations = [np.asarray(rotation, dtype=np.float64) for rotation in deferred.group.reciprocal_rotations]
    pair_indices = [index_of(point @ inverse) for point in pair_points]
    star_of: dict[int, int] = {}
    stars: list[set] = []
    for fraction in (pair_points @ inverse):
        if index_of(fraction) in star_of:
            continue
        members = set()
        for rotation in rotations:
            image = rotation @ fraction
            members.add(index_of(image))
            members.add(index_of(-image))
        number = len(stars)
        for member in members:
            if member in star_of:
                raise NotImplementedError("the stars of the deferred wedge overlap")
            star_of[member] = number
        stars.append(members)
    if sum(len(members) for members in stars) != size:
        raise NotImplementedError("the stars of the deferred wedge do not make up the mesh")
    # the representative of each star: its first point of the pairs (their labels and orbitals)
    chosen: dict[int, int] = {}
    for position, index in enumerate(pair_indices):
        chosen.setdefault(star_of[index], position)
    indices = tuple(chosen[number] for number in range(len(stars)))
    k_points = pair_points[list(indices)].copy()
    k_weights = np.asarray([len(members) / float(size) for members in stars], dtype=np.float64)
    if not complete:
        return WedgePlan(indices, k_points, k_weights, int(deferred.group.order))
    from ..Grid.space_group import periodic_grid_symmetry

    problem = system.input
    wedge_input = replace(
        problem,
        hybrid_kpoint_wedge=True,
        hybrid_kpoint_wedge_deferred=None,
        kpoints=None,
        kpoint_weights=None,
        monkhorst_pack_dimensions=tuple(int(value) for value in deferred.dimensions),
        monkhorst_pack_shift=tuple(float(value) for value in deferred.shift),
        space_group=problem.space_group if deferred.density_group is None else deferred.density_group,
        scf=replace(problem.scf, symmetrize_density=True),
    )
    symmetry = (
        system.density_symmetry
        if deferred.density_group is None
        else periodic_grid_symmetry(system.grid, deferred.density_group)
    )
    wedge_system = replace(system, input=wedge_input, density_symmetry=symmetry)
    found = mesh_stars(
        k_points, k_weights, deferred.dimensions, deferred.shift, reciprocal, symmetry,
        usable=_stencil_exact(wedge_system, symmetry),
    )
    return WedgePlan(indices, k_points, k_weights, int(deferred.group.order), wedge_system, found)


def _start_occupations(start) -> list:
    """The occupations of a k-point loop's result, per spin channel (0 to 1 per spin orbital)."""

    if hasattr(start, "occupations_up"):
        return [np.asarray(start.occupations_up), np.asarray(start.occupations_down)]
    return [np.asarray(start.occupations)]


def _restricted_start(start, plan: WedgePlan, pair_count: int):
    """The result ``start`` of the loop on the pairs at the points ``plan.indices`` alone (the wedge's
    representatives, with the labels of the pairs): their orbitals, levels, occupations and Ritz states."""

    from dataclasses import replace

    indices = list(plan.indices)

    def pooled(values, blocks):
        parts = _split(values, blocks)
        return np.concatenate([parts[index] for index in indices]) if indices else np.zeros(0)

    def states(found, channels: int):
        # the Ritz states by task (channel * n_k + k); those of the k-point sectors (a layout of their own) are left:
        # the wedge's sectors start afresh
        if not isinstance(found, dict) or any(not isinstance(key, (int, np.integer)) for key in found):
            return None
        moved = {}
        for channel in range(channels):
            for position, index in enumerate(indices):
                state = found.get(channel * pair_count + index)
                if state is not None:
                    moved[channel * len(indices) + position] = state
        return moved or None

    changes = dict(k_points=plan.k_points.copy(), k_weights=plan.k_weights.copy())
    if hasattr(start, "occupations_up"):
        channels = start.k_wavefunctions
        changes.update(
            k_wavefunctions=tuple(tuple(blocks[index] for index in indices) for blocks in channels),
            occupations_up=pooled(start.occupations_up, channels[0]),
            occupations_down=pooled(start.occupations_down, channels[1]),
            eigenvalues_up=pooled(start.eigenvalues_up, channels[0]),
            eigenvalues_down=pooled(start.eigenvalues_down, channels[1]),
            eigensolver_state=states(start.eigensolver_state, 2),
        )
    else:
        blocks = start.k_wavefunctions
        changes.update(
            k_wavefunctions=tuple(blocks[index] for index in indices),
            occupations=pooled(start.occupations, blocks),
            eigenvalues=pooled(start.eigenvalues, blocks),
            eigensolver_state=states(start.eigensolver_state, 1),
        )
    return replace(start, **changes)


# ---------------------------------------------------------------------------
# The outer DIIS of Bloch orbitals.
# ---------------------------------------------------------------------------


@dataclass
class BlochOccupiedBlock(OccupiedBlock):
    """The occupied Bloch factors of one (channel, k-point) of the window: Hermitian overlaps.

    ``weight`` is the number of points of the full mesh the k-point stands for
    (2 for a time-reversal pair, 1 for a self-partner point): the share of the
    k-point in the norm of the outer DIIS (``fock_diis._trace``).
    """

    weight: float = 1.0

    def overlap(self, other: OccupiedBlock) -> np.ndarray:
        """``Y_self^H Y_other`` on the host."""

        return np.asarray(np.conj(self.vectors).T @ np.asarray(other.vectors))

    @classmethod
    def combine(cls, terms) -> np.ndarray:
        """``Phi = sum_k Y_k C_k`` (complex, on the host)."""

        phi = None
        for block, part in terms:
            term = np.asarray(block.vectors) @ np.asarray(part)
            phi = term if phi is None else phi + term
        return phi


def _complement(phi: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """``empty`` orthogonalized against the orthonormal columns ``phi`` and among themselves (Loewdin, complex)."""

    rest = empty - phi @ (np.conj(phi).T @ empty)
    gram = np.conj(rest).T @ rest
    values, vectors = np.linalg.eigh(0.5 * (gram + np.conj(gram).T))
    keep = values > 1.0e-8 * max(float(values.max(initial=0.0)), 1.0e-300)
    return rest @ (vectors[:, keep] / np.sqrt(values[keep]))


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------


class BlochExchange:
    """The pieces of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` for a periodic cell with k-points (host).

    ``k_points`` and ``k_weights`` are the k-points solved: the full
    Monkhorst-Pack mesh reduced by time reversal (weights ``2/N``, or ``1/N``
    where ``-k = k``), or its irreducible wedge (``stars``,
    :func:`hybrid_mesh_stars`; weights the sizes of the stars over ``N``).  ``potentials`` replaces the pair potentials of the
    kernel of the run (:func:`periodic_pair_potentials`).  The SCF loops of
    the Fock loop are the k-point drivers (:meth:`run_scf`); they keep the
    orbitals and Ritz states of every k-point.
    """

    takes_hooks = False
    #: The k-point backend of the SCF loops (``make_kpoint_solver``): the host, where the operator is.
    kpoint_backend = "cpu"

    def __init__(self, system, definition, k_points, k_weights, potentials: PeriodicPairPotentials | None = None) -> None:
        self.k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
        self.k_weights = np.asarray(k_weights, dtype=np.float64)
        self.box = GridBox(system.grid)
        # The irreducible wedge of the mesh under the operations of the grid and time reversal
        # (PARSEC_HYBRID_KPOINT_WEDGE), checked here; None for a mesh reduced by time reversal only.
        self.stars = hybrid_mesh_stars(system, self.k_points, self.k_weights, self.box)
        # A uniform mesh reduced by time reversal only (or the wedge of the stars), or refused here (before any SCF).
        probe = bloch_orbital_set(
            self.k_points,
            self.k_weights,
            [np.zeros((self.box.size, 1), dtype=np.complex128)] * self.k_points.shape[0],
            [np.zeros(1)] * self.k_points.shape[0],
            2.0,
            self.box,
            self.stars,
        )
        # The counts of the full mesh the exchange sums over (the kernel's k_mesh), from the k-points themselves.
        self.mesh = (
            full_mesh_counts(self.k_points, probe.partners, self.box.reciprocal)
            if self.stars is None
            else tuple(int(value) for value in system.input.monkhorst_pack_dimensions)
        )
        self.potentials = (
            periodic_pair_potentials(system, definition, k_mesh=self.mesh) if potentials is None else potentials
        )
        # PARSEC_HYBRID_KPOINT_SECTORS=on: every SCF loop of this Fock loop splits its k-points into the sectors of
        # their little groups at any size (acceleration/SCF/kpoint_sectors.py, plan_sectors); off: the size rule of
        # PARSEC_KPOINT_SECTORS, as before
        any_size_key = _any_size_key()
        self.any_size_sectors = any_size_key is not None
        self.description = self._describe()
        self.batch_size = int(system.input.scf.exact_exchange.batch_size)
        self.spin_polarized = bool(system.input.scf.spin_polarized)
        # The Bloch-phase projectors of the k-points of this structure, kept for every SCF loop.
        self.projector_cache: dict = {}
        # the marks of the cache that every loop of this engine reads (kept through the move to the wedge)
        self._cache_marks: tuple = () if any_size_key is None else (any_size_key,)
        for key in self._cache_marks:
            self.projector_cache[key] = True

    def _describe(self) -> str:
        reduction = (
            "time-reversal-reduced mesh"
            if self.stars is None
            else (
                "irreducible wedge (space group and time reversal; the other points are the images of its factors "
                f"by {len(self.stars.gathers)} operation(s) of the grid) of the mesh"
            )
        )
        sectors = (
            "; the k-points of every SCF loop in the sectors of their little groups at any size "
            "(PARSEC_HYBRID_KPOINT_SECTORS=on)"
            if getattr(self, "any_size_sectors", False)
            else ""
        )
        return (
            f"{self.potentials.description}; {self.k_points.shape[0]} k-points of the {reduction} "
            f"{'x'.join(str(n) for n in self.mesh)}{sectors}"
        )

    # -- PARSEC_HYBRID_KPOINT_WEDGE=auto ---------------------------------------

    def after_start(self, system, start, *, callback: Callable | None = None):
        """``(system, start)`` of the Fock loop after its semilocal start ``start``
        (:func:`parsec_python.SCF.hybrid.run_hybrid_scf`).

        A hybrid single point under ``PARSEC_HYBRID_KPOINT_WEDGE=auto`` (``system.input.hybrid_kpoint_wedge_deferred``)
        runs its start on the time-reversal pairs, as ``off`` does.  A start that fills every state to 0 or 1
        (:func:`parsec_python.Grid.kpoints.partly_filled_states`: an insulator) moves the Fock loop to the wedge of
        the deferred group (:func:`hybrid_wedge_plan`): this engine takes its k-points and stars, and the returned
        system (the density averaged over the wedge's group) and start (the representatives' orbitals, levels,
        occupations and Ritz states of the start, their labels those of the pairs) are those of the Fock loop.  A
        start with a partly filled state returns both unchanged: the run stays bit for bit that of ``off``.
        ``callback`` receives the :class:`parsec_python.models.HybridMeshRouteRecord` of the decision.  Anything
        else (no deferred wedge, a wedge already) returns both unchanged.
        """

        from ..Grid.kpoints import HYBRID_WEDGE_FRACTIONAL_TOLERANCE, partly_filled_states
        from ..models import HybridMeshRouteRecord

        deferred = getattr(system.input, "hybrid_kpoint_wedge_deferred", None)
        if deferred is None or self.stars is not None or getattr(start, "k_wavefunctions", None) is None:
            return system, start
        count, largest = partly_filled_states(_start_occupations(start))
        pairs = int(self.k_points.shape[0])
        mesh_points = int(np.prod([int(value) for value in deferred.dimensions]))

        def note(route: str, points: int, operations: int, reason: str = "") -> None:
            if callback is not None:
                callback(
                    HybridMeshRouteRecord(
                        route=route,
                        points=int(points),
                        mesh_points=mesh_points,
                        pairs_points=pairs,
                        operations=int(operations),
                        partly_filled=int(count),
                        largest_fractional_occupation=float(largest),
                        tolerance=float(HYBRID_WEDGE_FRACTIONAL_TOLERANCE),
                        reason=reason,
                    )
                )

        if count:
            # a partly filled state: the pairs, as off (nothing of the run changes; the note names the wedge of on)
            try:
                plan = hybrid_wedge_plan(system, deferred, self.k_points, self.box.reciprocal, complete=False)
                note("pairs", len(plan.k_points), plan.operations)
            except NotImplementedError:
                note("pairs", 0, int(deferred.group.order))
            return system, start
        try:
            if tuple(int(value) for value in deferred.dimensions) != tuple(int(value) for value in self.mesh):
                raise NotImplementedError(
                    f"the pairs make up the {'x'.join(str(n) for n in self.mesh)} mesh, not the wedge's"
                )
            plan = hybrid_wedge_plan(system, deferred, self.k_points, self.box.reciprocal)
        except NotImplementedError as error:
            note("pairs", 0, int(deferred.group.order), reason=str(error))
            return system, start
        restricted = _restricted_start(start, plan, pairs)
        self._take_wedge(plan)
        note("wedge", len(plan.k_points), plan.operations)
        return plan.system, restricted

    def _take_wedge(self, plan: WedgePlan) -> None:
        """Run the Fock loop on the wedge of ``plan`` (its k-points, weights and stars; the kernel is the mesh's)."""

        from .kpoints import _FOCK_GUARD_KEY, _SECTORS_ABANDONED_KEY

        self.k_points = np.asarray(plan.k_points, dtype=np.float64)
        self.k_weights = np.asarray(plan.k_weights, dtype=np.float64)
        self.stars = plan.stars
        bloch_orbital_set(
            self.k_points,
            self.k_weights,
            [np.zeros((self.box.size, 1), dtype=np.complex128)] * self.k_points.shape[0],
            [np.zeros(1)] * self.k_points.shape[0],
            2.0,
            self.box,
            self.stars,
        )
        # the projectors and sector plans of the pairs' tasks go (other tasks, other workers); what the guard of the
        # Fock loop keeps stays, and so do the engine's own marks (PARSEC_HYBRID_KPOINT_SECTORS=on)
        kept = {
            key: self.projector_cache[key]
            for key in (_SECTORS_ABANDONED_KEY, _FOCK_GUARD_KEY, *self._cache_marks)
            if key in self.projector_cache
        }
        self.projector_cache.clear()
        self.projector_cache.update(kept)
        self.description = self._describe()

    # -- the SCF loops ------------------------------------------------------

    def run_scf(
        self,
        system,
        *,
        callback: Callable | None = None,
        warm_start=None,
        exchange_term=None,
        convergence_criterion: float | None = None,
        iteration_state_callback: Callable | None = None,
        mixer_factory: Callable | None = None,
        **hooks,
    ):
        """One SCF loop of the Fock loop (the semilocal start, or an inner loop with ``exchange_term``)."""

        from .kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from .single_point import refuse_exchange_hooks

        refuse_exchange_hooks(**hooks)
        keywords = dict(
            callback=callback,
            keep_wavefunctions=True,
            keep_states=True,
            # The host operator: the eigenproblems on the host as well (the device path brings its own engine).
            backend=self.kpoint_backend,
            warm_start=warm_start,
            convergence_criterion=convergence_criterion,
            exchange_term=exchange_term,
            mixer_factory=mixer_factory,
            # the projectors of the k-points, built once for every loop of the structure
            projector_cache=self.projector_cache,
        )
        if self.spin_polarized and warm_start is None:
            # The start of the Gamma loop (run_spin_scf), so that every route of a spin-polarized hybrid begins
            # in the same state: the per-species spin densities of initchrg.f90.
            from parsec_python.SCF.pbc import with_initial_spin_densities

            keywords["initial_spin_densities"] = with_initial_spin_densities(system).initial_spin_densities
        if iteration_state_callback is not None:
            keywords["iteration_state_callback"] = iteration_state_callback
        driver = run_scf_kpoints_spin_polarized if self.spin_polarized else run_scf_kpoints
        return driver(system, self.k_points, self.k_weights, **keywords)

    def start_system(self, system, definition):
        from .hybrid import semilocal_start_system

        return semilocal_start_system(system, definition)

    def transition_warm_start(self, system, start):
        """The first inner loop after the semilocal start: ``V_ion + V_H[rho] + V_xc^semilocal[rho]``, the
        Ritz states of every k-point kept (Quantum ESPRESSO after its first SCF)."""

        from .single_point import SCFWarmStart, state_for_new_potential

        ionic = np.asarray(system.ionic_potential, dtype=np.float64)
        if self.spin_polarized:
            hartree = np.asarray(start.scf_hartree_potential, dtype=np.float64)
            xc = system.evaluate_xc_spin_polarized(
                np.asarray(start.density_up, dtype=np.float64), np.asarray(start.density_down, dtype=np.float64)
            )
            potential = (ionic + hartree + xc.potential_up, ionic + hartree + xc.potential_down)
        else:
            hartree = np.asarray(start.hartree_potential, dtype=np.float64)
            xc = system.evaluate_xc(np.asarray(start.density, dtype=np.float64))
            potential = ionic + hartree + xc.potential
        return SCFWarmStart(
            input_potential=potential,
            hartree_potential=hartree,
            eigensolver_state=state_for_new_potential(start.eigensolver_state),
        )

    def next_warm_start(self, system, result):
        """The next inner loop: the mixed potential, the Hartree potential and the Ritz states of ``result``."""

        from .single_point import SCFWarmStart, state_for_new_potential

        if self.spin_polarized:
            return SCFWarmStart(
                input_potential=tuple(np.asarray(value) for value in result.next_potentials),
                hartree_potential=np.asarray(result.scf_hartree_potential),
                eigensolver_state=state_for_new_potential(result.eigensolver_state),
            )
        return SCFWarmStart(
            input_potential=np.asarray(result.next_effective_potential),
            hartree_potential=np.asarray(result.hartree_potential),
            eigensolver_state=state_for_new_potential(result.eigensolver_state),
        )

    def start_total_energy(self, system, start, exchange_energy: float, fraction: float) -> float:
        """``E_start - E_xc^start[rho] + E_xc^semilocal[rho] + a_x E_x(P)`` (Ry)."""

        if self.spin_polarized:
            semilocal = system.evaluate_xc_spin_polarized(
                np.asarray(start.density_up, dtype=np.float64), np.asarray(start.density_down, dtype=np.float64)
            ).total_energy
        else:
            semilocal = system.evaluate_xc(np.asarray(start.density, dtype=np.float64)).total_energy
        energies = start.energies
        return float(energies.total - energies.exchange_correlation + semilocal + fraction * exchange_energy)

    # -- the exchange --------------------------------------------------------

    def orbital_sets(self, result) -> tuple[BlochOrbitalSet, ...]:
        return bloch_orbital_sets_of(result, self.box, self.stars)

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        return [
            apply_bloch_exchange(
                orbital_set,
                self.potentials,
                columns=columns,
                known=None if known is None else known[index],
                batch_size=self.batch_size,
            )
            for index, orbital_set in enumerate(sets)
        ]

    def fock_operator(self, sets, applications, fraction: float) -> PeriodicFockOperator:
        channels = []
        for orbital_set, application in zip(sets, applications):
            if not application.has_electrons:
                channels.append(None)
                continue
            channels.append(
                BlochChannelExchange(
                    tuple(
                        compress_bloch_exchange(block, applied, fraction=fraction)
                        for block, applied in zip(orbital_set.orbitals, application.applied)
                    )
                )
            )
        return PeriodicFockOperator(
            channels=tuple(channels),
            fraction=float(fraction),
            reference_energy=float(sum(application.energy for application in applications)),
            spin_weight=float(sets[0].spin_weight),
        )

    def cross_energy(self, operator: PeriodicFockOperator, sets) -> float:
        """``E_x(P, P0)`` of the orbital sets ``sets`` with ``operator`` (Ry, unscaled)."""

        return float(
            sum(
                0.0 if channel is None else bloch_band_energy(channel.operators, orbital_set)
                for channel, orbital_set in zip(operator.channels, sets)
            )
        )

    def release(self, operator) -> None:
        """Nothing to free: the host operator goes with its last reference."""

    def drop_orbitals(self, sets, result=None) -> None:
        """Nothing to drop: host orbitals pin no eigensolver memory."""

    def occupied_blocks(self, sets) -> list:
        """Copies of the occupied Bloch factors of each set, one block per k-point (``Exx_Outer_Mixing: diis``)."""

        blocks = []
        for orbital_set in sets:
            channel = {}
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
                if occupied.size:
                    channel[t] = BlochOccupiedBlock(
                        np.array(block[:, occupied]),
                        np.array(values[occupied]),
                        float(orbital_set.point_counts[t]),
                    )
            blocks.append(channel)
        return blocks

    def natural_sets(self, sets, natural):
        """The orbital sets of the natural orbitals ``natural`` (per set ``{k: (Phi, n)}``) and the empty states."""

        result = []
        for orbital_set, blocks in zip(sets, natural):
            orbitals, occupations = [], []
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                empty = block[:, values <= OCCUPATION_THRESHOLD]
                phi, numbers = blocks.get(t, (None, np.zeros(0)))
                if phi is None:
                    columns, weights = empty, np.zeros(empty.shape[1])
                else:
                    phi = np.asarray(phi, dtype=np.complex128)
                    rest = _complement(phi, empty) if empty.shape[1] else empty
                    columns = np.hstack([phi, rest])
                    weights = np.concatenate([np.asarray(numbers, dtype=np.float64), np.zeros(rest.shape[1])])
                orbitals.append(np.ascontiguousarray(columns))
                occupations.append(weights)
            result.append(
                bloch_orbital_set(
                    orbital_set.k_points, orbital_set.weights, orbitals, occupations, orbital_set.spin_weight, self.box,
                    orbital_set.stars,
                )
            )
        return tuple(result)


def bloch_warm_start(previous_system, previous_result, system, unpolarized=None):
    """The warm start of the Fock loop of a moved structure from the hybrid k-point result before.

    Spin-polarized: the two mixed potentials with the ionic potential of the
    new structure in place of the old (``parsec.F90`` after ``domove``), the
    Hartree potential, the Ritz states of every task and the Bloch orbitals
    that build the first exchange operator.  Unpolarized: ``unpolarized``
    (:func:`parsec_python.SCF.warm_start_for_moved_atoms` with the
    extrapolation of the run, which carries the orbitals as well).
    """

    from .single_point import SCFWarmStart, state_for_new_potential, warm_start_for_moved_atoms

    if not hasattr(previous_result, "occupations_up"):
        build = warm_start_for_moved_atoms if unpolarized is None else unpolarized
        return build(previous_system, previous_result, system)
    shift = np.asarray(system.ionic_potential, dtype=np.float64) - np.asarray(previous_system.ionic_potential)
    return SCFWarmStart(
        input_potential=tuple(np.asarray(value, dtype=np.float64) + shift for value in previous_result.next_potentials),
        hartree_potential=np.asarray(previous_result.scf_hartree_potential, dtype=np.float64),
        eigensolver_state=state_for_new_potential(previous_result.eigensolver_state),
        exchange_orbitals=bloch_orbital_sets_of(previous_result, GridBox(previous_system.grid)),
    )


def bloch_nonlocal_forces(system, result) -> np.ndarray:
    """The nonlocal force of an unpolarized k-point result from its kept Bloch factors (Ry/bohr, ``(atoms, 3)``).

    ``4 w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`` summed over the k-points solved
    (the weights of the time-reversal representatives count their partners), as the step of
    :func:`parsec_python.SCF.kpoints.run_scf_kpoints` with ``compute_forces`` accumulates it.
    """

    from ..Forces.nonlocal_force import bloch_projector_gradients, nonlocal_forces
    from ..V_ion import build_nonlocal_projectors

    atoms = system.cell_atoms
    lattice = system.input.periodic_cell.lattice_vectors
    occupations = _split(result.occupations, result.k_wavefunctions)
    total = np.zeros((len(atoms), 3))
    for k_point, weight, vectors, values in zip(result.k_points, result.k_weights, result.k_wavefunctions, occupations):
        projectors = build_nonlocal_projectors(
            system.grid,
            atoms,
            system.pseudopotentials,
            system.input.pseudopotentials,
            lattice,
            k_point=k_point,
            periodic_dimensions=system.input.periodic_cell.dimensions,
        )
        gradients = bloch_projector_gradients(
            system.grid, atoms, system.pseudopotentials, system.input.pseudopotentials, lattice, k_point
        )
        total += nonlocal_forces(projectors, gradients, np.asarray(vectors), float(weight) * values, len(atoms))
    return total


# The engine factory of the device path (installed by parsec_python.acceleration), or None.
_device_engine = None


def set_device_engine(factory=None) -> None:
    """Install (or, with ``None``, remove) the factory of a device engine of the Fock loop with k-points.

    ``factory(system, definition, k_points, k_weights)`` returns an engine
    with the methods of :class:`BlochExchange` whose orbitals and operators
    live on the devices of the k-point pool, or ``None`` for a run whose
    k-points are solved on the CPU (the host engine then runs).  The
    performance counterpart lives in
    ``parsec_python.acceleration.SCF.hybrid_periodic``; importing
    :mod:`parsec_python.acceleration` installs it.
    """

    global _device_engine
    _device_engine = factory


def _any_size_key() -> str | None:
    """The key of the projector cache under which the SCF loops of a Fock loop split every k-point into its sectors
    (``PARSEC_HYBRID_KPOINT_SECTORS=on``), ``None`` under ``off`` and without the acceleration package (which solves
    the sectors)."""

    try:
        from ..acceleration.SCF.kpoint_sectors import ANY_SIZE_KEY, hybrid_kpoint_sectors_setting
    except ImportError:  # an installation without the acceleration package solves no sectors
        return None
    return ANY_SIZE_KEY if hybrid_kpoint_sectors_setting() == "on" else None


def _check_kpoint_hybrid_switches() -> None:
    """Read the switches that only the k-point loops of a hybrid read (``PARSEC_KPOINT_SECTOR_LEVELS``,
    ``PARSEC_KPOINT_SECTOR_GUARD_FOCK``, ``PARSEC_HYBRID_KPOINT_SECTORS``; ``PARSEC_HYBRID_KPOINT_WEDGE``, which the
    input reader reads for a single point with a Monkhorst-Pack mesh only) before its first loop, so that a bad value
    is an error before any SCF."""

    from ..Grid.kpoints import hybrid_kpoint_wedge_setting

    hybrid_kpoint_wedge_setting()
    try:
        from ..acceleration.SCF.kpoint_sectors import (
            hybrid_kpoint_sectors_setting,
            sector_guard_fock_setting,
            sector_levels_setting,
        )
    except ImportError:  # an installation without the acceleration package solves no sectors
        return
    sector_levels_setting()
    sector_guard_fock_setting()
    hybrid_kpoint_sectors_setting()


def _fock_guard_detail(guarded: dict) -> dict[str, float]:
    """The ``timings.detail`` entries of what the guarded inner loops of a Fock loop measured (``guarded``: the record
    the last of them left in the projector cache, :data:`parsec_python.SCF.kpoints._FOCK_GUARD_KEY`)."""

    values = dict(guarded.get("values") or {})
    detail = {"kpoint_guard_fock_loops": float(guarded.get("loops", 0))}
    for name in ("density_asymmetry", "fractional_share", "tightenings"):
        if f"kpoint_guard_{name}" in values:
            detail[f"kpoint_guard_fock_{name}"] = float(values[f"kpoint_guard_{name}"])
    detail["kpoint_guard_fock_exact"] = float(bool(guarded.get("exact", False)))
    return detail


def run_periodic_hybrid_scf(
    system,
    k_points,
    k_weights,
    *,
    callback: Callable | None = None,
    warm_start=None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable | None = None,
    compute_forces: bool = False,
):
    """The Fock loop of a hybrid functional of a periodic cell with k-points (see the module).

    ``k_points``/``k_weights``: the time-reversal-reduced Monkhorst-Pack mesh, or the irreducible wedge of a
    single point (``PARSEC_HYBRID_KPOINT_WEDGE``) (:func:`parsec_python.Grid.kpoints.kpoint_sampling` of a hybrid
    input).
    The result is that of the last inner loop (a
    :class:`~parsec_python.models.SinglePointResult` with ``k_points`` or a
    :class:`~parsec_python.models.SpinPolarizedSinglePointResult`) with the
    energies of the hybrid, the history of every loop, the orbitals and
    Ritz states of every k-point and the
    :class:`parsec_python.SCF.hybrid.FockLoopSummary` in ``exact_exchange``.
    A system that holds its exchange on devices (``exchange_device`` with a
    ``bloch_engine`` method) supplies its own engine.  ``compute_forces``
    adds the nonlocal force of an unpolarized result (``nonlocal_forces``,
    :func:`bloch_nonlocal_forces`), what
    :func:`parsec_python.Forces.compute_periodic_forces` of Andrew Grieder's
    routines reads.

    A spin-polarized loop in the sectors of its k-points
    (:mod:`parsec_python.acceleration.SCF.kpoint_sectors`) is guarded: the
    semilocal start by the three tests of ``PARSEC_KPOINT_SECTOR_GUARD`` (a
    start that fails one runs again with one task per k-point, and so do the
    inner loops after it), the inner loops by tests 2 and 3 at their end
    (``PARSEC_KPOINT_SECTOR_GUARD_FOCK``).  An inner loop at the final criterion
    that fails one ends the Fock loop: ``callback`` receives a
    :class:`parsec_python.models.SpinSectorFallbackRecord` (``route``
    ``kpoint``) and the Fock loop starts again from its semilocal start with
    one task per k-point, the run of ``PARSEC_KPOINT_SECTORS=off`` for this
    structure (``timings.detail``: ``kpoint_sector_abandoned`` = 1).  What the
    tests measured at the end of the last guarded inner loop is in
    ``timings.detail`` (``kpoint_guard_fock_*``).
    """

    from .hybrid import hybrid_definition, require_supported, run_hybrid_scf
    from .kpoints import _FOCK_GUARD_KEY, _SECTORS_ABANDONED_KEY, KPointSectorsAbandonedInFockLoop

    definition = require_supported(hybrid_definition(system.input.scf))
    if system.input.scf.fixed_occupations is not None:
        raise NotImplementedError(
            "the occupations of occup.in (a negative Fermi_Temp) belong to the Gamma point without k-point input; "
            "the k-point loops of a periodic hybrid find a Fermi level"
        )
    # the switches of the k-point sectors that only the loops of a hybrid read: a bad value stops the run here
    _check_kpoint_hybrid_switches()

    def make_engine():
        device = getattr(system, "exchange_device", None)
        factory = getattr(device, "bloch_engine", None)
        made = None
        if factory is not None:
            made = factory(system, definition, k_points, k_weights)
        elif _device_engine is not None:
            # The k-point pool on the GPUs: the exchange there too (None: the pool runs on the CPU).
            made = _device_engine(system, definition, k_points, k_weights)
        if made is None:
            made = BlochExchange(system, definition, k_points, k_weights)
        return made

    def fock_loop(engine):
        return run_hybrid_scf(
            system,
            callback=callback,
            warm_start=warm_start,
            convergence_criterion=convergence_criterion,
            iteration_state_callback=iteration_state_callback,
            engine=engine,
        )

    engine = make_engine()
    abandoned = None
    try:
        result = fock_loop(engine)
    except KPointSectorsAbandonedInFockLoop as error:
        # only the report leaves this block: the frames of the loop in the sectors, and their orbitals, go
        abandoned = (error.stage, error.report, dict(error.measured))
    if abandoned is not None:
        stage, report, measured = abandoned
        guarded = engine.projector_cache.get(_FOCK_GUARD_KEY) or {}
        # the engine of the loop in the sectors goes with what it holds on the devices before the new one is made
        engine = None
        import gc

        gc.collect()
        if callback is not None:
            from ..models import SpinSectorFallbackRecord

            callback(
                SpinSectorFallbackRecord(
                    stage=stage,
                    largest_fractional_occupation=report.largest,
                    tolerance=report.tolerance,
                    multiplets=tuple(report.lines)
                    + ("(the Fock loop starts again from its semilocal start with one task per k-point)",),
                    reason=report.reason,
                    measure=report.measure,
                    route="kpoint",
                )
            )
        engine = make_engine()
        engine.projector_cache[_SECTORS_ABANDONED_KEY] = True
        result = fock_loop(engine)
        detail = getattr(getattr(result, "timings", None), "detail", None)
        if isinstance(detail, dict):
            detail["kpoint_sector_abandoned"] = 1.0
            detail.update(_fock_guard_detail({**guarded, "values": measured}))
    else:
        detail = getattr(getattr(result, "timings", None), "detail", None)
        guarded = engine.projector_cache.get(_FOCK_GUARD_KEY)
        if isinstance(detail, dict) and guarded:
            detail.update(_fock_guard_detail(guarded))
    # A device engine hands host copies of the orbitals to what reads the result (the forces below too).
    finish = getattr(engine, "finish", None)
    if finish is not None:
        result = finish(result)
    if compute_forces and not system.input.scf.spin_polarized:
        from dataclasses import replace

        result = replace(result, nonlocal_forces=bloch_nonlocal_forces(system, result))
    return result


__all__ = [
    "BlochChannelExchange",
    "BlochExchange",
    "BlochOccupiedBlock",
    "PeriodicFockOperator",
    "bloch_nonlocal_forces",
    "bloch_orbital_sets_of",
    "bloch_warm_start",
    "full_mesh_counts",
    "hybrid_mesh_stars",
    "uniform_mesh_counts",
    "mesh_counts",
    "periodic_pair_potentials",
    "run_periodic_hybrid_scf",
    "set_device_engine",
]
