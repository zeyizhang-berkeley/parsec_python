"""Point-group sectors of a periodic grid whose axes are not the Cartesian axes: the frame of the grid indices.

The detection of :mod:`.axis_reflection` takes the signed permutations ``R`` of the Cartesian axes that map the atoms
and the grid onto themselves: on a grid whose axes are the Cartesian axes, with one step (or the steps of
:func:`.axis_reflection.unequal_step_grid`), ``R`` acts on the integer points as it acts on the positions.  A cell
with oblique lattice vectors (a hexagonal or primitive cubic cell, a hexagonal slab) has its grid points at
``r = S (i + s)``, ``S`` the matrix whose columns are the steps along the three grid axes and ``s`` the shift of the
grid.  An operation ``r -> R r + t`` of the cell maps the grid onto itself where ``W = S^-1 R S`` is an integer
matrix, and its sectors need the finite-difference operators to commute with it: ``W`` a signed permutation of the
axes (the centred first differences of the gradient, turned Cartesian by ``S^-T``, then turn with ``R``) that carries
every second difference of the Laplacian (the axes and the extra directions of the oblique stencil) onto one of the
same weight.  These are the operations the stencil-exact k-point wedge keeps
(:func:`parsec_python.Grid.kpoints.stencil_preserving_matrices`: a hexagonal cell keeps eight of its 24, the
primitive fcc cell 12 of 48).

In the frame of the grid indices, ``x = S^-1 r``, such an operation is the signed permutation ``W`` about the
origin, the grid is an orthorhombic box of one step (``x = i + s``) and the lattice the box of the counts.  The
detection runs there unchanged (:class:`IndexFrame`: the atoms at ``S^-1 r``, a grid of the same points and rows whose
cell is that box, the atom tolerance turned into index units by the smallest singular value of ``S``), with the
admissible operations restricted to the ``W`` above (``index_frame_operations`` of the grid of the frame, read by
:func:`.axis_reflection._integer_signed_permutation_parameters`).  Its orbits, sectors and characters are those of
the grid itself: only the rows enter them.  The operations of the reduction are the ``W``: the Cartesian ones are
``S W S^-1`` (:meth:`IndexFrame.cartesian`).

``PARSEC_OBLIQUE_SECTORS=auto`` (the default since 2026-11-23) takes such a frame for a bulk cell at the Gamma point
(the default routines' grid along the lattice vectors, the general grid of the Fortran-derived routines) and, with
``PARSEC_SLAB_WIRE_SECTORS=auto``, for a wire or slab whose periodic vectors are not along the axes; ``off`` keeps them
on the full grid.

The operations whose ``W`` is an integer matrix but no signed permutation (the three- and six-fold axes of a hexagonal
cell, the three- and four-fold axes of the primitive fcc and bcc cells that are no body diagonal of their grid) carry
the second differences of the Laplacian of both routine sets onto lines of the same weight (their extra directions are
the nearest neighbours of the lattice: ``u_1 + u_2`` of a hexagonal grid, ``u_i - u_j`` of fcc, ``u_1 + u_2 + u_3`` of
bcc), so the kinetic energy, the local and nonlocal pseudopotential, the densities and the finite-difference Hartree
potential commute with them (to the precision to which the cell is typed).  The centred first differences along the
three grid axes do not: such a ``W`` turns an axis onto an extra direction, along which the gradient takes no
difference of its own, so the discrete gradient of a GGA (its potential is off by 2e-3 to 8e-2 Ry on hcp Mg, diamond,
bcc Na and graphite) and the k-point term ``-2ik.grad`` are not invariant.  ``PARSEC_OBLIQUE_FULL_GROUP=auto`` (the
default since 2026-11-23, :func:`oblique_full_group_setting`) admits them for a bulk cell whose Hamiltonian has no
gradient (a local-density functional at the Gamma point; :func:`general_index_operations`); ``off`` keeps the signed
permutations alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations, product
import os

import numpy as np

_SWITCH = "PARSEC_OBLIQUE_SECTORS"
#: ``R = S W S^-1`` is an isometry within this (relative, entry by entry of ``R^T R - I``)
_ORTHOGONALITY_TOLERANCE = 1.0e-9
#: two second differences carry the same weight within this, relative to the largest weight
_WEIGHT_TOLERANCE = 1.0e-9


def oblique_sectors_setting() -> str:
    """``PARSEC_OBLIQUE_SECTORS``: ``auto`` (the default since 2026-11-23; opt-in from 2026-11-22) solves a cell whose
    lattice vectors are not along the Cartesian axes, or a cell on the general grid of the Fortran-derived routines,
    at the Gamma point in the sectors of the operations that map its grid and finite-difference operators onto
    themselves (:class:`IndexFrame`; a wire or slab of that kind with ``PARSEC_SLAB_WIRE_SECTORS=auto``), from the
    floor of ``PARSEC_PERIODIC_SECTOR_FLOOR`` on; ``off`` keeps every such cell on the full grid, as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


_FULL_GROUP_SWITCH = "PARSEC_OBLIQUE_FULL_GROUP"
#: The functionals whose potential reads no gradient of the density (``SCFSettings.xc_functional``): the operations of
#: :func:`general_index_operations` keep their discrete Hamiltonian at the Gamma point.
GRADIENT_FREE_FUNCTIONALS = frozenset({"ca", "pw92", "xalpha", "wigner", "hedin_lundqvist"})


def oblique_full_group_setting() -> str:
    """``PARSEC_OBLIQUE_FULL_GROUP``: ``auto`` (the default since 2026-11-23) or ``off``.

    ``auto``: the sectors of a bulk cell in the frame of its grid indices (``PARSEC_OBLIQUE_SECTORS=auto``) whose
    functional reads no gradient (:data:`GRADIENT_FREE_FUNCTIONALS`; the frame is taken at the Gamma point only) also
    take the operations whose action on the indices is an integer matrix but no signed permutation
    (:func:`general_index_operations`: the full hexagonal and cubic groups), where its sectors then cost less (rows times
    states plus the fixed cost of a sector, :func:`.axis_reflection.detect_point_group_reduction`); otherwise the
    sectors of the signed permutations, bit for bit.  ``off``: the signed permutations alone, as before 2026-11-23."""

    value = os.environ.get(_FULL_GROUP_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_FULL_GROUP_SWITCH} must be auto or off, not {value!r}")
    return value


def _diagonal(lattice: np.ndarray) -> bool:
    return bool(np.all(np.abs(lattice - np.diag(np.diag(lattice))) <= 1.0e-10))


def needs_index_frame(grid) -> bool:
    """Whether the Cartesian signed permutations of :mod:`.axis_reflection` do not act on the points of ``grid`` as on
    the positions, so that its detection needs :class:`IndexFrame`: a bulk grid of the default routines along oblique
    lattice vectors, a general grid of the Fortran-derived routines (an oblique cell, or sides of different steps), a
    wire or slab whose periodic vectors are not along the axes.  ``False`` for a cluster and the grids
    :func:`.axis_reflection._lattice_period` takes as they are."""

    if not (hasattr(grid, "cell") and hasattr(grid, "points_per_axis")):
        return False
    from .axis_reflection import confined_grid, unequal_step_grid

    lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    if confined_grid(grid):
        return not _diagonal(lattice)
    if tuple(getattr(grid, "boundary_modes", ())) != ("wrap", "wrap", "wrap"):
        # (the slab grid of the default routines, Grid.pbc.SlabRealSpaceGrid: no frame, no sectors)
        return False
    if getattr(grid, "is_general_periodic", False):
        return True
    if getattr(grid, "is_orthorhombic", True):
        return False
    return not (unequal_step_grid(grid) and _diagonal(lattice))


def step_matrix(grid) -> np.ndarray:
    """``S``: the columns are the steps of the grid along its three axes (bohr), ``r = S (i + s)``."""

    if hasattr(grid, "spacings") and hasattr(grid, "normalized_lattice"):
        # the general grid of the Fortran-derived routines and the grid of a wire or slab: r = A_n ((i + s) h)
        return np.asarray(grid.normalized_lattice, dtype=np.float64) * np.asarray(grid.spacings, dtype=np.float64)[None, :]
    return np.ascontiguousarray(np.asarray(grid.step_vectors, dtype=np.float64).T)


def stencil_lines(grid) -> list[tuple[tuple[int, int, int], float]]:
    """The second differences ``((offset, weight), ...)`` of the Laplacian of ``grid`` (offsets in grid steps, weights
    in 1/bohr^2): those of the default routines' oblique grid (:func:`parsec_python.Grid.kpoints.default_stencil_lines`),
    of the Fortran-derived general grid and of a wire or slab (``lapmvs``,
    :func:`parsec_python.Laplacian.nonorthogonal.pbc_grid_coefficients`), or the three axes of a box."""

    lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    if hasattr(grid, "spacings") and hasattr(grid, "normalized_lattice"):
        from parsec_python.Laplacian.nonorthogonal import pbc_grid_coefficients

        spacings = np.asarray(grid.spacings, dtype=np.float64)
        coefficients = pbc_grid_coefficients(lattice, spacings)
        return [
            (tuple(int(v) for v in offset), float(coefficient) / float(step) ** 2)
            for offset, coefficient, step in coefficients.stencil_terms(spacings)
        ]
    from parsec_python.Grid.kpoints import default_stencil_lines

    return default_stencil_lines(lattice, np.asarray(grid.points_per_axis, dtype=np.int64))


def _signed_permutations() -> list[np.ndarray]:
    found = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.int8)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            found.append(matrix)
    return found


def admissible_index_operations(grid) -> tuple[np.ndarray, ...]:
    """The signed permutations ``W`` of the grid axes whose ``R = S W S^-1`` is an isometry, that carry every second
    difference of the Laplacian onto one of the same weight, and (a wire or slab) keep its periodic and confined axes
    apart; the identity first."""

    steps = step_matrix(grid)
    inverse = np.linalg.inv(steps)
    lines = stencil_lines(grid)
    scale = max(abs(weight) for _, weight in lines)
    weights: dict[tuple[int, int, int], float] = {}
    for offset, weight in lines:
        weights[tuple(int(v) for v in offset)] = weight
        weights[tuple(-int(v) for v in offset)] = weight
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    from .axis_reflection import confined_grid

    periodic = np.arange(3) < int(getattr(grid, "periodicity", 3)) if confined_grid(grid) else np.ones(3, dtype=bool)
    identity = np.eye(3, dtype=np.int8)
    kept = [identity]
    for matrix in _signed_permutations():
        if np.array_equal(matrix, identity):
            continue
        sources = np.argmax(np.abs(matrix), axis=1)
        if np.any(periodic[sources] != periodic) or np.any(counts[sources] != counts):
            continue
        rotation = steps @ matrix.astype(np.float64) @ inverse
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > _ORTHOGONALITY_TOLERANCE:
            continue
        preserved = True
        for offset, weight in lines:
            image = tuple(int(v) for v in matrix.astype(np.int64) @ np.asarray(offset, dtype=np.int64))
            other = weights.get(image)
            if other is None or abs(other - weight) > _WEIGHT_TOLERANCE * scale:
                preserved = False
                break
        if preserved:
            kept.append(matrix)
    return tuple(kept)


def _unimodular_candidates() -> np.ndarray:
    """The 3x3 integer matrices with entries -1, 0 or 1 and determinant +-1 that are no signed permutation."""

    entries = np.asarray(list(product((-1, 0, 1), repeat=9)), dtype=np.int64).reshape(-1, 3, 3)
    determinants = np.rint(np.linalg.det(entries.astype(np.float64))).astype(np.int64)
    entries = entries[np.abs(determinants) == 1]
    absolute = np.abs(entries)
    signed = np.all(absolute.sum(axis=1) == 1, axis=1) & np.all(absolute.sum(axis=2) == 1, axis=1)
    return entries[~signed]


def general_index_operations(grid) -> tuple[np.ndarray, ...]:
    """The integer matrices ``W`` (entries -1, 0 or 1) that are no signed permutation and act on the indices of the
    bulk ``grid`` as an operation of its lattice that keeps its Laplacian: ``R = S W S^-1`` an isometry, ``W`` maps the
    lattice of the counts onto itself (``N^-1 W N`` integer, ``N`` the counts) and every second difference onto one of
    the same weight.  Empty for a wire or slab.

    These are the rotations and reflections of a hexagonal cell that are no signed permutation of its grid axes (16 of
    D6h), and those of the primitive fcc and bcc cells (36 of Oh).  The gradient of the grid does not commute with them
    (the module docstring): only a Hamiltonian without one keeps them (:func:`oblique_full_group_setting`)."""

    from .axis_reflection import confined_grid

    if confined_grid(grid):
        return ()
    steps = step_matrix(grid)
    inverse = np.linalg.inv(steps)
    lines = stencil_lines(grid)
    scale = max(abs(weight) for _, weight in lines)
    weights: dict[tuple[int, int, int], float] = {}
    for offset, weight in lines:
        weights[tuple(int(v) for v in offset)] = weight
        weights[tuple(-int(v) for v in offset)] = weight
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    candidates = _unimodular_candidates()
    # N^-1 W N integer: entry (i, j) times n_j / n_i
    lattice = candidates * counts[None, None, :]
    whole = np.all(lattice % counts[None, :, None] == 0, axis=(1, 2))
    candidates = candidates[whole]
    rotations = np.einsum("ij,gjk,kl->gil", steps, candidates.astype(np.float64), inverse)
    deviation = np.max(np.abs(np.einsum("gji,gjk->gik", rotations, rotations) - np.eye(3)[None]), axis=(1, 2))
    candidates = candidates[deviation <= _ORTHOGONALITY_TOLERANCE]
    kept = []
    for matrix in candidates:
        preserved = True
        for offset, weight in lines:
            image = tuple(int(v) for v in matrix @ np.asarray(offset, dtype=np.int64))
            other = weights.get(image)
            if other is None or abs(other - weight) > _WEIGHT_TOLERANCE * scale:
                preserved = False
                break
        if preserved:
            kept.append(matrix.astype(np.int8))
    return tuple(kept)


def signed_frame_grid(frame_grid):
    """``frame_grid`` without the operations of :func:`general_index_operations`: the grid of the frame whose detection
    takes the signed permutations alone (the same object where it has none)."""

    if not getattr(frame_grid, "index_frame_general_operations", None):
        return frame_grid
    cached = getattr(frame_grid, "_signed_frame_grid", None)
    if cached is not None:
        return cached
    copy = replace(frame_grid)
    object.__setattr__(copy, "index_frame_operations", frame_grid.index_frame_operations)
    object.__setattr__(frame_grid, "_signed_frame_grid", copy)
    return copy


@dataclass(frozen=True)
class IndexFrame:
    """The detection of the sectors of ``grid`` in the frame of its indices (the module docstring).

    ``frame_grid`` holds the points, rows and lookup of ``grid`` with the box of its counts as cell and one step per
    axis (``index_frame_operations``: the admissible signed permutations ``W``; ``index_frame_general_operations``: the
    integer matrices of :func:`general_index_operations` where ``general`` holds them); :meth:`atoms` gives the atoms at
    ``S^-1 r`` and :meth:`tolerance` a distance in index units.  ``operations`` are all admissible ``W``, the signed
    permutations first.
    """

    grid: object
    frame_grid: object
    steps: np.ndarray
    inverse: np.ndarray
    operations: tuple
    scale: float
    # the operations of :func:`general_index_operations` among ``operations`` (PARSEC_OBLIQUE_FULL_GROUP=auto)
    general: tuple = ()

    def atoms(self, atoms) -> tuple:
        """``atoms`` at their index coordinates ``S^-1 r`` (every other field kept)."""

        return tuple(
            replace(atom, position=self.inverse @ np.asarray(atom.position, dtype=np.float64)) for atom in atoms
        )

    def tolerance(self, distance: float) -> float:
        """A distance in bohr as a radius in index units that holds every point within it: ``distance / sigma_min(S)``."""

        return float(distance) * self.scale

    def cartesian(self, operation) -> np.ndarray:
        """``R = S W S^-1`` of an index operation ``W``."""

        return self.steps @ np.asarray(operation, dtype=np.float64) @ self.inverse

    @property
    def order(self) -> int:
        """The number of admissible operations (the identity among them)."""

        return len(self.operations)

    def signed(self) -> "IndexFrame":
        """This frame with the signed permutations alone (itself where it has no other operation)."""

        if not self.general:
            return self
        general = {np.asarray(op, dtype=np.int8).tobytes() for op in self.general}
        return replace(
            self,
            frame_grid=signed_frame_grid(self.frame_grid),
            operations=tuple(op for op in self.operations if np.asarray(op, dtype=np.int8).tobytes() not in general),
            general=(),
        )


def index_frame(grid, *, general: bool = False) -> IndexFrame:
    """The :class:`IndexFrame` of ``grid`` (:func:`needs_index_frame`); ``general`` also admits the operations of
    :func:`general_index_operations` (the caller has checked that the Hamiltonian keeps them)."""

    from parsec_python.Grid.pbc import PeriodicRealSpaceGrid
    from parsec_python.models import PeriodicCell

    from .axis_reflection import confined_grid

    steps = step_matrix(grid)
    inverse = np.linalg.inv(steps)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(getattr(grid, "shift", grid.settings.shift), dtype=np.float64)
    operations = admissible_index_operations(grid)
    integer = np.asarray(grid.integer_coordinates)
    coordinates = integer.astype(np.float64) + shift
    if confined_grid(grid):
        from parsec_python.Grid.confined import ConfinedPeriodicGrid

        periodicity = int(grid.periodicity)
        sides = np.ones(3, dtype=np.float64)
        sides[:periodicity] = counts[:periodicity]
        frame_grid = ConfinedPeriodicGrid(
            cell=PeriodicCell(np.diag(sides)),
            settings=replace(grid.settings, shift=tuple(float(v) for v in shift)),
            points_per_axis=counts,
            integer_coordinates=integer,
            coordinates=coordinates,
            lookup=grid.lookup,
            spacings=np.ones(3, dtype=np.float64),
            shift=shift,
            normalized_lattice=np.eye(3),
            gradient_transform=np.eye(3),
            periodicity=periodicity,
            confinement_radius=float(grid.confinement_radius) / float(np.min(np.asarray(grid.spacings))),
            index_min=grid.index_min,
            index_max=grid.index_max,
        )
    else:
        frame_grid = PeriodicRealSpaceGrid(
            cell=PeriodicCell(np.diag(counts.astype(np.float64))),
            settings=replace(grid.settings, shift=tuple(float(v) for v in shift), fortran_routines=False),
            points_per_axis=counts,
            integer_coordinates=integer,
            coordinates=coordinates,
            lookup=grid.lookup,
        )
    object.__setattr__(frame_grid, "index_frame_operations", frozenset(op.tobytes() for op in operations))
    others = general_index_operations(grid) if general else ()
    if others:
        object.__setattr__(frame_grid, "index_frame_general_operations", tuple(others))
    singular = np.linalg.svd(steps, compute_uv=False)
    return IndexFrame(
        grid=grid,
        frame_grid=frame_grid,
        steps=steps,
        inverse=inverse,
        operations=tuple(operations) + tuple(others),
        scale=1.0 / float(np.min(singular)),
        general=tuple(others),
    )


def symmetric_stencil_lines(grid, operations) -> list[tuple[tuple[int, int, int], float]] | None:
    """The second differences of ``grid`` (:func:`stencil_lines`) with the weight of every class of lines that the
    signed permutations ``operations`` (index frame) carry onto each other replaced by its mean, or ``None`` where
    every class holds one weight already (the operator of the grid is invariant as it is).

    A cell typed to ten decimals (a hexagonal ``a2 = (-a/2, a sqrt(3)/2)``) is symmetric to about 1e-11: the weights
    of its equivalent lines differ by as much, and so do the rows of its Laplacian that the operations exchange, more
    than the 5e-13 the reduction of an operator to a sector accepts.  The means change the operator by that much."""

    lines = stencil_lines(grid)
    keys = [tuple(int(v) for v in offset) for offset, _ in lines]
    index = {key: position for position, key in enumerate(keys)}
    index.update({tuple(-v for v in key): position for position, key in enumerate(keys)})
    parent = list(range(len(lines)))

    def root(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    for matrix in operations:
        matrix = np.asarray(matrix, dtype=np.int64)
        for position, key in enumerate(keys):
            image = tuple(int(v) for v in matrix @ np.asarray(key, dtype=np.int64))
            other = index.get(image)
            if other is None:
                raise ValueError("an operation of the sectors does not map the stencil of the grid onto itself")
            first, second = root(position), root(other)
            if first != second:
                parent[second] = first
    classes: dict[int, list[int]] = {}
    for position in range(len(lines)):
        classes.setdefault(root(position), []).append(position)
    weights = [float(weight) for _, weight in lines]
    if all(len({weights[p] for p in members}) == 1 for members in classes.values()):
        return None
    symmetric = list(weights)
    for members in classes.values():
        mean = float(np.mean([weights[p] for p in members]))
        for p in members:
            symmetric[p] = mean
    return [(keys[p], symmetric[p]) for p in range(len(lines))]


def lines_negative_laplacian(grid, lines):
    """``-nabla^2`` of ``grid`` from its second differences ``lines`` (offsets in grid steps, weights in 1/bohr^2):
    ``-sum_l w_l [c_0 u_r + sum_s c_s (u_(r - s o_l) + u_(r + s o_l))]``, the stencil of the builders of the oblique
    grids (:func:`parsec_python.Laplacian.finite_difference._build_oblique_negative_laplacian`,
    :func:`parsec_python.Laplacian.nonorthogonal.build_general_negative_laplacian`); a neighbour outside a wire or
    slab has no entry."""

    import scipy.sparse as sp

    from parsec_python.Laplacian.finite_difference import second_derivative_coefficients

    coefficients = second_derivative_coefficients(grid.settings.expansion_order)
    width = grid.settings.stencil_half_width
    size = int(grid.size)
    base = np.asarray(grid.integer_coordinates, dtype=np.int64)
    centre = 0.0
    for _offset, weight in lines:
        centre += -weight * coefficients[width]
    rows = [np.arange(size, dtype=np.int64)]
    columns = [np.arange(size, dtype=np.int64)]
    values = [np.full(size, centre, dtype=np.float64)]
    every = np.arange(size, dtype=np.int64)
    for offset, weight in lines:
        step = np.asarray(offset, dtype=np.int64)
        for shell in range(1, width + 1):
            value = -weight * coefficients[width + shell]
            for sign in (-1, 1):
                neighbours = grid.rows_for_integer_coordinates(base + sign * shell * step)
                present = neighbours >= 0
                rows.append(every[present])
                columns.append(neighbours[present])
                values.append(np.full(int(np.count_nonzero(present)), value, dtype=np.float64))
    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(size, size)
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def sector_negative_laplacian(grid, operations):
    """The Laplacian the sectors of an oblique cell take: that of :func:`lines_negative_laplacian` with the means of
    :func:`symmetric_stencil_lines`, or ``None`` where the operator of the grid is invariant under ``operations`` as it
    is (the sectors then take the grid's own)."""

    lines = symmetric_stencil_lines(grid, operations)
    return None if lines is None else lines_negative_laplacian(grid, lines)


__all__ = [
    "GRADIENT_FREE_FUNCTIONALS",
    "IndexFrame",
    "admissible_index_operations",
    "general_index_operations",
    "oblique_full_group_setting",
    "signed_frame_grid",
    "lines_negative_laplacian",
    "sector_negative_laplacian",
    "symmetric_stencil_lines",
    "index_frame",
    "needs_index_frame",
    "oblique_sectors_setting",
    "stencil_lines",
    "step_matrix",
]
