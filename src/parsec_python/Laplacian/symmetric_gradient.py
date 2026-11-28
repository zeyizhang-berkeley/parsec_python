"""A density gradient of an oblique periodic grid that commutes with the point group of its lattice.

The gradient of the periodic grids is the centred first difference along the three grid axes, turned Cartesian:
``grad f = sum_i m_i d_i f`` with ``m_i = S^-T e_i`` (``S`` the matrix whose columns are the steps of the grid, ``d_i``
the difference along axis ``i`` in index units; :func:`parsec_python.V_xc.pbe._oblique_gradient` on the default
routines' grid, :func:`parsec_python.Laplacian.nonorthogonal.cartesian_gradient` on the general grid of the
Fortran-derived routines, :func:`parsec_python.Laplacian.finite_difference.build_gradient` for the k-point term).  An
operation of the lattice acts on the grid indices as an integer matrix ``W`` (``R = S W S^-1``).  Where ``W`` is a
signed permutation it carries the axes onto axes and the gradient commutes with it; the three- and six-fold axes of a
hexagonal grid and the three- and four-fold axes of the primitive fcc and bcc grids turn an axis onto another lattice
direction (``u_1 + u_2``, ``u_i - u_j``, ``u_1 + u_2 + u_3``), along which the axis gradient takes no difference: the
PBE potential of a symmetric density is then not symmetric (2e-3 to 8e-2 Ry), nor is the k-point term ``-2ik.grad``.

``PARSEC_SYMMETRIC_GRADIENT=on`` replaces the axis gradient of such a grid by its average over the group ``G`` of
the integer index maps ``W`` of the lattice that map the grid and its Laplacian onto themselves
(:func:`lattice_index_operations`):

``D_sym = |G|^-1 sum_g R_g^-1 T_g^-1 D T_g = sum_d c_d d_d``,  ``c_d = |G|^-1 sum_(g, i: W_g e_i = +-d) +-R_g m_i``,

a sum of the same centred differences ``d_d`` (order ``Expansion_Order``, unit steps) along the orbit of the grid
axes under ``G`` (the nearest-neighbour directions of the lattice: three in the plane and the c axis of a hexagonal
grid, six of the primitive fcc grid, four of the bcc grid).  ``c_(W d) = R c_d`` holds for every ``g`` of ``G``, so
``D_sym (T_g f) = R_g T_g (D_sym f)``: the gradient commutes with the whole group.  It is exact for linear fields
(each ``d_d`` is, and the average of consistent gradients is consistent) and of the same order as the axis gradient,
each ``d_d`` being the centred difference of that order along a lattice vector of the length of the axis steps (the
orbit of a step under isometries).  By Schur's lemma it is the only gradient of these differences that commutes with
``G`` (the vector representation occurs once in the odd functions on each orbit of directions), which is also the
least-squares fit of the gradient to the differences along the orbit.  ``D_sym^T = -sum_d c_d d_d`` on the periodic
grid, the exact transpose of the GGA potential.

On the hexagonal, primitive fcc and bcc grids the directions are the lines of the Laplacian of both routine sets, and
``c_d = w_d S d`` with ``w_d`` the weight of the line (to round-off on the cubic grids, to the typed precision of a
hexagonal cell): by Cauchy-Schwarz ``|D|^2 <= sum_d w_d b_d^2 <= A`` for the symbols of every plane wave (``b``, ``a``
those of the first and second difference along a line, ``b^2 <= a``), so the free Bloch operator ``-L - 2ik.D + k^2``
has no level below zero at any ``k``, as on an orthogonal grid; the axis gradient of an oblique grid has such levels
(:func:`parsec_python.Grid.kpoints.lowest_free_levels`, which reads the symbols of the gradient taken).

The body-centred tetragonal grid (16 maps, 12 of them no signed permutation) and the body-centred and face-centred
orthorhombic grids (8 maps, 6) take the symmetric gradient too: it commutes with their groups, but there the lines of
the Laplacian carry other weights (``c_d`` differs from ``w_d S d`` by a third or more) and its free operator keeps a
few levels below zero, which the guard of ``PARSEC_KPOINT_MESH_GUARD`` reads as it reads those of the axis gradient.

A grid whose group holds signed permutations alone (an orthorhombic, tetragonal, monoclinic or rhombohedral cell),
a wire, a slab and a cluster keep the axis gradient, bit for bit: there the average equals it.

``off`` (the default): the axis gradient everywhere, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os

import numpy as np

SWITCH = "PARSEC_SYMMETRIC_GRADIENT"
_VALUES = ("off", "on")
#: ``R = S W S^-1`` is an isometry within this (entry by entry of ``R^T R - I``), as the index frame's operations
_ORTHOGONALITY_TOLERANCE = 1.0e-9
#: two second differences carry the same weight within this, relative to the largest weight
_WEIGHT_TOLERANCE = 1.0e-9


def symmetric_gradient_setting() -> str:
    """``PARSEC_SYMMETRIC_GRADIENT``: ``off`` (the default) or ``on`` (the module docstring); another value is an
    error."""

    value = os.environ.get(SWITCH, "off").strip().lower() or "off"
    if value not in _VALUES:
        raise ValueError(f"{SWITCH} must be off or on, not {value!r}")
    return value


@dataclass(frozen=True)
class GradientLines:
    """``grad f = sum_l vectors[l] d_l f`` with ``d_l f(i) = sum_(s=-M..M) w_s f(i + s offsets[l])`` (the centred
    first-difference weights of unit step, :func:`parsec_python.V_xc.pbe.first_derivative_coefficients`).

    ``offsets`` ``(L, 3)`` are integer grid offsets (the first non-zero entry positive), ``vectors`` ``(L, 3)`` the
    Cartesian coefficients in 1/bohr, ``order`` the number of operations averaged over."""

    offsets: np.ndarray
    vectors: np.ndarray
    order: int

    @property
    def count(self) -> int:
        return int(self.offsets.shape[0])


def _eligible(grid) -> bool:
    """A bulk periodic grid whose axes are not the Cartesian axes of one step: the default routines' oblique grid or
    the general grid of the Fortran-derived routines (every neighbour wraps)."""

    if not (hasattr(grid, "cell") and hasattr(grid, "points_per_axis")):
        return False
    if getattr(grid, "is_confined_periodic", False):
        return False
    if tuple(getattr(grid, "boundary_modes", ())) != ("wrap", "wrap", "wrap"):
        return False
    if int(getattr(grid.cell, "dimensions", 3)) != 3:
        return False
    if getattr(grid, "is_general_periodic", False):
        return True
    return not bool(getattr(grid, "is_orthorhombic", True))


def step_matrix(grid) -> np.ndarray:
    """``S``: the columns are the steps of the grid along its three axes (bohr)."""

    if getattr(grid, "is_general_periodic", False):
        return np.asarray(grid.normalized_lattice, dtype=np.float64) * np.asarray(grid.spacings, dtype=np.float64)[None, :]
    return np.ascontiguousarray(np.asarray(grid.step_vectors, dtype=np.float64).T)


def laplacian_lines(grid) -> list[tuple[tuple[int, int, int], float]]:
    """The second differences ``(offset, weight)`` of the Laplacian of a bulk grid (offsets in grid steps, weights in
    1/bohr^2): Fortran's ``lapmvs`` on the general grid, the plan of the default routines' oblique grid otherwise."""

    lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    if getattr(grid, "is_general_periodic", False):
        from .nonorthogonal import pbc_grid_coefficients

        spacings = np.asarray(grid.spacings, dtype=np.float64)
        coefficients = pbc_grid_coefficients(lattice, spacings)
        return [
            (tuple(int(v) for v in offset), float(coefficient) / float(step) ** 2)
            for offset, coefficient, step in coefficients.stencil_terms(spacings)
        ]
    from ..Grid.kpoints import default_stencil_lines

    return default_stencil_lines(lattice, np.asarray(grid.points_per_axis, dtype=np.int64))


def _signed_permutation(matrix: np.ndarray) -> bool:
    absolute = np.abs(matrix)
    return bool(np.all(absolute.sum(axis=0) == 1) and np.all(absolute.sum(axis=1) == 1))


def index_operations_of(steps, lines, counts) -> tuple[np.ndarray, ...]:
    """The integer index maps ``W`` (identity first) of the lattice of a grid with the step matrix ``steps`` (columns
    the steps of the three axes), the second differences ``lines`` (offset, weight) and ``counts`` points per axis
    that map its points and its Laplacian onto themselves: ``R = S W S^-1`` an isometry, ``N^-1 W N`` integer (``N``
    the counts: the grid of the cell onto itself) and every second difference onto one of the same weight.

    Such a ``W`` carries each axis onto a line of the Laplacian, so its columns are taken from the offsets of the lines
    and their negatives (every such map is found, whatever its entries)."""

    steps = np.asarray(steps, dtype=np.float64)
    inverse = np.linalg.inv(steps)
    scale = max(abs(weight) for _, weight in lines)
    weights: dict[tuple[int, int, int], float] = {}
    for offset, weight in lines:
        weights[tuple(int(v) for v in offset)] = weight
        weights[tuple(-int(v) for v in offset)] = weight
    columns = np.asarray(sorted(weights), dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    identity = np.eye(3, dtype=np.int64)
    found = [identity]
    for first, second, third in product(range(columns.shape[0]), repeat=3):
        matrix = np.column_stack((columns[first], columns[second], columns[third]))
        if abs(round(float(np.linalg.det(matrix.astype(np.float64))))) != 1 or np.array_equal(matrix, identity):
            continue
        # N^-1 W N integer: entry (i, j) times n_j / n_i
        if np.any((matrix * counts[None, :]) % counts[:, None] != 0):
            continue
        rotation = steps @ matrix.astype(np.float64) @ inverse
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > _ORTHOGONALITY_TOLERANCE:
            continue
        preserved = True
        for offset, weight in lines:
            image = tuple(int(v) for v in matrix @ np.asarray(offset, dtype=np.int64))
            other = weights.get(image)
            if other is None or abs(other - weight) > _WEIGHT_TOLERANCE * scale:
                preserved = False
                break
        if preserved:
            found.append(matrix)
    return tuple(found)


def lattice_index_operations(grid) -> tuple[np.ndarray, ...]:
    """:func:`index_operations_of` of ``grid``."""

    return index_operations_of(step_matrix(grid), laplacian_lines(grid), grid.points_per_axis)


def _canonical(offset: np.ndarray) -> tuple[tuple[int, int, int], int]:
    """``(offset with its first non-zero entry positive, the sign applied)``."""

    nonzero = np.flatnonzero(offset)
    sign = 1 if offset[nonzero[0]] > 0 else -1
    return tuple(int(v) for v in sign * offset), sign


def averaged_lines_of(steps, operations) -> GradientLines:
    """The group average of the axis gradient of a grid with the step matrix ``steps`` over the index maps
    ``operations``: the module docstring's ``c_d``."""

    steps = np.asarray(steps, dtype=np.float64)
    inverse = np.linalg.inv(steps)
    axis_vectors = inverse.T  # column i: m_i = S^-T e_i
    sums: dict[tuple[int, int, int], np.ndarray] = {}
    for matrix in operations:
        matrix = np.asarray(matrix, dtype=np.int64)
        rotation = steps @ matrix.astype(np.float64) @ inverse
        for axis in range(3):
            key, sign = _canonical(matrix[:, axis])
            vector = sign * (rotation @ axis_vectors[:, axis])
            if key in sums:
                sums[key] = sums[key] + vector
            else:
                sums[key] = vector
    order = len(operations)
    keys = sorted(sums, key=lambda key: (sum(abs(v) for v in key), tuple(-v for v in key)))
    offsets = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    vectors = np.asarray([sums[key] / float(order) for key in keys], dtype=np.float64).reshape(-1, 3)
    return GradientLines(offsets=offsets, vectors=vectors, order=order)


def averaged_gradient_lines(grid, operations=None) -> GradientLines:
    """:func:`averaged_lines_of` of ``grid`` over ``operations`` (:func:`lattice_index_operations` by default)."""

    if operations is None:
        operations = lattice_index_operations(grid)
    return averaged_lines_of(step_matrix(grid), operations)


def symmetric_lines_of(steps, lines, counts) -> GradientLines | None:
    """The symmetric gradient of a bulk oblique grid given by its step matrix, second differences and counts, or
    ``None`` where its group holds signed permutations alone (whatever the switch: the caller has read it)."""

    operations = index_operations_of(steps, lines, counts)
    if not any(not _signed_permutation(np.asarray(matrix)) for matrix in operations):
        return None
    return averaged_lines_of(steps, operations)


_CACHE = "_symmetric_gradient_lines"


def symmetric_gradient_lines(grid) -> GradientLines | None:
    """The :class:`GradientLines` that replace the axis gradient of ``grid`` under ``PARSEC_SYMMETRIC_GRADIENT=on``, or
    ``None`` (the switch off, a grid that is no oblique bulk grid, or a group of signed permutations alone: the axis
    gradient, bit for bit).  Computed once per grid."""

    if symmetric_gradient_setting() != "on" or not _eligible(grid):
        return None
    cached = getattr(grid, _CACHE, None)
    if cached is not None:
        return cached[0]
    lines = symmetric_lines_of(step_matrix(grid), laplacian_lines(grid), grid.points_per_axis)
    try:
        object.__setattr__(grid, _CACHE, (lines,))
    except (AttributeError, TypeError):
        pass
    return lines


_FULL_GROUP_SWITCH = "PARSEC_OBLIQUE_FULL_GROUP"


def full_group_wedge() -> bool:
    """Whether the stencil-exact k-point wedges keep the operations of a bulk oblique grid whose index map is no signed
    permutation but maps the Laplacian onto itself (:func:`parsec_python.Grid.kpoints.stencil_preserving_matrices`):
    ``PARSEC_SYMMETRIC_GRADIENT=on`` with ``PARSEC_OBLIQUE_FULL_GROUP=auto`` (its default; read as
    :func:`parsec_python.acceleration.Symmetry.oblique_frame.oblique_full_group_setting` reads it).  Both values are
    checked here, so that the input refuses a wrong one before the SCF.

    Such a map is an isometry of the crystal that carries the lines of the Laplacian onto lines of the same weight, so
    it is one of :func:`lattice_index_operations` of the grid, which then takes the symmetric gradient: the Bloch
    Hamiltonian ``-nabla^2 - 2ik.grad_sym + k^2`` at ``R k`` is that at ``k`` turned by the operation."""

    gradient = symmetric_gradient_setting()
    value = os.environ.get(_FULL_GROUP_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_FULL_GROUP_SWITCH} must be auto or off, not {value!r}")
    return gradient == "on" and value == "auto"


# ------------------------------------------------------------------- application on the periodic box
def _directional(box: np.ndarray, offset: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``d f(i) = sum_s w_s f(i + s offset)`` on the periodic box (``weights`` from ``-M`` to ``M``)."""

    moving = np.flatnonzero(offset)
    if moving.size == 1 and abs(int(offset[moving[0]])) == 1:
        from scipy.ndimage import correlate1d

        axis = int(moving[0])
        sign = int(offset[axis])
        return correlate1d(box, weights if sign > 0 else weights[::-1], axis=axis, mode="wrap")
    width = (weights.size - 1) // 2
    out = np.zeros_like(box)
    for shell in range(1, width + 1):
        forward = weights[width + shell]
        backward = weights[width - shell]
        if forward == 0.0 and backward == 0.0:
            continue
        shift = tuple(int(v) for v in shell * offset)
        if forward != 0.0:
            out += forward * np.roll(box, tuple(-v for v in shift), axis=(0, 1, 2))
        if backward != 0.0:
            out += backward * np.roll(box, shift, axis=(0, 1, 2))
    return out


def apply_gradient(values: np.ndarray, grid, lines: GradientLines, weights: np.ndarray) -> np.ndarray:
    """``(3, size)`` Cartesian gradient of a field of the periodic ``grid`` (rows = its box, row-major) with ``lines``
    and the unit-step first-difference ``weights``."""

    box = np.asarray(values, dtype=np.float64).reshape(grid.shape)
    weights = np.asarray(weights, dtype=np.float64)
    out = np.zeros((3, int(box.size)), dtype=np.float64)
    for offset, vector in zip(lines.offsets, lines.vectors):
        difference = _directional(box, offset, weights).reshape(-1)
        for component in range(3):
            if vector[component] != 0.0:
                out[component] += vector[component] * difference
    return out


def apply_gradient_transpose(fluxes: np.ndarray, grid, lines: GradientLines, weights: np.ndarray) -> np.ndarray:
    """``D_sym^T F = sum_l d_l^T (vectors[l] . F)``, the exact transpose of :func:`apply_gradient` (``d_l^T`` is the
    difference with the reversed weights on the periodic box)."""

    fluxes = np.asarray(fluxes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    reversed_weights = np.ascontiguousarray(weights[::-1])
    total = np.zeros(int(np.prod(grid.shape)), dtype=np.float64)
    for offset, vector in zip(lines.offsets, lines.vectors):
        projected = np.zeros(total.size, dtype=np.float64)
        for component in range(3):
            if vector[component] != 0.0:
                projected += vector[component] * fluxes[component]
        total += _directional(projected.reshape(grid.shape), offset, reversed_weights).reshape(-1)
    return total


def gradient_matrices(grid, lines: GradientLines, weights: np.ndarray):
    """The three sparse Cartesian matrices of :func:`apply_gradient` (CSR), for the k-point term and the meta-GGA."""

    import scipy.sparse as sp

    weights = np.asarray(weights, dtype=np.float64)
    width = (weights.size - 1) // 2
    size = int(grid.size)
    base = np.asarray(grid.integer_coordinates, dtype=np.int64)
    every = np.arange(size, dtype=np.int64)
    rows, columns, values = [[], [], []], [[], [], []], [[], [], []]
    for offset, vector in zip(lines.offsets, lines.vectors):
        for shell in range(-width, width + 1):
            weight = weights[width + shell]
            if shell == 0 or weight == 0.0:
                continue
            neighbours = np.asarray(grid.rows_for_integer_coordinates(base + shell * offset), dtype=np.int64)
            for component in range(3):
                if vector[component] == 0.0:
                    continue
                rows[component].append(every)
                columns[component].append(neighbours)
                values[component].append(np.full(size, weight * vector[component], dtype=np.float64))
    matrices = []
    for component in range(3):
        if not rows[component]:
            matrices.append(sp.csr_matrix((size, size)))
            continue
        matrix = sp.coo_matrix(
            (np.concatenate(values[component]), (np.concatenate(rows[component]), np.concatenate(columns[component]))),
            shape=(size, size),
        ).tocsr()
        matrix.sum_duplicates()
        matrices.append(matrix)
    return tuple(matrices)


__all__ = [
    "GradientLines",
    "SWITCH",
    "apply_gradient",
    "apply_gradient_transpose",
    "averaged_gradient_lines",
    "averaged_lines_of",
    "full_group_wedge",
    "gradient_matrices",
    "index_operations_of",
    "laplacian_lines",
    "lattice_index_operations",
    "step_matrix",
    "symmetric_gradient_lines",
    "symmetric_gradient_setting",
    "symmetric_lines_of",
]
