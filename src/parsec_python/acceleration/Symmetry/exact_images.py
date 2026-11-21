"""Whether the images of the atoms and of the grid under a symmetry group are exact.

The symmetry detector accepts a structure whose atoms are mapped onto atoms
of their species within a tolerance (1e-7 bohr by default) and a lattice
whose shift is symmetric within another (1e-10).  A field that is a sum of
one term per atom, such as the local ionic potential or the superposed
atomic density, is then invariant under the group to those tolerances only.

Where every image of an atom is an atom of its species in every bit, and
every grid point has the coordinates of the representative of its orbit up
to the signs and the order of its axes, the terms of such a sum at a point
are the terms at the representative, taken in another order of the atoms:
the two values differ by the round-off of one sum, and by nothing that the
tolerances of the detector allow.  (A group that permutes axes also adds
the three squares of a distance in another order, which is round-off of the
same size.)  The functions here test exactly that, with NumPy alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Rows of the grid that one pass of the coordinate test compares: the three
# arrays of a block stay in the cache.
_BLOCK_ROWS = 1 << 18


def group_operations(reduction) -> tuple[tuple[np.ndarray, np.ndarray], ...] | None:
    """Return the operations of ``reduction`` as (sources, factors) per operation.

    Component ``i`` of the image of a point ``p`` is
    ``factors[i] * p[sources[i]]``.  A reduction by axis reflections keeps
    the order of the axes; a signed-permutation reduction carries its 3x3
    matrices.  ``None`` for an object that says neither, which then has no
    operation to test.
    """

    matrices = getattr(reduction, "operations", None)
    if matrices is not None:
        operations = []
        for matrix in np.asarray(matrices, dtype=np.int64):
            if matrix.shape != (3, 3):
                return None
            absolute = np.abs(matrix)
            if not (
                np.all(absolute.sum(axis=0) == 1) and np.all(absolute.sum(axis=1) == 1)
            ):
                return None
            sources = np.argmax(absolute, axis=1)
            operations.append((sources, matrix[np.arange(3), sources]))
        return tuple(operations)
    signs = getattr(reduction, "signs", None)
    if signs is None:
        return None
    signs = np.asarray(signs, dtype=np.int64)
    if signs.ndim != 2 or signs.shape[1] != 3 or not np.all(np.abs(signs) == 1):
        return None
    axes = np.arange(3)
    return tuple((axes, row) for row in signs)


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    """The rows of ``points`` in lexicographic order."""

    return points[np.lexsort(points.T[::-1])]


def atom_images_exact(atoms: Sequence, operations, tolerance: float = 0.0) -> bool:
    """Whether every operation maps the atoms of every species onto themselves, bit for bit.

    The sign of a zero is not a difference: a nucleus on a mirror plane is
    its own image.  ``tolerance`` (bohr, for moved structures): an image may
    lie that far from its atom, every atom being the image of one; ``0``
    asks for the same bits.
    """

    by_species: dict[str, list] = {}
    for atom in atoms:
        by_species.setdefault(atom.symbol, []).append(atom.position)
    for positions in by_species.values():
        # Adding zero turns a negative zero into a positive one.
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3) + 0.0
        own = _sorted_rows(positions)
        tree = None
        for sources, factors in operations:
            image = positions[:, sources] * factors + 0.0
            if np.array_equal(_sorted_rows(image), own):
                continue
            if not tolerance > 0.0:
                return False
            if tree is None:
                from scipy.spatial import cKDTree

                tree = cKDTree(positions)
            distances, nearest = tree.query(image, k=1, workers=1)
            if not (np.all(distances <= tolerance) and np.unique(nearest).size == nearest.size):
                return False
    return True


def grid_images_exact(grid, reduction, operations, representative=None) -> bool:
    """Whether every grid point has the coordinates of its representative, up to signs and axis order.

    The orbits of ``reduction`` were formed on the integer lattice, where an
    image is exact; the real coordinates are ``(integer + shift) * spacing``
    evaluated in floating point.  Equal absolute values, sorted where an
    operation permutes axes, then make the coordinates of a point the image
    of those of its representative.  ``representative`` are the coordinates
    of the representative rows where the caller has gathered them.
    """

    coordinates = np.asarray(grid.coordinates)
    orbit = np.asarray(reduction.full_to_wedge)
    if coordinates.ndim != 2 or coordinates.shape != (orbit.size, 3):
        return False
    if representative is None:
        representative = coordinates[np.asarray(reduction.representative_rows)]
    permuted = any(not np.array_equal(sources, (0, 1, 2)) for sources, _ in operations)
    wanted = np.abs(representative)
    if permuted:
        wanted.sort(axis=1)
    for start in range(0, orbit.size, _BLOCK_ROWS):
        stop = start + _BLOCK_ROWS
        found = np.abs(coordinates[start:stop])
        if permuted:
            found.sort(axis=1)
        if not np.array_equal(found, wanted.take(orbit[start:stop], axis=0)):
            return False
    return True


def orbit_sum_points(
    grid, atoms: Sequence, reduction, *, atom_tolerance: float = 0.0
) -> tuple[np.ndarray | None, str | None]:
    """Return the points at which a sum over the atoms may stand for its whole orbit.

    They are the coordinates of the representative rows of ``reduction``,
    C-contiguous and in the order of the orbits, where the group has an
    operation beside the identity and the images of the atoms and of the
    grid points are exact (see the module documentation).  Otherwise
    ``None`` and, in words, the first condition that fails.
    ``atom_tolerance`` lets an image of an atom lie that far from its atom
    (:func:`atom_images_exact`): the positions of a moved structure, averaged
    over the group in floating point, may differ from exact images in their
    last bits, and the sums then differ from those of every point by about
    that much.
    """

    if reduction is None or int(getattr(reduction, "group_order", 1)) <= 1:
        return None, "no symmetry operation beside the identity"
    operations = group_operations(reduction)
    if operations is None:
        return None, "the operations of the symmetry reduction are not known"
    if not atoms or not atom_images_exact(atoms, operations, atom_tolerance):
        return None, (
            "an image of an atom is not an atom of its species in every bit"
            if not atom_tolerance > 0.0
            else f"an image of an atom is farther than {atom_tolerance:g} bohr from every atom of its species"
        )
    try:
        representative = np.ascontiguousarray(
            np.asarray(grid.coordinates)[np.asarray(reduction.representative_rows)],
            dtype=np.float64,
        )
        exact = grid_images_exact(grid, reduction, operations, representative)
    except (AttributeError, IndexError, TypeError):
        return None, "the symmetry reduction has no orbit maps"
    if not exact:
        return None, "an image of a grid point is not a grid point in every bit"
    return representative, None


__all__ = [
    "atom_images_exact",
    "grid_images_exact",
    "group_operations",
    "orbit_sum_points",
]
