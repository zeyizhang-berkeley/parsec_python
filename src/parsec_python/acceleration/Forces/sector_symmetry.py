"""The operations of the full group of a point-group reduction, as operations on the atoms of the forces.

A :class:`~parsec_python.acceleration.Symmetry.axis_reflection.PointGroupReduction` solves the sectors of a subgroup
``H`` of the group ``G`` of the structure (``full_operations``, with the translations ``full_translations`` of a
periodic cell in grid steps); a sector of a weighted representation solves one partner of each degenerate level and
counts it with its weight.  The nonlocal force of such states is the force of the density that the SCF used only after
an average over ``G``: the states of a level span a representation of ``G``, and the sum of a term over the partners is
``|G|^-1 sum_g g`` of the term of one of them, counted once per partner (Schur).  The local and Ewald terms read the
density, which the SCF has averaged over ``G`` already, so the average of the total over ``G`` changes them by
round-off only and keeps a relaxation in the group.

An operation ``r -> R r + t`` (``t`` in grid steps along the axes of a cell) acts on the atoms as
``R (r_a - tau) = r_b`` modulo the lattice with ``tau = -R^T t``, the representation of
:class:`parsec_python.Forces.periodic.PeriodicAtomSymmetryOperation` (forces turn with ``R`` alone,
:func:`parsec_python.Forces.symmetrize_positions` uses ``tau``).
"""

from __future__ import annotations

import numpy as np

#: Distance within which an image of an atom is that atom (bohr), as for the orbits of the reduction.
ATOM_TOLERANCE = 1.0e-7


def sector_reduction(system):
    """The :class:`PointGroupReduction` of the sectors of ``system``, or ``None`` (no sectors, or an Abelian group)."""

    from ..Symmetry.axis_reflection import PointGroupReduction

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if not isinstance(reduction, PointGroupReduction) or getattr(reduction, "full_operations", None) is None:
        return None
    return reduction


def sector_group_operations(reduction, atoms, grid, lattice_vectors=None):
    """The operations of ``G`` of ``reduction`` acting on ``atoms``, the identity first.

    ``lattice_vectors`` (rows) for a periodic cell: :class:`PeriodicAtomSymmetryOperation` with the images modulo the
    lattice and ``tau = -R^T t``; a cluster gets :class:`parsec_python.Forces.AtomSymmetryOperation`.  Raises
    ``ValueError`` where an operation carries an atom onto no atom of its species (the structure has left the group).
    """

    from parsec_python.Forces import AtomSymmetryOperation
    from parsec_python.Forces.periodic import PeriodicAtomSymmetryOperation

    from ..Symmetry.axis_reflection import atom_image_table

    matrices = np.asarray(reduction.full_operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "full_translations", None)
    shifts = None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    images = atom_image_table(tuple(atoms), matrices, shifts, grid, ATOM_TOLERANCE)
    if images is None:
        raise ValueError("an operation of the point group of the sectors carries an atom onto no atom of its species")
    count = len(atoms)
    for row in images:
        if np.unique(row).size != count:
            raise ValueError("an operation of the point group of the sectors does not permute the atoms")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the point group of the sectors hold no single identity")
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    if lattice_vectors is None:
        return tuple(AtomSymmetryOperation(matrix=matrices[g].copy(), image=images[g].copy()) for g in order)
    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    steps = None
    if shifts is not None:
        steps = np.diag(vectors) / np.asarray(grid.points_per_axis, dtype=np.float64)
    operations = []
    for g in order:
        translation = None
        if shifts is not None and np.any(shifts[g]):
            translation = -matrices[g].T @ (shifts[g] * steps)
        operations.append(
            PeriodicAtomSymmetryOperation(
                matrix=matrices[g].copy(), image=images[g].copy(), lattice_vectors=vectors.copy(), translation=translation
            )
        )
    return tuple(operations)


def sector_force_operations(system):
    """``G`` of the sectors of ``system`` on its atoms (:func:`sector_group_operations`), or ``None``."""

    reduction = sector_reduction(system)
    if reduction is None:
        return None
    cell = getattr(system.input, "periodic_cell", None)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    return sector_group_operations(
        reduction, system.atoms, grid, None if cell is None else np.asarray(cell.lattice_vectors, dtype=np.float64)
    )


__all__ = ["ATOM_TOLERANCE", "sector_force_operations", "sector_group_operations", "sector_reduction"]
