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

#: Distance (bohr) within which an image of a moved atom must fall on an atom of its species for the sectors of the
#: first structure to hold for the moved one (:func:`check_sector_atoms`).  An ionic loop that keeps the group (forces
#: and positions averaged over it, :func:`parsec_python.Forces.symmetrize_positions`) moves the atoms on it to the
#: round-off of the coordinates, about 1e-13 bohr per step in a cell of 100 bohr; the fold into the cell by lattice
#: vectors adds as much.  A displacement of 1e-6 bohr off the group changes the energy by ``|F| * 1e-6`` (below 1e-7
#: Ry) and the forces by ``k * 1e-6`` (about 1e-6 Ry/bohr for the bond stiffness of about 1 Ry/bohr^2), the level of
#: the force tolerance of the SCF loop, which the sectors would no longer resolve.  A real breaking of the group (a
#: manual move, molecular dynamics, an atom held by Move_Flag) is 1e-4 bohr and more.  The cluster path checks its
#: moved atoms to the same distance (:func:`.local_device.sector_atom_images`).  The operations of the forces of a
#: structure that this check accepts match its atoms to the same distance (:func:`sector_group_operations`).
MOVE_TOLERANCE = 1.0e-6


_FORTRAN_SECTOR_FORCES_SWITCH = "PARSEC_FORTRAN_SECTOR_FORCES"


def fortran_sector_forces_setting() -> str:
    """``PARSEC_FORTRAN_SECTOR_FORCES``: ``auto`` (the default) or ``off``.

    ``auto`` (since 2026-11-19) lets the forces, the relaxations by an optimizer and the restarts of a bulk cell at
    the Gamma point on the Fortran-derived routines (``Use_Fortran_Routines``, chosen by the input or by the parser
    for Cubic_Spline forces, an FHI pseudopotential, a core correction, ``Restart_Run``, another mixer or eigensolver,
    a double grid, ``Dynamic_Diag_Tol``) keep the sectors of the point group of the cell, as those of the default
    routines do under ``PARSEC_PERIODIC_SECTOR_FORCES=auto``: the terms of those routines (their Ewald sum, their
    local terms with the self-consistency and core-correction terms, of splines or the double grid) read the fields
    the sectors expand to the full grid, their nonlocal term is contracted in the sectors with the gradients of the
    projectors with their lattice images, and the total is averaged over the group of those routines and that of the
    sectors (:func:`fortran_force_operations`).  ``off`` keeps every forces or moving run of those routines on the
    full grid, bit for bit as before.  Read by the command line before the preparation, so that a value it does not
    know is refused before the SCF.
    """

    import os

    value = os.environ.get(_FORTRAN_SECTOR_FORCES_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_FORTRAN_SECTOR_FORCES_SWITCH} must be auto or off, not {value!r}")
    return value


def _operation_key(operation) -> tuple:
    """What an atom operation does to the atoms and the forces: its matrix (to 1e-9) and its permutation."""

    matrix = np.round(np.asarray(operation.matrix, dtype=np.float64), 9) + 0.0
    return tuple(matrix.reshape(-1).tolist()) + tuple(np.asarray(operation.image, dtype=np.int64).tolist())


def _composed(first, second):
    """The operation ``second`` after ``first`` (:class:`~parsec_python.Forces.periodic.PeriodicAtomSymmetryOperation`).

    ``R_1 (r_a - tau_1) = r_(p_1 a)`` and ``R_2 (r_b - tau_2) = r_(p_2 b)`` modulo the lattice give
    ``R_2 R_1 (r_a - tau_1 - R_1^T tau_2) = r_(p_2 p_1 a)``.
    """

    from dataclasses import replace

    matrix = np.asarray(second.matrix, dtype=np.float64) @ np.asarray(first.matrix, dtype=np.float64)
    image = np.asarray(second.image, dtype=np.int64)[np.asarray(first.image, dtype=np.int64)]
    translation = np.zeros(3)
    if getattr(first, "translation", None) is not None:
        translation = translation + np.asarray(first.translation, dtype=np.float64)
    if getattr(second, "translation", None) is not None:
        translation = translation + np.asarray(first.matrix, dtype=np.float64).T @ np.asarray(
            second.translation, dtype=np.float64
        )
    return replace(
        first,
        matrix=matrix,
        image=image,
        translation=None if not np.any(np.abs(translation) > 1.0e-12) else translation,
    )


#: Operations at most in the group :func:`fortran_force_operations` generates (48 rotations of a cubic lattice times
#: the pure translations of a supercell of several thousand atoms stay far below it).
_GENERATED_LIMIT = 200_000


def fortran_force_operations(routine_operations, sector_operations):
    """The group the forces and moved atoms of a cell of the Fortran-derived routines in sectors are averaged over.

    ``routine_operations`` are those the routines average over on the full grid
    (:func:`parsec_python.Forces.periodic.periodic_force_operations`: the space group the problem carries, else the
    point operations about the origin that keep the lattice, the grid and the atoms), ``sector_operations`` the group
    of the sectors (:func:`sector_symmetry_operations`: ``G`` of a point group, ``H`` of Abelian sectors, with their
    translations), which the moved atoms must keep (:func:`check_sector_atoms`).  Both are groups of symmetries of the
    structure, so the group they generate is one too, and the average over it keeps the forces of the density the SCF
    used: the operations of the routines themselves (the same average as on the full grid, operation for operation)
    where they hold those of the sectors, those of the sectors where they hold the routines', the group both generate
    otherwise (the routines' first, then the products).  ``routine_operations`` alone without sectors.
    """

    routine = tuple(routine_operations or ())
    sectors = tuple(sector_operations or ())
    if not sectors:
        return routine
    if not routine:
        return sectors
    routine_keys = {_operation_key(operation) for operation in routine}
    sector_keys = {_operation_key(operation) for operation in sectors}
    if sector_keys <= routine_keys:
        return routine
    if routine_keys <= sector_keys:
        return sectors
    generators = routine + tuple(operation for operation in sectors if _operation_key(operation) not in routine_keys)
    found = list(generators)
    keys = {_operation_key(operation) for operation in found}
    cursor = 0
    while cursor < len(found):
        element = found[cursor]
        cursor += 1
        for generator in generators:
            product = _composed(element, generator)
            key = _operation_key(product)
            if key not in keys:
                if len(found) >= _GENERATED_LIMIT:
                    raise ValueError(
                        "the operations of the Fortran-derived routines and of the sectors generate more than "
                        f"{_GENERATED_LIMIT} operations"
                    )
                keys.add(key)
                found.append(product)
    return tuple(found)


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

    from ..Symmetry.axis_reflection import group_atom_image_table

    matrices = np.asarray(reduction.full_operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "full_translations", None)
    shifts = None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    # the rows of H and of the coset representatives, composed (raises where a row is not a permutation); matched to the
    # distance of the check of moved atoms, so that a structure check_sector_atoms accepts has its operations (the
    # nearest atoms, the same table for atoms on the group); an oblique cell in the frame of its grid indices
    frame, image_atoms, image_grid, image_tolerance = _image_frame(
        reduction, atoms, grid, max(ATOM_TOLERANCE, MOVE_TOLERANCE)
    )
    images = group_atom_image_table(tuple(image_atoms), reduction, image_grid, image_tolerance)
    if images is None:
        raise ValueError("an operation of the point group of the sectors carries an atom onto no atom of its species")
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
        matrix, translation = _cartesian_operation(
            frame, matrices[g], None if shifts is None else shifts[g], steps
        )
        operations.append(
            PeriodicAtomSymmetryOperation(
                matrix=matrix, image=images[g].copy(), lattice_vectors=vectors.copy(), translation=translation
            )
        )
    return tuple(operations)


def _image_frame(reduction, atoms, grid, tolerance: float):
    """``(frame, atoms, grid, tolerance)`` in which the atom images of the operations of ``reduction`` are matched:
    those given, or for sectors detected in the frame of the grid indices of an oblique cell
    (:class:`..Symmetry.oblique_frame.IndexFrame`, ``index_frame`` of the reduction) the atoms at ``S^-1 r``, the grid of
    the frame and the tolerance in index units, where the operations act on the atoms as they act on the indices."""

    frame = getattr(reduction, "index_frame", None)
    if frame is None:
        return None, tuple(atoms), grid, float(tolerance)
    return frame, frame.atoms(tuple(atoms)), frame.frame_grid, frame.tolerance(tolerance)


def _cartesian_operation(frame, matrix, shift, steps):
    """``(R, tau)`` of the operation ``r -> R r + t`` of the sectors as an atom operation
    (:class:`parsec_python.Forces.periodic.PeriodicAtomSymmetryOperation`: ``R (r_a - tau) = r_b`` modulo the lattice,
    ``tau = -R^-1 t``, ``None`` without a translation).

    ``matrix`` and ``shift`` (grid steps) as the reduction holds them: Cartesian, the steps of the axes of a box giving
    ``t = shift * steps``; or, for an oblique cell in the frame of its grid indices (``frame``), the integer matrix
    ``W`` of the indices, ``R = S W S^-1`` and ``t = S shift`` (``S`` the steps of the grid axes as columns), so that
    ``tau = -S W^-1 shift``: ``W^T`` for a signed permutation, the integer inverse of an operation of
    ``PARSEC_OBLIQUE_FULL_GROUP`` that is none (a three-fold axis of a hexagonal or cubic cell)."""

    matrix = np.asarray(matrix, dtype=np.float64)
    moved = shift is not None and np.any(np.asarray(shift) != 0)
    if frame is None:
        translation = -matrix.T @ (np.asarray(shift, dtype=np.float64) * steps) if moved else None
        return matrix.copy(), translation
    rotation = frame.cartesian(matrix)
    absolute = np.abs(matrix)
    signed = bool(np.all(absolute.sum(axis=0) == 1) and np.all(absolute.sum(axis=1) == 1))
    inverse = matrix.T if signed else np.rint(np.linalg.inv(matrix))
    translation = -(frame.steps @ (inverse @ np.asarray(shift, dtype=np.float64))) if moved else None
    return rotation, translation


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


def abelian_cell_operations(system):
    """``H`` of the Abelian sectors of a periodic cell on its atoms, the identity first, or ``None``.

    A cell whose sectors are those of an Abelian group (an
    :class:`..Symmetry.axis_reflection.AxisReflectionReduction` or a
    :class:`..Symmetry.axis_reflection.SignedPermutationReduction` of order above one, not a
    :class:`..Symmetry.axis_reflection.PointGroupReduction`) solves every level, so its forces need no completion
    beyond that of the projectors of one atom per orbit (:func:`abelian_orbit_operations`, which takes these); but its
    moved atoms must keep ``H`` (:func:`check_sector_atoms`), and the round-off of its unsymmetrized forces on
    atoms that ``H`` fixes would move them (by about ``Min_Step`` in a steepest-descent step).  These operations
    average its forces and moved positions over ``H`` in the ionic loop (:func:`sector_symmetry_operations`).
    ``None`` for a cluster, a cell without sectors or with those of a point group.
    """

    from parsec_python.Forces.periodic import PeriodicAtomSymmetryOperation

    from ..Symmetry.axis_reflection import AxisReflectionReduction, PointGroupReduction, atom_image_table

    cell = getattr(getattr(system, "input", None), "periodic_cell", None)
    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if (
        cell is None
        or not isinstance(reduction, AxisReflectionReduction)
        or isinstance(reduction, PointGroupReduction)
        or int(getattr(reduction, "group_order", 1)) <= 1
    ):
        return None
    operations = getattr(reduction, "operations", None)
    if operations is None:
        operations = [np.diag(signs) for signs in np.asarray(reduction.signs)]
    matrices = np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "translations", None)
    shifts = None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    atoms = tuple(system.atoms)
    # (an oblique cell: in the frame of its grid indices, _image_frame)
    frame, image_atoms, image_grid, image_tolerance = _image_frame(
        reduction, atoms, grid, max(ATOM_TOLERANCE, MOVE_TOLERANCE)
    )
    images = atom_image_table(image_atoms, matrices, shifts, image_grid, image_tolerance)
    if images is None:
        raise ValueError("an operation of the group of the sectors carries an atom onto no atom of its species")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the group of the sectors hold no single identity")
    vectors = np.asarray(cell.lattice_vectors, dtype=np.float64)
    steps = None if shifts is None else np.diag(vectors) / np.asarray(grid.points_per_axis, dtype=np.float64)
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    operations = []
    for g in order:
        matrix, translation = _cartesian_operation(frame, matrices[g], None if shifts is None else shifts[g], steps)
        operations.append(
            PeriodicAtomSymmetryOperation(
                matrix=matrix,
                image=np.asarray(images[g], dtype=np.int64).copy(),
                lattice_vectors=vectors.copy(),
                translation=translation,
            )
        )
    return tuple(operations)


def abelian_orbit_operations(system):
    """``H`` of Abelian sectors whose projectors hold one atom per orbit, on the atoms, the identity first, or ``None``.

    Such sectors (:func:`..Symmetry.axis_reflection.attach_abelian_atom_orbits`) contract the projector gradients of
    the representative of every orbit, each ``n_a`` times the term of one atom of its orbit, and none of the other
    atoms (:func:`..Symmetry.axis_reflection.compress_gradients_for_sectors`).  ``H`` is Abelian with real
    one-dimensional characters, so every level is solved and the density matrix of the solved states is invariant
    under it: the term of atom ``h a`` is ``R_h`` times that of ``a``, and that of ``a`` is invariant under the
    operations that fix ``a``.  The average over ``H`` (:func:`parsec_python.Forces.symmetry.symmetrize_forces`) then
    gives every atom of an orbit its own term exactly: ``|H| / n_a`` operations carry ``a`` onto each of them.
    :class:`parsec_python.Forces.AtomSymmetryOperation` for a cluster, those of :func:`abelian_cell_operations` for a
    periodic cell.  ``None`` for a point-group reduction (:func:`sector_force_operations` completes those), sectors
    without the orbits of their atoms (the projectors of every atom, nothing to complete), or
    ``PARSEC_SECTOR_PROJECTOR_ORBITS=off``.
    """

    from parsec_python.Forces import AtomSymmetryOperation

    from ..Symmetry.axis_reflection import (
        AxisReflectionReduction,
        PointGroupReduction,
        abelian_sector_operations,
        atom_image_table,
        sector_projector_orbits_setting,
    )

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if (
        not isinstance(reduction, AxisReflectionReduction)
        or isinstance(reduction, PointGroupReduction)
        or int(getattr(reduction, "group_order", 1)) <= 1
    ):
        return None
    sizes = getattr(reduction, "atom_orbit_sizes", None)
    if (
        getattr(reduction, "atom_representatives", None) is None
        or sizes is None
        or not np.any(np.asarray(sizes) > 1)
        or sector_projector_orbits_setting() == "off"
    ):
        return None
    if getattr(getattr(system, "input", None), "periodic_cell", None) is not None:
        return abelian_cell_operations(system)
    matrices = abelian_sector_operations(reduction).astype(np.float64)
    shifts = getattr(reduction, "translations", None)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    images = atom_image_table(
        tuple(system.atoms),
        matrices,
        None if shifts is None else np.asarray(shifts, dtype=np.int64).reshape(-1, 3),
        grid,
        max(ATOM_TOLERANCE, MOVE_TOLERANCE),
    )
    if images is None:
        raise ValueError("an operation of the group of the sectors carries an atom onto no atom of its species")
    identity = np.flatnonzero(
        [np.array_equal(m, np.eye(3)) and (shifts is None or not np.any(shifts[g])) for g, m in enumerate(matrices)]
    )
    if identity.size != 1:
        raise ValueError("the operations of the group of the sectors hold no single identity")
    order = [int(identity[0])] + [g for g in range(len(matrices)) if g != identity[0]]
    return tuple(
        AtomSymmetryOperation(matrix=matrices[g].copy(), image=np.asarray(images[g], dtype=np.int64).copy())
        for g in order
    )


def sector_symmetry_operations(system):
    """The operations that keep the forces and moved atoms of a periodic cell in its sectors on their group.

    ``G`` of a point-group reduction (:func:`sector_force_operations`), ``H`` of Abelian sectors
    (:func:`abelian_cell_operations`), ``None`` otherwise (no sectors, one sector of the identity, a cluster, whose
    ionic loop keeps the point group of its atoms).
    """

    operations = sector_force_operations(system)
    return operations if operations is not None else abelian_cell_operations(system)


def _operation_text(matrix: np.ndarray, shift, frame=None) -> str:
    """An operation for a message: a signed permutation of the axes plus a shift in grid steps; for the sectors of the
    frame of the grid indices of an oblique cell (``frame``) the Cartesian operation ``R = S W S^-1`` and translation
    ``S t`` (bohr), with ``W`` and ``t`` named as acting on the grid indices."""

    rows = "; ".join(" ".join(str(int(value)) for value in row) for row in np.asarray(matrix).reshape(3, 3))
    text = f"[{rows}]" + ("" if shift is None or not np.any(shift) else f" + {np.asarray(shift).tolist()} grid steps")
    if frame is None:
        return text
    rotation = np.asarray(frame.cartesian(matrix), dtype=np.float64)
    cartesian = "; ".join(" ".join(f"{value:.4f}" for value in row) for row in np.round(rotation, 4) + 0.0)
    translation = ""
    if shift is not None and np.any(shift):
        bohr = np.round(np.asarray(frame.steps, dtype=np.float64) @ np.asarray(shift, dtype=np.float64), 4) + 0.0
        translation = " + ({}) bohr".format(", ".join(f"{value:.4f}" for value in bohr))
    return f"[{cartesian}]{translation} (on the grid indices {text})"


def check_sector_atoms(reduction, atoms, grid, lattice_vectors=None, *, tolerance: float = MOVE_TOLERANCE) -> None:
    """Raise ``ValueError`` unless the sectors of ``reduction`` hold for ``atoms`` (moved atoms of a prepared system).

    The sectors, the weights of their states and the projectors of one atom per orbit
    (:func:`..Symmetry.axis_reflection.compress_projectors_for_sectors`) were made for the group of the first
    structure.  They hold for moved atoms where every operation of the group of the sectors ``H`` (``operations``
    with ``translations``) and of the full group ``G`` (for a :class:`..Symmetry.axis_reflection.PointGroupReduction`)
    carries every atom within ``tolerance`` (bohr, :data:`MOVE_TOLERANCE`) onto an atom of its species, as a
    permutation, modulo the lattice of a periodic grid, and where the orbits of the atoms under ``H`` are those of the
    reduction (``atom_representatives`` and ``atom_orbit_sizes``).  ``G`` is the union of the right cosets ``H c``,
    so ``H`` and the coset representatives (``coset_operations`` with ``coset_translations``) are tested, ``|H| +
    [G:H]`` tree queries in place of ``|G|``.  An Abelian reduction is tested on ``H``, which is all its sectors use.

    ``grid`` is the grid of the sectors (a periodic grid matches images modulo its cell); ``lattice_vectors`` (rows),
    where given, must be those of that cell.  The error names the operation, the atom and how far its image lies from
    the nearest atom of its species.
    """

    from ..Symmetry.axis_reflection import _folded, _lattice_period, _matching_tree, _minimum_image, _query_workers

    atoms = tuple(atoms)
    if not atoms:
        return
    frame = getattr(reduction, "index_frame", None)
    cartesian = None
    if frame is not None:
        # Sectors of an oblique cell detected in the frame of its grid indices: the atoms and the grid of that frame,
        # where the operations of the reduction act (the cell itself is that of the frame's grid); the nearest atom is
        # found there and its distance measured in bohr (``cartesian``: the steps S), against ``tolerance`` in bohr.
        if lattice_vectors is not None and not np.allclose(
            np.asarray(lattice_vectors, dtype=np.float64).reshape(3, 3),
            np.asarray(frame.grid.cell.lattice_vectors, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("the lattice vectors are not those of the periodic grid of the sectors")
        atoms = frame.atoms(atoms)
        grid = frame.frame_grid
        cartesian = np.asarray(frame.steps, dtype=np.float64)
        lattice_vectors = None
    period = _lattice_period(grid)
    if lattice_vectors is not None:
        lattice = np.asarray(lattice_vectors, dtype=np.float64).reshape(3, 3)
        if period is None or not np.allclose(np.diag(lattice), period, rtol=0.0, atol=1.0e-10):
            raise ValueError("the lattice vectors are not those of the periodic grid of the sectors")
    operations = getattr(reduction, "operations", None)
    if operations is None:
        operations = [np.diag(signs) for signs in np.asarray(reduction.signs)]
    matrices = np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)
    shifts = getattr(reduction, "translations", None)
    steps = None
    if shifts is not None:
        steps = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=np.float64)) / np.asarray(
            grid.points_per_axis, dtype=np.float64
        )
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    symbols = np.asarray([atom.symbol for atom in atoms])
    species = [(symbol, np.flatnonzero(symbols == symbol)) for symbol in np.unique(symbols)]
    trees = {symbol: _matching_tree(positions[members], period) for symbol, members in species}
    order = int(matrices.shape[0])

    def images_of(matrix, shift, name: str) -> np.ndarray:
        images = np.empty(positions.shape[0], dtype=np.int64)
        for symbol, members in species:
            image = positions[members] @ matrix.T
            if shift is not None:
                image = image + np.asarray(shift, dtype=np.float64) * steps
            queries = image if period is None else _folded(image, period)
            distances, nearest = trees[symbol].query(queries, k=1, workers=_query_workers(queries.shape[0]))
            if cartesian is not None:
                difference = _minimum_image(image - positions[members][np.asarray(nearest)], period)
                distances = np.linalg.norm(difference @ cartesian.T, axis=1)
            worst = int(np.argmax(distances))
            if not distances[worst] <= tolerance:
                atom = int(members[worst])
                raise ValueError(
                    "the moved atoms leave the group of the symmetry sectors: "
                    f"{name} {_operation_text(matrix, shift, frame)} "
                    f"carries atom {atom + 1} ({symbol}) {float(distances[worst]):.3e} bohr from the nearest {symbol} "
                    f"atom (tolerance {tolerance:g} bohr). The sectors hold for the group of {order} operations of "
                    "the structure the calculation was prepared with: keep the atoms on that group, as a relaxation "
                    "with symmetrized forces does, or prepare the calculation without symmetry (symmetry='off', "
                    "PARSEC_PERIODIC_SYMMETRY=off for a periodic cell)"
                )
            images[members] = members[nearest]
        if np.unique(images).size != images.size:
            raise ValueError(
                "the moved atoms leave the group of the symmetry sectors: "
                f"{name} {_operation_text(matrix, shift, frame)} "
                "carries two atoms onto one"
            )
        return images

    table = np.stack(
        [
            images_of(matrices[g], None if shifts is None else shifts[g], f"operation {g + 1} of the {order} of H")
            for g in range(order)
        ]
    )
    cosets = getattr(reduction, "coset_operations", None)
    if cosets is not None and getattr(reduction, "full_operations", None) is not None:
        coset_matrices = np.asarray(cosets, dtype=np.float64).reshape(-1, 3, 3)
        coset_shifts = getattr(reduction, "coset_translations", None)
        for c in range(coset_matrices.shape[0]):
            images_of(
                coset_matrices[c],
                None if coset_shifts is None else coset_shifts[c],
                f"coset representative {c + 1} of the {coset_matrices.shape[0]} of G",
            )
    representatives = getattr(reduction, "atom_representatives", None)
    sizes = getattr(reduction, "atom_orbit_sizes", None)
    if representatives is None or sizes is None:
        return
    found = table.min(axis=0)
    found_sizes = 1 + np.count_nonzero(np.diff(np.sort(table, axis=0), axis=0), axis=0)
    representatives = np.asarray(representatives, dtype=np.int64).reshape(-1)
    sizes = np.asarray(sizes, dtype=np.int64).reshape(-1)
    if representatives.size != found.size or sizes.size != found.size:
        raise ValueError(
            f"the reduction knows the orbits of {representatives.size} atoms, the moved structure has {found.size}"
        )
    changed = np.flatnonzero((found != representatives) | (found_sizes != sizes))
    if changed.size:
        atom = int(changed[0])
        raise ValueError(
            "the moved atoms change the orbits of the atoms under the group of the symmetry sectors (atom "
            f"{atom + 1}: in the orbit of atom {int(found[atom]) + 1} with {int(found_sizes[atom])} atoms, prepared in "
            f"that of atom {int(representatives[atom]) + 1} with {int(sizes[atom])}): the projectors of one atom per "
            "orbit no longer hold; prepare the moved structure afresh"
        )


def sector_force_tolerance_refusal(system, *, moves: bool = True, atom_tolerance: float | None = None) -> str | None:
    """Why the forces (and, with ``moves``, the moves) of a prepared ``system`` would stop on the group of its
    sectors (or, for the moves, on the group its atoms must keep), or ``None``.

    ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` (``atom_tolerance`` where given) above :data:`MOVE_TOLERANCE` lets the
    detection take operations that carry the atoms onto each other only within that tolerance.  The SCF then runs,
    but the forces of its sectors match the atoms to :data:`MOVE_TOLERANCE` and raise ``ValueError`` after the whole
    SCF: the operations of the nonlocal term (:func:`sector_force_operations`, else :func:`abelian_orbit_operations`),
    the wedge sums of the local forces of a cluster (``local_device.sector_atom_images`` under
    ``PARSEC_LOCAL_FORCE_SUMS=wedge``, the default) and, with ``moves`` (a relaxation or molecular dynamics), the moved
    atoms (:func:`check_sector_atoms` for a cell, ``sector_atom_images`` for a cluster, as the driver's
    ``move_atoms``).  The moves are checked against the group the driver's ``move_atoms`` checks,
    ``system.move_symmetry_reduction``: a cluster keeps the group it detected where its orbitals take the full grid
    (``Spin_Symmetry_Sectors: off``, main.py's spin ``Restart_Run``, the stabilizer policy off), so its relaxation is
    checked here even without sectors; a system without that attribute is checked against the group of its sectors.
    A caller that will compute forces asks this after the preparation and refuses the run before its SCF.  The checks
    are those of the forces and moves themselves, so the runs refused here are those that stopped after their SCF
    before, and a structure accepted here passes them later.  ``None`` at a tolerance of :data:`MOVE_TOLERANCE` or
    less (the default 1e-7 bohr: every operation taken maps the atoms within it, and nothing is checked), and for a
    system whose forces and moves keep no group.
    """

    from ..Symmetry.axis_reflection import symmetry_atom_tolerance

    tolerance = symmetry_atom_tolerance() if atom_tolerance is None else float(atom_tolerance)
    if not tolerance > MOVE_TOLERANCE:
        return None

    def nontrivial(reduction):
        return reduction if reduction is not None and int(getattr(reduction, "group_order", 1)) > 1 else None

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = nontrivial(getattr(getattr(sectors, "decomposition", None), "reduction", None))
    moved = nontrivial(getattr(system, "move_symmetry_reduction", reduction)) if moves else None
    if reduction is None and moved is None:
        return None
    cell = getattr(getattr(system, "input", None), "periodic_cell", None)
    atoms = tuple(system.atoms)
    checking = reduction
    try:
        wedge_checked = False
        if reduction is not None:
            if sector_force_operations(system) is None:
                abelian_orbit_operations(system)
            if cell is None:
                from .local_device import local_force_sums_setting, sector_atom_images

                if local_force_sums_setting() == "wedge":
                    sector_atom_images(reduction, atoms, tolerance=MOVE_TOLERANCE)
                    wedge_checked = True
            else:
                abelian_cell_operations(system)
        if moved is not None and not (wedge_checked and moved is reduction):
            checking = moved
            if cell is None:
                from .local_device import sector_atom_images

                sector_atom_images(moved, atoms, tolerance=MOVE_TOLERANCE)
            else:
                check_sector_atoms(moved, atoms, system.grid, np.asarray(cell.lattice_vectors, dtype=np.float64))
    except ValueError as error:
        operations = getattr(checking, "full_operations", None)
        order = int(checking.group_order) if operations is None else int(np.asarray(operations).shape[0])
        return (
            f"PARSEC_SYMMETRY_ATOM_TOLERANCE={tolerance:g} bohr gave this calculation a group of {order} operations "
            f"that carries the atoms onto each other only to more than the {MOVE_TOLERANCE:g} bohr that its forces, "
            "relaxations and molecular dynamics require, so they would stop after the SCF "
            f"({error}). Refused before the SCF: symmetrize the positions onto the group, or leave "
            "PARSEC_SYMMETRY_ATOM_TOLERANCE at its default (1e-7 bohr) for a run with forces"
        )
    return None


__all__ = [
    "ATOM_TOLERANCE",
    "MOVE_TOLERANCE",
    "abelian_cell_operations",
    "abelian_orbit_operations",
    "check_sector_atoms",
    "fortran_force_operations",
    "fortran_sector_forces_setting",
    "sector_force_operations",
    "sector_force_tolerance_refusal",
    "sector_group_operations",
    "sector_reduction",
    "sector_symmetry_operations",
]
