"""The symmetry of a structure that replaces the atoms of the input after the input was read.

``Relax_Restart``, ``--restart-from`` and the restart of a dynamics run (``Restart_mode``) replace the coordinates of
the input (:func:`parsec_python.Relaxation.run.restart_structure`,
:func:`parsec_python.MolecularDynamics.run.restart_structure`).  PARSEC 1.5 reads those coordinates in ``initial``
(``initial.f90:294-305`` for ``relax_restart.dat``, ``:103-130`` for ``mdinit.dat``) and folds them into the cell
(``:322-326``) before it looks for the symmetry: ``parsec.F90:391`` calls ``initial``, ``:493`` ``symmetries``,
``:508`` ``inipbc`` (whose ``adjust_step`` fits the grid to the fractional translations), and ``init_var.F90:289``
``irrbz`` (the irreducible k-point wedge).  Its space group, k-point wedge and density average are those of the atoms
it solves.

The reader of this package finds the space group of a cell, and what follows from it, from the atoms of the input
(:mod:`parsec_python.Input.parsec_input`).  :func:`restarted_symmetry` reads the input again with the restarted atoms
for that search (:func:`parsec_python.Input.parsec_input.read_symmetry_of_structure`, on the routines of the run) and
takes what it changes: the space group, the grid of the Fortran-derived routines (its fractional translations), the
k-point mesh and its weights, the time-reversal pairs, the wedges of a hybrid, ``Symmetrize_Charge_Density``'s
average, the group of ``Output_GW`` and the reader's notes about them.  A restart whose atoms keep the group of the
input changes nothing (the same bits as before); one the reader refuses for the restarted atoms (a ``Move_Flag`` that
the wedge of the new group cannot follow, a symmetry the default routines do not compute) stops with the reason.

Not read again, because it is found from the atoms solved anyway: the point group of a cluster (the accelerated
driver's detection in ``acceleration/driver.py``, :func:`parsec_python.SCF.single_point.density_point_group`, the
relaxation's :func:`parsec_python.Relaxation.run.symmetry_decision`), and the sectors of a cell at the Gamma point and
of its k-points (``acceleration/Symmetry/axis_reflection.py``, ``acceleration/SCF/kpoint_sectors.py``), with the
force wedge of a Gamma cell.  A periodic problem that is not the input's (the supercell of
``PARSEC_KPOINT_SUPERCELL=on``, the cell of ``PARSEC_SUPERCELL_KPOINTS``) keeps its group where every operation
maps the restarted atoms onto atoms, and stops otherwise.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

SEED_NOTE = "Vector_Rng_Seed: -1"
"""The start of the reader's note of a seed taken from the clock: a second reading draws another one."""


def _group_key(group):
    """A space group as the set of its operations, translations modulo the lattice within 1e-8."""

    if group is None:
        return None
    operations = []
    for rotation, translation in zip(group.rotations, group.translations):
        fractions = tuple(int(round(float(value) * 1.0e8)) % 100_000_000 for value in translation)
        operations.append((tuple(tuple(int(entry) for entry in row) for row in rotation), fractions))
    return bool(getattr(group, "ignored", False)), tuple(sorted(operations))


def _array_key(values):
    if values is None:
        return None
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return array.shape, array.tobytes()


def _wedge_key(deferred):
    if deferred is None:
        return None
    return (
        _group_key(deferred.group),
        tuple(deferred.dimensions),
        tuple(float(value) for value in deferred.shift),
        _group_key(deferred.density_group),
    )


def _gw_key(settings):
    if settings is None:
        return None
    return _group_key(settings.space_group), tuple(settings.kgrid), tuple(settings.kshift), settings.fermi_level


def symmetry_signature(translation) -> tuple:
    """What the reader derives from the symmetry of the atoms of ``translation``, comparable between two readings."""

    problem = translation.problem
    return (
        _group_key(problem.space_group),
        _group_key(getattr(problem, "force_space_group", None)),
        bool(getattr(problem, "kpoint_tie_images", False)),
        bool(getattr(problem, "kpoint_written_ties", False)),
        bool(getattr(problem, "kpoint_shortest_representatives", False)),
        bool(problem.scf.symmetrize_density),
        problem.grid,
        None if problem.monkhorst_pack_dimensions is None else tuple(problem.monkhorst_pack_dimensions),
        None if problem.monkhorst_pack_shift is None else tuple(float(v) for v in problem.monkhorst_pack_shift),
        _array_key(problem.kpoints),
        _array_key(problem.kpoint_weights),
        bool(problem.kpoint_time_reversal),
        bool(problem.hybrid_kpoint_wedge),
        _wedge_key(problem.hybrid_kpoint_wedge_deferred),
        _gw_key(getattr(translation, "gw", None)),
    )


def reads_symmetry_of_atoms(translation) -> bool:
    """Whether the reader found symmetry from the atoms of ``translation``'s problem (what a restart reads again).

    A cluster: no (its point group is found from the atoms solved).  A cell on the Fortran-derived routines, a wire or
    a slab: yes (their space group sets the grid and the density average).  A bulk cell on the default routines: where
    it carries a space group, a k-point mesh (its wedge, or the time-reversal pairs that a group would reduce
    further) or a density average; its Gamma point otherwise takes the sectors the driver finds from the atoms.
    """

    problem = translation.problem
    if problem.periodic_cell is None:
        return False
    if getattr(getattr(translation, "routines", None), "fortran", False):
        return True
    if int(getattr(problem.grid, "periodicity", 3)) != 3:
        return True
    return bool(
        problem.space_group is not None
        or problem.has_kpoint_sampling
        or problem.scf.symmetrize_density
    )


def _describe(group) -> str:
    if group is None:
        return "no space group"
    return f"{group.order} space-group operation" + ("" if group.order == 1 else "s")


def _kept_group(translation, name: str):
    """A problem that is not the input's: its group stays where it maps the restarted atoms, else the run stops."""

    problem = translation.problem
    group = problem.space_group
    if group is None or group.order <= 1:
        return translation, []
    if int(getattr(problem.grid, "periodicity", 3)) == 3:
        from ..Grid.space_group import space_group_atom_images

        try:
            space_group_atom_images(group, problem.periodic_cell.lattice_vectors, problem.atoms)
        except ValueError as error:
            reason = str(error)
        else:
            return translation, []
    else:
        reason = "a wire or slab"
    raise ValueError(
        f"the structure of {name} does not keep the {_describe(group)} that this problem was prepared with "
        f"({reason}), and its group cannot be found again here: the problem is not that of the input (a supercell "
        "or the cell of a k-point mesh); give the structure in the Atom_Coord block of the input instead"
    )


def restarted_symmetry(translation, name: str):
    """``(translation, notes)``: ``translation``, whose atoms a restart replaced, with the symmetry of those atoms.

    ``name`` is the restart file (for the notes and the errors).  See the module text.  Raises ``ValueError`` where
    the reader refuses the input with the restarted atoms, or where the default routines do not compute it.
    """

    if not reads_symmetry_of_atoms(translation):
        return translation, []
    from .parsec_input import ParsecInputError, StructureCountMismatch, read_symmetry_of_structure
    from .routines import UseFortranRoutines

    problem = translation.problem
    positions = np.asarray([np.asarray(atom.position, dtype=np.float64) for atom in problem.atoms]).reshape(-1, 3)
    try:
        fresh = read_symmetry_of_structure(translation, positions)
    except StructureCountMismatch:
        return _kept_group(translation, name)
    except UseFortranRoutines as needed:
        reasons = "; ".join(needed.choice.reasons) or ", ".join(needed.choice.defaults)
        raise ValueError(
            f"the structure of {name} has another symmetry than the atoms of the input, and the default routines "
            f"do not compute the input with it ({reasons}): run it with Use_Fortran_Routines: .true., or give the "
            "structure in the Atom_Coord block of the input"
        ) from None
    except ParsecInputError as error:
        raise ValueError(f"with the structure of {name}: {error}") from error
    if symmetry_signature(fresh) == symmetry_signature(translation):
        return translation, []
    found = fresh.problem
    scf = problem.scf
    if bool(scf.symmetrize_density) != bool(found.scf.symmetrize_density):
        scf = replace(scf, symmetrize_density=found.scf.symmetrize_density)
    restarted = replace(
        problem,
        scf=scf,
        space_group=found.space_group,
        grid=found.grid,
        monkhorst_pack_dimensions=found.monkhorst_pack_dimensions,
        monkhorst_pack_shift=found.monkhorst_pack_shift,
        kpoints=found.kpoints,
        kpoint_weights=found.kpoint_weights,
        kpoint_time_reversal=found.kpoint_time_reversal,
        kpoint_tie_images=found.kpoint_tie_images,
        kpoint_written_ties=found.kpoint_written_ties,
        kpoint_shortest_representatives=found.kpoint_shortest_representatives,
        force_space_group=found.force_space_group,
        hybrid_kpoint_wedge=found.hybrid_kpoint_wedge,
        hybrid_kpoint_wedge_deferred=found.hybrid_kpoint_wedge_deferred,
    )
    # the reader's notes of the input's symmetry give way to those of the restarted atoms (a seed drawn from the
    # clock stays the one of the first reading, which the run uses)
    current = set(translation.warnings)
    fresh_notes = set(fresh.warnings)
    warnings = tuple(
        [note for note in translation.warnings if note in fresh_notes or note.startswith(SEED_NOTE)]
        + [note for note in fresh.warnings if note not in current and not note.startswith(SEED_NOTE)]
    )
    changes = {"problem": restarted, "warnings": warnings}
    if _gw_key(getattr(translation, "gw", None)) != _gw_key(getattr(fresh, "gw", None)):
        changes["gw"] = fresh.gw
    if _group_key(found.space_group) != _group_key(problem.space_group):
        what = (
            f"the atoms of {name} have another symmetry than those of the input ({_describe(found.space_group)}, "
            f"the input {_describe(problem.space_group)})"
        )
    else:
        what = f"the atoms of {name} change what follows from their symmetry ({_describe(found.space_group)})"
    note = (
        f"NOTE: {what}: the space group, the k-point wedge and the density average of this run are those of the "
        "restarted atoms, as PARSEC 1.5 finds them from the coordinates it reads (initial.f90 reads them before "
        "symmetries.f90 and irrbz.f90)"
    )
    return replace(translation, **changes), [note]


__all__ = ["SEED_NOTE", "reads_symmetry_of_atoms", "restarted_symmetry", "symmetry_signature"]
