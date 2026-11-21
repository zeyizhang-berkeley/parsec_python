"""What a command-line driver does for a relaxation, on any backend.

The readable and the accelerated drivers differ in how they prepare a system
and in the :class:`IonicBackend` they pass; reading a restart structure,
deciding about symmetry, running the loop with its recorder and reporting a
single point with forces are the same for both and live here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Callable

import numpy as np

from ..Forces import atom_symmetry_operations, operations_preserving
from ..V_ion import center_cluster_geometry, center_slab_geometry
from ..models import Atom
from .backend import moved_atoms
from .files import (
    RelaxRestart,
    parse_manual_structures,
    parse_relax_restart,
    parse_xyz_frames,
)
from .ionic_loop import IonicBackend, IonicOutcome, run_relaxation
from .recorder import RELAX_RESTART, RelaxationRecorder


MANUAL_FILE = "manual.dat"


def restart_structure(translation, directory: Path, source: Path | None = None):
    """Apply ``Relax_Restart`` (or an explicit restart file) to a translation.

    Returns ``(translation, restart, notes)``.  With ``Relax_Restart`` the
    coordinates of ``relax_restart.dat`` beside the input replace those of
    the input, as in ``initial.f90``, and its BFGS block, if it has one,
    continues the optimizer; a missing file is reported and the relaxation
    starts from scratch.  ``source`` names another file: a
    ``relax_restart.dat`` of PARSEC or of this package, or an XYZ file whose
    last frame gives the coordinates (the optimizer then starts afresh).

    The coordinates of a ``relax_restart.dat`` are used as they are: they
    were recentred by the run that wrote them, and the state of its optimizer
    refers to them.  An XYZ file stands for the ``Atom_Coord`` blocks of the
    input: its structure is recentred as theirs is (not in a ``manual`` run;
    a slab along z as ``initial.f90:201-228``, not in a restart), so that a
    file of any program can be given.

    With ``PARSEC_STANDARD_ORIENTATION=on`` (``translation.standard_orientation``)
    the structure of an XYZ file, in the frame of the input, is moved into the
    calculation frame.  A ``relax_restart.dat`` holds the frame of the run that
    wrote it (:func:`_restart_orientation`): this run reports in that frame.
    """

    settings = translation.relaxation
    notes: list[str] = []
    if source is None:
        if settings is None or not settings.restart:
            return translation, None, notes
        source = Path(directory) / RELAX_RESTART
        if not source.is_file():
            notes.append(
                f"WARNING! {RELAX_RESTART} not found; "
                "start structural relaxation from scratch"
            )
            return translation, None, notes
    source = Path(source)
    problem = translation.problem
    atom_count = len(problem.atoms)
    text = source.read_text(encoding="utf-8", errors="replace")
    restart = None
    if source.suffix.lower() == ".xyz":
        labels, positions = parse_xyz_frames(text)[-1]
        if len(labels) != atom_count:
            raise ValueError(
                f"{source.name} holds {len(labels)} atoms, the input {atom_count}"
            )
        # The coordinates are assigned by position in the file, so the file
        # must list the atoms as the input does: type by type.
        for index, (label, atom) in enumerate(zip(labels, problem.atoms)):
            specification = problem.pseudopotentials.get(atom.symbol)
            element = getattr(specification, "element_symbol", None) or atom.symbol
            if label.lower() not in {atom.symbol.lower(), element.lower()}:
                raise ValueError(
                    f"{source.name}: atom {index + 1} is {label}, the input has "
                    f"{atom.symbol} there; an XYZ file must list the atoms in the "
                    "order of the input"
                )
        recenter = problem.recenter_geometry
        orientation = getattr(translation, "standard_orientation", None)
        if orientation is not None:
            placed = moved_atoms(problem.atoms, positions, True)
            if recenter:
                placed = center_cluster_geometry(placed)
            positions = orientation.to_calculation(np.asarray([atom.position for atom in placed]))
            recenter = False
    else:
        restart = parse_relax_restart(text, atom_count)
        positions = restart.positions
        recenter = False
        translation, frame_notes = _restart_orientation(translation, restart, source.name)
        notes.extend(frame_notes)
    # the atoms keep their other fields with Andrew Grieder's routines (parsec_python.Input.routines)
    atoms = moved_atoms(
        problem.atoms, positions, not getattr(getattr(translation, "routines", None), "fortran", False)
    )
    slab_shift = 0.0
    if (
        restart is None
        and getattr(problem.grid, "periodicity", 3) == 2
        and getattr(settings, "method", "") != "manual"
        # PARSEC 1.5 re-centres a slab with Restart_Run too (initial.f90:287-289),
        # and not at all with Recenter_Atoms_In_Aperiodic_Directions: .false.
        and getattr(translation, "recenter_aperiodic", True)
    ):
        atoms, slab_shift = center_slab_geometry(atoms)
    notes.append(
        "WARNING: atom coordinates in input were replaced by coordinates in "
        f"{source.name}"
    )
    if restart is not None and restart.lbfgsb is not None and settings is not None:
        if settings.method == "bfgs":
            notes.append(" RESTART LBFGS FROM PREVIOUS RUN")
        else:
            restart = replace(restart, lbfgsb=None)
    translation = replace(
        translation,
        problem=replace(problem, atoms=atoms, recenter_geometry=recenter),
        slab_shift=slab_shift,
    )
    return translation, restart, notes


MOVE_FLAG_SUBGROUP_SWITCH = "PARSEC_MOVE_FLAG_SUBGROUP"


def move_flag_subgroup_setting() -> str:
    """``PARSEC_MOVE_FLAG_SUBGROUP``: ``auto`` (the default) or ``off``.

    A relaxation by an optimizer (``bfgs``, ``simple``) whose ``Move_Flag`` holds atoms that a symmetry operation of
    the structure exchanges with movable ones leaves the group of its first structure, but not the subgroup of the
    operations that map the fixed atoms onto fixed atoms (and so the movable ones onto movable ones): its forces and
    moved positions are averaged over that subgroup (:func:`parsec_python.Forces.operations_preserving`), so every
    structure keeps it.  ``auto`` keeps that subgroup: a cluster keeps it for its forces and, on the accelerated
    driver, for the sectors of its SCF, whose detection takes the fixed atoms as a species of their own
    (:func:`symmetry_fixed_atoms`); a periodic cell at the Gamma point keeps the sectors of the group its detection
    finds so (``acceleration.cli.periodic_sector_forces_refusal``).  ``off`` uses no symmetry for such a
    relaxation of a cluster and keeps every cell with a fixed atom on the full grid, as before 2026-11-19 (PARSEC 1.5
    takes no account of ``Move_Flag`` in its group).  A subgroup of the identity alone is no symmetry under both.
    """

    import os

    value = os.environ.get(MOVE_FLAG_SUBGROUP_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{MOVE_FLAG_SUBGROUP_SWITCH} must be auto or off, not {value!r}")
    return value


def _optimizer_fixes_atoms(translation) -> bool:
    """Whether ``translation`` is a relaxation by an optimizer (``bfgs``, ``simple``) whose Move_Flag fixes atoms."""

    settings = getattr(translation, "relaxation", None)
    movable = getattr(translation, "movable", None)
    return bool(
        settings is not None
        and settings.method in ("bfgs", "simple")
        and movable is not None
        and not all(bool(flag) for flag in movable)
    )


def _cluster_operations(translation):
    """The point-group operations of the structure of ``translation`` about its (recentred) origin."""

    problem = translation.problem
    atoms = (
        center_cluster_geometry(problem.atoms)
        if problem.recenter_geometry
        else tuple(problem.atoms)
    )
    return atom_symmetry_operations(atoms)
def _restart_orientation(translation, restart: RelaxRestart, name: str):
    """The frame of a ``relax_restart.dat`` against ``PARSEC_STANDARD_ORIENTATION``.

    Returns ``(translation, notes)``.  A file that records a frame (written
    with the switch on) holds its coordinates and its optimizer state in that
    calculation frame: the restart takes that frame for its reports, so that
    they stay in the frame of the first run whatever this one recentres (a
    ``Restart_Run`` does not), and needs the switch on.  A file without one
    holds the frame of the input as PARSEC placed it: the restart keeps it,
    and an orientation this run found is dropped.  Without the switch and
    without a recorded frame nothing changes.
    """

    current = getattr(translation, "standard_orientation", None)
    if restart.orientation is None:
        if current is None:
            return translation, []
        from ..Input.standard_orientation import SWITCH

        return replace(translation, standard_orientation=None), [
            f"{SWITCH}=on: {name} was written without the standard orientation; its coordinates and the state "
            "of its optimizer are in the frame of the input, which this restart keeps"
        ]
    from ..Input.standard_orientation import SWITCH, StandardOrientation, standard_orientation_setting

    recorded = StandardOrientation.from_restart_records(restart.orientation)
    if standard_orientation_setting() != "on":
        raise ValueError(
            f"{name} was written with {SWITCH}=on: its coordinates and the state of its optimizer are in the "
            f"calculation frame of that run; restart it with {SWITCH}=on (or start from an XYZ structure)"
        )
    return replace(translation, standard_orientation=recorded), [
        f"{SWITCH}=on: the coordinates of {name} are in the calculation frame it records; the reports are in "
        "the frame of the input of the run that wrote it"
    ]


def symmetry_decision(translation) -> tuple[bool, str | None]:
    """Whether the calculation may use the point group, and why not.

    PARSEC uses no symmetry with ``Ignore_Symmetry``, for manual movements
    and in molecular dynamics.  One more case is decided here: movable flags
    that a symmetry operation of the structure does not respect, because
    such a relaxation leaves the group it started in.  It keeps the subgroup
    of the operations that respect the flags where that subgroup is more than
    the identity and ``PARSEC_MOVE_FLAG_SUBGROUP=auto`` (the note says so), and
    uses no symmetry otherwise (:func:`move_flag_subgroup_setting`).
    """

    if translation.ignore_symmetry:
        return False, None
    if not translation.use_symmetry:
        return False, None
    if translation.relaxation is None:
        return True, None
    operations = _cluster_operations(translation)
    kept = operations_preserving(operations, translation.movable)
    if len(kept) != len(operations):
        if (
            len(kept) > 1
            and _optimizer_fixes_atoms(translation)
            and move_flag_subgroup_setting() == "auto"
        ):
            return True, (
                f"Move_Flag fixes atoms that {len(operations) - len(kept)} of the {len(operations)} symmetry "
                "operations of the structure exchange with movable ones; the relaxation keeps the subgroup of "
                f"the {len(kept)} operations that map the fixed atoms onto fixed atoms "
                f"({MOVE_FLAG_SUBGROUP_SWITCH}=auto; off uses no symmetry)."
            )
        return False, (
            "Move_Flag fixes atoms that a symmetry operation of the structure "
            "exchanges with movable ones; the relaxation uses no symmetry."
        )
    return True, None


def symmetry_fixed_atoms(translation) -> tuple[bool, ...] | None:
    """The atoms the symmetry detection of the accelerated driver takes as a species of their own, or ``None``.

    ``PARSEC_MOVE_FLAG_SUBGROUP=auto`` (:func:`move_flag_subgroup_setting`) and a relaxation by an optimizer
    (``bfgs``, ``simple``) with ``Move_Flag`` fixing atoms: the detection then finds the subgroup of the operations
    that map the fixed atoms onto fixed atoms, which the relaxation keeps (its forces and moved positions are
    averaged over the operations that respect the flags).  A periodic cell takes the fixed atoms so whenever it has
    any: its group is found by the driver.  A cluster takes them where the flags break its point group
    (:func:`symmetry_decision` keeps the subgroup); where every operation respects them the cluster keeps its whole
    group, found as before.  ``None`` under ``off``, without fixed atoms, for a cluster whose flags respect its group
    or leave the identity alone.  ``True`` marks a fixed atom.  ``Ignore_Symmetry`` is not read: the flags are
    harmless where no symmetry is used, and ``--symmetry on`` may overrule it.
    """

    if move_flag_subgroup_setting() != "auto" or not _optimizer_fixes_atoms(translation):
        return None
    fixed = tuple(not bool(flag) for flag in translation.movable)
    if getattr(translation.problem, "periodic_cell", None) is not None:
        return fixed
    operations = _cluster_operations(translation)
    kept = operations_preserving(operations, translation.movable)
    if len(kept) == len(operations) or len(kept) <= 1:
        return None
    return fixed


def manual_structures(translation, directory: Path):
    """Structures of a ``manual`` run from ``manual.dat``, or ``None``."""

    settings = translation.relaxation
    if settings is None or settings.method != "manual" or settings.max_movements == 0:
        return None
    scan = getattr(translation, "afm_scan", None)
    if scan is not None:
        # AFM_Scan (parsec_python.Embedding.afm): the scan points after the first.
        return scan.structures(scan.tip_positions)[1:]
    path = Path(directory) / MANUAL_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Minimization: manual with Movement_Num {settings.max_movements} "
            f"needs {MANUAL_FILE} beside the input"
        )
    return parse_manual_structures(
        path.read_text(encoding="utf-8", errors="replace"),
        len(translation.problem.atoms),
        settings.max_movements,
    )


def run_recorded_relaxation(
    translation,
    system,
    backend: IonicBackend,
    reporter,
    write: Callable[[str], None],
    *,
    directory: Path | None,
    input_directory: Path | None = None,
    restart: RelaxRestart | None = None,
    symmetrize: bool = True,
) -> IonicOutcome:
    """Run the relaxation of ``translation`` with the standard recorder.

    ``directory`` receives ``atom.cor``, ``relax_restart.dat`` and the XYZ
    trajectory.  ``manual.dat`` is read from ``input_directory``, by default
    the same one, where an AFM scan also writes its table after every point.
    """

    settings = translation.relaxation
    source = directory if input_directory is None else input_directory
    on_record = None
    scan = getattr(translation, "afm_scan", None)
    if scan is not None and source is not None:
        # AFM_Scan: afm_scan.dat after every tip position, so that a failure keeps the points before.
        from ..Output.afm_scan import write_scan_progress

        on_record = lambda records: write_scan_progress(source, scan, records)  # noqa: E731
    elements = [
        translation.problem.pseudopotentials[atom.symbol].element_symbol or atom.symbol
        for atom in system.atoms
    ]
    recorder = RelaxationRecorder(
        reporter=reporter,
        write=write,
        settings=settings,
        movable=translation.movable,
        directory=directory,
        elements=elements,
        warm_starts=getattr(backend, "warm_starts", True),
        **({} if on_record is None else {"on_record": on_record}),
        # PARSEC_STANDARD_ORIENTATION: atom.cor and the trajectory in the frame of the input
        orientation=getattr(translation, "standard_orientation", None),
    )
    try:
        return run_relaxation(
            system,
            settings,
            backend,
            movable=translation.movable,
            observer=recorder,
            restart=restart,
            manual_structures=(
                None if source is None else manual_structures(translation, source)
            ),
            symmetrize=symmetrize,
            output_level=translation.output_level,
        )
    finally:
        # Also when a structure failed: the files are complete on return.
        recorder.flush()


def run_single_point_with_forces(
    system,
    backend: IonicBackend,
    reporter,
    *,
    symmetrize: bool = True,
):
    """One SCF loop and its forces, reported; returns ``(result, forces)``."""

    started = time.perf_counter()
    result = backend.run_scf(system, callback=reporter.iteration, warm_start=None)
    elapsed = time.perf_counter() - started
    forces = backend.compute_forces(
        system, result, symmetrize=symmetrize, operations=None
    )
    reporter.finish(result, elapsed, forces=forces)
    return result, forces


def relaxation_header_lines(translation) -> list[str]:
    """What ``usrinputfile.F90`` prints about the minimization."""

    settings = translation.relaxation
    if settings is None:
        return [" No minimization!", ""]
    lines = [" Minimization data:", " ------------------"]
    if settings.method == "simple":
        lines += [
            " Simple minimization!",
            " This scheme is not robust!",
        ]
    elif settings.method == "bfgs":
        lines.append(" Limited-memory BFGS minimization!")
        if settings.bfgs_bound > 0.0:
            lines += [
                " Restricted BFGS",
                " Maximum displacement of atoms from initial position is "
                f"{settings.bfgs_bound:.6f}",
                "  bohrs along each cartesian direction.",
            ]
    else:
        lines.append(" atoms move according to info in manual.dat")
    lines.append(f" Maximum number of movements  = {settings.max_movements:4d}")
    if settings.method == "simple":
        lines += [
            f" Maximum step size  = {settings.maximum_step:8.5f} [au]",
            f" Mininum step size  = {settings.minimum_step:8.5f} [au]",
        ]
    if settings.method != "manual":
        lines.append(f" Minimum force  = {settings.minimum_force:8.5f} [Ry/au]")
    if settings.restart:
        lines += [
            " Restart relaxation, information on previous",
            " runs read from relax_restart.dat file.",
        ]
    if settings.method == "bfgs":
        lines.append(
            f" Number of corrections used in LBFGS update = {settings.bfgs_corrections:4d}"
        )
    movable = int(np.count_nonzero(np.asarray(translation.movable, dtype=bool)))
    lines.append(f" Movable atoms = {movable:6d} of {len(translation.movable):6d}")
    extrapolation = getattr(translation, "potential_extrapolation", "none")
    if extrapolation != "none":
        # Not a line of PARSEC, which always starts from the last potential.
        lines.append(f" Potential extrapolation after a move: {extrapolation}")
    if settings.method in {"bfgs", "simple"} and settings.scf_upscale > 1.0:
        # Not a line of PARSEC, which keeps Convergence_Criterion fixed.
        lines.append(
            f" SCF criterion up to {settings.scf_upscale:g} times Convergence_Criterion"
            " while the forces exceed Force_Min (Relaxation_SCF_Upscale)"
        )
    return [*lines, ""]


__all__ = [
    "MANUAL_FILE",
    "MOVE_FLAG_SUBGROUP_SWITCH",
    "manual_structures",
    "move_flag_subgroup_setting",
    "relaxation_header_lines",
    "restart_structure",
    "run_recorded_relaxation",
    "run_single_point_with_forces",
    "symmetry_decision",
    "symmetry_fixed_atoms",
]
