"""Molecular dynamics and manual movements in the sectors of the group their structures keep.

PARSEC 1.5 uses no point group for molecular dynamics and for manual movements (``parsec.F90``), and the accelerated
driver kept both on the full grid.  A trajectory whose structures keep a group in exact arithmetic gains nothing from
that: symmetric positions with symmetric forces and a symmetric state of the integrator stay symmetric.  Under
``PARSEC_MOVING_SECTORS=auto`` (the default) the accelerated command line solves such a run in the sectors of that
group, checks the group at every step and leaves the sectors where a structure leaves it; ``off`` keeps both on the
full grid, bit for bit as before.

Which runs take the sectors (:func:`plan_moving_sectors`):

* molecular dynamics of the microcanonical integrator (``Friction_Coefficient`` below 1e-6) whose next structures are
  fixed by the forces alone: a restart from ``mdinit.dat`` (its positions, step and accelerations decide), or a fresh
  run with one movable atom (its Maxwell velocity is removed with the centre of mass).  A fresh run of more movable
  atoms draws random velocities, and a Langevin run a random force at every step: both leave any group after their
  first structure and keep the full grid from the start;
* manual movements (``Minimization: manual``, also ``Movement_Num: 0``): the structures of ``manual.dat`` after the
  input.

A cluster keeps the operations of the point group of its first structure (signed permutations about the origin, the
atoms fixed by ``Move_Flag`` as a species of their own) that also keep the state of the integrator (the displacement
of the last step and the accelerations) or every structure of ``manual.dat`` with the permutation of the first: the
detection of the driver takes them as pseudo-atoms (``symmetry_constraints``) where they keep less than the atoms.
Where no operation beside the identity is left the run keeps the full grid of before.  A periodic cell at the Gamma
point keeps the sectors its detection finds (``PARSEC_PERIODIC_SECTOR_FORCES``, the Fortran-derived routines under
``PARSEC_FORTRAN_SECTOR_FORCES=auto``) where its basis reaches the floor of the sectors of an SCF
(``PARSEC_PERIODIC_SECTOR_FLOOR``: the full grid needs not average the structures of these runs, as it must those of
a relaxation) and the group of those sectors has :data:`MOVING_CELL_MINIMUM_ORDER` operations or more (the driver
drops a smaller one before it builds it); after its preparation the state of a restart and the structures of
``manual.dat`` are checked against that group, and the run is prepared again on the full grid (a NOTE says why) where
they do not keep it.  ``on`` takes the sectors of a cell at any size and order.

At every step (:class:`MovingSectors`): the forces are averaged over the group of the sectors (their nonlocal term
is completed over it in any case); after a step of the integrator the new structure, the orbits of its atoms and the
state are checked against the group (:data:`MOVE_TOLERANCE`, 1e-6 bohr, for the positions and for the displacement
and the accelerations times the squared time step) and, where they keep it, the state is averaged over it, which
removes the round-off that would otherwise accumulate (as a relaxation averages its moved positions).  A structure
that leaves the group is prepared again in the sectors of the group it keeps (a cluster, with the state or the
remaining structures as constraints) or on the full grid, from the potential and the Hartree potential of the last
structure with a first filter (the subspace of other sectors does not carry over); a cell, and a second re-planning,
take the full grid.  A NOTE in ``parsec.out`` says so.

Two spin channels take the sectors under ``PARSEC_MOVING_SPIN_SECTORS=auto`` (the default since 2026-11-22) as an
unpolarized run does (the plan, the checks and the averages above, both channels in the sectors of the group of the
labels), but for a start whose labels carry ``Initial_Spin_Polarization`` of both signs (an antiferromagnetic or
broken-symmetry start), which keeps both channels on the full grid with a NOTE: the sectors of its labels can hold a
symmetric state that two channels on the full grid leave (Fe2 and Co2 from ``+-0.8`` and ``+-0.7``: 10-28 mRy higher,
forces up to 0.125 Ry/bohr apart, no guard fired); ``on`` takes those too.  The spin guard of every SCF loop leaves the
sectors where it fires, as in a single point: weighted sectors go on to the abelian sectors of the group, or straight
to the full grid where those would stop the same levels (``PARSEC_WEIGHTED_GUARD_SHORTCUT``), and abelian sectors to
the full grid; the system is prepared again in place and the later structures keep what it holds.  A re-planned
structure starts from the potentials of both channels.  ``off`` keeps both channels on the full grid, as before
2026-11-22.
Not taken (the full grid as before): spinors, ``Ignore_Symmetry``, ``--symmetry off``, a run that switches the
symmetry off for another reason too (point charges, a field, FDET with ``AFM_Scan``, fold blocks),
``Restart_From_Wfndat``, ``main.py --mpi``, wires, slabs and k-points.  The MPI full-SCF runner takes the same plan
since 2026-11-22 on the layouts that hold one sector of the identity (``benchmarks/mpi_full_scf._moving_plan``): it
prepares a structure that leaves the group again on every rank, in the sectors of the group the structure keeps or as
that sector (``MPISectorContext.reprepare``); it kept that sector for every such run before, and keeps it for two spin
channels (both channels as that sector, its basis shared by the ranks).
Units bohr, Ry and atomic units of time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from parsec_python.models import Atom

SWITCH = "PARSEC_MOVING_SECTORS"

#: Distance (bohr) from its atom of the pseudo-atom that carries a vector of the state of the integrator (the
#: displacement of the last step, the accelerations) into the detection: the largest vector of a set is scaled to it.
#: Atoms are 1.4 bohr apart or more, so an image of a pseudo-atom can only fall on the pseudo-atom of the image of its
#: atom.
PSEUDO_ATOM_OFFSET = 0.25

#: Label of the pseudo-atoms of the detection (no species carries it).
_LABEL = "<moving sectors {kind} {index}>"


def moving_sectors_setting() -> str:
    """``PARSEC_MOVING_SECTORS``: ``auto`` (the default: the sectors where the full grid of a cluster times its states
    reaches :data:`MOVING_SECTOR_WORK`, and where the basis of a cell reaches the floor of the sectors of an SCF,
    ``PARSEC_PERIODIC_SECTOR_FLOOR``, with a group of :data:`MOVING_CELL_MINIMUM_ORDER` operations or more), ``on``
    (the sectors whatever the size and order) or ``off`` (molecular dynamics and manual movements on the full grid, as
    before 2026-11-21)."""

    value = os.environ.get(SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{SWITCH} must be auto, on or off, not {value!r}")
    return value


SPIN_SWITCH = "PARSEC_MOVING_SPIN_SECTORS"
#: The value of ``PARSEC_MOVING_SPIN_SECTORS`` where none is named: ``auto`` (one A100 node, first runs, eleven structures
#: of a breathing restart, the full grid of both channels -> the sectors: O2 triplet on 0.18 bohr, 1.9e7 points x states,
#: 170.6 -> 60.1 s, 3.4e-9 Ry and 6.6e-7 Ry/bohr from the full grid; Si28H36 with two channels, 2.9e7, 51.8 -> 40.2 s,
#: 3.2e-7 Ry (its criterion 1e-6) and 4.9e-7 Ry/bohr; the cation whose guard leaves the sectors at its first structure
#: 53.1 -> 54.8 s with the trajectory of the full grid; the polarized XPS atoms and ions the same energies, 21-24 ->
#: 18-19 s).  Antiferromagnetic labels keep the full grid under it (:data:`ANTIFERROMAGNETIC_REASON`).
SPIN_DEFAULT = "auto"
#: Why ``auto`` keeps a start whose labels carry ``Initial_Spin_Polarization`` of both signs on the full grid.  One A100
#: node, a single point of ``Minimization: manual`` with forces (``Movement_Num: 0``), PBE, a sphere of 14 bohr at
#: 0.25 bohr, both channels in the 8 sectors of C4v against the full grid: Fe2 at 3.8 bohr from ``+-0.8`` -78.62964813
#: Ry converged against -78.64828958 (the full grid with the flip, converged) and -78.65719336 (unconverged), forces
#: 0.125 Ry/bohr apart; Co2 at 3.9 bohr from ``+-0.7`` -105.79380344 against -105.80443025 (unconverged); the guard and
#: the stability test of the flip saw neither.  (Cr2 at 3.6 bohr from ``+-0.8`` agreed: 98.2 -> 48.1 s for eleven
#: structures on 0.25 bohr, energies within 1.1e-7 Ry, forces 4.5e-6 Ry/bohr; ``on`` takes such starts.)
ANTIFERROMAGNETIC_REASON = (
    "antiferromagnetic labels (Initial_Spin_Polarization of both signs: the sectors of the labels can hold a symmetric "
    f"state that two channels on the full grid leave; {SPIN_SWITCH}=on takes them)"
)


def moving_spin_sectors_setting() -> str:
    """``PARSEC_MOVING_SPIN_SECTORS``: ``auto`` (the default since 2026-11-22: the molecular dynamics and manual movements
    of two spin channels take the sectors as an unpolarized run does under ``PARSEC_MOVING_SECTORS``, both channels in
    the sectors of the group with the spin guard of every SCF loop, but for antiferromagnetic labels, which keep the full
    grid: :data:`ANTIFERROMAGNETIC_REASON`), ``on`` (antiferromagnetic labels too) or ``off`` (both channels on the full
    grid, as before 2026-11-22)."""

    value = os.environ.get(SPIN_SWITCH, SPIN_DEFAULT).strip().lower() or SPIN_DEFAULT
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{SPIN_SWITCH} must be auto, on or off, not {value!r}")
    return value


def antiferromagnetic_labels(problem) -> bool:
    """Whether the labels of the atoms of ``problem`` carry ``Initial_Spin_Polarization`` of both signs."""

    species = getattr(problem, "pseudopotentials", {}) or {}
    values = [
        float(getattr(species.get(atom.symbol), "initial_spin_polarization", 0.0) or 0.0) for atom in problem.atoms
    ]
    return bool(values) and min(values) < 0.0 < max(values)


#: Points of the full grid of a cluster times its states from which ``PARSEC_MOVING_SECTORS=auto`` takes the sectors,
#: 2^24 (1.7e7), the floor of the sectors of a periodic cell (``PARSEC_PERIODIC_SECTOR_FLOOR``).  Eleven structures of
#: molecular dynamics restarted from a breathing state (SCF to 1e-6 Ry), one A100 node, first runs in both orders, the
#: full grid -> the sectors: Ge5 (3.9e5, 24 operations) 9.5 -> 9.8 s, benzene (5.4e6, 8) 11.6-12.5 -> 16.4-17.1 s,
#: naphthalene (1.6e7, 8) 18.5-19.0 -> 22.3-22.8 s; Si28H36 at a spacing of 0.5 bohr (1.8e7, the 2 of a mirror)
#: 19.2-19.4 -> 17.2-18.1 s, naphthalene at 0.36 bohr (2.2e7, 8) 27.9-28.5 -> 23.6-23.8 s, Si28H36 (2.9e7, 2)
#: 27.1-29.0 -> 22.4-23.3 s, C60 (3.5e7, 24) 19.5-19.8 -> 15.0-15.4 s: on a small grid the sectors cost more per SCF
#: iteration than they save.
MOVING_SECTOR_WORK = 2**24

#: Operations of the group of the sectors from which ``PARSEC_MOVING_SECTORS=auto`` keeps the sectors of a periodic
#: cell (``minimum_sector_order`` of the preparation; fewer: the full grid).  Silicon 3^3 with the breathing defect
#: (216 atoms, 6.9e7 grid points x states, the default Chebyshev-Davidson solver), fresh molecular dynamics of one
#: movable atom, eleven structures, one A100 node, first runs in both orders, the full grid -> the sectors: on a bond
#: (C3v, whose sectors are the 2 of a mirror) the default routines 30.1-30.4 -> 31.4-31.7 s, the Fortran-derived
#: routines 27.7-27.8 -> 30.0-30.2 s, and a peak of device memory of 5.2 -> 11.9 GiB (the first SCF of the two sectors
#: costs more, 14.8 s against 9.6 s, than their later steps save, 1.16 s against 1.56 s); at (a, 0, 0) (C2v, 4
#: sectors) 23.6-23.8 -> 16.6-17.1 s and 24.2-24.4 -> 17.0 s, 5.2 -> 2.6 GiB.  Diamond 3^3 of ``make_diamond.py``
#: (Chebyshev filtering, spline forces) gains in 2 sectors all the same (67.2-67.5 -> 46.4-46.5 s), which ``on``
#: takes, and in 4 (60.5-60.6 -> 34.3-34.8 s).
MOVING_CELL_MINIMUM_ORDER = 4


def move_tolerance() -> float:
    from ..Forces.sector_symmetry import MOVE_TOLERANCE

    return float(MOVE_TOLERANCE)


@dataclass(frozen=True)
class MovingPlan:
    """What :func:`plan_moving_sectors` decided for a run that keeps the sectors.

    ``kind`` is ``molecular_dynamics`` or ``manual``.  ``fixed`` marks the atoms ``Move_Flag`` fixes in molecular
    dynamics (a species of their own for the detection; ``None`` where every atom moves).  ``constraints`` are the
    pseudo-atoms of a cluster whose state or structures keep less than its atoms.  ``structures`` are the structures
    of ``manual.dat`` after the input (bohr), ``vectors`` the vectors of the state of a restart that the group must
    keep (bohr: the displacement of the last step and the accelerations times the squared time step).  ``order`` is
    the order of the group a cluster keeps (``None`` for a cell, whose group the driver finds).  ``minimum_order``
    (a cell): the sectors of the group the driver finds are kept where it has that many operations or more
    (``minimum_sector_order`` of the preparation; ``0``: any group).
    """

    kind: str
    periodic: bool
    fixed: tuple[bool, ...] | None = None
    constraints: tuple[Atom, ...] = ()
    structures: tuple[np.ndarray, ...] = ()
    vectors: tuple[np.ndarray, ...] = ()
    order: int | None = None
    atoms_order: int | None = None
    minimum_order: int = 0

    def note(self, cell_order: int | None = None) -> str:
        """The NOTE line of ``parsec.out`` (``cell_order``: the operations of the sectors the driver found for a
        cell, which :meth:`MovingSectors.check_start` writes after the preparation)."""

        what = "molecular dynamics" if self.kind == "molecular_dynamics" else "manual movements"
        if self.periodic:
            kept = (
                f"the sectors of the {cell_order} operations its detection finds for the cell"
                if cell_order is not None
                else "the sectors of the group its detection finds for the cell"
                + (f" where they have {self.minimum_order} operations or more" if self.minimum_order > 1 else "")
            )
        elif self.order is not None and self.atoms_order is not None and self.order < self.atoms_order:
            kept = (
                f"sectors of the {self.order} of the {self.atoms_order} operations of the atoms that "
                + ("the state of the integrator keeps" if self.kind == "molecular_dynamics" else "every structure keeps")
                + " (the detection below names them)"
            )
        else:
            kept = f"sectors of the {self.order} operations of the atoms (the detection below names them)"
        return (
            f"NOTE: PARSEC 1.5 uses no point group for {what}; the accelerated driver keeps {kept}, checks the group "
            f"at every structure and leaves the sectors where a structure leaves it ({SWITCH}={moving_sectors_setting()})"
        )


def _kind(translation) -> str | None:
    if getattr(translation, "molecular_dynamics", None) is not None:
        return "molecular_dynamics"
    relaxation = getattr(translation, "relaxation", None)
    if relaxation is not None and relaxation.method == "manual":
        return "manual"
    return None


def _labelled(atoms, fixed) -> tuple[Atom, ...]:
    from ..driver import FIXED_ATOM_LABEL

    atoms = tuple(atoms)
    if fixed is None:
        return atoms
    return tuple(
        replace(atom, symbol=atom.symbol + FIXED_ATOM_LABEL) if flag else atom for atom, flag in zip(atoms, fixed)
    )


def keeping_operations(operations, vectors: Sequence[np.ndarray], tolerance: float):
    """The operations of ``operations`` (:class:`parsec_python.Forces.AtomSymmetryOperation`) under which every set
    of ``vectors`` (``(atoms, 3)``) is invariant to ``tolerance``: ``R u_a = u_{image(a)}``."""

    kept = []
    for operation in operations:
        matrix = np.asarray(operation.matrix, dtype=np.float64)
        image = np.asarray(operation.image, dtype=np.int64)
        if all(
            float(np.max(np.abs(np.asarray(vector) @ matrix.T - np.asarray(vector)[image]), initial=0.0)) <= tolerance
            for vector in vectors
        ):
            kept.append(operation)
    return tuple(kept)


def structure_operations(operations, atoms, positions: np.ndarray, tolerance: float):
    """The operations of ``operations`` (found for ``atoms``) that carry the structure ``positions`` (the same atoms
    moved) onto itself with the permutation they have for ``atoms``: the displacement is invariant."""

    reference = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    return keeping_operations(operations, (np.asarray(positions, dtype=np.float64) - reference,), tolerance)


def vector_constraints(atoms, vectors: Sequence[np.ndarray], kind: str) -> tuple[Atom, ...]:
    """Pseudo-atoms that only the operations keeping ``vectors`` keep: for each set ``u``, a pseudo-atom of its own
    label at ``r_a + d u_a / max|u|`` for every atom (``d`` = :data:`PSEUDO_ATOM_OFFSET`)."""

    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    found: list[Atom] = []
    for index, vector in enumerate(vectors):
        vector = np.asarray(vector, dtype=np.float64)
        largest = float(np.max(np.linalg.norm(vector, axis=1), initial=0.0))
        if not largest > 0.0:
            continue
        label = _LABEL.format(kind=kind, index=index)
        for position, value in zip(positions, vector * (PSEUDO_ATOM_OFFSET / largest)):
            found.append(Atom(label, tuple(float(x) for x in position + value)))
    return tuple(found)


def dynamics_vectors(state, time_step: float) -> tuple[np.ndarray, ...]:
    """The vectors of a state of the microcanonical integrator that its next step reads besides the positions and
    forces (bohr): the displacement of the last step and the accelerations times ``dt^2``."""

    current = np.asarray(state.current_positions, dtype=np.float64)
    previous = np.asarray(state.previous_positions, dtype=np.float64)
    accelerations = np.asarray(state.accelerations, dtype=np.float64)
    return (current - previous, accelerations * float(time_step) ** 2)


def _dynamics_start(translation, restart) -> tuple[tuple[np.ndarray, ...] | None, tuple | None, str | None]:
    """``(vectors, movable, reason)`` of the start of a molecular dynamics run: the vectors of its state, the movable
    flags, or why it leaves every group at its first step (``vectors`` ``None``)."""

    from parsec_python.MolecularDynamics.settings import MICROCANONICAL_FRICTION

    settings = translation.molecular_dynamics
    if float(settings.friction_coefficient) >= MICROCANONICAL_FRICTION:
        return None, None, "the Langevin integrator draws a random force at every step"
    if restart is not None:
        movable = tuple(bool(int(value)) for value in np.asarray(restart.movable))
        positions = np.asarray(restart.current_positions, dtype=np.float64)
        state = SimpleNamespace(
            current_positions=positions,
            previous_positions=np.asarray(restart.previous_positions, dtype=np.float64),
            accelerations=np.asarray(restart.accelerations, dtype=np.float64),
        )
        return dynamics_vectors(state, settings.time_step), movable, None
    movable = tuple(bool(flag) for flag in translation.movable)
    if sum(movable) > 1:
        return None, None, "the initial velocities are random (Maxwell velocities of the movable atoms)"
    # One movable atom: its Maxwell velocity is that of the centre of mass of the movable atoms, removed.
    count = len(movable)
    return (np.zeros((count, 3)), np.zeros((count, 3))), movable, None


def cell_below_floor(problem) -> str | None:
    """Why the basis of the periodic cell of ``problem`` (grid points times states) is below the floor of the
    sectors of an SCF in one process (``PARSEC_PERIODIC_SECTOR_FLOOR``, the driver's count), or ``None``."""

    from ..driver import _periodic_basis_below_floor, periodic_sector_floor_setting

    floor = periodic_sector_floor_setting()
    below = _periodic_basis_below_floor(problem, None, floor)
    if below is None:
        return None
    return (
        f"the {below[0]} grid points x {below[1]} states of the cell are below the floor of the sectors, {floor} "
        "entries (PARSEC_PERIODIC_SECTOR_FLOOR)"
    )


def fortran_fixed_atoms_lifted(translation, *, mpi_requested: bool = False, symmetry_argument: str | None = None):
    """Whether the command line takes the relaxation (``bfgs``, ``simple``) of a periodic cell on the Fortran-derived
    routines with atoms fixed by ``Move_Flag`` in the sectors of the subgroup of the fixed atoms
    (``fortran_fixed_atoms`` of :func:`..cli.periodic_sector_forces_refusal`), as the default routines are taken.

    ``off``: never (their full grid, as before); MPI: never (as before); ``on`` or ``--symmetry on``: whatever the
    size; ``auto``: where the basis of the cell reaches the floor of the sectors of an SCF
    (:func:`cell_below_floor`).  The displaced Si8 cube with Move_Flag (5,832 grid points x 22 states), five BFGS
    moves, one A100 node, first runs in both orders: 4.8-5.4 s on the full grid against 6.3-6.4 s in the sectors,
    slower at every step (SCF and forces 0.42 s against 0.59 s); diamond 2^3 (64 atoms, 6.3e6) 15.1-15.2 -> 14.3-14.4 s
    in the sectors, below the floor all the same.
    """

    setting = moving_sectors_setting()
    if setting == "off" or mpi_requested:
        return False
    if setting == "on" or symmetry_argument == "on":
        return True
    problem = translation.problem
    if problem.periodic_cell is None:
        return False
    return cell_below_floor(problem) is None


RUNNER_FORTRAN_FIXED_SWITCH = "PARSEC_RUNNER_FORTRAN_FIXED_SECTORS"


def runner_fortran_fixed_setting() -> str:
    """``PARSEC_RUNNER_FORTRAN_FIXED_SECTORS``: ``auto`` (default, since 2026-11-22) or ``off``.

    ``auto``: the full-SCF runner (``acceleration/benchmarks/mpi_full_scf.py``) takes a relaxation (``bfgs``,
    ``simple``) of a periodic cell on the Fortran-derived routines with atoms fixed by ``Move_Flag``
    (:func:`fortran_fixed_relaxation`) in the sectors of the subgroup that maps the fixed atoms onto fixed atoms, as
    it takes those of the default routines, the ranks dividing the sectors, whatever their number (main.py in one
    process keeps such a relaxation on the full grid below the floor of the sectors and for fewer than
    :data:`MOVING_CELL_MINIMUM_ORDER` operations, :func:`fortran_fixed_atoms_lifted`; the runner has no full grid to
    fall back to); not under ``PARSEC_MOVING_SECTORS=off``.  ``off``: refused by name, as before (main.py runs it)."""

    value = os.environ.get(RUNNER_FORTRAN_FIXED_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{RUNNER_FORTRAN_FIXED_SWITCH} must be auto or off, not {value!r}")
    return value


def runner_fortran_fixed_atoms(translation) -> bool:
    """Whether the full-SCF runner takes ``translation`` in the sectors of the subgroup of its fixed atoms
    (:func:`runner_fortran_fixed_setting`): a relaxation of :func:`fortran_fixed_relaxation` under ``auto``, with
    ``PARSEC_MOVING_SECTORS`` other than ``off``.  The same answer on every rank (the input and the switches)."""

    setting = runner_fortran_fixed_setting()
    if setting == "off" or moving_sectors_setting() == "off":
        return False
    return fortran_fixed_relaxation(translation)


def fortran_fixed_relaxation(translation) -> bool:
    """Whether ``translation`` is the relaxation (``bfgs``, ``simple``) of a periodic cell on the Fortran-derived
    routines with atoms fixed by ``Move_Flag``, the run :func:`fortran_fixed_atoms_lifted` is about."""

    problem = translation.problem
    relaxation = getattr(translation, "relaxation", None)
    movable = getattr(translation, "movable", None)
    return bool(
        problem.periodic_cell is not None
        and relaxation is not None
        and relaxation.method in ("bfgs", "simple")
        and movable is not None
        and not all(bool(flag) for flag in movable)
        and (
            getattr(getattr(translation, "routines", None), "fortran", False)
            or getattr(problem.grid, "fortran_routines", False)
        )
    )


def fortran_fixed_minimum_order(translation, *, lifted: bool, symmetry_argument: str | None = None) -> int:
    """The ``minimum_sector_order`` of the preparation of a relaxation that :func:`fortran_fixed_atoms_lifted` took
    into the sectors: :data:`MOVING_CELL_MINIMUM_ORDER` under ``auto`` (fewer operations: the full grid of those
    routines, as before), ``0`` under ``on``, with ``--symmetry on`` and for any other run.  Silicon 3^3 on the
    Fortran-derived routines (216 atoms, 6.9e7 grid points x states), five BFGS moves of the breathing defect, one
    A100 node, first runs in both orders, the full grid -> the sectors: the neighbour on +x+y+z fixed (C3v, the 2
    sectors of a mirror) 29.8-30.2 -> 32.2-32.5 s and 5.3 -> 12.3 GiB of device memory; the atom at (a, 0, 0) fixed
    (C2v, 4 sectors) 31.1-31.3 -> 24.5-25.1 s and 5.3 -> 2.4 GiB."""

    if not lifted or moving_sectors_setting() != "auto" or symmetry_argument == "on":
        return 0
    return MOVING_CELL_MINIMUM_ORDER if fortran_fixed_relaxation(translation) else 0


def plan_moving_sectors(
    translation,
    *,
    dynamics_restart=None,
    manual=None,
    mpi_requested: bool = False,
    symmetry_argument: str | None = None,
) -> tuple[MovingPlan | None, str | None]:
    """The plan of ``PARSEC_MOVING_SECTORS=auto`` for ``translation``, or ``(None, reason)``.

    ``dynamics_restart`` is the ``mdinit.dat`` of a restarted molecular dynamics run
    (:func:`parsec_python.MolecularDynamics.restart_structure`), ``manual`` the structures of ``manual.dat`` after the
    input (:func:`parsec_python.Relaxation.run.manual_structures`; ``None``: none).  ``reason`` is ``None`` for a run
    that is neither molecular dynamics nor manual movements, and under ``off``; otherwise it says why the full grid
    stays.  The switch is read for every run, so that a value it does not know is refused.
    """

    setting = moving_sectors_setting()
    # (read in every run, so that a value it does not know is refused)
    spin_setting = moving_spin_sectors_setting()
    kind = _kind(translation)
    if kind is None or setting == "off":
        return None, None
    problem = translation.problem
    scf = problem.scf
    reasons = set(getattr(translation, "symmetry_off_reasons", ()) or ())
    others = sorted(reasons - {kind if kind == "molecular_dynamics" else "manual"})
    # Two spin channels that the driver keeps on the full grid in one process (Spin_Symmetry_Sectors: off,
    # PARSEC_CUPY_DISTRIBUTED_FILTER=1, ...: driver._spin_sector_refusal) plan the full grid here as well, so that
    # parsec.out does not announce sectors that the preparation then drops (the same route and bits either way).
    spin_full_grid = None
    if bool(getattr(scf, "spin_polarized", False)) and spin_setting != "off" and not mpi_requested:
        from ..driver import _spin_sector_refusal

        spin_full_grid = _spin_sector_refusal(problem)
    for condition, reason in (
        (symmetry_argument == "off", "--symmetry off"),
        (bool(getattr(translation, "ignore_symmetry", False)), "Ignore_Symmetry"),
        (bool(others), "the symmetry is switched off for " + ", ".join(others) + " as well"),
        (getattr(translation, "afm_scan", None) is not None, "AFM_Scan"),
        (
            bool(getattr(translation, "restart_run", False) and getattr(translation, "restart_from_wfndat", False)),
            "Restart_From_Wfndat (its orbitals are full-grid states)",
        ),
        (bool(getattr(problem, "point_charges", ())), "point charges"),
        (getattr(translation, "polarizability_field", None) is not None, "Polarizability"),
        # (PARSEC_MOVING_SPIN_SECTORS=auto takes them as the unpolarized runs)
        (bool(getattr(scf, "spin_polarized", False)) and spin_setting == "off", "two spin channels"),
        (
            bool(getattr(scf, "spin_polarized", False)) and spin_setting == "auto" and antiferromagnetic_labels(problem),
            ANTIFERROMAGNETIC_REASON,
        ),
        (
            spin_full_grid is not None,
            f"two spin channels whose SCF keeps the full grid ({spin_full_grid})",
        ),
        (
            bool(getattr(scf, "self_consistent_spin_orbit", False) or getattr(scf, "noncollinear", False)),
            "spinors",
        ),
        (bool(mpi_requested), "MPI"),
        (
            problem.periodic_cell is not None and getattr(problem.grid, "periodicity", 3) != 3,
            "a wire or a slab",
        ),
    ):
        if condition:
            return None, f"{reason}: the full grid"
    vectors: tuple[np.ndarray, ...] = ()
    fixed = None
    structures: tuple[np.ndarray, ...] = ()
    if kind == "molecular_dynamics":
        start, movable, reason = _dynamics_start(translation, dynamics_restart)
        if start is None:
            return None, f"{reason}: the trajectory leaves every group after its first structure; the full grid"
        vectors = start
        if not all(movable):
            fixed = tuple(not flag for flag in movable)
    else:
        structures = tuple(np.asarray(item, dtype=np.float64) for item in (manual or ()))
    if problem.periodic_cell is not None:
        minimum = 0
        if setting == "auto" and symmetry_argument != "on":
            below = cell_below_floor(problem)
            if below is not None:
                return None, below + f" ({SWITCH}=on takes them): the full grid"
            minimum = MOVING_CELL_MINIMUM_ORDER
        return (
            MovingPlan(
                kind=kind, periodic=True, fixed=fixed, structures=structures, vectors=vectors, minimum_order=minimum
            ),
            None,
        )
    if setting == "auto":
        from .symmetry_off_inputs import estimated_full_grid_work

        work = estimated_full_grid_work(problem)
        if work is None or work < MOVING_SECTOR_WORK:
            return None, (
                f"the full grid of {'?' if work is None else f'{work:.2e}'} points x states is below the "
                f"{MOVING_SECTOR_WORK:.2e} from which the sectors are faster ({SWITCH}=on takes them): the full grid"
            )
    return _cluster_plan(translation, kind, fixed, structures, vectors)


def _cluster_plan(translation, kind, fixed, structures, vectors):
    """The plan of a cluster: the operations of its atoms that keep ``vectors`` and ``structures``."""

    from parsec_python.Forces import atom_symmetry_operations
    from parsec_python.V_ion import center_cluster_geometry

    from ..Symmetry.axis_reflection import symmetry_atom_tolerance

    problem = translation.problem
    atoms = center_cluster_geometry(problem.atoms) if problem.recenter_geometry else tuple(problem.atoms)
    tolerance = symmetry_atom_tolerance()
    group = atom_symmetry_operations(_labelled(atoms, fixed), tolerance=tolerance)
    if len(group) <= 1:
        return None, "the atoms keep no symmetry operation about the origin: the full grid"
    kept = keeping_operations(group, vectors, move_tolerance()) if vectors else group
    for positions in structures:
        if positions.shape != (len(atoms), 3):
            raise ValueError(f"a structure of manual.dat has {positions.shape[0]} atoms, the input {len(atoms)}")
        kept = structure_operations(kept, atoms, positions, tolerance)
    if len(kept) <= 1:
        what = "the state of the integrator keeps" if kind == "molecular_dynamics" else "the structures of manual.dat keep"
        return None, f"{what} no operation of the {len(group)} of the atoms but the identity: the full grid"
    constraints: tuple[Atom, ...] = ()
    if len(kept) < len(group):
        reference = np.asarray([atom.position for atom in atoms], dtype=np.float64)
        sets = vectors if kind == "molecular_dynamics" else tuple(positions - reference for positions in structures)
        constraints = vector_constraints(atoms, sets, "dynamics" if kind == "molecular_dynamics" else "manual")
    return (
        MovingPlan(
            kind=kind,
            periodic=False,
            fixed=fixed,
            constraints=constraints,
            structures=structures,
            vectors=vectors,
            order=len(kept),
            atoms_order=len(group),
        ),
        None,
    )


# --------------------------------------------------------------------------------------------------------------
# The group of a prepared system
# --------------------------------------------------------------------------------------------------------------


def sector_reduction(system):
    """The reduction of the sectors of ``system`` where it has more than the identity, else ``None``."""

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    if reduction is None or int(getattr(reduction, "group_order", 1)) <= 1:
        return None
    return reduction


def group_atom_operations(system, atoms=None):
    """The operations of the group the sectors of ``system`` keep, on ``atoms`` (default: those of ``system``).

    The full group ``G`` of a point-group reduction, the group of Abelian sectors: for a periodic cell those of
    :func:`..Forces.sector_symmetry.sector_symmetry_operations` (with the translations of the cell), for a cluster
    :class:`parsec_python.Forces.AtomSymmetryOperation` matched to :data:`MOVE_TOLERANCE`.  ``None`` without sectors;
    ``ValueError`` where an operation carries an atom onto no atom of its species.
    """

    reduction = sector_reduction(system)
    if reduction is None:
        return None
    atoms = tuple(system.atoms if atoms is None else atoms)
    cell = getattr(system.input, "periodic_cell", None)
    if cell is not None:
        from ..Forces.sector_symmetry import sector_symmetry_operations

        stand_in = SimpleNamespace(input=system.input, backend=system.backend, atoms=atoms, grid=system.grid)
        operations = sector_symmetry_operations(stand_in)
        if operations is None:
            raise ValueError("the sectors of the cell have no operations on its atoms")
        return tuple(operations)
    from parsec_python.Forces import atom_symmetry_operations

    from ..Symmetry.axis_reflection import abelian_sector_operations

    full = getattr(reduction, "full_operations", None)
    matrices = np.rint(
        np.asarray(full if full is not None else abelian_sector_operations(reduction), dtype=np.float64)
    ).reshape(-1, 3, 3)
    identity = [index for index, matrix in enumerate(matrices) if np.array_equal(matrix, np.eye(3))]
    if len(identity) != 1:
        raise ValueError("the operations of the sectors hold no single identity")
    order = identity + [index for index in range(len(matrices)) if index != identity[0]]
    operations = atom_symmetry_operations(atoms, tolerance=move_tolerance(), candidates=matrices[order])
    if len(operations) != len(matrices):
        raise ValueError(
            f"{len(matrices) - len(operations)} of the {len(matrices)} operations of the sectors carry an atom onto no "
            "atom of its species"
        )
    return operations


def check_structure(system, atoms) -> None:
    """Raise ``ValueError`` unless the sectors of ``system`` hold for ``atoms``: every operation of the group carries
    them onto themselves to :data:`MOVE_TOLERANCE` and their orbits are those of the preparation
    (:func:`..Forces.sector_symmetry.check_sector_atoms`, for a cluster as for a cell)."""

    from ..Forces.sector_symmetry import check_sector_atoms

    reduction = sector_reduction(system)
    if reduction is None:
        return
    cell = getattr(system.input, "periodic_cell", None)
    grid = getattr(reduction, "grid", None)
    if grid is None:
        grid = system.grid
    check_sector_atoms(
        reduction, atoms, grid, None if cell is None else np.asarray(cell.lattice_vectors, dtype=np.float64)
    )


def _detection_text(system) -> str:
    details = dict(getattr(getattr(system, "backend_info", None), "details", ()) or ())
    text = details.get("symmetry_detection")
    return str(text) if text else "the full grid"


# --------------------------------------------------------------------------------------------------------------
# The ionic loop
# --------------------------------------------------------------------------------------------------------------


class MovingSectors:
    """The per-structure check of :mod:`.moving_sectors` around the ionic backend of one run.

    ``prepare(problem, symmetry=..., constraints=..., fixed=...)`` prepares a system as the command line prepared the
    first one (backend, cache, sector forces); ``write`` takes the NOTE lines of ``parsec.out``.  :meth:`wrap` gives the
    backend of the loop, :meth:`after_step` is the hook of the molecular dynamics loop
    (:func:`parsec_python.MolecularDynamics.run_recorded_molecular_dynamics`).
    """

    def __init__(self, plan: MovingPlan, problem, *, prepare: Callable, write: Callable[[str], None], time_step=None):
        self.plan = plan
        self.problem = problem
        self._prepare = prepare
        self._write = write
        self._time_step = None if time_step is None else float(time_step)
        self._system = None
        self._broken: str | None = None
        self._last_state = None
        self._pending: dict[int, np.ndarray] = {}
        self._moves = 0
        #: re-plannings made, and whether the run has left the sectors for good
        self.replans = 0
        self.full_grid = False
        #: the structures (counting from 0) solved in sectors, for the reports and the tests
        self.sector_structures: list[int] = []

    # -- preparation ------------------------------------------------------------------------------------------

    def check_start(self, system) -> None:
        """After the preparation: the state of a restart and the structures of ``manual.dat`` must keep the group the
        driver found; otherwise ``system`` is prepared again (in place) on the full grid.  The NOTE of a cell's plan
        (with the operations of its sectors), or why it has none, goes into ``parsec.out`` here."""

        if sector_reduction(system) is None:
            self.full_grid = True
            self._write(f"NOTE: {SWITCH}: the first structure has no symmetry sectors ({_detection_text(system)})")
            return
        reason = None
        try:
            if self.plan.vectors:
                operations = group_atom_operations(system)
                kept = keeping_operations(operations, self.plan.vectors, move_tolerance())
                if len(kept) != len(operations):
                    reason = (
                        f"the state of the integrator keeps {len(kept)} of the {len(operations)} operations of the "
                        "sectors"
                    )
            if reason is None:
                for index, positions in enumerate(self.plan.structures):
                    atoms = tuple(replace(atom, position=tuple(row)) for atom, row in zip(system.atoms, positions))
                    try:
                        check_structure(system, atoms)
                    except ValueError as error:
                        reason = f"structure {index + 1} of manual.dat leaves the group of the sectors ({error})"
                        break
        except ValueError as error:
            reason = f"{error}"
        if reason is None:
            if self.plan.periodic:
                # (a cell: the NOTE of the plan once the driver found its sectors; a cluster wrote it before)
                self._write(self.plan.note(int(sector_reduction(system).group_order)))
            return
        self._write(f"NOTE: {SWITCH}: {reason}: the run is prepared again on the full grid")
        self._replace(system, self.problem, symmetry="off")
        self.full_grid = True

    # -- the backend ------------------------------------------------------------------------------------------

    def wrap(self, backend):
        """``backend`` with the forces, moves and warm starts of this run."""

        return replace(
            backend,
            run_scf=self._run_scf(backend.run_scf),
            compute_forces=self._forces(backend.compute_forces),
            update_geometry=self._update(backend.update_geometry),
            warm_start=self._warm(backend.warm_start),
        )

    def _run_scf(self, inner):
        def run_scf(system, *arguments, **keywords):
            self._system = system
            if not self.full_grid and sector_reduction(system) is not None:
                self.sector_structures.append(self._moves)
            return inner(system, *arguments, **keywords)

        return run_scf

    def _forces(self, inner):
        def compute_forces(system, result, *, symmetrize=False, operations=None):
            if not self.full_grid and sector_reduction(system) is not None:
                # averaged over the group of the sectors, which the structure keeps
                return inner(system, result, symmetrize=True, operations=group_atom_operations(system))
            return inner(system, result, symmetrize=symmetrize, operations=operations)

        return compute_forces

    def after_step(self, step):
        """The hook after a step of the integrator: the new structure and state checked against the group of the
        sectors, and the state averaged over it where they keep it."""

        system = self._system
        self._last_state = step.state
        if system is None or self.full_grid or sector_reduction(system) is None:
            return step
        from parsec_python.Forces import symmetrize_forces, symmetrize_positions

        state = step.state
        atoms = tuple(replace(atom, position=tuple(row)) for atom, row in zip(system.atoms, state.current_positions))
        try:
            check_structure(system, atoms)
            operations = group_atom_operations(system, atoms)
        except ValueError as error:
            self._broken = f"{error}"
            return step
        vectors = dynamics_vectors(state, self._time_step if self._time_step is not None else 0.0)
        kept = keeping_operations(operations, vectors, move_tolerance())
        if len(kept) != len(operations):
            self._broken = (
                f"the state of the integrator keeps {len(kept)} of the {len(operations)} operations of the sectors"
            )
            return step
        symmetric = replace(
            state,
            current_positions=symmetrize_positions(np.asarray(state.current_positions, dtype=np.float64), operations),
            previous_positions=symmetrize_positions(
                np.asarray(state.previous_positions, dtype=np.float64), operations
            ),
            accelerations=symmetrize_forces(np.asarray(state.accelerations, dtype=np.float64), operations),
        )
        self._last_state = symmetric
        return replace(step, state=symmetric, positions=symmetric.current_positions)

    def _update(self, inner):
        def update_geometry(system, atoms):
            self._moves += 1
            if self.full_grid or sector_reduction(system) is None:
                return inner(system, atoms)
            reason, self._broken = self._broken, None
            if reason is None:
                try:
                    check_structure(system, atoms)
                except ValueError as error:
                    reason = f"{error}"
            if reason is None:
                return inner(system, atoms)
            return self._replan(system, tuple(atoms), reason)

        return update_geometry

    def _warm(self, inner):
        def warm_start(previous_system, previous_result, system):
            old = self._pending.pop(id(system), None)
            if old is None:
                return inner(previous_system, previous_result, system)
            from parsec_python.SCF import SCFWarmStart
            from parsec_python.SCF.single_point import meta_gga_warm_start

            # parsec.F90 after domove (the last mixed potential with the ionic potential exchanged, the last Hartree
            # potential), without the subspace of the sectors of before: a first filter
            new = np.asarray(system.ionic_potential, dtype=np.float64)
            channels = getattr(previous_result, "spin", None)
            if channels is not None:
                # two spin channels (PARSEC_MOVING_SPIN_SECTORS=auto): the potential of each, no subspace of either
                start = SCFWarmStart(
                    input_potential=tuple(
                        np.asarray(channel.next_effective_potential, dtype=np.float64) - old + new
                        for channel in channels
                    ),
                    hartree_potential=np.asarray(previous_result.hartree_potential, dtype=np.float64),
                    eigensolver_state=(None, None),
                )
                return meta_gga_warm_start(start, previous_result)
            start = SCFWarmStart(
                input_potential=np.asarray(previous_result.next_effective_potential, dtype=np.float64)
                - old
                + new,
                hartree_potential=np.asarray(previous_result.hartree_potential, dtype=np.float64),
                eigensolver_state=None,
            )
            return meta_gga_warm_start(start, previous_result)

        return warm_start

    # -- re-planning ------------------------------------------------------------------------------------------

    def _replan(self, system, atoms, reason: str):
        """Prepare ``system`` again (in place) for ``atoms``: the sectors of the group they keep, or the full grid."""

        old = np.array(system.ionic_potential, dtype=np.float64, copy=True)
        problem = replace(self.problem, atoms=atoms, recenter_geometry=False)
        symmetry, constraints, fixed = "off", (), None
        if not self.plan.periodic and self.replans == 0:
            symmetry, constraints, fixed = self._cluster_replan(atoms)
        self.replans += 1
        self._replace(system, problem, symmetry=symmetry, constraints=constraints, fixed=fixed)
        if sector_reduction(system) is None:
            self.full_grid = True
        self._pending[id(system)] = old
        self._write(
            f"NOTE: {SWITCH}: structure {self._moves} leaves the group of the sectors ({reason}); prepared again"
            + (f" in the sectors it keeps: {_detection_text(system)}" if not self.full_grid else " on the full grid")
            + ", from the potential of the structure before"
        )
        return system

    def _cluster_replan(self, atoms):
        """``(symmetry, constraints, fixed)`` of a cluster structure that left the group: the operations of its atoms
        that keep the state of the integrator or the remaining structures of ``manual.dat``."""

        from parsec_python.Forces import atom_symmetry_operations

        from ..Symmetry.axis_reflection import symmetry_atom_tolerance

        fixed = self.plan.fixed
        tolerance = symmetry_atom_tolerance()
        group = atom_symmetry_operations(_labelled(atoms, fixed), tolerance=tolerance)
        if len(group) <= 1:
            return "off", (), None
        if self.plan.kind == "molecular_dynamics":
            state = self._last_state
            vectors = () if state is None else dynamics_vectors(state, self._time_step or 0.0)
            kept = keeping_operations(group, vectors, move_tolerance()) if vectors else group
            sets = vectors
        else:
            remaining = self.plan.structures[self._moves :]
            kept = group
            for positions in remaining:
                kept = structure_operations(kept, atoms, positions, tolerance)
            reference = np.asarray([atom.position for atom in atoms], dtype=np.float64)
            sets = tuple(positions - reference for positions in remaining)
        if len(kept) <= 1:
            return "off", (), None
        constraints = vector_constraints(atoms, sets, "replan") if len(kept) < len(group) else ()
        return "auto", constraints, fixed

    def _replace(self, system, problem, *, symmetry: str, constraints=(), fixed=None) -> None:
        from .symmetry_off_inputs import release_system, replace_system

        release_system(system)
        prepared = self._prepare(problem, symmetry=symmetry, constraints=tuple(constraints), fixed=fixed)
        if prepared.grid.size != system.grid.size:
            raise RuntimeError("the structure was prepared again on another grid")
        replace_system(system, prepared)


__all__ = [
    "MOVING_CELL_MINIMUM_ORDER",
    "MOVING_SECTOR_WORK",
    "ANTIFERROMAGNETIC_REASON",
    "SPIN_SWITCH",
    "antiferromagnetic_labels",
    "moving_spin_sectors_setting",
    "MovingPlan",
    "MovingSectors",
    "PSEUDO_ATOM_OFFSET",
    "RUNNER_FORTRAN_FIXED_SWITCH",
    "SWITCH",
    "cell_below_floor",
    "check_structure",
    "dynamics_vectors",
    "fortran_fixed_atoms_lifted",
    "fortran_fixed_minimum_order",
    "fortran_fixed_relaxation",
    "group_atom_operations",
    "keeping_operations",
    "moving_sectors_setting",
    "plan_moving_sectors",
    "runner_fortran_fixed_atoms",
    "runner_fortran_fixed_setting",
    "sector_reduction",
    "structure_operations",
    "vector_constraints",
]
