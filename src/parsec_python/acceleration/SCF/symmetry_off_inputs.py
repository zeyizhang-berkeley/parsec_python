"""Inputs that switch the point group off in PARSEC 1.5, kept in the sectors of the group they leave intact.

PARSEC 1.5 uses no point group for a cluster with ``Polarizability``, ``Add_Point_Charges``,
``Foldblock_Side_Length`` 2, a restart from ``wfn.dat`` (``Restart_From_Wfndat``) and, in this port, for the two
channels of a spin-polarized ``Restart_Run``.  The accelerated driver keeps the sectors of the subgroup each of them
leaves intact, every case behind a switch whose ``off`` restores the full grid of before, bit for bit (FDET keeps
the full grid; molecular dynamics and manual movements keep the sectors of the group their structures keep under
``PARSEC_MOVING_SECTORS``, :mod:`.moving_sectors`, which checks the group at every structure):

``PARSEC_POLARIZABILITY_SYMMETRY`` (``auto``|``on``|``off``)
    The seven field loops of ``polar.f90``; ``auto`` takes the route below where the points of the full grid times
    the states reach :data:`POLARIZABILITY_SECTOR_WORK` (1.2e7; below it the warm loops of one full grid are faster,
    and the run is that of ``off``), ``on`` whatever the size.  The zero-field loop solves in the sectors of the group ``G`` of the
    atoms.  A field along ``+-e_a`` keeps the stabilizer ``G_a = {R in G : R e_a = e_a}`` of its axis: the loops
    along an axis solve in the sectors of ``G_a`` (:func:`.driver.prepare_single_point` with an axis
    constraint).  A loop whose field direction ``d`` an operation ``R`` of ``G`` (one that also maps the active
    grid onto itself) carries from a loop ``d'`` solved later in the order of ``polar.f90`` is not solved: its
    energy is that of ``d'`` and its density the image ``rho_d(R r) = rho_d'(r)`` (its dipole ``R mu_d'``).  So the
    last loop (``-z``), whose forces and state the run reports, is always solved.  A Td or Oh cluster solves the
    zero field and ``-z`` only, a D2h molecule the zero field, ``-x``, ``-y`` and ``-z``.  A loop that starts on a
    new group starts from the potential of the zero-field loop with a first filter; the loops of one axis follow
    each other as in ``polar.f90`` (``pot%vnew = pot%vold`` and the saved subspace).
``PARSEC_POINT_CHARGE_SYMMETRY`` (``auto``|``off``)
    The group of the atoms that carries every point charge onto a charge of the same value: the charges are
    matched as atoms of their own label by the detection.
``PARSEC_FOLDBLOCK_SYMMETRY`` (``auto``|``off``)
    ``Foldblock_Side_Length`` 2 switches the symmetry off in PARSEC 1.5, which then takes the grid shift zero
    (``usrinputfile.F90`` 814-817, 1363-1382).  That grid stays; the detection keeps the operations that map its
    domain of whole 2 x 2 x 2 blocks onto itself (the permutations of the axes: the blocks ``[2m, 2m + 1]`` of a
    grid without shift are no images of each other under a sign change).
``PARSEC_SPIN_RESTART_SECTORS`` (``auto``|``on``|``off``)
    A spin-polarized ``Restart_Run`` of a cluster keeps the sectors where both channel densities of ``parsec.dat``
    are invariant under the group the system was prepared in (the atoms, with the point charges where they keep a
    group, on the grid: test 2 of :mod:`.spin_sectors`, 0.01 electrons); otherwise the run is prepared again on the
    full grid before its first iteration, as ``off`` prepares it.  ``auto`` does so where the points of the full
    grid times the states reach :data:`SPIN_RESTART_SECTOR_WORK` (2^25; a smaller restart is faster on the full
    grid, and is that of ``off``), ``on`` whatever the size.
``PARSEC_WFNDAT_RESTART_SECTORS`` (``auto``|``off``, default ``off``)
    ``Restart_From_Wfndat`` of a cluster: under ``auto``, where the density of ``wfn.dat`` is invariant (the same
    test), the run solves in the sectors from the stored potential and Hartree potential, with a first filter
    instead of the stored orbitals (full-grid states of one representation, which the sectors do not take);
    otherwise, and under ``off``, the full grid seeded with the stored orbitals, as before.  ``auto`` saves
    memory but not time: a converged file converges in one iteration of the seeded full grid.

Every switch is read when the run is prepared, so that a value it does not know is refused before the SCF.
Units bohr and Ry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import gc
import os
from typing import Callable

import numpy as np

from parsec_python.models import Atom

POLARIZABILITY_SWITCH = "PARSEC_POLARIZABILITY_SYMMETRY"
POINT_CHARGE_SWITCH = "PARSEC_POINT_CHARGE_SYMMETRY"
FOLDBLOCK_SWITCH = "PARSEC_FOLDBLOCK_SYMMETRY"
SPIN_RESTART_SWITCH = "PARSEC_SPIN_RESTART_SECTORS"
WFNDAT_RESTART_SWITCH = "PARSEC_WFNDAT_RESTART_SECTORS"

#: The label of the pseudo-atom that marks the axis of a field (no species carries it).
FIELD_AXIS_LABEL = "<electric field axis>"
#: Distance (bohr) of that pseudo-atom from the origin: any operation that moves the axis moves it by 1.4 bohr or more.
_FIELD_AXIS_DISTANCE = 1.0


def _auto_or_off(name: str, default: str = "auto") -> str:
    value = os.environ.get(name, default).strip().lower() or default
    if value not in {"auto", "off"}:
        raise ValueError(f"{name} must be auto or off, not {value!r}")
    return value


def polarizability_symmetry_setting() -> str:
    """``PARSEC_POLARIZABILITY_SYMMETRY``: ``auto`` (default: the sectors where the full grid times the states
    reach :data:`POLARIZABILITY_SECTOR_WORK`), ``on`` (the sectors whatever the size) or ``off`` (the seven loops on
    the full grid)."""

    value = os.environ.get(POLARIZABILITY_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{POLARIZABILITY_SWITCH} must be auto, on or off, not {value!r}")
    return value


#: Points of the full grid times states from which ``PARSEC_POLARIZABILITY_SYMMETRY=auto`` takes the sectors.  On one
#: A100 node (main.py, four GPUs, first runs) the seven loops took, sectors against the full grid: water (2.1e6) 9.2
#: against 6.5 s, benzene (5.4e6) 9.6 against 8.6 s, naphthalene (1.6e7) 12.6 against 13.6 s (benzene and
#: naphthalene: the means of two runs each), C35H36 (5.2e7) 9.2 against 14.3 s, C87H76 (1.6e8) 12.6 against 33.6 s,
#: C185H124 (5.3e8) 20.2 against 96.1 s: every new group is a preparation and a first filter, which the warm loops of
#: one full grid do not pay.  The two D2h molecules (three new groups, the most a molecule of a point group needs)
#: cross near 1.1e7; the threshold lies above that.
POLARIZABILITY_SECTOR_WORK = 1.2e7


def estimated_full_grid_work(problem) -> float | None:
    """Points of the full grid of a cluster (from the volume of its domain over ``h^3``) times its states, or
    ``None`` where the pseudopotentials cannot be read for the electron count."""

    grid = problem.grid
    spacing = float(grid.spacing)
    shape = getattr(grid, "domain_shape", "sphere")
    if shape == "box":
        volume = float(np.prod(np.asarray(grid.box_lengths, dtype=float)))
    elif shape == "ellipsoid":
        a, b, c = (float(value) for value in grid.domain_parameters)
        volume = 4.0 / 3.0 * np.pi * a * b * c
    elif shape == "cylinder":
        radius, length, _ = (float(value) for value in grid.domain_parameters)
        volume = np.pi * radius**2 * length
    else:
        volume = 4.0 / 3.0 * np.pi * float(grid.radius) ** 3
    points = volume / spacing**3
    states = problem.scf.number_of_states
    if states is None:
        try:
            from parsec_python.V_ion import load_pseudopotentials

            potentials = load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
            electrons = sum(float(potentials[atom.symbol].ionic_charge) for atom in problem.atoms)
        except Exception:
            return None
        electrons -= float(getattr(problem.scf, "net_charge", 0.0) or 0.0)
        states = int(np.ceil(0.5 * electrons)) + 6
    return float(points) * float(states)


def polarizability_in_sectors(translation) -> bool:
    """Whether the field loops of ``Polarizability`` take the sectors (``PARSEC_POLARIZABILITY_SYMMETRY``)."""

    setting = polarizability_symmetry_setting()
    if setting != "auto":
        return setting == "on"
    work = estimated_full_grid_work(translation.problem)
    return work is not None and work >= POLARIZABILITY_SECTOR_WORK


def point_charge_symmetry_setting() -> str:
    """``PARSEC_POINT_CHARGE_SYMMETRY``: ``auto`` (default) or ``off`` (point charges on the full grid)."""

    return _auto_or_off(POINT_CHARGE_SWITCH)


def foldblock_symmetry_setting() -> str:
    """``PARSEC_FOLDBLOCK_SYMMETRY``: ``auto`` (default) or ``off`` (fold blocks on the full grid)."""

    return _auto_or_off(FOLDBLOCK_SWITCH)


def spin_restart_sectors_setting() -> str:
    """``PARSEC_SPIN_RESTART_SECTORS``: ``auto`` (default: the sectors where the full grid times the states reach
    :data:`SPIN_RESTART_SECTOR_WORK`), ``on`` (the sectors whatever the size) or ``off`` (a spin restart on the
    full grid)."""

    value = os.environ.get(SPIN_RESTART_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{SPIN_RESTART_SWITCH} must be auto, on or off, not {value!r}")
    return value


#: Points of the full grid times states from which ``PARSEC_SPIN_RESTART_SECTORS=auto`` keeps the sectors of a spin
#: restart.  One A100 node (main.py, four GPUs), first runs, the restart from the parsec.dat of a converged run in the
#: sectors against the full grid: O2 (the test fixture, 4.0e4) 2.9 against 1.8 s; benzene (5.4e6) 5.5 against 4.4 s;
#: naphthalene (1.6e7) 5.5 against 4.9 s (means of two; two iterations, whose first filter costs more in the sectors);
#: C35H36 (5.2e7) 6.5 against 7.3 s; C87H76 (1.6e8) 11.9 against 21.4 s.  They cross near 2.8e7.
SPIN_RESTART_SECTOR_WORK = 2**25


def spin_restart_in_sectors(translation) -> bool:
    """Whether a spin-polarized ``Restart_Run`` of a cluster may keep the sectors (``PARSEC_SPIN_RESTART_SECTORS``):
    ``on``, or ``auto`` where :func:`estimated_full_grid_work` reaches :data:`SPIN_RESTART_SECTOR_WORK`."""

    setting = spin_restart_sectors_setting()
    if setting != "auto":
        return setting == "on"
    work = estimated_full_grid_work(translation.problem)
    return work is not None and work >= SPIN_RESTART_SECTOR_WORK


def wfndat_restart_sectors_setting() -> str:
    """``PARSEC_WFNDAT_RESTART_SECTORS``: ``off`` (default: ``wfn.dat`` seeds the full grid) or ``auto`` (the
    sectors from the stored potential where the stored density keeps the group).

    ``off`` stays the default: the sectors take none of the stored orbitals, so a restart from a converged file
    filters from the start where the full grid converges in one iteration (C87H76 on one A100 node, first runs, each
    measured twice on different nodes: ``auto`` 26.4 and 25.5 s, 8 iterations, 1.0 GiB per GPU; ``off`` 22.3 and 28.1 s, 1 iteration,
    5.8 GiB on its first GPU).
    """

    return _auto_or_off(WFNDAT_RESTART_SWITCH, default="off")


def read_settings() -> dict[str, str]:
    """Every switch of this module, read once so that a value it does not know is refused."""

    return {
        "polarizability": polarizability_symmetry_setting(),
        "point_charges": point_charge_symmetry_setting(),
        "foldblock": foldblock_symmetry_setting(),
        "spin_restart": spin_restart_sectors_setting(),
        "wfndat_restart": wfndat_restart_sectors_setting(),
    }


# --------------------------------------------------------------------------------------------------------------
# What the detection matches besides the atoms
# --------------------------------------------------------------------------------------------------------------


def point_charge_constraints(charges) -> tuple[Atom, ...]:
    """The point charges as pseudo-atoms, one label per charge value (``1e-10`` e apart)."""

    found = []
    for charge in charges:
        value = float(getattr(charge, "charge"))
        found.append(Atom(symbol=f"<point charge {value:.10e}>", position=np.asarray(charge.position, dtype=float)))
    return tuple(found)


def field_axis_constraint(axis: int) -> Atom:
    """The pseudo-atom that only the operations fixing ``e_axis`` keep (``R e_a = e_a``)."""

    position = np.zeros(3, dtype=float)
    position[int(axis)] = _FIELD_AXIS_DISTANCE
    return Atom(symbol=FIELD_AXIS_LABEL, position=position)


def orbits_of_real_atoms(reduction, count: int) -> None:
    """Keep the orbits of the first ``count`` (real) atoms of a reduction detected with pseudo-atoms after them.

    The pseudo-atoms have labels of their own, so the orbits of the real atoms are those the real atoms alone
    have under the group found: the first ``count`` entries.
    """

    representatives = getattr(reduction, "atom_representatives", None)
    sizes = getattr(reduction, "atom_orbit_sizes", None)
    if representatives is None or sizes is None or len(representatives) <= count:
        return
    kept = np.ascontiguousarray(np.asarray(representatives)[:count], dtype=np.int64)
    kept_sizes = np.ascontiguousarray(np.asarray(sizes)[:count], dtype=np.int64)
    if np.any(kept >= count):
        raise RuntimeError("a pseudo-atom of the symmetry detection shares an orbit with an atom")
    kept.setflags(write=False)
    kept_sizes.setflags(write=False)
    object.__setattr__(reduction, "atom_representatives", kept)
    object.__setattr__(reduction, "atom_orbit_sizes", kept_sizes)


# --------------------------------------------------------------------------------------------------------------
# The symmetry mode of the accelerated CLI
# --------------------------------------------------------------------------------------------------------------

#: Whether the switch of each reason of ``ParsecInputTranslation.symmetry_off_reasons`` keeps a group here.
_REASON_SWITCHES = {
    "polarizability": polarizability_in_sectors,
    "point_charges": lambda translation: point_charge_symmetry_setting() == "auto",
    "foldblock": lambda translation: foldblock_symmetry_setting() == "auto",
}


def kept_group_reasons(translation) -> tuple[str, ...]:
    """The reasons of a cluster input that switch PARSEC's group off and whose switch keeps a group here.

    Empty where the input has no such reason, where one of its reasons keeps the full grid (manual moves,
    molecular dynamics, FDET, a switch ``off``), for a periodic input, where the input says ``Ignore_Symmetry``,
    for spinors (self-consistent spin-orbit coupling or non-collinear spin: their solvers take no sectors), for two
    spin channels under ``Spin_Symmetry_Sectors: off`` (both on the full grid whatever the group), and
    for a relaxation whose ``Move_Flag`` fixes an atom that an operation of the atoms exchanges with a movable one
    (the test of :func:`parsec_python.Relaxation.run.symmetry_decision` on the whole group of the atoms, whatever
    ``PARSEC_MOVE_FLAG_SUBGROUP`` keeps of it; the group a constraint keeps, a
    subgroup of that of the atoms, passes wherever the atoms pass it: such a relaxation would leave the group of
    its sectors).  The caller still owns its own rules (the command line, the spinor route of the accelerated CLI).
    """

    reasons = tuple(getattr(translation, "symmetry_off_reasons", ()) or ())
    problem = translation.problem
    if not reasons or problem.periodic_cell is not None:
        return ()
    if translation.ignore_symmetry and "foldblock" not in reasons:
        # Ignore_Symmetry of the input (the fold blocks set it only where the input does not).
        return ()
    scf = getattr(problem, "scf", None)
    if getattr(scf, "self_consistent_spin_orbit", False) or getattr(scf, "noncollinear", False):
        return ()
    if getattr(scf, "spin_polarized", False) and getattr(scf, "spin_symmetry_sector_mode", "auto") == "off":
        # Spin_Symmetry_Sectors: off keeps both channels on the full grid, which a kept group would not change.
        return ()
    for reason in reasons:
        keeps = _REASON_SWITCHES.get(reason)
        if keeps is None or not keeps(translation):
            return ()
    if getattr(translation, "relaxation", None) is not None:
        from parsec_python.Relaxation.run import symmetry_decision

        keeps_group, note = symmetry_decision(replace(translation, use_symmetry=True, ignore_symmetry=False))
        # also where PARSEC_MOVE_FLAG_SUBGROUP=auto keeps the subgroup of the flags (keeps_group
        # with a note): the group a constraint keeps is not combined with that subgroup, so such a relaxation
        # takes the full grid, as this test decided before the subgroup existed
        if not keeps_group or note is not None:
            return ()
    return reasons


def kept_group_warnings(warnings, reasons) -> list[str]:
    """The parser's warnings of a run that keeps the group of :func:`kept_group_reasons` ``reasons``.

    Unchanged unless the fold blocks keep theirs: then the warning of ``Ignore_Symmetry`` (which the fold blocks
    set, and which the accelerated driver does not follow here) is left out, and PARSEC 1.5's warning that the
    fold blocks switch the symmetry off says what the driver keeps.
    """

    warnings = list(warnings)
    if "foldblock" not in tuple(reasons or ()):
        return warnings
    kept = []
    for warning in warnings:
        if warning.startswith("Ignore_Symmetry=true reproduces PARSEC's zero grid shift"):
            continue
        if warning.startswith("Foldblock_Side_Length") and "symmetry is switched off" in warning:
            warning += (
                "; the accelerated driver keeps the operations that map its domain of whole blocks onto itself "
                f"({FOLDBLOCK_SWITCH}={foldblock_symmetry_setting()}, NOTE below)"
            )
        kept.append(warning)
    return kept


def kept_group_note(reasons) -> str:
    """The NOTE line of parsec.out for :func:`kept_group_reasons`."""

    names = {
        "polarizability": f"Polarizability (PARSEC_POLARIZABILITY_SYMMETRY={polarizability_symmetry_setting()}: the "
        "zero field in the group of the atoms, each field axis in the operations that keep it, the loops that are "
        "images of a solved one not solved)",
        "point_charges": f"Add_Point_Charges ({POINT_CHARGE_SWITCH}={point_charge_symmetry_setting()}: the "
        "operations that carry the charges onto charges of their value)",
        "foldblock": f"Foldblock_Side_Length 2 ({FOLDBLOCK_SWITCH}={foldblock_symmetry_setting()}: on the grid of "
        "PARSEC 1.5 without shift, the operations that map the domain of whole blocks onto itself)",
    }
    return (
        "NOTE: PARSEC 1.5 uses no point group here; the accelerated driver keeps the sectors of the group the input "
        "leaves intact: " + "; ".join(names[reason] for reason in reasons)
    )


# --------------------------------------------------------------------------------------------------------------
# Polarizability
# --------------------------------------------------------------------------------------------------------------


def _key(matrix) -> bytes:
    return np.rint(np.asarray(matrix, dtype=np.float64)).astype(np.int8).tobytes()


def system_group(system) -> dict[bytes, np.ndarray]:
    """The signed permutations the SCF of ``system`` keeps (sectors and fields), the identity included."""

    from ..Symmetry.axis_reflection import abelian_sector_operations

    found = {_key(np.eye(3)): np.eye(3, dtype=np.int64)}
    reductions = []
    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reductions.append(getattr(getattr(sectors, "decomposition", None), "reduction", None))
    reductions.append(getattr(getattr(system, "scalar_field_adapter", None), "reduction", None))
    for reduction in reductions:
        if reduction is None or int(getattr(reduction, "group_order", 1)) <= 1:
            continue
        full = getattr(reduction, "full_operations", None)
        operations = full if full is not None else abelian_sector_operations(reduction)
        for operation in np.asarray(operations, dtype=np.int64).reshape(-1, 3, 3):
            found[_key(operation)] = operation
    return found


@dataclass(frozen=True)
class FieldPlan:
    """Which of the six field loops of ``polar.f90`` are solved and which are images of a solved one.

    ``solved`` are indices into :data:`parsec_python.SCF.polarizability.FIELD_DIRECTIONS` (1 to 6) in increasing
    order; ``images[i] = (j, R, rows)``: loop ``i`` is the image of the solved loop ``j > i`` under the signed
    permutation ``R`` (``R d_j = d_i``), ``rows`` the grid rows of the images ``R r`` of the grid points.
    ``group`` is the group ``G`` of the atoms (and point charges) on the lattice, as ``{key: R}``.
    """

    solved: tuple[int, ...]
    images: dict
    group: dict

    def stabilizer(self, axis: int) -> dict[bytes, np.ndarray]:
        """``G_a``: the operations of ``G`` that fix ``e_axis``."""

        unit = np.zeros(3, dtype=np.int64)
        unit[int(axis)] = 1
        return {key: matrix for key, matrix in self.group.items() if np.array_equal(matrix @ unit, unit)}

    @property
    def trivial(self) -> bool:
        return len(self.group) == 1


def field_axis(index: int) -> int:
    """The axis of the field of loop ``index`` (1, 2: x; 3, 4: y; 5, 6: z)."""

    return (int(index) - 1) // 2


def field_plan(grid, atoms, *, atom_tolerance: float, point_charges=()) -> FieldPlan:
    """:class:`FieldPlan` of a cluster on ``grid`` with ``atoms`` (and ``point_charges``)."""

    from parsec_python.SCF.polarizability import FIELD_DIRECTIONS
    from ..Symmetry.axis_reflection import _point_group_operations
    from .spin_sectors import _grid_images

    detection = tuple(atoms) + point_charge_constraints(point_charges)
    operations = _point_group_operations(grid, detection, atom_tolerance, 1.0e-10, None)
    group = {_key(matrix): np.rint(np.asarray(matrix, dtype=np.float64)).astype(np.int64) for matrix in operations}
    directions = [np.rint(np.asarray(direction, dtype=np.float64)).astype(np.int64) for direction in FIELD_DIRECTIONS]
    rows_of: dict[bytes, np.ndarray | None] = {}

    def rows(key, matrix):
        if key not in rows_of:
            rows_of[key] = _grid_images(grid, matrix)
        return rows_of[key]

    solved: list[int] = []
    images: dict[int, tuple] = {}
    for index in range(6, 0, -1):
        found = None
        for source in sorted(solved):
            for key, matrix in group.items():
                if not np.array_equal(matrix @ directions[source], directions[index]):
                    continue
                mapped = rows(key, matrix)
                if mapped is not None:
                    found = (source, matrix, mapped)
                    break
            if found is not None:
                break
        if found is None:
            solved.append(index)
        else:
            images[index] = found
    return FieldPlan(solved=tuple(sorted(solved)), images=images, group=group)


def _potential_warm_start(result):
    """The potentials of ``result`` as the start of a loop on another group: no subspace (a first filter)."""

    from parsec_python.SCF import SCFWarmStart

    channels = getattr(result, "spin", None)
    if channels is not None:
        potential = tuple(np.array(channel.input_effective_potential, dtype=np.float64) for channel in channels)
    else:
        potential = np.array(result.input_effective_potential, dtype=np.float64)
    return SCFWarmStart(
        input_potential=potential,
        hartree_potential=np.array(result.hartree_potential, dtype=np.float64),
        eigensolver_state=None,
    )


def _image_density(density, rows):
    """``rho'(R r) = rho(r)`` on the grid rows ``rows`` of the images ``R r``."""

    density = np.asarray(density, dtype=np.float64)
    image = np.empty_like(density)
    image[rows] = density
    return image


def release_system(system) -> None:
    """What the eigensolver of ``system`` keeps on the devices goes, before another preparation."""

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    if sectors is not None and int(getattr(sectors, "spin_channel_count", 1)) == 2:
        for index in (0, 1):
            sectors.for_spin_channel(index).reset()
    else:
        for solver in (sectors, getattr(system, "eigenproblem_solver", None)):
            reset = getattr(solver, "reset", None)
            if callable(reset):
                reset()
    gc.collect()


def _free_device_pools() -> None:
    try:
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
    except Exception:
        return
    for device in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()


class _FieldKeepingFallback:
    """The spin-sector fallback of one field loop, whose full grid carries the field of the loop.

    :class:`.driver._SpinSectorFallback` prepares the full grid from the input, without the field of the loop
    in the ionic potential of its reference; here that reference takes it as the loop of ``off`` does
    (:func:`parsec_python.SCF.polarizability.field_system` of the prepared ionic potential).  ``plain_reference``
    is the full-grid reference without the field once the fallback happened, else ``None``.
    """

    def __init__(self, inner, field: float, direction) -> None:
        self.inner = inner
        self.field = float(field)
        self.direction = direction
        self.plain_reference = None

    def run(self, system, run, **keywords):
        import copy

        from parsec_python.SCF.polarizability import field_system

        inner = copy.copy(self.inner)
        replace_with_full_grid = inner.replace_with_full_grid

        def with_field(target, stage, report) -> None:
            replace_with_full_grid(target, stage, report)
            plain = target.reference
            self.plain_reference = plain
            target.reference = field_system(
                plain, np.asarray(plain.ionic_potential, dtype=np.float64), self.field, self.direction
            )

        inner.replace_with_full_grid = with_field
        return inner.run(system, run, **keywords)


def replace_system(system, prepared) -> None:
    """Put the objects of the ``prepared`` system into ``system``, which every holder of it then sees."""

    kept = system.materialize_final_wavefunctions
    for item in fields(prepared):
        setattr(system, item.name, getattr(prepared, item.name))
    system.materialize_final_wavefunctions = kept
    gc.collect()
    _free_device_pools()


def symmetric_field_loop(
    system,
    *,
    problem,
    backend,
    symmetry_cache_directory,
    log=None,
    iteration_state_callback: Callable | None = None,
    symmetry: str = "auto",
):
    """The ``field_loop`` of :func:`parsec_python.cli._run_polarizability` under ``PARSEC_POLARIZABILITY_SYMMETRY``
    ``auto`` or ``on`` (:func:`kept_group_reasons`).

    ``system`` is the accelerated system of the zero-field loop, prepared from ``problem`` with the symmetry of
    the atoms; it is prepared again (and replaced in place) with the symmetry mode ``symmetry`` of the run for
    every field axis whose stabilizer differs from the group it holds (see the module).  ``log.write`` takes a
    line per new group.  A run whose mode is ``off`` takes no field loop (its caller passes none).
    """

    if str(symmetry).strip().lower() == "off":
        raise ValueError("the field loops in the sectors need a symmetry mode other than off")

    from parsec_python.Energy.dipole import cluster_dipole_moment
    from parsec_python.SCF.polarizability import (
        FIELD_DIRECTIONS,
        FieldRun,
        PolarizabilityResult,
        field_system,
        result_density,
        run_polarizability,
    )
    from ..Symmetry.axis_reflection import symmetry_atom_tolerance
    from ..driver import prepare_single_point, run_scf
    from .polarizability import field_runner, next_field_warm_start

    keywords = {} if iteration_state_callback is None else {"iteration_state_callback": iteration_state_callback}

    def write(line: str = "") -> None:
        if log is not None:
            log.write(line)

    def loop(reference, field, *, callback=None, before_run=None, after_run=None, **_ignored):
        field = float(field)
        if not np.isfinite(field) or field == 0.0:
            raise ValueError("the polarizability needs a nonzero finite field")
        runner = field_runner(system, iteration_state_callback=iteration_state_callback)
        continued = next_field_warm_start(system)
        plan = field_plan(
            system.grid,
            system.atoms,
            atom_tolerance=symmetry_atom_tolerance(),
            point_charges=getattr(system, "point_charges", ()),
        )
        if plan.trivial:
            # No operation beside the identity: the loops of polar.f90 in order on the system as it is.
            return run_polarizability(
                reference,
                field,
                scf_runner=runner,
                callback=callback,
                before_run=before_run,
                after_run=after_run,
                warm_start_builder=continued,
            )
        if plan.images:
            write(
                " Polarizability: loops "
                + ", ".join(
                    f"{_NAMES[index]} = image of {_NAMES[source]}" for index, (source, _, _) in sorted(plan.images.items())
                )
                + f" under operations of the {len(plan.group)} of the atoms "
                f"({POLARIZABILITY_SWITCH}={polarizability_symmetry_setting()})"
            )
        records: list = [None] * len(FIELD_DIRECTIONS)
        # Whether a loop of two spin channels left the sectors (the system holds the full grid from then on).
        left_sectors = [False]

        def record_of(run_system, result, direction) -> FieldRun:
            return FieldRun(
                direction=direction,
                total_energy=float(result.energies.total),
                dipole=cluster_dipole_moment(
                    result_density(result), run_system.grid, run_system.atoms, run_system.pseudopotentials
                ),
                converged=bool(result.converged),
                iterations=int(result.iterations),
            )

        def solve(index, warm_start):
            direction = FIELD_DIRECTIONS[index]
            if before_run is not None:
                before_run(index, direction)
            base = np.asarray(system.reference.ionic_potential, dtype=np.float64)
            run_system = field_system(system.reference, base, field, direction)
            fallback = getattr(system, "spin_sector_fallback", None)
            if fallback is None:
                result = runner(run_system, callback=callback, warm_start=warm_start)
            else:
                # Two spin channels in the sectors: where the guard of spin_sectors sends the loop to the full grid,
                # that grid carries the field of the loop, and the loops that follow solve there too.
                kept = _FieldKeepingFallback(fallback, field, direction)
                loop_system = replace(system, reference=run_system, spin_sector_fallback=kept)
                result = run_scf(loop_system, callback=callback, warm_start=warm_start, **keywords)
                if kept.plain_reference is not None:
                    run_system = loop_system.reference
                    loop_system.reference = kept.plain_reference
                    replace_system(system, loop_system)
                    left_sectors[0] = True
                    write(
                        f" Polarizability: the loop along {_NAMES[index]} left the spin sectors for the full grid, "
                        "where the loops that follow solve too"
                    )
            record = record_of(run_system, result, direction)
            records[index] = record
            if after_run is not None:
                after_run(index, run_system, result, record)
            # The images of this loop, in the order of polar.f90.
            for image, (source, matrix, rows) in sorted(plan.images.items()):
                if source != index:
                    continue
                image_direction = FIELD_DIRECTIONS[image]
                image_system = field_system(system.reference, base, field, image_direction)
                density = _image_density(result_density(result), rows)
                image_record = FieldRun(
                    direction=image_direction,
                    total_energy=record.total_energy,
                    dipole=cluster_dipole_moment(
                        density, image_system.grid, image_system.atoms, image_system.pseudopotentials
                    ),
                    converged=record.converged,
                    iterations=0,
                )
                records[image] = image_record
                if after_run is not None:
                    after_run(image, None, None, image_record, source=source, operation=matrix)
            return run_system, result

        # The zero field in the group of the atoms.
        run_system, result = solve(0, None)
        zero_start = _potential_warm_start(result)
        warm_start = continued(result)
        held = system_group(system)
        for axis in range(3):
            indices = [index for index in plan.solved if field_axis(index) == axis]
            if not indices:
                continue
            wanted = plan.stabilizer(axis)
            if set(held) != set(wanted) and not left_sectors[0]:
                # Another group: the system of this axis, from the potential of the zero field.
                del result, warm_start
                release_system(system)
                prepared = prepare_single_point(
                    problem,
                    backend=backend,
                    symmetry=symmetry,
                    symmetry_cache_directory=symmetry_cache_directory,
                    symmetry_constraints=(field_axis_constraint(axis),),
                )
                replace_system(system, prepared)
                del prepared
                held = system_group(system)
                detection = dict(getattr(system.backend_info, "details", ())).get("symmetry_detection", "")
                write(
                    f" Polarizability: the fields along {'xyz'[axis]} keep {len(wanted)} of the {len(plan.group)} "
                    f"operations of the atoms; prepared again: {detection}"
                )
                warm_start = zero_start
            for index in indices:
                run_system, result = solve(index, warm_start)
                warm_start = continued(result)
        if any(record is None for record in records):
            raise RuntimeError("a field loop of the polarizability was neither solved nor an image")
        return (
            PolarizabilityResult(field=field, runs=tuple(records), atom_count=len(system.atoms)),
            run_system,
            result,
        )

    return loop


_NAMES = ("0", "+x", "-x", "+y", "-y", "+z", "-z")


# --------------------------------------------------------------------------------------------------------------
# Restarts
# --------------------------------------------------------------------------------------------------------------


def restart_group_images(system):
    """``(matrix, rows)`` of the group a restart's densities are tested against (:func:`.spin_sectors.
    atom_group_images`): the operations of the atoms on the grid that also carry every point charge onto a charge of
    its value (``PARSEC_POINT_CHARGE_SYMMETRY=auto``: the group the system was prepared in, which the guard of the
    spin sectors tests too, ``driver``), on the grid of the system (a fold-block domain keeps its subgroup)."""

    from .spin_sectors import atom_group_images

    atoms = tuple(system.atoms)
    charges = tuple(getattr(system, "point_charges", ()) or ())
    if charges:
        atoms = atoms + point_charge_constraints(charges)
    return atom_group_images(system.grid, atoms)


def restart_density_breaking(system, densities):
    """:class:`.spin_sectors.FractionalOccupations` where ``densities`` (full-grid channels) break the group of
    :func:`restart_group_images` by more than test 2 of :mod:`.spin_sectors` allows, else ``None``."""

    from .spin_sectors import DENSITY_SYMMETRY_TOLERANCE, density_symmetry_breaking

    return density_symmetry_breaking(
        [np.asarray(density, dtype=np.float64) for density in densities],
        system.grid,
        tolerance=DENSITY_SYMMETRY_TOLERANCE,
        images=restart_group_images(system),
    )


def _switch_values(switch: str) -> str:
    """``NAME=value`` of each switch named in ``switch`` (comma separated)."""

    settings = {
        SPIN_RESTART_SWITCH: spin_restart_sectors_setting,
        WFNDAT_RESTART_SWITCH: wfndat_restart_sectors_setting,
    }
    return ", ".join(f"{name}={settings[name]()}" for name in (part.strip() for part in switch.split(",")))


def in_sectors(system) -> bool:
    """Whether the orbitals of ``system`` are solved in symmetry sectors of a group beyond the identity."""

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    return reduction is not None and int(getattr(reduction, "group_order", 1)) > 1


def full_grid_problem(problem, *, spin: bool):
    """``problem`` as the full grid of before prepares it: Spin_Symmetry_Sectors off for a spin restart."""

    if spin:
        return replace(problem, scf=replace(problem.scf, spin_symmetry_sectors="off"))
    return problem


def restart_sector_switch(translation) -> str | None:
    """The switch that keeps the sectors of a cluster's restart (spin ``Restart_Run``, ``Restart_From_Wfndat``),
    or ``None`` where the run has no such restart or a switch says ``off``.

    ``Restart_From_Wfndat`` answers to ``PARSEC_WFNDAT_RESTART_SECTORS`` and, for two spin channels, to
    ``PARSEC_SPIN_RESTART_SECTORS`` as well.
    """

    if not getattr(translation, "restart_run", False) or translation.problem.periodic_cell is not None:
        return None
    spin = bool(translation.problem.scf.spin_polarized)
    if getattr(translation, "restart_from_wfndat", False):
        if wfndat_restart_sectors_setting() != "auto":
            return None
        if spin and not spin_restart_in_sectors(translation):
            return None
        return WFNDAT_RESTART_SWITCH if not spin else f"{WFNDAT_RESTART_SWITCH}, {SPIN_RESTART_SWITCH}"
    if spin and spin_restart_in_sectors(translation):
        return SPIN_RESTART_SWITCH
    return None


def _wfndat_densities(path, system) -> tuple[np.ndarray, ...]:
    """The channel densities of ``wfn.dat`` on the grid of ``system`` (as :func:`parsec_python.SCF.restart.
    load_wfndat_restart` places them, without its subspace)."""

    from parsec_python.Output.parsec_dat import fortran_integer_coordinates
    from parsec_python.Output.wfn_dat import read_wfn_dat
    from parsec_python.SCF.restart import _is_periodic, check_wfn_dat

    grid = system.grid
    spin_count = 2 if getattr(system.input.scf, "spin_polarized", False) else 1
    data = read_wfn_dat(path, periodic=_is_periodic(grid), spin_count=spin_count, complex_values=False)
    check_wfn_dat(data, grid)
    rows = {tuple(int(value) for value in point): row for row, point in enumerate(fortran_integer_coordinates(grid))}
    targets = np.array([rows.get(tuple(int(v) for v in point), -1) for point in data.points])
    present = targets >= 0
    densities = []
    for channel in data.channels:
        values = np.zeros(grid.size, dtype=np.float64)
        values[targets[present]] = np.asarray(channel.density, dtype=np.float64)[present]
        densities.append(values)
    return tuple(densities)


def _parsec_dat_densities(path, system) -> tuple[np.ndarray, ...]:
    """The channel densities of ``parsec.dat`` on the grid of ``system``."""

    from parsec_python.Output.parsec_dat import read_parsec_dat
    from parsec_python.SCF.restart import restart_fields

    spin_count = 2 if getattr(system.input.scf, "spin_polarized", False) else 1
    data = read_parsec_dat(path, wavefunctions=False)
    return restart_fields(data, system.grid, spin_count=spin_count).densities


def restart_in_sectors(
    translation,
    system,
    warm_start,
    directory,
    *,
    backend,
    symmetry,
    symmetry_cache_directory,
    log=None,
    densities=None,
):
    """The warm start of a cluster's restart that :func:`restart_sector_switch` keeps in the sectors.

    Where ``system`` solves in sectors and the stored densities keep the group of the atoms on the grid (test 2 of
    :mod:`.spin_sectors`), the run stays there; the warm start of ``wfn.dat`` loses its stored orbitals (a first
    filter in the sectors).  Otherwise ``system`` is prepared again (in place) as the switches ``off`` prepare it
    -- ``symmetry`` with ``Spin_Symmetry_Sectors: off`` for ``parsec.dat``, the full grid for ``wfn.dat`` -- and
    the warm start is read again for it.  A system without sectors keeps ``warm_start``.  ``densities``: the
    channel densities of the file on the grid of ``system`` where the caller has read them (else they are read).
    """

    from pathlib import Path

    switch = restart_sector_switch(translation)
    if switch is None or warm_start is None or not in_sectors(system):
        return warm_start
    directory = Path(directory)
    wfndat = bool(getattr(translation, "restart_from_wfndat", False))
    spin = bool(translation.problem.scf.spin_polarized)
    source = directory / ("wfn.dat" if wfndat else "parsec.dat")
    if densities is None:
        densities = (_wfndat_densities if wfndat else _parsec_dat_densities)(source, system)
    densities = tuple(np.asarray(density, dtype=np.float64) for density in densities)
    from .spin_sectors import DENSITY_SYMMETRY_TOLERANCE, density_symmetry_breaking

    # The group of the atoms (with the point charges where the system was prepared with them) on the grid.
    images = restart_group_images(system)
    report = density_symmetry_breaking(densities, system.grid, tolerance=DENSITY_SYMMETRY_TOLERANCE, images=images)
    what = ("both channel densities" if spin else "the density") + f" of {source.name}"
    group_name = "the point group of the atoms" + (
        " and the point charges" if getattr(system, "point_charges", ()) else ""
    )

    def write(line: str) -> None:
        if log is not None:
            log.write(line)

    if report is None:
        largest = 0.0
        for _, rows in images:
            for density in densities:
                values = np.asarray(density, dtype=np.float64)
                largest = max(largest, float(system.grid.volume_element * np.sum(np.abs(values[rows] - values))))
        write(
            f"NOTE: Restart_Run: {what} keep {group_name} (largest int |rho(g r) - rho(r)| = "
            f"{largest:.3e} electrons < {DENSITY_SYMMETRY_TOLERANCE:.0e}): the run stays in the symmetry sectors"
            + (", from the stored potential with a first filter (the stored orbitals are full-grid states)" if wfndat else "")
            + f" ({_switch_values(switch)})"
        )
        if wfndat:
            return replace(warm_start, eigensolver_state=None)
        return warm_start
    write(
        f"NOTE: Restart_Run: {what} break {group_name} ({report.measure} = {report.largest:.3e} >= "
        f"{report.tolerance:.0e}): the run is prepared again on the full grid ({_switch_values(switch)})"
    )
    from ..driver import prepare_single_point

    problem = full_grid_problem(translation.problem, spin=spin)
    release_system(system)
    prepared = prepare_single_point(
        problem,
        backend=backend,
        symmetry="off" if wfndat else symmetry,
        symmetry_cache_directory=symmetry_cache_directory,
    )
    replace_system(system, prepared)
    del prepared
    if wfndat:
        from parsec_python.SCF.restart import load_wfndat_restart

        return load_wfndat_restart(source, system)[1]
    from parsec_python.SCF.restart import load_restart

    return load_restart(source, system, complex_wavefunctions=False, spinors=False)[1]


__all__ = [
    "FIELD_AXIS_LABEL",
    "FieldPlan",
    "field_axis",
    "field_axis_constraint",
    "field_plan",
    "foldblock_symmetry_setting",
    "full_grid_problem",
    "in_sectors",
    "kept_group_note",
    "kept_group_reasons",
    "orbits_of_real_atoms",
    "point_charge_constraints",
    "point_charge_symmetry_setting",
    "polarizability_symmetry_setting",
    "read_settings",
    "release_system",
    "replace_system",
    "restart_density_breaking",
    "restart_in_sectors",
    "restart_sector_switch",
    "kept_group_warnings",
    "restart_group_images",
    "spin_restart_in_sectors",
    "spin_restart_sectors_setting",
    "symmetric_field_loop",
    "system_group",
    "wfndat_restart_sectors_setting",
]
