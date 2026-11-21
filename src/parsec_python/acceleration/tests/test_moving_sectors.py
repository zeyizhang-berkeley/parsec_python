"""Molecular dynamics and manual movements in the sectors of the group their structures keep (``PARSEC_MOVING_SECTORS``).

The CPU tests check the plan (which runs keep a group, which operations, the pseudo-atoms of the detection) and the
routing of the command line; the GPU tests compare runs under ``auto`` with those of ``off`` (the full grid of
before): the energies, forces and trajectories of molecular dynamics restarted from a symmetric state, of a manual
path, of a fresh run whose random velocities keep the full grid (bit for bit), of a periodic cell, of a relaxation
with fixed atoms on the Fortran-derived routines, and of a run that leaves the group of its sectors on the way.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.tests.test_move_flag_subgroup import GE5, _cupy_ready, _ge5_input, _run_cli

#: Ten steps of 40 au of the microcanonical integrator from mdinit.dat.
def _dynamics(steps: int = 3, *, restart: bool = True, friction: float | None = None) -> str:
    text = (
        f"Molecular_Dynamics: .true.\nStep_num: {steps}\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\nTime_Step: 40.0\n"
    )
    if restart:
        text += "Restart_mode: .true.\n"
    if friction is not None:
        text += f"Friction_Coefficient: {friction}\n"
    return text


def _mdinit(directory: Path, current: np.ndarray, previous: np.ndarray, accelerations=None, movable=None) -> None:
    from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
    from parsec_python.MolecularDynamics.md_files import format_md_restart
    from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState

    count = current.shape[0]
    state = MolecularDynamicsState(
        masses=np.asarray(atomic_masses(["Ge"] * count), dtype=float),
        movable=np.ones(count, dtype=np.int64) if movable is None else np.asarray(movable, dtype=np.int64),
        current_positions=current,
        previous_positions=previous,
        velocities=np.zeros((count, 3)),
        accelerations=np.zeros((count, 3)) if accelerations is None else accelerations,
        temperature=300.0,
        frame=4,
        move_count=0,
        restarted=True,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")


def _breathing(directory: Path) -> None:
    _mdinit(directory, GE5 * 1.002, GE5 * 1.0015)


def _along_bond(directory: Path) -> None:
    """Atom 2 moving along its bond: C3v about that bond, 6 of the 24 operations of Td."""

    previous = GE5.copy()
    previous[1] = GE5[1] * 0.998
    _mdinit(directory, GE5.copy(), previous)


def _manual(directory: Path, count: int = 3) -> None:
    blocks = []
    for k in range(1, count + 1):
        structure = GE5.copy()
        structure[1] = GE5[1] * (1.0 + 0.01 * k)
        blocks.append("\n".join("  ".join(repr(float(v)) for v in row) for row in structure))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manual.dat").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _plan(directory: Path, extra: str, *, flags=None, environment=None):
    """``plan_moving_sectors`` of the Ge5 input of ``extra`` in ``directory`` (mdinit.dat and manual.dat read)."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.MolecularDynamics.run import restart_structure
    from parsec_python.Relaxation.run import manual_structures
    from parsec_python.acceleration.SCF.moving_sectors import plan_moving_sectors

    path = _ge5_input(directory, flags=flags, extra=extra)
    # (Ge5 is below the work from which auto takes the sectors: on, unless the test says otherwise)
    with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on", **(environment or {})}):
        translation = parse_parsec_input(path)
        translation, restart, _ = restart_structure(translation, directory)
        manual = None
        if translation.relaxation is not None and translation.relaxation.method == "manual":
            manual = manual_structures(translation, directory)
        return plan_moving_sectors(translation, dynamics_restart=restart, manual=manual)


class PlanTests(unittest.TestCase):
    """Which runs keep a group, and which (CPU)."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.count = 0

    def folder(self) -> Path:
        self.count += 1
        return self.root / f"case{self.count}"

    def test_the_switch_is_validated(self) -> None:
        from parsec_python.acceleration.SCF.moving_sectors import moving_sectors_setting

        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": ""}):
            self.assertEqual(moving_sectors_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "OFF"}):
            self.assertEqual(moving_sectors_setting(), "off")
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "On"}):
            self.assertEqual(moving_sectors_setting(), "on")
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_MOVING_SECTORS must be auto, on or off"):
                moving_sectors_setting()

    def test_auto_keeps_a_small_cluster_on_the_full_grid(self) -> None:
        from parsec_python.acceleration.SCF.moving_sectors import MOVING_SECTOR_WORK

        folder = self.folder()
        _breathing(folder)
        plan, reason = _plan(folder, _dynamics(), environment={"PARSEC_MOVING_SECTORS": "auto"})
        self.assertIsNone(plan)
        self.assertIn("below the", reason)
        self.assertEqual(MOVING_SECTOR_WORK, 2**24)
        folder = self.folder()
        _breathing(folder)
        plan, _ = _plan(folder, _dynamics(), environment={"PARSEC_MOVING_SECTORS": "on"})
        self.assertEqual(plan.order, 24)

    def test_random_velocities_and_the_langevin_force_keep_the_full_grid(self) -> None:
        plan, reason = _plan(self.folder(), _dynamics(restart=False))
        self.assertIsNone(plan)
        self.assertIn("initial velocities are random", reason)
        folder = self.folder()
        _breathing(folder)
        plan, reason = _plan(folder, _dynamics(friction=0.001))
        self.assertIsNone(plan)
        self.assertIn("Langevin", reason)

    def test_a_symmetric_restart_keeps_the_group_of_its_state(self) -> None:
        folder = self.folder()
        _breathing(folder)
        plan, reason = _plan(folder, _dynamics())
        self.assertIsNone(reason)
        self.assertEqual((plan.kind, plan.periodic, plan.order, plan.atoms_order), ("molecular_dynamics", False, 24, 24))
        self.assertEqual(plan.constraints, ())
        self.assertIn("(PARSEC_MOVING_SECTORS=", plan.note())
        folder = self.folder()
        _along_bond(folder)
        plan, _ = _plan(folder, _dynamics())
        self.assertEqual((plan.order, plan.atoms_order), (6, 24))
        # the pseudo-atoms of the displacement of the last step (the accelerations of the file are zero)
        self.assertEqual(len(plan.constraints), 5)
        self.assertIn("6 of the 24 operations", plan.note())
        folder = self.folder()
        previous = GE5.copy()
        previous[1] = previous[1] + np.array([0.001, -0.002, 0.0005])
        _mdinit(folder, GE5.copy(), previous)
        plan, reason = _plan(folder, _dynamics())
        self.assertIsNone(plan)
        self.assertIn("no operation of the 24", reason)

    def test_one_movable_atom_starts_at_rest(self) -> None:
        plan, reason = _plan(self.folder(), _dynamics(restart=False), flags=[0, 1, 0, 0, 0])
        self.assertIsNone(reason)
        self.assertEqual(plan.fixed, (True, False, True, True, True))
        # the operations of Td that fix atom 2 and map the fixed atoms onto fixed atoms: C3v
        self.assertEqual((plan.order, plan.atoms_order), (6, 6))

    def test_manual_movements_keep_what_every_structure_keeps(self) -> None:
        folder = self.folder()
        _manual(folder)
        plan, reason = _plan(folder, "Minimization: manual\nMovement_Num: 3\n")
        self.assertIsNone(reason)
        self.assertEqual((plan.kind, plan.order, plan.atoms_order, len(plan.structures)), ("manual", 6, 24, 3))
        self.assertEqual(len(plan.constraints), 15)
        plan, reason = _plan(self.folder(), "Minimization: manual\nMovement_Num: 0\n")
        self.assertEqual((plan.order, plan.constraints), (24, ()))
        plan, reason = _plan(
            self.folder(), "Minimization: manual\nMovement_Num: 0\n", environment={"PARSEC_MOVING_SECTORS": "off"}
        )
        self.assertEqual((plan, reason), (None, None))
        plan, reason = _plan(self.folder(), "Minimization: manual\nMovement_Num: 0\nIgnore_Symmetry: .true.\n")
        self.assertIsNone(plan)
        self.assertIn("Ignore_Symmetry", reason)
        # a single point or a relaxation by an optimizer is none of this
        self.assertEqual(_plan(self.folder(), ""), (None, None))
        self.assertEqual(_plan(self.folder(), "Minimization: bfgs\nMovement_Num: 2\n"), (None, None))

    def test_the_pseudo_atoms_restrict_the_detection_to_the_kept_operations(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.SCF.moving_sectors import vector_constraints
        from parsec_python.acceleration.SCF.symmetry_off_inputs import orbits_of_real_atoms
        from parsec_python.acceleration.Symmetry import detect_point_group_reduction, load_or_detect_reflection_reduction
        from parsec_python.models import Atom, GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=9.0))
        atoms = tuple(Atom("Ge", tuple(row)) for row in GE5)
        displacement = np.zeros_like(GE5)
        displacement[1] = GE5[1] * 0.002
        constraints = vector_constraints(atoms, (displacement,), "dynamics")
        self.assertEqual(len(constraints), 5)
        for case, extra, order in (("atoms", (), 24), ("constrained", constraints, 6)):
            with self.subTest(case=case):
                detected = atoms + tuple(extra)
                origin, _ = load_or_detect_reflection_reduction(grid, detected, cache_directory=None)
                reduction = detect_point_group_reduction(grid, detected, origin)
                full = getattr(reduction, "full_operations", None)
                found = len(full) if full is not None else reduction.group_order
                # Td in the weighted sectors of its subgroup; C3v (non-abelian) at least in those of one mirror
                self.assertGreater(found, 1)
                self.assertEqual(order % found, 0)
                if not extra:
                    self.assertEqual(found, order)
                orbits_of_real_atoms(reduction, len(atoms))
                if getattr(reduction, "atom_representatives", None) is not None:
                    self.assertEqual(len(reduction.atom_representatives), len(atoms))

    def cell(self, extra: str, *, restart: bool = True, positions=None):
        """The translation (and ``mdinit.dat`` state) of the displaced silicon cube with ``extra``."""

        from parsec_python.Input import parse_parsec_input
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.MolecularDynamics.run import restart_structure
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon

        folder = self.folder()
        current = displaced_positions(0.04) if positions is None else positions
        path = write_silicon(folder, extra, positions=current)
        if restart:
            state = MolecularDynamicsState(
                masses=np.asarray(atomic_masses(["Si"] * 8), dtype=float), movable=np.ones(8, dtype=np.int64),
                current_positions=current, previous_positions=displaced_positions(0.039),
                velocities=np.zeros((8, 3)), accelerations=np.zeros((8, 3)), temperature=300.0, frame=2,
                move_count=0, restarted=True,
            )
            (folder / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")
        translation = parse_parsec_input(path)
        if translation.molecular_dynamics is not None:
            translation, state, _ = restart_structure(translation, folder)
        else:
            state = None
        return translation, state

    def test_a_cell_below_the_floor_keeps_the_full_grid_under_auto(self) -> None:
        from parsec_python.acceleration.SCF.moving_sectors import MOVING_CELL_MINIMUM_ORDER, plan_moving_sectors

        translation, state = self.cell(_dynamics().replace("Time_Step: 40.0", "Time_Step: 30.0"))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}):
            plan, reason = plan_moving_sectors(translation, dynamics_restart=state)
            self.assertIsNone(plan)
            self.assertIn("grid points x", reason)
            self.assertIn("below the floor of the sectors", reason)
            self.assertIn("PARSEC_MOVING_SECTORS=on takes them", reason)
            # --symmetry on asks for the sectors: no floor, no smallest group
            plan, reason = plan_moving_sectors(translation, dynamics_restart=state, symmetry_argument="on")
            self.assertIsNone(reason)
            self.assertEqual(plan.minimum_order, 0)
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}):
            plan, reason = plan_moving_sectors(translation, dynamics_restart=state)
            self.assertIsNone(reason)
            self.assertTrue(plan.periodic)
            self.assertEqual(plan.minimum_order, MOVING_CELL_MINIMUM_ORDER)
            self.assertEqual(MOVING_CELL_MINIMUM_ORDER, 4)
            self.assertIn("where they have 4 operations or more", plan.note())
            self.assertIn("the sectors of the 8 operations its detection finds for the cell", plan.note(8))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}):
            plan, reason = plan_moving_sectors(translation, dynamics_restart=state)
            self.assertIsNone(reason)
            self.assertEqual(plan.minimum_order, 0)
            self.assertNotIn("operations or more", plan.note())
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FLOOR": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_SECTOR_FLOOR"):
                plan_moving_sectors(translation, dynamics_restart=state)

    def test_the_fortran_fixed_atoms_are_lifted_from_the_floor(self) -> None:
        from parsec_python.acceleration.SCF.moving_sectors import fortran_fixed_atoms_lifted

        translation, _ = self.cell(
            "Use_Fortran_Routines: .true.\nMove_Flag: 5\nMinimization: bfgs\nMovement_Num: 2\n", restart=False
        )
        for environment, keywords, expected in (
            ({"PARSEC_MOVING_SECTORS": "off", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}, {}, False),
            # the cube (5,832 points x its states) is below the floor of the sectors
            ({"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}, {}, False),
            ({"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}, {}, True),
            ({"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "1000"}, {}, True),
            ({"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}, {"symmetry_argument": "on"}, True),
            ({"PARSEC_MOVING_SECTORS": "on", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}, {}, True),
            # MPI keeps the full grid of before
            ({"PARSEC_MOVING_SECTORS": "on", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}, {"mpi_requested": True}, False),
            ({"PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}, {"mpi_requested": True}, False),
        ):
            with self.subTest(environment=environment, keywords=keywords):
                with mock.patch.dict(os.environ, environment):
                    self.assertIs(fortran_fixed_atoms_lifted(translation, **keywords), expected)
        from parsec_python.acceleration.SCF.moving_sectors import fortran_fixed_minimum_order, fortran_fixed_relaxation

        self.assertTrue(fortran_fixed_relaxation(translation))
        default_routines, _ = self.cell("Move_Flag: 5\nMinimization: bfgs\nMovement_Num: 2\n", restart=False)
        self.assertFalse(fortran_fixed_relaxation(default_routines))
        every_atom, _ = self.cell("Use_Fortran_Routines: .true.\nMinimization: bfgs\nMovement_Num: 2\n", restart=False)
        self.assertFalse(fortran_fixed_relaxation(every_atom))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "auto"}):
            self.assertEqual(fortran_fixed_minimum_order(translation, lifted=True), 4)
            self.assertEqual(fortran_fixed_minimum_order(translation, lifted=False), 0)
            self.assertEqual(fortran_fixed_minimum_order(translation, lifted=True, symmetry_argument="on"), 0)
            self.assertEqual(fortran_fixed_minimum_order(default_routines, lifted=True), 0)
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on"}):
            self.assertEqual(fortran_fixed_minimum_order(translation, lifted=True), 0)

    def test_the_driver_drops_sectors_below_the_minimum_order(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.driver import _below_minimum_sector_order, _minimum_sector_order_text

        two, four = SimpleNamespace(group_order=2), SimpleNamespace(group_order=4)
        self.assertTrue(_below_minimum_sector_order(two, 4))
        self.assertFalse(_below_minimum_sector_order(four, 4))
        self.assertFalse(_below_minimum_sector_order(two, 0))
        self.assertFalse(_below_minimum_sector_order(two, 1))
        self.assertFalse(_below_minimum_sector_order(None, 4))
        self.assertFalse(_below_minimum_sector_order(SimpleNamespace(group_order=1), 4))
        text = _minimum_sector_order_text(two, 4, "found it")
        self.assertTrue(text.startswith("skipped: the sectors of 2 operations"))
        self.assertIn("fewer than the 4", text)
        self.assertIn("PARSEC_MOVING_SECTORS=auto", text)
        self.assertIn("found: found it", text)

    def test_kept_operations(self) -> None:
        from parsec_python.Forces import atom_symmetry_operations
        from parsec_python.acceleration.SCF.moving_sectors import keeping_operations, structure_operations
        from parsec_python.models import Atom

        atoms = tuple(Atom("Ge", tuple(row)) for row in GE5)
        group = atom_symmetry_operations(atoms)
        self.assertEqual(len(group), 24)
        self.assertEqual(len(keeping_operations(group, (GE5 * 0.01,), 1e-6)), 24)
        moved = GE5.copy()
        moved[1] *= 1.01
        self.assertEqual(len(structure_operations(group, atoms, moved, 1e-7)), 6)
        moved[2] *= 1.01
        self.assertEqual(len(structure_operations(group, atoms, moved, 1e-7)), 4)
        moved[3] += (0.0, 0.0, 0.01)
        self.assertEqual(len(structure_operations(group, atoms, moved, 1e-7)), 1)


def _stand_in_system(atoms, reduction, grid):
    """What :mod:`.moving_sectors` reads of a prepared cluster: its sectors, atoms, grid and ionic potential."""

    from types import SimpleNamespace

    decomposition = SimpleNamespace(reduction=reduction)
    return SimpleNamespace(
        backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=decomposition)),
        backend_info=SimpleNamespace(details=(("symmetry_detection", "a stand-in"),)),
        input=SimpleNamespace(periodic_cell=None),
        atoms=tuple(atoms),
        grid=grid,
        ionic_potential=np.zeros(grid.size),
    )


class TrackerTests(unittest.TestCase):
    """:class:`MovingSectors` on a stand-in of a Ge5 cluster in the sectors of Td (CPU): the check of every step, the
    average of the state, the forces over the group, a structure that leaves the group prepared again."""

    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Symmetry import (
            detect_point_group_reduction,
            load_or_detect_reflection_reduction,
        )
        from parsec_python.models import Atom, GridSettings

        cls.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=9.0))
        cls.atoms = tuple(Atom("Ge", tuple(row)) for row in GE5)
        origin, _ = load_or_detect_reflection_reduction(cls.grid, cls.atoms, cache_directory=None)
        cls.reduction = detect_point_group_reduction(cls.grid, cls.atoms, origin)

    def tracker(self, plan_kind="molecular_dynamics", prepared=None):
        from parsec_python.acceleration.SCF.moving_sectors import MovingPlan, MovingSectors
        from parsec_python.models import SinglePointInput

        calls = []

        def prepare(problem, *, symmetry, constraints, fixed):
            calls.append((symmetry, len(constraints), fixed))
            return prepared

        plan = MovingPlan(kind=plan_kind, periodic=False, order=24, atoms_order=24)
        problem = SimpleNamespaceProblem(atoms=self.atoms)
        lines = []
        tracker = MovingSectors(plan, problem, prepare=prepare, write=lines.append, time_step=40.0)
        return tracker, calls, lines

    def step(self, forces):
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState, moldyn_step
        from parsec_python.MolecularDynamics.settings import MolecularDynamicsSettings

        settings = MolecularDynamicsSettings(
            enabled=True, restart=True, initial_temperature=300.0, final_temperature=300.0, temperature_step=0.0,
            time_step=40.0, step_num=3,
        )
        state = MolecularDynamicsState(
            masses=np.asarray(atomic_masses(["Ge"] * 5), dtype=float), movable=np.ones(5, dtype=np.int64),
            current_positions=GE5 * 1.002, previous_positions=GE5 * 1.0015, velocities=np.zeros((5, 3)),
            accelerations=np.zeros((5, 3)), temperature=300.0, frame=4, move_count=0, restarted=True,
        )
        return moldyn_step(settings, state, forces, -10.0)

    def test_a_symmetric_step_stays_in_the_sectors_with_its_state_averaged(self) -> None:
        tracker, calls, lines = self.tracker()
        system = _stand_in_system(self.atoms, self.reduction, self.grid)
        tracker._system = system
        forces = -GE5 * 1e-3
        step = self.step(forces)
        kept = tracker.after_step(step)
        self.assertIsNone(tracker._broken)
        np.testing.assert_allclose(kept.positions, step.positions, rtol=0, atol=1e-13)
        np.testing.assert_allclose(kept.state.accelerations, step.state.accelerations, rtol=0, atol=1e-18)
        self.assertIs(kept.positions, kept.state.current_positions)
        moved = []

        class Inner:
            @staticmethod
            def update_geometry(system, atoms):
                moved.append(atoms)
                return system

        update = tracker._update(Inner.update_geometry)
        from dataclasses import replace

        atoms = tuple(replace(atom, position=tuple(row)) for atom, row in zip(self.atoms, kept.positions))
        self.assertIs(update(system, atoms), system)
        self.assertEqual(len(moved), 1)
        self.assertEqual((calls, lines), ([], []))

    def test_a_step_that_leaves_the_group_is_prepared_again(self) -> None:
        from dataclasses import replace

        prepared = _stand_in_system(self.atoms, None, self.grid)
        prepared = type(prepared)(**{**vars(prepared), "ionic_potential": np.ones(self.grid.size)})
        tracker, calls, lines = self.tracker(prepared=prepared)
        system = _stand_in_system(self.atoms, self.reduction, self.grid)
        tracker._system = system
        forces = -GE5 * 1e-3
        forces[1] += (2e-3, -1e-3, 0.0)
        step = self.step(forces)
        self.assertIs(tracker.after_step(step), step)
        self.assertIsNotNone(tracker._broken)
        atoms = tuple(replace(atom, position=tuple(row)) for atom, row in zip(self.atoms, step.positions))
        released = []
        with mock.patch(
            "parsec_python.acceleration.SCF.symmetry_off_inputs.release_system", side_effect=released.append
        ), mock.patch(
            "parsec_python.acceleration.SCF.symmetry_off_inputs.replace_system",
            side_effect=lambda target, source: target.__dict__.update(vars(source)),
        ):
            update = tracker._update(lambda system, atoms: self.fail("the sectors kept a broken structure"))
            result = update(system, atoms)
        self.assertIs(result, system)
        self.assertEqual(released, [system])
        # the moved atoms keep no operation beside the identity: the full grid, prepared once
        self.assertEqual(calls, [("off", 0, None)])
        self.assertTrue(tracker.full_grid)
        self.assertIn("leaves the group of the sectors", lines[-1])
        self.assertIn("on the full grid", lines[-1])
        # the warm start of the structure prepared again: the potential with the ionic potentials exchanged
        from types import SimpleNamespace

        previous = SimpleNamespace(
            next_effective_potential=np.full(self.grid.size, 2.0),
            hartree_potential=np.full(self.grid.size, 3.0),
        )
        warm = tracker._warm(lambda *arguments: self.fail("the warm start of the sectors was asked"))
        start = warm(system, previous, system)
        np.testing.assert_array_equal(start.input_potential, np.full(self.grid.size, 3.0))
        np.testing.assert_array_equal(start.hartree_potential, np.full(self.grid.size, 3.0))
        self.assertIsNone(start.eigensolver_state)

    def test_the_forces_are_averaged_over_the_group_of_the_sectors(self) -> None:
        tracker, _, _ = self.tracker("manual")
        system = _stand_in_system(self.atoms, self.reduction, self.grid)
        seen = {}

        def inner(system, result, *, symmetrize=False, operations=None):
            seen.update(symmetrize=symmetrize, operations=operations)
            return "forces"

        self.assertEqual(tracker._forces(inner)(system, None, symmetrize=False, operations=None), "forces")
        self.assertTrue(seen["symmetrize"])
        self.assertEqual(len(seen["operations"]), 24)
        full = _stand_in_system(self.atoms, None, self.grid)
        tracker._forces(inner)(full, None, symmetrize=False, operations=None)
        self.assertEqual((seen["symmetrize"], seen["operations"]), (False, None))


@dataclass(frozen=True)
class SimpleNamespaceProblem:
    """A problem with atoms that ``dataclasses.replace`` can copy (the re-planned structure)."""

    atoms: tuple
    recenter_geometry: bool = True


class _Prepared(BaseException):
    """Raised by the stand-in of ``prepare_single_point``."""


class RoutingTests(unittest.TestCase):
    """What the command line prepares (the preparation itself is not run; CPU)."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.count = 0

    def keywords(self, path: Path, environment=None, options=()):
        from parsec_python.acceleration import cli as cli_module

        seen = {}

        def prepare(problem, **given):
            seen.update(given)
            raise _Prepared

        with mock.patch.dict(os.environ, environment or {}), mock.patch.object(
            cli_module, "prepare_single_point", side_effect=prepare
        ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            try:
                cli_module.main([str(path), "--quiet", "--no-archive", "--device", "cpu", *options])
            except _Prepared:
                return dict(seen)
        return None

    def test_a_cluster_takes_the_sectors_with_its_constraints(self) -> None:
        self.count += 1
        folder = self.root / f"case{self.count}"
        _along_bond(folder)
        path = _ge5_input(folder, extra=_dynamics())
        # Ge5 is below the work from which auto takes the sectors
        small = self.keywords(path)
        self.assertEqual(small["symmetry"], "off")
        self.assertNotIn("symmetry_constraints", small)
        auto = self.keywords(path, {"PARSEC_MOVING_SECTORS": "on"})
        self.assertEqual(auto["symmetry"], "auto")
        self.assertEqual(len(auto["symmetry_constraints"]), 5)
        off = self.keywords(path, {"PARSEC_MOVING_SECTORS": "off"})
        self.assertEqual(off["symmetry"], "off")
        self.assertNotIn("symmetry_constraints", off)
        self.assertEqual(
            self.keywords(path, {"PARSEC_MOVING_SECTORS": "on"}, options=("--symmetry", "off"))["symmetry"], "off"
        )
        # a fresh run of random velocities: the full grid, as before
        fresh = self.keywords(
            _ge5_input(self.root / "fresh", extra=_dynamics(restart=False)), {"PARSEC_MOVING_SECTORS": "on"}
        )
        self.assertEqual(fresh["symmetry"], "off")
        self.assertNotIn("symmetry_constraints", fresh)

    def test_a_cell_takes_the_sectors_its_detection_finds(self) -> None:
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import (
            BFGS,
            MANUAL,
            displaced_positions,
            write_silicon,
        )
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState

        folder = self.root / "cell"
        current, previous = displaced_positions(0.04), displaced_positions(0.039)
        path = write_silicon(folder, _dynamics(), positions=current)
        state = MolecularDynamicsState(
            masses=np.asarray(atomic_masses(["Si"] * 8), dtype=float), movable=np.ones(8, dtype=np.int64),
            current_positions=current, previous_positions=previous, velocities=np.zeros((8, 3)),
            accelerations=np.zeros((8, 3)), temperature=300.0, frame=2, move_count=0, restarted=True,
        )
        (folder / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")
        # (the test package sets PARSEC_PERIODIC_SECTOR_FLOOR=off: the cube takes the sectors)
        taken = self.keywords(path)
        self.assertEqual(taken["symmetry"], "auto")
        # sectors of 4 operations or more, under auto
        self.assertEqual(taken["minimum_sector_order"], 4)
        self.assertNotIn("minimum_sector_order", self.keywords(path, {"PARSEC_MOVING_SECTORS": "on"}))
        self.assertNotIn("minimum_sector_order", self.keywords(path, options=("--symmetry", "on")))
        # the floor of the sectors of an SCF applies (PARSEC_PERIODIC_SECTOR_FLOOR), but under on
        self.assertFalse(taken["sector_forces"])
        self.assertTrue(self.keywords(path, {"PARSEC_MOVING_SECTORS": "on"})["sector_forces"])
        self.assertEqual(self.keywords(path, {"PARSEC_MOVING_SECTORS": "off"})["symmetry"], "off")
        manual = write_silicon(self.root / "manual", MANUAL)
        self.assertEqual(self.keywords(manual)["symmetry"], "auto")
        self.assertEqual(self.keywords(manual, {"PARSEC_MOVING_SECTORS": "off"})["symmetry"], "off")
        # a relaxation with fixed atoms on the Fortran-derived routines: the sectors of the subgroup under auto
        fortran = write_silicon(
            self.root / "fortran", "Use_Fortran_Routines: .true.\n" + BFGS.replace("Minimization", "Move_Flag: 5\nMinimization")
        )
        kept = self.keywords(fortran)
        self.assertEqual(kept["symmetry"], "auto")
        self.assertEqual(kept["symmetry_fixed_atoms"], (False,) * 5 + (True,) * 3)
        # (sectors of 4 operations or more under auto; any group under on)
        self.assertEqual(kept["minimum_sector_order"], 4)
        self.assertNotIn("minimum_sector_order", self.keywords(fortran, {"PARSEC_MOVING_SECTORS": "on"}))
        self.assertEqual(self.keywords(fortran, {"PARSEC_MOVING_SECTORS": "off"})["symmetry"], "off")
        # with the default floor the cube keeps the full grid of before under auto (the same keywords as off), and
        # the sectors under on
        floor = {"PARSEC_PERIODIC_SECTOR_FLOOR": "auto"}
        for case in (path, manual, fortran):
            with self.subTest(case=case.parent.name):
                default = self.keywords(case, floor)
                self.assertEqual(default, self.keywords(case, {**floor, "PARSEC_MOVING_SECTORS": "off"}))
                self.assertEqual(default["symmetry"], "off")
                self.assertEqual(self.keywords(case, {**floor, "PARSEC_MOVING_SECTORS": "on"})["symmetry"], "auto")


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class DeviceTests(unittest.TestCase):
    """Runs under ``on`` (Ge5 is below the work from which ``auto`` takes the sectors) against ``off`` on the GPU;
    ``runs["auto"]`` holds the run in the sectors."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def pair(self, name: str, extra: str, prepare=None, *, flags=None, patches=()):
        runs = {}
        for mode in ("off", "on"):
            key = "auto" if mode == "on" else mode
            folder = self.root / name / mode
            if prepare is not None:
                prepare(folder)
            path = _ge5_input(folder, flags=flags, extra=extra)
            with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": mode}):
                if mode == "on" and patches:
                    with patches[0]:
                        runs[key] = _run_cli(path, {})
                else:
                    runs[key] = _run_cli(path, {})
            status, _, log, errors = runs[key]
            self.assertIn(status, (0, 4), f"{name} {mode}: {errors}\n{log[-3000:]}")
        return runs

    def assert_dynamics_match(self, runs, *, positions=1e-6, energies=1e-7) -> None:
        off, auto = runs["off"][1], runs["auto"][1]
        np.testing.assert_allclose(auto["dynamics_energies"], off["dynamics_energies"], rtol=0, atol=energies)
        np.testing.assert_allclose(auto["dynamics_positions"], off["dynamics_positions"], rtol=0, atol=positions)
        np.testing.assert_allclose(auto["dynamics_next_positions"], off["dynamics_next_positions"], rtol=0, atol=positions)
        np.testing.assert_allclose(auto["forces_total"], off["forces_total"], rtol=0, atol=1e-5)
        # the atoms moved
        self.assertGreater(float(np.max(np.abs(off["dynamics_next_positions"] - off["dynamics_positions"][0]))), 1e-4)

    def test_a_symmetric_restart_follows_the_trajectory_of_the_full_grid(self) -> None:
        runs = self.pair("breathing", _dynamics(), _breathing)
        self.assert_dynamics_match(runs)
        self.assertIn("PARSEC_MOVING_SECTORS=on", runs["auto"][2])
        self.assertNotIn("PARSEC_MOVING_SECTORS", runs["off"][2])
        self.assertNotIn("leaves the group", runs["auto"][2])

    def test_a_restart_that_keeps_a_subgroup_takes_its_sectors(self) -> None:
        runs = self.pair("bond", _dynamics(), _along_bond)
        self.assert_dynamics_match(runs)
        self.assertIn("6 of the 24 operations", runs["auto"][2])
        self.assertNotIn("leaves the group", runs["auto"][2])

    def test_a_manual_path_has_the_energies_and_forces_of_the_full_grid(self) -> None:
        runs = self.pair("manual", "Minimization: manual\nMovement_Num: 3\n", _manual)
        off, auto = runs["off"][1], runs["auto"][1]
        np.testing.assert_allclose(auto["relaxation_energies"], off["relaxation_energies"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(auto["relaxation_forces"], off["relaxation_forces"], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(auto["relaxation_positions"], off["relaxation_positions"])
        self.assertIn("6 of the 24 operations", runs["auto"][2])

    def test_random_velocities_keep_the_full_grid_bit_for_bit(self) -> None:
        runs = self.pair("fresh", _dynamics(2, restart=False) + "Rng_Seed: 7\n")
        off, auto = runs["off"][1], runs["auto"][1]
        for key in ("dynamics_positions", "dynamics_energies", "dynamics_next_positions", "forces_total"):
            np.testing.assert_array_equal(auto[key], off[key], err_msg=key)
        self.assertNotIn("PARSEC_MOVING_SECTORS", runs["auto"][2])

    def test_a_structure_that_leaves_the_group_is_prepared_again(self) -> None:
        from parsec_python.acceleration.SCF import moving_sectors

        original = moving_sectors.check_structure
        calls = {"count": 0}

        def failing(system, atoms):
            # the third structure (second move) leaves the group, as far as the run knows
            calls["count"] += 1
            if calls["count"] == 3:
                raise ValueError("a test that says so")
            return original(system, atoms)

        runs = self.pair(
            "replan", _dynamics(4), _along_bond, patches=(mock.patch.object(moving_sectors, "check_structure", failing),)
        )
        self.assertIn("leaves the group of the sectors (a test that says so); prepared again", runs["auto"][2])
        self.assert_dynamics_match(runs, energies=2e-7)


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class CellDeviceTests(unittest.TestCase):
    """The displaced silicon cube (Td about the origin): molecular dynamics restarted from a breathing state, and a
    relaxation with fixed atoms on the Fortran-derived routines."""

    def run_pair(self, name, extra, prepare=None, positions=None, modes=("off", "auto")):
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import _run

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            for mode in modes:
                directory = Path(folder) / name / mode
                if prepare is not None:
                    prepare(directory)
                # (the cube is below the floor of the sectors of an SCF, which the moving runs take too)
                with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": mode, "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}):
                    runs[mode] = _run(directory, extra, "auto", positions=positions)
                status, _, log, errors = runs[mode]
                self.assertIn(status, (0, 4), f"{name} {mode}: {errors}\n{log[-3000:]}")
        return runs

    def test_dynamics_of_the_cell_in_its_sectors(self) -> None:
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions

        current, previous = displaced_positions(0.04), displaced_positions(0.039)

        def prepare(directory):
            directory.mkdir(parents=True, exist_ok=True)
            state = MolecularDynamicsState(
                masses=np.asarray(atomic_masses(["Si"] * 8), dtype=float), movable=np.ones(8, dtype=np.int64),
                current_positions=current, previous_positions=previous, velocities=np.zeros((8, 3)),
                accelerations=np.zeros((8, 3)), temperature=300.0, frame=2, move_count=0, restarted=True,
            )
            (directory / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")

        runs = self.run_pair("md", _dynamics(3).replace("Time_Step: 40.0", "Time_Step: 30.0"), prepare, current)
        off, auto = runs["off"][1], runs["auto"][1]
        self.assertIn("Forces of the periodic cell in the sectors", runs["auto"][2])
        self.assertNotIn("Forces of the periodic cell in the sectors", runs["off"][2])
        # the NOTE of the plan, written once the sectors were found, with the operations of their group
        self.assertRegex(runs["auto"][2], r"keeps the sectors of the \d+ operations its detection finds for the cell")
        self.assertNotIn("has no symmetry sectors", runs["auto"][2])
        np.testing.assert_allclose(auto["dynamics_energies"], off["dynamics_energies"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(auto["dynamics_positions"], off["dynamics_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(auto["dynamics_next_positions"], off["dynamics_next_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(auto["forces_total"], off["forces_total"], rtol=0, atol=1e-5)

    def test_a_restart_state_that_breaks_the_group_of_the_cell_takes_the_full_grid(self) -> None:
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions

        current = displaced_positions(0.04)
        previous = current.copy()
        previous[2] += (0.002, -0.001, 0.0005)

        def prepare(directory):
            directory.mkdir(parents=True, exist_ok=True)
            state = MolecularDynamicsState(
                masses=np.asarray(atomic_masses(["Si"] * 8), dtype=float), movable=np.ones(8, dtype=np.int64),
                current_positions=current, previous_positions=previous, velocities=np.zeros((8, 3)),
                accelerations=np.zeros((8, 3)), temperature=300.0, frame=2, move_count=0, restarted=True,
            )
            (directory / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")

        runs = self.run_pair("broken", _dynamics(2).replace("Time_Step: 40.0", "Time_Step: 30.0"), prepare, current)
        self.assertIn("the run is prepared again on the full grid", runs["auto"][2])
        off, auto = runs["off"][1], runs["auto"][1]
        for key in ("dynamics_energies", "dynamics_positions", "dynamics_next_positions", "forces_total"):
            np.testing.assert_allclose(auto[key], off[key], rtol=0, atol=1e-10, err_msg=key)

    def test_sectors_of_two_operations_keep_the_full_grid_under_auto(self) -> None:
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions

        # fresh molecular dynamics of the one movable atom on the bond of the atom at the origin (listed first):
        # C3v about the bond, whose sectors are the 2 of a mirror
        positions = displaced_positions(0.04)[[4, 0, 1, 2, 3, 5, 6, 7]]
        runs = self.run_pair(
            "mirror",
            "Move_Flag: 1\n" + _dynamics(3, restart=False).replace("Time_Step: 40.0", "Time_Step: 30.0"),
            positions=positions,
            modes=("off", "auto", "on"),
        )
        off, auto, on = runs["off"][1], runs["auto"][1], runs["on"][1]
        self.assertIn("fewer than the 4 from which PARSEC_MOVING_SECTORS=auto takes them", runs["auto"][2])
        self.assertIn("has no symmetry sectors", runs["auto"][2])
        self.assertNotIn("Forces of the periodic cell in the sectors", runs["auto"][2])
        for key in ("dynamics_energies", "dynamics_positions", "dynamics_next_positions", "forces_total"):
            np.testing.assert_array_equal(auto[key], off[key], err_msg=key)
        # on keeps the two sectors, with the trajectory of the full grid
        self.assertIn("Forces of the periodic cell in the sectors", runs["on"][2])
        np.testing.assert_allclose(on["dynamics_energies"], off["dynamics_energies"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(on["dynamics_positions"], off["dynamics_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(on["dynamics_next_positions"], off["dynamics_next_positions"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(on["forces_total"], off["forces_total"], rtol=0, atol=1e-5)
        moved = np.asarray(off["dynamics_next_positions"]) - np.asarray(off["dynamics_positions"][0])
        self.assertGreater(float(np.max(np.abs(moved))), 1e-6)

    def assert_relaxations_match(self, sectors, off) -> None:
        np.testing.assert_allclose(sectors["relaxation_energies"], off["relaxation_energies"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(sectors["relaxation_forces"], off["relaxation_forces"], rtol=0, atol=1e-5)
        # (modulo the cell: the Fortran-derived routines fold the atoms into it, and the atom at the origin of the full
        # grid moves by round-off to the far side)
        difference = np.asarray(sectors["relaxation_positions"]) - np.asarray(off["relaxation_positions"])
        difference -= 10.26 * np.round(difference / 10.26)
        self.assertLess(float(np.max(np.abs(difference))), 1e-6)
        # the movable neighbours of the atom at the origin moved
        moved = np.asarray(off["relaxation_positions"][-1]) - np.asarray(off["relaxation_positions"][0])
        moved -= 10.26 * np.round(moved / 10.26)
        self.assertGreater(float(np.max(np.abs(moved))), 1e-4)

    def test_fixed_atoms_on_the_fortran_routines_relax_in_the_sectors_of_their_subgroup(self) -> None:
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions

        # the neighbours of the atom at the origin along +x+y+z and +x-y-z fixed (listed last): C2v, the 4 commuting
        # operations of the sectors
        runs = self.run_pair(
            "fortran",
            "Use_Fortran_Routines: .true.\nMove_Flag: 6\nMinimization: bfgs\nMovement_Num: 2\nForce_Min: 1.0e-5\n",
            positions=displaced_positions(0.04)[[0, 1, 2, 3, 6, 7, 4, 5]],
        )
        off, auto = runs["off"][1], runs["auto"][1]
        self.assertIn("Fortran-derived routines in the sectors", runs["auto"][2])
        self.assertIn("an Abelian group of 4 operations", runs["auto"][2])
        self.assertNotIn("Fortran-derived routines in the sectors", runs["off"][2])
        self.assert_relaxations_match(auto, off)

    def test_a_mirror_subgroup_of_the_fortran_routines_keeps_their_full_grid_under_auto(self) -> None:
        # three neighbours of the atom at the origin fixed: C3v about the bond of the fourth, whose sectors are the 2
        # of a mirror: the full grid of those routines under auto, bit for bit as off; the sectors under on
        runs = self.run_pair(
            "mirror",
            "Use_Fortran_Routines: .true.\nMove_Flag: 5\nMinimization: bfgs\nMovement_Num: 2\nForce_Min: 1.0e-5\n",
            modes=("off", "auto", "on"),
        )
        off, auto, on = runs["off"][1], runs["auto"][1], runs["on"][1]
        self.assertIn("fewer than the 4 from which", runs["auto"][2])
        self.assertNotIn("Fortran-derived routines in the sectors", runs["auto"][2])
        for key in ("relaxation_energies", "relaxation_forces", "relaxation_positions"):
            np.testing.assert_array_equal(auto[key], off[key], err_msg=key)
        self.assertIn("Fortran-derived routines in the sectors", runs["on"][2])
        self.assert_relaxations_match(on, off)


if __name__ == "__main__":
    unittest.main()
