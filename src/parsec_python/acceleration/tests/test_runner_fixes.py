"""Fixes of round 7 (2026-11-19) that change no result: refusals moved before the SCF, the restart of the MPI runner.

* The full-SCF runner reads ``Restart_Run`` as main.py does (``_restart_rules``, ``_load_restart_start``): PARSEC
  error 371 without ``parsec.dat``, ``Restart_From_Wfndat`` refused, the spin restart on the full grid (refused on
  the ranks, the rule of main.py under ``--serial-control``).
* One sector shared by ranks that hold several GPUs each (one rank per node) is refused with advice
  (``experimental/mpi_scf.single_sector_layout_refusal``, ``PARSEC_SINGLE_SECTOR_NODE_LAYOUT``).
* A spin cell that would reach the runner without a sector group is refused with the reason of a spin cell
  (``driver.spin_cell_group_refusal``).
* A group that ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` took beyond the 1e-6 bohr of the forces is refused before the SCF of
  a run with forces (``Forces/sector_symmetry.sector_force_tolerance_refusal``), and the note of the lost operations
  no longer recommends the tolerance to such runs.
* The switches of the local forces of a cell are read at the preparation (``validate_local_force_settings``).

CPU tests: the decisions are made from the input, the atoms and the group, before anything runs on a device.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.models import Atom

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


class _Comm:
    """A communicator of ``size`` ranks seen from ``rank``: every rank holds the same values."""

    def __init__(self, rank: int = 0, size: int = 1):
        self.rank, self.size = rank, size

    def bcast(self, value, root=0):
        return value

    def allgather(self, value):
        return [value] * self.size


def _restart_input(directory: Path, *, spin: bool = False, wfndat: bool = False) -> Path:
    text = (DATA / "H_cli_smoke.in").read_text()
    text += "\nRestart_Run: .true.\n"
    if wfndat:
        text += "Restart_From_Wfndat: .true.\n"
    if spin:
        text += "Spin_Polarization: .true.\n"
    path = directory / "parsec.in"
    path.write_text(text)
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    return path


class RunnerRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        from parsec_python.Input import parse_parsec_input

        self.runner = mpi_full_scf
        self.parse = parse_parsec_input
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_without_restart_nothing_changes(self) -> None:
        path = _restart_input(self.directory)
        path.write_text(path.read_text().replace("Restart_Run: .true.", ""))
        translation = self.parse(path)
        self.assertFalse(translation.restart_run)
        same, notes = self.runner._restart_rules(translation, _Comm())
        self.assertIs(same, translation)
        self.assertEqual(notes, [])
        self.assertIsNone(self.runner._load_restart_start(translation, None, None))

    def test_error_371_without_parsec_dat(self) -> None:
        translation = self.parse(_restart_input(self.directory))
        self.assertTrue(translation.restart_run)
        for size in (1, 4):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "PARSEC error 371"):
                self.runner._restart_rules(translation, _Comm(0, size))
        (self.directory / "parsec.dat").write_bytes(b"")
        same, notes = self.runner._restart_rules(translation, _Comm(0, 4))
        self.assertIs(same, translation)
        self.assertEqual(notes, [])

    def test_the_legacy_wfn_dat_is_refused_by_name(self) -> None:
        translation = self.parse(_restart_input(self.directory, wfndat=True))
        self.assertTrue(translation.restart_from_wfndat)
        with self.assertRaisesRegex(ValueError, "Restart_From_Wfndat is not available in the full-SCF runner"):
            self.runner._restart_rules(translation, _Comm())

    def test_a_spin_restart_takes_the_full_grid_or_is_refused(self) -> None:
        translation = self.parse(_restart_input(self.directory, spin=True))
        (self.directory / "parsec.dat").write_bytes(b"")
        self.assertEqual(translation.problem.scf.spin_symmetry_sector_mode, "auto")
        with self.assertRaisesRegex(ValueError, "spin-polarized calculation solves both channels on the full grid"):
            self.runner._restart_rules(translation, _Comm(0, 4))
        with self.assertRaisesRegex(ValueError, "--serial-control"):
            self.runner._restart_rules(translation, _Comm(0, 1))
        # --serial-control: the rule of main.py (Spin_Symmetry_Sectors: off for the restart)
        ruled, notes = self.runner._restart_rules(translation, _Comm(), serial_control=True)
        self.assertEqual(ruled.problem.scf.spin_symmetry_sector_mode, "off")
        self.assertEqual(len(notes), 1)
        self.assertIn("both channels on the full grid", notes[0])

    def test_off_keeps_the_former_runner(self) -> None:
        # PARSEC_MPI_RESTART_RUN=off: the keyword is ignored as before, without error 371 or refusals.
        for spin, wfndat in ((False, False), (True, False), (False, True)):
            directory = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, directory, True)
            translation = self.parse(_restart_input(directory, spin=spin, wfndat=wfndat))
            with mock.patch.dict(os.environ, {"PARSEC_MPI_RESTART_RUN": "off"}):
                self.assertEqual(self.runner.restart_run_setting(), "off")
                same, notes = self.runner._restart_rules(translation, _Comm(0, 4))
                self.assertIs(same, translation)
                self.assertEqual(notes, [])
                self.assertIsNone(self.runner._load_restart_start(translation, None, None))
        with mock.patch.dict(os.environ, {"PARSEC_MPI_RESTART_RUN": "on"}), self.assertRaises(ValueError):
            self.runner._restart_rules(self.parse(_restart_input(self.directory)), _Comm())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_MPI_RESTART_RUN", None)
            self.assertEqual(self.runner.restart_run_setting(), "auto")

    def test_the_root_loads_the_start_of_main(self) -> None:
        translation = self.parse(_restart_input(self.directory))
        log = SimpleNamespace(write=lambda text: None)
        with mock.patch("parsec_python.cli.load_restart_start", return_value="start") as load:
            self.assertEqual(self.runner._load_restart_start(translation, "system", log), "start")
        load.assert_called_once_with(translation, "system", log, translation.source.parent)

    def test_the_ionic_backend_takes_the_start_first(self) -> None:
        from parsec_python.Relaxation.backend import IonicBackend

        calls = []

        def run_scf(system, *args, warm_start=None, **keywords):
            calls.append(warm_start)
            return "result"

        def factory(extrapolation="none"):
            return IonicBackend(run_scf=run_scf, compute_forces=None, update_geometry=None, warm_start=None)

        translation = self.parse(_restart_input(self.directory))
        backend = self.runner._ionic_backend_of(translation, factory, restart="stored")
        backend.run_scf("system")
        backend.run_scf("system")
        self.assertEqual(calls, ["stored", None])
        plain = self.runner._ionic_backend_of(translation, factory)
        plain.run_scf("system")
        self.assertEqual(calls, ["stored", None, None])


class SingleSectorLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        from parsec_python.acceleration.experimental import mpi_scf

        self.mpi_scf = mpi_scf

    def test_the_layout_of_one_rank_per_node(self) -> None:
        refusal = self.mpi_scf.single_sector_layout_refusal
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            self.assertEqual(self.mpi_scf.single_sector_node_layout_setting(), "refuse")
            text = refusal(1, 2, 4)
            self.assertIsNotNone(text)
            for words in ("2 of their 8 GPUs", "--rank-per-gpu", "one node", "PARSEC_SINGLE_SECTOR_NODE_LAYOUT=allow"):
                self.assertIn(words, text)
            # what shares the basis over every GPU, or has no basis to share
            self.assertIsNone(refusal(1, 1, 4))
            self.assertIsNone(refusal(1, 2, 1))
            self.assertIsNone(refusal(2, 2, 4))
            self.assertIsNone(refusal(1, 8, 4, gpu_per_rank=True))
            self.assertIsNone(refusal(1, 2, 4, exchange=True))
        with mock.patch.dict(os.environ, {"PARSEC_SINGLE_SECTOR_NODE_LAYOUT": " ALLOW "}):
            self.assertIsNone(refusal(1, 2, 4))
        with mock.patch.dict(os.environ, {"PARSEC_SINGLE_SECTOR_NODE_LAYOUT": "on"}), self.assertRaises(ValueError):
            self.mpi_scf.single_sector_node_layout_setting()

    def test_configure_refuses_it_on_every_rank(self) -> None:
        context = self.mpi_scf.MPISectorContext(_Comm(0, 2))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            with self.assertRaisesRegex(self.mpi_scf.MPISCFError, "first GPU of each rank"):
                context.configure(1, (0, 1, 2, 3))
        self.assertTrue(context.failed)
        # One GPU per rank, or allow: the shared sector of before.
        shared = self.mpi_scf.MPISectorContext(_Comm(1, 2))
        shared.gpu_per_rank = True
        self.assertEqual(shared.configure(1, (0,)), {0: (0,)})
        with mock.patch.dict(os.environ, {"PARSEC_SINGLE_SECTOR_NODE_LAYOUT": "allow"}):
            allowed = self.mpi_scf.MPISectorContext(_Comm(1, 2))
            self.assertEqual(allowed.configure(1, (0, 1, 2, 3)), {0: (0,)})
            self.assertTrue(allowed.shared)
        # several sectors on ranks with the GPUs of their nodes: as before
        several = self.mpi_scf.MPISectorContext(_Comm(0, 2))
        groups = several.configure(4, (0, 1, 2, 3))
        self.assertFalse(several.shared)
        self.assertEqual(set(groups), set(several.owned_sectors))
        self.assertEqual(len(several.owned_sectors), 2)


class NestedLayoutTests(unittest.TestCase):
    """(r7 merge of fixes and nested) The single-sector layout of one rank per node is refused only where nested
    sharing does not spread it over the GPUs of every node (``nested_sharing_applies``): after a CHEBDAV first solve
    or under ``PARSEC_MPI_NESTED_SHARING=off``."""

    def setUp(self) -> None:
        from parsec_python.acceleration.experimental import mpi_scf

        self.mpi_scf = mpi_scf

    def test_the_refusal_yields_to_nested_sharing(self) -> None:
        refusal = self.mpi_scf.single_sector_layout_refusal
        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto"}):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            self.assertIsNone(refusal(1, 2, 4, divided=True))
            # a CHEBDAV first solve: the refusal, which names Eigensolver: chebff first
            text = refusal(1, 2, 4, divided=False)
            self.assertIn("Name Eigensolver: chebff", text)
            self.assertIn("--rank-per-gpu", text)
        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "off"}):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            text = refusal(1, 2, 4, divided=True)
            self.assertIn("Leave PARSEC_MPI_NESTED_SHARING at auto (it is off)", text)
            self.assertIn("on the first GPU of each rank only", text)
            with mock.patch.dict(os.environ, {"PARSEC_SINGLE_SECTOR_NODE_LAYOUT": "allow"}):
                self.assertIsNone(refusal(1, 2, 4, divided=True))

    def test_configure_takes_the_nested_route_or_refuses(self) -> None:
        from parsec_python.acceleration.tests.thread_mpi import run_ranks

        def configure(thread):
            context = self.mpi_scf.MPISectorContext(thread.comm)
            context.node_devices = 4
            context.divided_bases = True
            try:
                groups = context.configure(1, (0, 1, 2, 3))
            except self.mpi_scf.MPISCFError as error:
                return str(error)
            return context.shared, context.nested, groups

        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto"}):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            for shared, nested, groups in run_ranks(2, configure):
                self.assertTrue(shared and nested)
                self.assertEqual(groups, {0: (0, 1, 2, 3)})
        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "off"}):
            os.environ.pop("PARSEC_SINGLE_SECTOR_NODE_LAYOUT", None)
            for outcome in run_ranks(2, configure):
                self.assertIn("first GPU of each rank", outcome)

    def test_the_spin_fallback_plan_counts_the_nested_layout(self) -> None:
        from parsec_python.acceleration.driver import SpinFallbackPlan

        def plan():
            return SpinFallbackPlan(basis_bytes=21 * 2**30, device_bytes=80 * 2**30, devices=2, divided=True,
                                    shareable=True, setting="auto", mpi=True, rank_devices=4, gpu_per_rank=False)

        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "auto"}):
            nested = plan()
            self.assertTrue(nested.nested)
            self.assertFalse(nested.first_gpu_shares)
            self.assertEqual(nested.sharing_devices, 8)
            self.assertIn("over the GPUs of its node", nested.layout_text())
        with mock.patch.dict(os.environ, {"PARSEC_MPI_NESTED_SHARING": "off"}):
            before = plan()
            self.assertFalse(before.nested)
            self.assertTrue(before.first_gpu_shares)
            self.assertEqual(before.sharing_devices, 2)


class SpinCellRefusalTests(unittest.TestCase):
    def test_the_reason_is_that_of_a_spin_cell(self) -> None:
        from parsec_python.acceleration.driver import spin_cell_group_refusal

        text = spin_cell_group_refusal("PARSEC_PERIODIC_SYMMETRY=off")
        self.assertIn("spin-polarized periodic cell", text)
        self.assertIn("PARSEC_PERIODIC_SYMMETRY=off", text)
        self.assertIn("main.py", text)
        self.assertNotIn("PARSEC_CUPY_SINGLE_SECTOR", text)


class LocalForceSwitchTests(unittest.TestCase):
    def test_every_switch_is_read_at_once(self) -> None:
        from parsec_python.acceleration.Forces.periodic_local import (
            periodic_local_balls_setting,
            periodic_local_orbits_setting,
            validate_local_force_settings,
        )

        names = (
            "PARSEC_PERIODIC_FORCE_SERIES",
            "PARSEC_PERIODIC_SECTOR_LOCAL_FORCES",
            "PARSEC_PERIODIC_LOCAL_ORBITS",
            "PARSEC_PERIODIC_LOCAL_BALLS",
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in names:
                os.environ.pop(name, None)
            validate_local_force_settings()
            for name in names:
                with self.subTest(name=name), mock.patch.dict(os.environ, {name: "of"}):
                    with self.assertRaisesRegex(ValueError, name):
                        validate_local_force_settings()
            for value in ("0", "false", "No"):
                with mock.patch.dict(
                    os.environ, {"PARSEC_PERIODIC_LOCAL_ORBITS": value, "PARSEC_PERIODIC_LOCAL_BALLS": value}
                ):
                    self.assertEqual(periodic_local_orbits_setting(), "off")
                    self.assertEqual(periodic_local_balls_setting(), "off")
                    validate_local_force_settings()

    def test_the_command_line_reads_them_before_the_run(self) -> None:
        from parsec_python.acceleration import cli

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        source = directory / "parsec.in"
        source.write_text((DATA / "H_cli_smoke.in").read_text())
        shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
        with (
            mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_BALLS": "of"}),
            mock.patch("sys.stderr") as stderr,
            mock.patch.object(cli, "shared_vector_seed", side_effect=AssertionError("ran past the check")),
        ):
            arguments = SimpleNamespace(
                profile_repeats=1, bands_from=None, input=source, pp_dir=None, esdf_lenient=False
            )
            self.assertEqual(cli._run(arguments, SimpleNamespace(device="cpu")), 2)
        printed = "".join(str(call.args[0]) for call in stderr.write.call_args_list if call.args)
        self.assertIn("PARSEC_PERIODIC_LOCAL_BALLS", printed)


def _ch4(offset: float = 0.0):
    """A methane-like cluster (Td about the origin), its hydrogens moved by ``offset`` bohr off the group."""

    d = 1.19
    return (
        Atom("C", (0.0, 0.0, 0.0)),
        Atom("H", (d, d, d * (1.0 + offset))),
        Atom("H", (-d, -d, d)),
        Atom("H", (-d, d, -d)),
        Atom("H", (d, -d, -d)),
    )


def _cluster_system(atoms, tolerance):
    from parsec_python.Grid import build_cluster_grid
    from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction
    from parsec_python.models import GridSettings

    grid = build_cluster_grid(GridSettings(radius=5.0, spacing=0.5, expansion_order=4))
    reduction = SignedPermutationReduction.detect(grid, atoms, atom_tolerance=tolerance)
    system = SimpleNamespace(
        backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=SimpleNamespace(reduction=reduction))),
        input=SimpleNamespace(periodic_cell=None),
        atoms=atoms,
        grid=grid,
    )
    return system, reduction


class AtomToleranceForcesTests(unittest.TestCase):
    def test_a_loose_group_is_refused_before_the_forces(self) -> None:
        from parsec_python.acceleration.Forces.sector_symmetry import sector_force_tolerance_refusal

        noisy = _ch4(2.0e-5 / 1.19)
        system, reduction = _cluster_system(noisy, 1.0e-4)
        self.assertGreater(reduction.group_order, 1)
        text = sector_force_tolerance_refusal(system, atom_tolerance=1.0e-4)
        self.assertIsNotNone(text)
        self.assertIn("PARSEC_SYMMETRY_ATOM_TOLERANCE=0.0001", text)
        self.assertIn("before the SCF", text)
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4"}):
            self.assertEqual(sector_force_tolerance_refusal(system), text)
        # a single point checks what its forces check: the wedge sums of the local forces (the default), not the
        # moves; with the full-grid sums and sectors without atom orbits its forces never match the atoms
        self.assertEqual(sector_force_tolerance_refusal(system, moves=False, atom_tolerance=1.0e-4), text)
        with mock.patch.dict(os.environ, {"PARSEC_LOCAL_FORCE_SUMS": "full"}):
            self.assertIsNone(sector_force_tolerance_refusal(system, moves=False, atom_tolerance=1.0e-4))
            self.assertEqual(sector_force_tolerance_refusal(system, moves=True, atom_tolerance=1.0e-4), text)
        # the default tolerance checks nothing; exact atoms pass at any tolerance
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_SYMMETRY_ATOM_TOLERANCE", None)
            self.assertIsNone(sector_force_tolerance_refusal(system))
        exact, _ = _cluster_system(_ch4(), 1.0e-4)
        self.assertIsNone(sector_force_tolerance_refusal(exact, atom_tolerance=1.0e-4))
        # no sectors: nothing to check
        none = SimpleNamespace(backend=SimpleNamespace(symmetry_eigensolver=None), atoms=noisy)
        self.assertIsNone(sector_force_tolerance_refusal(none, atom_tolerance=1.0e-4))

    def test_a_cluster_with_full_grid_orbitals_is_checked_against_its_moves(self) -> None:
        # Spin_Symmetry_Sectors: off (or main.py's spin Restart_Run rule) leaves the orbitals and the Hartree solve on
        # the full grid, but the driver's move_atoms still checks the moved atoms against the group the preparation
        # detected (system.move_symmetry_reduction): a relaxation is refused before its SCF.
        from parsec_python.acceleration.Forces.local_device import sector_atom_images
        from parsec_python.acceleration.Forces.sector_symmetry import sector_force_tolerance_refusal
        from parsec_python.acceleration.SCF.single_point import AcceleratedPreparedSinglePointSystem

        noisy = _ch4(2.0e-5 / 1.19)
        sectors, reduction = _cluster_system(noisy, 1.0e-4)
        self.assertGreater(reduction.group_order, 1)
        with self.assertRaises(ValueError):
            sector_atom_images(reduction, noisy)  # what move_atoms does after the first SCF
        full_grid = SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=None),
            input=SimpleNamespace(periodic_cell=None),
            atoms=noisy,
            grid=sectors.grid,
            move_symmetry_reduction=reduction,
        )
        text = sector_force_tolerance_refusal(full_grid, atom_tolerance=1.0e-4)
        self.assertIsNotNone(text)
        self.assertIn("before the SCF", text)
        self.assertIn("a group of", text)
        # a single point on the full grid: its forces keep no group, nothing to refuse
        self.assertIsNone(sector_force_tolerance_refusal(full_grid, moves=False, atom_tolerance=1.0e-4))
        # the prepared system of the driver carries the field (and keeps it through dataclasses.replace)
        prepared = AcceleratedPreparedSinglePointSystem(
            reference=SimpleNamespace(atoms=noisy, input=SimpleNamespace(periodic_cell=None), grid=sectors.grid),
            backend=SimpleNamespace(symmetry_eigensolver=None),
            backend_info=None,
            move_symmetry_reduction=reduction,
        )
        self.assertEqual(sector_force_tolerance_refusal(prepared, atom_tolerance=1.0e-4), text)
        from dataclasses import replace

        self.assertIs(replace(prepared, boundary_tail_maximum=None).move_symmetry_reduction, reduction)
        self.assertIsNone(AcceleratedPreparedSinglePointSystem(
            reference=prepared.reference, backend=prepared.backend, backend_info=None
        ).move_symmetry_reduction)
        # exact atoms pass, the default tolerance checks nothing
        _, exact_reduction = _cluster_system(_ch4(), 1.0e-4)
        exact = SimpleNamespace(**{**vars(full_grid), "atoms": _ch4(), "move_symmetry_reduction": exact_reduction})
        self.assertIsNone(sector_force_tolerance_refusal(exact, atom_tolerance=1.0e-4))
        self.assertIsNone(sector_force_tolerance_refusal(full_grid, atom_tolerance=1.0e-7))
        trivial = SimpleNamespace(**{**vars(full_grid), "move_symmetry_reduction": None})
        self.assertIsNone(sector_force_tolerance_refusal(trivial, atom_tolerance=1.0e-4))

    def test_the_note_keeps_the_tolerance_from_runs_with_forces(self) -> None:
        from parsec_python.acceleration.Symmetry.axis_reflection import (
            SignedPermutationReduction,
            detect_point_group_reduction,
            lost_operations_note,
        )
        from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model
        from parsec_python.acceleration.tests.test_periodic_symmetry_detection import _diamond

        atoms, sides = _diamond((1, 1, 1), side=6.0)
        grid, _ = _periodic_model(sides, 8, 0.0, atoms, {"C": 2.0})
        for scale, forces in ((1.0e-5, True), (1.0e-7, False)):
            with self.subTest(scale=scale):
                rng = np.random.default_rng(7)
                noisy = [
                    Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.normal(0.0, scale, 3))) for atom in atoms
                ]
                strict = detect_point_group_reduction(
                    grid, noisy, SignedPermutationReduction.detect(grid, noisy, atom_tolerance=1.0e-9),
                    atom_tolerance=1.0e-9,
                )
                note = lost_operations_note(grid, noisy, strict, atom_tolerance=1.0e-9)
                self.assertIsNotNone(note)
                self.assertIn("PARSEC_SYMMETRY_ATOM_TOLERANCE", note)
                self.assertEqual("serves a run without forces only" in note, forces)
                self.assertEqual("refused before the SCF" in note, forces)


if __name__ == "__main__":
    unittest.main()
