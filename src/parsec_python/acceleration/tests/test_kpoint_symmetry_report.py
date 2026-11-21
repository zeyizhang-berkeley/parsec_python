"""The k-point route names its cell group, and the readers of the density group take it without its table (CPU).

* ``kpoint_symmetry_detection`` (2026-11-21): the cell group of the k-point sectors, its translations and the atom
  tolerance it was built at (``PARSEC_SYMMETRY_ATOM_TOLERANCE``, ``PARSEC_KPOINT_SECTOR_TOLERANCE``), or why there are
  none, written when the solver is made, before its first solve, where a run prints it (the command line,
  ``SCF.kpoint_solver.start_kpoint_setup_notes``), in ``timings.notes`` of a k-point SCF and in its K-point solver
  detail block of ``parsec.out``;
* the readers of a :class:`parsec_python.Grid.space_group.PeriodicGridSymmetry` in the accelerated path (the orbit
  table of the symmetrized density in Gamma sectors, the invariance test of the k-point sectors) and the k-point SCF
  with a symmetrized density give the same integers and energies without the table (``PARSEC_PERIODIC_IMAGE_ROWS=on``,
  the rows of ``auto`` at any size) as with ``off``.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.SCF.kpoint_solver import KPointJob
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.models import Atom

_ROOT = Path(__file__).resolve().parents[4]
_DIAMOND = _ROOT / "examples" / "3d_diamond_benchmark"


def _silicon_system():
    from parsec_python.SCF.pbc import prepare_periodic_single_point
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = silicon("pbe", spacing=0.855)
        return problem, prepare_periodic_single_point(problem)


def _groups(grid, group):
    from parsec_python.Grid.space_group import periodic_grid_symmetry

    # (on: the rows at any size; these grids are below the 2^17 points from which auto keeps no table)
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "on"}):
        rows = periodic_grid_symmetry(grid, group)
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "off"}):
        table = periodic_grid_symmetry(grid, group)
    return rows, table


class DetectionLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = _silicon_system()

    def _job(self, states=22, cache=None, extras=None):
        geometry = {
            "grid": self.system.grid,
            "atoms": tuple(self.system.cell_atoms),
            "k_points": np.zeros((1, 3)),
            "weights": np.ones(1),
            "lattice": np.asarray(self.problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
            "cache": {} if cache is None else cache,
        }
        return KPointJob(1, states, lambda *a: None, lambda *a: None, lambda *a, **k: {}, geometry=geometry,
                         extras=extras)

    def test_the_cell_group_and_its_tolerance(self) -> None:
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0), \
                mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "", "PARSEC_KPOINT_SECTOR_TOLERANCE": ""}):
            cache: dict = {}
            notes: list = []
            plans = sectors.plan_sectors(self._job(cache=cache), notes)
            self.assertIsNotNone(plans)
            cell = sectors.CellGroup(self.system.grid, self.system.cell_atoms)
            self.assertEqual(len(notes), 1)
            line = notes[0]
            self.assertIn(f"cell group of {cell.order} operations", line)
            translated = sum(1 for shift in cell.shifts if np.any(shift != 0))
            self.assertIn(f"{translated} of them with a translation", line)
            self.assertIn("atom tolerance of 1.0e-07 bohr", line)
            self.assertIn("PARSEC_KPOINT_SECTOR_TOLERANCE=follow", line)
            self.assertIn("1 of 1 k-point(s) split", line)
            # the next loop of the structure takes the plans of the cache and the same line
            again: list = []
            self.assertIs(sectors.plan_sectors(self._job(cache=cache), again)[0], plans[0])
            self.assertEqual(again, notes)
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0), \
                mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-5"}):
            notes = []
            sectors.plan_sectors(self._job(), notes)
            self.assertIn("atom tolerance of 1.0e-05 bohr", notes[0])
            # a loop with forces: at most 1e-6
            notes = []
            sectors.plan_sectors(self._job(extras=lambda *a, **k: None), notes)
            self.assertIn("atom tolerance of 1.0e-06 bohr", notes[0])
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0), \
                mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-5",
                                             "PARSEC_KPOINT_SECTOR_TOLERANCE": "fixed"}):
            notes = []
            sectors.plan_sectors(self._job(), notes)
            self.assertIn("atom tolerance of 1.0e-07 bohr", notes[0])
            self.assertIn("PARSEC_KPOINT_SECTOR_TOLERANCE=fixed", notes[0])

    def test_why_there_are_none(self) -> None:
        notes: list = []
        self.assertIsNone(sectors.plan_sectors(self._job(), notes))
        self.assertIn("below the floor of the k-point sectors", notes[0])
        self.assertIn(f"{self.system.grid.size} grid points x 22 states", notes[0])
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"}):
            notes = []
            self.assertIsNone(sectors.sector_solver(self._job(), lambda *a, **k: None, notes=notes))
            self.assertEqual(notes, ["disabled by option (PARSEC_KPOINT_SECTORS=off; one task per k-point)"])
        job = self._job()
        notes = []
        self.assertIsNone(sectors.sector_solver(replace(job, geometry=None), lambda *a, **k: None, notes=notes))
        self.assertIn("takes no sectors", notes[0])

    def test_the_line_of_a_kpoint_scf(self) -> None:
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.SCF.kpoints import run_scf_kpoints

        side = float(np.asarray(self.problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        problem = replace(self.problem, scf=replace(self.problem.scf, convergence_criterion=1.0e-6))
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        from parsec_python.SCF.kpoint_solver import start_kpoint_setup_notes, stop_kpoint_setup_notes

        for floor, expected in ((0, "cell group of"), (1 << 40, "below the floor")):
            with self.subTest(floor=floor), mock.patch.object(sectors, "_MINIMUM_ENTRIES", floor), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # what the command line writes (parsec.out and the standard output), in order with the iterations
                written: list[str] = []
                token = start_kpoint_setup_notes(written.append)
                try:
                    result = run_scf_kpoints(
                        prepare_periodic_single_point(problem), k_points, np.array([0.5, 0.5]),
                        callback=lambda step: written.append("iteration"),
                    )
                finally:
                    stop_kpoint_setup_notes(token)
                line = result.timings.notes["kpoint_symmetry_detection"]
                self.assertIn(expected, line)
                printed = ParsecTextReporter._kpoint_detail_lines(result.timings)
                self.assertIn(f" kpoint_symmetry_detection = {line}", printed)
                self.assertLess(printed.index(f" kpoint_symmetry_detection = {line}"), 4)
                # at the set-up: once, before the first iteration
                self.assertEqual(written[0], f" kpoint_symmetry_detection = {line}")
                self.assertEqual(written.count(written[0]), 1)
                self.assertIn("iteration", written[1:])

    def test_the_setup_notes_of_a_run(self) -> None:
        from parsec_python.SCF import kpoint_solver

        class Solver:
            def __init__(self, text):
                self.kpoint_symmetry_detection = text

        made = iter([Solver("group A"), Solver("group A"), Solver("group B"), Solver(None), Solver("group C")])
        outer: list[str] = []
        inner: list[str] = []
        with mock.patch.object(kpoint_solver, "_parallel_solver", lambda job, **options: next(made)):
            kpoint_solver.make_kpoint_solver(None)  # no run prints them: nothing written
            token = kpoint_solver.start_kpoint_setup_notes(outer.append)
            try:
                kpoint_solver.make_kpoint_solver(None)
                nested = kpoint_solver.start_kpoint_setup_notes(inner.append)
                kpoint_solver.make_kpoint_solver(None)
                kpoint_solver.make_kpoint_solver(None)
                kpoint_solver.stop_kpoint_setup_notes(nested)
                kpoint_solver.stop_kpoint_setup_notes(nested)  # (a token no longer active is ignored)
                kpoint_solver.make_kpoint_solver(None)
            finally:
                kpoint_solver.stop_kpoint_setup_notes(token)
        self.assertEqual(outer, [" kpoint_symmetry_detection = group A", " kpoint_symmetry_detection = group C"])
        self.assertEqual(inner, [" kpoint_symmetry_detection = group B"])
        self.assertEqual(kpoint_solver._SETUP_NOTES, [])


class DensityGroupReaderTests(unittest.TestCase):
    def test_the_orbit_table_of_the_gamma_sectors(self) -> None:
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.Grid.space_group import find_space_group
        from parsec_python.acceleration.SCF.density_symmetry import build_wedge_density_symmetry
        from parsec_python.acceleration.Symmetry import detect_point_group_reduction, load_or_detect_reflection_reduction
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        cell = PeriodicCell(lattice_vectors=np.eye(3) * 6.0)
        grid = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=4))
        atoms = (Atom("Si", (0.0, 0.0, 0.0)),)
        origin, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
        reduction = detect_point_group_reduction(grid, atoms, origin)
        rows, table = _groups(grid, find_space_group(cell.lattice_vectors, atoms))
        self.assertFalse(rows.holds_table)
        ours = build_wedge_density_symmetry(rows, reduction)
        theirs = build_wedge_density_symmetry(table, reduction)
        np.testing.assert_array_equal(ours.orbit_images, theirs.orbit_images)
        self.assertEqual(ours.orbit_images.dtype, theirs.orbit_images.dtype)

    def test_the_invariance_of_the_kpoint_sectors(self) -> None:
        from parsec_python.Grid.space_group import find_grid_space_group

        problem, system = _silicon_system()
        grid = system.grid
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        group = find_grid_space_group(lattice, system.cell_atoms, grid.points_per_axis, grid.shift)
        self.assertGreater(group.order, 1)
        rows, table = _groups(grid, group)
        cell = sectors.CellGroup(grid, system.cell_atoms)
        point = np.array([[0.11, 0.07, 0.05]])
        ours = sectors.PotentialInvariance(cell, rows, point, np.ones(1), lattice)
        theirs = sectors.PotentialInvariance(cell, table, point, np.ones(1), lattice)
        np.testing.assert_array_equal(ours._sampled, theirs._sampled)
        decided = [ours.in_density_group(index) for index in range(cell.order)]
        self.assertEqual(decided, [theirs.in_density_group(index) for index in range(cell.order)])
        self.assertTrue(any(decided[1:]))

    @unittest.skipUnless((_DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
    def test_a_kpoint_scf_with_the_symmetrized_density(self) -> None:
        # the benchmark's input on the default routines (the 3 x 3 x 3 mesh reduced by Td about an atom, the density
        # averaged over it each iteration), in the k-point sectors and without them: the same energies bit for bit
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.Grid.kpoints import kpoint_sampling

        directory = Path(tempfile.mkdtemp(prefix="parsec_rows_"))
        self.addCleanup(shutil.rmtree, directory, True)
        subprocess.run(
            [sys.executable, str(_DIAMOND / "make_diamond.py"), "1", "1", "1", str(directory / "parsec.in"),
             "--kmesh", "3", "3", "3"],
            check=True, capture_output=True,
        )
        (directory / "C_POTRE.DAT").write_bytes((_DIAMOND / "C_POTRE.DAT").read_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(directory / "parsec.in").problem
        problem = replace(problem, scf=replace(problem.scf, convergence_criterion=1.0e-8))
        self.assertTrue(problem.scf.symmetrize_density)
        # one worker thread on the CPU: the k-points then add their densities in one order and the run repeats itself
        # bit for bit (with several workers two runs of either setting differ by about 2e-14 Ry)
        steady = {"PARSEC_DEVICE": "cpu", "PARSEC_KPOINT_BACKEND": "", "PARSEC_KPOINT_WORKERS": "1"}
        for floor in (0, 1 << 40):
            results = {}
            # (on: the rows at this size, below the 2^17 points from which auto keeps no table)
            for setting in ("on", "off"):
                with self.subTest(floor=floor, setting=setting), mock.patch.object(sectors, "_MINIMUM_ENTRIES", floor), \
                        mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": setting, **steady}), \
                        warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    system = prepare_periodic_single_point(problem)
                    self.assertEqual(system.density_symmetry.holds_table, setting == "off")
                    points, weights = kpoint_sampling(problem)
                    results[setting] = run_scf_kpoints(system, points, weights)
            auto, off = results["on"], results["off"]
            self.assertTrue(auto.converged and off.converged)
            self.assertEqual(auto.energies.total, off.energies.total)
            self.assertEqual(len(auto.history), len(off.history))
            np.testing.assert_array_equal(np.asarray(auto.density), np.asarray(off.density))


if __name__ == "__main__":
    unittest.main()
