"""The size floor of the Gamma sectors of a periodic cell (``PARSEC_PERIODIC_SECTOR_FLOOR``).

CPU: the switch and the basis of a cell against the floor.  GPU: the silicon cube (13,824 points x 22 states) keeps
the full grid under ``auto`` and the sectors of its point group under ``off``, with the same energy.

The package of these tests sets ``PARSEC_PERIODIC_SECTOR_FLOOR=off`` for the other tests (``__init__.py``), whose
small cells exercise the sectors; the tests here set the switch themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon


def _silicon(directory: Path, extra: str = ""):
    from parsec_python.Input import parse_parsec_input

    settings = {
        "Grid_Spacing": "0.57",
        "Expansion_Order": "8",
        "States_Num": "22",
        "Convergence_Criterion": "1.0e-9",
        "Diag_Tolerance": "1.0e-10",
        "Correlation_Type": "pbe",
    }
    path = write_silicon(directory, extra, positions=displaced_positions(0.0), settings=settings)
    return parse_parsec_input(path).problem


SPIN = "Spin_Polarization: true\nInitial_Spin_Polarization: 0.1\n"


class FloorSettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.driver import periodic_sector_floor_setting

        for value, expected in (("", 1 << 24), ("auto", 1 << 24), ("OFF", 0), ("1000", 1000), ("0", 0)):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FLOOR": value}):
                self.assertEqual(periodic_sector_floor_setting(), expected)
        for value in ("on", "-3", "1e6"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FLOOR": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_SECTOR_FLOOR"):
                    periodic_sector_floor_setting()

    def test_the_basis_of_a_cell_against_the_floor(self) -> None:
        from parsec_python.acceleration.driver import _periodic_basis_below_floor

        with tempfile.TemporaryDirectory() as name:
            problem = _silicon(Path(name))
        found = _periodic_basis_below_floor(problem, None, 1 << 24)
        self.assertIsNotNone(found)
        rows, states = found
        self.assertEqual(states, 22)
        self.assertLess(rows * states, 1 << 24)
        self.assertIsNone(_periodic_basis_below_floor(problem, None, rows * states))
        self.assertIsNone(_periodic_basis_below_floor(problem, None, 0))
        # an MPI run keeps its sectors (its ranks share no full grid)
        self.assertIsNone(_periodic_basis_below_floor(problem, object(), 1 << 24))
        # and so do forces and moves averaged over the group of the sectors
        self.assertIsNone(_periodic_basis_below_floor(problem, None, 1 << 24, sector_forces=True))

    def test_the_points_are_counted_without_building_the_grid(self) -> None:
        from dataclasses import replace

        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.acceleration.SCF.gamma_hybrid_route import periodic_grid_points

        with tempfile.TemporaryDirectory() as name:
            problem = _silicon(Path(name))
        for spacing in (0.57, 0.45, 0.3):
            for rule in ("round", "fortran"):
                with self.subTest(spacing=spacing, rule=rule):
                    case = replace(problem, grid=replace(problem.grid, spacing=spacing, points_rule=rule))
                    self.assertEqual(
                        periodic_grid_points(case), build_periodic_grid(case.periodic_cell, case.grid).size
                    )
        with mock.patch("parsec_python.Grid.pbc.build_periodic_grid", side_effect=AssertionError("built")):
            self.assertEqual(periodic_grid_points(problem), 18**3)


class SpinFloorTests(unittest.TestCase):
    """The floor of a spin-polarized cell (``PARSEC_PERIODIC_SPIN_SECTOR_FLOOR``)."""

    def test_the_switch(self) -> None:
        from parsec_python.acceleration.driver import periodic_spin_sector_floor_setting

        for value, expected in (("", 0), ("auto", 0), ("OFF", 0), ("2097152", 1 << 21), ("0", 0)):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": value}):
                self.assertEqual(periodic_spin_sector_floor_setting(), expected)
        for value in ("on", "-3", "1e6"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_SPIN_SECTOR_FLOOR"):
                    periodic_spin_sector_floor_setting()

    def test_each_kind_of_cell_compares_with_its_own_floor(self) -> None:
        from parsec_python.acceleration import driver

        with tempfile.TemporaryDirectory() as name:
            plain = _silicon(Path(name) / "plain")
            spin = _silicon(Path(name) / "spin", SPIN)
        rows, states = driver._periodic_basis_below_floor(spin, None, 1 << 40)
        basis = rows * states
        decide = driver._cell_below_sector_floor
        found, floor, switch = decide(spin, None, spin_polarized=True, floor=0, spin_floor=basis + 1)
        self.assertEqual((found, floor, switch), ((rows, states), basis + 1, "PARSEC_PERIODIC_SPIN_SECTOR_FLOOR"))
        self.assertIsNone(decide(spin, None, spin_polarized=True, floor=basis + 1, spin_floor=basis)[0])
        self.assertIsNone(decide(spin, None, spin_polarized=True, floor=0, spin_floor=0)[0])
        found, _, switch = decide(plain, None, spin_polarized=False, floor=basis + 1, spin_floor=0)
        self.assertEqual((found, switch), ((rows, states), "PARSEC_PERIODIC_SECTOR_FLOOR"))
        # an MPI run, forces averaged over the group and the inputs of other switches keep the sectors
        self.assertIsNone(decide(spin, object(), spin_polarized=True, floor=0, spin_floor=basis + 1)[0])
        self.assertIsNone(
            decide(spin, None, spin_polarized=True, floor=0, spin_floor=basis + 1, sector_forces=True)[0]
        )
        with mock.patch.object(driver, "periodic_sector_refusal", return_value="LDA+U"):
            self.assertIsNone(decide(spin, None, spin_polarized=True, floor=0, spin_floor=basis + 1)[0])

    def test_opposite_starts_keep_their_route(self) -> None:
        from dataclasses import replace

        from parsec_python.acceleration import driver

        with tempfile.TemporaryDirectory() as name:
            spin = _silicon(Path(name), SPIN)
        label = spin.atoms[0].symbol
        species = spin.pseudopotentials[label]

        def started(values):
            atoms = tuple(
                replace(atom, symbol=f"Si_{index % len(values)}") for index, atom in enumerate(spin.atoms)
            )
            table = {
                f"Si_{index}": replace(species, initial_spin_polarization=value) for index, value in enumerate(values)
            }
            return replace(spin, atoms=atoms, pseudopotentials=table)

        self.assertFalse(driver._opposite_spin_starts(spin))
        for values, expected in (
            ((0.5, -0.5), True),
            ((0.1, -0.4, 0.0), True),
            ((0.5, 0.2), False),
            ((-0.5, -0.2), False),
            ((0.5, 0.0), False),
            ((0.0, 0.0), False),
        ):
            with self.subTest(values=values):
                problem = started(values)
                self.assertIs(driver._opposite_spin_starts(problem), expected)
                found = driver._cell_below_sector_floor(
                    problem, None, spin_polarized=True, floor=0, spin_floor=1 << 40
                )[0]
                self.assertEqual(found is None, expected)
        # a label without atoms does not count
        unused = replace(spin, pseudopotentials={label: species, "Si_x": replace(species, initial_spin_polarization=-0.5)})
        self.assertFalse(driver._opposite_spin_starts(unused))


def _cli_route(directory: Path, extra: str, environment=None) -> dict:
    """The keywords the accelerated command line prepares the displaced silicon cube with (not run)."""

    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    import parsec_python.acceleration.cli as cli_module
    from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import _Prepared

    path = write_silicon(directory, extra)
    seen: dict = {}

    def prepare(problem, **keywords):
        seen.update(keywords)
        raise _Prepared

    variables = {"PARSEC_PERIODIC_SECTOR_FORCES": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "auto", **(environment or {})}
    with mock.patch.dict(os.environ, variables), mock.patch.object(
        cli_module, "prepare_single_point", side_effect=prepare
    ), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            cli_module.main([str(path), "--quiet", "--no-archive", "--device", "cpu"])
        except _Prepared:
            return seen
    raise AssertionError("the command line did not prepare")


class FloorForcesRoutingTests(unittest.TestCase):
    """Forces and relaxations in the sectors keep them below the floor: the command line says so to the driver."""

    def test_forces_and_relaxations_ask_for_the_sectors(self) -> None:
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import BFGS, DYNAMICS, FORCES, SIMPLE

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for index, extra in enumerate((FORCES, BFGS, SIMPLE)):
                with self.subTest(extra=extra):
                    seen = _cli_route(root / f"moving{index}", extra)
                    self.assertEqual(seen["symmetry"], "auto")
                    self.assertIs(seen["sector_forces"], True)
            # an SCF alone, dynamics (full grid) and the switch off: the floor decides as before
            for index, (extra, environment) in enumerate(
                (("", None), (DYNAMICS, None), (FORCES, {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}))
            ):
                with self.subTest(extra=extra, environment=environment):
                    self.assertIs(_cli_route(root / f"other{index}", extra, environment)["sector_forces"], False)


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # pragma: no cover
        return False


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class FloorRouteTests(unittest.TestCase):
    def _run(self, setting: str):
        from parsec_python.acceleration import driver

        with tempfile.TemporaryDirectory() as name:
            problem = _silicon(Path(name))
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FLOOR": setting}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy", symmetry="auto")
                result = driver.run_scf(system)
        return dict(system.backend_info.details).get("symmetry_detection", ""), float(result.energies.total)

    def test_a_small_cell_keeps_the_full_grid(self) -> None:
        floor_text, floor_energy = self._run("auto")
        sector_text, sector_energy = self._run("off")
        self.assertIn("below the floor of the sectors", floor_text)
        self.assertIn("PARSEC_PERIODIC_SECTOR_FLOOR", floor_text)
        self.assertIn("exact", sector_text)
        self.assertAlmostEqual(floor_energy, sector_energy, delta=1e-8)
        # a floor at the size of the cell keeps the sectors
        text, energy = self._run("1")
        self.assertIn("exact", text)
        self.assertAlmostEqual(energy, sector_energy, delta=1e-12)


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class SpinFloorRouteTests(unittest.TestCase):
    def _run(self, setting: str):
        from parsec_python.acceleration import driver

        with tempfile.TemporaryDirectory() as name:
            problem = _silicon(Path(name), SPIN)
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": setting}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy", symmetry="auto")
                result = driver.run_scf(system)
        return dict(system.backend_info.details).get("symmetry_detection", ""), float(result.energies.total)

    def test_a_floor_keeps_a_small_spin_cell_on_the_full_grid(self) -> None:
        floor_text, floor_energy = self._run(str(1 << 21))
        sector_text, sector_energy = self._run("off")
        default_text, default_energy = self._run("auto")
        self.assertIn("below the floor of the sectors", floor_text)
        self.assertIn("PARSEC_PERIODIC_SPIN_SECTOR_FLOOR", floor_text)
        self.assertIn("exact", sector_text)
        self.assertAlmostEqual(floor_energy, sector_energy, delta=2e-6)
        # the default takes no floor: the route of off, bit for bit
        self.assertEqual(default_text, sector_text)
        self.assertEqual(default_energy, sector_energy)


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class FloorRelaxationTests(unittest.TestCase):
    """A steepest-descent step of the displaced silicon cube (below the floor) under the default floor: the forces and
    moved positions stay averaged over the group of the sectors, so the four atoms Td fixes do not move (on the full
    grid their round-off forces move them by about 1e-4 bohr), as with the floor off."""

    def test_the_atoms_the_group_fixes_stay_under_the_default_floor(self) -> None:
        import numpy as np

        from parsec_python.acceleration.backends.native import native_available
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import STEEPEST_DESCENT, _run

        if not native_available():
            self.skipTest("native extension is unavailable")
        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for setting in ("auto", "off"):
                with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FLOOR": setting}):
                    runs[setting] = _run(Path(name) / setting, STEEPEST_DESCENT, "auto")
        for setting, (status, _, log, errors) in runs.items():
            self.assertIn(status, (0, 4), f"{setting}: {errors}\n{log[-3000:]}")
            self.assertIn("Forces of the periodic cell in the sectors", log)
            self.assertNotIn("below the floor of the sectors", log)
        moved = np.asarray(runs["auto"][1]["relaxation_positions"])
        self.assertEqual(moved.shape, (2, 8, 3))
        np.testing.assert_allclose(moved[1, :4], moved[0, :4], rtol=0, atol=1e-12)
        self.assertGreater(float(np.abs(moved[1, 4:] - moved[0, 4:]).max()), 1.0e-3)
        # the same run as with the floor off, bit for bit
        np.testing.assert_array_equal(moved, np.asarray(runs["off"][1]["relaxation_positions"]))
        np.testing.assert_array_equal(
            runs["auto"][1]["relaxation_energies"], runs["off"][1]["relaxation_energies"]
        )


if __name__ == "__main__":
    unittest.main()
