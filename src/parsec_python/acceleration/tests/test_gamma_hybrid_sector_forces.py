"""Forces and relaxations of a small Gamma-point hybrid (``PARSEC_GAMMA_HYBRID_SECTOR_FORCES``).

An unpolarized hybrid of a bulk cell at the Gamma point takes the real sectors of its abelian group where its basis
reaches ``2^24`` entries (``PARSEC_GAMMA_HYBRID_SECTORS=auto``); a smaller cell keeps the full grid.  The command line
counts its forces and its relaxations by an optimizer as forces of the sectors (``PARSEC_PERIODIC_SECTOR_FORCES=auto``),
which it averages over the group of the sectors; on the full grid of the default routines nothing averaged them (the
Fortran-derived routines average theirs over their own operations).  The switch:

* ``auto`` (the default since 2026-11-21): the full grid, its forces and moved positions averaged over the operations
  of the cell (``Forces.periodic.periodic_force_operations``, as the Fortran-derived routines do there) on the default
  routines; the Fortran-derived routines as before;
* ``sectors``: the sectors whatever the size of the basis;
* ``off``: the full grid without the average, as before.

CPU: the switch and the decisions of the driver.  GPU: the displaced silicon cube with PBE0 (5,832 points x 22 states,
below the floor): the averaged forces of the full grid equal those of the full grid and of the sectors, a BFGS
relaxation moves as on the full grid, steepest descent keeps the atoms the group fixes where the full grid moves them by
round-off, and a cell of the Fortran-derived routines keeps its own route.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import parsec_python.acceleration.cli as cli_module
from parsec_python.acceleration.SCF import gamma_hybrid_route as route_module
from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import (
    FORCES,
    RELAXATION,
    SILICON_SETTINGS,
    STEEPEST_DESCENT,
    displaced_positions,
    write_silicon,
)

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
    from parsec_python.acceleration.backends.native import native_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

    def native_available() -> bool:
        return False

_SWITCH = "PARSEC_GAMMA_HYBRID_SECTOR_FORCES"
#: The silicon cube of the sector-force tests with PBE0 and a tight Fock loop.
HYBRID_SETTINGS = {**SILICON_SETTINGS, "Correlation_Type": "pbe0"}
HYBRID_EXTRA = "Exx_Tolerance: 1e-10 Ry\n"


def _problem(*, hybrid=True, spin=False, fortran=False, periodic=True):
    return SimpleNamespace(
        scf=SimpleNamespace(is_hybrid=hybrid, spin_polarized=spin),
        grid=SimpleNamespace(fortran_routines=fortran),
        periodic_cell=object() if periodic else None,
    )


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (
            ("", "auto"),
            ("AUTO", "auto"),
            ("sectors", "sectors"),
            (" Sectors ", "sectors"),
            ("off", "off"),
            (" Off ", "off"),
        ):
            with mock.patch.dict(os.environ, {_SWITCH: value}):
                self.assertEqual(route_module.gamma_hybrid_sector_forces_setting(), expected)
        for value in ("on", "1", "kpoint", "full"):
            with mock.patch.dict(os.environ, {_SWITCH: value}):
                with self.assertRaisesRegex(ValueError, _SWITCH):
                    route_module.gamma_hybrid_sector_forces_setting()

    def test_the_sectors_below_the_floor(self) -> None:
        from parsec_python.acceleration import driver

        hybrid = _problem()
        semilocal = _problem(hybrid=False)
        with mock.patch.object(route_module, "sector_basis_reached", return_value=False):
            for sectors, forces_setting, expected in (
                ("auto", "auto", False),
                ("auto", "sectors", True),
                ("auto", "off", False),
                ("off", "sectors", False),
                ("kpoint", "sectors", False),
            ):
                variables = {"PARSEC_GAMMA_HYBRID_SECTORS": sectors, _SWITCH: forces_setting}
                with self.subTest(sectors=sectors, forces=forces_setting), mock.patch.dict(os.environ, variables):
                    self.assertIs(driver._gamma_hybrid_in_sectors(hybrid, None, sector_forces=True), expected)
                    # an SCF alone below the floor, an MPI context, a semilocal functional: as before
                    self.assertFalse(driver._gamma_hybrid_in_sectors(hybrid, None))
                    self.assertFalse(driver._gamma_hybrid_in_sectors(hybrid, object(), sector_forces=True))
                    self.assertFalse(driver._gamma_hybrid_in_sectors(semilocal, None, sector_forces=True))
        with mock.patch.object(route_module, "sector_basis_reached", return_value=True):
            for forces_setting in ("auto", "sectors", "off"):
                with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "auto", _SWITCH: forces_setting}):
                    self.assertTrue(driver._gamma_hybrid_in_sectors(hybrid, None))
                    self.assertTrue(driver._gamma_hybrid_in_sectors(hybrid, None, sector_forces=True))

    def test_the_average_on_the_full_grid(self) -> None:
        from parsec_python.acceleration import driver

        def decide(problem, *, context=None, forces=True, symmetry="auto", periodic="auto", reached=False,
                   refusal=None, **variables):
            environment = {"PARSEC_GAMMA_HYBRID_SECTORS": "auto", _SWITCH: "auto", **variables}
            with mock.patch.object(route_module, "sector_basis_reached", return_value=reached), mock.patch.object(
                driver, "periodic_sector_refusal", return_value=refusal
            ), mock.patch.dict(os.environ, environment):
                return driver._gamma_hybrid_full_grid_forces(
                    problem, context, sector_forces=forces, symmetry_mode=symmetry, periodic_symmetry_mode=periodic
                )

        self.assertTrue(decide(_problem()))
        # the default routines alone: the Fortran-derived ones average their forces on the full grid already
        self.assertFalse(decide(_problem(fortran=True)))
        for label, value in (
            ("no forces of the sectors", decide(_problem(), forces=False)),
            ("semilocal", decide(_problem(hybrid=False))),
            ("spin", decide(_problem(spin=True))),
            ("cluster", decide(_problem(periodic=False))),
            ("MPI", decide(_problem(), context=object())),
            ("symmetry off", decide(_problem(), symmetry="off")),
            ("PARSEC_PERIODIC_SYMMETRY=off", decide(_problem(), periodic="off")),
            ("above the floor (the sectors)", decide(_problem(), reached=True)),
            ("a cell the sectors do not take", decide(_problem(), refusal="an oblique cell")),
            ("sectors", decide(_problem(), **{_SWITCH: "sectors"})),
            ("off", decide(_problem(), **{_SWITCH: "off"})),
            ("hybrid sectors off", decide(_problem(), PARSEC_GAMMA_HYBRID_SECTORS="off")),
            ("hybrid sectors kpoint", decide(_problem(), PARSEC_GAMMA_HYBRID_SECTORS="kpoint")),
        ):
            with self.subTest(label):
                self.assertFalse(value)

    def test_the_detail_of_the_system(self) -> None:
        def system(details):
            return SimpleNamespace(backend_info=SimpleNamespace(details=details))

        marked = system(((route_module.FULL_GRID_FORCES_DETAIL, route_module.FULL_GRID_FORCES_VALUE + "; ..."),))
        self.assertTrue(route_module.full_grid_forces_averaged(marked))
        self.assertFalse(route_module.full_grid_forces_averaged(system(())))
        self.assertFalse(route_module.full_grid_forces_averaged(system((("symmetry_mode", "auto"),))))
        self.assertFalse(route_module.full_grid_forces_averaged(SimpleNamespace()))
        self.assertIsNone(cli_module._hybrid_full_grid_forces_line(system(())))

    def test_the_command_line_refuses_a_wrong_value(self) -> None:
        # read in every run before the preparation (a semilocal cell here; the preparation is not reached)
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import _Prepared

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            path = write_silicon(Path(name), FORCES)
            errors = StringIO()
            with mock.patch.dict(os.environ, {_SWITCH: "sometimes"}), mock.patch.object(
                cli_module, "prepare_single_point", side_effect=_Prepared
            ), redirect_stdout(StringIO()), redirect_stderr(errors):
                status = cli_module.main([str(path), "--quiet", "--no-archive", "--device", "cpu"])
        self.assertEqual(status, 2)
        self.assertIn(f"{_SWITCH} must be auto, sectors or off", errors.getvalue())


def _run(directory: Path, extra: str, setting: str, *, positions=None, settings=None):
    """The accelerated command line on the silicon cube with PBE0: ``(status, archive, parsec.out, stderr)``."""

    path = write_silicon(
        directory, extra + HYBRID_EXTRA, positions=positions, settings=HYBRID_SETTINGS if settings is None else settings
    )
    errors = StringIO()
    variables = {_SWITCH: setting, "PARSEC_PERIODIC_SECTOR_FORCES": "auto", "PARSEC_GAMMA_HYBRID_SECTORS": "auto"}
    with mock.patch.dict(os.environ, variables), redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", "--backend", "cupy", "--gpus", "1"])
    archive = {}
    if (directory / "parsec_python_results.npz").is_file():
        with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
            archive = {key: data[key] for key in data.files}
    log = (directory / "parsec.out").read_text(encoding="utf-8") if (directory / "parsec.out").is_file() else ""
    return status, archive, log, errors.getvalue()


_SECTOR_LINE = "Forces of the periodic cell in the sectors of its point group"
_AVERAGE_LINE = "Forces of the periodic hybrid on the full grid (below the floor of its sectors)"
_SYMMETRIZING = "symmetrizing forces according to symmetry operations"


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class HybridSectorForceTests(unittest.TestCase):
    """The PBE0 silicon cube below the floor: the full grid with the average (auto), the sectors (sectors) and the
    full grid as before (off)."""

    def test_the_cell_is_below_the_floor(self) -> None:
        from parsec_python.Input import parse_parsec_input

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            path = write_silicon(Path(name), FORCES + HYBRID_EXTRA, settings=HYBRID_SETTINGS)
            problem = parse_parsec_input(path).problem
        self.assertFalse(route_module.sector_basis_reached(problem))

    def test_averaged_forces_equal_those_of_the_full_grid_and_of_the_sectors(self) -> None:
        # the atom at the origin moved along z as well: C2v about the origin, Abelian, every level solved
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", FORCES, "off", positions=positions)
            averaged = _run(root / "averaged", FORCES, "auto", positions=positions)
            sectors = _run(root / "sectors", FORCES, "sectors", positions=positions)
        for label, (status, _, log, errors) in (("full", full), ("averaged", averaged), ("sectors", sectors)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
        self.assertIn(_SECTOR_LINE + ": an Abelian group of 4 operations", sectors[2])
        self.assertIn(_AVERAGE_LINE + ": the total and the moved positions averaged over the 4 operations", averaged[2])
        self.assertIn(_SYMMETRIZING, averaged[2])
        for log in (full[2], averaged[2]):
            self.assertNotIn(_SECTOR_LINE, log)
        for log in (full[2], sectors[2]):
            self.assertNotIn(_AVERAGE_LINE, log)
        self.assertNotIn(_SYMMETRIZING, full[2])
        self.assertGreater(float(np.abs(full[1]["forces_total"]).max()), 1.0e-3)
        # the same SCF as the full grid; its forces averaged over C2v about the origin
        self.assertAlmostEqual(float(averaged[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=1e-10)
        np.testing.assert_allclose(averaged[1]["forces_total"], full[1]["forces_total"], rtol=0, atol=2e-6)
        np.testing.assert_allclose(averaged[1]["forces_total"], sectors[1]["forces_total"], rtol=0, atol=2e-6)
        # the moved atom at the origin: no force across the mirror planes of C2v
        np.testing.assert_allclose(averaged[1]["forces_total"][0, :2], 0.0, rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(sectors[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=1e-7)

    def test_a_relaxation_moves_as_on_the_full_grid(self) -> None:
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", RELAXATION, "off", positions=positions)
            averaged = _run(root / "averaged", RELAXATION, "auto", positions=positions)
            sectors = _run(root / "sectors", RELAXATION, "sectors", positions=positions)
        for label, (status, _, log, errors) in (("full", full), ("averaged", averaged), ("sectors", sectors)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
        self.assertIn(_SECTOR_LINE + ": an Abelian group of 4 operations", sectors[2])
        self.assertIn(_AVERAGE_LINE, averaged[2])
        for run in (averaged, sectors):
            self.assertEqual(np.asarray(run[1]["relaxation_positions"]).shape, (3, 8, 3))
            for key in ("relaxation_positions", "relaxation_final_positions", "relaxation_energies"):
                np.testing.assert_allclose(run[1][key], full[1][key], rtol=0, atol=1e-6, err_msg=key)
        # the atom at the origin stays on the z axis of C2v
        moved = np.asarray(averaged[1]["relaxation_positions"])
        np.testing.assert_allclose(moved[:, 0, :2], 0.0, rtol=0, atol=1e-12)

    def test_steepest_descent_keeps_the_atoms_the_group_fixes(self) -> None:
        # Td about the origin: the simple scheme of domove.f90 moves every nonzero force component by about Min_Step,
        # so on the full grid the round-off forces of the atoms that the group fixes move them; with the forces and
        # positions averaged over the group (the full grid of auto, the abelian sectors of sectors) they stay.  The
        # atoms with forces move as on the full grid.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", STEEPEST_DESCENT, "off")
            averaged = _run(root / "averaged", STEEPEST_DESCENT, "auto")
            sectors = _run(root / "sectors", STEEPEST_DESCENT, "sectors")
        for label, (status, _, log, errors) in (("full", full), ("averaged", averaged), ("sectors", sectors)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-3000:]}")
        self.assertIn(_SECTOR_LINE, sectors[2])
        self.assertIn(_AVERAGE_LINE + ": the total and the moved positions averaged over the 24 operations", averaged[2])
        reference = np.asarray(full[1]["relaxation_positions"])
        for label, run in (("averaged", averaged), ("sectors", sectors)):
            with self.subTest(label):
                self.assertIn(_SYMMETRIZING, run[2])
                moved = np.asarray(run[1]["relaxation_positions"])
                self.assertEqual(moved.shape, (2, 8, 3))
                # the atom at the origin is fixed by every operation about it
                np.testing.assert_allclose(moved[1, 0], moved[0, 0], rtol=0, atol=1e-12)
                self.assertGreater(float(np.abs(moved[1] - moved[0]).max()), 1.0e-3)
                same = np.abs(moved[1] - reference[1]) <= 1.0e-6
                kept = np.abs(moved[1] - moved[0]) <= 1.0e-12
                self.assertTrue(bool(np.all(same | kept)), f"{moved[1] - reference[1]}")
                self.assertTrue(bool(np.all(np.abs(reference[1] - reference[0])[~same] < 1.0e-3)))
                np.testing.assert_allclose(
                    run[1]["relaxation_energies"], full[1]["relaxation_energies"], rtol=0, atol=1e-6
                )
        # the full grid moves the atom at the origin by round-off forces
        self.assertGreater(float(np.abs(reference[1, 0] - reference[0, 0]).max()), 1.0e-6)

    def test_the_fortran_derived_routines_keep_their_route(self) -> None:
        # Cubic_Spline puts the cell on the Fortran-derived routines, which average their forces over their own
        # operations on the full grid: auto runs them as off does
        settings = {**HYBRID_SETTINGS}
        positions = displaced_positions(origin_shift=0.05)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            full = _run(root / "full", FORCES + "Cubic_Spline: .true.\n", "off", positions=positions, settings=settings)
            kept = _run(root / "kept", FORCES + "Cubic_Spline: .true.\n", "auto", positions=positions, settings=settings)
        for label, (status, _, log, errors) in (("full", full), ("kept", kept)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-3000:]}")
            self.assertIn("Fortran-derived", log)
            self.assertIn(_SYMMETRIZING, log)
            self.assertNotIn(_SECTOR_LINE, log)
            self.assertNotIn(_AVERAGE_LINE, log)
        np.testing.assert_allclose(kept[1]["forces_total"], full[1]["forces_total"], rtol=0, atol=1e-10)
        self.assertAlmostEqual(float(kept[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=1e-10)


if __name__ == "__main__":
    unittest.main()
