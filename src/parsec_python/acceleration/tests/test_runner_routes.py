"""Routes of the MPI full-SCF runner decided from the input (CPU tests).

* An unpolarized hybrid of a bulk cell at the Gamma point runs its SCF on the ranks (``driver._mpi_periodic_hybrid``):
  the abelian sectors where its basis reaches the floor of the sectors, the one sector of the identity below it and
  under ``PARSEC_GAMMA_HYBRID_SECTORS=off``; ``kpoint``, spin and its forces stay refused
  (``benchmarks/mpi_full_scf._refuse_unsupported_periodic``).
* Molecular dynamics and manual movements take the plan of ``PARSEC_MOVING_SECTORS`` on the layouts that hold one
  sector of the identity (``benchmarks/mpi_full_scf._moving_plan``).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HYBRIDS = ROOT / "examples" / "3d_hybrid_functionals"


def _silicon_hybrid(directory: Path, cells: int, extra: str = "") -> Path:
    """The Si PBE0 cell of ``examples/3d_hybrid_functionals/Si64_gamma`` with ``cells`` conventional cells per edge."""

    side = 10.26 * cells
    base = np.array([[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0], [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1]], float)
    atoms = [(row + 4 * np.array([i, j, k])) * 2.565 for i in range(cells) for j in range(cells) for k in range(cells)
             for row in base]
    lines = [
        "Periodic_System: .true.", "Boundary_Conditions: bulk", "begin Cell_Shape", f"  {side} 0.0 0.0",
        f"  0.0 {side} 0.0", f"  0.0 0.0 {side}", "end Cell_Shape", "Lattice_Vector_Scale: 1.0 bohr",
        "Coordinate_Unit: Cartesian_Bohr", "Grid_Spacing: 0.300", "Expansion_Order: 12", "Correlation_Type: pbe0",
        f"States_Num: {20 * cells ** 3 + 4}", "Fermi_Temp: 1 K", extra, "Atom_Types_Num: 1", "Atom_Type: Si",
        "Pseudopotential_Format: fhipp", "Potential_Num: 3", "Local_Component: d", "Core_Cutoff_Radius: 2.1",
        "begin Electron_Per_Orbital", "2 2 0", "end Electron_Per_Orbital", "begin Atom_Coord",
    ]
    lines += [f"  {x:.10f} {y:.10f} {z:.10f}" for x, y, z in atoms]
    lines += ["end Atom_Coord"]
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(HYBRIDS / "Si_FHIPP.DAT", directory / "Si_FHIPP.DAT")
    (directory / "parsec.in").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory / "parsec.in"


def _parse(path: Path):
    from parsec_python.Input import parse_parsec_input

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


class PeriodicHybridRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_sectors_above_the_floor_and_the_identity_below(self) -> None:
        from parsec_python.acceleration.driver import _mpi_periodic_hybrid

        small = _parse(_silicon_hybrid(self.folder / "si8", 1)).problem
        large = _parse(_silicon_hybrid(self.folder / "si64", 2)).problem
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "auto"}):
            # 34^3 points x 24 states is below 2^24, 68^3 x 164 above it
            self.assertIs(_mpi_periodic_hybrid(small, object()), False)
            self.assertIs(_mpi_periodic_hybrid(large, object()), True)
            # one process decides elsewhere (driver._gamma_hybrid_in_sectors)
            self.assertIsNone(_mpi_periodic_hybrid(large, None))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "off"}):
            self.assertIs(_mpi_periodic_hybrid(large, object()), False)
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "kpoint"}):
            self.assertIsNone(_mpi_periodic_hybrid(large, object()))

    def test_a_spin_hybrid_is_not_taken(self) -> None:
        from parsec_python.acceleration.driver import _mpi_periodic_hybrid

        problem = _parse(_silicon_hybrid(self.folder / "spin", 1, "Spin_Polarization: true")).problem
        self.assertIsNone(_mpi_periodic_hybrid(problem, object()))

    def test_the_runner_takes_the_scf_and_refuses_the_forces(self) -> None:
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _refuse_unsupported_periodic

        scf = _parse(_silicon_hybrid(self.folder / "scf", 1))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "auto"}):
            _refuse_unsupported_periodic(scf)
        forces = _parse(_silicon_hybrid(self.folder / "forces", 1, "Skip_Force: .false."))
        with self.assertRaisesRegex(ValueError, "hybrid functional are not available in the full-SCF runner"):
            _refuse_unsupported_periodic(forces)
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_HYBRID_SECTORS": "kpoint"}):
            with self.assertRaisesRegex(ValueError, "a hybrid functional of a periodic cell is not available"):
                _refuse_unsupported_periodic(scf)


class MovingPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def _ge5_dynamics(self, extra: str = ""):
        """Ge5 (Td) restarted from a breathing state: every operation kept (as the moving-sector tests)."""

        from parsec_python.acceleration.tests.test_move_flag_subgroup import GE5, _ge5_input
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.MolecularDynamics.run import restart_structure

        block = ("Molecular_Dynamics: .true.\nStep_num: 2\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\n"
                 "Time_Step: 40.0\nRestart_mode: .true.\n")
        path = _ge5_input(self.folder, extra=block + extra)
        current, previous = GE5 * 1.002, GE5 * 1.0015
        state = MolecularDynamicsState(
            masses=np.asarray(atomic_masses(["Ge"] * 5), dtype=float), movable=np.ones(5, dtype=np.int64),
            current_positions=current, previous_positions=previous, velocities=current - previous,
            accelerations=np.zeros((5, 3)), temperature=300.0, frame=3, move_count=0, restarted=True,
        )
        (self.folder / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")
        translation, restart, _notes = restart_structure(_parse(path), path.parent)
        return translation, restart

    def test_the_layouts_that_hold_the_sector_of_the_identity(self) -> None:
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _moving_plan

        translation, restart = self._ge5_dynamics()
        keywords = dict(dynamics_restart=restart, hybrid=False, serial_control=False)
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on"}):
            plan = _moving_plan(translation, None, one_sector=True, **keywords)
            self.assertIsNotNone(plan)
            self.assertEqual((plan.kind, plan.periodic, plan.order), ("molecular_dynamics", False, 24))
            # one rank per node on several nodes, a hybrid and the serial control keep what they did before
            self.assertIsNone(_moving_plan(translation, None, one_sector=False, **keywords))
            self.assertIsNone(_moving_plan(translation, None, one_sector=True, dynamics_restart=restart, hybrid=True,
                                           serial_control=False))
            self.assertIsNone(_moving_plan(translation, None, one_sector=True, dynamics_restart=restart, hybrid=False,
                                           serial_control=True))
            self.assertIsNone(_moving_plan(translation, "off", one_sector=True, **keywords))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "off"}):
            self.assertIsNone(_moving_plan(translation, None, one_sector=True, **keywords))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "auto"}):
            # the full grid of Ge5 times its states is below the 2^24 of auto
            self.assertIsNone(_moving_plan(translation, None, one_sector=True, **keywords))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "sometimes"}):
            with self.assertRaises(ValueError):
                _moving_plan(translation, None, one_sector=True, **keywords)

    def test_two_spin_channels_keep_the_sector_of_the_identity(self) -> None:
        # PARSEC_MOVING_SPIN_SECTORS takes them into the sectors in one process only: the ranks keep the one sector of
        # the identity of both channels under every value, and still refuse a value it does not know
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _moving_plan

        keywords = dict(hybrid=False, serial_control=False, one_sector=True)
        SPIN = "Spin_Polarization: .true.\n"
        for setting in ("auto", "on", "off"):
            with self.subTest(setting=setting), mock.patch.dict(
                os.environ, {"PARSEC_MOVING_SECTORS": "on", "PARSEC_MOVING_SPIN_SECTORS": setting}
            ):
                translation, restart = self._ge5_dynamics(SPIN)
                self.assertTrue(translation.problem.scf.spin_polarized)
                self.assertIsNone(_moving_plan(translation, None, dynamics_restart=restart, **keywords))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on", "PARSEC_MOVING_SPIN_SECTORS": "maybe"}):
            translation, restart = self._ge5_dynamics(SPIN)
            with self.assertRaises(ValueError):
                _moving_plan(translation, None, dynamics_restart=restart, **keywords)

    def test_a_single_point_has_no_plan(self) -> None:
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _moving_plan
        from parsec_python.acceleration.tests.test_move_flag_subgroup import _ge5_input

        translation = _parse(_ge5_input(self.folder / "single"))
        with mock.patch.dict(os.environ, {"PARSEC_MOVING_SECTORS": "on"}):
            self.assertIsNone(_moving_plan(translation, None, dynamics_restart=None, one_sector=True, hybrid=False,
                                           serial_control=False))


if __name__ == "__main__":
    unittest.main()
