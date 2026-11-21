"""The Bloch block of the finite-difference Laplacian as the kinetic term of a k-point (``PARSEC_KPOINT_KINETIC``).

``build_bloch_negative_laplacian`` gives the exact Bloch block of the supercell's matrix: the two gauges are
unitarily related, the spectra of a mesh of ``m`` points along each axis are together the spectrum of the
supercell of ``m`` cells, and the SCF of a cell with that mesh is the Gamma-point SCF of the supercell.  On the
device the sparse route and the stencil kernel apply the Hamiltonian of the host.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian, kpoint_kinetic_setting
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


def _box(side: float = 6.0, points: int = 9, order: int = 4):
    return build_periodic_grid(
        PeriodicCell(side * np.eye(3)), PeriodicGridSettings(spacing=side / points, expansion_order=order, shift=(0.5,) * 3)
    )


class BlochLaplacianTests(unittest.TestCase):
    def test_the_two_gauges_and_gamma(self) -> None:
        grid = _box()
        k = np.array([0.3, -0.7, 1.1])
        u_gauge = build_bloch_negative_laplacian(grid, k, "u")
        psi_gauge = build_bloch_negative_laplacian(grid, k, "psi")
        phases = sp.diags(np.exp(1j * np.asarray(grid.coordinates) @ k))
        self.assertLess(abs(phases.conj() @ psi_gauge @ phases - u_gauge).max(), 1.0e-13)
        self.assertEqual(abs(u_gauge - u_gauge.conj().T).max(), 0.0)
        self.assertEqual(abs(build_bloch_negative_laplacian(grid, np.zeros(3)) - build_negative_laplacian(grid)).max(), 0.0)
        # a time-reversal invariant k: the twisted Laplacian is real
        self.assertEqual(build_bloch_negative_laplacian(grid, np.pi / 6.0 * np.array([1.0, 0.0, 1.0]), "psi").dtype, np.float64)

    def test_the_mesh_spectra_are_the_spectrum_of_the_supercell(self) -> None:
        side, points = 6.0, 9
        cell = _box(side, points)
        supercell = build_periodic_grid(
            PeriodicCell(2 * side * np.eye(3)),
            PeriodicGridSettings(spacing=side / points, expansion_order=4, shift=(0.5,) * 3),
        )
        full = np.sort(np.linalg.eigvalsh(build_negative_laplacian(supercell).toarray()))
        mesh = np.sort(np.concatenate([
            np.linalg.eigvalsh(build_bloch_negative_laplacian(cell, np.pi / side * np.array(bits, float)).toarray())
            for bits in np.ndindex(2, 2, 2)
        ]))
        np.testing.assert_allclose(mesh, full, rtol=0, atol=1.0e-11)

    def test_the_switch(self) -> None:
        for value, expected in (("", "gradient"), ("Bloch", "bloch"), ("AUTO", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": value}):
                self.assertEqual(kpoint_kinetic_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "u"}):
            with self.assertRaises(ValueError):
                kpoint_kinetic_setting()


class BlochSCFTests(unittest.TestCase):
    def test_a_mesh_is_the_gamma_point_of_the_supercell(self) -> None:
        # the conventional cube of silicon with the 2 x 2 x 2 mesh of the time-reversal invariant k (Bloch
        # kinetic term) against the cube of eight cubes at Gamma, SCF to 1e-9 Ry; on the device where the
        # acceleration package runs the k-points (the pooled solver)
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf as run_scf_gamma
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
            base = replace(base, scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
                           eigensolver=replace(base.eigensolver, tolerance=1.0e-10))
            side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
            k_points = np.array([np.array(b, dtype=float) * np.pi / side for b in np.ndindex(2, 2, 2)])
            weights = np.full(8, 1.0 / 8.0)
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "bloch"}):
                mesh = run_scf_kpoints(prepare_periodic_single_point(base), k_points, weights)
            lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=float)
            atoms = [
                Atom(atom.symbol, tuple(np.asarray(atom.position) + np.array(shift, float) @ lattice))
                for shift in np.ndindex(2, 2, 2) for atom in base.atoms
            ]
            supercell = replace(
                base, atoms=atoms, periodic_cell=replace(base.periodic_cell, lattice_vectors=2.0 * lattice),
                scf=replace(base.scf, number_of_states=8 * base.scf.number_of_states),
            )
            gamma = run_scf_gamma(prepare_periodic_single_point(supercell))
        self.assertTrue(mesh.converged and gamma.converged)
        self.assertAlmostEqual(mesh.energies.total, gamma.energies.total / 8.0, delta=1.0e-8)


class MeshAndSupercellTests(unittest.TestCase):
    """``SCF/kpoint_supercell.py``: a mesh of the time-reversal invariant k-points as the supercell, and back."""

    def test_the_two_transforms(self) -> None:
        from parsec_python.SCF import kpoint_supercell as transforms
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
        mesh = replace(base, monkhorst_pack_dimensions=(2, 2, 2), monkhorst_pack_shift=(0.0, 0.0, 0.0))
        switches = {"PARSEC_KPOINT_SUPERCELL": "auto", "PARSEC_SUPERCELL_KPOINTS": "auto"}
        with mock.patch.dict(os.environ, switches):
            supercell, repeats = transforms.supercell_of_trim_mesh(mesh)
            self.assertEqual(repeats, (2, 2, 2))
            self.assertEqual(len(supercell.atoms), 8 * len(base.atoms))
            self.assertEqual(supercell.scf.number_of_states, 8 * base.scf.number_of_states)
            np.testing.assert_array_equal(transforms._grid_counts(supercell), 2 * transforms._grid_counts(base))
            self.assertFalse(supercell.has_kpoint_sampling)
            # a mesh of k = +-1/4 (no shift) is not time-reversal invariant
            self.assertIsNone(transforms.supercell_of_trim_mesh(replace(base, monkhorst_pack_dimensions=(2, 2, 2))))
            cell, back = transforms.kpoints_of_supercell(supercell)
            self.assertEqual(back, (2, 2, 2))
            self.assertEqual(len(cell.atoms), len(base.atoms))
            self.assertEqual(cell.scf.number_of_states, base.scf.number_of_states + 6)
            np.testing.assert_allclose(cell.periodic_cell.lattice_vectors, base.periodic_cell.lattice_vectors)
            self.assertEqual((cell.monkhorst_pack_dimensions, cell.monkhorst_pack_shift), ((2, 2, 2), (0.0, 0.0, 0.0)))
            # the face-centred cube itself repeats no smaller orthorhombic cell
            self.assertIsNone(transforms.kpoints_of_supercell(base))
            # the supercell of the mesh needs no kinetic setting; the mesh of the supercell is refused under an
            # explicit gradient form (it reproduces the supercell with the Bloch form only)
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "gradient"}):
                self.assertIsNotNone(transforms.supercell_of_trim_mesh(mesh))
                self.assertIsNone(transforms.kpoints_of_supercell(supercell))
        # auto by default since 2026-11-19 (the supercell where it applies); off keeps the k-points
        self.assertIsNotNone(transforms.supercell_of_trim_mesh(mesh))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SUPERCELL": "off"}):
            self.assertIsNone(transforms.supercell_of_trim_mesh(mesh))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DefaultTrimRouteTests(unittest.TestCase):
    """``PARSEC_KPOINT_SUPERCELL=auto`` (the default) on the command line: a single point of silicon with the
    Gamma-centred 2 x 2 x 2 mesh in a process of one GPU runs as the Gamma point of the supercell, reports the
    energies of the cell in its energy lines (those of the k-points with the Bloch form), writes the ``parsec.dat``
    and the result archive of the cell, which a ``Restart_Run`` (the k-points) and ``--bands-from`` read, takes a
    charged cell and a restarted structure as the k-points do, keeps the k-points with several GPUs in the process,
    and ``off`` keeps the k-points."""

    @staticmethod
    def _environment(switches):
        from pathlib import Path

        environment = {key: value for key, value in os.environ.items()
                       if key not in ("PARSEC_KPOINT_SUPERCELL", "PARSEC_KPOINT_KINETIC", "PARSEC_CUPY_DEVICES",
                                      "PARSEC_KPOINT_SUPERCELL_GPUS")}
        # the 8-atom cube is below the floor of the default route (PARSEC_KPOINT_SUPERCELL_MINIMUM, 2^24 entries)
        environment["PARSEC_KPOINT_SUPERCELL_MINIMUM"] = "0"
        # one GPU in the process: the default route keeps the k-points with more (PARSEC_KPOINT_SUPERCELL_GPUS)
        environment["PARSEC_CUPY_DEVICES"] = "0"
        for name, value in switches.items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        source = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return environment

    def _call(self, directory, switches, arguments=("parsec.in",)):
        import subprocess
        import sys

        completed = subprocess.run([sys.executable, "-m", "parsec_python", *arguments], cwd=directory,
                                   env=self._environment(switches), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
        return completed

    def _run(self, directory, switches, *, restart: bool = False, fortran: bool = False, charge: int = 0,
             arguments=()):
        import re

        from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

        if not (directory / "parsec.in").is_file():
            _silicon_input(directory, spacing=0.64125, order=12)
        text = (directory / "parsec.in").read_text()
        text = re.sub(r"(?im)^ *Restart_run:.*$", f" Restart_run: {'.true.' if restart else '.false.'}", text)
        text = re.sub(r"(?im)^ *Net_charges:.*$", f" Net_charges: {charge}", text)
        if fortran and "Use_Fortran_Routines" not in text:
            text = text.replace(" Restart_run:", " Use_Fortran_Routines: .true.\n Restart_run:", 1)
        (directory / "parsec.in").write_text(text)
        completed = self._call(directory, switches, ("parsec.in", *arguments))
        out = (directory / "parsec.out").read_text()
        totals = [float(value) for value in re.findall(r"Total Energy =\s+(-?\d+\.\d+)", out)]
        cell = re.findall(r"Total energy of the input cell .*= +(-?\d+\.\d+)", out)
        return completed.stdout, out, totals, (float(cell[-1]) if cell else None)

    @staticmethod
    def _supercell_total(out):
        import re

        found = re.findall(r"Total energy of the supercell of 8 cells .*= +(-?\d+\.\d+)", out)
        return float(found[-1]) if found else None

    @staticmethod
    def _electrons(out):
        import re

        return float(re.findall(r"Number of electrons =\s+(-?\d+\.\d+)", out)[-1])

    def _bands_from(self, directory):
        """``--bands-from`` of the archive of the run in ``directory`` with the input of the cell and a path."""
        from parsec_python.Output.bands import load_band_structure_archive

        path = ("begin bandstruc\n1 -0.25 0.0 0.0  0.25 0.0 0.0  left-right\nend bandstruc\n"
                "bandstruc_points 2\n")
        (directory / "bands.in").write_text((directory / "parsec.in").read_text() + path)
        completed = self._call(directory, {}, ("bands.in", "--bands-from", "parsec_python_results.npz", "--quiet"))
        self.assertIn("no SCF", (directory / "parsec_bands.out").read_text(), completed.stdout[-2000:])
        return load_band_structure_archive(directory / "parsec_python_bands.npz")

    def test_the_default_route_and_its_files(self) -> None:
        import tempfile
        from pathlib import Path

        from parsec_python.Output.parsec_dat import read_parsec_dat

        folder = Path(tempfile.mkdtemp(prefix="parsec_trim_cli_"))
        stdout, out, totals, per_cell = self._run(folder / "auto", {})
        self.assertIn("supercell of 8 cells", stdout)
        self.assertIn("supercell of 8 cells", out)
        self.assertIsNotNone(per_cell)
        # the energy lines are those of the cell, the supercell's total closes the report
        self.assertAlmostEqual(per_cell, totals[-1], places=7)
        self.assertAlmostEqual(self._supercell_total(out), 8.0 * per_cell, places=6)
        self.assertEqual(self._electrons(out), 8 * 32.0)
        data = read_parsec_dat(folder / "auto" / "parsec.dat", wavefunctions=False)
        self.assertEqual(int(data.ndim), 16 ** 3)
        self.assertEqual(len(data.levels[0]), 8)
        with np.load(folder / "auto" / "parsec_python_results.npz") as archive:
            self.assertEqual(archive["atom_coordinates_bohr"].shape, (8, 3))
            self.assertEqual(archive["output_effective_potential_ry"].size, 16 ** 3)
            self.assertAlmostEqual(float(archive["energy_total_ry"]), per_cell, places=7)
        # the k-points with the Bloch form: the same energy per cell
        _, _, bloch, none = self._run(folder / "bloch", {"PARSEC_KPOINT_SUPERCELL": "off",
                                                         "PARSEC_KPOINT_KINETIC": "bloch"})
        self.assertIsNone(none)
        self.assertAlmostEqual(per_cell, bloch[-1], delta=2.0e-7)
        # --bands-from of the cell's input reads the archive of either: the same bands
        bands = self._bands_from(folder / "auto")
        reference = self._bands_from(folder / "bloch")
        np.testing.assert_allclose(bands.eigenvalues, reference.eigenvalues, atol=2.0e-6)
        self.assertAlmostEqual(bands.fermi_level, reference.fermi_level, delta=2.0e-6)
        # off: the k-points with the gradient form, 3.2e-5 Ry per atom above (2.7e-5 by the estimate)
        stdout_off, _, gradient, _ = self._run(folder / "off", {"PARSEC_KPOINT_SUPERCELL": "off"})
        self.assertNotIn("supercell", stdout_off)
        self.assertGreater(gradient[-1] - per_cell, 0.0)
        self.assertLess((gradient[-1] - per_cell) / 8.0, 1.0e-4)
        # Restart_Run of the cell from the file of the supercell: the k-points on the Fortran-derived routines (which
        # a Restart_Run selects), from the converged density: the energy of that route from the start, in fewer
        # iterations
        _, _, fortran, _ = self._run(folder / "fortran", {}, fortran=True)
        stdout_restart, _, restarted, _ = self._run(folder / "auto", {}, restart=True)
        self.assertIn("Restart_Run", stdout_restart)
        self.assertAlmostEqual(restarted[-1], fortran[-1], delta=2.0e-6)
        self.assertLess(len(restarted), len(fortran))

    def test_a_charged_cell_and_a_restarted_structure(self) -> None:
        import tempfile
        from pathlib import Path

        from parsec_python.Input import parse_parsec_input
        from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

        folder = Path(tempfile.mkdtemp(prefix="parsec_trim_cli_"))
        bloch = {"PARSEC_KPOINT_SUPERCELL": "off", "PARSEC_KPOINT_KINETIC": "bloch"}
        # Net_Charges is the charge of one cell: the supercell holds 8 x 31 electrons, the energy that of the
        # k-points with the Bloch form
        stdout, out, totals, per_cell = self._run(folder / "charged", {}, charge=1)
        self.assertIn("supercell of 8 cells", stdout)
        self.assertEqual(self._electrons(out), 8 * 31.0)
        _, out_k, charged, _ = self._run(folder / "charged_k", bloch, charge=1)
        self.assertEqual(self._electrons(out_k), 31.0)
        self.assertAlmostEqual(per_cell, charged[-1], delta=2.0e-7)
        self.assertAlmostEqual(per_cell, totals[-1], places=7)
        # --restart-from of a structure of the cell: the supercell is made of its atoms
        directory = folder / "moved"
        _silicon_input(directory, spacing=0.64125, order=12)
        atoms = parse_parsec_input(directory / "parsec.in").problem.atoms
        angstrom = 0.529177210903
        rows = [f"Si {(x + (0.1 if index == 0 else 0.0)) * angstrom:.10f} {y * angstrom:.10f} {z * angstrom:.10f}"
                for index, (x, y, z) in enumerate(atom.position for atom in atoms)]
        (directory / "cell.xyz").write_text("\n".join([str(len(rows)), "moved", *rows]) + "\n")
        (folder / "moved_k").mkdir()
        for name in ("Si_POTRE.DAT", "cell.xyz"):
            (folder / "moved_k" / name).write_bytes((directory / name).read_bytes())
        # the k-points without the space group of the input's atoms, which the moved atom breaks (the k-points keep
        # that group after a restart and would reduce the mesh and average the density with it), on the grid of the
        # input (Ignore_Symmetry alone would put a point on the origin)
        (folder / "moved_k" / "parsec.in").write_text(
            (directory / "parsec.in").read_text()
            + " Ignore_Symmetry: .true.\n begin Grid_Shift\n 0.5 0.5 0.5\n end Grid_Shift\n"
        )
        stdout, out, totals, moved = self._run(directory, {}, arguments=("--restart-from", "cell.xyz"))
        self.assertIn("supercell of 8 cells", stdout)
        self.assertIn("cell.xyz", out)
        _, _, moved_k, _ = self._run(folder / "moved_k", bloch, arguments=("--restart-from", "cell.xyz"))
        self.assertAlmostEqual(moved, moved_k[-1], delta=2.0e-7)
        # the moved atom raises the energy (the restart was applied; about 1.3 mRy, harmonic in the displacement)
        _, _, still, _ = self._run(folder / "still", bloch)
        self.assertGreater(moved - still[-1], 1.0e-4)

    def test_several_gpus_keep_the_kpoints(self) -> None:
        import tempfile
        from pathlib import Path

        from parsec_python.acceleration.backends.cupy import cupy_device_count
        from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

        folder = Path(tempfile.mkdtemp(prefix="parsec_trim_cli_"))
        _silicon_input(folder, spacing=0.64125, order=12)
        if cupy_device_count() > 1:
            # every visible GPU (the default of PARSEC_CUPY_DEVICES): the pools of the k-points divide them
            completed = self._call(folder, {"PARSEC_CUPY_DEVICES": None}, ("parsec.in", "--dry-run"))
            self.assertIn("GPUs in one process", completed.stdout)
            completed = self._call(folder, {"PARSEC_CUPY_DEVICES": "0,1"}, ("parsec.in", "--dry-run"))
            self.assertIn("2 GPUs in one process", completed.stdout)
            completed = self._call(folder, {"PARSEC_CUPY_DEVICES": "0,1", "PARSEC_KPOINT_SUPERCELL_GPUS": "2"},
                                   ("parsec.in", "--dry-run"))
            self.assertIn("supercell of 8 cells", completed.stdout)
        completed = self._call(folder, {"PARSEC_CUPY_DEVICES": "0"}, ("parsec.in", "--gpus", "1", "--dry-run"))
        self.assertIn("supercell of 8 cells", completed.stdout)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceBlochKineticTests(unittest.TestCase):
    def test_the_device_operators_apply_the_host_hamiltonian(self) -> None:
        import cupy as cp

        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
        from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture

        _PeriodicFixture.setUpClass()
        fx = _PeriodicFixture
        side = float(np.asarray(fx.cell.lattice_vectors)[0, 0])
        rng = np.random.default_rng(7)
        block = rng.normal(size=(fx.grid.size, 5)) + 1j * rng.normal(size=(fx.grid.size, 5))
        for label, k in (("general", fx.k_point), ("time-reversal invariant", np.pi / side * np.array([1.0, 0.0, 1.0]))):
            host = KPointKohnShamHamiltonian(
                fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, k,
                bloch_laplacian=build_bloch_negative_laplacian(fx.grid, k, "u"),
            )
            expected = host.apply(block)
            scale = float(np.max(np.abs(expected)))
            for fused in (False, True):
                converter = DeviceBlochConverter(CupyArrayBackend(0))
                converter.fused_stencil = fused
                operator = converter.operator(host)
                if fused:
                    self.assertIsNotNone(operator.stencil, label)
                result = cp.asnumpy(operator.apply(cp.asarray(block)))
                self.assertLess(float(np.max(np.abs(result - expected))), 1.0e-12 * scale, f"{label} fused={fused}")


if __name__ == "__main__":
    unittest.main()
