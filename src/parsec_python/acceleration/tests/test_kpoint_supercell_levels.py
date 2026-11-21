"""The levels of the cell's ``parsec.dat`` and result archive after a mesh solved as its supercell, on the GPU.

``PARSEC_KPOINT_SUPERCELL=auto`` solves the Gamma-centred 2 x 2 x 2 mesh of the silicon cube (and the 2 x 2 x 1 mesh
of a tetragonal cell) as the Gamma point of its supercell; ``PARSEC_KPOINT_SUPERCELL_LEVELS=auto`` gives every level
its k-point by its Bloch phases (``SCF/supercell_levels.py``).  The k-points, weights, levels and occupations of the
files are those of the k-point route with the Bloch kinetic term (the same Hamiltonian), with the supercell in the
weighted sectors (the default), in the abelian ones, in sectors (abelian and weighted) without the translations of the
cell and on the full grid; a non-self-consistent run from either ``parsec.dat`` gives the same levels and Fermi
level.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

BLOCH = {"PARSEC_KPOINT_SUPERCELL": "off", "PARSEC_KPOINT_KINETIC": "bloch"}
#: Levels of a k-point compared: the occupied 16 and four empty ones (the top of the supercell's 192 states and of
#: each k-point's 24 are buffer states of different convergence on the two routes).
COMPARED = 20


def _environment(switches):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("PARSEC_KPOINT_SUPERCELL", "PARSEC_KPOINT_KINETIC", "PARSEC_CUPY_DEVICES",
                                          "PARSEC_ORBITAL_GROUP", "PARSEC_PERIODIC_TRANSLATIONS"))}
    # the 8-atom cube is below the floor of the default route, and its coarse grid above the bound of the estimate
    environment["PARSEC_KPOINT_SUPERCELL_MINIMUM"] = "0"
    environment["PARSEC_KPOINT_SUPERCELL_TOLERANCE"] = "1.0"
    environment["PARSEC_CUPY_DEVICES"] = "0"
    environment.update({name: value for name, value in switches.items() if value is not None})
    source = str(Path(__file__).resolve().parents[3])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def _write(directory: Path, *, mesh=(2, 2, 2), sides=(10.26, 10.26, 10.26), extra: str = "") -> Path:
    from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

    path = _silicon_input(directory, spacing=0.64125, order=12, sides=sides)
    text = path.read_text()
    if tuple(mesh) != (2, 2, 2):
        text = text.replace(" begin Monkhorst_Pack_Grid\n 2 2 2\n", " begin Monkhorst_Pack_Grid\n "
                            + " ".join(str(value) for value in mesh) + "\n")
    path.write_text(text + extra)
    return path


def _run(directory: Path, switches, arguments=()):
    completed = subprocess.run([sys.executable, "-m", "parsec_python", "parsec.in", *arguments], cwd=directory,
                               env=_environment(switches), capture_output=True, text=True)
    return completed


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SupercellLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_trim_levels_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.folder, ignore_errors=True)

    def _pair(self, name, switches, *, mesh=(2, 2, 2), sides=(10.26, 10.26, 10.26), arguments=()):
        from parsec_python.Output.parsec_dat import read_parsec_dat

        found = {}
        for route, extra in (("supercell", switches), ("kpoints", {**switches, **BLOCH})):
            directory = self.folder / f"{name}_{route}"
            _write(directory, mesh=mesh, sides=sides)
            completed = _run(directory, extra, arguments)
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
            out = (directory / "parsec.out").read_text()
            self.assertEqual("supercell of" in completed.stdout, route == "supercell", completed.stdout[-2000:])
            data = read_parsec_dat(directory / "parsec.dat", wavefunctions=False)
            with np.load(directory / "parsec_python_results.npz") as archive:
                stored = {key: np.asarray(archive[key]) for key in ("eigenvalues_ry", "k_weights", "n_kpoints")}
            found[route] = (directory, out, data, stored)
        return found

    def _compare(self, found):
        _, out, cell, cell_archive = found["supercell"]
        _, _, kpoints, kpoints_archive = found["kpoints"]
        self.assertNotIn("dealt to the k-points", out)
        np.testing.assert_allclose(cell.periodic.k_points, kpoints.periodic.k_points, atol=1.0e-12)
        np.testing.assert_allclose(cell.periodic.k_weights, kpoints.periodic.k_weights, atol=1.0e-12)
        self.assertEqual(len(cell.levels[0]), len(kpoints.levels[0]))
        worst = 0.0
        for mine, theirs in zip(cell.levels[0], kpoints.levels[0]):
            np.testing.assert_allclose(mine.energies[:COMPARED], theirs.energies[:COMPARED], rtol=0, atol=2.0e-6)
            np.testing.assert_allclose(mine.occupations[:COMPARED], theirs.occupations[:COMPARED], rtol=0, atol=1.0e-6)
            worst = max(worst, float(np.abs(mine.energies[:COMPARED] - theirs.energies[:COMPARED]).max()))
        count = int(cell_archive["n_kpoints"])
        self.assertEqual(count, int(kpoints_archive["n_kpoints"]))
        np.testing.assert_allclose(cell_archive["k_weights"], kpoints_archive["k_weights"], atol=1.0e-12)
        mine = cell_archive["eigenvalues_ry"].reshape(count, -1)[:, :COMPARED]
        theirs = kpoints_archive["eigenvalues_ry"].reshape(count, -1)[:, :COMPARED]
        np.testing.assert_allclose(np.sort(mine, axis=1), np.sort(theirs, axis=1), rtol=0, atol=2.0e-6)
        return worst

    def _nscf(self, found):
        """A non-self-consistent run from each ``parsec.dat``: the levels and the Fermi level of both."""
        levels = {}
        block = "\n Restart_run: .true.\n begin nscf_kpoints\n 0.0 0.0 0.0 1\n 0.25 0.0 0.0 1\n 0.5 0.5 0.0 1\n end nscf_kpoints\n"
        for route, (directory, _, _, _) in found.items():
            target = directory.parent / f"{directory.name}_nscf"
            shutil.copytree(directory, target)
            text = re.sub(r"(?im)^ *Restart_run:.*$", "", (target / "parsec.in").read_text())
            (target / "parsec.in").write_text(text + block)
            # (the nscf k-points of both with the Bloch kinetic term: only the file differs)
            completed = _run(target, BLOCH)
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
            with np.load(target / "parsec_python_nscf.npz") as archive:
                levels[route] = (float(archive["fermi_level_ry"]), np.asarray(archive["nscf_eigenvalues_ry"]))
        self.assertAlmostEqual(levels["supercell"][0], levels["kpoints"][0], delta=2.0e-6)
        np.testing.assert_allclose(levels["supercell"][1], levels["kpoints"][1], rtol=0, atol=2.0e-6)

    def test_weighted_sectors(self) -> None:
        found = self._pair("weighted", {})
        print(f"\n weighted sectors: largest level difference {self._compare(found):.2e} Ry")
        self._nscf(found)

    def test_abelian_sectors(self) -> None:
        print(f"\n abelian sectors: {self._compare(self._pair('abelian', {'PARSEC_ORBITAL_GROUP': 'abelian'})):.2e} Ry")

    def test_sectors_without_the_translations(self) -> None:
        # sectors about the origin alone: the orbitals formed on the full grid tell the k-points (the abelian ones
        # state by state, the weighted ones the star of every solved state, over which its partners are spread)
        found = self._pair("notrans", {"PARSEC_PERIODIC_TRANSLATIONS": "off", "PARSEC_ORBITAL_GROUP": "abelian"})
        print(f"\n abelian sectors without the translations: {self._compare(found):.2e} Ry")
        found = self._pair("notransw", {"PARSEC_PERIODIC_TRANSLATIONS": "off"})
        print(f"\n weighted sectors without the translations: {self._compare(found):.2e} Ry")

    def test_the_full_grid(self) -> None:
        found = self._pair("full", {}, arguments=("--symmetry", "off"))
        print(f"\n full grid: {self._compare(found):.2e} Ry")

    def test_a_tetragonal_cell_with_two_points_on_two_axes(self) -> None:
        # (equal steps: the Bloch kinetic term of the k-points takes a box whose rows are its points)
        found = self._pair("tetragonal", {}, mesh=(2, 2, 1), sides=(10.26, 10.26, 15.39))
        print(f"\n tetragonal 2 x 2 x 1: {self._compare(found):.2e} Ry")


if __name__ == "__main__":
    unittest.main()
