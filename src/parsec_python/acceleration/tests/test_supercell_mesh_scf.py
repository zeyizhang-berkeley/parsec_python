"""A perfect supercell at the Gamma point and the same supercell as a smaller cell with k-points
(``PARSEC_SUPERCELL_KPOINTS=auto``, ``SCF/supercell_mesh.py``): the same total energy per atom.

Diamond of 3 x 3 x 3 cubes (216 atoms, 12 points per cubic edge, SCF to 1e-8 Ry) through ``parsec_python.main`` on
one GPU, each run in its own process: the Gamma point of the supercell (its sectors with the three face-centring
translations), the cube with the 3 x 3 x 3 mesh reduced by time reversal alone (the route before 2026-11-19), the
same mesh reduced to the wedge of the cube's operations (the default of the route), and the cell of 1 x 3 x 3
cubes with the 3 x 1 x 1 mesh; and the same supercell with ``Net_Charges: -1`` at the Gamma point and as the cube
with the wedge of the mesh (1/27 of the charge in the cube).
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

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

ROOT = Path(__file__).resolve().parents[4]
DIAMOND = ROOT / "examples" / "3d_diamond_benchmark"
SOURCE = ROOT / "src"


def _energy(text: str) -> float:
    found = re.findall(r"Total Energy =\s+(-?\d+\.\d+)", text)
    if not found:
        raise AssertionError("no total energy in parsec.out")
    return float(found[-1])


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SupercellAsCellWithKPointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="parsec_supercell_mesh_scf_"))
        spacing = 3.567 / 0.529177210903 / 12 * (1.0 - 1.0e-9)
        for folder, extra in ((cls.directory, []), (cls.directory / "charged", ["--set", "Net_Charges: -1"])):
            folder.mkdir(exist_ok=True)
            subprocess.run(
                [sys.executable, str(DIAMOND / "make_diamond.py"), "3", "3", "3", str(folder / "parsec.in"),
                 "--set", f"Grid_Spacing: {spacing:.14f}", "--set", "Convergence_Criterion: 1e-8 Ry",
                 "--set", "Diag_Tolerance: 1e-9", "--set", "Max_Iter: 100", *extra],
                check=True, capture_output=True,
            )
            shutil.copy(DIAMOND / "C_POTRE.DAT", folder / "C_POTRE.DAT")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def _run(self, label: str, switches: dict, source: Path | None = None) -> tuple[float, str]:
        work = self.directory / label
        work.mkdir()
        for name in ("parsec.in", "C_POTRE.DAT"):
            shutil.copy((source or self.directory) / name, work / name)
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("PARSEC_SUPERCELL_KPOINTS", "PARSEC_KPOINT_KINETIC"))
        }
        environment["PYTHONPATH"] = os.pathsep.join([str(SOURCE), environment.get("PYTHONPATH", "")])
        environment.update(switches)
        completed = subprocess.run(
            [sys.executable, "-m", "parsec_python.main", "parsec.in", "--device", "gpu", "--gpus", "1"],
            cwd=work, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, f"{label}: {completed.stderr[-3000:]}")
        return _energy((work / "parsec.out").read_text()), completed.stdout

    def test_the_same_energy_per_atom(self) -> None:
        gamma, _ = self._run("gamma", {})
        on = {"PARSEC_SUPERCELL_KPOINTS": "auto"}
        reversal, note = self._run("reversal", {**on, "PARSEC_SUPERCELL_KPOINTS_WEDGE": "off"})
        self.assertIn("mesh 3 x 3 x 3", note)
        self.assertIn("time reversal alone", note)
        wedge, note = self._run("wedge", on)
        # 12 points per cube edge: the 48 operations of Oh map the grid (Td of the 18-point grid of make_diamond.py)
        self.assertRegex(note, r"irreducible wedge of (24|48) operations")
        slab, note = self._run("mesh311", {**on, "PARSEC_SUPERCELL_KPOINTS_MESH": "3x1x1"})
        self.assertIn("the cell of 1 x 3 x 3 of them", note)
        # 27 cubes: the cube with 27 k-points, the cell of nine cubes with three
        self.assertAlmostEqual(27.0 * reversal, gamma, delta=2.0e-6)
        self.assertAlmostEqual(27.0 * wedge, gamma, delta=2.0e-6)
        self.assertAlmostEqual(wedge, reversal, delta=5.0e-8)
        self.assertAlmostEqual(3.0 * slab, gamma, delta=2.0e-6)

    def test_a_charged_supercell(self) -> None:
        # Net_Charges -1 of the 216 atoms: each cube takes 1/27 of it (with the uniform background the same density
        # and energy per cube); before 2026-11-19 every cube took the whole charge
        source = self.directory / "charged"
        gamma, _ = self._run("charged_gamma", {}, source)
        cube, note = self._run("charged_wedge", {"PARSEC_SUPERCELL_KPOINTS": "auto"}, source)
        self.assertIn("Net_Charges -0.03703703704 in the cell, 1/27 of the supercell's", note)
        self.assertAlmostEqual(27.0 * cube, gamma, delta=2.0e-6)


if __name__ == "__main__":
    unittest.main()
