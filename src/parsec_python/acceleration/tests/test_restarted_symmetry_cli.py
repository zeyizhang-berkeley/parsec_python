"""The k-points of a restarted structure on the command line (``parsec_python.Input.restarted_symmetry``).

The 8-atom silicon cube of ``examples/3d_Si`` with the Gamma-centred 2 x 2 x 2 mesh on the k-points
(``PARSEC_KPOINT_SUPERCELL=off``), 16 points per edge, ``Expansion_Order`` 12, on one GPU:

* ``--restart-from`` an XYZ file with atom 0 moved 0.1 bohr: the wedge and the density average of the restarted atoms
  give the energy of the k-points without any group on the same grid (``Ignore_Symmetry`` with ``Grid_Shift`` 0.5,
  since ``Ignore_Symmetry`` alone puts a point on the origin), with both kinetic forms.  Before 2026-11-20 the restart
  kept the 48 operations of the input: 8.9 mRy per cell too low with the Bloch form.
* a ``relax_restart.dat`` of the input's own atoms: the wedge of the input and the final energy of the run without
  the restart, to the printed digit.
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
from parsec_python.tests.parsec15_tables import start_parsec15_tables

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

CODATA_ANGSTROM = 0.529177210903


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class RestartedKpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # energies pinned with PARSEC 1.5's tables
        start_parsec15_tables(cls)
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_restartgroup_cli_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.folder, ignore_errors=True)

    @staticmethod
    def _environment(switches):
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("PARSEC_KPOINT_SUPERCELL", "PARSEC_KPOINT_KINETIC", "PARSEC_CUPY_DEVICES")}
        environment["PARSEC_KPOINT_SUPERCELL"] = "off"
        environment["PARSEC_CUPY_DEVICES"] = "0"
        environment.update(switches)
        source = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return environment

    def _input(self, name, extra=()):
        from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

        directory = self.folder / name
        _silicon_input(directory, spacing=0.64125, order=12, extra=extra)
        return directory

    def _run(self, directory, switches, arguments=()):
        completed = subprocess.run([sys.executable, "-m", "parsec_python", "parsec.in", *arguments], cwd=directory,
                                   env=self._environment(switches), capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
        out = (directory / "parsec.out").read_text()
        totals = [float(value) for value in re.findall(r"Total Energy =\s+(-?\d+\.\d+)", out)]
        lines = [line for line in out.splitlines() if "Total Energy =" in line or "Energy/atom" in line]
        return out, totals, lines

    def _moved_xyz(self, directory):
        from parsec_python.Input import parse_parsec_input

        atoms = parse_parsec_input(directory / "parsec.in").problem.atoms
        rows = [f"Si {(x + (0.1 if index == 0 else 0.0)) * CODATA_ANGSTROM:.10f} {y * CODATA_ANGSTROM:.10f} "
                f"{z * CODATA_ANGSTROM:.10f}" for index, (x, y, z) in enumerate(atom.position for atom in atoms)]
        (directory / "moved.xyz").write_text("\n".join([str(len(rows)), "moved", *rows]) + "\n")

    def test_a_moved_structure_takes_its_own_wedge(self) -> None:
        no_group = (" Ignore_Symmetry: .true.", " begin Grid_Shift", " 0.5 0.5 0.5", " end Grid_Shift")
        for kinetic in ("bloch", "gradient"):
            with self.subTest(kinetic=kinetic):
                switches = {"PARSEC_KPOINT_KINETIC": kinetic}
                restart = self._input(f"restart_{kinetic}")
                reference = self._input(f"nogroup_{kinetic}", extra=no_group)
                for directory in (restart, reference):
                    self._moved_xyz(directory)
                out, restarted, _ = self._run(restart, switches, ("--restart-from", "moved.xyz"))
                self.assertIn("another symmetry", out)
                self.assertNotIn("by 48 operations", out)
                _, without, _ = self._run(reference, switches, ("--restart-from", "moved.xyz"))
                self.assertAlmostEqual(restarted[-1], without[-1], delta=2.0e-7)
                if kinetic == "bloch":
                    # the TRIM measurement of the supercell and of the k-points without a group (2026-11-19, at a
                    # criterion of 1e-7 Ry; the stale group gave -63.69524890)
                    self.assertAlmostEqual(restarted[-1], -63.68637552, delta=1.0e-6)

    def test_an_unmoved_restart_keeps_its_energies(self) -> None:
        from parsec_python.Input import parse_parsec_input
        from parsec_python.Relaxation.files import RelaxRestart, format_relax_restart

        plain = self._input("plain")
        plain_out, totals, _ = self._run(plain, {})
        restart = self._input("unmoved")
        positions = np.asarray([atom.position for atom in parse_parsec_input(restart / "parsec.in").problem.atoms])
        (restart / "same.relax_restart.dat").write_text(
            format_relax_restart(RelaxRestart(positions=positions, iteration=1, movable_atoms=len(positions))))
        out, restarted, _ = self._run(restart, {}, ("--restart-from", "same.relax_restart.dat"))
        self.assertIn("replaced by coordinates in same.relax_restart.dat", out)
        self.assertNotIn("another symmetry", out)
        # the wedge note of the input (48 operations) in both reports
        notes = [line for line in plain_out.splitlines() if "operations of the space group" in line]
        self.assertTrue(notes)
        self.assertEqual(notes, [line for line in out.splitlines() if "operations of the space group" in line])
        # The final energy to the printed digit: the GPU's sums are not bitwise reproducible from one run to the next
        # (an intermediate iteration may differ in the last printed digit, between two runs of the plain input too);
        # that the restarted input is the input's, value for value, is pinned by tests/unit/test_restarted_symmetry.py
        # and, bit for bit on one CPU thread, by the A100 checks of 2026-11-20.
        self.assertAlmostEqual(restarted[-1], totals[-1], delta=1.0e-8)


if __name__ == "__main__":
    unittest.main()
