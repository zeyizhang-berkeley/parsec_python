"""Fractional pairs of degenerate levels of one label of the abelian spin sectors (``PARSEC_SPIN_GUARD_DOUBLETS``).

The abelian sectors hold the e_g of Oh in one sector (A_g of D2h); test 3 of the spin guard counts a fractional
multiplet only where its levels lie in two or more sectors, so the symmetric ensemble of such a pair passes.  bcc Mn2
with start polarizations +0.5 and -0.5 (PBE, 300 K) keeps a half-filled pair in each channel there, a state the full
grid leaves for a lower one.  ``note`` (the default) keeps the result and says so; ``stop`` sends the run to the full
grid.  CPU tests of the detector and the NOTE; the route on a CUDA device.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


def _channel(values, filled, sectors):
    return SimpleNamespace(
        eigenvalues=np.asarray(values, dtype=float),
        occupations=np.asarray(filled, dtype=float),
        representations=np.asarray(sectors, dtype=np.int64),
    )


class DetectorTests(unittest.TestCase):
    def test_only_fractional_pairs_of_one_label_count(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import SECTOR_DOUBLET_REASON, fractional_sector_doublets

        pair = _channel([-0.6, -0.282, -0.282, 0.1], [1.0, 0.5, 0.5, 0.0], [1, 1, 1, 2])
        report = fractional_sector_doublets([pair, pair])
        self.assertEqual(report.reason, SECTOR_DOUBLET_REASON)
        self.assertTrue(report.doublet)
        self.assertAlmostEqual(report.largest, 0.5)
        self.assertEqual(len(report.lines), 2)
        # partners in two sectors (test 3 counts those), integer occupations, levels apart: none
        self.assertIsNone(fractional_sector_doublets([_channel([-0.6, -0.282, -0.282], [1, 0.5, 0.5], [1, 1, 3])]))
        self.assertIsNone(fractional_sector_doublets([_channel([-0.6, -0.282, -0.282], [1, 1, 1], [1, 1, 1])]))
        self.assertIsNone(fractional_sector_doublets([_channel([-0.6, -0.282, -0.280], [1, 0.5, 0.5], [1, 1, 1])]))
        # a Fermi-Dirac tail below the tolerance
        self.assertIsNone(fractional_sector_doublets([_channel([-0.282, -0.282], [0.99, 0.99], [2, 2])]))

    def test_the_note(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.acceleration.SCF.spin_sectors import SECTOR_DOUBLET_REASON
        from parsec_python.acceleration.driver import _SpinSectorFallback
        from parsec_python.models import SpinSectorFallbackRecord

        record = SpinSectorFallbackRecord(
            stage="the SCF loop",
            largest_fractional_occupation=0.5,
            tolerance=0.05,
            multiplets=("up channel: 2 levels",),
            reason=SECTOR_DOUBLET_REASON,
            route="sector_doublet",
            doublet=True,
        )
        text = "\n".join(spin_sector_fallback_lines(record))
        self.assertIn("The result is kept", text)
        self.assertIn("PARSEC_SPIN_GUARD_DOUBLETS=stop", text)
        self.assertNotIn("weighted", text)
        seen = []
        report = SimpleNamespace(largest=0.5, tolerance=0.05, lines=(), reason=SECTOR_DOUBLET_REASON, measure="m")
        system = SimpleNamespace(scalar_field_adapter=SimpleNamespace(doublet_report=("the SCF loop", report)))
        _SpinSectorFallback._say_doublet(system, seen.append)
        self.assertEqual(seen[0].route, "sector_doublet")


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # pragma: no cover
        return False


def _manganese_input() -> str:
    return """Spin_Polarization: .true.
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.40000000 5.40000000 5.40000000
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.22
Expansion_Order: 12
States_Num: 15
Fermi_Temp: 300.0
Max_Iter: 500
Convergence_Criterion: 1.0e-6
Diag_Tolerance: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.2
Correlation_Type: pbe
Atom_Types_Num: 2
Atom_Type: Mn_a
Element_Symbol: Mn
Local_Component: s
Initial_Spin_Polarization: 0.5
begin Atom_Coord
  0.0 0.0 0.0
end Atom_Coord
Atom_Type: Mn_b
Element_Symbol: Mn
Local_Component: s
Initial_Spin_Polarization: -0.5
begin Atom_Coord
  2.7 2.7 2.7
end Atom_Coord
"""


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class SectorDoubletRouteTests(unittest.TestCase):
    def test_note_keeps_the_bits_and_stop_takes_the_full_grid(self) -> None:
        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for label, environment in (
                ("off", {"PARSEC_SPIN_GUARD_DOUBLETS": "off"}),
                ("note", {}),
                ("stop", {"PARSEC_SPIN_GUARD_DOUBLETS": "stop"}),
                ("full", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
            ):
                directory = Path(name) / label
                directory.mkdir()
                (directory / "parsec.in").write_text(_manganese_input())
                for target in ("Mn_a_POTRE.DAT", "Mn_b_POTRE.DAT", "Mn_POTRE.DAT"):
                    shutil.copyfile(pseudopotential_file("tests/data/spin_pbe_Mn_POTRE.DAT"), directory / target)
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy", "--gpus", "1"])
                log = (directory / "parsec.out").read_text(encoding="utf-8")
                self.assertEqual(status, 0, f"{label}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    runs[label] = (float(data["energy_total_ry"]), log)
        phrase = "fractional degenerate levels of one sector"
        self.assertNotIn(phrase, runs["off"][1])
        self.assertIn(phrase, runs["note"][1])
        self.assertIn("The result is kept", runs["note"][1])
        self.assertEqual(runs["note"][0], runs["off"][0])
        self.assertIn(phrase, runs["stop"][1])
        self.assertIn("Spin_Symmetry_Sectors:", runs["stop"][1])
        # the full grid from the atomic densities: the result without symmetry, below the symmetric ensemble
        self.assertAlmostEqual(runs["stop"][0], runs["full"][0], delta=1e-6)
        self.assertLess(runs["full"][0], runs["off"][0] - 5e-3)


if __name__ == "__main__":
    unittest.main()
