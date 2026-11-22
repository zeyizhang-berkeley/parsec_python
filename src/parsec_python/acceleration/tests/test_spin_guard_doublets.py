"""Doublets whose partners are states of one weighted sector (``PARSEC_SPIN_GUARD_DOUBLETS``).

CPU: the switch, the partner labels, the plans that hold a representation twice in one sector and the lines of
``parsec.out``.  GPU: fcc Fe4 with 21 electrons at 100 K, whose E doublet holds half an electron per partner in both
channels: ``note`` keeps the bits of ``off`` and says so, ``stop`` goes to the full grid.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

FCC = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))


class SwitchTests(unittest.TestCase):
    def test_the_values(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import spin_guard_doublets_setting

        for value, expected in (("", "note"), ("note", "note"), ("STOP", "stop"), ("off", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_DOUBLETS": value}):
                self.assertEqual(spin_guard_doublets_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_DOUBLETS": "auto"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SPIN_GUARD_DOUBLETS"):
                spin_guard_doublets_setting()


class LabelTests(unittest.TestCase):
    def test_without_flags_the_labels_of_before(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import PARTNER_LABEL_STRIDE as stride, partner_levels

        labels = partner_levels([-1.0, -0.5, -0.5, 0.2], [1.0, 0.5, 0.5, 0.0], [1, 1, 1, 2], [1, 1, 1, 3])[2]
        self.assertEqual(labels.tolist(), [stride, stride, stride, 2 * stride, 2 * stride + 1, 2 * stride + 2])

    def test_degenerate_states_of_a_flagged_sector_are_partners(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import partly_filled_multiplets, partner_levels

        values, filled, sectors = [-1.0, -0.5, -0.5, 0.2], [1.0, 0.5, 0.5, 0.0], [1, 1, 1, 2]
        before = partner_levels(values, filled, sectors, [1, 1, 1, 3])
        self.assertIsNone(partly_filled_multiplets([before[0]], [before[1]], [before[2]], fixed=False))
        found = partner_levels(values, filled, sectors, [1, 1, 1, 3], repeated=[True, False])
        report = partly_filled_multiplets([found[0]], [found[1]], [found[2]], fixed=False)
        self.assertIsNotNone(report)
        self.assertAlmostEqual(report.largest, 1.0)

    def test_levels_apart_stay_in_one_label(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import MULTIPLET_WIDTH, _repeated_offsets

        values = [0.3, 0.1, 0.1 + 0.5 * MULTIPLET_WIDTH, 0.1 + 3.0 * MULTIPLET_WIDTH, 0.2]
        offsets = _repeated_offsets(values, [1, 1, 1, 1, 2], [True, False])
        self.assertEqual(offsets.tolist(), [0, 0, 1, 0, 0])
        with self.assertRaisesRegex(ValueError, "outside"):
            _repeated_offsets([0.0], [3], [True, False])

    def test_a_long_run_stays_within_the_labels_of_its_sector(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import PARTNER_LABEL_STRIDE as stride, partner_levels

        count = 3 * stride
        labels = partner_levels(np.zeros(count), np.zeros(count), np.ones(count, dtype=int), np.ones(count, dtype=int),
                                repeated=[True])[2]
        self.assertEqual(int(labels.max()) // stride, 1)
        self.assertTrue(np.all(labels[1:] != labels[:-1]))


class PlanTests(unittest.TestCase):
    @staticmethod
    def _plan(side, basis, symbol):
        from parsec_python.acceleration.Symmetry import orbital_groups as groups
        from parsec_python.acceleration.tests.test_orbital_group_arrays import _captured_plan_inputs
        from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model
        from parsec_python.models import Atom

        atoms = [Atom(symbol, tuple(side * np.array(position))) for position in basis]
        grid, _ = _periodic_model((side,) * 3, 24, 0.0, atoms, {symbol: 2.0})
        case = _captured_plan_inputs(grid, atoms)
        return groups.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"])

    def test_fcc_holds_the_two_doublets_in_sectors_of_weight_one(self) -> None:
        plan = self._plan(7.65, FCC, "Al")
        held = [index for index, irreps in enumerate(plan.repeated_irreps) if irreps]
        self.assertEqual(len(held), 2)
        self.assertTrue(all(plan.weights[index] == 1 for index in held))

    def test_bcc_holds_none(self) -> None:
        plan = self._plan(5.42, ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)), "Fe")
        self.assertFalse(any(plan.repeated_irreps))


class FullGridDoubletTests(unittest.TestCase):
    """The note of a spin cell that its floor keeps on the full grid: pairs of degenerate levels alone."""

    @staticmethod
    def _channel(values, filled):
        from types import SimpleNamespace

        return SimpleNamespace(eigenvalues=np.asarray(values), occupations=np.asarray(filled))

    def test_only_partly_filled_pairs_count(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import partly_filled_doublets

        pair = self._channel([-1.0, -0.5, -0.5, 0.2], [1.0, 0.5, 0.5, 0.0])
        triplet = self._channel([-1.0, -0.5, -0.5, -0.5], [1.0, 1 / 3, 1 / 3, 1 / 3])
        filled = self._channel([-1.0, -0.5, -0.5, 0.2], [1.0, 1.0, 1.0, 0.0])
        apart = self._channel([-1.0, -0.5, -0.49, 0.2], [1.0, 0.5, 0.5, 0.0])
        report = partly_filled_doublets([pair, triplet])
        self.assertEqual(len(report.lines), 1)
        self.assertTrue(report.lines[0].startswith("up channel: 2 levels"))
        self.assertAlmostEqual(report.largest, 0.5)
        for channel in (triplet, filled, apart):
            self.assertIsNone(partly_filled_doublets([channel]))

    def test_only_a_system_of_the_floor_says_it(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.driver import _say_floor_doublet

        result = SimpleNamespace(spin=(self._channel([-0.5, -0.5], [0.5, 0.5]),))
        for flag, count in ((True, 1), (False, 0)):
            records = []
            _say_floor_doublet(SimpleNamespace(spin_floor_full_grid=flag), result, records.append)
            self.assertEqual(len(records), count)
        self.assertEqual(records, [])
        records = []
        _say_floor_doublet(SimpleNamespace(spin_floor_full_grid=True), result, records.append)
        self.assertEqual(records[0].route, "floor_doublet")
        _say_floor_doublet(SimpleNamespace(spin_floor_full_grid=True), result, None)

    def test_the_lines(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        text = "\n".join(
            spin_sector_fallback_lines(
                SpinSectorFallbackRecord(
                    stage="the SCF loop",
                    largest_fractional_occupation=0.5,
                    tolerance=0.05,
                    multiplets=("up channel: 2 levels",),
                    reason="a partly filled pair of degenerate levels",
                    route="floor_doublet",
                    doublet=True,
                )
            )
        )
        self.assertIn("full grid of a small spin cell", text)
        self.assertIn("The result is kept", text)
        self.assertIn("up channel: 2 levels", text)


class LineTests(unittest.TestCase):
    def test_the_note_says_the_result_is_kept(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        record = SpinSectorFallbackRecord(
            stage="the SCF loop",
            largest_fractional_occupation=0.5,
            tolerance=0.05,
            multiplets=("up channel: 2 level(s)",),
            reason="a partly filled doublet within one sector",
            route="doublet",
            doublet=True,
        )
        text = "\n".join(spin_sector_fallback_lines(record))
        self.assertIn("The result is kept", text)
        self.assertIn("PARSEC_SPIN_GUARD_DOUBLETS=stop", text)
        self.assertIn("up channel: 2 level(s)", text)


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # pragma: no cover
        return False


def _iron_input() -> str:
    side = 6.82
    coordinates = "\n".join(f"  {x * side:.10f} {y * side:.10f} {z * side:.10f}" for x, y, z in FCC)
    return f"""Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side:.8f} {side:.8f} {side:.8f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.22
Expansion_Order: 12
States_Num: 24
Fermi_Temp: 100.0
Max_Iter: 100
Convergence_Criterion: 1.0e-6
Diag_Tolerance: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.2
Correlation_Type: pbe
Spin_Polarization: true
Initial_Spin_Polarization: 0.0
Net_Charges: 11
Atom_Types_Num: 1
Atom_Type: Fe
Local_Component: s
begin Atom_Coord
{coordinates}
end Atom_Coord
"""


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class DoubletRouteTests(unittest.TestCase):
    def test_note_keeps_the_bits_and_stop_leaves_the_sectors(self) -> None:
        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for mode in ("off", "note", "stop"):
                directory = Path(name) / mode
                directory.mkdir()
                (directory / "parsec.in").write_text(_iron_input())
                shutil.copyfile(pseudopotential_file("tests/data/spin_pbe_Fe_POTRE.DAT"), directory / "Fe_POTRE.DAT")
                environment = {"PARSEC_SPIN_GUARD_DOUBLETS": mode, "PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": "off"}
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy"])
                log = (directory / "parsec.out").read_text(encoding="utf-8")
                self.assertEqual(status, 0, f"{mode}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    runs[mode] = (float(data["energy_total_ry"]), log)
        phrase = "a partly filled doublet within one sector"
        self.assertNotIn(phrase, runs["off"][1])
        self.assertIn(phrase, runs["note"][1])
        self.assertIn("The result is kept", runs["note"][1])
        self.assertEqual(runs["note"][0], runs["off"][0])
        self.assertIn(phrase, runs["stop"][1])
        self.assertIn("goes straight to the", runs["stop"][1])
        self.assertIn("Spin_Symmetry_Sectors:", runs["stop"][1])
        self.assertAlmostEqual(runs["stop"][0], runs["off"][0], delta=1e-5)

    def test_the_note_of_the_full_grid_changes_no_number(self) -> None:
        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for label, environment in (
                ("floor", {"PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": str(1 << 21)}),
                ("full", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
            ):
                directory = Path(name) / label
                directory.mkdir()
                (directory / "parsec.in").write_text(_iron_input())
                shutil.copyfile(pseudopotential_file("tests/data/spin_pbe_Fe_POTRE.DAT"), directory / "Fe_POTRE.DAT")
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy"])
                log = (directory / "parsec.out").read_text(encoding="utf-8")
                self.assertEqual(status, 0, f"{label}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    runs[label] = (float(data["energy_total_ry"]), np.array(data["eigenvalues_ry"]), log)
        phrase = "a partly filled pair of degenerate levels"
        self.assertIn(phrase, runs["floor"][2])
        self.assertNotIn(phrase, runs["full"][2])
        self.assertEqual(runs["floor"][0], runs["full"][0])
        np.testing.assert_array_equal(runs["floor"][1], runs["full"][1])


if __name__ == "__main__":
    unittest.main()
