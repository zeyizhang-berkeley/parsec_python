"""The fixed cost of a solved sector in the choice of the weighted plan of a periodic cell (``PARSEC_SECTOR_OVERHEAD``).

CPU: the switch; the plan of least rows x states plus a cost per sector solves fewer sectors on a small cell and is the
plan of before at cost zero; the reduction keeps its route and its full group ``G`` (its fields, forces and moves keep
their average over ``G``); the driver's cost of a sector (none in an MPI run or under ``off``).  The SCF of the plans
of fewer sectors against those of before on A100 is in the README row of the switch.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.Symmetry import orbital_groups as og
from parsec_python.acceleration.tests.test_orbital_group_arrays import _captured_plan_inputs, _diamond, _rocksalt


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.Symmetry.axis_reflection import SECTOR_OVERHEAD_ENTRIES, sector_overhead_entries

        for value, expected in (("", SECTOR_OVERHEAD_ENTRIES), ("auto", SECTOR_OVERHEAD_ENTRIES), ("OFF", 0),
                                ("0", 0), ("4096", 4096)):
            with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": value}):
                self.assertEqual(sector_overhead_entries(), expected)
        for value in ("on", "-1", "2.5"):
            with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_SECTOR_OVERHEAD"):
                    sector_overhead_entries()


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {"diamond cell": _captured_plan_inputs(*_diamond(1)),
                     "rock-salt cell": _captured_plan_inputs(*_rocksalt(1)),
                     "diamond 2x2x2": _captured_plan_inputs(*_diamond(2))}

    def test_cost_zero_is_the_plan_of_before(self) -> None:
        for name, case in self.cases.items():
            with self.subTest(case=name):
                before = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"])
                again = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"],
                                              sector_cost=0.0)
                self.assertEqual(repr(before), repr(again))
                self.assertEqual(before.least_cost, before.cost)

    def test_a_cost_per_sector_solves_fewer_sectors(self) -> None:
        for name, case in self.cases.items():
            with self.subTest(case=name):
                before = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"])
                # a sector as costly as a whole small cell
                plan = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"],
                                             sector_cost=4.0)
                self.assertLess(len(plan.characters), len(before.characters))
                self.assertGreater(len(plan.subgroup), 1)
                # the least rows x states found on the way is the plan of before
                self.assertEqual(plan.least_cost, before.cost)
                self.assertGreaterEqual(plan.cost, before.cost)
                # no other admitted plan costs less in total
                table = og.multiplication_table(case["operations"], modulus=case["modulus"])
                representations = og.real_irreducible_characters(table)
                total = plan.cost + 4.0 * len(plan.characters)
                for members in og.subgroups(table, case["required"]):
                    if len(members) <= 1:
                        continue
                    other = og._plan_for(table, representations, members, table.shape[0])
                    if other is not None:
                        self.assertGreaterEqual(other.cost + 4.0 * len(other.characters), total - 1e-12)


class ReductionTests(unittest.TestCase):
    def test_the_route_and_its_full_group_are_kept(self) -> None:
        from parsec_python.acceleration.Symmetry import (
            PointGroupReduction,
            ReflectionRepresentationDecomposition,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )

        for name, (grid, atoms) in (("diamond cell", _diamond(1)), ("rock-salt cell", _rocksalt(1)),
                                    ("diamond 2x2x2", _diamond(2))):
            with self.subTest(case=name):
                abelian = SignedPermutationReduction.detect(grid, atoms)
                before = detect_point_group_reduction(grid, atoms, abelian)
                cheaper = detect_point_group_reduction(grid, atoms, abelian, sector_cost=4.0)
                self.assertIsInstance(before, PointGroupReduction)
                self.assertIsInstance(cheaper, PointGroupReduction)
                np.testing.assert_array_equal(cheaper.full_operations, before.full_operations)
                if getattr(before, "full_translations", None) is not None:
                    np.testing.assert_array_equal(cheaper.full_translations, before.full_translations)
                self.assertLess(len(cheaper.sector_weights), len(before.sector_weights))
                self.assertLess(cheaper.group_order, before.group_order)
                # the decomposition of the fewer sectors covers the grid with every level weighted once
                decomposition = ReflectionRepresentationDecomposition.build(grid, cheaper)
                self.assertEqual(len(decomposition.sector_sizes), len(cheaper.sector_weights))


class DriverCostTests(unittest.TestCase):
    def test_the_cost_of_a_sector_of_a_cell(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration import driver

        grid, _ = _diamond(1)
        problem = SimpleNamespace(scf=SimpleNamespace(number_of_states=20))
        reference = SimpleNamespace(grid=grid)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": "auto"}):
            cost = driver._periodic_sector_cost(problem, reference, None)
            self.assertAlmostEqual(cost, (1 << 21) / (grid.size * 20))
            self.assertEqual(driver._periodic_sector_cost(problem, reference, object()), 0.0)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": "off"}):
            self.assertEqual(driver._periodic_sector_cost(problem, reference, None), 0.0)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": "1000"}):
            self.assertAlmostEqual(driver._periodic_sector_cost(problem, reference, None), 1000 / (grid.size * 20))
        # a cell whose basis exceeds the fixed cost of a sector keeps the plan of before under auto, not under an
        # integer cost
        large = SimpleNamespace(scf=SimpleNamespace(number_of_states=(1 << 21) // grid.size + 1))
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": "auto"}):
            self.assertEqual(driver._periodic_sector_cost(large, reference, None), 0.0)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": str(1 << 21)}):
            self.assertGreater(driver._periodic_sector_cost(large, reference, None), 0.0)

    def test_spin_keeps_the_plan_of_before(self) -> None:
        """A spin-polarized (non-collinear, spin-orbit) cell takes no fixed cost under any value: its guard needs the
        partners of the multiplets of G in distinct sectors or in states of weight above one."""

        from types import SimpleNamespace

        from parsec_python.acceleration import driver

        grid, _ = _diamond(1)
        reference = SimpleNamespace(grid=grid)
        for name in ("spin_polarized", "noncollinear", "self_consistent_spin_orbit"):
            problem = SimpleNamespace(scf=SimpleNamespace(number_of_states=20, **{name: True}))
            for setting in ("auto", "1000", str(1 << 30)):
                with self.subTest(kind=name, setting=setting), mock.patch.dict(
                    os.environ, {"PARSEC_SECTOR_OVERHEAD": setting}
                ):
                    self.assertEqual(driver._periodic_sector_cost(problem, reference, None), 0.0)
        # a wrong value is still refused
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": "on"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SECTOR_OVERHEAD"):
                driver._periodic_sector_cost(
                    SimpleNamespace(scf=SimpleNamespace(number_of_states=20, spin_polarized=True)), reference, None
                )


def _held_in_one_weight_one_sector(plan, table) -> int:
    """The real representations of ``G`` of two or more dimensions whose solved partners all lie in one sector of
    weight 1: several states of one sector label, which the multiplet tests of the spin guard do not count."""

    representations = og.real_irreducible_characters(table)
    members = np.asarray(plan.subgroup, dtype=np.int64)
    count = 0
    for representation in representations:
        if representation.dimension < 2:
            continue
        holders = []
        for index, character in enumerate(plan.characters):
            signs = np.array([character[h] for h in plan.subgroup], dtype=np.float64)
            if round(float(signs @ representation.character[members]) / len(members)) > 0:
                holders.append(index)
        if len(holders) == 1 and plan.weights[holders[0]] == 1:
            count += 1
    return count


class SpinGuardPlanTests(unittest.TestCase):
    """Why spin keeps the plan of before: in bcc Fe2 the plan of fewer sectors (the subgroup of E and the body-centring
    translation, weights 1 and 1) holds every multiplet of G in one sector of weight 1, the plan of the least rows x
    states none."""

    def test_bcc_iron(self) -> None:
        from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model
        from parsec_python.models import Atom

        side = 5.42
        atoms = [Atom("Fe", (0.0, 0.0, 0.0)), Atom("Fe", (side / 2,) * 3)]
        grid, _ = _periodic_model((side,) * 3, 24, 0.0, atoms, {"Fe": 2.0})
        case = _captured_plan_inputs(grid, atoms)
        table = og.multiplication_table(case["operations"], modulus=case["modulus"])
        before = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"])
        fewer = og.orbital_group_plan(case["operations"], modulus=case["modulus"], required=case["required"],
                                      sector_cost=1.0)
        self.assertEqual(len(case["operations"]), 96)
        self.assertEqual(len(before.subgroup), 32)
        self.assertEqual(_held_in_one_weight_one_sector(before, table), 0)
        self.assertEqual((len(fewer.subgroup), fewer.weights), (2, (1, 1)))
        self.assertGreater(_held_in_one_weight_one_sector(fewer, table), 0)


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available
        from parsec_python.acceleration.backends.native import native_available

        return bool(cupy_available()) and bool(native_available())
    except Exception:  # pragma: no cover
        return False


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device and the native extension")
class SectorOverheadRelaxationTests(unittest.TestCase):
    """A steepest-descent step of the displaced silicon cube (Td about the origin with the face-centring translations)
    under the default fixed cost of a sector and under ``off``: fewer sectors, the same full group, the atoms it fixes
    stay, and the energies, forces and moved atoms of the plan of before within the SCF criterion."""

    def test_fewer_sectors_the_same_group_and_results(self) -> None:
        from pathlib import Path
        import tempfile

        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import STEEPEST_DESCENT, _run

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for setting in ("auto", "off"):
                with mock.patch.dict(os.environ, {"PARSEC_SECTOR_OVERHEAD": setting}):
                    runs[setting] = _run(Path(name) / setting, STEEPEST_DESCENT, "auto")
        lines = {}
        for setting, (status, _, log, errors) in runs.items():
            self.assertIn(status, (0, 4), f"{setting}: {errors}\n{log[-3000:]}")
            found = [line for line in log.splitlines() if "Forces of the periodic cell in the sectors" in line]
            self.assertTrue(found, setting)
            lines[setting] = found[0]
        # the forces and moves of both are averaged over the same full group
        average = lines["off"].split("averaged over ")[1].split(" operations")[0]
        self.assertIn(f"averaged over {average} operations", lines["auto"])
        detection = {
            setting: next(line for line in log.splitlines() if "symmetry_detection = " in line)
            for setting, (_, _, log, _) in runs.items()
        }
        self.assertIn("subgroup of 4 ", detection["auto"])
        self.assertNotEqual(detection["auto"], detection["off"])
        moved = {setting: np.asarray(run[1]["relaxation_positions"]) for setting, run in runs.items()}
        self.assertEqual(moved["auto"].shape, (2, 8, 3))
        # the four atoms the group fixes do not move
        np.testing.assert_allclose(moved["auto"][1, :4], moved["auto"][0, :4], rtol=0, atol=1e-12)
        np.testing.assert_allclose(moved["auto"], moved["off"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            runs["auto"][1]["relaxation_energies"], runs["off"][1]["relaxation_energies"], rtol=0, atol=1e-7
        )


#: bcc Fe in its cubic cell (a 5.42 bohr, h 0.22), PBE, spin-polarized at 100 K: the full grid leaves the state of the
#: full group, which the spin guard of the plan of the least rows x states detects (found 2026-11-21).
_IRON_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.42000000 5.42000000 5.42000000
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.22
Expansion_Order: 12
States_Num: 16
Fermi_Temp: 100.0
Max_Iter: 100
Convergence_Criterion: 1.0e-6
Diag_Tolerance: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.2
Correlation_Type: pbe
Atom_Types_Num: 1
Atom_Type: Fe
Local_Component: s
begin Atom_Coord
  0.0000000000 0.0000000000 0.0000000000
  2.7100000000 2.7100000000 2.7100000000
end Atom_Coord
Spin_Polarization: true
Initial_Spin_Polarization: 0.3
"""


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device and the native extension")
class SpinGuardRegressionTests(unittest.TestCase):
    """Spin-polarized bcc Fe2 at 100 K under the default switches (the fixed cost of a sector and the device rule at
    ``auto``) reaches the energy of the full grid: the run keeps the plan of the least rows x states, whose guard
    leaves the weighted sectors, instead of the two sectors of weight 1 that held the symmetric state 0.125 Ry above."""

    def test_the_default_reaches_the_full_grid(self) -> None:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        from pathlib import Path
        import shutil
        import tempfile

        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        results = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for label, environment in (
                ("default", {"PARSEC_SECTOR_OVERHEAD": "auto", "PARSEC_PERIODIC_SECTOR_DEVICES": "auto"}),
                ("full", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
            ):
                directory = Path(name) / label
                directory.mkdir()
                (directory / "parsec.in").write_text(_IRON_INPUT)
                shutil.copyfile(pseudopotential_file("tests/data/spin_pbe_Fe_POTRE.DAT"), directory / "Fe_POTRE.DAT")
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy"])
                log = (directory / "parsec.out").read_text(encoding="utf-8") if (directory / "parsec.out").is_file() else ""
                self.assertEqual(status, 0, f"{label}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    results[label] = (float(data["energy_total_ry"]), log)
        default_energy, default_log = results["default"]
        full_energy, _ = results["full"]
        # the plan of before (a subgroup of 32 of the 96 operations), not the two sectors of weight 1
        detection = next(line for line in default_log.splitlines() if "symmetry_detection = " in line)
        self.assertIn("subgroup of 32 ", detection)
        self.assertAlmostEqual(default_energy, full_energy, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
