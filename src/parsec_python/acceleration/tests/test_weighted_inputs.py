"""Weighted sectors of Fix_Number_Eigenvalues, libxc, occup.in and collinear spin (``Symmetry/weighted_inputs.py``).

Until 2026-11-19 these inputs kept the abelian group of the orbitals (``driver._orbital_group_refusal``); with their
switches at ``auto`` a Td cluster takes the weighted sectors of D2d.  Checked on the host: the switches and the
refusals; the occupations of ``occup.in`` per partner of a weighted state and the refusal of unequal ones; the Fermi
level of two spin channels with weights against the list of every partner; the partner levels of the spin guard,
which catch a partly filled multiplet held by one weighted state.  On a GPU: GeH4 (Td) in the weighted sectors
against the abelian ones (the switch ``off``) and the full grid, with ``Fix_Number_Eigenvalues``, with ``occup.in``
(equal partners; unequal partners start again in the abelian sectors), with libxc's PBE (where libxc loads), and
spin-polarized (the neutral molecule and its anion, symmetric states; the cation, whose partly filled t2 shell the
guard sends to the abelian sectors and on to the full grid), energies and forces.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.Occupations import (
    UnequalPartnerOccupations,
    partner_occupations,
    spin_fermi_occupations,
    spin_fixed_occupations,
    weighted_fixed_occupations,
)
from parsec_python.models import Atom

from ..Symmetry.weighted_inputs import (
    WEIGHTED_INPUT_SWITCHES,
    abelian_orbitals,
    forced_abelian_reason,
    weighted_input_settings,
)
from ..backends.cupy import cupy_available
from ..backends.native import native_available

_BOND = 2.9 / np.sqrt(3.0)
# GeH4 in Td: the weighted sectors of D2d (weights 1, 1, 3, 3), the abelian group D2.
GEH4 = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
    Atom("H", (_BOND * x, _BOND * y, _BOND * z)) for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
)


def _problem_namespace(**scf):
    defaults = dict(
        spin_polarized=False,
        noncollinear=False,
        self_consistent_spin_orbit=False,
        is_hybrid=False,
        is_meta_gga=False,
        xc_functional="pbe",
        fixed_occupations=None,
        spin_symmetry_sector_mode="auto",
        symmetrize_density=False,
    )
    defaults.update(scf)
    return SimpleNamespace(
        scf=SimpleNamespace(**defaults),
        pseudopotentials={},
        eigensolver=SimpleNamespace(fixed_state_count=False),
    )


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {name: "" for name in WEIGHTED_INPUT_SWITCHES.values()}):
            self.assertEqual(set(weighted_input_settings().values()), {"auto"})
        for name in WEIGHTED_INPUT_SWITCHES.values():
            with mock.patch.dict(os.environ, {name: "OFF"}):
                self.assertIn("off", weighted_input_settings().values())
            with mock.patch.dict(os.environ, {name: "abelian"}):
                with self.assertRaises(ValueError):
                    weighted_input_settings()

    def test_refusals(self) -> None:
        from .. import driver

        refusal = driver._orbital_group_refusal
        fixed = _problem_namespace(fixed_occupations=(1.0, 0.0))
        counted = _problem_namespace()
        counted.eigensolver.fixed_state_count = True
        libxc = _problem_namespace(xc_functional="libxc:gga_x_pbe+gga_c_pbe")
        spin = _problem_namespace(spin_polarized=True)
        cases = (
            ("occupations", fixed, False, None),
            ("fixed_count", counted, False, None),
            ("libxc", libxc, False, None),
            ("spin", spin, True, None),
        )
        with mock.patch.dict(os.environ, {name: "auto" for name in WEIGHTED_INPUT_SWITCHES.values()}):
            for kind, problem, polarized, _ in cases:
                with self.subTest(kind=kind):
                    self.assertIsNone(refusal(problem, polarized, None))
                    with mock.patch.dict(os.environ, {WEIGHTED_INPUT_SWITCHES[kind]: "off"}):
                        self.assertIsNotNone(refusal(problem, polarized, None))
                    with abelian_orbitals("left"):
                        self.assertEqual(forced_abelian_reason(), "left")
                        self.assertEqual(refusal(problem, polarized, None), "left")
                    self.assertIsNone(forced_abelian_reason())
            # on several MPI ranks occup.in and spin take them too since 2026-11-21 (PARSEC_ORBITAL_GROUP_MPI=auto);
            # under off they keep the abelian group there, as before; Fix_Number_Eigenvalues takes them either way
            context = SimpleNamespace(rank=0, size=4, exchange=False)
            self.assertIsNone(refusal(fixed, False, context))
            self.assertIsNone(refusal(spin, True, context))
            self.assertIsNone(refusal(counted, False, context))
            with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MPI": "off"}):
                self.assertIsNotNone(refusal(fixed, False, context))
                self.assertIsNotNone(refusal(spin, True, context))
                self.assertIsNone(refusal(counted, False, context))
                self.assertIsNone(refusal(spin, True, None))
            # spin on the full grid, with spin-orbit coupling, or with a hybrid keeps it; a cell takes it
            self.assertIsNotNone(refusal(_problem_namespace(spin_polarized=True, spin_symmetry_sector_mode="off"),
                                         True, None))
            cell = _problem_namespace(spin_polarized=True)
            cell.periodic_cell = object()
            self.assertIsNone(refusal(cell, True, None))
            self.assertIsNotNone(refusal(_problem_namespace(self_consistent_spin_orbit=True), True, None))
            self.assertIsNotNone(refusal(_problem_namespace(spin_polarized=True, is_hybrid=True), True, None))


class OccupationTests(unittest.TestCase):
    def test_partners_take_consecutive_entries(self) -> None:
        levels = np.array([-1.0, -0.5, -0.2, 0.1])
        weights = np.array([1, 3, 1, 3])
        np.testing.assert_array_equal(
            partner_occupations(levels, weights, [1, 0.5, 0.5, 0.5, 0.25, 0, 0]), [1.0, 0.5, 0.25, 0.0]
        )
        result = weighted_fixed_occupations(levels, weights, 5.5, [1, 0.5, 0.5, 0.5, 0.25, 0, 0])
        self.assertAlmostEqual(result.electron_count, 5.5, places=12)
        # the partners of the last state past States_Num are empty
        with self.assertRaises(UnequalPartnerOccupations) as caught:
            partner_occupations(levels, weights, [1, 1, 1, 1, 0, 0.5, 0])
        self.assertEqual((caught.exception.first, caught.exception.entries), (6, (0.5, 0.0, 0.0)))
        with self.assertRaises(UnequalPartnerOccupations):
            partner_occupations(levels, weights, [1, 1, 1, 0.5, 0, 0, 0])
        with self.assertRaises(ValueError):
            weighted_fixed_occupations(levels, weights, 9.0, [1, 1, 1, 1, 0, 0, 0])
        with self.assertRaises(ValueError):
            partner_occupations(levels, weights, np.ones(9))

    def test_partners_of_a_level_that_two_sectors_hold(self) -> None:
        # The e of Td lies in the sectors A1 and B1 of D2d, weight 1 each (partner mask bit 2), here at -0.2 Ry with
        # the eigensolver's 1e-9 Ry between its partners; the t2 states of weight 3 hold no such level.
        levels = np.array([-1.0, -0.5, -0.2, -0.2 + 1.0e-9, 0.1])
        weights = np.array([1, 3, 1, 1, 3])
        masks = [4, 0, 4, 4, 0]
        alike = [1, 1, 1, 1, 0.5, 0.5, 0, 0, 0]
        unequal = [1, 1, 1, 1, 1, 0, 0, 0, 0]
        np.testing.assert_array_equal(partner_occupations(levels, weights, alike, partners=masks), [1, 1, 0.5, 0.5, 0])
        # one partner filled and one empty: the average over Td would give each a half
        with self.assertRaises(UnequalPartnerOccupations) as caught:
            partner_occupations(levels, weights, unequal, partners=masks)
        self.assertEqual((caught.exception.first, caught.exception.positions), (5, (5, 6)))
        self.assertEqual(caught.exception.entries, (1.0, 0.0))
        with self.assertRaises(UnequalPartnerOccupations):
            weighted_fixed_occupations(levels, weights, 10.0, unequal, partners=masks)
        # without masks nothing is compared across states (the call of before)
        np.testing.assert_array_equal(partner_occupations(levels, weights, unequal), [1, 1, 1, 0, 0])
        # masks without a common representation, or levels farther apart than PARTNER_LEVEL_WIDTH: no partners
        partner_occupations(levels, weights, unequal, partners=[4, 0, 4, 8, 0])
        apart = levels.copy()
        apart[3] = -0.2 + 2.0e-4
        partner_occupations(apart, weights, unequal, partners=masks)
        # a level of another sector between the partners: their entries are still compared, the positions listed
        between = np.array([-1.0, -0.2, -0.2 + 1.0e-9, -0.2 + 2.0e-9, 0.1])
        with self.assertRaises(UnequalPartnerOccupations) as caught:
            partner_occupations(between, np.array([1, 1, 3, 1, 3]), [1, 1, 0.5, 0.5, 0.5, 0, 0, 0, 0],
                                partners=[0, 4, 0, 4, 0])
        self.assertEqual(caught.exception.positions, (2, 6))
        self.assertIn("states 2, 6", str(caught.exception))
        with self.assertRaises(ValueError):
            partner_occupations(levels, weights, alike, partners=masks[:4])

    def test_spin_partners_of_a_level_that_two_sectors_hold(self) -> None:
        up = np.array([-1.0, -0.3, -0.3 + 1.0e-9])
        down = np.array([-0.9, -0.2, -0.2 + 1.0e-9])
        weights = (np.array([1, 1, 1]), np.array([1, 1, 1]))
        masks = ([0, 4, 4], [0, 4, 4])
        result = spin_fixed_occupations(up, down, 4.0, [1, 0.5, 0.5], [1, 0.5, 0.5], weights=weights, partners=masks)
        np.testing.assert_array_equal(result.occupations[1], [1.0, 0.5, 0.5])
        with self.assertRaises(UnequalPartnerOccupations) as caught:
            spin_fixed_occupations(up, down, 4.0, [1, 0.5, 0.5], [1, 1, 0], weights=weights, partners=masks)
        self.assertEqual(caught.exception.channel, "down")
        # the masks of one channel alone
        spin_fixed_occupations(up, down, 4.0, [1, 0.5, 0.5], [1, 1, 0], weights=weights, partners=(masks[0], None))

    def test_plans_name_the_levels_that_several_sectors_hold(self) -> None:
        # Td in D2d: e (representation 2 of Td) in the two sectors of weight 1; Oh in D4h: eg and eu, two sectors each;
        # D4h keeps its e levels in merged sectors of weight 2 (no level shared between solved states).
        import itertools

        from ..Symmetry.orbital_groups import orbital_group_plan

        signed = []
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                matrix = np.zeros((3, 3), dtype=np.int64)
                for row, (column, sign) in enumerate(zip(permutation, signs)):
                    matrix[row, column] = sign
                signed.append(matrix)
        corners = {(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)}
        groups = {
            "Td": [m for m in signed if {tuple(m @ np.array(v)) for v in corners} == corners],
            "Oh": signed,
            "D4h": [m for m in signed if abs(m[2, 2]) == 1],
        }
        for name, shared in (("Td", 2), ("Oh", 4), ("D4h", 0)):
            with self.subTest(group=name):
                plan = orbital_group_plan(groups[name])
                self.assertEqual(len(plan.partner_irreps), len(plan.weights))
                holding = [index for index, held in enumerate(plan.partner_irreps) if held]
                self.assertEqual(len(holding), shared)
                self.assertTrue(all(plan.weights[index] == 1 for index in holding))

    def test_spin_fermi_level_with_weights(self) -> None:
        generator = np.random.default_rng(7)
        up = np.sort(generator.normal(size=9))
        down = np.sort(generator.normal(size=9))
        ones = (np.ones(9), np.ones(9))
        for temperature in (0.0, 80.0, 3000.0):
            plain = spin_fermi_occupations(up, down, 7.0, temperature)
            weighted = spin_fermi_occupations(up, down, 7.0, temperature, weights=ones)
            np.testing.assert_allclose(weighted.occupations[0], plain.occupations[0], rtol=0, atol=1e-14)
            np.testing.assert_allclose(weighted.occupations[1], plain.occupations[1], rtol=0, atol=1e-14)
            weights = (np.array([1, 3, 1, 3, 1, 2, 2, 1, 1]), np.array([1, 1, 3, 1, 2, 1, 1, 3, 1]))
            compact = spin_fermi_occupations(up, down, 11.0, temperature, weights=weights)
            listed = spin_fermi_occupations(np.repeat(up, weights[0]), np.repeat(down, weights[1]), 11.0, temperature)
            for channel in (0, 1):
                np.testing.assert_allclose(
                    np.repeat(compact.occupations[channel], weights[channel]), listed.occupations[channel],
                    rtol=0, atol=1e-9,
                )
            self.assertAlmostEqual(compact.electron_count, 11.0, places=8)
            self.assertAlmostEqual(compact.fermi_level, listed.fermi_level, places=8)

    def test_spin_fixed_occupations_with_weights(self) -> None:
        up = np.array([-1.0, -0.5, 0.2])
        down = np.array([-0.9, -0.4, 0.3])
        weights = (np.array([1, 3, 1]), np.array([1, 3, 1]))
        result = spin_fixed_occupations(up, down, 8.0, [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], weights=weights)
        np.testing.assert_array_equal(result.occupations[0], [1.0, 1.0, 0.0])
        self.assertEqual(result.electrons, (4.0, 4.0))
        with self.assertRaises(ValueError):
            spin_fixed_occupations(up, down, 7.0, [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], weights=weights)
        with self.assertRaises(UnequalPartnerOccupations) as caught:
            spin_fixed_occupations(up, down, 7.0, [1, 1, 1, 1, 0], [1, 1, 0.5, 0.5, 0], weights=weights)
        self.assertEqual(caught.exception.channel, "down")


class GuardTests(unittest.TestCase):
    def test_partner_levels(self) -> None:
        from ..SCF.spin_sectors import PARTNER_LABEL_STRIDE, partly_filled_multiplets, partner_levels

        values, filled, sectors = partner_levels([-1.0, -0.5, 0.2], [1.0, 1.0, 0.0], [1, 4, 2], [1, 3, 1])
        np.testing.assert_array_equal(values, [-1.0, -0.5, -0.5, -0.5, 0.2])
        np.testing.assert_allclose(filled, [1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
        np.testing.assert_array_equal(
            sectors, np.array([1, 4, 4, 4, 2]) * PARTNER_LABEL_STRIDE + np.array([0, 0, 1, 2, 0])
        )
        # one electron in a triplet held by one weighted state: a partly filled multiplet
        report = partly_filled_multiplets([values], [filled], [sectors], fixed=False)
        self.assertIsNotNone(report)
        self.assertIn("4.1 4.2 4.3", report.lines[0])
        # a filled triplet is not
        values, filled, sectors = partner_levels([-1.0, -0.5, 0.2], [1.0, 3.0, 0.0], [1, 4, 2], [1, 3, 1])
        self.assertIsNone(partly_filled_multiplets([values], [filled], [sectors], fixed=False))
        # nor are equal occupations of occup.in
        values, filled, sectors = partner_levels([-1.0, -0.5, 0.2], [1.0, 1.0, 0.0], [1, 4, 2], [1, 3, 1])
        self.assertIsNone(partly_filled_multiplets([values], [filled], [sectors], fixed=True))

    def test_reducer_of_weighted_sectors(self) -> None:
        from ..SCF.spin_sectors import SpinSectorReducer, SpinSectorsAbandoned, fractional_occupations

        reduction = SimpleNamespace(
            sector_weights=np.array([1, 1, 3, 3]), wedge_size=4, multiplicities=np.ones(4)
        )
        reducer = SimpleNamespace(reduction=reduction, weights=np.ones(4))
        adapter = SpinSectorReducer(reducer, hybrid=False)
        self.assertEqual(adapter.sector_weights.tolist(), [1, 1, 3, 3])
        # one electron in a T2 level (sector 4, weight 3) of the down channel at the start
        report = adapter.start_spectrum_report(
            [np.array([-1.0, -0.5, 0.2]), np.array([-1.0, -0.5, 0.2])],
            [np.array([1.0, 3.0, 0.0]), np.array([1.0, 1.0, 0.0])],
            [np.array([1, 4, 2]), np.array([1, 4, 2])],
        )
        self.assertIsNotNone(report)
        self.assertTrue(report.lines[0].startswith("down channel"))

        def channel(occupations, sector_states):
            values = np.repeat([-1.0, -0.5, 0.2], sector_states[2])
            return SimpleNamespace(
                eigenvalues=values,
                occupations=np.asarray(occupations, dtype=float),
                representations=np.repeat(sector_states[0], sector_states[2]),
                sector_states=sector_states,
                density=np.zeros(4),
            )

        states = (np.array([1, 4, 2]), np.array([1.0, 1.0, 0.0]), np.array([1, 3, 1]))
        partial = channel([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0], states)
        # the repeated labels of the public lists alone would hide the multiplet of one weighted state
        self.assertIsNone(fractional_occupations([partial]))
        with self.assertRaises(SpinSectorsAbandoned):
            adapter.check_spin_result(SimpleNamespace(spin=(partial, partial)), final=True)
        filled = channel([1.0, 1.0, 1.0, 1.0, 0.0], (states[0], np.array([1.0, 3.0, 0.0]), states[2]))
        adapter.check_spin_result(SimpleNamespace(spin=(filled, filled)), final=True)
        # partners of a level in two sectors of weight one, one filled and one empty: their mean is fractional
        pair = SimpleNamespace(
            eigenvalues=np.array([-1.0, -0.5, -0.5]),
            occupations=np.array([1.0, 1.0, 0.0]),
            representations=np.array([1, 1, 2]),
        )
        self.assertIsNone(fractional_occupations([pair]))
        self.assertIsNotNone(fractional_occupations([pair], unequal_partners=True))


class FallbackTests(unittest.TestCase):
    """``driver._SpinSectorFallback.run``: the weighted sectors, then the abelian ones, then (spin) the full grid."""

    def _system(self, weighted: bool):
        from ..Symmetry import PointGroupReduction

        reduction = mock.MagicMock(spec=PointGroupReduction) if weighted else object()
        sectors = SimpleNamespace(decomposition=SimpleNamespace(reduction=reduction))
        return SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=sectors),
            scalar_field_adapter=SimpleNamespace(check_spin_result=lambda result, final: None),
        )

    def _fallback(self, spin: bool):
        from .. import driver

        fallback = driver._SpinSectorFallback(
            SimpleNamespace(pseudopotentials={}), (), backend="auto", symmetry="auto",
            symmetry_cache_directory=None, spin=spin,
        )
        replaced = []

        def to_abelian(system, stage, report):
            replaced.append(("abelian", stage, report.reason))
            system.backend = self._system(False).backend

        def to_full_grid(system, stage, report):
            replaced.append(("full", stage, report.reason))
            system.backend = SimpleNamespace(symmetry_eigensolver=None)

        fallback.replace_with_abelian = to_abelian
        fallback.replace_with_full_grid = to_full_grid
        return fallback, replaced

    def test_unequal_partners_start_again_in_the_abelian_sectors(self) -> None:
        fallback, replaced = self._fallback(spin=False)
        system = self._system(True)
        calls, records = [], []

        def run(system, callback=None, warm_start=None, **keywords):
            calls.append(warm_start)
            if len(calls) == 1:
                raise UnequalPartnerOccupations(2, (1.0, 0.5, 0.5), -0.4)
            return "result"

        self.assertEqual(fallback.run(system, run, callback=records.append), "result")
        self.assertEqual([item[0] for item in replaced], ["abelian"])
        self.assertEqual([record.route for record in records], ["weighted"])
        self.assertEqual(records[0].reason, "unequal occupations of degenerate partners")
        # in the abelian sectors the same error is the run's own
        fallback, replaced = self._fallback(spin=False)

        def failing(system, **keywords):
            raise UnequalPartnerOccupations(2, (1.0, 0.5, 0.5), -0.4)

        with self.assertRaises(UnequalPartnerOccupations):
            fallback.run(self._system(False), failing)
        self.assertEqual(replaced, [])

    def test_a_spin_guard_goes_on_to_the_full_grid(self) -> None:
        from ..SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        report = FractionalOccupations(largest=0.33, tolerance=0.05, lines=("x",))
        fallback, replaced = self._fallback(spin=True)
        calls, records = [], []

        def run(system, callback=None, warm_start=None, **keywords):
            calls.append(system.backend.symmetry_eigensolver)
            if len(calls) <= 2:
                raise SpinSectorsAbandoned("the SCF loop", report)
            return "result"

        self.assertEqual(fallback.run(self._system(True), run, callback=records.append), "result")
        self.assertEqual([item[0] for item in replaced], ["abelian", "full"])
        self.assertEqual([record.route for record in records], ["weighted", "spin"])
        # an abelian run goes to the full grid at once, as before
        fallback, replaced = self._fallback(spin=True)
        calls.clear()
        records.clear()

        def once(system, callback=None, warm_start=None, **keywords):
            calls.append(system.backend.symmetry_eigensolver)
            if len(calls) == 1:
                raise SpinSectorsAbandoned("the SCF loop", report)
            return "result"

        self.assertEqual(fallback.run(self._system(False), once, callback=records.append), "result")
        self.assertEqual([item[0] for item in replaced], ["full"])
        self.assertEqual([record.route for record in records], ["spin"])


def _weighted_spin_at_any_size(case: unittest.TestCase) -> None:
    """The spin tests of the weighted sectors run on small systems, whatever floor ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR``
    of the environment would keep the abelian group of (none by default, ``test_weighted_mpi_route.py``)."""

    patcher = mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_SPIN_FLOOR": "off"})
    patcher.start()
    case.addCleanup(patcher.stop)


def _device_available() -> bool:
    return cupy_available() and native_available()


def _libxc_available() -> bool:
    try:
        from parsec_python.V_xc.libxc_library import load_libxc

        load_libxc()
    except Exception:  # noqa: BLE001 - no library, or one that does not load here
        return False
    return True


def _write_problem(directory: Path, extra=()):
    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.force_reference import load_cases, write_case

    case = dict(load_cases()["ge2h2_nlcc_tiny"])
    case["input"] = list(case["input"]) + list(extra)
    problem = parse_parsec_input(write_case(case, directory)).problem
    return replace(
        problem,
        atoms=GEH4,
        recenter_geometry=False,
        grid=replace(problem.grid, radius=9.0),
        scf=replace(problem.scf, number_of_states=8, convergence_criterion=1.0e-8),
    )


@unittest.skipUnless(_device_available(), "CUDA runtime and native extension are unavailable")
class DeviceTests(unittest.TestCase):
    """Complete calculations of GeH4 on a GPU."""

    def setUp(self) -> None:
        _weighted_spin_at_any_size(self)

    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.directory = Path(directory.name)
        cls.problem = _write_problem(cls.directory)

    def _run(self, problem, symmetry, environment=None, *, forces=True):
        from parsec_python.acceleration import driver

        records = []
        with mock.patch.dict(os.environ, environment or {}):
            system = driver.prepare_single_point(problem, symmetry=symmetry)
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system, callback=records.append)
            computed = driver.compute_forces(system, result) if forces else None
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        reduction = None if sectors is None else sectors.decomposition.reduction
        return SimpleNamespace(
            system=system, result=result, forces=computed, reduction=reduction, records=records,
            details=dict(system.backend_info.details),
        )

    def _assert_weighted(self, run) -> None:
        from ..Symmetry import PointGroupReduction

        self.assertIsInstance(run.reduction, PointGroupReduction)
        self.assertEqual(sorted(int(w) for w in run.reduction.sector_weights), [1, 1, 3, 3])
        # the e level of Td in the two sectors of weight 1 (the same representation of Td), none in the others
        pairs = list(zip(run.reduction.sector_partner_irreps, (int(w) for w in run.reduction.sector_weights)))
        shared = [tuple(held) for held, weight in pairs if weight == 1]
        self.assertEqual(len(shared), 2)
        self.assertTrue(shared[0] and shared[0] == shared[1])
        self.assertEqual([tuple(held) for held, weight in pairs if weight == 3], [(), ()])

    def _assert_abelian(self, run) -> None:
        from ..Symmetry import PointGroupReduction

        self.assertIsNotNone(run.reduction)
        self.assertNotIsInstance(run.reduction, PointGroupReduction)

    def _compare(self, weighted, abelian, full, *, energy=1e-8, full_energy=1e-6, force=2e-6) -> None:
        energies = {name: float(run.result.energies.total) for name, run in
                    (("weighted", weighted), ("abelian", abelian), ("full", full))}
        self.assertAlmostEqual(energies["weighted"], energies["abelian"], delta=energy, msg=str(energies))
        self.assertAlmostEqual(energies["weighted"], energies["full"], delta=full_energy, msg=str(energies))
        if weighted.forces is not None:
            np.testing.assert_allclose(weighted.forces.total, abelian.forces.total, rtol=0, atol=force)
            np.testing.assert_allclose(weighted.forces.total, full.forces.total, rtol=0, atol=10 * force)

    def test_fixed_state_count(self) -> None:
        problem = replace(self.problem, eigensolver=replace(self.problem.eigensolver, fixed_state_count=True))
        weighted = self._run(problem, "on")
        abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_FIXED_COUNT": "off"})
        full = self._run(problem, "off")
        self._assert_weighted(weighted)
        self._assert_abelian(abelian)
        self._compare(weighted, abelian, full)
        # every sector kept the levels it holds among the lowest States_Num (counted with their weights)
        counts = weighted.system.backend.symmetry_eigensolver._state.sector_state_counts
        weights = np.asarray(weighted.reduction.sector_weights)
        self.assertGreaterEqual(int(np.dot(counts, weights)), 8)

    def test_occupations_of_equal_partners(self) -> None:
        # The cation GeH4(0.6+), a1^2 t2^5.4 with a1* empty: the three t2 partners alike, below a gap, so the order of
        # the levels holds and the run must stay in the weighted sectors.
        values = (1.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0)
        problem = replace(self.problem, scf=replace(self.problem.scf, fermi_temperature_kelvin=-1.0,
                                                    fixed_occupations=values, net_charge=0.6))
        weighted = self._run(problem, "on")
        abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_OCCUPATIONS": "off"})
        full = self._run(problem, "off")
        self._assert_abelian(abelian)
        self.assertEqual([type(record).__name__ for record in weighted.records if not hasattr(record, "iteration")],
                         [])
        self._assert_weighted(weighted)
        np.testing.assert_allclose(weighted.result.occupations[:8], values, atol=0)
        self._compare(weighted, abelian, full)

    def test_occupations_of_a_level_that_two_sectors_hold(self) -> None:
        # The e level of Td lies in the sectors A1 and B1 of D2d (weight 1 each).  A file that fills its two partners
        # [1, 0] asks for a state that the weighted density, averaged over Td, cannot hold: it gives each partner a
        # half (before the check: 2.6e-3 Ry from the abelian route and the full grid on A100, with no NOTE).  The run
        # starts again in the abelian sectors; [0.5, 0.5] stays in the weighted ones.
        states = 24
        base = replace(self.problem, scf=replace(self.problem.scf, number_of_states=states,
                                                 convergence_criterion=1.0e-7, max_iterations=200))
        levels = self._run(base, "on", forces=False)
        self._assert_weighted(levels)
        values = np.asarray(levels.result.eigenvalues, dtype=float)[:states]
        sectors = np.asarray(levels.result.representations, dtype=np.int64)[:states] - 1
        shared = [bool(held) for held in levels.reduction.sector_partner_irreps]
        pairs = [
            position for position in range(4, states - 2)
            if abs(values[position + 1] - values[position]) < 1.0e-6
            and sectors[position] != sectors[position + 1]
            and shared[sectors[position]] and shared[sectors[position + 1]]
            and values[position + 2] - values[position + 1] > 1.0e-4
            and values[position] - values[position - 1] > 1.0e-4
        ]
        self.assertTrue(pairs, msg=str(values))
        pair = pairs[0]

        def with_file(first, second):
            entries = np.zeros(states)
            entries[0] = 1.0
            entries[1:4] = 2.0 / 3.0
            entries[pair], entries[pair + 1] = first, second
            return replace(base, scf=replace(base.scf, fermi_temperature_kelvin=-1.0,
                                             fixed_occupations=tuple(float(value) for value in entries)))

        weighted = self._run(with_file(1.0, 0.0), "on", forces=False)
        abelian = self._run(with_file(1.0, 0.0), "on", {"PARSEC_ORBITAL_GROUP_OCCUPATIONS": "off"}, forces=False)
        self._assert_abelian(weighted)
        notes = [record for record in weighted.records if type(record).__name__ == "SpinSectorFallbackRecord"]
        self.assertEqual([(note.route, note.stage) for note in notes], [("weighted", "the occupations of occup.in")])
        self.assertIn(f"states {pair + 1} to {pair + 2}", notes[0].multiplets[0])
        self.assertAlmostEqual(float(weighted.result.energies.total), float(abelian.result.energies.total),
                               delta=1e-10)
        alike = self._run(with_file(0.5, 0.5), "on", forces=False)
        self._assert_weighted(alike)
        self.assertEqual([record for record in alike.records if type(record).__name__ == "SpinSectorFallbackRecord"],
                         [])
        # the symmetric ensemble is another state (A100 review run: 2.56e-3 Ry)
        self.assertGreater(abs(float(alike.result.energies.total) - float(abelian.result.energies.total)), 1.0e-3)

    def test_occupations_of_unequal_partners_start_again_in_the_abelian_sectors(self) -> None:
        # two of the three t2 partners filled, the third half: a state the weighted sectors cannot hold
        values = (1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0)
        problem = replace(self.problem, scf=replace(self.problem.scf, fermi_temperature_kelvin=-1.0,
                                                    fixed_occupations=values))
        weighted = self._run(problem, "on", forces=False)
        abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_OCCUPATIONS": "off"}, forces=False)
        self._assert_abelian(weighted)
        notes = [record for record in weighted.records if type(record).__name__ == "SpinSectorFallbackRecord"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].route, "weighted")
        self.assertIn("weighted_sectors", weighted.details)
        self.assertAlmostEqual(float(weighted.result.energies.total), float(abelian.result.energies.total), delta=1e-10)

    @unittest.skipUnless(_libxc_available(), "libxc is not available (PARSEC_LIBXC_LIBRARY)")
    def test_libxc_pbe(self) -> None:
        problem = _write_problem(self.directory, ("Correlation_Type: pbe", "LibXC_Functional: gga_x_pbe gga_c_pbe"))
        self.assertTrue(problem.scf.is_libxc)
        weighted = self._run(problem, "on")
        abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_LIBXC": "off"})
        full = self._run(problem, "off")
        self._assert_weighted(weighted)
        self._assert_abelian(abelian)
        self._compare(weighted, abelian, full)
        # and the code's own PBE in the weighted sectors (A100: 2.1e-7 Ry from libxc's, a functional that differs
        # from the code's in its digits)
        own = self._run(_write_problem(self.directory, ("Correlation_Type: pbe",)), "on", forces=False)
        self.assertAlmostEqual(float(own.result.energies.total), float(weighted.result.energies.total), delta=1e-6)

    def _spin_problem(self, charge: float):
        return replace(self.problem, scf=replace(self.problem.scf, spin_polarized=True, net_charge=charge))

    def test_spin_symmetric_states(self) -> None:
        for charge in (0.0, -1.0):
            with self.subTest(charge=charge):
                problem = self._spin_problem(charge)
                weighted = self._run(problem, "on")
                abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_SPIN": "off"})
                full = self._run(problem, "off")
                self._assert_abelian(abelian)
                left = [record for record in weighted.records if type(record).__name__ == "SpinSectorFallbackRecord"]
                # the closed-shell molecule and the anion (its extra electron in a1*): symmetric states that the
                # weighted sectors keep
                self.assertEqual(left, [])
                self._assert_weighted(weighted)
                self._compare(weighted, abelian, full, full_energy=2e-6)
                moments = [float(run.result.spin[0].electron_count - run.result.spin[1].electron_count)
                           for run in (weighted, full)]
                self.assertAlmostEqual(moments[0], moments[1], delta=1e-6)
                self.assertEqual(len(weighted.result.spin[0].eigenvalues), len(weighted.result.spin[0].occupations))

    def test_spin_step_of_a_relaxation(self) -> None:
        # The anion stretched along its breathing mode (Td kept): the weighted sectors move with the atoms, from the
        # potentials of the first structure, as a relaxation step does; against the full grid of the moved atoms.
        from parsec_python.acceleration import driver

        problem = self._spin_problem(-1.0)
        first = self._run(problem, "on")
        self._assert_weighted(first)
        moved_atoms = tuple(Atom(atom.symbol, 1.03 * np.asarray(atom.position)) for atom in first.system.atoms)
        moved = driver.update_geometry(first.system, moved_atoms)
        warm = driver.run_scf(moved, warm_start=driver.warm_start_for_moved_atoms(first.system, first.result, moved))
        warm_forces = driver.compute_forces(moved, warm)
        full = self._run(replace(problem, atoms=moved_atoms), "off")
        self.assertTrue(warm.converged)
        self.assertAlmostEqual(float(warm.energies.total), float(full.result.energies.total), delta=2e-6)
        np.testing.assert_allclose(warm_forces.total, full.forces.total, rtol=0, atol=2e-5)
        self.assertGreater(float(np.abs(full.forces.total).max()), 1e-3)

    def test_spin_cation_leaves_the_weighted_sectors(self) -> None:
        problem = self._spin_problem(1.0)
        weighted = self._run(problem, "on", forces=False)
        abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_SPIN": "off"}, forces=False)
        notes = [record for record in weighted.records if type(record).__name__ == "SpinSectorFallbackRecord"]
        self.assertGreaterEqual(len(notes), 1)
        self.assertEqual(notes[0].route, "weighted")
        # the run ends where the abelian route ends (its own guard may send both to the full grid)
        self.assertAlmostEqual(
            float(weighted.result.energies.total), float(abelian.result.energies.total), delta=1e-6
        )


@unittest.skipUnless(_device_available(), "CUDA runtime and native extension are unavailable")
class SpinFixtureTests(unittest.TestCase):
    """The spin fixtures of ``test_spin_sectors.py`` (O2 and the O atom, D4h of the grid) in the weighted sectors."""

    def setUp(self) -> None:
        _weighted_spin_at_any_size(self)

    def solve(self, *arguments, **options):
        import shutil

        from .test_spin_sectors import _solve

        directory, system, result = _solve(*arguments, **options)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return system, result

    def test_oxygen_molecule(self) -> None:
        from parsec_python.acceleration import driver

        from ..Symmetry import PointGroupReduction

        full_system, full = self.solve("o2_ca", "off")
        records = []
        system, weighted = self.solve("o2_ca", "auto", records=records)
        self.assertEqual([type(record).__name__ for record in records if not hasattr(record, "iteration")], [])
        reduction = system.backend.symmetry_eigensolver.decomposition.reduction
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertIn(2, [int(w) for w in reduction.sector_weights])
        self.assertAlmostEqual(weighted.energies.total, full.energies.total, delta=1e-9)
        for index in range(2):
            count = min(len(full.spin[index].eigenvalues), len(weighted.spin[index].eigenvalues))
            np.testing.assert_allclose(
                weighted.spin[index].eigenvalues[:count], full.spin[index].eigenvalues[:count], atol=2e-8
            )
            np.testing.assert_allclose(
                weighted.spin[index].occupations[:count], full.spin[index].occupations[:count], atol=1e-9
            )
            np.testing.assert_allclose(weighted.spin[index].density, full.spin[index].density, atol=1e-7)
            self.assertEqual(weighted.spin[index].sector_states[2].dtype.kind, "i")
        np.testing.assert_allclose(
            driver.compute_forces(system, weighted).total, driver.compute_forces(full_system, full).total, atol=1e-7
        )

    def test_oxygen_atom_leaves_the_weighted_then_the_abelian_sectors(self) -> None:
        from parsec_python.models import SpinSectorFallbackRecord

        _, full = self.solve("o_spherical_ca", "off", drop=("Fermi_Temp",))
        records = []
        system, result = self.solve("o_spherical_ca", "auto", drop=("Fermi_Temp",), records=records)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual([note.route for note in notes], ["weighted", "spin"])
        # the p shell of the minority channel is one state of weight 3 holding a third of an electron per partner
        self.assertEqual(notes[0].stage, "the start Hamiltonian")
        self.assertEqual(notes[0].reason, "a partly filled multiplet")
        self.assertIn("down channel: 3 levels", notes[0].multiplets[0])
        self.assertEqual(result.energies.total, full.energies.total)
        details = dict(result.backend.details)
        self.assertTrue(details["orbital_symmetry"].startswith("full grid (spin-polarized"))
        self.assertIsNone(system.spin_sector_fallback)

    def test_oxygen_atom_with_equal_partners_of_occup_in(self) -> None:
        # The fixture's occup.in spreads the minority p electron over the three p levels: equal partners, which the
        # weighted sectors hold (the density keeps the full group on the full grid too).
        from ..Symmetry import PointGroupReduction

        _, full = self.solve("o_spherical_ca", "off")
        records = []
        system, weighted = self.solve("o_spherical_ca", "auto", records=records)
        self.assertEqual([type(record).__name__ for record in records if not hasattr(record, "iteration")], [])
        self.assertIsInstance(system.backend.symmetry_eigensolver.decomposition.reduction, PointGroupReduction)
        self.assertAlmostEqual(weighted.energies.total, full.energies.total, delta=1e-8)
        for index in range(2):
            np.testing.assert_allclose(weighted.spin[index].density, full.spin[index].density, atol=1e-6)


@unittest.skipUnless(_device_available(), "CUDA runtime and native extension are unavailable")
class GammaSpinTests(unittest.TestCase):
    """Spin-polarized diamond vacancies of 2 x 2 x 2 cubes at the Gamma point (``test_gamma_spin_sectors.py``)."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto"})
        patcher.start()
        self.addCleanup(patcher.stop)
        _weighted_spin_at_any_size(self)

    def test_the_negative_vacancy(self) -> None:
        # V- (S = 3/2: a1 and t2^3 in the up channel), Td about the vacancy: the weighted sectors of D2d against the
        # abelian ones of D2, at the origin and at (a/2, a/2, 0) (the sectors centred on the vacancy).
        import shutil

        from parsec_python.models import SpinSectorFallbackRecord

        from ..Symmetry import PointGroupReduction
        from .test_gamma_spin_sectors import _magnetization, _sector_run, _vacancy

        energies = {}
        for name, site in (("origin", (0.0, 0.0, 0.0)), ("off", (0.5, 0.5, 0.0))):
            directory, problem = _vacancy(site, charge=1)
            self.addCleanup(shutil.rmtree, directory, True)
            for value in ("auto", "off"):
                records: list = []
                with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_SPIN": value}):
                    system, result = _sector_run(problem, records=records)
                with self.subTest(site=name, switch=value):
                    self.assertFalse(any(isinstance(record, SpinSectorFallbackRecord) for record in records))
                    self.assertTrue(result.converged)
                    reduction = system.backend.symmetry_eigensolver.decomposition.reduction
                    self.assertEqual(isinstance(reduction, PointGroupReduction), value == "auto")
                    up, down = result.spin
                    self.assertAlmostEqual(
                        _magnetization(system.reference.grid, up.density, down.density), 3.0, delta=1.0e-8
                    )
                energies[(name, value)] = result.energies.total
        # (A100: 9e-10 Ry at the origin, 5.8e-8 Ry off it, at the criterion of the fixture)
        for name in ("origin", "off"):
            self.assertAlmostEqual(energies[(name, "auto")], energies[(name, "off")], delta=1.0e-7, msg=str(energies))

    def test_the_neutral_vacancy_leaves_the_weighted_sectors(self) -> None:
        # The neutral vacancy (t2^2 in the up channel) at 10 K with test 1 deferred: the weighted sectors hold the
        # t2 shell as one state of weight 3 with two thirds of an electron per partner, which test 3 finds
        # (the density, averaged over Td, passes test 2); the run stops at that record here.
        import shutil

        from parsec_python.acceleration import driver
        from parsec_python.models import SpinSectorFallbackRecord

        from ..SCF.spin_sectors import SpinSectorReducer
        from .test_gamma_spin_sectors import _vacancy

        class Stop(Exception):
            pass

        fired: list = []

        def callback(record):
            if isinstance(record, SpinSectorFallbackRecord):
                fired.append(record)
                raise Stop

        directory, problem = _vacancy((0.0, 0.0, 0.0), temperature=10.0)
        self.addCleanup(shutil.rmtree, directory, True)
        with mock.patch.object(SpinSectorReducer, "start_spectrum_report", return_value=None):
            system = driver.prepare_single_point(problem, backend="auto")
            with self.assertRaises(Stop):
                driver.run_scf(system, callback=callback)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].route, "weighted")
        self.assertEqual(fired[0].stage, "the SCF loop")
        self.assertEqual(fired[0].reason, "fractional occupations")
        self.assertGreater(fired[0].largest_fractional_occupation, 0.3)


@unittest.skipUnless(_device_available(), "CUDA runtime and native extension are unavailable")
class GammaCellTests(unittest.TestCase):
    """The unpolarized neutral vacancy of diamond 2 x 2 x 2 cubes (Td about it) with occup.in and Fix_Number_Eigenvalues."""

    def _problem(self, **scf):
        import shutil

        from .test_gamma_spin_sectors import _vacancy

        directory, problem = _vacancy((0.0, 0.0, 0.0))
        self.addCleanup(shutil.rmtree, directory, True)
        return replace(problem, scf=replace(problem.scf, spin_polarized=False, **scf))

    def _run(self, problem, environment):
        from parsec_python.acceleration import driver
        from parsec_python.models import SpinSectorFallbackRecord

        from ..Symmetry import PointGroupReduction

        import warnings

        records = []
        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="auto")
            result = driver.run_scf(system, callback=records.append)
        reduction = system.backend.symmetry_eigensolver.decomposition.reduction
        notes = [record.route for record in records if isinstance(record, SpinSectorFallbackRecord)]
        return result, isinstance(reduction, PointGroupReduction), notes

    def test_occupations(self) -> None:
        states = int(4 * 63 // 2 * 1.1) + 8
        # the t2 shell in the gap with two electrons: a third per partner (equal), or the first partner filled
        equal = (1.0,) * 125 + (1 / 3, 1 / 3, 1 / 3) + (0.0,) * (states - 128)
        aufbau = (1.0,) * 126 + (0.0,) * (states - 126)
        for name, values in (("equal", equal), ("aufbau", aufbau)):
            # (one partner of a degenerate shell filled does not converge in the abelian sectors either, A100: 200
            # iterations at 1e-10 Ry; 20 iterations of both runs are compared)
            limit = {"max_iterations": 20} if name == "aufbau" else {}
            problem = self._problem(fermi_temperature_kelvin=-1.0, fixed_occupations=values, **limit)
            weighted, kept, notes = self._run(problem, {"PARSEC_ORBITAL_GROUP_OCCUPATIONS": "auto"})
            abelian, abelian_kept, _ = self._run(problem, {"PARSEC_ORBITAL_GROUP_OCCUPATIONS": "off"})
            with self.subTest(occupations=name):
                self.assertFalse(abelian_kept)
                if name == "aufbau":
                    # one partner of the t2 shell filled: the run starts again in the abelian sectors
                    self.assertEqual(notes, ["weighted"])
                    self.assertFalse(kept)
                    self.assertEqual(weighted.iterations, abelian.iterations)
                    self.assertAlmostEqual(weighted.energies.total, abelian.energies.total, delta=1e-6)
                    continue
                self.assertTrue(weighted.converged and abelian.converged)
                self.assertTrue(kept)
                self.assertEqual(notes, [])
                self.assertAlmostEqual(weighted.energies.total, abelian.energies.total, delta=1e-8)

    def test_fixed_state_count(self) -> None:
        problem = self._problem()
        problem = replace(problem, eigensolver=replace(problem.eigensolver, fixed_state_count=True))
        weighted, kept, notes = self._run(problem, {"PARSEC_ORBITAL_GROUP_FIXED_COUNT": "auto"})
        abelian, abelian_kept, _ = self._run(problem, {"PARSEC_ORBITAL_GROUP_FIXED_COUNT": "off"})
        self.assertTrue(kept)
        self.assertFalse(abelian_kept)
        self.assertEqual(notes, [])
        self.assertAlmostEqual(weighted.energies.total, abelian.energies.total, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
