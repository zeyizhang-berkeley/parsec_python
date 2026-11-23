"""The Fermi level of a gap at the root of the electron count (``PARSEC_FERMI_GAP_ROOT``).

The bisection of ``flevel.f90`` stops at the first level whose count is within
1e-10 of the target.  In a gap the count differs from the target only by the
thermal tails, so the level it returns can lie anywhere in the gap where those
are below the tolerance, set by the bracket of the search and the levels it was
given: the three searches of this package put CO at -0.3494 (the unpolarized
bisection), -0.4486 (the bisection of ``flevel.f90`` that the NSCF run repeats)
and anywhere else for other level lists.  With ``on`` (default) every search
returns the root of the count, ``midgap + (kT/2) ln(g_h/g_l)`` for a highest
filled level of degeneracy ``g_h`` and a lowest empty one of ``g_l``; metals and
narrow gaps keep the level of the bisection bit for bit; ``off`` restores the
bisection everywhere.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.Occupations import fermi_occupations, spin_fermi_occupations, weighted_fermi_occupations
from parsec_python.Occupations.fermi_dirac import (
    BOLTZMANN_RYDBERG_PER_KELVIN,
    FERMI_GAP_ROOT_ENV,
    fermi_gap_root_setting,
    gap_fermi_level,
)
from parsec_python.Occupations.spin import weighted_spin_fermi_occupations
from parsec_python.SCF.kpoint_weights import KPointWeights
from parsec_python.SCF.nscf import fermi_level_from_levels

# The converged levels (Ry) of CO, PBE, 1160.45 K, of the core-level example: one filled level at
# -0.666, an empty pair at -0.150 and -0.149 (a gap of 70 kT).
CO = np.array([
    -2.1398824130, -1.0364550731, -0.8632279908, -0.8630961588, -0.6660753672, -0.1498425462,
    -0.1485366185, 0.0386746657, 0.1238278309, 0.1402783253, 0.1571645774, 0.2287693773,
    0.2461171193, 0.2826481553, 0.2968879961, 0.3201591045,
])
CO_TEMPERATURE = 1160.451812
#: The levels of the bisections on b72375f60 (bit for bit): the unpolarized search and the search of
#: flevel.f90 (NSCF); the root of the count, which every search reaches to a few units in the last
#: place (ROOT_PLACES: the Newton steps end within four of them, from the level each bisection found).
CO_BISECTION = -0.34938620284010424
CO_FLEVEL = -0.44860386971874994
CO_ROOT = -0.4101940865403358
ROOT_PLACES = 4.0 * np.spacing(abs(CO_ROOT))


def _switch(value: str | None):
    environment = {key: item for key, item in os.environ.items() if key != FERMI_GAP_ROOT_ENV}
    if value is not None:
        environment[FERMI_GAP_ROOT_ENV] = value
    return mock.patch.dict(os.environ, environment, clear=True)


def _two_level_root(highest: float, lowest: float, filled: float, empty: float, kbt: float) -> float:
    return 0.5 * (highest + lowest) + 0.5 * kbt * np.log(filled / empty)


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with _switch(None):
            self.assertEqual(fermi_gap_root_setting(), "on")
        with _switch("OFF"):
            self.assertEqual(fermi_gap_root_setting(), "off")
        with _switch("auto"), self.assertRaisesRegex(ValueError, "PARSEC_FERMI_GAP_ROOT must be on or off"):
            fermi_gap_root_setting()


class GapRootTests(unittest.TestCase):
    def test_every_search_returns_the_same_root_and_off_the_bisections(self) -> None:
        kbt = BOLTZMANN_RYDBERG_PER_KELVIN * CO_TEMPERATURE
        with _switch("off"):
            self.assertEqual(fermi_occupations(CO, 10.0, CO_TEMPERATURE).fermi_level, CO_BISECTION)
            self.assertEqual(
                weighted_fermi_occupations(CO, np.full(CO.size, 0.25), 2.5, CO_TEMPERATURE).fermi_level, CO_BISECTION
            )
            self.assertEqual(fermi_level_from_levels([[CO]], [1.0], 10.0, CO_TEMPERATURE), CO_FLEVEL)
        with _switch(None):
            result = fermi_occupations(CO, 10.0, CO_TEMPERATURE)
            self.assertAlmostEqual(result.fermi_level, CO_ROOT, delta=ROOT_PLACES)
            self.assertAlmostEqual(
                weighted_fermi_occupations(CO, np.full(CO.size, 0.25), 2.5, CO_TEMPERATURE).fermi_level, CO_ROOT,
                delta=ROOT_PLACES,
            )
            self.assertAlmostEqual(fermi_level_from_levels([[CO]], [1.0], 10.0, CO_TEMPERATURE), CO_ROOT,
                                   delta=ROOT_PLACES)
            # both channels of a spin-polarized run hold the same levels: the same root
            spin = spin_fermi_occupations(CO, CO, 10.0, CO_TEMPERATURE)
            self.assertAlmostEqual(spin.fermi_level, CO_ROOT, delta=ROOT_PLACES)
            self.assertEqual(spin.electron_count, 10.0)
        # electrons above the filled levels = holes below them at the root
        filled = CO[:5]
        empty = CO[5:]
        holes = np.sum(1.0 / (np.exp(-(filled - CO_ROOT) / kbt) + 1.0))
        electrons = np.sum(1.0 / (np.exp((empty - CO_ROOT) / kbt) + 1.0))
        self.assertAlmostEqual(electrons / holes, 1.0, delta=1.0e-12)
        self.assertEqual(result.electron_count, 10.0)

    def test_the_root_does_not_depend_on_the_levels_far_above(self) -> None:
        # The bisection's level moves with the bracket, which the highest level sets; the root moves
        # only by the tails of the levels added.
        with _switch(None):
            reference = fermi_occupations(CO, 10.0, CO_TEMPERATURE).fermi_level
        moved = []
        for extra in (0.25, 0.5, 2.0, 3.0):
            levels = np.append(CO, CO[-1] + extra)
            with self.subTest(extra=extra):
                with _switch(None):
                    self.assertAlmostEqual(fermi_occupations(levels, 10.0, CO_TEMPERATURE).fermi_level, reference,
                                           delta=1.0e-14)
                with _switch("off"):
                    moved.append(fermi_occupations(levels, 10.0, CO_TEMPERATURE).fermi_level)
        # -0.3800, -0.4418, -0.3776, -0.3457 Ry: 13 kT apart
        self.assertGreater(max(moved) - min(moved), 0.09)

    def test_degeneracies_and_weights(self) -> None:
        temperature = 300.0
        kbt = BOLTZMANN_RYDBERG_PER_KELVIN * temperature
        levels = np.array([-1.0, -0.5, -0.5, -0.5, 0.2, 0.9])
        with _switch(None):
            plain = fermi_occupations(levels, 8.0, temperature)
            self.assertAlmostEqual(plain.fermi_level, _two_level_root(-0.5, 0.2, 3.0, 1.0, kbt), delta=1.0e-15)
            # the same levels as one weighted state each: weight 3 at -0.5
            weighted = weighted_fermi_occupations(np.array([-1.0, -0.5, 0.2, 0.9]), np.array([1.0, 3.0, 1.0, 1.0]),
                                                  8.0, temperature)
            self.assertAlmostEqual(weighted.fermi_level, plain.fermi_level, delta=1.0e-15)
            # k-point weights, as KPointWeights pools them (equal and unequal weights): the second k-point
            # has every level 0.01 Ry higher, so its filled edge leads the holes and the first k-point's
            # empty edge the electrons: midgap -0.145, plus (kT/2) ln(3 w_2 / w_1) (to the tails of the
            # other edges, 1.3e-5 Ry for unequal weights).
            second = levels + 0.01
            stacked = np.concatenate([levels, second])
            uniform = KPointWeights(np.array([0.5, 0.5]), True, 0.5)
            unequal = KPointWeights(np.array([0.25, 0.75]), False, 0.25)
            _, uniform_level = uniform.occupy(stacked, 8.0, temperature, 2.0)
            _, unequal_level = unequal.occupy(stacked, 8.0, temperature, 2.0,
                                              unequal.state_weights([levels, second], [0, 1]))
            self.assertAlmostEqual(uniform_level, -0.145 + 0.5 * kbt * np.log(3.0), delta=1.0e-12)
            self.assertAlmostEqual(unequal_level, -0.145 + 0.5 * kbt * np.log(9.0), delta=2.0e-5)
            # spin channels with weights (weighted sectors): the same root as one list
            spin = weighted_spin_fermi_occupations(levels[:5], levels[:5], (np.ones(5), np.ones(5)), 8.0, temperature)
            self.assertAlmostEqual(spin.fermi_level, plain.fermi_level, delta=1.0e-15)
            # a state of weight 3 in each channel for the level at -0.5
            spin = weighted_spin_fermi_occupations(np.array([-1.0, -0.5, 0.2]), np.array([-1.0, -0.5, 0.2]),
                                                   (np.array([1.0, 3.0, 1.0]),) * 2, 8.0, temperature)
            self.assertAlmostEqual(spin.fermi_level, plain.fermi_level, delta=1.0e-15)

    def test_a_gap_wider_than_the_cutoff_of_the_occupations(self) -> None:
        # 200 kT: the occupations with the cut-off at 36.7 kT are exactly 0 and 1 over most of the gap; the
        # root of the exact count is still the middle (equal degeneracies).
        temperature = 100.0
        kbt = BOLTZMANN_RYDBERG_PER_KELVIN * temperature
        levels = np.array([-0.8, -0.3, -0.3 + 200.0 * kbt, 0.4])
        with _switch(None):
            result = fermi_occupations(levels, 4.0, temperature)
        self.assertAlmostEqual(result.fermi_level, -0.3 + 100.0 * kbt, delta=1.0e-15)
        np.testing.assert_array_equal(result.occupations, [1.0, 1.0, 0.0, 0.0])

    def test_metals_and_narrow_gaps_keep_the_bisection(self) -> None:
        generator = np.random.default_rng(7)
        metal = np.sort(generator.uniform(-1.0, 0.0, 300))
        narrow = np.array([-0.6, -0.31, -0.30, -0.30 + 12.0 * BOLTZMANN_RYDBERG_PER_KELVIN * 500.0, 0.2])
        cases = [(metal, 211.0, 1000.0), (metal, 100.0, 80.0), (narrow, 6.0, 500.0)]
        for levels, electrons, temperature in cases:
            with self.subTest(electrons=electrons, temperature=temperature):
                with _switch("off"):
                    off = fermi_occupations(levels, electrons, temperature)
                with _switch(None):
                    on = fermi_occupations(levels, electrons, temperature)
                self.assertEqual(on.fermi_level, off.fermi_level)
                np.testing.assert_array_equal(on.occupations, off.occupations)
        # a metal whose filled levels happen to hold the count at the bisection's level
        equal = np.array([-0.5, -0.1, -0.1 + 1.0e-4, 0.3])
        with _switch("off"):
            off = fermi_occupations(equal, 3.0, 1000.0)
        with _switch(None):
            on = fermi_occupations(equal, 3.0, 1000.0)
        self.assertEqual(on.fermi_level, off.fermi_level)

    def test_no_root_where_the_count_is_not_a_filling_of_whole_levels(self) -> None:
        kbt = BOLTZMANN_RYDBERG_PER_KELVIN * CO_TEMPERATURE
        self.assertIsNone(gap_fermi_level(CO, None, 5.0 + 3.0e-5, kbt, -0.40))
        self.assertIsNone(gap_fermi_level(CO, None, 5.0, kbt, CO[4]))  # on a level
        self.assertIsNone(gap_fermi_level(CO, None, 5.0, 0.0, -0.40))
        # zero weights do not count; a gap without an empty level of weight has no root
        weights = np.ones(CO.size)
        weights[5:] = 0.0
        self.assertIsNone(gap_fermi_level(CO, weights, 5.0, kbt, -0.40))
        weights[6] = 1.0
        root = gap_fermi_level(CO, weights, 5.0, kbt, -0.40)
        self.assertAlmostEqual(root, _two_level_root(CO[4], CO[6], 1.0, 1.0, kbt), delta=1.0e-12)

    def test_zero_temperature_is_unchanged(self) -> None:
        for value in ("off", None):
            with _switch(value):
                result = fermi_occupations(CO, 10.0, 0.0)
            self.assertEqual(result.fermi_level, 0.5 * (CO[4] + CO[5]))


if __name__ == "__main__":
    unittest.main()
