"""Verdicts of the consistency matrix (``acceleration/benchmarks/consistency_matrix.py``) on non-finite numbers.

A NaN force, slope, energy or level compares false with every tolerance: the matrix judged ``d > tolerance`` for the
forces, so a NaN force passed.  Every comparison now takes ``_within`` (finite and within the tolerance); finite
numbers keep their verdicts.  CPU only: the checks are judged on hand-made results.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.acceleration.benchmarks import consistency_matrix as matrix


def _result(free, forces=None, bands=None):
    result = matrix.Result(status=0)
    result.free = free
    result.energy = free
    if forces is not None:
        result.forces = np.asarray(forces, dtype=float)
    if bands is not None:
        result.bands = np.asarray(bands, dtype=float)
    return result


def _force_results(force, slope, steps=matrix.FORCE_STEPS):
    """A base result whose force on atom 0 along x is ``force`` and displaced results whose free energies give the
    central difference ``slope`` at both ``steps``."""

    results = {"base": _result(-1.0, [[force, 0.0, 0.0]])}
    for step in steps:
        for sign in (1.0, -1.0):
            # -(E(+h) - E(-h)) / 2h = slope
            results[(0, 0, step, sign)] = _result(-1.0 - sign * slope * step)
    return results


class ConsistencyMatrixVerdictTests(unittest.TestCase):
    def check(self):
        return matrix.ForceCheck("force", "f_test", "", None, ((0, 0),), None)

    def test_a_force_within_the_tolerance_passes(self) -> None:
        verdicts = self.check().judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-7))
        self.assertEqual([verdict for verdict, _ in verdicts], ["PASS"])

    def test_a_force_off_the_slope_fails(self) -> None:
        verdicts = self.check().judge(_force_results(1.0e-3, 2.0e-3))
        self.assertEqual([verdict for verdict, _ in verdicts], ["FAIL"])

    def test_a_non_finite_force_or_slope_fails(self) -> None:
        for force, slope in ((float("nan"), 1.0e-3), (1.0e-3, float("nan")), (float("inf"), 1.0e-3)):
            with self.subTest(force=force, slope=slope):
                verdicts = self.check().judge(_force_results(force, slope))
                self.assertEqual([verdict for verdict, _ in verdicts], ["FAIL"])
        # (an exempt atom too: a NaN is never a step that may miss)
        exempt = matrix.ForceCheck("force", "f_test", "", None, ((0, 0),), None,
                                   exempt={0: (matrix.FORCE_STEPS, "test")})
        self.assertEqual([v for v, _ in exempt.judge(_force_results(float("nan"), 1.0e-3))], ["FAIL"])

    def test_a_case_with_its_own_steps_moves_and_judges_at_those_steps(self) -> None:
        fine = matrix.ForceCheck("force", "f_test", "", matrix.Structure("t", str, np.zeros((1, 3)), ()),
                                 ((0, 0),), None, steps=matrix.FINE_STEPS)
        moves = {key[2] for key in fine.jobs() if key != "base"}
        self.assertEqual(moves, set(matrix.FINE_STEPS))
        verdicts = fine.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-7, matrix.FINE_STEPS))
        self.assertEqual([verdict for verdict, _ in verdicts], ["PASS"])
        self.assertIn(f"slope({matrix.FINE_STEPS[0]:g})", verdicts[0][1])
        # the cases of the matrix that take them
        names = {check.name: check.steps for check in matrix.force_checks()}
        self.assertEqual(names["f_au2_pbe_fine"], matrix.FINE_STEPS)
        self.assertEqual(names["f_au2c_pbe_fine"], matrix.FINE_STEPS)
        self.assertEqual(names["f_c8p_d_grid"], matrix.FORCE_STEPS)

    def test_a_bounded_exemption_fails_beyond_its_bound(self) -> None:
        bounded = matrix.ForceCheck("force", "f_test", "", None, ((0, 0),), None,
                                    exempt={0: (matrix.FORCE_STEPS, "test", 5.0e-4)})
        self.assertEqual([v for v, _ in bounded.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-4))], ["EXEMPT"])
        self.assertIn("exempt up to 5e-04", bounded.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-4))[0][1])
        self.assertEqual([v for v, _ in bounded.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-3))], ["FAIL"])
        self.assertEqual([v for v, _ in bounded.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-7))], ["PASS"])
        # without a bound, as before
        unbounded = matrix.ForceCheck("force", "f_test", "", None, ((0, 0),), None,
                                      exempt={0: (matrix.FORCE_STEPS, "test")})
        self.assertEqual([v for v, _ in unbounded.judge(_force_results(1.0e-3, 1.0e-3 + 1.0e-3))], ["EXEMPT"])

    def test_non_finite_energies_forces_and_levels_fail_the_comparisons(self) -> None:
        symmetry = matrix.SymmetryCheck("symmetry", "s_test", "")
        same = {"sym": _result(-1.0, [[0.0, 0.0, 0.0]]), "off": _result(-1.0, [[0.0, 0.0, 0.0]])}
        self.assertEqual(symmetry.judge(same)[0][0], "PASS")
        for sym in (_result(float("nan"), [[0.0, 0.0, 0.0]]), _result(-1.0, [[float("nan"), 0.0, 0.0]])):
            self.assertEqual(symmetry.judge({"sym": sym, "off": same["off"]})[0][0], "FAIL")
        repeat = matrix.RepeatCheck("repeat", "r_test", "")
        self.assertEqual(repeat.judge({0: _result(float("nan")), 1: _result(-1.0)})[0][0], "FAIL")
        # band levels: NaN at the same places in both runs is left out, a level finite in one run only fails
        a = _result(-1.0, bands=[[0.1, 0.2, float("nan")]])
        b = _result(-1.0, bands=[[0.1, 0.2, float("nan")]])
        self.assertIsNot(matrix.compare_extras(a, b)[1], False)
        c = _result(-1.0, bands=[[0.1, float("nan"), 0.3]])
        self.assertIs(matrix.compare_extras(a, c)[1], False)
        d = _result(-1.0, bands=[[float("nan")] * 3])
        self.assertIs(matrix.compare_extras(d, d)[1], False)


if __name__ == "__main__":
    unittest.main()
