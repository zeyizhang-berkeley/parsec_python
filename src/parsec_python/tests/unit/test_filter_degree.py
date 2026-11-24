"""The later-SCF Chebyshev degree near convergence and its restoration in a stalled loop (``SCF/filter_degree``).

``PARSEC_FILTER_DEGREE_STALL=off`` must lower the degree exactly as PARSEC 1.5's rule (``parsec.F90`` l.1942-1952:
one per iteration beyond ``Chebyshev_Degree_Decrease_After`` while the weighted residual is below the approach
criterion, down to the minimum, never raised); ``auto`` must agree with it while the residual halves within
:data:`STALL_WINDOW` iterations of the residual at the last reset (the anchor), and give the starting degree back, for
good, after :data:`STALL_WINDOW` iterations without halving it.  The NOTE goes to one record, and to none where
PARSEC's spectral-width floor already keeps the degree.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

import parsec_python.Eigensolvers.subspace as subspace
from parsec_python.SCF.filter_degree import (
    DEFAULT,
    STALL_DECADE,
    STALL_PROGRESS,
    STALL_WINDOW,
    SWITCH,
    LaterFilterDegree,
    filter_degree_stall_setting,
)


def parsec_rule(residuals, *, degree=15, minimum=10, approach=1.0e-4, after=4):
    """The degrees of PARSEC 1.5's rule after each iteration (the code this module replaced in the loops)."""

    degrees = []
    for iteration, residual in enumerate(residuals, start=1):
        if iteration > after and residual < approach and degree > minimum:
            degree -= 1
        degrees.append(degree)
    return degrees


def schedule_degrees(residuals, *, restore, degree=15, minimum=10, approach=1.0e-4, after=4):
    schedule = LaterFilterDegree(degree=degree, minimum=minimum, restore=restore)
    return [schedule.after_iteration(i, r, approach, after) for i, r in enumerate(residuals, start=1)], schedule


#: dropped at iteration 6 (9e-5 below the approach criterion 1e-4), then a plateau that never halves 9e-5
STALLED = [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 9e-5, 8e-5] + [8.5e-5] * 40


class SwitchTests(unittest.TestCase):
    def test_default_and_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SWITCH, None)
            self.assertEqual(filter_degree_stall_setting(), DEFAULT)
        for value, expected in (("auto", "auto"), (" OFF ", "off"), ("", DEFAULT)):
            with mock.patch.dict(os.environ, {SWITCH: value}):
                self.assertEqual(filter_degree_stall_setting(), expected)
        with mock.patch.dict(os.environ, {SWITCH: "on"}):
            with self.assertRaises(ValueError):
                filter_degree_stall_setting()

    def test_for_loop_reads_the_switch(self):
        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            self.assertFalse(LaterFilterDegree.for_loop(15, 10).restore)
        with mock.patch.dict(os.environ, {SWITCH: "auto"}):
            self.assertTrue(LaterFilterDegree.for_loop(15, 10).restore)


class ScheduleTests(unittest.TestCase):
    def test_off_is_parsec_rule_on_random_histories(self):
        rng = np.random.default_rng(20261123)
        for _ in range(200):
            residuals = 10.0 ** rng.uniform(-7.0, -2.0, size=int(rng.integers(1, 60)))
            approach = float(10.0 ** rng.uniform(-6.0, -3.0))
            after = int(rng.integers(0, 6))
            minimum = int(rng.integers(4, 14))
            degree = int(rng.integers(minimum - 2, 25))
            expected = parsec_rule(residuals, degree=degree, minimum=minimum, approach=approach, after=after)
            got, _ = schedule_degrees(residuals, restore=False, degree=degree, minimum=minimum, approach=approach,
                                      after=after)
            self.assertEqual(got, expected)

    def test_steady_decay_never_restores(self):
        # any decay that halves the residual within the window, from just below the approach criterion
        for rate in (0.4, 0.8, 0.9, 0.92, 0.95):
            residuals = list(1.2e-4 * rate ** np.arange(400))
            got, schedule = schedule_degrees(residuals, restore=True)
            self.assertIsNone(schedule.restored_at, rate)
            self.assertEqual(got, parsec_rule(residuals))

    def test_auto_restores_after_a_stall(self):
        got, schedule = schedule_degrees(STALLED, restore=True)
        off = parsec_rule(STALLED)
        restored = 6 + STALL_WINDOW
        self.assertEqual(schedule.restored_at, restored)
        self.assertEqual(got[: restored - 1], off[: restored - 1])
        self.assertEqual(got[restored - 1:], [15] * (len(STALLED) - restored + 1))
        self.assertEqual(off[-1], 10)

    def test_a_halving_of_the_anchor_restarts_the_count(self):
        plateau = [8.5e-5] * (STALL_WINDOW - 2)
        # 4e-5 halves the anchor 9e-5: a new anchor; 3.9e-5 does not halve it again within the window
        residuals = [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 9e-5] + plateau + [4e-5] + [3.9e-5] * (STALL_WINDOW - 1) + [1e-6]
        got, schedule = schedule_degrees(residuals, restore=True)
        self.assertIsNone(schedule.restored_at)
        self.assertEqual(got, parsec_rule(residuals))
        # two more iterations on the plateau before the halving give the degree back
        residuals = [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 9e-5] + plateau + [8.5e-5, 8.5e-5, 4e-5]
        _, schedule = schedule_degrees(residuals, restore=True)
        self.assertEqual(schedule.restored_at, 6 + STALL_WINDOW)
        self.assertEqual(STALL_PROGRESS, 0.5)

    def test_progress_is_measured_against_the_anchor_not_the_running_minimum(self):
        # 0.97 per iteration: the residual takes 23 iterations to halve the anchor, so the count reaches the window;
        # a running minimum would also have counted every 0.9 decay (test_steady_decay_never_restores) as a stall
        residuals = list(9.9e-5 * 0.97 ** np.arange(60))
        _, schedule = schedule_degrees(residuals, restore=True)
        self.assertEqual(schedule.restored_at, 1 + STALL_WINDOW + 4)

    def test_restoration_is_final(self):
        residuals = [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 9e-5] + [9.5e-5] * STALL_WINDOW + [1e-6, 1e-7, 1e-8]
        got, schedule = schedule_degrees(residuals, restore=True)
        self.assertIsNotNone(schedule.restored_at)
        self.assertEqual(got[-3:], [15, 15, 15])

    def test_no_restoration_below_the_decade_of_the_approach_criterion(self):
        # dropped at 9e-5 (approach 1e-4), converged on to 5e-6 (below 1e-5), then a plateau: another cause
        residuals = [1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 9e-5, 3e-5, 1e-5, 5e-6] + [6e-6] * (2 * STALL_WINDOW)
        got, schedule = schedule_degrees(residuals, restore=True)
        self.assertIsNone(schedule.restored_at)
        self.assertEqual(got, parsec_rule(residuals))
        # an erratic loop of a loose criterion (5e-3 Ry, approach 0.5): dropped at the fifth iteration, converged
        # thirty-fold, then swinging
        erratic = [0.36, 0.24, 0.17, 0.45, 0.37, 0.37, 0.16, 0.14, 0.042, 0.23, 0.19, 0.077, 0.076, 0.019, 0.073,
                   0.087, 0.082, 0.013, 0.14, 0.065, 0.027, 0.24, 0.082, 0.072, 0.068, 0.023] * 3
        got, schedule = schedule_degrees(erratic, restore=True, approach=0.5)
        self.assertIsNone(schedule.restored_at)
        self.assertEqual(got, parsec_rule(erratic, approach=0.5))
        self.assertEqual(STALL_DECADE, 0.1)

    def test_no_restoration_before_the_first_drop(self):
        residuals = [1e-2] * 30
        got, schedule = schedule_degrees(residuals, restore=True)
        self.assertEqual(got, [15] * 30)
        self.assertIsNone(schedule.restored_at)


class NoteTests(unittest.TestCase):
    def setUp(self):
        self._floor = subspace._LAST_SPECTRAL_FLOOR[0]
        subspace._LAST_SPECTRAL_FLOOR[0] = 0

    def tearDown(self):
        subspace._LAST_SPECTRAL_FLOOR[0] = self._floor

    def test_the_note_goes_to_one_record(self):
        schedule = LaterFilterDegree(degree=15, minimum=10, restore=True)
        notes = []
        for iteration, residual in enumerate(STALLED, start=1):
            notes.append(schedule.take_note())
            schedule.after_iteration(iteration, residual, 1.0e-4, 4)
            notes.append(schedule.take_note())
        lines = [note for note in notes if note]
        self.assertEqual(len(lines), 1)
        (line,) = lines[0]
        self.assertIn("returns to 15", line)
        self.assertIn(f"lowered to {schedule.lowest}", line)
        self.assertIn(f"after iteration {6 + STALL_WINDOW}", line)
        self.assertIn(SWITCH, line)
        _, off = schedule_degrees(STALLED, restore=False)
        self.assertEqual(off.take_note(), ())

    def test_no_note_where_the_spectral_width_floor_keeps_the_degree(self):
        for floor, expect_note, shown in ((15, False, None), (16, False, None), (11, True, 11), (8, True, 10)):
            subspace._LAST_SPECTRAL_FLOOR[0] = floor
            _, schedule = schedule_degrees(STALLED, restore=True)
            note = schedule.take_note()
            self.assertEqual(bool(note), expect_note, floor)
            if note:
                self.assertIn(f"lowered to {shown}", note[0])

    def test_the_floor_of_the_table(self):
        self.assertEqual([subspace.spectral_width_floor(0.0, width) for width in (1500, 1300, 1100, 900, 500)],
                         [15, 11, 9, 8, 0])
        self.assertEqual(subspace._adapt_polynomial_degree(10, 0.0, 1500.0), 15)
        self.assertEqual(subspace.last_spectral_width_floor(), 15)
        self.assertEqual(subspace._adapt_polynomial_degree(12, 0.0, 500.0), 12)
        self.assertEqual(subspace.last_spectral_width_floor(), 0)


if __name__ == "__main__":
    unittest.main()
