"""More states where an SCF loop stalls on a truncated spectrum (``SCF/state_extension``, PARSEC_STATES_CLOSE_MULTIPLET).

The 8-atom silicon cube at the Gamma point (h 0.5 bohr, ``Fermi_Temp`` 300 K) with ``States_Num: 21`` cuts its six-fold
lowest conduction level after five states, each occupied 8e-5: the reference solver keeps its weighted residual near
5e-6 Ry, so a criterion of 1e-8 Ry is never met.  Checked here (CPU): the stall test and the size of an extension on
synthetic levels, the switch values (``off`` by default), and the reference SCF of that cube with ``auto`` (the
partner and the next level once it stalls, converged) and ``off`` (``States_Num`` kept, not converged).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python.SCF.state_extension import (
    MAXIMUM_EXTENSIONS,
    SWITCH,
    MultipletCloser,
    close_multiplet_setting,
    identity_reduction,
    occupied_threshold,
)

_ROOT = Path(__file__).resolve().parents[4]
# the levels of the Si8 cube in its sixth iteration: states 15-21 returned, the working set from state 15 to 27
_LEVELS = np.array([0.8403848, 0.8403951, 0.8753137, 0.8753154, 0.8753208, 0.8753224, 0.8753249])
_WORKING = np.array([0.8403848, 0.8403951, 0.8753137, 0.8753154, 0.8753208, 0.8753224, 0.8753249, 0.8753278,
                     1.019065, 1.019069, 1.0190733, 1.0749568, 1.4000634])
_FERMI = 0.8578
_STALLED = [0.068, 0.041, 2.3e-3, 9.4e-4, 2.4e-4, 1.0e-4, 2.9e-5, 1.8e-5, 1.3e-5, 5.1e-6, 4.6e-6, 5.3e-6, 5.4e-6, 4.6e-6,
            4.3e-6, 4.0e-6, 4.6e-6, 5.4e-6, 4.6e-6, 4.4e-6]
_FALLING = [0.068, 0.041, 2.3e-3, 9.4e-4, 2.4e-4, 1.0e-4, 2.9e-5, 9.0e-6, 2.5e-6, 7.0e-7, 2.0e-7, 6.0e-8, 2.0e-8]


def _channel(occupation=7.9e-5, label=""):
    return (label, _LEVELS, np.where(_LEVELS > 0.86, occupation, 1.0), _WORKING)


def _closer(criterion=1.0e-8, temperature=300.0, **keywords):
    return MultipletCloser(1000, criterion, temperature, **keywords)


class CloserTests(unittest.TestCase):
    def test_a_stalled_loop_takes_the_buffer_down_to_the_threshold(self) -> None:
        closer = _closer()
        change = closer.check(len(_STALLED) + 1, 7, (_channel(),), _STALLED, _FERMI)
        self.assertIsNotNone(change)
        # the sixth partner (held 8e-5) and the next level, whose occupation is below 3e-8
        self.assertEqual((change.previous, change.states), (7, 9))
        text = "\n".join(change.lines())
        self.assertIn("States_Num 7 -> 9", text)
        self.assertIn(SWITCH, text)
        self.assertEqual(closer.restart_iteration, len(_STALLED) + 1)

    def test_no_change_where_the_loop_converges_or_the_top_is_empty(self) -> None:
        closer = _closer()
        self.assertIsNone(closer.check(len(_FALLING) + 1, 7, (_channel(),), _FALLING, _FERMI))
        # too early: the window must start three iterations after the start of the loop
        self.assertIsNone(closer.check(6, 7, (_channel(),), _STALLED[:5], _FERMI))
        # the highest state holds less than the threshold
        self.assertIsNone(closer.check(len(_STALLED) + 1, 7, (_channel(1.0e-9),), _STALLED, _FERMI))
        # a loose criterion: the residual is already below it
        self.assertIsNone(_closer(1.0e-5).check(len(_STALLED) + 1, 7, (_channel(),), _STALLED, _FERMI))
        # no working levels kept by the solver
        no_levels = ("", _LEVELS, _channel()[2], None)
        self.assertIsNone(closer.check(len(_STALLED) + 1, 7, (no_levels,), _STALLED, _FERMI))
        self.assertEqual(closer.extensions, 0)

    def test_the_window_starts_again_after_an_extension(self) -> None:
        closer = _closer()
        self.assertIsNotNone(closer.check(len(_STALLED) + 1, 7, (_channel(),), _STALLED, _FERMI))
        longer = _STALLED + [5.0e-6] * 12
        self.assertIsNone(closer.check(len(_STALLED) + 6, 7, (_channel(),), longer[: len(_STALLED) + 5], _FERMI))
        self.assertIsNotNone(closer.check(len(longer) + 1, 7, (_channel(),), longer, _FERMI))
        self.assertEqual(closer.extensions, 2)

    def test_limits(self) -> None:
        closer = MultipletCloser(8, 1.0e-8, 300.0)
        change = closer.check(len(_STALLED) + 1, 7, (_channel(),), _STALLED, _FERMI)
        self.assertEqual(change.states, 8)
        closer = _closer()
        residuals = list(_STALLED)
        for _ in range(MAXIMUM_EXTENSIONS):
            self.assertIsNotNone(closer.check(len(residuals) + 1, 7, (_channel(),), residuals, _FERMI))
            residuals += [5.0e-6] * 12
        self.assertIsNone(closer.check(len(residuals) + 1, 7, (_channel(),), residuals, _FERMI))
        # the inner loops of a hybrid keep a count and add none
        self.assertIsNone(_closer(extend=False).check(len(_STALLED) + 1, 7, (_channel(),), _STALLED, _FERMI))

    def test_the_threshold_follows_the_criterion(self) -> None:
        self.assertAlmostEqual(occupied_threshold(1.0e-8), 3.0e-8)
        self.assertAlmostEqual(occupied_threshold(2.0e-4), 6.0e-4)
        with self.assertRaises(ValueError):
            occupied_threshold(0.0)

    def test_a_continued_loop_keeps_the_raised_count(self) -> None:
        closer = MultipletCloser(30, 1.0e-8, 300.0)
        self.assertEqual(closer.carried(21, (SimpleNamespace(requested_states=23), None)), 23)
        self.assertEqual(closer.carried(21, (SimpleNamespace(requested_states=21),)), 21)
        self.assertEqual(closer.carried(21, (None,)), 21)
        # never beyond what the grid holds
        self.assertEqual(closer.carried(21, (SimpleNamespace(requested_states=40),)), 21)

    def test_the_larger_need_of_two_channels(self) -> None:
        empty = ("up", _LEVELS, np.where(_LEVELS > 0.86, 0.0, 1.0), _WORKING)
        change = _closer().check(len(_STALLED) + 1, 7, (empty, _channel(label="down")), _STALLED, _FERMI)
        self.assertEqual(change.channel, "down")

    def test_identity_reduction(self) -> None:
        self.assertTrue(identity_reduction(SimpleNamespace(reduction=SimpleNamespace(wedge_size=9, full_size=9))))
        self.assertFalse(identity_reduction(SimpleNamespace(reduction=SimpleNamespace(wedge_size=3, full_size=9))))
        spin = SimpleNamespace(reducer=SimpleNamespace(reduction=SimpleNamespace(wedge_size=9, full_size=9)))
        self.assertTrue(identity_reduction(spin))
        self.assertFalse(identity_reduction(SimpleNamespace()))

    def test_setting(self) -> None:
        with mock.patch.dict(os.environ, {SWITCH: ""}):
            self.assertEqual(close_multiplet_setting(), "off")
        with mock.patch.dict(os.environ, {SWITCH: "AUTO"}):
            self.assertEqual(close_multiplet_setting(), "auto")
        with mock.patch.dict(os.environ, {SWITCH: "on"}):
            with self.assertRaises(ValueError):
                close_multiplet_setting()


_SI8 = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
 10.26 10.26 10.26
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.5
Expansion_Order: 8
States_Num: 21
Fermi_Temp: 300.0
Convergence_Criterion: 1.0e-8
Max_Iter: 40
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
  0.0 0.0 0.0
  0.0 5.13 5.13
  5.13 0.0 5.13
  5.13 5.13 0.0
  2.565 2.565 2.565
  2.565 7.695 7.695
  7.695 2.565 7.695
  7.695 7.695 2.565
end Atom_Coord
"""


class ReferenceLoopTests(unittest.TestCase):
    """The reference SCF of the Si8 cube (CPU, a few seconds per run)."""

    def _run(self, value: str):
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            folder = Path(name)
            (folder / "parsec.in").write_text(_SI8)
            shutil.copy(_ROOT / "examples/3d_Si/Si_POTRE.DAT", folder / "Si_POTRE.DAT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                translation = parse_parsec_input(folder / "parsec.in")
            steps = []
            with mock.patch.dict(os.environ, {SWITCH: value}):
                system = prepare_periodic_single_point(translation.problem)
                result = run_scf(system, callback=steps.append)
        return result, steps

    def test_auto_closes_the_level_and_converges(self) -> None:
        result, steps = self._run("auto")
        self.assertTrue(result.converged)
        notes = [step for step in steps if step.state_count_note]
        self.assertEqual(len(notes), 1)
        self.assertIn("States_Num 21 -> 23", "\n".join(notes[0].state_count_note))
        self.assertEqual(np.asarray(result.eigenvalues).size, 23)
        self.assertLess(len(steps), 35)

    def test_off_keeps_the_states(self) -> None:
        result, steps = self._run("off")
        self.assertFalse(result.converged)
        self.assertFalse(any(step.state_count_note for step in steps))
        self.assertEqual(np.asarray(result.eigenvalues).size, 21)


if __name__ == "__main__":
    unittest.main()
