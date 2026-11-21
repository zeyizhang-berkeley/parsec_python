"""The note of an SCF loop that did not converge with its highest computed state occupied (``Output/state_truncation``).

An 8-atom silicon cube at the Gamma point with ``States_Num: 20`` and ``Fermi_Temp: 300`` cuts its six-fold lowest
conduction level after four states, each occupied 1e-4: the eigensolver returns another part of the multiplet in every
iteration and the loop keeps its residual near 2e-4 Ry.  Checked here (CPU): the note for that spectrum, none for a
converged loop or an empty highest state, the blocks of k-points and spin channels, and the report of ``parsec.out``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.Output.state_truncation import truncated_states_note

# the last iteration of the Si8 cube (A100): states 14-16 the top of the valence band, 17-20 a part of the six-fold level
_SI8_LEVELS = np.array([0.8404112559, 0.8404112785, 0.8404112888, 0.8753238423, 0.8753239959, 0.8753241562,
                        0.8753242553])
_SI8_OCCUPATIONS = np.array([0.9999, 0.9999, 0.9999, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4])


def _result(levels, occupations, *, converged=False, k_points=None, spin=None):
    return SimpleNamespace(
        converged=converged, eigenvalues=np.asarray(levels), occupations=np.asarray(occupations), k_points=k_points,
        spin=spin,
    )


class NoteTests(unittest.TestCase):
    def test_the_cut_multiplet(self) -> None:
        lines = truncated_states_note(_result(_SI8_LEVELS, _SI8_OCCUPATIONS))
        self.assertEqual(len(lines), 4)
        self.assertIn("state 7 at 0.875324 Ry, the last of 4 levels within 1e-05 Ry", lines[0])
        self.assertIn("occupation of 1.0e-04", lines[0])
        self.assertIn("raise States_Num", lines[2])

    def test_no_note(self) -> None:
        self.assertEqual(truncated_states_note(_result(_SI8_LEVELS, _SI8_OCCUPATIONS, converged=True)), [])
        empty = _SI8_OCCUPATIONS.copy()
        empty[3:] = 1.0e-9
        self.assertEqual(truncated_states_note(_result(_SI8_LEVELS, empty)), [])
        self.assertEqual(truncated_states_note(SimpleNamespace(converged=False)), [])

    def test_a_single_top_state(self) -> None:
        lines = truncated_states_note(_result(_SI8_LEVELS[:4], _SI8_OCCUPATIONS[:4]))
        self.assertIn("state 4 at 0.875324 Ry, holds", lines[0])

    def test_kpoints_and_spin_channels(self) -> None:
        levels = np.concatenate([_SI8_LEVELS, _SI8_LEVELS + 0.1])
        occupations = np.concatenate([np.full(7, 0.5) * [1, 1, 1, 0, 0, 0, 0], _SI8_OCCUPATIONS])
        lines = truncated_states_note(_result(levels, occupations, k_points=np.zeros((2, 3))))
        self.assertIn("(k-point 2)", lines[0])
        lines = truncated_states_note(_result(levels, occupations, spin=(None, None)))
        self.assertIn("(down)", lines[0])
        # a spin-polarized k-point result keeps the channels apart
        spin = SimpleNamespace(
            converged=False, k_points=np.zeros((2, 3)), eigenvalues_up=levels, occupations_up=np.zeros(14),
            eigenvalues_down=levels, occupations_down=occupations,
        )
        self.assertIn("(down, k-point 2)", truncated_states_note(spin)[0])


class ReportTests(unittest.TestCase):
    def test_the_self_consistent_soc_report_prints_the_note(self) -> None:
        # the spinors of two k-points (k-major), the second with its highest computed spinor occupied 1e-4
        from parsec_python.Output.parsec_output import ParsecTextReporter

        written = []
        translation = SimpleNamespace(
            problem=SimpleNamespace(periodic_cell=None, grid=SimpleNamespace(), scf=SimpleNamespace(spin_polarized=False))
        )
        reporter = ParsecTextReporter(written.append, translation)
        levels = np.concatenate([_SI8_LEVELS, _SI8_LEVELS + 0.1])
        occupations = np.concatenate([np.array([1, 1, 1, 0, 0, 0, 0], dtype=float), _SI8_OCCUPATIONS])
        for converged in (False, True):
            result = SimpleNamespace(
                converged=converged, eigenvalues=levels, occupations=occupations, magnetic_moment_per_state=0 * levels,
                k_points=np.zeros((2, 3)), k_weights=np.full(2, 0.5), magnetic_moment_vector=None,
                magnetic_moment=None, density=None, atoms=(), energies=SimpleNamespace(total=-1.0), timings=None,
            )
            reporter.finish_self_consistent_soc(result, 1.0)
        unconverged, converged_text = written
        status_line = unconverged.index("Maximum SCF iterations reached without convergence.")
        self.assertLess(status_line, unconverged.index("NOTE: the highest computed state (k-point 2), state 7"))
        self.assertNotIn("NOTE: the highest computed state", converged_text)

    def test_the_report_prints_the_note_after_the_status(self) -> None:
        # a silicon atom whose three p levels hold two electrons, two of them computed, stopped after two iterations
        import shutil
        import tempfile
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        from pathlib import Path

        from parsec_python.cli import main

        data = Path(__file__).resolve().parents[1] / "data"
        lines = [
            "Boundary_Sphere_Radius: 6.0", "Grid_Spacing: 0.6", "Expansion_Order: 6", "States_Num: {states}",
            "Max_Iter: 2", "Convergence_Criterion: 1.0e-10", "Fermi_Temp: 300.0", "Atom_Types_Num: 1",
            "Atom_Type: Si", "Local_Component: p", "begin Atom_Coord", "  0.0 0.0 0.0", "end Atom_Coord",
        ]
        logs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for states in (3, 4):
                directory = Path(name) / str(states)
                directory.mkdir()
                (directory / "parsec.in").write_text("\n".join(lines).format(states=states) + "\n", encoding="utf-8")
                shutil.copy(data / "Si_POTRE.DAT", directory / "Si_POTRE.DAT")
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    status = main([str(directory / "parsec.in"), "--quiet", "--no-archive"])
                self.assertEqual(status, 3)
                logs[states] = (directory / "parsec.out").read_text(encoding="utf-8")
        text = logs[3]
        status_line = text.index("Maximum SCF iterations reached without convergence.")
        note = text.index("NOTE: the highest computed state, state 3")
        self.assertLess(status_line, note)
        self.assertIn("raise States_Num past the multiplet", text)
        # all three p levels computed: the highest one holds 1/3 of two electrons, but it closes its multiplet only
        # in the eigensolver's sense; the note names the occupation (and the levels within 1e-5 Ry)
        self.assertIn("NOTE: the highest computed state, state 4", logs[4])


if __name__ == "__main__":
    unittest.main()
