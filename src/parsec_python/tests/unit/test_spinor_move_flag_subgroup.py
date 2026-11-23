"""The Move_Flag subgroup for the spinors of non-magnetic spin-orbit coupling (``PARSEC_SPINOR_MOVE_FLAG_SUBGROUP``).

A relaxation whose ``Move_Flag`` fixes atoms that an operation of the space group exchanges with movable ones keeps
the subgroup of the operations that map the fixed atoms onto fixed atoms (``PARSEC_MOVE_FLAG_SPACE_GROUP``).  Since
2026-11-22 the spinors of self-consistent spin-orbit coupling without spin polarization or non-collinear magnetism
take it too, on both routine sets: the mesh is reduced by it and time reversal and the forces completed over it.
``off`` keeps the routes of before: the time-reversal pairs on the default routines, the refusal by name on the
Fortran-derived ones.  Checked here: the parser (k-points, group, notes, refusals) and a coarse CPU relaxation step of
the default routines against the whole mesh.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling, spinor_move_flag_subgroup_setting
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError, parse_parsec_input

_AU = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
10.0 10.0 10.0
end Cell_Shape
Grid_Spacing: {spacing}
Expansion_Order: 6
States_Num: 44
Fermi_Temp: 1000.0
Max_Iter: 150
Convergence_Criterion: 1.0e-7
Mixing_Method: Anderson
Mixing_Param: 0.3
Eigensolver: chebff
{extra}
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 2 2
end Monkhorst_Pack_Grid
{shift}
Atom_Types_Num: 1
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Move_Flag: -1
Begin atom_coord
  5.0 5.0 5.0 1
  7.9 7.9 7.9 0
  2.1 2.1 2.1 1
End atom_coord
Correlation_Type: CA
"""
# a chain along (111) about the centre of the cube (D3d, 12 operations); the atom on +x+y+z fixed keeps C3v
_RELAX = "SO_from_scratch: .true.\nMinimization: bfgs\nMovement_Num: {moves}"
_ZERO_SHIFT = "begin Monkhorst_Pack_Shift\n0 0 0\nend Monkhorst_Pack_Shift"
_SWITCH = "PARSEC_SPINOR_MOVE_FLAG_SUBGROUP"


def _text(*, fortran: bool = False, extra: str = "", moves: int = 2, spacing: float = 0.6, shift=None) -> str:
    body = _RELAX.format(moves=moves) + ("\nUse_Fortran_Routines: .true." if fortran else "") + extra
    return _INPUT.format(
        spacing=spacing, extra=body, shift=(_ZERO_SHIFT if fortran else "") if shift is None else shift
    )


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(_AU, self.directory / "Au_POTRE.DAT")

    def parse(self, text: str, environment=None):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(path)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(_SWITCH, None)
            self.assertEqual(spinor_move_flag_subgroup_setting(), "auto")
        for value, expected in (("auto", "auto"), (" OFF ", "off"), ("", "auto")):
            with mock.patch.dict(os.environ, {_SWITCH: value}):
                self.assertEqual(spinor_move_flag_subgroup_setting(), expected)
        with mock.patch.dict(os.environ, {_SWITCH: "on"}):
            with self.assertRaisesRegex(ValueError, "auto or off"):
                spinor_move_flag_subgroup_setting()


class ParserTests(_Directory):
    def test_default_routines_take_the_subgroup_wedge(self) -> None:
        translation = self.parse(_text())
        self.assertFalse(translation.routines.fortran)
        problem = translation.problem
        self.assertEqual(problem.space_group.order, 6)
        self.assertFalse(problem.kpoint_time_reversal)
        self.assertEqual(len(kpoint_sampling(problem)[0]), 4)
        self.assertTrue(any("the 6 operations that map the fixed atoms onto fixed atoms" in line
                            for line in translation.warnings))
        # off: the whole mesh as its time-reversal pairs, as before
        before = self.parse(_text(), {_SWITCH: "off"}).problem
        self.assertIsNone(before.space_group)
        self.assertTrue(before.kpoint_time_reversal)
        self.assertEqual(len(kpoint_sampling(before)[0]), 8)

    def test_fortran_routines_take_the_subgroup_and_off_refuses(self) -> None:
        translation = self.parse(_text(fortran=True))
        self.assertTrue(translation.routines.fortran)
        self.assertEqual(translation.problem.space_group.order, 6)
        self.assertEqual(len(kpoint_sampling(translation.problem)[0]), 4)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.parse(_text(fortran=True), {_SWITCH: "off"})

    def test_magnetic_spinors_keep_their_routes(self) -> None:
        # non-collinear magnetism keeps the whole mesh on the default routines, whatever the switch
        for environment in ({}, {_SWITCH: "off"}):
            with self.subTest(environment=environment):
                translation = self.parse(_text(extra="\nNon_Collinear_Magnetism: .true."), environment)
                self.assertIsNone(translation.problem.space_group)
                self.assertEqual(len(kpoint_sampling(translation.problem)[0]), 8)

    def test_moving_atoms_that_keep_the_group_are_unchanged(self) -> None:
        # every atom movable: the wedge of the whole group under both values
        text = _text().replace("  7.9 7.9 7.9 0", "  7.9 7.9 7.9 1")
        for environment in ({}, {_SWITCH: "off"}):
            with self.subTest(environment=environment):
                problem = self.parse(text, environment).problem
                self.assertEqual(problem.space_group.order, 12)
                self.assertEqual(len(kpoint_sampling(problem)[0]), 4)


def _archive(directory: Path):
    from parsec_python.tests.archives import read_archive

    return read_archive(directory / "parsec_python_results.npz")


class RunTests(_Directory):
    """A coarse cell on the CPU: the first structure of the subgroup's wedge against the whole mesh."""

    def run_cli(self, text: str, environment=None) -> dict:
        from parsec_python.cli import main as cli_main

        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            code = cli_main([str(path), "--quiet"])
        log = (self.directory / "parsec.out").read_text(encoding="utf-8")
        # (4: the relaxation stopped at Movement_Num before Force_Min, as these short ones do)
        self.assertIn(code, (0, 4), log[-2000:])
        archive = _archive(self.directory)
        return {"forces": np.asarray(archive["forces_total"]), "k": int(archive["n_kpoints"]), "log": log}

    def test_the_subgroup_forces_are_those_of_the_whole_mesh(self) -> None:
        text = _text(moves=1, spacing=0.6).replace("Convergence_Criterion: 1.0e-7", "Convergence_Criterion: 1.0e-8")
        wedge = self.run_cli(text)
        whole = self.run_cli(text, {_SWITCH: "off", "PARSEC_KPOINT_TIME_REVERSAL": "off"})
        self.assertEqual((wedge["k"], whole["k"]), (4, 8))
        self.assertGreater(float(np.max(np.abs(whole["forces"]))), 1.0e-3)
        np.testing.assert_allclose(wedge["forces"], whole["forces"], atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
