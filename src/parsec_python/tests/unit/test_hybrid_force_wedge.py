"""Forces and relaxations of a periodic hybrid on the irreducible wedge of its mesh (``PARSEC_HYBRID_KPOINT_FORCE_WEDGE``).

H2 along z in a cube with the 3 x 3 x 1 mesh (default routines): ``off`` keeps the 5 time-reversal pairs; ``auto`` (the
default) runs the wedge of ``PARSEC_HYBRID_KPOINT_WEDGE`` (``on``: 3 points throughout; ``auto``: the pairs for the semilocal start,
the wedge for the Fock loop of the insulating start, kept by the later structures of a relaxation), the forces
completed and averaged over the wedge's group: the same energies and forces.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.tests.unit.test_periodic_ionic import _H, _WEDGE_INPUT, _WEDGE_MESH, CommandLineTests

_HYBRID = _WEDGE_INPUT.replace("Correlation_Type: ca\n", "Correlation_Type: pbe0\nExx_Tolerance: 1e-9 Ry\n")


def _settings(force: str, wedge: str) -> dict:
    return {"PARSEC_HYBRID_KPOINT_FORCE_WEDGE": force, "PARSEC_HYBRID_KPOINT_WEDGE": wedge}


#: the 3 x 3 x 1 mesh without a shift block (PARSEC_MP_UNSHIFTED_MESH=fortran: the mesh of a zero shift, with the note
#: of a hybrid's mesh), on the default routines and on the Fortran-derived ones
_UNSHIFTED_MESH = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n3 3 1\nend Monkhorst_Pack_Grid\n"


class HybridForceWedgeTests(unittest.TestCase):
    def _folder(self, extra: str, *, mesh: str = _WEDGE_MESH, flag: str = "") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        text = _HYBRID.format(extra=mesh + extra, mixing="0.1")
        if flag:
            text = text.replace("Local_Component: s\n", f"Local_Component: s\n{flag}\n")
        (folder / "parsec.in").write_text(text)
        (folder / "H_POTRE.DAT").write_bytes(_H.read_bytes())
        return folder

    def _parse_with(self, extra: str, settings: dict, *, mesh: str, flag: str = ""):
        from parsec_python.Input import parse_parsec_input

        with mock.patch.dict(os.environ, settings):
            return parse_parsec_input(self._folder(extra, mesh=mesh, flag=flag) / "parsec.in")

    @staticmethod
    def _mesh_note(translation) -> str:
        notes = [note for note in translation.warnings if note.startswith("Monkhorst_Pack_Grid without")]
        return notes[0] if len(notes) == 1 else ""

    def test_the_switches_and_notes_of_each_set_of_routines(self) -> None:
        """PARSEC_KPOINT_FORCE_WEDGE=off keeps a hybrid's forces on the pairs on the default routines only (the
        Fortran-derived routines do not read it, for semilocal forces either); PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off on
        both; the note of a mesh without a shift block gives the reason of each."""

        for fortran in (False, True):
            mesh = f"Use_Fortran_Routines: {'.true.' if fortran else '.false.'}\n" + _UNSHIFTED_MESH
            for extra in ("Skip_Force: .false.\n", "Minimization: bfgs\n"):
                label = (fortran, extra)
                # (auto, the default of PARSEC_HYBRID_KPOINT_WEDGE before 2026-11-23, keeps the wedge aside; on, the
                # default since, takes it at once)
                semilocal_off = self._parse_with(
                    extra, {"PARSEC_KPOINT_FORCE_WEDGE": "off", "PARSEC_HYBRID_KPOINT_WEDGE": "auto"}, mesh=mesh
                )
                self.assertEqual(semilocal_off.problem.hybrid_kpoint_wedge_deferred is not None, fortran, label)
                semilocal_default = self._parse_with(extra, {"PARSEC_KPOINT_FORCE_WEDGE": "off"}, mesh=mesh)
                self.assertEqual(semilocal_default.problem.hybrid_kpoint_wedge, fortran, label)
                self.assertIsNone(semilocal_default.problem.hybrid_kpoint_wedge_deferred, label)
                hybrid_off = self._parse_with(extra, {"PARSEC_HYBRID_KPOINT_FORCE_WEDGE": "off"}, mesh=mesh)
                self.assertIsNone(hybrid_off.problem.hybrid_kpoint_wedge_deferred, label)
                self.assertFalse(hybrid_off.problem.hybrid_kpoint_wedge, label)
                note = self._mesh_note(hybrid_off)
                self.assertIn("forces and moving atoms keep the time-reversal pairs", note, label)
                self.assertIn("PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off", note, label)
                self.assertEqual("PARSEC_KPOINT_FORCE_WEDGE=off" in note, not fortran, label)
                if not fortran:
                    note = self._mesh_note(semilocal_off)
                    self.assertIn("PARSEC_KPOINT_FORCE_WEDGE=off", note, label)
                else:
                    self.assertTrue(any(text.startswith("Forces of the hybrid") for text in semilocal_off.warnings))

    def test_a_fixed_atom_that_the_group_moves_keeps_the_pairs_with_its_reason(self) -> None:
        """The mirror z -> -z exchanges the atoms, the second of which Move_Flag 1 fixes: the relaxation keeps the
        pairs, and the note names the fixed atoms (not 'no operation ... maps the grid')."""

        mesh = "Use_Fortran_Routines: .false.\n" + _UNSHIFTED_MESH
        free = self._parse_with("Minimization: bfgs\n", {}, mesh=mesh)
        self.assertTrue(free.problem.hybrid_kpoint_wedge)
        deferred = self._parse_with("Minimization: bfgs\n", {"PARSEC_HYBRID_KPOINT_WEDGE": "auto"}, mesh=mesh)
        self.assertIsNotNone(deferred.problem.hybrid_kpoint_wedge_deferred)
        for wedge in ("on", "auto"):
            fixed = self._parse_with(
                "Minimization: bfgs\n", {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}, mesh=mesh, flag="Move_Flag: 1"
            )
            self.assertIsNone(fixed.problem.hybrid_kpoint_wedge_deferred)
            self.assertFalse(fixed.problem.hybrid_kpoint_wedge)
        note = self._mesh_note(fixed)
        self.assertIn("keeps the fixed atoms (Move_Flag) apart from the movable ones", note)
        self.assertFalse(any(text.startswith("Forces of the hybrid") for text in fixed.warnings))
        # a single point without forces keeps the reason of a single point
        alone = self._parse_with("Ignore_Symmetry: .true.\n", {}, mesh=mesh)
        self.assertIn("or the symmetry is off (Ignore_Symmetry", self._mesh_note(alone))
        self.assertNotIn("Move_Flag", self._mesh_note(alone))

    def _parse(self, extra: str, force: str, wedge: str = "auto"):
        from parsec_python.Input import parse_parsec_input

        with mock.patch.dict(os.environ, _settings(force, wedge)):
            return parse_parsec_input(self._folder(extra) / "parsec.in")

    def _run(self, extra: str, force: str, wedge: str = "auto") -> Path:
        from parsec_python.cli import main as reference_main

        folder = self._folder(extra)
        output, error = StringIO(), StringIO()
        previous = os.getcwd()
        os.chdir(folder)
        try:
            with mock.patch.dict(os.environ, _settings(force, wedge)):
                with redirect_stdout(output), redirect_stderr(error):
                    code = reference_main([str(folder / "parsec.in"), "--no-archive"])
        finally:
            os.chdir(previous)
        self.assertIn(code, (0, 4), error.getvalue()[-3000:])
        return folder

    @staticmethod
    def _energy(folder: Path) -> float:
        return float(re.findall(r"Total Energy =\s+(-?\d+\.\d+)", (folder / "parsec.out").read_text())[-1])

    def test_the_mesh_of_each_setting(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        for extra in ("Skip_Force: .false.\n", "Minimization: bfgs\n"):
            off = self._parse(extra, "off").problem
            self.assertFalse(off.hybrid_kpoint_wedge)
            self.assertIsNone(off.hybrid_kpoint_wedge_deferred)
            self.assertEqual(len(kpoint_sampling(off)[0]), 5)
            on = self._parse(extra, "auto", "on").problem
            self.assertTrue(on.hybrid_kpoint_wedge)
            self.assertGreater(on.space_group.order, 1)
            self.assertTrue(on.scf.symmetrize_density)
            self.assertEqual(len(kpoint_sampling(on)[0]), 3)
            auto = self._parse(extra, "auto", "auto").problem
            self.assertFalse(auto.hybrid_kpoint_wedge)
            self.assertIsNotNone(auto.hybrid_kpoint_wedge_deferred)
            self.assertEqual(auto.hybrid_kpoint_wedge_deferred.group.order, on.space_group.order)
            np.testing.assert_array_equal(kpoint_sampling(auto)[0], kpoint_sampling(off)[0])
            self.assertEqual(sum(note.startswith("Forces of the hybrid") for note in self._parse(extra, "auto").warnings), 1)
            self.assertFalse(any(note.startswith("Forces of the hybrid") for note in self._parse(extra, "off").warnings))
            # the wedge's switch off keeps the pairs whatever the forces' switch says
            alone = self._parse(extra, "auto", "off").problem
            self.assertFalse(alone.hybrid_kpoint_wedge)
            self.assertIsNone(alone.hybrid_kpoint_wedge_deferred)
        dynamics = self._parse(
            "Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 300\nT_Step: 10\nTime_Step: 20\nStep_Num: 2\n", "auto", "on"
        ).problem
        self.assertFalse(dynamics.hybrid_kpoint_wedge)
        self.assertIsNone(dynamics.hybrid_kpoint_wedge_deferred)
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_FORCE_WEDGE": "on"}):
            from parsec_python.Grid.kpoints import hybrid_kpoint_force_wedge_setting

            with self.assertRaises(ValueError):
                hybrid_kpoint_force_wedge_setting()

    def test_the_forces_of_the_wedge_are_those_of_the_pairs(self) -> None:
        found = {}
        for label, force, wedge in (("off", "off", "auto"), ("on", "auto", "on"), ("auto", "auto", "auto")):
            folder = self._run("Skip_Force: .false.\n", force, wedge)
            found[label] = (self._energy(folder), CommandLineTests._final_forces(folder))
        forces = found["off"][1]
        self.assertGreater(float(np.abs(forces[:, 2]).max()), 1.0e-2)
        for label in ("on", "auto"):
            self.assertAlmostEqual(found[label][0], found["off"][0], delta=2.0e-7, msg=label)
            np.testing.assert_allclose(found[label][1], forces, atol=1.0e-5, err_msg=label)
            np.testing.assert_allclose(found[label][1][:, :2], 0.0, atol=1.0e-6, err_msg=label)

    def test_a_relaxation_keeps_the_wedge(self) -> None:
        extra = "Minimization: bfgs\nMovement_Num: 3\nForce_Min: 1e-4 Ry/bohr\n"
        found = {}
        for label, force, wedge in (("off", "off", "auto"), ("on", "auto", "on"), ("auto", "auto", "auto")):
            folder = self._run(extra, force, wedge)
            text = (folder / "parsec.out").read_text()
            found[label] = (self._energy(folder), CommandLineTests._final_forces(folder), text)
        for label in ("on", "auto"):
            self.assertAlmostEqual(found[label][0], found["off"][0], delta=1.0e-6, msg=label)
            np.testing.assert_allclose(found[label][1], found["off"][1], atol=1.0e-5, err_msg=label)
        # auto: the first structure's start decides (one note), the later structures keep its wedge
        self.assertEqual(found["auto"][2].count("The Fock loop runs on the irreducible wedge"), 1)
        # an insulator's Fock loops on the wedge end with every state 0 or 1: no note of a metal on the wedge
        for label in ("on", "auto", "off"):
            self.assertNotIn("a metal, held in the state of the wedge", found[label][2], label)

    def test_a_metallic_end_on_the_wedge_is_noted(self) -> None:
        """A Fock loop that ``auto`` moved to the wedge and that ends with a partly filled state gets a NOTE (the
        first structure: after its insulating start; the later ones: the wedge kept from the structure before); the
        run is not changed.  The occupations of the H2 cell are 0/1, so the test makes every count after the first
        (the start's) report partly filled states."""

        from parsec_python.Grid import kpoints

        real = kpoints.partly_filled_states
        calls = []

        def counted(occupations, *args, **kwargs):
            calls.append(1)
            found = real(occupations, *args, **kwargs)
            return found if len(calls) == 1 else (2, 0.25)

        extra = "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 1e-4 Ry/bohr\n"
        with mock.patch.object(kpoints, "partly_filled_states", side_effect=counted):
            folder = self._run(extra, "auto", "auto")
        text = (folder / "parsec.out").read_text()
        self.assertEqual(text.count("The Fock loop runs on the irreducible wedge"), 1)
        notes = text.count("a metal, held in the state of the wedge")
        carried = text.count("This structure kept the wedge its relaxation took at the first structure")
        self.assertGreaterEqual(carried, 1)
        # one note per Fock loop on the wedge: the first structure's (after its start) and every later one's
        self.assertEqual(notes, carried + 1)
        self.assertEqual(len(calls), notes + 1)


class MetallicWedgeNoteTests(unittest.TestCase):
    """``BlochExchange.after_loop`` and the lines of its record, without a run: the state a metal's Fock loop ends in,
    on the wedge or on the time-reversal pairs, and the outer iteration at which it became unstable."""

    def test_the_record_and_its_lines(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Output.parsec_output import hybrid_mesh_route_lines
        from parsec_python.SCF.hybrid_periodic import BlochExchange
        from parsec_python.models import ExchangeLoopIteration

        engine = object.__new__(BlochExchange)
        engine.k_points = np.zeros((4, 3))
        records = []
        metal = SimpleNamespace(occupations=[np.array([1.0, 1.0, 0.5, 0.0]), np.array([1.0, 1.0, 0.5, 0.0])])
        insulator = SimpleNamespace(occupations=[np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])])
        # dexx 1e-3, 1e-6, 1e-4: rose tenfold at outer iteration 3
        loop = tuple(
            mock.Mock(spec=ExchangeLoopIteration, iteration=i, estimated_error=e, total_energy=None)
            for i, e in ((0, None), (1, 1.0e-3), (2, 1.0e-6), (3, 1.0e-4))
        )
        # the pairs of a mesh of 8 points: the NOTE of a metal on the pairs, with the outer iteration of the instability
        engine.mesh = (2, 2, 2)
        engine.after_loop(None, insulator, callback=records.append, records=loop)
        self.assertEqual(records, [])
        engine.after_loop(None, metal, callback=records.append, records=loop)
        record = records.pop()
        self.assertEqual((record.route, record.points, record.pairs_points, record.mesh_points), ("pairs_partly_filled", 4, 4, 8))
        self.assertEqual((record.partly_filled, record.instability), (2, 3))
        text = "\n".join(hybrid_mesh_route_lines(record))
        self.assertIn("the Fock loop on the 4 time-reversal pairs of the 8-point mesh", text)
        self.assertIn("ends with 2 partly filled state(s) at the Fermi level", text)
        self.assertIn("became unstable at outer iteration 3", text)
        self.assertIn("are not comparable with those of structures that keep a group", text)
        engine.after_loop(None, metal, callback=records.append)
        self.assertIsNone(records.pop().instability)
        # a metal's start (partly filled) whose Fock loop ends with every state 0 or 1: the broken state of the pairs
        engine._start_partly_filled = 3
        engine.after_loop(None, insulator, callback=records.append, records=loop)
        record = records.pop()
        self.assertEqual((record.route, record.partly_filled, record.start_partly_filled), ("pairs_partly_filled", 0, 3))
        text = "\n".join(hybrid_mesh_route_lines(record))
        self.assertIn("ends with every state 0 or 1, its semilocal start with 3 partly filled state(s)", text)
        self.assertIn("became unstable at outer iteration 3", text)
        engine.after_loop(None, insulator, callback=records.append)
        text = "\n".join(hybrid_mesh_route_lines(records.pop()))
        self.assertIn("filled the levels the start held partly to 0 or 1", text)
        engine.after_loop(None, metal, callback=records.append)
        text = "\n".join(hybrid_mesh_route_lines(records.pop()))
        self.assertIn("did not become unstable", text)
        engine._start_partly_filled = 0
        # one point: no equivalent k-points, no NOTE
        engine.mesh = (1, 1, 1)
        engine.after_loop(None, metal, callback=records.append, records=loop)
        self.assertEqual(records, [])
        # the wedge from the start (PARSEC_HYBRID_KPOINT_WEDGE=on, the default)
        engine.mesh = (2, 2, 2)
        engine.stars = SimpleNamespace(gathers=[None] * 16)
        engine.after_loop(None, insulator, callback=records.append, records=loop)
        self.assertEqual(records, [])
        engine.after_loop(None, metal, callback=records.append)
        record = records.pop()
        self.assertEqual((record.route, record.reason, record.operations, record.pairs_points), ("wedge_partly_filled", "wedge", 16, 0))
        text = "\n".join(hybrid_mesh_route_lines(record))
        self.assertIn("a metal, held in the state of the wedge", text)
        self.assertIn("On the time-reversal pairs of the mesh its Fock loop may fill", text)
        self.assertNotIn("became unstable", text)
        # the wedge that auto took after an insulating start, or that a relaxation carried
        for how in ("start", "carried"):
            engine._wedge_taken = (how, 8, 8, 16)
            engine.after_loop(None, insulator, callback=records.append)
            self.assertEqual(records, [])
            engine.after_loop(None, metal, callback=records.append, records=loop)
            record = records.pop()
            self.assertEqual((record.route, record.points, record.pairs_points, record.mesh_points), ("wedge_partly_filled", 4, 8, 8))
            self.assertEqual((record.operations, record.partly_filled, record.reason, record.instability), (16, 2, how, 3))
            self.assertAlmostEqual(record.largest_fractional_occupation, 0.5)
            text = "\n".join(hybrid_mesh_route_lines(record))
            self.assertIn("ends with 2 partly filled state(s)", text)
            self.assertIn("On the 8 time-reversal pairs of the 8-point mesh", text)
            self.assertIn("became unstable at outer iteration 3", text)
            self.assertEqual("This structure kept the wedge" in text, how == "carried")
            self.assertIn("PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off", text)


class WedgeSectorStatesTests(unittest.TestCase):
    """``PARSEC_HYBRID_WEDGE_SECTOR_STATES``: the Ritz states of the k-point sectors of the start on the pairs move with
    their k-points to the wedge (``SCF/hybrid_periodic.py``, ``_restricted_sector_states``)."""

    def test_the_layout_of_the_wedge(self) -> None:
        from parsec_python.SCF.hybrid_periodic import _restricted_sector_states

        carried = {
            "layout": [(0, -1, 10), (1, 0, 5), (1, 1, 5), (2, 0, 6), (2, 1, 4), (3, -1, 10)],
            "counts": [8, 5, 5, 6, 4, 8],
            "states": {0: "a", 1: "b", 3: "d", 4: "e", 5: "f"},
        }
        found = _restricted_sector_states(carried, (0, 2, 3), 4, 1)["kpoint_sectors"]
        self.assertEqual(found["layout"], [(0, -1, 10), (1, 0, 6), (1, 1, 4), (2, -1, 10)])
        self.assertEqual(found["counts"], [8, 6, 4, 8])
        self.assertEqual(found["states"], {0: "a", 1: "d", 2: "e", 3: "f"})
        # two channels: the tasks channel * n_k + k of both
        spin = {"layout": [(0, -1, 3), (1, -1, 3), (2, -1, 3), (3, -1, 3)], "counts": [2, 2, 2, 2], "states": {}}
        found = _restricted_sector_states(spin, (1,), 2, 2)["kpoint_sectors"]
        self.assertEqual(found["layout"], [(0, -1, 3), (1, -1, 3)])
        self.assertIsNone(_restricted_sector_states({"layout": [(1, -1, 3)], "counts": [2]}, (0,), 2, 1))

    def test_the_switch(self) -> None:
        from parsec_python.SCF.hybrid_periodic import wedge_sector_states_setting

        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_WEDGE_SECTOR_STATES": ""}):
            self.assertEqual(wedge_sector_states_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_WEDGE_SECTOR_STATES": "on"}):
            with self.assertRaises(ValueError):
                wedge_sector_states_setting()


if __name__ == "__main__":
    unittest.main()
