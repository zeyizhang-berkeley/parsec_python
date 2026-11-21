"""Standard orientation of a cluster (``PARSEC_STANDARD_ORIENTATION``, :mod:`parsec_python.Input.standard_orientation`).

The search finds the frame and the centre of the largest group of signed permutations, symmetrizes the atoms
exactly in it and maps the reports back; the input reader applies it only with the switch on and only where the
group grows.  CPU only, no SCF.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Forces import assemble_forces, atom_symmetry_operations, operations_preserving, total_torque
from parsec_python.Forces.local import LocalForceTerms
from parsec_python.Energy.dipole import DipoleMoment
from parsec_python.Input.parsec_input import ParsecInputError, parse_parsec_input
from parsec_python.Input.standard_orientation import (
    SWITCH,
    TOLERANCE_SWITCH,
    dipole_to_reference,
    find_standard_orientation,
    forces_to_reference,
    standard_orientation_setting,
    standard_orientation_tolerance,
)
from parsec_python.models import Atom


DATA = Path(__file__).resolve().parents[1] / "data"
_A = 1.18
METHANE = np.array([[0.0, 0.0, 0.0], [_A, _A, _A], [-_A, -_A, _A], [-_A, _A, -_A], [_A, -_A, -_A]])
METHANE_LABELS = ["C", "H", "H", "H", "H"]


def _rotation(z: float, y: float, x: float) -> np.ndarray:
    """``Rz(z) Ry(y) Rx(x)``, angles in degrees."""

    def axis(angle, first, second):
        cosine, sine = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        matrix = np.eye(3)
        matrix[first, first] = matrix[second, second] = cosine
        matrix[first, second], matrix[second, first] = -sine, sine
        return matrix

    return axis(z, 0, 1) @ axis(y, 2, 0) @ axis(x, 1, 2)


RANDOM = _rotation(37.0, -21.0, 66.0)


@contextmanager
def _environment(**values):
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _group_order(positions, labels, tolerance=1.0e-12) -> int:
    atoms = [Atom(label, position) for label, position in zip(labels, positions)]
    return len(atom_symmetry_operations(atoms, tolerance=tolerance))


class StandardOrientationSearchTest(unittest.TestCase):
    def test_methane_rotated_about_z_is_put_back(self):
        positions = METHANE @ _rotation(10.0, 0.0, 0.0).T
        search = find_standard_orientation(positions, METHANE_LABELS)
        self.assertEqual(search.before, (2, 4))
        self.assertEqual(search.after, (4, 24))
        orientation = search.orientation
        self.assertIsNotNone(orientation)
        # the frame nearest the input: the aligned methane itself
        np.testing.assert_allclose(search.positions, METHANE, atol=1.0e-13)
        self.assertAlmostEqual(abs(np.linalg.det(orientation.rotation)), 1.0, places=12)
        self.assertGreater(np.linalg.det(orientation.rotation), 0.0)
        np.testing.assert_allclose(orientation.to_reference(search.positions), positions, atol=1.0e-13)
        self.assertEqual(_group_order(search.positions, METHANE_LABELS), 24)

    def test_arbitrary_rotation_and_offset(self):
        positions = METHANE @ RANDOM.T + np.array([0.3, -0.2, 0.7])
        search = find_standard_orientation(positions, METHANE_LABELS)
        self.assertEqual(search.after, (4, 24))
        orientation = search.orientation
        # exactly symmetric: every operation maps the atoms onto each other bit for bit
        self.assertEqual(_group_order(search.positions, METHANE_LABELS, tolerance=0.0 + 1.0e-300), 24)
        np.testing.assert_allclose(orientation.to_reference(search.positions), positions, atol=1.0e-12)
        np.testing.assert_allclose(
            orientation.to_calculation(positions), search.positions, atol=1.0e-12
        )
        # the bounding box of the calculation frame is centred at the origin
        middle = 0.5 * (search.positions.min(axis=0) + search.positions.max(axis=0))
        np.testing.assert_allclose(middle, 0.0, atol=1.0e-14)

    def test_small_offset_is_a_translation(self):
        positions = METHANE + np.array([0.05, 0.0, 0.0])
        search = find_standard_orientation(positions, METHANE_LABELS)
        # C2v about the x axis: its four operations commute; Td gives the weighted sectors of D2d
        self.assertEqual(search.before, (4, 4))
        self.assertEqual(search.after, (4, 24))
        np.testing.assert_allclose(search.orientation.rotation, np.eye(3), atol=1.0e-14)
        np.testing.assert_allclose(search.orientation.centre, [0.05, 0.0, 0.0], atol=1.0e-14)

    def test_single_atom_goes_to_the_origin(self):
        search = find_standard_orientation(np.array([[0.05, -0.02, 0.0]]), ["Na"])
        self.assertEqual(search.before, (2, 2))
        self.assertEqual(search.after, (8, 48))
        np.testing.assert_array_equal(search.positions, np.zeros((1, 3)))
        np.testing.assert_allclose(search.orientation.centre, [0.05, -0.02, 0.0], atol=1.0e-15)

    def test_aligned_input_is_kept(self):
        # ammonia with its threefold axis along (1, 1, 1): C3v of six signed permutations, two of which commute;
        # a frame with a mirror normal along an axis would keep two operations in all, and is not taken
        ammonia = np.array([[0.3, 0.3, 0.3], [1.6, -0.6, -0.6], [-0.6, 1.6, -0.6], [-0.6, -0.6, 1.6]])
        for positions, labels in (
            (METHANE, METHANE_LABELS),
            (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]]), ["C", "O", "O"]),
            (ammonia, ["N", "H", "H", "H"]),
        ):
            search = find_standard_orientation(positions, labels)
            self.assertIsNone(search.orientation)
            self.assertIsNone(search.positions)

    def test_rounded_coordinates_are_symmetrized_within_the_tolerance(self):
        # five decimals in angstrom, as a database gives them
        positions = np.round(METHANE @ RANDOM.T * 0.529177, 5) / 0.529177
        search = find_standard_orientation(positions, METHANE_LABELS)
        self.assertEqual(search.after, (4, 24))
        self.assertLess(search.orientation.displacement, 1.0e-4)
        self.assertGreater(search.orientation.displacement, 1.0e-7)
        self.assertEqual(_group_order(search.positions, METHANE_LABELS, tolerance=1.0e-300), 24)
        strict = find_standard_orientation(positions, METHANE_LABELS, tolerance=1.0e-7)
        self.assertIsNone(strict.orientation)

    def test_other_groups(self):
        side = 2.95
        hexafluoride = np.array(
            [[0, 0, 0], [side, 0, 0], [-side, 0, 0], [0, side, 0], [0, -side, 0], [0, 0, side], [0, 0, -side]],
            dtype=float,
        )
        water = np.array([[0.0, 0.0, 0.22], [1.43, 0.0, -0.88], [-1.43, 0.0, -0.88]])
        angles = np.arange(6) * np.pi / 3.0
        benzene = np.vstack(
            [
                np.column_stack([2.64 * np.cos(angles), 2.64 * np.sin(angles), np.zeros(6)]),
                np.column_stack([4.68 * np.cos(angles), 4.68 * np.sin(angles), np.zeros(6)]),
            ]
        )
        carbon_dioxide = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.2]])
        cases = (
            (hexafluoride, ["S"] + ["F"] * 6, (8, 48)),
            (water, ["O", "H", "H"], (4, 4)),
            (benzene, ["C"] * 6 + ["H"] * 6, (8, 8)),
            (carbon_dioxide, ["C", "O", "O"], (8, 16)),
        )
        for positions, labels, expected in cases:
            with self.subTest(labels=labels[:2]):
                rotated = positions @ RANDOM.T
                search = find_standard_orientation(rotated, labels)
                self.assertEqual(search.after, expected)
                self.assertEqual(_group_order(search.positions, labels), expected[1])
                np.testing.assert_allclose(
                    search.orientation.to_reference(search.positions), rotated, atol=1.0e-12
                )

    def test_labels_restrict_the_group(self):
        # a fixed hydrogen (Move_Flag): the operations may not exchange it with a movable one
        labels = [("C", True), ("H", False), ("H", True), ("H", True), ("H", True)]
        search = find_standard_orientation(METHANE @ RANDOM.T, labels)
        self.assertIsNotNone(search.orientation)
        self.assertEqual(search.after, (2, 2))
        atoms = [Atom("X" if not label[1] else label[0], p) for label, p in zip(labels, search.positions)]
        for operation in atom_symmetry_operations(atoms, tolerance=1.0e-12):
            self.assertEqual(operation.image[1], 1)

    def test_movable_flags_count_as_the_relaxation_decides(self):
        # Relaxation.run.symmetry_decision: a group of the species that exchanges a fixed atom with a movable one
        # is not used at all under PARSEC_MOVE_FLAG_SUBGROUP=off (move_flag_subgroup=False), so a frame where that
        # happens counts as the identity alone; under auto (a relaxation by an optimizer, since 2026-11-20 in this
        # score) it counts by the subgroup of the operations that map the fixed atoms onto fixed atoms: C3v (one
        # fixed H) and C2v (two) in the frame of Td, instead of a mirror
        for movable, former, subgroup in (
            ([True, False, True, True, True], (2, 2), (2, 6)),
            ([True, False, True, True, False], (2, 2), (4, 4)),
            ([False, True, True, True, True], (4, 24), (4, 24)),
            ([True] * 5, (4, 24), (4, 24)),
        ):
            for rule, expected in ((False, former), (True, subgroup)):
                with self.subTest(movable=movable, subgroup=rule):
                    search = find_standard_orientation(
                        METHANE @ RANDOM.T, METHANE_LABELS, movable=movable, move_flag_subgroup=rule
                    )
                    self.assertEqual(search.before, (1, 1))
                    self.assertEqual(search.after, expected)
                    atoms = [Atom(label, position) for label, position in zip(METHANE_LABELS, search.positions)]
                    operations = atom_symmetry_operations(atoms)
                    self.assertEqual(len(operations_preserving(operations, movable)), expected[1])
                    if not rule:
                        self.assertEqual(len(operations), expected[1])
        # None reads PARSEC_MOVE_FLAG_SUBGROUP (auto by default)
        for setting, expected in (("auto", (4, 4)), ("off", (2, 2))):
            with self.subTest(setting=setting), _environment(PARSEC_MOVE_FLAG_SUBGROUP=setting):
                search = find_standard_orientation(
                    METHANE @ RANDOM.T, METHANE_LABELS, movable=[True, False, True, True, False]
                )
                self.assertEqual(search.after, expected)

    def test_two_fixed_hydrogens_keep_the_frame_of_the_input(self):
        # Cs in the input frame (a mirror normal along z, the other axes turned by 23 degrees about it); the frame of
        # the C2v that respects the fixed atoms gives D2d for the species, half of which exchange fixed and movable
        # atoms: under PARSEC_MOVE_FLAG_SUBGROUP=off the relaxation would use no symmetry there, so the input frame is
        # kept; under auto it keeps that C2v (4 commuting operations), which wins over the mirror
        normal = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        angle = np.radians(23.0)
        axis = np.cos(angle) * np.array([1.0, 0.0, 0.0]) + np.sin(angle) * np.cross(normal, [1.0, 0.0, 0.0])
        frame = np.vstack([axis, np.cross(normal, axis), normal])
        movable = [True, False, True, True, False]
        search = find_standard_orientation(METHANE @ frame.T, METHANE_LABELS, movable=movable, move_flag_subgroup=False)
        self.assertEqual(search.before, (2, 2))
        self.assertIsNone(search.orientation)
        kept = find_standard_orientation(METHANE @ frame.T, METHANE_LABELS, movable=movable, move_flag_subgroup=True)
        self.assertEqual((kept.before, kept.after), ((2, 2), (4, 4)))
        atoms = [Atom(label, position) for label, position in zip(METHANE_LABELS, kept.positions)]
        operations = atom_symmetry_operations(atoms)
        self.assertEqual((len(operations), len(operations_preserving(operations, movable))), (8, 4))
        # labels alone would take that frame
        labelled = find_standard_orientation(METHANE @ frame.T, list(zip(METHANE_LABELS, movable)))
        self.assertEqual(labelled.after, (4, 4))
        atoms = [Atom(label, position) for label, position in zip(METHANE_LABELS, labelled.positions)]
        self.assertEqual(len(atom_symmetry_operations(atoms)), 8)

    def test_grid_shift_keeps_the_operations_of_the_grid(self):
        # a shift of 1/2 along x alone keeps no swap of x with another axis
        search = find_standard_orientation(METHANE @ RANDOM.T, METHANE_LABELS, grid_shift=(0.5, 0.0, 0.0))
        self.assertEqual(search.after, (4, 8))

    def test_switch_values(self):
        with _environment(**{SWITCH: None, TOLERANCE_SWITCH: None}):
            self.assertEqual(standard_orientation_setting(), "off")
            self.assertEqual(standard_orientation_tolerance(), 1.0e-3)
        with _environment(**{SWITCH: " ON ", TOLERANCE_SWITCH: "1e-5"}):
            self.assertEqual(standard_orientation_setting(), "on")
            self.assertEqual(standard_orientation_tolerance(), 1.0e-5)
        for name, value in ((SWITCH, "yes"), (TOLERANCE_SWITCH, "0"), (TOLERANCE_SWITCH, "0.5"), (TOLERANCE_SWITCH, "x")):
            with self.subTest(name=name, value=value), _environment(**{name: value}):
                with self.assertRaises(ValueError):
                    standard_orientation_setting() if name == SWITCH else standard_orientation_tolerance()


class StandardOrientationReportsTest(unittest.TestCase):
    def setUp(self):
        self.reference = METHANE @ RANDOM.T + np.array([0.3, -0.2, 0.7])
        search = find_standard_orientation(self.reference, METHANE_LABELS)
        self.orientation = search.orientation
        self.positions = search.positions

    def test_forces_rotate_back_and_the_torque_is_taken_again(self):
        generator = np.random.default_rng(7)
        terms = [generator.normal(size=(5, 3)) for _ in range(5)]
        forces = assemble_forces(
            self.positions,
            terms[0],
            LocalForceTerms(terms[1], terms[2], terms[3]),
            terms[4],
        )
        reported = forces_to_reference(forces, self.orientation)
        rotation = self.orientation.rotation
        np.testing.assert_allclose(reported.positions, self.orientation.to_reference(self.positions))
        for name in ("ion_ion", "nonlocal_pseudopotential", "unsymmetrized", "symmetrized", "total"):
            np.testing.assert_allclose(getattr(reported, name), getattr(forces, name) @ rotation.T)
        np.testing.assert_allclose(reported.local.ionic, forces.local.ionic @ rotation.T)
        np.testing.assert_allclose(reported.net_force, rotation @ forces.net_force)
        expected = rotation @ forces.torque + np.cross(self.orientation.centre, rotation @ forces.symmetrized.sum(axis=0))
        np.testing.assert_allclose(reported.torque, expected, atol=1.0e-12)
        np.testing.assert_allclose(reported.torque, total_torque(reported.positions, reported.symmetrized))
        self.assertIs(forces_to_reference(forces, None), forces)

    def test_dipole_about_the_origin_of_the_input(self):
        dipole = DipoleMoment(charge=-1.0, dipole=np.array([0.1, -0.4, 0.25]))
        reported = dipole_to_reference(dipole, self.orientation)
        np.testing.assert_allclose(
            reported.dipole, self.orientation.rotation @ dipole.dipole - self.orientation.centre
        )
        self.assertEqual(reported.charge, -1.0)


class StandardOrientationInputTest(unittest.TestCase):
    def setUp(self):
        self._scratch = tempfile.TemporaryDirectory()
        self.directory = Path(self._scratch.name)
        shutil.copy(DATA / "spin_ca_C_POTRE.DAT", self.directory / "C_POTRE.DAT")
        shutil.copy(DATA / "spin_ca_H_POTRE.DAT", self.directory / "H_POTRE.DAT")

    def tearDown(self):
        self._scratch.cleanup()

    def _input(self, positions, *extra: str, flags=None) -> Path:
        rows = [" ".join(f"{value:.12f}" for value in row) for row in positions]
        move = []
        if flags is not None:
            rows = [f"{row} {int(flag)}" for row, flag in zip(rows, flags)]
            move = ["Move_Flag: -1"]
        lines = [
            "Boundary_Sphere_Radius: 8",
            "Grid_Spacing: 0.35",
            "States_Num: 6",
            "Atom_Types_Num: 2",
            "Atom_Type: C",
            "Local_Component: p",
            *move,
            "begin Atom_Coord",
            rows[0],
            "end Atom_Coord",
            "Atom_Type: H",
            "Local_Component: s",
            *move,
            "begin Atom_Coord",
            *rows[1:],
            "end Atom_Coord",
            *extra,
        ]
        path = self.directory / "parsec.in"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_off_leaves_the_input_alone(self):
        positions = METHANE @ RANDOM.T + 0.3
        path = self._input(positions)
        with _environment(**{SWITCH: None}):
            default = parse_parsec_input(path)
        with _environment(**{SWITCH: "off"}):
            off = parse_parsec_input(path)
        for translation in (default, off):
            self.assertIsNone(translation.standard_orientation)
            np.testing.assert_array_equal(
                np.array([atom.position for atom in translation.problem.atoms]),
                np.array([[float(f"{value:.12f}") for value in row] for row in positions]),
            )
            self.assertFalse(any(SWITCH in warning for warning in translation.warnings))
        self.assertEqual(repr(default.problem), repr(off.problem))

    def test_on_orients_a_rotated_cluster(self):
        positions = METHANE @ RANDOM.T + 0.3
        with _environment(**{SWITCH: "on"}):
            translation = parse_parsec_input(self._input(positions, "Minimization: BFGS"))
        orientation = translation.standard_orientation
        self.assertIsNotNone(orientation)
        self.assertEqual((orientation.abelian_after, orientation.order_after), (4, 24))
        calculation = np.array([atom.position for atom in translation.problem.atoms])
        self.assertEqual(_group_order(calculation, METHANE_LABELS), 24)
        # the reference frame is the input after PARSEC's recentring of its bounding box
        reference = orientation.to_reference(calculation)
        shift = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
        np.testing.assert_allclose(reference, positions - shift, atol=1.0e-10)
        np.testing.assert_allclose(orientation.recentring, shift, atol=1.0e-10)
        self.assertTrue(any(f"{SWITCH}=on: the cluster is calculated" in w for w in translation.warnings))

    def test_on_keeps_an_aligned_cluster(self):
        with _environment(**{SWITCH: "on"}):
            translation = parse_parsec_input(self._input(METHANE))
        self.assertIsNone(translation.standard_orientation)
        np.testing.assert_array_equal(np.array([atom.position for atom in translation.problem.atoms]), METHANE)
        self.assertTrue(any("keeps the largest group" in w for w in translation.warnings))

    def test_reports_and_restart_structures(self):
        from parsec_python.Output.parsec_output import ParsecTextReporter, standard_orientation_lines
        from parsec_python.Relaxation.files import format_xyz_frame
        from parsec_python.Relaxation.run import restart_structure

        positions = METHANE @ RANDOM.T + 0.3
        with _environment(**{SWITCH: "on"}):
            translation = parse_parsec_input(self._input(positions, "Minimization: BFGS"))
        orientation = translation.standard_orientation
        lines = standard_orientation_lines(translation)
        self.assertTrue(any("Standard orientation" in line for line in lines))
        calculation = np.array([atom.position for atom in translation.problem.atoms])
        reporter = ParsecTextReporter(lambda text: None, translation)
        np.testing.assert_allclose(reporter._reported_positions(calculation), orientation.to_reference(calculation))
        # an XYZ structure stands for the coordinates of the input: recentred as they are, then turned
        xyz = self.directory / "start.xyz"
        moved = positions + 0.01 * np.array([[0, 0, 0], [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
        xyz.write_text(format_xyz_frame(METHANE_LABELS, moved, step=0), encoding="utf-8")
        restarted, restart, _ = restart_structure(translation, self.directory, source=xyz)
        self.assertIsNone(restart)
        self.assertFalse(restarted.problem.recenter_geometry)
        shift = 0.5 * (moved.min(axis=0) + moved.max(axis=0))
        np.testing.assert_allclose(
            np.array([atom.position for atom in restarted.problem.atoms]),
            orientation.to_calculation(moved - shift),
            atol=1.0e-12,
        )
        with _environment(**{SWITCH: "off"}):
            plain = parse_parsec_input(self._input(positions, "Minimization: BFGS"))
        self.assertEqual(standard_orientation_lines(plain), [])

    def test_refusals_and_errors(self):
        positions = METHANE @ RANDOM.T
        for extra, reason in (
            (("Ignore_Symmetry: true",), "Ignore_Symmetry"),
            (("Molecular_Dynamics: true",), "uses no point group"),
        ):
            with self.subTest(reason=reason), _environment(**{SWITCH: "on"}):
                translation = parse_parsec_input(self._input(positions, *extra))
                self.assertIsNone(translation.standard_orientation)
                self.assertTrue(any("is not applied" in w and reason in w for w in translation.warnings))
        for name, value in ((SWITCH, "maybe"), (TOLERANCE_SWITCH, "-1")):
            with self.subTest(name=name), _environment(**{SWITCH: "on", name: value}):
                with self.assertRaises(ParsecInputError):
                    parse_parsec_input(self._input(positions))

    def test_initial_density_files_are_refused_and_afm_scan_false_is_not(self):
        positions = METHANE @ RANDOM.T
        for extra in (
            ("Initial_Density: file", "ML_Density_File: density.npz"),
            ("Initial_Density: charge3net", "ML_Density_File: density.npz"),
        ):
            with self.subTest(extra=extra), _environment(**{SWITCH: "on"}):
                translation = parse_parsec_input(self._input(positions, *extra))
                self.assertIsNone(translation.standard_orientation)
                self.assertTrue(
                    any("is not applied" in w and "initial density" in w for w in translation.warnings)
                )
        with _environment(**{SWITCH: "on"}):
            translation = parse_parsec_input(self._input(positions, "AFM_Scan: .false."))
        self.assertIsNotNone(translation.standard_orientation)

    def test_move_flag_relaxation_keeps_its_group(self):
        from parsec_python.Relaxation.run import symmetry_decision

        normal = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        angle = np.radians(23.0)
        axis = np.cos(angle) * np.array([1.0, 0.0, 0.0]) + np.sin(angle) * np.cross(normal, [1.0, 0.0, 0.0])
        positions = METHANE @ np.vstack([axis, np.cross(normal, axis), normal]).T
        flags = [1, 0, 1, 1, 0]
        for setting in ("off", "on"):
            with self.subTest(setting=setting), _environment(**{SWITCH: setting, "PARSEC_MOVE_FLAG_SUBGROUP": "off"}):
                translation = parse_parsec_input(self._input(positions, "Minimization: BFGS", flags=flags))
                self.assertIsNone(translation.standard_orientation)
                self.assertEqual(symmetry_decision(translation), (True, None))
        # PARSEC_MOVE_FLAG_SUBGROUP=auto (the default): the frame of the C2v that the relaxation keeps (4 of the 8
        # operations of the species there); off of the orientation keeps the input frame and its mirror
        with _environment(**{SWITCH: "on", "PARSEC_MOVE_FLAG_SUBGROUP": "auto"}):
            translation = parse_parsec_input(self._input(positions, "Minimization: BFGS", flags=flags))
            self.assertEqual(translation.standard_orientation.order_after, 4)
            keeps, note = symmetry_decision(translation)
            self.assertTrue(keeps)
            self.assertIn("keeps the subgroup of the 4 operations", note)
        with _environment(**{SWITCH: "off", "PARSEC_MOVE_FLAG_SUBGROUP": "auto"}):
            translation = parse_parsec_input(self._input(positions, "Minimization: BFGS", flags=flags))
            self.assertIsNone(translation.standard_orientation)
            self.assertEqual(symmetry_decision(translation), (True, None))
        # a rotated methane whose fixed carbon every operation keeps: the whole group
        with _environment(**{SWITCH: "on"}):
            translation = parse_parsec_input(
                self._input(METHANE @ RANDOM.T, "Minimization: BFGS", flags=[0, 1, 1, 1, 1])
            )
        self.assertEqual(translation.standard_orientation.order_after, 24)
        self.assertEqual(symmetry_decision(translation), (True, None))

    def test_relax_restart_records_the_frame(self):
        from dataclasses import replace

        from parsec_python.Input.standard_orientation import StandardOrientation
        from parsec_python.Relaxation.files import RelaxRestart, format_relax_restart, parse_relax_restart
        from parsec_python.Relaxation.run import restart_structure

        # off the origin by more than 0.1 bohr: the first run recentres, a Restart_Run does not
        positions = METHANE @ RANDOM.T + np.array([0.7, -0.4, 0.3])
        relax = ("Minimization: BFGS", "Relax_Restart: true")
        with _environment(**{SWITCH: "on"}):
            first = parse_parsec_input(self._input(positions, *relax))
            again = parse_parsec_input(self._input(positions, *relax, "Restart_Run: true"))
        orientation = first.standard_orientation
        self.assertGreater(np.max(np.abs(again.standard_orientation.centre - orientation.centre)), 0.1)
        records = orientation.restart_records()
        recorded = StandardOrientation.from_restart_records(records)
        for name in ("rotation", "centre", "recentring"):
            np.testing.assert_array_equal(getattr(recorded, name), getattr(orientation, name))
        self.assertEqual(replace(recorded, rotation=None, centre=None, recentring=None),
                         replace(orientation, rotation=None, centre=None, recentring=None))
        calculation = np.array([atom.position for atom in first.problem.atoms])
        moved = calculation * 1.01
        text = format_relax_restart(RelaxRestart(moved, 1, 5, orientation=records))
        self.assertEqual(parse_relax_restart(text, 5).orientation, records)
        (self.directory / "relax_restart.dat").write_text(text, encoding="utf-8")
        with _environment(**{SWITCH: "on"}):
            for translation in (first, again):
                restarted, restart, notes = restart_structure(translation, self.directory)
                np.testing.assert_array_equal(
                    np.array([atom.position for atom in restarted.problem.atoms]), moved
                )
                # the frame the file records, also where the Restart_Run reads the input without recentring
                np.testing.assert_array_equal(restarted.standard_orientation.rotation, orientation.rotation)
                np.testing.assert_array_equal(restarted.standard_orientation.centre, orientation.centre)
                self.assertTrue(any("calculation frame it records" in note for note in notes))
        with _environment(**{SWITCH: "off"}):
            plain = parse_parsec_input(self._input(positions, *relax))
            with self.assertRaises(ValueError):
                restart_structure(plain, self.directory)
        # a file written without the switch holds the frame of the input, which the restart keeps
        (self.directory / "relax_restart.dat").write_text(
            format_relax_restart(RelaxRestart(moved, 1, 5)), encoding="utf-8"
        )
        with _environment(**{SWITCH: "on"}):
            restarted, _, notes = restart_structure(first, self.directory)
        self.assertIsNone(restarted.standard_orientation)
        self.assertTrue(any("written without the standard orientation" in note for note in notes))
        with _environment(**{SWITCH: "off"}):
            restarted, _, notes = restart_structure(plain, self.directory)
        self.assertIsNone(restarted.standard_orientation)
        self.assertFalse(any(SWITCH in note for note in notes))

    def test_relax_restart_block_follows_the_bfgs_state(self):
        import json
        from dataclasses import replace

        from parsec_python.Relaxation.files import ORIENTATION_HEADER, format_relax_restart, parse_relax_restart

        text = json.loads((DATA / "relax_restart_parsec15_h2_bfgs.json").read_text(encoding="utf-8"))[
            "relax_restart_dat"
        ]
        records = (" 1.0 0.0 0.0", " 0.0 1.0 0.0", " 0.0 0.0 1.0", " 0.1 0.0 0.0", " 0.0 0.0 0.0",
                   " 1 1 8 16", " 1e-14 0.001")
        plain = parse_relax_restart(text, 2)
        self.assertIsNone(plain.orientation)
        marked = parse_relax_restart(text.rstrip("\n") + "\n" + ORIENTATION_HEADER + "\n" + "\n".join(records), 2)
        self.assertEqual(marked.orientation, records)
        self.assertIsNotNone(marked.lbfgsb)
        self.assertEqual(format_relax_restart(replace(marked, orientation=None)), format_relax_restart(plain))
        with self.assertRaises(ValueError):
            parse_relax_restart(text.rstrip("\n") + "\n" + ORIENTATION_HEADER + "\n" + "\n".join(records[:3]), 2)


if __name__ == "__main__":
    unittest.main()
