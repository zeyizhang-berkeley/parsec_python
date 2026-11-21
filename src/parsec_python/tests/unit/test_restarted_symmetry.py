"""The symmetry of a restarted structure (``parsec_python.Input.restarted_symmetry``).

``Relax_Restart``, ``--restart-from`` (a ``relax_restart.dat`` or an XYZ file) and the restart of a dynamics run
replace the atoms of the input after the reader found the space group of a cell from them.  Before 2026-11-20 the
problem kept that group: a structure that breaks it had its k-point mesh reduced and its density averaged by
operations it does not have (8.9 mRy per cell for the 8-atom silicon cube with one atom moved 0.1 bohr).  These tests
check the decision on every path: a restart whose atoms keep the input's group keeps the translation as it was (the
same objects, so the same bits); one that breaks it takes what the reader finds for the restarted atoms, which is
what it finds for an input that holds them in its Atom_Coord block; a dynamics restart and a cluster change nothing
here (no group, or the point group found from the atoms solved); a problem that is not the input's keeps its group
where every operation maps the atoms and stops otherwise; the reader's refusals stop the restart.  One test checks
the energy of the k-points of a moved structure against the k-points without any group on the same grid.
"""

from __future__ import annotations

from dataclasses import fields, replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
SIDE = 10.26
SITES = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
         (0.75, 0.25, 0.75), (0.75, 0.75, 0.25))
# relax_restart.dat and the reader's XYZ unit (Relaxation.files.BOHR_TO_ANGSTROM)
ANGSTROM = 0.529177


def _positions(moves=None) -> np.ndarray:
    positions = np.asarray(SITES, dtype=np.float64) * SIDE
    for index, vector in (moves or {}).items():
        positions[index] = positions[index] + np.asarray(vector, dtype=np.float64)
    return positions


def _write_input(directory: Path, *, positions=None, spacing=0.64125, order=12, extra=(), mesh=True) -> Path:
    """The silicon cube of examples/3d_Si with the Gamma-centred 2 x 2 x 2 mesh (no shift block)."""

    positions = _positions() if positions is None else np.asarray(positions, dtype=np.float64)
    lines = [
        "Periodic_System: .true.", "Boundary_Conditions: bulk", "begin Cell_Shape", f" {SIDE} {SIDE} {SIDE}",
        "end Cell_Shape", "Lattice_Vector_Scale: 1.0 bohr", "Coordinate_Unit: Cartesian_Bohr",
        f"Grid_Spacing: {spacing}", f"Expansion_Order: {order}", "States_Num: 24", "Fermi_Temp: 300.0",
        "Correlation_Type: ca", "Convergence_Criterion: 1.0e-9", "Diag_Tolerance: 1.0e-10", "Max_Iter: 100",
        "Mixing_Method: Anderson", "Mixing_Param: 0.3",
    ]
    if mesh:
        lines += ["Kpoint_Method: mp", "begin Monkhorst_Pack_Grid", " 2 2 2", "end Monkhorst_Pack_Grid"]
    lines += [*extra, "Atom_Types_Num: 1", "Atom_Type: Si", "Local_Component: p", "begin Atom_Coord"]
    lines += [" " + " ".join(repr(float(value)) for value in row) for row in positions]
    lines += ["end Atom_Coord", ""]
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "Si_POTRE.DAT", directory / "Si_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse(path: Path):
    from parsec_python.Input import parse_parsec_input

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


def _write_xyz(path: Path, positions) -> Path:
    rows = [f"Si {x * ANGSTROM:.12f} {y * ANGSTROM:.12f} {z * ANGSTROM:.12f}" for x, y, z in positions]
    path.write_text("\n".join([str(len(rows)), "restart", *rows]) + "\n", encoding="utf-8")
    return path


def _write_relax_restart(directory: Path, positions) -> Path:
    from parsec_python.Relaxation.files import RelaxRestart, format_relax_restart

    path = directory / "relax_restart.dat"
    path.write_text(
        format_relax_restart(RelaxRestart(positions=np.asarray(positions, dtype=np.float64), iteration=1,
                                          movable_atoms=len(positions))),
        encoding="utf-8",
    )
    return path


def _order(group) -> int:
    return 0 if group is None else int(group.order)


class _Folder(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="parsec_restartgroup_"))
        self.addCleanup(shutil.rmtree, self.folder, True)

    def assert_unchanged(self, translation, restarted, notes) -> None:
        """The restarted translation is the input's but for the atoms: the same values, so the same bits."""

        for field in fields(translation.problem):
            if field.name in ("atoms", "recenter_geometry"):
                continue
            new, old = getattr(restarted.problem, field.name), getattr(translation.problem, field.name)
            if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
                np.testing.assert_array_equal(new, old, err_msg=field.name)
            else:
                self.assertEqual(new, old, field.name)
        self.assertIs(restarted.problem.space_group, translation.problem.space_group)
        self.assertEqual(restarted.warnings, translation.warnings)
        self.assertIs(restarted.gw, translation.gw)
        self.assertFalse(any("another symmetry" in note for note in notes), notes)

    def assert_as_input(self, restarted, reference) -> None:
        """What follows from the symmetry is that of an input that holds the restarted atoms."""

        from parsec_python.Input.restarted_symmetry import symmetry_signature

        self.assertEqual(symmetry_signature(restarted), symmetry_signature(reference))
        self.assertEqual(restarted.problem.space_group, reference.problem.space_group)


class DefaultRoutinesTests(_Folder):
    """The k-points of the default routines: the irreducible wedge of the Gamma-centred mesh."""

    def test_an_unmoved_restart_keeps_everything(self) -> None:
        from parsec_python.Relaxation.run import restart_structure

        translation = _parse(_write_input(self.folder))
        self.assertEqual(_order(translation.problem.space_group), 48)
        # --restart-from of an XYZ file and of a relax_restart.dat, and Relax_Restart beside the input
        for source in (_write_xyz(self.folder / "same.xyz", _positions()), _write_relax_restart(self.folder,
                                                                                              _positions())):
            restarted, _, notes = restart_structure(translation, self.folder, source)
            self.assert_unchanged(translation, restarted, notes)
        relaxing = _parse(_write_input(self.folder / "relax", extra=("Minimization: bfgs", "Movement_Num: 1",
                                                                   "Relax_Restart: .true.")))
        _write_relax_restart(self.folder / "relax", _positions())
        restarted, restart, notes = restart_structure(relaxing, self.folder / "relax")
        self.assertIsNotNone(restart)
        self.assertTrue(any("relax_restart.dat" in note for note in notes))
        self.assert_unchanged(relaxing, restarted, notes)

    def test_a_moved_atom_takes_the_group_of_the_restarted_atoms(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling_details
        from parsec_python.Grid.space_group import space_group_atom_images
        from parsec_python.Relaxation.run import restart_structure

        translation = _parse(_write_input(self.folder))
        moved = _positions({0: (0.1, 0.0, 0.0)})
        restarted, _, notes = restart_structure(translation, self.folder, _write_xyz(self.folder / "moved.xyz", moved))
        group = restarted.problem.space_group
        self.assertLess(_order(group), 48)
        self.assertTrue(any("another symmetry" in note and "48" in note for note in notes), notes)
        lattice = restarted.problem.periodic_cell.lattice_vectors
        # the group of the run maps the atoms it solves; the input's does not
        space_group_atom_images(group, lattice, restarted.problem.atoms)
        with self.assertRaises(ValueError):
            space_group_atom_images(translation.problem.space_group, lattice, restarted.problem.atoms)
        # the wedge of the moved atoms holds more points, with the weights of the whole mesh
        before = kpoint_sampling_details(translation.problem)
        after = kpoint_sampling_details(restarted.problem)
        self.assertGreater(len(after.weights), len(before.weights))
        self.assertAlmostEqual(float(np.sum(after.weights)), 1.0, places=12)
        # the reader's note of the mesh names the restarted group
        self.assertTrue(any(f"by {_order(group)} operations" in note for note in restarted.warnings))
        self.assertFalse(any("by 48 operations" in note for note in restarted.warnings))
        # the same as an input that holds the restarted atoms
        reference = _parse(_write_input(self.folder / "reference", positions=[a.position for a in
                                                                              restarted.problem.atoms]))
        self.assert_as_input(restarted, reference)
        self.assertEqual(restarted.problem.grid, translation.problem.grid)

    def test_a_shifted_mesh(self) -> None:
        from parsec_python.Relaxation.run import restart_structure

        extra = ("begin Monkhorst_Pack_Shift", " 0.5 0.5 0.5", "end Monkhorst_Pack_Shift")
        translation = _parse(_write_input(self.folder, extra=extra))
        self.assertGreater(_order(translation.problem.space_group), 1)
        moved = _positions({3: (0.0, 0.07, -0.04)})
        restarted, _, notes = restart_structure(translation, self.folder, _write_xyz(self.folder / "m.xyz", moved))
        self.assertLess(_order(restarted.problem.space_group), _order(translation.problem.space_group))
        reference = _parse(_write_input(self.folder / "ref", positions=[a.position for a in restarted.problem.atoms],
                                        extra=extra))
        self.assert_as_input(restarted, reference)

    def test_a_relaxation_restart_of_a_symmetric_structure_keeps_its_group(self) -> None:
        # atom 0 moved along (1, 1, 1): the six operations of C3v about it, which the forces of a relaxation keep
        # (the force wedge, PARSEC_KPOINT_FORCE_WEDGE); its relax_restart.dat after a step along the same axis
        from parsec_python.Relaxation.run import restart_structure

        relax = ("Minimization: bfgs", "Movement_Num: 2", "Relax_Restart: .true.")
        translation = _parse(_write_input(self.folder, positions=_positions({0: (0.05, 0.05, 0.05)}), extra=relax))
        self.assertEqual(_order(translation.problem.space_group), 6)
        _write_relax_restart(self.folder, _positions({0: (0.08, 0.08, 0.08)}))
        restarted, _, notes = restart_structure(translation, self.folder)
        self.assertAlmostEqual(float(restarted.problem.atoms[0].position[0]), 0.08, places=10)
        self.assert_unchanged(translation, restarted, notes)
        # a step off the axis breaks it: the wedge of the forces is that of the restarted atoms
        _write_relax_restart(self.folder, _positions({0: (0.08, 0.05, 0.03)}))
        restarted, _, notes = restart_structure(translation, self.folder)
        self.assertLess(_order(restarted.problem.space_group), 6)
        self.assertTrue(any("another symmetry" in note for note in notes))
        reference = _parse(_write_input(self.folder / "ref", positions=[a.position for a in restarted.problem.atoms],
                                        extra=relax))
        self.assert_as_input(restarted, reference)

    def test_a_dynamics_restart_changes_nothing(self) -> None:
        # Molecular dynamics uses no group (PARSEC): the reader's choice does not depend on the atoms
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.MolecularDynamics.run import restart_structure

        dynamics = ("Molecular_Dynamics: .true.", "Restart_mode: .true.", "Tinit: 300", "T_Final: 200",
                    "T_Step: 10", "Time_Step: 20", "Step_Num: 1")
        translation = _parse(_write_input(self.folder, extra=dynamics))
        self.assertIsNotNone(translation.molecular_dynamics)
        current = _positions({0: (0.1, 0.0, 0.0)})
        count = len(current)
        state = MolecularDynamicsState(
            masses=np.full(count, 28.0855 * 1822.888486), movable=np.ones(count, dtype=np.int64),
            current_positions=current, previous_positions=current - 0.01, velocities=np.zeros((count, 3)),
            accelerations=np.zeros((count, 3)), temperature=300.0, frame=3,
        )
        (self.folder / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")
        restarted, restart, notes = restart_structure(translation, self.folder)
        self.assertIsNotNone(restart)
        self.assertEqual(_order(restarted.problem.space_group), _order(translation.problem.space_group))
        self.assert_unchanged(translation, restarted, notes)


class FortranRoutinesTests(_Folder):
    """Use_Fortran_Routines: the group also sets the grid (its fractional translations, adjust_step)."""

    def test_unmoved_and_moved(self) -> None:
        from parsec_python.Relaxation.run import restart_structure

        extra = ("Use_Fortran_Routines: .true.",)
        translation = _parse(_write_input(self.folder, extra=extra))
        self.assertTrue(translation.routines.fortran)
        self.assertEqual(_order(translation.problem.space_group), 48)
        self.assertTrue(translation.problem.grid.fractional_translations)
        restarted, _, notes = restart_structure(translation, self.folder, _write_relax_restart(self.folder,
                                                                                               _positions()))
        self.assert_unchanged(translation, restarted, notes)
        moved = _positions({0: (0.1, 0.0, 0.0)})
        restarted, _, notes = restart_structure(translation, self.folder, _write_relax_restart(self.folder, moved))
        self.assertLess(_order(restarted.problem.space_group), 48)
        self.assertEqual(restarted.problem.grid.fractional_translations, ())
        self.assertTrue(restarted.routines.fortran)
        reference = _parse(_write_input(self.folder / "ref", positions=[a.position for a in restarted.problem.atoms],
                                        extra=extra))
        self.assert_as_input(restarted, reference)

    def test_an_atom_outside_the_cell_is_folded_for_the_search(self) -> None:
        # initial.f90:322-326 folds the restarted atoms into the cell before the search: an atom one lattice vector
        # away gives the group (and the grid) of the input
        from parsec_python.Relaxation.run import restart_structure

        translation = _parse(_write_input(self.folder, extra=("Use_Fortran_Routines: .true.",)))
        shifted = _positions({5: (SIDE, 0.0, -SIDE)})
        restarted, _, notes = restart_structure(translation, self.folder, _write_relax_restart(self.folder, shifted))
        np.testing.assert_allclose(restarted.problem.atoms[5].position, shifted[5])
        self.assert_unchanged(translation, restarted, notes)


class ClusterTests(_Folder):
    """A cluster: the reader keeps no group; the run finds the point group from the atoms it solves."""

    def test_a_cluster_restart(self) -> None:
        from parsec_python.Input.restarted_symmetry import reads_symmetry_of_atoms
        from parsec_python.Relaxation.run import _cluster_operations, restart_structure, symmetry_decision

        lines = [
            "Boundary_Sphere_Radius: 10.0", "Grid_Spacing: 0.7", "Expansion_Order: 6", "States_Num: 6",
            "Correlation_Type: ca", "Coordinate_Unit: Cartesian_Bohr", "Minimization: bfgs", "Movement_Num: 1",
            "Atom_Types_Num: 1", "Atom_Type: Si", "Local_Component: p", "begin Atom_Coord", " 0.0 0.0 -2.2",
            " 0.0 0.0 2.2", "end Atom_Coord", "",
        ]
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / "Si_POTRE.DAT", self.folder / "Si_POTRE.DAT")
        (self.folder / "parsec.in").write_text("\n".join(lines), encoding="utf-8")
        translation = _parse(self.folder / "parsec.in")
        self.assertFalse(reads_symmetry_of_atoms(translation))
        restarted, _, notes = restart_structure(
            translation, self.folder, _write_relax_restart(self.folder, [[0.0, 0.3, -2.2], [0.0, 0.0, 2.2]])
        )
        self.assert_unchanged(translation, restarted, notes)
        # what the relaxation and the drivers detect is the group of the restarted atoms
        self.assertLess(len(_cluster_operations(restarted)), len(_cluster_operations(translation)))
        self.assertEqual(symmetry_decision(restarted), symmetry_decision(translation))


class OtherProblemTests(_Folder):
    """A problem with another number of atoms than the input (a supercell, the cell of a mesh)."""

    def test_its_group_stays_where_it_maps_the_atoms(self) -> None:
        from parsec_python.Grid.space_group import find_space_group
        from parsec_python.Input.restarted_symmetry import restarted_symmetry

        translation = _parse(_write_input(self.folder))
        # the four atoms of one fcc sublattice, as a problem of its own
        atoms = translation.problem.atoms[:4]
        own = find_space_group(translation.problem.periodic_cell.lattice_vectors, atoms)
        problem = replace(translation.problem, atoms=atoms, space_group=own)
        kept, notes = restarted_symmetry(replace(translation, problem=problem), "cell.xyz")
        self.assertIs(kept.problem, problem)
        self.assertEqual(notes, [])
        # the diamond group exchanges the sublattices: it cannot stay, and it cannot be found again
        problem = replace(translation.problem, atoms=atoms)
        with self.assertRaisesRegex(ValueError, "not that of the input"):
            restarted_symmetry(replace(translation, problem=problem), "cell.xyz")


class RefusalTests(_Folder):
    def test_what_the_reader_refuses_stops_the_restart(self) -> None:
        from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
        from parsec_python.Input.restarted_symmetry import restarted_symmetry
        from parsec_python.Input.routines import RoutineChoice, UseFortranRoutines

        translation = _parse(_write_input(self.folder))
        target = "parsec_python.Input.parsec_input.read_symmetry_of_structure"
        with mock.patch(target, side_effect=UnsupportedParsecOptionError("a reason")):
            with self.assertRaisesRegex(ValueError, "with the structure of moved.xyz: a reason"):
                restarted_symmetry(translation, "moved.xyz")
        with mock.patch(target, side_effect=UseFortranRoutines(RoutineChoice().with_reason("a feature"))):
            with self.assertRaisesRegex(ValueError, "Use_Fortran_Routines: .true."):
                restarted_symmetry(translation, "moved.xyz")


class SignatureTests(unittest.TestCase):
    def test_groups_compare_modulo_the_lattice(self) -> None:
        from parsec_python.Grid.space_group import find_space_group
        from parsec_python.Input.restarted_symmetry import _group_key
        from parsec_python.models import Atom

        group = find_space_group(np.eye(3) * SIDE, [Atom("Si", position) for position in _positions()])
        self.assertTrue(any(np.any(np.asarray(t) != 0.0) for t in group.translations))
        moved = replace(group, translations=tuple(
            tuple(float(value) + 1.0 + 1.0e-12 for value in translation) for translation in group.translations
        ))
        self.assertEqual(_group_key(moved), _group_key(group))
        reordered = replace(group, rotations=group.rotations[::-1], translations=group.translations[::-1])
        self.assertEqual(_group_key(reordered), _group_key(group))
        other = replace(group, translations=((0.0, 0.0, 1.0e-6),) + group.translations[1:])
        self.assertNotEqual(_group_key(other), _group_key(group))


class EnergyTests(_Folder):
    """The SCF of the k-points of a moved structure (reference solver, 12 points per edge, order 8)."""

    def test_the_restarted_kpoints_are_those_without_a_group(self) -> None:
        # The restart of the moved cube against the k-points without any group on the same grid (Ignore_Symmetry with
        # Grid_Shift 0.5: Ignore_Symmetry alone puts a point on the origin).  Before 2026-11-20 the restart reduced the
        # mesh and averaged the density with the 48 operations of the input, 5e-3 Ry below.
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.Relaxation.run import restart_structure
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        def energy(problem):
            k_points, weights = kpoint_sampling(problem)[:2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_scf_kpoints(prepare_periodic_single_point(problem), k_points, weights)
            self.assertTrue(result.converged)
            return float(result.energies.total), len(weights)

        moved = _positions({0: (0.1, 0.0, 0.0)})
        translation = _parse(_write_input(self.folder, spacing=0.855, order=8))
        restarted, _, _ = restart_structure(translation, self.folder, _write_xyz(self.folder / "moved.xyz", moved))
        unsymmetric = _parse(_write_input(
            self.folder / "none", spacing=0.855, order=8, positions=[a.position for a in restarted.problem.atoms],
            extra=("Ignore_Symmetry: .true.", "begin Grid_Shift", " 0.5 0.5 0.5", "end Grid_Shift"),
        ))
        self.assertEqual(restarted.problem.grid.shift, unsymmetric.problem.grid.shift)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "gradient"}):
            found, points = energy(restarted.problem)
            reference, all_points = energy(unsymmetric.problem)
            stale, stale_points = energy(replace(restarted.problem, space_group=translation.problem.space_group))
        self.assertLess(points, all_points)
        self.assertAlmostEqual(found, reference, delta=2.0e-7)
        self.assertGreater(abs(stale - reference), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
