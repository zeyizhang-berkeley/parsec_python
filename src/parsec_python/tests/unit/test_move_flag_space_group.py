"""The space group a relaxation of a bulk cell keeps where its Move_Flag breaks the group (PARSEC_MOVE_FLAG_SPACE_GROUP).

A relaxation by an optimizer whose fixed atoms an operation of the space group exchanges with movable ones keeps the
operations that map the fixed atoms onto fixed atoms.  Under ``auto`` (the default since 2026-11-21) the parser
carries that subgroup: the Monkhorst-Pack mesh is reduced by it and time reversal, the density averaged over it, and
the forces of the wedge completed and averaged over it, on the default and on the Fortran-derived routines.  ``off``
keeps the routes of before: the time-reversal pairs of the whole mesh on the default routines, a refusal on the
Fortran-derived ones.

The CPU systems: H2 along z in a cube (the 16 operations of D4h; Move_Flag 1 fixes the second atom, C4v keeps it) and
the 8-atom diamond cube with its last atom fixed (Td about it: each rotation needs a translation of the face-centred
lattice that the space group of the atoms, one translation per rotation, need not hold).
"""

from __future__ import annotations

import itertools
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling
from parsec_python.Grid.space_group import (
    atoms_with_fixed_species,
    find_space_group,
    respects_flags,
    space_group_atom_images,
)
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.models import Atom

_ROOT = Path(__file__).resolve().parents[4]
_H = _ROOT / "examples" / "0d_AuH" / "H_POTRE.DAT"
_C = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_SWITCH = "PARSEC_MOVE_FLAG_SPACE_GROUP"

_H2 = """
Periodic_System: .true.
Boundary_Conditions bulk
begin Cell_Shape
 6.0 0.0 0.0
 0.0 6.0 0.0
 0.0 0.0 6.0
end Cell_Shape
Lattice_Vector_Scale: 1.0
Grid_Spacing: 0.3
Expansion_Order: 8
Correlation_Type: ca
States_Num: 3
Convergence_Criterion: 1e-8 Ry
Max_Iter: 80
Eigensolver: chebdav
Diag_Tolerance: 1e-9
Fermi_Temp: 300 K
Mixing_Param: 0.1
Memory_Param: 6
Mixing_Safeguard: true
{extra}
Atom_Types_Num: 1
Coordinate_Unit: cartesian_bohr
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
{flag}
begin Atom_Coord
 3.0 3.0 2.4
 3.0 3.0 3.6
end Atom_Coord
"""

_SHIFTED_MESH = (
    "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n3 3 1\nend Monkhorst_Pack_Grid\n"
    "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"
)
# the nine points of that mesh with equal weights (the whole mesh; reciprocal-lattice coordinates)
_WHOLE_MESH = (
    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
    + "".join(
        f" {a:.15f} {b:.15f} 0.0 {1.0 / 9.0:.15f}\n" for a, b in itertools.product((0.0, 1.0 / 3.0, -1.0 / 3.0), repeat=2)
    )
    + "end Kpoint_List\n"
)

_A = 3.567 / 0.529177210903
_BASIS = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
_DIAMOND = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {edge:.12f} 0.0 0.0
  0.0 {edge:.12f} 0.0
  0.0 0.0 {edge:.12f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing:.14f}
Expansion_Order: 8
Correlation_Type: pbe
States_Num: 22
{extra}
Atom_Types_Num: 1
Atom_Type: C
Pseudopotential_Format: martins_new
Local_Component: d
Cubic_Spline: true
Move_Flag: 7
begin Atom_Coord
{atoms}end Atom_Coord
"""
_DIAMOND_MESH = (
    "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n"
    "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"
)


_AU = _ROOT / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_AU_HCP = 5.45
# hcp Au2 (P6_3/mmc, 24 operations) with atom 0 fixed (D3h about it, 12 operations, of which the 4 of an orthogonal
# subgroup keep the finite-difference operators of the oblique grid), or an Au2 pair along z in a cube (D4h; the
# second atom fixed: C4v)
_AU2 = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
{cell}end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing}
Expansion_Order: 6
Correlation_Type: ca
States_Num: 20
Fermi_Temp: 1000.0
Minimization: bfgs
{extra}
Atom_Types_Num: 1
Atom_Type: Au
Pseudopotential_Format: martins_new
Local_Component: s
Move_Flag: -1
begin Atom_Coord
{atoms}end Atom_Coord
"""
_HCP_MESH = (
    "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n4 4 2\nend Monkhorst_Pack_Grid\n"
    "begin Monkhorst_Pack_Shift\n0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n"
)


def _hcp_text(extra: str) -> str:
    lattice = np.array(
        [[_AU_HCP, 0.0, 0.0], [-0.5 * _AU_HCP, 0.5 * np.sqrt(3.0) * _AU_HCP, 0.0], [0.0, 0.0, 1.633 * _AU_HCP]]
    )
    atoms = np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]) @ lattice
    return _AU2.format(
        cell="".join(f" {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}\n" for r in lattice),
        spacing=0.3,
        extra=extra,
        atoms="".join(f"  {p[0]:.12f} {p[1]:.12f} {p[2]:.12f} {k}\n" for k, p in enumerate(atoms)),
    )


def _pair_text(extra: str) -> str:
    return _AU2.format(
        cell=" 6.2 0.0 0.0\n 0.0 6.2 0.0\n 0.0 0.0 6.2\n",
        spacing=0.31,
        extra=extra,
        atoms="  3.1 3.1 2.0 1\n  3.1 3.1 4.2 0\n",
    )


def _diamond_atoms():
    return [
        Atom("C", [round((b + shift) * _A, 12) for b in basis])
        for shift in (0.0, 0.25)
        for basis in _BASIS
    ]


class _Folder(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        (self.folder / "H_POTRE.DAT").write_bytes(_H.read_bytes())
        (self.folder / "C_POTRE.DAT").write_bytes(_C.read_bytes())
        (self.folder / "Au_POTRE.DAT").write_bytes(_AU.read_bytes())

    def gold(self, text: str, *, environment=None):
        (self.folder / "parsec.in").write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(self.folder / "parsec.in")

    def h2(self, extra: str, *, flag: str = "Move_Flag: 1", environment=None, functional: str = "ca"):
        text = _H2.format(extra=extra, flag=flag).replace("Correlation_Type: ca", f"Correlation_Type: {functional}")
        (self.folder / "parsec.in").write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(self.folder / "parsec.in")

    def diamond(self, extra: str, *, environment=None):
        text = _DIAMOND.format(
            edge=_A,
            spacing=_A / 16 * (1.0 - 1.0e-9),
            extra=extra,
            atoms="".join(f"  {a.position[0]:.12f} {a.position[1]:.12f} {a.position[2]:.12f}\n" for a in _diamond_atoms()),
        )
        (self.folder / "parsec.in").write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(self.folder / "parsec.in")


class HelperTests(unittest.TestCase):
    def test_labelled_atoms_give_the_site_group_of_a_fixed_atom(self) -> None:
        lattice = np.eye(3) * _A
        atoms = _diamond_atoms()
        movable = [True] * 7 + [False]
        group = find_space_group(lattice, atoms)
        self.assertEqual(group.order, 48)
        self.assertFalse(respects_flags(group, lattice, atoms, movable))
        # the operations of the group itself that keep the last atom: their one translation each fixes it for few
        images = space_group_atom_images(group, lattice, atoms)
        naive = sum(1 for row in images if row[7] == 7)
        labelled = atoms_with_fixed_species(atoms, movable)
        self.assertEqual([atom.symbol for atom in labelled].count("C|fixed"), 1)
        kept = find_space_group(lattice, labelled)
        self.assertEqual(kept.order, 24)  # Td about the fixed atom
        self.assertLess(naive, kept.order)
        self.assertTrue(respects_flags(kept, lattice, atoms, movable))
        with self.assertRaises(ValueError):
            atoms_with_fixed_species(atoms, movable[:-1])


class ParserTests(_Folder):
    def test_default_routines_reduce_the_mesh_by_the_subgroup(self) -> None:
        auto = self.h2("Use_Fortran_Routines: .false.\nMinimization: bfgs\n" + _SHIFTED_MESH)
        problem = auto.problem
        self.assertEqual(problem.space_group.order, 8)  # C4v about the bond
        self.assertTrue(problem.scf.symmetrize_density)
        self.assertFalse(problem.kpoint_time_reversal)
        self.assertEqual(len(kpoint_sampling(problem)[0]), 3)
        self.assertTrue(any(line.startswith("Move_Flag fixes atoms") and f"{_SWITCH}=auto" in line
                            for line in auto.warnings))
        for environment in ({_SWITCH: "off"}, {"PARSEC_MOVE_FLAG_SUBGROUP": "off"}):
            with self.subTest(environment=environment):
                off = self.h2("Use_Fortran_Routines: .false.\nMinimization: bfgs\n" + _SHIFTED_MESH,
                              environment=environment).problem
                self.assertIsNone(off.space_group)
                self.assertTrue(off.kpoint_time_reversal)
                self.assertEqual(len(kpoint_sampling(off)[0]), 5)
        # flags that the group respects, and the forces of a single point, keep the whole group as before
        free = self.h2("Use_Fortran_Routines: .false.\nMinimization: bfgs\n" + _SHIFTED_MESH, flag="").problem
        self.assertEqual(free.space_group.order, 16)

    def test_fortran_routines_take_the_subgroup_instead_of_refusing(self) -> None:
        extra = "Use_Fortran_Routines: .true.\nMinimization: bfgs\n" + _SHIFTED_MESH
        auto = self.h2(extra).problem
        self.assertEqual(auto.space_group.order, 8)
        self.assertEqual(len(kpoint_sampling(auto)[0]), 3)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.h2(extra, environment={_SWITCH: "off"})
        # a weighted Kpoint_List is a wedge of the input's group: still refused
        weighted = (
            "Use_Fortran_Routines: .true.\nMinimization: bfgs\nKpoint_Method: manual\n"
            "Kpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n 0.0 0.0 0.0 1.0\n 0.33333333333333 0.0 0.0 4.0\n"
            " 0.33333333333333 0.33333333333333 0.0 4.0\nend Kpoint_List\n"
        )
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.h2(weighted)

    def test_collinear_spin_takes_the_subgroup_and_spinors_keep_the_route_of_before(self) -> None:
        spin = self.h2("Spin_Polarization: .true.\nMinimization: bfgs\n" + _SHIFTED_MESH)
        self.assertTrue(spin.routines.fortran)  # a spin-polarized cell with k-points
        self.assertEqual(spin.problem.space_group.order, 8)
        self.assertEqual(len(kpoint_sampling(spin.problem)[0]), 3)
        # non-collinear magnetism: the refusal of the Fortran-derived routines, the whole mesh of the default ones
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.h2("Non_Collinear_magnetism: .true.\nMinimization: bfgs\n" + _SHIFTED_MESH)
        default = self.h2(
            "Use_Fortran_Routines: .false.\nNon_Collinear_magnetism: .true.\nMinimization: bfgs\n" + _SHIFTED_MESH
        ).problem
        self.assertIsNone(default.space_group)
        self.assertEqual(len(kpoint_sampling(default)[0]), 9)

    def test_hybrids_on_a_mesh_keep_the_route_of_before(self) -> None:
        # a hybrid keeps the time-reversal pairs of its mesh, so the subgroup saves no point, and its density
        # average can hold a state that exact exchange breaks: the refusal of the Fortran-derived routines and the
        # whole mesh of the default ones, as before
        hybrid = "Minimization: bfgs\n" + _SHIFTED_MESH
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
            self.h2("Use_Fortran_Routines: .true.\n" + hybrid, functional="pbe0")
        default = self.h2("Use_Fortran_Routines: .false.\n" + hybrid, functional="pbe0").problem
        self.assertIsNone(default.space_group)
        self.assertEqual(len(kpoint_sampling(default)[0]), 5)

    def test_unknown_values_stop_the_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, _SWITCH):
            self.h2("Use_Fortran_Routines: .false.\n" + _SHIFTED_MESH, environment={_SWITCH: "maybe"})

    def test_the_site_group_of_a_fixed_atom_of_a_supercell(self) -> None:
        for routines in (".false.", ".true."):
            with self.subTest(routines=routines):
                problem = self.diamond(f"Use_Fortran_Routines: {routines}\nMinimization: bfgs\n" + _DIAMOND_MESH).problem
                self.assertEqual(problem.space_group.order, 24)
                self.assertEqual(len(kpoint_sampling(problem)[0]), 4)
                lattice = np.eye(3) * _A
                self.assertTrue(respects_flags(problem.space_group, lattice, _diamond_atoms(), [True] * 7 + [False]))

    def test_the_gamma_density_average_of_the_fortran_routines(self) -> None:
        # Symmetrize_Charge_Density at the Gamma point (the Fortran-derived routines): the density of the moved
        # structures is averaged over the operations they keep, not over the group of the first structure
        extra = "Minimization: bfgs\nSymmetrize_Charge_Density: .true.\n"
        auto = self.h2(extra)
        self.assertTrue(auto.routines.fortran)
        self.assertTrue(auto.problem.scf.symmetrize_density)
        self.assertEqual(auto.problem.space_group.order, 8)
        off = self.h2(extra, environment={_SWITCH: "off"})
        self.assertEqual(off.problem.space_group.order, 16)

    def test_oblique_cells_keep_the_operations_that_keep_the_stencil(self) -> None:
        # hcp Au2 with atom 0 fixed: of the 12 operations of D3h about it, the three- and six-fold ones turn the grid
        # axes onto the extra Laplacian direction of the oblique grid (their wedge was 7.7e-3 Ry off the whole mesh
        # at 0.3 bohr); the 4 that keep it give 14 points of the 4x4x2 mesh, on both routine sets
        for routines in (".false.", ".true."):
            with self.subTest(routines=routines):
                text = _hcp_text(f"Use_Fortran_Routines: {routines}\n" + _HCP_MESH)
                auto = self.gold(text)
                self.assertEqual(auto.routines.fortran, routines == ".true.")
                self.assertEqual(auto.problem.space_group.order, 4)
                self.assertTrue(auto.problem.scf.symmetrize_density)
                self.assertEqual(len(kpoint_sampling(auto.problem)[0]), 14)
                if routines == ".false.":
                    off = self.gold(text, environment={_SWITCH: "off"}).problem
                    self.assertIsNone(off.space_group)
                    # the time-reversal pairs of the 32 points, each point of the zone boundary with all its shortest
                    # images (PARSEC_KPOINT_FOLD=shortest, the default): 29; one folded point each (centred): 20
                    self.assertEqual(len(kpoint_sampling(off)[0]), 29)
                    centred = self.gold(text, environment={_SWITCH: "off", "PARSEC_KPOINT_FOLD": "centred"}).problem
                    self.assertEqual(len(kpoint_sampling(centred)[0]), 20)
                else:
                    with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag fixes atoms"):
                        self.gold(text, environment={_SWITCH: "off"})

    def test_the_gamma_density_average_without_a_kept_subgroup(self) -> None:
        # Symmetrize_Charge_Density of a Gamma relaxation whose flags break the group, where no subgroup is kept:
        # the moved structures leave the group, so the density is not averaged (off: over the whole group, as before)
        extra = "Use_Fortran_Routines: .true.\nSymmetrize_Charge_Density: .true.\n"
        cases = (
            # spin-orbit coupling: its subgroup is not validated
            ("spinors", lambda environment: self.gold(_pair_text(extra + "SO_from_scratch: .true.\n"),
                                                      environment=environment)),
            # a subgroup whose translations the grid of the whole group does not take (the search finds none)
            ("search", lambda environment: self.h2("Minimization: bfgs\n" + extra, environment=environment)),
        )
        for name, parse in cases:
            with self.subTest(case=name):
                with mock.patch("parsec_python.Input.parsec_input._fortran_flag_group", return_value=None):
                    auto = parse(None)
                self.assertTrue(auto.routines.fortran)
                self.assertIsNone(auto.problem.space_group)
                self.assertFalse(auto.problem.scf.symmetrize_density)
                self.assertTrue(any("Symmetrize_Charge_Density averages nothing" in line for line in auto.warnings))
                off = parse({_SWITCH: "off"})
                self.assertEqual(off.problem.space_group.order, 16)
                self.assertTrue(off.problem.scf.symmetrize_density)
        # with a subgroup found, spin-orbit coupling keeps the route above and scalar orbitals average over it
        self.assertEqual(self.h2("Minimization: bfgs\n" + extra).problem.space_group.order, 8)


class CommandLineTests(_Folder):
    """The forces and moves of the subgroup's wedge are those of the whole mesh (reference CLI, CPU)."""

    def _relax(self, extra: str, environment=None):
        from parsec_python.cli import main as reference_main

        (self.folder / "parsec.in").write_text(
            _H2.format(extra=extra + "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 1e-5 Ry/bohr\n",
                       flag="Move_Flag: 1"),
            encoding="utf-8",
        )
        previous = os.getcwd()
        os.chdir(self.folder)
        try:
            with mock.patch.dict(os.environ, environment or {}):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
                    code = reference_main([str(self.folder / "parsec.in")])
        finally:
            os.chdir(previous)
        self.assertIn(code, (0, 4), error.getvalue()[-2000:])
        with np.load(self.folder / "parsec_python_results.npz") as archive:
            return (
                np.asarray(archive["relaxation_energies"]),
                np.asarray(archive["relaxation_forces"]),
                np.asarray(archive["relaxation_positions"]),
            )

    def _compare(self, wedge, whole):
        energies, forces, positions = wedge
        self.assertEqual(energies.shape, whole[0].shape)
        self.assertGreater(energies.size, 1)
        np.testing.assert_allclose(energies, whole[0], atol=1.0e-7, rtol=0.0)
        self.assertGreater(float(np.abs(whole[1]).max()), 1.0e-3)
        np.testing.assert_allclose(forces, whole[1], atol=2.0e-6, rtol=0.0)
        np.testing.assert_allclose(positions, whole[2], atol=1.0e-6, rtol=0.0)
        # the fixed atom stays, the moving one stays on the axis of C4v
        np.testing.assert_allclose(positions[:, 1], np.broadcast_to(positions[0, 1], positions[:, 1].shape),
                                   atol=1.0e-12)
        np.testing.assert_allclose(positions[:, 0, :2], np.full(positions[:, 0, :2].shape, 3.0), atol=1.0e-8)

    def test_default_routines(self) -> None:
        wedge = self._relax("Use_Fortran_Routines: .false.\n" + _SHIFTED_MESH)
        pairs = self._relax("Use_Fortran_Routines: .false.\n" + _SHIFTED_MESH, {_SWITCH: "off"})
        self._compare(wedge, pairs)

    def test_fortran_routines(self) -> None:
        wedge = self._relax("Use_Fortran_Routines: .true.\n" + _SHIFTED_MESH)
        whole = self._relax("Use_Fortran_Routines: .true.\n" + _WHOLE_MESH)
        self._compare(wedge, whole)

    def test_fortran_routines_with_collinear_spin(self) -> None:
        extra = "Use_Fortran_Routines: .true.\nSpin_Polarization: .true.\n"
        self._compare(self._relax(extra + _SHIFTED_MESH), self._relax(extra + _WHOLE_MESH))


if __name__ == "__main__":
    unittest.main()
