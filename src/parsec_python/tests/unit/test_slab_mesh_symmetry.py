"""The space group of a slab with a Monkhorst-Pack mesh (``PARSEC_SLAB_MESH_SYMMETRY``, 2026-11-19).

PARSEC 1.5 searches no symmetry for a slab (``parsec.F90:470-486``, ``ierr = 2``) unless a band structure, spin-orbit
coupling or an NSCF run asks for it, so its mesh is reduced by time reversal alone.  Under ``auto`` (the default) a
slab of scalar or collinear-spin orbitals with a native LDA or GGA takes the operations of its group that keep its
grid and its mesh and are signed permutations of the lattice directions (``full``: all that keep its grid and its
mesh): the mesh is reduced to their wedge, the density symmetrized and the forces averaged with them.  Checked here:

* the switch and its values;
* :func:`parsec_python.Grid.confined.slab_grid_layout` against the grid the builder makes;
* :func:`parsec_python.Grid.space_group.slab_mesh_subgroup`: operations whose fractional translation is no whole
  number of steps, and operations reversing ``z`` on planes that are not symmetric about ``z = 0``, are dropped;
* the parser: h-BN with ``Monkhorst_Pack_Grid 6 6 1`` runs 13 points under ``auto`` (4 operations), 7 under ``full``
  (D3h, 12 operations) and PARSEC's 20 under ``off``; the group maps the grid and the mesh onto themselves and the
  wedge weights sum to one; graphene keeps the operations with the translation ``(1/3, -1/3)`` on a grid of 3n points
  and drops them on another; hybrids, a
  ``Symmetrize_Charge_Density: .false.``, a mesh of one point, a weighted ``Kpoint_List``, molecular dynamics and a
  ``Move_Flag`` that breaks the group keep the former route;
* the report prints the group instead of PARSEC's "skipping non-trivial symmetry operations";
* the reference k-point SCF of a small h-BN slab: ``auto`` gives the energy of the whole mesh, ``full`` differs by the
  error of the finite differences;
* :func:`parsec_python.Grid.space_group.slab_mesh_wedge_exactness`: a four-fold rotation without a diagonal mirror on a
  mesh through the zone boundary computes points of the time-reversal wedge only modulo a reciprocal vector, so an S4
  slab on the unshifted 4 x 4 x 1 mesh keeps the former route under ``auto`` (the shifted mesh takes the wedge), and a
  group that reduces no point (a two-fold axis alone) is not used;
* charged sheets that are not symmetric about ``z = 0`` drop the operations reversing ``z`` (an h-BN capacitor: the
  energy and the dipole of ``off``), symmetric sheets keep them; the Ylm-projected DOS keeps the former route.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.confined import build_confined_periodic_grid, slab_grid_layout
from parsec_python.Grid.kpoints import kpoint_sampling
from parsec_python.Grid.space_group import (
    SLAB_MESH_SYMMETRY_ENV,
    confined_grid_symmetry,
    find_wire_slab_space_group,
    slab_mesh_subgroup,
    slab_mesh_symmetry_setting,
    slab_mesh_wedge_exactness,
)
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_ROOT = Path(__file__).resolve().parents[4]
_HBN = _ROOT / "examples" / "3d_hybrid_functionals"
_C_POTRE = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_A_HBN = 4.73
_A_GRAPHENE = 4.6487


def _honeycomb_cell(a: float) -> np.ndarray:
    return np.array([[a, 0.0, 0.0], [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.0]])


_HBN_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: slab
begin Cell_Shape
  4.7300000000 0.0000000000 0.0
  -2.3650000000 4.0963001599 0.0
end Cell_Shape
Boundary_Sphere_Radius: 6.00
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.40
Expansion_Order: 6
Correlation_Type: {xc}
States_Num: 8
Fermi_Temp: 1 K
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
{mesh}
end Monkhorst_Pack_Grid
{extra}
Atom_Types_Num: 2

Atom_Type: B
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 1.8
begin Electron_Per_Orbital
2 1 0
end Electron_Per_Orbital
begin Atom_Coord
     0.0000000000    0.0000000000    0.0000000000
end Atom_Coord
{flag_b}
Atom_Type: N
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 1.6
begin Electron_Per_Orbital
2 3 0
end Electron_Per_Orbital
begin Atom_Coord
     0.0000000000    2.7308667733    {z_n}
end Atom_Coord
{flag_n}
"""

_GRAPHENE_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: slab
begin Cell_Shape
  4.6487000000 0.0000000000 0.0
  -2.3243500000 4.0258922946 0.0
end Cell_Shape
Boundary_Sphere_Radius: 6.0
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing}
Expansion_Order: 6
Correlation_Type: pbe
States_Num: 8
Fermi_Temp: 500 K
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
9 9 1
end Monkhorst_Pack_Grid
{extra}
Atom_Types_Num: 1
Atom_Type: C
Local_Component: d
{flag}
begin Atom_Coord
     0.0000000000    0.0000000000    0.0000000000
     0.0000000000    2.6839281964    0.0000000000
end Atom_Coord
"""


_SQUARE_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: slab
begin Cell_Shape
  7.0 0.0 0.0
  0.0 7.0 0.0
end Cell_Shape
Boundary_Sphere_Radius: 5.00
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.45
Expansion_Order: 6
Correlation_Type: ca
States_Num: 14
Fermi_Temp: 500 K
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
4 4 1
end Monkhorst_Pack_Grid
{extra}
Atom_Types_Num: 1

Atom_Type: C
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 1.6
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
begin Atom_Coord
{coords}
end Atom_Coord
"""

# four carbon atoms related by S4 about z (no mirror, no in-plane two-fold axis)
_S4_COORDS = "   1.6  0.6  0.8\n  -0.6  1.6 -0.8\n  -1.6 -0.6  0.8\n   0.6 -1.6 -0.8"
# two carbon atoms related by the two-fold axis z alone
_C2_COORDS = "   1.6  0.6  0.8\n  -1.6 -0.6  0.8"
_SHIFT = "begin Monkhorst_Pack_Shift\n0.5 0.5 0.0\nend Monkhorst_Pack_Shift"


def _sheets(*sheets) -> str:
    """``add_charged_sheets`` lines: one type per ``(charge, z)``."""

    lines = ["add_charged_sheets: .true.", f"num_charged_sheets: {len(sheets)}"]
    for charge, height in sheets:
        lines += [f"sheet_charge: {charge}", "begin Sheet_Coord", f" 0.0 0.0 {height}", "end Sheet_Coord"]
    return "\n".join(lines)


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        for name in ("B_FHIPP.DAT", "N_FHIPP.DAT", "C_FHIPP.DAT"):
            shutil.copy(_HBN / name, self.directory / name)
        shutil.copy(_C_POTRE, self.directory / "C_POTRE.DAT")

    def parse_text(self, text: str, setting: str | None = None):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        environment = {} if setting is None else {SLAB_MESH_SYMMETRY_ENV: setting}
        with mock.patch.dict(os.environ, environment):
            if setting is None:
                os.environ.pop(SLAB_MESH_SYMMETRY_ENV, None)
            return parse_parsec_input(path)

    def hbn(self, setting=None, *, mesh="6 6 1", extra="", xc="pbe", z_n=0.0, flag_b="", flag_n=""):
        return self.parse_text(
            _HBN_INPUT.format(mesh=mesh, extra=extra, xc=xc, z_n=f"{z_n:.10f}", flag_b=flag_b, flag_n=flag_n),
            setting,
        )


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SLAB_MESH_SYMMETRY_ENV, None)
            self.assertEqual(slab_mesh_symmetry_setting(), "auto")
        for value, expected in (("auto", "auto"), ("OFF", "off"), (" off ", "off"), ("", "auto"), ("Full", "full")):
            with mock.patch.dict(os.environ, {SLAB_MESH_SYMMETRY_ENV: value}):
                self.assertEqual(slab_mesh_symmetry_setting(), expected)
        with mock.patch.dict(os.environ, {SLAB_MESH_SYMMETRY_ENV: "on"}):
            with self.assertRaisesRegex(ValueError, "auto, full or off"):
                slab_mesh_symmetry_setting()


class LayoutAndSubgroupTests(unittest.TestCase):
    def settings(self, spacing=0.4, shift=(0.0, 0.0, 0.5), radius=6.0):
        return PeriodicGridSettings(
            spacing=spacing,
            expansion_order=6,
            shift=shift,
            periodicity=2,
            confinement_radius=radius,
        )

    def test_layout_is_the_builders(self) -> None:
        for a, spacing, shift, radius in (
            (_A_HBN, 0.4, (0.0, 0.0, 0.5), 6.0),
            (_A_GRAPHENE, 0.25, (0.0, 0.0, 0.5), 5.3),
            (_A_HBN, 0.3, (0.0, 0.0, 0.0), 4.0),
        ):
            with self.subTest(a=a, spacing=spacing, shift=shift):
                cell = PeriodicCell(_honeycomb_cell(a), dimensions=2)
                settings = self.settings(spacing, shift, radius)
                grid = build_confined_periodic_grid(cell, settings)
                counts, planes = slab_grid_layout(cell, settings)
                np.testing.assert_array_equal(counts, grid.points_per_axis[:2])
                built = np.unique(np.round(grid.coordinates[:, 2], 12))
                np.testing.assert_allclose(np.sort(planes), built, atol=1.0e-12)

    def graphene_group(self):
        atoms = (
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("C", (0.0, _A_GRAPHENE / np.sqrt(3.0), 0.0)),
        )
        group, dropped = find_wire_slab_space_group(_honeycomb_cell(_A_GRAPHENE), 2, 6.0, atoms)
        self.assertEqual(dropped, 0)
        self.assertEqual(group.order, 24)
        return group

    def test_translations_of_no_whole_number_of_steps_are_dropped(self) -> None:
        group = self.graphene_group()
        planes = (np.arange(-10, 10) + 0.5) * 0.25
        # 18 points: the translation (1/3, -1/3) is 6 steps; 20 points: no whole number
        kept, dropped = slab_mesh_subgroup(group, (18, 18), (0.0, 0.0), planes)
        self.assertIs(kept, group)
        self.assertEqual(dropped, 0)
        kept, dropped = slab_mesh_subgroup(group, (20, 20), (0.0, 0.0), planes)
        self.assertEqual(dropped, 12)
        self.assertEqual(kept.order, 12)
        self.assertTrue(np.all(np.abs(kept.translation_vectors) < 1.0e-12))

    def test_z_reversal_needs_symmetric_planes(self) -> None:
        group = self.graphene_group()
        asymmetric = np.arange(-10, 10) * 0.25  # -2.5 ... 2.25
        kept, dropped = slab_mesh_subgroup(group, (18, 18), (0.0, 0.0), asymmetric)
        self.assertEqual(kept.order, 12)
        self.assertTrue(np.all(kept.rotation_matrices[:, 2, 2] == 1))
        self.assertEqual(dropped, 12)

    def test_subgroup_maps_the_grid(self) -> None:
        cell = PeriodicCell(_honeycomb_cell(_A_GRAPHENE), dimensions=2)
        settings = self.settings(0.25, (0.0, 0.0, 0.5), 5.3)
        grid = build_confined_periodic_grid(cell, settings)
        counts, planes = slab_grid_layout(cell, settings)
        group, _ = slab_mesh_subgroup(self.graphene_group(), counts, (0.0, 0.0), planes)
        table = confined_grid_symmetry(grid, group)
        self.assertEqual(table.images.shape, (group.order, grid.size))
        for row in table.images:
            self.assertEqual(np.unique(row).size, grid.size)


class ParserTests(_Directory):
    def test_hbn_mesh(self) -> None:
        off = self.hbn("off")
        self.assertEqual(off.problem.space_group.order, 1)
        self.assertEqual(off.problem.space_group.crystal_system, "")
        self.assertEqual(len(kpoint_sampling(off.problem)[0]), 20)
        for setting, order, count in ((None, 4, 13), ("auto", 4, 13), ("full", 12, 7)):
            with self.subTest(setting=setting):
                auto = self.hbn(setting)
                group = auto.problem.space_group
                self.assertEqual(group.order, order)
                self.assertTrue(auto.problem.scf.symmetrize_density)
                k_points, weights = kpoint_sampling(auto.problem)
                self.assertEqual(len(k_points), count)
                self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
                self.assertTrue(any("PARSEC_SLAB_MESH_SYMMETRY" in note for note in auto.warnings))
                # the grid is that of off
                self.assertEqual(auto.problem.grid, off.problem.grid)

    def test_auto_keeps_the_signed_permutations(self) -> None:
        # h-BN: of D3h only the operations whose in-plane lattice matrix is a signed permutation (identity, the mirror
        # z -> -z, the swap of a1 and a2 and its product with it): 4 operations, 13 points
        exact = self.hbn("auto")
        group = exact.problem.space_group
        self.assertEqual(group.order, 4)
        for matrix in group.rotation_matrices:
            plane = np.abs(np.asarray(matrix)[:2, :2])
            np.testing.assert_array_equal(plane.sum(axis=0), [1, 1])
            np.testing.assert_array_equal(plane.sum(axis=1), [1, 1])
        k_points, weights = kpoint_sampling(exact.problem)
        self.assertEqual(len(k_points), 13)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertEqual(exact.problem.grid, self.hbn("off").problem.grid)

    def test_wedge_unfolds_to_the_mesh(self) -> None:
        for setting in ("auto", "full"):
            with self.subTest(setting=setting):
                self.unfold(setting)

    def unfold(self, setting: str) -> None:
        auto = self.hbn(setting)
        off = self.hbn("off")
        lattice = auto.problem.periodic_cell.lattice_vectors
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        wedge, weights = kpoint_sampling(auto.problem)
        full, full_weights = kpoint_sampling(off.problem)
        # every star of the wedge (rotations and time reversal) lies in the mesh, and their weights add up to those
        # of the time-reversal pairs of off
        rotations = auto.problem.space_group.reciprocal_rotations.astype(np.float64)

        def fractions(points):
            return points @ np.linalg.inv(reciprocal)

        mesh = {tuple(np.round(np.mod(f * 6.0, 6.0)).astype(int) % 6) for f in fractions(full)}
        mesh |= {tuple((-np.array(key)) % 6) for key in mesh}
        self.assertEqual(len(mesh), 36)
        covered = set()
        for point in fractions(wedge):
            for rotation in rotations:
                for sign in (1.0, -1.0):
                    image = sign * (rotation @ point) * 6.0
                    self.assertLess(float(np.max(np.abs(image - np.rint(image)))), 1.0e-8)
                    covered.add(tuple(np.rint(image).astype(int) % 6))
        self.assertEqual(covered, mesh)
        self.assertAlmostEqual(float(np.sum(full_weights)), 1.0, places=12)

    def test_report_prints_the_group(self) -> None:
        from parsec_python.Output.kpoint_output import space_group_lines

        auto = self.hbn("auto")
        lines = space_group_lines(auto.problem.space_group, info=1, norot=-1)
        self.assertFalse(any("skipping non-trivial" in line for line in lines))

    def test_graphene_keeps_translations_on_a_grid_of_three_n(self) -> None:
        for spacing, setting, order, count in (
            (0.25, "full", 24, 12), (0.23, "full", 12, 12), (0.25, "auto", 8, 25), (0.23, "auto", 4, 25)
        ):
            with self.subTest(spacing=spacing, setting=setting):
                translation = self.parse_text(_GRAPHENE_INPUT.format(spacing=spacing, extra="", flag=""), setting)
                self.assertEqual(translation.problem.space_group.order, order)
                self.assertEqual(len(kpoint_sampling(translation.problem)[0]), count)
                off = self.parse_text(_GRAPHENE_INPUT.format(spacing=spacing, extra="", flag=""), "off")
                self.assertEqual(translation.problem.grid, off.problem.grid)
                self.assertEqual(len(kpoint_sampling(off.problem)[0]), 41)

    def test_former_route_where_auto_does_not_apply(self) -> None:
        cases = {
            "hybrid": dict(xc="pbe0"),
            "no symmetrization": dict(extra="Symmetrize_Charge_Density: .false."),
            "one point": dict(mesh="1 1 1"),
            "dynamics": dict(extra="Molecular_Dynamics: .true.\nStep_num: 2"),
        }
        for name, arguments in cases.items():
            with self.subTest(case=name):
                translation = self.hbn("auto", **arguments)
                group = translation.problem.space_group
                self.assertTrue(group is None or group.order == 1, name)
        # Move_Flag 1 moves the first carbon of graphene only: the operations that exchange the two break the group
        moved = self.parse_text(
            _GRAPHENE_INPUT.format(spacing=0.25, extra="Minimization: bfgs", flag="Move_Flag: 1"), "auto"
        )
        self.assertEqual(moved.problem.space_group.order, 1)
        self.assertEqual(len(kpoint_sampling(moved.problem)[0]), 41)
        self.assertTrue(any("Move_Flag" in note and "slab" in note for note in moved.warnings))
        both = self.parse_text(_GRAPHENE_INPUT.format(spacing=0.25, extra="Minimization: bfgs", flag=""), "full")
        self.assertEqual(both.problem.space_group.order, 24)
        moved = self.parse_text(
            _GRAPHENE_INPUT.format(spacing=0.25, extra="Minimization: bfgs", flag="Move_Flag: 1"), "full"
        )
        self.assertEqual(moved.problem.space_group.order, 1)

    def test_buckled_relaxation_keeps_c3v(self) -> None:
        # N lifted 0.3 bohr off the plane: C3v (no z -> -z), the same wedge of 7 points under full, the mirror that
        # swaps a1 and a2 under auto (13 points)
        for setting, order, count in (("full", 6, 7), ("auto", 2, 13)):
            with self.subTest(setting=setting):
                translation = self.hbn(setting, extra="Minimization: bfgs", z_n=0.3)
                self.assertEqual(translation.problem.space_group.order, order)
                self.assertTrue(np.all(translation.problem.space_group.rotation_matrices[:, 2, 2] == 1))
                self.assertEqual(len(kpoint_sampling(translation.problem)[0]), count)

    def test_unknown_value_stops_the_parser(self) -> None:
        with self.assertRaisesRegex(Exception, "PARSEC_SLAB_MESH_SYMMETRY must be auto, full or off"):
            self.hbn("bogus")


class ScfTests(_Directory):
    """The reference k-point SCF of a small h-BN slab: the wedge with the averaged density against the whole mesh."""

    def scf(self, setting: str):
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        text = _HBN_INPUT.format(mesh="3 3 1", extra="", xc="ca", z_n="0.0000000000", flag_b="", flag_n="")
        text = text.replace("Grid_Spacing: 0.40", "Grid_Spacing: 0.45").replace(
            "Boundary_Sphere_Radius: 6.00", "Boundary_Sphere_Radius: 5.00"
        )
        problem = self.parse_text(text + "Convergence_Criterion: 1e-8 Ry\n", setting).problem
        k_points, weights = kpoint_sampling(problem)
        system = prepare_wire_slab_single_point(problem)
        result = run_scf_kpoints(system, k_points, weights)
        self.assertTrue(result.converged)
        return len(k_points), result, system

    def test_wedge_gives_the_energy_of_the_mesh(self) -> None:
        count_off, off, system_off = self.scf("off")
        count_exact, exact, system_exact = self.scf("auto")
        count_auto, auto, system_auto = self.scf("full")
        self.assertEqual((count_off, count_exact, count_auto), (5, 4, 3))
        self.assertEqual(system_off.density_symmetry.images.shape[0], 1)
        self.assertEqual(system_exact.density_symmetry.images.shape[0], 4)
        self.assertEqual(system_auto.density_symmetry.images.shape[0], 12)
        # auto: the Hamiltonian of every k-point follows the operations to round-off, the wedge is the whole mesh
        self.assertAlmostEqual(exact.energies.total, off.energies.total, delta=2.0e-7)
        np.testing.assert_allclose(exact.density, off.density, atol=1.0e-6 * float(np.max(off.density)))
        # full: the three-fold rotations follow the Bloch term -2 i k . nabla only to the error of its differences
        # along the lattice directions, large on this coarse grid (h = 0.45 bohr, order 6: 5.1e-4 Ry on A100) and
        # vanishing with it
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-3)

    def test_asymmetric_sheets_keep_the_energy_and_the_dipole(self) -> None:
        # a capacitor across a planar h-BN sheet (+0.05 e at z = -4, -0.05 e at z = +4) breaks z -> -z: auto keeps the
        # swap of a1 and a2 only (A100 run of an earlier version: auto averaged over z -> -z, -1.05e-2 Ry and the
        # dipole erased)
        results = {}
        for setting in ("off", "auto"):
            text = _HBN_INPUT.format(mesh="3 3 1", extra=_sheets((0.05, -4.0), (-0.05, 4.0)), xc="ca",
                                     z_n="0.0000000000", flag_b="", flag_n="")
            text = text.replace("Grid_Spacing: 0.40", "Grid_Spacing: 0.45").replace(
                "Boundary_Sphere_Radius: 6.00", "Boundary_Sphere_Radius: 5.00"
            )
            problem = self.parse_text(text + "Convergence_Criterion: 1e-8 Ry\n", setting).problem
            from parsec_python.SCF.kpoints import run_scf_kpoints
            from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

            k_points, weights = kpoint_sampling(problem)
            system = prepare_wire_slab_single_point(problem)
            result = run_scf_kpoints(system, k_points, weights)
            self.assertTrue(result.converged)
            z = np.asarray(system.grid.coordinates)[:, 2]
            dipole = float(np.sum(np.asarray(result.density) * z) * system.grid.volume_element)
            results[setting] = (problem.space_group, len(k_points), float(result.energies.total), dipole)
        group, count, energy, dipole = results["auto"]
        self.assertEqual((results["off"][0].order, results["off"][1]), (1, 5))
        self.assertEqual((group.order, count), (2, 4))
        self.assertTrue(np.all(group.rotation_matrices[:, 2, 2] == 1))
        self.assertAlmostEqual(energy, results["off"][2], delta=2.0e-7)
        self.assertGreater(abs(results["off"][3]), 0.1)
        self.assertAlmostEqual(dipole, results["off"][3], delta=1.0e-5)


class ExactnessTests(unittest.TestCase):
    """:func:`slab_mesh_wedge_exactness` on the in-plane parts of signed permutations (``z`` kept)."""

    @staticmethod
    def rotations(*planes):
        return np.array([[[a, b, 0], [c, d, 0], [0, 0, 1]] for a, b, c, d in planes], dtype=np.int64)

    def test_four_fold_rotation_without_mirror(self) -> None:
        square = np.diag([7.0, 7.0, 1.0])
        c4 = self.rotations((1, 0, 0, 1), (-1, 0, 0, -1), (0, -1, 1, 0), (0, 1, -1, 0))
        # (1/2, 1/4) and (1/4, 1/2) fall into one star; their folded points are images only modulo a reciprocal vector
        self.assertEqual(slab_mesh_wedge_exactness(square, (4, 4, 1), (0.0, 0.0, 0.0), c4), (False, 6, 10))
        self.assertEqual(slab_mesh_wedge_exactness(square, (6, 6, 1), (0.0, 0.0, 0.0), c4), (False, 11, 20))
        self.assertEqual(slab_mesh_wedge_exactness(square, (3, 3, 1), (0.5, 0.5, 0.0), c4), (False, 3, 5))
        # no point on the zone boundary, or too few points to join two pairs
        exact, wedge, pairs = slab_mesh_wedge_exactness(square, (4, 4, 1), (0.5, 0.5, 0.0), c4)
        self.assertTrue(exact)
        self.assertLess(wedge, pairs)
        self.assertTrue(slab_mesh_wedge_exactness(square, (3, 3, 1), (0.0, 0.0, 0.0), c4)[0])
        self.assertTrue(slab_mesh_wedge_exactness(square, (2, 2, 1), (0.0, 0.0, 0.0), c4)[0])

    def test_mirrors_and_swaps_are_exact(self) -> None:
        square = np.diag([7.0, 7.0, 1.0])
        hexagonal = _honeycomb_cell(_A_HBN)
        d4 = self.rotations(
            (1, 0, 0, 1), (-1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0),
            (0, -1, 1, 0), (0, 1, -1, 0), (1, 0, 0, -1), (-1, 0, 0, 1),
        )
        mirrors = self.rotations((1, 0, 0, 1), (-1, 0, 0, -1), (1, 0, 0, -1), (-1, 0, 0, 1))
        swaps = self.rotations((1, 0, 0, 1), (-1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0))
        for n in range(1, 9):
            for shift in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)):
                with self.subTest(n=n, shift=shift):
                    self.assertTrue(slab_mesh_wedge_exactness(square, (n, n, 1), shift, d4)[0])
                    self.assertTrue(slab_mesh_wedge_exactness(square, (n, n + 1, 1), shift, mirrors)[0])
                    self.assertTrue(slab_mesh_wedge_exactness(hexagonal, (n, n, 1), shift, swaps)[0])
        self.assertEqual(slab_mesh_wedge_exactness(hexagonal, (6, 6, 1), (0.0, 0.0, 0.0), swaps), (True, 13, 20))
        self.assertEqual(slab_mesh_wedge_exactness(square, (4, 4, 1), (0.0, 0.0, 0.0), d4), (True, 6, 10))


class ExclusionTests(_Directory):
    def square(self, setting, coords, extra=""):
        return self.parse_text(_SQUARE_INPUT.format(coords=coords, extra=extra), setting)

    def test_s4_slab_through_the_zone_boundary_keeps_the_former_route(self) -> None:
        # A100 run of an earlier version: auto (E, C2, IC4, IC4) on the unshifted 4 x 4 x 1 mesh 2.4e-5 Ry off
        unshifted = self.square("auto", _S4_COORDS)
        self.assertEqual(unshifted.problem.space_group.order, 1)
        self.assertEqual(len(kpoint_sampling(unshifted.problem)[0]), 10)
        self.assertTrue(any("four-fold rotation" in note for note in unshifted.warnings))
        shifted = self.square("auto", _S4_COORDS, _SHIFT)
        self.assertEqual(shifted.problem.space_group.order, 4)
        self.assertEqual(len(kpoint_sampling(shifted.problem)[0]), 4)
        self.assertEqual(len(kpoint_sampling(self.square("off", _S4_COORDS, _SHIFT).problem)[0]), 8)
        # full takes the rotations with the error of the finite differences
        self.assertEqual(self.square("full", _S4_COORDS).problem.space_group.order, 4)

    def test_group_that_reduces_no_point_is_not_used(self) -> None:
        for setting in ("auto", "full"):
            with self.subTest(setting=setting):
                translation = self.square(setting, _C2_COORDS)
                self.assertEqual(translation.problem.space_group.order, 1)
                self.assertEqual(len(kpoint_sampling(translation.problem)[0]), 10)

    def test_charged_sheets(self) -> None:
        asymmetric = self.hbn("auto", extra=_sheets((0.05, -4.0), (-0.05, 4.0)))
        group = asymmetric.problem.space_group
        self.assertEqual(group.order, 2)
        self.assertTrue(np.all(group.rotation_matrices[:, 2, 2] == 1))
        self.assertEqual(len(kpoint_sampling(asymmetric.problem)[0]), 13)
        self.assertTrue(any("charged sheets" in note for note in asymmetric.warnings))
        # sheets that are their own mirror image keep z -> -z
        for sheets in (((0.05, -4.0), (0.05, 4.0)), ((0.1, 0.0),)):
            with self.subTest(sheets=sheets):
                symmetric = self.hbn("auto", extra=_sheets(*sheets)).problem.space_group
                self.assertEqual(symmetric.order, 4)
                self.assertTrue(np.any(symmetric.rotation_matrices[:, 2, 2] == -1))
        self.assertEqual(self.hbn("full", extra=_sheets((0.05, -4.0), (-0.05, 4.0))).problem.space_group.order, 6)
        self.assertEqual(self.hbn("off", extra=_sheets((0.05, -4.0), (-0.05, 4.0))).problem.space_group.order, 1)

    def test_ylm_projected_dos_keeps_the_former_route(self) -> None:
        projected = self.hbn("auto", extra="Create_DOS: .true.\nYLM_DOS_L: 1")
        self.assertEqual(projected.problem.space_group.order, 1)
        self.assertEqual(len(kpoint_sampling(projected.problem)[0]), 20)
        self.assertTrue(any("YLM_DOS_L" in note and "time reversal alone" in note for note in projected.warnings))
        total = self.hbn("auto", extra="Create_DOS: .true.")
        self.assertEqual(total.problem.space_group.order, 4)
        self.assertEqual(len(kpoint_sampling(total.problem)[0]), 13)


if __name__ == "__main__":
    unittest.main()
