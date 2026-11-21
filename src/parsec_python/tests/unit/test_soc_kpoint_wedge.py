"""The irreducible k-point wedge of non-magnetic spin-orbit coupling on the default routines (2026-11-19).

PARSEC 1.5 reduces the Monkhorst-Pack mesh of a spin-orbit run with ``irrbz`` and time reversal as any other run that
is not non-collinear (``parsec.F90:477``, ``init_var.F90:289``, ``irrbz.f90:96-99``); the default routines kept the
whole mesh of a spinor run (cubic Au2 4x4x4: 64 points against 10).  ``PARSEC_SOC_KPOINT_WEDGE=auto`` reduces the mesh
of self-consistent spin-orbit coupling without spin polarization or non-collinear magnetism by the operations that map
the grid, the mesh and the grid's finite-difference operators onto themselves, with time reversal and the density
averaged (on an orthogonal grid the group of scalar orbitals, on an oblique one the subgroup that keeps the stencil);
``grid`` by the operations of scalar orbitals; ``off`` keeps the whole mesh.  Checked here:

* the parser: the k-points, weights and group of the wedge (the same as a zero ``Monkhorst_Pack_Shift`` block on the
  default routines; on the cube and under ``grid`` as many points as scalar orbitals), the stencil-preserving
  subgroups of fcc and hcp cells and time reversal alone where no operation keeps the stencil, the whole mesh with
  ``off``, for collinear spin and for non-collinear magnetism, the time-reversal pairs of moving atoms, the switch's
  validation, and the note of a mesh without a shift block, which says what the run makes of the mesh;
* runs on the CPU: energy and forces of the wedge (completed over its group) and of the time-reversal pairs (unequal
  weights in the force of the spinor k-point loop) against the whole mesh.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling, soc_kpoint_wedge_setting
from parsec_python.Input.parsec_input import parse_parsec_input

_AU = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
{cell}
Grid_Spacing: {spacing}
Expansion_Order: 6
States_Num: {states}
Fermi_Temp: 1000.0
Max_Iter: 150
Convergence_Criterion: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.3
Eigensolver: chebff
{extra}
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
{mesh}
end Monkhorst_Pack_Grid
{shift}
Atom_Types_Num: 1
Total_Atom_Num: {count}
Atom_Type: Au
SO_PSP: {so_psp}
Local_Component: s
Begin atom_coord
{coordinates}
End atom_coord
Correlation_Type: {xc}
"""
_FCC = "begin Cell_Shape\n0.0 3.85 3.85\n3.85 0.0 3.85\n3.85 3.85 0.0\nend Cell_Shape\nLattice_Vector_Scale: 1.0"
_CUBE = "begin Cell_Shape\n6.2 6.2 6.2\nend Cell_Shape"
_ZERO_SHIFT = "begin Monkhorst_Pack_Shift\n0 0 0\nend Monkhorst_Pack_Shift"
_SOC = "SO_from_scratch: .true."
_NOTE = "Monkhorst_Pack_Grid without Monkhorst_Pack_Shift"


_A_HCP, _C_HCP = 5.45, 8.9
_HCP = (
    "begin Cell_Shape\n"
    f"{_A_HCP} 0.0 0.0\n{-_A_HCP / 2} {_A_HCP * 3 ** 0.5 / 2} 0.0\n0.0 0.0 {_C_HCP}\n"
    "end Cell_Shape\nLattice_Vector_Scale: 1.0"
)


def _hexagonal(fractional) -> str:
    lattice = np.array([[_A_HCP, 0.0, 0.0], [-_A_HCP / 2, _A_HCP * 3 ** 0.5 / 2, 0.0], [0.0, 0.0, _C_HCP]])
    return "\n".join("  " + " ".join(f"{v:.10f}" for v in np.asarray(f) @ lattice) for f in fractional)


def _text(*, cell="cube", mesh=None, spacing=None, extra=_SOC, shift="", xc="CA", so_psp=".true."):
    mesh = mesh or ("4 4 2" if cell == "hcp" else "4 4 4")
    if cell == "fcc":
        geometry = dict(cell=_FCC, count=1, states=16, coordinates="  0.0 0.0 0.0", spacing=spacing or 0.35)
    elif cell == "hcp":
        # hcp Au2 (P6_3/mmc, 24 operations)
        coordinates = _hexagonal([(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)])
        geometry = dict(cell=_HCP, count=2, states=32, coordinates=coordinates, spacing=spacing or 0.3)
    elif cell == "c3":
        # two triangles about the hexagonal axis whose mirror lines differ, and an atom on the axis: C3 alone, whose
        # rotations turn the grid axes of the hexagonal cell onto its extra Laplacian direction
        coordinates = _hexagonal(
            [(0.1, 0.2, 0.0), (0.8, 0.9, 0.0), (0.1, 0.9, 0.0), (0.3, 0.1, 0.5), (0.9, 0.2, 0.5), (0.8, 0.7, 0.5),
             (0.0, 0.0, 0.3)]
        )
        geometry = dict(cell=_HCP, count=7, states=100, coordinates=coordinates, spacing=spacing or 0.45)
    else:
        # bcc Au2 with the second atom moved along (1, 1, 1): C3v about the first atom, forces along the axis
        geometry = dict(
            cell=_CUBE, count=2, states=30, coordinates="  0.0 0.0 0.0\n  3.35 3.35 3.35", spacing=spacing or 0.31
        )
    return _INPUT.format(mesh=mesh, extra=extra, shift=shift, xc=xc, so_psp=so_psp, **geometry)


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(_AU, self.directory / "Au_POTRE.DAT")

    def parse(self, text: str, environment=None):
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            return parse_parsec_input(path)

    def note(self, translation) -> str:
        notes = [line for line in translation.warnings if line.startswith(_NOTE)]
        self.assertEqual(len(notes), 1, notes)
        return notes[0]


class ParserTests(_Directory):
    def test_the_wedge_of_a_spin_orbit_single_point(self) -> None:
        # (cell, switch, operations, points, the same points as scalar orbitals)
        for cell, setting, order, wedge, as_scalar in (
            ("cube", "auto", 6, 13, True),
            ("cube", "grid", 6, 13, True),
            # 13 stars, one of which holds shortest images on the zone boundary that its 12 operations do not relate
            # (PARSEC_KPOINT_FOLD=shortest splits it, Grid.kpoints.irreducible_monkhorst_pack with ties)
            ("fcc", "auto", 12, 14, False),
            ("fcc", "grid", 48, 8, True),
            ("hcp", "auto", 8, 14, False),
            ("hcp", "grid", 24, 8, True),
        ):
            with self.subTest(cell=cell, setting=setting):
                environment = {"PARSEC_SOC_KPOINT_WEDGE": setting}
                translation = self.parse(_text(cell=cell), environment)
                problem = translation.problem
                self.assertFalse(translation.routines.fortran)
                self.assertTrue(problem.scf.self_consistent_spin_orbit)
                self.assertEqual(problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
                self.assertEqual(problem.space_group.order, order)
                self.assertTrue(problem.scf.symmetrize_density)
                k_points, weights = kpoint_sampling(problem)
                self.assertEqual(len(k_points), wedge)
                self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
                self.assertIn(f"reduced to its irreducible wedge by {order} operations", self.note(translation))
                # the same as a zero shift block on these routines
                block = self.parse(
                    _text(cell=cell, shift=_ZERO_SHIFT, extra=_SOC + "\nUse_Fortran_Routines: .false."), environment
                )
                np.testing.assert_array_equal(kpoint_sampling(block.problem)[0], k_points)
                np.testing.assert_array_equal(kpoint_sampling(block.problem)[1], weights)
                # the scalar orbitals of the same cell under PARSEC_SCALAR_KPOINT_WEDGE=grid keep every operation that
                # maps the grid onto itself (their group before 2026-11-21)
                scalar = kpoint_sampling(
                    self.parse(
                        _text(cell=cell, extra="", so_psp=".false."),
                        {**environment, "PARSEC_SCALAR_KPOINT_WEDGE": "grid"},
                    ).problem
                )
                self.assertEqual(len(scalar[0]) == wedge and np.array_equal(scalar[0], k_points), as_scalar)
                if setting == "auto":
                    # and by default the operations of the spin-orbit wedge (PARSEC_SCALAR_KPOINT_WEDGE=auto)
                    default = kpoint_sampling(
                        self.parse(_text(cell=cell, extra="", so_psp=".false."), environment).problem
                    )
                    np.testing.assert_array_equal(default[0], k_points)
                    np.testing.assert_array_equal(default[1], weights)
                if not as_scalar:
                    self.assertIn("do not map the finite-difference operators of the oblique grid", " ".join(translation.warnings))
        # the fcc cell keeps all 48 operations on the Fortran-derived routines' grid: 8 points
        fortran = self.parse(_text(cell="fcc", shift=_ZERO_SHIFT))
        self.assertTrue(fortran.routines.fortran)
        self.assertEqual(len(kpoint_sampling(fortran.problem)[0]), 8)

    def test_the_kept_operations_map_the_stencil_onto_itself(self) -> None:
        from parsec_python.Grid.kpoints import stencil_preserving_operations
        from parsec_python.Grid.pbc import default_periodic_grid_counts
        from parsec_python.Laplacian.lattice_metric import second_derivative_plan

        for cell, kept in (("cube", 6), ("fcc", 12), ("hcp", 8), ("c3", 1)):
            with self.subTest(cell=cell):
                problem = self.parse(_text(cell=cell), {"PARSEC_SOC_KPOINT_WEDGE": "grid"}).problem
                lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
                counts = default_periodic_grid_counts(problem.periodic_cell, problem.grid)
                group = problem.space_group  # the operations that map the grid and the mesh onto themselves
                self.assertIsNotNone(group)
                subgroup = stencil_preserving_operations(group, lattice, counts)
                self.assertEqual(subgroup.order, kept)
                if kept == group.order:
                    self.assertIs(subgroup, group)
                # every kept operation is a signed permutation of the grid axes that keeps each second difference
                plan = second_derivative_plan(lattice / np.asarray(counts, dtype=float)[:, None])
                lines = {tuple(np.eye(3, dtype=int)[axis]): plan.axial[axis] for axis in range(3)}
                lines.update(dict(zip(plan.directions, plan.weights)))
                lines.update({tuple(-v for v in key): value for key, value in list(lines.items())})
                rotations = {tuple(map(tuple, matrix)) for matrix in subgroup.rotation_matrices}
                for matrix in subgroup.rotation_matrices:
                    index_map = np.rint(np.asarray(counts)[:, None] * matrix / np.asarray(counts)[None, :]).astype(int)
                    self.assertTrue(np.all(np.abs(index_map).sum(axis=0) == 1))
                    for offset, weight in lines.items():
                        image = tuple(int(v) for v in index_map @ np.asarray(offset))
                        self.assertAlmostEqual(lines[image], weight, delta=1.0e-9 * max(lines.values()))
                    # a group: closed under products
                    for other in subgroup.rotation_matrices:
                        self.assertIn(tuple(map(tuple, matrix @ other)), rotations)

    def test_time_reversal_alone_where_no_operation_keeps_the_stencil(self) -> None:
        translation = self.parse(_text(cell="c3"))
        problem = translation.problem
        self.assertFalse(translation.routines.fortran)
        self.assertIsNone(problem.space_group)
        # the 36 time-reversal pairs of the 4x4x4 mesh, of which the pairs with several shortest images on the zone
        # boundary run as all of them (PARSEC_KPOINT_FOLD=shortest); 36 under PARSEC_KPOINT_FOLD=centred
        self.assertEqual(len(kpoint_sampling(problem)[0]), 48)
        centred = self.parse(_text(cell="c3"), {"PARSEC_KPOINT_FOLD": "centred"}).problem
        self.assertEqual(len(kpoint_sampling(centred)[0]), 36)
        self.assertIn("reduced by time reversal alone", self.note(translation))
        self.assertIn(
            "the k-point mesh is reduced by time reversal alone: the 2 operations of the space group besides the "
            "identity",
            " ".join(translation.warnings),
        )
        grid = self.parse(_text(cell="c3"), {"PARSEC_SOC_KPOINT_WEDGE": "grid"}).problem
        self.assertEqual(grid.space_group.order, 3)
        self.assertLess(len(kpoint_sampling(grid)[0]), 36)

    def test_off_keeps_the_whole_mesh(self) -> None:
        off = {"PARSEC_SOC_KPOINT_WEDGE": "off"}
        for extra in (_SOC, _SOC + "\nSkip_Force: .false.", _SOC + "\nMolecular_Dynamics: .true.\nStep_Num: 2"):
            with self.subTest(extra=extra):
                translation = self.parse(_text(extra=extra), off)
                problem = translation.problem
                self.assertIsNone(problem.monkhorst_pack_shift)
                self.assertIsNone(problem.space_group)
                self.assertFalse(problem.scf.symmetrize_density)
                self.assertFalse(problem.kpoint_time_reversal)
                k_points, weights = kpoint_sampling(problem)
                self.assertEqual(len(k_points), 64)
                np.testing.assert_array_equal(weights, np.full(64, 1.0 / 64.0))
                self.assertIn("the whole mesh (spin-orbit coupling with PARSEC_SOC_KPOINT_WEDGE=off)", self.note(translation))
        with mock.patch.dict(os.environ, {"PARSEC_SOC_KPOINT_WEDGE": "nonsense"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SOC_KPOINT_WEDGE"):
                soc_kpoint_wedge_setting()
            # every input stops at the parser, also one without spin-orbit coupling
            with self.assertRaisesRegex(Exception, "PARSEC_SOC_KPOINT_WEDGE"):
                self.parse(_text(extra="", so_psp=".false."))
        with mock.patch.dict(os.environ, {"PARSEC_SOC_KPOINT_WEDGE": ""}):
            self.assertEqual(soc_kpoint_wedge_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_SOC_KPOINT_WEDGE": " Grid "}):
            self.assertEqual(soc_kpoint_wedge_setting(), "grid")
        with mock.patch.dict(os.environ, {"PARSEC_SOC_KPOINT_WEDGE": "on"}):
            with self.assertRaisesRegex(ValueError, "auto, grid or off"):
                soc_kpoint_wedge_setting()

    def test_magnetic_spinors_keep_the_whole_mesh(self) -> None:
        for extra, words in (
            (_SOC + "\nSpin_Polarization: .true.", "spin-polarized spin-orbit coupling"),
            (_SOC + "\nNon_Collinear_magnetism: .true.", "non-collinear magnetism"),
        ):
            with self.subTest(extra=extra):
                translation = self.parse(_text(extra=extra))
                problem = translation.problem
                self.assertIsNone(problem.space_group)
                self.assertFalse(problem.kpoint_time_reversal)
                self.assertEqual(len(kpoint_sampling(problem)[0]), 64)
                note = self.note(translation)
                self.assertIn(f"the whole mesh ({words}", note)
                # the note says why, and recommends no route that reduces by operations the moments break
                self.assertIn("PARSEC 1.5 reduces such a mesh by the space group of the atoms", note)
                self.assertNotIn("Use_Fortran_Routines", note)
                for setting in ("off", "grid"):
                    with mock.patch.dict(os.environ, {"PARSEC_SOC_KPOINT_WEDGE": setting}):
                        again = self.parse(_text(extra=extra)).problem
                    np.testing.assert_array_equal(again.kpoints, problem.kpoints)

    def test_forces_take_the_wedge_and_moving_atoms_the_pairs(self) -> None:
        forces = self.parse(_text(extra=_SOC + "\nSkip_Force: .false."))
        self.assertEqual(forces.problem.space_group.order, 6)
        self.assertEqual(len(kpoint_sampling(forces.problem)[0]), 13)
        relaxation = self.parse(_text(extra=_SOC + "\nMinimization: bfgs\nMovement_Num: 2"))
        self.assertEqual(len(kpoint_sampling(relaxation.problem)[0]), 13)
        for extra, environment, words in (
            (_SOC + "\nSkip_Force: .false.", {"PARSEC_KPOINT_FORCE_WEDGE": "off"}, "PARSEC_KPOINT_FORCE_WEDGE=off"),
            (_SOC + "\nMolecular_Dynamics: .true.\nStep_Num: 2", {}, "molecular dynamics"),
        ):
            with self.subTest(extra=extra):
                translation = self.parse(_text(extra=extra), environment)
                problem = translation.problem
                self.assertIsNone(problem.space_group)
                self.assertTrue(problem.kpoint_time_reversal)
                k_points, weights = kpoint_sampling(problem)
                self.assertEqual(len(k_points), 36)  # (64 + 8)/2: the 8 points with -k = k and 28 pairs
                self.assertEqual(sorted(set(np.round(weights * 64).astype(int))), [1, 2])
                note = self.note(translation)
                self.assertIn("the whole mesh, run as its time-reversal pairs", note)
                self.assertIn(words, note)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_TIME_REVERSAL": "off", "PARSEC_KPOINT_FORCE_WEDGE": "off"}):
            whole = self.parse(_text(extra=_SOC + "\nSkip_Force: .false.")).problem
        self.assertFalse(whole.kpoint_time_reversal)
        self.assertEqual(len(kpoint_sampling(whole)[0]), 64)

    def test_the_note_of_scalar_and_hybrid_meshes(self) -> None:
        scalar = self.parse(_text(extra="", so_psp=".false."))
        self.assertIn("reduced to its irreducible wedge by 6 operations of the space group and time reversal", self.note(scalar))
        # PARSEC_HYBRID_KPOINT_WEDGE=on: a hybrid single point takes the wedge of the scalar orbitals; auto (the
        # default since 2026-11-20) the pairs of off for its semilocal start, with that wedge kept aside for the Fock
        # loop of an insulating start; off keeps the time-reversal pairs and names the switch
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": "on"}):
            hybrid = self.parse(_text(extra="", so_psp=".false.", xc="PBE0"))
        self.assertTrue(hybrid.problem.hybrid_kpoint_wedge)
        self.assertEqual(hybrid.problem.space_group.order, scalar.problem.space_group.order)
        self.assertEqual(len(kpoint_sampling(hybrid.problem)[0]), len(kpoint_sampling(scalar.problem)[0]))
        self.assertIn("the exchange of the hybrid functional sums over the whole mesh through the images", self.note(hybrid))
        deferred = self.parse(_text(extra="", so_psp=".false.", xc="PBE0"))
        self.assertFalse(deferred.problem.hybrid_kpoint_wedge)
        self.assertEqual(deferred.problem.hybrid_kpoint_wedge_deferred.group.order, scalar.problem.space_group.order)
        self.assertEqual(len(kpoint_sampling(deferred.problem)[0]), 36)
        self.assertIn("for the semilocal start of the hybrid; its Fock loop takes the irreducible wedge", self.note(deferred))
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": "off"}):
            pairs = self.parse(_text(extra="", so_psp=".false.", xc="PBE0"))
        self.assertIsNone(pairs.problem.space_group)
        self.assertEqual(len(kpoint_sampling(pairs.problem)[0]), 36)
        self.assertIn("reduced by time reversal only (k ~ -k): PARSEC_HYBRID_KPOINT_WEDGE=off", self.note(pairs))
        block = self.parse(_text(cell="fcc", extra="", so_psp=".false.", shift=_ZERO_SHIFT))
        self.assertFalse(any(line.startswith(_NOTE) for line in block.warnings))
        with mock.patch.dict(os.environ, {"PARSEC_MP_UNSHIFTED_MESH": "port"}):
            port = self.parse(_text())
        self.assertFalse(any(line.startswith(_NOTE) for line in port.warnings))
        self.assertEqual(len(kpoint_sampling(port.problem)[0]), 64)

    def test_the_fortran_routines_are_unchanged(self) -> None:
        for environment in ({}, {"PARSEC_SOC_KPOINT_WEDGE": "off"}):
            with self.subTest(environment=environment):
                translation = self.parse(_text(cell="fcc", extra=_SOC + "\nUse_Fortran_Routines: .true."), environment)
                self.assertTrue(translation.routines.fortran)
                self.assertIn("by 48 operations of the space group and time reversal", self.note(translation))
                self.assertEqual(len(kpoint_sampling(translation.problem)[0]), 8)
        # where the Fortran-derived routines refuse the wedge by name, their fall-back note alone describes the mesh
        moved = self.parse(_text(extra=_SOC + "\nUse_Fortran_Routines: .true."))
        self.assertIn("is not used", self.note(moved))
        self.assertIn("reduced by time reversal alone", self.note(moved))


def _archive(directory: Path):
    from parsec_python.tests.archives import read_archive

    return read_archive(directory / "parsec_python_results.npz")


class RunTests(_Directory):
    """Coarse cells on the CPU: what the wedge and the pairs give against the whole mesh."""

    def run_cli(self, text: str, environment=None) -> dict:
        from parsec_python.cli import main as cli_main

        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            code = cli_main([str(path), "--quiet"])
        log = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertEqual(code, 0, log[-2000:])
        archive = _archive(self.directory)
        self.assertTrue(bool(archive["converged"]))
        return {
            "energy": float(archive["energy_total_ry"]),
            "forces": np.asarray(archive["forces_total"]),
            "k": int(archive["n_kpoints"]),
        }

    def test_the_wedge_forces_are_those_of_the_whole_mesh(self) -> None:
        text = _text(mesh="2 2 2", spacing=0.6, extra=_SOC + "\nSkip_Force: .false.")
        wedge = self.run_cli(text)
        whole = self.run_cli(text, {"PARSEC_SOC_KPOINT_WEDGE": "off"})
        self.assertEqual((wedge["k"], whole["k"]), (4, 8))
        self.assertGreater(float(np.max(np.abs(whole["forces"]))), 1.0e-2)
        self.assertAlmostEqual(wedge["energy"], whole["energy"], delta=2.0e-6)
        np.testing.assert_allclose(wedge["forces"], whole["forces"], atol=2.0e-5)

    def test_the_pair_forces_are_those_of_the_whole_mesh(self) -> None:
        # 3 x 1 x 1: Gamma and the pair +-1/3 along x, weights 1/3 and 2/3 (unequal weights in the spinor force)
        text = _text(mesh="3 1 1", spacing=0.6, extra=_SOC + "\nSkip_Force: .false.")
        pairs = self.run_cli(text, {"PARSEC_KPOINT_FORCE_WEDGE": "off"})
        whole = self.run_cli(text, {"PARSEC_SOC_KPOINT_WEDGE": "off"})
        self.assertEqual((pairs["k"], whole["k"]), (2, 3))
        self.assertGreater(float(np.max(np.abs(whole["forces"]))), 1.0e-2)
        self.assertAlmostEqual(pairs["energy"], whole["energy"], delta=2.0e-6)
        np.testing.assert_allclose(pairs["forces"], whole["forces"], atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
