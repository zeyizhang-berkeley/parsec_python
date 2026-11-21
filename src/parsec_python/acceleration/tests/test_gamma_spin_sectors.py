"""Spin-polarized bulk cells at the Gamma point in the sectors of the accelerated driver (``PARSEC_GAMMA_SPIN_SECTORS``).

* The switch, the refusals of the route (:func:`parsec_python.acceleration.SCF.gamma_spin.gamma_spin_sector_refusal`
  through ``driver.periodic_sector_refusal``) and the routing of the command line (``_gamma_spin_on_device``,
  ``_reference_only_features``): what stays on the reference CLI's 1x1x1 k-point driver.
* Complete runs on CuPy against that driver (``run_scf_kpoints_spin_polarized`` at k = 0, what ``off`` runs): the O2
  molecule in a periodic box (triplet, the eight sectors of D2h), two O2 molecules related by the body-centre
  translation (sixteen operations, eight with a translation, a moment of 4), to 1e-8 Ry and 1e-8 in the moment.
* The guard: the O atom in a periodic cube (a partly filled minority p shell across three sectors) leaves the sectors
  and is then the run of ``Spin_Symmetry_Sectors: off`` on the full grid of the driver, bit for bit.
* The operations of its density test (test 2) on a cell (``PARSEC_SPIN_GUARD_CELL_GROUP``): a diamond vacancy off the
  origin, whose sectors are centred on it, is tested against Td about it (CPU: the group, the row maps, model densities;
  CuPy: V- passes at and off the origin with the same energy, the broken neutral vacancy fires test 2 at both with the
  same asymmetry, where the operations about the origin alone kept it off the origin); a perfect crystal about a bond
  centre keeps every operation about the origin in the group tested, also where the closure reaches its limit; under a
  looser ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` the operations beyond the sector group are those of the atoms to 1e-6 bohr.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.SCF import gamma_spin

_O2_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side} {side} {side}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.55
Expansion_Order: 8
States_Num: {states}
Fermi_Temp: 80
Max_Iter: 150
Convergence_Criterion: 1.0e-10
Diag_Tolerance: 1.0e-10
Eigensolver: chebff
Correlation_Type: ca
Spin_Polarization: .true.
{extra}Atom_Types_Num: 1
Atom_Type: O
Local_Component: p
begin Atom_Coord
{coordinates}end Atom_Coord
"""


def _o2_cell(*, pair: bool = False, extra: str = "", coordinates: str | None = None):
    """O2 along z about the origin of a periodic cube of 9.9 bohr (18 points a side), or (``pair``) with a second
    molecule at the body centre of a cube of 11 bohr: ``(directory, translation)``."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.spin_reference import pseudopotential_file

    side = 11.0 if pair else 9.9
    positions = [(0.0, 0.0, 1.1414), (0.0, 0.0, -1.1414)]
    if pair:
        positions += [(5.5, 5.5, 5.5 + 1.1414), (5.5, 5.5, 5.5 - 1.1414)]
    if coordinates is None:
        coordinates = "".join(f"   {x:.6f}   {y:.6f}   {z:.6f}\n" for x, y, z in positions)
    directory = Path(tempfile.mkdtemp(prefix="parsec_gspin_"))
    (directory / "parsec.in").write_text(
        _O2_INPUT.format(side=side, states=16 if pair else 10, extra=extra, coordinates=coordinates)
    )
    shutil.copyfile(pseudopotential_file("tests/data/spin_ca_O_POTRE.DAT"), directory / "O_POTRE.DAT")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        translation = parse_parsec_input(directory / "parsec.in")
    return directory, translation


def _magnetization(grid, up, down) -> float:
    return float(grid.integrate(np.asarray(up) - np.asarray(down)))


_DIAMOND_A = 3.567 / 0.529177210903
_VACANCY_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side} {side} {side}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing}
Expansion_Order: 8
States_Num: {states}
Fermi_Temp: {temperature}
Max_Iter: 200
Convergence_Criterion: 1.0e-10
Diag_Tolerance: 1.0e-9
Eigensolver: chebff
Correlation_Type: ca
Spin_Polarization: .true.
{extra}Atom_Types_Num: 1
Atom_Type: C
Local_Component: p
Initial_Spin_Polarization: 0.5
begin Atom_Coord
{coordinates}end Atom_Coord
"""


def _vacancy(site, *, charge: int = 0, temperature: float = 80.0):
    """Diamond 2 x 2 x 2 cubes (18 points per cube edge, CA) with the atom at ``site`` (in units of the cube edge)
    removed, spin-polarized, ``charge`` electrons added: ``(directory, problem)``."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.spin_reference import pseudopotential_file

    fcc = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
    basis = np.vstack([fcc, fcc + 0.25])
    cells = np.array([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    side = 2.0 * _DIAMOND_A
    positions = (cells[:, None, :] + basis[None]).reshape(-1, 3) * _DIAMOND_A
    positions = np.where(positions >= 0.5 * side, positions - side, positions)
    keep = np.linalg.norm(positions - _DIAMOND_A * np.asarray(site, dtype=float), axis=1) > 1.0e-8
    positions = positions[keep]
    directory = Path(tempfile.mkdtemp(prefix="parsec_gspin_vacancy_"))
    (directory / "parsec.in").write_text(
        _VACANCY_INPUT.format(
            side=side,
            spacing=f"{_DIAMOND_A / 18:.10f}",
            states=int(4 * len(positions) // 2 * 1.1) + 8,
            temperature=temperature,
            extra="" if not charge else f"Net_Charges: {-charge}\n",
            coordinates="".join(f"  {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in positions),
        )
    )
    shutil.copyfile(pseudopotential_file("tests/data/spin_ca_C_POTRE.DAT"), directory / "C_POTRE.DAT")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = parse_parsec_input(directory / "parsec.in").problem
    return directory, problem


class SwitchAndRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        directory, self.translation = _o2_cell()
        self.addCleanup(shutil.rmtree, directory, True)
        self.problem = self.translation.problem

    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("off", "off"), ("0", "off"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": value}):
                self.assertEqual(gamma_spin.gamma_spin_sector_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "on"}), self.assertRaises(ValueError):
            gamma_spin.gamma_spin_sector_setting()

    def test_what_the_route_takes_and_refuses(self) -> None:
        from parsec_python.acceleration.driver import periodic_sector_refusal

        problem, scf = self.problem, self.problem.scf
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto"}):
            self.assertIsNone(gamma_spin.gamma_spin_sector_refusal(problem))
            self.assertIsNone(periodic_sector_refusal(problem))
            for changed, name in (
                (replace(scf, spin_symmetry_sectors="off"), "Spin_Symmetry_Sectors: off"),
                (replace(scf, symmetrize_density=True), "Symmetrize_Charge_Density"),
                (replace(scf, noncollinear=True), "non-collinear"),
                (replace(scf, self_consistent_spin_orbit=True), "spin-orbit"),
                (replace(scf, xc_functional="pbe0"), "hybrid"),
            ):
                with self.subTest(name=name):
                    self.assertIn(name, periodic_sector_refusal(replace(problem, scf=changed)))
            # the refusals of the unpolarized route come after
            self.assertIn("k-point", periodic_sector_refusal(replace(problem, monkhorst_pack_dimensions=(1, 1, 2))))
            species = {
                symbol: replace(item, spin_orbit=True) for symbol, item in problem.pseudopotentials.items()
            }
            self.assertIn("spin-orbit species", periodic_sector_refusal(replace(problem, pseudopotentials=species)))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "off"}):
            # the message of before
            self.assertEqual(
                periodic_sector_refusal(problem), "spin polarization, spin-orbit coupling or non-collinear magnetism"
            )

    def _default_routines(self):
        # O2 across the face z = 0 of the cube on the default routines (the forces of the Fortran-derived routines,
        # which the fixture's atom outside the cell takes, keep the full grid)
        directory, translation = _o2_cell(
            extra="Use_Fortran_Routines: .false.\n", coordinates="   0.0 0.0 1.2\n   0.0 0.0 8.7\n"
        )
        self.addCleanup(shutil.rmtree, directory, True)
        return translation

    def test_the_command_line_route(self) -> None:
        translation = self._default_routines()

        def route(changed=translation, *, mpi=False, cupy=True):
            return gamma_spin.gamma_spin_on_sector_route(changed, mpi=mpi, backend_is_cupy=lambda: cupy)

        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", "PARSEC_SPIN_SECTOR_FORCES": "auto"}):
            self.assertTrue(route())
            self.assertFalse(route(mpi=True))
            self.assertFalse(route(cupy=False))
            for name, value in (
                ("restart_run", True),
                ("polarizability_field", object()),
            ):
                with self.subTest(name=name):
                    self.assertFalse(route(replace(translation, **{name: value})))
            # forces and relaxations by an optimizer keep the sectors where the rule of an unpolarized cell does;
            # molecular dynamics, manual movements and fixed atoms take the full grid of the driver
            for name, value in (
                ("compute_forces", True),
                ("relaxation", SimpleNamespace(method="bfgs")),
                ("relaxation", SimpleNamespace(method="manual")),
                ("molecular_dynamics", object()),
            ):
                with self.subTest(name=name, value=value):
                    moving = replace(translation, **{name: value})
                    self.assertTrue(route(moving))
                    self.assertFalse(route(moving, mpi=True))
                    self.assertFalse(gamma_spin.gamma_spin_on_sector_route(
                        moving, mpi=False, backend_is_cupy=lambda: True, supercell=True))
                    with mock.patch.dict(os.environ, {"PARSEC_SPIN_SECTOR_FORCES": "off"}):
                        self.assertFalse(route(moving))
                        self.assertTrue(route())
                    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}):
                        self.assertFalse(route(moving))
            fixed = replace(translation, relaxation=SimpleNamespace(method="bfgs"), movable=(True, False))
            self.assertTrue(route(fixed))
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_SECTOR_FORCES": "on"}), self.assertRaises(ValueError):
                route()
            unpolarized = replace(
                translation, problem=replace(translation.problem, scf=replace(translation.problem.scf, spin_polarized=False))
            )
            self.assertFalse(route(unpolarized))
            # the Fortran-derived routines of the fixture: forces on the k-point driver, the single point in sectors
            self.assertTrue(route(self.translation))
            self.assertFalse(route(replace(self.translation, compute_forces=True)))
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "off"}):
            self.assertFalse(route())
            self.assertFalse(route(replace(translation, compute_forces=True)))

    def test_a_full_grid_that_fits_nowhere_keeps_the_kpoint_driver(self) -> None:
        # Molecular dynamics, manual movements, Move_Flag and forces with Ignore_Symmetry solve both channels on the
        # full grid: where its estimate fits neither one device nor the GPUs of the process (the V- 5^3 cell on one
        # A100 under off, 999 atoms), the k-point driver of before, with a NOTE; the shared layout where it fits.
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.driver import SpinFallbackPlan

        translation = self._default_routines()
        gib = 2**30

        def plan(setting, basis=22 * gib):
            return lambda problem: SpinFallbackPlan(
                basis_bytes=basis, device_bytes=80 * gib, devices=4, divided=True, shareable=True,
                setting=setting, mpi=False, moves=True,
            )

        dynamics = replace(translation, molecular_dynamics=object())
        cases = (
            dynamics,
            replace(translation, relaxation=SimpleNamespace(method="manual")),
            replace(translation, relaxation=SimpleNamespace(method="bfgs"), movable=(True, False)),
        )
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", "PARSEC_SPIN_SECTOR_FORCES": "auto"}):
            for moving in cases:
                with self.subTest(moving=moving.relaxation or "dynamics"):
                    with mock.patch.object(driver, "spin_full_grid_plan", plan("off")):
                        notes = []
                        self.assertFalse(gamma_spin.gamma_spin_on_sector_route(
                            moving, mpi=False, backend_is_cupy=lambda: True, notes=notes))
                        self.assertEqual(len(notes), 1)
                        self.assertIn("keeps the 1x1x1 k-point driver", notes[0])
                    with mock.patch.object(driver, "spin_full_grid_plan", plan("auto")):
                        notes = []
                        self.assertTrue(gamma_spin.gamma_spin_on_sector_route(
                            moving, mpi=False, backend_is_cupy=lambda: True, notes=notes))
                        self.assertEqual(notes, [])
                        self.assertEqual(gamma_spin.spin_full_grid_moves_plan(moving).layout, "shared")
                    with mock.patch.object(driver, "spin_full_grid_plan", plan("off", basis=6 * gib)):
                        self.assertTrue(gamma_spin.gamma_spin_on_sector_route(
                            moving, mpi=False, backend_is_cupy=lambda: True))
                        self.assertEqual(gamma_spin.spin_full_grid_moves_plan(moving).layout, "device")
            forces = replace(translation, compute_forces=True)
            with mock.patch.object(driver, "spin_full_grid_plan", plan("off")):
                # forces in the sectors and a single point ask no plan; Ignore_Symmetry (symmetry_off) does
                self.assertIsNone(gamma_spin.spin_full_grid_moves_plan(forces))
                self.assertIsNone(gamma_spin.spin_full_grid_moves_plan(translation, symmetry_off=True))
                self.assertTrue(gamma_spin.gamma_spin_on_sector_route(forces, mpi=False, backend_is_cupy=lambda: True))
                self.assertFalse(gamma_spin.gamma_spin_on_sector_route(
                    forces, mpi=False, backend_is_cupy=lambda: True, symmetry_off=True))
                self.assertTrue(gamma_spin.gamma_spin_on_sector_route(
                    translation, mpi=False, backend_is_cupy=lambda: True, symmetry_off=True))
        from parsec_python.acceleration import cli

        self.assertTrue(cli._spin_symmetry_off(SimpleNamespace(symmetry="off"), translation))
        self.assertFalse(cli._spin_symmetry_off(SimpleNamespace(symmetry=None), translation))
        self.assertTrue(cli._spin_symmetry_off(SimpleNamespace(symmetry=None), replace(translation, ignore_symmetry=True)))
        self.assertFalse(cli._spin_symmetry_off(SimpleNamespace(symmetry="auto"), replace(translation, ignore_symmetry=True)))

    def test_the_forces_switch(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("off", "off"), ("no", "off"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_SECTOR_FORCES": value}):
                self.assertEqual(gamma_spin.spin_sector_forces_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_SECTOR_FORCES": "on"}), self.assertRaises(ValueError):
            gamma_spin.spin_sector_forces_setting()

    def test_the_mpi_runner_takes_the_cell(self) -> None:
        # The full-SCF runner divides the sectors of both channels among its ranks: single points, and under
        # PARSEC_SPIN_SECTOR_FORCES=auto the forces and relaxations that keep the sectors, and (since 2026-11-21) the
        # moves without a point group as one sector of the identity of both channels (_spin_moves_without_group);
        # _refuse_spin_ionic_steps refuses what main.py keeps on the k-point driver, and every one under off; off of
        # PARSEC_GAMMA_SPIN_SECTORS refuses the cell as before.
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        translation = self._default_routines()
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", "PARSEC_SPIN_SECTOR_FORCES": "auto"}):
            mpi_full_scf._refuse_unsupported_periodic(self.translation)
            forces = replace(translation, compute_forces=True)
            mpi_full_scf._refuse_spin_ionic_steps(forces, 4)
            mpi_full_scf._refuse_spin_ionic_steps(replace(translation, relaxation=SimpleNamespace(method="bfgs")), 4)
            # (with PARSEC_MOVE_FLAG_SUBGROUP=auto) the flags of a single point with forces move no atom: it
            # keeps the sectors; a relaxation with fixed atoms of a spin cell keeps the full grid of both channels.
            mpi_full_scf._refuse_spin_ionic_steps(replace(forces, movable=(True, False)), 4)
            self.assertFalse(mpi_full_scf._spin_moves_without_group(replace(translation, compute_forces=False)))
            for changed in (
                replace(translation, molecular_dynamics=object()),
                replace(translation, relaxation=SimpleNamespace(method="manual")),
                replace(translation, relaxation=SimpleNamespace(method="bfgs"), movable=(True, False)),
                # forces that keep the sectors, prepared without symmetry (Ignore_Symmetry, --symmetry off)
                forces,
            ):
                with self.subTest(changed=changed):
                    self.assertTrue(mpi_full_scf._spin_moves_without_group(changed))
                    mpi_full_scf._refuse_spin_ionic_steps(changed, 4)
            for changed, reason in (
                (replace(self.translation, compute_forces=True), "Fortran-derived routines"),
                (replace(self.translation, molecular_dynamics=object()), "molecular dynamics uses no symmetry"),
            ):
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    self.assertFalse(mpi_full_scf._spin_moves_without_group(changed))
                    mpi_full_scf._refuse_spin_ionic_steps(changed, 4)
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}):
                moving = replace(translation, molecular_dynamics=object())
                self.assertFalse(mpi_full_scf._spin_moves_without_group(moving))
                with self.assertRaisesRegex(ValueError, "molecular dynamics"):
                    mpi_full_scf._refuse_spin_ionic_steps(moving, 4)
            self.assertFalse(mpi_full_scf._spin_moves_without_group(
                replace(translation, molecular_dynamics=object()), supercell=True))
            with self.assertRaisesRegex(ValueError, "supercell"):
                mpi_full_scf._refuse_spin_ionic_steps(forces, 4, supercell=True)
            mpi_full_scf._refuse_spin_ionic_steps(replace(translation, molecular_dynamics=object()), 1)
            with (
                mock.patch.dict(os.environ, {"PARSEC_SPIN_SECTOR_FORCES": "off"}),
                self.assertRaisesRegex(ValueError, "forces, relaxation and molecular dynamics of a spin"),
            ):
                mpi_full_scf._refuse_spin_ionic_steps(forces, 4)
        with (
            mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "off"}),
            self.assertRaisesRegex(ValueError, "spin polarization"),
        ):
            mpi_full_scf._refuse_unsupported_periodic(self.translation)

    def test_the_reference_only_features(self) -> None:
        from parsec_python.acceleration import cli

        features = cli._reference_only_features(self.translation, accelerated_backend_selected=False)
        self.assertIn("spin polarization of a periodic system or of spin-orbit species", features)
        self.assertEqual(
            cli._reference_only_features(
                self.translation, accelerated_backend_selected=False, gamma_spin_on_device=True
            ),
            [],
        )
        arguments = SimpleNamespace(backend="auto", mpi=False)
        with (
            mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto"}),
            mock.patch.object(cli, "resolve_backend", return_value=SimpleNamespace(selected="cupy")),
        ):
            self.assertTrue(cli._gamma_spin_on_device(arguments, self.translation))
            self.assertFalse(cli._gamma_spin_on_device(SimpleNamespace(backend="auto", mpi=True), self.translation))
        with (
            mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto"}),
            mock.patch.object(cli, "resolve_backend", return_value=SimpleNamespace(selected="native")),
        ):
            self.assertFalse(cli._gamma_spin_on_device(arguments, self.translation))


class GuardGroupOfACellTests(unittest.TestCase):
    def test_the_operations_of_the_atoms_modulo_the_lattice(self) -> None:
        # The conventional diamond cube with its atom at the origin removed (a vacancy, Td about the origin): the
        # seven atoms listed in the cell keep only the six permutations of the axes; modulo the lattice all 24
        # operations of Td.  A cluster (no cell) keeps the operations of its listed atoms.
        from parsec_python.Forces.symmetry import atom_symmetry_operations
        from parsec_python.acceleration.SCF.spin_sectors import _periodic_atom_matrices
        from parsec_python.models import Atom

        a = 6.74
        basis = [(0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0), (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
                 (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)]
        atoms = [Atom("C", tuple(a * np.asarray(b, dtype=float))) for b in basis]
        self.assertEqual(len(atom_symmetry_operations(atoms)), 6)
        cell = SimpleNamespace(cell=SimpleNamespace(lattice_vectors=np.diag([a, a, a])))
        matrices = _periodic_atom_matrices(cell, atoms)
        self.assertEqual(len(matrices), 24)
        self.assertTrue(all(abs(round(float(np.linalg.det(m)))) == 1 for m in matrices))
        self.assertIsNone(_periodic_atom_matrices(SimpleNamespace(), atoms))
        # a tetragonal cell: no permutation that exchanges the long axis
        tetragonal = SimpleNamespace(cell=SimpleNamespace(lattice_vectors=np.diag([a, a, 2 * a])))
        long = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", (0.0, 0.0, a))]
        self.assertEqual(len(_periodic_atom_matrices(tetragonal, long)), 16)

    def test_the_switch(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import spin_guard_cell_group_setting

        for value, expected in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": value}):
                self.assertEqual(spin_guard_cell_group_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": "on"}), self.assertRaises(ValueError):
            spin_guard_cell_group_setting()

    def test_a_vacancy_off_the_origin(self) -> None:
        # Diamond 2 x 2 x 2 cubes with the atom at (a/2, a/2, 0) removed.  The spin sectors take D2 about the vacancy
        # with its translations (periodic_abelian_reduction, the group the driver gives a spin cell).  Test 2 tests
        # generators of Td about the vacancy, the 24 operations of the cell group, which hold every element of the
        # sector group and every operation about the origin; its rows are the images R r + t on the periodic grid.
        # The operations about the origin alone (PARSEC_SPIN_GUARD_CELL_GROUP=off, the test of before) are the swap
        # of x and y and the identity.
        from parsec_python.acceleration.SCF import spin_sectors
        from parsec_python.acceleration.SCF.kpoint_sectors import CellGroup
        from parsec_python.acceleration.Symmetry.axis_reflection import abelian_sector_operations

        for shift in (0.0, 0.5):
            with self.subTest(shift=shift):
                grid, atoms = _vacancy_cell(2, (0.5, 0.5, 0.0), shift=shift)
                sectors = _spin_cell_sectors(grid, atoms)
                self.assertEqual(sectors.group_order, 4)
                self.assertTrue(np.any(np.asarray(sectors.translations) != 0))
                images = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
                self.assertEqual(CellGroup(grid, atoms).order, 24)
                self.assertEqual(spin_sectors.guard_group_order(grid, images), 24)
                closure = spin_sectors._AffineClosure(grid.points_per_axis)
                for affine, _ in images:
                    closure.add(affine)
                for matrix, translation in zip(abelian_sector_operations(sectors), sectors.translations):
                    self.assertIn(spin_sectors._affine(matrix, translation), closure)
                for matrix in spin_sectors._periodic_atom_matrices(grid, atoms):
                    self.assertIn(spin_sectors._affine(matrix, np.zeros(3)), closure)
                sides = np.diag(np.asarray(grid.cell.lattice_vectors))
                steps = sides / np.asarray(grid.points_per_axis)
                points = np.asarray(grid.coordinates)
                for affine, rows in images:
                    moved = points @ affine[:3, :3].T + affine[:3, 3] * steps
                    difference = np.asarray(points)[rows] - moved
                    np.testing.assert_allclose(difference - sides * np.rint(difference / sides), 0.0, atol=1.0e-9)
                with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": "off"}):
                    before = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
                self.assertEqual(spin_sectors.guard_group_order(grid, before), 2)
                # The V- state (a1 t2^3 in the up channel) passes; the neutral vacancy's t2^2 in the x and y
                # partners (different sectors of D2, invariant under the swap of x and y) breaks the threefold axes
                # about the vacancy: test 2 fires now, by more than the noise bound, and did not before.
                bulk, partners = _defect_densities(grid, atoms, 6.74 * np.array((0.5, 0.5, 0.0)))
                symmetric = [bulk + partners[0] + partners[1] + partners[2], bulk]
                self.assertIsNone(spin_sectors.density_symmetry_breaking(symmetric, grid, images=images))
                broken = [bulk + partners[0] + partners[1], bulk]
                report = spin_sectors.density_symmetry_breaking(broken, grid, images=images)
                self.assertGreater(report.largest, spin_sectors.DENSITY_SYMMETRY_NOISE_BOUND)
                self.assertIn("up channel", report.lines[0])
                self.assertIn("grid steps", report.lines[0])
                self.assertIsNone(spin_sectors.density_symmetry_breaking(broken, grid, images=before))

    def test_a_vacancy_off_the_origin_of_a_larger_cell(self) -> None:
        # 3 x 3 x 3 cubes: the cell group of the detection keeps the swap of x and y of Td about the vacancy at
        # (a/2, a/2, 0) (2026-11-19: the translations of the other operations are searched among more candidates
        # than it tries), and so does the sector group.  The operations from the environments of the atoms are all
        # 24, whatever the cell group holds.
        from parsec_python.acceleration.SCF import spin_sectors

        grid, atoms = _vacancy_cell(3, (0.5, 0.5, 0.0))
        sectors = _spin_cell_sectors(grid, atoms)
        found = spin_sectors._environment_operations(grid, atoms, 1.0e-7)
        self.assertEqual(len(found), 23)
        images = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
        self.assertEqual(spin_sectors.guard_group_order(grid, images), 24)
        bulk, partners = _defect_densities(grid, atoms, 6.74 * np.array((0.5, 0.5, 0.0)))
        self.assertIsNone(
            spin_sectors.density_symmetry_breaking([bulk + sum(partners), bulk], grid, images=images)
        )
        report = spin_sectors.density_symmetry_breaking([bulk + partners[0] + partners[1], bulk], grid, images=images)
        self.assertGreater(report.largest, spin_sectors.DENSITY_SYMMETRY_NOISE_BOUND)

    def test_the_closure_and_the_recent_cell_group(self) -> None:
        # The group the operations generate (modulo the counts of the grid), with a limit that leaves it as it was;
        # the cell group of the last detection, for the same atoms, grid, tolerance and switches only.
        from parsec_python.acceleration.SCF import spin_sectors
        from parsec_python.acceleration.SCF.kpoint_sectors import CellGroup, recent_cell_group_operations

        grid, atoms = _vacancy_cell(2, (0.5, 0.5, 0.0))
        cell = CellGroup(grid, atoms)
        recent = recent_cell_group_operations(grid, atoms)
        np.testing.assert_array_equal(np.asarray(recent[0]), np.asarray(cell.matrices))
        np.testing.assert_array_equal(np.asarray(recent[1]), np.asarray(cell.shifts))
        self.assertIsNone(recent_cell_group_operations(grid, atoms[1:]))
        self.assertIsNone(recent_cell_group_operations(grid, atoms, atom_tolerance=1.0e-5))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY_CENTRE": "off"}):
            self.assertIsNone(recent_cell_group_operations(grid, atoms))
        operations = [spin_sectors._affine(matrix, shift) for matrix, shift in zip(cell.matrices, cell.shifts)]
        closure = spin_sectors._AffineClosure(grid.points_per_axis)
        for operation in operations[1:]:
            closure.add(operation)
        self.assertEqual(closure.order, 24)
        self.assertTrue(all(operation in closure for operation in operations))
        twofold = next(item for item in operations[1:] if spin_sectors._is_diagonal(item[:3, :3]))
        threefold = next(
            item for item in operations if np.array_equal(np.abs(item[:3, :3]), np.roll(np.eye(3, dtype=int), 1, axis=1))
        )
        small = spin_sectors._AffineClosure(grid.points_per_axis)
        self.assertTrue(small.add(twofold))
        self.assertEqual(small.order, 2)
        self.assertFalse(small.add(threefold, limit=4))
        self.assertEqual(small.order, 2)
        self.assertNotIn(threefold, small)
        self.assertTrue(small.add(threefold))
        self.assertEqual(small.order, 12)

    def test_a_crystal_of_many_cells_and_a_cluster(self) -> None:
        # A perfect supercell (every atom of one environment) takes the cell group about its symmetric site, with its
        # half-cell translations: Td with the eight translations of 2 x 2 x 2 cubes.  A cluster keeps its operations,
        # the same arrays under either value of the switch.
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.SCF import spin_sectors
        from parsec_python.models import Atom, GridSettings

        grid, atoms = _vacancy_cell(2, None)
        self.assertIsNone(spin_sectors._environment_operations(grid, atoms, 1.0e-7))
        sectors = _spin_cell_sectors(grid, atoms)
        images = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
        self.assertEqual(spin_sectors.guard_group_order(grid, images), 192)
        methane = (
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("H", (1.2, 1.2, 1.2)),
            Atom("H", (-1.2, -1.2, 1.2)),
            Atom("H", (-1.2, 1.2, -1.2)),
            Atom("H", (1.2, -1.2, -1.2)),
        )
        cluster = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=4))
        found = {}
        for value in ("auto", "off"):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": value}):
                found[value] = spin_sectors.atom_group_images(cluster, methane, sectors=sectors)
        self.assertEqual(len(found["auto"]), len(found["off"]))
        for (matrix, rows), (matrix_off, rows_off) in zip(found["auto"], found["off"]):
            self.assertEqual(np.asarray(matrix).shape, (3, 3))
            np.testing.assert_array_equal(matrix, matrix_off)
            np.testing.assert_array_equal(rows, rows_off)
            self.assertEqual(np.asarray(rows).dtype, np.int64)

    def test_the_operations_about_the_origin_at_the_limit_of_the_group(self) -> None:
        # A perfect diamond crystal whose origin is a bond centre: its cell group is Td about an atom with the
        # half-cell translations, the operations about the origin D3d (12), whose twofold axes and inversion carry one
        # sublattice onto the other and, with the cell group, generate the 48 x 4 n^3 operations of the crystal.
        # Where the group would grow beyond the limit of the closure with one of them (6 x 6 x 6 cubes: 41,472
        # operations; 2 x 2 x 2 cubes, 1,536, under a limit of 1,024), that one is tested on its own: the group tested
        # holds all 12, as under off, and a density with 5 % more electrons on one sublattice fires test 2 under
        # either value (2026-11-19: the limit dropped 6 of the 12, the inversion among them, and auto passed it).
        from parsec_python.acceleration.SCF import spin_sectors

        for repeats, limit in ((2, 1024), (6, spin_sectors._GUARD_GROUP_LIMIT)):
            with self.subTest(repeats=repeats), mock.patch.object(spin_sectors, "_GUARD_GROUP_LIMIT", limit):
                grid, atoms, sublattice = _crystal_about_a_bond_centre(repeats, 8)
                origin = spin_sectors._periodic_atom_matrices(grid, atoms)
                self.assertEqual(len(origin), 12)
                self.assertIsNone(spin_sectors._environment_operations(grid, atoms, 1.0e-7))
                sectors = _spin_cell_sectors(grid, atoms)
                images = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
                self.assertLessEqual(len(images), 12)
                order = spin_sectors.guard_group_order(grid, images)
                self.assertEqual(order, 48 * 4 * repeats**3)
                self.assertGreater(order, limit)
                closure = spin_sectors._AffineClosure(grid.points_per_axis)
                for affine, _ in images:
                    closure.add(affine)
                for matrix in origin:
                    self.assertIn(spin_sectors._affine(matrix, np.zeros(3)), closure)
                # the atoms sit on grid points (the move is one grid step): one value per atom, 1.05 on a sublattice
                steps = np.diag(np.asarray(grid.cell.lattice_vectors)) / np.asarray(grid.points_per_axis)
                points = np.rint(np.asarray([atom.position for atom in atoms]) / steps).astype(np.int64)
                rows = grid.rows_for_integer_coordinates(points)
                self.assertTrue(np.all(rows >= 0))
                density = np.zeros(grid.size)
                density[rows] = np.where(sublattice, 1.05, 1.0) / grid.volume_element
                with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": "off"}):
                    before = spin_sectors.atom_group_images(grid, atoms, sectors=sectors)
                for found in (images, before):
                    report = spin_sectors.density_symmetry_breaking([density, density], grid, images=found)
                    self.assertIsNotNone(report)
                    self.assertAlmostEqual(report.largest, 0.05 * len(atoms), delta=1.0e-6 * len(atoms))
                # the operations of the cell group alone keep the sublattices
                kept = [item for item in images if not np.array_equal(item[0][:3, 3], np.zeros(3))]
                self.assertIsNone(spin_sectors.density_symmetry_breaking([density, density], grid, images=kept))

    def test_a_looser_atom_tolerance(self) -> None:
        # PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-3 and the atoms of a diamond vacancy (2 x 2 x 2 cubes, (a/2, a/2, 0)) off
        # by up to 3e-4 bohr: the sectors take D2 about the vacancy at that tolerance and average the Hamiltonian over
        # it alone.  The operations beyond the sector group are taken to 1e-6 bohr (none hold these atoms), so a
        # density averaged over the sector group, as the sectors give it, passes; the threefold axes that hold the
        # atoms to 3e-4 bohr measure the structure (above the tolerance of the test).  The atoms in place keep Td.
        from parsec_python.acceleration.SCF import spin_sectors
        from parsec_python.acceleration.SCF.kpoint_sectors import CellGroup
        from parsec_python.acceleration.Symmetry import load_or_detect_reflection_reduction
        from parsec_python.acceleration.Symmetry.axis_reflection import (
            _grid_row_mapping_operation,
            abelian_sector_operations,
            periodic_abelian_reduction,
        )
        from parsec_python.models import Atom

        grid, atoms = _vacancy_cell(2, (0.5, 0.5, 0.0))
        random = np.random.default_rng(7)
        moved = [
            Atom("C", tuple(np.asarray(atom.position) + random.uniform(-1.0, 1.0, 3) * 3.0e-4 / np.sqrt(3.0)))
            for atom in atoms
        ]
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-3"}):
            origin, _ = load_or_detect_reflection_reduction(grid, moved, cache_directory=None)
            sectors = periodic_abelian_reduction(grid, moved, origin, atom_tolerance=1.0e-3)
            self.assertEqual(sectors.group_order, 4)
            images = spin_sectors.atom_group_images(grid, moved, sectors=sectors)
            self.assertEqual(spin_sectors.guard_group_order(grid, images), 4)
            exact = spin_sectors.atom_group_images(grid, atoms, sectors=_spin_cell_sectors(grid, atoms))
            self.assertEqual(spin_sectors.guard_group_order(grid, exact), 24)
        bulk, _ = _defect_densities(grid, moved, 6.74 * np.array((0.5, 0.5, 0.0)))
        average = np.zeros_like(bulk)
        operations = abelian_sector_operations(sectors)
        for matrix, shift in zip(operations, sectors.translations):
            average += bulk[_grid_row_mapping_operation(grid, matrix, 1.0e-10, translation=shift)]
        average /= len(operations)
        self.assertIsNone(spin_sectors.density_symmetry_breaking([average, average], grid, images=images))
        loose = CellGroup(grid, moved, atom_tolerance=1.0e-3)
        self.assertEqual(loose.order, 24)
        threefold = [
            (spin_sectors._affine(matrix, shift), loose.row_map(index))
            for index, (matrix, shift) in enumerate(zip(loose.matrices, loose.shifts))
            if np.array_equal(np.abs(matrix), np.roll(np.eye(3, dtype=int), 1, axis=1))
        ]
        report = spin_sectors.density_symmetry_breaking([average, average], grid, images=threefold)
        self.assertGreater(report.largest, spin_sectors.DENSITY_SYMMETRY_TOLERANCE)


def _vacancy_cell(repeats: int, site, *, points_per_cube: int = 12, shift: float = 0.0, a: float = 6.74):
    """Diamond cubes of ``a`` bohr repeated ``repeats`` times along each axis, the atom at ``site`` (in units of
    ``a``; ``None``: none) removed, on a periodic grid of ``points_per_cube`` points per cube edge:
    ``(grid, atoms)``."""

    from parsec_python.Grid.pbc import build_periodic_grid
    from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

    fcc = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
    basis = np.vstack([fcc, fcc + 0.25])
    cells = np.array([(i, j, k) for i in range(repeats) for j in range(repeats) for k in range(repeats)], float)
    side = a * repeats
    positions = np.mod((cells[:, None, :] + basis[None]).reshape(-1, 3) * a, side)
    if site is not None:
        keep = np.linalg.norm(positions - a * np.asarray(site, dtype=float), axis=1) > 1.0e-8
        assert np.count_nonzero(~keep) == 1
        positions = positions[keep]
    grid = build_periodic_grid(
        PeriodicCell(lattice_vectors=np.diag([side] * 3)),
        PeriodicGridSettings(spacing=a / points_per_cube, expansion_order=4, shift=(shift,) * 3),
    )
    assert tuple(int(value) for value in grid.points_per_axis) == (points_per_cube * repeats,) * 3
    return grid, [Atom("C", tuple(position)) for position in positions]


def _spin_cell_sectors(grid, atoms):
    """The group of the spin sectors of a cell (``driver``: no weighted sectors for two channels, the commuting
    involutions of the cell with their translations)."""

    from parsec_python.acceleration.Symmetry import load_or_detect_reflection_reduction
    from parsec_python.acceleration.Symmetry.axis_reflection import periodic_abelian_reduction

    origin, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
    return periodic_abelian_reduction(grid, atoms, origin)


def _defect_densities(grid, atoms, centre):
    """Gaussians on the atoms (modulo the cell) and the three partners ``x^2, y^2, z^2`` times a Gaussian about
    ``centre``, one electron each: ``(bulk, partners)``."""

    sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=float))
    points = np.asarray(grid.coordinates, dtype=float)

    def image(difference):
        return difference - sides * np.rint(difference / sides)

    bulk = np.zeros(points.shape[0])
    for atom in atoms:
        difference = image(points - np.asarray(atom.position))
        bulk += np.exp(-np.sum(difference * difference, axis=1) / 0.9)
    difference = image(points - np.asarray(centre, dtype=float))
    radial = np.exp(-np.sum(difference * difference, axis=1) / 2.0)
    partners = []
    for axis in range(3):
        lobe = difference[:, axis] ** 2 * radial
        partners.append(lobe / (grid.volume_element * np.sum(lobe)))
    return bulk, partners


def _lobe(grid, centre, direction):
    """One electron in the partner ``(n . r)^2`` times the Gaussian of :func:`_defect_densities` about ``centre``."""

    sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=float))
    difference = np.asarray(grid.coordinates, dtype=float) - np.asarray(centre, dtype=float)
    difference -= sides * np.rint(difference / sides)
    radial = np.exp(-np.sum(difference * difference, axis=1) / 2.0)
    lobe = (difference @ np.asarray(direction, dtype=float)) ** 2 * radial
    return lobe / (grid.volume_element * np.sum(lobe))


def _crystal_about_a_bond_centre(repeats: int, points_per_cube: int):
    """A perfect diamond crystal of ``repeats`` cubes a side moved by ``-a/8`` along each axis: the origin of the
    cell is a bond centre (D3d about it), the symmetric site of the cell group an atom.  ``(grid, atoms, sublattice)``,
    ``sublattice`` true for the atoms of the face-centred sites before the move."""

    from parsec_python.models import Atom

    a = 6.74
    grid, atoms = _vacancy_cell(repeats, None, points_per_cube=points_per_cube, a=a)
    positions = np.asarray([atom.position for atom in atoms], dtype=float)
    fractional = np.mod(positions / a, 1.0)
    sublattice = np.all(np.abs(fractional * 2.0 - np.rint(fractional * 2.0)) < 1.0e-9, axis=1)
    moved = [Atom("C", tuple(np.mod(position - a / 8.0, repeats * a))) for position in positions]
    return grid, moved, sublattice


def _sector_run(problem, *, records=None):
    from parsec_python.acceleration import driver

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = driver.prepare_single_point(problem, backend="auto")
        result = driver.run_scf(system, callback=None if records is None else records.append)
    return system, result


def _kpoint_run(problem):
    import parsec_python.acceleration  # noqa: F401  (the device k-point solver)
    from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = prepare_periodic_single_point(problem)
        result = run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1))
    return system, result


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class GammaSpinSectorRunTests(unittest.TestCase):
    # The abelian sectors of the cells (the weighted ones of PARSEC_ORBITAL_GROUP_SPIN: test_weighted_inputs.py).
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "auto", "PARSEC_ORBITAL_GROUP_SPIN": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_same_as_the_kpoint_driver(self, translation, *, sectors: int, moment: float, translations: int = 0):
        system, result = _sector_run(translation.problem)
        details = dict(result.backend.details)
        self.assertTrue(result.converged)
        self.assertTrue(details["orbital_symmetry"].startswith("CuPy real"), details["orbital_symmetry"])
        self.assertIn(f"both channels in the {sectors} sectors", details["spin_symmetry_sectors"])
        if translations:
            self.assertIn(f"{translations} of them with a translation", details["symmetry_detection"])
        kpoint_system, kpoint = _kpoint_run(translation.problem)
        self.assertTrue(kpoint.converged)
        self.assertAlmostEqual(result.energies.total, kpoint.energies.total, delta=1.0e-8)
        up, down = result.spin
        magnetization = _magnetization(system.reference.grid, up.density, down.density)
        self.assertAlmostEqual(magnetization, moment, delta=1.0e-8)
        self.assertAlmostEqual(
            magnetization, _magnetization(kpoint_system.grid, kpoint.density_up, kpoint.density_down), delta=1.0e-8
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(up.eigenvalues)), np.sort(np.ravel(kpoint.eigenvalues_up))[: len(up.eigenvalues)],
            rtol=0, atol=1.0e-7,
        )
        # the densities of both routes on the full grid
        np.testing.assert_allclose(np.asarray(up.density), np.asarray(kpoint.density_up), rtol=0, atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(down.density), np.asarray(kpoint.density_down), rtol=0, atol=1.0e-6)

    def test_oxygen_molecule_in_a_box(self) -> None:
        directory, translation = _o2_cell()
        self.addCleanup(shutil.rmtree, directory, True)
        self.assert_same_as_the_kpoint_driver(translation, sectors=8, moment=2.0)

    def test_two_molecules_related_by_the_body_centre(self) -> None:
        directory, translation = _o2_cell(pair=True)
        self.addCleanup(shutil.rmtree, directory, True)
        self.assert_same_as_the_kpoint_driver(translation, sectors=16, moment=4.0, translations=8)

    def test_the_guard_leaves_the_sectors_for_the_full_grid(self) -> None:
        from parsec_python.acceleration.tests.test_kpoint_sector_guard import _oxygen
        from parsec_python.models import SpinSectorFallbackRecord

        directory, problem = _oxygen("80", criterion="1.0e-8")
        self.addCleanup(shutil.rmtree, directory, True)
        records: list = []
        _, left = _sector_run(problem, records=records)
        self.assertTrue(any(isinstance(record, SpinSectorFallbackRecord) for record in records))
        details = dict(left.backend.details)
        self.assertIn("left the sectors", details["spin_symmetry_sectors"])
        self.assertIn("full grid", details["orbital_symmetry"])
        _, full = _sector_run(replace(problem, scf=replace(problem.scf, spin_symmetry_sectors="off")))
        self.assertIn("Spin_Symmetry_Sectors: off", dict(full.backend.details)["symmetry_detection"])
        self.assertEqual(left.iterations, full.iterations)
        self.assertEqual(left.energies.total, full.energies.total)

    def test_forces_in_the_sectors_are_those_of_the_full_grid(self) -> None:
        # The default routines (the sector forces of a cell are theirs): O2 across the face z = 0 of the cube, whose
        # D4h about the origin holds the molecule modulo the lattice.  The forces of the driver in the sectors (both
        # channels, symmetrized over the group) against those of its full grid (Spin_Symmetry_Sectors: off).
        from parsec_python.acceleration import driver

        coordinates = "   0.0 0.0 1.2\n   0.0 0.0 8.7\n"
        directory, translation = _o2_cell(extra="Use_Fortran_Routines: .false.\n", coordinates=coordinates)
        self.addCleanup(shutil.rmtree, directory, True)
        problem = translation.problem
        found = {}
        for mode in ("auto", "off"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(
                    replace(problem, scf=replace(problem.scf, spin_symmetry_sectors=mode)), backend="auto"
                )
                result = driver.run_scf(system)
                forces = driver.compute_forces(system, result, symmetrize=mode == "auto")
            self.assertTrue(result.converged)
            found[mode] = (result, np.asarray(forces.total))
        self.assertIn("8 sectors", dict(found["auto"][0].backend.details)["spin_symmetry_sectors"])
        self.assertAlmostEqual(found["auto"][0].energies.total, found["off"][0].energies.total, delta=1.0e-9)
        np.testing.assert_allclose(found["auto"][1], found["off"][1], rtol=0, atol=1.0e-8)
        self.assertGreater(abs(found["auto"][1][0, 2]), 1.0)

    def test_the_command_line_runs_the_sectors_and_off_the_kpoint_driver(self) -> None:
        # parsec_python.main on the O2 box: auto on the accelerated driver (no note of the reference solver), off on
        # the reference CLI's 1x1x1 k-point driver; the same total energy in parsec.out.
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        import re

        from parsec_python.acceleration import cli

        directory, _ = _o2_cell()
        self.addCleanup(shutil.rmtree, directory, True)
        energies = {}
        for mode in ("auto", "off"):
            output, error = StringIO(), StringIO()
            with (
                mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": mode}),
                redirect_stdout(output),
                redirect_stderr(error),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                code = cli.main([str(directory / "parsec.in")])
            self.assertEqual(code, 0, error.getvalue()[-2000:])
            self.assertEqual("reference parsec_python.cli solver" in error.getvalue(), mode == "off")
            text = (directory / "parsec.out").read_text()
            energies[mode] = float(re.findall(r"Total Energy =\s+(-?\d+\.\d+)", text)[-1])
            if mode == "auto":
                self.assertIn("orbital_symmetry = CuPy real one-dimensional", text)
        self.assertAlmostEqual(energies["auto"], energies["off"], delta=2.0e-8)

    def test_a_vacancy_off_the_origin(self) -> None:
        # V- in diamond 2 x 2 x 2 cubes (S = 3/2, a1 t2^3 in the up channel) with the vacancy at the origin and at
        # (a/2, a/2, 0), nine grid steps away: the same cell moved by a translation of the grid.  Both stay in the four
        # sectors of D2 about the vacancy (about the origin; with its translations), test 2 tests the 24 operations of
        # Td about the vacancy and passes (A100: 1.9e-8 and 5.9e-7 electrons), the energies agree (1.0e-8 Ry at the
        # criterion of 1e-10 Ry) and the moment is 3.
        from parsec_python.acceleration.SCF.spin_sectors import guard_group_order
        from parsec_python.models import SpinSectorFallbackRecord

        energies = {}
        for name, site in (("origin", (0.0, 0.0, 0.0)), ("off", (0.5, 0.5, 0.0))):
            directory, problem = _vacancy(site, charge=1)
            self.addCleanup(shutil.rmtree, directory, True)
            records: list = []
            system, result = _sector_run(problem, records=records)
            with self.subTest(site=name):
                self.assertFalse(any(isinstance(record, SpinSectorFallbackRecord) for record in records))
                self.assertTrue(result.converged)
                self.assertIn("both channels in the 4 sectors", dict(result.backend.details)["spin_symmetry_sectors"])
                grid, images = system.scalar_field_adapter._grid_and_images()
                self.assertEqual(guard_group_order(grid, images), 24)
                up, down = result.spin
                self.assertAlmostEqual(_magnetization(system.reference.grid, up.density, down.density), 3.0, delta=1.0e-8)
            energies[name] = result.energies.total
        self.assertAlmostEqual(energies["origin"], energies["off"], delta=1.0e-7)

    def test_the_neutral_vacancy_off_the_origin_fires_test_2(self) -> None:
        # The neutral vacancy (t2^2 in the up channel) at 10 K, test 1 deferred as the unconverged start levels of a
        # large cell defer it: the sectors fill the t2 levels of two partners, integer occupations that test 3 passes,
        # a density that breaks the threefold axes of Td about the vacancy.  Test 2 fires at the origin and at
        # (a/2, a/2, 0) with the same asymmetry (A100: 0.6991 electrons both); the operations about the origin alone
        # (PARSEC_SPIN_GUARD_CELL_GROUP=off, the test of before) keep that state off the origin (4e-7 electrons under
        # the swap of x and y, its only operation).  The runs stop at the record of the fallback.
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.SCF.spin_sectors import SpinSectorReducer
        from parsec_python.models import SpinSectorFallbackRecord

        class Stop(Exception):
            pass

        def run(problem):
            fired: list = []

            def callback(record):
                if isinstance(record, SpinSectorFallbackRecord):
                    fired.append(record)
                    raise Stop

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="auto")
                try:
                    result = driver.run_scf(system, callback=callback)
                except Stop:
                    result = None
            return fired, result, system

        measures = {}
        with mock.patch.object(SpinSectorReducer, "start_spectrum_report", return_value=None):
            for name, site in (("origin", (0.0, 0.0, 0.0)), ("off", (0.5, 0.5, 0.0))):
                directory, problem = _vacancy(site, temperature=10.0)
                self.addCleanup(shutil.rmtree, directory, True)
                fired, _, _ = run(problem)
                with self.subTest(site=name):
                    self.assertEqual(len(fired), 1)
                    self.assertEqual(fired[0].stage, "the SCF loop")
                    self.assertIn("breaks the point group", fired[0].reason)
                    self.assertGreater(fired[0].largest_fractional_occupation, 0.5)
                measures[name] = fired[0].largest_fractional_occupation
            self.assertAlmostEqual(measures["origin"], measures["off"], delta=1.0e-6)
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_GUARD_CELL_GROUP": "off"}):
                fired, kept, system = run(problem)
        self.assertEqual(fired, [])
        self.assertTrue(kept.converged)
        up, down = kept.spin
        self.assertAlmostEqual(_magnetization(system.reference.grid, up.density, down.density), 2.0, delta=1.0e-6)

    def test_off_keeps_the_route_of_before(self) -> None:
        from parsec_python.acceleration import driver

        directory, translation = _o2_cell()
        self.addCleanup(shutil.rmtree, directory, True)
        with mock.patch.dict(os.environ, {"PARSEC_GAMMA_SPIN_SECTORS": "off"}):
            self.assertIsNotNone(driver.periodic_sector_refusal(translation.problem))
            from parsec_python.acceleration import cli

            arguments = SimpleNamespace(backend="auto", mpi=False)
            self.assertFalse(cli._gamma_spin_on_device(arguments, translation))


if __name__ == "__main__":
    unittest.main()
