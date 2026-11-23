"""The Hartree potential of a cluster as the derivative of its Hartree energy (:mod:`parsec_python.Hartree.variational`)."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import scipy.sparse.linalg as spl

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Forces import compute_forces
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import build_atomic_tail, solve_hartree, valence_point_charges
from parsec_python.Hartree.boundary import point_charge_potential
from parsec_python.Hartree.variational import (
    HARTREE_VARIATIONAL_SWITCH,
    FieldLayout,
    HartreeAdjoint,
    VariationalHartree,
    ZeroBoundaryHostSolver,
    coulomb_sums,
    hartree_variational_setting,
    tail_charges,
)
from parsec_python.Input import parse_parsec_input
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.SCF import run_scf, update_geometry
from parsec_python.SCF.single_point import prepare_single_point

DATA = Path(__file__).resolve().parents[1] / "data"
KT = 6.33327186e-06


def _na2(*, forces_wanted=True, spacing=0.7, radius=7.5, tolerance=1.0e-11, grid=None):
    """Na2 in a small sphere: Solver_Lpole 2, which the boundary plan raises to 6 with the atomic tail."""

    return SinglePointInput(
        atoms=[Atom("Na", [-3.03, 0.11, 0.07]), Atom("Na", [3.02, -0.05, 0.04])],
        pseudopotentials={"Na": SpeciesPotential(DATA / "Na_POTRE.DAT", 0, use_spline=True)},
        grid=GridSettings(spacing=spacing, radius=radius, expansion_order=6) if grid is None else grid,
        scf=SCFSettings(
            max_iterations=300, number_of_states=6, convergence_criterion=1.0e-10, fermi_temperature_kelvin=500.0
        ),
        hartree=HartreeSettings(
            multipole_order=2, max_iterations=4000, relative_tolerance=tolerance, absolute_tolerance=1.0e-15
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-11),
        forces_wanted=forces_wanted,
    )


def _switch(value):
    return mock.patch.dict(os.environ, {HARTREE_VARIATIONAL_SWITCH: value})


def _unset():
    environment = {name: value for name, value in os.environ.items() if name != HARTREE_VARIATIONAL_SWITCH}
    return mock.patch.dict(os.environ, environment, clear=True)


class AdjointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(GridSettings(spacing=0.6, radius=5.0, expansion_order=6))
        self.operator = build_negative_laplacian(self.grid)
        self.lu = spl.splu(self.operator.tocsc())
        self.settings = HartreeSettings(
            multipole_order=4, relative_tolerance=1.0e-13, absolute_tolerance=1.0e-16, max_iterations=6000
        )
        radius = np.linalg.norm(self.grid.coordinates, axis=1)
        self.density = np.exp(-0.25 * radius**2) * (1.0 + 0.3 * self.grid.coordinates[:, 0])
        self.other = np.exp(-0.5 * np.linalg.norm(self.grid.coordinates - [1.0, -0.5, 0.8], axis=1) ** 2)

    def poisson(self, density):
        return solve_hartree(density, self.grid, self.operator, self.settings).potential

    def test_the_transpose_is_that_of_the_multipole_boundary(self) -> None:
        adjoint = HartreeAdjoint(self.grid, self.settings.multipole_order)
        layout = FieldLayout(self.grid)

        def transposed(density):
            u = self.lu.solve(density)
            return 8.0 * math.pi * u - adjoint.transpose(layout, u)

        forward = float(self.other @ self.poisson(self.density))
        backward = float(self.density @ transposed(self.other))
        self.assertLess(abs(forward - backward), 1.0e-10 * abs(forward))
        # K itself is not symmetric: the reason for the module
        swapped = float(self.density @ self.poisson(self.other))
        self.assertGreater(abs(forward - swapped), 1.0e-6 * abs(forward))

    def test_the_coulomb_sums_are_those_of_the_pairs(self) -> None:
        rng = np.random.default_rng(5)
        points = rng.uniform(-6.0, 6.0, (37, 3)) + [0.0, 0.0, 9.0]
        positions = rng.uniform(-2.0, 2.0, (5, 3))
        charges = rng.uniform(0.5, 4.0, 5)
        weights = rng.standard_normal(37)
        difference = points[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(difference, axis=2)
        potential = 2.0 * (charges / distance).sum(axis=1)
        forces = 2.0 * charges[:, None] * np.einsum("p,pa,paj->aj", weights, 1.0 / distance**3, difference)
        for chunk in (10, 80, 10_000):
            with self.subTest(chunk=chunk):
                np.testing.assert_allclose(coulomb_sums(points, positions, charges, chunk_pairs=chunk), potential, rtol=1.0e-13)
                np.testing.assert_allclose(
                    coulomb_sums(points, positions, charges, weights=weights, chunk_pairs=chunk), forces, rtol=1.0e-12
                )
        np.testing.assert_allclose(point_charge_potential(points, positions, charges), potential, rtol=1.0e-13)


class PotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        with _switch("on"):
            self.system = prepare_single_point(_na2(tolerance=1.0e-13))
        self.assertTrue(self.system.hartree_boundary.atomic_tail)
        self.helper = self.system.hartree_variational
        self.volume = self.system.grid.volume_element

    def plain(self, density):
        system = self.system
        return solve_hartree(
            density, system.grid, system.negative_laplacian, system.input.hartree,
            boundary_tail=system.hartree_boundary_tail,
        )

    def test_the_energy_is_kept_and_the_potential_is_its_derivative(self) -> None:
        density = np.asarray(self.system.initial_density, dtype=float)
        result = self.plain(density)
        varied = self.helper.potential(density, result, tail_charges(self.system)).potential
        energy = 0.5 * self.volume * float(density @ result.potential)
        self.assertLess(abs(0.5 * self.volume * float(density @ varied) - energy), 1.0e-11 * abs(energy))
        x = self.system.grid.coordinates
        change = density * (x[:, 0] - 0.4 * x[:, 2])
        change -= change.sum() / change.size
        step = 1.0e-3

        def hartree_energy(values):
            return 0.5 * self.volume * float(values @ self.plain(values).potential)

        slope = (hartree_energy(density + step * change) - hartree_energy(density - step * change)) / (2.0 * step)
        self.assertLess(abs(self.volume * float(change @ varied) - slope), 1.0e-9 * abs(slope))
        # the potential of the Poisson solve is not the derivative
        self.assertGreater(abs(self.volume * float(change @ result.potential) - slope), 1.0e-6 * abs(slope))

    def test_the_tail_force_is_the_derivative_of_the_tail_energy(self) -> None:
        system = self.system
        density = np.asarray(system.initial_density, dtype=float)
        positions, charges = valence_point_charges(system.atoms, system.pseudopotentials, system.electron_count)
        lu = spl.splu(system.negative_laplacian.tocsc())
        order = system.input.hartree.multipole_order

        def tail_energy(current):
            tail = build_atomic_tail(system.grid, current, charges, order)
            return 0.5 * self.volume * float(density @ lu.solve(tail.full(system.grid.size)))

        forces = self.helper.tail_forces(density, positions, charges)
        step = 1.0e-3
        for atom, axis in ((0, 0), (0, 2), (1, 1)):
            parts = []
            for sign in (1.0, -1.0):
                moved = np.array(positions, dtype=float)
                moved[atom, axis] += sign * step
                parts.append(tail_energy(moved))
            slope = -(parts[0] - parts[1]) / (2.0 * step)
            with self.subTest(atom=atom, axis=axis):
                self.assertLess(abs(forces[atom, axis] - slope), 1.0e-9 + 1.0e-6 * abs(slope))
        self.assertGreater(float(np.abs(forces).max()), 1.0e-7)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "slab"), ("auto", "auto"), (" ON ", "on"), ("off", "off"), ("Slab", "slab")):
            with _switch(value):
                self.assertEqual(hartree_variational_setting(), expected)
        with _unset():
            self.assertEqual(hartree_variational_setting(), "slab")
            # a cluster keeps the Poisson solution by default (slab), also where auto would take V'
            self.assertIsNone(prepare_single_point(_na2(forces_wanted=True)).hartree_variational)
        with _switch("bogus"), self.assertRaisesRegex(ValueError, HARTREE_VARIATIONAL_SWITCH):
            prepare_single_point(_na2())

    def test_where_it_applies(self) -> None:
        cases = (
            ("auto", _na2(forces_wanted=True), True),
            ("auto", _na2(forces_wanted=False), False),
            ("on", _na2(forces_wanted=False), True),
            ("off", _na2(forces_wanted=True), False),
            ("slab", _na2(forces_wanted=True), False),
        )
        for value, problem, applies in cases:
            with self.subTest(value=value, forces=problem.forces_wanted), _switch(value):
                system = prepare_single_point(problem)
                self.assertEqual(system.hartree_variational is not None, applies)
        parsec_boundary = _na2()
        parsec_boundary = replace(parsec_boundary, hartree=replace(parsec_boundary.hartree, boundary_tolerance=None))
        for value, applies in (("auto", False), ("on", True)):
            with self.subTest(value=value, boundary="PARSEC's"), _switch(value):
                self.assertEqual(prepare_single_point(parsec_boundary).hartree_variational is not None, applies)
        direct = _na2()
        direct = replace(direct, hartree=replace(direct.hartree, boundary_method="direct"))
        box = _na2(grid=GridSettings(spacing=0.7, radius=7.5, domain_shape="box", box_lengths=(14.0, 10.0, 10.0), expansion_order=6))
        for name, problem in (("direct", direct), ("box", box)):
            with self.subTest(name=name), _switch("on"):
                self.assertIsNone(prepare_single_point(problem).hartree_variational)

    def test_auto_decides_at_the_first_solve(self) -> None:
        """A small sphere takes V'; a large one keeps the Poisson solution, bit for bit that of ``off``."""

        with _switch("auto"):
            small = prepare_single_point(_na2())
            self.assertIsNone(small.hartree_variational.active)
            run_scf(small)
        self.assertTrue(small.hartree_variational.active)
        self.assertGreater(small.hartree_variational.estimate, small.hartree_variational.threshold)
        self.assertIn("on (estimated force change", small.hartree_variational.report())
        results = {}
        for value in ("auto", "off"):
            with _switch(value):
                system = prepare_single_point(_na2(radius=14.0))
                result = run_scf(system)
                results[value] = (system, result, compute_forces(system, result, symmetrize=False))
        helper = results["auto"][0].hartree_variational
        self.assertFalse(helper.active)
        self.assertLess(helper.estimate, helper.threshold)
        self.assertEqual(helper.solves, 1)
        self.assertEqual(results["auto"][1].energies.total, results["off"][1].energies.total)
        np.testing.assert_array_equal(results["auto"][2].unsymmetrized, results["off"][2].unsymmetrized)
        self.assertIsNone(results["auto"][2].hartree_boundary)

    def test_the_input_reader_records_forces(self) -> None:
        text = (DATA / "H2_parsec.in").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            (Path(directory) / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
            path.write_text(text, encoding="utf-8")
            self.assertFalse(parse_parsec_input(path).problem.forces_wanted)
            for extra, wanted in (
                ("Calculate_Force: .true.", True),
                ("Minimization: manual", True),
                # a manual movement computes forces, which Skip_Force says are not wanted
                ("Minimization: manual\nSkip_Force: true", False),
                # a relaxation computes them whatever Skip_Force says
                ("Minimization: bfgs\nSkip_Force: true", True),
            ):
                with self.subTest(extra=extra):
                    path.write_text(text + "\n" + extra + "\n", encoding="utf-8")
                    self.assertEqual(parse_parsec_input(path).problem.forces_wanted, wanted)


class DecisionTests(unittest.TestCase):
    """The decision of auto per structure, and the potential a result keeps."""

    def setUp(self) -> None:
        self.grid = build_cluster_grid(GridSettings(spacing=0.6, radius=5.0, expansion_order=6))
        self.operator = build_negative_laplacian(self.grid)
        self.settings = HartreeSettings(
            multipole_order=4, relative_tolerance=1.0e-12, absolute_tolerance=1.0e-16, max_iterations=6000
        )
        radius = np.linalg.norm(self.grid.coordinates, axis=1)
        self.density = np.exp(-0.3 * radius**2)
        self.result = solve_hartree(self.density, self.grid, self.operator, self.settings)
        self.estimates = []

    def helper(self, estimate):
        def estimator(density, residual, atoms):
            self.estimates.append(len(atoms))
            return np.full((len(atoms), 3), estimate)

        return VariationalHartree(
            self.grid, self.settings, ZeroBoundaryHostSolver(self.operator, self.grid), estimator=estimator
        )

    @staticmethod
    def atoms(shift):
        return (Atom("Na", [-1.5 - shift, 0.0, 0.0]), Atom("Na", [1.5, 0.0, 0.0]))

    def test_a_run_that_keeps_the_poisson_solution_decides_again_when_an_atom_moves_outward(self) -> None:
        helper = self.helper(0.0)
        self.assertIs(helper.potential(self.density, self.result, None, self.atoms(0.0)), self.result)
        self.assertFalse(helper.active)
        # less than one grid spacing (0.6 bohr) outward: the decision stands, nothing is solved
        self.assertIs(helper.potential(self.density, self.result, None, self.atoms(0.5)), self.result)
        self.assertEqual((len(helper.decisions), helper.solves), (1, 1))
        # more than one spacing from where it decided: a new decision
        self.assertIs(helper.potential(self.density, self.result, None, self.atoms(0.7)), self.result)
        self.assertEqual((len(helper.decisions), helper.solves), (2, 2))
        self.assertEqual(self.estimates, [2, 2])

    def test_a_result_keeps_the_poisson_potential(self) -> None:
        helper = self.helper(1.0)
        varied = helper.potential(self.density, self.result, None, self.atoms(0.0))
        self.assertTrue(helper.active)
        poisson = self.result.potential
        self.assertGreater(float(np.abs(varied.potential - poisson).max()), 1.0e-6)
        np.testing.assert_array_equal(helper.electrostatic(varied.potential), poisson)
        np.testing.assert_array_equal(helper.electrostatic(np.array(varied.potential)), poisson)
        # anything else is left as it is
        other = varied.potential + 1.0
        self.assertIs(helper.electrostatic(other), other)
        # a run that takes V' keeps it when atoms move inward or outward
        helper.potential(self.density, self.result, None, self.atoms(2.0))
        self.assertEqual(len(helper.decisions), 1)


class SlopeTests(unittest.TestCase):
    """The force of an atom against the slope of the SCF free energy at a step of 1e-3 bohr."""

    def slope_and_force(self, value):
        with _switch(value):
            system = prepare_single_point(_na2())
        positions = np.array([atom.position for atom in system.atoms], dtype=float)

        def at(current):
            return update_geometry(system, [replace(atom, position=p) for atom, p in zip(system.atoms, current)])

        def free_energy(result):
            f = np.clip(np.asarray(result.occupations, dtype=float).ravel(), 1.0e-300, 1.0 - 1.0e-16)
            return float(result.energies.total) + 2.0 * KT * 500.0 * float(np.sum(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)))

        start = at(positions)
        result = run_scf(start)
        force = compute_forces(start, result, symmetrize=False)
        step = 1.0e-3
        energies = []
        for sign in (1.0, -1.0):
            moved = positions.copy()
            moved[0, 0] += sign * step
            energies.append(free_energy(run_scf(at(moved))))
        self.result = result
        return -(energies[0] - energies[1]) / (2.0 * step), force

    def test_the_force_follows_the_energy(self) -> None:
        slope, force = self.slope_and_force("on")
        self.assertIsNotNone(force.hartree_boundary)
        self.assertLess(abs(force.unsymmetrized[0, 0] - slope), 1.0e-7)
        # the loop ran with V', the result keeps V_H of the Poisson solve
        result = self.result
        ran_with = result.output_effective_potential - result.ionic_potential - result.xc_potential
        self.assertGreater(float(np.abs(ran_with - result.hartree_potential).max()), 1.0e-6)
        slope_off, force_off = self.slope_and_force("off")
        result = self.result
        ran_with = result.output_effective_potential - result.ionic_potential - result.xc_potential
        np.testing.assert_allclose(ran_with, result.hartree_potential, rtol=0, atol=1.0e-10)
        self.assertIsNone(force_off.hartree_boundary)
        # the potential of the Poisson solve: 1.1e-6 Ry/bohr from the slope
        self.assertGreater(abs(force_off.unsymmetrized[0, 0] - slope_off), 5.0e-7)


if __name__ == "__main__":
    unittest.main()
