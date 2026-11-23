"""The Hartree potential of a slab as the derivative of its Hartree energy (``PARSEC_HARTREE_VARIATIONAL``).

The slab solve ``V_H = G rho`` takes the continuum potential of the grid planes as its boundary values; ``G`` is not
symmetric, so ``V_H`` is not the derivative of ``(h^3/2) rho.G rho`` and the SCF does not make the energy stationary.
Checked here (CPU, a hydrogen slab whose density reaches the edge of the domain): the transposes of the sheet sums and
of ``G``, the energy identity ``rho.V' = rho.V_H``, where the potential applies, and the force against the slope of
the SCF free energy with and without it.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.Forces.wire_slab import compute_wire_slab_forces
from parsec_python.Hartree.confined import solve_confined_hartree
from parsec_python.Hartree.variational import HARTREE_VARIATIONAL_SWITCH, ConfinedVariationalHartree
from parsec_python.models import Atom, HartreeSettings
from parsec_python.Output.force_output import force_report_lines
from parsec_python.SCF import run_scf
from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point, update_wire_slab_geometry
from parsec_python.tests.unit.test_wire_slab_ionic import hydrogen_problem

KT = 6.33327186e-06
TIGHT = HartreeSettings(relative_tolerance=1.0e-14, absolute_tolerance=1.0e-16, max_iterations=40000)


def _problem(periodicity=2, *, forces_wanted=True, radius=3.0):
    atoms = [Atom("H", (0.31, 0.12, 0.66)), Atom("H", (-0.22, 0.05, -0.71))]
    problem = hydrogen_problem(periodicity, atoms, radius=radius)
    species = {symbol: replace(value, use_spline=True) for symbol, value in problem.pseudopotentials.items()}
    return replace(
        problem,
        pseudopotentials=species,
        forces_wanted=forces_wanted,
        scf=replace(problem.scf, convergence_criterion=1.0e-11, max_iterations=200, fermi_temperature_kelvin=500.0),
    )


def _switch(value):
    return mock.patch.dict(os.environ, {HARTREE_VARIATIONAL_SWITCH: value})


class TransposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with _switch("off"):
            cls.system = prepare_wire_slab_single_point(_problem())

    def test_the_transpose_of_the_sheet_sums(self) -> None:
        plan = self.system.hartree_plan
        rng = np.random.default_rng(3)
        density = rng.standard_normal(self.system.grid.size)
        values = rng.standard_normal(plan.stencil.size)
        first = float(values @ plan.boundary.boundary_values(density))
        second = float(plan.boundary.boundary_transpose(values) @ density)
        self.assertLess(abs(first - second), 1.0e-12 * abs(first))

    def test_the_transpose_of_the_hartree_operator_and_the_energy(self) -> None:
        system = self.system
        coordinates = system.grid.coordinates
        rho = np.asarray(system.initial_density)
        other = np.exp(-np.sum((coordinates - np.array([0.4, -0.3, 1.8])) ** 2, axis=1))

        def solve(values):
            return solve_confined_hartree(values, system.hartree_plan, system.negative_laplacian, TIGHT, None,
                                          raise_on_nonconvergence=False).potential

        helper = ConfinedVariationalHartree(system.hartree_plan, system.negative_laplacian, TIGHT)
        transposed = helper.transpose(rho)
        reference = float(rho @ solve(other))
        self.assertLess(abs(float(transposed @ other) - reference), 1.0e-11 * abs(reference))
        # G itself is not symmetric where the density reaches the edge
        self.assertGreater(abs(float(other @ solve(rho)) - reference), 1.0e-8 * abs(reference))
        poisson = solve(rho)
        corrected = 0.5 * (poisson + transposed)
        self.assertLess(abs(float(rho @ (corrected - poisson))), 1.0e-11 * abs(float(rho @ poisson)))


class SwitchTests(unittest.TestCase):
    def test_where_it_applies(self) -> None:
        with _switch("off"):
            self.assertIsNone(prepare_wire_slab_single_point(_problem()).hartree_variational)
        for value in ("on", "auto", "slab", ""):
            with _switch(value):
                self.assertIsInstance(prepare_wire_slab_single_point(_problem()).hartree_variational,
                                      ConfinedVariationalHartree)
                self.assertIsNone(prepare_wire_slab_single_point(_problem(forces_wanted=False)).hartree_variational)
                self.assertIsNone(prepare_wire_slab_single_point(_problem(1)).hartree_variational)
        with _switch("sometimes"):
            with self.assertRaisesRegex(ValueError, "PARSEC_HARTREE_VARIATIONAL"):
                prepare_wire_slab_single_point(_problem())

    def test_the_force_report_says_so(self) -> None:
        for value in ("on", "off"):
            with _switch(value):
                system = prepare_wire_slab_single_point(_problem())
                forces = compute_wire_slab_forces(system, run_scf(system))
            lines = [line for line in force_report_lines(forces, periodic=True) if "Hartree potential of the SCF" in line]
            with self.subTest(value=value):
                if value == "on":
                    self.assertIn("solves of A u = rho", forces.hartree_variational)
                    self.assertEqual(len(lines), 1)
                else:
                    self.assertIsNone(forces.hartree_variational)
                    self.assertEqual(lines, [])


class SlopeTests(unittest.TestCase):
    """The force of an atom along z against the slope of the SCF free energy: the Richardson value of the central
    differences at 1e-3 and 2e-3 bohr (each alone carries an h^2 term of 1.8e-6 and 7.3e-6 Ry/bohr here)."""

    def slope_and_force(self, value):
        with _switch(value):
            system = prepare_wire_slab_single_point(_problem())
            result = run_scf(system)
            forces = compute_wire_slab_forces(system, result, symmetrize=False)
            slopes = []
            for step in (1.0e-3, 2.0e-3):
                energies = []
                for sign in (1.0, -1.0):
                    atoms = list(system.input.atoms)
                    position = np.array(atoms[0].position, dtype=float)
                    position[2] += sign * step
                    atoms[0] = replace(atoms[0], position=tuple(position))
                    moved = run_scf(update_wire_slab_geometry(system, atoms))
                    f = np.clip(np.asarray(moved.occupations, dtype=float).ravel(), 1.0e-300, 1.0 - 1.0e-16)
                    entropy = float(np.sum(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)))
                    energies.append(float(moved.energies.total) + 2.0 * KT * 500.0 * entropy)
                slopes.append(-(energies[0] - energies[1]) / (2.0 * step))
        force = float(forces.unsymmetrized[0, 2])
        return (4.0 * slopes[0] - slopes[1]) / 3.0, force, result

    def test_the_force_follows_the_energy(self) -> None:
        slope, force, result = self.slope_and_force("on")
        self.assertLess(abs(force - slope), 1.0e-7)
        # the loop ran with V', the result keeps V_H of the Poisson solve
        ran_with = result.output_effective_potential - result.ionic_potential - result.xc_potential
        self.assertGreater(float(np.abs(ran_with - result.hartree_potential).max()), 1.0e-6)
        slope_off, force_off, result_off = self.slope_and_force("off")
        ran_with = result_off.output_effective_potential - result_off.ionic_potential - result_off.xc_potential
        np.testing.assert_allclose(ran_with, result_off.hartree_potential, rtol=0, atol=1.0e-10)
        # the potential of the Poisson solve: 8.5e-5 Ry/bohr from the slope
        self.assertGreater(abs(force_off - slope_off), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
