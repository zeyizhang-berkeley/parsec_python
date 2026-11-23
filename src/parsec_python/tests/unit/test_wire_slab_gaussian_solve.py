"""The potential of the Gaussian charges of a wire or slab is solved tightly (``PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE``).

``V_G`` is solved once per structure from ``n_g`` (``ionpot_slab.f90``'s ``hpotcg``); its error enters the energy at first
order and moves with the atoms, so the energy over small steps carries its slope (7.4e-5 Ry/bohr in F_z of B in an
hBN slab at the Hartree tolerance of a run).  Checked here (CPU, a hydrogen slab and wire): ``off`` keeps the solve of
the run's settings bit for bit, ``auto`` reaches a relative residual of 1e-12, and at a fixed density the force of the
Gaussian charges in its analytic form (``PARSEC_WIRE_SLAB_GAUSSIAN_FORCE=auto``) is the central difference of
``-dV rho . V_G`` of the tight solves, while the finite-difference gradient form is not at h 0.6 bohr.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from parsec_python.Forces.wire_slab import wire_slab_local_forces
from parsec_python.Hartree.confined import solve_confined_hartree
from parsec_python.models import Atom, HartreeSettings
from parsec_python.SCF import wire_slab as wire_slab_scf
from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point, update_wire_slab_geometry
from parsec_python.tests.unit.test_wire_slab_ionic import hydrogen_problem

_ENV = wire_slab_scf.GAUSSIAN_SOLVE_ENV
_RUN = HartreeSettings(relative_tolerance=1.0e-7)


def _problem(periodicity):
    atoms = [Atom("H", (0.31, 0.12, 0.66)), Atom("H", (-0.22, 0.05, -0.71))]
    return replace(hydrogen_problem(periodicity, atoms), hartree=_RUN)


def _residual(system):
    rhs, _ = system.hartree_plan.right_hand_side(np.asarray(system.short_range.gaussian_density))
    start = system.negative_laplacian @ np.asarray(system.short_range.gaussian_density) - rhs
    final = system.negative_laplacian @ np.asarray(system.gaussian_potential) - rhs
    return float(np.linalg.norm(final) / np.linalg.norm(start))


class GaussianSolveTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertEqual(wire_slab_scf.gaussian_solve_setting(), expected)
        with mock.patch.dict(os.environ, {_ENV: "tight"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE must be auto or off"):
                prepare_wire_slab_single_point(_problem(2))

    def test_off_keeps_the_solve_of_the_run_and_auto_is_tight(self) -> None:
        for periodicity in (1, 2):
            problem = _problem(periodicity)
            with mock.patch.dict(os.environ, {_ENV: "off"}):
                former = prepare_wire_slab_single_point(problem)
            with mock.patch.dict(os.environ, {_ENV: "auto"}):
                tight = prepare_wire_slab_single_point(problem)
            direct = solve_confined_hartree(
                former.short_range.gaussian_density, former.hartree_plan, former.negative_laplacian, problem.hartree,
                former.short_range.gaussian_density,
            ).potential
            with self.subTest(periodicity=periodicity):
                np.testing.assert_array_equal(former.gaussian_potential, direct)
                self.assertLess(_residual(tight), 2.0e-12)
                self.assertGreater(_residual(former), 1.0e-9)

    def test_the_gaussian_force_is_the_slope_of_the_energy_of_the_tight_solve(self) -> None:
        problem = _problem(2)
        with mock.patch.dict(os.environ, {_ENV: "auto"}):
            system = prepare_wire_slab_single_point(problem)
            density = np.asarray(system.initial_density)
            tight = HartreeSettings(relative_tolerance=1.0e-13, absolute_tolerance=1.0e-15, max_iterations=20000)
            hartree = solve_confined_hartree(density, system.hartree_plan, system.negative_laplacian, tight).potential
            gradient = np.vstack([matrix @ hartree for matrix in system.gradient])
            zero = np.zeros(system.grid.size)
            arguments = (system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials, density, zero, zero)
            terms = wire_slab_local_forces(*arguments, gradient, hartree_potential=hartree)
            with mock.patch.dict(os.environ, {"PARSEC_WIRE_SLAB_GAUSSIAN_FORCE": "gradient"}):
                former = wire_slab_local_forces(*arguments, gradient, hartree_potential=hartree)
            step = 2.0e-3
            dv = system.grid.volume_element
            base = np.array([atom.position for atom in system.atoms], dtype=float)
            for axis in (0, 2):
                values = []
                for sign in (1.0, -1.0):
                    moved = base.copy()
                    moved[0, axis] += sign * step
                    other = update_wire_slab_geometry(
                        system, [replace(atom, position=p) for atom, p in zip(system.atoms, moved)]
                    )
                    values.append(-dv * float(density @ np.asarray(other.gaussian_potential)))
                slope = -(values[0] - values[1]) / (2.0 * step)
                with self.subTest(axis=axis):
                    # the analytic slope of the Gaussian (PARSEC_WIRE_SLAB_GAUSSIAN_FORCE=auto); the finite-difference
                    # gradient of the potential does not resolve the Gaussian at h 0.6 (2.2e-3 Ry/bohr off)
                    self.assertAlmostEqual(float(terms.gaussian[0, axis]), slope, delta=2.0e-6)
                    self.assertGreater(abs(float(former.gaussian[0, axis]) - slope), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
