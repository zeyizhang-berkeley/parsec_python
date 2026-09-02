from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.V_xc import pbe, pbe_energy_partials
from parsec_python.models import GridSettings


BENCHMARK = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "0_CH4_CF4"
    / "python_pbe"
)


class PBEFunctionalTests(unittest.TestCase):
    def test_pointwise_partials_match_energy_finite_differences(self) -> None:
        density = np.array([1.0e-5, 0.01, 0.1, 1.0])
        sigma = np.array([1.0e-8, 1.0e-4, 0.03, 0.7])
        _energy, derivative_density, derivative_sigma = pbe_energy_partials(
            density, sigma
        )
        for index in range(density.size):
            dn = max(1.0e-10, density[index] * 1.0e-6)
            ds = max(1.0e-12, sigma[index] * 1.0e-6)
            plus = density.copy()
            minus = density.copy()
            plus[index] += dn
            minus[index] -= dn
            finite_n = (
                pbe_energy_partials(plus, sigma)[0][index]
                - pbe_energy_partials(minus, sigma)[0][index]
            ) / (2.0 * dn)
            plus_sigma = sigma.copy()
            minus_sigma = sigma.copy()
            plus_sigma[index] += ds
            minus_sigma[index] -= ds
            finite_sigma = (
                pbe_energy_partials(density, plus_sigma)[0][index]
                - pbe_energy_partials(density, minus_sigma)[0][index]
            ) / (2.0 * ds)
            self.assertAlmostEqual(finite_n, derivative_density[index], places=8)
            self.assertAlmostEqual(
                finite_sigma, derivative_sigma[index], delta=3.0e-8
            )

    def test_grid_potential_is_discrete_energy_derivative(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.7,
                radius=3.2,
                expansion_order=8,
                shift=(0.5, 0.5, 0.5),
            )
        )
        radius_squared = np.einsum(
            "ij,ij->i", grid.coordinates, grid.coordinates
        )
        density = 0.2 * np.exp(-0.5 * radius_squared) + 0.01
        direction = np.random.default_rng(2).normal(size=grid.size)
        direction /= np.linalg.norm(direction)
        result = pbe(density, grid)
        step = 1.0e-5
        finite_difference = (
            pbe(density + step * direction, grid).total_energy
            - pbe(density - step * direction, grid).total_energy
        ) / (2.0 * step)
        variational_derivative = (
            grid.volume_element * np.dot(result.potential, direction)
        )
        self.assertAlmostEqual(
            finite_difference, variational_derivative, delta=1.0e-8
        )


class CoreHoleInputTests(unittest.TestCase):
    def test_c_1s_species_keeps_physical_element_and_electron_count_data(self) -> None:
        translation = parse_parsec_input(BENCHMARK / "CH4" / "FS_1s" / "parsec.in")
        problem = translation.problem
        self.assertEqual(problem.scf.xc_functional, "pbe")
        self.assertEqual(problem.scf.net_charge, 1.0)
        carbon = problem.pseudopotentials["C-1s"]
        self.assertEqual(carbon.element_symbol, "C")
        self.assertAlmostEqual(carbon.atomic_energy_correction, -35.46960)
        potentials = load_pseudopotentials(
            problem.pseudopotentials,
            xc_functional=problem.scf.xc_functional,
        )
        self.assertAlmostEqual(potentials["C-1s"].ionic_charge, 5.0)
        self.assertEqual(potentials["C-1s"].symbol, "C")

    def test_ares_grid_counts_are_reproduced(self) -> None:
        for molecule, expected_shape, expected_active in (
            ("CH4", (102, 102, 102), 539_152),
            ("CF4", (182, 182, 182), 3_103_688),
        ):
            problem = parse_parsec_input(
                BENCHMARK / molecule / "IS" / "parsec.in"
            ).problem
            grid = build_cluster_grid(problem.grid)
            self.assertEqual(grid.shape, expected_shape)
            self.assertEqual(grid.size, expected_active)


if __name__ == "__main__":
    unittest.main()
