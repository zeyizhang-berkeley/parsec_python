"""Parity tests for recurrence-based Hartree multipoles."""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.acceleration.Hartree.fast_multipole import (
    FastMultipoleExpansion,
    density_multipoles_fast,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import density_multipoles
from parsec_python.models import GridSettings


class FastMultipoleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(
            GridSettings(spacing=0.7, radius=3.2, expansion_order=4)
        )
        coordinates = self.grid.coordinates
        density = np.exp(-0.8 * np.sum(coordinates * coordinates, axis=1))
        density *= 2.0 / self.grid.integrate(density)
        self.density = density * (
            1.0 + 0.08 * coordinates[:, 0] - 0.03 * coordinates[:, 1]
        )

    def test_moments_and_boundary_values_match_scipy_harmonics(self) -> None:
        reference = density_multipoles(self.density, self.grid, order=9)
        accelerated = density_multipoles_fast(self.density, self.grid, order=9)
        self.assertIsInstance(accelerated, FastMultipoleExpansion)
        for key, expected in reference.moments.items():
            self.assertAlmostEqual(accelerated.moments[key].real, expected.real, 11)
            self.assertAlmostEqual(accelerated.moments[key].imag, expected.imag, 11)

        points = np.asarray(
            [
                (4.1, 0.3, -0.2),
                (-0.4, 4.3, 0.7),
                (0.2, -0.6, 4.5),
                (-3.8, -2.1, 1.4),
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            accelerated.potential(points),
            reference.potential(points),
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_validation_matches_reference_contract(self) -> None:
        with self.assertRaises(ValueError):
            density_multipoles_fast(self.density[:-1], self.grid)
        with self.assertRaises(ValueError):
            density_multipoles_fast(self.density, self.grid, order=-1)


if __name__ == "__main__":
    unittest.main()
