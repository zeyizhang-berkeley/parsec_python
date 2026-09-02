"""Public-API checks for the concept-package layout."""

from __future__ import annotations

import unittest

import parsec_python as public_api
from parsec_python.Energy import total_energy
from parsec_python.Grid import RealSpaceGrid, build_cluster_grid
from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.Hartree import solve_hartree
from parsec_python.Laplacian import (
    apply_negative_laplacian_boundary,
    build_negative_laplacian,
    neighbor_shells,
    second_derivative_coefficients,
)
from parsec_python.Mixer import (
    AndersonMixer,
    ResidualMetrics,
    potential_residual_metrics,
)
from parsec_python.Occupations import (
    BOLTZMANN_RYDBERG_PER_KELVIN,
    OccupationResult,
    density_from_orbitals,
    fermi_occupations,
)
from parsec_python.Pseudopotential import (
    ParsecRadialSpline,
    parsec_radial_integral,
    read_parsec_pseudopotential,
)
from parsec_python.V_ion import build_nonlocal_projectors
from parsec_python.V_xc import XCResult, ca_lda


class ConceptKernelImportTests(unittest.TestCase):
    def test_grid_and_laplacian_public_api_uses_concept_implementations(
        self,
    ) -> None:
        self.assertIs(public_api.RealSpaceGrid, RealSpaceGrid)
        self.assertIs(public_api.build_cluster_grid, build_cluster_grid)
        self.assertIs(
            public_api.apply_negative_laplacian_boundary,
            apply_negative_laplacian_boundary,
        )
        self.assertIs(
            public_api.build_negative_laplacian,
            build_negative_laplacian,
        )
        self.assertIs(
            public_api.second_derivative_coefficients,
            second_derivative_coefficients,
        )
        self.assertEqual(
            neighbor_shells.__module__,
            "parsec_python.Laplacian.finite_difference",
        )

    def test_xc_occupation_mixer_and_energy_public_api(self) -> None:
        self.assertIs(public_api.XCResult, XCResult)
        self.assertIs(public_api.ca_lda, ca_lda)
        self.assertEqual(
            public_api.BOLTZMANN_RYDBERG_PER_KELVIN,
            BOLTZMANN_RYDBERG_PER_KELVIN,
        )
        self.assertIs(public_api.OccupationResult, OccupationResult)
        self.assertIs(public_api.density_from_orbitals, density_from_orbitals)
        self.assertIs(public_api.fermi_occupations, fermi_occupations)
        self.assertIs(public_api.AndersonMixer, AndersonMixer)
        self.assertIs(public_api.ResidualMetrics, ResidualMetrics)
        self.assertIs(
            public_api.potential_residual_metrics,
            potential_residual_metrics,
        )
        self.assertIs(public_api.total_energy, total_energy)

    def test_potential_and_hamiltonian_public_api(self) -> None:
        self.assertIs(
            public_api.read_parsec_pseudopotential,
            read_parsec_pseudopotential,
        )
        self.assertIs(public_api.ParsecRadialSpline, ParsecRadialSpline)
        self.assertIs(
            public_api.parsec_radial_integral,
            parsec_radial_integral,
        )
        self.assertIs(
            public_api.build_nonlocal_projectors,
            build_nonlocal_projectors,
        )
        self.assertIs(public_api.solve_hartree, solve_hartree)
        self.assertIs(
            public_api.KohnShamHamiltonian,
            KohnShamHamiltonian,
        )


if __name__ == "__main__":
    unittest.main()
