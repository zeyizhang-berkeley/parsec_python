"""Parity and work-counter tests for the optional native Poisson CG solver."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.Hartree.native_poisson import (
    NativePoissonSolver,
    solve_native_poisson,
)
from parsec_python.acceleration.backends.native import (
    NativeConjugateGradientBackend,
    native_available,
    native_build_info,
)
from parsec_python.Hartree.poisson import (
    HartreeResult,
    MultipoleExpansion,
    _conjugate_gradient,
)
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import prepare_single_point
from parsec_python.models import HartreeSettings


NATIVE_AVAILABLE = native_available()
DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _negative_laplacian_1d(size: int) -> sp.csr_matrix:
    """Small SPD Dirichlet operator with canonical CSR storage."""
    return sp.diags(
        (
            -np.ones(size - 1),
            2.0 * np.ones(size),
            -np.ones(size - 1),
        ),
        offsets=(-1, 0, 1),
        format="csr",
        dtype=np.float64,
    )


@unittest.skipUnless(
    NATIVE_AVAILABLE,
    "parsec_accelerated_native has not been built",
)
class NativeConjugateGradientTests(unittest.TestCase):
    def test_build_info_lists_native_cg_kernel(self) -> None:
        self.assertIn(
            "ConjugateGradientSolver",
            tuple(native_build_info()["implemented_kernels"]),
        )
        self.assertIn(
            "MultipoleBoundaryBuilder",
            tuple(native_build_info()["implemented_kernels"]),
        )

    def test_worker_team_scales_with_sparse_problem_size(self) -> None:
        size = 65_498
        solver = NativeConjugateGradientBackend(
            sp.eye(size, format="csr", dtype=np.float64)
        )
        configured_maximum = int(native_build_info()["openmp_max_threads"])

        self.assertEqual(
            solver.worker_count,
            min(configured_maximum, (size + 8191) // 8192),
        )

    def test_warm_start_at_solution_uses_only_initial_matvec(self) -> None:
        operator = _negative_laplacian_1d(17)
        exact = np.linspace(-0.7, 1.1, operator.shape[0])
        rhs = np.asarray(operator @ exact)
        settings = HartreeSettings(
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
        )

        result = NativePoissonSolver(operator).solve(rhs, exact, settings)

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.matrix_vector_products, 1)
        self.assertEqual(result.initial_residual_norm, 0.0)
        self.assertEqual(result.residual_norm, 0.0)
        np.testing.assert_array_equal(result.potential, exact)

    def test_chronological_guess_exactly_predicts_linear_rhs_sequence(self) -> None:
        """The two-step predictor changes only the CG starting vector."""

        operator = sp.eye(19, format="csr", dtype=np.float64)
        origin = np.linspace(-0.4, 0.7, operator.shape[0])
        direction = np.linspace(0.03, -0.02, operator.shape[0])
        solver = NativePoissonSolver(operator)
        settings = HartreeSettings(
            relative_tolerance=1.0e-13,
            absolute_tolerance=1.0e-15,
        )

        with patch.dict(
            os.environ,
            {"PARSEC_HARTREE_CHRONOLOGICAL_GUESS": "1"},
        ):
            solver.solve(origin, np.zeros_like(origin), settings)
            solver.solve(origin + direction, np.zeros_like(origin), settings)
            predicted = solver.solve(
                origin + 2.0 * direction,
                np.zeros_like(origin),
                settings,
            )

        self.assertEqual(solver.chronological_prediction_calls, 1)
        self.assertAlmostEqual(solver.last_chronological_alpha, 1.0, places=14)
        self.assertEqual(predicted.iterations, 0)
        np.testing.assert_allclose(
            predicted.potential,
            origin + 2.0 * direction,
            rtol=0.0,
            atol=2.0e-16,
        )

    def test_chronological_guess_can_be_disabled(self) -> None:
        operator = sp.eye(11, format="csr", dtype=np.float64)
        solver = NativePoissonSolver(operator)
        zeros = np.zeros(operator.shape[0])

        with patch.dict(
            os.environ,
            {"PARSEC_HARTREE_CHRONOLOGICAL_GUESS": "0"},
        ):
            for scale in (1.0, 2.0, 3.0):
                result = solver.solve(scale * np.ones_like(zeros), zeros)

        self.assertEqual(solver.chronological_prediction_calls, 0)
        self.assertIsNone(solver.last_chronological_alpha)
        self.assertEqual(result.iterations, 1)

    def test_solution_residual_and_counters_match_reference(self) -> None:
        operator = _negative_laplacian_1d(73)
        rng = np.random.default_rng(8814)
        rhs = rng.normal(size=operator.shape[0])
        initial = rng.normal(scale=0.1, size=operator.shape[0])
        settings = HartreeSettings(
            relative_tolerance=2.0e-10,
            absolute_tolerance=3.0e-14,
            max_iterations=200,
        )
        reference = _conjugate_gradient(operator, rhs, initial, settings)

        result = solve_native_poisson(operator, rhs, initial, settings)

        np.testing.assert_allclose(
            result.potential,
            reference[0],
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        self.assertEqual(result.converged, reference[1])
        self.assertEqual(result.iterations, reference[2])
        self.assertEqual(result.matrix_vector_products, reference[3])
        self.assertAlmostEqual(result.residual_norm, reference[4], places=11)
        self.assertAlmostEqual(
            result.initial_residual_norm, reference[5], places=13
        )
        self.assertAlmostEqual(
            result.residual_norm,
            float(np.linalg.norm(rhs - operator @ result.potential)),
            places=13,
        )

    def test_matvec_budget_policy_matches_reference(self) -> None:
        operator = _negative_laplacian_1d(41)
        rhs = np.linspace(-1.0, 2.0, operator.shape[0])
        initial = np.zeros_like(rhs)
        settings = HartreeSettings(
            relative_tolerance=1.0e-15,
            absolute_tolerance=0.0,
            max_iterations=3,
        )
        reference = _conjugate_gradient(operator, rhs, initial, settings)

        result = NativePoissonSolver(operator).solve(
            rhs,
            initial,
            settings,
            raise_on_nonconvergence=False,
        )

        self.assertFalse(result.converged)
        self.assertFalse(result.breakdown)
        self.assertEqual(result.iterations, reference[2])
        self.assertEqual(result.matrix_vector_products, reference[3])
        # initial A*x + two iteration A*p + explicit final A*x
        self.assertEqual(result.matrix_vector_products, 4)
        np.testing.assert_allclose(result.potential, reference[0], rtol=1e-14, atol=1e-14)

    def test_non_spd_breakdown_matches_reference(self) -> None:
        operator = -sp.eye(9, format="csr", dtype=np.float64)
        rhs = np.arange(1.0, 10.0)
        initial = np.zeros_like(rhs)
        settings = HartreeSettings(max_iterations=50)
        reference = _conjugate_gradient(operator, rhs, initial, settings)

        result = NativePoissonSolver(operator).solve(
            rhs,
            initial,
            settings,
            raise_on_nonconvergence=False,
        )

        self.assertFalse(result.converged)
        self.assertTrue(result.breakdown)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.matrix_vector_products, 3)
        self.assertEqual(result.matrix_vector_products, reference[3])
        np.testing.assert_array_equal(result.potential, reference[0])
        self.assertEqual(result.residual_norm, reference[4])

    def test_low_level_backend_and_scalar_controls_are_modular(self) -> None:
        operator = _negative_laplacian_1d(25)
        exact = np.sin(np.linspace(0.0, np.pi, operator.shape[0]))
        rhs = np.asarray(operator @ exact)
        backend = NativeConjugateGradientBackend(operator)

        payload = backend.solve(
            rhs,
            np.zeros_like(rhs),
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100,
        )
        result = NativePoissonSolver(operator).solve(
            rhs,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100,
        )

        np.testing.assert_allclose(payload["solution"], exact, atol=2.0e-12)
        np.testing.assert_allclose(result.potential, exact, atol=2.0e-12)

    def test_nonconvergence_raises_by_default_and_can_be_inspected(self) -> None:
        operator = _negative_laplacian_1d(13)
        rhs = np.ones(operator.shape[0])
        solver = NativePoissonSolver(operator)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            solver.solve(rhs, max_iterations=1)
        result = solver.solve(
            rhs,
            max_iterations=1,
            raise_on_nonconvergence=False,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.matrix_vector_products, 2)

    def test_result_converts_to_reference_hartree_result(self) -> None:
        operator = _negative_laplacian_1d(7)
        exact = np.arange(7.0)
        rhs = np.asarray(operator @ exact)
        native = NativePoissonSolver(operator).solve(rhs, exact)
        boundary = MultipoleExpansion(order=0, moments={(0, 0): 0.0j})

        result = native.as_hartree_result(boundary)

        self.assertIsInstance(result, HartreeResult)
        self.assertIs(result.boundary, boundary)
        self.assertIs(result.potential, native.potential)
        self.assertIs(result.right_hand_side, native.right_hand_side)

    def test_physical_boundary_corrected_rhs_matches_python_solve(self) -> None:
        problem = parse_parsec_input(DATA / "H_cli_smoke.in").problem
        system = prepare_single_point(problem)
        initial = -system.ionic_potential
        reference = system.solve_hartree(system.initial_density, initial)

        native = NativePoissonSolver(system.negative_laplacian).solve(
            reference.right_hand_side,
            initial,
            system.input.hartree,
        )

        self.assertTrue(native.converged)
        np.testing.assert_allclose(
            native.potential,
            reference.potential,
            rtol=3.0e-11,
            atol=3.0e-11,
        )
        self.assertLessEqual(native.residual_norm, native.tolerance * 1.05)
        self.assertEqual(
            native.matrix_vector_products,
            reference.matrix_vector_products,
        )


if __name__ == "__main__":
    unittest.main()
