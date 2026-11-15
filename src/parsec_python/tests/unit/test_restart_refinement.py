"""Checkpoint reconstruction must finish before an SCF density can be used."""
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.SCF.restart_refinement import (
    RestartRefinementError, RestartRefinementSettings, from_environment,
    guarded_solver, refine, refine_kpoints,
)


class RestartRefinementTests(unittest.TestCase):
    def test_filter_policy_is_explicit_and_validated(self):
        with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "levels",
                                     "PARSEC_RESTART_REFINE_LOWER_BOUND": "smart",
                                     "PARSEC_RESTART_REFINE_MAX_PASSES": "40"}):
            policy = from_environment(SimpleNamespace(tolerance=1e-4))
        self.assertEqual(policy.lower_bound_rule, "smart")
        self.assertEqual(policy.tolerance, 1e-4)
        self.assertEqual(policy.max_passes, 40)  # Explicit historical limits remain valid.
        with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "levels"}, clear=True):
            default = from_environment(SimpleNamespace(tolerance=1e-4))
        self.assertEqual(default, RestartRefinementSettings(1e-4))
        self.assertEqual(default.max_passes, 100)
        self.assertEqual(default.stable_passes, 2)
        self.assertEqual(default.lower_bound_rule, "inherit")
        with self.assertRaisesRegex(ValueError, "inherit or smart"):
            RestartRefinementSettings(1e-4, lower_bound_rule="unknown")

    def test_disabled_preserves_callable_and_ignores_debug(self):
        solve = object()
        self.assertIs(guarded_solver(solve, None), solve)
        with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "off", "PARSEC_DEBUG": "audit"}):
            self.assertIsNone(from_environment(SimpleNamespace(tolerance=1e-4)))
        with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "levels", "PARSEC_DEBUG": "off",
                                     "PARSEC_RESTART_REFINE_MAX_PASSES": "7"}):
            self.assertEqual(from_environment(SimpleNamespace(tolerance=1e-6)),
                             RestartRefinementSettings(1e-6, max_passes=7))

    def test_in_place_spectrum_is_snapshotted_and_needs_consecutive_passes(self):
        values = np.array([2.])
        sequence = iter([1., 1.0001, .5, .5001, .5002])
        def step():
            values[0] = next(sequence)
            return (values,)
        report = refine(step, (values,), RestartRefinementSettings(.001, 5), label="test")
        self.assertEqual(report["passes"], 5)

    def test_bad_spectra_fail_closed(self):
        policy = RestartRefinementSettings(.001, 2)
        for result in [(np.array([np.nan]),), (np.zeros(2),), ()]:
            with self.subTest(result=result), self.assertRaises(RestartRefinementError):
                refine(lambda: result, (np.zeros(1),), policy, label="bad")
        with self.assertRaisesRegex(RestartRefinementError, "no new density"):
            values = iter([(np.ones(1),), (np.zeros(1),)])
            refine(lambda: next(values), (np.zeros(1),), policy, label="limit")

    def test_guard_releases_wrappers_and_reuses_identical_hamiltonian(self):
        calls, releases = [], []
        operator, settings = object(), object()
        def solve(op, count, *, settings, state):
            calls.append((op, count, settings, state))
            index = len(calls)
            vectors = SimpleNamespace(release_intermediate_storage=lambda: releases.append(index))
            return SimpleNamespace(eigenvalues=np.array([1. / index if index < 3 else 0.]),
                                   state=index, vectors=vectors)
        guarded = guarded_solver(solve, RestartRefinementSettings(.001, 6))
        result = guarded(operator, 1, settings=settings)
        self.assertEqual(len(calls), 5)
        self.assertEqual(releases, [1, 2, 3, 4])
        self.assertTrue(all(op is operator and config is settings for op, _, config, _ in calls))
        guarded(operator, 1, settings=settings, state=result.state)
        self.assertEqual(len(calls), 6)  # Later SCF step: exactly one ordinary call.

    def test_real_chebff_reconstruction_improves_against_dense_eigenpairs(self):
        from scipy.sparse.linalg import aslinearoperator
        from parsec_python.Eigensolvers import EigvalSettings, ChebFFSettings, SubspaceSettings, solve_eigval
        diagonal = np.linspace(-2., 15., 64)
        operator = aslinearoperator(np.diag(diagonal))
        settings = EigvalSettings(safety_buffer=4,
            chebff=ChebFFSettings(filter_cycles=1, polynomial_degree=2, block_size=4),
            subspace=SubspaceSettings(polynomial_degree=12, block_size=4))
        original = solve_eigval(operator, 4, settings=settings)
        rebuilt = guarded_solver(solve_eigval, RestartRefinementSettings(1e-10))(
            operator, 4, settings=settings)
        self.assertGreater(np.max(np.abs(original.eigenvalues - diagonal[:4])), .01)
        np.testing.assert_allclose(rebuilt.eigenvalues, diagonal[:4], rtol=0, atol=1e-8)
        residuals = np.linalg.norm(operator @ rebuilt.vectors - rebuilt.vectors * rebuilt.eigenvalues, axis=0)
        self.assertLess(float(np.max(residuals)), 2e-5)

    def test_smart_bound_reduces_work_on_a_shifted_spectrum_at_same_accuracy(self):
        from scipy.sparse.linalg import aslinearoperator
        from parsec_python.Eigensolvers import EigvalSettings, ChebFFSettings, SubspaceSettings, solve_eigval
        # A rigid potential shift changes eigenvalues, not eigenvectors. The
        # zero-clamped monotone bound should not be needed to preserve accuracy.
        diagonal = np.r_[np.linspace(-2., -.2, 12), np.linspace(-.05, 156., 116)] - 15.
        operator = aslinearoperator(np.diag(diagonal))
        settings = EigvalSettings(safety_buffer=4,
            chebff=ChebFFSettings(filter_cycles=3, polynomial_degree=15, block_size=4),
            subspace=SubspaceSettings(polynomial_degree=15, block_size=4))
        passes = {}
        for rule in ("inherit", "smart"):
            reports = []
            def record(event, **fields):
                if event == "restart_level_refinement":
                    reports.append(fields)
            with patch("parsec_python.SCF.restart_refinement.debug.emit", side_effect=record):
                result = guarded_solver(solve_eigval, RestartRefinementSettings(
                    1e-9, 80, lower_bound_rule=rule))(operator, 8, settings=settings)
            np.testing.assert_allclose(result.eigenvalues, diagonal[:8], rtol=0, atol=2e-8)
            residual = np.linalg.norm(operator @ result.vectors - result.vectors * result.eigenvalues, axis=0)
            self.assertLess(float(np.max(residual)), 5e-5)
            self.assertTrue(reports[-1]["passed"])
            passes[rule] = reports[-1]["passes"]
        self.assertLess(passes["smart"], passes["inherit"] / 2)
        self.assertEqual(settings.subspace.lower_bound_rule, "monotone")

    def test_override_is_limited_to_extra_scalar_passes(self):
        from parsec_python.Eigensolvers import EigvalSettings
        settings, operator = EigvalSettings(), object()
        calls = []
        def solve(op, count, *, settings, state):
            self.assertIs(op, operator)
            calls.append(settings)
            return SimpleNamespace(eigenvalues=np.zeros(1), state=len(calls), vectors=object())
        guarded = guarded_solver(solve, RestartRefinementSettings(.001, 2, lower_bound_rule="smart"))
        result = guarded(operator, 1, settings=settings)
        self.assertIs(calls[0], settings)  # Original initial CHEBFF.
        self.assertEqual([x.subspace.lower_bound_rule for x in calls[1:]], ["smart", "smart"])
        self.assertTrue(all(x.chebff is settings.chebff for x in calls))
        self.assertTrue(all(x.subspace.polynomial_degree == settings.subspace.polynomial_degree for x in calls))
        guarded(operator, 1, settings=settings, state=result.state)
        self.assertIs(calls[-1], settings)  # Ordinary later SCF uses original policy.

    def test_kpoint_gate_checks_all_tasks_and_keeps_potential(self):
        potentials = {"V": np.array([1., 2.])}
        calls = []
        outputs = iter([[np.array([0.]), np.array([1.])],
                        [np.array([0.]), np.array([.1])],
                        [np.array([0.]), np.array([.1])],
                        [np.array([0.]), np.array([.1])]])
        class Solver:
            def solve(self, p, degree, tolerance):
                self_test.assertIs(p, potentials)
                calls.append((degree, tolerance))
                return next(outputs)
        self_test = self
        result = refine_kpoints(Solver(), potentials, 15, None,
                                [np.array([0.]), np.array([2.])], RestartRefinementSettings(.001))
        self.assertEqual(len(calls), 4)
        np.testing.assert_array_equal(result[1], [.1])

    def test_kpoint_override_restores_factory_on_success_and_failure(self):
        from parsec_python.Eigensolvers import EigvalSettings
        from parsec_python.SCF.kpoint_solver import job_eigval_settings
        settings, potentials = EigvalSettings(), {"V": np.array([1., 2.])}
        for fail in (False, True):
            with self.subTest(failure=fail):
                factory = lambda degree, tolerance=None: settings
                job = SimpleNamespace(eigval_settings=factory)
                calls = []
                owner = self
                class Solver:
                    def solve(self, p, degree, tolerance):
                        owner.assertIs(p, potentials)
                        configured = job_eigval_settings(self.job, degree, tolerance)
                        owner.assertEqual(configured.subspace.lower_bound_rule, "smart")
                        calls.append((degree, tolerance))
                        return [np.array([float(len(calls)) if fail else 0.])] * 2
                solver = Solver(); solver.job = job
                policy = RestartRefinementSettings(.001, 2, lower_bound_rule="smart")
                if fail:
                    with self.assertRaises(RestartRefinementError):
                        refine_kpoints(solver, potentials, 15, 1e-5, [np.zeros(1)] * 2, policy)
                else:
                    refine_kpoints(solver, potentials, 15, 1e-5, [np.zeros(1)] * 2, policy)
                self.assertIs(job.eigval_settings, factory)
                self.assertEqual(calls, [(15, 1e-5)] * 2)
                self.assertIs(job_eigval_settings(job, 15, 1e-5), settings)

    def test_unsupported_kpoint_override_fails_before_extra_solves(self):
        solver = SimpleNamespace(solve=lambda *args: self.fail("unsupported solver was called"))
        with self.assertRaisesRegex(TypeError, "KPointJob"):
            refine_kpoints(solver, {}, 15, None, [np.zeros(1)],
                           RestartRefinementSettings(.001, lower_bound_rule="smart"))

    def test_reconstruction_failure_prevents_scf_callbacks(self):
        from parsec_python.tests.unit.test_strict_scf_eigensolver import StrictSCFEigensolverTests
        from parsec_python.SCF import prepare_single_point, run_scf
        from parsec_python.SCF.single_point import SCFWarmStart
        system = prepare_single_point(StrictSCFEigensolverTests()._problem())
        callbacks = []
        warm = SCFWarmStart(input_potential=system.ionic_potential.copy(),
                            hartree_potential=-system.ionic_potential,
                            restart_refinement=RestartRefinementSettings(1e-30, 2))
        with self.assertRaises(RestartRefinementError):
            run_scf(system, warm_start=warm, iteration_state_callback=callbacks.append)
        self.assertEqual(callbacks, [])

    def test_checkpoint_loading_uses_explicit_policy_only(self):
        import tempfile
        from pathlib import Path
        from parsec_python.tests.unit.test_strict_scf_eigensolver import StrictSCFEigensolverTests
        from parsec_python.SCF import prepare_single_point, run_scf
        from parsec_python.SCF.restart import load_restart
        from parsec_python.Output.parsec_dat import parsec_dat_from_result, write_parsec_dat
        from parsec_python.models import WavefunctionSelection
        system = prepare_single_point(StrictSCFEigensolverTests()._problem())
        result = run_scf(system)
        data, _ = parsec_dat_from_result(result, system.grid, WavefunctionSelection())
        with tempfile.TemporaryDirectory() as directory:
            path = write_parsec_dat(Path(directory) / "parsec.dat", data)
            with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "off"}):
                _, old = load_restart(path, system)
            with patch.dict(os.environ, {"PARSEC_RESTART_REFINE": "levels"}):
                _, new = load_restart(path, system)
        self.assertIsNone(old.restart_refinement)
        self.assertEqual(new.restart_refinement.tolerance, system.input.eigensolver.tolerance)
        np.testing.assert_array_equal(old.input_potential, new.input_potential)
        self.assertIsNone(new.eigensolver_state)


if __name__ == "__main__":
    unittest.main()
