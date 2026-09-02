"""Parity tests for the optional C++/OpenMP backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.native import (
    NativeHamiltonianBackend,
    build_native_negative_laplacian,
    native_available,
    native_build_info,
    native_unavailable_reason,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import GridSettings


NATIVE_AVAILABLE = native_available()


class NativeBackendImportTests(unittest.TestCase):
    def test_optional_extension_status_is_actionable(self) -> None:
        if NATIVE_AVAILABLE:
            self.assertIsNone(native_unavailable_reason())
        else:
            reason = native_unavailable_reason()
            self.assertIsInstance(reason, str)
            self.assertIn("pip install", reason)
            self.assertIn("parsec_python/acceleration/native", reason)


@unittest.skipUnless(
    NATIVE_AVAILABLE,
    "parsec_accelerated_native has not been built",
)
class NativeBackendParityTests(unittest.TestCase):
    @staticmethod
    def _grid():
        return build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.4,
                expansion_order=8,
            )
        )

    @staticmethod
    def _projector_operator(size: int) -> NonlocalProjectorOperator:
        rng = np.random.default_rng(773)
        projectors = sp.random(
            size,
            5,
            density=0.24,
            format="csc",
            random_state=rng,
            data_rvs=lambda count: rng.normal(size=count),
            dtype=np.float64,
        )
        projectors.sum_duplicates()
        projectors.sort_indices()
        signs = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0])
        labels = tuple((index, 0, 0) for index in range(5))
        return NonlocalProjectorOperator(projectors, signs, labels)

    def test_build_info_reports_float64_openmp_kernel(self) -> None:
        info = native_build_info()
        self.assertEqual(info["module"], "parsec_accelerated_native")
        self.assertEqual(info["dtype"], "float64")
        self.assertTrue(info["openmp_enabled"])
        self.assertTrue(info["fixed_summation_order"])
        detected = int(info["openmp_detected_processors"])
        expected_default = max(1, detected - 4)
        self.assertEqual(int(info["openmp_default_threads"]), expected_default)
        self.assertEqual(
            int(info["openmp_reserved_threads"]),
            detected - expected_default,
        )
        self.assertGreaterEqual(int(info["openmp_max_threads"]), 1)
        if os.environ.get("OMP_NUM_THREADS"):
            self.assertEqual(info["openmp_thread_source"], "OMP_NUM_THREADS")
        else:
            self.assertEqual(
                info["openmp_thread_source"],
                "detected_processors_minus_4",
            )
            self.assertEqual(int(info["openmp_max_threads"]), expected_default)

    def test_omp_num_threads_explicitly_overrides_default(self) -> None:
        environment = dict(os.environ)
        environment["OMP_NUM_THREADS"] = "2"
        code = (
            "import json, parsec_accelerated_native as native; "
            "print(json.dumps(native.build_info()))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        info = json.loads(completed.stdout)
        self.assertEqual(info["openmp_thread_source"], "OMP_NUM_THREADS")
        self.assertEqual(int(info["openmp_max_threads"]), 2)

    def test_native_laplacian_buffers_match_reference_csr(self) -> None:
        settings = [
            GridSettings(
                spacing=0.8,
                radius=2.4,
                expansion_order=order,
            )
            for order in (2, 4, 8, 12, 20)
        ]
        settings.append(
            GridSettings(
                spacing=0.9,
                radius=2.0,
                expansion_order=6,
                domain_shape="box",
                box_lengths=(3.2, 3.6, 4.0),
            )
        )
        for grid_settings in settings:
            with self.subTest(
                order=grid_settings.expansion_order,
                shape=grid_settings.domain_shape,
            ):
                grid = build_cluster_grid(grid_settings)
                reference = build_negative_laplacian(grid)
                native = build_native_negative_laplacian(grid)

                self.assertEqual(native.shape, reference.shape)
                self.assertTrue(native.has_sorted_indices)
                np.testing.assert_array_equal(native.indptr, reference.indptr)
                np.testing.assert_array_equal(native.indices, reference.indices)
                np.testing.assert_allclose(
                    native.data,
                    reference.data,
                    rtol=2.0e-15,
                    atol=0.0,
                )

    def test_fused_vector_and_block_actions_match_scipy_factorization(self) -> None:
        grid = self._grid()
        kinetic = build_negative_laplacian(grid)
        nonlocal_operator = self._projector_operator(grid.size)
        local = np.linspace(-1.75, 0.4, grid.size, dtype=np.float64)
        backend = NativeHamiltonianBackend(kinetic, nonlocal_operator)
        backend.update_local(local)

        rng = np.random.default_rng(9901)
        for vectors in (
            rng.normal(size=grid.size),
            rng.normal(size=(grid.size, 7)),
        ):
            expected = np.asarray(kinetic @ vectors)
            if vectors.ndim == 1:
                expected = expected + local * vectors
                coefficients = nonlocal_operator.projectors.T @ vectors
                coefficients = nonlocal_operator.signs * coefficients
            else:
                expected = expected + local[:, None] * vectors
                coefficients = nonlocal_operator.projectors.T @ vectors
                coefficients = nonlocal_operator.signs[:, None] * coefficients
            expected = expected + np.asarray(
                nonlocal_operator.projectors @ coefficients
            )
            actual = backend.apply(vectors)
            np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

        self.assertEqual(backend.statistics.applications, 2)
        self.assertEqual(backend.statistics.vectors_applied, 8)

    def test_local_update_changes_only_the_diagonal_term(self) -> None:
        grid = self._grid()
        kinetic = build_negative_laplacian(grid)
        nonlocal_operator = self._projector_operator(grid.size)
        backend = NativeHamiltonianBackend(kinetic, nonlocal_operator)
        vector = np.linspace(-0.5, 0.75, grid.size, dtype=np.float64)

        first = np.linspace(-0.2, 0.3, grid.size)
        second = np.linspace(0.6, -0.4, grid.size)
        backend.update_local(first)
        first_result = backend.apply(vector)
        backend.update_local(second)
        second_result = backend.apply(vector)

        np.testing.assert_allclose(
            second_result - first_result,
            (second - first) * vector,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        self.assertEqual(backend.statistics.local_updates, 2)

    def test_zero_projector_case_is_supported(self) -> None:
        grid = self._grid()
        kinetic = build_negative_laplacian(grid)
        empty = NonlocalProjectorOperator(
            sp.csc_matrix((grid.size, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            (),
        )
        backend = NativeHamiltonianBackend(kinetic, empty)
        local = np.full(grid.size, -0.125)
        backend.update_local(local)
        vectors = np.ones((grid.size, 3))
        np.testing.assert_allclose(
            backend.apply(vectors),
            kinetic @ vectors + local[:, None] * vectors,
            rtol=2.0e-14,
            atol=2.0e-14,
        )


if __name__ == "__main__":
    unittest.main()
