"""Correctness tests for exact-key startup shortcuts."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.Laplacian import (
    DeferredNativeNegativeLaplacian,
)
from parsec_python.acceleration.Symmetry import (
    AxisReflectionReduction,
    ReflectionRepresentationDecomposition,
)
from parsec_python.acceleration.Symmetry.operator_cache import (
    _MEMORY_BUNDLES,
    _MEMORY_BUNDLES_LOCK,
    load_or_build_reduced_operators,
)
from parsec_python.acceleration.backends.native import native_available
from parsec_python.Grid import build_cluster_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.models import Atom, GridSettings


class DeferredFiniteDifferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.8,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )

    def test_descriptor_count_and_key_match_discrete_operator_inputs(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        reference = build_negative_laplacian(self.grid)
        repeated = DeferredNativeNegativeLaplacian(self.grid)

        self.assertEqual(descriptor.shape, reference.shape)
        self.assertEqual(descriptor.nnz, reference.nnz)
        self.assertEqual(descriptor.cache_key, repeated.cache_key)
        self.assertFalse(descriptor.materialized)

        changed_grid = build_cluster_grid(
            GridSettings(
                spacing=0.75,
                radius=2.8,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        changed = DeferredNativeNegativeLaplacian(changed_grid)
        self.assertNotEqual(descriptor.cache_key, changed.cache_key)

    def test_descriptor_reuses_exact_nnz_cache(self) -> None:
        directory = Path.cwd() / ".tmp" / f"nnz-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = DeferredNativeNegativeLaplacian(
                self.grid,
                cache_directory=directory,
            )
            self.assertEqual(first.nnz_cache_status, "miss-written")
            self.assertIsNotNone(first.nnz_cache_path)
            self.assertTrue(first.nnz_cache_path.is_file())

            second = DeferredNativeNegativeLaplacian(
                self.grid,
                cache_directory=directory,
            )
            self.assertEqual(second.nnz, first.nnz)
            self.assertEqual(second.cache_key, first.cache_key)
            self.assertEqual(second.nnz_cache_status, "memory-hit")
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    @unittest.skipUnless(
        native_available(),
        "parsec_accelerated_native has not been built",
    )
    def test_materialization_is_exact_and_memoized(self) -> None:
        descriptor = DeferredNativeNegativeLaplacian(self.grid)
        expected = build_negative_laplacian(self.grid)

        first = descriptor.materialize()
        second = descriptor.materialize()

        self.assertIs(first, second)
        self.assertTrue(descriptor.materialized)
        np.testing.assert_array_equal(first.indptr, expected.indptr)
        np.testing.assert_array_equal(first.indices, expected.indices)
        np.testing.assert_array_equal(first.data, expected.data)

    def test_exact_seed_cache_hit_never_materializes_full_operator(self) -> None:
        reduction = AxisReflectionReduction.detect(
            self.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, reduction
        )
        full = build_negative_laplacian(self.grid)
        values = np.linspace(-0.3, 0.7, self.grid.size)
        projectors = sp.csc_matrix(values[:, None])
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.asarray((1.0,), dtype=np.float64),
            labels=((0, 0, 0),),
        )
        directory = Path.cwd() / ".tmp" / f"startup-cache-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            first = load_or_build_reduced_operators(
                decomposition,
                full,
                nonlocal_operator,
                cache_directory=directory,
                kinetic_key_seed="a" * 64,
                decomposition_key_seed="b" * 64,
            )
            # An arbitrary sentinel cannot be converted to CSR.  A successful
            # second call proves the exact-key hit was accepted before any
            # full-grid materialization/canonicalization was attempted.
            sentinel = object()
            second = load_or_build_reduced_operators(
                decomposition,
                sentinel,
                nonlocal_operator,
                cache_directory=directory,
                kinetic_key_seed="a" * 64,
                decomposition_key_seed="b" * 64,
            )

            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "hit")
            for left, right in zip(
                first.stencil_metadata,
                second.stencil_metadata,
                strict=True,
            ):
                np.testing.assert_array_equal(left.neighbors, right.neighbors)
                np.testing.assert_array_equal(
                    left.coefficient_codes, right.coefficient_codes
                )
                np.testing.assert_array_equal(
                    left.coefficient_palette, right.coefficient_palette
                )
        finally:
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()

    def test_resident_operator_bundle_uses_bounded_memory_hit(self) -> None:
        reduction = AxisReflectionReduction.detect(
            self.grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        decomposition = ReflectionRepresentationDecomposition.build(
            self.grid, reduction
        )
        full = build_negative_laplacian(self.grid)
        values = np.linspace(-0.3, 0.7, self.grid.size)
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix(values[:, None]),
            signs=np.asarray((1.0,), dtype=np.float64),
            labels=((0, 0, 0),),
        )
        directory = Path.cwd() / ".tmp" / f"operator-memory-test-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=False)
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "PARSEC_ACCELERATED_RESIDENT": "1",
                        "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "1",
                    },
                ),
                _MEMORY_BUNDLES_LOCK,
            ):
                _MEMORY_BUNDLES.clear()
            with patch.dict(
                os.environ,
                {
                    "PARSEC_ACCELERATED_RESIDENT": "1",
                    "PARSEC_RESIDENT_OPERATOR_CACHE_SIZE": "1",
                },
            ):
                first = load_or_build_reduced_operators(
                    decomposition,
                    full,
                    nonlocal_operator,
                    cache_directory=directory,
                    kinetic_key_seed="c" * 64,
                    decomposition_key_seed="d" * 64,
                )
                second = load_or_build_reduced_operators(
                    decomposition,
                    object(),
                    nonlocal_operator,
                    cache_directory=directory,
                    kinetic_key_seed="c" * 64,
                    decomposition_key_seed="d" * 64,
                )
            self.assertEqual(first.cache_info.status, "miss-written")
            self.assertEqual(second.cache_info.status, "memory-hit")
            self.assertIs(second.stencil_metadata, first.stencil_metadata)
            self.assertIs(second.nonlocal_operators, first.nonlocal_operators)
        finally:
            with _MEMORY_BUNDLES_LOCK:
                _MEMORY_BUNDLES.clear()
            for generated in directory.iterdir():
                generated.unlink()
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
