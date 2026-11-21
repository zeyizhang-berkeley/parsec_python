"""The filters of several devices and of shared bases for an operator without the stencil-major kernels.

``PARSEC_CUPY_STENCIL_MAJOR=0`` (or a failed build of that layout) leaves a sector operator with the CSR-order
compact stencil, or the generic CSR.  The other devices of its sector then get replicas built from its CSR
(``distributed_filter.replica_csr``), and they, the ranks of a shared basis and the GPUs of a nested group filter
without CUDA graphs (``filter_graph.PlainBlockFilter``), with the arithmetic of the filter of one device.  Before
2026-11-21 these layouts stopped with an ``AttributeError`` (``neighbors``, ``chebyshev_recurrence``).
"""

import gc
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy
from parsec_python.acceleration.Eigensolvers import filter_graph
from parsec_python.acceleration.Eigensolvers.chebyshev import (
    FilterBlock,
    chebff_filter,
    subspace_filter,
    subspace_filter_blocks,
)
from parsec_python.acceleration.Eigensolvers.distributed_filter import starting_sigmas

# The filter of one device without graphs, in FP64, for the comparisons.
_ONE_DEVICE = dict(PARSEC_CUPY_DISTRIBUTED_FILTER="0", PARSEC_CUPY_FILTER_GRAPHS="1", PARSEC_CUPY_BATCH_FILTERS="0",
                   PARSEC_CUPY_MIXED_FILTER="off")


def _assert_filtered_alike(case, actual, expected, repeated):
    """``actual`` to the last bit where the filter of one device repeats its own bits (``repeated``, a second run of
    it), and within round-off otherwise: the projector products of a CSR operator (cuSPARSE, ``B.T X`` and ``B C``)
    need not sum in the same order on every run (on A100 a few columns of two filters of one device differed in the
    last bit)."""

    if np.array_equal(expected.view(np.uint64), repeated.view(np.uint64)):
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    else:
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14 * scale)
        np.testing.assert_allclose(repeated, expected, rtol=0, atol=1e-14 * scale)


class GraphCapabilityTests(unittest.TestCase):
    """Which operators the graphs can record, without a device."""

    def test_an_operator_without_the_stencil_major_kernels_is_filtered_without_graphs(self):
        for stencil in (None, SimpleNamespace(apply=None)):
            operator = SimpleNamespace(compact_finite_difference=stencil, projector_count=0,
                                       custom_projector_projection=None)
            self.assertFalse(filter_graph.graphs_can_filter(operator))

    def test_a_stencil_major_operator_with_projectors_needs_its_tree_projection(self):
        stencil = object.__new__(filter_graph.CuPyStencilMajorFiniteDifference)
        bare = SimpleNamespace(compact_finite_difference=stencil, projector_count=0, custom_projector_projection=None)
        self.assertTrue(filter_graph.graphs_can_filter(bare))
        projected = SimpleNamespace(compact_finite_difference=stencil, projector_count=3,
                                    custom_projector_projection=object())
        self.assertTrue(filter_graph.graphs_can_filter(projected))
        unprojected = SimpleNamespace(compact_finite_difference=stencil, projector_count=3,
                                      custom_projector_projection=None)
        self.assertFalse(filter_graph.graphs_can_filter(unprojected))


@unittest.skipUnless(cupy_available(), "CUDA unavailable")
class CSRDeviceFilterTests(unittest.TestCase):
    rows = 257

    def tearDown(self):
        gc.collect()

    def operator(self, *, stencil_major="0", compact=True):
        n = self.rows
        kinetic = sp.diags([-0.2 * np.ones(n - 1), 2.0 * np.ones(n), -0.2 * np.ones(n - 1)], [-1, 0, 1], format="csr")
        potential = np.linspace(-0.4, 0.3, n)
        projectors = sp.csr_matrix(np.random.default_rng(18).normal(size=(n, 3)) * 0.02)
        signs = np.array([1.0, -1.0, 1.0])
        with patch.dict(os.environ, {"PARSEC_CUPY_STENCIL_MAJOR": stencil_major, "PARSEC_CUPY_COMPACT_FD": "1",
                                     "PARSEC_CUPY_FUSED_PROJECTOR_SCATTER": "1",
                                     "PARSEC_CUPY_PROJECTOR_PROJECTION": "1"}):
            return CuPyHamiltonian(kinetic, potential, projectors=projectors, projector_signs=signs,
                                   retain_generic_laplacian=False, use_compact_finite_difference=compact)

    def test_the_plain_filter_gives_the_filter_of_one_device(self):
        cp, _ = require_cupy()
        from parsec_python.acceleration.backends.cupy_compact import CuPyCompactFiniteDifference

        host = np.random.default_rng(5).normal(size=(self.rows, 19))
        for compact in (True, False):
            operator = self.operator(compact=compact)
            if compact:
                self.assertIsInstance(operator.compact_finite_difference, CuPyCompactFiniteDifference)
            else:
                self.assertIsNone(operator.compact_finite_difference)
            self.assertFalse(filter_graph.graphs_can_filter(operator))
            workspace = filter_graph.operator_filter(operator)
            self.assertIsInstance(workspace, filter_graph.PlainBlockFilter)
            self.assertIs(filter_graph.operator_filter(operator), workspace)
            self.assertFalse(workspace.recorded)
            for reset in (False, True):
                with self.subTest(compact=compact, reset=reset), patch.dict(os.environ, _ONE_DEVICE):
                    # The SUBSPACE plan (two degrees), sigma carried from block to block.
                    expected, repeated = (
                        cp.asnumpy(subspace_filter(operator, cp.asarray(host, order="F"), 7, 2, 0.0, 4.0,
                                                   reset_recurrence_per_block=reset, mixed_precision=False))
                        for _ in range(2))
                    blocks = subspace_filter_blocks(19, 6, 7, 2)
                    actual = cp.asnumpy(workspace.apply(cp.asarray(host, order="F"), blocks, 0.0, 4.0, 0.0, reset))
                    _assert_filtered_alike(self, actual, expected, repeated)
                    # CHEBFF blocks with their own reference, in place, the sigma carried into each block given per
                    # block (as the ranks of a shared basis pass it).
                    expected, repeated = (
                        cp.asnumpy(chebff_filter(operator, cp.asarray(host, order="F"), 7, 0.0, 4.0, -0.5,
                                                 reset_recurrence_per_block=reset))
                        for _ in range(2))
                    uniform = tuple(FilterBlock(start, min(start + 6, 19), 7) for start in range(0, 19, 6))
                    target = cp.asarray(host, order="F")
                    result = workspace.apply(target, uniform, 0.0, 4.0, -0.5, reset,
                                             initial_sigma=starting_sigmas(uniform, 0.0, 4.0, reset, -0.5),
                                             out=target)
                    self.assertIs(result, target)
                    _assert_filtered_alike(self, cp.asnumpy(result), expected, repeated)

    def test_the_devices_of_a_csr_sector_filter_like_its_owner(self):
        cp, _ = require_cupy()
        count = cp.cuda.runtime.getDeviceCount()
        if count < 2:
            self.skipTest("two GPUs required")
        from parsec_python.acceleration.backends.cupy_compact import CuPyCompactFiniteDifference

        group = tuple(range(min(count, 4)))
        host = np.random.default_rng(31).normal(size=(self.rows, 23))
        common = dict(PARSEC_CUPY_FILTER_GRAPHS="1", PARSEC_CUPY_BATCH_FILTERS="0", PARSEC_CUPY_MIXED_FILTER="off",
                      PARSEC_CUPY_FILTER_COLUMN_MAJOR="1", PARSEC_CUPY_DEVICES=",".join(map(str, group)))
        for compact in (True, False):
            with cp.cuda.Device(group[0]):
                operator = self.operator(compact=compact)
                operator.distributed_filter_devices = group
                for shift in (0.0, 0.031):
                    operator.update_local_potential(np.linspace(-0.4, 0.3, self.rows) + shift)
                    for reset in (False, True):
                        with self.subTest(compact=compact, shift=shift, reset=reset):
                            x = cp.asarray(host, order="F")
                            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER="0", **common):
                                expected, repeated = (
                                    cp.asnumpy(subspace_filter(operator, x, 9, 3, 1.2, 6.0,
                                                               reset_recurrence_per_block=reset))
                                    for _ in range(2))
                            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER="1", **common):
                                actual = subspace_filter(operator, x, 9, 3, 1.2, 6.0, reset_recurrence_per_block=reset)
                            self.assertEqual(int(actual.device.id), group[0])
                            _assert_filtered_alike(self, cp.asnumpy(actual), expected, repeated)
                            np.testing.assert_array_equal(cp.asnumpy(x), host)
                worker = operator._distributed_filter
                self.assertEqual(set(worker.replicas), set(group) - {group[0]})
                for device, replica in worker.replicas.items():
                    held = replica.compact_finite_difference
                    if compact:
                        self.assertIsInstance(held, CuPyCompactFiniteDifference)
                        stencil = operator.compact_finite_difference
                        for name in ("indptr", "indices", "coefficient_codes", "coefficient_palette"):
                            mine, theirs = getattr(held, name), getattr(stencil, name)
                            self.assertEqual(int(mine.device.id), device)
                            np.testing.assert_array_equal(cp.asnumpy(mine), cp.asnumpy(theirs))
                    else:
                        self.assertIsNone(held)
                        self.assertIsNotNone(replica.negative_laplacian)
                for workspace in worker.graphs.values():
                    self.assertIsInstance(workspace, filter_graph.PlainBlockFilter)

    def test_a_stencil_major_sector_keeps_its_graphs(self):
        operator = self.operator(stencil_major="1")
        if operator.custom_projector_projection is None:
            self.skipTest("no tree projection on this device")
        self.assertTrue(filter_graph.graphs_can_filter(operator))
        self.assertIsInstance(filter_graph.device_filter(operator), filter_graph.BlockFilterGraphs)


if __name__ == "__main__":
    unittest.main()
