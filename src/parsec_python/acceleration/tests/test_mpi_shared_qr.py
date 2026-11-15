"""CPU integration of shared-storage exchanges and the robust QR control flow.

NumPy replaces device storage and SciPy replaces only the cuBLAS triangular
solve. Actual MPI/GPU timing and allocator peaks require the GPU A/B check.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve_triangular

from parsec_python.acceleration.Eigensolvers import mpi_shared_basis as shared
from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout, SectorDeviceGroup
from parsec_python.acceleration.tests.test_mpi_shared_basis import _ThreadComm, _ThreadWorld


class _HostArrays:
    cuda = SimpleNamespace(Device=lambda device: nullcontext())
    asnumpy = staticmethod(np.asarray)

    def __getattr__(self, name):
        return getattr(np, name)


class SharedQRStorageTests(unittest.TestCase):
    def exchange(self, matrix, layout, width, *, trim=None, qr=False, padded=True):
        size = len(layout.parts)
        world = _ThreadWorld(size)
        rows = matrix.shape[0]
        backing_by_id, columns = {}, []
        for rank in range(size):
            plan = shared._RowExchange(SimpleNamespace(size=size, rank=rank), layout, rows, width)
            capacity = plan.inplace_storage()[1] if padded else rows * layout.widths[rank]
            backing = np.empty(max(1, capacity))
            block = backing[:rows * layout.widths[rank]].reshape((rows, layout.widths[rank]), order="F")
            for column, offset, count in layout.spans(rank):
                block[:, offset:offset + count] = matrix[:, column:column + count]
            if trim is not None:
                block = block[:, :layout.leading(trim).widths[rank]]
            backing_by_id[id(block)] = backing
            columns.append(block)
        if trim is not None:
            layout, matrix = layout.leading(trim), matrix[:, :trim]

        def view(xp, block, offset, shape):
            return backing_by_id[id(block)][offset:offset + int(np.prod(shape))].reshape(shape, order="F")

        def solve(factor, block):
            block[...] = solve_triangular(factor, block.T, lower=True).T

        def run(rank):
            group = shared.MPIShardGroup.__new__(shared.MPIShardGroup)
            group.rank, group.size, group.owner, group.device = rank, size, 0, 0
            group.comm = _ThreadComm(world, rank)
            group.separate_row_blocks = 0
            group._round_width = lambda n: width
            group._sync = lambda: None

            def exchange(send, sc, receive, rc):
                group.comm.Alltoallv(
                    [send, (sc, np.cumsum([0] + sc[:-1]))],
                    [receive, (rc, np.cumsum([0] + rc[:-1]))])

            def sum_(packed):
                result = np.empty_like(packed)
                group.comm.Allreduce(packed, result)
                return result

            group._exchange = exchange
            group._sum = sum_
            group._broadcast_matrix = lambda factor, n: np.array(group.comm.bcast(factor, root=0), order="F")
            group.ritz = lambda operator, basis, stages: (None, basis, None)
            group.to_rows = lambda *args: self.fail("candidate allocated a second basis through to_rows")
            original = columns[rank]
            if qr:
                _, result, _ = group.orthonormal_ritz(None, shared.SharedBasis(group, original, layout), None)
                self.assertIs(result.local, original)
            else:
                row_basis, plan = group._to_rows_in_place(original, layout, rows)
                np.testing.assert_array_equal(row_basis, matrix[plan.edges[rank]:plan.edges[rank + 1]])
                # A change in canonical state order must reach the right owner.
                row_basis *= np.arange(layout.columns) + 1
                group._to_columns_in_place(row_basis, original, plan)
            return original.copy(), group.separate_row_blocks

        with patch.object(shared, "require_cupy", return_value=(_HostArrays(), None)), \
                patch.object(shared, "dense_solve_on_device", return_value=False), \
                patch.object(shared, "_solve_lower_transpose_in_place", side_effect=solve), \
                patch.object(SectorDeviceGroup, "_room", side_effect=lambda block: backing_by_id[id(block)].size), \
                patch.object(SectorDeviceGroup, "_view", side_effect=view), \
                patch.dict(os.environ, PARSEC_MPI_SHARED_QR="inplace", PARSEC_DENSE_BUFFER_BYTES="256"):
            with ThreadPoolExecutor(size) as pool:
                results = list(pool.map(run, range(size)))
        whole = np.empty_like(matrix)
        for rank, (block, separate) in enumerate(results):
            if padded:
                self.assertEqual(separate, 0)
            for column, offset, count in layout.spans(rank):
                whole[:, column:column + count] = block[:, offset:offset + count]
        return whole, [separate for _, separate in results]

    def test_canonical_exchanges_cover_uneven_empty_and_trimmed_ranks(self):
        for rows, widths in ((17, (6, 5, 2)), (3, (0, 1, 0, 5)), (100, (9, 9, 9)), (2, (2, 0, 0))):
            edges = np.cumsum((0,) + widths)
            layout = ColumnLayout(tuple(zip(edges[:-1], edges[1:])))
            matrix = np.arange(rows * sum(widths), dtype=float).reshape(rows, -1)
            for width in (1, 2, 7, 100):
                for trim in (None, max(0, sum(widths) - 3)):
                    with self.subTest(rows=rows, widths=widths, round_width=width, trim=trim):
                        actual, _ = self.exchange(matrix, layout, width, trim=trim)
                        expected = matrix if trim is None else matrix[:, :trim]
                        np.testing.assert_array_equal(actual, expected * (np.arange(expected.shape[1]) + 1))

    def test_interleaved_filter_layout_and_external_unpadded_storage(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import interleaved_ranges

        layout = ColumnLayout(interleaved_ranges(uniform_filter_blocks(97, 6, 10), 3))
        matrix = np.arange(113 * 97, dtype=float).reshape(113, 97)
        for trim in (None, 29, 4):
            for padded in (True, False):
                actual, separate = self.exchange(matrix, layout, 2, trim=trim, padded=padded)
                expected = matrix if trim is None else matrix[:, :trim]
                np.testing.assert_array_equal(actual, expected * (np.arange(expected.shape[1]) + 1))
                if not padded and trim is None:
                    self.assertGreater(sum(separate), 0)

    def test_three_pass_qr_preserves_the_ill_conditioned_subspace(self):
        from parsec_python.acceleration.tests.test_mpi_shared_basis import ShiftedCholeskyTests

        matrix = ShiftedCholeskyTests.basis(rows=401, columns=23, condition=1e9)
        layout = ColumnLayout(((0, 8), (8, 14), (14, 23)))
        actual, _ = self.exchange(matrix, layout, 3, qr=True)
        np.testing.assert_allclose(actual.T @ actual, np.eye(23), rtol=0, atol=2e-12)
        self.assertLess(np.linalg.norm(matrix - actual @ (actual.T @ matrix)) / np.linalg.norm(matrix), 2e-13)
        # Compare the same three passes in canonical order on one host.
        reference = matrix.copy()
        for step in range(3):
            factor = shared._host_shifted_cholesky(reference.T @ reference, 401, shifted=step == 0)
            reference = solve_triangular(factor, reference.T, lower=True).T
        # The projection, not individual QR vectors, is the physical invariant.
        self.assertLess(np.linalg.norm(reference - actual @ (actual.T @ reference)), 2e-6)

    def test_a_rank_deficient_basis_still_fails_the_robust_fallback(self):
        layout = ColumnLayout(((0, 8), (8, 14), (14, 23)))
        with self.assertRaisesRegex(RuntimeError, "could not be orthonormalized"):
            self.exchange(np.zeros((101, 23)), layout, 3, qr=True)


if __name__ == "__main__":
    unittest.main()
