"""Numerical invariants and bounded scratch under uneven MPI decompositions."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.Eigensolvers.dense_memory import (
    broadcast_matrix, buffer_bytes, lower_sum_in_place, mirror_lower,
    permute_columns_in_place, sum_in_place,
)
from parsec_python.acceleration.tests.thread_mpi import run_ranks


class DenseMemoryTests(unittest.TestCase):
    def test_column_permutation_preserves_storage_and_bounds_the_actual_gather(self):
        scratch = []

        class Tracked(np.ndarray):
            def __getitem__(self, key):
                result = super().__getitem__(key)
                if isinstance(key, tuple) and isinstance(key[1], np.ndarray):
                    scratch.append(result.nbytes)
                return result

        rng = np.random.default_rng(82)
        for dtype in (np.float64, np.complex128):
            for rows, columns in ((31, 17), (0, 7), (5, 0)):
                for budget in (16, 432):
                    with self.subTest(dtype=dtype, shape=(rows, columns), budget=budget):
                        backing = np.arange((rows + 4) * (columns + 2), dtype=dtype).reshape(
                            rows + 4, columns + 2, order="F")
                        block = backing[2:2 + rows, 1:1 + columns].view(Tracked)
                        original, outside = np.array(block), backing.copy()
                        order = rng.permutation(columns)
                        pointer = block.__array_interface__["data"][0]
                        scratch.clear()
                        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES=str(budget)):
                            self.assertIs(permute_columns_in_place(block, order, np), block)
                            np.testing.assert_array_equal(block, original[:, order])
                            permute_columns_in_place(block, np.argsort(order), np)
                        np.testing.assert_array_equal(backing, outside)
                        self.assertEqual(block.__array_interface__["data"][0], pointer)
                        self.assertTrue(all(n <= max(budget, columns * np.dtype(dtype).itemsize)
                                            for n in scratch))

    def test_invalid_column_permutation_does_not_modify_the_matrix(self):
        matrix = np.arange(12.).reshape(3, 4)
        original = matrix.copy()
        for order in ([0, 1, 1, 3], [0, 1, 2], [-1, 1, 2, 3], [0., 1., 2., 3.]):
            with self.assertRaises(ValueError):
                permute_columns_in_place(matrix, order, np)
            np.testing.assert_array_equal(matrix, original)

    def test_mirror_ignores_poisoned_upper_triangle_and_preserves_dtype(self):
        rng = np.random.default_rng(512)
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="144"):
            for dtype in (np.float64, np.complex128):
                raw = rng.normal(size=(17, 17)).astype(dtype)
                if dtype == np.complex128:
                    raw += 1j * rng.normal(size=raw.shape)
                raw[np.triu_indices(17, 1)] = np.nan
                expected = np.tril(raw) + np.tril(raw, -1).conj().T
                actual = mirror_lower(raw, np)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(np.isnan(raw[0, -1]))
                self.assertIs(mirror_lower(raw, np, copy=False), raw)
                np.testing.assert_array_equal(raw, expected)

    def test_sum_reuses_storage_and_bounds_every_collective(self):
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="80"):
            for size in (1, 3, 7):
                for transport in ("host", "cuda"):
                    def run(ctx):
                        local = np.arange(54, dtype=np.float64).reshape(2, 3, 9) + ctx.rank
                        def reduce(send, receive):
                            self.assertLessEqual(send.nbytes, 80)
                            ctx.comm.Allreduce(send, receive)
                        total = sum_in_place(local, np, reduce, transport=transport)
                        self.assertIs(total, local)
                        expected = np.arange(54).reshape(2, 3, 9) * size + size * (size - 1) / 2
                        np.testing.assert_array_equal(total, expected)
                    run_ranks(size, run)

    def test_lower_reduction_is_exact_and_does_not_grow_with_state_count(self):
        rng = np.random.default_rng(6)
        original = rng.normal(size=(19, 19)) + 1j * rng.normal(size=(19, 19))
        original[np.triu_indices(19, 1)] = np.nan
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="256"):
            for size in (1, 3, 6):
                def run(ctx):
                    local = original.copy() * (ctx.rank + 1)
                    def reduce(send, receive):
                        self.assertLessEqual(send.nbytes, 256)
                        ctx.comm.Reduce(send, receive, root=size - 1)
                    result = lower_sum_in_place(local, np, reduce, root=ctx.rank == size - 1, transport="host")
                    if ctx.rank == size - 1:
                        self.assertIs(result, local)
                        expected = sum(np.tril(original * (rank + 1)) for rank in range(size))
                        np.testing.assert_array_equal(result, expected)
                    else:
                        self.assertIsNone(result)
                run_ranks(size, run)

    def test_broadcast_reuses_fortran_root_and_covers_ragged_final_chunk(self):
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="80"):
            for dtype in (np.float64, np.complex128):
                for order in ("C", "F"):
                    original = np.array(np.arange(63).reshape(7, 9), dtype=dtype, order=order)
                    def run(ctx):
                        def bcast(piece):
                            self.assertLessEqual(piece.nbytes, 80)
                            ctx.comm.Bcast(piece, root=2)
                        result = broadcast_matrix(original if ctx.rank == 2 else None, original.shape,
                                                  dtype, np, bcast, root=ctx.rank == 2, transport="host")
                        np.testing.assert_array_equal(result, original)
                        self.assertTrue(result.flags.f_contiguous)
                        if ctx.rank == 2 and order == "F":
                            self.assertIs(result, original)
                    run_ranks(4, run)

    def test_invalid_budget_fails_before_collectives(self):
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="0"):
            with self.assertRaises(ValueError):
                buffer_bytes()

    def test_cuda_reduction_finishes_queued_copy_before_receive_buffer_reuse(self):
        # A stream double delays device copies while MPI immediately writes
        # its receive buffer. Reusing a pooled allocation without a stream
        # wait would corrupt all but the final chunk.
        pending = []
        class DeviceArray(np.ndarray):
            def __setitem__(self, key, value):
                pending.append(lambda: np.ndarray.__setitem__(self, key, value))
        def sync():
            for copy in pending:
                copy()
            pending.clear()
        pool = np.empty(10, dtype=np.float64)
        xp = SimpleNamespace(empty_like=lambda piece: pool[:piece.size],
                             cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=sync)))
        original = np.arange(43, dtype=np.float64)
        local = original.copy().view(DeviceArray)
        def reduce(send, receive):
            receive[...] = np.asarray(send) * 2
        with patch.dict(os.environ, PARSEC_DENSE_BUFFER_BYTES="80"):
            sum_in_place(local, xp, reduce, transport="cuda")
        sync()
        np.testing.assert_array_equal(local, original * 2)


if __name__ == "__main__":
    unittest.main()
