"""A sector that grows during the SCF keeps its vectors (``PARSEC_CUPY_SECTOR_WARM_GROWTH``).

The switch and the exchange of saved columns among ranks run on the host; the first solves need a CUDA device (one
GPU named twice stands for a group of devices).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.Eigensolvers.subspace import SubspaceSettings
from parsec_python.acceleration.backends.cupy import require_cupy
from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout
from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup
from parsec_python.acceleration.Eigensolvers.sector_growth import (
    SavedColumns,
    host_room,
    new_column_scale,
    sector_warm_growth_setting,
    staged_on_every_rank,
)
from parsec_python.acceleration.tests.test_mpi_shared_basis import _ThreadComm, _ThreadWorld
from parsec_python.acceleration.tests.test_ritz_handover import _ROUTE, _clean, _device_count, _operator


class SwitchTests(unittest.TestCase):
    def test_values(self):
        for raw, value in (("auto", "auto"), ("", "auto"), ("off", "off"), (" AUTO ", "auto")):
            with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH": raw}):
                self.assertEqual(sector_warm_growth_setting(), value)
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH": "on"}):
            with self.assertRaises(ValueError):
                sector_warm_growth_setting()

    def test_the_new_columns_get_about_unit_norm(self):
        rows = 30000
        column = np.random.default_rng(1).uniform(-1.0, 1.0, rows)
        self.assertAlmostEqual(float(np.linalg.norm(column * new_column_scale(rows))), 1.0, delta=0.02)


class AgreementTests(unittest.TestCase):
    """The ranks of a shared basis take one start; the host room is counted while a copy is staged."""

    def test_one_rank_that_could_not_stage_turns_every_rank_to_random_vectors(self):
        from types import SimpleNamespace

        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import SharedBasis

        size = 4
        for failing in (None, 0, 2):
            world = _ThreadWorld(size)

            def rank_job(rank):
                vectors = SharedBasis.__new__(SharedBasis)
                vectors.group = SimpleNamespace(comm=_ThreadComm(world, rank))
                return staged_on_every_rank(vectors, rank != failing)

            with ThreadPoolExecutor(max_workers=size) as pool:
                answers = [future.result(timeout=60) for future in [pool.submit(rank_job, r) for r in range(size)]]
            self.assertEqual(answers, [failing is None] * size)
        # A basis of one process asks no one.
        self.assertTrue(staged_on_every_rank(np.zeros((3, 2)), True))
        self.assertFalse(staged_on_every_rank(np.zeros((3, 2)), False))

    def test_staged_bytes_count_until_the_copy_is_done(self):
        from parsec_python.acceleration.Eigensolvers import sector_growth

        with patch.object(sector_growth, "host_available_bytes", lambda: 1000):
            with host_room(500) as first:
                self.assertTrue(first)
                with host_room(250) as second:
                    self.assertTrue(second)
                with host_room(251) as third:
                    self.assertFalse(third)
            with host_room(500) as again:
                self.assertTrue(again)
            with host_room(501) as large:
                self.assertFalse(large)
        with patch.object(sector_growth, "host_available_bytes", lambda: None):
            with host_room(1 << 60) as unknown:
                self.assertTrue(unknown)

    def test_the_ranks_of_a_node_divide_its_room(self):
        from parsec_python.acceleration.Eigensolvers import sector_growth

        before = sector_growth._host_processes
        try:
            with patch.object(sector_growth, "host_available_bytes", lambda: 1000):
                sector_growth.set_host_processes(4)
                with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST": ""}):
                    self.assertEqual(sector_growth.host_processes(), 4)
                    # half of the available 1000 bytes among four ranks: 125 each
                    with host_room(125) as fits:
                        self.assertTrue(fits)
                    with host_room(126) as fits:
                        self.assertFalse(fits)
                with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST": "process"}):
                    # the whole half for every process, as before
                    self.assertEqual(sector_growth.host_processes(), 1)
                    with host_room(500) as fits:
                        self.assertTrue(fits)
                    with host_room(501) as fits:
                        self.assertFalse(fits)
                sector_growth.set_host_processes(0)
                self.assertEqual(sector_growth.host_processes(), 1)
        finally:
            sector_growth._host_processes = before

    def test_the_host_switch(self):
        from parsec_python.acceleration.Eigensolvers.sector_growth import warm_growth_host_setting

        for raw, value in (("", "node"), ("node", "node"), (" PROCESS ", "process")):
            with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST": raw}):
                self.assertEqual(warm_growth_host_setting(), value)
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST": "rank"}):
            with self.assertRaises(ValueError):
                warm_growth_host_setting()


class ExchangeTests(unittest.TestCase):
    """The saved columns reach the ranks that hold them in the layout of the larger basis."""

    rows = 7

    def _run(self, old, new, kept, round_bytes):
        size = len(old.parts)
        world = _ThreadWorld(size)
        coded = (np.arange(1, self.rows + 1)[:, None] * 1000.0 + np.arange(kept)[None, :])

        def rank_job(rank):
            group = MPIShardGroup.__new__(MPIShardGroup)
            group.comm, group.rank, group.size = _ThreadComm(world, rank), rank, size
            group._mpi, group.exchanged_bytes = None, 0
            columns = [c for start, stop in old.parts[rank] for c in range(start, stop)]
            saved = SavedColumns(kept, old, {rank: np.asfortranarray(coded[:, columns])})
            got = {}

            def write(names, values):
                for index, column in enumerate(names):
                    got[column] = values[:, index].copy()

            with patch.dict(os.environ, {"PARSEC_MPI_SHARED_EXCHANGE_BYTES": str(round_bytes)}):
                group._exchange_saved(self.rows, new, saved, write)
            return got

        with ThreadPoolExecutor(max_workers=size) as pool:
            results = [future.result(timeout=60) for future in [pool.submit(rank_job, r) for r in range(size)]]
        for rank, got in enumerate(results):
            expected = [c for start, stop in new.parts[rank] for c in range(start, min(stop, kept))]
            self.assertEqual(sorted(got), expected)
            for column, values in got.items():
                np.testing.assert_array_equal(values, coded[:, column])

    def test_contiguous_to_contiguous(self):
        self._run(ColumnLayout(((0, 4), (4, 7), (7, 10))), ColumnLayout(((0, 5), (5, 10), (10, 15))), 10, 1 << 20)

    def test_interleaved_layouts_in_small_rounds(self):
        old = ColumnLayout((((0, 2), (5, 7)), ((2, 4), (7, 9)), ((4, 5), (9, 10))))
        new = ColumnLayout((((0, 3), (8, 11)), ((3, 6), (11, 13)), ((6, 8), (13, 15))))
        self._run(old, new, 10, 8 * self.rows * 3)

    def test_a_rank_without_columns(self):
        self._run(ColumnLayout(((0, 6), (6, 6))), ColumnLayout(((0, 3), (3, 9))), 6, 8 * self.rows)


@unittest.skipUnless(_device_count() >= 1, "a CUDA device required")
class GrowthTests(unittest.TestCase):
    """The first solve of a larger count from the saved vectors, against one from random vectors."""

    rows, states, grown = 257, 12, 18

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)
        dense = self.cp.asnumpy(self.op @ self.cp.eye(self.rows, dtype=self.cp.float64, order="F"))
        self.exact = np.linalg.eigvalsh(0.5 * (dense + dense.T))

    def tearDown(self):
        gc.collect()

    def _solver(self, method="chebff", capable=False):
        from parsec_python.Eigensolvers.chebdav import ChebDavSettings
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalSolver

        settings = EigvalSettings(
            initial_method=method, safety_buffer=0, chebff=ChebFFSettings(polynomial_degree=12, filter_cycles=3),
            chebdav=ChebDavSettings(), subspace=SubspaceSettings(polynomial_degree=8, degree_delta=1))
        return CuPyEigvalSolver(self.op, settings=settings, retain_vectors_on_device=True,
                                compute_subspace_residuals=False, handover_capable=capable)

    def _errors(self, grow, method="chebff"):
        solver = self._solver(method)
        for _ in range(12):
            solver.solve(self.states)
        if grow:
            self.assertTrue(solver.stage_growth_start(self.grown))
            self.assertIsNone(solver.device_state)
        else:
            solver.reset()
        result = solver.solve(self.grown)
        self.assertEqual(result.solver_path, method)
        self.assertEqual(result.eigenvalues.shape, (self.grown,))
        return np.abs(result.eigenvalues[: self.states] - self.exact[: self.states]).max(), result

    def _environment(self, **extra):
        return _clean(**_ROUTE, PARSEC_CUPY_EXCHANGE_CONCURRENT="0", **extra)

    def test_the_occupied_levels_stay_converged(self):
        with self._environment():
            warm, _ = self._errors(True)
            cold, _ = self._errors(False)
        self.assertLess(warm, 0.05 * cold)

    def _cold_equal(self, solver):
        """The solve after a refused staging is the solve of a fresh solver, bit for bit."""
        grown = solver.solve(self.grown)
        fresh = self._solver(solver.settings.initial_method).solve(self.grown)
        np.testing.assert_array_equal(grown.eigenvalues, fresh.eigenvalues)
        self.assertEqual(solver.device_state.solves_completed, 1)

    def test_a_chebdav_solver_grows_from_random_vectors_as_before(self):
        with self._environment():
            solver = self._solver("chebdav")
            for _ in range(12):
                solver.solve(self.states)
            self.assertFalse(solver.stage_growth_start(self.grown))
            self.assertIsNone(solver.device_state)
            self._cold_equal(solver)

    def test_the_method_of_the_next_solve_decides(self):
        with self._environment():
            solver = self._solver()
            for _ in range(3):
                solver.solve(self.states)
            dav = replace(solver.settings, initial_method="chebdav")
            self.assertFalse(solver.stage_growth_start(self.grown, dav))
            self.assertEqual(solver.settings.initial_method, "chebdav")

    def test_a_failed_copy_grows_from_random_vectors(self):
        from parsec_python.acceleration.Eigensolvers import eigval

        def failing(vectors, keep):
            raise MemoryError("injected")

        with self._environment():
            solver = self._solver()
            for _ in range(3):
                solver.solve(self.states)
            with patch.object(eigval, "saved_columns", failing):
                self.assertFalse(solver.stage_growth_start(self.grown))
            self.assertIsNone(solver.device_state)
            self.assertEqual(solver.growth_record["failed"], 1)
            self._cold_equal(solver)

    def test_without_room_on_the_host_a_sector_grows_from_random_vectors(self):
        from parsec_python.acceleration.Eigensolvers import sector_growth

        with self._environment():
            solver = self._solver()
            for _ in range(3):
                solver.solve(self.states)
            # 257 x 12 doubles need 24,672 bytes: they fit in 49,344 available, not in 49,343.
            with patch.object(sector_growth, "host_available_bytes", lambda: 2 * 8 * self.rows * self.states - 1):
                self.assertFalse(solver.stage_growth_start(self.grown))
            self.assertEqual(solver.growth_record["no_room"], 1)
            self._cold_equal(solver)
            for _ in range(2):
                solver.solve(self.grown)
            with patch.object(sector_growth, "host_available_bytes", lambda: 2 * 8 * self.rows * self.grown):
                self.assertTrue(solver.stage_growth_start(self.grown + 1))
            self.assertEqual(solver.growth_record["saved"], 1)

    def test_a_basis_on_the_host_is_read_where_it_lies(self):
        with self._environment():
            solver = self._solver()
            for _ in range(12):
                solver.solve(self.states)
            solver.offload_state_to_host()
            self.assertIsInstance(solver.device_state.subspace.vectors, np.ndarray)
            solver.stage_growth_start(self.grown)
            result = solver.solve(self.grown)
        self.assertLess(np.abs(result.eigenvalues[: self.states] - self.exact[: self.states]).max(), 1e-4)

    def test_a_hand_over_basis_is_kept_by_its_span(self):
        with self._environment(PARSEC_CUPY_RITZ_HANDOVER="triangular", PARSEC_CUPY_RITZ_HANDOVER_COLUMNS="8"):
            solver = self._solver(capable=True)
            for _ in range(12):
                solver.solve(self.states)
            self.assertIsNotNone(solver.device_state.subspace.handover)
            self.assertTrue(solver.stage_growth_start(self.grown))
            result = solver.solve(self.grown)
        self.assertLess(np.abs(result.eigenvalues[: self.states] - self.exact[: self.states]).max(), 1e-4)

    def test_a_basis_that_devices_share(self):
        with self._environment(PARSEC_CUPY_DISTRIBUTED_STATE="1"):
            self.op.distributed_filter_devices = (0, 0)
            warm, result = self._errors(True)
            self.assertTrue(hasattr(result.vectors, "group"))
            cold, _ = self._errors(False)
        self.assertLess(warm, 0.05 * cold)

    def test_the_vectors_of_a_first_solve_alone_are_not_kept(self):
        with self._environment():
            solver = self._solver()
            solver.solve(self.states)
            self.assertFalse(solver.stage_growth_start(self.grown))
            self.assertIsNone(solver.device_state)
            grown = solver.solve(self.grown)
            fresh = self._solver().solve(self.grown)
            np.testing.assert_array_equal(grown.eigenvalues, fresh.eigenvalues)
            self.assertEqual(solver.device_state.solves_completed, 1)

    def test_a_grown_basis_counts_as_refined(self):
        with self._environment():
            solver = self._solver()
            solver.solve(self.states)
            solver.solve(self.states)
            self.assertTrue(solver.stage_growth_start(self.grown - 2))
            solver.solve(self.grown - 2)
            self.assertEqual(solver.device_state.solves_completed, 3)
            self.assertTrue(solver.stage_growth_start(self.grown))

    def test_a_reset_forgets_the_saved_columns(self):
        with self._environment():
            solver = self._solver()
            solver.solve(self.states)
            solver.solve(self.states)
            solver.stage_growth_start(self.grown)
            solver.reset()
            fresh = self._solver()
            np.testing.assert_array_equal(solver.solve(self.grown).eigenvalues, fresh.solve(self.grown).eigenvalues)


if __name__ == "__main__":
    unittest.main()
