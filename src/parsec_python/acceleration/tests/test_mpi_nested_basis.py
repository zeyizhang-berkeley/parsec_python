"""A sector basis shared by ranks over all their GPUs must reproduce one device (GPU tests).

Two ranks of two GPUs each (threads with a fake communicator, devices 0-1 and 2-3 of one node) solve a small
Hamiltonian with the CHEBFF first solve and later passes of :mod:`distributed_state` on a
:class:`mpi_nested_basis.NestedShardGroup`, under both transports and the block and rows routes of the Ritz step;
the Ritz values and the subspace are those of one device.  Also: the DLARNV columns of every device, the density
over the devices, the helpers of the small solve kept on the root's rank, a failing device thread that releases
the others, and one rank of two GPUs (``R = 1``).
"""

from __future__ import annotations

import copy
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy


def _device_count():
    if not cupy_available():
        return 0
    cp, _ = require_cupy()
    return int(cp.cuda.runtime.getDeviceCount())


_COMMON = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_FILTER='0',
               PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1', PARSEC_CUPY_RITZ_ROTATION='reuse',
               PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1', PARSEC_CUPY_STREAMING_RITZ='0', PARSEC_CUPY_DEVICE_RANDOM='1')
_TIMEOUT = 300.0


def _hamiltonian(n):
    """A small Hamiltonian with projectors on the current device."""
    return CuPyHamiltonian(sp.diags((-np.ones(n-1), 2.2*np.ones(n), -np.ones(n-1)), (-1, 0, 1)),
                           np.linspace(-.1, .1, n),
                           (sp.csr_matrix(np.random.default_rng(8).normal(size=(n, 2))*.002), np.array([1., -1.])),
                           retain_generic_laplacian=False)


def _host(array):
    if hasattr(array, '__cuda_array_interface__'):
        cp, _ = require_cupy()
        with cp.cuda.Device(int(array.device.id)):
            return cp.asnumpy(array)
    return np.array(array, copy=True)


def _set(buffer, values):
    values = np.asarray(values).reshape(buffer.shape)
    if hasattr(buffer, '__cuda_array_interface__'):
        cp, _ = require_cupy()
        with cp.cuda.Device(int(buffer.device.id)):
            buffer.set(np.ascontiguousarray(values))
            cp.cuda.get_current_stream().synchronize()
    else:
        buffer[...] = values


class _World:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=_TIMEOUT)
        self.slots = [None] * size
        self.mail = {}
        self.condition = threading.Condition()


class _Done:
    def Wait(self):
        return None


class _Pending:
    def __init__(self, comm, buffer, source, tag):
        self.comm, self.buffer, self.key = comm, buffer, (comm.space, int(source), comm.rank, int(tag))

    def Wait(self):
        world = self.comm.world
        with world.condition:
            if not world.condition.wait_for(lambda: world.mail.get(self.key), timeout=_TIMEOUT):
                raise TimeoutError(f'no message {self.key}')
            data = world.mail[self.key].pop(0)
        _set(self.buffer, data)


class _RankComm:
    """The part of an mpi4py communicator that a nested hub calls, between threads that stand for ranks."""

    def __init__(self, world, rank, space=0):
        self.world, self.rank, self.size, self.space = world, rank, world.size, space
        self._dups = 0

    def _post(self, value):
        self.world.slots[self.rank] = value
        self.world.barrier.wait()
        values = list(self.world.slots)
        self.world.barrier.wait()
        return values

    def bcast(self, value, root=0):
        return copy.deepcopy(self._post(value if self.rank == root else None)[root])

    def allgather(self, value):
        return [copy.deepcopy(item) for item in self._post(value)]

    def Bcast(self, buffer, root=0):
        values = self._post(_host(buffer) if self.rank == root else None)
        if self.rank != root:
            _set(buffer, values[root])

    def Allreduce(self, send, receive, op=None):
        values = self._post(_host(send))
        total = np.array(values[0], copy=True)
        for value in values[1:]:
            total = total + value
        _set(receive, total)

    def Dup(self):
        self._dups += 1
        return _RankComm(self.world, self.rank, space=self.space * 100 + self._dups)

    def Isend(self, buffer, dest, tag=0):
        data = _host(buffer)
        with self.world.condition:
            self.world.mail.setdefault((self.space, self.rank, int(dest), int(tag)), []).append(data)
            self.world.condition.notify_all()
        return _Done()

    def Irecv(self, buffer, source, tag=0):
        return _Pending(self, buffer, source, tag)


def _later_state(first):
    """The state of the later passes after a first solve, as ``eigval._state_from_first`` makes it."""
    from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState
    return DeviceSubspaceState(operator_dimension=first.state.operator_dimension,
                               working_states=first.state.wanted_states, eigenvalues=first.state.eigenvalues,
                               vectors=first.state.vectors, filter_lower_bound=None, first_filter=True,
                               filters_completed=0)


def _run_ranks(size, function):
    world = _World(size)
    with ThreadPoolExecutor(max_workers=size) as pool:
        futures = [pool.submit(function, rank, _RankComm(world, rank)) for rank in range(size)]
        try:
            return [future.result(timeout=_TIMEOUT) for future in futures]
        except BaseException:
            world.barrier.abort()
            raise


def _whole(basis):
    """The columns of a nested basis that this rank holds: ``{column: values}``."""
    cp, _ = require_cupy()
    held = {}
    for index, block in enumerate(basis.local_blocks):
        virtual = basis.group.rank + index
        with cp.cuda.Device(int(block.device.id)):
            values = cp.asnumpy(block)
        for column, offset, count in basis.layout.spans(virtual):
            for k in range(count):
                held[column + k] = values[:, offset + k]
    return held


def _assemble(parts, columns):
    held = {}
    for part in parts:
        held.update(part)
    return np.stack([held[column] for column in range(columns)], axis=1)


def _same_subspace(testcase, expected, actual, tolerance=1e-8):
    values = np.linalg.svd(expected.T @ actual, compute_uv=False)
    np.testing.assert_allclose(values, np.ones_like(values), atol=tolerance)


@unittest.skipUnless(_device_count() >= 4, 'four CUDA devices required')
class NestedGroupTests(unittest.TestCase):
    rows = 1500
    states = 40

    @classmethod
    def setUpClass(cls):
        cls.cp, _ = require_cupy()

    def _reference(self, settings, later):
        import importlib
        from parsec_python.acceleration.Eigensolvers.subspace import SubspaceSettings, run_subspace_filter
        run_chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp = self.cp
        with cp.cuda.Device(0), patch.dict(os.environ, _COMMON):
            operator = _hamiltonian(self.rows)
            first = run_chebff(operator, self.states, settings=settings)
            passes = [(cp.asnumpy(first.eigenvalues), cp.asnumpy(first.vectors))]
            state = _later_state(first)
            for _ in range(later):
                result = run_subspace_filter(operator, state, settings=SubspaceSettings(), compute_residuals=False)
                passes.append((cp.asnumpy(result.eigenvalues), cp.asnumpy(result.vectors)))
                state = result.state
        return passes

    def _nested(self, settings, later, *, ranks=2, local=2, transport='cuda', route='block', environment=None):
        from parsec_python.acceleration.Eigensolvers.subspace import SubspaceSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import (
            run_distributed_chebff, run_distributed_subspace)
        from parsec_python.acceleration.Eigensolvers.mpi_nested_basis import attach_nested_group
        cp = self.cp

        def rank_run(rank, comm):
            devices = tuple(range(rank * local, rank * local + local))
            with cp.cuda.Device(devices[0]):
                operator = _hamiltonian(self.rows)
                group = attach_nested_group(operator, comm, devices, ritz_route=route, transport=transport)
                first = run_distributed_chebff(operator, self.states, settings=settings, group=group)
                passes = [(cp.asnumpy(first.eigenvalues), _whole(first.vectors))]
                state = _later_state(first)
                for _ in range(later):
                    result = run_distributed_subspace(operator, state, settings=SubspaceSettings(),
                                                      compute_residuals=False, group=group)
                    passes.append((cp.asnumpy(result.eigenvalues), _whole(result.vectors)))
                    state = result.state
                helpers = (group._groups[0]._condition_helper(), group._groups[0]._eigh_helper())
                record = dict(passes=group.passes, seconds=group.seconds, remote=group.remote_bytes,
                              exchanged=group.exchanged_bytes, helpers=helpers)
                group.hub.close()
                return passes, record

        with patch.dict(os.environ, {**_COMMON, **(environment or {})}):
            return _run_ranks(ranks, rank_run)

    def _compare(self, reference, results, atol=1e-10):
        for step, (values, vectors) in enumerate(reference):
            for rank_passes, _record in results:
                np.testing.assert_allclose(rank_passes[step][0], values, rtol=0, atol=atol)
            whole = _assemble([rank_passes[step][1] for rank_passes, _record in results], self.states)
            np.testing.assert_allclose(whole.T @ whole, np.eye(self.states), atol=1e-9)
            _same_subspace(self, vectors, whole)

    def test_two_ranks_of_two_gpus_reproduce_one_device(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        reference = self._reference(settings, later=2)
        for transport in ('cuda', 'host'):
            for route in ('block', 'rows'):
                with self.subTest(transport=transport, route=route):
                    results = self._nested(settings, 2, transport=transport, route=route)
                    self._compare(reference, results)
                    for _passes, record in results:
                        self.assertEqual(record['passes'], 4 + 2)
                        self.assertGreater(record['remote'], 0)
                        # The condition helper is the root's second device; the eigh helper would be rank 1's
                        # first, outside a collective: the root solves itself.
                        self.assertEqual(record['helpers'][0], 1)
                        self.assertIsNone(record['helpers'][1])

    def test_one_rank_of_two_gpus_reproduces_one_device(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=3)
        reference = self._reference(settings, later=1)
        results = self._nested(settings, 1, ranks=1, local=2)
        self._compare(reference, results)
        self.assertEqual(results[0][1]['remote'], 0)

    def test_one_rank_of_four_gpus_keeps_both_helpers_on_its_devices(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=3)
        reference = self._reference(settings, later=1)
        results = self._nested(settings, 1, ranks=1, local=4)
        self._compare(reference, results)
        self.assertEqual(results[0][1]['helpers'], (1, 2))

    def test_random_basis_and_density_of_every_device(self):
        from parsec_python.Eigensolvers.lapack_random import LapackRandom as HostRandom
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.lapack_random import LapackRandom
        from parsec_python.acceleration.Eigensolvers.mpi_nested_basis import attach_nested_group
        cp = self.cp
        rows, states = 300, 24
        expected = HostRandom().uniform_minus_1_1((rows, states), column_major=True)
        weights = np.linspace(2.0, 0.5, 17)

        def builder(block, held, volume):
            return cp.asnumpy((block * block) @ cp.asarray(held)) * volume

        def rank_run(rank, comm):
            devices = (2 * rank, 2 * rank + 1)
            with cp.cuda.Device(devices[0]):
                operator = _hamiltonian(rows)
                group = attach_nested_group(operator, comm, devices, ritz_route='block')
                generator = LapackRandom()
                basis = group.random_basis(generator, rows, group.column_layout(uniform_filter_blocks(states, 4, 3)))
                after = generator.seed
                density = basis.density(builder, weights, 0.5)
                leading = basis[:, :10]
                group.hub.close()
                return _whole(basis), after, density, leading.shape, leading.nbytes

        with patch.dict(os.environ, _COMMON):
            results = _run_ranks(2, rank_run)
        np.testing.assert_array_equal(_assemble([result[0] for result in results], states), expected)
        stream = HostRandom()
        stream.uniform_minus_1_1((rows, states), column_major=True)
        self.assertEqual({result[1] for result in results}, {stream.seed})
        total = results[0][2] + results[1][2]
        np.testing.assert_allclose(total, ((expected[:, :17] ** 2) @ weights) * 0.5, rtol=1e-13, atol=1e-15)
        self.assertEqual({result[3] for result in results}, {(rows, 10)})

    def test_a_failing_device_releases_the_others(self):
        from parsec_python.acceleration.Eigensolvers.mpi_nested_basis import _HubFailure, _NodeHub
        world = _World(1)
        hub = _NodeHub(_RankComm(world, 0), (0, 1, 2), 'host')

        def fail():
            raise ValueError('device trouble')

        def wait_in_collective(index):
            return lambda: hub.call(index, 'bcast', (index, 0))

        def wait_for_message():
            return hub.receive(2, 'object', None, 0, 5)

        outcomes = hub.run([fail, wait_in_collective(1), wait_for_message])
        self.assertIsInstance(outcomes[0][1], ValueError)
        self.assertIsInstance(outcomes[1][1], _HubFailure)
        self.assertIsInstance(outcomes[2][1], _HubFailure)
        # The hub serves the next step.
        outcomes = hub.run([wait_in_collective(0), wait_in_collective(1), wait_in_collective(2)])
        self.assertEqual([value for _ok, value in outcomes], [0, 0, 0])
        # A thread that finishes while the others wait in a collective ends their waits.
        outcomes = hub.run([lambda: 7, wait_in_collective(1), wait_in_collective(2)])
        self.assertEqual(outcomes[0], (True, 7))
        self.assertIsInstance(outcomes[1][1], _HubFailure)
        hub.close()


if __name__ == '__main__':
    unittest.main()
