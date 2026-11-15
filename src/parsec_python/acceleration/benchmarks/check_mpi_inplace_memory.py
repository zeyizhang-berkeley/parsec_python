"""Actual MPI A/B of the two-share and one-share FP64 Ritz routes."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np


def main(output):
    import cupy as cp
    from mpi4py import MPI
    from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
    from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup, SharedBasis

    comm = MPI.COMM_WORLD
    rows, states = 131071, 257
    os.environ['PARSEC_MPI_SHARED_EXCHANGE_BYTES'] = str(8 * rows * 3)
    os.environ['PARSEC_DENSE_BUFFER_BYTES'] = str(1024 * 1024)

    class Peak(cp.cuda.MemoryHook):
        name = 'parsec-mpi-live-peak'
        def __init__(self):
            self.live = self.peak = 0
            self.allocations = {}
        def malloc_postprocess(self, **kw):
            if kw['mem_ptr']:
                self.allocations[kw['pmem_id']] = kw['mem_size']
                self.live += kw['mem_size']
                self.peak = max(self.peak, self.live)
        def free_postprocess(self, **kw):
            self.live -= self.allocations.pop(kw['pmem_id'], 0)

    class Operator:
        def __init__(self):
            self.effective_potential = cp.linspace(1., 5., rows)
        def apply_into(self, block, out):
            cp.multiply(self.effective_potential[:, None], block, out=out)

    def measure(route):
        os.environ['PARSEC_MPI_SHARED_RITZ'] = route
        pool, peak = cp.cuda.MemoryPool(), Peak()
        with cp.cuda.using_allocator(pool.malloc), peak:
            operator = Operator()
            group = MPIShardGroup(comm, operator, transport='host')
            layout = group.column_layout(uniform_filter_blocks(states, 6, 10))
            columns = group.allocate_columns(rows, layout)
            source = np.asfortranarray(np.random.default_rng(71 + comm.rank).normal(size=columns.shape))
            columns.set(source)
            pointer = columns.data.ptr
            values, basis, whitened = group.ritz(operator, SharedBasis(group, columns, layout), stages=False)
            cp.cuda.get_current_stream().synchronize()
            measured_peak = peak.peak
            rotated = cp.asnumpy(basis.local)
            reused = basis.local.data.ptr == pointer
            fallback = group.separate_row_blocks
            del columns, basis, whitened, group, operator
        gc.collect()
        pool.free_all_blocks()
        return np.asarray(values), rotated, dict(peak_cupy_live_bytes=measured_peak,
                                                reuses_columns=reused, separate_row_blocks=fallback)

    old_values, old_basis, old = measure('rounds')
    values, basis, new = measure('inplace')
    signs = np.sign(np.sum(old_basis * basis, axis=0))
    value_error = comm.allreduce(float(np.max(np.abs(values - old_values))), op=MPI.MAX)
    vector_error = comm.allreduce(float(np.max(np.abs(basis * signs - old_basis))), op=MPI.MAX)
    records = comm.allgather(dict(rank=comm.rank, baseline=old, optimized=new))
    passed = (value_error < 1e-10 and vector_error < 1e-8
              and all(r['optimized']['peak_cupy_live_bytes'] < r['baseline']['peak_cupy_live_bytes']
                      and r['optimized']['reuses_columns'] and not r['optimized']['separate_row_blocks'] for r in records))
    if comm.rank == 0:
        record = dict(rows=rows, states=states, ranks=comm.size, dtype='float64', passed=passed,
                      max_eigenvalue_difference=value_error, max_vector_difference_after_signs=vector_error,
                      baseline_peak_bytes=max(r['baseline']['peak_cupy_live_bytes'] for r in records),
                      optimized_peak_bytes=max(r['optimized']['peak_cupy_live_bytes'] for r in records), ranks_detail=records,
                      metric='CuPy live allocations during full MPI Ritz step; excludes CUDA context/library memory')
        Path(output).write_text(json.dumps(record, indent=2) + '\n')
        print('MPI_INPLACE_MEMORY', json.dumps(record), flush=True)
    return 0 if passed else 1
