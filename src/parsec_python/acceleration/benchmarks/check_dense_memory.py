"""Compare the pinned and optimized FP64 Ritz solve on one actual GPU.

Reports CuPy live allocation peaks (including input matrices and solver work,
excluding CUDA context/library allocations). Each solve uses a fresh pool.
The caller supplies the hash-verified baseline rayleigh_ritz.py.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=4096)
    args = parser.parse_args()
    import cupy as cp
    from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import solve_whitened_ritz_on_device
    name = 'parsec_python.acceleration.Eigensolvers._reference_ritz'
    spec = importlib.util.spec_from_file_location(name, args.reference)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(73)
    n = args.states
    overlap = rng.normal(size=(n, n)) * (0.1 / n)
    overlap = overlap + overlap.T + np.eye(n)
    projection = rng.normal(size=(n, n)) / np.sqrt(n)
    projection = projection + projection.T

    class Peak(cp.cuda.MemoryHook):
        name = 'parsec-live-allocation-peak'
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

    def measure(solve):
        pool = cp.cuda.MemoryPool()
        hook = Peak()
        with cp.cuda.using_allocator(pool.malloc), hook:
            s, h = cp.asarray(overlap), cp.asarray(projection)
            values, coefficients, whitened = solve(s, h)
            cp.cuda.get_current_stream().synchronize()
            peak = hook.peak
            # The residual is measured after the solve's peak, independently
            # of its existing condition and orthogonality audits.
            residual = float(cp.abs(h @ coefficients - (s @ coefficients) * values).max().item())
            result = cp.asnumpy(values), cp.asnumpy(coefficients)
            del s, h, values, coefficients, whitened
        gc.collect()
        pool.free_all_blocks()
        return result, dict(peak_cupy_live_bytes=peak, max_eigen_residual=residual)

    (old_values, old_vectors), old = measure(reference.solve_whitened_ritz_on_device)
    (values, vectors), new = measure(lambda s, h: solve_whitened_ritz_on_device(s, h, overwrite_grams=True))
    signs = np.sign(np.sum(old_vectors * vectors, axis=0))
    value_error = float(np.max(np.abs(values - old_values)))
    vector_error = float(np.max(np.abs(vectors * signs - old_vectors)))
    passed = (value_error < 1e-10 and vector_error < 1e-8
              and max(old['max_eigen_residual'], new['max_eigen_residual']) < 1e-10
              and new['peak_cupy_live_bytes'] < old['peak_cupy_live_bytes'])
    record = dict(states=n, dtype='float64', baseline=old, optimized=new, overwrite_grams=True,
                  max_eigenvalue_difference=value_error, max_vector_difference_after_signs=vector_error,
                  saved_live_bytes=old['peak_cupy_live_bytes'] - new['peak_cupy_live_bytes'], passed=passed,
                  metric='CuPy live allocations; includes input matrices; excludes CUDA context/library allocations')
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
