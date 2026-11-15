"""Real MPI/CUDA audit verification; no SCF convergence claim."""
import json
import os
from pathlib import Path

import numpy as np


def main(output, expected_ranks=64):
    import cupy as cp
    from mpi4py import MPI
    from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
    from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup, SharedBasis
    from parsec_python.acceleration.Eigensolvers.debug_audit import shared_ritz_audit

    comm = MPI.COMM_WORLD
    if comm.size != expected_ranks:
        raise ValueError('Diagnostic validation must use the full allocation')
    rows, states = 4093, 509
    class Operator:
        def __init__(self):
            self.effective_potential = cp.linspace(-2., 3., rows)
        def apply_into(self, block, out):
            cp.multiply(self.effective_potential[:, None], block, out=out)
    operator = Operator()
    group = MPIShardGroup(comm, operator, transport='host')
    layout = group.column_layout(uniform_filter_blocks(states, 6, 10))
    columns = group.allocate_columns(rows, layout)
    initial = np.empty(columns.shape, order='F')
    for first, offset, count in layout.spans(comm.rank):
        for index in range(count):
            initial[:, offset + index] = np.random.default_rng(91 + first + index).normal(size=rows)
    columns.set(initial)
    values, basis, _ = group.orthonormal_ritz(operator, SharedBasis(group, columns, layout), None)
    before = basis.local.copy()
    indices = np.concatenate([np.arange(first, first + count) for first, _, count in layout.spans(comm.rank)])
    reference = cp.linalg.norm(operator.effective_potential[:, None] * before - before * cp.asarray(values[indices]), axis=0)
    maximum = max(comm.allgather(float(cp.max(reference)) if reference.size else 0.))
    previous = os.environ.get('PARSEC_DEBUG')
    try:
        os.environ['PARSEC_DEBUG'] = 'audit'
        report = shared_ritz_audit(group, basis, values, phase='64_rank_validation')
    finally:
        if previous is None:
            os.environ.pop('PARSEC_DEBUG', None)
        else:
            os.environ['PARSEC_DEBUG'] = previous
    unchanged = all(comm.allgather(bool(cp.array_equal(before, basis.local))))
    passed = (unchanged and report['checked_states'] == states and report['all_finite']
              and abs(maximum - report['maximum_residual_ry']) < 1e-11)
    if comm.rank == 0:
        Path(output).write_text(json.dumps(dict(passed=passed, bitwise_basis_unchanged=unchanged,
            ranks=comm.size, reference_maximum=maximum, audit=report), indent=2) + '\n')
        print('ALL PASSED debug audit' if passed else 'DEBUG AUDIT FAILED', flush=True)
    return 0 if passed else 1
