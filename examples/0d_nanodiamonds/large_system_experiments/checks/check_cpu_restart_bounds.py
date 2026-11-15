"""Work/accuracy probe; CPU fixtures are not large-GPU performance results."""
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import LinearOperator

from parsec_python.Eigensolvers import (
    EigvalSettings, ChebFFSettings, SubspaceSettings, solve_eigval,
)
from parsec_python.SCF.restart_refinement import RestartRefinementSettings, guarded_solver


def main():
    reports = []
    for complex_basis in (False, True):
        rng = np.random.default_rng(37)
        matrix = rng.standard_normal((128, 128))
        if complex_basis:
            matrix = matrix + 1j * rng.standard_normal(matrix.shape)
        unitary, _ = np.linalg.qr(matrix)
        for shift in (0., -15.):
            spectrum = np.r_[np.linspace(-2., -.2, 12), np.linspace(-.05, 156., 116)] + shift
            hamiltonian = (unitary * spectrum) @ unitary.conj().T
            reference = np.linalg.eigvalsh(hamiltonian)[:8]
            for rule in ("inherit", "smart"):
                work = {"hamiltonian_vector_products": 0}
                def apply(vectors):
                    work["hamiltonian_vector_products"] += 1 if vectors.ndim == 1 else vectors.shape[1]
                    return hamiltonian @ vectors
                operator = LinearOperator(hamiltonian.shape, dtype=hamiltonian.dtype,
                                          matvec=apply, matmat=apply)
                settings = EigvalSettings(safety_buffer=4,
                    chebff=ChebFFSettings(filter_cycles=3, polynomial_degree=15, block_size=4),
                    subspace=SubspaceSettings(polynomial_degree=15, block_size=4))
                recorded = []
                def record(event, **fields):
                    if event == "restart_level_refinement":
                        recorded.append(fields)
                started = perf_counter()
                with patch("parsec_python.SCF.restart_refinement.debug.emit", side_effect=record):
                    result = guarded_solver(solve_eigval, RestartRefinementSettings(
                        1e-9, 80, lower_bound_rule=rule))(operator, 8, settings=settings)
                seconds = perf_counter() - started
                # Independent dense check is outside the work counter/timer.
                residual = float(np.linalg.norm(hamiltonian @ result.vectors -
                    result.vectors * result.eigenvalues, axis=0).max())
                error = float(np.abs(result.eigenvalues - reference).max())
                assert error < 2e-8 and residual < 5e-5, (error, residual)
                assert recorded[-1]["passed"]
                reports.append(dict(complex_basis=complex_basis, potential_shift_ry=shift,
                    lower_bound_rule=rule, passes=recorded[-1]["passes"],
                    seconds=seconds, level_error_ry=error, maximum_residual_ry=residual,
                    **work))
    output = dict(passed=True, scope="CPU fixed-Hamiltonian work/accuracy probe; not GPU speedup",
                  fixture_max_passes=80, gpu_candidate_max_passes=40, reports=reports)
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
