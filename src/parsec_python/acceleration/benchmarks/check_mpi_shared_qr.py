"""Same-allocation A/B of robust shared-basis QR, with unchanged FP64 Ritz.

Run under MPI with one visible GPU per rank. Memory is live CuPy allocation
usage, not device-wide NVML usage. The check deliberately reaches the QR
fallback; it does not replace full-SCF energy/density validation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def main(output, *, rows=524287, states=509, expected_ranks=64, repeats=2):
    import cupy as cp
    from mpi4py import MPI

    from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
    from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import MPIShardGroup, SharedBasis
    from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import GeneralizedRitzStabilityError

    comm = MPI.COMM_WORLD
    if comm.size != expected_ranks:
        raise ValueError(f"expected {expected_ranks} MPI ranks, received {comm.size}")
    if rows < states or states < 4 or repeats < 1:
        raise ValueError("need rows >= states >= 4 and at least one A/B repeat")
    os.environ["PARSEC_MPI_SHARED_RITZ"] = "inplace"
    os.environ["PARSEC_MPI_SHARED_EXCHANGE_BYTES"] = str(8 * rows)
    os.environ["PARSEC_DENSE_BUFFER_BYTES"] = str(1024**2)

    class Peak(cp.cuda.MemoryHook):
        name = "parsec-shared-qr-live-peak"

        def __init__(self):
            self.live = self.peak = 0
            self.allocations = {}

        def malloc_postprocess(self, **kw):
            if kw["mem_ptr"]:
                self.allocations[kw["pmem_id"]] = kw["mem_size"]
                self.live += kw["mem_size"]
                self.peak = max(self.peak, self.live)

        def free_postprocess(self, **kw):
            self.live -= self.allocations.pop(kw["pmem_id"], 0)

    class Operator:
        def __init__(self):
            self.effective_potential = cp.linspace(1., 5., rows)

        def apply_into(self, block, out):
            cp.multiply(self.effective_potential[:, None], block, out=out)

    def measure(route):
        os.environ["PARSEC_MPI_SHARED_QR"] = route
        pool, peak = cp.cuda.MemoryPool(), Peak()
        with cp.cuda.using_allocator(pool.malloc):
            with peak:
                operator = Operator()
                group = MPIShardGroup(comm, operator, transport="host")
                layout = group.column_layout(uniform_filter_blocks(states, 6, 10))
                columns = group.allocate_columns(rows, layout)
                local = np.empty(columns.shape, dtype=np.float64, order="F")
                for first, offset, count in layout.spans(comm.rank):
                    for index in range(count):
                        column = first + index
                        values = np.random.default_rng(71 + column).normal(size=rows)
                        if column == 3:
                            values = np.random.default_rng(72).normal(size=rows) + 1e-6 * values
                        local[:, offset + index] = values
                columns.set(local)
                pointer = columns.data.ptr
                filtered = SharedBasis(group, columns, layout)
                refused = False
                try:
                    group.ritz(operator, filtered, stages=False)
                except GeneralizedRitzStabilityError as error:
                    filtered, refused = error.filtered, True
                if not all(comm.allgather(refused)):
                    raise AssertionError("the ill-conditioned input must reach the robust fallback on every rank")
                # Match the production caller: leave the exception handler
                # before QR so its traceback does not retain failed Ritz work.
                gc.collect()
                cp.cuda.get_current_stream().synchronize()
                peak.peak = peak.live
                comm.Barrier()
                started = perf_counter()
                eigenvalues, basis, whitened = group.orthonormal_ritz(operator, filtered, None)
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter() - started
                memory = peak.peak
                rotated = cp.asnumpy(basis.local)
                reused = basis.local.data.ptr == pointer
                separate = group.separate_row_blocks
            # Accuracy diagnostics allocate additional storage only after the
            # measured phase. Validate both orthogonality and Galerkin residual.
            row_basis = group.to_rows(basis.local, layout, rows)
            # Use the solver's row split rather than assuming divisible rows.
            from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import row_edges
            edges = row_edges(rows, comm.size)
            potential = operator.effective_potential[edges[comm.rank]:edges[comm.rank + 1]]
            grams = group._sum(cp.ascontiguousarray(cp.stack((
                row_basis.T @ row_basis,
                row_basis.T @ (row_basis * potential[:, None])))))
            overlap, projected = cp.asnumpy(grams)
            orthogonality = float(np.max(np.abs(overlap - np.eye(states))))
            galerkin = float(np.max(np.abs(projected - overlap * np.asarray(eigenvalues))))
            del columns, basis, whitened, filtered, row_basis, grams, operator, group
        gc.collect()
        pool.free_all_blocks()
        return np.asarray(eigenvalues), rotated, dict(
            seconds=float(comm.allreduce(elapsed, op=MPI.MAX)),
            peak_cupy_live_bytes=int(comm.allreduce(memory, op=MPI.MAX)),
            reuses_columns=all(comm.allgather(reused)), separate_row_blocks=sum(comm.allgather(separate)),
            orthogonality_max_abs=orthogonality, galerkin_residual_max_abs=galerkin)

    records = []
    passed = True
    for repeat in range(repeats):
        # Reverse A/B order on alternate repeats to expose warmup/order bias.
        sequence = ("rows", "inplace") if repeat % 2 == 0 else ("inplace", "rows")
        pair = {route: measure(route) for route in sequence}
        old_values, old_basis, old = pair["rows"]
        values, basis, new = pair["inplace"]
        signs = np.sign(np.sum(old_basis * basis, axis=0))
        eigenvalue_error = float(comm.allreduce(float(np.max(np.abs(values - old_values))), op=MPI.MAX))
        local_error = float(np.max(np.abs(basis * signs - old_basis))) if basis.size else 0.
        vector_error = float(comm.allreduce(local_error, op=MPI.MAX))
        accuracy = (eigenvalue_error < 1e-10 and vector_error < 1e-8
                    and all(r["orthogonality_max_abs"] < 5e-10 and r["galerkin_residual_max_abs"] < 1e-9
                            for r in (old, new)))
        improved_memory = new["peak_cupy_live_bytes"] < old["peak_cupy_live_bytes"]
        passed &= accuracy and improved_memory and new["reuses_columns"] and not new["separate_row_blocks"]
        records.append(dict(repeat=repeat, order=sequence, baseline=old, candidate=new,
                            eigenvalue_max_abs_difference=eigenvalue_error,
                            vector_max_abs_difference_after_signs=vector_error,
                            candidate_to_baseline_memory_ratio=new["peak_cupy_live_bytes"] / old["peak_cupy_live_bytes"],
                            candidate_to_baseline_time_ratio=new["seconds"] / old["seconds"]))
    if comm.rank == 0:
        result = dict(rows=rows, states=states, ranks=comm.size, dtype="float64", passed=bool(passed),
                      repeats=records, scope="robust QR plus subsequent Ritz, not full SCF",
                      metric="live CuPy allocations; excludes CUDA context/library memory",
                      adoption="requires full-SCF precision and time/memory tradeoff review")
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        print("MPI_SHARED_QR", json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=524287)
    parser.add_argument("--states", type=int, default=509)
    parser.add_argument("--expected-ranks", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.output, rows=arguments.rows, states=arguments.states,
                          expected_ranks=arguments.expected_ranks, repeats=arguments.repeats))
