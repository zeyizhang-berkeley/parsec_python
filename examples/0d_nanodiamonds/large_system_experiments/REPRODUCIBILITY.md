# Inputs, source identity, runtime and verification

## 1. Source identity

The baseline Git commit is `2c961072e0232ab1a41f85be481bb2f8876f6fc5`.
This branch contains memory/communicator changes, switchable diagnostics,
fixed-potential reconstruction and the smart-bound candidate. At publication,
the SHA256 values of 27 changed source/test files matched the measured GPU
bundle; see [provenance.json](data/provenance.json). Those hashes describe the
historical measured code, before the subsequent default-limit change to 100.
They are preserved rather than relabeled as evidence for the new defaults.

Historical labels such as `2c96107+memory-…`, `+qr-…`, `+restartfix-…` and
`+restartbounds-…` identify frozen snapshots, not Git commits that can be checked
out by those names. The historical measured `smart` bundle SHA256 is
`9f332b0cb1b01756953c44d7f0940e01cf695aeacd4776975cfc7227ae5dd6c2`;
the GB200 runtime SHA256 is
`59067e1e373f0a5e5114cc47c0ad6edcf708ac13ff3eae2248d241fb5ed8b4b8`.
Enabling every change in this branch does not reproduce every historical version.

That `9f332b…` bundle is not the latest source state. The subsequent 100-limit
scientific bundle is `7ebe86ec8f56d10c7a23e224fe5ac7b694cdeaed16ee9e27897c3a177e3521b0`.
The completed communication-only probe used its unchanged scientific source with
a separate adapter bundle; its hashes, acceptance evidence and scope are in
[COMMUNICATION_DIAGNOSTIC.md](COMMUNICATION_DIAGNOSTIC.md) and
[communication_probe_20261115.json](data/communication_probe_20261115.json).
The newer optional MPI tracing code has targeted CPU validation, while its
large-memory GPU validation remains pending. Do not attribute the earlier GPU
results to that instrumentation.

| Stage | Changes | Evidence boundary |
|---|---|---|
| base | Baseline Git revision | Existing benchmarks and H100 control without symmetry |
| memory-v1 | Dense-intermediate reuse, sector communicators | Full-SCF memory decrease; timing trade-off not fully isolated |
| memory-v2 | MPI Ritz layout reuse within one allocation | Module accuracy, capacity and short SCF checks |
| QR candidate | Shared-storage robust QR | Module/short-SCF checks passed; large-system validation unfinished |
| restart levels | Additional fixed-potential solves | Historical GPU bundle: original accuracy thresholds, at most 40 extra passes |
| restart smart | Smart lower bound during extra reconstruction | 95,648-electron restart converged; matched A/B and array review open |

## 2. Structures and inputs

The historical construction procedure is in [structures/README.md](../structures/README.md).
Larger clusters use the same diamond-fragment generator, shared dangling-site
completion and hydrogen passivation. These commands generate structures/inputs
without running SCF:

```bash
python examples/0d_nanodiamonds/structures/generate_nanodiamonds.py \
  --output-root /tmp/parsec_generated_clusters \
  --maximum-carbon-atoms 31000 --maximum-multiplier 21

python examples/3d_diamond_benchmark/make_diamond.py \
  13 13 13 /tmp/parsec_gamma13.in
python examples/3d_diamond_benchmark/make_diamond.py \
  10 10 10 /tmp/parsec_k2_10.in --kmesh 2 2 2
```

Freshly generated solver defaults are not the original historical inputs. For
exact input review, use the frozen `parsec.in` files in [inputs/](inputs/) and
verify [inputs.json](data/inputs.json). The k11 H100 comparison selected
`--symmetry off` in the launcher. Its input's `Ignore_Symmetry: false` alone does
not determine the execution mode; launcher settings must also be recorded.

Use the pseudopotentials already in the repository:

| Purpose | File | SHA256 |
|---|---|---|
| Cluster C | `examples/0d_nanodiamonds/C_POTRE.DAT` | `2e073cb81f66327a15d82a3955b8822c72f13494d31ec51799b2b3c122598c8c` |
| Cluster H | `examples/0d_nanodiamonds/H_POTRE.DAT` | `202df2aba56d7aa99a515909add249b643cde27f1aff67bc79b4ceeebf2af191` |
| Periodic PBE C | `examples/3d_diamond_benchmark/C_POTRE.DAT` | `d6d64ac88cd8a72e6593f5cd8be7e78841d891abab9f758852e3727117844784` |

## 3. Runtime switches

All MPI ranks must receive consistent settings. Installation follows the main
repository README. The site launcher must bind one MPI rank to one GPU for the
shared MPI route; 64 ranks must not accidentally select the same device.

| Variable | Use in this campaign | Scope |
|---|---|---|
| `PARSEC_MPI_SHARED_RITZ=inplace` | Reuse column/row basis storage | `rounds`/`rows` controls remain; library `auto` default unchanged |
| `PARSEC_MPI_SHARED_QR=inplace` | Robust-QR candidate | Default `rows`; real shared-basis path validated so far |
| `PARSEC_DENSE_BUFFER_BYTES=67108864` | Bounded dense scratch | Per buffer, not total GPU memory |
| `PARSEC_MPI_SHARED_TRANSPORT=host` | Host-staged MPI | CUDA path needs separate environment/numerical validation |
| `PARSEC_DEBUG=off/summary/audit` | Switchable diagnostics | Audit adds Hq, reductions and synchronization |
| `PARSEC_DEBUG_RANKS=leaders` | Sector/pool leader records | Retain each leader when collecting logs |
| `PARSEC_MPI_TRACE=1` | Existing exchange-stage and buffer-layout metadata | Default off; all ranks, independent of debug leader selection; adds logging, not MPI calls or device synchronization |
| `PARSEC_RESTART_REFINE=off/levels` | Fixed-potential reconstruction | Default off; requires the actual `load_restart` path |
| `PARSEC_RESTART_REFINE_MAX_PASSES=100` | Current default for extra reconstruction passes | Explicit values override it; failure stops before accepting an unqualified density |
| `PARSEC_RESTART_REFINE_LOWER_BOUND=inherit/smart` | Extra-pass lower-bound rule | Default inherit; SCF accuracy threshold unchanged |

The current default for ordinary SCF is also 100 iterations when `Max_Iter` is
omitted. These are independent limits: `Max_Iter` counts density/potential SCF
updates, whereas `PARSEC_RESTART_REFINE_MAX_PASSES` bounds additional solves at
the fixed loaded potential before the first resumed density is accepted.
The restart policy still requires two consecutive stable passes at the input
`Diag_Tolerance`; the final SCF criterion is unchanged. Existing explicit input
limits and the historical 40-pass measurements are unchanged. The higher caps
allow more work and do not establish convergence or a performance improvement.

Optional MPI trace records contain metadata rather than orbital values, with a
cache of at most 256 layout signatures per group. Preserve per-rank records and
source identity when diagnosing a missing exchange stage. Disable tracing for
performance comparisons after correctness is established; metadata processing
and flushed output add overhead. See the [communication diagnostic](COMMUNICATION_DIAGNOSTIC.md)
for the completed probe and the limits of its evidence.

For an already allocated GPU pool, the portable fresh cluster/Gamma entry point
is `parsec_python.acceleration.benchmarks.mpi_full_scf`. For example, on a Slurm
allocation with GPU binding and 16×4 GB200s, run from the repository root:

```bash
export PYTHONPATH="$PWD/src"
export PARSEC_MPI_SHARED_RITZ=inplace
export PARSEC_MPI_SHARED_QR=inplace
export PARSEC_DENSE_BUFFER_BYTES=67108864
export PARSEC_MPI_SHARED_TRANSPORT=host
export PARSEC_DEBUG=summary

srun --nodes=16 --ntasks-per-node=4 --gpus-per-task=1 \
  python -m parsec_python.acceleration.benchmarks.mpi_full_scf \
  --input examples/0d_nanodiamonds/large_system_experiments/inputs/cluster/k19_C23191H2884/parsec.in \
  --output-dir /tmp/parsec_k19_run --devices 0 --symmetry on \
  --pp-dir examples/0d_nanodiamonds
```

This is a site-dependent launcher example, not the TaskKit submission used in the
campaign or a complete performance-reproduction configuration. Record native/GPU
backends, BLAS/OpenMP threads, CPU binding, GPU-aware MPI, cache placement and
checkpoint I/O when measuring performance. The actual campaign froze these settings.
For the complex-k-point MPI entry point, see the
[periodic benchmark](../../3d_diamond_benchmark/README.md).

**Do not use `--restart-from parsec.dat` for density/potential recovery.** That
benchmark flag is for coordinate/geometry restart. The diagnostic adapter prepares
the same system, calls `SCF.restart.load_restart`, validates the parent hash/grid/
input/GPU/runtime identity and explicitly passes the resulting `SCFWarmStart` to
SCF. The essential API form is:

```python
from parsec_python.SCF.restart import load_restart

fields, warm = load_restart(checkpoint_path, prepared_system)
assert not (fields.extrapolate or fields.missing_points or fields.dropped_points)
# Pass warm_start=warm to the matching prepared-system SCF driver.
# The MPI adapter must broadcast the same problem and policy to all ranks.
```

Enabling `levels` without using the recovery entry point does not trigger
reconstruction. Multi-GB parent checkpoints are not distributed here; their
hashes, scientific inputs and runtime identities allow holders to verify them.
The independent CPU fixtures run without those checkpoint files.

## 4. Executed validation and commands

The publication CPU regression used the same frozen scientific source as the GPU
candidate: **161 tests, 134 passed, 27 skipped, zero failures** across 12 modules.
The module list and unittest summary are in [provenance.json](data/provenance.json).
Skipped checks are not GPU passes.

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -m unittest \
  parsec_python.acceleration.tests.test_dense_memory \
  parsec_python.acceleration.tests.test_ritz_dense \
  parsec_python.acceleration.tests.test_mpi_shared_basis \
  parsec_python.acceleration.tests.test_mpi_shared_qr \
  parsec_python.acceleration.tests.test_mpi_scf \
  parsec_python.acceleration.tests.test_kpoint_pools \
  parsec_python.acceleration.tests.test_column_ritz \
  parsec_python.acceleration.tests.test_debug_audit \
  parsec_python.tests.unit.test_restart_refinement \
  parsec_python.tests.unit.test_diagnostics \
  parsec_python.tests.unit.test_strict_scf_eigensolver \
  parsec_python.tests.unit.test_parsec_dat_io

python examples/0d_nanodiamonds/large_system_experiments/checks/check_cpu_restart_bounds.py
python examples/0d_nanodiamonds/large_system_experiments/checks/check_cpu_kpoint_restart.py
```

The independent spectral fixture allows 80 extra passes to compare convergence
work; the historical GPU candidate used 40. Both limits are recorded in the
JSON. The current default of 100 does not retroactively change those runs. CPU
work ratios must not be presented as GPU speedups.

The evidence validator intentionally checks source hashes against the published
measured snapshot. Run it in a separate checkout of commit
`15cf1d7bbc3f38a95983207299c6bf824e736dca`, before the default-limit change:

```bash
python examples/0d_nanodiamonds/large_system_experiments/checks/validate_report.py
```

GPU module checks are in `acceleration/benchmarks/`: `check_dense_memory`,
`check_mpi_inplace_memory`, `check_mpi_shared_qr` and `check_debug_audit`.
Their results cover the actual measured shapes/dtypes, not every switch combination.

## 5. Preemption recovery and timing boundaries

Scientific code is separate from the scheduler. Each allocation uses immutable
source/runtime/input identities. Submission checks current resources and pending
jobs, and does not replay ambiguous submission intents. Numerical or identity
failures stop the affected path; preemptibility is not a reason to retry incorrect
calculations automatically.

Recovery uses a validated healthy parent. Diagnostic forks do not promote their
own output to replace that parent. Partial reconstruction state is not serialized,
so an interruption loses completed reconstruction passes. Each allocation is
bounded to eight hours and retries are bounded. The original control and new
candidate share the previously authorized 51-allocation budget; three allocations
were transferred to the candidate without increasing the total.

`wall_seconds` measures one launch-to-finish attempt; `reported_program_seconds`
is the program's own timer; `restart_level_refinement.seconds` covers extra
reconstruction only. Queues, interrupted work and cumulative cross-attempt cost
are separate. A resumed fragment is not the cost of a fresh complete calculation,
and residual-audit overhead must be reported when interpreting diagnostic timings.

## 6. Regenerate figures and validate records

```bash
python examples/0d_nanodiamonds/large_system_experiments/checks/plot_results.py
python examples/0d_nanodiamonds/large_system_experiments/checks/analyze_utilization.py
python examples/0d_nanodiamonds/large_system_experiments/checks/validate_report.py
```

Figures use Matplotlib and the curated CSV/JSON. Report validation checks scientific
input/source hashes, electron counts, comparison arithmetic and local Markdown links.
Full numerical acceptance still requires the actual SCF arrays; checking a report
is not a substitute for validating the calculation.
