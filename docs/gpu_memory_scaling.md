# GPU memory and MPI sector groups

This branch keeps float64/complex128, physical inputs, eigensolver thresholds,
SCF convergence criteria and the existing stability fallback. Its changes are
storage ownership, bounded scratch buffers and the communicator used for each
symmetry sector. It does not implement distributed dense diagonalization.

## Decomposition

Launch **one MPI process per GPU**, with that process seeing its assigned GPU
as device 0. `mpi_full_scf --devices 0 --symmetry on` now permits several MPI
processes per node. The TaskKit adapter in the optimization campaign sets the
visibility explicitly. Do not launch several processes on the same GPU.

For P ranks and S symmetry sectors:

- P > S: each sector gets a balanced, contiguous rank group. Each rank holds
  columns of that sector's basis on one GPU. The groups may span nodes.
- P <= S: the existing cyclic sector ownership and node-local device groups
  remain in use.
- With hybrid exchange, the existing exchange ownership rules remain in use.

For 64 ranks and 4 sectors, ranks 0–15, 16–31, 32–47 and 48–63 form separate
communicators. On eight nodes with eight GPUs each, each sector spans two
nodes. Filtering, basis transposes, Gram reductions, condition estimation and
Ritz rotation use the sector communicator. SCF commands and the density sum
use the world communicator. Only the sector leader contributes Ritz metadata;
every member contributes only the density of its own orbital columns.

Contiguous groups are balanced by rank count. They are not yet weighted by
measured sector costs, GPU memory or heterogeneous node performance.

## Storage

`Eigensolvers/dense_memory.py` supplies NumPy/CuPy helpers shared by the real
and complex solvers:

- Mirror lower triangles in bounded tiles; ignore undefined upper entries.
- Sum disposable Gram storage in place, with bounded host/device buffers.
- Reduce complex lower triangles without O(m²) cached index arrays or a full
  packed triangle; the root consumes the existing matrix storage.
- Broadcast coefficients into their final Fortran-ordered allocation.

`PARSEC_DENSE_BUFFER_BYTES` sets the maximum size of each dense communication
buffer, default 64 MiB. Every rank of a communicator must use the same value.
Several buffers can coexist; this is not a total-process memory limit.
Host buffers remain alive until their upload completes. CUDA transport still
requires a CUDA-aware MPI implementation; this setting does not enable it.

The real generalized Ritz solve releases completed whitening intermediates
before the eigensolver workspace is allocated, releases factor/vector
intermediates before its audit, and subtracts the audit identity in place.
Its condition limit remains 1e8 and its orthogonality threshold remains 5e-10.
The MPI owner explicitly permits consumption of its disposable Gram pair.
The public dense-solve API preserves its inputs by default.

`PARSEC_MPI_SHARED_RITZ=inplace` uses one allocation for the column and row
distributions for Gamma, nonsymmetric clusters and the new MPI sector groups.
Reverse rounds first apply H, copy outgoing columns, and receive rows behind
the remaining column prefix. Lower Gram blocks use only the row suffix that
has already arrived. Independent row tiles rotate this same allocation;
forward rounds restore columns after each send completes. Uneven ranks use
the exact padding computed by `_RowExchange.inplace_storage()`. This capacity
also suffices after leading-column trims. A failed stability audit restores
the original filtered columns bit for bit before the existing fallback.

Production random bases reserve this padding at creation. As with the existing
node-local single-allocation route, a consumed basis owns its backing allocation;
callers must not put unrelated live arrays in the unused tail of that allocation.
Externally supplied arrays without enough padding use a separate row allocation; the timing report
counts these events as `separate_row_blocks`. The default `rows` fallback
orthonormalization still temporarily needs a second row block; the opt-in QR
candidate below reuses storage there as well. Neither path reduces precision.

The v2 campaign selects `inplace` explicitly. `rounds` (two shares) and `rows`
(three shares at the Ritz peak) remain available, and the library's `auto`
default is unchanged. Changed reduction/chunking
orders can change round-off and eigenvector signs or degenerate rotations;
physical comparisons use the original accuracy criteria.

## Remaining limits

For a sector with N grid points, m states and q ranks, the main FP64 basis
storage in `inplace` costs approximately `8 * N * m / q` bytes per GPU,
compared with `2 * 8 * N * m / q` in `rounds`, before imbalance, padding and
buffers. Increasing q reduces this part. It does not divide the full sector
operator or the root's dense m×m solve among GPUs.

The dense root and condition helper still need O(m²) memory and the dense
solve remains O(m³) work. Host memory, MPI communication, grid/projector
storage, SCF iterations and allocation time can also set the practical limit.
A sum of all GPU memories is therefore not a usable single-memory budget.
The existing explicit host dense backend remains a separate capacity/speed
tradeoff (`PARSEC_CUPY_RITZ_DENSE_BACKEND=host` together with
`PARSEC_CUPY_RITZ_EIGH_BACKEND=host`). It is not silently selected here.

## Validation

### Candidate: shared storage for robust QR (2026-11-13)

`PARSEC_MPI_SHARED_QR=inplace` opts into the candidate robust-orthogonalization
path. Its default remains `rows`, the existing implementation. Use together
with `PARSEC_MPI_SHARED_RITZ=inplace` so initial column allocations reserve
the small padding needed by uneven exchange layouts. This candidate is not
enabled in the immutable v2 capacity campaign. A subsequent 64-GB200 module
A/B reduced live CuPy allocations by 25.54%, with equal eigenvalues and
sign-aligned vectors. Small Gamma-256 and cluster-3,480 full-SCF comparisons
passed energy/density checks but did not reduce whole-program sampled peaks.
Large-case adoption remains pending. Detailed evidence is in the
[large-system experiment report](../examples/0d_nanodiamonds/large_system_experiments/RESULTS.md).

The real shared-basis solver used by clusters and Gamma-point calculations
can enter shifted Cholesky QR after its generalized Ritz stability check
rejects a filtered basis. Previously this allocated a second complete row
basis, even when the usual Ritz step reused column storage. A resumed
125,272-valence-electron cluster failed at exactly this allocation: 116.04
GiB live, then a request for another 115.12 GiB on a 184.31 GiB GPU.

The candidate moves columns into their existing allocation in decreasing
exchange rounds, permutes bounded row tiles to canonical state order, and
performs the same three shifted Cholesky QR passes. It restores the original
exchange order in bounded tiles and returns columns in increasing rounds.
The Cholesky shifts, dense backend selection, precision, stability audit
and subsequent Ritz solve are preserved. The host-factor route also uses
the existing in-place device triangular solve to avoid a tall output copy.
The permutation scratch is at most `PARSEC_DENSE_BUFFER_BYTES` or one full
row, whichever is larger, plus an O(states) integer index vector. External
unpadded arrays still work, with a separate row allocation counted in
`separate_row_blocks`. Root dense eigensolver workspace remains a separate
bottleneck. Complex k-point eigensolver code is not changed by this candidate.

`tests/test_mpi_shared_qr.py` runs CPU integration checks for canonical
exchanges, uneven/interleaved/empty/trimmed layouts, three-pass QR on an
ill-conditioned basis, and failure on a rank-deficient basis. GPU tests in
`test_mpi_shared_basis.py` compare both dense backends and the full transition
from a refused Ritz step to QR. Run the same-allocation 64-rank A/B with:

```
mpiexec -n 64 python -m parsec_python.acceleration.benchmarks.check_mpi_shared_qr \
  --expected-ranks 64 --output qr-ab.json
```

It checks eigenvalues, vector/subspace accuracy, orthogonality and Galerkin
residual, records live CuPy allocation peaks and elapsed time, and reverses
A/B order on the second repeat. It measures the fallback plus its subsequent
Ritz solve, not a full SCF. Adoption also requires matched full-SCF initial
states, energy/density checks and a measured time/memory tradeoff review.

CPU regression modules:

```
python -m unittest \
  parsec_python.acceleration.tests.test_dense_memory \
  parsec_python.acceleration.tests.test_ritz_dense \
  parsec_python.acceleration.tests.test_mpi_shared_basis \
  parsec_python.acceleration.tests.test_mpi_shared_qr \
  parsec_python.acceleration.tests.test_mpi_scf \
  parsec_python.acceleration.tests.test_kpoint_pools \
  parsec_python.acceleration.tests.test_column_ritz
```

Tests cover poisoned upper triangles, real/complex arrays, uneven chunk sizes,
communicators with uneven sector groups (including 64 ranks / 4 sectors),
exactly-once metadata ownership, stability rejection and first/subsequent
k-point eigensolves and SCFs. The expanded publication regression on the
frozen candidate ran 161 tests: 134 passed and 27 were skipped. The exact
module inventory and independent complex-k-point fixture results are included
in the experiment report; skipped CUDA checks are not counted as GPU passes.

`benchmarks/check_mpi_shared_basis.py` checks the actual MPI/CuPy exchanges and
Ritz steps. `benchmarks/check_dense_memory.py --reference PATH --output JSON`
compares against the pinned baseline dense module on an actual GPU, including
live CuPy allocation peaks, eigenvalues, sign-aligned eigenvectors and eigen
residuals. This live-allocation metric excludes CUDA context/library memory;
the capacity jobs separately sample device-wide memory with `nvidia-smi`.
`check_mpi_inplace_memory.main(output)` compares full MPI Ritz live peaks for
`rounds` and `inplace` on the same ranks, including padded bases and bounded
buffers. It also compares their Ritz values and sign-aligned vectors.

The campaign gates larger jobs on small full-SCF energy, density, charge and
checkpoint-restart comparisons. A converged SCF is distinct from an OOM,
time limit, failed accuracy check or merely queued allocation. Capacity is
reported in valence electrons, with GPU counts and source/input identities.

## Optional numerical diagnostics

Set the same environment on every MPI rank:

```bash
export PARSEC_DEBUG=summary       # or audit; off (the default) disables it
export PARSEC_DEBUG_RANKS=leaders # or all; each sector/pool owner is a leader
export PARSEC_DEBUG_DIR=/path/to/run/debug # optional, separate rank JSONL files
export PARSEC_DEBUG_BUFFER_BYTES=67108864 # audit scratch target, 64 MiB
```

Every record is also printed as `PARSEC_DEBUG { ... }`, with an event, UTC
Unix timestamp, global rank, process ID and sequence number. Summary mode
records restart contents, SCF residuals/energies, spectral intervals, Ritz
condition estimates, and QR fallback reasons. Audit mode additionally scans
scalar fields (including the physical density integral), records QR Gram
diagonals, and measures actual `||H q_j - epsilon_j q_j||` in Ry. A basis with
an unrotated handover is explicitly reported as skipped, rather than tested
as though it already contained eigenvectors. Column norm diagnostics do not
replace the solver's off-diagonal orthogonality audit.

The real shared-basis residual check works on bounded column tiles and
retains no second full basis. Three columns are its scratch floor when one
column is larger than a third of the requested budget; the log reports that
floor. The complex pool reuses its existing applied row blocks and bounds
additional residual tiles. These checks add Hamiltonian applications,
reductions and synchronization, so audit timings are diagnostic timings,
not production performance measurements. Turning diagnostics on never
changes filter counts, convergence thresholds, QR routing or the returned
residual policy. Missing historical CHEBFF residuals are JSON `null`, not
zero. A log file error falls back to stdout and is reported without leaving
other MPI ranks stranded in a collective.

For a restart investigation compare the same frozen parent checkpoint and
Hamiltonian. Record the parent SHA256 and distinguish a diagnostic fork
from a production continuation. A correct file hash and conserved electron
count do not prove that the restarted eigensubspace is accurate.
