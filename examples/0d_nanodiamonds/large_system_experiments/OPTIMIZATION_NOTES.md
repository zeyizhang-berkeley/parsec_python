# Optimization opportunities identified by large-system tests

This document separates implemented changes, measured benefits and proposed work.
See [RESULTS.md](RESULTS.md) for numerical evidence and
[GPU_UTILIZATION.md](GPU_UTILIZATION.md) for the utilization investigation.
The objective is lower complete-calculation time or peak memory at unchanged
physical settings and validated accuracy.

## 1. GPU memory is not one pooled allocation

For a sector with $N$ grid points, $m$ working states and $q$ participating ranks,
one FP64 basis requires approximately

$$
M_{basis}\simeq 8Nm/q\quad\text{bytes per GPU}.
$$

Complex128 uses 16 bytes per element. Actual storage also includes uneven-layout
padding, Hamiltonian/projector data, potential/density, communication buffers,
memory pools and library workspaces. The leader's dense $m\times m$ matrices and
eigensolver retain $O(m^2)$ storage and $O(m^3)$ work; more ranks do not reduce
these costs proportionally.

At fixed physical density and grid accuracy, both $N$ and $m$ often grow with
electron count, giving the planning approximation

$$
M_{basis}\propto N_e^2/q.
$$

Even with perfect basis distribution, doubling GPUs would therefore often support
only about $\sqrt 2$ times as many electrons. This is not a measured capacity law:
vacuum, sector imbalance, k-point pools, dense storage and QR fallbacks change the
actual boundary.

## 2. Implemented: distribute a sector across nodes

Sources: [mpi_scf.py](../../../src/parsec_python/acceleration/experimental/mpi_scf.py),
[mpi_shared_basis.py](../../../src/parsec_python/acceleration/Eigensolvers/mpi_shared_basis.py),
[symmetry.py](../../../src/parsec_python/acceleration/Eigensolvers/symmetry.py).

These diamond fragments use D2 symmetry with four sectors. Four is a property of
these structures and the symmetry group used, not a universal cluster limit.
The previous node-local assignment restricted the usefulness of additional nodes.
When total ranks $P>S$, the new path builds balanced, contiguous rank groups per
sector, allowing a group to span nodes. With 64 ranks and four sectors, each group
has 16 ranks: four four-GPU nodes on the observed GB200 allocation.

Filtering, layout conversion, Gram reduction and Ritz rotation use the sector
communicator; SCF commands and global density reduction use the world communicator.
Only the sector leader submits eigenvalue metadata, while each member contributes
the density of its own orbital columns. This avoids duplicated counts. Existing
paths are retained for $P\le S$ and specialized hybrid-exchange ownership.

The assignment is not a lookup table for C23191H2884. Tests include uneven rank
and sector counts, empty blocks and exactly-once metadata collection. It still
balances rank counts, rather than measured sector time/memory, and does not yet
implement heterogeneous-GPU load balancing.

## 3. Implemented: reuse dense intermediates and MPI Ritz storage

Sources: [dense_memory.py](../../../src/parsec_python/acceleration/Eigensolvers/dense_memory.py),
[rayleigh_ritz.py](../../../src/parsec_python/acceleration/Eigensolvers/rayleigh_ritz.py),
[chebff_columns.py](../../../src/parsec_python/acceleration/Eigensolvers/chebff_columns.py).

- Mirror lower triangles in bounded blocks, avoiding full-matrix index arrays or
  an additional packed triangle.
- Sum explicitly consumable Gram temporaries in place; public APIs preserve caller
  inputs unless consumption is requested.
- Release consumed whitening, factor and vector intermediates before allocating
  subsequent workspaces.
- Broadcast into the final Fortran-order matrix to avoid another coefficient copy.
- Bound communication scratch with `PARSEC_DENSE_BUFFER_BYTES`, default 64 MiB.
  This is a per-buffer bound, not a total-process memory limit.

`PARSEC_MPI_SHARED_RITZ=inplace` reuses one large allocation for column and row
layouts. Reverse exchange rounds reuse storage only after its send has completed;
row tiles rotate in place and are then restored to column layout. Padding accounts
for uneven layouts and must remain sufficient after trimming. External arrays with
insufficient padding retain the separately allocated row-block path, which is
counted in diagnostics.

If the stability audit fails, the original filtered columns must be restored
before the existing fallback runs. Saving memory must not overwrite live inputs,
release asynchronous-transfer buffers early or relax orthogonality thresholds.
The library's `auto` policy was not changed to force this experimental mode;
memory-v2 selected `inplace` explicitly.

Module savings and the 64-H100 capacity improvement are documented in RESULTS.
Timing changes have not all been isolated. Different reduction groupings alter
rounding order; bitwise equality is not promised for every physical problem.

## 4. Implemented candidate: reuse the large basis during robust QR

During recovery at 125,272 electrons, a previous path requested another approximately
115.12 GiB row basis while approximately 116.04 GiB was already live, exceeding the
observed 184.31 GiB GB200 capacity. A single-allocation ordinary Ritz route does
not imply a single-allocation stability fallback. This peak becomes visible only
for sufficiently large or difficult subspaces.

`PARSEC_MPI_SHARED_QR=inplace` makes the three-pass shifted Cholesky QR reuse the
basis allocation. Bounded row permutations and triangular solves preserve the
shift, condition checks and subsequent Ritz calculation. The default remains
`rows`. This candidate concerns real shared-basis cluster/Gamma paths; its results
do not establish a complex-k-point QR improvement.

The 64-GB200 QR-plus-Ritz module check reduced live memory by 25.54%. Module accuracy
and two small full-SCF comparisons passed, but small-program peaks did not fall
and the 125,272-electron validation is unfinished. It remains a candidate.

## 5. Implemented candidates: fixed-potential reconstruction and smart bounds

Sources: [restart_refinement.py](../../../src/parsec_python/SCF/restart_refinement.py),
[restart.py](../../../src/parsec_python/SCF/restart.py),
[kpoints.py](../../../src/parsec_python/SCF/kpoints.py).

Loading a healthy potential does not restore a healthy eigensubspace. Three fixed
initial filtering cycles can leave a new random subspace insufficiently refined;
the resulting density can then damage the mixing history. The improved first
resumed residual after additional cycles is the main evidence for this diagnosis.

The `levels` policy keeps the Hamiltonian fixed and performs extra solves until
all requested levels stabilize for two consecutive passes. It retains only an
O(states) host spectrum snapshot, not another full wavefunction basis. If the
configured limit is reached, it fails before constructing/mixing a new density,
and that output must not become a new healthy parent automatically. The current
default limit is 100 extra passes; the reported GPU experiments used an explicit
40-pass limit. Ordinary SCF has a separate `Max_Iter` limit, also defaulting to
100 when omitted. Neither change relaxes the accuracy thresholds.

The `smart` option changes the filtering lower bound during extra reconstruction
passes only. In the large example, the old monotone bound retained an early value
near +0.193 Ry; the new bound followed the Ritz spectrum toward approximately
−0.064 Ry. Initial-solve settings, ordinary SCF settings objects, degrees,
precision and acceptance thresholds are unchanged. The reconstructed subspace
and bound history persist, so later numerical trajectories need not be bitwise equal.

Scalar/sector and built-in k-point/spin paths share the policy. The k-point settings
factory is restored on success or exception. Custom solvers without a KPointJob
explicitly reject the override. Spin-orbit checkpoint refinement is not supported.

Independent CPU real/complex spectral checks reduced the shifted-spectrum pass
counts from 47/50 to 10/11. Small physical k-point tests agreed with ARPACK levels
to approximately 1e-14 Ry. These work counts are not large-GPU speedups. The actual
95,648-electron smart restart needed 33 extra passes and met the original SCF
criterion; its matched control and final array comparisons remain open.

## 6. Next stage: prioritize data redistribution

The completed large restart provides specific timing evidence:

| Rank 0 / sector 0 phase | Accumulated seconds |
|---|---:|
| to_rows | 2,534.67 |
| to_columns | 1,277.12 |
| Gram construction | 264.51 |
| Rotation | 219.02 |
| Reduction | 149.11 |
| Dense solve and coefficient broadcast | 89.84 |
| Filter | 89.67 |
| Hamiltonian apply | 8.24 |

![Large-restart phase profile](figures/restart_profile.svg)

The two conversions total 3,811.79 seconds, or 73.85% of the 5,161.30-second
diagonalization. One rank's `exchanged_bytes` counter is approximately 8.14e12;
this is cumulative logical send volume, not whole-machine network throughput.
Timers include packing, host/device movement and collective waits. Different ranks
or overlapping phases cannot be summed as wall time. This describes one audited
restart workload, not every full-SCF run.

Priorities below are hypotheses, not measured improvements:

1. **Reduce conversions or reuse layouts.** Determine whether row/column state can
   persist across Ritz work or consecutive exchange rounds can be combined. Track
   every Hamiltonian, density and filtering consumer so fewer transfers do not
   require another large basis.
2. **Overlap communication and computation.** Double-buffer safe tiles and overlap
   host/device copies, MPI and local Gram work without changing precision. Include
   the extra buffers in measured peaks.
3. **Validate the transport.** The measured path stages through the host. Verify
   available MPI providers, GPU-aware communication and topology before an isolated
   A/B; an environment-variable change alone does not establish device-direct support.
4. **Preserve reconstruction work.** Evaluate eigensubspace/wavefunction or partial
   reconstruction checkpoints, including I/O, capacity and cross-GPU portability.
   Density/potential alone cannot preserve completed reconstruction passes.
5. **General layout and load balancing.** Use sector/k-point state counts, memory
   and timing to allocate ranks. Preserve fast node-local transfers inside a general
   hierarchy; do not hard-code a four-sector or one-structure decomposition.
6. **Distribute large dense problems when justified.** Leader $m^2$ storage and $m^3$
   work may eventually dominate. The present profile prioritizes layout movement
   ahead of a distributed dense eigensolver.

## 7. Adoption criteria

Freeze each candidate independently and preserve a control using the same input
and healthy parent. Check full SCF, total/per-electron energy, density, charge,
occupied/unoccupied levels and actual eigenpair residuals. Then measure performance
with audit disabled, on the same allocation, reversing A/B order and repeating.
Record live allocations and whole-device sampled peaks separately.

A substantial time regression is not accepted merely because memory decreases.
A trade-off is considered only when one resource improves enough to justify the
other, using explicit measurements. Deliver validated families/sizes incrementally
before extending to larger clusters, Gamma cells and complex k-point workloads.
