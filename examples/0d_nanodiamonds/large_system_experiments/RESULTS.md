# Experimental results and evidence limits

The main scientific snapshot is **2026-11-15 00:46 PST**. See
[runs.csv](data/runs.csv) for stable `record_id` values and
[runs.json](data/runs.json) for the corresponding detailed records and hashes.
The later utilization investigation is documented separately in
[GPU_UTILIZATION.md](GPU_UTILIZATION.md).

## 1. Resources

| GPU | Observed visible memory/GPU | CPU architecture | Visible GPUs/node | Sizes used in this campaign | Allocation type |
|---|---:|---|---:|---|---|
| A100-SXM4-80GB, existing NERSC reference | 79.25 GiB | See original benchmark | 4 | 1–16 GPUs, at most 4 nodes | Historical reference |
| H100 | Approximately 79.11 GiB | x86_64 | 8 | 16-GPU checks; 32/64-GPU capacity runs | Overseas preemptible |
| H200 | Approximately 139.80 GiB | x86_64 | 8 | 16-GPU module and short checks | Overseas preemptible |
| GB200 | Approximately 184.31 GiB | aarch64 | 4 | Early 8/16-GPU runs; primarily 64 GPUs for later work | Overseas preemptible |

The actual 64-GB200 topology is **16 nodes × 4 GPUs**; 64 H100s use
**8 nodes × 8 GPUs**. Equal GPU counts do not imply equal topology or network paths.
The domestic 64-A100 request was stopped while queued; this report contains no
completed result from that allocation. There is also insufficient evidence for a
64-H200 large-system capacity boundary.

The runtime inventory in [hardware.json](data/hardware.json) records Python 3.12.4,
NumPy 2.2.6, SciPy 1.15.3, CuPy 14.2.0 and mpi4py 4.1.1. The main GB200 bundle uses
MPICH 4.3.2 and native extension 0.6.1. The MPI array path explicitly uses host
staging. Each 64-GPU allocation is bounded to eight hours; preemptible resources
do not guarantee uninterrupted execution.

## 2. Systems, accuracy settings and electron counts

Clusters are ideal diamond fragments with hydrogen passivation; the larger inputs
were not subjected to a new geometry relaxation. The valence-electron convention is
$N_e=4N_C+N_H$, not the sum of atomic numbers.

| Cluster | Composition | Atoms | Valence electrons | Boundary-sphere radius | Requested levels |
|---|---|---:|---:|---:|---:|
| k11 | C4601H988 | 5,589 | 19,392 | 28.0 Å | 9,702 |
| k18 | C19395H2556 | 21,951 | 80,136 | See frozen run input | See run input |
| k19 | C23191H2884 | 26,075 | 95,648 | 45.8 Å | 47,830 |
| k20 | C26469H3148 | 29,617 | 109,024 | 48.5 Å | 54,518 |
| k21 | C30457H3444 | 33,901 | 125,272 | 49.4 Å | 62,642 |

Cluster settings are LDA/Ceperley–Alder, 0.20 Å spacing, eighth-order finite
differences and 500 K Fermi–Dirac occupations. Periodic calculations use PBE bulk
diamond, with eight carbon atoms per conventional cell. Replication by
$n_1\times n_2\times n_3$ gives

$$
N_{atoms}=8n_1n_2n_3,\qquad N_e=32n_1n_2n_3.
$$

The periodic grid spacing is 0.37448072665553 bohr (approximately 0.198 Å), with
eighth-order finite differences. `k2` and `k3` denote 2×2×2 and 3×3×3
Monkhorst–Pack meshes, not supercell replication counts. The solver uses the
symmetry-reduced k points and weights; $N_e$ is not multiplied by the number of
k points.

The original settings are retained: `Convergence_Criterion=2e-4 Ry`,
`Diag_Tolerance=1e-4`, and FP64/complex128. Initial CHEBFF uses three cycles of
degree 30; subsequent SUBSPACE filtering uses degree 15. Condition-number checks,
orthogonalization fallbacks and mixing settings remain in effect. Cluster and
periodic carbon pseudopotentials differ; their hashes in
[inputs.json](data/inputs.json) must not be interchanged.

## 3. Completed large samples and failures

These are **selected completed samples and neighboring failures**, not a complete
binary search of capacity. Every row below uses 64 GPUs. Time covers the recorded
attempt only, excluding queues and work lost in earlier allocations.

| Version/resource | System | Valence electrons | Outcome | SCF steps | Attempt time/s | Peak MiB/GPU |
|---|---|---:|---|---:|---:|---:|
| memory-v2 / H100 | Cluster C19395H2556 | 80,136 | Converged | 12 | 3,957.20 | 72,727 |
| memory-v2 / H100 | Cluster C23191H2884 | 95,648 | OOM | — | 146.89 | Not fully recovered |
| memory-v2 / H100 | Gamma, 11×11×11 | 42,592 | Converged | 12 | 3,764.06 | 75,297 |
| memory-v2 / H100 | k3, 8×8×8 | 16,384 | Converged | 10 | 2,447.04 | 74,901 |
| memory-v2 / H100 | k3, 9×9×9 | 23,328 | OOM | — | 143.14 | Not fully recovered |
| memory-v2 / GB200 | Cluster C23191H2884 | 95,648 | Fresh calculation converged | 14 | 2,360.24 | 97,057 |
| memory-v2 / GB200 | Cluster C26469H3148 | 109,024 | OOM during recovery | — | 989.68 | Not fully recovered |
| memory-v2 / GB200 | Cluster C30457H3444 | 125,272 | OOM during recovery | — | 1,226.84 | Not fully recovered |
| memory-v2 / GB200 | Gamma, 13×13×13 | 70,304 | Converged after recovery | 10 | 2,734.98 | 185,689 |
| memory-v2 / GB200 | Gamma, 13×13×14 | 75,712 | OOM | — | 554.07 | Not fully recovered |
| memory-v2 / GB200 | k2, 10×10×10 | 32,000 | Fresh calculation converged | 8 | 5,173.08 | 148,971 |
| memory-v2 / GB200 | k2, 11×11×11 | 42,592 | OOM | — | 89.69 | Not fully recovered |
| memory-v2 / GB200 | k3, 11×11×11 | 42,592 | OOM | — | 89.72 | Not fully recovered |

The Gamma result at 70,304 electrons, for example, is already close to the observed
GB200 memory capacity. Cluster failures also depend on whether robust QR requests
a second large basis. A capacity claim therefore needs the system family, grid,
k mesh, symmetry, source identity and execution path, as well as GPU count.
Successful initialization, one saved checkpoint or the absence of OOM alone is
not a completed capacity validation.

The earlier 16-A100 C9449H1572 reference has 39,368 electrons, ten SCF steps,
208.5 seconds of program time and approximately 60.0 GiB peak memory. It uses
node-local software/cache and different topology and timing boundaries. It is
not a direct hardware-speed comparison with these TaskKit end-to-end measurements.

## 4. Measured optimization effects

### 4.1 Full SCF: memory-v1

The same C4601H988 input and runtime, without symmetry, on 64 H100s:

| Measurement | Baseline | memory-v1 |
|---|---:|---:|
| SCF steps | 11 | 11 |
| Total energy/Ry | −53752.0912609566 | −53752.091260952875 |
| Maximum sampled memory/MiB | 61,269 | 42,989 |
| End-to-end attempt time/s | 1,350.6375 | 1,827.5936 |

Memory decreased by 29.84%; the total-energy difference was
$3.7253\times10^{-9}$ Ry. Time increased by 35.31%. These runs used different
allocations and include communication/checkpoint I/O, so the timing difference
cannot be attributed entirely to one kernel. This supports a memory-saving
claim, but not unconditional adoption under the requested time/memory trade-off.
See [full_scf_memory_pair.json](data/full_scf_memory_pair.json).

The 80,136-electron cluster failed with OOM on 64 H100s under memory-v1 and
converged under memory-v2. That is observed capacity progress; precise performance
attribution still requires a matched A/B on the same allocation.

### 4.2 Module workspace measurements

| Check | Configuration | Old → new live bytes | Reduction | Numerical evidence |
|---|---|---:|---:|---|
| Dense generalized Ritz | 4,096 states, GB200/H100/H200 | 1,615,072,768 → 1,212,419,584 | 24.93% | Equal levels and sign-aligned vectors |
| MPI Ritz | 131,071 rows, 257 states, 8 GB200s | 82,977,280 → 49,291,776 | 40.60% | Equal levels and aligned vectors |
| MPI Ritz | Same shape, 16 H100s/H200s | 47,260,160 → 30,417,408 | 35.64% | Equal levels; vector difference approximately 1.73e-17 |
| Robust QR followed by Ritz | 524,287 rows, 509 states, 64 GB200s | 100,418,560 → 74,766,848 | 25.54% | Equal levels/aligned vectors; orthogonality error approximately 2.66e-15 |

These measurements describe **live CuPy allocations**, excluding CUDA context
and library memory. They are not `nvidia-smi` or full-SCF peak reductions. The two
QR timings with reversed execution order produced ratios of 1.351 and 0.869,
so they do not establish a stable speedup. See
[module_comparisons.json](data/module_comparisons.json).

The QR candidate passed full-SCF energy, density and charge comparisons for
Gamma-256 and cluster-3,480. Their program peaks remained 921 and 1,857 MiB:
workspace savings did not lower the full-program peak in these small samples.
See [qr_full_scf_pairs.json](data/qr_full_scf_pairs.json).

## 5. Checkpoint reconstruction and the latest candidate

The `parsec.dat` recovery path used here restores charge, potential and levels,
but not wavefunctions, the full eigensubspace or mixer history. The repository
has other wavefunction I/O capabilities; this statement concerns the actual
campaign recovery path, not every checkpoint format in the project.

Starting from a healthy SCF-5 checkpoint of the 125,272-electron system, the first
resumed SCF residual was 9.0906888391 with three initial filtering cycles, versus
0.1377455741 with six cycles—approximately 66 times smaller. No QR fallback was
observed in the six-cycle probe. This supports insufficient subspace reconstruction
as an important cause, but does not prove that all divergence mechanisms are fixed.

`PARSEC_RESTART_REFINE=levels` performs additional solves at fixed loaded potential.
Every requested level must change by no more than the input `Diag_Tolerance` for
two consecutive passes before a new density is constructed. The GPU candidate
allows at most 40 extra passes and fails closed. The `smart` candidate additionally
uses the existing dynamic filter-lower-bound rule during those extra passes only.
Polynomial degrees, FP64, physical parameters and the final SCF criterion are unchanged.

### 5.1 Short comparisons on the same allocation

| System | Electrons | inherit → smart time/s | Total-energy difference/Ry | Peak MiB/GPU |
|---|---:|---:|---:|---:|
| Gamma | 256 | 37.3399 → 33.2425 | 0 | 917 → 917 |
| Cluster C795H300 | 3,480 | 43.3704 → 41.5646 | +2.79397e-9 | 1,857 → 1,857 |

Each pair starts from its own identical healthy parent checkpoint and converges
in one resumed SCF step. Gamma's final array fingerprints also match. Cluster
fingerprints differ; direct numerical array comparisons remain to be completed.
These are single audited pairs, not proof of repeatable 11% or 4% speedups.

### 5.2 Completed smart restart at 95,648 electrons

- 16 nodes × 4 GB200s, 64 MPI ranks, four sectors with 16 ranks each.
- 33 extra reconstruction passes, taking 4,729.07 seconds; final maximum adjacent
  level change 9.53870e-5 Ry, with two consecutive passes satisfying 1e-4.
- One subsequent SCF step converged: weighted residual 6.65673e-5, against the
  original 2e-4 criterion.
- End-to-end time 5,239.92 seconds; reported program time 5,204.62 seconds;
  sampled peak 97,105 MiB = 94.83 GiB per GPU.
- Total energy −268541.72651565075 Ry versus the healthy parent
  −268541.7257309258 Ry: **−7.84725e-4 Ry total**, or −8.20430e-9 Ry per
  valence electron. Both normalizations are reported.
- Integrated density 95,648.00000000525; occupation charge 95,648.00000000003.
  Final density, potential, levels and occupations were all finite.
- Actual maximum working-state $\|Hq-\epsilon q\|$ values across sector leaders
  were approximately 0.00717–0.00815 Ry. **Level stability is not an eigenvector
  residual criterion**; this does not establish 1e-4 residual accuracy for every
  working state.

The matching `inherit` control began immediately afterward on the same allocation
and was unfinished at the snapshot. Final density/occupied-level comparisons,
the complete timing ratio and further post-restart evolution remain open.
One resumed step from a converged parent does not mean a fresh calculation needs
one step. The historical fresh run took 14 steps and 2,360.24 seconds; that and
this strict 5,239.92-second reconstruction are not a speed A/B.

![Reconstruction progress](figures/reconstruction_progress.svg)

The historical `inherit` curve was interrupted after 29 passes; no convergence
point is invented. At pass 20, `smart` had approximately 29.23% lower adjacent-level
change with similar elapsed reconstruction time. This is a process-level observation.
See [large_restart_result.json](data/large_restart_result.json) and
[reconstruction_curve.csv](data/reconstruction_curve.csv).

## 6. Conclusions not established

There is no demonstrated speedup across all cluster/Gamma/k-point families, no
complete 64-A100/H200 capacity boundary, and no completed 125,272-electron accuracy
validation of the robust-QR candidate. Small-system timings do not replace large
matched experiments. If the platform does not identify an interruption's cause,
it remains an interruption/failure rather than an asserted preemption. Production
adoption requires the remaining accuracy and benefit checks for each candidate.
