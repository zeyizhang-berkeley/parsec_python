# Large-system, multi-node GPU experiments

This directory records PARSEC Python experiments from November 11–15, 2026.
The main scientific-results snapshot is **2026-11-15 00:46 PST**, followed by a
publication CPU regression. A subsequent **01:14 PST** check found the matching
`inherit` control still running: 10 extra reconstruction passes, with maximum
adjacent-pass level change of 1.76443e-3 Ry. No final timing ratio was available.
The later November 15 [communication diagnostic](COMMUNICATION_DIAGNOSTIC.md)
adds a completed 64-GB200 transport probe and documents optional exchange tracing;
it does not replace those historical scientific-result snapshots.

The objective is to calculate larger systems on more nodes while preserving the
physical problem and accuracy settings, and to measure memory, communication and
algorithmic improvements separately. Electron counts mean **pseudopotential
valence electrons** throughout.

This is an experimental branch. Implemented features, measured benefits and
unvalidated candidates are distinguished explicitly. SCF convergence alone does
not complete an independent accuracy review or establish a GPU's maximum capacity.

## Main findings

- **Full-SCF memory measurement:** for a 19,392-electron cluster without symmetry
  on 64 H100s, memory-v1 reduced the sampled per-GPU peak from 61,269 to 42,989 MiB
  (**29.84%**). The total-energy difference was $3.73\times10^{-9}$ Ry. The runs
  took 1,350.64 and 1,827.59 seconds on different allocations: an observed 35.31%
  increase in time, not a speedup.
- **Completed large systems:** memory-v2 has converged records on 64 H100s for an
  80,136-electron cluster, 42,592-electron Gamma-point cell and 16,384-electron
  cell with a 3×3×3 k mesh. On 64 GB200s, completed examples include a 95,648-electron
  cluster, 70,304-electron Gamma cell and 32,000-electron cell with a 2×2×2 k mesh.
  These are completed samples under their own settings, not universal limits.
- **Module workspace savings:** isolated dense Ritz, MPI Ritz and robust-QR checks
  measured approximately 24.93%, 35.64–40.60% and 25.54% lower live CuPy allocations.
  These percentages do not describe full-program peak memory.
- **Latest completed restart:** the `smart` candidate reconstructed the fixed-potential
  subspace of the 95,648-electron cluster in 33 extra passes on 64 GB200s, then met
  the original SCF criterion in one resumed step. End-to-end time was 5,239.92 seconds;
  sampled peak memory was 97,105 MiB. Its matching `inherit` control was unfinished
  at the recorded checks.
- **Next optimization target:** in that audited restart, rank 0/sector 0 spent
  approximately 3,812 seconds in the two layout conversions, about 74% of the
  diagonalization time. These timers include packing, transfers and waits.

## Reading guide

| File | Contents |
|---|---|
| [RESULTS.md](RESULTS.md) | Resources, systems, capacity observations, full-SCF/module comparisons, restart diagnostics and accuracy limits |
| [GPU_UTILIZATION.md](GPU_UTILIZATION.md) | Why the 64-GB200 repair job has low utilization; comparison with the earlier 16-GB200 job |
| [COMMUNICATION_DIAGNOSTIC.md](COMMUNICATION_DIAGNOSTIC.md) | Completed 64-GB200 communication probe, its memory/duration limits, and optional all-rank exchange tracing |
| [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md) | Source changes, memory/communication bottlenecks and proposed validation work |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Inputs, source identity, environment, switches and verification commands |
| [data/runs.csv](data/runs.csv) | 46 curated scientific run records, including completed and failed attempts |
| [data/runs.json](data/runs.json) | The same runs with available input/runtime hashes |
| [data/large_restart_result.json](data/large_restart_result.json) | SCF, charge and eigenpair audits for the 95,648-electron restart |
| [data/large_restart_profile.json](data/large_restart_profile.json) | Per-rank timing and memory measurements for that restart |
| [data/communication_probe_20261115.json](data/communication_probe_20261115.json) | Sanitized aggregate from the completed communication-only diagnostic |
| [data/provenance.json](data/provenance.json) | Baseline/frozen identities, hashes of 27 changed source/test files, and regression results |
| [inputs/](inputs/) | 12 actual cluster/periodic input files, including successful and failed sizes |
| [checks/](checks/) | Independent CPU accuracy checks, evidence validation and figure regeneration |

Sampled device peaks, live CuPy allocations, total runtime and reconstruction time
are separate measurements. Raw scheduler logs, credentials and large checkpoints
remain in experiment storage. This directory publishes scientific inputs, selected
numerical evidence and content hashes; it does not turn interrupted attempts into
completed results.

Earlier 1–16 A100 benchmarks are in the [parent README](../README.md) and
[performance directory](../performance/README.md). Those measurements use different
launch/storage conditions and topology. Their timings cannot be divided directly
by the present end-to-end measurements to claim a hardware speedup.
