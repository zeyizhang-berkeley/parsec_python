# Incremental large-system optimization

Each stage delivers one independently selectable change, a frozen source
inventory, a same-input comparison, precision evidence, and a documented
decision. An improvement is reported for the tested system and hardware;
untested cluster/Gamma/k-point paths remain separate validation work.

## Evidence already obtained

The 19,392-electron cluster on 64 H100s completed 11 SCF steps in both the
original and memory-v1 runs. Sampled peak memory fell from 61,269 to 42,989
MiB per GPU (29.84%); final total energies differed by 3.73e-9 Ry. Observed
wall time increased from 1,350.64 to 1,827.59 seconds (35.31%). These were
different allocations, so this is evidence for memory reduction with an
observed time cost, not an isolated timing result or an unconditional
recommendation. It does not yet satisfy the user's preference for a large
advantage when accepting a regression in the other metric.

Smaller dense/QR/Ritz module tests show reduced temporary storage, but do
not establish the whole-program peak for larger systems. The completed
95,648-electron calculation establishes capacity, not an old/new comparison.

## Stage 1: shorten missing-wavefunction reconstruction

Candidate: `PARSEC_RESTART_REFINE_LOWER_BOUND=smart`, used only with
`PARSEC_RESTART_REFINE=levels`. Default remains `inherit`. See
[restart_refinement.md](restart_refinement.md) for scope and limits.

The target is the 95,648-electron cluster on 64 GB200s. Small 256-electron
Gamma and 3,480-electron cluster pairs precede the large paired run as short
compatibility checks. All pairs use exact pinned healthy parents and the
same runtime, grid, pseudopotentials, FP64 controls, initial filtering work,
40-pass reconstruction cap and original full-SCF convergence thresholds.

Record separately:

- Reconstruction passes, Hamiltonian/filter work and reconstruction wall time.
- Full SCF convergence, iteration count, solve time and end-to-end time.
- Maximum per-GPU sampled memory and instrumented live allocation peaks.
- Final energy (also per electron), density norms, electron count, occupied
  and all-requested level differences, and actual eigenpair residuals.
- Allocation interruption, startup, checkpoint I/O and diagnostic overhead.

Current stage: the large smart candidate has completed a converged restart;
the matched inherit control and final array-accuracy review are pending.
The expanded publication regression has 134 passes and 27 skips (161 total).
Independent real/complex fixtures and physical k-point checks also passed.
No large-system speedup is yet claimed.

GPU update, 2026-11-15 00:46 PST: the 64-GB200 comparison is running. Both
small same-parent pairs completed full SCF. Gamma-256 had identical energy
and final field fingerprints, 917 MiB sampled peak in both runs, and observed
wall times 37.34/33.24 seconds (inherit/smart). Cluster-3,480 had a 2.79e-9 Ry
energy difference, 1,857 MiB in both runs, and times 43.37/41.56 seconds;
its final fields are not bitwise identical and still require numeric array
comparison. These single audit-enabled pairs do not establish stable speedup.

The 95,648-electron candidate passed reconstruction after 33 extra passes
(4,729.07 s), then converged in one resumed SCF step with weighted residual
6.66e-5 against the original 2e-4 threshold. End-to-end time was 5,239.92 s
and sampled peak 97,105 MiB. The energy differs from the healthy parent by
-7.84725e-4 Ry (-8.20430e-9 Ry/electron). Working-state residual maxima across
the four sectors remain 0.00717–0.00815 Ry; level stability is not a claim
that every eigenvector residual is below 1e-4. The paired large control and
final accuracy/performance review remain open. Public evidence and the
next communication bottleneck are documented in the
[large-system experiment report](../examples/0d_nanodiamonds/large_system_experiments/README.md).

Adoption needs a complete large-case precision review and a repeatable
benefit. The first audit-enabled pair establishes behavior; matched runs
with audit disabled, repeated in reversed order on the same allocation,
establish timing. A faster reconstruction that worsens the full SCF or
accuracy is rejected. An inconclusive or failed candidate remains opt-in.

## Following stages

Use the measured large-case profile to choose the next independent change:
reduce peak copies in orthogonalization/Ritz rotation, reduce cross-node
communication, or retain reusable work buffers. Change one mechanism per
comparison. Numerical algorithms must pass complete SCF checks; memory-only
changes should preserve the mathematical operations and be checked for
aliasing and buffer lifetime errors.

Extend each useful result to large Gamma and complex k-point cases, then
test the next electron count. A first result in one family can be delivered
with its scope stated while those extensions continue. Neither relaxed
accuracy nor a larger allocation is counted as a code optimization.
