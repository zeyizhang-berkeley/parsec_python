# Experimental checkpoint reconstruction

`parsec.dat` in this campaign stores charge, potential and eigenvalues. It
does not restore wavefunctions, the eigensolver subspace or mixer history.
Rebuilding a large subspace with a fixed `FF_MaxIter` can therefore produce
a poor first density even when the stored potential is healthy.

The optional repair holds the loaded potential fixed after the initial
solve and runs additional existing SUBSPACE passes before constructing a
density. All requested Ritz levels must change by at most the input
`Diag_Tolerance` on two consecutive passes. It stops with
`RestartRefinementError` if the bounded reconstruction fails, before the
new density reaches mixing or an iteration checkpoint callback.

```bash
export PARSEC_RESTART_REFINE=levels
export PARSEC_RESTART_REFINE_MAX_PASSES=40
export PARSEC_RESTART_REFINE_LOWER_BOUND=inherit  # or opt-in smart; see below
export PARSEC_DEBUG=summary  # audit also measures actual Hq - epsilon*q
```

Use the same environment on all MPI ranks. `PARSEC_RESTART_REFINE=off` is
the default and preserves the old fixed-pass path. Logging and the repair
are separate switches. The repair only attaches to `load_restart`; a
programmatic `SCFWarmStart` can explicitly supply
`RestartRefinementSettings`. A saved eigensolver state bypasses rebuilding.

The common policy is used by the scalar/sector SCF loop, its isolated spin
channels, and the k-point loops with and without spin polarization. An
enabled spin-orbit checkpoint restart is explicitly unsupported. Other
drivers must opt into the policy rather than silently claiming coverage.

The policy retains only small host spectra, O(requested states), and
releases obsolete orbital wrappers before the next solve. It adds no full
device basis. It preserves the Hamiltonian, discretization, occupations,
precision controls and final SCF threshold. Extra first-step work is a
deliberate cost of rebuilding a checkpoint that lacks wavefunctions.

Level stability is **not** a rigorous eigenvector residual bound, and does
not certify final SCF convergence. Debug residual audits and complete SCF
energy/density comparisons remain required before adopting this option.
It does not repair a checkpoint whose mixed potential has already diverged.

## Experimental reconstruction bound policy (2026-11-14)

`PARSEC_RESTART_REFINE_LOWER_BOUND=smart` applies the existing SUBSPACE smart
filter-bound rule to the **additional fixed-potential reconstruction passes**.
`inherit`, the default, keeps the original behavior. The initial solve and
subsequent ordinary SCF calls retain their original settings. The reconstructed
subspace, including its bound history, is retained; this is an algorithmic
change and does not promise bitwise-identical SCF trajectories.

The motivation is the 95,648-electron diagnostic: after 29 extra passes its
monotone lower bound remained at approximately +0.193 Ry, and its maximum
level change was 2.10e-4 Ry, above the unchanged 1e-4 Ry gate. A bound inherited
from early approximate Ritz levels can make filtering unnecessarily weak.
The candidate changes that bound policy, without changing the polynomial
degree, Hamiltonian, FP64 arithmetic, pass limit or acceptance tolerances.
It copies settings objects only, with no additional wavefunction array.

The override covers scalar/sector and built-in k-point solvers, including
spin channels. The k-point settings factory is restored even if reconstruction
fails. A custom k-point solver without a KPointJob is rejected explicitly.

CPU validation: 54 tests passed and 17 CUDA-only tests were skipped. Four
independently diagonalized real/complex fixtures passed accuracy checks. In
the shifted-spectrum fixtures, reconstruction dropped from 47/50 to 10/11
passes; these are work-count probes, **not GPU speed measurements**. Small
physical complex-k-point fixtures with and without spin agree with ARPACK to
about 1e-14 Ry. The subsequent expanded CPU regression passed 134 tests with
27 skips. Large GPU evidence is summarized below; performance adoption is
still pending.

The isolated `restart_bounds64_gb200_20261114_v1` campaign in the parent
RSDFT workspace compares inherit/smart using the same healthy checkpoint,
64 GB200s and unchanged physical inputs. The old frozen campaign is retained
as a control. Neither completion of a reconstruction gate nor a CPU result
constitutes production adoption.

## Evidence as of 2026-11-13

For the 125,272-electron cluster on 64 GB200s, the healthy SCF-5 checkpoint
was held fixed. Its first resumed SCF residual was 9.0906888391 with three
initial filtering cycles. A six-cycle diagnostic produced 0.1377455741,
about 66 times smaller. All four sectors reported no QR fallback in these
six cycles. This supports insufficient subspace reconstruction as a cause
of the bad restart; it does not yet validate the adaptive repair or full
SCF convergence.

The repair's CPU regression suite has 49 passes and 17 CUDA-only skips.
An additional actual two-k-point SCF test, with and without spin, agrees
with an independent ARPACK solve to 1.05e-14 Ry in levels and 1.56e-14 in
maximum density difference for its small fixture. Large GPU validation is
kept separate from these CPU results.

Validation campaign:
`campaigns/parsec_memory_20261111/restartfix64_gb200_20261113_v2` in the
parent RSDFT workspace. It compares a 256-electron Gamma-point checkpoint
with the repair off/on, then runs complete 95,648- and 125,272-electron
cluster restarts from pinned healthy parents. These are diagnostic runs;
formal capacity submissions remain held until the repair is validated.

The first complete 64-GB200 Gamma checkpoint comparison has finished:
256 electrons required eight SCF steps with the option off and one with
it on. Both used the original SCF threshold and had a 917 MiB sampled
peak. The total energy difference was -6.4051e-7 Ry (-2.5020e-9 Ry per
electron). The density L1 difference was 0.00643464 electrons and the
maximum occupied-level difference 2.85e-5 Ry. Empty requested levels do
change (maximum across all requested levels 0.0110147 Ry); the final
all-working-state residual maximum decreased from 0.02870 to 0.007168 Ry.
Thus this is not bitwise spectral parity. At that stage the large checks
were still pending; no production adoption was claimed.

## GPU update, 2026-11-15 00:46 PST

The smart-bound candidate has completed the 95,648-electron restart on
64 GB200s: 33 extra reconstruction passes, then one converged resumed SCF
step. Reconstruction took 4,729.07 seconds; total elapsed time was 5,239.92
seconds and sampled peak 97,105 MiB. Weighted SCF residual was 6.66e-5
against the original 2e-4 threshold. Energy differed from the healthy parent
by -7.84725e-4 Ry (-8.20430e-9 Ry/electron). Actual working-state residual
maxima across sectors were 0.00717–0.00815 Ry, so this does not certify
every eigenvector to 1e-4 Ry.

The inherit control on the same allocation is still running. Final array
accuracy and matched timing review remain required, and the 125,272-electron
repair is not yet validated. The
[detailed report](../examples/0d_nanodiamonds/large_system_experiments/README.md)
contains inputs, numerical evidence, profiling and remaining optimization
work; no production-adoption decision has been made.
