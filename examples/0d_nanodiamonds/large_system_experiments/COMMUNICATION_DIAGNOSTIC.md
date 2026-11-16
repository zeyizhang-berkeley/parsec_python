# Communication diagnosis on 64 GB200 GPUs

## Completed communication probe

The November 15 communication-only probe passed on **64 GB200 GPUs across
16 nodes**, with four MPI ranks and four GPUs per node. All **64 unique rank
reports** passed the required checks, and all 64 completion receipts reported
successful exits across the expected 16 hosts. The sanitized aggregate is in
[communication_probe_20261115.json](data/communication_probe_20261115.json).

This experiment tested the communication path separately from a DFT calculation.
It used four sector communicators of 16 ranks each, matching the communicator
sizes used by the large nanodiamond restart. It checked global control operations
and sector membership, then exercised sector `Alltoallv` with nonuniform counts
and deterministic FP64 integer patterns. Every received element was compared
exactly with its expected pattern.

Two transfer paths were tested at logical send sizes of **1, 64 and 512 MiB per
rank**:

- Host arrays through MPI.
- GPU arrays copied to the host, exchanged through MPI, and copied back to the GPU.

The six path/size combinations were cycled through the phase loop. A phase round
means one path at one size; it is not an SCF iteration or six complete exchanges.

| Measurement | Observed result |
|---|---:|
| GPU topology | 16 nodes × 4 GB200 GPUs |
| MPI topology | 64 ranks; four sectors × 16 ranks |
| Completed phase rounds per rank | 1,000 |
| Active phase-loop duration | **164.93 s** |
| Launch-to-finish duration | **233.34 s** |
| Sampled maximum device memory per GPU | **1,727 MiB** |
| Logical bytes sent, summed over ranks | 11.7001 TiB |
| Required path/size combinations | 6 of 6 on every rank |
| Data comparison | Exact FP64 integer-pattern equality |

The configured stopping conditions were 1,000 phase rounds or 600 active seconds.
The run reached the **round cap after about 165 seconds**. It was **not a
ten-minute endurance test**. Launch duration includes work outside the active
phase loop. Neither duration is an isolated network transfer timer: allocation,
pattern generation, validation, synchronization and staging contribute to the
measurement. Logical send volume includes self-destinations and transfers within
a node; it must not be divided by elapsed time and presented as NIC bandwidth.

The accepted diagnostic result was recorded at **2026-11-15 19:34:27 PST**.
The platform subsequently showed `killed` because the diagnostic controller
released the allocation after completion. This cleanup state does not contradict
the completed rank evidence. Platform stream collection also encountered a query
timeout, so completeness is established from the saved summary and rank receipts,
not from an assumption that every platform log line was collected.

## What this result establishes

The result shows that these transfer patterns, communicator sizes, rank placement
and GPU bindings worked in that allocation. It makes a persistent failure in
every small-buffer exchange less likely, but it does not identify the cause of
the large DFT failures or prove that the transport is reliable under every load.

The device-memory peak was about **1.69 GiB per GPU**. The completed large
95,648-electron restart previously reached **94.83 GiB per GPU**. This probe did
not reproduce that resident-memory footprint, orbital layout, long-lived buffer
reuse, eigensolver sequence, or full SCF calculation. Its passing result therefore
does not establish large-DFT memory stability, numerical accuracy, or the maximum
system size. The preemptible resource also supplies no long-term availability
guarantee. Existing full-SCF submission and production-adoption holds remain
separate from this communication-only acceptance.

## Optional tracing for the actual DFT exchanges

The branch now includes an observational trace switch:

```bash
export PARSEC_MPI_TRACE=1
```

The switch is **off by default**. When enabled consistently on all ranks, it emits
`PARSEC_MPI_TRACE` JSON independently of `PARSEC_DEBUG` and its leader selection.
It covers the existing real shared-basis and complex-column `Alltoallv` wrappers.
Records identify world and communicator ranks, communicator size, a process-local
group instance, exchange sequence, call site, transport and buffer layout.

Stage records surround the existing producer synchronization, host staging,
MPI entry/exit, upload and, where already present, post-upload synchronization.
An upload marked `enqueued` does not claim that device execution has completed.
The helper reads metadata such as shape, dtype, byte size, counts and offsets;
it does not read orbital values or retain array references. It retains at most
**256 layout signatures per communicator-group object**, emitting the full layout
when a signature is first observed or reappears after eviction.

Tracing adds **no MPI operation, device synchronization, numerical array operation
or mathematical change**. It does add CPU metadata work and flushed logging, so
instrumented wall times cannot be treated as an uninstrumented performance result.
Best-effort logging failures are reported without changing the solver's transport
sequence. The live collector handles ordinary debug records and a rolling
five-minute trace window separately; saved per-rank logs remain the primary
evidence when platform collection is incomplete.

The targeted source CPU validation recorded **45 passed tests, 7 skipped tests
and no failures**, including checks of the actual exchange-wrapper operation
order with tracing on/off. Separate adapter checks enforce the frozen trace
policy. Skipped tests are not GPU passes. The first instrumented large-memory
DFT allocation subsequently ran, but terminated before SCF, as detailed below.
**Full GPU numerical validation remains pending.**

The instrumented comparison keeps the 95,648-electron inherited-bound case,
healthy parent checkpoint, reconstruction cap of 100, `Max_Iter=100`,
`FF_MaxIter=3`, precision controls and full-SCF policy unchanged. The trace is
intended to locate the first missing stage or inconsistent exchange metadata
under the actual memory footprint. Any later correction still requires numerical
validation before a performance or production-adoption claim.

## First large-memory trace outcome, reviewed November 16

The first instrumented allocation ran the **C23191H2884 nanodiamond cluster**
(26,075 atoms and **95,648 valence electrons**) on **64 GB200 GPUs, 16 nodes ×
4 GPUs**. It retained the healthy parent, inherited spectral lower bound,
reconstruction cap of 100, `Max_Iter=100` and `FF_MaxIter=3`. The sanitized record is
[restart_trace_termination_20261116.json](data/restart_trace_termination_20261116.json).

| Measurement at the last completed reconstruction pass | Result |
|---|---:|
| Completed extra reconstruction passes | 33 |
| Next reconstruction pass started | 34 |
| Maximum adjacent-pass level change | **0.0001622911340486699 Ry** |
| Required level-change threshold | 0.0001 Ry |
| Consecutive passing checks | 0 of 2 required |
| Elapsed reconstruction time | 4,749.62 s |
| Completed SCF iterations | **0** |
| Latest four sector eigenpair audits | All checked values finite |

Level changes continued to decrease over the last five completed passes, from
0.000209894 to 0.000162291 Ry. This is progress toward the reconstruction gate,
not an SCF convergence result. The gate had not passed when the allocation ended.
Finite audit values also do not establish converged eigenpairs or final SCF
precision; the audit residual and adjacent-pass level change are different
quantities.

The platform recorded failure at **2026-11-15 22:53:43.619 PST**. Launcher logs
showed **SIGTERM on all 16 workers** at approximately 22:53:38–39 PST, and all 16
instances reported exit code 2. The inspected platform reason fields were empty.
These observations establish a coordinated termination signature, but **do not
identify its initiating cause or prove preemption**. The eight-hour allocation
guard had not expired and had not requested a stop.

The inspected terminal-window queries returned no rows for `Alltoallv`,
`Traceback` or `Segmentation`. This particular query evidence did not reproduce
the native MPI fault seen in earlier attempts. Absence from these keyword queries
does not rule out an unreported fault or establish a platform root cause.

The collected trace also has a coverage limitation: the latest per-rank trace
records were approximately **182 seconds older than the final reconstruction
record**. Therefore, a last observed `mpi_enter` or staging event cannot locate
the terminating operation. The saved count review at **2026-11-15 22:38:28 PST**
found **472,832 of 472,832 resolved sender/receiver count pairs matched**, covering
1,847 fully resolved collective sequences. Another 911 complete-entry sequences
lacked sufficient layout metadata, and 213 sequences had partial rank coverage.
These counts describe that earlier snapshot, not complete terminal coverage. No
mismatch was observed in its resolved subset; unresolved coverage remains
unresolved.

A separate monitoring defect left the status summary stale: MPI trace records
use `stage`, while ordinary debug records use `event`. The summarizer previously
assumed every record contained `event`. The corrected summarizer streams the
mixed log, skips irrelevant or incomplete records, and successfully reproduced
the terminal pass-33 summary. This change affects monitoring only.

This allocation produced **no accepted full-SCF result and no validated speedup
or memory reduction**. A conditional recovery supervisor was started at
**2026-11-16 01:08 PST**, after 35 controller regression tests passed locally and
on the development machine. That timestamp records supervisor startup, not GPU
readiness or scientific acceptance. The recovery release changes only the
scheduler allocation/retry fields in the manifest; scientific source, adapters,
inputs, sequence, runtime and healthy parent remain identical.

The controller permits up to 11 allocations within the existing campaign budget.
It retries only after verifying the exact terminal allocation, all 16 worker exit
codes and SIGTERM records, and the absence of contradictory failure/stop/expiry
evidence. Missing evidence defers without consuming a new allocation; numerical,
native or memory errors block recovery. The classification remains
`external_termination_unclassified`, not confirmed preemption. Terminal log
queries cover the completion-centered window with five minutes on each side,
after allowing at least five minutes for ingestion. Collection drains for a
bounded five to ten minutes and continues to label platform coverage as partial.
The parent does not contain wavefunctions, so recovery must rebuild them; it
cannot resume directly at reconstruction pass 34.

## Bootstrap interruption and recovery-controller correction

The first recovery allocation was submitted on **2026-11-16 at 01:18:27.881 PST**,
started at 01:19:27.640 and failed at **01:21:45.164 PST**. All 16 workers received
SIGTERM and reported exit code 2; the initiating cause remains unknown. All 16
nodes had committed staging records with the expected source revision, but the
application had not published its readiness marker or started the DFT sequence.
This attempt therefore added **no reconstruction or SCF result**. The sanitized
incident record is
[restart_bootstrap_recovery_20261116.json](data/restart_bootstrap_recovery_20261116.json).

The recovery controller then stayed in `waiting_for_terminal_evidence` until the
status review around **10:00 PST**. Its original gate required `ready.json` even
when termination occurred before readiness, so a legitimate startup interruption
could never satisfy that gate. This was a controller limitation, separate from
the numerical solver and from the unknown cause of the termination.

The corrected gate recognizes an interrupted bootstrap only with verified frozen
entrypoints, consistent committed staging records, an otherwise empty application
control area, all 16 SIGTERM/exit-code records, and failure queries covering the
entire short allocation lifetime (at most 300 seconds). Missing evidence still
defers; positive contradictions block. Additional broad `Error`/`failed` searches
apply only to this verified bootstrap phase, preventing ordinary numerical debug
fields such as `maximum_norm_squared_error` from being mistaken for program failures.
The classification remains an **unclassified external termination during
bootstrap**, not confirmed preemption.

The private controller correction passed **44 regression tests**, including
**31 retry-policy tests**, locally and on the development machine. It was deployed
at **10:09:15 PST** and the CPU supervisor restart was verified. The GPU bundle,
scientific source, precision controls and healthy checkpoint were unchanged. The
allocation cap remained 11, with one used and ten not yet submitted at patch time.
Supervisor recovery alone does not establish that a replacement GPU allocation
has started or that any numerical optimization has been validated.

## Native MPI fault in the second recovery allocation

The second recovery allocation reached the MPI readiness checks on **64 GB200
GPUs, 16 nodes × 4 GPUs**. The platform recorded failure at
**2026-11-16 18:17:20.927 UTC (10:17:20.927 PST)**. This attempt used the same
pinned scientific-source and runtime archives, and archived 95,648-electron
case as the preceding recovery release. It **did not reach checkpoint reconstruction or SCF**. The
sanitized evidence is in
[readiness_native_failure_20261116.json](data/readiness_native_failure_20261116.json).

| Readiness evidence | Observed result |
|---|---|
| MPI startup | All 64 unique ranks reported startup |
| Shared-basis checks | `ALL PASSED` recorded |
| GPU unit tests on the coordinator rank | 16 tests, `OK`; reported test time 1.335 s |
| Native stack evidence | 14 observed rows containing `PMPI_Alltoallv_c+0x548` |
| Native signal evidence | 14 observed SIGSEGV rows with fault address `0x110` |
| QR comparison completion | No completion result observed |
| Terminal worker review | All 16 workers received SIGTERM and reported exit code 2 |
| Reconstruction / SCF results | None |
| Recovery decision | Blocked because of positive native-failure evidence |

The native fault distinguishes this attempt from the earlier bootstrap
interruption. SIGTERM and exit code 2 describe the terminal worker state; they
do not establish preemption or explain the earlier segmentation fault. The
inspected platform reason fields did not establish an initiating cause, and the
allocation guard had not requested a stop. The 14 stack and signal rows are
**observed log counts**, not a complete census of failed MPI ranks. The saved
SIGTERM query extract is also truncated; the all-worker terminal aggregate
comes from the separate terminal review.

The recorded sequence establishes successful shared-basis checks and successful
coordinator GPU unit tests before failure, with no subsequent QR completion
record. It does **not** identify the exact application call that entered the
faulting collective. In this frozen adapter, readiness runs in check mode with
no physical case attached, so its MPI tracing remained **off** even though the
later DFT case enabled tracing. Consequently, the available output lacks the
all-rank exchange-entry/exit metadata needed to connect the native frame to a
specific readiness call and buffer layout.

The cause therefore remains unresolved between the application caller, the MPI
implementation, and the loaded runtime environment. The observed symbol and
fault address are evidence of a native failure, not proof of an MPI-library bug
or an invalid application buffer. A useful follow-up must isolate the QR check
from the inherited readiness sequence, capture stage boundaries on every rank,
and identify the loaded MPI components before interpreting another failure.
This section records the failed attempt only; it makes no claim that a new
diagnostic has started or succeeded.

The earlier communication-only pass remains valid for its tested transfer
patterns and low-memory workload. It cannot overrule this readiness failure or
establish that the large DFT calculation is stable. This attempt adds **no
accepted numerical result, validated speedup, or validated memory reduction**.

## Source identity

The successful communication probe used scientific source
`2c96107+restart100-aaca2df3fea923efe323`, unchanged from the 100-limit base release.
Its separate adapter bundle SHA256 was
`7d1d5949b4a4686f3ccfde7cb9c05c1b2752314f27df63938c2f28996b2cef1a`.
The underlying scientific base bundle was
`7ebe86ec8f56d10c7a23e224fe5ac7b694cdeaed16ee9e27897c3a177e3521b0`,
with GB200 runtime
`59067e1e373f0a5e5114cc47c0ad6edcf708ac13ff3eae2248d241fb5ed8b4b8`.

These identities describe the **completed communication probe**. The terminated
instrumented DFT allocation used source
`2c96107+restarttrace-4a91d62cd7c813410426`, with release SHA256
`9d7c43a81a8f335e8401215df2bb2d3b0facd6bf40c5cde8e5a5171cccb4e449`.
The metadata-only recovery release is
`ca0af36744679fe413c0acf39a7aef8f6230f2aa42a32af66d094b25b0331569`;
it retains the same scientific source revision and source inventory.
Historical `smart` results continue to belong to their earlier frozen bundle.
Internal endpoints, hosts, process identifiers, raw logs and checkpoints are
intentionally absent from the public aggregate.
