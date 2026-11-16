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
policy. Skipped tests are not GPU passes. **Validation of the new tracing code on the real large-memory GPU
case is pending**; this document does not claim that the instrumented DFT task has
started or completed.

The intended next comparison keeps the 95,648-electron inherited-bound case,
healthy parent checkpoint, reconstruction cap of 100, `Max_Iter=100`,
`FF_MaxIter=3`, precision controls and full-SCF policy unchanged. The trace is
intended to locate the first missing stage or inconsistent exchange metadata
under the actual memory footprint. Any later correction still requires numerical
validation before a performance or production-adoption claim.

## Source identity

The successful communication probe used scientific source
`2c96107+restart100-aaca2df3fea923efe323`, unchanged from the 100-limit base release.
Its separate adapter bundle SHA256 was
`7d1d5949b4a4686f3ccfde7cb9c05c1b2752314f27df63938c2f28996b2cef1a`.
The underlying scientific base bundle was
`7ebe86ec8f56d10c7a23e224fe5ac7b694cdeaed16ee9e27897c3a177e3521b0`,
with GB200 runtime
`59067e1e373f0a5e5114cc47c0ad6edcf708ac13ff3eae2248d241fb5ed8b4b8`.

These identities describe the **completed communication probe**, not the new
instrumented DFT candidate. Historical `smart` results continue to belong to
their earlier frozen bundle. Internal endpoints, hosts, process identifiers,
raw logs and checkpoints are intentionally absent from the public aggregate.
