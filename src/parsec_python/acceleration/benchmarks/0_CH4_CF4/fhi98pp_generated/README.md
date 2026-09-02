# fhi98pp interface smoke test

These files are an independent generation test, not the pseudopotentials used
for the matched ARES reproduction under `../python_pbe`.  That calculation
uses ARES's exact supplied UPFs so PP-generation choices do not enter the code
comparison.

Both references were generated through `python -m pp_generation` using the
local FHI98PP carbon Troullier-Martins template, PBE, automatic local-channel
screening, the QE FHI-to-UPF converter, and the repository UPF-to-POTRE
converter.

| reference | ionic charge | reference electrons | selected local l | ghost-free | AE-PS correction (Ry) |
|---|---:|---:|---:|---:|---:|
| ordinary C | 4 | 4 | 2 (d) | yes | -64.80126 |
| C-1s full hole | 5 | 4 | 2 (d) | yes | -35.46520 |

The core-hole run automatically emits `C-1s_POTRE.DAT` and
`C-1s_FHIPP.DAT` aliases.  `C-1s_generated.report.json` records the full
configuration, channel tests, energies, paths and SHA-256 hashes.

These template defaults are only a functional interface/ghost-screening
test.  Production pseudopotentials still require cutoff convergence,
logarithmic-derivative and chemical-transferability validation.  In
particular, the automatically selected d-local channel differs from the
p-local supplied ARES benchmark PP and must not be substituted into only one
side of that comparison.
