# FHI98PP versus ATOM: audited recommendation

## Decision

Use **FHI98PP as the reference backend for the core-hole milestone**, and use
**ATOM as an independent generator/checker and the preferred source of broad
native formats**. “Better” is requirement-dependent; neither code replaces a
modern transferability and solid-state validation protocol.

| Capability | FHI98PP | ATOM 6.x |
|---|---|---|
| NCPP construction | Hamann and Troullier--Martins | Troullier--Martins (`tm2`) |
| Core-hole evidence | Checked Si 2p and P 1s full-core-hole regressions | Arbitrary atomic configurations are possible, but this checkout has no validated core-hole workflow |
| Ghost analysis | Gonze spectrum criterion, bound spectra, and energy-dependent logarithmic derivatives through `pswatch` | Gonze spectrum criterion in `kb_conv`/`atom_all`, plus separate log-derivative workflow |
| Local-channel handling | `pswatch -l L`; historically one choice per run | Explicit local channel or supremum; portal automation tries all physical channels |
| Relativity | Nonrelativistic/scalar-relativistic, spin-unpolarized | Nonrelativistic, spin-polarized, and relativistic/spin-orbit paths |
| NLCC | Supported through `rnlc` | Supported; improved smooth GGA construction documented |
| Defaults | 66 unique TM and 65 Hamann element templates; README warns not to use uncritically | Built-in defaults through Og; separate portal inputs cover 94 elements |
| Formats | CPI/FHI; conversion needed for modern formats | PARSEC POTRE, UPF v2, ABINIT PSP8, SIESTA PSF, CPW2000/POTKB |
| Code base | Compact historical fixed-form Fortran, validated but difficult to evolve | Documented modern Fortran 90 and actively modernized in this checkout |
| License evidence in checkout | README defers to the distribution source; clarify before source translation | GPLv2 file/source notices; check GPLv2-or-later compatibility before copying into GPLv3 project |

FHI98PP wins the immediate decision because the requested feature is not just
“an input occupation can be changed”; it has checked core-hole energies,
electron counts, PARSEC smoke cases, and ghost-free reference products. ATOM
is the better long-term cross-check and exporter reference.

## What the PARSEC portal actually does

The portal advertises an ATOM 5.803 Troullier--Martins library. Its downloaded
metadata (generated 2026-08-21) contains:

- 94 elements and 188 accepted records: CA-LDA and PBE for each element;
- `tm2` for all 188 records;
- 148 relativistic and 40 nonrelativistic generating inputs;
- NLCC disabled in all 188 records;
- 136 records where at least one candidate local channel is marked ghosted;
- 52 records where every candidate is safe;
- zero cases where the published `suggestedLocal` is not marked safe.

Thus the portal is a pre-generated, metadata-backed library, not merely a call
to `atom_all Element`. Its important construction step is an automated
`kb_conv` gate over candidate local channels followed by selection of a safe
one. The Python layer now applies the same principle to both backends.

Passing that gate does **not** prove broad transferability. Some portal source
titles explicitly say “Guess,” all entries disable NLCC, and atomic ghost
screening alone does not test equations of state, chemistry, or cutoff
convergence.

## Required quality gates

A production NCPP should pass all relevant levels:

1. generation convergence and finite radial arrays;
2. reference eigenvalue and norm conservation inside every cutoff;
3. correct Coulomb tail, smooth origin behavior, and absence of unintended nodes;
4. all candidate KB local channels classified by the Gonze spectral test;
5. all-electron/semilocal/KB logarithmic derivatives over a useful energy window;
6. excited, ionized, and chemically plausible atomic configurations;
7. kinetic-energy/grid-spacing convergence and projector convergence;
8. molecule/solid benchmarks and equation-of-state comparisons;
9. XC, relativity, NLCC, valence, ionic-charge, and core-hole metadata consistency;
10. format round trips and downstream-reader smoke tests.

Core-hole potentials additionally require explicit FCH versus XCH electron
counting and an atomic reference-energy correction for total-energy
comparisons. A ghost-free result is necessary, not sufficient.

## Verified Python baseline

- Si 2p-hole PBE/TM: AE `-285.79258 Ha`, pseudoatom `-5.87811 Ha`, ionic
  charge 5, all three local choices tested, selected d-local.
- P 1s-hole PBE/TM: AE `-262.28942 Ha`, pseudoatom `-9.57763 Ha`, ionic
  charge 6, all three local choices tested, selected d-local.
- Reference CPI/FHIPP, UPF, and POTRE files are byte-identical when the
  reference prefix is used.
- Independent ATOM Si CA generation produces PARSEC, UPF, PSP8, SIESTA, and
  CPW2000 artifacts.
- Ghosts, indeterminate/ill-defined FHI results, invalid core holes,
  unsupported backend/format pairs, and nonconverged ATOM output are rejected.

## Primary references

- Fuchs and Scheffler, FHI98PP methodology: https://doi.org/10.1016/S0010-4655(98)00201-X
- Troullier and Martins NCPP: https://doi.org/10.1103/PhysRevB.43.1993
- Hamann generalized NCPP: https://doi.org/10.1103/PhysRevB.40.2980
- Gonze, Stumpf, and Scheffler analysis: https://doi.org/10.1103/PhysRevB.44.8503
- Kleinman--Bylander form: https://doi.org/10.1103/PhysRevLett.48.1425
- PseudoDojo grading: https://doi.org/10.1016/j.cpc.2018.01.012
- SSSP protocol: https://doi.org/10.1038/s41524-018-0127-2
- Hamann ONCV: https://doi.org/10.1103/PhysRevB.88.085117
- Core-hole pseudopotential XANES: https://doi.org/10.1103/PhysRevB.66.195107
- UPF documentation: https://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format
- PSP8 documentation: https://docs.abinit.org/developers/psp8_info/
- PARSEC portal: https://parsecdev.vercel.app/pseudopotentials.html
