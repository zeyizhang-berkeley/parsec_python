# Python reproduction of the ARES CH4/CF4 C 1s benchmark

This directory reproduces the PBE part of the supplied ARES benchmark with
the native Python real-space solver.  It does not call ARES or PARSEC.

## Directory correspondence

The supplied `0_ARES_CH4_CF4` tree contains three distinct pieces:

- `CH4/` is an earlier standalone/manual ARES trial (`ares.in`, `POSCAR`,
  ordinary C/H UPFs, Slurm script and output).  Its zero-length `ares.log` is
  not the production reference.
- `CF4/` contains only the source `CF4.xyz` geometry; it is not a calculation.
- `test_CH4_CF4/` is the complete, generated eight-case validation suite and
  is the reference used here.  Its `setup.py` builds cases, `run*.qsub`/`.sh`
  launch them, and `analyze.py` forms binding energies and shifts.  Salvaged
  and Slurm files are run history, not additional physical cases.

The production calculation tree is

```text
0_ARES_CH4_CF4/test_CH4_CF4/
  pbe/{CH4,CF4}/{IS,FS_1s}/
  b3lyp/{CH4,CF4}/{IS,FS_1s}/
```

`IS` is the neutral ground-state calculation with the ordinary carbon
pseudopotential.  `FS_1s` is the fully screened final state with a carbon
pseudopotential generated after removing one 1s electron.  The ARES `b3lyp`
tree first performs the same PBE SCF and then evaluates one non-self-consistent
B3LYP energy; it is not a B3LYP SCF calculation.

Inside each terminal ARES case, `ares.in` holds numerical controls, `POSCAR`
the coordinates, `ISOCAR` the species-to-PP assignment, and the copied UPFs
the actual pseudopotentials.  `ares.log` is the scalar/energy report and
`ares.band` the eigenvalue output; `srun.err` is launcher stderr.

The Python PBE tree mirrors the physical cases:

```text
python_pbe/
  CH4/{IS,FS_1s}/
  CF4/{IS,FS_1s}/
  pseudopotentials/
  analyze.py
```

The input label `C-1s` means “carbon with one full 1s hole.”  It remains tied
to physical element `C` through `Element_Symbol: C`, so the label changes PP
selection without inventing a chemical element.  The final-state inputs use
`Net_Charges: 1 e`; the C-1s PP has five valence electrons, leaving the same
explicit electron count as the ARES `DCHARGE=-1` convention.

## Numerical correspondence

The molecular coordinates, grid spacings, finite-difference order, number of
states, electron temperature, PBE pseudopotentials, and spherical domains are
taken from the ARES runs.  The radii are written with enough precision to give
the same Cartesian box and active point counts:

| molecule | spacing | Cartesian box | active points |
|---|---:|---:|---:|
| CH4 | 0.12 angstrom | 102 x 102 x 102 | 539,152 |
| CF4 | 0.07 angstrom | 182 x 182 x 182 | 3,103,688 |

The Python SCF uses its audited Chebyshev-filtered subspace solver and Anderson
potential mixing.  ARES reports a density residual, whereas the Python solver
tests a charge-weighted potential residual; the numerical tolerances therefore
do not have identical meanings.  The CF4 threshold is `5e-5 Ry`, at which its
total energy is already stable much more tightly than the core-level chemical
shift being tested.

ARES inputs request spin polarization, but their final spin-up and spin-down
bands are paired and equal for these closed-shell, fully screened cases.  The
present Python calculations are therefore unpolarized.  A genuinely magnetic
or fractional-hole calculation will require a spin-PBE implementation before
it is a valid reproduction.

## Pseudopotential and all-electron reference

The POTRE files were converted from the exact UPF files in the supplied ARES
case, which were themselves produced by FHI98PP.  For example, from the
repository root:

```powershell
.venv312\Scripts\python.exe src\tools\upf_to_parsec.py `
  examples\0_CH4_CF4\0_ARES_CH4_CF4\test_CH4_CF4\pbe\CH4\IS\C.pbe-mt-cpi.UPF `
  examples\0_CH4_CF4\python_pbe\pseudopotentials\C_POTRE.DAT

.venv312\Scripts\python.exe src\tools\upf_to_parsec.py `
  --allow-ionized-reference `
  examples\0_CH4_CF4\0_ARES_CH4_CF4\test_CH4_CF4\pbe\CH4\FS_1s\C.pbe-mt-cpi-1s.UPF `
  examples\0_CH4_CF4\python_pbe\pseudopotentials\C-1s_POTRE.DAT
```

The converter resamples the UPF mesh by cubic interpolation in log r. The
files here were made again that way on 23 November 2026; before, they were
the linear resampling of its first version (`--interpolation linear`),
whose values between the points of the UPF mesh rippled: their fourth
differences were as large as their second ones (pseudo wave functions and
valence density; 0.15 to 0.27 of them for the C and F potentials), and the
eigenvalues of the pseudo atom, with the screening potential of FHI98PP,
were 0.06 (H) to 0.33 mRy (F) off those of the generator (the cubic files:
within 2.1e-7 Ry). The four files agree with those of
[`../../0d_xps_core_levels/pseudopotentials`](../../0d_xps_core_levels/README.md),
generated from the inputs of the paper, to 1e-11 in every value (the last
printed digit), apart from the name of the source in their second line. The UPF files of
the ARES case are not part of the repository;
[`../paper_pp_inputs`](../paper_pp_inputs/README.md) holds the FHI98PP
inputs that regenerate them.

The initial and final carbon pseudopotentials have different ionic charges,
so their raw pseudopotential total energies do not share an energy zero.  Each
species input therefore includes the FHI98PP all-electron-minus-pseudo atomic
reference correction.  Binding energies must be formed from the reported
`Reference-corrected all-electron total`,

```text
BE(C 1s) = E_AE(FS_1s) - E_AE(IS).
```

The chemically transferable validation quantity is the shift

```text
Delta BE = BE(CF4) - BE(CH4).
```

## Running

From any one of the four case directories:

```powershell
..\..\..\..\..\.venv312\Scripts\python.exe `
  ..\..\..\..\..\src\parsec_python\main.py parsec.in `
  --backend auto --symmetry auto --log parsec.out `
  --no-archive
```

Remove `--no-archive` and add `--output parsec_result.npz` when the converged
grid fields are needed.  CF4 has more than three million active points, so its
archive is intentionally omitted for an energy-only reproduction.

After all four jobs finish, compare directly with the supplied ARES logs:

```powershell
.venv312\Scripts\python.exe `
  examples\0_CH4_CF4\python_pbe\analyze.py
```

## PBE reproduction results

All four Python calculations converged.  Energies below are the
reference-corrected all-electron values, in eV:

| system | Python | ARES | Python - ARES |
|---|---:|---:|---:|
| CH4 IS | -1101.53972976 | -1101.48773900 | -0.05199076 |
| CH4 FS_1s | -804.38248842 | -804.34015900 | -0.04232942 |
| CF4 IS | -11909.65954868 | -11909.43066320 | -0.22888548 |
| CF4 FS_1s | -11601.94393513 | -11601.71550480 | -0.22843033 |

The physically relevant energy differences reproduce much more closely:

| quantity | Python | ARES | Python - ARES |
|---|---:|---:|---:|
| BE(CH4) | 297.15724134 | 297.14758000 | +0.00966134 |
| BE(CF4) | 307.71561355 | 307.71515840 | +0.00045515 |
| CF4-CH4 shift | 10.55837221 | 10.56757840 | -0.00920619 |

The larger offsets in individual absolute totals are systematic discretization
and implementation differences.  They cancel within each matched ordinary/
core-hole PP pair; the final chemical-shift disagreement is about 9 meV.
With the linearly resampled tables used before 23 November 2026 (the same
code and defaults, measured on 23 November 2026) the totals are 0.019 (CH4
IS) to 0.117 eV (CF4 FS_1s) lower, the binding energies 297.15419064 and
307.70893835 eV and the shift 10.55474771 eV (13 meV below ARES); SCF
iterations are the same.

For these first runs (the kept logs, of 23 November 2026, one NVIDIA A100
per case and four cases side by side on a node, no symmetry cache, after one
throwaway warm-up run on each GPU), CH4 IS/FS used 28/21 SCF iterations and
CF4 IS/FS used 40/37.  CF4 wall times were 43/40 seconds.  Its dominant cost
was the Hartree linear solve (15/14 seconds), followed by GPU diagonalization
(12/11 seconds); PBE evaluation itself was not the dominant kernel. Since 2026-11-07 that linear
solve runs on the GPU by default (`PARSEC_HARTREE_LINEAR_BACKEND`,
`src/parsec_python/acceleration/Hartree/GPU_HARTREE.md`);
`PARSEC_HARTREE_LINEAR_BACKEND=native` takes the host CG of the logs of
1 September 2026, whose CF4 runs took 793/627 seconds on the tested hybrid
machine (host CG 446/407 seconds, GPU diagonalization 225/111 seconds), with
the same energies to the printed digits. A `martins_new` projector is zero
beyond the largest cutoff radius of its channels (`PARSEC_PROJECTOR_TAIL=auto`,
since 22 November 2026). The cubic files differ from their local potential by
up to 2.6e-4 Ry (C, C-1s) and 7e-4 Ry (F) at the first one or two mesh points beyond
that radius, where the interpolation reaches across it; with
`PARSEC_PROJECTOR_TAIL=off` those points enter the projectors and the totals
are 4.3e-6 (CH4 IS) to 8.8e-6 Ry (CF4 FS_1s) lower.

## B3LYP status

The ARES one-shot B3LYP refinement cannot be represented by another local
`V_xc(r)` switch.  It contains nonlocal Hartree-Fock exchange and therefore
requires occupied-orbital pair densities and Poisson solves (or an equivalent
exact-exchange algorithm), in addition to B88 exchange and LYP/VWN
correlation.  It is deliberately not approximated in this PBE milestone.
Implementing it as a separate post-SCF energy module preserves both the code
architecture and the physical meaning of the ARES result.

That module now exists (`Energy_Refinement: b3lyp`).  The same four cases
with the refinement, the comparison with the ARES `b3lyp` tree and the reason
why the energy parts of the two codes differ are in
[`../python_pbe_b3lyp`](../python_pbe_b3lyp/README.md).  Its tables convert
Rydberg to eV with the CODATA constant; the tables above use the `[eV]` lines
of `parsec.out` (13.6058 eV per Rydberg), which puts their binding energies
2.3 to 2.4 meV higher.
