# ML initial-density small-molecule regression suite

This directory compares three ways to initialize the same PBE real-space DFT
calculation:

- `sad`: the current PARSEC-compatible superposition of pseudopotential atomic
  valence densities;
- `scdp`: the SCDP prediction saved by the old code framework;
- `charge3net`: the ChargE3Net prediction saved by the old code framework.

C60 is intentionally excluded. The suite contains the other 14 directories
from `tests/tests/small_molecules`, giving 42 calculations in total.

## What is held fixed

Within each molecule, all three inputs use the same atoms, spherical domain,
0.2 Angstrom grid spacing, eighth-order finite differences, PBE functional,
PBE H/C/N/O norm-conserving pseudopotentials, eigensolver, mixing, temperature,
and convergence criterion. Only `Initial_Density` changes.

The four files in `pseudopotentials/` are those of the library,
`src/pp_generation/NCPPs/PBE/{H,C,N,O}/PARSEC` (Troullier–Martins potentials
of FHI98PP, resampled from its UPF files by `src/tools/upf_to_parsec.py` with
cubic interpolation in log r). Until 23 November 2026 they were the linear
resampling of the same UPF files (`--interpolation linear`), whose values
between the points of the UPF mesh rippled; the outputs and tables below are
those of the cubic files.

The canonical inputs use CHEBDAV degree 15 for the first nonlinear
diagonalization and the existing GPU PARSEC `SUBSPACE` filter on every later
SCF iteration. This was selected only after a 42-case comparison against
CHEBFF; the CHEBFF output of every input is kept beside it
(`parsec_chebff.out`).

The legacy model arrays are volume densities in `e/angstrom^3` on 70x70x70
(7 Angstrom radius) or 90x90x90 (9 Angstrom radius) Cartesian lattices. Their
portable files retain explicit origin and voxel-vector metadata. The old
positive-box coordinates are translated by the same bounding-box midpoint
used to recenter the nuclei, then linearly sampled at the active PARSEC grid
points. Each imported field has the correct valence-electron integral before
normalization; normalization remains enabled as the safe general default.

These are offline regression runs: the SCDP and ChargE3Net repositories are
not invoked. A named provider plus `ML_Density_File` preserves the density's
provenance while loading a previously generated prediction. To run the
models instead, remove the `ML_Density_File` line from an input (the model
then runs before the SCF), or predict the file first:

```powershell
.\.venv312\Scripts\python.exe src\parsec_python\ml_density.py predict `
  --input examples\ml_initial_density\small_molecules\H2O\scdp\parsec.in `
  --pp-dir examples\ml_initial_density\pseudopotentials
```

The [quick start](quickstart/README.md) and the
[tutorial](../../src/parsec_python/MLDensity/README.md) describe the model
environments; [stored and direct predictions](#stored-and-direct-predictions)
compares the two kinds of start.

## Layout

```text
ml_initial_density/
  pseudopotentials/            shared PBE H/C/N/O POTRE files
  small_molecules/
    H2O/                        one directory per molecule
      densities/
        scdp.npz
        charge3net.npz
      sad/parsec.in
      sad/parsec.out
      sad/parsec_chebff.out
      scdp/parsec.in
      scdp/parsec.out
      charge3net/parsec.in
      charge3net/parsec.out
  case_manifest.csv             source-grid and legacy-file provenance
  results_summary.csv           current CHEBDAV convergence/energy/timings
  results_summary_chebff.csv    CHEBFF convergence/energy/timings
  eigensolver_comparison.csv    controlled CHEBFF/CHEBDAV comparison
  chebdav_degree_ch4.csv        degree-15/degree-20 CH4 check
  prepare_small_molecule_cases.py
  run_small_molecule_suite.py
  compare_eigensolvers.py
  plot_chebff_chebdav.py       Figure-5-style convergence/timing plot
  figures/
    chebff_chebdav_iterations_timing.png
    chebff_chebdav_iterations_timing.pdf
    chebff_chebdav_figure_data.csv
```

## Rebuild and run

From the repository root, rebuild the cases from the local legacy result tree:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\prepare_small_molecule_cases.py
```

Validate every generated input without starting an SCF calculation:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py --dry-run
```

Run all 42 calculations with the fastest automatically selected backend:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py
```

Run only one molecule or one method:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\run_small_molecule_suite.py `
  --molecule H2O --method scdp
```

The runner writes `parsec.out` in each method directory and regenerates
`results_summary.csv`. To run one input without the suite helper:

```powershell
.\.venv312\Scripts\python.exe src\parsec_python\main.py `
  examples\ml_initial_density\small_molecules\H2O\scdp\parsec.in `
  --pp-dir examples\ml_initial_density\pseudopotentials --no-archive
```

Regenerate the CHEBFF/CHEBDAV comparison figure and its paired audit table:

```powershell
.\.venv312\Scripts\python.exe `
  examples\ml_initial_density\plot_chebff_chebdav.py
```

The timing panels report `solver_wall_seconds`: setup plus SCF time measured
inside the calculation driver. They do not include Python interpreter startup
in the outer suite launcher. Dashed horizontal lines show the mean for each
initial-density method, matching the convention used in Figure 5 of the paper.

## Results from the current code

The outputs are first runs (no symmetry cache) of 23 November 2026, each on
one NVIDIA A100 (80 GB) with 32 CPUs of a Perlmutter GPU node, four runs side
by side on a node, after one throwaway warm-up run on each GPU; the outputs
before were those of 2 September 2026 on a laptop GPU (RTX 5070 Laptop). All
42 calculations converged. Iteration counts are:

| Molecule | SAD | SCDP | ChargE3Net | Final-energy spread (meV) |
|---|---:|---:|---:|---:|
| C10H8 | 13 | 10 | 8 | 0.044 |
| C2H2 | 13 | 8 | 8 | 0.003 |
| C2H4 | 13 | 9 | 8 | 0.003 |
| C2H6 | 16 | 15 | 13 | <0.001 |
| C3H8O | 13 | 14 | 18 | 0.007 |
| C6H6 | 14 | 10 | 11 | 0.003 |
| CH3CH2CH2OH | 19 | 14 | 18 | <0.001 |
| CH3CH2OCH3 | 12 | 13 | 19 | 0.012 |
| CH3CH2OH | 15 | 12 | 17 | <0.001 |
| CH3CHOHCH3 | 19 | 17 | 20 | <0.001 |
| CH3CN | 16 | 12 | 12 | 0.001 |
| CH4 | 17 | 12 | 9 | 0.002 |
| CO2 | 10 | 12 | 11 | <0.001 |
| H2O | 13 | 12 | 13 | <0.001 |

The mean CHEBDAV iteration counts are 14.50 (SAD), 12.14 (SCDP), and
13.21 (ChargE3Net), compared with CHEBFF's 21.00, 22.00, and 22.07. On the
A100 a run takes 2 to 4 seconds: CHEBDAV was faster in 22 of the 42
calculations, and the summed solver time was 115.41 seconds against
116.15 seconds for CHEBFF (1.01x). A second run of all 84 inputs on the
same day gave 22 of 42 and 114.50 against 115.26 seconds. Per input the two
solvers differ by -0.9 to +0.7 seconds, more than repeated runs of the same
input (0.18 s standard deviation of their difference), and the differences
nearly cancel in the sum. On the laptop GPU of the first audit CHEBDAV was faster
in 40/42 calculations and reduced the summed solver time from 439.85 to
328.56 seconds (1.34x); the two exceptions were SAD cases whose reduced
iteration count did not repay the more expensive first eigensolve. Degree 20
saved one SAD iteration for CH4 (16 against 17) and no other, so the
source-valid minimum degree 15 is used.

The maximum final-energy spread among the three starts is
`3.21e-6 Ry = 0.044 meV`, well below the `2e-4 Ry` SCF criterion. The largest
CHEBDAV-versus-CHEBFF final-energy difference was `4.429e-5 Ry`, also below
that criterion. Detailed results are in `results_summary.csv` and
`eigensolver_comparison.csv` (its CHEBDAV columns are the runs of
`parsec.out`); each `parsec_chebff.out` is the CHEBFF run corresponding to
the current `parsec.out`.

Against all 84 runs with the former files on the same code and defaults
(23 November 2026), the cubic pseudopotential files (see "What is held
fixed") raise every total energy by 1.05e-3 Ry (H2O) to 1.16e-2 Ry (C10H8),
1.05e-3 to 1.6e-3 Ry per C, N or O atom (the hydrogen file is purely local
and changed little), and change two CHEBDAV iteration counts (CH4 SAD 18 to
17, CH3CHOHCH3 SCDP 19 to 17) and one CHEBFF count (CH3CN ChargE3Net 24 to
25). The other changes against the outputs of 2 September (C2H2 ChargE3Net
with CHEBDAV, 10 to 8; C2H2 and CO2 with CHEBFF) come with the former files
too on the current code, whose energies with those files lie 3.2e-5 (CH4) to
3.4e-4 Ry (C10H8) above the former ones by the defaults of 23 November 2026
(for these PBE inputs, the Kleinman-Bylander quadrature on the logarithmic
mesh).

## Why CHEBDAV reveals the ML benefit

Old-architecture `diagmeth=3` used `first_filt` for the first SCF iteration
and `chefsi1` (Chebyshev-filtered subspace iteration) thereafter. Current
CHEBFF is the closest PARSEC implementation of that policy, and both current
first-solver choices already use the same `SUBSPACE` implementation after the
first iteration. There is therefore no missing “chebsup” later-SCF path.

CHEBFF intentionally performs a fixed amount of approximate first-filter
work without accepting individual Ritz pairs by residual. CHEBDAV spends more
time building and residual-testing a better first eigensubspace. For these
ML-density tests, that additional first-step work prevents eigensolver error
from obscuring differences in the starting Kohn--Sham potential. On the
laptop GPU of the first audit the fewer later SCF/subspace iterations
recovered its cost in nearly every case; on an A100, where these runs take
seconds, the two about balance.

## Stored and direct predictions

The stored fields are the starting densities of the old code
(`*_init_rho.npy`): already normalized, so their electron count says nothing
about the raw model output, and predicted on boxes whose points lie off this
grid (by 0.087 Å in y for water), so they are imported by linear
interpolation, which lowers the density peaks near the nuclei (water,
ChargE3Net: 1.007 instead of 1.123 e/bohr³ at the maximum, 5.9 % L1
difference). Predicting at the exact grid points with `ml_density.py predict`
changed the SCF iterations as follows (A100, 2026-10-18, with the linearly
resampled pseudopotential files of that time; final energies within 1.1e-6 Ry
of SAD; the direct predictions need the model environments and were not
repeated with the cubic files, with which the stored starts of these four
molecules take the iterations of the table above, CH4 SAD 17 instead of 18):

| Molecule | SAD | ChargE3Net stored / direct | SCDP fast stored / direct | SCDP accurate, direct |
|---|---:|---:|---:|---:|
| H2O | 13 | 13 / 9 | 12 / 12 | 11 |
| CH4 | 18 | 9 / 7 | 12 / 10 | 7 |
| C6H6 | 14 | 11 / 15 | 10 / 9 | 9 |
| C10H8 | 13 | 8 / 8 | 10 / 7 | 8 |

The geometries of this suite are not relaxed: water has O–H bonds of 1.10 Å
(0.96 Å at equilibrium) and CO2 C=O bonds of 1.40 Å (1.16 Å), outside the
equilibrium structures of QM9 on which both models were trained. SCDP's fast
model gives 7.21 instead of 8 electrons for this water and 7.77 for a relaxed
one. The SCF result does not depend on the start; the comparison of starts
does.

See [the complete interface guide](../../src/parsec_python/MLDensity/README.md)
for direct-model environments, schemas, units, cache behavior, normalization,
and model-domain limitations.
