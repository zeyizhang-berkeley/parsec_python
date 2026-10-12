# Core-level binding energies of molecules: published benchmark sets

The cases of two papers on the pseudopotential ΔSCF method for core-level
binding energies and its refinement step PP-PBE(B3LYP), run with the solver
of this repository:

- `jctc2022_1s`: Q. Xu, D. Prendergast, J. Qian, *Real-Space Pseudopotential
  Method for the Calculation of 1s Core-Level Binding Energies*, J. Chem.
  Theory Comput. **18**, 5471 (2022). B, C, N and O 1s of 54 molecules
  (Tables 1, 2, 3, S1 and S2).
- `jctc2022_protonated_water`: Figure 5(a) of the same paper. O 1s of the
  hydronium oxygen in the clusters H3O+(H2O)n, n = 1 to 20, relative to the
  water molecule.
- `jctc2024_third_row`: L. Liu, Q. Xu, L. dos Anjos Cunha, H. Xin,
  M. Head-Gordon, J. Qian, *Real-Space Pseudopotential Method for the
  Calculation of Third-Row Elements X-ray Photoelectron Spectroscopic
  Signatures*, J. Chem. Theory Comput. **20**, 6134 (2024). S and P 1s, 2s and
  2p of 21 molecules (Tables S1, S4 and S5, structures optimized with PBE).

The structures and the FHI98PP inputs of the pseudopotentials are those of the
Supporting Information of the papers; the values of the papers' tables are the
reference.

## Layout

```text
0d_xps_core_levels/
  xps_cases.py              shared steps: inputs, queue, log reader
  pseudopotentials/         the potentials of both sets and species.json
  jctc2022_1s/
    paper/structures/<B|C|N|O>/<nn>.xyz   molecules of Table S1, in its order
    paper/fhi98pp_inputs/*.ini            inputs of the potentials
    paper/table_s1.csv                    Table S1
    cases/<set>/<nn>/{IS,FS}/parsec.in    neutral molecule and 1s core hole
    reference_energies.csv                totals of every case
    setup.py  run.py  analyze.py
    RESULTS.md
  jctc2022_protonated_water/
    paper/structures/<nn>.xyz             H3O+(H2O)n, global minimum of the published file
    paper/figure_5a.csv                   the points of Figure 5(a), read from its image
    cases/<nn>/{IS,FS}/parsec.in          cluster (charge +1) and 1s hole on the hydronium oxygen
    reference_energies.csv
    setup.py  run.py  analyze.py
    RESULTS.md
  jctc2024_third_row/
    paper/structures/<formula>.xyz
    paper/fhi98pp_inputs/*.ini
    paper/tables.csv                      Tables S1, S4, S5
    cases/<formula>/{IS,FS_<S|P>-<1s|2s|2p>}/parsec.in
    reference_energies.csv
    setup.py  run.py  analyze.py
    RESULTS.md
```

`paper/` holds copies of files of the Supporting Information (on figshare
under CC BY-NC 4.0:
[10.1021/acs.jctc.2c00474.s002](https://doi.org/10.1021/acs.jctc.2c00474.s002),
[10.1021/acs.jctc.4c00535.s002](https://doi.org/10.1021/acs.jctc.4c00535.s002))
and the tables of its PDF files typed into CSV.

`pseudopotentials/` holds the potentials generated from those inputs
(Troullier–Martins, PBE, no core correction; FHI98PP, M. Fuchs and
M. Scheffler, Comput. Phys. Commun. **119**, 67 (1999)) in the format the
solver reads. `species.json` gives for each the local component, the number
of valence electrons and the all-electron and pseudo energies of the
generating atom, from which the `Atomic_Energy_Correction` of the inputs
follows. The local component is the highest channel of the input: p for B, C,
N, O and F, d for Si, P, S, Cl and Br, s for H. A potential with a core hole
(`C-1s`, `S-2p`, ...) is generated with one electron removed from that shell.

## Running

From `jctc2022_1s`, `jctc2022_protonated_water` or `jctc2024_third_row`, with
the Python environment of the repository:

```powershell
python run.py              # all cases; --sets C, --only C/02 C/15, ... select some
python analyze.py          # tables against the paper
python analyze.py --check  # exit code 1 if a run differs from the reference
```

`run.py` starts one ordinary run of the solver per case, with the Poisson
solves on the GPU (`--cpu-poisson` for the CPU), and skips cases whose log
`parsec.out` is finished. A case has `Energy_Refinement: b3lyp`, and its log
ends with the two totals that are read:

```text
PBE all-electron total (as above)           :         -80.96260980 [Ry]
B3LYP-refined all-electron total            :         -81.00444122 [Ry]
```

A binding energy is the difference of these totals between the final state
(`FS`, the atom with the hole carries the core-hole potential and the molecule
the charge +1) and the initial state (`IS`). `reference_energies.csv` holds
the totals, the parts of the B3LYP energy and the times of every case of the
runs reported below; without logs `analyze.py` works from it. `--check`
compares new logs with it (0.003 eV) and with the paper (tolerances in the
script).

`setup.py` rewrites the inputs, for example with another `--spacing`.

The protonated-water clusters were run on GPUs with 80 GB; on the 8 GB GPU of
the tested laptop only the two smallest were run. `run.py --largest 3e6`
selects the cases up to that many grid points.

## Settings

| | papers (ARES) | here |
|---|---|---|
| domain | sphere, Dirichlet boundary, multipole boundary values to l = 9 | sphere 5.0 Å beyond the outermost nucleus, same boundary treatment |
| grid | 0.07 to 0.12 Å (2022 runs kept in `../0_CH4_CF4`), 0.10 Å (2024 input) | 0.10 Å, finite differences of order 8 |
| spin | spin-polarized runs that end unpolarized | unpolarized |
| atomic reference energies | from the potential generation | the same (spin-unpolarized atoms) |
| refinement | `Lrefine=T` | `Energy_Refinement: b3lyp` |

With 0.10 Å the binding energy of CF4 is within 0.002 eV of a run with
0.07 Å (`../0_CH4_CF4/python_pbe_b3lyp`). The totals of CH4 change by 0.0003 eV
at most when the wall is moved from 4 Å to 8 Å beyond the outermost
nucleus. `States_Num` is the number of occupied states plus at least 8,
rounded up to a multiple of 16.

## Results

All 208 runs converged. Energies in eV; "PBE / refined" stands for PP-PBE /
PP-PBE(B3LYP). The complete tables are `RESULTS.md` in each folder.

2022 set:

| quantity | paper | here |
|---|---|---|
| Table 1, atomic energies of the generation, 16 spin-unpolarized values | | within 0.001 |
| Table 2, PP-PBE of the four reference molecules | 198.04, 296.71, 411.94, 544.93 | 198.037, 296.713, 411.942, 544.928 |
| Table S1, the same with its spin-polarized atomic energies, PBE | 191.60, 288.16, 401.86, 534.46 | 191.601, 288.155, 401.859, 534.460 |
| Table S1, the same, refined | 191.77, 288.15, 401.64, 534.09 | 191.795, 288.161, 401.639, 534.092 |
| Table S1, shifts of the other 50 molecules, PBE / refined | | mean absolute difference 0.007 / 0.010, largest 0.029 / 0.041 |
| Table 3, error against experiment, PBE (B, C, N, O, all) | 0.23, 0.20, 0.15, 0.11, 0.16 | 0.229, 0.206, 0.151, 0.108, 0.167 |
| Table 3, the same, refined | 0.11, 0.18, 0.14, 0.12, 0.14 | 0.109, 0.176, 0.142, 0.116, 0.139 |
| Table S2, N shifts with 0 to 5 times the exact exchange | 0.15, 0.14, 0.16, 0.21, 0.26, 0.31 | 0.155, 0.142, 0.164, 0.208, 0.252, 0.302 |

- The absolute values of Table S1 use atomic energies of spin-polarized atoms.
  The constant per element that this takes off a binding energy follows from
  Table 1 of the paper (B 6.436, C 8.558, N 10.083, O 10.468) and is applied
  for the third and fourth row; shifts do not depend on it.
- Table S1 prints the values of Cl3CCN and HNCO (N set) in each other's rows:
  here Cl3CCN 1.500 / 1.819 and HNCO 1.942 / 2.098, in the table 1.95 / 2.10
  and 1.49 / 1.82. They are compared exchanged.
- The errors of Table 3 count the reference molecule of each set.
- Table S2 is evaluated from the parts of the B3LYP energy that every case
  lists; it needs no further runs.

Protonated water (Figure 5(a) of the 2022 paper), binding energy of the 1s
level of the hydronium oxygen relative to the water molecule of the 2022 set:

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PBE, here | 9.931 | 7.891 | 6.189 | 5.631 | 5.548 | 5.027 | 5.281 | 4.805 | 4.568 | 4.213 |
| PBE, figure | 9.94 | 7.86 | 6.17 | 5.59 | 5.50 | 4.98 | 5.29 | 4.80 | 4.56 | 4.18 |
| refined, here | 9.984 | 7.976 | 6.297 | 5.744 | 5.665 | 5.144 | 5.417 | 4.937 | 4.697 | 4.344 |
| refined, figure | 9.99 | 7.97 | 6.31 | 5.71 | 5.62 | 5.10 | 5.42 | 4.93 | 4.69 | 4.32 |

| n | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PBE, here | 3.932 | 3.694 | 3.716 | 3.517 | 3.373 | 3.517 | 3.437 | 3.345 | 3.234 | 3.170 |
| PBE, figure | 3.92 | 3.62 | 3.68 | 3.49 | 3.35 | 3.51 | 3.40 | 3.32 | 3.18 | 3.15 |
| refined, here | 4.071 | 3.823 | 3.856 | 3.660 | 3.513 | 3.658 | 3.581 | 3.486 | 3.378 | 3.315 |
| refined, figure | 4.05 | 3.75 | 3.82 | 3.64 | 3.49 | 3.66 | 3.56 | 3.45 | 3.32 | 3.30 |

- The paper gives the figure only. Its points were read from the image
  (0.013 eV per pixel; the squares of the PBE curve lie partly under the
  circles of the refined one), which leaves an uncertainty of about 0.03.
  Mean absolute difference 0.028 / 0.025, largest 0.074 / 0.073 (n = 12).
- The structures are the first entry of each published file, the global
  minimum. The hole is on the oxygen with three hydrogens bound (Figure S2 of
  the paper); the initial state has the charge +1, the final state +2.

2024 set:

| quantity | paper | here |
|---|---|---|
| Table S1, S 1s of H2S and P 1s of P(CH3)3, PBE | 2492.11, 2162.71 | 2492.118, 2162.706 |
| Table S1, the same, refined | 2492.11, 2162.77 | 2492.130, 2162.780 |
| Table S4, S 2s of S8 and P 2s of P4, PBE | 224.3, 185.72 | 224.363, 185.719 |
| Table S4, the same, refined | 224.44, 185.71 | 224.430, 185.702 |
| Tables S1, S4, S5, shifts of 33 further entries, PBE / refined | | mean absolute difference 0.018 / 0.022, largest 0.055 / 0.055 |

- A potential with a 2p hole is spherical, so a 2p binding energy here is the
  average over the multiplet; the paper lists 2p3/2 values. Only shifts are
  compared for 2p.
- Two entries of the paper's tables are not in this set: S 2s of PSF3 and
  P 2p of PF3, which differ from the tables by 0.10 and 0.17.

## Time and memory

First calculations (no cache, no restart) in double precision on a laptop with
32 logical cores, an RTX 5070 Laptop GPU with 8 GB and 32 GB of memory:

| set | runs | all runs | median run | shortest | longest |
|---|---:|---:|---:|---:|---:|
| 2022, 54 molecules | 108 | 5.2 h | 95 s | 26 s (CO) | 1410 s (B(OC2H5)3) |
| 2024, 21 molecules | 60 | 2.8 h | 142 s | 41 s (H2S) | 804 s (P(CF3)3) |

| run | grid points | occupied / computed states | SCF steps | SCF [s] | pair solves | refinement [s] | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CO, neutral | 723,160 | 5 / 16 | 24 | 19 | 15 | 5 | 26 | 2.1 |
| CH3CN, neutral | 1,246,488 | 8 / 16 | 44 | 73 | 36 | 18 | 96 | 2.9 |
| P(CF3)3, neutral | 1,854,640 | 40 / 48 | 35 | 93 | 820 | 683 | 781 | 5.5 |
| B(OC2H5)3, neutral | 3,246,592 | 30 / 48 | 58 | 549 | 465 | 849 | 1410 | 11.3 |

The refinement is one Poisson solve per pair of occupied orbitals, 0.3 s on
0.7 million points and 1.8 s on 3.2 million, and takes about half of the total
time. The GPU held 7.7 of its 8.0 GiB during B(OC2H5)3.

The protonated-water clusters ran on A100 GPUs with 80 GB (Perlmutter, one run
per GPU, solver defaults): 40 runs, 13.9 GPU hours, 56 minutes on 16 GPUs.
The two smallest clusters were also run on the laptop; their totals are the
same in all eight printed decimals.

| run | grid points | occupied / computed states | SCF steps | SCF [s] | pair solves | refinement [s] | whole run [s] | host memory [GiB] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H3O+(H2O)1 | 1,360,280 | 8 / 16 | 38 | 39 | 36 | 10 | 56 | 3.3 |
| H3O+(H2O)5 | 2,278,264 | 24 / 32 | 44 | 81 | 300 | 121 | 211 | 5.6 |
| H3O+(H2O)10 | 3,260,032 | 44 / 64 | 38 | 93 | 990 | 655 | 764 | 10.2 |
| H3O+(H2O)15 | 3,939,528 | 64 / 80 | 44 | 164 | 2080 | 1764 | 1953 | 14.4 |
| H3O+(H2O)20 | 4,140,056 | 84 / 96 | 39 | 158 | 3570 | 3180 | 3369 | 16.0 |

There the refinement is 89 % of the time: 0.26 s per pair solve on 1.4 million
points, 0.88 s on 4.1 million.
