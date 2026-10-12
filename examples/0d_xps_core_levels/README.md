# Core-level binding energies of molecules: two published benchmark sets

The cases of two papers on the pseudopotential ΔSCF method for core-level
binding energies and its refinement step PP-PBE(B3LYP), run with the solver
of this repository:

- `jctc2022_1s`: Q. Xu, D. Prendergast, J. Qian, *Real-Space Pseudopotential
  Method for the Calculation of 1s Core-Level Binding Energies*, J. Chem.
  Theory Comput. **18**, 5471 (2022). B, C, N and O 1s of 54 molecules
  (Tables 1, 2, 3, S1 and S2).
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

From `jctc2022_1s` or `jctc2024_third_row`, with the Python environment of the
repository:

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

All 168 runs converged. Energies in eV; "PBE / refined" stands for PP-PBE /
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
