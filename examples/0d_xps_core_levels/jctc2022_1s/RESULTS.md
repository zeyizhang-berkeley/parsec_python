# 1s binding energies of the 2022 set

Written by `analyze.py --write` from `reference_energies.csv`.

## Atomic energies of the pseudopotential generation (Table 1, spin-unpolarized), eV

| element | all-electron atom | paper | all-electron ion | paper | pseudo-atom | paper | pseudo-ion | paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | -669.486 | -669.486 | -462.410 | -462.410 | -70.483 | -70.483 | -128.688 | -128.688 |
| C | -1027.632 | -1027.632 | -723.254 | -723.254 | -145.942 | -145.943 | -240.665 | -240.665 |
| N | -1481.742 | -1481.741 | -1061.468 | -1061.468 | -262.446 | -262.446 | -401.742 | -401.742 |
| O | -2040.902 | -2040.902 | -1486.074 | -1486.074 | -428.168 | -428.168 | -619.405 | -619.405 |

## Reference molecules: absolute binding energies, eV

The paper's Table S1 uses atomic energies of spin-polarized atoms. The constant per element that this takes off a binding energy follows from its Table 1 (B 6.436, C 8.558, N 10.083, O 10.468) and is applied in the fifth column.

| set | molecule | method | here | paper Table 2 | here, constant applied | paper Table S1 | difference |
|---|---|---|---:|---:|---:|---:|---:|
| B | P(CH3)3BH3 | PP-PBE | 198.037 | 198.04 | 191.601 | 191.60 | +0.001 |
| B | P(CH3)3BH3 | PP-PBE(B3LYP) | 198.231 |  | 191.795 | 191.77 | +0.025 |
| C | CH3SiH3 | PP-PBE | 296.713 | 296.71 | 288.155 | 288.16 | -0.005 |
| C | CH3SiH3 | PP-PBE(B3LYP) | 296.719 |  | 288.161 | 288.15 | +0.011 |
| N | (C2H5)2NH | PP-PBE | 411.942 | 411.94 | 401.859 | 401.86 | -0.001 |
| N | (C2H5)2NH | PP-PBE(B3LYP) | 411.722 |  | 401.639 | 401.64 | -0.001 |
| O | S(CH3)2O | PP-PBE | 544.928 | 544.93 | 534.460 | 534.46 | +0.000 |
| O | S(CH3)2O | PP-PBE(B3LYP) | 544.560 |  | 534.092 | 534.09 | +0.002 |

## B 1s: shifts relative to P(CH3)3BH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(CH3)3BH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H3BN(CH2CH3)3 | 0.27 | 0.242 | 0.24 | +0.002 | 0.302 | 0.29 | +0.012 |
| BH3CO | 2.17 | 2.174 | 2.17 | +0.004 | 2.209 | 2.21 | -0.001 |
| B(CH2CH3)3 | 2.57 | 2.186 | 2.18 | +0.006 | 2.336 | 2.35 | -0.014 |
| B(OC2H5)3 | 4.67 | 4.413 | 4.42 | -0.007 | 4.871 | 4.87 | +0.001 |
| B(OCH3)3 | 4.87 | 4.679 | 4.68 | -0.001 | 5.134 | 5.13 | +0.004 |
| BBr3 | 6.07 | 5.820 | 5.83 | -0.010 | 6.144 | 6.17 | -0.026 |
| BCl3 | 6.87 | 6.445 | 6.45 | -0.005 | 6.863 | 6.88 | -0.017 |
| BF3 | 9.87 | 9.349 | 9.35 | -0.001 | 9.999 | 10.01 | -0.011 |

## C 1s: shifts relative to CH3SiH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| CH3SiH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| CH4 | 0.49 | 0.431 | 0.43 | +0.001 | 0.530 | 0.53 | -0.000 |
| CH3SH | 1.10 | 1.099 | 1.09 | +0.009 | 1.146 | 1.14 | +0.006 |
| CH3OH | 2.11 | 2.045 | 2.04 | +0.005 | 2.186 | 2.18 | +0.006 |
| H2NCN | 3.15 | 2.974 | 2.98 | -0.006 | 3.284 | 3.30 | -0.016 |
| HCN | 3.19 | 3.130 | 3.13 | -0.000 | 3.311 | 3.32 | -0.009 |
| NH2CHO | 4.14 | 3.887 | 3.89 | -0.003 | 4.310 | 4.30 | +0.010 |
| HNCO | 5.58 | 5.337 | 5.34 | -0.003 | 5.956 | 5.96 | -0.004 |
| CO | 5.89 | 6.004 | 6.01 | -0.006 | 6.354 | 6.36 | -0.006 |
| CH2F2 | 6.05 | 5.781 | 5.78 | +0.001 | 6.123 | 6.12 | +0.003 |
| CCl4 | 6.08 | 5.871 | 5.89 | -0.019 | 6.009 | 6.05 | -0.041 |
| CO2 | 7.38 | 7.093 | 7.10 | -0.007 | 7.823 | 7.82 | +0.003 |
| CHF3 | 8.79 | 8.434 | 8.44 | -0.006 | 8.961 | 8.96 | +0.001 |
| COF2 | 9.33 | 8.885 | 8.89 | -0.005 | 9.614 | 9.62 | -0.006 |
| CF4 | 11.54 | 10.994 | 11.00 | -0.006 | 11.712 | 11.72 | -0.008 |

## N 1s: shifts relative to (C2H5)2NH, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| (C2H5)2NH | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H2N(CH2)3OCH3 | 0.30 | 0.343 | 0.35 | -0.007 | 0.276 | 0.28 | -0.004 |
| CH3CH2NH2 | 0.35 | 0.419 | 0.39 | +0.029 | 0.493 | 0.46 | +0.033 |
| (CH3)3CCN | 0.48 | 0.428 | 0.41 | +0.018 | 0.659 | 0.65 | +0.009 |
| CH3NH2 | 0.57 | 0.659 | 0.66 | -0.001 | 0.728 | 0.73 | -0.002 |
| H2N(CH2)3OH | 0.70 | 0.522 | 0.52 | +0.002 | 0.611 | 0.61 | +0.001 |
| NH3 | 1.02 | 1.087 | 1.08 | +0.007 | 1.237 | 1.22 | +0.017 |
| CH3CN | 1.02 | 0.883 | 0.89 | -0.007 | 1.082 | 1.09 | -0.008 |
| CH3SCN | 1.42 | 1.121 | 1.14 | -0.019 | 1.197 | 1.22 | -0.023 |
| ClCH2CN | 1.49 | 1.299 | 1.31 | -0.011 | 1.523 | 1.53 | -0.007 |
| NH2CHO | 1.72 | 1.999 | 2.00 | -0.001 | 2.110 | 2.11 | +0.000 |
| Cl3CCN | 1.86 | 1.500 | 1.49 | +0.010 | 1.819 | 1.82 | -0.001 |
| HNCO | 1.86 | 1.942 | 1.95 | -0.008 | 2.098 | 2.10 | -0.002 |
| ClCN | 1.87 | 1.512 | 1.52 | -0.008 | 1.744 | 1.76 | -0.016 |
| HCN | 2.22 | 2.277 | 2.28 | -0.003 | 2.427 | 2.44 | -0.013 |

Table S1 prints the values of Cl3CCN and HNCO in each other's rows; the "paper" columns above have them exchanged.

## O 1s: shifts relative to S(CH3)2O, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| S(CH3)2O | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| (H2N)2CO | 0.52 | 0.496 | 0.51 | -0.014 | 0.653 | 0.65 | +0.003 |
| NH2CHO | 1.07 | 0.991 | 0.99 | +0.001 | 1.090 | 1.08 | +0.010 |
| CH3CHO | 1.95 | 1.861 | 1.85 | +0.011 | 1.958 | 1.95 | +0.008 |
| (CH3)2O | 2.36 | 2.146 | 2.15 | -0.004 | 2.246 | 2.25 | -0.004 |
| CH3OH | 2.53 | 2.488 | 2.49 | -0.002 | 2.576 | 2.58 | -0.004 |
| CH2O | 2.77 | 2.872 | 2.87 | +0.002 | 2.897 | 2.90 | -0.003 |
| COCl2 | 3.05 | 2.953 | 2.95 | +0.003 | 3.143 | 3.14 | +0.003 |
| H2O | 3.25 | 3.275 | 3.26 | +0.015 | 3.475 | 3.46 | +0.015 |
| HNCO | 3.49 | 3.627 | 3.60 | +0.027 | 3.799 | 3.77 | +0.029 |
| H2CCO | 3.58 | 3.550 | 3.56 | -0.010 | 3.658 | 3.68 | -0.022 |
| OCS | 3.63 | 3.730 | 3.73 | +0.000 | 3.906 | 3.91 | -0.004 |
| COF2 | 4.10 | 3.911 | 3.90 | +0.011 | 4.183 | 4.17 | +0.013 |
| BH3CO | 5.38 | 5.011 | 5.00 | +0.011 | 5.368 | 5.35 | +0.018 |
| CO | 5.90 | 6.022 | 6.04 | -0.018 | 6.114 | 6.14 | -0.026 |

## Summary, eV

Mean absolute error (MAE) of the shifts against experiment over the molecules of a set, the reference molecule counted as in Table S1; differences from the paper over the other molecules.

| set | molecules | method | MAE here | MAE in the paper | mean abs. difference from the paper | largest |
|---|---:|---|---:|---:|---:|---:|
| B | 9 | PP-PBE | 0.229 | 0.23 | 0.005 | 0.010 |
| B | 9 | PP-PBE(B3LYP) | 0.109 | 0.11 | 0.011 | 0.026 |
| C | 15 | PP-PBE | 0.206 | 0.20 | 0.006 | 0.019 |
| C | 15 | PP-PBE(B3LYP) | 0.176 | 0.18 | 0.008 | 0.041 |
| N | 15 | PP-PBE | 0.151 | 0.15 | 0.009 | 0.029 |
| N | 15 | PP-PBE(B3LYP) | 0.142 | 0.14 | 0.010 | 0.033 |
| O | 15 | PP-PBE | 0.108 | 0.11 | 0.009 | 0.027 |
| O | 15 | PP-PBE(B3LYP) | 0.116 | 0.12 | 0.012 | 0.029 |
| all | 54 | PP-PBE | 0.167 | 0.16 | 0.007 | 0.029 |
| all | 54 | PP-PBE(B3LYP) | 0.139 | 0.14 | 0.010 | 0.041 |

## N 1s shifts refined with alpha times the exact exchange of B3LYP (Table S2), eV

From the parts of the B3LYP energy of each case. Errors against experiment, the reference molecule counted.

| alpha | MAE here | paper | mean error here | paper |
|---:|---:|---:|---:|---:|
| 0 | 0.155 | 0.15 | +0.020 | +0.02 |
| 1 | 0.142 | 0.14 | +0.075 | +0.08 |
| 2 | 0.164 | 0.16 | +0.130 | +0.13 |
| 3 | 0.208 | 0.21 | +0.185 | +0.19 |
| 4 | 0.252 | 0.26 | +0.240 | +0.25 |
| 5 | 0.302 | 0.31 | +0.295 | +0.31 |
