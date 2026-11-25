# 1s binding energies of the 2022 set

Written by `analyze.py --write` from the logs in `cases/`.

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
| B | P(CH3)3BH3 | PP-PBE | 198.041 | 198.04 | 191.605 | 191.60 | +0.005 |
| B | P(CH3)3BH3 | PP-PBE(B3LYP) | 198.235 |  | 191.799 | 191.77 | +0.029 |
| C | CH3SiH3 | PP-PBE | 296.719 | 296.71 | 288.161 | 288.16 | +0.001 |
| C | CH3SiH3 | PP-PBE(B3LYP) | 296.725 |  | 288.167 | 288.15 | +0.017 |
| N | (C2H5)2NH | PP-PBE | 411.946 | 411.94 | 401.863 | 401.86 | +0.003 |
| N | (C2H5)2NH | PP-PBE(B3LYP) | 411.726 |  | 401.643 | 401.64 | +0.003 |
| O | S(CH3)2O | PP-PBE | 544.925 | 544.93 | 534.457 | 534.46 | -0.003 |
| O | S(CH3)2O | PP-PBE(B3LYP) | 544.557 |  | 534.089 | 534.09 | -0.001 |

## B 1s: shifts relative to P(CH3)3BH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(CH3)3BH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H3BN(CH2CH3)3 | 0.27 | 0.241 | 0.24 | +0.001 | 0.301 | 0.29 | +0.011 |
| BH3CO | 2.17 | 2.173 | 2.17 | +0.003 | 2.208 | 2.21 | -0.002 |
| B(CH2CH3)3 | 2.57 | 2.184 | 2.18 | +0.004 | 2.334 | 2.35 | -0.016 |
| B(OC2H5)3 | 4.67 | 4.413 | 4.42 | -0.007 | 4.870 | 4.87 | -0.000 |
| B(OCH3)3 | 4.87 | 4.678 | 4.68 | -0.002 | 5.133 | 5.13 | +0.003 |
| BBr3 | 6.07 | 5.822 | 5.83 | -0.008 | 6.146 | 6.17 | -0.024 |
| BCl3 | 6.87 | 6.446 | 6.45 | -0.004 | 6.865 | 6.88 | -0.015 |
| BF3 | 9.87 | 9.349 | 9.35 | -0.001 | 9.999 | 10.01 | -0.011 |

## C 1s: shifts relative to CH3SiH3, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| CH3SiH3 | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| CH4 | 0.49 | 0.428 | 0.43 | -0.002 | 0.526 | 0.53 | -0.004 |
| CH3SH | 1.10 | 1.099 | 1.09 | +0.009 | 1.145 | 1.14 | +0.005 |
| CH3OH | 2.11 | 2.046 | 2.04 | +0.006 | 2.186 | 2.18 | +0.006 |
| H2NCN | 3.15 | 2.974 | 2.98 | -0.006 | 3.284 | 3.30 | -0.016 |
| HCN | 3.19 | 3.129 | 3.13 | -0.001 | 3.311 | 3.32 | -0.009 |
| NH2CHO | 4.14 | 3.889 | 3.89 | -0.001 | 4.312 | 4.30 | +0.012 |
| HNCO | 5.58 | 5.338 | 5.34 | -0.002 | 5.958 | 5.96 | -0.002 |
| CO | 5.89 | 6.002 | 6.01 | -0.008 | 6.352 | 6.36 | -0.008 |
| CH2F2 | 6.05 | 5.780 | 5.78 | +0.000 | 6.123 | 6.12 | +0.003 |
| CCl4 | 6.08 | 5.875 | 5.89 | -0.015 | 6.013 | 6.05 | -0.037 |
| CO2 | 7.38 | 7.091 | 7.10 | -0.009 | 7.821 | 7.82 | +0.001 |
| CHF3 | 8.79 | 8.436 | 8.44 | -0.004 | 8.962 | 8.96 | +0.002 |
| COF2 | 9.33 | 8.887 | 8.89 | -0.003 | 9.615 | 9.62 | -0.005 |
| CF4 | 11.54 | 10.995 | 11.00 | -0.005 | 11.712 | 11.72 | -0.008 |

## N 1s: shifts relative to (C2H5)2NH, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| (C2H5)2NH | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| H2N(CH2)3OCH3 | 0.30 | 0.341 | 0.35 | -0.009 | 0.275 | 0.28 | -0.005 |
| CH3CH2NH2 | 0.35 | 0.419 | 0.39 | +0.029 | 0.493 | 0.46 | +0.033 |
| (CH3)3CCN | 0.48 | 0.427 | 0.41 | +0.017 | 0.659 | 0.65 | +0.009 |
| CH3NH2 | 0.57 | 0.659 | 0.66 | -0.001 | 0.728 | 0.73 | -0.002 |
| H2N(CH2)3OH | 0.70 | 0.523 | 0.52 | +0.003 | 0.612 | 0.61 | +0.002 |
| NH3 | 1.02 | 1.085 | 1.08 | +0.005 | 1.235 | 1.22 | +0.015 |
| CH3CN | 1.02 | 0.883 | 0.89 | -0.007 | 1.081 | 1.09 | -0.009 |
| CH3SCN | 1.42 | 1.123 | 1.14 | -0.017 | 1.198 | 1.22 | -0.022 |
| ClCH2CN | 1.49 | 1.299 | 1.31 | -0.011 | 1.523 | 1.53 | -0.007 |
| NH2CHO | 1.72 | 1.998 | 2.00 | -0.002 | 2.109 | 2.11 | -0.001 |
| Cl3CCN | 1.86 | 1.500 | 1.49 | +0.010 | 1.819 | 1.82 | -0.001 |
| HNCO | 1.86 | 1.943 | 1.95 | -0.007 | 2.098 | 2.10 | -0.002 |
| ClCN | 1.87 | 1.511 | 1.52 | -0.009 | 1.743 | 1.76 | -0.017 |
| HCN | 2.22 | 2.276 | 2.28 | -0.004 | 2.425 | 2.44 | -0.015 |

Table S1 prints the values of Cl3CCN and HNCO in each other's rows; the "paper" columns above have them exchanged.

## O 1s: shifts relative to S(CH3)2O, eV

| molecule | experiment | PP-PBE here | paper | difference | PP-PBE(B3LYP) here | paper | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| S(CH3)2O | 0.00 | 0.000 | 0.00 | +0.000 | 0.000 | 0.00 | +0.000 |
| (H2N)2CO | 0.52 | 0.493 | 0.51 | -0.017 | 0.650 | 0.65 | -0.000 |
| NH2CHO | 1.07 | 0.997 | 0.99 | +0.007 | 1.096 | 1.08 | +0.016 |
| CH3CHO | 1.95 | 1.850 | 1.85 | +0.000 | 1.947 | 1.95 | -0.003 |
| (CH3)2O | 2.36 | 2.139 | 2.15 | -0.011 | 2.239 | 2.25 | -0.011 |
| CH3OH | 2.53 | 2.486 | 2.49 | -0.004 | 2.574 | 2.58 | -0.006 |
| CH2O | 2.77 | 2.868 | 2.87 | -0.002 | 2.893 | 2.90 | -0.007 |
| COCl2 | 3.05 | 2.947 | 2.95 | -0.003 | 3.137 | 3.14 | -0.003 |
| H2O | 3.25 | 3.263 | 3.26 | +0.003 | 3.463 | 3.46 | +0.003 |
| HNCO | 3.49 | 3.621 | 3.60 | +0.021 | 3.794 | 3.77 | +0.024 |
| H2CCO | 3.58 | 3.547 | 3.56 | -0.013 | 3.655 | 3.68 | -0.025 |
| OCS | 3.63 | 3.725 | 3.73 | -0.005 | 3.901 | 3.91 | -0.009 |
| COF2 | 4.10 | 3.905 | 3.90 | +0.005 | 4.177 | 4.17 | +0.007 |
| BH3CO | 5.38 | 5.001 | 5.00 | +0.001 | 5.358 | 5.35 | +0.008 |
| CO | 5.90 | 6.016 | 6.04 | -0.024 | 6.108 | 6.14 | -0.032 |

## Summary, eV

Mean absolute error (MAE) of the shifts against experiment over the molecules of a set, the reference molecule counted as in Table S1; differences from the paper over the other molecules.

| set | molecules | method | MAE here | MAE in the paper | mean abs. difference from the paper | largest |
|---|---:|---|---:|---:|---:|---:|
| B | 9 | PP-PBE | 0.229 | 0.23 | 0.004 | 0.008 |
| B | 9 | PP-PBE(B3LYP) | 0.109 | 0.11 | 0.010 | 0.024 |
| C | 15 | PP-PBE | 0.205 | 0.20 | 0.005 | 0.015 |
| C | 15 | PP-PBE(B3LYP) | 0.175 | 0.18 | 0.009 | 0.037 |
| N | 15 | PP-PBE | 0.150 | 0.15 | 0.009 | 0.029 |
| N | 15 | PP-PBE(B3LYP) | 0.142 | 0.14 | 0.010 | 0.033 |
| O | 15 | PP-PBE | 0.109 | 0.11 | 0.008 | 0.024 |
| O | 15 | PP-PBE(B3LYP) | 0.114 | 0.12 | 0.011 | 0.032 |
| all | 54 | PP-PBE | 0.167 | 0.16 | 0.007 | 0.029 |
| all | 54 | PP-PBE(B3LYP) | 0.138 | 0.14 | 0.010 | 0.037 |

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
