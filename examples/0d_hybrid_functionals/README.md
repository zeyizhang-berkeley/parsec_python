# Self-consistent hybrid functionals: seven molecules against Quantum ESPRESSO

PBE0, HSE06, HSE03, B3LYP, B3LYP5, B3PW91 and Hartree-Fock, self-consistent (generalized Kohn-Sham with adaptively compressed exchange, ACE), for H2, H2O, CH4, CO, N2, benzene and triplet O2 (spin-polarized), compared with Quantum ESPRESSO 7.5 (+ libxc 7.0.0) using the same FHI98PP pseudopotentials and geometries. The quantity compared is E_hyb - E_PBE of each code at its own discretization, where both codes' PBE-level conventions cancel to first order; Fock energies and HOMOs are compared directly.

## Contents

- `<molecule>/<functional>/parsec.in` for `pbe`, `pbe0`, `hse06`, `hse03`, `b3lyp`, `b3lyp5`, `b3pw91`, `hf` (O2: with `occup.in`, 7 up and 5 down electrons), and `parsec_python.out`, the reference log of that input (one NVIDIA A100; paths shortened to repository paths).
- `H_FHIPP.DAT`, `C_FHIPP.DAT`, `N_FHIPP.DAT`, `O_FHIPP.DAT`: copies of `src/pp_generation/NCPPs/PBE/<E>/FHI/` (Troullier-Martins, FHI98PP, PBE, no core correction). The inputs find them with `--pp-dir` (one folder for all elements; the library keeps one folder per element).
- `qe/`: the Quantum ESPRESSO inputs, their settings, the UPF note and the QE numbers (`qe/README.md`).
- `libxc/run_libxc.py`: the same molecules with libxc's functionals (section "libxc's functionals").

## Run one case

From the repository root:

```powershell
python src\parsec_python\main.py examples\0d_hybrid_functionals\H2O\pbe0\parsec.in `
  --pp-dir examples\0d_hybrid_functionals --no-archive
```

The hybrid runs on the GPU with CuPy (`--device gpu`, the default when one is found); without CuPy the accelerated CLI routes the input to the reference solver, which is much slower at this grid. `parsec.out` is written beside the input; compare it with `parsec_python.out`.

Settings of every input: grid spacing 0.16 bohr, sphere = largest atom distance from the centre + 10 bohr, finite-difference order 12, `Convergence_Criterion 1e-6 Ry`, `Cubic_Spline: .true.` for every species, `Exx_Tolerance 1e-8 Ry` (default 1e-6 Ry; tight for the comparison: the default changes the energies by < 1e-6 Ry). Everything else is the default: PBE start of the Fock loop, ACE of the `States_Num` orbitals, Kronecker-product pair solver (the pair densities of the symmetry-sector orbitals in reflection-parity classes), `Exx_Inner_Criterion adaptive`, `Exx_Outer_Mixing diis`, HSE06 with w = 0.11 1/bohr; O2 keeps both spin channels in the symmetry sectors (`Spin_Symmetry_Sectors auto`).

With the adaptive inner criterion the PBE start of a spin-unpolarized hybrid stops at a residual of 1e-4 Ry. The energy its log prints for the start (the last `Total Energy` before `Fock loop`, with the note that it is the energy of a start) is therefore not a converged PBE energy: here it lies 1.6e-07 to 1.4e-05 Ry above the energy of the `pbe` input (benzene the most). E_PBE in the tables below is always the energy of the `pbe` input. The O2 start (spin-polarized) runs to `Convergence_Criterion`: its printed energy equals E_PBE to the printed digits.

The logs are of 24 November 2026, made at the defaults of that day. Against the logs of 23 November 2026 they differ by the Fourier filter of the tables, a default since 23 November 2026 (`PARSEC_LOCAL_FILTER=all`, section "PBE level": the totals move by -3.4e-5 (H2) to +2.0e-4 Ry (CO), E_hyb - E_PBE by at most 7.7e-6 Ry (N2 HF)), by the exact Coulomb tail of the spline of the local tables (`PARSEC_SPLINE_COULOMB_TAIL`, since 23 November 2026; at most 2.1e-6 Ry, benzene) and by the weighted sectors of the hybrids (below); the Kleinman-Bylander energies are integrated exactly and the FHI tables continued below their first sample since 23 November 2026 (section "Known offsets"). `PARSEC_FORTRAN_NUMERICS=on` gives the totals of the logs of 20 November 2026, before all of these (section "Known offsets"). H2, N2, CO and CH4 have the point groups D4h, C4v and Td on the grid, and since 15 November (Td) and 18 November 2026 (D4h, C4v; `PARSEC_ORBITAL_GROUP_MAX_COST` 0.75) a semilocal run solves the weighted one-dimensional sectors of a larger subgroup instead of the abelian group (`src/parsec_python/acceleration/README.md`, "Weighted sectors of a larger subgroup"). Since 19 November 2026 (`PARSEC_ORBITAL_GROUP_SPIN` and `PARSEC_ORBITAL_GROUP_OCCUPATIONS` auto) the spin-polarized O2 `pbe` input with its `occup.in` takes them too (D4h, the six sectors of a subgroup of 8): it needs 25 SCF iterations against 29 on the abelian route (38 in the logs of 20 November 2026), with the same total energy and HOMOs within 0.006 meV. The total energies of the weighted route are those of the abelian route to the printed digits, so E_PBE and every dE below do not depend on it; the HOMOs of CO and CH4 differ by 0.009 and 0.008 meV. The PBE runs of H2, CH4, CO and N2 took 5 to 6 s (Timings); with `PARSEC_ORBITAL_GROUP=abelian` H2 needs 20 SCF iterations instead of 30, CO 32 instead of 35, CH4 36 instead of 37 and N2 25 instead of 24 (A100, 2026-11-24). The difference is not a defect of the weighted route: its mixer and norms take the orbit multiplicities (the algebra of the full grid), both routes agree at the first step (total energy within 1e-8 Ry, every term within 1e-6 Ry) and part at the second, where one Chebyshev filter pass per sector, from sector blocks of other sizes, leaves densities that differ by about 1e-5 (Hartree energy 3.4e-5 Ry); the PBE potential carries that difference on (LDA: 24 steps on both routes), and near the criterion of 1e-6 Ry the charge-weighted potential error of the weighted run drifts at 1-3e-6 Ry for several steps. Small changes move the count by up to ten steps either way (A100, 2026-11-19, weighted against abelian: `States_Num` 8 20 and 20 steps, 12 21 and 20; `Grid_Spacing` 0.17 24 and 25; `Diag_Tolerance` 1e-8 25 and 19; `Convergence_Criterion` 1e-7 31 and 22; and on 2026-11-24 CH4 37 steps with the even first counts of the sectors, 31 with `PARSEC_WEIGHTED_FIRST_SHARE=orbitals`), always with the same energies.

The hybrids of these clusters take the weighted sectors as well since 23 November 2026 (`PARSEC_ORBITAL_GROUP_HYBRID=auto`; `src/parsec_python/acceleration/README.md`, "Hybrid functionals"), and the logs of the 35 hybrid inputs of CH4, CO, N2, H2 and O2 are made on that route, their PBE start included (O2: 25 SCF iterations): the line `symmetry_detection` names the point group and the subgroup whose sectors the orbitals use (the abelian subgroup in the logs before 24 November 2026), and the Fock loop solves 0.59 (Td) and 0.80 to 0.85 (D4h, C4v) of the pair densities of the abelian route (CH4 PBE0 67 against 114, O2 PBE0 359 against 447). Benzene and H2O have no larger group on the grid and keep their route. `PARSEC_ORBITAL_GROUP_HYBRID=off` runs the abelian sectors of the earlier logs; on the same node, right after the logs (A100, 24 November 2026, warm kernel caches), it gives the total energies of the logs to the printed digits (CO HF within 3e-8 Ry at its `Exx_Tolerance` of 1e-8 Ry), the Fock energies within 6.1e-6 Ry (CO HF; first order in the orbitals, within the reproducibility given below) and the HOMOs within 0.04 meV, and takes 1 to 59 % more seconds for 34 of the 35 inputs (CO HF 16.7 against 10.5 s, CH4 HF 20.0 against 12.7 s; O2 HSE06 3 % fewer). With empty kernel caches (first runs on a node, a fresh CuPy and CUDA cache for each run, four side by side) PBE0 takes 3.4 to 7.9 s more than with warm caches: CH4 13.6 s against 20.1 s on the abelian route, CO 15.4 against 21.6, N2 14.7 against 18.2, H2 14.1 against 16.8 and O2 25.5 against 33.0 s; H2O 22.8 s, benzene 39.9 s and the CH4 `pbe` input 10.5 s (A100, 24 November 2026). An unpolarized hybrid whose PBE start or an inner loop ends with a partly filled degenerate level starts again in the abelian sectors, with a NOTE; the closed shells of this folder and the O2 inputs with their `occup.in` do not (no such NOTE in the logs).

## PBE level

| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | difference (Ry) | trapezoid E_l: QE / this code (Ry) | HOMO this code (eV) | HOMO QE (eV) |
|---|---|---|---|---|---|---|
| H2 | -2.32974875 | -2.32974716 | -1.59e-06 | 0 / 0 | -10.3765 | -10.3762 |
| H2O | -34.34610161 | -34.34601311 | -8.85e-05 | +1.75e-4 / +1.69e-4 | -7.2391 | -7.2384 |
| CH4 | -16.15717622 | -16.15717314 | -3.08e-06 | -1.8e-5 / -1.91e-5 | -9.4649 | -9.4647 |
| CO | -43.31658892 | -43.31651838 | -7.05e-05 | +8.7e-5 / +7.82e-5 | -9.0619 | -9.0617 |
| N2 | -39.83021701 | -39.83021536 | -1.65e-06 | -4.9e-5 / -4.99e-5 | -10.2963 | -10.2961 |
| C6H6 | -75.47021559 | -75.47016414 | -5.15e-05 | - / -1.52e-4 | -6.3306 | -6.3306 |
| O2 (triplet) | -63.71099604 | -63.71080847 | -1.88e-04 | +2.73e-4 / +2.63e-4 | -6.8743 / -11.6060 | -6.8736 / -11.6064 |

QE: 150 Ry, benzene 120 Ry (QE's H2O total at 150 Ry is 6e-5 Ry above its 250 Ry value; benzene's at 120 Ry is 6e-5 Ry above its 160 Ry value). Trapezoid E_l: the change of the PBE energy when the Kleinman-Bylander energies E_l are integrated by PARSEC's trapezoid on the logarithmic mesh instead of exactly, in QE (UPF files with those E_l) and in this code (`PARSEC_KB_QUADRATURE=trapezoid`, the quadrature of the logs before 23 November 2026; section "Known offsets"). The two agree within 1e-5 Ry (2 to 10 %), so the default quadrature of the logs removes that part of the difference to QE: with the trapezoid the differences would be -1.59e-06 (H2), +8.07e-05 (H2O), -2.21e-05 (CH4), +7.61e-06 (CO), -5.15e-05 (N2), -2.03e-04 (C6H6) and +7.57e-05 Ry (O2). Most of what remains is QE's cutoff error for oxygen at 150 Ry. QE's converged totals (H2O -34.346079, O2 -63.710930 Ry at 600 Ry, with the change from a 20 to a 28 bohr box at 300 Ry added; 400 to 600 Ry change them by less than 1e-7 Ry, the box by at most 4.3e-6 Ry; CO -43.316584, N2 -39.830244 and CH4 -16.157192 Ry at 400 to 500 Ry, with the change to a 24 bohr box) lie 6.6e-5 (H2O, CO), 1.2e-4 (O2), 2.8e-5 (N2) and 1.9e-5 Ry (CH4) below those at 150 Ry, and the PBE totals of the inputs lie within 6.6e-5 Ry of them at every h from 0.20 to 0.08 bohr (A100, 24 November 2026; at h 0.20, 0.16, 0.14, 0.12, 0.10 and 0.08: H2O +2.7e-5, -2.3e-5, +3.4e-6, -4.7e-6, -5.2e-6, -5.6e-6; O2 +3.1e-5, -6.6e-5, -1.5e-5, -2.6e-5, -2.4e-5, -2.4e-5; CO -7.4e-6, -4.9e-6, -3.9e-6, -5.3e-6, -1.2e-5, -1.1e-5; N2 -5.6e-5, +2.6e-5, +1.9e-5, +2.3e-5, +2.2e-5, +2.0e-5; CH4 +1.3e-5, +1.5e-5, +1.5e-5, +1.6e-5, +1.6e-5, +1.6e-5 Ry). That rests on the Fourier filter of the tables, a default since 23 November 2026 (`PARSEC_LOCAL_FILTER=all`; `src/parsec_python/acceleration/README.md`, row `PARSEC_LOCAL_FILTER`): a table sampled at the grid points holds Fourier components beyond the resolution of the grid, which make the total depend on where the atoms sit relative to the grid points, oxygen's above all. With the tables of the files (`PARSEC_LOCAL_FILTER=off`, the logs before 24 November 2026) an O atom on a grid point lay about 1.7e-3 Ry higher than one between grid points at h 0.16 (7e-4 Ry at h 0.12); the inputs, with the O atoms midway between grid lines in x and y, lay on the low side (O2 -2.6e-4 and H2O -2.0e-4 Ry against QE's converged totals at h 0.16, still -1.0e-4 and -5.7e-5 Ry at h 0.08), and only averages over eight placements half a grid step apart came within about 3.5e-5 Ry of QE for h <= 0.14. With the filter a rigid translation over one grid step changes the totals by at most 3.2e-5 Ry at h 0.16 (O2) and 7.4e-6 Ry at h 0.12 (N2) (the same row). The trapezoid's change does not depend on h (to 1e-7 Ry from h 0.20 to 0.08, measured with the tables of the files).

## Hybrids against QE

dE = E_hyb - E_PBE (Ry), each code at its own discretization (this code: the `pbe` input); ddE = this code - QE; Fock = a_x E_x^HF of the final orbitals (Ry); HOMO in eV (O2: up / down channel). QE rows: internal `pbe0`, internal `hse` with `screening_parameter = 0.11`, libxc 402 / 475 / 401 for B3LYP / B3LYP5 / B3PW91, `hf`; HSE03 see below.

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | HOMO this code | HOMO QE |
|---|---|---|---|---|---|---|---|---|
| pbe0 | H2 | -0.00613212 | -0.00613599 | +3.9e-06 | -0.32884518 | -0.32884726 | -12.0031 | -12.0031 |
| pbe0 | H2O | +0.00543638 | +0.00540135 | +3.5e-05 | -1.96909017 | -1.96908409 | -9.0476 | -9.0473 |
| pbe0 | CH4 | -0.01578016 | -0.01579183 | +1.2e-05 | -1.47996488 | -1.47996664 | -11.0059 | -11.0058 |
| pbe0 | CO | +0.02214185 | +0.02213003 | +1.2e-05 | -2.33983329 | -2.33983251 | -10.7435 | -10.7436 |
| pbe0 | N2 | +0.02829892 | +0.02828276 | +1.6e-05 | -2.24328669 | -2.24328729 | -12.1982 | -12.1982 |
| pbe0 | C6H6 | +0.00193371 | +0.00190234 | +3.1e-05 | -5.73059512 | -5.73059923 | -7.2996 | -7.2996 |
| pbe0 | O2 (triplet) | +0.04027942 | +0.04022361 | +5.6e-05 | -3.16307261 | -3.16306389 | -9.0168 / -12.9364 | -9.0165 / -12.9364 |
| hse06 | H2 | -0.00510537 | -0.00510630 | +9.3e-07 | -0.26809256 | -0.26809615 | -11.5837 | -11.5836 |
| hse06 | H2O | +0.00763114 | +0.00760713 | +2.4e-05 | -1.72404478 | -1.72404677 | -8.6327 | -8.6324 |
| hse06 | CH4 | -0.01247333 | -0.01247594 | +2.6e-06 | -1.23633057 | -1.23633660 | -10.5919 | -10.5918 |
| hse06 | CO | +0.02494657 | +0.02494017 | +6.4e-06 | -2.03395681 | -2.03395215 | -10.3277 | -10.3276 |
| hse06 | N2 | +0.03111904 | +0.03111482 | +4.2e-06 | -1.93747323 | -1.93747648 | -11.7847 | -11.7846 |
| hse06 | C6H6 | +0.01177538 | +0.01174862 | +2.7e-05 | -4.81796927 | -4.81797872 | -6.9029 | -6.9027 |
| hse06 | O2 (triplet) | +0.04314418 | +0.04310624 | +3.8e-05 | -2.79499072 | -2.79497840 | -8.6018 / -12.5202 | -8.6017 / -12.5203 |
| hse03 | H2 | -0.04563310 | -0.04563432 | +1.2e-06 | -0.27069373 | -0.27069810 | -11.8641 | -11.8640 |
| hse03 | H2O | -0.16327523 | -0.16329488 | +2.0e-05 | -1.73369171 | -1.73369653 | -8.9100 | -8.9096 |
| hse03 | CH4 | -0.17677844 | -0.17679047 | +1.2e-05 | -1.24598234 | -1.24599087 | -10.8678 | -10.8676 |
| hse03 | CO | -0.18687972 | -0.18688571 | +6.0e-06 | -2.04587647 | -2.04587594 | -10.6066 | -10.6065 |
| hse03 | N2 | -0.18041735 | -0.18042079 | +3.4e-06 | -1.94921626 | -1.94922256 | -12.0683 | -12.0681 |
| hse03 | C6H6 | -0.60497967 | -0.60503878 | +5.9e-05 | -4.85274024 | -4.85275277 | -7.1613 | -7.1611 |
| hse03 | O2 (triplet) | -0.21599460 | -0.21604556 | +5.1e-05 | -2.80906213 | -2.80905330 | -8.8929 / -12.8024 | -8.8927 / -12.8024 |
| b3lyp | H2 | -0.02680913 | -0.02681196 | +2.8e-06 | -0.26262478 | -0.26263902 | -11.8165 | -11.8156 |
| b3lyp | H2O | -0.05079237 | -0.05074475 | -4.8e-05 | -1.57306909 | -1.57308427 | -8.8204 | -8.8190 |
| b3lyp | CH4 | -0.04323335 | -0.04323551 | +2.2e-06 | -1.18350703 | -1.18352905 | -10.7775 | -10.7768 |
| b3lyp | CO | -0.03601068 | -0.03598029 | -3.0e-05 | -1.87085772 | -1.87086635 | -10.5665 | -10.5657 |
| b3lyp | N2 | -0.01996664 | -0.01992990 | -3.7e-05 | -1.79306493 | -1.79307506 | -12.0173 | -12.0165 |
| b3lyp | C6H6 | -0.00960584 | -0.00963286 | +2.7e-05 | -4.58671014 | -4.58674653 | -7.0361 | -7.0354 |
| b3lyp | O2 (triplet) | -0.04263702 | -0.04254443 | -9.3e-05 | -2.52868875 | -2.52872814 | -8.7554 / -12.7450 | -8.7531 / -12.7432 |
| b3lyp5 | H2 | -0.01329495 | -0.01329778 | +2.8e-06 | -0.26253673 | -0.26255086 | -11.7249 | -11.7240 |
| b3lyp5 | H2O | +0.00680306 | +0.00685061 | -4.8e-05 | -1.57283141 | -1.57284672 | -8.7285 | -8.7270 |
| b3lyp5 | CH4 | +0.01252744 | +0.01252528 | +2.2e-06 | -1.18329663 | -1.18331869 | -10.6856 | -10.6848 |
| b3lyp5 | CO | +0.03569325 | +0.03572351 | -3.0e-05 | -1.87060699 | -1.87061573 | -10.4744 | -10.4735 |
| b3lyp5 | N2 | +0.05162715 | +0.05166384 | -3.7e-05 | -1.79283706 | -1.79284712 | -11.9241 | -11.9233 |
| b3lyp5 | C6H6 | +0.20188876 | +0.20186172 | +2.7e-05 | -4.58619582 | -4.58623198 | -6.9457 | -6.9449 |
| b3lyp5 | O2 (triplet) | +0.04403703 | +0.04412960 | -9.3e-05 | -2.52848429 | -2.52852392 | -8.6693 / -12.6425 | -8.6670 / -12.6407 |
| b3pw91 | H2 | -0.02627804 | -0.02628176 | +3.7e-06 | -0.26359148 | -0.26360699 | -11.7625 | -11.7617 |
| b3pw91 | H2O | -0.06939688 | -0.06941150 | +1.5e-05 | -1.57583588 | -1.57585112 | -8.7475 | -8.7463 |
| b3pw91 | CH4 | -0.07561963 | -0.07563168 | +1.2e-05 | -1.18510393 | -1.18512474 | -10.7573 | -10.7565 |
| b3pw91 | CO | -0.06880292 | -0.06879803 | -4.9e-06 | -1.87284891 | -1.87286506 | -10.4836 | -10.4829 |
| b3pw91 | N2 | -0.06051587 | -0.06051616 | +2.9e-07 | -1.79576020 | -1.79577674 | -11.9072 | -11.9065 |
| b3pw91 | C6H6 | -0.19686823 | -0.19690742 | +3.9e-05 | -4.58795091 | -4.58798564 | -7.1159 | -7.1152 |
| b3pw91 | O2 (triplet) | -0.07640252 | -0.07640513 | +2.6e-06 | -2.53131770 | -2.53135931 | -8.6552 / -12.7339 | -8.6531 / -12.7323 |
| hf | H2 | +0.06327136 | +0.06326826 | +3.1e-06 | -1.31609966 | -1.31610309 | -16.1765 | -16.1764 |
| hf | H2O | +0.58002290 | +0.58000211 | +2.1e-05 | -7.90795486 | -7.90796261 | -13.7688 | -13.7692 |
| hf | CH4 | +0.44474097 | +0.44472939 | +1.2e-05 | -5.93861603 | -5.93865321 | -14.8433 | -14.8434 |
| hf | CO | +0.78939764 | +0.78940883 | -1.1e-05 | -9.39275046 | -9.39273655 | -15.0718 | -15.0715 |
| hf | N2 | +0.82313063 | +0.82314955 | -1.9e-05 | -8.96337621 | -8.96339912 | -16.6723 | -16.6722 |
| hf | C6H6 | +2.17027792 | +2.17025095 | +2.7e-05 | -22.95270707 | -22.95275275 | -9.1339 | -9.1340 |
| hf | O2 (triplet) | +1.02030798 | +1.02026324 | +4.5e-05 | -12.70095689 | -12.70095410 | -15.0131 / -15.6681 | -15.0133 / -15.6676 |

Over the table: |ddE| <= 9.3e-05 Ry, |Fock this code - Fock QE| <= 4.6e-05 Ry, |HOMO this code - HOMO QE| <= 2.29 meV. The totals are variational and reproduce to 1e-8 Ry between runs converged to `Exx_Tolerance 1e-8 Ry`; the Fock energies and the levels are first order in the orbitals and reproduce to about 3e-5 Ry and 3e-4 eV (the last digit of a HOMO can change from one such run to the next).

Bound LUMOs (CO, N2, benzene, O2 down channel) agree within 2.6 meV for benzene and O2. Those of CO and N2 in the hybrids (-0.6 to -1.4 eV) are bound weakly enough to feel the sphere: up to +9 meV at 10 bohr of vacuum, +0.2 / +0.8 meV (PBE0) with 4 bohr more. The LUMOs of H2, H2O, CH4 are diffuse domain states in both codes and those of HF are box states: not compared.

## Known offsets

**The Kleinman-Bylander quadrature.** Fortran PARSEC integrates the Kleinman-Bylander energies E_l = int u_l^2 (V_l - V_loc) dr by the trapezoid in r on the logarithmic mesh, which exceeds the converged integral by sinh(dx)/dx - 1 = 9.9e-5 relative, a constant that does not vanish with h; QE uses Simpson's rule. The logs of this folder (default `PARSEC_KB_QUADRATURE=auto` since 23 November 2026) integrate E_l exactly on the logarithmic mesh (Simpson's value to 1e-12 relative). The trapezoid, with which the logs before 23 November 2026 were made, changes the PBE totals of this code by the values of the PBE table, within 1e-5 Ry of the change it makes in QE, so the default removes that part of the difference to QE. In E_hyb - E_PBE it mostly cancels, but not completely when the hybrid changes how the occupied orbitals see the projectors: from the logs of 20 November to those of 23 November 2026 (both with the tables of the files) dE moved by +3.3e-5 Ry (O2 HF), +3.1e-5 (C6H6 HF), +2.4e-5 (N2 HF), +2.1e-5 (CO HF), +1.5e-5 (H2O HF), +1.0e-5 to +1.1e-5 (O2 PBE0, HSE06, HSE03), 1.4e-6 to 9.2e-6 for the other PBE0 and HSE rows, at most 6.2e-6 for B3LYP, B3LYP5 and B3PW91, and 0 for H2 (no projectors), about the opposite of the changes measured in QE with the trapezoid E_l (-2.3e-5 N2 HF, -2.1e-5 CO HF, -1.6e-5 H2O HF, -1.1e-5 O2 PBE0, -7e-6 N2 PBE0, <= 3e-6 B3LYP). The reader also continues the FHI tables below their first radial sample since then (`PARSEC_FHI_ORIGIN=auto`), which moves these totals by at most 1e-8 Ry. `PARSEC_FORTRAN_NUMERICS=on` gives the totals of the logs of 20 November 2026 to the printed digits (A100, 24 November 2026: the seven `pbe` inputs with their SCF iteration counts, the H2O, CO and N2 HF and the O2 and CH4 PBE0 inputs, whose Fock loops now run on the weighted sectors: CO HF within 3e-8 Ry, its `Exx_Tolerance`). `PARSEC_LOCAL_FILTER=off` with `PARSEC_SPLINE_COULOMB_TAIL=off` gives the totals of the logs of 23 November 2026 on the same twelve inputs (CO HF within 3e-8 Ry), and with `PARSEC_KB_QUADRATURE=trapezoid` and `PARSEC_FHI_ORIGIN=off` added those of 20 November 2026 again.

**QE's cutoff error for LYP (and for O).** QE's E_hyb - E_PBE at 150 Ry is not converged to 1e-5 Ry for the LYP functionals and for oxygen. The same QE build at higher cutoffs (L = 24 bohr, ecutfock = 2 ecutwfc):

| molecule | functional | QE 150 Ry | QE at ecut | ecut (Ry) | this code | this code - QE 150 | this code - QE at ecut |
|---|---|---|---|---|---|---|---|
| H2O | pbe0 | +0.00540135 | +0.00542384 | 300 | +0.00543638 | +3.5e-05 | +1.3e-05 |
| H2O | b3lyp | -0.05074475 | -0.05078664 | 300 | -0.05079237 | -4.8e-05 | -5.7e-06 |
| N2 | pbe0 | +0.02828276 | +0.02829726 | 250 | +0.02829892 | +1.6e-05 | +1.7e-06 |
| N2 | b3lyp | -0.01992990 | -0.01995836 | 250 | -0.01996664 | -3.7e-05 | -8.3e-06 |
| N2 | hf | +0.82314955 | +0.82314394 | 250 | +0.82313063 | -1.9e-05 | -1.3e-05 |
| O2 (triplet) | pbe0 | +0.04022361 | +0.04026462 | 250 | +0.04027942 | +5.6e-05 | +1.5e-05 |
| O2 (triplet) | b3lyp | -0.04254443 | -0.04262701 | 250 | -0.04263702 | -9.3e-05 | -1.0e-05 |
| C6H6 | pbe0 | +0.00190234 | +0.00190887 | 160 (QE 120 above) | +0.00193371 | +3.1e-05 | +2.5e-05 |
| C6H6 | hse06 | +0.01174862 | +0.01175731 | 160 (QE 120 above) | +0.01177538 | +2.7e-05 | +1.8e-05 |
| C6H6 | hse03 | -0.60503878 | -0.60502880 | 160 (QE 120 above) | -0.60497967 | +5.9e-05 | +4.9e-05 |

Against QE at the higher cutoff the remaining differences of the cases above are within 2.5e-5 Ry (benzene PBE0; N2 HF -1.3e-5), apart from benzene's HSE03 (+4.9e-5, the derived reference, section "HSE03"), three of them are below 1e-5 and O2 B3LYP at 1.0e-5 (the logs before 23 November 2026 needed this code's KB coefficients in QE as well for that comparison). B3LYP5 has the same differences as B3LYP to 1.3e-7 Ry; B3PW91, with the same exchange and PW91 correlation, does not show the LYP effect. On our side, E_hyb - E_PBE changes by at most 1.0e-5 Ry between h 0.16, 0.14 and 0.12 bohr (H2O PBE0 and B3LYP, N2 PBE0, B3LYP and HF; A100, 24 November 2026).

**B3PW91 constants.** The PW91 and PW92 pieces of this code keep Fortran PARSEC's constants, which differ from libxc's in the last digits (PW92 A, PW91 nu and C_x, the Rasolt-Geldart rs^3 term). On the converged densities libxc's constants change the B3PW91 energy by -0.3e-6 (H2) to -2.6e-6 Ry (O2).

**B3LYP definitions.** `b3lyp` is B3LYP with VWN-RPA (libxc HYB_GGA_XC_B3LYP, Gaussian's), `b3lyp5` with VWN5. QE's internal `B3LYP` uses VWN5 and its own B88/LYP, which differ from libxc's by about 1e-4 Ry for H2O; QE's `B3LYP-V1R` interpolates VWN-RPA in spin differently (8e-4 Ry for O2). The references are therefore QE with libxc 402 and 475.

## libxc's functionals

The same inputs with the functionals of libxc (the optional library, `LibXC_Functional`; see "Functionals of
libxc" in `src/parsec_python/README.md`): `libxc/run_libxc.py` copies a case, replaces its `Correlation_Type` by
the libxc keyword and runs `main.py` (one A100, libxc 7.1.2 CUDA build, the libxc functional evaluated on the
GPU, 2026-11-24):

```bash
PARSEC_LIBXC_LIBRARY=.../libxc.so python examples/0d_hybrid_functionals/libxc/run_libxc.py --out OUT
```

**The code's functionals and libxc's** (total energies, Ry; libxc - this code):

| molecule | `pbe` | `gga_x_pbe gga_c_pbe` | `b3lyp` | `hyb_gga_xc_b3lyp` | `pbe0` | `hyb_gga_xc_pbeh` |
|---|---|---|---|---|---|---|
| H2 | -2.32974875 | -1e-8 | -2.35655788 | 0 | -2.33588087 | -1e-8 |
| H2O | -34.34610161 | -7e-8 | -34.39689398 | 0 | -34.34066523 | -8e-8 |
| CH4 | -16.15717622 | -6e-8 | -16.20040957 | 0 | -16.17295638 | -6e-8 |
| CO | -43.31658892 | -9e-8 | -43.35259960 | 0 | -43.29444707 | -1.0e-7 |
| N2 | -39.83021701 | -1.0e-7 | -39.85018365 | 0 | -39.80191809 | -9e-8 |
| O2 (triplet) | -63.71099604 | -1.3e-7 | -63.75363306 | 0 | | |

B3LYP is the same functional in both (also with spin: O2), to the printed digits, and so are the exchange
energies `a_x E_x`. The PBE of this code differs from libxc's by the PW92 `A` of its correlation
(`(1 - ln 2)/pi^2` against libxc's 0.0310907): 1e-8 to 1.3e-7 Ry here, and by the same in PBE0.

**libxc's HSE06 (428) and HSE03 (427) against Quantum ESPRESSO with libxc** (QE's `hse06_libxc` and
`hse03_libxc` rows, `qe/README.md`): dE = E_hyb - E_PBE (this code: the `pbe` input), Fock = `a_x E_x` (Ry).

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | dFock |
|---|---|---|---|---|---|---|---|
| hse06 (libxc 428) | H2 | -0.00631394 | -0.00631791 | +4.0e-06 | -0.26801640 | -0.26801734 | +9.4e-07 |
| hse06 (libxc 428) | H2O | +0.00752046 | +0.00749483 | +2.6e-05 | -1.72361563 | -1.72361325 | -2.4e-06 |
| hse06 (libxc 428) | CH4 | -0.01276879 | -0.01278614 | +1.7e-05 | -1.23611081 | -1.23611379 | +3.0e-06 |
| hse06 (libxc 428) | CO | +0.02582488 | +0.02581579 | +9.1e-06 | -2.03362773 | -2.03362469 | -3.0e-06 |
| hse06 (libxc 428) | N2 | +0.03242095 | +0.03241806 | +2.9e-06 | -1.93708885 | -1.93708486 | -4.0e-06 |
| hse03 (libxc 427) | H2 | -0.04684159 | -0.04684600 | +4.4e-06 | -0.27061719 | -0.27061929 | +2.1e-06 |
| hse03 (libxc 427) | H2O | -0.16338408 | -0.16340967 | +2.6e-05 | -1.73326402 | -1.73326301 | -1.0e-06 |
| hse03 (libxc 427) | CH4 | -0.17707847 | -0.17709741 | +1.9e-05 | -1.24576310 | -1.24576806 | +5.0e-06 |
| hse03 (libxc 427) | CO | -0.18600097 | -0.18601185 | +1.1e-05 | -2.04554917 | -2.04554848 | -6.9e-07 |
| hse03 (libxc 427) | N2 | -0.17911528 | -0.17911944 | +4.2e-06 | -1.94883343 | -1.94883094 | -2.5e-06 |

The semilocal part from libxc and the screened exact exchange of this code reproduce QE's libxc hybrids within
2.6e-5 Ry, the level of the code's own HSE06 against QE's (table above). libxc's HSE06 differs from the code's
(QE's form, `hse06` above) by -1.2e-3 (H2) to +1.3e-3 Ry (N2): libxc's full-range wPBEh(w = 0) exchange and
its rescaling of s.

## Why `Cubic_Spline` is set

Without it, this code (as PARSEC) interpolates r V(r) linearly between the samples of the FHI logarithmic mesh (dr/r = 2.4 %). Against the smooth potential that QE Fourier-transforms this raises the PBE energy of the inputs by 8.8e-5 (H2), 3.2e-4 (H2O), 9.3e-4 (O2), 9.7e-4 (CH4), 1.6e-3 (CO), 2.5e-3 (N2) and 5.2e-3 Ry (benzene) at the defaults of the logs (A100, 24 November 2026; with the tables of the files, `PARSEC_LOCAL_FILTER=off`, about +2.2e-3 Ry per O, +4e-4 per C and +1.5e-5 per H), makes the dependence on h erratic, and lowers the H2O HOMO by 2.5 (PBE) to 3.9 meV (HF; PBE0 2.9, B3LYP 2.8 meV). `Cubic_Spline: .true.` (PARSEC's spline of the radial table) removes it (1e-7 Ry).

## HSE03

`hse03` is HSE in QE's (= VASP's) form with w_HF = 0.15/sqrt(2) = 0.106066 and w_PBE = 0.15*2^(1/3) = 0.188988 1/bohr. QE's internal HSE has one w for both parts, and libxc's HSE03 (427) writes the full-range exchange as wPBEh(w = 0) instead of PBE and rescales s differently. The reference is therefore derived: hse03_libxc + (hse06 - hse06_libxc), which is HSE03 in QE's form if the libxc-minus-QE form difference does not depend on w. The s-rescaling part of that difference does depend on w: on our densities it makes the derived value 2e-6 (H2) to 9e-6 Ry (benzene) too high. The derived Fock energies and HOMOs are estimates made the same way. Benzene (+4.9e-5 Ry against QE at 160 Ry, HSE06 +1.8e-5) is the largest HSE03 difference; the two functionals share all code except the two w values, so the excess is attributed to the derived reference.

## Timings (indicative)

Wall time of `main.py` (program wall, PBE start included) on one A100 of a shared node, the logs of 24 November 2026 (the node's kernel cache, warmed by two discarded hybrid runs on each GPU before the timed runs, four inputs side by side, one per GPU), against QE on 4 MPI ranks of a laptop (different hardware):

| molecule | grid points | PBE (s) | hybrids (s) | HF (s) | QE, hybrids and HF (s) |
|---|---|---|---|---|---|
| H2 | 1,252,368 | 5 | 7-15 | 8 | 53-158 |
| H2O | 1,568,208 | 8 | 13-23 | 18 | 86-294 |
| CH4 | 1,788,656 | 5 | 10-20 | 13 | 95-283 |
| CO | 1,387,496 | 6 | 9-17 | 11 | 114-299 |
| N2 | 1,376,312 | 5 | 9-18 | 11 | 99-258 |
| C6H6 | 3,241,288 | 12 | 32-54 | 35 | 404-744 |
| O2 (triplet) | 1,413,968 | 10 | 19-29 | 23 | 257-654 |

Outer Fock iterations to an estimated exchange error below 1e-8 Ry: 3-4 for PBE0, HSE and B3*, 6-7 for HF (QE, whose loop runs to 1e-10 Ry and starts the libxc hybrids from the functional without its exact exchange and HF from Slater exchange: 4-6 and 10-16). The table gives outer iterations / inner SCF iterations of the Fock loop (summed over its outer iterations); the PBE start takes 13-25 SCF iterations before them, the `pbe` input 24-37. The exact exchange itself (pair solves and ACE builds) is 0.4-19.6 s of a run.

| molecule | pbe0 | hse06 | hse03 | b3lyp | b3lyp5 | b3pw91 | hf |
|---|---|---|---|---|---|---|---|
| H2 | 3 / 43 | 3 / 32 | 3 / 26 | 3 / 26 | 3 / 24 | 3 / 26 | 6 / 38 |
| H2O | 4 / 35 | 3 / 35 | 3 / 36 | 3 / 23 | 3 / 37 | 4 / 30 | 7 / 54 |
| CH4 | 4 / 25 | 4 / 41 | 4 / 36 | 4 / 40 | 4 / 37 | 4 / 32 | 7 / 73 |
| CO | 4 / 35 | 4 / 37 | 4 / 32 | 4 / 32 | 4 / 32 | 4 / 32 | 7 / 43 |
| N2 | 4 / 28 | 4 / 37 | 4 / 33 | 4 / 32 | 4 / 35 | 4 / 32 | 7 / 45 |
| C6H6 | 4 / 41 | 4 / 65 | 4 / 47 | 4 / 48 | 4 / 46 | 4 / 46 | 7 / 57 |
| O2 (triplet) | 4 / 40 | 4 / 51 | 4 / 42 | 4 / 42 | 4 / 48 | 4 / 39 | 7 / 56 |
