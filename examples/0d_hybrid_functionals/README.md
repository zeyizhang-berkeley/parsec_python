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

With the adaptive inner criterion the PBE start of a spin-unpolarized hybrid stops at a residual of 1e-4 Ry. The energy its log prints for the start (the last `Total Energy` before `Fock loop`, with the note that it is the energy of a start) is therefore not a converged PBE energy: here it lies 4.0e-08 to 1.4e-05 Ry above the energy of the `pbe` input (benzene the most). E_PBE in the tables below is always the energy of the `pbe` input. The O2 start (spin-polarized) runs to `Convergence_Criterion`: its printed energy equals E_PBE to the printed digits.

The logs are of 23 November 2026, made at the defaults of that day (section "Known offsets": the Kleinman-Bylander energies integrated exactly, the FHI tables continued below their first sample); `PARSEC_KB_QUADRATURE=trapezoid` with `PARSEC_FHI_ORIGIN=off` gives the totals of the logs of 20 November 2026 before them. H2, N2, CO and CH4 have the point groups D4h, C4v and Td on the grid, and since 15 November (Td) and 18 November 2026 (D4h, C4v; `PARSEC_ORBITAL_GROUP_MAX_COST` 0.75) a semilocal run solves the weighted one-dimensional sectors of a larger subgroup instead of the abelian group (`src/parsec_python/acceleration/README.md`, "Weighted sectors of a larger subgroup"). Since 19 November 2026 (`PARSEC_ORBITAL_GROUP_SPIN` and `PARSEC_ORBITAL_GROUP_OCCUPATIONS` auto) the spin-polarized O2 `pbe` input with its `occup.in` takes them too (D4h, the six sectors of a subgroup of 8): it needs 27 SCF iterations against 29 on the abelian route (38 against 29 in the logs of 20 November 2026), with the same total energy and HOMOs within 0.005 meV. The total energies of the weighted route are those of the abelian route to the printed digits, so E_PBE and every dE below do not depend on it; the HOMOs of CO and CH4 differ by 0.011 and 0.002 meV. The PBE runs of H2, CH4, CO and N2 took 5 to 6 s (Timings); H2 needed 36 SCF iterations instead of 20 (4.7 to 5.3 s against 3.9 s with `PARSEC_ORBITAL_GROUP=abelian` on the same node, 2026-11-19), CO 36 instead of 32, CH4 37 instead of 36 (2026-11-23). The difference is not a defect of the weighted route: its mixer and norms take the orbit multiplicities (the algebra of the full grid), both routes agree at the first step (total energy within 1e-8 Ry, every term within 1e-6 Ry) and part at the second, where one Chebyshev filter pass per sector, from sector blocks of other sizes, leaves densities that differ by about 1e-5 (Hartree energy 3.4e-5 Ry); the PBE potential carries that difference on (LDA: 24 steps on both routes), and near the criterion of 1e-6 Ry the charge-weighted potential error of the weighted run drifts at 1-3e-6 Ry for several steps. Small changes move the count by up to ten steps either way (A100, 2026-11-19, weighted against abelian: `States_Num` 8 20 and 20 steps, 12 21 and 20; `Grid_Spacing` 0.17 24 and 25; `Diag_Tolerance` 1e-8 25 and 19; `Convergence_Criterion` 1e-7 31 and 22; and on 2026-11-23 CH4 37 steps with the even first counts of the sectors, 31 with `PARSEC_WEIGHTED_FIRST_SHARE=orbitals`), always with the same energies. The hybrids keep the abelian group, the PBE start of the O2 hybrids included (29 SCF iterations).

## PBE level

| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | difference (Ry) | trapezoid E_l: QE / this code (Ry) | HOMO this code (eV) | HOMO QE (eV) |
|---|---|---|---|---|---|---|
| H2 | -2.32971479 | -2.32974716 | +3.24e-05 | 0 / 0 | -10.3764 | -10.3762 |
| H2O | -34.34627903 | -34.34601311 | -2.66e-04 | +1.75e-4 / +1.69e-4 | -7.2389 | -7.2384 |
| CH4 | -16.15716676 | -16.15717314 | +6.38e-06 | -1.8e-5 / -1.91e-5 | -9.4648 | -9.4647 |
| CO | -43.31679330 | -43.31651838 | -2.75e-04 | +8.7e-5 / +7.81e-5 | -9.0616 | -9.0617 |
| N2 | -39.83037726 | -39.83021536 | -1.62e-04 | -4.9e-5 / -4.99e-5 | -10.2963 | -10.2961 |
| C6H6 | -75.47036355 | -75.47016414 | -1.99e-04 | - / -1.52e-4 | -6.3307 | -6.3306 |
| O2 (triplet) | -63.71119430 | -63.71080847 | -3.86e-04 | +2.73e-4 / +2.63e-4 | -6.8739 / -11.6056 | -6.8736 / -11.6064 |

QE: 150 Ry, benzene 120 Ry (QE's H2O total at 150 Ry is 6e-5 Ry above its 250 Ry value; benzene's at 120 Ry is 6e-5 Ry above its 160 Ry value). Trapezoid E_l: the change of the PBE energy when the Kleinman-Bylander energies E_l are integrated by PARSEC's trapezoid on the logarithmic mesh instead of exactly, in QE (UPF files with those E_l) and in this code (`PARSEC_KB_QUADRATURE=trapezoid`, the quadrature of the logs before 23 November 2026; section "Known offsets"). The two agree within 1e-5 Ry (2 to 10 %), so the default quadrature of the logs removes that part of the difference to QE: with the trapezoid the differences were +3.24e-05 (H2), -9.67e-05 (H2O), -1.27e-05 (CH4), -1.97e-04 (CO), -2.12e-04 (N2), -3.51e-04 (C6H6) and -1.23e-04 Ry (O2). What remains is how the grid samples the pseudopotentials, oxygen's above all: at a given h the total depends on where the atoms sit relative to the grid points, because the sum of the local potential times the density over the grid points picks up its Fourier components beyond the resolution of the grid. An O atom on a grid point lies about 1.7e-3 Ry higher than one between grid points at h 0.16 (7e-4 Ry at h 0.12), and the inputs put the O atoms midway between grid lines in x and y, on the low side (O2 -2.6e-4 and H2O -2.0e-4 Ry at h 0.16). Averaged over the eight placements half a grid step apart along each axis, the PBE totals of H2O, O2, CO and N2 lie within 1e-4 Ry of QE's converged values at h 0.16 and agree with them within about 3.5e-5 Ry for h <= 0.14 (QE's: H2O -34.346079, O2 -63.710930 Ry at 600 Ry, with the change from a 20 to a 28 bohr box at 300 Ry added; 400 to 600 Ry change them by less than 1e-7 Ry, the box by at most 4.3e-6 Ry); at the placement of the inputs H2O is still -5.7e-5 and O2 -1.0e-4 Ry below them at h 0.08, CH4 within 6e-5 Ry at every h from 0.20 to 0.08 (A100, 23 November 2026). The trapezoid's change does not depend on h (to 1e-7 Ry from h 0.20 to 0.08): at h 0.16 it hid the placement error of the inputs, but with it the averages over placements lie +1.7e-4 (H2O) and +2.5e-4 Ry (O2) above QE at h 0.12 and 0.10.

## Hybrids against QE

dE = E_hyb - E_PBE (Ry), each code at its own discretization (this code: the `pbe` input); ddE = this code - QE; Fock = a_x E_x^HF of the final orbitals (Ry); HOMO in eV (O2: up / down channel). QE rows: internal `pbe0`, internal `hse` with `screening_parameter = 0.11`, libxc 402 / 475 / 401 for B3LYP / B3LYP5 / B3PW91, `hf`; HSE03 see below.

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | HOMO this code | HOMO QE |
|---|---|---|---|---|---|---|---|---|
| pbe0 | H2 | -0.00613246 | -0.00613599 | +3.5e-06 | -0.32884160 | -0.32884726 | -12.0030 | -12.0031 |
| pbe0 | H2O | +0.00543583 | +0.00540135 | +3.4e-05 | -1.96909373 | -1.96908409 | -9.0474 | -9.0473 |
| pbe0 | CH4 | -0.01577989 | -0.01579183 | +1.2e-05 | -1.47996506 | -1.47996664 | -11.0058 | -11.0058 |
| pbe0 | CO | +0.02214132 | +0.02213003 | +1.1e-05 | -2.33983770 | -2.33983251 | -10.7432 | -10.7436 |
| pbe0 | N2 | +0.02829557 | +0.02828276 | +1.3e-05 | -2.24329292 | -2.24328729 | -12.1982 | -12.1982 |
| pbe0 | C6H6 | +0.00193165 | +0.00190234 | +2.9e-05 | -5.73059507 | -5.73059923 | -7.2997 | -7.2996 |
| pbe0 | O2 (triplet) | +0.04027976 | +0.04022361 | +5.6e-05 | -3.16308375 | -3.16306389 | -9.0164 / -12.9359 | -9.0165 / -12.9364 |
| hse06 | H2 | -0.00510568 | -0.00510630 | +6.2e-07 | -0.26808897 | -0.26809615 | -11.5836 | -11.5836 |
| hse06 | H2O | +0.00763053 | +0.00760713 | +2.3e-05 | -1.72404861 | -1.72404677 | -8.6325 | -8.6324 |
| hse06 | CH4 | -0.01247303 | -0.01247594 | +2.9e-06 | -1.23633090 | -1.23633660 | -10.5919 | -10.5918 |
| hse06 | CO | +0.02494597 | +0.02494017 | +5.8e-06 | -2.03396136 | -2.03395215 | -10.3274 | -10.3276 |
| hse06 | N2 | +0.03111561 | +0.03111482 | +7.9e-07 | -1.93747948 | -1.93747648 | -11.7847 | -11.7846 |
| hse06 | C6H6 | +0.01177338 | +0.01174862 | +2.5e-05 | -4.81796909 | -4.81797872 | -6.9030 | -6.9027 |
| hse06 | O2 (triplet) | +0.04314444 | +0.04310624 | +3.8e-05 | -2.79500184 | -2.79497840 | -8.6014 / -12.5198 | -8.6017 / -12.5203 |
| hse03 | H2 | -0.04563330 | -0.04563432 | +1.0e-06 | -0.27069015 | -0.27069810 | -11.8640 | -11.8640 |
| hse03 | H2O | -0.16327578 | -0.16329488 | +1.9e-05 | -1.73369616 | -1.73369653 | -8.9097 | -8.9096 |
| hse03 | CH4 | -0.17677813 | -0.17679047 | +1.2e-05 | -1.24598252 | -1.24599087 | -10.8677 | -10.8676 |
| hse03 | CO | -0.18688020 | -0.18688571 | +5.5e-06 | -2.04588086 | -2.04587594 | -10.6063 | -10.6065 |
| hse03 | N2 | -0.18042071 | -0.18042079 | +8.0e-08 | -1.94922248 | -1.94922256 | -12.0683 | -12.0681 |
| hse03 | C6H6 | -0.60498160 | -0.60503878 | +5.7e-05 | -4.85274047 | -4.85275277 | -7.1615 | -7.1611 |
| hse03 | O2 (triplet) | -0.21599428 | -0.21604556 | +5.1e-05 | -2.80907334 | -2.80905330 | -8.8924 / -12.8019 | -8.8927 / -12.8024 |
| b3lyp | H2 | -0.02680872 | -0.02681196 | +3.2e-06 | -0.26262190 | -0.26263902 | -11.8164 | -11.8156 |
| b3lyp | H2O | -0.05079164 | -0.05074475 | -4.7e-05 | -1.57307200 | -1.57308427 | -8.8202 | -8.8190 |
| b3lyp | CH4 | -0.04323376 | -0.04323551 | +1.8e-06 | -1.18350717 | -1.18352905 | -10.7775 | -10.7768 |
| b3lyp | CO | -0.03600822 | -0.03598029 | -2.8e-05 | -1.87086127 | -1.87086635 | -10.5662 | -10.5657 |
| b3lyp | N2 | -0.01996725 | -0.01992990 | -3.7e-05 | -1.79306996 | -1.79307506 | -12.0173 | -12.0165 |
| b3lyp | C6H6 | -0.00960678 | -0.00963286 | +2.6e-05 | -4.58671019 | -4.58674653 | -7.0362 | -7.0354 |
| b3lyp | O2 (triplet) | -0.04263554 | -0.04254443 | -9.1e-05 | -2.52869753 | -2.52872814 | -8.7550 / -12.7446 | -8.7531 / -12.7432 |
| b3lyp5 | H2 | -0.01329455 | -0.01329778 | +3.2e-06 | -0.26253393 | -0.26255086 | -11.7248 | -11.7240 |
| b3lyp5 | H2O | +0.00680379 | +0.00685061 | -4.7e-05 | -1.57283430 | -1.57284672 | -8.7283 | -8.7270 |
| b3lyp5 | CH4 | +0.01252703 | +0.01252528 | +1.8e-06 | -1.18329681 | -1.18331869 | -10.6855 | -10.6848 |
| b3lyp5 | CO | +0.03569573 | +0.03572351 | -2.8e-05 | -1.87061060 | -1.87061573 | -10.4741 | -10.4735 |
| b3lyp5 | N2 | +0.05162656 | +0.05166384 | -3.7e-05 | -1.79284204 | -1.79284712 | -11.9241 | -11.9233 |
| b3lyp5 | C6H6 | +0.20188782 | +0.20186172 | +2.6e-05 | -4.58619589 | -4.58623198 | -6.9458 | -6.9449 |
| b3lyp5 | O2 (triplet) | +0.04403851 | +0.04412960 | -9.1e-05 | -2.52849327 | -2.52852392 | -8.6689 / -12.6421 | -8.6670 / -12.6407 |
| b3pw91 | H2 | -0.02627809 | -0.02628176 | +3.7e-06 | -0.26358859 | -0.26360699 | -11.7624 | -11.7617 |
| b3pw91 | H2O | -0.06939703 | -0.06941150 | +1.4e-05 | -1.57583873 | -1.57585112 | -8.7473 | -8.7463 |
| b3pw91 | CH4 | -0.07561938 | -0.07563168 | +1.2e-05 | -1.18510409 | -1.18512474 | -10.7572 | -10.7565 |
| b3pw91 | CO | -0.06880257 | -0.06879803 | -4.5e-06 | -1.87285244 | -1.87286506 | -10.4833 | -10.4829 |
| b3pw91 | N2 | -0.06051820 | -0.06051616 | -2.0e-06 | -1.79576516 | -1.79577674 | -11.9073 | -11.9065 |
| b3pw91 | C6H6 | -0.19686935 | -0.19690742 | +3.8e-05 | -4.58795088 | -4.58798564 | -7.1161 | -7.1152 |
| b3pw91 | O2 (triplet) | -0.07640171 | -0.07640513 | +3.4e-06 | -2.53132653 | -2.53135931 | -8.6548 / -12.7334 | -8.6531 / -12.7323 |
| hf | H2 | +0.06327050 | +0.06326826 | +2.2e-06 | -1.31608532 | -1.31610309 | -16.1763 | -16.1764 |
| hf | H2O | +0.58002336 | +0.58000211 | +2.1e-05 | -7.90797071 | -7.90796261 | -13.7685 | -13.7692 |
| hf | CH4 | +0.44474218 | +0.44472939 | +1.3e-05 | -5.93861780 | -5.93865321 | -14.8432 | -14.8434 |
| hf | CO | +0.78940101 | +0.78940883 | -7.8e-06 | -9.39276237 | -9.39273655 | -15.0714 | -15.0715 |
| hf | N2 | +0.82312295 | +0.82314955 | -2.7e-05 | -8.96340022 | -8.96339912 | -16.6718 | -16.6722 |
| hf | C6H6 | +2.17027387 | +2.17025095 | +2.3e-05 | -22.95270652 | -22.95275275 | -9.1341 | -9.1340 |
| hf | O2 (triplet) | +1.02031202 | +1.02026324 | +4.9e-05 | -12.70100569 | -12.70095410 | -15.0127 / -15.6676 | -15.0133 / -15.6676 |

Over the table: |ddE| <= 9.1e-05 Ry, |Fock this code - Fock QE| <= 5.2e-05 Ry, |HOMO this code - HOMO QE| <= 1.87 meV. The totals are variational and reproduce to 1e-8 Ry between runs converged to `Exx_Tolerance 1e-8 Ry`; the Fock energies and the levels are first order in the orbitals and reproduce to about 3e-5 Ry and 3e-4 eV (the last digit of a HOMO can change from one such run to the next).

Bound LUMOs (CO, N2, benzene, O2 down channel) agree within 2.1 meV for benzene and O2. Those of CO and N2 in the hybrids (-0.6 to -1.4 eV) are bound weakly enough to feel the sphere: up to +9 meV at 10 bohr of vacuum, +0.2 / +0.8 meV (PBE0) with 4 bohr more. The LUMOs of H2, H2O, CH4 are diffuse domain states in both codes and those of HF are box states: not compared.

## Known offsets

**The Kleinman-Bylander quadrature.** Fortran PARSEC integrates the Kleinman-Bylander energies E_l = int u_l^2 (V_l - V_loc) dr by the trapezoid in r on the logarithmic mesh, which exceeds the converged integral by sinh(dx)/dx - 1 = 9.9e-5 relative, a constant that does not vanish with h; QE uses Simpson's rule. The logs of this folder (23 November 2026, default `PARSEC_KB_QUADRATURE=auto`) integrate E_l exactly on the logarithmic mesh (Simpson's value to 1e-12 relative). The trapezoid, with which the logs before that date were made, changes the PBE totals of this code by the values of the PBE table, within 1e-5 Ry of the change it makes in QE, so the default removes that part of the difference to QE. In E_hyb - E_PBE it mostly cancels, but not completely when the hybrid changes how the occupied orbitals see the projectors: against the former logs dE moved by +3.3e-5 Ry (O2 HF), +3.1e-5 (C6H6 HF), +2.4e-5 (N2 HF), +2.1e-5 (CO HF), +1.5e-5 (H2O HF), +1.0e-5 to +1.1e-5 (O2 PBE0, HSE06, HSE03), 1.4e-6 to 9.2e-6 for the other PBE0 and HSE rows, at most 6.2e-6 for B3LYP, B3LYP5 and B3PW91, and 0 for H2 (no projectors), about the opposite of the changes measured in QE with the trapezoid E_l (-2.3e-5 N2 HF, -2.1e-5 CO HF, -1.6e-5 H2O HF, -1.1e-5 O2 PBE0, -7e-6 N2 PBE0, <= 3e-6 B3LYP). The reader also continues the FHI tables below their first radial sample since then (`PARSEC_FHI_ORIGIN=auto`), which moves these totals by at most 1e-8 Ry. `PARSEC_KB_QUADRATURE=trapezoid` with `PARSEC_FHI_ORIGIN=off`, or `PARSEC_FORTRAN_NUMERICS=on`, gives the totals of the former logs to the printed digits (checked on the seven `pbe` inputs; the pair also on the H2O, CO and N2 HF and the O2 PBE0 inputs).

**QE's cutoff error for LYP (and for O).** QE's E_hyb - E_PBE at 150 Ry is not converged to 1e-5 Ry for the LYP functionals and for oxygen. The same QE build at higher cutoffs (L = 24 bohr, ecutfock = 2 ecutwfc):

| molecule | functional | QE 150 Ry | QE at ecut | ecut (Ry) | this code | this code - QE 150 | this code - QE at ecut |
|---|---|---|---|---|---|---|---|
| H2O | pbe0 | +0.00540135 | +0.00542384 | 300 | +0.00543583 | +3.4e-05 | +1.2e-05 |
| H2O | b3lyp | -0.05074475 | -0.05078664 | 300 | -0.05079164 | -4.7e-05 | -5.0e-06 |
| N2 | pbe0 | +0.02828276 | +0.02829726 | 250 | +0.02829557 | +1.3e-05 | -1.7e-06 |
| N2 | b3lyp | -0.01992990 | -0.01995836 | 250 | -0.01996725 | -3.7e-05 | -8.9e-06 |
| N2 | hf | +0.82314955 | +0.82314394 | 250 | +0.82312295 | -2.7e-05 | -2.1e-05 |
| O2 (triplet) | pbe0 | +0.04022361 | +0.04026462 | 250 | +0.04027976 | +5.6e-05 | +1.5e-05 |
| O2 (triplet) | b3lyp | -0.04254443 | -0.04262701 | 250 | -0.04263554 | -9.1e-05 | -8.5e-06 |
| C6H6 | pbe0 | +0.00190234 | +0.00190887 | 160 (QE 120 above) | +0.00193165 | +2.9e-05 | +2.3e-05 |
| C6H6 | hse06 | +0.01174862 | +0.01175731 | 160 (QE 120 above) | +0.01177338 | +2.5e-05 | +1.6e-05 |
| C6H6 | hse03 | -0.60503878 | -0.60502880 | 160 (QE 120 above) | -0.60498160 | +5.7e-05 | +4.7e-05 |

Against QE at the higher cutoff the remaining differences of the cases above are within 2.3e-5 Ry (benzene PBE0; N2 HF -2.1e-5), apart from benzene's HSE03 (+4.7e-5, the derived reference, section "HSE03"), and four of them are below 1e-5 (the logs before 23 November 2026 needed this code's KB coefficients in QE as well for that comparison). B3LYP5 has the same differences as B3LYP to 1e-7 Ry; B3PW91, with the same exchange and PW91 correlation, does not show the LYP effect. On our side, E_hyb - E_PBE changes by at most 1.3e-5 Ry between h 0.16, 0.14 and 0.12 bohr (H2O, N2).

**B3PW91 constants.** The PW91 and PW92 pieces of this code keep Fortran PARSEC's constants, which differ from libxc's in the last digits (PW92 A, PW91 nu and C_x, the Rasolt-Geldart rs^3 term). On the converged densities libxc's constants change the B3PW91 energy by -0.3e-6 (H2) to -2.6e-6 Ry (O2).

**B3LYP definitions.** `b3lyp` is B3LYP with VWN-RPA (libxc HYB_GGA_XC_B3LYP, Gaussian's), `b3lyp5` with VWN5. QE's internal `B3LYP` uses VWN5 and its own B88/LYP, which differ from libxc's by about 1e-4 Ry for H2O; QE's `B3LYP-V1R` interpolates VWN-RPA in spin differently (8e-4 Ry for O2). The references are therefore QE with libxc 402 and 475.

## libxc's functionals

The same inputs with the functionals of libxc (the optional library, `LibXC_Functional`; see "Functionals of
libxc" in `src/parsec_python/README.md`): `libxc/run_libxc.py` copies a case, replaces its `Correlation_Type` by
the libxc keyword and runs `main.py` (one A100, libxc 7.1.2 CUDA build, the libxc functional evaluated on the
GPU, 2026-11-23):

```bash
PARSEC_LIBXC_LIBRARY=.../libxc.so python examples/0d_hybrid_functionals/libxc/run_libxc.py --out OUT
```

**The code's functionals and libxc's** (total energies, Ry; libxc - this code):

| molecule | `pbe` | `gga_x_pbe gga_c_pbe` | `b3lyp` | `hyb_gga_xc_b3lyp` | `pbe0` | `hyb_gga_xc_pbeh` |
|---|---|---|---|---|---|---|
| H2 | -2.32971479 | -1e-8 | -2.35652351 | 0 | -2.33584725 | -1e-8 |
| H2O | -34.34627903 | -7e-8 | -34.39707067 | 0 | -34.34084320 | -7e-8 |
| CH4 | -16.15716676 | -6e-8 | -16.20040052 | 0 | -16.17294665 | -6e-8 |
| CO | -43.31679330 | -9e-8 | -43.35280152 | 0 | -43.29465198 | -1.0e-7 |
| N2 | -39.83037726 | -9e-8 | -39.85034451 | 0 | -39.80208169 | -9e-8 |
| O2 (triplet) | -63.71119430 | -1.3e-7 | -63.75382984 | 0 | | |

B3LYP is the same functional in both (also with spin: O2), to the printed digits, and so are the exchange
energies `a_x E_x`. The PBE of this code differs from libxc's by the PW92 `A` of its correlation
(`(1 - ln 2)/pi^2` against libxc's 0.0310907): 1e-8 to 1.3e-7 Ry here, and by the same in PBE0.

**libxc's HSE06 (428) and HSE03 (427) against Quantum ESPRESSO with libxc** (QE's `hse06_libxc` and
`hse03_libxc` rows, `qe/README.md`): dE = E_hyb - E_PBE (this code: the `pbe` input), Fock = `a_x E_x` (Ry).

| functional | molecule | dE this code | dE QE | ddE | Fock this code | Fock QE | dFock |
|---|---|---|---|---|---|---|---|
| hse06 (libxc 428) | H2 | -0.00631429 | -0.00631791 | +3.6e-06 | -0.26801282 | -0.26801734 | +4.5e-06 |
| hse06 (libxc 428) | H2O | +0.00752002 | +0.00749483 | +2.5e-05 | -1.72361913 | -1.72361325 | -5.9e-06 |
| hse06 (libxc 428) | CH4 | -0.01276876 | -0.01278614 | +1.7e-05 | -1.23611104 | -1.23611379 | +2.8e-06 |
| hse06 (libxc 428) | CO | +0.02582462 | +0.02581579 | +8.8e-06 | -2.03363231 | -2.03362469 | -7.6e-06 |
| hse06 (libxc 428) | N2 | +0.03241786 | +0.03241806 | -2.0e-07 | -1.93709509 | -1.93708486 | -1.0e-05 |
| hse03 (libxc 427) | H2 | -0.04684184 | -0.04684600 | +4.2e-06 | -0.27061363 | -0.27061929 | +5.7e-06 |
| hse03 (libxc 427) | H2O | -0.16338446 | -0.16340967 | +2.5e-05 | -1.73326753 | -1.73326301 | -4.5e-06 |
| hse03 (libxc 427) | CH4 | -0.17707843 | -0.17709741 | +1.9e-05 | -1.24576333 | -1.24576806 | +4.7e-06 |
| hse03 (libxc 427) | CO | -0.18600112 | -0.18601185 | +1.1e-05 | -2.04555350 | -2.04554848 | -5.0e-06 |
| hse03 (libxc 427) | N2 | -0.17911830 | -0.17911944 | +1.1e-06 | -1.94883960 | -1.94883094 | -8.7e-06 |

The semilocal part from libxc and the screened exact exchange of this code reproduce QE's libxc hybrids within
2.5e-5 Ry, the level of the code's own HSE06 against QE's (table above). libxc's HSE06 differs from the code's
(QE's form, `hse06` above) by -1.2e-3 (H2) to +1.3e-3 Ry (N2): libxc's full-range wPBEh(w = 0) exchange and
its rescaling of s.

## Why `Cubic_Spline` is set

Without it, this code (as PARSEC) interpolates r V(r) linearly between the samples of the FHI logarithmic mesh (dr/r = 2.4 %). Against the smooth potential that QE Fourier-transforms this shifts the energy by about +2.2e-3 Ry per O, +4e-4 per C and +1.5e-5 per H, makes the dependence on h erratic, and lowers the H2O HOMO by 3 to 4.5 meV. `Cubic_Spline: .true.` (PARSEC's spline of the radial table) removes it (1e-7 Ry).

## HSE03

`hse03` is HSE in QE's (= VASP's) form with w_HF = 0.15/sqrt(2) = 0.106066 and w_PBE = 0.15*2^(1/3) = 0.188988 1/bohr. QE's internal HSE has one w for both parts, and libxc's HSE03 (427) writes the full-range exchange as wPBEh(w = 0) instead of PBE and rescales s differently. The reference is therefore derived: hse03_libxc + (hse06 - hse06_libxc), which is HSE03 in QE's form if the libxc-minus-QE form difference does not depend on w. The s-rescaling part of that difference does depend on w: on our densities it makes the derived value 2e-6 (H2) to 9e-6 Ry (benzene) too high. The derived Fock energies and HOMOs are estimates made the same way. Benzene (+4.7e-5 Ry against QE at 160 Ry, HSE06 +1.6e-5) is the largest HSE03 difference; the two functionals share all code except the two w values, so the excess is attributed to the derived reference.

## Timings (indicative)

Wall time of `main.py` (program wall, PBE start included) on one A100 of a shared node, the logs of 23 November 2026 (the node's kernel cache, warmed by two discarded hybrid runs on each GPU before the timed runs, four inputs side by side, one per GPU), against QE on 4 MPI ranks of a laptop (different hardware):

| molecule | grid points | PBE (s) | hybrids (s) | HF (s) | QE, hybrids and HF (s) |
|---|---|---|---|---|---|
| H2 | 1,252,368 | 5 | 8-17 | 10 | 53-158 |
| H2O | 1,568,208 | 7 | 13-23 | 18 | 86-294 |
| CH4 | 1,788,656 | 6 | 13-24 | 21 | 95-283 |
| CO | 1,387,496 | 6 | 13-22 | 16 | 114-299 |
| N2 | 1,376,312 | 6 | 11-19 | 13 | 99-258 |
| C6H6 | 3,241,288 | 11 | 31-54 | 36 | 404-744 |
| O2 (triplet) | 1,413,968 | 11 | 24-29 | 29 | 257-654 |

Outer Fock iterations to an estimated exchange error below 1e-8 Ry: 3-4 for PBE0, HSE and B3*, 6-7 for HF (QE, whose loop runs to 1e-10 Ry and starts the libxc hybrids from the functional without its exact exchange and HF from Slater exchange: 4-6 and 10-16). The table gives outer iterations / inner SCF iterations of the Fock loop (summed over its outer iterations); the PBE start takes 13-29 SCF iterations before them, the `pbe` input 24-37. The exact exchange itself (pair solves and ACE builds) is 0.4-19.7 s of a run.

| molecule | pbe0 | hse06 | hse03 | b3lyp | b3lyp5 | b3pw91 | hf |
|---|---|---|---|---|---|---|---|
| H2 | 3 / 39 | 3 / 40 | 3 / 27 | 3 / 26 | 3 / 21 | 3 / 27 | 6 / 38 |
| H2O | 4 / 35 | 3 / 39 | 3 / 27 | 3 / 23 | 3 / 37 | 4 / 30 | 7 / 54 |
| CH4 | 4 / 29 | 4 / 40 | 4 / 42 | 4 / 36 | 4 / 37 | 4 / 32 | 7 / 72 |
| CO | 4 / 39 | 4 / 53 | 4 / 36 | 4 / 32 | 4 / 31 | 4 / 31 | 7 / 44 |
| N2 | 4 / 28 | 4 / 38 | 4 / 37 | 4 / 38 | 4 / 35 | 4 / 30 | 7 / 45 |
| C6H6 | 4 / 41 | 4 / 60 | 4 / 48 | 4 / 48 | 4 / 46 | 4 / 47 | 7 / 57 |
| O2 (triplet) | 4 / 43 | 4 / 51 | 4 / 42 | 4 / 43 | 4 / 47 | 4 / 39 | 7 / 52 |
