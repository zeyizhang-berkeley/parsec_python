# Meta-GGAs and meta-GGA hybrids: six molecules against Quantum ESPRESSO

SCAN (this code's native SCAN), r2SCAN and TPSS (libxc) and the meta-GGA hybrids TPSSh and r2SCAN0 (libxc's
semilocal part with this code's exact exchange), self-consistent, for H2O, CH4, NH3, N2, CO and triplet O2
(spin-polarized), compared with Quantum ESPRESSO 7.5 (+ libxc 7.0.0) using the same FHI98PP pseudopotentials and
geometries; forces of a distorted water molecule. The quantity compared is E_mGGA - E_PBE of each code at its own
discretization, where both codes' PBE-level conventions cancel to first order; HOMOs, exact-exchange energies and
forces are compared directly. Bulk Si is in [`../3d_meta_gga`](../3d_meta_gga/README.md).

## Contents

- `<molecule>/<variant>/parsec.in` (O2: with `occup.in`, 7 up and 5 down electrons) and `parsec_python.out`, the
  reference log of that input (one NVIDIA A100; paths shortened to repository paths, the libxc library as
  `$PARSEC_LIBXC_LIBRARY`). Variants: `pbe`, `scan`, `r2scan`, `tpss` for the six molecules; `tpssh`, `r2scan0` for
  the five closed shells; `pbe_h013`, `r2scan_h013` (grid spacing 0.13 bohr) for H2O and O2; `H2O_dist/pbe`,
  `H2O_dist/r2scan`: water with one O-H bond stretched to 1.10 A, with forces.
- Pseudopotentials (copies of the project's library `src/pp_generation/NCPPs/PBE`, found with `--pp-dir`):

  | file | from | used by |
  |---|---|---|
  | `H_FHIPP.DAT`, `C_FHIPP.DAT`, `N_FHIPP.DAT`, `O_FHIPP.DAT` | `NCPPs/PBE/<element>/FHI` (Troullier-Martins, PBE, no core correction) | the inputs here, `python -m parsec_python.acceleration.benchmarks.meta_gga_check`, `tests/unit/test_meta_gga_scf.py` |
  | `H_POTRE.DAT`, `Li_POTRE.DAT` | `NCPPs/PBE/<element>/PARSEC` | `tests/unit/test_meta_gga_scf.py` (LiH: Li has a nonlinear core correction) |

- `qe/`: the Quantum ESPRESSO inputs of the compared rows, their settings and QE's cutoff convergence
  (`qe/README.md`).
- `run_examples.py` (one process per input, `main.py --gpus 1`; `--install` copies the logs of a run folder into the
  example) and `collect_results.py` (the tables below from the logs).

## Run

r2SCAN, TPSS and the hybrids are libxc's: build or install libxc and point `PARSEC_LIBXC_LIBRARY` at it ("Functionals
of libxc" in `src/parsec_python/README.md`; the logs here used libxc 7.1.2, CUDA build, evaluated on the GPU). From
the repository root:

```bash
export PARSEC_LIBXC_LIBRARY=.../libxc-7.1.2-cuda-sm80/lib/libxc.so
python src/parsec_python/main.py examples/0d_meta_gga/H2O/r2scan/parsec.in --pp-dir examples/0d_meta_gga --no-archive
python examples/0d_meta_gga/run_examples.py --out RUNS          # every input, logs in RUNS/<molecule>/<variant>/
python examples/0d_meta_gga/collect_results.py [--runs RUNS]
```

`parsec.out` is written beside the input; compare it with `parsec_python.out`. SCAN needs no library.

Settings of every input (those of `../0d_hybrid_functionals`): grid spacing 0.16 bohr, sphere = largest atom distance
from the centre + 10 bohr, `Expansion_Order` 12, `Convergence_Criterion` 1e-6 Ry, `Max_Iter` 200, `Cubic_Spline` for
every species; the hybrids `Exx_Tolerance` 1e-8 Ry (default 1e-6 Ry, tight for the comparison). Everything else is
the default.

### The functional in the input

Three spellings, each used here:

```text
Correlation_Type: scan                                  # this code's SCAN (scan/)
LibXC_Functional: mgga_x_r2scan mgga_c_r2scan           # libxc by name (r2scan/, and the hybrids:
LibXC_Functional: hyb_mgga_xc_tpssh                     #   tpssh/, r2scan0/)
Use_LibXC: .true.                                       # PARSEC 1.5's keywords, libxc by number (tpss/):
Func_Id_X: 202                                          #   MGGA_X_TPSS
Func_Id_C: 231                                          #   MGGA_C_TPSS
```

`LibXC_Functional: mgga_x_tpss mgga_c_tpss` is the same functional as the 1.5 form (the run's settings are equal).
In 1.5's form an exchange-correlation functional in `Func_Id_X` (`457` for TPSSh) gets 1.5's default `Func_Id_C`
(LDA correlation) added, as 1.5 does, with a warning; `Func_Id_C: 0` leaves it out. The hybrids here are therefore
written with `LibXC_Functional`. `python -m parsec_python.V_xc.libxc --list` lists libxc's functionals.

### Defaults and how to switch them

Each log names what it did under "Correlation data".

- **Density gradient.** `auto`: the orbital form `g = sum_n f_n 2 psi_n (G psi_n)` for functionals smooth across
  `tau = tau_W` (SCAN, r2SCAN, r2SCAN0: "density gradient: orbital form") and the stencil gradient of the density for
  those with a kink there (TPSS, TPSSh: "density form"). `PARSEC_META_GGA_GRADIENT=orbital` or `density` forces one.
- **Floors of `tau`.** libxc meta-GGAs floor `tau` of each channel at `LibXC_Tau_Threshold` (1e-12 Hartree/bohr^3:
  "tau floor 1e-12"); TPSS-type functionals also at `tau_W` with the exact derivatives ("tau >= tau_W floor (Fermi-hole
  curvature; the functional has a kink at tau_W)"), so that all their components see `z = tau_W/tau <= 1`.
  `LibXC_Enforce_FHC: .false.` evaluates them at the `tau` of the orbitals, `.true.` floors any libxc meta-GGA. libxc's
  own clamp (`sigma -> 8 n tau`, not carried into the derivatives) is never used; Quantum ESPRESSO switches it off as
  well. The native SCAN has no floor (`PARSEC_SCAN_TAU_FLOOR=weizsaecker` adds the `tau_W` one).
- **First iteration and mixing.** The first iteration is PBE of the start density (`vtau = 0`), as PARSEC 1.5 starts
  its libxc meta-GGAs; the mixer mixes `(V, vtau[, W])` of all channels jointly with Anderson's least squares
  weighted by the density (`PARSEC_META_GGA_ANDERSON=weighted`; `plain` for PARSEC's metric). The filter's upper bound
  takes in Gershgorin's bound of the orbital term (`PARSEC_META_GGA_FILTER_BOUND=gershgorin`).
- **Meta-GGA hybrids** run the Fock loop of the hybrids (PBE start, ACE, `Exx_*` keywords) with meta-GGA SCFs as
  inner loops; libxc gives the exact-exchange fraction (TPSSh 0.10, r2SCAN0 0.25).

All switches: "Meta-GGA functionals" and "Functionals of libxc" in `src/parsec_python/README.md`.

## PBE level

QE at 300 Ry (N2, CO: 150 Ry); HOMO in eV (O2: up / down channel).

| molecule | E_PBE this code (Ry) | E_PBE QE (Ry) | QE cutoff (Ry) | difference (Ry) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) |
|---|---|---|---|---|---|---|---|
| H2O | -34.34627903 | -34.34607194 | 300 | -2.07e-04 | -7.2389 | -7.2383 | -0.6 |
| CH4 | -16.15716676 | -16.15718789 | 300 | +2.11e-05 | -9.4648 | -9.4644 | -0.4 |
| NH3 | -23.46695052 | -23.46695167 | 300 | +1.15e-06 | -6.1613 | -6.1604 | -0.9 |
| N2 | -39.83037726 | -39.83021459 | 150 | -1.63e-04 | -10.2963 | -10.2959 | -0.4 |
| CO | -43.31679330 | -43.31651727 | 150 | -2.76e-04 | -9.0616 | -9.0616 | +0.0 |
| O2 | -63.71119430 | -63.71092564 | 300 | -2.69e-04 | -6.8739 / -11.6056 | -6.8741 / -11.6063 | +0.2 / +0.7 |

The PBE inputs equal those of `../0d_hybrid_functionals` (its README explains the remaining offsets: the grid,
oxygen the slowest, and QE's cutoff at 150 Ry). The logs are of 23 November 2026, whose defaults integrate the
Kleinman-Bylander energies exactly on the logarithmic mesh, as QE's Simpson rule does, and continue the FHI tables
below their first sample (`../0d_hybrid_functionals/README.md`, "Known offsets"). PARSEC's trapezoid, with which the
logs before that date were made, changes the PBE totals by +1.69e-4 (H2O), -1.91e-5 (CH4), +1.70e-5 (NH3), -4.99e-5
(N2), +7.81e-5 (CO) and +2.63e-4 Ry (O2), which for H2O and O2 hid most of the grid error of oxygen: the differences
to QE were -3.79e-05, +2.07e-06, +1.81e-05, -2.13e-04, -1.98e-04 and -5.44e-06 Ry. In E - E_PBE of the meta-GGAs and their hybrids below it
cancelled to 0.018 mRy. `PARSEC_KB_QUADRATURE=trapezoid` with `PARSEC_FHI_ORIGIN=off` (and `PARSEC_CA_CONTINUITY=off`,
`PARSEC_FERMI_GAP_ROOT=off`) gives the totals, SCF iterations and forces of those logs to the printed digits (checked on
eight inputs: H2O PBE and r2SCAN, CH4 TPSSh, N2 SCAN, O2 TPSS, CO r2SCAN0, O2 PBE at h = 0.13 bohr and the distorted
water with r2SCAN). CH4, N2, CO and O2 have the point groups Td, D4h, C4v and D4h on the grid, and a PBE run solves
the weighted one-dimensional sectors of a larger subgroup (D2d for Td, a subgroup of order 8 or 4 for D4h and C4v)
instead of the abelian group of the logs of 8 November 2026: CH4, N2 and CO since 15 and 18 November, the
spin-polarized O2 with its `occup.in` since 20 November (27 SCF iterations against 29 on the abelian route, at
h = 0.13 bohr 42 against 38; in the logs of 20 November 38 against 29 and 42 against 37). Their energies are those of
the abelian runs to the printed digits, and their HOMOs within 0.011 meV (CO). The meta-GGAs and their hybrids keep
the abelian group (`src/parsec_python/acceleration/README.md`, "Weighted sectors of a larger subgroup").

## Meta-GGAs against QE

E - E_PBE in mRy, each code against its own PBE (QE: at the cutoff of the column, 300 Ry where QE's convergence rows
exist, TPSS of H2O and O2 250 Ry, else 150 Ry; `qe/README.md`); HOMO in eV (O2: up / down). "Max_Iter": the run
stopped at `Max_Iter` 200 with the energy settled (see "SCAN" below).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | QE cutoff (Ry) | this code - QE (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | SCF iterations | gradient form |
|---|---|---|---|---|---|---|---|---|---|---|
| H2O | SCAN | +1.406 | +1.762 | 300 | -0.356 | -7.5583 | -7.5973 | +39.0 | 46 | orbital |
| H2O | r2SCAN | -2.122 | -2.024 | 300 | -0.099 | -7.5682 | -7.5688 | +0.6 | 31 | orbital |
| H2O | TPSS | -11.073 | -11.083 | 250 | +0.010 | -7.3158 | -7.3158 | +0.0 | 38 | density |
| CH4 | SCAN (Max_Iter) | -4.041 | -3.924 | 300 | -0.117 | -9.8256 | -9.8252 | -0.4 | 200 | orbital |
| CH4 | r2SCAN | -1.505 | -1.443 | 300 | -0.062 | -9.8133 | -9.8132 | -0.1 | 29 | orbital |
| CH4 | TPSS | -39.429 | -39.357 | 150 | -0.072 | -9.6570 | -9.6566 | -0.4 | 28 | density |
| NH3 | SCAN (Max_Iter) | +6.224 | +6.516 | 300 | -0.292 | -6.5036 | -6.4947 | -8.9 | 200 | orbital |
| NH3 | r2SCAN | +4.859 | +4.822 | 300 | +0.037 | -6.4626 | -6.4625 | -0.1 | 60 | orbital |
| NH3 | TPSS | -17.647 | -17.640 | 150 | -0.007 | -6.2285 | -6.2282 | -0.3 | 61 | density |
| N2 | SCAN | +38.794 | +39.273 | 150 | -0.479 | -10.7707 | -10.8024 | +31.7 | 42 | orbital |
| N2 | r2SCAN | +35.241 | +35.240 | 150 | +0.001 | -10.7266 | -10.7267 | +0.1 | 39 | orbital |
| N2 | TPSS | +8.423 | +8.392 | 150 | +0.031 | -10.4620 | -10.4621 | +0.1 | 27 | density |
| CO | SCAN (Max_Iter) | +22.778 | +22.822 | 150 | -0.044 | -9.5955 | -9.5976 | +2.1 | 200 | orbital |
| CO | r2SCAN | +19.605 | +19.736 | 150 | -0.131 | -9.5128 | -9.5134 | +0.6 | 38 | orbital |
| CO | TPSS | +1.578 | +1.589 | 150 | -0.011 | -9.2827 | -9.2832 | +0.5 | 43 | density |
| O2 | SCAN | +19.209 | +18.248 | 300 | +0.961 | -7.4972 / -11.6000 | -7.5098 / -11.6164 | +12.6 / +16.4 | 43 | orbital |
| O2 | r2SCAN | +13.155 | +13.471 | 300 | -0.316 | -7.4421 / -11.6894 | -7.4424 / -11.6907 | +0.3 / +1.3 | 28 | orbital |
| O2 | TPSS | +1.765 | +1.786 | 250 | -0.022 | -7.0623 / -11.5646 | -7.0621 / -11.5653 | -0.2 / +0.7 | 30 | density |

r2SCAN and TPSS agree within 0.14 mRy (O2 r2SCAN 0.32 mRy) and their HOMOs within 1.3 meV. SCAN agrees within
0.04 to 0.48 mRy (O2 0.96 mRy) and its HOMOs within 40 meV: SCAN is grid-sensitive in every code (QE's own SCAN
E - E_PBE moves by 0.1 to 0.7 mRy between 150 and 300 Ry, `qe/README.md`).

**Why a meta-GGA differs more than PBE at the same grid: quadrature.** The energy is `h^3 sum_i e(n_i, sigma_i,
tau_i)` over the grid points. `tau` and the iso-orbital indicator `alpha = (tau - tau_W)/tau_unif` vary on shorter
length scales than the density, so the sum carries a quadrature (aliasing) error that changes sign from one grid
spacing to the next; a plane-wave code has the same error on its dense FFT grid. The derivative stencil of `sigma`
and `tau`, the boundary of the domain and the form of the density gradient contribute at most 0.07 mRy: with the
energy density summed on a twice finer grid (the orbitals interpolated spectrally) r2SCAN and TPSS agree with QE to
0.04 mRy and 2.5 meV at h = 0.20, 0.16 and 0.13 bohr (measured with the former quadrature). SCAN's steep switching function in `alpha` makes its error the
largest. `python -m parsec_python.acceleration.benchmarks.meta_gga_accuracy --input <parsec.in> --refine 2` estimates
this error for an input ("Accuracy and grid spacing" in `src/parsec_python/README.md`).

## Grid spacing

r2SCAN at h = 0.13 bohr (`*_h013` inputs, E_PBE at the same h) against QE at 300 Ry:

| molecule | h (bohr) | E_r2SCAN - E_PBE this code (mRy) | QE 300 Ry (mRy) | this code - QE (mRy) | HOMO this code (eV) | dHOMO (meV) | SCF iterations |
|---|---|---|---|---|---|---|---|
| H2O | 0.16 | -2.122 | -2.024 | -0.099 | -7.5682 | +0.6 | 31 |
| H2O | 0.13 | -1.968 | -2.024 | +0.056 | -7.5680 | +0.8 | 50 |
| O2 | 0.16 | +13.155 | +13.471 | -0.316 | -7.4421 / -11.6894 | +0.3 / +1.3 | 28 |
| O2 | 0.13 | +13.595 | +13.471 | +0.124 | -7.4418 / -11.6903 | +0.6 / +0.4 | 38 |

The difference changes sign and shrinks (the quadrature error above). Advice:

- r2SCAN, TPSS and other smooth meta-GGAs: `Grid_Spacing` 0.16 bohr gives E - E_PBE within about 0.1 mRy (O2 0.32)
  and levels within 2 meV; 0.13 bohr for energy differences to 0.05-0.1 mRy with first-row atoms (1.9 times the grid
  points of 0.16, 2.2 to 2.4 times the time here). 0.20 bohr is too coarse for oxygen (O2 r2SCAN: 1 mRy, 14 meV).
- SCAN: 0.13 bohr or finer, and still up to 0.2 mRy and 25 meV; prefer r2SCAN (built to remove this sensitivity)
  unless SCAN itself is required.
- LUMOs of small molecules near 0 eV are diffuse and confined by `Boundary_Sphere_Radius`: not compared here.

## Meta-GGA hybrids against QE

QE at 150 Ry, `ecutfock` 300 Ry; E - E_PBE (mRy) with QE's PBE at 150 Ry; a_x E_x = the exact-exchange energy of the
final orbitals (TPSSh 0.10, r2SCAN0 0.25 of E_x^HF). QE 7.5 gives no O2 TPSSh or r2SCAN0 (`qe/README.md`).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | a_x E_x this code (Ry) | a_x E_x QE (Ry) | difference (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | inner SCF per outer iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | TPSSh | -8.109 | -8.186 | +0.077 | -0.78721597 | -0.78721403 | -0.002 | -8.0321 | -8.0323 | +0.2 | [6, 9, 16] |
| H2O | r2SCAN0 | +11.275 | +11.303 | -0.027 | -1.97348079 | -1.97347758 | -0.003 | -9.2896 | -9.2906 | +1.0 | [11, 11, 14, 7] |
| CH4 | TPSSh | -41.475 | -41.426 | -0.049 | -0.59343237 | -0.59342921 | -0.003 | -10.2561 | -10.2556 | -0.5 | [12, 10, 15, 10] |
| CH4 | r2SCAN0 | -3.794 | -3.745 | -0.049 | -1.48112834 | -1.48112872 | +0.000 | -11.2308 | -11.2306 | -0.2 | [7, 14, 9, 7] |
| NH3 | TPSSh | -16.872 | -16.873 | +0.002 | -0.67180073 | -0.67180198 | +0.001 | -6.8438 | -6.8434 | -0.4 | [7, 15, 11] |
| NH3 | r2SCAN0 | +11.185 | +11.158 | +0.028 | -1.68127018 | -1.68127045 | +0.000 | -7.9245 | -7.9234 | -1.1 | [9, 6, 17, 9] |
| N2 | TPSSh | +18.453 | +18.419 | +0.034 | -0.89752227 | -0.89752649 | +0.004 | -11.2067 | -11.2070 | +0.3 | [6, 6, 9] |
| N2 | r2SCAN0 | +61.922 | +61.921 | +0.000 | -2.24537259 | -2.24536486 | -0.008 | -12.4856 | -12.4856 | +0.0 | [8, 6, 17, 5] |
| CO | TPSSh | +9.787 | +9.795 | -0.008 | -0.93593705 | -0.93593890 | +0.002 | -9.9365 | -9.9371 | +0.6 | [6, 9, 9] |
| CO | r2SCAN0 | +43.972 | +44.056 | -0.084 | -2.34422886 | -2.34421579 | -0.013 | -11.0409 | -11.0414 | +0.5 | [6, 15, 13, 10, 9] |

E - E_PBE agrees within 0.09 mRy, the exact exchange within 0.013 mRy and the HOMOs within 1.1 meV: the meta-GGA part
at the level of the semilocal r2SCAN and TPSS above (QE's own 150 Ry rows of r2SCAN and TPSS are 0.005 to 0.12 mRy
from 300 Ry), the exact exchange at the level of the GGA hybrids of `../0d_hybrid_functionals`. The Fock loops take 3
or 4 outer iterations (CO r2SCAN0 5) to an estimated exchange error below 1e-8 Ry.

## Forces

The distorted water molecule (one O-H bond 1.10 A), forces after the net force is removed (Ry/bohr; y = 0 by
symmetry), against QE at 200 Ry; the last column is QE's own change from 150 to 200 Ry:

| functional | atom | this code (x, z) | QE 200 Ry (x, z) | this code - QE 200 Ry (x, z) | QE 150 - QE 200 Ry (x, z) |
|---|---|---|---|---|---|
| PBE | O | +0.151514, +0.054943 | +0.151003, +0.055011 | +5.1e-04, -6.8e-05 | -5.3e-05, -4.8e-05 |
| PBE | H1 (1.10 A) | -0.122991, -0.086583 | -0.123481, -0.086608 | +4.9e-04, +2.5e-05 | +3.3e-05, +2.7e-05 |
| PBE | H2 | -0.028523, +0.031639 | -0.027522, +0.031597 | -1.0e-03, +4.2e-05 | +2.0e-05, +2.2e-05 |
| r2SCAN | O | +0.149115, +0.071892 | +0.149053, +0.073281 | +6.2e-05, -1.4e-03 | -2.0e-04, -2.5e-04 |
| r2SCAN | H1 (1.10 A) | -0.136135, -0.094422 | -0.136854, -0.095122 | +7.2e-04, +7.0e-04 | +1.2e-04, +1.2e-04 |
| r2SCAN | H2 | -0.012980, +0.022530 | -0.012198, +0.021841 | -7.8e-04, +6.9e-04 | +7.8e-05, +1.3e-04 |

The largest difference is 1.0e-3 Ry/bohr for PBE and 1.4e-3 for r2SCAN at h = 0.16 bohr (at 0.13, with the former
quadrature: 2.6e-4 and 5e-4; the default quadrature moved the forces at 0.16 by up to 7e-5 Ry/bohr). The meta-GGA forces are the derivatives of the grid energy with the force formulas of the GGAs (the orbital
term does not depend on the atoms; finite differences of the energy match them as for PBE), so their difference to
QE is the grid error of the energy, larger for a meta-GGA as above.

## SCAN

- **Residual floor.** The SCAN runs of CH4, NH3 and CO stop at `Max_Iter` 200 ("Maximum SCF iterations reached
  without convergence") with the total energy settled to the printed 1e-8 Ry from iteration 26, 52 and 30 on, while
  the weighted residual of the last 100 iterations stays between 1.8e-6 and 1.8e-4: points of density 1e-5 to 1e-3, where SCAN's switching
  function in `alpha` amplifies the last digits of `tau - tau_W`. r2SCAN, built to remove that sensitivity, converges
  as a GGA does. For SCAN energies, `Convergence_Criterion: 1e-4 Ry` stops these runs after 35 to 53 iterations
  within 1e-8 Ry (CH4, CO) and 1.3e-6 Ry (NH3) of the energies above.
- **Fine grids.** The density-weighted Anderson mixer lets the low-density values of `vtau` and `W` drift; on fine
  grids that can produce states localized on a few points (SCAN H2O at h = 0.13 bohr: passing excursions, residual
  4.6e-5 after 80 iterations). `PARSEC_META_GGA_ANDERSON=plain` converges it (54 iterations), as does r2SCAN.

## Costs

`main.py` on one A100: program wall time (setup included, one process per input) / SCF iterations (hybrids: the PBE
start + the inner iterations of the Fock loop) / peak device memory of the process. The logs of 23 November 2026:
four inputs side by side, one per GPU, in the node's kernel cache after two discarded runs on each GPU. Grid: 1.4 to 1.8 million points
at h = 0.16 bohr (H2O 1,568,208), 3 million at 0.13.

| input | PBE | SCAN | r2SCAN | TPSS | TPSSh | r2SCAN0 |
|---|---|---|---|---|---|---|
| H2O, h 0.16 | 7 s / 29 / 1.2 GiB | 31 s / 46 / 3.0 GiB | 25 s / 31 / 2.7 GiB | 23 s / 38 / 2.5 GiB | 30 s / 19 + 31 / 3.0 GiB | 47 s / 19 + 43 / 3.2 GiB |
| CH4, h 0.16 | 5 s / 37 / 0.9 GiB | 110 s / 200 (Max_Iter) / 3.7 GiB | 28 s / 29 / 3.2 GiB | 23 s / 28 / 3.0 GiB | 33 s / 13 + 47 / 3.5 GiB | 46 s / 13 + 37 / 3.8 GiB |
| NH3, h 0.16 | 16 s / 54 / 2.2 GiB | 145 s / 200 (Max_Iter) / 4.5 GiB | 55 s / 60 / 3.7 GiB | 41 s / 61 / 3.7 GiB | 43 s / 22 + 33 / 4.2 GiB | 68 s / 22 + 41 / 4.4 GiB |
| N2, h 0.16 | 5 s / 24 / 0.8 GiB | 23 s / 42 / 2.4 GiB | 23 s / 39 / 2.3 GiB | 17 s / 27 / 2.1 GiB | 19 s / 14 + 21 / 2.6 GiB | 30 s / 14 + 36 / 2.8 GiB |
| CO, h 0.16 | 6 s / 36 / 1.2 GiB | 92 s / 200 (Max_Iter) / 2.8 GiB | 25 s / 38 / 2.5 GiB | 21 s / 43 / 2.3 GiB | 25 s / 27 + 24 / 2.8 GiB | 41 s / 27 + 53 / 3.0 GiB |
| O2, h 0.16 | 10 s / 27 / 1.1 GiB | 40 s / 43 / 3.5 GiB | 32 s / 28 / 2.5 GiB | 28 s / 30 / 2.3 GiB |  |  |
| H2O, h 0.13 | 17 s / 49 / 1.8 GiB |  | 61 s / 50 / 4.7 GiB |  |  |  |
| O2, h 0.13 | 18 s / 42 / 1.6 GiB |  | 69 s / 38 / 4.2 GiB |  |  |  |
| H2O distorted (forces), h 0.16 | 12 s / 44 / 2.1 GiB |  | 29 s / 27 / 3.5 GiB |  |  |  |

A meta-GGA run takes 2.6 to 4.6 times the time of PBE on the same route here (H2O, NH3; the SCAN runs stopped at
`Max_Iter` more); the PBE runs of CH4, N2 and CO in their weighted sectors (above) took 5 to 6 s, against 6 to 9 s in
the logs of 8 November on the abelian route, which the meta-GGAs keep, and O2 10 and 18 s (h = 0.16 and 0.13 bohr)
against 12 and 23 s. An SCF iteration costs 2 to 3 times as much (`tau` of the orbitals, the orbital term
`-div(vtau grad)` in every Hamiltonian application and the wider Chebyshev interval it brings: H2O 0.18 s per iteration with PBE, 0.35 s with TPSS, 0.46-0.49 s with SCAN and r2SCAN), and the set-up
before the SCF ("Pre-SCF setup/reporting wall time" of the log) takes 8 to 11 s for a meta-GGA against 1.2 to 1.9 s
for PBE (16 to 17 s at h = 0.13 bohr). The meta-GGA hybrids add 2 to 22 s to the semilocal meta-GGA run.
