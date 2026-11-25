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
| H2O | -34.34610161 | -34.34607194 | 300 | -2.97e-05 | -7.2391 | -7.2383 | -0.8 |
| CH4 | -16.15717622 | -16.15718789 | 300 | +1.17e-05 | -9.4649 | -9.4644 | -0.5 |
| NH3 | -23.46694450 | -23.46695167 | 300 | +7.17e-06 | -6.1613 | -6.1604 | -0.9 |
| N2 | -39.83021701 | -39.83021459 | 150 | -2.42e-06 | -10.2963 | -10.2959 | -0.4 |
| CO | -43.31658892 | -43.31651727 | 150 | -7.17e-05 | -9.0619 | -9.0616 | -0.3 |
| O2 | -63.71099604 | -63.71092564 | 300 | -7.04e-05 | -6.8743 / -11.6060 | -6.8741 / -11.6063 | -0.2 / +0.3 |

The PBE inputs equal those of `../0d_hybrid_functionals`, whose section "PBE level" gives QE's converged totals and
the totals of these inputs from h = 0.20 to 0.08 bohr (within 6.6e-5 Ry of QE at every h). The logs are of 24 November
2026, made at the defaults of that day. Against the logs of 23 November 2026 they differ by the Fourier filter of the
local potential and the projectors, a default since 23 November 2026 (`PARSEC_LOCAL_FILTER=all`;
`src/parsec_python/acceleration/README.md`, row `PARSEC_LOCAL_FILTER`): a table sampled at the grid points holds
Fourier components beyond the resolution of the grid, which made the total depend on where the atoms, oxygen above
all, sit relative to the grid points. The filter moves the PBE totals by +1.77e-4 (H2O), -9.5e-6 (CH4), +6.0e-6 (NH3),
+1.60e-4 (N2), +2.04e-4 (CO) and +1.98e-4 Ry (O2), at h = 0.13 bohr by +1.07e-4 (H2O) and +2.89e-4 Ry (O2); with the
tables of the files the differences to QE above were -2.07e-4, +2.11e-5, +1.15e-6, -1.63e-4, -2.76e-4 and -2.69e-4 Ry.
A rigid translation over one grid step now changes the totals by at most 3.2e-5 Ry at h = 0.16 bohr (O2) and 7.4e-6 Ry
at 0.12 (N2) (the same row). What remains, at most 7.2e-5 Ry, is QE's cutoff of 150 Ry for N2 and CO (QE's converged
totals lie 2.8e-5 and 6.6e-5 Ry below those at 150 Ry) and the grid error of oxygen (O2 -6.6e-5 and H2O -2.3e-5 Ry
against QE's converged totals; at h = 0.12 and 0.10 bohr O2 -2.6e-5 and -2.4e-5, H2O -4.7e-6 and -5.2e-6 Ry). In
E - E_PBE of the meta-GGAs and their hybrids below the filter cancels to 0.007 mRy (N2 r2SCAN0).

Since 23 November 2026 the Kleinman-Bylander energies are integrated exactly on the logarithmic mesh, as QE's
Simpson rule does, and the FHI tables are continued below their first sample (`../0d_hybrid_functionals/README.md`,
"Known offsets"). PARSEC's trapezoid (`PARSEC_KB_QUADRATURE=trapezoid`), with which the logs before that date were made,
changes the PBE totals by +1.69e-4 (H2O), -1.91e-5 (CH4), +1.70e-5 (NH3), -4.99e-5 (N2), +7.81e-5 (CO) and +2.63e-4 Ry
(O2), the same at every h from 0.20 to 0.08 bohr (H2O, CO, O2 with the tables of the files) and with the filter as
without it (H2O and O2 at h = 0.16, 0.12 and 0.10 bohr, within 1e-7 Ry). With the tables of the files it hid most of
the placement error of oxygen in these inputs at h = 0.16 bohr; with the filter, where the placement no longer
matters, it puts H2O and O2 that much above QE at every h (1.6e-4 and 2.4e-4 Ry above QE's converged totals at
h = 0.12 and 0.10 bohr). `PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics: the trapezoid, the tables of the files
and the other values of `src/parsec_python/fortran_numerics.py`) with `PARSEC_ORBITAL_GROUP_META_GGA=off` gives the
totals, SCF iterations, Fock-loop counts, HOMOs and forces of the logs of 20 November 2026 to the printed digits, and
so do `PARSEC_LOCAL_FILTER=off`, `PARSEC_KB_QUADRATURE=trapezoid`, `PARSEC_FHI_ORIGIN=off`, `PARSEC_CA_CONTINUITY=off`,
`PARSEC_FERMI_GAP_ROOT=off`, `PARSEC_SPLINE_COULOMB_TAIL=off` and `PARSEC_ORBITAL_GROUP_META_GGA=off` together
(checked on eight inputs: H2O PBE and r2SCAN, CH4 TPSSh, N2 SCAN, O2 TPSS, CO r2SCAN0, O2 PBE at h = 0.13 bohr and the
distorted water with r2SCAN; the umbrella adds a NOTE on the boundary values of its Hartree solve).
`PARSEC_LOCAL_FILTER=off` alone gives the totals of the logs of 23 November 2026 to the printed digits for the
meta-GGAs of CH4, N2, CO and O2, and those of the other inputs, whose logs of that day were made before the exact
Coulomb tail of the spline of the local tables (`PARSEC_SPLINE_COULOMB_TAIL`, a default of the same date), within 3e-8
to 2.1e-7 Ry (all 40 inputs).

CH4, N2, CO and O2 have the point groups Td, D4h, C4v and D4h on the grid, and a PBE run solves the weighted
one-dimensional sectors of a larger subgroup (D2d for Td, a subgroup of order 8 or 4 for D4h and C4v) instead of the
abelian group of the logs of 8 November 2026 (`PARSEC_ORBITAL_GROUP=abelian`): CH4, N2 and CO since 15 and 18
November, the spin-polarized O2 with its `occup.in` since 20 November (25 SCF iterations against 29 on the abelian
route, at h = 0.13 bohr 43 against 38). Their energies are those of the abelian runs to the printed digits, and their
HOMOs within 0.014 meV (O2 at h = 0.13 bohr). Since 23 November 2026 the meta-GGAs of these clusters take the weighted
sectors too (`PARSEC_ORBITAL_GROUP_META_GGA`; `src/parsec_python/acceleration/README.md`, "Meta-GGA functionals on the
device"); their hybrids keep the abelian group. The total energies are those of the abelian route
(`PARSEC_ORBITAL_GROUP_META_GGA=off`) to the printed digits (CO r2SCAN within 1e-8 Ry) and the HOMOs within 0.1 meV
(N2 r2SCAN 0.094 meV), while the SCF iteration counts change either way (CO TPSS 30 against 43, N2 SCAN 47 against 64,
N2 r2SCAN 35 against 25; CH4 and CO SCAN stop at `Max_Iter` on both routes; A100, 24 November 2026).

## Meta-GGAs against QE

E - E_PBE in mRy, each code against its own PBE (QE: at the cutoff of the column, 300 Ry where QE's convergence rows
exist, TPSS of H2O and O2 250 Ry, else 150 Ry; `qe/README.md`); HOMO in eV (O2: up / down). "Max_Iter": the run
stopped at `Max_Iter` 200 with the energy settled (see "SCAN" below).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | QE cutoff (Ry) | this code - QE (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | SCF iterations | gradient form |
|---|---|---|---|---|---|---|---|---|---|---|
| H2O | SCAN | +1.406 | +1.762 | 300 | -0.356 | -7.5586 | -7.5973 | +38.7 | 45 | orbital |
| H2O | r2SCAN | -2.122 | -2.024 | 300 | -0.098 | -7.5684 | -7.5688 | +0.4 | 31 | orbital |
| H2O | TPSS | -11.070 | -11.083 | 250 | +0.013 | -7.3159 | -7.3158 | -0.1 | 30 | density |
| CH4 | SCAN (Max_Iter) | -4.041 | -3.924 | 300 | -0.116 | -9.8256 | -9.8252 | -0.4 | 200 | orbital |
| CH4 | r2SCAN | -1.505 | -1.443 | 300 | -0.061 | -9.8134 | -9.8132 | -0.2 | 26 | orbital |
| CH4 | TPSS | -39.429 | -39.357 | 150 | -0.072 | -9.6571 | -9.6566 | -0.5 | 28 | density |
| NH3 | SCAN (Max_Iter) | +6.226 | +6.516 | 300 | -0.290 | -6.5035 | -6.4947 | -8.8 | 200 | orbital |
| NH3 | r2SCAN | +4.861 | +4.822 | 300 | +0.038 | -6.4626 | -6.4625 | -0.1 | 56 | orbital |
| NH3 | TPSS | -17.645 | -17.640 | 150 | -0.005 | -6.2284 | -6.2282 | -0.2 | 46 | density |
| N2 | SCAN | +38.798 | +39.273 | 150 | -0.474 | -10.7707 | -10.8024 | +31.7 | 47 | orbital |
| N2 | r2SCAN | +35.246 | +35.240 | 150 | +0.006 | -10.7265 | -10.7267 | +0.2 | 35 | orbital |
| N2 | TPSS | +8.429 | +8.392 | 150 | +0.037 | -10.4619 | -10.4621 | +0.2 | 27 | density |
| CO | SCAN (Max_Iter) | +22.778 | +22.822 | 150 | -0.044 | -9.5959 | -9.5976 | +1.7 | 200 | orbital |
| CO | r2SCAN | +19.605 | +19.736 | 150 | -0.131 | -9.5132 | -9.5134 | +0.2 | 33 | orbital |
| CO | TPSS | +1.582 | +1.589 | 150 | -0.007 | -9.2830 | -9.2832 | +0.2 | 30 | density |
| O2 | SCAN | +19.207 | +18.248 | 300 | +0.959 | -7.4976 / -11.6005 | -7.5098 / -11.6164 | +12.2 / +15.9 | 41 | orbital |
| O2 | r2SCAN | +13.151 | +13.471 | 300 | -0.319 | -7.4425 / -11.6899 | -7.4424 / -11.6907 | -0.1 / +0.8 | 26 | orbital |
| O2 | TPSS | +1.767 | +1.786 | 250 | -0.019 | -7.0627 / -11.5650 | -7.0621 / -11.5653 | -0.6 / +0.3 | 34 | density |

r2SCAN and TPSS agree within 0.14 mRy (O2 r2SCAN 0.32 mRy) and their HOMOs within 0.8 meV. SCAN agrees within
0.04 to 0.47 mRy (O2 0.96 mRy) and its HOMOs within 39 meV: SCAN is grid-sensitive in every code (QE's own SCAN
E - E_PBE moves by 0.1 to 0.7 mRy between 150 and 300 Ry, `qe/README.md`).

**Why a meta-GGA differs more than PBE at the same grid: quadrature.** The energy is `h^3 sum_i e(n_i, sigma_i,
tau_i)` over the grid points. `tau` and the iso-orbital indicator `alpha = (tau - tau_W)/tau_unif` vary on shorter
length scales than the density, so the sum carries a quadrature (aliasing) error that changes sign from one grid
spacing to the next; a plane-wave code has the same error on its dense FFT grid. The derivative stencil of `sigma`
and `tau`, the boundary of the domain and the form of the density gradient contribute at most 0.07 mRy: with the
energy density summed on a twice finer grid (the orbitals interpolated spectrally) r2SCAN and TPSS agree with QE to
0.04 mRy and 2.5 meV at h = 0.20, 0.16 and 0.13 bohr (measured with the former quadrature and the tables of the
files). SCAN's steep switching function in `alpha` makes its error the largest.
`python -m parsec_python.acceleration.benchmarks.meta_gga_accuracy --input <parsec.in> --refine 2` estimates
this error for an input ("Accuracy and grid spacing" in `src/parsec_python/README.md`).

## Grid spacing

r2SCAN at h = 0.13 bohr (`*_h013` inputs, E_PBE at the same h) against QE at 300 Ry:

| molecule | h (bohr) | E_r2SCAN - E_PBE this code (mRy) | QE 300 Ry (mRy) | this code - QE (mRy) | HOMO this code (eV) | dHOMO (meV) | SCF iterations |
|---|---|---|---|---|---|---|---|
| H2O | 0.16 | -2.122 | -2.024 | -0.098 | -7.5684 | +0.4 | 31 |
| H2O | 0.13 | -1.966 | -2.024 | +0.057 | -7.5681 | +0.7 | 46 |
| O2 | 0.16 | +13.151 | +13.471 | -0.319 | -7.4425 / -11.6899 | -0.1 / +0.8 | 26 |
| O2 | 0.13 | +13.596 | +13.471 | +0.125 | -7.4422 / -11.6906 | +0.2 / +0.1 | 36 |

The difference changes sign and shrinks (the quadrature error above). Advice:

- r2SCAN, TPSS and other smooth meta-GGAs: `Grid_Spacing` 0.16 bohr gives E - E_PBE within about 0.1 mRy (O2 0.32)
  and levels within 1 meV; 0.13 bohr for energy differences to 0.05-0.1 mRy with first-row atoms (1.9 times the grid
  points of 0.16, 2.2 to 2.3 times the time here). 0.20 bohr is too coarse for oxygen (O2 r2SCAN: 1.1 mRy, 13 meV;
  H2O 0.23 mRy, 1.9 meV).
- SCAN: 0.13 bohr or finer, and still up to 0.2 mRy and 26 meV (H2O and O2 at h = 0.13 bohr: +0.22 and -0.20 mRy,
  HOMOs -13 and -26 meV); prefer r2SCAN (built to remove this sensitivity) unless SCAN itself is required.
- LUMOs of small molecules near 0 eV are diffuse and confined by `Boundary_Sphere_Radius`: not compared here.

## Meta-GGA hybrids against QE

QE at 150 Ry, `ecutfock` 300 Ry; E - E_PBE (mRy) with QE's PBE at 150 Ry; a_x E_x = the exact-exchange energy of the
final orbitals (TPSSh 0.10, r2SCAN0 0.25 of E_x^HF). QE 7.5 gives no O2 TPSSh or r2SCAN0 (`qe/README.md`).

| molecule | functional | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | a_x E_x this code (Ry) | a_x E_x QE (Ry) | difference (mRy) | HOMO this code (eV) | HOMO QE (eV) | dHOMO (meV) | inner SCF per outer iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O | TPSSh | -8.107 | -8.186 | +0.079 | -0.78721461 | -0.78721403 | -0.001 | -8.0324 | -8.0323 | -0.1 | [6, 9, 16] |
| H2O | r2SCAN0 | +11.277 | +11.303 | -0.026 | -1.97347719 | -1.97347758 | +0.000 | -9.2899 | -9.2906 | +0.7 | [11, 11, 14, 7] |
| CH4 | TPSSh | -41.475 | -41.426 | -0.049 | -0.59343233 | -0.59342921 | -0.003 | -10.2562 | -10.2556 | -0.6 | [12, 10, 20, 8] |
| CH4 | r2SCAN0 | -3.794 | -3.745 | -0.049 | -1.48112817 | -1.48112872 | +0.001 | -11.2309 | -11.2306 | -0.3 | [7, 14, 9, 7] |
| NH3 | TPSSh | -16.870 | -16.873 | +0.003 | -0.67180098 | -0.67180198 | +0.001 | -6.8438 | -6.8434 | -0.4 | [7, 15, 19] |
| NH3 | r2SCAN0 | +11.187 | +11.158 | +0.030 | -1.68126858 | -1.68127045 | +0.002 | -7.9245 | -7.9234 | -1.1 | [12, 19, 14, 8] |
| N2 | TPSSh | +18.460 | +18.419 | +0.041 | -0.89751981 | -0.89752649 | +0.007 | -11.2067 | -11.2070 | +0.3 | [6, 6, 9] |
| N2 | r2SCAN0 | +61.929 | +61.921 | +0.008 | -2.24536646 | -2.24536486 | -0.002 | -12.4855 | -12.4856 | +0.1 | [8, 6, 17, 5] |
| CO | TPSSh | +9.791 | +9.795 | -0.004 | -0.93593534 | -0.93593890 | +0.004 | -9.9369 | -9.9371 | +0.2 | [6, 9, 9] |
| CO | r2SCAN0 | +43.973 | +44.056 | -0.083 | -2.34422449 | -2.34421579 | -0.009 | -11.0413 | -11.0414 | +0.1 | [6, 15, 13, 10, 9] |

E - E_PBE agrees within 0.09 mRy, the exact exchange within 0.009 mRy and the HOMOs within 1.1 meV: the meta-GGA part
at the level of the semilocal r2SCAN and TPSS above (QE's own 150 Ry rows of r2SCAN and TPSS are 0.005 to 0.12 mRy
from 300 Ry), the exact exchange at the level of the GGA hybrids of `../0d_hybrid_functionals`. The Fock loops take 3
or 4 outer iterations (CO r2SCAN0 5) to an estimated exchange error below 1e-8 Ry.

## Forces

The distorted water molecule (one O-H bond 1.10 A), forces after the net force is removed (Ry/bohr; y = 0 by
symmetry), against QE at 200 Ry; the last column is QE's own change from 150 to 200 Ry:

| functional | atom | this code (x, z) | QE 200 Ry (x, z) | this code - QE 200 Ry (x, z) | QE 150 - QE 200 Ry (x, z) |
|---|---|---|---|---|---|
| PBE | O | +0.150979, +0.054999 | +0.151003, +0.055011 | -2.4e-05, -1.2e-05 | -5.3e-05, -4.8e-05 |
| PBE | H1 (1.10 A) | -0.123467, -0.086603 | -0.123481, -0.086608 | +1.4e-05, +4.9e-06 | +3.3e-05, +2.7e-05 |
| PBE | H2 | -0.027513, +0.031604 | -0.027522, +0.031597 | +9.3e-06, +6.9e-06 | +2.0e-05, +2.2e-05 |
| r2SCAN | O | +0.148551, +0.071933 | +0.149053, +0.073281 | -5.0e-04, -1.3e-03 | -2.0e-04, -2.5e-04 |
| r2SCAN | H1 (1.10 A) | -0.136599, -0.094436 | -0.136854, -0.095122 | +2.6e-04, +6.9e-04 | +1.2e-04, +1.2e-04 |
| r2SCAN | H2 | -0.011952, +0.022503 | -0.012198, +0.021841 | +2.5e-04, +6.6e-04 | +7.8e-05, +1.3e-04 |

The largest difference is 2.4e-5 Ry/bohr for PBE, within QE's own change from 150 to 200 Ry, and 1.3e-3 for r2SCAN at
h = 0.16 bohr (1.1e-4 and 5.8e-4 at h = 0.13 bohr). With the tables of the files (`PARSEC_LOCAL_FILTER=off`, the logs
before 24 November 2026) the PBE forces at h = 0.16 bohr were up to 1.0e-3 Ry/bohr from QE's (r2SCAN 1.4e-3), the
force of the egg-box term that the Fourier filter removes (PBE section above). The meta-GGA forces are the derivatives of the
grid energy with the force formulas of the GGAs (the orbital term does not depend on the atoms; finite differences of
the energy match them as for PBE), so their difference to QE is the grid error of the energy, larger for a meta-GGA as
above.

## SCAN

- **Residual floor.** The SCAN runs of CH4, NH3 and CO stop at `Max_Iter` 200 ("Maximum SCF iterations reached
  without convergence") with the total energy settled to the printed 1e-8 Ry from iteration 24, 51 and 28 on, while
  the weighted residual of the last 100 iterations stays between 2.3e-6 and 1.5e-4: points of density 1e-5 to 1e-3,
  where SCAN's switching function in `alpha` amplifies the last digits of `tau - tau_W`. r2SCAN, built to remove that
  sensitivity, converges as a GGA does. For SCAN energies, `Convergence_Criterion: 1e-4 Ry` stops these runs after 41,
  35 and 61 iterations with the energies above within 1e-8 Ry (CH4, CO) and 1.3e-6 Ry (NH3).
- **Fine grids.** The density-weighted Anderson mixer lets the low-density values of `vtau` and `W` drift; on fine
  grids that can produce states localized on a few points (SCAN H2O at h = 0.13 bohr: two passing excursions of the
  weighted residual, to 3.2 and 3.6 at iterations 16 and 18, and 86 iterations to convergence).
  `PARSEC_META_GGA_ANDERSON=plain` converges it in 48 iterations to the same energy; r2SCAN converges with either
  mixer (46 iterations with the default).

## Costs

`main.py` on one A100: program wall time (setup included, one process per input) / SCF iterations (hybrids: the PBE
start + the inner iterations of the Fock loop) / peak device memory of the process. The logs of 24 November 2026:
four inputs side by side, one per GPU, in the node's kernel cache after three discarded runs on each GPU (H2O r2SCAN0,
CH4 r2SCAN and O2 SCAN at h = 0.30 bohr). Grid: 1.4 to 1.8 million points at h = 0.16 bohr (H2O 1,568,208), 3 million
at 0.13. The abelian route of CH4, N2, CO and O2 (`PARSEC_ORBITAL_GROUP=abelian` for PBE,
`PARSEC_ORBITAL_GROUP_META_GGA=off` for the meta-GGAs) and the 40 inputs with the tables of the files
(`PARSEC_LOCAL_FILTER=off`) were timed in the same way on the same day.

| input | PBE | SCAN | r2SCAN | TPSS | TPSSh | r2SCAN0 |
|---|---|---|---|---|---|---|
| H2O, h 0.16 | 7 s / 33 / 1.2 GiB | 30 s / 45 / 3.0 GiB | 25 s / 31 / 2.7 GiB | 21 s / 30 / 2.5 GiB | 30 s / 19 + 31 / 3.0 GiB | 47 s / 19 + 43 / 3.3 GiB |
| CH4, h 0.16 | 5 s / 37 / 0.9 GiB | 54 s / 200 (Max_Iter) / 1.9 GiB | 19 s / 26 / 1.9 GiB | 17 s / 28 / 1.7 GiB | 34 s / 13 + 50 / 3.5 GiB | 46 s / 13 + 37 / 3.8 GiB |
| NH3, h 0.16 | 19 s / 64 / 2.2 GiB | 145 s / 200 (Max_Iter) / 4.5 GiB | 51 s / 56 / 3.7 GiB | 34 s / 46 / 3.7 GiB | 48 s / 38 + 41 / 4.5 GiB | 74 s / 38 + 53 / 4.6 GiB |
| N2, h 0.16 | 5 s / 24 / 0.8 GiB | 22 s / 47 / 1.9 GiB | 20 s / 35 / 1.9 GiB | 16 s / 27 / 1.7 GiB | 20 s / 14 + 21 / 2.6 GiB | 29 s / 14 + 36 / 2.8 GiB |
| CO, h 0.16 | 6 s / 35 / 1.2 GiB | 77 s / 200 (Max_Iter) / 2.1 GiB | 22 s / 33 / 2.1 GiB | 17 s / 30 / 2.0 GiB | 26 s / 27 + 24 / 2.8 GiB | 41 s / 27 + 53 / 3.0 GiB |
| O2, h 0.16 | 10 s / 25 / 1.1 GiB | 36 s / 41 / 3.2 GiB | 28 s / 26 / 2.1 GiB | 26 s / 34 / 2.0 GiB |  |  |
| H2O, h 0.13 | 18 s / 49 / 1.8 GiB |  | 58 s / 46 / 4.7 GiB |  |  |  |
| O2, h 0.13 | 19 s / 43 / 1.6 GiB |  | 62 s / 36 / 3.7 GiB |  |  |  |
| H2O distorted (forces), h 0.16 | 12 s / 44 / 2.1 GiB |  | 29 s / 27 / 3.5 GiB |  |  |  |

A meta-GGA run takes 1.8 to 4.6 times the time of PBE on the same route here (the SCAN runs that stop at `Max_Iter`
more); the PBE runs of CH4, N2, CO and O2 in their weighted sectors (above) took 4.9 to 9.8 s against 5.2 to 11.1 s on
the abelian route (CH4 8.4 -> 5.4 s), and O2 at h = 0.13 bohr 18.9 against 20.3 s. The meta-GGAs gain more from the
weighted sectors: CH4 (Td) SCAN 109 -> 54 s, r2SCAN 27 -> 19 s and TPSS 23 -> 17 s against the abelian route, GPU
peak 3.0-3.6 -> 1.7-1.9 GiB, an SCF iteration 45 to 56 % cheaper; N2, CO and O2 (D4h and C4v, whose subgroups of 8
and 4 have the order of their abelian groups) save 7 to 24 % per iteration and 0.3 to 0.6 GiB, with whole runs from
23 % shorter (N2 SCAN 29 -> 22 s, 47 iterations against 64) to 13 % longer (CO r2SCAN 20 -> 22 s, 33 iterations
against 25) as the iteration counts change. An SCF iteration (SCF wall time over iterations) costs 1.5 to 3.3 times as
much as with PBE on the same route (`tau` of the orbitals, the orbital term `-div(vtau grad)` in every Hamiltonian
application and the wider Chebyshev interval it brings: H2O 0.18 s per iteration with PBE, 0.37 s with TPSS,
0.46-0.49 s with SCAN and r2SCAN), and the set-up before the SCF ("Pre-SCF setup/reporting wall time" of the log)
takes 8 to 11 s for a meta-GGA against 1.1 to 1.7 s for PBE (16 to 17 s at h = 0.13 bohr). The meta-GGA hybrids,
which keep the abelian group, add 4 to 23 s to the semilocal meta-GGA run on the abelian route and 3 to 27 s to the
run of the default route. The Fourier filter costs no measurable time here: with the tables of the files the 40
inputs took 1297 s together against 1306 s, and an SCF iteration of the default is 3 % dearer in the median, within
the spread of single runs (-22 to +16 %).
