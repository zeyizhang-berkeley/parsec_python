# Self-consistent hybrid functionals of periodic cells: Si, C, MgO, an h-BN layer and O2 in a box against Quantum ESPRESSO

PBE0, HSE06, Hartree-Fock and B3LYP, self-consistent (generalized Kohn-Sham with adaptively compressed exchange, ACE),
for bulk crystals with k-point meshes (Si 2x2x2 and 4x4x4, C and MgO 3x3x3), a Si cell with a displaced atom (forces),
a two-dimensional h-BN layer (6x6x1) and triplet O2 in a periodic box (spin-polarized, Gamma point), each on one
NVIDIA A100, compared with Quantum ESPRESSO 7.5 (+ libxc 7.0.0) using the same FHI98PP pseudopotentials, geometries
and Gamma-centred k-point meshes. The quantity compared is E_hyb - E_PBE of each code at its own discretization, where
both codes' PBE-level conventions cancel to first order; Fock energies, gaps and forces are compared directly. A
64-atom Si cell (Gamma point on one GPU, a 2x2x2 mesh on four) gives the cost of a larger cell.

## Contents

- `<system>/<input>/parsec.in` and `parsec_python.out`, the log of that input on one A100 (four for `Si64_k222`;
  paths shortened to repository paths):

  | system | inputs |
  |---|---|
  | `Si_k222`, `Si_k444`: diamond Si, primitive fcc cell, a = 10.26 bohr, h = 0.25 bohr | `pbe`, `pbe0`, `hse06`, `hse06_default`, `hf`; `b3lyp` (2x2x2) |
  | `Si_disp_k222`: the same cell with atom 2 moved by +0.05 bohr along x (forces) | `pbe`, `pbe0`, `hse06` |
  | `C_k333`: diamond, a = 6.74 bohr, h = 0.14 bohr | `pbe`, `pbe0`, `hse06` |
  | `MgO_k333`: rocksalt, a = 7.96 bohr, h = 0.14 bohr (Mg with its core correction) | `pbe`, `pbe0`, `hse06` |
  | `hBN_k661`: h-BN monolayer, a = 4.73 bohr, slab of half height 15 bohr, h = 0.15 bohr (B with its core correction) | `pbe`, `pbe0`, `hse06` (defaults: padded slab kernel), `pbe0_periodic`, `hse06_periodic` (Quantum ESPRESSO's convention) |
  | `O2_box`: O2 (1.2075 A) in a periodic cube of 12 bohr, Gamma point, `occup.in` with 7 up and 5 down electrons, h = 0.16 bohr | `pbe`, `pbe0`, `hse06` |
  | `Si64_gamma`, `Si64_k222`: 64-atom Si cell (2x2x2 conventional cells), h = 0.30 bohr, 140 states | `pbe`, `pbe0` (timing) |

- `Si_FHIPP.DAT`, `C_FHIPP.DAT`, `Mg_FHIPP.DAT`, `O_FHIPP.DAT`, `B_FHIPP.DAT`, `N_FHIPP.DAT`: copies of
  `src/pp_generation/NCPPs/PBE/<E>/FHI/` (Troullier-Martins, FHI98PP, PBE). The inputs find them with `--pp-dir`.
- `run_examples.py`: runs every input (or those named) on the GPUs of one node, one process per input;
  `collect_results.py`: the tables below from the logs.
- `qe/`: the Quantum ESPRESSO inputs, their settings, the UPF note and the QE numbers (`qe/README.md`).

## Run

One input, from the repository root:

```powershell
python src\parsec_python\main.py examples\3d_hybrid_functionals\Si_k222\pbe0\parsec.in `
  --pp-dir examples\3d_hybrid_functionals --no-archive --gpus 1
```

`parsec.out` is written beside the input; compare it with `parsec_python.out`. Every input of this example, one after
another on one node (about 1 h 3 min of program time on one A100, 39 min of it the two O2 hybrids), and the tables:

```powershell
python examples\3d_hybrid_functionals\run_examples.py            # or: run_examples.py Si_k222 hBN_k661/pbe0
python examples\3d_hybrid_functionals\collect_results.py --logs examples\3d_hybrid_functionals --runs examples\3d_hybrid_functionals\runs.jsonl
```

`collect_results.py` without arguments reads the reference logs. On the GPU path a periodic hybrid takes one of
three routes; the first two give the numbers of the host loop to round-off
(`src/parsec_python/acceleration/README.md`, "Hybrid functionals of periodic cells on the device"):

- **k-points** (Si, C, MgO, h-BN, `Si64_k222`): `main.py` hands the input to the reference solver with a NOTE; its
  k-point eigensolvers run on the GPUs (one worker per GPU), and the pair densities, their cuFFT solves with
  `K(q + G)`, ACE of every k-point and the exchange term of the Bloch Hamiltonians run there too.
- **Gamma point without spin** (`Si64_gamma`): the accelerated driver's device Fock loop of isolated systems with a
  real-to-complex cuFFT pair solver in place of the Kronecker solver. Since 19 November 2026 a large enough cell
  (grid points x states at least 2^24, `PARSEC_GAMMA_HYBRID_SECTORS=auto`) solves its orbitals in the real sectors of
  its abelian group with the half-cell translations (32 for `Si64_gamma`), and its pair densities in the classes of
  the seven half-cell translations, on the 34^3 box of each class (`PARSEC_CUPY_EXCHANGE_TRANSLATIONS=auto`; `off`
  solves them on the full grid).
- **Gamma point with spin and the fixed occupations of `occup.in`** (`O2_box`): the Fock loop on real orbitals of the
  reference solver, on the host; this route has no device path for two spin channels of a periodic cell (a
  spin-polarized Gamma point without `occup.in` takes the 1x1x1 mesh of the k-point route on the GPUs instead).

## Settings of the inputs

Grid spacings of the matching QE comparison (Si 0.25, C and MgO 0.14, h-BN 0.15, O2 0.16 bohr), finite-difference
order 12, `Convergence_Criterion 1e-6 Ry`, `Cubic_Spline: .true.` for every species (as in
`examples/0d_hybrid_functionals`), `Exx_Tolerance 1e-8 Ry` (default 1e-6 Ry; tight for the comparison),
`Fermi_Temp 1 K` (O2: `-1 K` with `occup.in`), `States_Num` = occupied + 4 (O2: 11 per channel; Si64: 140). The
k-points are QE's `K_POINTS automatic n n n 0 0 0` written out as a `Kpoint_List` of every point of the mesh with
equal weights; a hybrid reduces the list by time reversal only (Si 2x2x2: 8 points, 4x4x4: 36, 3x3x3: 14, h-BN
6x6x1: 20), and its exchange sums over the whole mesh, as QE's with `nqx = n`. (`Monkhorst_Pack_Grid n n n`, with
or without a `Monkhorst_Pack_Shift` block of zeros, gives the same points, but it also turns on the space group of
PARSEC's k-point machinery, which the displaced Si cell does not admit on its grid; the lists keep every input
alike.) The Si64 cells are timing cases: `Si64_k222` takes `Monkhorst_Pack_Grid 2 2 2` without a shift block,
which is PARSEC 1.5's Gamma-centred mesh since 18 November 2026 (0 and 1/2: the irreducible wedge of 4 points for
PBE; PBE0 runs its PBE start on all 8 points, each its own time-reversal partner, and, since that start fills every state to 0 or 1, its Fock loop on the same wedge, the other points the images of the wedge's Bloch factors in the exchange, as a NOTE after the start says (`PARSEC_HYBRID_KPOINT_WEDGE=auto` of 20 November 2026; `off`: all 8 points throughout, the route of its log of 19 November 2026).
`PARSEC_MP_UNSHIFTED_MESH=port` keeps this code's former mesh of such an input (points at +-1/4 of the reciprocal
vectors: 8 points for PBE, 4 time-reversal pairs for PBE0). Everything else is the default: PBE start of the Fock loop, ACE of the
`States_Num` states of every k-point, the adaptive inner criterion, outer DIIS, HSE06 with omega = 0.11 1/bohr in
both parts. The Fock-loop keywords that differ from input to input are those of the Coulomb singularity:

| input | `Exx_Divergence` | `Exx_Auxiliary_Exponent` (bohr^2) | `Exx_Slab_Coulomb` | QE row |
|---|---|---|---|---|
| `pbe0`, `hf`, `b3lyp` | default (`auto`) | default | default (`padded`, h-BN) | `gbx` |
| `hse06` (bulk, O2) | `gygi-baldereschi` | QE's 10/ecutwfc | | `gb` |
| `hse06_default`, `hse06` of h-BN | default (`auto`) | default | default (`padded`) | `gb` |
| `pbe0_periodic` (h-BN) | `gygi-baldereschi-x` | 0.1 | `periodic` | `gbx` |
| `hse06_periodic` (h-BN) | `gygi-baldereschi` | 0.1 | `periodic` | `gb` |

## The Coulomb singularity: this code's defaults and QE's convention

The exchange of a periodic cell sums `K(q + G)` over the q-points of the mesh (`q = k - k'`) and the reciprocal
lattice; for the `1/r` kernel `K = 8 pi/|q + G|^2` (Ry) diverges at `q + G = 0`. Both codes treat the element at
`q + G = 0` by the method of Gygi and Baldereschi: an auxiliary function with the same singularity, summed on the mesh
and integrated analytically, gives the element that makes the sum over the mesh equal the Brillouin-zone integral
for the auxiliary function. QE's `x_gamma_extrapolation = .true.` (its default, rows `gbx`) adds the extrapolation of
Nguyen and de Gironcoli: weights 8/7 on the mesh and 0 on the points of the coarser double mesh, which removes the
`1/N_k` term of the error. QE's auxiliary function `exp(-a |q+G|^2)/|q+G|^2` has the exponent `a = 10/ecutwfc` bohr^2.

This code (`Exx_Divergence`, `Hartree/exchange_kernel.py`) computes the same element in closed form (lattice sums by
Ewald's identity) and reproduces QE's to 1e-15 for every QE row of this comparison (a unit test of the kernel). Its
default `auto` is:

- **the `1/r` part with the extrapolation**, QE's default `gbx`: the PBE0, HF and B3LYP inputs at the defaults are
  like-for-like with QE's default rows. The element of the extrapolated `1/r` kernel does not depend on the
  exponent: Si 2x2x2 PBE0 with `Exx_Divergence: gygi-baldereschi-x` and QE's exponent 1/6 gives the total energy of
  the default to 1e-8 Ry (both -15.59135195 Ry).
- **the `erfc` part (HSE) without it**, QE's `gb`, with the exponent in the limit `a -> 0` instead of QE's
  `10/ecutwfc`. The short-range kernel has no divergence (`K -> 2 pi/omega^2` Ry bohr^2 as `q + G -> 0`); the auxiliary
  function only improves the sampling of the smooth integrand, and the extrapolation makes that worse. On Gaussian
  model pair densities in the Si cell at a 4x4x4 mesh (the validation of the kernel) the `1/r` element errs by
  4.8e-4 Ry without the extrapolation and by 2e-8 with it, the `erfc` element by 1.7e-6 without it and by 4.4e-4 with
  it. The exponent of the `erfc` element matters on coarse meshes, where the element carries a k-point error of the
  whole sum: the `hse06` inputs of the bulk cells and of O2 pass QE's `10/ecutwfc` (like-for-like with `gb`), and
  the `hse06_default` inputs show what the default gives: with the exponent a -> 0, E_hyb - E_PBE of Si lies 9.9e-4 Ry below the value with QE's exponent at 2x2x2 (gap +3.4 meV) and 4.6e-7 Ry below it at 4x4x4 (gap the same to 0.1 meV). The two elements are two quadratures of the same Brillouin-zone integral; they agree once the mesh resolves the short-range kernel.

QE's `gb` rows of the `1/r` kernel (not compared here) converge with the mesh like `1/N_k` (Si: `E_x(gb) - E_x(gbx)`
= 0.32, 0.067 and 0.026 Ry at 1, 2 and 3 points per direction); its `gbx` rows and this code's default converge faster.

**Slabs.** QE 7.5's `assume_isolated = '2D'` cuts the Hartree, local and Ewald terms along z but not the exact
exchange: its Fock term is that of the stack of layers 30 bohr apart, with the three-dimensional element. This code's
default `Exx_Slab_Coulomb: padded` zero-pads the pair densities to twice the height of the slab box (here 404 planes
of 0.15 bohr) and solves them with the bulk kernel of the padded box and the `auto` element: the images along z sit
beyond the density (2e-6 to 9e-6 Ry on model densities), and the in-plane mesh converges as fast as in a bulk cell.
With the extrapolation of the `1/r` part the height of the box drops out: QE's `gbx` row of h-BN (PBE0, 3x3x1) changes
by 5.5e-6 Ry from a cell height of 20 to 30 bohr and by 6e-7 Ry from 30 to 60 bohr, while its `gb` row drifts like
`1/c` (7.4e-4 Ry from 30 to 40 bohr) toward the truncated value; the short-range HSE exchange hardly sees the vacuum
(QE `gb`: 6e-7 Ry from 30 to 40 bohr). So the default `pbe0` of h-BN is compared with QE's `gbx` row and `hse06` with
its `gb` row; the `*_periodic` inputs take QE's convention itself (the bulk kernel on the unpadded slab box of 202
planes, 30.3 bohr against QE's 30.0, with QE's treatment and exponent) and give the defaults' numbers to
1.7e-6 Ry (PBE0) and 9e-8 Ry (HSE06) in E_hyb - E_PBE and 0.2 meV in the gaps. `Exx_Slab_Coulomb: truncated` (the Coulomb kernel cut along z, not run here) is exact for an isolated
layer but converges only as `N_k^-1.5` with the in-plane mesh (a cusp at `Q_par = 0` that no element removes): for
h-BN PBE0 its E_hyb - E_PBE lies 1.6e-3 Ry below the padded value at 3x3x1 and 2.4e-4 Ry at 6x6x1.

## PBE level

| system | E_PBE this code (Ry) | E_PBE QE (Ry) | difference (Ry) | gap this code (eV) | gap QE (eV) |
|---|---|---|---|---|---|
| Si_k222 | -15.57493931 | -15.57477255 | -1.7e-04 | 0.6102 | 0.6105 |
| Si_k444 | -15.74873691 | -15.74854796 | -1.9e-04 | 0.7003 | 0.7006 |
| Si_disp_k222 | -15.57440787 | -15.57432061 | -8.7e-05 | 0.4109 | 0.4113 |
| C_k333 | -22.75885110 | -22.75901823 | +1.7e-04 | 4.1753 | 4.1749 |
| MgO_k333 | -35.13822874 | -35.13885358 | +6.2e-04 | 4.6617 | 4.6579 |
| hBN_k661 | -25.80785832 | -25.80737855 | -4.8e-04 | 4.6983 | 4.6973 |
| O2_box | -63.71114036 | -63.71072842 | -4.1e-04 | 2.4874 | 2.4864 |

QE: Si 60, C 120, MgO 150, h-BN 100, O2 150 Ry. The logs (23 November 2026) integrate the Kleinman-Bylander energies
E_l exactly on the logarithmic mesh, which is QE's Simpson rule to 1e-12 relative, so the differences hold QE's cutoff
error of the totals (Si -1.4e-4, C -5.5e-5, MgO -1.0e-4, h-BN -5.3e-4 Ry per cell against QE's converged total) and
this code's grid (oxygen converges slowest: MgO about +5e-4 Ry at h 0.14; the error depends on where the atoms sit
relative to the grid points, which a crystal fixes, so it does not average out, `../0d_hybrid_functionals/README.md`,
"PBE level"); Si differs from QE by QE's cutoff error.
PARSEC's trapezoid on the logarithmic mesh, with which the logs before that date were made (`qe/README.md`;
`PARSEC_KB_QUADRATURE=trapezoid`, with `PARSEC_FHI_ORIGIN=off` the totals of those logs to the printed digits, as
`PARSEC_FORTRAN_NUMERICS=on` gives them for every cell but h-BN, -1.4e-5 Ry there), changes the totals of the `pbe`
inputs by -3.34e-4 (Si_k222, Si_disp_k222), -3.40e-4 (Si_k444), -3.3e-5 (C_k333), +1.41e-4 (MgO_k333), -2.2e-6
(hBN_k661) and +2.63e-4 Ry (O2_box); the differences were -5.0e-4, -5.3e-4, -4.2e-4, +1.3e-4, +7.7e-4, -4.8e-4 and
-1.5e-4 Ry with it. QE with the same E_l (UPF files with PARSEC's E_l) moved by -3.6e-4 Ry for Si on the 2x2x2 mesh and
-1e-6 Ry for h-BN on a 3x3x1 mesh, within 7 % and 3e-6 Ry of this code's change. Its controls of C (+1.2e-4) and MgO
(+2.9e-4 Ry per cell) were made at the Gamma point alone, not on the 3x3x3 mesh of the comparison: the change is first
order in E_l, weighted by the projections of the occupied states on the projectors, and these depend on the k-points.
At the Gamma point alone this code's trapezoid changes Si, C and MgO by -3.1e-4, +1.3e-4 and +2.8e-4 Ry (QE -3.3e-4,
+1.2e-4 and +2.9e-4 Ry, within 9 %), on the 3x3x3 mesh C and MgO by -3.3e-5 and +1.41e-4 Ry (the same inputs with the
k-point list cut to the Gamma point, A100, 23 November 2026). The gaps agree within 0.4 meV for Si and C, 1.0 meV for
h-BN and O2 and 3.8 meV for MgO (1.8 meV with the trapezoid, which lowers the MgO gap by 2.0 meV).

## Hybrids against QE

dE = E_hyb - E_PBE (Ry), each code at its own discretization (this code: the `pbe` input of the same system); ddE =
this code - QE; "ddE (QE moved)": against the same QE inputs with every atom moved off QE's FFT grid (next section);
Fock = a_x E_x^HF of the final orbitals (Ry); gap = conduction band minimum - valence band maximum over the points of
the mesh (eV; O2: lowest empty minus highest occupied level of both channels).

| system | input | q+G=0 / slab | dE this code | dE QE | ddE | ddE (QE moved) | Fock this code | Fock QE | difference | gap this code | gap QE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Si_k222 | pbe0 | default (= QE gbx) | -0.01641264 | -0.01641613 | +3.5e-06 | +3.2e-06 | -1.09610450 | -1.09610497 | +4.7e-07 | 2.2890 | 2.2893 |
| Si_k222 | hse06 | gb, a = 10/ecutwfc | -0.02017563 | -0.02018365 | +8.0e-06 | +5.5e-06 | -0.86447419 | -0.86447055 | -3.6e-06 | 1.6229 | 1.6231 |
| Si_k222 | hse06_default | default (a -> 0) | -0.02116980 | -0.02018365 | -9.9e-04 | -9.9e-04 | -0.86546836 | -0.86447055 | -1.0e-03 | 1.6263 | 1.6231 |
| Si_k222 | hf | default (= QE gbx) | +0.50036207 | +0.50037506 | -1.3e-05 |  | -4.46161527 | -4.46160434 | -1.1e-05 | 7.9361 | 7.9362 |
| Si_k222 | b3lyp | default (= QE gbx) | +0.03822767 | +0.03826345 | -3.6e-05 |  | -0.88258201 | -0.88257822 | -3.8e-06 | 2.2126 | 2.2131 |
| Si_k444 | pbe0 | default (= QE gbx) | -0.01057606 | -0.01058443 | +8.4e-06 | +8.3e-06 | -1.07070730 | -1.07070707 | -2.3e-07 | 1.9814 | 1.9817 |
| Si_k444 | hse06 | gb, a = 10/ecutwfc | -0.00743248 | -0.00744763 | +1.5e-05 | +1.1e-05 | -0.83294076 | -0.83294089 | +1.3e-07 | 1.3271 | 1.3274 |
| Si_k444 | hse06_default | default (a -> 0) | -0.00743294 | -0.00744763 | +1.5e-05 | +1.1e-05 | -0.83294122 | -0.83294089 | -3.3e-07 | 1.3271 | 1.3274 |
| Si_k444 | hf | default (= QE gbx) | +0.53144051 | +0.53144092 | -4.1e-07 |  | -4.36743650 | -4.36746762 | +3.1e-05 | 6.7855 | 6.7861 |
| Si_disp_k222 | pbe0 | default (= QE gbx) | -0.01639032 | -0.01639308 | +2.8e-06 |  | -1.09607306 | -1.09607990 | +6.8e-06 | 2.0796 | 2.0801 |
| Si_disp_k222 | hse06 | gb, a = 10/ecutwfc | -0.02015524 | -0.02016056 | +5.3e-06 |  | -0.86444547 | -0.86444827 | +2.8e-06 | 1.4145 | 1.4149 |
| C_k333 | pbe0 | default (= QE gbx) | -0.00224134 | -0.00224824 | +6.9e-06 | +5.2e-06 | -1.60903011 | -1.60903339 | +3.3e-06 | 6.2156 | 6.2152 |
| C_k333 | hse06 | gb, a = 10/ecutwfc | -0.00440271 | -0.00443196 | +2.9e-05 | +7.2e-06 | -1.36981048 | -1.36981661 | +6.1e-06 | 5.5515 | 5.5510 |
| MgO_k333 | pbe0 | default (= QE gbx) | +0.27007272 | +0.27005382 | +1.9e-05 | +1.9e-05 | -1.89795905 | -1.89795571 | -3.3e-06 | 7.2158 | 7.2118 |
| MgO_k333 | hse06 | gb, a = 10/ecutwfc | +0.22883771 | +0.22876483 | +7.3e-05 | +2.0e-05 | -1.65476674 | -1.65476926 | +2.5e-06 | 6.5023 | 6.4986 |
| hBN_k661 | pbe0 | default: padded slab | +0.01060131 | +0.01050458 | +9.7e-05 | +9.8e-05 | -1.68614650 | -1.68611913 | -2.7e-05 | 6.3735 | 6.3753 |
| hBN_k661 | pbe0_periodic | periodic, gbx, a = 10/ecutwfc | +0.01060296 | +0.01050458 | +9.8e-05 | +9.9e-05 | -1.68614545 | -1.68611913 | -2.6e-05 | 6.3733 | 6.3753 |
| hBN_k661 | hse06 | default: padded slab | +0.01121085 | +0.01110674 | +1.0e-04 | +9.6e-05 | -1.44208592 | -1.44206026 | -2.6e-05 | 5.7016 | 5.7089 |
| hBN_k661 | hse06_periodic | periodic, gb, a = 10/ecutwfc | +0.01121094 | +0.01110674 | +1.0e-04 | +9.6e-05 | -1.44208581 | -1.44206026 | -2.6e-05 | 5.7016 | 5.7089 |
| O2_box | pbe0 | default (= QE gbx) | +0.04050163 | +0.04044334 | +5.8e-05 | +4.6e-05 | -3.16300446 | -3.16297699 | -2.7e-05 | 6.1976 | 6.1969 |
| O2_box | hse06 | gb, a = 10/ecutwfc | +0.03747237 | +0.03711984 | +3.5e-04 | +3.7e-05 | -2.80022005 | -2.80020128 | -1.9e-05 | 5.4192 | 5.4187 |

- **Si and C** (meshes 2x2x2 to 4x4x4; PBE0, HSE06, HF, B3LYP): ddE within 8.4e-6 Ry for PBE0, 1.3e-5 for HF and 1.5e-5 for HSE06 of Si; C's HSE06 lies +2.9e-5 Ry from QE's reference row and +7.2e-6 from the QE run with the atoms off its grid, and every HSE06 row of Si and C is within 1.1e-5 Ry of those runs. Fock energies within 6.8e-6 Ry (HF 3.1e-5, four times the exact exchange of PBE0); gaps within 0.6 meV. B3LYP lies 3.6e-5 Ry below QE with its Fock energy within 4e-6 Ry: the difference is in the semilocal B88, LYP and VWN-RPA part (libxc 402 in QE), not in the exchange.
- **MgO** (3x3x3): PBE0 +1.9e-5 Ry, HSE06 +7.3e-5 Ry against the reference and +2.0e-5 Ry against QE with the atoms moved (QE's cutoff error of E_hyb - E_PBE: 1.2e-5 Ry); Fock -3.3e-6 and +2.5e-6 Ry; the gaps 4.0 and 3.7 meV above QE's, as at the PBE level (oxygen at h 0.14).
- **h-BN** (6x6x1): +9.7e-5 to +1.0e-4 Ry against QE at 100 Ry, which is QE's own cutoff error of this difference (9.7e-5 Ry). Against QE at 160 Ry (`qe/*_e160.in`) the default PBE0 is +1.2e-6 Ry from QE's `gbx` row (QE's `gb` change from 100 to 160 Ry added to it) and the default HSE06 +9.3e-6 Ry from its `gb` row. Fock energies within 2.7e-5 Ry. The gaps (PBE0 -1.8 meV, HSE06 -7.3 meV) end at the conduction band minimum at Gamma, the nearly-free-electron state of the layer, which reaches into the vacuum where the two codes' cells differ.
- **O2** (Gamma, spin, `occup.in`): PBE0 +5.8e-5 Ry (QE's cutoff error 3.1e-5 Ry; +4.6e-5 against QE with the molecule moved), HSE06 +3.5e-4 Ry against the reference but +3.7e-5 Ry against the moved QE (next section); Fock within 2.7e-5 Ry; gaps within 0.7 meV.
- **`hse06_default`**: the default exponent of the HSE element, above.

## QE's HSE energies and its grid

QE evaluates the semilocal part of HSE (the short-range PBE exchange that HSE removes) inside its gradient-correction
loop, which skips the grid points where the density or its gradient falls below a threshold. For HSE that loop holds
the whole short-range exchange term, its local-density part included, so every point of QE's FFT grid where the
gradient of the density vanishes by symmetry (an atom site, a bond centre, the O2 bond midpoint) loses
`-0.25 e_x^SR(n) dV`, and QE's HSE energy comes out too low by that amount; PBE and PBE0 lose only a gradient
correction that vanishes at such a point. In the reference inputs both atoms of Si and of C, the Mg of MgO and the B
of h-BN are points of QE's grid (the origin, and a quarter of the 36 points along each vector of Si and C), and so is
the O2 bond midpoint. The `qe/*_moved.in` inputs move every atom by a quarter of a grid step along a1 (O2: half a
step along x, y and z), so that no such point is on the grid:

| system | input | dE QE | dE QE moved | change | dE this code | ddE | ddE (moved) |
|---|---|---|---|---|---|---|---|
| Si_k222 | pbe0 | -0.01641613 | -0.01641586 | +2.7e-07 | -0.01641264 | +3.5e-06 | +3.2e-06 |
| Si_k222 | hse06 | -0.02018365 | -0.02018111 | +2.5e-06 | -0.02017563 | +8.0e-06 | +5.5e-06 |
| Si_k444 | pbe0 | -0.01058443 | -0.01058437 | +6.0e-08 | -0.01057606 | +8.4e-06 | +8.3e-06 |
| Si_k444 | hse06 | -0.00744763 | -0.00744394 | +3.7e-06 | -0.00743248 | +1.5e-05 | +1.1e-05 |
| C_k333 | pbe0 | -0.00224824 | -0.00224656 | +1.7e-06 | -0.00224134 | +6.9e-06 | +5.2e-06 |
| C_k333 | hse06 | -0.00443196 | -0.00440995 | +2.2e-05 | -0.00440271 | +2.9e-05 | +7.2e-06 |
| MgO_k333 | pbe0 | +0.27005382 | +0.27005349 | -3.3e-07 | +0.27007272 | +1.9e-05 | +1.9e-05 |
| MgO_k333 | hse06 | +0.22876483 | +0.22881796 | +5.3e-05 | +0.22883771 | +7.3e-05 | +2.0e-05 |
| hBN_k661 | pbe0 | +0.01050458 | +0.01050371 | -8.7e-07 | +0.01060131 | +9.7e-05 | +9.8e-05 |
| hBN_k661 | hse06 | +0.01110674 | +0.01111519 | +8.5e-06 | +0.01121085 | +1.0e-04 | +9.6e-05 |
| O2_box | pbe0 | +0.04044334 | +0.04045604 | +1.3e-05 | +0.04050163 | +5.8e-05 | +4.6e-05 |
| O2_box | hse06 | +0.03711984 | +0.03743523 | +3.2e-04 | +0.03747237 | +3.5e-04 | +3.7e-05 |

With the atoms off its grid, QE's E_hyb - E_PBE of HSE06 rises by 2.5e-6 (Si 2x2x2) to 3.2e-4 Ry (O2), that of PBE0 changes by at most 1.3e-5 Ry (O2), and this code's HSE06 agrees with the moved QE as well as its PBE0 does (Si and C within 1.1e-5 Ry, MgO 2.0e-5, O2 3.7e-5; h-BN 9.6e-5, QE's cutoff error). This code evaluates the semilocal part at every point of its grid (which holds the atom at the origin
too), with no skip where the gradient vanishes.

## Forces

| functional | this code | QE | difference | hybrid - PBE: difference |
|---|---|---|---|---|
| pbe | -0.018054 | -0.01806913 | +1.5e-05 |  |
| pbe0 | -0.018967 | -0.01898746 | +2.0e-05 | +5.3e-06 |
| hse06 | -0.018876 | -0.01889558 | +2.0e-05 | +4.4e-06 |

Atom 2 of `Si_disp_k222` (moved by +0.05 bohr along x), forces after the net force is set to zero, as QE's; this
code prints them to 1e-6 Ry/bohr. The PBE forces of the two codes differ by 1.5e-5 Ry/bohr (grid against cutoff), the hybrid forces by 2.0e-5 Ry/bohr; the change of the force from PBE to the hybrid agrees with QE's to 5.3e-6 (PBE0) and 4.4e-6 Ry/bohr (HSE06). The exchange operator of norm-conserving pseudopotentials adds no
explicit force term (Hellmann-Feynman forces of the hybrid orbitals and density). Atom 1 sits exactly on a grid point;
the gradient of its projectors there is the slope of the smooth `c r^l` of the table (`PARSEC_PROJECTOR_CENTER=auto`),
so its force before the net force is removed, 0.01806459 Ry/bohr with PBE, agrees with central differences of the
energy (SCF 1e-10 Ry, Fortran-derived routines, steps 1e-4, 1e-3 and 2e-3 bohr) within 1.6e-11, 2.2e-9 and 9.2e-9
Ry/bohr (PBE), and within 1.2e-6 (PBE0) and 4.8e-7 Ry/bohr (HSE06) at all three steps. The step of 1e-4 bohr lies
inside the first samples of the Si FHI table; since 23 November 2026 the reader continues the table below them
(`PARSEC_FHI_ORIGIN=auto`), while the former reading (`PARSEC_FHI_ORIGIN=off`, the padding of pseudo.f90, 25 to 48 %
above `c r` there) gives a slope 7.9e-6 (PBE), 7.8e-6 (PBE0) and 8.5e-6 Ry/bohr (HSE06) below the force at that step.
With the gradient `nonloc.F90` forms at the point (`PARSEC_PROJECTOR_CENTER=tip`, the former forces: 0.01805669
Ry/bohr with PBE) the force lies 7.9e-6 (PBE), 1.0e-5 (PBE0) and 9.4e-6 Ry/bohr (HSE06) below the slope at every step;
under `tip` the rows above read -0.018050, -0.018963 and -0.018871 (A100, 23 November 2026).

## Error budget

| source | size | how measured |
|---|---|---|
| QE's cutoff, E_hyb - E_PBE | Si 1.4e-6, C 3.7e-6, MgO 1.2e-5, O2 3.1e-5, h-BN 9.7e-5 Ry | QE's PBE0 `gb` at the next cutoff of a scan (Si 80, C 140, MgO 180, O2 200, h-BN 160 Ry; for h-BN `qe/*_e160.in`) |
| QE's cutoff, gaps | Si 0.3, C 0.1, MgO 0.2, O2 0.2, h-BN 1.8 meV | the same scans |
| QE's HSE grid points | HSE06 E_hyb - E_PBE: O2 3.2e-4, MgO 5.3e-5, C 2.2e-5, h-BN 8.5e-6, Si <= 3.7e-6 Ry; PBE0 <= 1.3e-5 Ry (O2) | `qe/*_moved.in` (above) |
| this code's grid, E_hyb - E_PBE | h-BN PBE0 4e-6 Ry between h 0.18 and 0.15 | the same input at h 0.18; `examples/0d_hybrid_functionals`: 1.3e-5 Ry between h 0.16 and 0.12 (H2O, N2) |
| Fock loop | `Exx_Tolerance 1e-8 Ry` (estimated exchange error); QE: dexx <= 1e-10 Ry | the logs |
| k-point mesh | none between the codes: the same mesh in both (the reference is QE's row at the same mesh and q+G = 0 treatment, not the converged limit) | |
| q+G = 0 element | 1e-15 relative | unit test of the kernel against QE's `exxdiv` of every row |
| Kleinman-Bylander quadrature | none: both codes integrate E_l exactly (the former trapezoid: a constant offset of the totals, PBE level, and up to 1.1e-5 Ry in E_hyb - E_PBE, O2) | this code with `PARSEC_KB_QUADRATURE=trapezoid`; QE with those E_l |
| h-BN box height (`*_periodic`) | 30.3 against 30.0 bohr: none for `gbx` and HSE (above) | QE at cell heights 20 to 60 bohr |

## Timings

| input | GPUs | program (s) | SCF (s) | exchange (s) | solver (s) | pair solves | outer | peak GiB |
|---|---|---|---|---|---|---|---|---|
| Si_k222/pbe | 1 | 3.5 | 2.9 |  |  |  |  | 0.7 |
| Si_k222/pbe0 | 1 | 7.3 | 6.6 | 0.27 | 0.09 | 6736 | 4 | 1.8 |
| Si_k222/hse06 | 1 | 7.5 | 6.9 | 0.30 | 0.09 | 6736 | 4 | 1.8 |
| Si_k222/hse06_default | 1 | 7.7 | 7.1 | 0.28 | 0.09 | 6736 | 4 | 1.8 |
| Si_k222/hf | 1 | 8.5 | 7.9 | 0.47 | 0.14 | 11920 | 7 | 1.9 |
| Si_k222/b3lyp | 1 | 7.7 | 7.0 | 0.32 | 0.10 | 8288 | 5 | 1.8 |
| Si_k444/pbe | 1 | 18.4 | 17.5 |  |  |  |  | 1.6 |
| Si_k444/pbe0 | 1 | 34.7 | 34.3 | 5.66 | 2.77 | 306432 | 5 | 2.8 |
| Si_k444/hse06 | 1 | 30.9 | 30.1 | 4.59 | 2.25 | 249216 | 4 | 2.8 |
| Si_k444/hse06_default | 1 | 30.0 | 29.6 | 4.61 | 2.26 | 249216 | 4 | 2.8 |
| Si_k444/hf | 1 | 43.7 | 43.3 | 9.47 | 4.49 | 498432 | 8 | 3.0 |
| Si_disp_k222/pbe | 1 | 4.0 | 3.1 |  |  |  |  | 0.7 |
| Si_disp_k222/pbe0 | 1 | 7.8 | 6.9 | 0.41 | 0.22 | 6736 | 4 | 1.8 |
| Si_disp_k222/hse06 | 1 | 8.2 | 7.4 | 0.37 | 0.12 | 6736 | 4 | 1.8 |
| C_k333/pbe | 1 | 11.5 | 10.7 |  |  |  |  | 1.3 |
| C_k333/pbe0 | 1 | 13.6 | 12.9 | 1.05 | 0.47 | 40102 | 4 | 2.2 |
| C_k333/hse06 | 1 | 12.3 | 11.6 | 0.79 | 0.34 | 30872 | 3 | 2.2 |
| MgO_k333/pbe | 1 | 36.6 | 35.7 |  |  |  |  | 1.9 |
| MgO_k333/pbe0 | 1 | 35.1 | 34.2 | 1.03 | 0.53 | 40102 | 4 | 2.9 |
| MgO_k333/hse06 | 1 | 31.9 | 31.1 | 0.80 | 0.40 | 30872 | 3 | 2.9 |
| hBN_k661/pbe | 1 | 87.8 | 81.5 |  |  |  |  | 4.5 |
| hBN_k661/pbe0 | 1 | 194.2 | 187.5 | 11.88 | 10.16 | 95664 | 5 | 7.2 |
| hBN_k661/hse06 | 1 | 188.9 | 182.3 | 9.61 | 8.27 | 77800 | 4 | 7.2 |
| hBN_k661/pbe0_periodic | 1 | 192.3 | 185.5 | 6.90 | 5.37 | 95664 | 5 | 6.8 |
| hBN_k661/hse06_periodic | 1 | 184.1 | 177.5 | 5.72 | 4.35 | 77800 | 4 | 6.8 |
| O2_box/pbe | 1 | 15.2 | 14.1 |  |  |  |  | 2.0 |
| O2_box/pbe0 | 1 | 1176.5 | 1175.5 | 7.73 | 1.91 | 548 | 5 | 0.5 |
| O2_box/hse06 | 1 | 1180.5 | 1179.5 | 6.32 | 1.56 | 447 | 4 | 0.5 |
| Si64_gamma/pbe | 1 | 4.1 | 2.3 |  |  |  |  | 0.5 |
| Si64_gamma/pbe0 | 1 | 22.4 | 20.6 | 1.25 | 0.34 | 67008 | 6 | 1.2 |
| Si64_k222/pbe | 4 | 13.7 | 11.4 |  |  |  |  | 0.7 |
| Si64_k222/pbe0 | 4 | 137.8 | 136.0 | 41.83 | 29.00 | 2918400 | 6 | 14.4 |

Program wall time of `main.py` (first runs of each input, with the GPU kernels of the code compiled on the node by
discarded runs of each system at another grid spacing on the same GPU, four inputs side by side, one per GPU, on an
A100 node that other jobs shared: indicative; the logs of 23 November 2026, those of `Si64_k222` of 22 November 2026
and of `Si64_gamma` of 21 November 2026), the
seconds of its SCF (PBE start and Fock loop), of the exchange builds (pair solves on the GPU and ACE; in the cuFFT
solver alone), the pair solves, the outer iterations and the peak device memory (`acceleration/benchmarks/
periodic_hybrid_run.py`). The exchange builds take 3 % (MgO, h-BN with the periodic kernel) to 17 % (Si 4x4x4; 22 % for HF) of the SCF, and 6 % and 31 % for Si64 at the Gamma point and on the 2x2x2 mesh; the rest is mostly the eigensolver of the k-points. Against the code of 21 November 2026 (the two trees side by side on one allocation) the k-point inputs take the same eigensolver steps in less time: Si 4x4x4 PBE 30.9 -> 17.9 s, PBE0 50.4 -> 33.1 s, h-BN PBE 220.4 -> 87.4 s. The sparse products of the Bloch Hamiltonians on the device in a fixed order (`PARSEC_ORDERED_SUMS=auto`) are faster (Si 4x4x4 PBE: the Chebyshev filter of the same 37,200 operator applications 8.9 s, against 19.7 s with `off`, which takes 30.2 s in all), and the preparation of the h-BN slab takes 6 s (38 s with `PARSEC_CONFINED_SETUP=readable`, 48 s before). The two O2 hybrids run the Fock loop on the host (the third route above): 1,177 to 1,181 s, against 15 s for the PBE input of the same cell on the GPU. Si64: at the Gamma point (68^3 points, 128 occupied states, 67,008 pair solves) PBE0 takes 22.4 s on one GPU in the 32 real sectors of the abelian group of the cell with the half-cell translations (a hybrid keeps no weighted sectors; SCF 20.6 s, 1.25 s of it in the exchange, and 1.2 GiB; in the runs of 20 November 2026 SCF 20.1 s against 34.6 s and 8.2 GiB on the full grid with `PARSEC_GAMMA_HYBRID_SECTORS=off` on the same node, the same energy to the printed digits), and PBE 4.1 s in the 10 weighted sectors of a subgroup of 32 of the 192 operations of the cell (README of the accelerated driver, "Translations by half the cell"; in the runs of 20 November 2026 4.9 s against 8.0 s and 4.1 GiB on the full grid (`PARSEC_PERIODIC_SYMMETRY=off`, the route of the log of 8 November 2026: 8.6 s, 4.3 GiB) on the same node, with the same energy to 1e-8 Ry (the 15.0 s of the log of 19 November 2026 included the compilation of the CUDA kernels of the sectors)). On the 2x2x2 mesh (64^3 points) PBE0 runs its PBE start on all 8 points of PARSEC 1.5's Gamma-centred mesh, each its own time-reversal partner, and, the start being an insulator, its Fock loop on the wedge of 4 points (`PARSEC_HYBRID_KPOINT_WEDGE`; since 19 November 2026 the last inner loop keeps the sectors, `PARSEC_KPOINT_SECTOR_LEVELS`): 2.9 million pair solves, 138 s on four GPUs, 42 s of it in the exchange, 14.4 GiB per GPU (the log of 19 November 2026 solved all 8 points throughout: 4.8 million pair solves, 375 s, 76 s in the exchange, 25.7 GiB; the same total energy to the printed digits); PBE solves its irreducible wedge of 4 points (Gamma, X, M and R, the density symmetrized), each split into the sectors of its little group (64 tasks; since 22 November 2026 the 32 of them that are images P_g u of solved sectors under operations of the little group are not solved, `PARSEC_KPOINT_SECTOR_CLASSES`), in 14 s (23 s in the log of 21 November 2026, which solved all 64). With the former mesh of this input (`PARSEC_MP_UNSHIFTED_MESH=port`, points at +-1/4) the code of 8 November 2026 took 168 s for PBE0 (4 time-reversal pairs, 2.4 million pair solves) and 129 s for PBE (all 8 points, without sectors). The same `Grid_Spacing` gives 68 points per side on the Gamma-point route and 64 on the k-point route.
