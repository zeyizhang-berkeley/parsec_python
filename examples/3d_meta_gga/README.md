# Meta-GGAs of bulk Si against Quantum ESPRESSO

SCAN (this code's native SCAN), r2SCAN and TPSS (libxc) of bulk Si, fcc with two atoms, on Gamma-centred 2x2x2 and
4x4x4 meshes, compared with Quantum ESPRESSO 7.5 + libxc 7.0.0 using the same FHI98PP potential and cell. The
quantities compared are E_mGGA - E_PBE of each code at its own discretization (per cell) and the band gap on the
mesh. The molecules, the meta-GGA hybrids, the keywords and the defaults are in
[`../0d_meta_gga/README.md`](../0d_meta_gga/README.md).

## Contents

- `Si_k222/<functional>/parsec.in`, `Si_k444/<functional>/parsec.in` for `pbe`, `scan`, `r2scan`, `tpss`, and
  `parsec_python.out`, the reference log of that input (one NVIDIA A100; paths shortened to repository paths, the libxc
  library as `$PARSEC_LIBXC_LIBRARY`).
- `Si_FHIPP.DAT`: a copy of `src/pp_generation/NCPPs/PBE/Si/FHI/Si_FHIPP.DAT` (Troullier-Martins, PBE, no core
  correction), found with `--pp-dir`.
- `qe/`: the Quantum ESPRESSO inputs, their settings and the cutoff error (`qe/README.md`).
- `run_examples.py` (one process per input, `main.py --gpus 1`; `--install` copies the logs of a run folder into the
  example) and `collect_results.py` (the tables below from the logs).

## Run

From the repository root, with libxc for r2SCAN and TPSS (see "Functionals of libxc" in
`src/parsec_python/README.md`):

```bash
export PARSEC_LIBXC_LIBRARY=.../libxc-7.1.2-cuda-sm80/lib/libxc.so
python src/parsec_python/main.py examples/3d_meta_gga/Si_k444/r2scan/parsec.in --pp-dir examples/3d_meta_gga --no-archive
python examples/3d_meta_gga/run_examples.py --out RUNS          # every input, logs in RUNS/<mesh>/<functional>/
python examples/3d_meta_gga/collect_results.py [--runs RUNS]
```

Settings of every input: a = 10.26 bohr (lattice vectors (0, a/2, a/2), ...), atoms at 0 and a/4 (1, 1, 1),
`Grid_Spacing` 0.20 bohr, `Expansion_Order` 12, `Cubic_Spline`, `States_Num` 8, `Fermi_Temp` 10 K (occupations 1
and 0), `Convergence_Criterion` 1e-6 Ry. The mesh is `Monkhorst_Pack_Grid n n n` with a zero
`Monkhorst_Pack_Shift`: QE's Gamma-centred mesh, reduced to its irreducible wedge (3 and 8 k-points) with the
density symmetrized; these cells run on the Fortran-derived routines (the default for these grids, as the log says).
The functional lines are those of the molecules: `Correlation_Type: scan`, `LibXC_Functional: mgga_x_r2scan
mgga_c_r2scan`, and TPSS in PARSEC 1.5's form (`Use_LibXC`, `Func_Id_X: 202`, `Func_Id_C: 231`). The default
density gradient (`auto`) is the orbital form for SCAN and r2SCAN and the density form for TPSS (its log says so).

## Against Quantum ESPRESSO

QE at 200 Ry, dual 8. PBE:

| mesh | E this code (Ry/cell) | E QE (Ry/cell) | difference (Ry) | gap this code (eV) | gap QE (eV) | dgap (meV) |
|---|---|---|---|---|---|---|
| 2x2x2 | -15.57491210 | -15.57491928 | +7.18e-06 | 0.6102 | 0.6102 | -0.0 |
| 4x4x4 | -15.74871176 | -15.74871754 | +5.78e-06 | 0.7002 | 0.7002 | -0.0 |

The logs are of 24 November 2026. Their defaults integrate the Kleinman-Bylander energies exactly on the logarithmic
mesh (Simpson's value to 1e-12 relative, as QE) and Fourier-filter the local potential and the projectors to the grid
(`PARSEC_LOCAL_FILTER=all`, the default since 23 November 2026): the PBE totals lie 7.2e-6 and 5.8e-6 Ry per cell
above QE's. With the tables of the file (`PARSEC_LOCAL_FILTER=off`, the logs of 23 November 2026, reproduced to the
printed digits on the same tree) they lay 1.6e-6 and 1.5e-6 Ry below: the filter raises the totals by 8.8e-6 and
7.2e-6 Ry per cell at this spacing and placement, changes E - E_PBE of the meta-GGAs by at most 0.003 mRy, and
leaves the gaps (within 0.1 meV) and the SCF iterations as they were. PARSEC's trapezoid on the logarithmic mesh, with which the logs
before 23 November 2026 were made, puts the totals 3.34e-4 and 3.40e-4 Ry lower (`PARSEC_KB_QUADRATURE=trapezoid`
on the defaults), as QE with those coefficients moves by -3.3e-4 to -3.6e-4 Ry per cell (`qe/README.md`);
`PARSEC_FORTRAN_NUMERICS=on`, which takes the trapezoid and the file's tables together with PARSEC 1.5's other
numerics, gives the totals of those logs to the printed digits (-15.57525517 and -15.74905936 Ry). In E - E_PBE the
trapezoid cancels to 3 microRy: the meta-GGA rows below moved by at most 0.003 mRy, the gaps by at most 0.1 meV.
Meta-GGAs, E - E_PBE in mRy per cell:

| functional | mesh | E - E_PBE this code (mRy) | E - E_PBE QE (mRy) | this code - QE (mRy) | gap this code (eV) | gap QE (eV) | dgap (meV) | SCF iterations | gradient form |
|---|---|---|---|---|---|---|---|---|---|
| SCAN | 2x2x2 | -7.777 | -7.687 | -0.089 | 0.7954 | 0.7932 | +2.2 | 18 | orbital |
| SCAN | 4x4x4 | -9.588 | -9.443 | -0.145 | 0.9653 | 0.9605 | +4.8 | 17 | orbital |
| r2SCAN | 2x2x2 | +0.407 | +0.516 | -0.109 | 0.7651 | 0.7648 | +0.3 | 15 | orbital |
| r2SCAN | 4x4x4 | -5.981 | -5.888 | -0.092 | 0.8563 | 0.8560 | +0.3 | 16 | orbital |
| TPSS | 2x2x2 | +14.548 | +14.650 | -0.102 | 0.9631 | 0.9628 | +0.3 | 18 | density |
| TPSS | 4x4x4 | +20.197 | +20.288 | -0.091 | 0.8711 | 0.8709 | +0.2 | 17 | density |

E - E_PBE agrees within 0.09 to 0.15 mRy per cell, which is QE's own scatter of the meta-GGA totals between 120 and
250 Ry (+-0.15 mRy per cell, `qe/README.md`); the gaps of r2SCAN and TPSS within 0.3 meV, SCAN's within 2 to 5 meV
(SCAN's grid sensitivity, see `../0d_meta_gga/README.md`; QE's SCAN gap moves by 1-2 meV with its cutoff). At
h = 0.16 bohr the differences are the same to 0.08 mRy (r2SCAN -0.10 and -0.09, SCAN -0.16 and -0.10 mRy).

## TPSS at the Gamma point alone

In a cell sampled at the Gamma point alone, every occupied orbital can have a zero gradient at the same grid points
(in Si the bond centres, at the density maximum), where `tau` and `sigma` vanish together. TPSS-type functionals
(`z = tau_W/tau`; the density form of the gradient under `auto`) may then stall or diverge, whatever
`LibXC_Tau_Threshold` and `LibXC_Enforce_FHC` say, and the orbital form (`PARSEC_META_GGA_GRADIENT=orbital`) does
not converge reliably either. The `tpss` input of `Si_k222` with `Monkhorst_Pack_Grid 1 1 1` (`main.py`, `Max_Iter
200`, `Convergence_Criterion 1e-6 Ry`; A100, 24 November 2026):

- density form: at h = 0.25 bohr the SCF stalls (SRE between 4.5e-3 and 1.7e-1 in the last 100 of 200 iterations);
  at h = 0.40 it converges in 21 iterations.
- orbital form: not converged in 200 iterations at h = 0.25 bohr (SRE 1.7e-5 to 3.6e-4 in the last 100) nor at
  h = 0.16 (3.1e-6 to 6.7e-5); at the last iteration E - E_PBE lies +0.18 and -0.25 mRy from QE's
  (-43.673 mRy, QE 7.5 at Gamma) and the gap 43 and 6 meV below QE's 3.801 eV.

The k-point loop of the check script below (`--kmesh 1 --functional tpss`, the same settings) behaves alike: the
density form stalls at h = 0.25 and diverges at h = 0.40 (the energy jumps to some +200 Ry within ten iterations),
the orbital form ends its 200 iterations at an SRE of 1.2e-4 (h = 0.25) and 4.4e-6 (h = 0.16, after transient
blow-ups of the energy). The same holds with the tables of the file (`PARSEC_LOCAL_FILTER=off`), and for the code of
8 November 2026, with which this section was first written: on both routes its orbital form also ended 200
iterations unconverged (at h = 0.25 the check script reached an SRE of 1.8e-6, E - E_PBE within 0.05 mRy of QE's and
the gap 25 meV below it), so the iteration counts given then (125 and 173) are not reproduced; small numerical
changes move such a loop between stalling and converging. Use a k-mesh: the 2x2x2 and 4x4x4 rows above converge in
17 and 18 iterations. SCAN and r2SCAN, written with `alpha`, are not affected.

## Costs

`main.py` on one A100 (the logs: first run of each input with an empty kernel cache, after one discarded run on each
GPU, four inputs side by side, one per GPU; program wall time, setup included, one process per input; peak device
memory of the process):

| mesh | irreducible k-points | functional | SCF iterations | wall (s) | peak device memory (GiB) |
|---|---|---|---|---|---|
| 2x2x2 | 3 | PBE | 12 | 6.3 | 0.7 |
| 2x2x2 | 3 | SCAN | 18 | 14.0 | 0.9 |
| 2x2x2 | 3 | r2SCAN | 15 | 12.3 | 0.9 |
| 2x2x2 | 3 | TPSS | 18 | 13.0 | 0.9 |
| 4x4x4 | 8 | PBE | 11 | 9.9 | 0.9 |
| 4x4x4 | 8 | SCAN | 17 | 21.2 | 1.0 |
| 4x4x4 | 8 | r2SCAN | 16 | 19.6 | 1.0 |
| 4x4x4 | 8 | TPSS | 17 | 19.6 | 1.0 |

The orbital terms of a meta-GGA (`tau` of the orbitals of every k-point and `-div(vtau grad)` in every Bloch
Hamiltonian) and the evaluation of libxc on the GPU make a meta-GGA run 2.0 to 2.2 times as long as PBE here. The
logs (24 November 2026) are 1.3 to 3.2 s shorter than those of 23 November 2026 (8.9 to 22.6 s): the finiteness
checks on the device no longer compile CUB reductions on a first run, and the kernels of a run are compiled ahead on
background threads (`PARSEC_CUPY_COMPILE_AHEAD=auto`, since 24 November 2026; with `off` these inputs take 0.3 to
1.2 s longer, A100, 24 November 2026), while the projectors filtered to the grid (`PARSEC_LOCAL_FILTER=all`) make
each application of a Bloch Hamiltonian a little dearer. The k-point loop of these meta-GGAs takes no symmetry
sectors: since 23 November 2026 a meta-GGA may solve them (`PARSEC_KPOINT_SECTOR_META_GGA=auto`), but they need a
bulk orthorhombic box of points, and the grid of this fcc cell is not one (the logs say so).

## The check script

`python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check` writes the same cell with a functional,
grid spacing and mesh (also Gamma alone and the full mesh as an explicit k-point list), runs the k-point loop in its
own process (on the GPUs where CuPy finds them) and prints energy, band edges, gap, iterations and timings as JSON:

```bash
python -m parsec_python.acceleration.benchmarks.meta_gga_periodic_check --functional pbe,scan,r2scan,tpss \
    --kmesh 1,2,4 --spacing 0.20 [--gradient density|orbital] [--term-timing] [--program] --out results
```

`r2scan` and `tpss` are libxc's there (`LibXC_Functional`); `scan` is the native SCAN.
