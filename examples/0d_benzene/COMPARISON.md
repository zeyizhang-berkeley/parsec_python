# Benzene comparison

Both calculations use the same geometry, grid, order-12 finite differences,
CA/PZ LDA functional, 20 states, order-one double grid, and canonical PARSEC
C/H pseudopotentials. The Python values come from `parsec_python.out`, run
with the code of 22 November 2026 on one NVIDIA A100. The input gives no
`Cubic_Spline` line, so Python interpolates both tables by cubic splines,
its default since 22 November 2026, where PARSEC 1.5 interpolates
linearly. The columns "linear tables" are the same command with
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off`.

## Final energies

| Quantity [Ry] | Python | PARSEC | Python - PARSEC | Python, linear tables | Linear - PARSEC |
|---|---:|---:|---:|---:|---:|
| Eigenvalue energy | -26.80136519 | -26.80659505 | 0.00522986 | -26.80047913 | 0.00611592 |
| Hartree energy | 264.06913753 | 264.06624712 | 0.00289041 | 264.06795951 | 0.00171239 |
| Integral of `Vxc*rho` | -32.40011813 | -32.39957539 | -0.00054274 | -32.39989016 | -0.00031477 |
| Exchange-correlation energy | -24.78546884 | -24.78505779 | -0.00041105 | -24.78529630 | -0.00023851 |
| Electron-ion energy | -586.97968885 | -586.97255581 | -0.00713304 | -586.97566381 | -0.00310800 |
| Ion-ion energy | 207.86811275 | 207.86811275 | 0.00000000 | 207.86811275 | 0.00000000 |
| **Total energy** | **-75.39064796** | **-75.38863314** | **-0.00201482** | **-75.38863333** | **-0.00000019** |

With PARSEC 1.5's linear tables the total energy agrees to `2e-7 Ry`. The
default cubic splines lower it by `2.0e-3 Ry` (`1.7e-4 Ry` per atom), the
difference between the two interpolations of these tables. The individual SCF-dependent terms differ more than the final total because
each calculation stops on its potential-residual criterion at a different
point in its mixing trajectory. Their double-counting corrections cancel
closely in the total energy.

## Eigenvalues

| Quantity | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| HOMO, state 15 [Ry] | -0.4819702509 | -0.4821855251 | 0.0002152742 |
| LUMO, state 16 [Ry] | -0.0979606962 | -0.0981788182 | 0.0002181220 |
| HOMO-LUMO gap [Ry] | 0.3840095547 | 0.3840067069 | 0.0000028478 |

Across all 20 states, the maximum absolute eigenvalue difference is
`0.0003736716 Ry` (state 18) and the RMS difference is `0.0001965890 Ry`.
With PARSEC 1.5's linear tables (HOMO `-0.4819995091 Ry`, LUMO
`-0.0979891855 Ry`, gap `0.3840103236 Ry`) the maximum is `0.0003613903 Ry`
(state 18) and the RMS `0.0002051253 Ry`, and the differences are close to
a common shift of about `2e-4 Ry` of all levels, which comes from where
each SCF loop stops: with `Convergence_Criterion: 1.0e-8` the Python HOMO is
`-0.4820798948 Ry` (cubic splines: `-0.4820507993 Ry`), and with either
interpolation the levels of the run at the input's criterion lie 0.8e-4 to
1.1e-4 Ry above the tightly converged ones for states 1 to 17 and up to
1.8e-4 Ry above for the three highest. The cubic splines move the levels by
`-1.7e-4 Ry` (the lowest) to `+3.3e-5 Ry`: the five lowest levels then lie
`6e-5` to `1.3e-4 Ry` from PARSEC's, and most of the others move `1e-5` to
`3e-5 Ry` farther away.

## Convergence and timing

| Quantity | Python | PARSEC |
|---|---:|---:|
| SCF iterations | 10 | 17 |
| Final plain SRE | 0.0065385021 | 0.0026821908 |
| Final charge-weighted SRE | 0.0001036044 | 0.0001465128 |
| Diagonalization time | 0.89 s | 4.89 s |
| Hartree time | 0.07 s | 5.29 s |
| SCF time | 1.13 s | 9.92 s |
| Program wall time | 2.30 s | 10.63 s |

The timing comparison is not like-for-like. Python ran as one process on an
A100 GPU and solved the eight `D_2h` representations there; its program
time includes about 1.2 s of set-up before the SCF. PARSEC used eight
MPI ranks on the CPU, separated the same eight representations, and worked
with a 33,512-point reduced domain. PARSEC also reports forces and MPI
statistics that the current Python single-point path does not calculate.

The Python result in this table uses the native CHEBFF initialization
(`FF_MaxIter` 3, the default since 5 November 2026) and one source-compatible
SUBSPACE filter per later SCF step. The older adapted ARPACK-first trajectory
and the full-grid run of 28 July 2026 (15 SCF iterations, `-75.38863311 Ry`)
are no longer part of the package.
