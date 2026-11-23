# Benzene comparison

Both calculations use the same geometry, grid, order-12 finite differences,
CA/PZ LDA functional, 20 states, order-one double grid, and canonical PARSEC
C/H pseudopotentials. The Python values come from `parsec_python.out`, run
with the code of 23 November 2026 on one NVIDIA A100. The input gives no
`Cubic_Spline` line, so Python interpolates both tables by cubic splines,
its default since 22 November 2026, where PARSEC 1.5 interpolates
linearly; since 23 November 2026 it also normalizes the projectors by the
quadrature of the logarithmic mesh and uses the continuous CA coefficients
(`README.md`). The columns "linear tables" are the same command with
`PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics; on 22 November
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` gave these numbers,
and the code of 23 November gives every one of them again under
`PARSEC_FORTRAN_NUMERICS=on`, the tightly converged HOMO below included).

## Final energies

| Quantity [Ry] | Python | PARSEC | Python - PARSEC | Python, linear tables | Linear - PARSEC |
|---|---:|---:|---:|---:|---:|
| Eigenvalue energy | -26.80136899 | -26.80659505 | 0.00522606 | -26.80047913 | 0.00611592 |
| Hartree energy | 264.06933189 | 264.06624712 | 0.00308477 | 264.06795951 | 0.00171239 |
| Integral of `Vxc*rho` | -32.40048919 | -32.39957539 | -0.00091380 | -32.39989016 | -0.00031477 |
| Exchange-correlation energy | -24.78588394 | -24.78505779 | -0.00082615 | -24.78529630 | -0.00023851 |
| Electron-ion energy | -586.97994109 | -586.97255581 | -0.00738528 | -586.97566381 | -0.00310800 |
| Ion-ion energy | 207.86811275 | 207.86811275 | 0.00000000 | 207.86811275 | 0.00000000 |
| **Total energy** | **-75.39088888** | **-75.38863314** | **-0.00225574** | **-75.38863333** | **-0.00000019** |

With PARSEC 1.5's numerics the total energy agrees to `2e-7 Ry`. The
defaults lower it by `2.3e-3 Ry` (`1.9e-4 Ry` per atom): `2.0e-3 Ry` from
the cubic splines of these tables and `2.4e-4 Ry` from the quadrature of
the projector norms and the continuous CA correlation. The individual SCF-dependent terms differ more than the final total because
each calculation stops on its potential-residual criterion at a different
point in its mixing trajectory. Their double-counting corrections cancel
closely in the total energy.

## Eigenvalues

| Quantity | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| HOMO, state 15 [Ry] | -0.4819699258 | -0.4821855251 | 0.0002155993 |
| LUMO, state 16 [Ry] | -0.0979605604 | -0.0981788182 | 0.0002182578 |
| HOMO-LUMO gap [Ry] | 0.3840093654 | 0.3840067069 | 0.0000026585 |

Across all 20 states, the maximum absolute eigenvalue difference is
`0.0003723440 Ry` (state 18) and the RMS difference is `0.0001949505 Ry`.
With PARSEC 1.5's linear tables (HOMO `-0.4819995091 Ry`, LUMO
`-0.0979891855 Ry`, gap `0.3840103236 Ry`) the maximum is `0.0003613903 Ry`
(state 18) and the RMS `0.0002051253 Ry`, and the differences are close to
a common shift of about `2e-4 Ry` of all levels, which comes from where
each SCF loop stops: with `Convergence_Criterion: 1.0e-8` the Python HOMO is
`-0.4820798948 Ry` (defaults: `-0.4820504531 Ry`), and with either
numerics the levels of the run at the input's criterion lie 0.8e-4 to
1.1e-4 Ry above the tightly converged ones for states 1 to 17 and up to
1.8e-4 Ry above for the three highest. The defaults move the levels by
`-1.6e-4 Ry` (the lowest ones) to `+3.2e-5 Ry`: the five lowest levels then
lie `7e-5` to `1.3e-4 Ry` from PARSEC's, and most of the others move `1e-5`
to `3e-5 Ry` farther away.

## Convergence and timing

| Quantity | Python | PARSEC |
|---|---:|---:|
| SCF iterations | 10 | 17 |
| Final plain SRE | 0.0065354917 | 0.0026821908 |
| Final charge-weighted SRE | 0.0001035457 | 0.0001465128 |
| Diagonalization time | 0.99 s | 4.89 s |
| Hartree time | 0.07 s | 5.29 s |
| SCF time | 1.25 s | 9.92 s |
| Program wall time | 2.37 s | 10.63 s |

The timing comparison is not like-for-like. Python ran as one process on an
A100 GPU (after one throwaway warm-up run on it, three other runs on the
node) and solved the eight `D_2h` representations there; its program
time includes about 1.1 s of set-up before the SCF. PARSEC used eight
MPI ranks on the CPU, separated the same eight representations, and worked
with a 33,512-point reduced domain. PARSEC also reports forces and MPI
statistics that the current Python single-point path does not calculate.

The Python result in this table uses the native CHEBFF initialization
(`FF_MaxIter` 3, the default since 5 November 2026) and one source-compatible
SUBSPACE filter per later SCF step. The older adapted ARPACK-first trajectory
and the full-grid run of 28 July 2026 (15 SCF iterations, `-75.38863311 Ry`)
are no longer part of the package.
