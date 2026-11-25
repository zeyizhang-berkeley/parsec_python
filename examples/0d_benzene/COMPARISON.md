# Benzene comparison

Both calculations use the same geometry, grid, order-12 finite differences,
CA/PZ LDA functional, 20 states, order-one double grid, and canonical PARSEC
C/H pseudopotentials. The Python values come from `parsec_python.out`, run
with the code of 24 November 2026 on one NVIDIA A100. The input gives no
`Cubic_Spline` line, so Python interpolates both tables by cubic splines,
its default since 22 November 2026, where PARSEC 1.5 interpolates
linearly; since 23 November 2026 it also normalizes the projectors by the
quadrature of the logarithmic mesh and uses the continuous CA coefficients,
and since 24 November 2026 it Fourier-filters the local potentials and the
projectors to the grid, where PARSEC 1.5 samples the tables as they are
(`README.md`). The columns "linear tables" are the same command with
`PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics; on 22 November
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` gave these numbers,
and the code of 23 and of 24 November gives every one of them again under
`PARSEC_FORTRAN_NUMERICS=on`, the tightly converged HOMO below included).

## Final energies

| Quantity [Ry] | Python | PARSEC | Python - PARSEC | Python, linear tables | Linear - PARSEC |
|---|---:|---:|---:|---:|---:|
| Eigenvalue energy | -26.80772159 | -26.80659505 | -0.00112654 | -26.80047913 | 0.00611592 |
| Hartree energy | 264.06936806 | 264.06624712 | 0.00312094 | 264.06795951 | 0.00171239 |
| Integral of `Vxc*rho` | -32.39990733 | -32.39957539 | -0.00033194 | -32.39989016 | -0.00031477 |
| Exchange-correlation energy | -24.78544548 | -24.78505779 | -0.00038769 | -24.78529630 | -0.00023851 |
| Electron-ion energy | -586.97796814 | -586.97255581 | -0.00541233 | -586.97566381 | -0.00310800 |
| Ion-ion energy | 207.86811275 | 207.86811275 | 0.00000000 | 207.86811275 | 0.00000000 |
| **Total energy** | **-75.39740094** | **-75.38863314** | **-0.00876780** | **-75.38863333** | **-0.00000019** |

With PARSEC 1.5's numerics the total energy agrees to `2e-7 Ry`. The
defaults lower it by `8.8e-3 Ry` (`7.3e-4 Ry` per atom): `6.5e-3 Ry` from
the Fourier filter of the tables to the grid (the egg-box term of this
placement on the 0.4-bohr grid, `README.md`), `2.0e-3 Ry` from the cubic
splines of these tables and `2.4e-4 Ry` from the quadrature of the
projector norms and the continuous CA correlation. The individual
SCF-dependent terms differ more than the final total because
each calculation stops on its potential-residual criterion at a different
point in its mixing trajectory. Their double-counting corrections cancel
closely in the total energy.

## Eigenvalues

| Quantity | Python | PARSEC | Python - PARSEC |
|---|---:|---:|---:|
| HOMO, state 15 [Ry] | -0.4815044408 | -0.4821855251 | 0.0006810843 |
| LUMO, state 16 [Ry] | -0.0974614196 | -0.0981788182 | 0.0007173986 |
| HOMO-LUMO gap [Ry] | 0.3840430212 | 0.3840067069 | 0.0000363143 |

Across all 20 states, the maximum absolute eigenvalue difference is
`0.0008321140 Ry` (state 9) and the RMS difference is `0.0005558450 Ry`.
With PARSEC 1.5's linear tables (HOMO `-0.4819995091 Ry`, LUMO
`-0.0979891855 Ry`, gap `0.3840103236 Ry`) the maximum is `0.0003613903 Ry`
(state 18) and the RMS `0.0002051253 Ry`, and the differences are close to
a common shift of about `2e-4 Ry` of all levels, which comes from where
each SCF loop stops: with `Convergence_Criterion: 1.0e-8` the Python HOMO is
`-0.4820798948 Ry` (defaults: `-0.4815872255 Ry`), and with either
numerics the levels of the run at the input's criterion lie 0.8e-4 to
1.1e-4 Ry above the tightly converged ones for states 1 to 17 and up to
1.8e-4 Ry above for the three highest. The defaults move the levels by
`-1.1e-3 Ry` (state 9) to `+6.2e-4 Ry` (state 14) from those of PARSEC 1.5's
numerics, almost all of it from the Fourier filter, which also changes how
the grid splits the degenerate pairs (states 2 and 3 lie `1.0e-3 Ry` apart in
PARSEC and with PARSEC 1.5's numerics, `4.7e-5 Ry` with the defaults; states
12 and 13 `7.5e-5` and `4.1e-5 Ry`, against `3.1e-4 Ry`): the levels then lie
between `-8.3e-4` and `+8.3e-4 Ry` from PARSEC's.

## Convergence and timing

| Quantity | Python | PARSEC |
|---|---:|---:|
| SCF iterations | 10 | 17 |
| Final plain SRE | 0.0065219178 | 0.0026821908 |
| Final charge-weighted SRE | 0.0001029167 | 0.0001465128 |
| Diagonalization time | 0.89 s | 4.89 s |
| Hartree time | 0.07 s | 5.29 s |
| SCF time | 1.12 s | 9.92 s |
| Program wall time | 2.32 s | 10.63 s |

The timing comparison is not like-for-like. Python ran as one process on an
A100 GPU (after one throwaway warm-up run on it, three other runs on the
node) and solved the eight `D_2h` representations there; its program
time includes about 1.2 s of set-up before the SCF. PARSEC used eight
MPI ranks on the CPU, separated the same eight representations, and worked
with a 33,512-point reduced domain. PARSEC also reports forces and MPI
statistics that the current Python single-point path does not calculate.

The Python result in this table uses the native CHEBFF initialization
(`FF_MaxIter` 3, the default since 5 November 2026) and one source-compatible
SUBSPACE filter per later SCF step. The older adapted ARPACK-first trajectory
and the full-grid run of 28 July 2026 (15 SCF iterations, `-75.38863311 Ry`)
are no longer part of the package.
