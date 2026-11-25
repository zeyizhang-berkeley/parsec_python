# H2 physical comparison result

Date: 2026-11-24 (Python run; the run of 2026-11-05 took the abelian D_2h
sectors and gave -2.29319727 Ry in 17 SCF iterations). The comparison of
2026-07-28 used an earlier Python program, which diagonalized the full grid with an ARPACK first
solve and converged in 7 SCF iterations to -2.293197288860 Ry; that program
is no longer part of the package.

Both calculations used the files in this directory: the same 861-point
Martins-new `H_POTRE.DAT`, H-H distance, real-space grid, finite-difference
order, CA-LDA functional, temperature, state count, and convergence criterion.

The PARSEC values come from `parsec_reference.out`. The Python values come
from `parsec_python.out`: `python src\parsec_python\main.py
examples\h2_full_nonlocal\parsec.in --no-archive --log parsec_python.out`
with the code of 24 November 2026 on one NVIDIA A100 (no symmetry cache,
after one throwaway warm-up run on the GPU, four runs side by side on the
node; paths shortened to repository paths). The input gives no
`Cubic_Spline` line, so the H table is interpolated by cubic splines, the
default since 22 November 2026, and since 23 November 2026 the projector
norms take the quadrature of the logarithmic mesh and the CA correlation
the continuous coefficients, and since 24 November 2026 the local potential
and the projectors are Fourier-filtered to the grid (`PARSEC_LOCAL_FILTER`);
the column "linear tables" is the same command with
`PARSEC_FORTRAN_NUMERICS=on`, the numerics of PARSEC 1.5 (code of 24 November
2026, the same digits as on 23 November;
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` gave them on 22
November). The report prints the energy terms with eight
decimals. Differences below are Python minus PARSEC.

| Quantity | PARSEC [Ry] | Python [Ry] | Python - PARSEC [Ry] | Python, linear tables [Ry] | Linear - PARSEC [Ry] |
|---|---:|---:|---:|---:|---:|
| Total energy | -2.2931972800 | -2.2930261900 | +1.7109e-04 | -2.2931972800 | +0.0000e+00 |
| Eigenvalue energy | -1.5116605700 | -1.5117192500 | -5.8680e-05 | -1.5118798700 | -2.1930e-04 |
| Hartree energy | 2.5892819900 | 2.5894749100 | +1.9292e-04 | 2.5894847500 | +2.0276e-04 |
| Integral Vxc*rho | -1.6953933500 | -1.6955271300 | -1.3378e-04 | -1.6955153800 | -1.2203e-04 |
| Exchange-correlation energy | -1.2986859700 | -1.2987903700 | -1.0440e-04 | -1.2987789100 | -9.2940e-05 |
| Electron-ion energy | -7.0454730400 | -7.0457541900 | -2.8115e-04 | -7.0458615000 | -3.8846e-04 |
| Ion-ion energy | 1.4111386700 | 1.4111386700 | +0.0000e+00 | 1.4111386700 | +0.0000e+00 |
| Occupied eigenvalue | -0.7558302836 | -0.7558596245 | -2.9341e-05 | -0.7559399368 | -1.0965e-04 |

Convergence and runtime:

| Quantity | PARSEC | Python |
|---|---:|---:|
| Converged SCF iteration | 19 | 17 |
| Final weighted SRE [Ry] | 6.58059e-05 | 1.54542e-04 |
| Full active grid points | 179,944 | 179,944 |
| Grid points diagonalized directly | eight symmetry sectors of 22,493 | six weighted sectors of a subgroup of order 8 of D_4h |
| Nonlocal projector columns | 6 | 6 |
| Wall time [s] | 19.30 (CPU) | 2.73 (A100 GPU) |

## Interpretation

With PARSEC 1.5's numerics the total energy is reproduced to the
eight printed decimal places in Rydberg for this matched physical case. The
defaults raise it by `1.7e-4 Ry`: `+1.93e-4 Ry` from the Fourier filter of
the table to the grid, `-1.5e-5 Ry` from the cubic splines of this table and
`-7.1e-6 Ry` from the quadrature of the projector norms and the continuous
CA correlation; both converge in 17 iterations. The
separate energy terms agree at the `1e-4 Ry` scale; their leading
differences cancel in the stationary total energy, and each code stops on
its potential-residual criterion (`2e-4 Ry`) at a different point of its
mixing trajectory. The occupied eigenvalue follows the potential at that
point: it differs by `1.1e-4 Ry` with PARSEC 1.5's numerics, the size of
its own change between the last two Python iterations (`1.0e-4 Ry`), and by
`2.9e-5 Ry` with the defaults; the run of 2026-11-05 (D_2h sectors) stopped
at another point and gave `1.5e-6 Ry`.

PARSEC diagonalizes the eight representations of its abelian subgroup `D_2h`
separately. H2 on the z axis has the point group `D_4h` on the grid, and the
Python run (since 2026-11-18, `PARSEC_ORBITAL_GROUP_MAX_COST=0.75`) solves the
six one-dimensional sectors of a subgroup of order 8 with weights 1, 1, 2, 1,
1, 2: a weight-2 sector holds one partner of each degenerate pair, and its
partner is counted twice. `PARSEC_ORBITAL_GROUP_MAX_COST=0.5` keeps the
`D_2h` sectors of before. The Python first solve is CHEBFF. Unoccupied states
are not paired one-for-one in this table because the two codes distribute the
states differently among the representations, and unoccupied states are
converged less tightly; the Python log lists 16 levels for `States_Num: 16`,
since the weighted state that holds the degenerate pair 16-17 counts for the
one state left (`PARSEC_WEIGHTED_STATE_COUNT`; under `off` it reports both
partners, 17 levels, with the same energy). Degenerate subspaces may also
rotate.

This result validates a full radial potential and the nonlocal KB path. It
does not validate PARSEC's Ono-Hirose double-grid algorithm.

## Canonical PARSEC H potential

The stock `<PARSEC_FORTRAN_ROOT>/tests/H2` case uses PARSEC's canonical
863-point, one-channel H potential and `Double_Grid_Order: 3`. Python accepts
that setting; its double grid averages only the nonlocal projectors, which
this potential does not have, so it gives the order-one result
(`examples/h2_canonical_nodg`). A trial order-one PARSEC input with that
one-channel potential crashes in PARSEC's `nonloc` routine before SCF, so it
cannot supply a matched order-one reference. The stock canonical result is
compared with the order-one Python run in `examples/h2_canonical_nodg`.
