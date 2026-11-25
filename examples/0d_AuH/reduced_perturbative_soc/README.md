Scaled-down AuH molecule (same geometry and pseudopotentials as
`examples/0d_AuH/`, reduced grid resolution for a fast, apples-to-apples
comparison) exercising perturbative spin-orbit coupling combined with
collinear spin polarization -- the mode `parsec_python.cli` runs whenever
`Spin_Polarization=true` and any species has `SO_PSP=true` but
`SO_from_scratch`/`SCF_SO` is not set.

`parsec.in` is the PARSEC.py-flavored input (used by
`tests/unit/test_0d_AuH_fortran_reference.py`). `parsec_fortran.in` is the
same physical system in Fortran PARSEC's own input, used to generate
`fortran_reference.out`:

```
mpiexec -n 2 <PARSEC_FORTRAN_ROOT>/src/parsec-ubuntu_gnu-mpifort-gfortran-16.2.1.linux_ubuntu.mpi
```

(git commit `f432777`, config `ubuntu_gnu`; PARSEC_ACCELERATED environment
not involved -- run from a directory containing only `parsec_fortran.in`
renamed to `parsec.in` plus the two POTRE files.)

PARSEC.py values from `src/parsec_python/reference_main.py parsec.in`, code
of 24 November 2026 on the CPU of an A100 node. The input gives no
`Cubic_Spline` lines, so PARSEC.py interpolates both tables by cubic splines,
its default since 22 November 2026, where Fortran PARSEC interpolates
linearly; since 23 November 2026 it normalizes the Kleinman-Bylander
projectors by the quadrature of the logarithmic mesh and takes the CA
coefficients that make the correlation continuous at `r_s = 1`, and since 24
November 2026 it Fourier-filters the local potentials and the scalar
projectors to the grid before they are sampled (`PARSEC_LOCAL_FILTER`; the
spin-orbit projectors as the file gives them). With
`PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics: linear unfiltered
tables, its Kleinman-Bylander quadrature and CA coefficients; code of 24
November 2026, the same numbers as on 23 November and as
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` gave on 22
November) the run compares as follows. Converged
scalar-relativistic (pre-SOC) spin-polarized total energy: Fortran
-68.32490203 Ry vs PARSEC.py -68.32490805 Ry (6.0e-6 Ry). Converged
perturbative-SOC state 1 eigenvalue/<S_z>: Fortran -0.818415 Ry / 0.9779 vs
PARSEC.py -0.818297 Ry / 0.9778. With the defaults PARSEC.py gives
-68.14849805 Ry (0.176 Ry above: +0.168 Ry from the Fourier filter, +1.02e-2
Ry from the cubic splines, mostly of the Au table, and -1.8e-3 Ry from the
quadrature of the projector norms and the CA coefficients) and -0.776312 Ry /
0.9677 for state 1. The grid of 0.5 bohr of this reduced input is far too
coarse for the Au table, so the filter moves the energy that much: the same
molecule without spin-orbit coupling and spin polarization (`main.py` on one
A100) lies 0.637 Ry (the file's tables) and 0.469 Ry (filtered) below its
energy at a spacing of 0.15 bohr (-67.67992844 Ry, both forms within 1.1e-5
Ry), and from 0.4 bohr on the two forms agree within 2.1e-3 Ry. The
comparison with Fortran PARSEC is one of numerics on the same coarse grid,
hence made with PARSEC 1.5's numerics. States
inside tightly-spaced
near-degenerate multiplets (Au's d-manifold) show larger <S_z> spread
between the two codes, expected basis-sensitivity rather than a
discrepancy.
