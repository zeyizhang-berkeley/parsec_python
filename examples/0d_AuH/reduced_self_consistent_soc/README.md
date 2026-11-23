Scaled-down AuH molecule (same geometry and pseudopotentials as
`examples/0d_AuH/`, reduced grid resolution) exercising self-consistent
spin-orbit coupling.

`parsec.in` (PARSEC.py-flavored: `SO_from_scratch=true`, no
`Spin_Polarization`) is used by
`tests/unit/test_0d_AuH_fortran_reference.py`, and so is
`parsec_spin_polarized.in`, the same input with `Spin_Polarization: .true.`.
That combination runs on the reference solver (`reference_main.py`:
-68.33264566 Ry with the defaults, -68.34106409 Ry with PARSEC 1.5's
numerics; see below); the GPU path does not cover it (`main.py --gpus 1` stops
with an input error). `parsec_fortran.in`
is the same physical system in Fortran PARSEC's own input, used to
generate `fortran_reference.out`:

```
mpiexec -n 2 <PARSEC_FORTRAN_ROOT>/src/parsec-ubuntu_gnu-mpifort-gfortran-16.2.1.linux_ubuntu.mpi
```

(git commit `f432777`, config `ubuntu_gnu`; run from a directory containing
only `parsec_fortran.in` renamed to `parsec.in` plus the two POTRE files.)

Note: Fortran PARSEC's own input parser hard-rejects (`ierr=151`,
`usrinputfile.F90:2574-2580`) any species with `SO_PSP=true` unless
`Spin_Polarization=true` is also set -- this holds even for `SCF_SO`, so
`parsec_fortran.in` sets `Spin_Polarization: .true.` despite PARSEC.py's
side not using it. AuH's ground state has zero net magnetic moment in both
codes, so the comparison is still meaningful. PARSEC.py values from
`src/parsec_python/reference_main.py parsec.in`, code of 23 November 2026
on the CPU of an A100 node: the input gives no `Cubic_Spline` lines, so
PARSEC.py interpolates both tables by cubic splines, its default since
22 November 2026, where Fortran PARSEC interpolates linearly, and since
23 November 2026 it normalizes the Kleinman-Bylander projectors by the
quadrature of the logarithmic mesh and takes the CA coefficients that make
the correlation continuous at `r_s = 1`. With
`PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics: linear tables, its
Kleinman-Bylander quadrature and CA coefficients; code of 23 November 2026,
the number `PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` gave
on 22 November) the converged self-consistent SOC total energy matches to 5.0e-6
Ry (Fortran -68.34107156 Ry vs PARSEC.py -68.34106654 Ry); the defaults
give -68.33264812 Ry, 8.4e-3 Ry above (+1.02e-2 Ry from the cubic splines,
mostly of the Au table, and -1.8e-3 Ry from the quadrature of the projector
norms and the CA coefficients). Kramers
double-degeneracy holds to high precision in both codes. Individual
eigenvalues agree to ~1e-3 Ry (6e-4 Ry with PARSEC 1.5's numerics, 1.1e-3 Ry
with the defaults); <S_z> within near-degenerate pairs diverges more, same
basis-sensitivity caveat as the perturbative case.
