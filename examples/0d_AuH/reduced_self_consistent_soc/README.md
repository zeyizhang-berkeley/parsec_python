Scaled-down AuH molecule (same geometry and pseudopotentials as
`examples/0d_AuH/`, reduced grid resolution) exercising self-consistent
spin-orbit coupling.

`parsec.in` (PARSEC.py-flavored: `SO_from_scratch=true`, no
`Spin_Polarization`) is used by
`tests/unit/test_0d_AuH_fortran_reference.py`, and so is
`parsec_spin_polarized.in`, the same input with `Spin_Polarization: .true.`.
That combination runs on the reference solver (`reference_main.py`:
-68.33088737 Ry with the default cubic splines, -68.34106409 Ry with linear
tables; see below); the GPU path does not cover it (`main.py --gpus 1` stops
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
`src/parsec_python/reference_main.py parsec.in`, code of 22 November 2026
on the CPU of an A100 node: the input gives no `Cubic_Spline` lines, so
PARSEC.py interpolates both tables by cubic splines, its default since
22 November 2026, where Fortran PARSEC interpolates linearly. With
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` (linear
tables) the converged self-consistent SOC total energy matches to 5.0e-6
Ry (Fortran -68.34107156 Ry vs PARSEC.py -68.34106654 Ry); the default
cubic splines give -68.33088983 Ry, 1.02e-2 Ry above, the difference
between the two interpolations, mostly of the Au table. Kramers
double-degeneracy holds to high precision in both codes. Individual
eigenvalues agree to ~1e-3 Ry (6e-4 Ry with linear tables, 1.2e-3 Ry with
splines); <S_z> within near-degenerate pairs diverges more, same
basis-sensitivity caveat as the perturbative case.
