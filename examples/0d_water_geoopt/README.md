# Water molecule: BFGS geometry optimization

The geometry-optimization example `0d_water_geoopt` of Fortran PARSEC,
unchanged: a water molecule in a sphere of 12 bohr on a 0.3 bohr grid, CA-LDA,
`Minimization: BFGS`, `Force_Min: 0.02` Ry/bohr.

```powershell
python src\parsec_python\main.py examples\0d_water_geoopt\parsec.in --no-archive
```

The run computes seven structures in about 7 s on one NVIDIA A100 (code of
24 November 2026, after one throwaway warm-up run on the GPU, three other
runs on the node) and ends with exit status 0. Beside the input it writes
`parsec.out`, `atom.cor`, `relax_restart.dat`, `iterate.dat` and
`relaxation.xyz`; see
[Forces, relaxation and molecular dynamics](../../src/parsec_python/FORCES_AND_RELAXATION.md).

## Expected trajectory

`atom_cor_fortran_reference.out` is the `atom.cor` of Fortran PARSEC for this
input. Energy per atom in eV and largest force in Ry/bohr of every structure:

| Structure | Fortran PARSEC | This package | This package, `PARSEC_FORTRAN_NUMERICS=on` |
|---:|---|---|---|
| 0 | -155.04550, 0.22661 | -155.03875, 0.23468 | -155.04550, 0.22661 |
| 1 | -154.56606, 0.21984 | -154.56348, 0.22093 | -154.56607, 0.21984 |
| 2 | -155.26126, 0.08884 | -155.27072, 0.08655 | -155.26126, 0.08884 |
| 3 | -155.31577, 0.08506 | -155.32624, 0.08012 | -155.31577, 0.08505 |
| 4 | -155.42371, 0.10606 | -155.39331, 0.15382 | -155.42422, 0.10540 |
| 5 | -155.46132, 0.03881 | -155.45598, 0.03309 | -155.46147, 0.03862 |
| 6 | -155.47162, 0.01352 | -155.46910, 0.01396 | -155.47161, 0.01389 |

The input gives no `Cubic_Spline` lines, so this package interpolates both
tables by cubic splines, its default since 22 November 2026; PARSEC 1.5
interpolates linearly. Since 23 November 2026 the projector norms also take
the quadrature of the logarithmic mesh and the CA correlation the
coefficients that make it continuous at `r_s = 1`, and since 24 November
2026 the local potentials and the projectors are Fourier-filtered to the
grid before they are sampled (`PARSEC_LOCAL_FILTER`). Together the defaults
raise the energy of the first structure by 6.7e-3 eV per atom: the filter
raises it by 1.23e-2 eV per atom (2.7e-3 Ry for the molecule), the splines,
the quadrature and the CA coefficients lower it by 5.5e-3. The filter's
share is the egg-box term of this placement: on this 0.3-bohr grid the
energy of these Troullier-Martins tables varies by 4.0e-3 Ry and the forces
by 3.0e-2 Ry/bohr when the molecule moves by one grid step, and by 5.2e-4 Ry
and 9.9e-3 Ry/bohr with the filtered tables (`PARSEC_LOCAL_FILTER` in the
[accelerated README](../../src/parsec_python/acceleration/README.md)). The
force noise of the file's tables is of the size of `Force_Min`, so the
trajectory differs from the second structure on (positions 4e-3 to 8.5e-2
bohr apart from that of the file's tables, `PARSEC_LOCAL_FILTER=off`); it
also takes seven structures, and the last one is 2.5e-3 eV per atom above
Fortran's. The last column is the same run with

```text
PARSEC_FORTRAN_NUMERICS=on
```

in the environment (PARSEC 1.5's numerics; code of 24 November 2026, the
same digits as on 23 November and as `PARSEC_CUBIC_SPLINE_DEFAULT=off
PARSEC_PROJECTOR_TAIL=off` on 22 November). The input leaves the SCF criterion at PARSEC's default
of 2e-4 Ry, so the two codes stop their SCF loops at slightly different
points and the last digits differ (the Chebyshev degree of the later SCF
steps of this package may decrease after the fourth step instead of the
fifth). With PARSEC 1.5's numerics and `Convergence_Criterion: 1.0e-8` and
`Max_Iter: 200` added to both inputs, all seven energies agree in every
printed digit, the largest forces within 1e-5 Ry/bohr and the coordinates
within 1e-6 bohr (`atom.cor` of this package under
`PARSEC_FORTRAN_NUMERICS=on` is that of 22 November with linear tables,
byte for byte).

The second structure lies higher in energy than the first: the first step of
L-BFGS-B moves along the force by the inverse of its norm, and the line search
then shortens it. Fortran PARSEC does the same.

## Stopping and continuing

```text
Movement_Num: 2
```

stops the run after two movements with exit status 4 (not relaxed). A second
run with

```text
Relax_Restart: true
```

or with `--restart-from relax_restart.dat` on the command line continues from
the saved structure and optimizer state. The file is PARSEC's: either code
continues a run of the other.
