# Benzene PARSEC/Python benchmark

This directory contains the unchanged benzene example input and
pseudopotentials copied from:

```text
<PARSEC_FORTRAN_ROOT>/examples/benchmarks/0d_benzene
```

The copied files are byte-for-byte identical to the WSL sources:

| File | SHA-256 |
|---|---|
| `parsec.in` | `c213ed4eceef790e7bdb86d6f63fc49e513af82a785a01a7b7011f54adb44764` |
| `C_POTRE.DAT` | `2e073cb81f66327a15d82a3955b8822c72f13494d31ec51799b2b3c122598c8c` |
| `H_POTRE.DAT` | `202df2aba56d7aa99a515909add249b643cde27f1aff67bc79b4ceeebf2af191` |

The system has six carbon and six hydrogen atoms, 30 valence electrons,
20 requested states, a 16-bohr spherical domain, 0.4-bohr spacing, and the
default order-12 finite-difference stencil. The omitted double-grid order
defaults to one in PARSEC, so no input change was required.

Carbon uses its `p` channel as the local potential. Its `s` channel supplies
one Kleinman-Bylander projector per carbon atom, giving six nonlocal projector
columns. Hydrogen has only its local `s` channel. Neither potential has a
nonlinear core correction.

## Run

From the repository root:

```powershell
python src\parsec_python\main.py examples\0d_benzene\parsec.in --dry-run
python src\parsec_python\main.py examples\0d_benzene\parsec.in --no-archive
```

The physical calculation writes `parsec.out` in this directory. The
retained `parsec_python.out` is the second command with
`--log parsec_python.out`, run with the code of 24 November 2026 on one
NVIDIA A100 (no symmetry cache, after one throwaway warm-up run on the GPU,
four runs side by side on the node; paths shortened to repository paths).
The native CHEBFF/SUBSPACE run solved the eight `D_2h` representations of
the 268,096-point grid separately and converged in 10 SCF iterations:

| Quantity | Python |
|---|---:|
| Total energy | -75.39740094 Ry |
| Charge-weighted SRE | 0.0001029167 |
| SCF time | 1.12 s |
| Total wall time | 2.32 s |

The input gives no `Cubic_Spline` line, so both tables are interpolated by
cubic splines, the default since 22 November 2026 (the log says so in its
header); since 23 November 2026 the Kleinman-Bylander projectors are
normalized by the quadrature of the logarithmic mesh (a line of the atom
data says so) and the CA correlation takes the coefficients that make it
continuous at `r_s = 1` (`PARSEC_KB_QUADRATURE`, `PARSEC_CA_CONTINUITY`);
since 24 November 2026 the local potentials and the projectors are
Fourier-filtered to the grid before they are sampled (`PARSEC_LOCAL_FILTER`,
a `Fourier filter` line of the atom data per species), which lowers this
total by `6.5e-3 Ry` (`-75.39088894 Ry` with `PARSEC_LOCAL_FILTER=off`).
That is the egg-box term of this placement on the 0.4-bohr grid: over five
placements of the molecule between the grid points the file's tables spread
its energy by `1.1e-2 Ry` and the filtered ones by `2.3e-3 Ry`, with the same
mean (`PARSEC_LOCAL_FILTER` in the
[accelerated README](../../src/parsec_python/acceleration/README.md)).
PARSEC 1.5 interpolates linearly and samples the tables as they are; with
`PARSEC_FORTRAN_NUMERICS=on` (PARSEC 1.5's numerics, code of 24 November 2026;
the same digits as on 23 November) the same command gives -75.38863333 Ry,
also in 10 SCF iterations.

## PARSEC reference

The upstream example did not include a reference output. A new reference was
therefore calculated from the unchanged WSL files with the matching PARSEC
OpenMPI build and eight MPI ranks. Copies are stored here as
`parsec_reference.out` and `parsec_reference_time.txt`.

PARSEC found the eight-operation `D_2h` subgroup and reduced the Hamiltonian
from 268,096 to 33,512 grid points per representation. It converged in 17 SCF
iterations to `-75.38863314 Ry`. The Python-minus-PARSEC total-energy
difference is `-0.00876780 Ry` (`-7.3e-4 Ry` per atom) with the defaults:
the Fourier filter of the tables (`-6.5e-3 Ry`), the cubic splines of these
tables (`-2.0e-3 Ry`), and the quadrature of the projector norms and the
continuous CA correlation (`-2.4e-4 Ry`). With PARSEC 1.5's numerics it is
`-0.00000019 Ry`.

See `COMPARISON.md` for the energy components, frontier eigenvalues, convergence
values, and timing caveats.
