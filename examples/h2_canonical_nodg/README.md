# H2 with PARSEC's canonical pseudopotential (no double grid)

This directory preserves the full canonical hydrogen pseudopotential copied
from:

```text
<PARSEC_FORTRAN_ROOT>/tests/H2/H_POTRE.DAT
```

That file contains 863 radial points. It is unrelated to the six-point
synthetic fixture under `tests/data`, which exists only for fast unit tests.

The stock PARSEC H2 input uses `Double_Grid_Order: 3`. This input sets the
order to one; the Python solver runs the Ono-Hirose double grid too, but it
averages only the nonlocal projectors, and this one-channel H potential has
none, so the order does not change this calculation.

## Run the Python calculation

From the repository root:

```powershell
python src\parsec_python\main.py examples\h2_canonical_nodg\parsec.in --no-archive
```

The calculation writes a PARSEC-style text report to `parsec.out` in
this directory. The retained `parsec_python.out` is this command with
`--log parsec_python.out`, run with the code of 24 November 2026 on one
NVIDIA A100 (no symmetry cache, after one throwaway warm-up run on the GPU;
paths shortened to repository paths). H2 on the z axis has the
point group D4h on the grid (16 signed permutations); since 18 November 2026
(`PARSEC_ORBITAL_GROUP_MAX_COST=0.75`) the orbitals use the six weighted
one-dimensional sectors of a subgroup of order 8 (weights 1, 1, 2, 1, 1, 2),
where the run of 5 November used the eight sectors of the abelian D2h. With
the CHEBFF-first/SUBSPACE-later eigensolver the calculation converged in 19
SCF iterations:

| Quantity | Python result |
|---|---:|
| Total energy | -2.24251534 Ry |
| Occupied eigenvalue | -0.7441864646 Ry |
| Plain SRE | 0.0020026608 |
| Charge-weighted SRE | 0.0001277575 |

This H table is a Troullier-Martins potential with a core radius of 0.15
bohr, close to the bare Coulomb potential of the proton, on a grid of 0.378
bohr. Since 24 November 2026 the local potential is Fourier-filtered to the
grid before it is sampled (`PARSEC_LOCAL_FILTER`; the atom data of the log
says so and notes that the grid is coarse for this table, whose filtered
form departs from the file's by 9.8 Ry inside 0.15 bohr). Sampled as the
file gives it, the table aliases strongly: the same command under
`PARSEC_LOCAL_FILTER=off` gives -2.13713415 Ry in 17 iterations (the result
of the code of 23 November, occupied eigenvalue -0.7196461204 Ry), 0.105 Ry
higher. The filtered energy is the nearer one to the limit of fine grids
(the same input on one A100 with smaller `Grid_Spacing`):

| `Grid_Spacing` [ang] | 0.20 | 0.15 | 0.10 | 0.05 |
|---|---:|---:|---:|---:|
| Filtered (default) [Ry] | -2.24251534 | -2.26853263 | -2.27239350 | -2.27508500 |
| File's table (`PARSEC_LOCAL_FILTER=off`) [Ry] | -2.13713415 | -2.20831581 | -2.23707460 | (-2.2776 after the 50 steps of the input, not converged) |

The file's table approaches the limit erratically (-2.23365698 Ry at 0.075
ang, above its value at 0.10 ang); the filtered one falls monotonically. On
the grid of this input the filtered energy lies 0.033 Ry above the filtered
value at 0.05 ang, the file's 0.138 Ry.

The input gives no `Cubic_Spline` line, so the H table is interpolated by
cubic splines, the default since 22 November 2026 (the log says so in its
header; PARSEC 1.5 interpolates linearly), and since 23 November 2026 the CA
correlation takes the coefficients that make it continuous at `r_s = 1`
(`PARSEC_CA_CONTINUITY`). On this dense 863-point table without projectors
those two make little difference: with `PARSEC_FORTRAN_NUMERICS=on` (PARSEC
1.5's numerics, its unfiltered linear tables and CA coefficients among them;
code of 24 November 2026, the same digits as on 23 November) the run gives
-2.13713415 Ry in 17 iterations, the occupied eigenvalue -0.7196461994 Ry,
the plain SRE 0.0031773149 and the charge-weighted SRE 0.0000777048.

The eigenvalue table lists 16 levels for `States_Num: 16`, as the full grid
does: the weighted state that holds the degenerate pair 16-17 counts for the
one state left (`PARSEC_WEIGHTED_STATE_COUNT`; under `off` it reports both
partners, 17 levels, with the same energy).
`PARSEC_ORBITAL_GROUP_MAX_COST=0.5` gives the D2h route of before (16 levels).

The report follows PARSEC's organization and labels for the grid, atoms,
pseudopotential, eigenstates, SCF iterations, energy decomposition, and SRE
values. It identifies itself as `PARSEC-PYTHON` and explicitly lists the
PARSEC-only results that this implementation does not calculate.

## Comparison with the stock PARSEC H2 result

The following comparison is useful as a numerical check, but it is **not a
strict reproduction test**: the stock PARSEC run uses double-grid order three,
which gives the same Python energies here (no nonlocal projectors), but no
PARSEC run of this exact input exists (see below). PARSEC samples the table
as the file gives it, so the comparison is made with PARSEC 1.5's numerics
(`PARSEC_FORTRAN_NUMERICS=on`):

| Quantity | Python, DG=1, PARSEC 1.5's numerics | Stock PARSEC, DG=3 | Python - PARSEC |
|---|---:|---:|---:|
| Total energy [Ry] | -2.13713415 | -2.13713410 | -0.00000005 |
| Occupied eigenvalue [Ry] | -0.7196461994 | -0.7194900830 | -0.0001561164 |
| Eigenvalue energy [Ry] | -1.43929240 | -1.43898017 | -0.00031223 |
| Hartree energy [Ry] | 2.49259088 | 2.49240374 | 0.00018714 |
| Integral of `Vxc*rho` [Ry] | -1.63991186 | -1.63979721 | -0.00011465 |
| Exchange-correlation energy [Ry] | -1.25644225 | -1.25635504 | -0.00008721 |
| Electron-ion energy [Ry] | -6.74648299 | -6.74608367 | -0.00039932 |
| Ion-ion energy [Ry] | 1.41113867 | 1.41113867 | 0.00000000 |

The total energy is stationary; the separate terms and the occupied level
follow the potential at which each SCF stops within the criterion of
2e-4 Ry (the Python level moves by 8e-5 Ry between its last two iterations).
With the defaults (the filtered table) the total lies 0.105 Ry below the
stock PARSEC result, by the aliasing of the file's table described above;
the electron-ion energy carries most of it (-7.03214727 Ry, 0.286 Ry below).

PARSEC currently segfaults in its `nonloc` routine before SCF when this
one-channel H potential is run with double-grid order one, both on the WSL
native filesystem and on the Windows-mounted filesystem. Therefore this
directory does not contain a valid `parsec_reference.out` and must not be used
for an order-one reproduction claim. The near agreement with stock PARSEC is
encouraging; a controlled canonical comparison would run PARSEC's unmodified
input in both programs. With `Double_Grid_Order: 3` this input gives the
Python energies above bit for bit, with the defaults and with PARSEC 1.5's
numerics, since the double grid averages only the nonlocal projectors and
this potential has none.
