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
`--log parsec_python.out`, run with the code of 22 November 2026 on one
NVIDIA A100 (paths shortened to repository paths). H2 on the z axis has the
point group D4h on the grid (16 signed permutations); since 18 November 2026
(`PARSEC_ORBITAL_GROUP_MAX_COST=0.75`) the orbitals use the six weighted
one-dimensional sectors of a subgroup of order 8 (weights 1, 1, 2, 1, 1, 2),
where the run of 5 November used the eight sectors of the abelian D2h. With
the CHEBFF-first/SUBSPACE-later eigensolver the calculation converged in 17
SCF iterations (20 on the D2h route of 5 November, with the same total
energy):

| Quantity | Python result |
|---|---:|
| Total energy | -2.13713415 Ry |
| Occupied eigenvalue | -0.7196461976 Ry |
| Plain SRE | 0.0031773148 |
| Charge-weighted SRE | 0.0000777048 |

The input gives no `Cubic_Spline` line, so the H table is interpolated by
cubic splines, the default since 22 November 2026 (the log says so in its
header; PARSEC 1.5 interpolates linearly). On this dense 863-point table
the difference is small: with `PARSEC_CUBIC_SPLINE_DEFAULT=off
PARSEC_PROJECTOR_TAIL=off`, PARSEC 1.5's linear tables, the run has the
same total energy and SCF history, and the occupied eigenvalue is
-0.7196461994 Ry and the plain SRE 0.0031773149.

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
PARSEC run of this exact input exists (see below):

| Quantity | Python, DG=1 | Stock PARSEC, DG=3 | Python - PARSEC |
|---|---:|---:|---:|
| Total energy [Ry] | -2.13713415 | -2.13713410 | -0.00000005 |
| Occupied eigenvalue [Ry] | -0.7196461976 | -0.7194900830 | -0.0001561146 |
| Eigenvalue energy [Ry] | -1.43929240 | -1.43898017 | -0.00031223 |
| Hartree energy [Ry] | 2.49259088 | 2.49240374 | 0.00018714 |
| Integral of `Vxc*rho` [Ry] | -1.63991186 | -1.63979721 | -0.00011465 |
| Exchange-correlation energy [Ry] | -1.25644225 | -1.25635504 | -0.00008721 |
| Electron-ion energy [Ry] | -6.74648298 | -6.74608367 | -0.00039931 |
| Ion-ion energy [Ry] | 1.41113867 | 1.41113867 | 0.00000000 |

With PARSEC 1.5's linear tables (the two `off` switches above) only two
entries change, in the last digits: the occupied eigenvalue to
-0.7196461994 Ry and the electron-ion energy to -6.74648299 Ry.

The total energy is stationary; the separate terms and the occupied level
follow the potential at which each SCF stops within the criterion of
2e-4 Ry (the Python level moves by 8e-5 Ry between its last two iterations).

PARSEC currently segfaults in its `nonloc` routine before SCF when this
one-channel H potential is run with double-grid order one, both on the WSL
native filesystem and on the Windows-mounted filesystem. Therefore this
directory does not contain a valid `parsec_reference.out` and must not be used
for an order-one reproduction claim. The near agreement with stock PARSEC is
encouraging; a controlled canonical comparison would run PARSEC's unmodified
input in both programs. With `Double_Grid_Order: 3` this input gives the
Python energies above bit for bit, since the double grid averages only the
nonlocal projectors and this potential has none.
