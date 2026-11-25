# SiNa4 cluster: BFGS relaxation with fixed occupations

The test `tests/relaxation` of Fortran PARSEC, unchanged. It exercises three
input features beside the relaxation itself:

- an old-format Martins pseudopotential, `Si_POTR.DAT`
  (`Pseudopotential_Format: martins_old` with `Core_Cutoff_Radius`,
  `Potential_Num` and the block `Electron_Per_Orbital`);
- occupations read from `occup.in` (`Fermi_Temp: -1`);
- a coarse grid (0.8 bohr) in a small sphere (4 Angstrom), so it runs in
  seconds.

```powershell
python src\parsec_python\main.py examples\0d_SiNa4_relaxation\parsec.in --no-archive
```

The run computes four structures and ends with exit status 0.

## Expected trajectory

`atom_cor_fortran_reference.out` is the `atom.cor` of Fortran PARSEC for this
input. Energy per atom in eV and largest force in Ry/bohr (this package: code
of 24 November 2026 on one NVIDIA A100, 2.2 s after one throwaway warm-up
run on the GPU, three other runs on the node):

| Structure | Fortran PARSEC | This package, PARSEC boundary | This package, default | `PARSEC_FORTRAN_NUMERICS=on` |
|---:|---|---|---|---|
| 0 | -23.45045, 0.37619 | -23.51274, 0.37473 | -23.51275, 0.37473 | -23.45045, 0.37619 |
| 1 | -24.82806, 0.15740 | -24.87791, 0.15004 | -24.87796, 0.15004 | -24.82806, 0.15740 |
| 2 | -25.23815, 0.05513 | -25.24949, 0.06047 | -25.24957, 0.06047 | -25.23815, 0.05513 |
| 3 | -25.31114, 0.01485 | -25.34355, 0.01830 | -25.34368, 0.01830 | -25.31114, 0.01485 |

The sphere is small, so the default Hartree boundary of this package adds the
atomic tail that PARSEC's multipole expansion leaves out, and the energies
move by up to about 1.3e-4 eV per atom. The input line

```text
Hartree_Boundary_Tolerance: off
```

restores PARSEC's boundary ("PARSEC boundary"). The input gives no
`Cubic_Spline` lines, so both tables are interpolated by cubic splines, the
default since 22 November 2026; PARSEC 1.5 interpolates linearly, and the
splines lower the energies by 0.9e-3 to 1.2e-3 eV per atom. The quadrature
of the projector norms on the logarithmic mesh and the continuous CA
correlation, the defaults since 23 November 2026, raise them by 1e-5 to
2e-5 eV per atom. Since 24 November 2026 the local potentials and the
projectors are Fourier-filtered to the grid before they are sampled
(`PARSEC_LOCAL_FILTER`): on this coarse grid that lowers the first structure
by 6.1e-2 eV per atom, and the relaxation, again four structures, ends 3.2e-2
eV per atom below the run with `PARSEC_LOCAL_FILTER=off` (-25.31214 eV per
atom, the default column of 23 November). Neither form is converged at 0.8
bohr: the first structure on finer grids (both forms, `Force_Min: 10` for a
single point) gives -23.4921 (0.4 bohr), -23.4821 (0.3) and -23.4725 eV per
atom (0.2 bohr), the forms within 5e-4 eV per atom of each other there; at
0.8 bohr the file's tables lie 2.1e-2 eV per atom above the value at 0.2
bohr and the filtered ones 4.0e-2 below it, at 0.6 bohr 1.2e-2 and 1.8e-2
below it. With

```text
PARSEC_FORTRAN_NUMERICS=on
```

in the environment (last column; it gives an input with a radius PARSEC's
boundary, and PARSEC 1.5's unfiltered tables, Kleinman-Bylander quadrature and CA
coefficients) every printed digit of the Fortran trajectory is reproduced
(code of 24 November 2026, as on 23 November; on 22 November PARSEC's boundary with
`PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off` did the same). See "Physical scope" in the
[package README](../../src/parsec_python/README.md).

`occup.in` holds two sections. A spin-unpolarized run reads the first one
only, as Fortran PARSEC does, and says so in a warning.
