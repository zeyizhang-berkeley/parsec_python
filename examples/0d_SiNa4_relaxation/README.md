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
of 22 November 2026 on one NVIDIA A100, 2.4 s):

| Structure | Fortran PARSEC | This package, PARSEC boundary | This package, default | PARSEC boundary and linear tables |
|---:|---|---|---|---|
| 0 | -23.45045, 0.37619 | -23.45152, 0.37621 | -23.45153, 0.37621 | -23.45045, 0.37619 |
| 1 | -24.82806, 0.15740 | -24.82897, 0.15749 | -24.82901, 0.15749 | -24.82806, 0.15740 |
| 2 | -25.23815, 0.05513 | -25.23934, 0.05504 | -25.23942, 0.05504 | -25.23815, 0.05513 |
| 3 | -25.31114, 0.01485 | -25.31203, 0.01481 | -25.31216, 0.01481 | -25.31114, 0.01485 |

The sphere is small, so the default Hartree boundary of this package adds the
atomic tail that PARSEC's multipole expansion leaves out, and the energies
move by up to about 1e-4 eV per atom. The input line

```text
Hartree_Boundary_Tolerance: off
```

restores PARSEC's boundary ("PARSEC boundary"). The input gives no
`Cubic_Spline` lines, so both tables are interpolated by cubic splines, the
default since 22 November 2026; PARSEC 1.5 interpolates linearly, and the
splines lower the energies by 0.9e-3 to 1.2e-3 eV per atom. With PARSEC's
boundary and

```text
PARSEC_CUBIC_SPLINE_DEFAULT=off PARSEC_PROJECTOR_TAIL=off
```

in the environment (last column) every printed digit of the Fortran
trajectory is reproduced. See "Physical scope" in the
[package README](../../src/parsec_python/README.md).

`occup.in` holds two sections. A spin-unpolarized run reads the first one
only, as Fortran PARSEC does, and says so in a warning.
