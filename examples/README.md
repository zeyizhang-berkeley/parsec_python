# Examples and benchmarks

This directory contains runnable PARSEC-style inputs, pseudopotentials, and
selected reference results. It sits beside `src/` so calculation data is kept
separate from the importable `parsec_python` package.

## Set up the environment

From the repository root in PowerShell:

```powershell
.\.venv312\Scripts\Activate.ps1
python -m pip install -r src\parsec_python\requirements.txt
```

The default `auto` backend will use the installed C++/OpenMP extension and
CuPy/CUDA runtime when available, and will report any fallback in the output.
The solver itself never calls the PARSEC Fortran executable.

## Run an example

The general command from the repository root is:

```powershell
python src\parsec_python\main.py examples\CASE\parsec.in --no-archive
```

For example, run naphthalene with automatic backend and symmetry selection:

```powershell
python src\parsec_python\main.py `
  examples\0d_naphthalene\parsec.in `
  --backend auto --symmetry auto --no-archive
```

You can also work directly inside a case directory. From
`examples\0d_naphthalene`:

```powershell
..\..\.venv312\Scripts\Activate.ps1
python ..\..\src\parsec_python\main.py parsec.in --no-archive
```

Each `parsec.in` expects its `<Atom_Type>_POTRE.DAT` files in the same
directory unless `--pp-dir DIRECTORY` is supplied.

Before starting a large calculation, validate the input and resolve the
backend without building the grid:

```powershell
python src\parsec_python\main.py examples\0d_Si28H36\parsec.in --dry-run
```

## Available cases

| Directory | System and purpose | Suggested starting command |
|---|---|---|
| `h2_canonical_nodg` | H2 with PARSEC's canonical H potential and double-grid order one; small introductory case. | `python src\parsec_python\main.py examples\h2_canonical_nodg\parsec.in --no-archive` |
| `h2_full_nonlocal` | H2 with a full nonlocal POTRE and paired PARSEC reference output; best small component-parity case. | `python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --no-archive` |
| `0d_benzene` | Twelve-atom benzene cluster with exact symmetry; medium physical and timing comparison. | `python src\parsec_python\main.py examples\0d_benzene\parsec.in --no-archive` |
| `0d_naphthalene` | Eighteen-atom naphthalene benchmark used for GPU/C++ and symmetry optimization. | `python src\parsec_python\main.py examples\0d_naphthalene\parsec.in --no-archive` |
| `0d_Si28H36` | Larger hydrogen-passivated silicon cluster; use this after validating the environment on a smaller case. | `python src\parsec_python\main.py examples\0d_Si28H36\parsec.in --no-archive` |
| `0d_nanodiamonds` | Hydrogen-terminated nanodiamonds from C1H4 to C9449H1572 (up to 39,368 valence electrons): Fortran PARSEC comparisons up to C795H300 and the multi-GPU benchmark series with A100 reference times and memory. | See [`0d_nanodiamonds/README.md`](0d_nanodiamonds/README.md). |
| `0_CH4_CF4` | ARES comparison, FHI98PP-generated ordinary/core-hole potentials, and four spin-unpolarized PBE delta-SCF inputs. | See the core-hole commands below. |
| `0d_hybrid_functionals` | Self-consistent hybrids (PBE0, HSE06, HSE03, B3LYP, B3LYP5, B3PW91, HF) of H2, H2O, CH4, CO, N2, benzene and triplet O2 on the GPU, compared with Quantum ESPRESSO using the same FHI98PP potentials. | `python src\parsec_python\main.py examples\0d_hybrid_functionals\H2O\pbe0\parsec.in --pp-dir examples\0d_hybrid_functionals --no-archive`; see [`0d_hybrid_functionals/README.md`](0d_hybrid_functionals/README.md). |
| `3d_hybrid_functionals` | Self-consistent hybrids (PBE0, HSE06, HF, B3LYP) of periodic cells on the GPU: Si, C and MgO with k-point meshes, Si forces, an h-BN layer (slab) and triplet O2 in a box, compared with Quantum ESPRESSO using the same FHI98PP potentials and meshes; a 64-atom Si cell for timing. | `python src\parsec_python\main.py examples\3d_hybrid_functionals\Si_k222\pbe0\parsec.in --pp-dir examples\3d_hybrid_functionals --no-archive`; see [`3d_hybrid_functionals/README.md`](3d_hybrid_functionals/README.md). |
| `ml_initial_density` | Fourteen-molecule SAD/SCDP/ChargE3Net suite with portable predicted densities and CHEBFF/CHEBDAV reference outputs. | See [`ml_initial_density/README.md`](ml_initial_density/README.md). |
| `3d_Si` | Periodic boundary conditions for bulk silicon with an 8 atom conventional unit cell | `python src\parsec_python\main.py examples\3d_Si\parsec.in --no-archive` |
| `0d_meta_gga` | Meta-GGAs (the native SCAN; r2SCAN and TPSS of libxc) and meta-GGA hybrids (TPSSh, r2SCAN0) of H2O, CH4, NH3, N2, CO and triplet O2 on the GPU, compared with Quantum ESPRESSO using the same FHI98PP potentials (E - E_PBE, HOMO, exact exchange, forces of a distorted H2O, grid spacing 0.16 and 0.13 bohr); also the potentials of the meta-GGA unit tests. | `python src\parsec_python\main.py examples\0d_meta_gga\H2O\r2scan\parsec.in --pp-dir examples\0d_meta_gga --no-archive` (libxc: `PARSEC_LIBXC_LIBRARY`); see [`0d_meta_gga/README.md`](0d_meta_gga/README.md). |
| `3d_meta_gga` | Meta-GGAs (PBE, SCAN, r2SCAN, TPSS) of bulk Si (fcc, two atoms) on Gamma-centred 2x2x2 and 4x4x4 meshes, against Quantum ESPRESSO with the same FHI98PP potential (E - E_PBE, gap). | `python src\parsec_python\main.py examples\3d_meta_gga\Si_k444\r2scan\parsec.in --pp-dir examples\3d_meta_gga --no-archive`; see [`3d_meta_gga/README.md`](3d_meta_gga/README.md). |

`0d_naphthalene/gpu_cpp` is a retained self-contained historical run folder.
For new calculations, prefer `0d_naphthalene/parsec.in`; the canonical
launcher already selects GPU plus native C++ components automatically.

## Core-hole PBE examples

The Python calculations are under `0_CH4_CF4/python_pbe`. The initial-state
and C 1s core-hole final-state cases can be run independently:

```powershell
python src\parsec_python\main.py examples\0_CH4_CF4\python_pbe\CH4\IS\parsec.in --no-archive
python src\parsec_python\main.py examples\0_CH4_CF4\python_pbe\CH4\FS_1s\parsec.in --no-archive
python src\parsec_python\main.py examples\0_CH4_CF4\python_pbe\CF4\IS\parsec.in --no-archive
python src\parsec_python\main.py examples\0_CH4_CF4\python_pbe\CF4\FS_1s\parsec.in --no-archive
```

The final-state inputs use the explicit species label `C-1s` and therefore
resolve `C-1s_POTRE.DAT`. The supplied ARES directory is reference data, not
an executable dependency of the Python solver. See
[`0_CH4_CF4/python_pbe/README.md`](0_CH4_CF4/python_pbe/README.md) for the
scientific comparison, and
[`0_CH4_CF4/python_pbe_b3lyp/README.md`](0_CH4_CF4/python_pbe_b3lyp/README.md)
for B3LYP refinements of the same cases.

## Outputs, caches, and reproducibility

By default, outputs are placed beside the input:

- `parsec.out`: PARSEC-shaped text report;
- `parsec_python_results.npz`: arrays and metadata.

Use `--no-archive` when only the text report is needed. Use `--log NAME.out`
and `--output NAME.npz` to avoid overwriting an existing result:

```powershell
python src\parsec_python\main.py examples\0d_benzene\parsec.in `
  --log benzene_new.out --output benzene_new.npz
```

Automatic symmetry keeps no persistent cache unless a directory is named: a
first calculation of a structure is fastest without one. For repeated
calculations of the same structure and grid, `--symmetry-cache DIRECTORY`
keeps exact-key symmetry maps and operators there. The cache changes setup
time only, not the physical result or fresh SCF state. Earlier versions wrote
it to `.parsec_cache/symmetry` beside the input; that directory is no longer
read unless it is named:

```powershell
python src\parsec_python\main.py examples\0d_naphthalene\parsec.in `
  --symmetry-cache examples\0d_naphthalene\.parsec_cache\symmetry --no-archive
```

Selected `parsec_reference.out`, `parsec_python.out`, and comparison documents
are retained as validation evidence. The retained Python outputs that list
symmetry-cache entries were written while the cache beside the input was the
default; a run of the current code reports
`symmetry_cache_directory = disabled` instead.
The reference coverage is:

| Case | Python reference | Independent reference |
|---|---|---|
| `h2_canonical_nodg` | `parsec_python.out` | Canonical PP/input regression |
| `h2_full_nonlocal` | `parsec_python.out` | `parsec_reference.out` from Fortran PARSEC |
| `0d_benzene` | `parsec_python.out` | `parsec_reference.out` from Fortran PARSEC |
| `0d_naphthalene` | `parsec_python.out` | `parsec_reference.out` and `parsec_reference_time.txt` from the 28-rank Fortran run |
| `0d_Si28H36` | `parsec_python.out` | `parsec_reference.out` and `parsec_reference_time.txt` from the 30-rank Fortran run |
| `0_CH4_CF4/python_pbe` | One `parsec_python.out` in each IS/FS directory | Corresponding ARES PBE `ares.log` files and the comparison in its README |
| `ml_initial_density` | CHEBDAV `parsec.out` and retained CHEBFF `parsec_chebff.out` | Paired SAD/SCDP/ChargE3Net regression data |

The default numerics of the Python solver depart from PARSEC 1.5 where that
is more accurate: cubic splines of the radial tables with the exact Coulomb
tail, projectors that end at their cutoff, the exact Kleinman-Bylander
quadrature on the logarithmic mesh, the continuous Perdew-Zunger correlation,
FHI98PP tables continued to the nucleus, local potentials and projectors
Fourier-filtered to the grid (which removes most of the dependence of the
total on where the atoms sit on the grid), smooth LDA+U projectors, the Fermi
level of a gapped system at the root of its electron count, and Hartree
boundary values at a multipole order chosen by an estimate from the geometry
(`src/parsec_python/README.md`, `src/parsec_python/acceleration/README.md`).
`PARSEC_FORTRAN_NUMERICS=on` restores PARSEC 1.5's numerics in one go,
including PARSEC's boundary for an input that gives its domain; the Fortran
comparisons quoted in the example READMEs are reproduced under it. Of the
cases above, `0d_Si28H36` gives -263.51147821 Ry under it against
-263.51147856 Ry of its Fortran reference, and -263.52977897 Ry with the
defaults (filtered tables, multipole order 10 with the atomic tail). Its
`parsec_python.out` (-263.51946553 Ry) was made with the code of 22 November
2026 and PARSEC's boundary; `PARSEC_HARTREE_BOUNDARY=legacy` with
`PARSEC_KB_QUADRATURE=trapezoid`, `PARSEC_CA_CONTINUITY=off`,
`PARSEC_FERMI_GAP_ROOT=off`, `PARSEC_LOCAL_FILTER=off` and
`PARSEC_SPLINE_COULOMB_TAIL=off` reproduces it.

Example outputs are made again when a change of the code concerns them
directly (their subject, a comparison their README states, or their tables).
The others keep the outputs of their date: later default changes move their
totals far less than chemical accuracy (those of 23 November 2026 by up to
about 2.3e-3 Ry per Ag or Au atom and 1e-6 to 3e-4 Ry per light atom; the
grid filter of 24 November 2026 by the egg-box term of their placement,
typically 1e-5 to 4e-4 Ry per atom on the grids of the examples, more for a
species far from resolved, such as Au at h 0.5 bohr) and add lines such as
the Kleinman-Bylander quadrature and the Fourier filter of each species to
`parsec.out`.

Reference timings are hardware-specific; energies and convergence histories
are the portable validation quantities. Ordinary new outputs, NumPy result
archives, CUDA caches, and `.parsec_cache` directories are intentionally
ignored by Git. The older readable-Python naphthalene run is preserved locally
under `0d_naphthalene/reference_readable`; the current optimized input remains
at the case root.

## Comparing execution modes

Use explicit backends only for controlled comparisons:

```powershell
python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --backend scipy --symmetry off --no-archive
python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --backend native --symmetry auto --no-archive
python src\parsec_python\main.py examples\h2_full_nonlocal\parsec.in --backend cupy --symmetry auto --no-archive
```

For ordinary work, `--backend auto --symmetry auto` is recommended. Compare
energies, convergence histories, grid settings, pseudopotential hashes, and
accuracy before comparing wall time.
