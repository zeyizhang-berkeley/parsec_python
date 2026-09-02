# PARSEC.py

PARSEC.py is a native Python implementation of isolated, real-space
Kohn--Sham DFT based on the algorithms in the PARSEC Fortran code. It reads
PARSEC-style `parsec.in` files and PARSEC `*_POTRE.DAT` norm-conserving
pseudopotentials directly. It does not call PARSEC, WSL, MPI, or Fortran at
runtime.

The maintained solver now has one canonical package: `src/parsec_python`.
Readable scientific modules and optimized execution backends live in that
package together. Normal calculations automatically select the fastest
accuracy-preserving combination available on the machine.

The former implementation is preserved in the GitHub branch
[`old-architecture`](https://github.com/zeyizhang-berkeley/parsec_python/tree/old-architecture)
and is intentionally absent from the default `main` branch.

## Quick start

Python 3.12 and the repository's `.venv312` environment are recommended on
Windows:

```powershell
cd C:\Users\zigze\Desktop\parsec_python
.\.venv312\Scripts\Activate.ps1
python -m pip install -r src\parsec_python\requirements.txt
```

Place `parsec.in` and its pseudopotentials in one calculation directory:

```text
calculation/
├── parsec.in
├── C_POTRE.DAT
└── H_POTRE.DAT
```

From the repository root, validate an input without building the grid:

```powershell
python src\parsec_python\main.py path\to\calculation\parsec.in --dry-run
```

Run the calculation:

```powershell
python src\parsec_python\main.py path\to\calculation\parsec.in --no-archive
```

The default text result is `parsec.out` beside the input. Unless
`--no-archive` is used, arrays and metadata are also stored in
`parsec_python_results.npz`.

When `src` is importable, the equivalent package command is:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m parsec_python path\to\calculation\parsec.in --no-archive
```

For example:

```powershell
python src\parsec_python\main.py `
  examples\0d_naphthalene\parsec.in `
  --no-archive
```

The launcher uses `--backend auto` and `--symmetry auto` by default. On a
configured workstation this normally combines C++/OpenMP preparation and
Hartree kernels with a CuPy Hamiltonian/eigensolver. It falls back to an exact
supported backend when an optional accelerator is unavailable, and records
every selection in `parsec.out`.

Useful options include:

| Option | Meaning |
|---|---|
| `--backend auto|scipy|native|cupy` | Select automatic hybrid execution or a controlled comparison backend. |
| `--symmetry auto|on|off` | Detect exact supported symmetries, require them, or force the full grid. |
| `--no-symmetry-cache` | Disable persistent exact-key representation caches. |
| `--pp-dir DIR` | Search an additional directory for `*_POTRE.DAT`. |
| `--dry-run` | Parse and validate without constructing the grid. |
| `--no-archive` | Write only the text output. |
| `--profile-operator` | Time individual Hamiltonian actions. |
| `--resident` | Submit to a warmed local worker while creating fresh SCF state. |
| `--debug` | Re-raise failures with a traceback. |

## Source layout

```text
examples/                          runnable calculations and benchmark data
├── README.md                      case guide and execution commands
├── h2_canonical_nodg/
├── h2_full_nonlocal/
├── 0d_benzene/
├── 0d_naphthalene/
├── 0d_Si28H36/
└── 0_CH4_CF4/
src/
├── parsec_python/                 canonical DFT package
│   ├── main.py                    default command-line launcher
│   ├── __main__.py                python -m parsec_python
│   ├── driver.py                  readable preparation/reference workflow
│   ├── reference_main.py          explicit readable CLI launcher
│   ├── models.py                  shared physical/settings/result data
│   ├── Input/                     parsec.in translation and validation
│   ├── Grid/                      isolated real-space domains
│   ├── Laplacian/                 high-order finite differences
│   ├── Pseudopotential/           PARSEC POTRE reader and radial splines
│   ├── V_ion/                     local potential and nonlocal projectors
│   ├── Hartree/                   Poisson solve and open-boundary data
│   ├── V_xc/                      CA/PZ LDA and spin-unpolarized PBE
│   ├── Hamiltonian/               matrix-free Kohn--Sham operator
│   ├── Eigensolvers/              PARSEC-style ChebFF/CHEBDAV/subspace code
│   ├── Occupations/               Fermi level, occupations, and density
│   ├── Mixer/                     SCF potential mixing
│   ├── Energy/                    total-energy bookkeeping
│   ├── SCF/                       self-consistent single-point loop
│   ├── Output/                    PARSEC-shaped text reporting
│   ├── acceleration/              internal optimized implementation
│   │   ├── backends/              SciPy, CuPy, and native adapters
│   │   ├── Eigensolvers/          device-resident GPU eigensolvers
│   │   ├── Symmetry/              automatic representation decomposition
│   │   ├── native/                optional C++/OpenMP extension
│   │   ├── resident.py            warmed-process execution
│   │   └── tests/                 backend, parity, and GPU tests
│   └── tests/                     scientific and architecture tests
├── pp_generation/                 general FHI98PP generation package
└── tools/
    └── upf_to_parsec.py           UPF-to-PARSEC converter
```

There are no longer separate `new_architecture` and
`accelerated_architecture` packages. `parsec_python.acceleration` is an
internal implementation layer, not a second public solver.
Runnable inputs and benchmark reference data are deliberately outside the
importable package under [`examples/`](examples/README.md).

## Scientific flow

The isolated single-point workflow is:

1. Parse the atomic geometry and numerical settings from `parsec.in`.
2. Read each species' `*_POTRE.DAT` file.
3. Center the cluster and build its spherical or box-shaped Cartesian grid.
4. Construct the high-order finite-difference representation of
   `-nabla^2`.
5. Interpolate and superpose local ionic potentials; build separable
   Kleinman--Bylander nonlocal projector factors.
6. Superpose atomic starting densities and normalize the electron count.
7. Detect exact supported spatial symmetry and, when useful, construct
   representation-sector operators.
8. For each SCF iteration, solve Poisson's equation for `V_H`, evaluate
   `V_xc`, apply

   `H = -nabla_FD^2 + V_ion,local + V_H + V_xc + V_NL`,

   filter/diagonalize the occupied subspace, rebuild occupations and density,
   mix the effective potential, and evaluate the energy.
9. Stop according to the translated PARSEC convergence criterion or
   `Max_Iter`, then write the PARSEC-shaped report.

The Hamiltonian and nonlocal potential remain matrix-free during iterative
eigensolution. Static terms are prepared once; only density-dependent local
fields change during SCF.

## Modular Python API

With `src` on `PYTHONPATH`, individual scientific pieces can be used without
running SCF:

```python
from parsec_python import (
    build_cluster_grid,
    build_negative_laplacian,
    ca_lda,
    pbe,
    read_parsec_pseudopotential,
    solve_hartree,
)
```

The public workflow functions use accelerated execution by default:

```python
from parsec_python import prepare_single_point, run_scf, run_single_point

system = prepare_single_point("calculation/parsec.in", backend="auto")
result = run_scf(system)
```

For audits and algorithm study, explicit aliases retain the readable SciPy
path:

```python
from parsec_python import (
    prepare_reference_single_point,
    run_reference_scf,
    run_reference_single_point,
)
```

## Acceleration policy

Acceleration changes execution, not the physical model:

- C++/OpenMP handles loop-heavy finite-difference setup, radial interpolation,
  multipole boundaries, CA-LDA, sparse Hamiltonian application, and conjugate
  gradient where profitable.
- CuPy keeps large orbital blocks, Chebyshev filtering, orthogonalization,
  Rayleigh--Ritz work, nonlocal projections, and suitable Poisson operations
  on the GPU.
- Exact symmetry-representation decomposition reduces independent sectors
  automatically; unsupported or absent symmetry safely uses the full domain.
- NumPy/SciPy remain the reference implementation and handle small dense
  operations where GPU launch/transfer costs would be slower.
- Backend decisions, fallbacks, timings, and symmetry choices are reported.

See [the acceleration guide](src/parsec_python/acceleration/README.md) and
[the optimization audit](src/parsec_python/acceleration/ACCELERATION_AUDIT.md)
for implementation and parity details.

## Optional native extension

The C++/OpenMP extension is optional but recommended for fastest automatic
execution. Its source is in `src/parsec_python/acceleration/native`. A built
wheel can be installed into `.venv312` with:

```powershell
python -m pip install --force-reinstall --no-deps `
  src\parsec_python\acceleration\native\dist\parsec_accelerated_native-*.whl
```

Verify the loaded binary:

```powershell
python -c "import parsec_accelerated_native as n; print(n.__file__); print(n.build_info())"
```

The OpenMP default uses detected logical processors minus four unless
`OMP_NUM_THREADS` explicitly overrides it.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s src\parsec_python\tests -p "test_*.py" -v
python -m unittest discover -s src\parsec_python\acceleration\tests -p "test_*.py" -v
python -m unittest discover -s src\pp_generation\tests -p "test_*.py" -v
```

The acceleration suite includes real CUDA tests when CuPy and a CUDA device
are available; otherwise those runtime-specific tests skip cleanly.

## Pseudopotential utilities

`src/pp_generation` is a general FHI98PP orchestration package, independent of
the DFT solver. It supports ordinary and core-hole configurations within the
capabilities of the installed generator. Generated potentials must still be
validated for ghost states, transferability, exchange-correlation consistency,
and relativistic requirements before production use.

Convert a norm-conserving UPF file to PARSEC format with:

```powershell
python src\tools\upf_to_parsec.py input.UPF output_POTRE.DAT
```

## Documentation

- [PARSEC algorithm map](src/parsec_python/PARSEC_ALGORITHM.md)
- [Python implementation map](src/parsec_python/PYTHON_IMPLEMENTATION.md)
- [Package architecture](src/parsec_python/ARCHITECTURE.md)
- [Acceleration implementation](src/parsec_python/acceleration/README.md)
- [Pseudopotential generation](src/pp_generation/README.md)

## License

See [LICENSE](LICENSE).
