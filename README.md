# PARSEC.py

Python version of the real-space DFT workflow used in this repository.

This package currently contains two entry paths:

- `src/main.py`: the older monolithic driver.
- `src/main_new.py`: the refactored driver with separated input/setup/output/solver modules.

The recent cleanup work in this repo was done around `main_new.py`. If you want the cleaner code path, use `main_new.py`.

## Repo Layout

- `src/main_new.py`: small orchestrator for the refactored workflow.
- `src/rsdft_input.py`: manual input, file input, unit conversion, ML-grid path resolution.
- `src/rsdft_setup.py`: grid spacing, domain construction, recentering, output-path generation.
- `src/rsdft_backend.py`: CPU/GPU backend selection.
- `src/rsdft_solver.py`: the real-space DFT calculation and SCF loop.
- `src/rsdft_output.py`: output log, density saves, timing summaries, optional `wfn.dat`.
- `src/GUI/gui_input_generator.py`: GUI for generating `.in` input files.
- `src/elements_new.csv`: element lookup table used during setup.

## Environment Setup

This package does not currently ship a lockfile or `requirements.txt`, so the
environment needs to be created manually.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it using your shell's normal virtual-environment activation command,
then install the core runtime packages used by the current code:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib
```

For GPU runs, also install a CuPy build that matches your CUDA setup.

If Matplotlib or CuPy complain about cache or temp-directory permissions on
your machine, point the following environment variables to writable folders:

- `MPLCONFIGDIR`
- `TEMP`
- `TMP`
- `CUPY_CACHE_DIR`

### PowerShell quick start

If you are using PowerShell and want the environment activated in your current
shell, load the repo helper script with dot-sourcing:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\Enter-ParsecEnv.ps1
```

That command:

- activates the `parsec_python` Conda environment when available
- otherwise falls back to `.\.conda_env`
- sets `MPLCONFIGDIR`, `CUPY_CACHE_DIR`, `TEMP`, and `TMP`
- clears inherited `PYTHONPATH` / `PYTHONHOME` values that can interfere with the env
- sets `CUDA_PATH` if CUDA 13.0 is installed in the default Windows location
- changes into `src\`

After that, run:

```powershell
python main_new.py
```

To stay in the repo root instead of changing into `src\`, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\Enter-ParsecEnv.ps1 -RepoRoot
python .\src\main_new.py
```

## Native C++ Scaffold

This repo now includes an additive scaffold for future C++/OpenMP versions of:

- `pseudoDiag`
- `pseudoNL_original`

The scaffold lives in:

- `src/native/`
- `src/V_ion/pseudoDiag_cpp.py`
- `src/V_ion/pseudoNL_original_cpp.py`

Important:

- the existing Python implementations are unchanged
- the new native entry points are not wired into the default backend yet
- the C++ functions are placeholders that currently raise a clear runtime error
- the purpose of this scaffold is to establish the build system, module layout,
  and wrapper API before porting the hot loops

### Build prerequisites on Windows

To build the extension from source you will need:

- Visual Studio Build Tools with MSVC C++
- OpenMP support
- CMake
- Ninja

These tools were not available on `PATH` in the current shell, so the scaffold
was added but not compiled here.

### Build commands

From the repo root inside your Python environment:

```powershell
python -m pip install --upgrade pip
python -m pip install scikit-build-core pybind11
python -m pip install -v .
```

After a successful build, you can inspect the extension with:

```powershell
python -c "import rsdft_native; print(rsdft_native.build_info())"
```

The expected result now looks like:

```python
{'scaffold': False, 'implemented_kernels': ('pseudo_diag_omp', 'pseudo_nl_omp'), 'openmp_enabled': True, 'openmp_max_threads': 8}
```

### Wrapper usage

Once the extension is built, the additive wrappers can be imported directly:

```python
from V_ion.pseudoDiag_cpp import pseudoDiag
from V_ion.pseudoNL_original_cpp import pseudoNL
```

Those wrappers intentionally mirror the future native call sites without
replacing the existing Python modules.

### Enable native kernels in the driver

The refactored driver now defaults to the native C++/OpenMP ionic kernels when
the `rsdft_native` extension is installed. To run explicitly with the native
diagonal and nonlocal setup, you can still set:

```powershell
$env:PARSEC_NATIVE_PSEUDODIAG = "1"
$env:PARSEC_NATIVE_PSEUDONL = "1"
$env:OMP_NUM_THREADS = "8"
python src\main_new.py --cpu samples\h\h_smoke.json
```

You can also enable only one native kernel:

```powershell
$env:PARSEC_NATIVE_PSEUDODIAG = "1"
Remove-Item Env:PARSEC_NATIVE_PSEUDONL -ErrorAction SilentlyContinue
python src\main_new.py --cpu samples\h\h_smoke.json
```

To force the legacy Python ionic kernels for comparison, set either variable to
`0` before running:

```powershell
$env:PARSEC_NATIVE_PSEUDODIAG = "0"
$env:PARSEC_NATIVE_PSEUDONL = "0"
python src\main_new.py --cpu samples\h\h_smoke.json
```

## Running The Refactored Driver

From this package directory:

```bash
python src/main_new.py
```

That starts the interactive/manual-input flow.

To run from an input file:

```bash
python src/main_new.py path/to/your_input.in
```

Supported input formats:

- `.in`
- `.inp`
- `.json`
- `.txt`
- `.dat`
- `.mat`

## Input Modes

`main_new.py` supports two input modes:

- Manual terminal entry.
- File-driven input from one of the supported formats above.

For manual input, the program can optionally prompt for solver/grid overrides such as:

- `tol`
- `maxits`
- `fd_order`
- `Fermi_temp`
- `poldeg`
- `diagmeth`
- `CG_prec`
- `adaptiveScheme`
- `use_gpu`
- `recenter_atoms`
- `nev`
- `grid_spacing`
- `sphere_radius`

## Density Initialization Modes

Supported density initialization values are:

- `sad`
- `ml`
- `sad_ml_grid`

Behavior:

- `sad`: superposition of atomic densities on the default atom-derived grid.
- `ml`: use an ML-predicted density/grid `.npy` file.
- `sad_ml_grid`: use SAD density on a grid defined by `.npy + POSCAR`.

If `density_method` is `ml` or `sad_ml_grid` and the user does not explicitly set `diagmeth`, the refactored driver defaults to `diagmeth = 2`.

## Recenter Option

The refactored driver now supports a dedicated setting:

```text
recenter_atoms = 0
```

or

```text
recenter_atoms = 1
```

Notes:

- Default is `1`.
- It only affects the `sad` + default-grid path.
- It is ignored for ML-grid-based runs.

The GUI generator also exposes the same setting with a checkbox labeled:

- `Recenter atoms for default SAD grid`

## Output Behavior

### File-driven runs

If you launch from an input file, the main `.out` file is written next to that input file.

### Manual-input runs

If you launch through manual input, the refactored driver now creates a folder named after the run basename and places the generated files inside it.

Example pattern:

```text
./H2O_sad_diagmeth3_5A_0p2A/
```

Inside that folder you may see:

- `<base>.out`
- `<base>_init_rho_grid.npy`
- `<base>_init_rho_bohr3.npy`
- `<base>_init_rho.npy`
- `<base>_conv_rho_grid.npy`
- `<base>_conv_rho_bohr3.npy`
- `<base>_conv_rho.npy`
- `<base>_wfn.dat` if `save_wfn = 1`

## GUI Input Generator

Launch it from this package directory:

```bash
python src/GUI/gui_input_generator.py
```

The GUI can:

- Load `.xyz` and POSCAR-like structure files.
- Export `.in` files for the solver.
- Set density mode and optional ML-grid files.
- Set numerical options such as `diagmeth`, `save_wfn`, `use_gpu`, and `recenter_atoms`.

## Notes On GPU

The refactored code can select GPU implementations with:

```text
use_gpu = 1
```

but only if the required GPU modules import successfully in the current environment.

If GPU support is missing, the run exits with a clear error message listing the missing GPU modules.

## Suggested Comparison Workflow

If you want to compare the old and new drivers on the same case:

1. Activate your Python environment.
2. Run the old path with `src/main.py`.
3. Run the refactored path with `src/main_new.py` on the same input.
4. Compare:
   - total energies
   - convergence behavior
   - generated density files
   - optional `wfn.dat`

## Current Status

`main.py` was intentionally left untouched during the refactor so the old workflow remains available.

The current recommended development path is:

- use `src/main_new.py`
- keep new changes in the refactored modules
- use the GUI generator for consistent `.in` file generation
