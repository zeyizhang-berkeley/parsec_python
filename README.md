# PARSEC Python CPU

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
