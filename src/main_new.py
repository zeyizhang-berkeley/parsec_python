"""Clean RSDFT entry point.

This file intentionally stays small: it wires together input parsing,
problem setup, backend selection, output-file initialization, and the
actual RSDFT solve implemented in ``rsdft_solver.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rsdft_backend import select_solver_backend
from rsdft_input import has_diagmeth_override, load_elements, prepare_input_data
from rsdft_models import SolverSettings
from rsdft_output import initialize_output_file, write_rsdft_parameter_output
from rsdft_setup import prepare_system
from rsdft_solver import run_rsdft_calculation


def main(argv: list[str] | None = None):
    """Run one RSDFT job.

    Input:
        argv: Optional CLI arguments. When provided, ``argv[0]`` is treated
            as an input-file path. When omitted, the function uses
            ``sys.argv[1:]``.

    Output:
        Returns ``None``. Side effects are the generated output files and
        printed run diagnostics.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    base_dir = Path(__file__).resolve().parent

    # 1. Load default solver settings and periodic-table metadata.
    settings = SolverSettings()
    settings.normalize()

    elem, n_elements = load_elements(base_dir)
    cli_input_path = argv[0] if argv else None

    # 2. Gather geometry, charge state, density mode, and optional overrides.
    input_data = prepare_input_data(elem, settings, cli_input_path, base_dir)

    if input_data.density_method in ["ml", "sad_ml_grid"] and not has_diagmeth_override(input_data.settings_overrides):
        settings.diagmeth = 2
        print("No diagonalization override supplied for ML density input; using diagmeth=2 by default.")

    applied = settings.apply_overrides(input_data.settings_overrides)
    if applied:
        print("Applied solver setting overrides:")
        for name, value in applied.items():
            print(f"  {name}: {value}")

    # 3. Select CPU/GPU implementations, then derive the simulation domain.
    backend = select_solver_backend(settings)
    print(f"Successfully loaded {len(input_data.atoms)} atomic species.")
    print(f"Using {backend.label.upper()} backend for available solver stages.")

    problem = prepare_system(input_data, settings, elem, n_elements)
    print(f"Atoms {problem.input_data.atoms}")
    print(f"n_atom {problem.input_data.n_atom}")
    print(f"Z_charge {problem.input_data.z_charge}")
    print(f"Grid spacing (h): {problem.h}")
    print(f"Number of eigenvalues (nev): {problem.nev}")
    print(f"Domain: {problem.domain}")

    # 4. Initialize the log/output files and launch the actual SCF solve.
    initialize_output_file(problem.paths.output_file, backend.label, problem.input_data.density_method)
    write_rsdft_parameter_output(
        problem.paths.output_file,
        problem.nev,
        problem.input_data.atoms,
        problem.input_data.n_atom,
        problem.domain,
        problem.h,
        settings.poldeg,
        settings.fd_order,
    )
    run_rsdft_calculation(problem, elem, n_elements, backend)


if __name__ == "__main__":
    main()
