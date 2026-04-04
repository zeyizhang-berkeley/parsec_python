"""Backend-selection helpers for the refactored RSDFT driver.

The job of this file is to expose one function, ``select_solver_backend()``,
which returns a ``SolverBackend`` bundle pointing at either the CPU or GPU
implementations of the diagonalization and Hartree-solver routines.
"""

from __future__ import annotations

import numpy as np

from Eigensolvers.chefsi1 import chefsi1 as chefsi1_cpu
from Eigensolvers.chsubsp import chsubsp as chsubsp_cpu
from Eigensolvers.first_filt import first_filt as first_filt_cpu
from Eigensolvers.lanczos import lanczos as lanczos_cpu
from Eigensolvers.occupations import occupations as occupations_cpu
from Eigensolvers.pcg import pcg as pcg_cpu
from V_ion.pseudoDiag import pseudoDiag as pseudo_diag_cpu
from V_ion.pseudoNL_original import pseudoNL as pseudo_nl_cpu

from rsdft_models import SolverBackend, SolverSettings

try:
    import cupy as cp
    from cupy._core import set_reduction_accelerators, set_routine_accelerators
except ImportError:
    cp = None
    set_reduction_accelerators = None
    set_routine_accelerators = None

try:
    from Eigensolvers.chefsi1_gpu import chefsi1 as chefsi1_gpu
except ImportError:
    chefsi1_gpu = None

try:
    from Eigensolvers.chsubsp_gpu import chsubsp as chsubsp_gpu
except ImportError:
    chsubsp_gpu = None

try:
    from Eigensolvers.first_filt_gpu import first_filt as first_filt_gpu
except ImportError:
    first_filt_gpu = None

try:
    from Eigensolvers.lanczos_gpu import lanczos as lanczos_gpu
except ImportError:
    lanczos_gpu = None

try:
    from Eigensolvers.occupations_gpu import occupations as occupations_gpu
except ImportError:
    occupations_gpu = None

try:
    from Eigensolvers.pcg_gpu import pcg as pcg_gpu
except ImportError:
    pcg_gpu = None

try:
    from V_ion.pseudoDiag_gpu import pseudoDiag as pseudo_diag_gpu
except ImportError:
    pseudo_diag_gpu = None

try:
    from V_ion.pseudoNL_original_gpu import pseudoNL as pseudo_nl_gpu
except ImportError:
    pseudo_nl_gpu = None


def configure_gpu_runtime() -> None:
    """Apply CuPy runtime settings needed by the GPU path."""
    if cp is None or set_reduction_accelerators is None or set_routine_accelerators is None:
        raise SystemExit("GPU backend requested but CuPy is not available.")
    set_reduction_accelerators([])
    set_routine_accelerators([])


def select_solver_backend(settings: SolverSettings) -> SolverBackend:
    """Return the CPU or GPU function bundle used by the solver.

    Input:
        settings: Normalized run settings. Only ``settings.use_gpu`` is used
            here, but the whole settings object is passed for clarity.

    Output:
        A ``SolverBackend`` dataclass containing the function references and
        array module for the selected execution path.
    """
    if settings.use_gpu:
        required_gpu_modules = {
            "cupy": cp,
            "pseudoDiag_gpu": pseudo_diag_gpu,
            "pseudoNL_original_gpu": pseudo_nl_gpu,
            "first_filt_gpu": first_filt_gpu,
            "chefsi1_gpu": chefsi1_gpu,
            "lanczos_gpu": lanczos_gpu,
            "chsubsp_gpu": chsubsp_gpu,
            "occupations_gpu": occupations_gpu,
            "pcg_gpu": pcg_gpu,
        }
        missing = [name for name, impl in required_gpu_modules.items() if impl is None]
        if missing:
            raise SystemExit(
                "GPU backend requested but the following GPU modules could not be imported: "
                + ", ".join(missing)
            )

        configure_gpu_runtime()
        return SolverBackend(
            label="gpu",
            pseudo_diag=pseudo_diag_gpu,
            pseudo_nl=pseudo_nl_gpu,
            first_filt=first_filt_gpu,
            chefsi1=chefsi1_gpu,
            lanczos=lanczos_gpu,
            chsubsp=chsubsp_gpu,
            occupations=occupations_gpu,
            pcg=pcg_gpu,
            array_module=cp,
            cupy_module=cp,
        )

    return SolverBackend(
        label="cpu",
        pseudo_diag=pseudo_diag_cpu,
        pseudo_nl=pseudo_nl_cpu,
        first_filt=first_filt_cpu,
        chefsi1=chefsi1_cpu,
        lanczos=lanczos_cpu,
        chsubsp=chsubsp_cpu,
        occupations=occupations_cpu,
        pcg=pcg_cpu,
        array_module=np,
        cupy_module=cp,
    )
