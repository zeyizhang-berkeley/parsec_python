"""Optional Hamiltonian execution backends."""

from .base import BoundHamiltonian, HamiltonianBackend
from .cupy import (
    CuPyHamiltonian,
    CuPyUnavailableError,
    cupy_available,
    cupy_device_count,
)
from .native import (
    NativeConjugateGradientBackend,
    NativeHamiltonianBackend,
    build_native_negative_laplacian,
    native_available,
)
from .scipy import ScipyHamiltonianBackend
from .selection import BackendSelection, resolve_backend

__all__ = [
    "BackendSelection",
    "BoundHamiltonian",
    "CuPyHamiltonian",
    "CuPyUnavailableError",
    "HamiltonianBackend",
    "NativeConjugateGradientBackend",
    "NativeHamiltonianBackend",
    "ScipyHamiltonianBackend",
    "build_native_negative_laplacian",
    "cupy_available",
    "cupy_device_count",
    "native_available",
    "resolve_backend",
]
