"""CuPy-resident PARSEC CHEBFF, CHEBDAV, and later-SCF SUBSPACE primitives."""

from .chebdav import (
    ChebDavSettings,
    DeviceChebDavResult,
    DeviceChebDavState,
    run_chebdav,
)
from .chebff import (
    ChebFFCycle,
    ChebFFSettings,
    DeviceChebFFResult,
    DeviceChebFFState,
    run_chebff,
)
from .chebyshev import (
    FilterBlock,
    chebff_filter,
    chebyshev_filter,
    subspace_filter,
    subspace_filter_blocks,
)
from .eigval import (
    CuPyEigvalDeviceState,
    CuPyEigvalResult,
    CuPyEigvalSolver,
    EigvalSettings,
)
from .symmetry import (
    CuPySymmetryEigvalResult,
    CuPySymmetryEigvalState,
    CuPySymmetryOrbitals,
    CuPySymmetrySCFEigensolver,
)
from .lapack_random import LapackRandom, PARSEC_RANDOM_ARRAY_SEED
from .orthogonalize import DeviceOrthonormalizationResult, orthonormalize
from .rayleigh_ritz import (
    DeviceRayleighRitzResult,
    GeneralizedRitzStabilityError,
    generalized_rayleigh_ritz,
    generalized_ritz_requested,
    rayleigh_ritz,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound
from .subspace import (
    DeviceSubspaceResult,
    DeviceSubspaceState,
    SubspaceSettings,
    adapt_polynomial_degree,
    run_subspace_filter,
)

__all__ = [
    "ChebDavSettings",
    "ChebFFCycle",
    "ChebFFSettings",
    "CuPyEigvalDeviceState",
    "CuPyEigvalResult",
    "CuPyEigvalSolver",
    "CuPySymmetryEigvalResult",
    "CuPySymmetryEigvalState",
    "CuPySymmetryOrbitals",
    "CuPySymmetrySCFEigensolver",
    "DeviceChebFFResult",
    "DeviceChebFFState",
    "DeviceChebDavResult",
    "DeviceChebDavState",
    "DeviceOrthonormalizationResult",
    "DeviceRayleighRitzResult",
    "GeneralizedRitzStabilityError",
    "DeviceSubspaceResult",
    "DeviceSubspaceState",
    "EigvalSettings",
    "FilterBlock",
    "LanczosBoundResult",
    "LapackRandom",
    "PARSEC_RANDOM_ARRAY_SEED",
    "SubspaceSettings",
    "adapt_polynomial_degree",
    "chebff_filter",
    "chebyshev_filter",
    "generalized_rayleigh_ritz",
    "generalized_ritz_requested",
    "lanczos_upper_bound",
    "orthonormalize",
    "rayleigh_ritz",
    "run_chebdav",
    "run_chebff",
    "run_subspace_filter",
    "subspace_filter",
    "subspace_filter_blocks",
]
