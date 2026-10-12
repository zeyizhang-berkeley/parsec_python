"""Semilocal parts of the B3LYP exchange-correlation energy on the grid.

B3LYP is evaluated here as an *energy only*, for a density that another
functional made self-consistent (the PP-PBE(B3LYP) refinement of Xu,
Prendergast and Qian, J. Chem. Theory Comput. 18, 5471 (2022); ``Lrefine``
in ARES).  No potential is returned: nothing in this module enters a
Hamiltonian.  The definition is the one of Gaussian, libxc
(``HYB_GGA_XC_B3LYP``) and ARES,

``E_xc = 0.20 E_x[HF] + 0.08 E_x[Slater] + 0.72 E_x[B88]
        + 0.19 E_c[VWN-RPA] + 0.81 E_c[LYP]``,

where ``E_x[B88]`` is the complete Becke 1988 exchange (Slater plus its
gradient correction) and the local correlation is the Vosko-Wilk-Nusair fit
to the *RPA* correlation energy of the uniform gas ("VWN1RPA" in ARES,
functional III of Gaussian), not the fit to the Ceperley-Alder data (VWN5).
The exact (Hartree-Fock) exchange is the subject of
:mod:`parsec_python.V_xc.exact_exchange`; this module provides the four
density functionals.

All four are written for a spin-unpolarized density ``n = n_up + n_down``
with ``n_up = n_down``, the only case the solver has.  Their gradient
variable is ``sigma = |D n|**2`` with the same centered finite-difference
first derivative ``D`` of the active grid, extended by zero outside the
domain, that :mod:`parsec_python.V_xc.pbe` uses.

Densities are in electrons/bohr**3.  Pointwise functions return energies per
volume in Hartree atomic units; :func:`b3lyp_semilocal_energies` integrates
them and converts to the Rydberg convention of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from ..Grid import RealSpaceGrid
from .pbe import _embed_active, _gather_active, first_derivative_coefficients


# Mixing coefficients of B3LYP (Stephens, Devlin, Chabalowski, Frisch 1994).
B3LYP_EXACT_EXCHANGE = 0.20
B3LYP_SLATER_EXCHANGE = 0.08
B3LYP_B88_EXCHANGE = 0.72
B3LYP_VWN_CORRELATION = 0.19
B3LYP_LYP_CORRELATION = 0.81

# Slater (Dirac) exchange of the unpolarized gas: e_x = -C_x n**(4/3).
_SLATER_CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
# Becke 1988: beta of the gradient correction, fitted to the rare-gas atoms.
_B88_BETA = 0.0042
# Lee-Yang-Parr 1988 (Colle-Salvetti) constants.
_LYP_A = 0.04918
_LYP_B = 0.132
_LYP_C = 0.2533
_LYP_D = 0.349
_LYP_CF = 0.3 * (3.0 * np.pi * np.pi) ** (2.0 / 3.0)
# Vosko-Wilk-Nusair 1980, paramagnetic gas, Hartree: (A, x0, b, c).  "rpa"
# is their fit to the RPA correlation energy (the B3LYP of Gaussian, libxc
# and ARES), "vwn5" their fit to the Ceperley-Alder Monte Carlo energies.
_VWN_PARAMETERS = {
    "rpa": (0.0310907, -0.409286, 13.0720, 42.7198),
    "vwn5": (0.0310907, -0.10498, 3.72744, 12.9352),
}
# The vacuum branch of pbe.py: below it every energy density is zero.
_DENSITY_THRESHOLD = 1.0e-20


def _checked(density: np.ndarray, sigma: np.ndarray | None = None):
    density = np.asarray(density, dtype=np.float64)
    if np.any(density < -1.0e-14):
        raise ValueError("the functional requires a nonnegative density")
    if sigma is None:
        return density, None
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != density.shape:
        raise ValueError("density and squared gradient must have the same shape")
    if np.any(sigma < -1.0e-14):
        raise ValueError("the functional requires a nonnegative squared gradient")
    return density, np.maximum(sigma, 0.0)


def slater_exchange_energy_density(density: np.ndarray) -> np.ndarray:
    """Slater exchange energy per volume, ``-(3/4)(3/pi)**(1/3) n**(4/3)``."""

    density, _ = _checked(density)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    energy[positive] = -_SLATER_CX * n * np.cbrt(n)
    return energy


def b88_exchange_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Becke 1988 exchange energy per volume, Slater part included.

    For each spin density ``n_s = n/2`` with ``x = |grad n_s| / n_s**(4/3)``,

    ``e_s = -(3/2)(3/(4 pi))**(1/3) n_s**(4/3)
            - beta n_s**(4/3) x**2 / (1 + 6 beta x asinh(x))``,

    and the unpolarized energy density is ``2 e_s``
    (A. D. Becke, Phys. Rev. A 38, 3098 (1988)).
    """

    density, sigma = _checked(density, sigma)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    spin_density_43 = (0.5 * n) * np.cbrt(0.5 * n)
    # |grad n_s| = |grad n| / 2.
    x = 0.5 * np.sqrt(sigma[positive]) / spin_density_43
    correction = _B88_BETA * x * x / (1.0 + 6.0 * _B88_BETA * x * np.arcsinh(x))
    energy[positive] = -_SLATER_CX * n * np.cbrt(n) - 2.0 * spin_density_43 * correction
    return energy


def vwn_correlation_energy_density(
    density: np.ndarray,
    parametrization: str = "rpa",
) -> np.ndarray:
    """Vosko-Wilk-Nusair correlation energy per volume of the unpolarized gas.

    With ``x = sqrt(r_s)``, ``X(x) = x**2 + b x + c`` and
    ``Q = sqrt(4c - b**2)`` the energy per electron is (S. H. Vosko, L. Wilk
    and M. Nusair, Can. J. Phys. 58, 1200 (1980), Eq. 4.4)

    ``eps_c = A { ln(x**2/X) + (2b/Q) atan(Q/(2x+b))
                 - (b x0/X(x0)) [ ln((x-x0)**2/X)
                                  + (2(b+2 x0)/Q) atan(Q/(2x+b)) ] }``.

    ``parametrization`` is ``"rpa"`` (the B3LYP of Gaussian, libxc and ARES)
    or ``"vwn5"``.
    """

    try:
        a, x0, b, c = _VWN_PARAMETERS[parametrization]
    except KeyError:
        raise ValueError("VWN parametrization must be 'rpa' or 'vwn5'") from None
    density, _ = _checked(density)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    x = np.sqrt(np.cbrt(3.0 / (4.0 * np.pi * n)))
    big_x = x * x + b * x + c
    big_x0 = x0 * x0 + b * x0 + c
    q = np.sqrt(4.0 * c - b * b)
    arctangent = np.arctan(q / (2.0 * x + b))
    epsilon = a * (
        np.log(x * x / big_x)
        + (2.0 * b / q) * arctangent
        - (b * x0 / big_x0)
        * (
            np.log((x - x0) * (x - x0) / big_x)
            + (2.0 * (b + 2.0 * x0) / q) * arctangent
        )
    )
    energy[positive] = n * epsilon
    return energy


def lyp_correlation_energy_density(density: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Lee-Yang-Parr correlation energy per volume of a closed shell.

    The gradient-only form of B. Miehlich, A. Savin, H. Stoll and H. Preuss,
    Chem. Phys. Lett. 157, 200 (1989), Eq. 2, with ``n_up = n_down = n/2``
    and ``|grad n_up|**2 = |grad n_down|**2 = sigma/4`` reduces to

    ``e_c = -a n / (1 + d n**(-1/3))
            - a b exp(-c n**(-1/3)) / (1 + d n**(-1/3))
              * [ C_F n - n**(-5/3) sigma (1/24 + 7 delta/72) ]``,

    ``delta = c n**(-1/3) + d n**(-1/3) / (1 + d n**(-1/3))`` and
    ``C_F = (3/10)(3 pi**2)**(2/3)``.
    """

    density, sigma = _checked(density, sigma)
    energy = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    n = density[positive]
    inverse_cube_root = 1.0 / np.cbrt(n)
    denominator = 1.0 + _LYP_D * inverse_cube_root
    delta = _LYP_C * inverse_cube_root + _LYP_D * inverse_cube_root / denominator
    gradient_term = (
        inverse_cube_root**5 * sigma[positive] * (1.0 / 24.0 + 7.0 * delta / 72.0)
    )
    energy[positive] = (
        -_LYP_A * n / denominator
        - _LYP_A
        * _LYP_B
        * np.exp(-_LYP_C * inverse_cube_root)
        / denominator
        * (_LYP_CF * n - gradient_term)
    )
    return energy


def density_gradient_squared(density: np.ndarray, grid: RealSpaceGrid) -> np.ndarray:
    """Return ``sigma = |D n|**2`` on the active grid.

    ``D`` is the centered first derivative of the order of the grid, with the
    density extended by zero outside the active domain: the gradient of
    :func:`parsec_python.V_xc.pbe`.
    """

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active real-space grid")
    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing
    full_density = _embed_active(density, grid)
    sigma = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        derivative_box = correlate1d(
            full_density, weights, axis=axis, mode="constant", cval=0.0
        )
        gradient = _gather_active(derivative_box, grid)
        sigma += gradient * gradient
    return sigma


@dataclass(frozen=True)
class B3LYPSemilocalEnergies:
    """Integrated density-functional parts of B3LYP, in Rydberg.

    ``b88_exchange`` is the complete B88 exchange, Slater part included.
    ``vwn_correlation`` belongs to ``vwn_parametrization`` (``"rpa"`` for
    B3LYP).
    """

    slater_exchange: float
    b88_exchange: float
    vwn_correlation: float
    lyp_correlation: float
    vwn_parametrization: str = "rpa"


def b3lyp_semilocal_energies(
    valence_density: np.ndarray,
    grid: RealSpaceGrid,
    core_density: np.ndarray | None = None,
    *,
    vwn_parametrization: str = "rpa",
) -> B3LYPSemilocalEnergies:
    """Integrate Slater, B88, VWN and LYP for one density of the grid.

    ``core_density`` is the frozen nonlinear-core-correction density.  It is
    added to the valence density exactly as the self-consistent functional
    adds it (:func:`parsec_python.V_xc.pbe`), so that the refinement replaces
    the exchange-correlation energy of one and the same density.
    """

    valence = np.asarray(valence_density, dtype=np.float64)
    if valence.shape != (grid.size,):
        raise ValueError("density does not match the active real-space grid")
    if core_density is None:
        density = valence
    else:
        core = np.asarray(core_density, dtype=np.float64)
        if core.shape != valence.shape:
            raise ValueError("core and valence densities must have the same shape")
        density = valence + core
    if np.any(density < -1.0e-14):
        raise ValueError("the functional requires a nonnegative density")
    density = np.maximum(density, 0.0)
    sigma = density_gradient_squared(density, grid)
    # Hartree -> Rydberg at the public boundary, as in pbe.py.
    scale = 2.0 * grid.volume_element

    def integral(values: np.ndarray) -> float:
        return float(scale * np.sum(values))

    return B3LYPSemilocalEnergies(
        slater_exchange=integral(slater_exchange_energy_density(density)),
        b88_exchange=integral(b88_exchange_energy_density(density, sigma)),
        vwn_correlation=integral(
            vwn_correlation_energy_density(density, vwn_parametrization)
        ),
        lyp_correlation=integral(lyp_correlation_energy_density(density, sigma)),
        vwn_parametrization=vwn_parametrization,
    )


def b3lyp_exchange_correlation_energy(
    exact_exchange: float,
    semilocal: B3LYPSemilocalEnergies,
) -> float:
    """Combine the five parts with the B3LYP weights (same unit as the input)."""

    if semilocal.vwn_parametrization != "rpa":
        raise ValueError(
            "B3LYP is defined with the RPA parametrization of VWN, not "
            f"{semilocal.vwn_parametrization!r}"
        )
    return float(
        B3LYP_EXACT_EXCHANGE * exact_exchange
        + B3LYP_SLATER_EXCHANGE * semilocal.slater_exchange
        + B3LYP_B88_EXCHANGE * semilocal.b88_exchange
        + B3LYP_VWN_CORRELATION * semilocal.vwn_correlation
        + B3LYP_LYP_CORRELATION * semilocal.lyp_correlation
    )


__all__ = [
    "B3LYP_B88_EXCHANGE",
    "B3LYP_EXACT_EXCHANGE",
    "B3LYP_LYP_CORRELATION",
    "B3LYP_SLATER_EXCHANGE",
    "B3LYP_VWN_CORRELATION",
    "B3LYPSemilocalEnergies",
    "b3lyp_exchange_correlation_energy",
    "b3lyp_semilocal_energies",
    "b88_exchange_energy_density",
    "density_gradient_squared",
    "lyp_correlation_energy_density",
    "slater_exchange_energy_density",
    "vwn_correlation_energy_density",
]
