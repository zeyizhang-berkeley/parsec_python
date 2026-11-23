"""Forces of the local pseudopotential on an isolated grid (``forcloc.f90``).

Three grid integrals act on atom ``a`` at ``R_a``.  With ``d = r - R_a``:

``F_ion  = integral rho(r)         * V_local'(|d|) * d/|d|``
``F_scf  = integral dV_hxc(r)      * rho_atom'(|d|) * d/|d|``
``F_core = integral V_xc(r)        * rho_core'(|d|) * d/|d|``

``F_ion`` is the Hellmann-Feynman force of the local electron-ion energy
(Kronik et al., J. Chem. Phys. 115, 4322 (2001), Eq. 9).  ``F_core`` is the
derivative of the exchange-correlation energy through the frozen core density
of a nonlinear core correction.  ``F_scf`` corrects for an SCF loop that
stopped short of its fixed point (Chan, Bohnen and Ho, Phys. Rev. B 47, 4771
(1993)): ``dV_hxc`` is the Hartree-plus-XC potential made from the last
density minus the one that entered the last eigenproblem, which is the last
``V_out - V_in`` because the ionic potential cancels.

Every integral is the plain grid sum times the volume element.  Grid points
closer to the atom than its first radial sample are left out of all three
terms, as in PARSEC, under ``PARSEC_SPLINE_FORCE_CENTER=tip``; under ``auto``
(the default) every point within ``2 rs(2)`` of the atom
takes the slope of the smooth behaviour of the local potential and of the core
density, ``s(c) d/c``
(:func:`parsec_python.Pseudopotential.radial_derivatives.spline_force_center_setting`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..Grid import RealSpaceGrid
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..Pseudopotential.radial_derivatives import (
    density_derivative_table,
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    linear_slope_setting,
    local_potential_derivative_table,
    smooth_center_radius,
    spline_force_center_setting,
    valence_density_table,
)


@dataclass(frozen=True)
class LocalForceTerms:
    """The three ``forcloc.f90`` terms, each ``(atoms, 3)`` in Ry/bohr."""

    ionic: np.ndarray
    self_consistency: np.ndarray
    core_correction: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.ionic + self.self_consistency + self.core_correction


@dataclass(frozen=True)
class SpeciesForceTables:
    """Radial tables one species contributes to the local forces."""

    local_potential_derivative: np.ndarray
    valence_density: np.ndarray
    valence_density_derivative: np.ndarray
    core_density_derivative: np.ndarray | None


def species_force_tables(
    potential: ParsecPseudopotential, specification: SpeciesPotential
) -> SpeciesForceTables:
    """Build the derivative tables of one species (``pseudo.f90``)."""

    valence = valence_density_table(
        potential, read_from_file=specification.read_valence_density
    )
    core = (
        density_derivative_table(potential, potential.core_density)
        if potential.has_nonlinear_core_correction
        else None
    )
    return SpeciesForceTables(
        local_potential_derivative=local_potential_derivative_table(
            potential, specification.local_angular_momentum
        ),
        valence_density=valence,
        valence_density_derivative=density_derivative_table(potential, valence),
        core_density_derivative=core,
    )


def local_forces(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    density: np.ndarray,
    potential_residual: np.ndarray,
    xc_potential: np.ndarray,
) -> LocalForceTerms:
    """Evaluate the local-pseudopotential force terms on the full grid.

    ``density`` is the valence density of the last SCF step in
    electrons/bohr**3, ``potential_residual`` its ``V_out - V_in`` in Ry and
    ``xc_potential`` the XC potential made from ``density`` plus the frozen
    core density.

    The atomic valence density of the self-consistency term is interpolated
    linearly also for a species with ``Cubic_Spline``: ``forcloc.f90`` passes
    the derivative table together with the spline coefficients of the
    density itself there, which does not give a derivative of the density.

    The radial slopes of a species with linear tables (no ``Cubic_Spline``) follow
    ``PARSEC_LINEAR_TABLE_SLOPES``
    (:func:`parsec_python.Pseudopotential.radial_derivatives.linear_slope_setting`):
    the tables of ``pseudo.f90`` by default, the slopes of the linear
    interpolants the energy samples under ``interpolant``.
    """

    density = np.asarray(density, dtype=np.float64)
    residual = np.asarray(potential_residual, dtype=np.float64)
    xc = np.asarray(xc_potential, dtype=np.float64)
    for name, field in (
        ("density", density),
        ("potential_residual", residual),
        ("xc_potential", xc),
    ):
        if field.shape != (grid.size,):
            raise ValueError(f"{name} must hold one value per grid point")

    count = len(atoms)
    ionic = np.zeros((count, 3))
    self_consistency = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    tables: dict[str, SpeciesForceTables] = {}
    half_width = grid.settings.stencil_half_width
    volume = grid.volume_element
    center_setting = spline_force_center_setting()
    slopes = linear_slope_setting()

    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        if atom.symbol not in tables:
            tables[atom.symbol] = species_force_tables(potential, specification)
        species = tables[atom.symbol]

        relative = grid.coordinates - np.asarray(atom.position, dtype=np.float64)
        radius = np.sqrt(np.einsum("ij,ij->i", relative, relative))
        # within 2 rs(2) the slopes of the smooth behaviour (PARSEC_SPLINE_FORCE_CENTER=auto), which
        # also reach the points closer than rs(2); without it those are left out (forcloc.f90)
        center = smooth_center_radius(potential, center_setting)
        included = radius >= potential.radii[0] if center == 0.0 else radius > 0.0
        inverse = np.zeros_like(radius)
        inverse[included] = 1.0 / radius[included]

        potential_slope = evaluate_local_potential_derivative(
            potential,
            radius,
            specification.local_angular_momentum,
            table=species.local_potential_derivative,
            use_spline=specification.use_spline,
            spline_padding_width=half_width,
            direct=specification.old_interpolation,
            center=center,
            slopes=slopes,
        )
        ionic[index] = (potential_slope * density * inverse) @ relative

        density_slope = evaluate_density_derivative(
            potential,
            radius,
            species.valence_density,
            table=species.valence_density_derivative,
            slopes=slopes,
        )
        self_consistency[index] = (density_slope * residual * inverse) @ relative

        if species.core_density_derivative is not None:
            core_slope = evaluate_density_derivative(
                potential,
                radius,
                potential.core_density,
                table=species.core_density_derivative,
                use_spline=specification.use_spline,
                spline_padding_width=half_width,
                center=center,
                slopes=slopes,
            )
            core_correction[index] = (core_slope * xc * inverse) @ relative

    return LocalForceTerms(
        ionic=ionic * volume,
        self_consistency=self_consistency * volume,
        core_correction=core_correction * volume,
    )


__all__ = [
    "LocalForceTerms",
    "SpeciesForceTables",
    "local_forces",
    "species_force_tables",
]
