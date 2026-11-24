"""Forces of the local pseudopotential and the core density on a periodic grid (``forpbc.f90``, ``force_loc_pb.f90``).

With ``d = r - R_a - T`` over the periodic images ``T``, the two grid integrals acting on atom ``a`` are

``F_ion  = integral rho(r)  grad_r V_a(r - R_a)``
``F_core = integral V_xc(r) * sum_T rho_core'(|d|) d/|d|``

``F_ion`` differentiates the local ionic potential of :func:`parsec_python.V_ion.ewald.ewald_local_ionic_potential`
term by term: the short-ranged ``V_a(d) + 2 Z erf(eta d)/d`` over the image balls, and the reciprocal series
``-2 Z (4 pi/V) sum_(G != 0) w(G) cos(G.(r - R_a))`` with ``w(G) = exp(-G^2/4 eta^2)/G^2``, whose gradient is
``+2 Z (4 pi/V) sum_G w(G) G sin(G.(r - R_a))``.  The constant it drops (the ``G = 0`` term) does not depend on the
positions.  ``F_core`` is the derivative of the exchange-correlation energy through the frozen core density of a
nonlinear core correction, over the image balls of :func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density`.

The local potential ignores ``Cubic_Spline`` as the Fortran periodic one does and always uses the cubic spline of the
table (see :data:`parsec_python.V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`); the core density follows ``Cubic_Spline``,
as ``corecd.f90``.  The radial slopes keep ``forcloc.f90``'s rule closer than the first sample (zero) but switch to the
Coulomb slope (the potential) or to zero (the core density) at ``rs(ns-1)``, where the sampled tables do, not at
``forcloc.f90``'s ``rs(npoint-6)``: for a table of fewer than eight samples that radius lies at or inside the first
sample, and every grid point took the Coulomb slope (forces of a six-sample test table off by 0.08 Ry/bohr, and NaN in
the sums of :mod:`parsec_python.acceleration.Forces.periodic_local` for an atom on a grid point).  Between the two radii
the slopes are now those of the spline the energy samples.  The POTRE and FHI tables of the tree put ``rs(npoint-6)``
beyond 65 bohr; a table that is the Coulomb tail to round-off from a few bohr on is summed by the ball sums of the
acceleration package only out to the larger of the Gaussian tail and that Coulomb onset (10 to 25 bohr), so its forces
keep their bits there.  A table that is not the Coulomb tail to round-off before its end is summed out to ``rs(ns-1)``
(``examples/h2_full_nonlocal/H_POTRE.DAT``: onset ``rs(ns-1)`` = 115.6 bohr, ``rs(npoint-6)`` = 108.6 bohr; the forces
of an H2 cell with it change by up to 3.2e-5 Ry/bohr), and the readable sums below (every grid point for every image)
reach ``rs(npoint-6)`` in large cells (a 72-bohr Si cube at fixed smooth fields: 1.5e-2 of 97 Ry/bohr).

Closer than the first radial sample the slopes keep ``forcloc.f90``'s zero unless ``PARSEC_SPLINE_FORCE_CENTER=auto``
gives them the smooth behaviour of the table, which it does for a table whose first sample lies below 3e-3 bohr (every
POTRE table of the tree, and the FHI tables, which ``PARSEC_FHI_ORIGIN=auto`` continues to 1e-5 bohr).  A coarser table
(the six-sample test tables, 0.1 bohr; the H and He FHI tables under ``PARSEC_FHI_ORIGIN=off``, 6.25e-3 and 3.1e-3
bohr) leaves a grid point inside its first sample out, where the energy has the slope of the padded spline (a
hexagonal cell with the test core table and a grid point 0.06 bohr from an atom: 8.9e-5 Ry/bohr off the slope of the
energy; the Fortran-derived routines take the spline slope there).

Fortran's periodic path (``force_loc_pb.f90``) has no term for the SCF residual, so ``self_consistency`` is zero here;
the forces are first-order accurate in the density, as the Hellmann-Feynman forces need a converged SCF.

Every integral is the plain grid sum times the volume element.  This is the reference (dense) evaluation, with the
cost of the reference local potential: atoms x grid points x images.  The FFT version belongs to the
acceleration package.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..Pseudopotential.radial_derivatives import (
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    linear_slope_setting,
    smooth_center_radius,
    spline_force_center_setting,
)
from ..V_ion import ionic_potential as _ionic
from ..V_ion.ewald import ewald_local_image_sets, local_potential_uses_spline
from .local import LocalForceTerms, species_force_tables


def _erf_over_r_slope(charge: float, radius: np.ndarray, eta: float) -> np.ndarray:
    """``d/dr [2 Z erf(eta r)/r]``, smooth through ``r = 0`` (where it vanishes)."""

    radius = np.asarray(radius, dtype=float)
    result = np.zeros_like(radius)
    far = radius > 1.0e-8
    r = radius[far]
    result[far] = 2.0 * charge * (2.0 * eta * np.exp(-((eta * r) ** 2)) / (np.sqrt(np.pi) * r) - erf(eta * r) / (r * r))
    return result


# Like the local ionic potential (:func:`parsec_python.V_ion.ewald.set_local_potential_provider`), the sums can be made
# by a faster routine with the same result: the acceleration package installs the ball sums and the one FFT of
# :mod:`parsec_python.acceleration.Forces.periodic_local` when it is imported.  ``dense_periodic_local_forces`` is the
# readable definition they are tested against.
_local_force_provider = None


def set_local_force_provider(provider) -> None:
    """Install (or, with ``None``, remove) the routine that :func:`periodic_local_forces` hands the sums to.

    The provider takes the arguments of :func:`periodic_local_forces` after ``eta`` is left out and returns its
    :class:`LocalForceTerms`, or ``None`` to decline.
    """

    global _local_force_provider
    _local_force_provider = provider


def periodic_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
    xc_potential: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms:
    """Local-pseudopotential force terms of a periodic cell, ``(atoms, 3)`` in Ry/bohr.

    ``density`` is the valence density of the last SCF step in electrons/bohr**3 and ``xc_potential`` the
    exchange-correlation potential made from it plus the frozen core density.  An installed provider forms the sums
    (unless the splitting parameter ``eta`` is given, which only the readable routine takes).
    """

    if _local_force_provider is not None and eta is None:
        provided = _local_force_provider(
            grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, tolerance=tolerance
        )
        if provided is not None:
            return provided
    return dense_periodic_local_forces(
        grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, eta=eta, tolerance=tolerance
    )


def dense_periodic_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
    xc_potential: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms:
    """The sums of :func:`periodic_local_forces` as defined: over every grid point and image, and every reciprocal vector."""

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    density = np.asarray(density, dtype=np.float64)
    xc = np.asarray(xc_potential, dtype=np.float64)
    for name, field in (("density", density), ("xc_potential", xc)):
        if field.shape != (grid.size,):
            raise ValueError(f"{name} must hold one value per grid point")

    count = len(atoms)
    ionic = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    if count == 0:
        return LocalForceTerms(ionic, np.zeros((0, 3)), core_correction)
    volume = float(abs(np.linalg.det(lattice_vectors)))
    eta, real_cutoff, cell_center, cell_radius, all_translations, reciprocal_points = ewald_local_image_sets(
        lattice_vectors, eta, tolerance
    )
    g_squared = np.sum(reciprocal_points**2, axis=1) if reciprocal_points.size else None
    weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared if g_squared is not None else None
    half_width = grid.settings.stencil_half_width
    tables: dict = {}
    center = spline_force_center_setting()
    slopes = linear_slope_setting()  # PARSEC_LINEAR_TABLE_SLOPES (the core density of a species with linear tables)

    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        if atom.symbol not in tables:
            tables[atom.symbol] = species_force_tables(potential, specification)
        species = tables[atom.symbol]
        charge = potential.ionic_charge
        position = np.asarray(atom.position, dtype=np.float64)

        reaching = np.linalg.norm(position + all_translations - cell_center, axis=1) <= real_cutoff + cell_radius
        for translation in all_translations[reaching]:
            relative = grid.coordinates - (position + translation)
            radius = np.linalg.norm(relative, axis=1)
            slope = evaluate_local_potential_derivative(
                potential,
                radius,
                specification.local_angular_momentum,
                table=species.local_potential_derivative,
                use_spline=local_potential_uses_spline(specification),
                spline_padding_width=half_width,
                center=smooth_center_radius(potential, center),
                # the Coulomb slope where the sampled potential becomes -2 Z/r (not forcloc.f90's rs(npoint-6))
                tail=potential.interpolation_cutoff,
            ) + _erf_over_r_slope(charge, radius, eta)
            inverse = np.zeros_like(radius)
            positive = radius > 0.0
            inverse[positive] = 1.0 / radius[positive]
            ionic[index] += (slope * density * inverse) @ relative

        if reciprocal_points.size:
            phase = (grid.coordinates - position) @ reciprocal_points.T
            sine_density = density @ np.sin(phase)  # (G,)
            ionic[index] += 2.0 * charge * (4.0 * np.pi / volume) * ((weight * sine_density) @ reciprocal_points)

        if species.core_density_derivative is not None:
            support = _ionic.atomic_density_support_radius(potential, specification, core=True)
            for translation in _ionic._periodic_image_translations(lattice_vectors, support):
                rows, relative, radius = _ionic._support_finder(grid, position + translation, support)
                inside = radius < support
                if not np.any(inside):
                    continue
                rows, relative, radius = rows[inside], relative[inside], radius[inside]
                slope = evaluate_density_derivative(
                    potential,
                    radius,
                    potential.core_density,
                    table=species.core_density_derivative,
                    use_spline=specification.use_spline,
                    spline_padding_width=half_width,
                    center=smooth_center_radius(potential, center),
                    slopes=slopes,
                    tail=potential.interpolation_cutoff,
                )
                inverse = np.zeros_like(radius)
                positive = radius > 0.0
                inverse[positive] = 1.0 / radius[positive]
                core_correction[index] += (slope * xc[rows] * inverse) @ relative

    cell_volume_element = grid.volume_element
    return LocalForceTerms(
        ionic=ionic * cell_volume_element,
        self_consistency=np.zeros((count, 3)),
        core_correction=core_correction * cell_volume_element,
    )


__all__ = ["dense_periodic_local_forces", "periodic_local_forces", "set_local_force_provider"]
