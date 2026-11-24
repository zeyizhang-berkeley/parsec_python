"""Atomic forces of a three-dimensional periodic (bulk) cell, in Ry/bohr.

Fortran PARSEC evaluates the forces of a bulk cell in ``forpbc.f90`` /
``force_loc_pb.f90`` (local pseudopotential and core correction in G space,
from the FFT of the density and of V_xc), ``ewald_sum.f90`` (ion-ion) and
``forcnloc.f90`` (Kleinman-Bylander projectors with their lattice replicas,
the Bloch phases ``right`` for k-points), then symmetrizes them over its
space group (``symm_forces.f90``) and removes the net force.  This module
has the same structure, but every term is the derivative of the energy this
port computes, which differs from Fortran's in its local potential:

* The local ionic potential of a periodic cell is the real-space Ewald split
  of :func:`parsec_python.V_ion.ewald_local_ionic_potential`,

  ``V(r) = sum_a sum_T f_a(|r - R_a - T|) + sum_a L_a(r - R_a)``,

  ``f_a(r) = V_a(r) + 2 Z_a erf(eta r)/r`` (lattice translations ``T`` within
  the real-space cutoff of that function) and
  ``L_a(x) = -2 Z_a (4 pi/V) sum_(G != 0) exp(-G^2/4 eta^2)/G^2 cos(G.x)``,
  not Fortran's G-space ``v_first``/``pot_local``.  Its force is

  ``F_a = dV sum_r rho(r) [sum_T f_a'(|d_T|) d_T/|d_T|
          + 2 Z_a (4 pi/V) sum_G w_G G sin(G.(r - R_a))]``

  with ``d_T = r - R_a - T`` and the same translations and G vectors as the
  potential (:func:`periodic_local_forces`).  With Fortran's G-space
  potential (``PeriodicGridSettings.local_potential = "g_space"``,
  :mod:`parsec_python.V_ion.gspace_local`) the local force is its derivative
  ``-(dV/Omega) sum_G v_a(|G|) G Im(P(G) exp(-i G.R_a))``, ``P(G)`` the FFT of
  the density (``force_loc_pb.f90``).
* The ion-ion force is the derivative of
  :func:`parsec_python.V_ion.ewald_ion_ion_energy` with its translations and
  G vectors (Andrew Grieder's :func:`parsec_python.Forces.ion_ion.ewald_ion_ion_forces`, which replaced the one
  of this module: the same sums to 1e-14).  Both real-space sums select
  their lattice images as the energy does, for every cell shape: the
  ion-ion sum the translations within the cutoff plus the diameter of the
  atom set (and skips an image no pair reaches), the local potential, per
  atom, the images whose centre lies within the cutoff plus the
  circumscribed radius of the cell from the centre of the cell.  The force
  is the derivative of the energy with exactly those terms.
* Radial functions are differentiated as the energy samples them: the slope
  of the linear interpolant (of ``r V`` for the local potential, of the
  density, of the projector) or the derivative of the cubic spline where the
  input selects one.  The forces of an isolated system
  (:mod:`parsec_python.Forces.local`, :mod:`.nonlocal_force`) interpolate
  instead PARSEC's tabulated slopes (``pseudo.f90``), which differ from the
  interpolant's by about a percent point by point.

The nonlinear-core-correction term ``dV sum_r V_xc(r) sum_T rho_c'(|d_T|)
d_T/|d_T|`` is the one of ``force_loc_pb.f90`` (``vsc * dnc``).  The
self-consistency correction of ``forcloc.f90`` (``(V_out - V_in)`` times the
slope of the atomic valence density) is added as for an isolated system; it
vanishes at self-consistency and ``forpbc.f90`` has no counterpart.

The nonlocal force for k-points uses the Bloch projectors of
:func:`parsec_python.V_ion.build_nonlocal_projectors`,
``beta^k(r) = sum_T exp(-i k.(r - T)) beta(r - R_a - T)``:

``F_a = 4 sum_k w_k sum_n f_nk sum_q s_q Re(conj(<beta^k_q|u_nk>) <grad beta^k_q|u_nk>)``,

``grad beta^k`` carrying the same phases (``forcnloc.f90``'s
``real(zvywd(0) conjg(zvywd(1:3)))`` with ``right``).  LDA+U (``J = 0``) and
TS-vdW forces are those of :func:`parsec_python.Forces.total.correction_forces`,
which already sum the lattice images.

Symmetrization: PARSEC averages over its space group ``symm``
(``symm_forces.f90``).  Where the problem carries that group
(``SinglePointInput.space_group``: ``Monkhorst_Pack_Shift``,
``Symmetrize_Charge_Density`` or a weighted ``Kpoint_List`` with forces) the
forces are averaged over its operations, non-symmorphic ones included
(:func:`space_group_symmetry_operations`); otherwise, as before,
:func:`periodic_symmetry_operations` keeps the point operations about the
origin that map the lattice, the grid, the atoms (modulo lattice vectors) and
the k-point set onto themselves.  The nonlocal forces of a k-point set reduced
to the irreducible wedge are completed over the group in any case
(:func:`reduced_kpoint_operations`).  A result of
``run_scf_kpoints_spin_polarized`` (collinear spin with k-points) sums both
channels; ``Double_Grid_Order > 1`` takes the projector gradients on the fine
points.  The torque is not reported for a periodic cell, as in
``forcnloc.f90``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.special import erf

from ..Grid.cell import fractional_coordinates
from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from ..Pseudopotential import radial_derivatives as _radial_derivatives
from ..Pseudopotential.radial_derivatives import (
    density_derivative_table,
    evaluate_density_derivative,
    evaluate_local_potential_derivative,
    local_potential_derivative_table,
    radial_projector_with_derivative,
    spline_derivative,
)
from ..V_ion.ewald import (
    _cell_center_and_radius,
    _ewald_cutoffs,
    _lattice_points,
    _reciprocal_lattice,
    local_potential_tolerance,
)
from ..V_ion import ionic_potential as _ionic_potential
from ..V_ion.ionic_potential import (
    NonlocalProjectorOperator,
    _periodic_image_translations,
    _projector_support_radius,
    build_nonlocal_projectors,
)
from .local import LocalForceTerms
from .nonlocal_force import (
    NonlocalProjectorGradients,
    gradient_columns_for,
    real_harmonics_with_gradients,
)
from .ion_ion import ewald_ion_ion_forces  # Andrew Grieder's (equal to 1e-14 in the comparison of both sets, the same time)
from .symmetry import POSITION_TOLERANCE, SpaceGroupAtomOperation, symmetrize_forces


# ---------------------------------------------------------------------------
# Radial slopes of the interpolants the energy uses.

#: How the forces of these routines differentiate the radial tables of a species with linear tables (``Cubic_Spline:
#: .false.``, or no value under ``PARSEC_CUBIC_SPLINE_DEFAULT=off``):
#: ``"interpolant"`` (default) the slopes of the linear interpolants the energy samples (local potential, densities,
#: projectors), the exact derivative of the computed energy; ``"tabulated"`` the slope table ``dvw`` of
#: ``pseudo.f90`` for the projectors only, as Fortran's ``forcnloc.f90`` (its nonlocal forces to 1e-6 Ry/bohr);
#: ``"tables"`` every slope from the tables of ``pseudo.f90`` (``dvion``, ``drhodr``, ``ddenc`` and ``dvw``,
#: interpolated linearly as ``forcloc.f90`` and ``forcnloc.f90`` do), the forces of the clusters and of the default
#: routines of a cell (``PARSEC_LINEAR_TABLE_SLOPES=tabulated``).  The energy of linear tables is rough on the scale
#: of a relaxation step, so its exact derivative is a noisy force.  On 2-atom fcc and 8-atom cubic Si cells at
#: h = 0.6-0.64 bohr the projector slopes differ by up to 1e-2 Ry/bohr.  The environment variable
#: ``PARSEC_PERIODIC_FORCE_SLOPES`` selects one for :func:`compute_periodic_forces`.
RADIAL_SLOPES = ("interpolant", "tabulated", "tables")
SLOPES_ENV = "PARSEC_PERIODIC_FORCE_SLOPES"


#: The round-off distance (bohr): under ``PARSEC_PROJECTOR_CENTER=off`` the distance below which ``1/|d|`` of a
#: projector gradient is clamped, at most the first radial knot (an atom within round-off of a grid point got the
#: gradient of this distance instead of ``1/1e-15``, found when both sets of routines were compared), and the radius
#: within which ``PARSEC_SPLINE_FORCE_CENTER=tip`` drops the spline slopes.  Far below any physical displacement,
#: far above round-off.  The projectors take the same rule for their harmonics under ``off``
#: (``V_ion.ionic_potential.PROJECTOR_CENTER_ROUND_OFF``); ``auto`` and ``tip`` clamp at the first radial knot.
ROUND_OFF_DISTANCE = _ionic_potential.PROJECTOR_CENTER_ROUND_OFF


def _projector_inverse_radius(radius: np.ndarray, potential: ParsecPseudopotential, setting: str):
    """``(1/|d|, inside)`` of the projector gradients with the slopes of the interpolant.

    ``PARSEC_PROJECTOR_CENTER=tip`` (:func:`parsec_python.V_ion.ionic_potential.projector_center_setting`): the
    distance clamped below to the first radial sample ``rs(2)`` (``nonloc.F90``'s ``rnorm``), for a point on the
    atom too, and ``inside`` the points closer than ``rs(2)``: their projector is ``w(rs(2)) P(d/rs(2))``, whose
    gradient ``w(rs(2)) dP/du / rs(2)`` (:func:`real_harmonics_with_gradients` with ``inside``) is its derivative for
    displacements within ``rs(2)``.  ``off``: ``min(rs(2), ROUND_OFF_DISTANCE)``, zero on the atom, as
    before (``inside`` ``None``).  (``auto`` does not come here: :func:`_interpolant_gradient_terms`.)  The 1/|d| of a
    round-off distance (an atom within round-off of a grid point) was
    first clamped to ``ROUND_OFF_DISTANCE`` because ``1/1e-15`` made ``w(r0) grad Y`` 1e15 (Si64, h 0.513: nonlocal
    forces of 82 Ry/bohr); with the true direction of the harmonics down to 1e-6 bohr that left ``w(rs(2))/1e-6``
    at such a point, 13 (carbon POTRE) to 1000 (FHI tables) times the slope of a p projector.
    """

    if setting == "tip":
        center = float(potential.radii[0])
        inside = radius < center
        return 1.0 / np.maximum(radius, center), (inside if np.any(inside) else None)
    inverse = np.zeros_like(radius)
    nonzero = radius > 0.0
    inverse[nonzero] = 1.0 / np.maximum(radius[nonzero], min(potential.radii[0], ROUND_OFF_DISTANCE))
    return inverse, None


def _interpolant_gradient_terms(radius, potential, radial_grid, spline, setting: str):
    """``(w, w', 1/|d|, inside)`` of the projector gradients with the slopes of the interpolant or the spline.

    ``PARSEC_PROJECTOR_CENTER=auto``: everything at
    :func:`parsec_python.V_ion.ionic_potential.projector_gradient_distance`, which is ``|d|`` from 3e-3 bohr on and
    that distance closer to the atom, the smooth slope of the table.
    ``tip`` and ``off``: at the true distance, with :func:`_projector_inverse_radius`.
    """

    setting = _ionic_potential.projector_gradient_rule(potential, setting)
    if setting == "floor":
        at = _ionic_potential.projector_gradient_distance(radius, potential)
        if spline is not None:
            return spline(at), spline_derivative(spline, at), 1.0 / at, None
        radial = np.interp(at, potential.radii, radial_grid, left=radial_grid[0], right=0.0)
        return radial, interpolant_slope(at, potential.radii, radial_grid), 1.0 / at, None
    if spline is not None:
        clamped = np.maximum(radius, potential.radii[0])
        radial = spline(clamped)
        slope = np.where(radius > potential.radii[0], spline_derivative(spline, clamped), 0.0)
    else:
        radial = np.interp(radius, potential.radii, radial_grid, left=radial_grid[0], right=0.0)
        slope = interpolant_slope(radius, potential.radii, radial_grid)
    inverse, inside = _projector_inverse_radius(radius, potential, setting)
    return radial, slope, inverse, inside


def radial_slope_setting() -> str:
    """``PARSEC_PERIODIC_FORCE_SLOPES``: ``interpolant`` (default), ``tabulated`` or ``tables`` (:data:`RADIAL_SLOPES`)."""

    import os

    value = os.environ.get(SLOPES_ENV, "interpolant").strip().lower() or "interpolant"
    if value not in RADIAL_SLOPES:
        raise ValueError(f"{SLOPES_ENV} must be one of " + ", ".join(RADIAL_SLOPES))
    return value


#: The switch of the radial slopes of the local forces next to an atom (:func:`spline_force_center_setting`).
SPLINE_CENTER_ENV = _radial_derivatives.SPLINE_FORCE_CENTER_ENV


def spline_force_center_setting() -> str:
    """``PARSEC_SPLINE_FORCE_CENTER``: the radial slopes of the local potential and of the core density in the local
    forces at grid points next to an atom
    (:func:`parsec_python.Pseudopotential.radial_derivatives.spline_force_center_setting`, where the values are
    described).

    On the Fortran-derived routines of a cell (:func:`periodic_local_forces`,
    :mod:`parsec_python.acceleration.Forces.periodic_spline` and :mod:`parsec_python.acceleration.Forces.periodic_device`):
    ``auto`` the slope of the smooth behaviour within ``2 rs(2)`` of the atom; ``tip`` (the default
    of 2026-11-21) the points within ``min(rs(2), 1e-6 bohr)`` of the atom (:data:`ROUND_OFF_DISTANCE`) take the
    symmetric value at the tip of the cone of a spline, zero, every other point the slope of the spline or of the
    interpolant (a grid point within round-off of an atom took the slope of the cone along the direction of the
    round-off before: up to 1.5e-5 Ry/bohr on the POTRE diamond cubes, 5.6e-3 on an O FHI table); ``fortran`` every
    spline slope closer than the first radial sample left out; ``off`` the spline slope at every positive distance.
    """

    return _radial_derivatives.spline_force_center_setting()


def spline_center_radius(potential: ParsecPseudopotential, setting: str) -> float:
    """The radius below which :func:`without_spline_center` drops a spline slope (0 under ``off``; ``auto`` as ``tip``,
    whose radius lies inside that of the rule of ``auto``, :func:`smooth_center_radius`)."""

    if setting == "off":
        return 0.0
    first = float(potential.radii[0])
    return first if setting == "fortran" else min(first, ROUND_OFF_DISTANCE)


def without_spline_center(slope, distances, potential: ParsecPseudopotential, setting: str):
    """``slope`` with the values closer to the atom than :func:`spline_center_radius` set to zero (for a spline
    slope; the caller decides), unchanged under ``off`` of :func:`spline_force_center_setting`."""

    if setting == "off":
        return slope
    return np.where(np.asarray(distances) < spline_center_radius(potential, setting), 0.0, slope)


smooth_center_radius = _radial_derivatives.smooth_center_radius
with_smooth_center = _radial_derivatives.with_smooth_center



def interpolant_slope(points: np.ndarray, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Derivative of ``np.interp(points, knots, values)`` (constant outside).

    The slope of the segment ``[knots[j], knots[j+1])`` that contains each
    point, the segment ``np.interp`` itself uses; zero below the first and
    from the last knot on, where ``np.interp`` continues the end values.
    """

    x = np.asarray(points, dtype=float)
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(values, dtype=float)
    segment = np.searchsorted(knots, x, side="right") - 1
    valid = (segment >= 0) & (segment < knots.size - 1)
    slope = np.zeros_like(x)
    if np.any(valid):
        lower = segment[valid]
        slope[valid] = (values[lower + 1] - values[lower]) / (knots[lower + 1] - knots[lower])
    return slope


def local_potential_slope(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    local_l: int,
    *,
    use_spline: bool = False,
    spline_padding_width: int | None = None,
    direct: bool = False,
) -> np.ndarray:
    """``dV/dr`` of :meth:`ParsecPseudopotential.local_potential` itself.

    The three branches of that method: the derivative of the padded spline;
    with ``direct`` the slope of the linear interpolant of ``V``; otherwise
    ``(p'(r) - p(r)/r)/r`` with ``p`` the linear interpolant of ``r V`` (zero
    at and below the first sample, where the value is constant).  Beyond the
    table (``r >= rs(ns-1)``; the spline from its own cutoff,
    :meth:`ParsecPseudopotential.local_spline_cutoff`) the Coulomb tail ``-2Z/r`` gives ``2Z/r^2``.
    """

    radius = np.asarray(distances, dtype=float)
    values = potential.local_channel_potential(local_l)
    radii = potential.radii
    result = np.zeros_like(radius)
    inside = radius < (potential.local_spline_cutoff(local_l) if use_spline else potential.interpolation_cutoff)
    outside = ~inside
    result[outside] = 2.0 * potential.ionic_charge / (radius[outside] * radius[outside])
    if not np.any(inside):
        return result
    sample = radius[inside]
    if use_spline:
        if spline_padding_width is None:
            raise ValueError("spline_padding_width is required when use_spline=True")
        spline = ParsecRadialSpline.from_positive_grid(radii, values, spline_padding_width)
        result[inside] = spline_derivative(spline, sample)
        return result
    if direct:
        result[inside] = interpolant_slope(sample, radii, values)
        return result
    slope = np.zeros_like(sample)
    beyond = sample > radii[0]
    if np.any(beyond):
        r = sample[beyond]
        weighted = radii * values
        interpolated = np.interp(r, radii, weighted)
        slope[beyond] = (interpolant_slope(r, radii, weighted) - interpolated / r) / r
    result[inside] = slope
    return result


def _erf_over_r_slope_over_r(charge: float, radius: np.ndarray, eta: float) -> np.ndarray:
    """``g'(r)/r`` for ``g(r) = 2 Z erf(eta r)/r``, finite at ``r = 0``.

    ``g'(r)/r = 2Z [2 eta r exp(-eta^2 r^2)/sqrt(pi) - erf(eta r)]/r^3``; below
    ``eta r = 1e-2`` its series ``2Z eta^3 (2/sqrt(pi)) (-2/3 + 2x^2/5 - x^4/7)``.
    """

    x = eta * np.asarray(radius, dtype=float)
    result = np.empty_like(x)
    small = x < 1.0e-2
    if np.any(small):
        xs = x[small]
        result[small] = (
            2.0 * charge * eta**3 * (2.0 / np.sqrt(np.pi))
            * (-2.0 / 3.0 + 0.4 * xs * xs - xs**4 / 7.0)
        )
    large = ~small
    if np.any(large):
        r = radius[large]
        xl = x[large]
        result[large] = (
            2.0 * charge
            * (2.0 * xl * np.exp(-xl * xl) / np.sqrt(np.pi) - erf(xl))
            / (r * r * r)
        )
    return result


def _density_slope(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    table: np.ndarray,
    *,
    spline: ParsecRadialSpline | None = None,
) -> np.ndarray:
    """Slope of an atomic density as :func:`superpose_atomic_density` samples it.

    Linear interpolation (first value continued inwards) or ``spline``, zero
    from ``rs(ns-1)`` on, where the density is dropped.
    """

    radius = np.asarray(distances, dtype=float)
    result = np.zeros_like(radius)
    inside = radius < potential.interpolation_cutoff
    if not np.any(inside):
        return result
    if spline is not None:
        result[inside] = spline_derivative(spline, radius[inside])
    else:
        result[inside] = interpolant_slope(radius[inside], potential.radii, table)
    return result


def _valence_density_table(
    potential: ParsecPseudopotential, specification: SpeciesPotential
) -> np.ndarray:
    """The atomic valence density :func:`superpose_atomic_density` superposes."""

    if potential.filtered_valence_density is not None:
        return np.asarray(potential.filtered_valence_density, dtype=float)
    if specification.read_valence_density:
        return np.asarray(potential.valence_density, dtype=float)
    radii = potential.radii
    table = np.zeros_like(radii)
    for angular_momentum, wavefunction in potential.radial_wavefunctions.items():
        table += (
            potential.channel_occupations.get(angular_momentum, 0.0)
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * radii * radii)
        )
    return table


#: An atomic density is summed over the images out to where it falls below
#: this fraction of its largest value (the energy keeps the rest of the
#: table, out to ``rs(ns-1)`` near 118 bohr, whose terms are smaller by more
#: than sixteen orders of magnitude).
DENSITY_TAIL_FRACTION = 1.0e-16


def _density_reach(potential: ParsecPseudopotential, table: np.ndarray) -> float:
    """Radius beyond which ``table`` stays below :data:`DENSITY_TAIL_FRACTION` of its peak."""

    values = np.abs(np.asarray(table, dtype=float))
    peak = float(np.max(values, initial=0.0))
    cutoff = float(potential.interpolation_cutoff)
    if peak == 0.0:
        return 0.0
    above = np.flatnonzero(values > DENSITY_TAIL_FRACTION * peak)
    last = min(int(above[-1]) + 1, values.size - 1)
    return min(cutoff, float(potential.radii[last]))


def _reaching_translations(grid, position: np.ndarray, translations: np.ndarray, radius: float):
    """The translations whose image ball of ``radius`` can reach a grid point.

    A cheap bounding-sphere test around the cell of the grid; an image it
    drops has no grid point within ``radius``, so it adds exactly nothing to
    a sum that stops at ``radius``.
    """

    coordinates = grid.coordinates
    centre = 0.5 * (coordinates.min(axis=0) + coordinates.max(axis=0))
    extent = float(np.max(np.linalg.norm(coordinates - centre, axis=1)))
    distance = np.linalg.norm(position + translations - centre, axis=1)
    return translations[distance <= radius + extent]


# ---------------------------------------------------------------------------
# Ion-ion (Ewald) forces.


# ---------------------------------------------------------------------------
# Local pseudopotential, self-consistency and core-correction forces.

_local_force_provider = None


def set_local_force_provider(provider=None) -> None:
    """Install (or, with ``None``, remove) the local terms of :func:`periodic_local_forces`.

    ``provider(grid, atoms, potentials, specifications, lattice_vectors, density,
    potential_residual, xc_potential)`` returns the :class:`LocalForceTerms` of the fields it
    built, or ``None`` for a cell it does not take (the sums below then run).  ``import parsec_python``
    installs the forces of the fast potential of ``Cubic_Spline`` species on the orthorhombic grid
    (``parsec_python.acceleration.Forces.periodic_spline``, with
    ``parsec_python.acceleration.V_ion.periodic_setup.install``); the accelerated command line installs
    the derivatives of the device fields of ``parsec_python.acceleration.V_ion.periodic_device``
    together with those fields (``parsec_python.acceleration.Forces.periodic_device.device_periodic_forces``),
    which hand the ``Cubic_Spline`` species to the former.
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
    potential_residual: np.ndarray,
    xc_potential: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms:
    """The three local terms of a periodic cell on its full grid, Ry/bohr.

    ``density`` (e/bohr^3), ``potential_residual`` (``V_out - V_in``, Ry) and
    ``xc_potential`` (Ry) are full-grid fields as for
    :func:`parsec_python.Forces.local_forces`.  ``ionic`` is the derivative
    of ``dV sum_r rho V_ion`` with ``V_ion`` from
    :func:`parsec_python.V_ion.ewald_local_ionic_potential` (same ``eta``,
    ``tolerance``, translations and G vectors), or, where
    ``grid.settings.local_potential`` is ``"g_space"``, from Fortran's
    G-space potential (:func:`parsec_python.V_ion.gspace_local.gspace_local_forces`,
    the term of ``force_loc_pb.f90``); ``core_correction`` the
    derivative of the exchange-correlation energy through the superposed core
    density and ``self_consistency`` the ``forcloc.f90`` correction with the
    superposed atomic valence densities, both with their lattice images.

    A provider installed with :func:`set_local_force_provider` takes the
    cell first (see there), except under ``PARSEC_PERIODIC_FORCE_SLOPES=tables``
    with a species of linear tables: the slopes of ``pseudo.f90``'s
    tables are summed here.
    """

    tables = radial_slope_setting() == "tables" and any(
        not specifications[atom.symbol].use_spline for atom in atoms
    )
    if _local_force_provider is not None and not tables:
        provided = _local_force_provider(
            grid,
            atoms,
            potentials,
            specifications,
            lattice_vectors,
            density,
            potential_residual,
            xc_potential,
        )
        if provided is not None:
            return provided
    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
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
    if count == 0:
        return LocalForceTerms(ionic, self_consistency, core_correction)

    volume = float(abs(np.linalg.det(lattice_vectors)))
    # The derivative of the potential the SCF used: where an installed provider
    # (the FFT Ewald split of the acceleration package) returned the converged
    # sum, the converged sum's translations and G vectors.
    tolerance = local_potential_tolerance(grid, lattice_vectors, tolerance)
    eta, real_cutoff, reciprocal_cutoff = _ewald_cutoffs(volume, eta, tolerance)
    reciprocal_vectors = _reciprocal_lattice(lattice_vectors)
    # The images of ewald_local_ionic_potential (same expressions, same sets).
    cell_center, cell_radius = _cell_center_and_radius(lattice_vectors)
    atom_reach = max([cell_radius] + [float(np.linalg.norm(np.asarray(atom.position) - cell_center)) for atom in atoms])
    all_translations = _lattice_points(
        lattice_vectors, reciprocal_vectors, real_cutoff + cell_radius + atom_reach
    )
    reciprocal_points = _lattice_points(reciprocal_vectors, lattice_vectors, reciprocal_cutoff)
    reciprocal_points = reciprocal_points[np.any(reciprocal_points != 0.0, axis=1)]
    reciprocal_weight = (
        np.exp(-np.sum(reciprocal_points**2, axis=1) / (4.0 * eta**2))
        / np.sum(reciprocal_points**2, axis=1)
        if reciprocal_points.size
        else None
    )
    coordinates = grid.coordinates
    padding = grid.settings.stencil_half_width
    residual_present = bool(np.any(residual != 0.0))
    xc_present = bool(np.any(xc != 0.0))
    valence_tables: dict[str, np.ndarray] = {}
    core_splines: dict[str, ParsecRadialSpline | None] = {}
    # PARSEC_PERIODIC_FORCE_SLOPES=tables: the derivative tables of pseudo.f90 per species (dvion, drhodr, ddenc)
    slope_tables: dict[tuple[str, str], np.ndarray] = {}
    center = spline_force_center_setting()
    g_space = getattr(grid.settings, "local_potential", "real_space") == "g_space"
    if g_space:
        # Fortran's G-space potential has no real-space piece (its force is
        # added after the loop); the core and valence terms below remain.
        all_translations = all_translations[:0]

    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = float(potential.ionic_charge)
        position = np.asarray(atom.position, dtype=np.float64)

        # Short-range Ewald piece: every translation of the potential (whose
        # radial function is always interpolated as r*V or by the spline:
        # Old_Interpolation_Format applies to clusters only).
        def local_slope(points, _p=potential, _s=specification, _symbol=atom.symbol):
            if tables and not _s.use_spline:
                # PARSEC_PERIODIC_FORCE_SLOPES=tables: dvion of pseudo.f90, interpolated as forcloc.f90 does
                key = (_symbol, "local")
                if key not in slope_tables:
                    slope_tables[key] = local_potential_derivative_table(_p, _s.local_angular_momentum)
                return evaluate_local_potential_derivative(
                    _p, points, _s.local_angular_momentum, table=slope_tables[key], slopes="tabulated"
                )
            slope = local_potential_slope(
                _p, points, _s.local_angular_momentum, use_spline=_s.use_spline, spline_padding_width=padding
            )
            if _s.use_spline:
                # the cone at the atom (spline_force_center_setting)
                slope = without_spline_center(slope, points, _p, center)
            return slope

        # the slope of the smooth behaviour next to the atom (PARSEC_SPLINE_FORCE_CENTER=auto)
        smooth = smooth_center_radius(potential, center)
        smooth_slope = local_slope(np.array([smooth])) if smooth > 0.0 else None
        reaching = np.linalg.norm(position + all_translations - cell_center, axis=1) <= real_cutoff + cell_radius
        for translation in all_translations[reaching]:
            relative = coordinates - (position + translation)
            radius = np.sqrt(np.einsum("ij,ij->i", relative, relative))
            slope_over_r = np.zeros_like(radius)
            nonzero = radius > 0.0
            slope = local_slope(radius[nonzero])
            if smooth_slope is not None:
                slope = _radial_derivatives.apply_smooth_center(slope, radius[nonzero], smooth, smooth_slope)
            slope_over_r[nonzero] = slope / radius[nonzero]
            slope_over_r += _erf_over_r_slope_over_r(charge, radius, eta) * nonzero
            ionic[index] += (density * slope_over_r) @ relative

        # Long-range piece: dV sum_r rho sin(G.(r - R)) per G.
        if reciprocal_weight is not None and not g_space:
            phase = (coordinates - position) @ reciprocal_points.T
            sines = density @ np.sin(phase)
            ionic[index] += (
                2.0 * charge * (4.0 * np.pi / volume)
                * ((reciprocal_weight * sines) @ reciprocal_points)
            )

        # Atomic valence (self-consistency) and core densities with images.
        if residual_present:
            if atom.symbol not in valence_tables:
                valence_tables[atom.symbol] = _valence_density_table(potential, specification)
            table = valence_tables[atom.symbol]
            cutoff = _density_reach(potential, table)
            for translation in _reaching_translations(
                grid,
                position,
                _periodic_image_translations(lattice_vectors, cutoff),
                cutoff,
            ):
                relative = coordinates - (position + translation)
                radius = np.sqrt(np.einsum("ij,ij->i", relative, relative))
                if not np.any(radius < cutoff):
                    continue
                if tables and not specification.use_spline:
                    key = (atom.symbol, "valence")
                    if key not in slope_tables:
                        slope_tables[key] = density_derivative_table(potential, table)
                    slope = evaluate_density_derivative(potential, radius, table, table=slope_tables[key])
                else:
                    slope = _density_slope(potential, radius, table)
                nonzero = radius > 0.0
                weight = np.zeros_like(radius)
                weight[nonzero] = residual[nonzero] * slope[nonzero] / radius[nonzero]
                self_consistency[index] += weight @ relative
        if xc_present and potential.has_nonlinear_core_correction:
            if atom.symbol not in core_splines:
                core_splines[atom.symbol] = (
                    ParsecRadialSpline.from_positive_grid(
                        potential.radii, potential.core_density, padding
                    )
                    if specification.use_spline
                    else None
                )
            spline = core_splines[atom.symbol]
            cutoff = _density_reach(potential, potential.core_density)

            def core_slope(points, _p=potential, _c=spline, _symbol=atom.symbol):
                if tables and _c is None:
                    # PARSEC_PERIODIC_FORCE_SLOPES=tables: ddenc of pseudo.f90, interpolated as forcloc.f90 does
                    key = (_symbol, "core")
                    if key not in slope_tables:
                        slope_tables[key] = density_derivative_table(_p, _p.core_density)
                    return evaluate_density_derivative(_p, points, _p.core_density, table=slope_tables[key])
                slope = _density_slope(_p, points, _p.core_density, spline=_c)
                if _c is not None:
                    slope = without_spline_center(slope, points, _p, center)
                return slope

            # the slope of the smooth behaviour next to the atom (PARSEC_SPLINE_FORCE_CENTER=auto)
            smooth = smooth_center_radius(potential, center)
            smooth_slope = core_slope(np.array([smooth])) if smooth > 0.0 else None
            for translation in _reaching_translations(
                grid,
                position,
                _periodic_image_translations(lattice_vectors, cutoff),
                cutoff,
            ):
                relative = coordinates - (position + translation)
                radius = np.sqrt(np.einsum("ij,ij->i", relative, relative))
                if not np.any(radius < cutoff):
                    continue
                slope = core_slope(radius)
                if smooth_slope is not None:
                    slope = _radial_derivatives.apply_smooth_center(slope, radius, smooth, smooth_slope)
                nonzero = radius > 0.0
                weight = np.zeros_like(radius)
                weight[nonzero] = xc[nonzero] * slope[nonzero] / radius[nonzero]
                core_correction[index] += weight @ relative

    volume_element = grid.volume_element
    ionic = ionic * volume_element
    if g_space:
        from ..V_ion.gspace_local import gspace_local_forces

        ionic = gspace_local_forces(
            grid, atoms, potentials, specifications, lattice_vectors, density
        )
    return LocalForceTerms(
        ionic=ionic,
        self_consistency=self_consistency * volume_element,
        core_correction=core_correction * volume_element,
    )


# ---------------------------------------------------------------------------
# Nonlocal (Kleinman-Bylander) forces.


def build_periodic_projector_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    k_point: np.ndarray | None = None,
    *,
    radial_slopes: str = "interpolant",
    atom_indices=None,
) -> NonlocalProjectorGradients:
    """``grad beta_q`` of the periodic projectors of ``build_nonlocal_projectors``.

    The images, supports, column order and radial interpolation are those of
    :func:`parsec_python.V_ion.build_nonlocal_projectors` with
    ``lattice_vectors`` (and ``k_point``); the gradient of each image is
    ``w(|d|) grad Y + w'(|d|) Y d/|d|`` with ``w'`` the slope of the
    interpolant (or of the spline at ``max(|d|, rs(2))``), and for a k-point
    it carries the image's Bloch phase ``exp(-i k.(r - T))``.  Columns are
    complex for a k-point.  Under ``PARSEC_PROJECTOR_CENTER=auto`` a point
    within 3e-3 bohr of the atom takes all of this at that distance
    (:func:`_interpolant_gradient_terms`).

    ``radial_slopes="tabulated"`` takes instead what ``nonloc.F90`` stores
    for ``forcnloc.f90``: the slope table ``dvw`` of ``pseudo.f90``
    (:func:`parsec_python.Pseudopotential.radial_derivatives.radial_projector_with_derivative`)
    with the distance clamped to ``rs(2)`` also in ``d/|d|``, as the forces of
    an isolated system do.  That reproduces Fortran's nonlocal forces, under
    every value of ``PARSEC_PROJECTOR_CENTER``, but is not the derivative of
    the sampled projector (see :data:`RADIAL_SLOPES`).

    ``atom_indices`` (the representative atoms of the orbits of symmetry
    sectors) samples the projectors of those atoms only: their columns, in the
    order of the projectors and with their atom indices in the labels, each
    the same column as in the gradients of every atom.
    """

    if radial_slopes not in RADIAL_SLOPES:
        raise ValueError("radial_slopes must be one of " + ", ".join(RADIAL_SLOPES))
    tabulated = radial_slopes in ("tabulated", "tables")
    wanted = _selected_atoms(len(atoms), atom_indices)
    center_setting = _ionic_potential.projector_center_setting()

    order = int(getattr(grid.settings, "double_grid_order", 1))
    if order > 1:
        return _double_grid_periodic_gradients(
            grid,
            atoms,
            potentials,
            specifications,
            np.asarray(lattice_vectors, dtype=np.float64),
            k_point,
            order=order,
            tabulated=tabulated,
            wanted=wanted,
        )
    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    complex_valued = k_point is not None
    value_dtype = complex if complex_valued else float
    k_vector = None if k_point is None else np.asarray(k_point, dtype=float)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    sqrt_dv = np.sqrt(grid.volume_element)
    coordinates = grid.coordinates

    for atom_index, atom in enumerate(atoms):
        if wanted is not None and not wanted[atom_index]:
            continue
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)
        support_radius = _projector_support_radius(potential)
        images = []
        for translation in _periodic_image_translations(lattice_vectors, support_radius):
            # The rows and distances of the projector builder (the dense scan,
            # or the grid-ball query the acceleration package installs: the
            # same rows, displacements and distances).
            support_rows, relative, radius = _ionic_potential._support_finder(
                grid, position + translation, support_radius
            )
            if support_rows.size:
                phase = (
                    None
                    if k_vector is None
                    else np.exp(-1j * ((coordinates[support_rows] - translation) @ k_vector))
                )
                images.append((support_rows, relative, radius, phase))
        if not images:
            continue
        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            radial_grid, sign = potential.radial_projector(angular_momentum, local_l)
            slope_table = (
                radial_projector_with_derivative(potential, angular_momentum, local_l)[1]
                if tabulated
                else None
            )
            spline = (
                ParsecRadialSpline.from_positive_grid(
                    potential.radii, radial_grid, grid.settings.stencil_half_width
                )
                if specification.use_spline
                else None
            )
            channel_columns = list(range(column, column + 2 * angular_momentum + 1))
            column += len(channel_columns)
            for harmonic_index in range(len(channel_columns)):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
            for support_rows, relative, radius, phase in images:
                if tabulated:
                    # nonloc.F90: rnorm = max(|d|, rs(2)) in w, w' and d/|d|.
                    clamped = np.maximum(radius, potential.radii[0])
                    if spline is not None:
                        radial = spline(clamped)
                        slope = spline_derivative(spline, clamped)
                    else:
                        radial = np.interp(clamped, potential.radii, radial_grid, right=0.0)
                        slope = np.interp(clamped, potential.radii, slope_table, right=0.0)
                    inverse = 1.0 / clamped
                    inside = None
                else:
                    # the distance of the gradient and the centre radius of the harmonics (PARSEC_PROJECTOR_CENTER)
                    radial, slope, inverse, inside = _interpolant_gradient_terms(
                        radius, potential, radial_grid, spline, center_setting
                    )
                unit = relative * inverse[:, None]
                harmonics, harmonic_gradients = real_harmonics_with_gradients(
                    angular_momentum, unit, inverse, inside=inside
                )
                for harmonic_index, target in enumerate(channel_columns):
                    gradient = sqrt_dv * (
                        radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                        + (slope * harmonics[:, harmonic_index])[:, None] * unit
                    )
                    if phase is not None:
                        gradient = gradient * phase[:, None]
                    rows.append(support_rows)
                    columns.append(np.full(support_rows.size, target, dtype=np.int64))
                    for component in range(3):
                        values[component].append(gradient[:, component].astype(value_dtype))

    def assemble(component: int) -> sp.csc_matrix:
        if column == 0:
            return sp.csc_matrix((grid.size, 0), dtype=value_dtype)
        return sp.coo_matrix(
            (
                np.concatenate(values[component]),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


def _selected_atoms(count: int, atom_indices) -> np.ndarray | None:
    """A mask of the atoms of ``atom_indices`` among ``count`` atoms, or ``None`` for every atom."""

    if atom_indices is None:
        return None
    selected = np.asarray(atom_indices, dtype=np.int64).reshape(-1)
    if selected.size and (selected.min() < 0 or selected.max() >= count):
        raise ValueError("an atom index of the projector gradients is outside the atoms")
    wanted = np.zeros(count, dtype=bool)
    wanted[selected] = True
    return wanted


def _double_grid_periodic_gradients(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    k_point: np.ndarray | None,
    *,
    order: int,
    tabulated: bool,
    wanted: np.ndarray | None = None,
) -> NonlocalProjectorGradients:
    """Gradients of the double-grid projectors of a periodic cell.

    The projectors of :func:`parsec_python.V_ion.double_grid.build_double_grid_projectors`
    with ``lattice_vectors`` (and ``k_point``): every coarse point of an image
    sums ``w(dx) w(dy) w(dz)`` times the projector at its fine points, and
    carries the Bloch phase ``exp(-i k.(r - T))`` of the coarse point itself
    (``nonloc.F90:505-657``, also for its ``vylmd``).  The gradient of each
    fine sample is that of :func:`build_periodic_projector_gradients`: the
    slope of the interpolant with the true distance (the derivative of the
    sampled projector; under ``PARSEC_PROJECTOR_CENTER=auto`` at 3e-3 bohr
    for a fine point closer to the atom), or with ``tabulated`` Fortran's ``dvw`` with the
    distance clamped to ``rs(2)`` (``nonloc.F90:544-568``).  The orthorhombic
    grid only (the input reader refuses the double grid on the general one).
    """

    from ..V_ion.double_grid import double_grid_support

    center_setting = _ionic_potential.projector_center_setting()
    k_vector = None if k_point is None else np.asarray(k_point, dtype=float)
    value_dtype = complex if k_vector is not None else float
    spacing = float(grid.spacing)
    sqrt_dv = np.sqrt(grid.volume_element)
    coordinates = grid.coordinates
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0
    for atom_index, atom in enumerate(atoms):
        if wanted is not None and not wanted[atom_index]:
            continue
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)
        support_radius = _projector_support_radius(potential)
        images = []
        for translation in _periodic_image_translations(lattice_vectors, support_radius):
            support = double_grid_support(
                coordinates, position + translation, support_radius, order, spacing
            )
            if support.rows.size:
                phase = (
                    None
                    if k_vector is None
                    else np.exp(-1j * ((coordinates[support.rows] - translation) @ k_vector))
                )
                images.append((support, phase))
        if not images:
            continue
        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            radial_grid, sign = potential.radial_projector(angular_momentum, local_l)
            slope_table = (
                radial_projector_with_derivative(potential, angular_momentum, local_l)[1]
                if tabulated
                else None
            )
            spline = (
                ParsecRadialSpline.from_positive_grid(
                    potential.radii, radial_grid, grid.settings.stencil_half_width
                )
                if specification.use_spline
                else None
            )
            width = 2 * angular_momentum + 1
            channel_columns = list(range(column, column + width))
            column += width
            for harmonic_index in range(width):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
            for support, phase in images:
                points, fine = support.radius.shape
                radius = support.radius.reshape(-1)
                relative = support.relative.reshape(-1, 3)
                clamped = np.maximum(radius, potential.radii[0])
                if tabulated:
                    if spline is not None:
                        radial = spline(clamped)
                        slope = spline_derivative(spline, clamped)
                    else:
                        radial = np.interp(clamped, potential.radii, radial_grid, right=0.0)
                        slope = np.interp(clamped, potential.radii, slope_table, right=0.0)
                    inverse = 1.0 / clamped
                    inside = None
                else:
                    # the distance of the gradient and the centre radius of the harmonics (PARSEC_PROJECTOR_CENTER)
                    radial, slope, inverse, inside = _interpolant_gradient_terms(
                        radius, potential, radial_grid, spline, center_setting
                    )
                unit = relative * inverse[:, None]
                harmonics, harmonic_gradients = real_harmonics_with_gradients(
                    angular_momentum, unit, inverse, inside=inside
                )
                for harmonic_index, target in enumerate(channel_columns):
                    gradient = sqrt_dv * (
                        radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                        + (slope * harmonics[:, harmonic_index])[:, None] * unit
                    )
                    summed = np.einsum(
                        "pm,pmc->pc", support.weights, gradient.reshape(points, fine, 3)
                    )
                    if phase is not None:
                        summed = summed * phase[:, None]
                    rows.append(support.rows)
                    columns.append(np.full(support.rows.size, target, dtype=np.int64))
                    for component in range(3):
                        values[component].append(summed[:, component].astype(value_dtype))

    def assemble(component: int) -> sp.csc_matrix:
        if column == 0:
            return sp.csc_matrix((grid.size, 0), dtype=value_dtype)
        return sp.coo_matrix(
            (
                np.concatenate(values[component]),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


def periodic_nonlocal_forces(
    operator: NonlocalProjectorOperator,
    gradients: NonlocalProjectorGradients,
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    atom_count: int,
    *,
    weight: float = 1.0,
    degeneracy: float = 2.0,
) -> np.ndarray:
    """``2 degeneracy weight sum_n f_n sum_q s_q Re(conj(c_qn) g_qn)`` per atom.

    ``c = B^H u`` and ``g_c = (grad_c B)^H u`` with Euclidean-normalized
    orbitals ``u`` (real at Gamma, Bloch factors at a k-point), ``f_n`` in
    ``[0, 1]``.  ``degeneracy`` is 2 for spin-degenerate orbitals and 1 for
    the orbitals of one spin channel; ``weight`` the k-point weight.
    """

    occupations = np.asarray(occupations, dtype=np.float64)
    if operator.projectors.shape[1] == 0 or occupations.size == 0:
        return np.zeros((atom_count, 3), dtype=np.float64)
    columns = gradient_columns_for(operator.labels, gradients.labels)
    occupied = np.flatnonzero(occupations != 0.0)
    orbitals = np.asarray(wavefunctions)[:, occupied]
    overlaps = np.asarray(operator.projectors.conj().T @ orbitals)
    factor = 2.0 * degeneracy * float(weight)
    weights = factor * operator.signs[:, None] * np.conj(overlaps) * occupations[occupied][None, :]
    owners = np.fromiter(
        (label[0] for label in operator.labels), dtype=np.int64, count=len(operator.labels)
    )
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    for component in range(3):
        slopes = np.asarray(gradients.components[component].conj().T @ orbitals)[columns]
        per_projector = np.real(np.einsum("qn,qn->q", weights, slopes))
        forces[:, component] = np.bincount(owners, weights=per_projector, minlength=atom_count)
    return forces


# ---------------------------------------------------------------------------
# Symmetry operations of a periodic cell.


@dataclass(frozen=True)
class PeriodicAtomSymmetryOperation(SpaceGroupAtomOperation):
    """An operation of a bulk cell: ``matrix @ (R_a - translation) = R_image[a] + T``.

    The representation of :class:`parsec_python.Forces.SpaceGroupAtomOperation`
    (shared with wires and slabs) with ``periodicity`` 3: ``lattice_vectors``
    (rows) gives the lattice vectors ``T``;
    :func:`parsec_python.Forces.symmetrize_positions` removes them before it
    averages positions.  ``translation`` is the Cartesian fractional
    translation ``A t`` of a space-group operation (``symop`` of
    ``symmetries.f90``: ``T(r) = U (r - A t)``); ``None`` for the point
    operations about the origin of :func:`periodic_symmetry_operations`.
    Forces transform with ``matrix`` alone.
    """


_CANDIDATE_MATRICES = None


def _lattice_point_operations(lattice_vectors: np.ndarray, tolerance: float = 1.0e-8):
    """Cartesian rotations ``R`` (with their fractional matrices) of the lattice.

    ``r = f @ A`` (rows of ``A`` the lattice vectors); ``R r`` has the
    fractional coordinates ``f @ N`` with ``N = A R^T A^-1``.  The candidates
    are the unimodular matrices ``N`` with entries -1, 0, 1 whose ``R`` is
    orthogonal, the identity first.
    """

    global _CANDIDATE_MATRICES
    if _CANDIDATE_MATRICES is None:
        candidates = np.asarray(list(product((0.0, 1.0, -1.0), repeat=9))).reshape(-1, 3, 3)
        unimodular = np.abs(np.abs(np.linalg.det(candidates)) - 1.0) < 1.0e-9
        _CANDIDATE_MATRICES = candidates[unimodular]
    vectors = np.asarray(lattice_vectors, dtype=float)
    inverse = np.linalg.inv(vectors)
    # R = (A^-1 N A)^T for every candidate at once.
    rotations = np.transpose(inverse[None, :, :] @ _CANDIDATE_MATRICES @ vectors[None, :, :], (0, 2, 1))
    products = rotations @ np.transpose(rotations, (0, 2, 1))
    orthogonal = np.all(np.abs(products - np.eye(3)[None, :, :]) <= tolerance, axis=(1, 2))
    found = [
        (rotations[index].copy(), _CANDIDATE_MATRICES[index].copy())
        for index in np.flatnonzero(orthogonal)
    ]
    found.sort(key=lambda item: 0 if np.allclose(item[1], np.eye(3)) else 1)
    return found


def _grid_invariant(grid, matrix: np.ndarray) -> bool:
    """Whether ``f -> f @ N`` maps the grid points onto grid points."""

    counts = np.asarray(grid.points_per_axis, dtype=float)
    shift = np.asarray(getattr(grid, "shift", np.zeros(3)), dtype=float)
    scaled = np.diag(counts) @ matrix @ np.diag(1.0 / counts)
    if not np.allclose(scaled, np.round(scaled), atol=1.0e-9):
        return False
    offset = shift @ scaled - shift
    return bool(np.allclose(offset, np.round(offset), atol=1.0e-9))


def _kpoints_invariant(rotation: np.ndarray, k_points, lattice_vectors) -> bool:
    """Whether ``k -> R k`` maps the set onto itself modulo G and ``k -> -k``."""

    if k_points is None:
        return True
    k_points = np.asarray(k_points, dtype=float)
    vectors = np.asarray(lattice_vectors, dtype=float)
    # Reciprocal fractional coordinates kappa = k . a_i / (2 pi).
    fractions = k_points @ vectors.T / (2.0 * np.pi)
    rotated = (k_points @ rotation.T) @ vectors.T / (2.0 * np.pi)
    for point in rotated:
        same = fractions - point[None, :]
        opposite = fractions + point[None, :]
        if not (
            np.any(np.all(np.abs(same - np.round(same)) < 1.0e-8, axis=1))
            or np.any(np.all(np.abs(opposite - np.round(opposite)) < 1.0e-8, axis=1))
        ):
            return False
    return True


class _PeriodicImageMatcher:
    """The atom each mapped atom falls on, modulo the lattice: the search of ``symm_forces.f90`` with ``tol``.

    For every mapped atom ``a`` (fractional coordinates ``mapped[a]``) the atoms of its species whose Cartesian
    minimum-image distance ``|(mapped[a] - f_b - round(mapped[a] - f_b)) A|`` is at most ``tolerance`` are its
    candidates; the operation maps the atoms where every atom has exactly one.  The candidates are those of the
    distances of every pair (the loop over the atoms of before 2026-11-19, ``O(N^2)`` per operation: 2.5 s for the
    24 operations of 1,728 atoms); a tree of the fractional coordinates of each species (periodic, side 1) gives the
    atoms within ``tolerance / sigma_min(A)`` of each mapped atom, a superset of them, whose distances are then
    those of the same expression, so the images and the decisions are the same.
    """

    def __init__(self, fractions: np.ndarray, symbols, vectors: np.ndarray, tolerance: float) -> None:
        from scipy.spatial import cKDTree

        self.fractions = np.asarray(fractions, dtype=np.float64)
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.tolerance = float(tolerance)
        self.symbols = np.asarray(list(symbols))
        count = self.fractions.shape[0]
        smallest = float(np.linalg.svd(self.vectors, compute_uv=False).min()) if count else 1.0
        # |df A| >= sigma_min |df|: every candidate lies within this fractional distance (with room for round-off)
        self.radius = self.tolerance / smallest * (1.0 + 1.0e-6) + 1.0e-12
        self.species = []
        for symbol in dict.fromkeys(self.symbols.tolist()):
            members = np.flatnonzero(self.symbols == symbol)
            self.species.append((members, cKDTree(self._wrapped(self.fractions[members]), boxsize=1.0)))

    @staticmethod
    def _wrapped(values: np.ndarray) -> np.ndarray:
        wrapped = values - np.floor(values)
        wrapped[wrapped >= 1.0] = 0.0
        return wrapped

    def _distances(self, point: np.ndarray, others: np.ndarray) -> np.ndarray:
        difference = point[None, :] - self.fractions[others]
        difference -= np.round(difference)
        return np.linalg.norm(difference @ self.vectors, axis=1)

    def images(self, mapped: np.ndarray):
        """``image[a]`` of every mapped atom, or ``None`` where an atom has no candidate or more than one."""

        mapped = np.asarray(mapped, dtype=np.float64)
        image = np.full(self.fractions.shape[0], -1, dtype=np.int64)
        for members, tree in self.species:
            queries = self._wrapped(mapped[members])
            counts = np.asarray(tree.query_ball_point(queries, self.radius, return_length=True), dtype=np.int64)
            if np.any(counts == 0):
                return None
            single = counts == 1
            if np.any(single):
                _, nearest = tree.query(queries[single], k=1)
                chosen = members[np.asarray(nearest, dtype=np.int64)]
                atoms = members[single]
                difference = mapped[atoms] - self.fractions[chosen]
                difference -= np.round(difference)
                if np.any(np.linalg.norm(difference @ self.vectors, axis=1) > self.tolerance):
                    return None
                image[atoms] = chosen
            for local in np.flatnonzero(~single):
                atom = int(members[local])
                others = members[np.asarray(tree.query_ball_point(queries[local], self.radius), dtype=np.int64)]
                candidates = others[self._distances(mapped[atom], others) <= self.tolerance]
                if candidates.size != 1:
                    return None
                image[atom] = int(candidates[0])
        return image


def periodic_symmetry_operations(
    atoms: Sequence[Atom],
    lattice_vectors: np.ndarray,
    grid=None,
    k_points: np.ndarray | None = None,
    *,
    tolerance: float = POSITION_TOLERANCE,
) -> tuple[PeriodicAtomSymmetryOperation, ...]:
    """Point operations about the origin that leave the periodic energy invariant.

    An operation is kept when it maps the lattice onto itself, every atom
    onto an atom of the same species modulo lattice vectors (within
    ``tolerance`` bohr, ``tol`` of ``symm_forces.f90``), the grid points onto
    grid points (where ``grid`` is given) and the k-points onto k-points
    modulo reciprocal lattice vectors and time reversal.  Operations with a
    fractional translation (non-symmorphic) are not searched.  The identity
    comes first.
    """

    vectors = np.asarray(lattice_vectors, dtype=float)
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    symbols = [atom.symbol for atom in atoms]
    fractions = fractional_coordinates(positions, vectors) if len(atoms) else positions
    matcher = _PeriodicImageMatcher(fractions, symbols, vectors, tolerance)
    found: list[PeriodicAtomSymmetryOperation] = []
    for rotation, matrix in _lattice_point_operations(vectors):
        if grid is not None and not _grid_invariant(grid, matrix):
            continue
        if not _kpoints_invariant(rotation, k_points, vectors):
            continue
        image = matcher.images(fractions @ matrix)
        if image is None or np.unique(image).size != len(atoms):
            continue
        found.append(
            PeriodicAtomSymmetryOperation(
                matrix=rotation.copy(), image=image, lattice_vectors=vectors.copy()
            )
        )
    if not found or not np.allclose(found[0].matrix, np.eye(3)):
        raise RuntimeError("the identity must be a symmetry operation")
    return tuple(found)


def space_group_symmetry_operations(
    atoms: Sequence[Atom],
    lattice_vectors: np.ndarray,
    group,
) -> tuple[PeriodicAtomSymmetryOperation, ...]:
    """The operations of a space group as atom operations (``symm_forces.f90``).

    ``group`` is a :class:`parsec_python.Grid.space_group.SpaceGroup` (Fortran
    ``symm``), non-symmorphic operations included.  Operation ``T(r) = U (r -
    A t)`` maps atom ``a`` onto atom ``image[a]`` modulo a lattice vector
    (:func:`parsec_python.Grid.space_group.space_group_atom_images`, the
    search and tolerance of ``symm_forces.f90``); averaging forces over these
    operations is ``symm_forces``' ``F_sym(a) = (1/N) sum_T U^T F(T a)``
    written as ``F_sym(T a) += U F(a)`` (the same sum over a group).  The
    identity (first operation of ``symgen``) comes first.
    """

    from ..Grid.space_group import space_group_atom_images

    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    images = space_group_atom_images(group, vectors, atoms)
    rotations = np.asarray(group.cartesian_rotations, dtype=np.float64).reshape(-1, 3, 3)
    translations = group.translation_vectors @ vectors
    operations = tuple(
        PeriodicAtomSymmetryOperation(
            matrix=rotations[index].copy(),
            image=images[index].copy(),
            lattice_vectors=vectors.copy(),
            translation=translations[index].copy(),
        )
        for index in range(group.order)
    )
    if not np.allclose(operations[0].matrix, np.eye(3)) or np.any(operations[0].translation):
        raise RuntimeError("the identity must be the first space-group operation")
    return operations


def _space_group(system):
    """The space group a periodic problem carries (``SinglePointInput.space_group``), or ``None``."""

    return getattr(system.input, "space_group", None)


def periodic_force_operations(system, k_points: np.ndarray | None = None):
    """The operations the forces of a periodic cell are averaged over by default.

    Where the problem carries Fortran's space group (``Monkhorst_Pack_Shift``,
    ``Symmetrize_Charge_Density`` or a weighted ``Kpoint_List`` with forces;
    see :func:`parsec_python.Input.parse_parsec_input`), its operations,
    non-symmorphic ones included, as ``symm_forces.f90`` with ``symm``;
    otherwise the point operations of :func:`periodic_symmetry_operations`
    (lattice, grid, atoms and k-points), as before.  Where the problem carries a
    larger ``force_space_group`` (a wedge of the default routines that keeps
    the grid's finite-difference operators, ``PARSEC_KPOINT_FORCE_GROUP=crystal``),
    its operations: those that map the grid and the mesh onto themselves; the
    k-point sum is completed over the wedge's group
    (:func:`reduced_kpoint_operations`).
    """

    group = getattr(system.input, "force_space_group", None) or _space_group(system)
    lattice = system.input.periodic_cell.lattice_vectors
    if group is not None:
        return space_group_symmetry_operations(system.atoms, lattice, group)
    return periodic_symmetry_operations(system.atoms, lattice, system.grid, k_points)


def reduced_kpoint_operations(system, result):
    """The operations that complete the k-point sum of a reduced k-point set, or ``None``.

    A k-point set reduced by the space group (``Monkhorst_Pack_Shift`` with a
    non-trivial group, Fortran's ``irrbz``) or weighted unequally (a weighted
    ``Kpoint_List``, taken as such a wedge) gives the nonlocal forces of its
    own points only.  Summed over the whole Brillouin zone they are
    ``(1/N) sum_g U_g F(g^-1 a)`` of the wedge forces (each k-point's star
    ``g k``, with ``-k`` by time reversal already in the weights), the average
    over the group: Fortran gets it from ``symm_forces`` at the end.  ``None``
    for every other run (Gamma, unreduced or equally weighted sets, no group).
    """

    group = _space_group(system)
    k_points = getattr(result, "k_points", None)
    if group is None or group.order <= 1 or k_points is None:
        return None
    weights = np.asarray(getattr(result, "k_weights", np.ones(len(k_points))), dtype=float)
    reduced = getattr(system.input, "monkhorst_pack_shift", None) is not None or bool(
        weights.size > 1 and np.any(weights != weights[0])
    )
    if not reduced:
        return None
    return space_group_symmetry_operations(
        system.atoms, system.input.periodic_cell.lattice_vectors, group
    )


# ---------------------------------------------------------------------------
# Assembly.


def _is_spin_kpoint_result(result) -> bool:
    """A result of ``run_scf_kpoints_spin_polarized`` (two channels, k-points)."""

    return getattr(result, "k_points", None) is not None and hasattr(result, "density_up")


def _kpoint_blocks(result):
    """``(k_point, weight, vectors, occupations, degeneracy)`` of a k-point result.

    A collinear spin-polarized result yields the up and then the down block
    of each k-point, each orbital holding one electron.
    """

    vectors = getattr(result, "k_wavefunctions", None)
    if vectors is None:
        raise ValueError(
            "the k-point result holds no orbitals: run the k-point SCF with "
            "keep_wavefunctions=True to compute forces"
        )
    k_points = np.asarray(result.k_points, dtype=float)
    weights = np.asarray(result.k_weights, dtype=float)
    if _is_spin_kpoint_result(result):
        channels = (
            (vectors[0], np.asarray(result.occupations_up, dtype=float)),
            (vectors[1], np.asarray(result.occupations_down, dtype=float)),
        )
        degeneracy = 1.0
    else:
        channels = ((vectors, np.asarray(result.occupations, dtype=float)),)
        degeneracy = 2.0
    offsets = [0] * len(channels)
    for index in range(len(k_points)):
        for channel, (blocks, occupations) in enumerate(channels):
            block = blocks[index]
            count = np.asarray(block).shape[1]
            start = offsets[channel]
            yield (
                k_points[index],
                float(weights[index]),
                block,
                occupations[start : start + count],
                degeneracy,
            )
            offsets[channel] = start + count


def meta_gga_core_potential(result, xc_potential):
    """``dE_xc/dn_core`` (Ry) of the core-correction force: ``xc_potential``, or for a meta-GGA in the orbital form
    of the density gradient the record's ``core_potential`` (its local potential ``de/dn`` lacks the divergence of
    ``W = de/dg``, which reaches ``n_core`` through the gradient of the core density)."""

    record = getattr(result, "meta_gga", None)
    core = None if record is None else getattr(record, "core_potential", None)
    return xc_potential if core is None else np.asarray(core, dtype=np.float64)


#: The switch of the exchange-correlation potential of the core-correction force of a cell (:func:`core_force_xc_setting`).
CORE_FORCE_XC_ENV = "PARSEC_CORE_FORCE_XC"


def core_force_xc_setting() -> str:
    """``PARSEC_CORE_FORCE_XC``: the exchange-correlation potential of the core-correction force of a 3-D cell.

    PARSEC's ``potshift`` (``Zero_Average_Vxc``, the default of a 3-D cell without spin) removes the average of
    ``V_xc`` from the potential the SCF loop keeps.  The core-correction force ``dV sum_r V_xc(r) d rho_c(r - R_a)/dR_a``
    is the derivative of ``E_xc[rho + rho_c]`` only with the potential itself: with the average ``c`` removed it gains
    ``c dQ_c/dR_a``, where ``Q_c = dV sum_r rho_c(r - R_a)`` is the grid sum of the core density, which changes as the
    atom moves against the grid (5 to 7 % of the force of a Ge atom 0.01 to 0.1 bohr off its site, 3e-4 to 6e-3
    Ry/bohr).
    PARSEC 1.5 forms that term in G-space (``forpbc.f90``), where the average drops out.

    ``auto`` (default): the average is added back (:func:`core_force_xc_average`).  ``shifted``: the potential as
    the loop kept it, bit for bit as before.
    """

    import os

    value = os.environ.get(CORE_FORCE_XC_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "shifted"):
        raise ValueError(f"{CORE_FORCE_XC_ENV} must be auto or shifted, not {value!r}")
    return value


def _host_array(values) -> np.ndarray:
    getter = getattr(values, "get", None)
    return np.asarray(getter() if callable(getter) else values, dtype=np.float64)


def _has_core_charge(potential) -> bool:
    """Whether a pseudopotential carries a nonzero core density (``None`` and an array of zeros mean none)."""

    core = getattr(potential, "core_density", None)
    return core is not None and bool(np.any(np.asarray(core) != 0.0))


def core_force_xc_average(system, result) -> float:
    """The average of ``V_xc`` that ``potshift`` removed from the potential of ``result``, to be added back in the
    core-correction force (``PARSEC_CORE_FORCE_XC=auto``); ``0.0`` where nothing was removed or nothing needs it.

    Zero under ``shifted``, for a calculation without ``SCFSettings.zero_average_xc_potential``, for a wire or slab
    (``potshift`` acts on 3-D cells only), without a species with a core density, and for a meta-GGA, whose core term
    takes ``dE_xc/dn_core`` of its own record, which keeps its average.  Otherwise the potential of the density of the
    result is evaluated once more without the shift (``evaluate_xc`` of the host system with ``zero_average=False``; a
    collinear spin result: the mean of the channel averages, which is what the averaged potential of the core term
    lost; collinear spin with spin-orbit coupling and an explicit ``Zero_Average_Vxc``: the same from the output
    densities of both channels that the spinor result keeps for ``parsec.dat``, since 2026-11-22, before which its
    core term kept the shifted potential).  Zero for a non-collinear spinor result, which keeps no channel densities.
    """

    if core_force_xc_setting() == "shifted":
        return 0.0
    # the host system of an accelerated one: its evaluate_xc takes the full-grid density of the result and the
    # zero_average keyword (the device evaluator of the loop takes the loop's layout)
    reference = getattr(system, "reference", None)
    owner = reference if hasattr(reference, "evaluate_xc") else (system if hasattr(system, "evaluate_xc") else None)
    settings = getattr(system, "input", None)
    scf = getattr(settings, "scf", None)
    if owner is None or scf is None or not getattr(scf, "zero_average_xc_potential", False):
        return 0.0
    if int(getattr(getattr(settings, "grid", None), "periodicity", 3) or 3) != 3:
        return 0.0
    if getattr(scf, "is_meta_gga", False):
        return 0.0
    potentials = getattr(system, "pseudopotentials", None) or getattr(owner, "pseudopotentials", {})
    # the readers give a table without a core charge an array of zeros, not None: test the values, so a cell without
    # a core charge pays no second evaluation of the functional (the term it would feed is zero anyway)
    if not any(_has_core_charge(potential) for potential in potentials.values()):
        return 0.0
    channels = getattr(result, "spin", None)
    if channels is not None or hasattr(result, "density_up"):
        up, down = (
            (channels[0].density, channels[1].density)
            if channels is not None
            else (result.density_up, result.density_down)
        )
        xc = owner.evaluate_xc_spin_polarized(_host_array(up), _host_array(down), zero_average=False)
        return 0.5 * (float(np.mean(_host_array(xc.potential_up))) + float(np.mean(_host_array(xc.potential_down))))
    if getattr(scf, "spin_polarized", False):
        # collinear spin with spin-orbit coupling: the spinor result keeps the output densities of both channels in
        # the fields of parsec.dat (kind 2, spin-orbit), the densities its potential was made of
        fields = getattr(result, "parsec_dat_fields", None) or {}
        densities = fields.get("densities")
        if hasattr(result, "spinors") and fields.get("kind") == 2 and densities is not None:
            xc = owner.evaluate_xc_spin_polarized(_host_array(densities[0]), _host_array(densities[1]), zero_average=False)
            return 0.5 * (float(np.mean(_host_array(xc.potential_up))) + float(np.mean(_host_array(xc.potential_down))))
        # a spin result that keeps the total density alone (a non-collinear magnetization): the channel averages
        # cannot be formed again, the potential stays as the loop left it
        return 0.0
    xc = owner.evaluate_xc(_host_array(result.density), zero_average=False)
    return float(np.mean(_host_array(xc.potential)))


def core_force_xc(system, result, xc_potential):
    """``xc_potential`` with the average of :func:`core_force_xc_average` added (the potential of the core term)."""

    shift = core_force_xc_average(system, result)
    return xc_potential if shift == 0.0 else xc_potential + shift


def periodic_force_fields(system, result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(density, V_out - V_in, V_xc)`` of a periodic result for the local terms.

    A k-point loop returns the mixed potential as its input potential; its
    last input potential, where kept, is ``scf_input_potential``.  A
    collinear spin-polarized k-point result gives the total density and the
    averages over its two channels (``forpbc.f90``/``forcloc.f90``), from the
    potentials ``keep_wavefunctions`` kept.  The potential of the core term is that of ``PARSEC_CORE_FORCE_XC``
    (:func:`core_force_xc`).
    """

    if _is_spin_kpoint_result(result):
        inputs = getattr(result, "scf_input_potentials", None)
        xc_channels = getattr(result, "xc_potentials", None)
        if inputs is None or xc_channels is None:
            raise ValueError(
                "the spin-polarized k-point result holds no potentials: run it "
                "with keep_wavefunctions=True to compute forces"
            )
        outputs = (
            result.scf_potentials["potential_up"],
            result.scf_potentials["potential_down"],
        )
        density = np.asarray(result.density_up, dtype=np.float64) + np.asarray(
            result.density_down, dtype=np.float64
        )
        residual = 0.5 * (
            (np.asarray(outputs[0]) - np.asarray(inputs[0]))
            + (np.asarray(outputs[1]) - np.asarray(inputs[1]))
        )
        xc = 0.5 * (np.asarray(xc_channels[0]) + np.asarray(xc_channels[1]))
        return density, residual, meta_gga_core_potential(result, core_force_xc(system, result, xc))

    density = np.asarray(result.density, dtype=np.float64)
    output_potential = np.asarray(result.output_effective_potential, dtype=np.float64)
    if getattr(result, "k_points", None) is not None:
        last_input = getattr(result, "scf_input_potential", None)
        residual = (
            np.zeros_like(density)
            if last_input is None
            else output_potential - np.asarray(last_input, dtype=np.float64)
        )
    else:
        residual = output_potential - np.asarray(result.input_effective_potential)
    xc = core_force_xc(system, result, np.asarray(result.xc_potential, dtype=np.float64))
    return density, residual, meta_gga_core_potential(result, xc)


def compute_periodic_forces(
    system,
    result,
    *,
    symmetrize: bool = True,
    operations=None,
):
    """Forces of a finished SCF loop of a periodic cell (Gamma or k-points).

    ``system`` is a :class:`parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem`
    and ``result`` the :class:`parsec_python.models.SinglePointResult` of
    :func:`parsec_python.SCF.run_scf` (Gamma, also with collinear spin through
    :func:`parsec_python.SCF.run_spin_scf`) or of
    :func:`parsec_python.SCF.run_scf_kpoints` with ``keep_wavefunctions=True``.
    The order of ``forcnloc.f90``: ion-ion + local, nonlocal (+ LDA+U, TS),
    symmetrization, net force removed.  ``operations`` default to
    :func:`periodic_symmetry_operations` of the present atoms.
    """

    from time import perf_counter

    from .total import assemble_forces, correction_forces

    started = perf_counter()
    cell = system.input.periodic_cell
    if cell is None:
        raise ValueError("compute_periodic_forces needs a periodic system")
    lattice = np.asarray(cell.lattice_vectors, dtype=np.float64)
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    if any(getattr(spec, "spin_orbit", False) for spec in specifications.values()):
        raise NotImplementedError(
            "the forces of spin-orbit species come from the spinor drivers "
            "(parsec_python.Forces.spin_orbit), not from a scalar result"
        )

    repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
    density, residual, xc_potential = periodic_force_fields(system, result)
    local = periodic_local_forces(
        system.grid,
        atoms,
        system.pseudopotentials,
        specifications,
        lattice,
        density,
        residual,
        xc_potential,
    )
    slopes = radial_slope_setting()
    k_points = getattr(result, "k_points", None)
    separable = np.zeros((len(atoms), 3))
    if k_points is not None:
        # The blocks come k-point by k-point (both spin channels of one
        # k-point together): one set of Bloch projectors at a time.
        cache: dict[bytes, tuple] = {}
        for k_point, weight, vectors, occupations, degeneracy in _kpoint_blocks(result):
            key = np.asarray(k_point, dtype=np.float64).tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = (
                    build_nonlocal_projectors(
                        system.grid,
                        atoms,
                        system.pseudopotentials,
                        specifications,
                        lattice,
                        k_point=k_point,
                    ),
                    build_periodic_projector_gradients(
                        system.grid,
                        atoms,
                        system.pseudopotentials,
                        specifications,
                        lattice,
                        k_point,
                        radial_slopes=slopes,
                    ),
                )
            operator, gradients = cache[key]
            separable += periodic_nonlocal_forces(
                operator,
                gradients,
                vectors,
                occupations,
                len(atoms),
                weight=weight,
                degeneracy=degeneracy,
            )
        completion = reduced_kpoint_operations(system, result)
        if completion is not None:
            # The k-points of the wedge stand for their stars.
            separable = symmetrize_forces(separable, completion)
    else:
        gradients = build_periodic_projector_gradients(
            system.grid,
            atoms,
            system.pseudopotentials,
            specifications,
            lattice,
            radial_slopes=slopes,
        )
        spin = getattr(result, "spin", None)
        separable = periodic_nonlocal_forces(
            system.nonlocal_operator,
            gradients,
            np.asarray(result.wavefunctions),
            np.asarray(result.occupations),
            len(atoms),
            degeneracy=2.0 if spin is None else 1.0,
        )
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = periodic_force_operations(system, k_points)
    return assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        seconds=perf_counter() - started,
        periodic=True,
        keep_torque=True,
        **correction_forces(system, result),
    )


__all__ = [
    "RADIAL_SLOPES",
    "SLOPES_ENV",
    "SPLINE_CENTER_ENV",
    "PeriodicAtomSymmetryOperation",
    "build_periodic_projector_gradients",
    "compute_periodic_forces",
    "interpolant_slope",
    "local_potential_slope",
    "CORE_FORCE_XC_ENV",
    "core_force_xc",
    "core_force_xc_average",
    "core_force_xc_setting",
    "periodic_force_fields",
    "periodic_local_forces",
    "periodic_nonlocal_forces",
    "periodic_force_operations",
    "periodic_symmetry_operations",
    "radial_slope_setting",
    "reduced_kpoint_operations",
    "space_group_symmetry_operations",
    "spline_center_radius",
    "spline_force_center_setting",
    "without_spline_center",
]
