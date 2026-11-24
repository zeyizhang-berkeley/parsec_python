"""Radial derivatives of PARSEC pseudopotential tables, as forces use them.

``pseudo.f90`` differentiates four radial tables once per species: the local
potential, the atomic valence density, the frozen core density and the
Kleinman-Bylander radial projector.  ``forcloc.f90`` and ``nonloc.F90`` then
interpolate those derivative tables at the atom-grid distances.  This module
translates both steps.

The radial grid of a ``MARTINS_NEW`` file is logarithmic,

``r_j = a*exp(b*(j-1)) - c``,  ``j = 2, ..., npoint``,

with ``c = a`` for every Martins format and ``c = 0`` for an FHIPP file
(:attr:`ParsecPseudopotential.radial_offset`), so a derivative with respect
to the index ``j`` becomes a radial derivative through
``dj/dr = 1/(b*(r_j + c))``.  PARSEC takes the index derivative with a
central stencil of nine points on each side (``mor = 9`` in ``pseudo.f90``)
whose weights ``fornberg.f90`` stores with ten digits; the same ten digits
are used here.  Fortran indices ``j <= 1`` are padding that continues the
first tabulated value, and file sample ``p`` (zero based) is Fortran index
``p + 2``.

Every table returned here is aligned with ``potential.radii``.
"""

from __future__ import annotations

import numpy as np

from .potre import ParsecPseudopotential
from .radial_spline import ParsecRadialSpline


#: ``mor`` of ``pseudo.f90``: stencil points on each side.
DERIVATIVE_HALF_WIDTH = 9

#: ``coe(1:9)`` of ``fornberg(1, 9, coe)``, with the digits PARSEC stores.
#: ``coe(-k) = -coe(k)`` and ``coe(0) = 0``.
FIRST_DERIVATIVE_WEIGHTS = np.array(
    [
        0.9000000000e00,
        -0.3272727273e00,
        0.1272727273e00,
        -0.4405594406e-01,
        0.1258741259e-01,
        -0.2797202797e-02,
        0.4495504496e-03,
        -0.4627725216e-04,
        0.2285296403e-05,
    ]
)

#: Local-potential derivatives at radii up to this one are set to zero.
_LOCAL_DERIVATIVE_NOISE_RADIUS = 3.0e-4

#: ``forcloc.f90`` switches to the analytic tail at ``rs(npoint-6)``, which is
#: this index of the positive radial samples (a table of at least seven).
_TAIL_START_INDEX = -7

#: The switch of the radial slopes of the forces for tables the energy interpolates linearly
#: (:func:`linear_slope_setting`).
LINEAR_SLOPES_ENV = "PARSEC_LINEAR_TABLE_SLOPES"
LINEAR_SLOPES = ("tabulated", "interpolant")


def linear_slope_setting() -> str:
    """``PARSEC_LINEAR_TABLE_SLOPES``: how forces differentiate a radial table that the energy interpolates linearly.

    A species with linear tables (``Cubic_Spline: .false.``, PARSEC 1.5's default for a species without the label,
    here under ``PARSEC_CUBIC_SPLINE_DEFAULT=off``) samples its local potential (``r V``), its core density
    and its projectors by linear interpolation between the radial samples, so the energy is piecewise smooth in the
    atom positions.

    * ``tabulated`` (default): the slope tables of ``pseudo.f90`` (a central difference of eighteenth order on the
      logarithmic grid), interpolated linearly, as ``forcloc.f90`` and ``forcnloc.F90`` use them: the forces of
      PARSEC 1.5 to round-off, but not the derivative of the energy the run evaluates.
    * ``interpolant``: the slope of the linear interpolant the energy samples (the exact derivative of the computed
      energy, except within 3e-3 bohr of a grid point, where ``PARSEC_PROJECTOR_CENTER=auto`` takes the projector
      gradient at that distance: there the nonlocal term misses the derivative by up to 2.6e-2 Ry/bohr on an Ag
      table; exact under ``tip``), the radial slopes the Fortran-derived routines of a cell take already
      (``PARSEC_PERIODIC_FORCE_SLOPES``).  The local forces of a cluster are then summed by the readable routine on
      the host instead of the device kernel (``acceleration/Forces/compute.py``): a molecular dynamics run takes about
      twice the wall time.

    It applies to the local potential and the atomic densities of clusters, wires and slabs, to the core density of
    the default routines of a cell, and to the projectors of clusters, wires, slabs, the double grid and of the
    default routines of a cell (Gamma point, k-point meshes, the scalar part of spin-orbit runs).  ``Cubic_Spline``
    species are not affected (their slopes are the spline derivatives), nor are the periodic local potentials of the
    default routines (always the spline) or the Fortran-derived routines of a cell (``PARSEC_PERIODIC_FORCE_SLOPES``).
    """

    import os

    value = os.environ.get(LINEAR_SLOPES_ENV, "tabulated").strip().lower() or "tabulated"
    if value not in LINEAR_SLOPES:
        raise ValueError(f"{LINEAR_SLOPES_ENV} must be one of " + ", ".join(LINEAR_SLOPES) + f", not {value!r}")
    return value


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


def linear_projector_slope(
    distances: np.ndarray,
    radii: np.ndarray,
    projector: np.ndarray,
    tabulated: np.ndarray,
    slopes: str,
) -> np.ndarray:
    """Radial slope of a projector the energy samples as ``np.interp(distances, radii, projector, right=0.0)``.

    ``slopes="tabulated"``: ``dvw`` of ``pseudo.f90`` (``tabulated``) interpolated linearly, as ``nonloc.F90``;
    ``"interpolant"``: the slope of the interpolant itself (zero below the first and beyond the last sample).
    """

    if slopes == "interpolant":
        return interpolant_slope(distances, radii, projector)
    return np.interp(distances, radii, tabulated, right=0.0)


def linear_local_potential_slope(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    local_l: int,
    *,
    direct: bool = False,
) -> np.ndarray:
    """``dV/dr`` of the linear branches of :meth:`ParsecPseudopotential.local_potential`.

    With ``p`` the linear interpolant of ``r V``, ``V = p/r`` and the slope is ``(p'(r) - p(r)/r)/r``; ``direct``
    (``Old_Interpolation_Format``) interpolates ``V`` itself.  Zero at and below the first sample, where the value
    is constant, and the Coulomb slope ``2Z/r^2`` from ``rs(ns-1)`` on, where the potential is ``-2Z/r``.
    """

    radius = np.asarray(distances, dtype=float)
    values = potential.local_channel_potential(local_l)
    radii = potential.radii
    result = np.zeros_like(radius)
    inside = radius < potential.interpolation_cutoff
    outside = ~inside
    result[outside] = 2.0 * potential.ionic_charge / (radius[outside] * radius[outside])
    if not np.any(inside):
        return result
    sample = radius[inside]
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


def _stencil_range(count: int) -> tuple[int, int]:
    """First and one-past-last file sample with a complete PARSEC stencil.

    Fortran indices ``mor <= j <= npoint - mor`` with ``npoint = count + 1``
    and sample ``p = j - 2``.
    """

    return DERIVATIVE_HALF_WIDTH - 2, count - DERIVATIVE_HALF_WIDTH


def log_grid_derivative(
    potential: ParsecPseudopotential,
    values: np.ndarray,
    *,
    first: int | None = None,
) -> np.ndarray:
    """Differentiate one radial table as ``pseudo.f90`` does.

    The two padding entries below the first sample continue ``values[0]``.
    Samples without a complete stencil are returned as zero; every caller
    applies its own rule there.  ``first`` moves the first differentiated
    sample upwards (the projector tables start at Fortran index ``2*mor``).
    """

    table = np.asarray(values, dtype=float)
    radii = potential.radii
    if table.shape != radii.shape:
        raise ValueError("values must be aligned with the radial grid")
    start, stop = _stencil_range(table.size)
    if first is not None:
        start = max(start, int(first))
    result = np.zeros_like(table)
    if stop <= start:
        return result
    # Fortran indices 0 and 1 hold the continued first value.
    padded = np.concatenate((np.full(2, table[0]), table))
    centre = np.arange(start, stop) + 2
    accumulated = np.zeros(centre.size)
    for offset, weight in enumerate(FIRST_DERIVATIVE_WEIGHTS, start=1):
        accumulated += weight * (padded[centre + offset] - padded[centre - offset])
    result[start:stop] = accumulated / (
        potential.radial_parameter_b
        * (radii[start:stop] + potential.radial_offset)
    )
    return result


def local_potential_derivative_table(
    potential: ParsecPseudopotential, local_l: int
) -> np.ndarray:
    """``dvion`` of ``pseudo.f90``: dV_local/dr in Ry/bohr on the radial grid.

    The last nine samples take the Coulomb value ``2*Z/r**2`` and samples at
    radii up to ``3e-4`` bohr are zero, where the numerical derivative is
    noise.  A table continued below its first sample by ``PARSEC_FHI_ORIGIN=auto``
    (``origin_samples``) is smooth there: no sample of it is zeroed.
    """

    potential.validate_local_channel(local_l)
    radii = potential.radii
    table = log_grid_derivative(potential, potential.local_channel_potential(local_l))
    _, stop = _stencil_range(radii.size)
    table[stop:] = 2.0 * potential.ionic_charge / (radii[stop:] * radii[stop:])
    if not getattr(potential, "origin_samples", 0):
        table[radii <= _LOCAL_DERIVATIVE_NOISE_RADIUS] = 0.0
    return table


def density_derivative_table(
    potential: ParsecPseudopotential, density: np.ndarray
) -> np.ndarray:
    """``drhodr``/``ddenc`` of ``pseudo.f90`` for a volume-density table.

    The density is continued by its first value below the grid and its
    derivative is zero where the stencil is incomplete.
    """

    return log_grid_derivative(potential, density)


def valence_density_table(
    potential: ParsecPseudopotential, *, read_from_file: bool
) -> np.ndarray:
    """Spherical atomic valence density ``rho_r`` in electrons/bohr**3.

    Without ``Read_Vcd`` PARSEC rebuilds it from the reduced radial functions
    and their occupations, ``sum_l f_l*u_l(r)**2/(4*pi*r**2)``.  A Fourier
    filtered potential carries the filtered density, which ``fourier_f.f90``
    puts in place of either.
    """

    if potential.filtered_valence_density is not None:
        return np.asarray(potential.filtered_valence_density, dtype=float)
    if read_from_file:
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


def radial_projector_with_derivative(
    potential: ParsecPseudopotential, angular_momentum: int, local_l: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalized KB radial projector, its radial derivative and its sign.

    The first array and the sign are those of
    :meth:`ParsecPseudopotential.radial_projector`.  The derivative is
    ``dvw`` of ``pseudo.f90``: the stencil acts on the projector before its
    inner samples are replaced, the samples up to Fortran index ``2*mor``
    take the derivative at ``2*mor``, and the last nine are zero.

    For a Fourier-filtered channel both tables are those ``fourier_f.f90``
    puts in their place at every file radius: the filtered projector
    (``fint``) and its radial derivative ``dfint`` (PARSEC 1.5
    ``math/fourier_f.f90:265-268``; the previous tree stored ``fint`` as the
    slope as well, a bug 1.5 fixed).
    """

    if angular_momentum in potential.filtered_projectors:
        projector, sign = potential.radial_projector(angular_momentum, local_l)
        slopes = potential.filtered_projector_slopes
        if angular_momentum not in slopes:
            raise ValueError(
                "the Fourier-filtered projector of "
                f"l={angular_momentum} carries no radial derivative (dfint)"
            )
        return projector, np.array(slopes[angular_momentum], dtype=float), sign
    projector, sign = potential.radial_projector(angular_momentum, local_l)
    delta_v = potential.potential_difference(
        potential.channel_potentials[angular_momentum], local_l
    )
    radial_wave = potential.radial_wavefunctions[angular_momentum]
    denominator = potential.kb_denominator(radial_wave * radial_wave * delta_v)
    raw = delta_v * radial_wave / potential.radii / np.sqrt(abs(denominator))
    constant_through = 2 * DERIVATIVE_HALF_WIDTH - 2
    derivative = log_grid_derivative(potential, raw, first=constant_through)
    if derivative.size > constant_through:
        derivative[:constant_through] = derivative[constant_through]
    return projector, derivative, sign


def tail_start_radius(potential: ParsecPseudopotential) -> float:
    """Radius from which ``forcloc.f90`` uses the analytic Coulomb tail.

    ``rlarge = rs(npoint-6)`` (forcloc.f90:153) with ``npoint`` the Fortran
    index of the last sample (``count + 1``: the file samples are
    ``rs(2:npoint)``, pseudo.f90:409-410).  For a table of seven samples or
    more that is a file radius.  A shorter table points into the padding,
    where pseudo.f90:843 sets ``rs(j) = j - 1`` for ``j <= 1``: the tail
    radius is then zero or negative and every point from the first sample
    outwards takes the Coulomb tail, as in PARSEC.
    """

    count = int(potential.radii.size)
    if count >= -_TAIL_START_INDEX:
        return float(potential.radii[_TAIL_START_INDEX])
    # Fortran index npoint - 6 = count - 5 <= 1: the padding value j - 1.
    return float(count - 6)


def spline_derivative(spline: ParsecRadialSpline, points: np.ndarray) -> np.ndarray:
    """First derivative ``y1`` of PARSEC's ``splint`` at ``points``."""

    queries = np.asarray(points, dtype=float).reshape(-1)
    lower = np.searchsorted(spline.knots, queries, side="right") - 1
    lower = np.clip(lower, 0, spline.knots.size - 2)
    upper = lower + 1
    step = spline.knots[upper] - spline.knots[lower]
    left_weight = (spline.knots[upper] - queries) / step
    right_weight = (queries - spline.knots[lower]) / step
    return (
        (spline.values[upper] - spline.values[lower]) / step
        - (3.0 * left_weight * left_weight - 1.0)
        / 6.0
        * step
        * spline.second_derivatives[lower]
        + (3.0 * right_weight * right_weight - 1.0)
        / 6.0
        * step
        * spline.second_derivatives[upper]
    )


#: The switch of the radial slopes of the local potential and the core density next to an atom
#: (:func:`spline_force_center_setting`).
SPLINE_FORCE_CENTER_ENV = "PARSEC_SPLINE_FORCE_CENTER"

#: Bohr: under ``PARSEC_SPLINE_FORCE_CENTER=auto`` a table whose first radial sample ``rs(2)`` lies below this distance
#: gives the grid points closer to its atom than ``c = 2 rs(2)`` the slope of the smooth behaviour
#: (:func:`smooth_center_radius`).  At or beyond it (the H and He FHI tables, first sample 6.25e-3 bohr, and coarse
#: test tables) steps of 1e-3 and 2e-3 bohr resolve the padded first interval ``[0, rs(2)]``: the energy follows the
#: spline there, and the table keeps the slopes of its route.  H FHI at fixed density, an atom 1e-11 to 1.4e-2 bohr
#: from a grid point (CPU): with the rule (``c`` = 1.25e-2 bohr) the local term was up to 3.4e-5 Ry/bohr off the
#: central differences of the local energy at steps of 1e-3 and 2e-3 bohr, against 2.3e-5 with the spline slope
#: (Fortran-derived routines of a cell); the ``forcloc.f90`` skip of clusters is 4.3e-5 off at 6e-3 bohr.  The limit
#: of the projector rule (``V_ion.ionic_potential.PROJECTOR_GRADIENT_FLOOR``).
SMOOTH_CENTER_TABLE_LIMIT = 3.0e-3


def spline_force_center_setting() -> str:
    """``PARSEC_SPLINE_FORCE_CENTER``: the radial slopes of the local potential and of the core density in the local
    forces at grid points next to an atom, on every route (clusters, wires, slabs and both routine sets of a cell).

    The energy samples the local potential and the core density at the grid points.  Below its first radial sample
    ``rs(2)`` a table has no data: the padded cubic spline of a ``Cubic_Spline`` table continues the first value to the
    origin, takes a slope of its own there (carbon POTRE 3.6e-3 Ry/bohr, first radial sample 1.3e-5 bohr; FHI tables
    Si 0.035 (4.5e-4 bohr), Mg 3.4e-3, O 0.73, C 0.42 (1.04e-3 bohr)) and relaxes to the table over the first
    intervals.  Over a step of 1e-3 bohr the energy averages that interval out when it is narrower than the step (the
    FHI tables, whose first samples lie at 4.5e-4 to 1.25e-3 bohr) and follows the smooth behaviour ``V(0) + a r^2``,
    whose slope ``2 a d`` is neither the slope of the spline at the point nor the zero of ``forcloc.f90``'s skip.
    (The FHI numbers are those of the tables as the file gives them, ``PARSEC_FHI_ORIGIN=off``; under its default
    ``auto`` the reader continues an FHI table to 1e-5 bohr: :func:`..fhipp.fhi_origin_setting`.)

    * ``auto`` (default since 2026-11-22): every grid point closer to its atom than ``c = 2 rs(2)``
      (:func:`smooth_center_radius`) takes the slope of the smooth behaviour, ``s(c) d/c`` with ``s`` the slope the
      route takes at ``c`` (the derivative of the spline or of the linear interpolant, or the interpolated slope table
      of ``pseudo.f90``), so ``s/d`` is the constant ``s(c)/c`` there and joins the slope of the route continuously at
      ``c``.  Farther points keep the slope of their route, bit for bit, so a table sampled down to the origin (POTRE:
      ``rs(2)`` 4e-7 to 8e-5 bohr) keeps its own slope beyond ``2 rs(2)``.  Tables whose first sample lies at or beyond
      3e-3 bohr keep ``tip`` (:data:`SMOOTH_CENTER_TABLE_LIMIT`).  A deliberate departure from PARSEC 1.5
      (``forcloc.f90`` leaves the points closer than ``rs(2)`` out); the energies do not change.
    * ``tip``: the forces as of 2026-11-21: on the Fortran-derived routines of a cell the spline slopes at every
      point farther than ``min(rs(2), 1e-6 bohr)`` (the exact derivative of the energy at the point, cone included),
      on the default routines of a cell, clusters, wires and slabs the rule of ``forcloc.f90`` (the points closer
      than ``rs(2)`` left out).  Between 1e-6 bohr and ``rs(2)`` the Fortran-derived forces are then off the slope of
      the energy over realistic steps by about ``V'(0+)`` times the density at the point (8-9e-4 Ry/bohr for carbon
      FHI at 1e-6 to 5e-4 bohr from a grid point), and the ``forcloc.f90`` rule by ``2 a d`` times it.
    * ``fortran``: the rule of ``forcloc.f90`` on the Fortran-derived routines as well (as ``tip`` elsewhere).
    * ``off``: the spline slope at every positive distance on the Fortran-derived routines, as before 2026-11-21 (an
      atom within round-off of a grid point took the slope of the cone in a random direction); as ``tip`` elsewhere.
    """

    from ..fortran_numerics import switch_value

    value = switch_value(SPLINE_FORCE_CENTER_ENV, "auto")
    if value not in ("auto", "tip", "fortran", "off"):
        raise ValueError(f"{SPLINE_FORCE_CENTER_ENV} must be auto, tip, fortran or off, not {value!r}")
    return value


def smooth_center_radius(potential: ParsecPseudopotential, setting: str) -> float:
    """``c`` of the rule of ``PARSEC_SPLINE_FORCE_CENTER=auto`` for ``potential``, ``0.0`` where it does not apply.

    ``2 rs(2)`` under ``auto`` for a table whose first radial sample ``rs(2)`` lies below
    :data:`SMOOTH_CENTER_TABLE_LIMIT`; ``0.0`` (no rule) under the other values and for coarser tables.  The rule
    reaches the padded first interval of the table and as far again, where the spline still relaxes from its padding.

    The radius was chosen at fixed density between ``2 rs(2)`` and ``max(2 rs(2), 1.5e-3 bohr)`` (they differ for the
    Si and Mg FHI tables and the POTRE tables): the local term of an atom 1e-11 to 3e-3 bohr from a grid point against
    the central differences of the local energy at steps of 1e-3 and 2e-3 bohr, C, O, N, B, Si, Mg FHI and C, Ge, Au
    POTRE tables, clusters and the Fortran-derived routines of a cell (CPU, 234 positions).  The two steps disagree
    wherever ``V'(r)/r`` changes between 1e-3 and 2e-3 bohr (carbon POTRE: -813 to -796 Ry/bohr^2, 2.4e-5 Ry/bohr;
    oxygen FHI up to 1.5e-4).  The wider radius sits between the two steps and is closer to the worse of them
    (summed 5.50e-3 against 5.69e-3 Ry/bohr; ``tip`` 9.59e-2), ``2 rs(2)`` is closer to their extrapolation to zero step
    ``(4 F(1e-3) - F(2e-3))/3`` (4.22e-3 against 4.32e-3; ``tip`` 9.10e-2), and only ``2 rs(2)`` keeps the own slope
    of the tables sampled to the origin beyond their first intervals (carbon POTRE 5e-4 bohr from a grid point: 3.0e-6
    off that extrapolation with its own slope, 1.6e-5 with the wider radius).  The core terms (B, Mg FHI, Ge POTRE) are
    within 5e-7 Ry/bohr of the central differences with either radius.
    """

    if setting != "auto":
        return 0.0
    first = float(potential.radii[0])
    if first >= SMOOTH_CENTER_TABLE_LIMIT:
        return 0.0
    return 2.0 * first


def with_smooth_center(slope_of, distances: np.ndarray, radius: float) -> np.ndarray:
    """``slope_of(distances)`` with the values of the distances below ``radius`` replaced by ``slope_of(radius) d/radius``.

    ``slope_of`` maps an array of distances to the slopes of a route; ``radius`` is :func:`smooth_center_radius`
    (``0.0``: ``slope_of(distances)`` itself).
    """

    distance = np.asarray(distances, dtype=float)
    slope = slope_of(distance)
    if radius <= 0.0 or not np.any(distance < radius):
        return slope
    return apply_smooth_center(slope, distance, radius, slope_of(np.array([float(radius)])))


def apply_smooth_center(slope: np.ndarray, distances: np.ndarray, radius: float, slope_at_radius) -> np.ndarray:
    """:func:`with_smooth_center` for a caller that has the slopes and the slope at ``radius`` already."""

    if radius <= 0.0:
        return slope
    distance = np.asarray(distances, dtype=float)
    at_radius = float(np.asarray(slope_at_radius, dtype=float).reshape(-1)[0])
    return np.where(distance < radius, (at_radius / radius) * distance, slope)


def evaluate_local_potential_derivative(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    local_l: int,
    *,
    table: np.ndarray | None = None,
    use_spline: bool = False,
    spline_padding_width: int | None = None,
    direct: bool = False,
    center: float = 0.0,
    slopes: str = "tabulated",
    tail: float | None = None,
) -> np.ndarray:
    """dV_local/dr at ``distances`` with the rules of ``forcloc.f90``.

    Points closer than the first radial sample return zero: ``forcloc.f90``
    leaves them out of every force term.  From ``rs(npoint-6)`` outwards the
    value is the Coulomb derivative ``2*Z/r**2``.  In between the default is
    PARSEC's linear interpolation of ``r**2 * dV/dr``; ``direct``
    interpolates ``dV/dr`` itself (``Old_Interpolation_Format``,
    ``forcloc.f90:201-203``); with ``use_spline`` it is the derivative of
    the padded cubic spline of the potential, whichever of the two is set,
    and ``2*Z/r**2`` from the spline's own cutoff on where that lies closer
    (:meth:`~parsec_python.Pseudopotential.ParsecPseudopotential.local_spline_cutoff`, ``PARSEC_SPLINE_COULOMB_TAIL``).
    ``center`` (:func:`smooth_center_radius`, ``PARSEC_SPLINE_FORCE_CENTER=auto``): the distances below it take the
    slope of the smooth behaviour, ``s(center) d/center`` (:func:`with_smooth_center`).
    ``tail`` replaces ``rs(npoint-6)`` as the radius of the Coulomb tail: the local forces of the default routines of a
    cell pass ``rs(ns-1)``, where :meth:`ParsecPseudopotential.local_potential` itself switches to ``-2 Z/r``, so that
    they differentiate the potential they sample (a table of fewer than eight samples puts ``rs(npoint-6)`` at or
    inside its first sample, :func:`tail_start_radius`, which gave every point the Coulomb slope).

    ``slopes="interpolant"`` (:func:`linear_slope_setting`) replaces the two
    linear branches by the slope of the potential the energy samples
    (:func:`linear_local_potential_slope`, with its Coulomb tail from
    ``rs(ns-1)``); points closer than the first sample still return zero.
    A linear table under ``slopes="interpolant"`` keeps that exact slope next to the atom too: ``center`` is not
    applied to it (the smooth behaviour of ``PARSEC_SPLINE_FORCE_CENTER`` replaces the slopes of a route; the
    interpolant is the derivative of the energy itself).
    """

    if center > 0.0 and not (slopes == "interpolant" and not use_spline):
        return with_smooth_center(
            lambda points: evaluate_local_potential_derivative(
                potential,
                points,
                local_l,
                table=table,
                use_spline=use_spline,
                spline_padding_width=spline_padding_width,
                direct=direct,
                slopes=slopes,
                tail=tail,
            ),
            distances,
            center,
        )
    radius = np.asarray(distances, dtype=float)
    radii = potential.radii
    result = np.zeros_like(radius)
    if slopes == "interpolant" and not use_spline:
        return linear_local_potential_slope(potential, radius, local_l, direct=direct)
    # forcloc.f90:180-188: a point closer than rs(2) is skipped before the
    # tail test (which matters only where the tail starts inside rs(2)).
    # rs(npoint-6) of forcloc.f90 (clusters, wires, slabs), or the radius the caller passes (``tail``: rs(ns-1) for the
    # default routines of a cell, where the sampled potential becomes -2 Z/r)
    start = tail_start_radius(potential) if tail is None else float(tail)
    if use_spline:
        # the slope of the potential the energy samples: -2 Z/r from the spline's cutoff (PARSEC_SPLINE_COULOMB_TAIL;
        # rs(ns-1) under off, so that the minimum keeps the radius above)
        start = min(start, potential.local_spline_cutoff(local_l))
    tail = (radius >= start) & (radius >= radii[0])
    result[tail] = 2.0 * potential.ionic_charge / (radius[tail] * radius[tail])
    inner = (radius >= radii[0]) & ~tail
    if not np.any(inner):
        return result
    sample = radius[inner]
    if use_spline:
        if spline_padding_width is None:
            raise ValueError("spline_padding_width is required when use_spline=True")
        spline = ParsecRadialSpline.from_positive_grid(
            radii, potential.local_channel_potential(local_l), spline_padding_width
        )
        result[inner] = spline_derivative(spline, sample)
        return result
    if table is None:
        table = local_potential_derivative_table(potential, local_l)
    if direct:
        result[inner] = np.interp(sample, radii, table)
        return result
    result[inner] = np.interp(sample, radii, table * radii * radii) / (sample * sample)
    return result


def evaluate_density_derivative(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    density: np.ndarray,
    *,
    table: np.ndarray | None = None,
    use_spline: bool = False,
    spline_padding_width: int | None = None,
    center: float = 0.0,
    slopes: str = "tabulated",
    tail: float | None = None,
) -> np.ndarray:
    """d(rho_atom)/dr at ``distances`` with the rules of ``forcloc.f90``.

    Zero closer than the first radial sample and from ``rs(npoint-6)``
    outwards; linear interpolation of the derivative table in between, or
    the derivative of the padded cubic spline of ``density``.  ``center`` as in
    :func:`evaluate_local_potential_derivative` (the core density; the valence density of the self-consistency term
    is called without it), not applied under ``slopes="interpolant"`` without a spline.  ``tail`` replaces
    ``rs(npoint-6)`` (the core density of the default routines of a cell: ``rs(ns-1)``, where the sampled density is
    dropped; see :func:`evaluate_local_potential_derivative`).

    ``slopes="interpolant"`` (:func:`linear_slope_setting`, no spline): the
    slope of the linear interpolant of ``density`` that
    :func:`parsec_python.V_ion.superpose_atomic_density` samples, zero below
    the first sample and from ``rs(ns-1)`` on, where the density is dropped.
    """

    if center > 0.0 and not (slopes == "interpolant" and not use_spline):
        return with_smooth_center(
            lambda points: evaluate_density_derivative(
                potential,
                points,
                density,
                table=table,
                use_spline=use_spline,
                spline_padding_width=spline_padding_width,
                slopes=slopes,
                tail=tail,
            ),
            distances,
            center,
        )
    radius = np.asarray(distances, dtype=float)
    radii = potential.radii
    result = np.zeros_like(radius)
    if slopes == "interpolant" and not use_spline:
        inside = (radius >= radii[0]) & (radius < potential.interpolation_cutoff)
        if np.any(inside):
            result[inside] = interpolant_slope(radius[inside], radii, np.asarray(density, dtype=float))
        return result
    start = tail_start_radius(potential) if tail is None else float(tail)
    inner = (radius >= radii[0]) & (radius < start)
    if not np.any(inner):
        return result
    sample = radius[inner]
    if use_spline:
        if spline_padding_width is None:
            raise ValueError("spline_padding_width is required when use_spline=True")
        spline = ParsecRadialSpline.from_positive_grid(
            radii, density, spline_padding_width
        )
        result[inner] = spline_derivative(spline, sample)
        return result
    if table is None:
        table = density_derivative_table(potential, density)
    result[inner] = np.interp(sample, radii, table)
    return result


__all__ = [
    "DERIVATIVE_HALF_WIDTH",
    "FIRST_DERIVATIVE_WEIGHTS",
    "SMOOTH_CENTER_TABLE_LIMIT",
    "SPLINE_FORCE_CENTER_ENV",
    "apply_smooth_center",
    "density_derivative_table",
    "evaluate_density_derivative",
    "evaluate_local_potential_derivative",
    "interpolant_slope",
    "LINEAR_SLOPES",
    "LINEAR_SLOPES_ENV",
    "linear_local_potential_slope",
    "linear_projector_slope",
    "linear_slope_setting",
    "local_potential_derivative_table",
    "log_grid_derivative",
    "radial_projector_with_derivative",
    "smooth_center_radius",
    "spline_derivative",
    "spline_force_center_setting",
    "tail_start_radius",
    "valence_density_table",
    "with_smooth_center",
]
