"""Fourier filter of the local pseudopotential to the grid (``PARSEC_LOCAL_FILTER``).

The local potential of an atom at ``R`` enters the energy as the sum ``h^3 sum_i V(|r_i - R|) rho(r_i)`` over the grid
points ``r_i``.  The sum depends on where ``R`` sits between the points through the components of ``V rho`` at the
reciprocal vectors of the grid (``|G| >= 2 q0`` with ``q0 = pi/h``): the "egg-box" effect.  A pseudopotential table
holds components far beyond ``q0`` near the nucleus (the O, N and C tables of FHI98PP at ``h = 0.16`` bohr: 1.9e-3,
8.9e-4 and 3.0e-4 Ry from an atom on a grid point to one between points, and spurious forces up to 3.9e-2 Ry/bohr on
O), which the grid cannot represent but samples.  Briggs, Sullivan and Bernholc (Phys. Rev. B 54, 14362 (1996)) filter
them out of the table before it is sampled; PARSEC 1.5 does so for periodic systems only (``fourier_f.f90``, whose
filtered ``vion`` no periodic consumer reads) and not at all for clusters.  This package filters the local potential
and the projectors by default (``all``); ``off`` keeps the file's tables as PARSEC 1.5 does (and is one of the values
of ``PARSEC_FORTRAN_NUMERICS=on``).

The filter here splits off the Coulomb tail with a Gaussian charge of width ``rc``,

``V(r) = [V(r) + 2 Z erf(r/rc)/r] - 2 Z erf(r/rc)/r``,

filters the short-ranged first part in reciprocal space and keeps the second one exact::

    f(q)   = 4 pi integral r^2 V_s(r) j0(q r) dr        (V_s the short-ranged part)
    V_f(r) = 1/(2 pi^2) integral q^2 f(q) F(q) j0(q r) dq - 2 Z erf(r/rc)/r

with the mask ``F(q) = erfc((q/q0 - c)/w)/2`` (:data:`MASK_CENTRE` ``c = 1.5``, :data:`MASK_WIDTH` ``w = 0.2``): ``F``
is 1 to 2e-4 at ``q0``, where the grid's own cutoff lies, 1/2 at ``1.5 q0`` and 2e-4 at ``2 q0``.  ``rc`` is
:data:`SPLIT_STEPS` grid steps (4 h), so that the Gaussian part has no components beyond ``q0`` (``exp(-(q0 rc)^2/4)``
is 2e-17) and the result does not depend on ``rc``.  The mask is smooth, so the filtered table approaches ``-2 Z/r``
as ``exp(-(w q0 r)^2/4)`` beyond the core: 10 to 14 grid steps beyond the Coulomb onset of the file it is ``-2 Z/r`` to
1e-14 Ry (O of FHI98PP: 3.6 bohr at ``h = 0.16``, 10.4 at 0.6, against 2.1 for the file), and from the last radius
where it departs by more the table is set to ``-2 Z/r`` exactly.  Both transforms are sine transforms of ``r V`` on a
uniform radial mesh of ``h/32`` (at most 0.01 bohr) to :func:`_mesh_length`, the forward one by the discrete sine
transform and the backward one summed at the radii of the table, so the filtered table has the radii, the
interpolation and every consumer of the file's table (the local potential on the grid, its periodic images, its slope
in the forces, on every route and on the GPU).  The short-ranged part is made from the cubic spline of the table with
the origin padding of the run's stencil and from ``-2 Z/r`` beyond the Coulomb onset of the file
(:func:`parsec_python.V_ion.ionic_potential.coulomb_onset_radius`), so it does not depend on how the spline ends.

``h`` is the longest grid step (:func:`filter_step`): the reciprocal vectors of a grid with steps ``s_m`` are at least
``2 pi/max |s_m|`` long, oblique or not, so ``q0 = pi/max |s_m|`` is the cutoff of every grid.

``all`` filters the Kleinman-Bylander projectors with the same mask as well (:func:`filtered_projector_tables`):
the projectors alias as the local potential does, and with the local part filtered alone their aliasing is what
remains (O2 at ``h = 0.16``: the net force oscillates by 1.9e-3 Ry/bohr from the projectors and by 9e-5 from
the local part), and where the two partly cancelled
before (the Troullier-Martins O table of ``examples/0d_water_geoopt`` at ``h = 0.3``) the local filter alone even
raises the egg-box effect.  The projectors keep a compact support through the mask-function method of Wang (Phys.
Rev. B 64, 201107 (2001)): ``R0 = 1.5 rc`` instead of ``rc`` (3.4 times the points of a projector).  A species
whose input filters its projectors (``Alpha_Filter`` of a periodic system, ``fourier_f.f90``) keeps those, and with
``Double_Grid_Order`` above 1 (which treats the projectors its own way) only the local potential is filtered; a species
whose table ends within the support of the filtered projectors (a test table of six radii) keeps both tables of its
file.  The spin-orbit projectors are not filtered (they are evaluated on the larger sphere, zero beyond ``rc``), nor
are the core-correction and atomic valence densities.

The mask-function projectors change components at and below ``q0`` on coarse grids (up to 1.6e-3 relative at
``0.75 q0`` for the d projector of the C POTRE table at ``h = 0.45``; about 1e-7 at ``h = 0.16``).  Where a grid is far
too coarse for a species the energy of one placement can move either way; the means over placements agree within the
spread of the placements, at ``h <= 0.3`` every difference is within chemical accuracy, and the spread is usually,
not always, smaller.  Five rigid placements without re-centring, error of the mean against a fine grid, ``off`` ->
``all``: Cu2 of the NCPP library -0.445 -> -0.463 Ry at ``h = 0.40`` (placements spread over 0.30 and 0.18 Ry),
-0.140 -> -0.143 at 0.35, -3.96e-2 -> -3.91e-2 at 0.30; benzene -1.06e-2 -> -1.08e-2 at 0.40 (spread 1.1e-2 and
2.3e-3), -3.9e-3 -> -3.2e-3 at 0.35 and -2.4e-3 -> -7.8e-4 at 0.30; N2 of the NCPP library -1.24e-3 -> -1.52e-3 at
0.30 (spread 7.8e-4 and 4.0e-4) (one re-centred placement of Cu2 and benzene at 0.40: -0.262 -> -0.350 and
-2.8e-3 -> -9.2e-3).  On a grid far too coarse for a species the filter can expose an error that the aliasing of the
file's tables was cancelling (an H2 slab at Gamma and X at ``h = 0.6``, 0.2 Ry); parsec.out notes such grids
(:func:`core_departure`, :data:`COARSE_NOTE_LIMIT`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from threading import Lock
from typing import Mapping

import numpy as np
from scipy.fft import dst
from scipy.special import erf, erfc, spherical_jn

from .potre import ParsecPseudopotential
from .radial_spline import ParsecRadialSpline

LOCAL_FILTER_ENV = "PARSEC_LOCAL_FILTER"
LOCAL_FILTER_DEFAULT = "all"
LOCAL_FILTER_VALUES = ("off", "on", "all")

#: Centre and width of the mask ``F(q) = erfc((q/q0 - MASK_CENTRE)/MASK_WIDTH)/2`` in units of ``q0 = pi/h``.
MASK_CENTRE = 1.5
MASK_WIDTH = 0.2
#: The mask is summed out to where it falls below this (``2.97 q0``).
MASK_FLOOR = 1.0e-17
#: Width of the Gaussian charge of the Coulomb split, in grid steps.
SPLIT_STEPS = 4.0
#: Spacing of the uniform radial mesh of the transforms, in grid steps, and its largest value (bohr).
MESH_STEPS = 1.0 / 32.0
MESH_LARGEST = 0.01
#: Beyond the last radius where the filtered table departs from ``-2 Z/r`` by more than this (Ry), it is ``-2 Z/r``.
TAIL_TOLERANCE = 1.0e-14
#: ``all``: the filtered projectors end at this multiple of the support radius of the file's projectors (the mask
#: function of :func:`filtered_projector_tables`).
PROJECTOR_SUPPORT_FACTOR = 1.5
#: parsec.out notes a grid that is coarse for a species where the filtered local table departs from the file's by more
#: than this inside the support radius (Ry, the ``r^2``-weighted root mean square of :func:`core_departure`).
COARSE_NOTE_LIMIT = 0.05

_TABLES: OrderedDict = OrderedDict()
_TABLES_LOCK = Lock()
_TABLES_SIZE = 64


def local_filter_setting() -> str:
    """``PARSEC_LOCAL_FILTER``: ``all`` (the default: the local potential Fourier-filtered to the grid, the module
    docstring, and the Kleinman-Bylander projectors with the same mask, :func:`filtered_projector_tables`), ``on`` (the
    local potential only) or ``off`` (the file's tables, as PARSEC 1.5, and the bits of the code before the default;
    also under ``PARSEC_FORTRAN_NUMERICS=on``).  An unknown value is an error."""

    from ..fortran_numerics import switch_value

    value = switch_value(LOCAL_FILTER_ENV, LOCAL_FILTER_DEFAULT)
    if value not in LOCAL_FILTER_VALUES:
        raise ValueError(f"{LOCAL_FILTER_ENV} must be one of {', '.join(LOCAL_FILTER_VALUES)}, not {value!r}")
    return value


def filter_step(grid) -> float:
    """The longest grid step of ``grid`` (bohr): ``max |s_m|`` of its step vectors, or the spacing of a cluster grid."""

    vectors = getattr(grid, "step_vectors", None)
    if vectors is not None:
        return float(np.max(np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)))
    return float(grid.spacing)


def filter_mask(q: np.ndarray, step: float) -> np.ndarray:
    """``F(q)`` of the module docstring for the grid step ``step``."""

    q0 = np.pi / float(step)
    return 0.5 * erfc((np.asarray(q, dtype=np.float64) / q0 - MASK_CENTRE) / MASK_WIDTH)


def _mesh_length(onset: float, step: float) -> float:
    """Length of the uniform radial mesh: the short-ranged part (the Coulomb onset of the table and six widths of the
    Gaussian charge), the decay of the mask's ringing (``14/(w q0)`` beyond them, where ``exp(-(w q0 r)^2/4)`` is
    below 1e-21), and as much again so that the odd periodic continuation of the sine transform stays away."""

    reach = max(float(onset), 6.0 * SPLIT_STEPS * step)
    ringing = 14.0 / (MASK_WIDTH * np.pi / step)
    return 2.0 * (reach + ringing) + 1.0


def filtered_local_table(
    potential: ParsecPseudopotential, local_l: int, step: float, padding: int
) -> np.ndarray:
    """The local potential of ``potential`` (channel ``local_l``) filtered for a grid of longest step ``step`` (bohr),
    at the radii of the table (Ry).  ``padding`` is the origin padding of the cubic spline (``Expansion_Order/2``)."""

    from ..V_ion.ionic_potential import coulomb_onset_radius

    step = float(step)
    if not (np.isfinite(step) and step > 0.0):
        raise ValueError("the grid step of the local filter must be positive")
    potential.validate_local_channel(local_l)
    if potential.filtered_local_potentials:
        # always the file's table (its Coulomb onset too), never a table filtered before
        potential = replace(potential, filtered_local_potentials={})
    values = np.asarray(potential.channel_potentials[local_l], dtype=np.float64)
    radii = np.asarray(potential.radii, dtype=np.float64)
    key = (radii.tobytes(), values.tobytes(), float(potential.ionic_charge), step, int(padding))
    with _TABLES_LOCK:
        cached = _TABLES.get(key)
    if cached is not None:
        return cached.copy()

    charge = 2.0 * float(potential.ionic_charge)
    split = SPLIT_STEPS * step
    onset = coulomb_onset_radius(potential, local_l)
    length = _mesh_length(onset, step)
    spacing = min(MESH_STEPS * step, MESH_LARGEST)
    count = int(np.ceil(length / spacing))
    spacing = length / count
    mesh = spacing * np.arange(1, count)
    spline = ParsecRadialSpline.from_positive_grid(radii, values, int(padding))
    inside = (mesh < onset) & (mesh < potential.interpolation_cutoff)
    weighted = -charge * np.ones_like(mesh)  # r V: -2 Z beyond the Coulomb onset
    weighted[inside] = mesh[inside] * spline(mesh[inside])
    short = weighted + charge * erf(mesh / split)  # r V_s
    # the sine transform of r V_s at q_k = k pi/length (trapezoid; r V_s vanishes at 0 and at length)
    transform = spacing * dst(short, type=1) / 2.0
    q = (np.pi / length) * np.arange(1, count)
    mask = filter_mask(q, step)
    kept = mask > MASK_FLOOR
    weights = (2.0 / length) * transform[kept] * mask[kept]  # (2/pi) dq G(q) F(q), dq = pi/length
    q = q[kept]
    filtered_short = np.empty_like(radii)
    reach = radii < 0.5 * length
    for start in range(0, radii.size, 512):
        block = radii[start : start + 512]
        filtered_short[start : start + 512] = (np.sin(np.outer(block, q)) @ weights) / block
    filtered_short[~reach] = 0.0
    table = filtered_short - charge * erf(radii / split) / radii
    coulomb = -charge / radii
    significant = np.flatnonzero(np.abs(table - coulomb) > TAIL_TOLERANCE)
    last = int(significant[-1]) if significant.size else -1
    table[last + 1 :] = coulomb[last + 1 :]
    if not np.all(np.isfinite(table)):
        raise ValueError(f"the filtered local potential of {potential.source} is not finite")
    with _TABLES_LOCK:
        _TABLES[key] = table.copy()
        while len(_TABLES) > _TABLES_SIZE:
            _TABLES.popitem(last=False)
    return table


def projector_mask_function(x: np.ndarray) -> np.ndarray:
    """The real-space mask ``m(x) = exp(-x^2/(1 - x^2))`` of :func:`filtered_projector_tables` (``x = r/R0``; zero
    from ``x = 1`` on, smooth there to every order)."""

    x = np.asarray(x, dtype=np.float64)
    inside = x < 1.0
    result = np.zeros_like(x)
    result[inside] = np.exp(-x[inside] ** 2 / (1.0 - x[inside] ** 2))
    return result


def filtered_projector_tables(
    potential: ParsecPseudopotential, local_l: int, step: float, padding: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]:
    """The normalized Kleinman-Bylander projectors of every nonlocal channel filtered for a grid of longest step
    ``step`` (bohr), their radial derivatives, both at the radii of the table, and the radius ``R0`` they end at
    (``PARSEC_LOCAL_FILTER=all``).

    The mask-function method of Wang (Phys. Rev. B 64, 201107 (2001)): with the smooth mask ``m(r/R0)``
    (:func:`projector_mask_function`, ``R0 = gamma rc``, ``rc`` the support radius of the file's projector,
    ``gamma`` :data:`PROJECTOR_SUPPORT_FACTOR`), ``p/m`` is filtered in reciprocal space and multiplied by ``m``
    again::

        g(q)   = 4 pi integral_0^rc r^2 (p/m)(r) j_l(q r) dr
        p_f(r) = m(r/R0) / (2 pi^2) integral q^2 g(q) G(q) j_l(q r) dq

    with ``G(q)`` the mask ``F(q)`` of the local potential (:func:`filter_mask`).  The mask function is smooth over
    ``R0``, so the components of ``p_f`` stay within a few ``1/R0`` of those of the filtered ``p/m``, and ``p_f`` ends
    at ``R0`` without a jump.  ``p`` is the cubic spline of the file's projector on a uniform radial mesh of ``h/32``
    (at most 0.01 bohr); the slope is the derivative of ``p_f`` (``q j_l'(q r)`` and ``m'``).  A low-pass that cuts
    the projectors below ``q0`` was tried and refused: it removes components the orbitals have on a coarse grid
    (64-atom diamond, C POTRE at ``h = 0.374``: +1.0 Ry).  ``F`` itself is 1 to 2e-4 up to ``q0``, but the mask
    function still changes the components of ``p_f`` at and below ``q0`` on a coarse grid (up to 1.6e-3 relative at
    ``0.75 q0`` for the d projector of the C POTRE table at ``h = 0.45``; about 1e-7 at ``h = 0.16``).  A table whose
    second-to-last radius does not reach ``R0`` (every table of the pseudopotential libraries reaches six times its
    projector support; a test table of six radii does not) gives no filtered projectors: the species keeps the file's.
    """

    step = float(step)
    if not (np.isfinite(step) and step > 0.0):
        raise ValueError("the grid step of the projector filter must be positive")
    radii = np.asarray(potential.radii, dtype=np.float64)
    support = float(potential.projector_support_radius)
    outer = PROJECTOR_SUPPORT_FACTOR * support
    if not projector_support_fits(potential):
        # a table that ends within the support of the filtered projectors (the mask function would vanish where the
        # projector does not; a test table of a few radii) gives no filtered projectors
        return {}, {}, support
    spacing = min(MESH_STEPS * step, MESH_LARGEST)
    count = int(np.ceil(support / spacing))
    spacing = support / count
    mesh = spacing * np.arange(0, count + 1)
    weights = np.full(mesh.size, spacing)
    weights[0] = weights[-1] = 0.5 * spacing
    mask_mesh = projector_mask_function(mesh / outer)
    q0 = np.pi / step
    q_step = np.pi / (2.0 * outer)
    q = q_step * np.arange(1, int(np.ceil((MASK_CENTRE + 8.0 * MASK_WIDTH) * q0 / q_step)) + 1)
    lowpass = filter_mask(q, step)
    inside = radii < outer
    x = radii[inside] / outer
    mask = projector_mask_function(x)
    # d m/dr = m * (-2 x / (1 - x^2)^2) / R0
    mask_slope = mask * (-2.0 * x / (1.0 - x * x) ** 2) / outer
    projectors: dict[int, np.ndarray] = {}
    slopes: dict[int, np.ndarray] = {}
    for channel in sorted(potential.radial_wavefunctions):
        if channel == local_l or channel not in potential.channel_potentials:
            continue
        table, _ = potential.radial_projector(channel, local_l)
        spline = ParsecRadialSpline.from_positive_grid(radii, table, int(padding))
        values = spline(np.maximum(mesh, radii[0])) / mask_mesh
        transform = 4.0 * np.pi * (spherical_jn(channel, np.outer(q, mesh)) @ (weights * mesh * mesh * values))
        coefficients = q_step * q * q * transform * lowpass / (2.0 * np.pi * np.pi)
        argument = np.outer(radii[inside], q)
        smooth = spherical_jn(channel, argument) @ coefficients
        smooth_slope = spherical_jn(channel, argument, derivative=True) @ (coefficients * q)
        filtered = np.zeros_like(radii)
        slope = np.zeros_like(radii)
        filtered[inside] = mask * smooth
        slope[inside] = mask * smooth_slope + mask_slope * smooth
        if not (np.all(np.isfinite(filtered)) and np.all(np.isfinite(slope))):
            raise ValueError(f"the filtered projector l={channel} of {potential.source} is not finite")
        projectors[channel] = filtered
        slopes[channel] = slope
    return projectors, slopes, outer


def projector_support_fits(potential: ParsecPseudopotential) -> bool:
    """Whether the table reaches the support of its filtered projectors (``PROJECTOR_SUPPORT_FACTOR`` times its
    projector support radius) before its last radius: every table of the pseudopotential libraries does (six times
    and more), a test table of six radii does not."""

    radii = np.asarray(potential.radii, dtype=np.float64)
    return bool(PROJECTOR_SUPPORT_FACTOR * float(potential.projector_support_radius) < float(radii[-2]))


def nonlocal_channels(potential: ParsecPseudopotential, local_l: int) -> list[int]:
    """The channels of ``potential`` that have a Kleinman-Bylander projector with the local channel ``local_l``."""

    return [l for l in sorted(potential.radial_wavefunctions) if l != local_l and l in potential.channel_potentials]


def core_departure(potential: ParsecPseudopotential, local_l: int, table: np.ndarray, padding: int = 6) -> float:
    """How far the filtered local ``table`` departs from the file's inside the projector support radius ``rc``: the
    ``r^2``-weighted root mean square of the difference of the two cubic splines on a uniform mesh, Ry.  The weight
    keeps a structure of the file the grid cannot hold at the very centre (the spike of the Fe table of the tests at
    0.03 bohr, 5 Ry in either form at every spacing) from counting; the H table of ``examples/0d_water_geoopt`` gives
    0.107 at ``h = 0.6``, 0.028 at 0.45 and 0.006 at 0.3, its O 0.41, 0.086 and 0.020, C of ``examples/0d_benzene``
    0.067 at 0.6 and 0.014 at 0.4, Cu of the NCPP library 0.026 at 0.6."""

    radii = np.asarray(potential.radii, dtype=np.float64)
    support = float(potential.projector_support_radius)
    r = np.linspace(1.0e-4, support, 4000)
    file_values = ParsecRadialSpline.from_positive_grid(radii, np.asarray(potential.channel_potentials[local_l]),
                                                        int(padding))(r)
    filtered_values = ParsecRadialSpline.from_positive_grid(radii, np.asarray(table), int(padding))(r)
    weight = r * r
    difference = filtered_values - file_values
    return float(np.sqrt(np.sum(weight * difference * difference) / np.sum(weight)))


def filtered_local_reach(potential: ParsecPseudopotential, local_l: int) -> float:
    """The radius (bohr) a sum over periodic images of the local table must reach: 0 for the file's table (the rules of
    the routes stand), the Coulomb onset of a filtered table, whose ringing reaches 10 to 14 grid steps beyond the
    core (:func:`parsec_python.V_ion.ionic_potential.coulomb_onset_radius`)."""

    if local_l not in potential.filtered_local_potentials:
        return 0.0
    from ..V_ion.ionic_potential import coulomb_onset_radius

    return float(coulomb_onset_radius(potential, local_l))


def apply_local_filters(
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping,
    grid,
) -> dict[str, ParsecPseudopotential]:
    """``potentials`` with the local channel of every species filtered to ``grid`` under ``PARSEC_LOCAL_FILTER`` ``on``
    and ``all`` (the default) (:attr:`ParsecPseudopotential.filtered_local_potentials`), under ``all`` its projectors too
    (``filtered_projectors``, ``filtered_projector_slopes``, ``filtered_projector_radius``); unchanged under ``off``."""

    result = dict(potentials)
    setting = local_filter_setting()
    if setting == "off":
        return result
    step = filter_step(grid)
    padding = int(grid.settings.stencil_half_width)
    double_grid = int(getattr(grid.settings, "double_grid_order", 1) or 1)
    for symbol, specification in specifications.items():
        if symbol not in result:
            continue
        potential = result[symbol]
        local_l = int(specification.local_angular_momentum)
        filter_projectors = setting == "all" and not potential.filtered_projectors and double_grid <= 1
        if filter_projectors and nonlocal_channels(potential, local_l) and not projector_support_fits(potential):
            # the local part filtered alone is the configuration of on, worse where the local and the projector
            # aliasing partly cancel: such a species keeps both tables of its file
            result[symbol] = replace(
                potential,
                local_filter_note="the tables of the file: the table ends within the support of the filtered projectors",
            )
            continue
        table = filtered_local_table(potential, local_l, step, padding)
        changes = {"filtered_local_potentials": {local_l: table}}
        departure = core_departure(potential, local_l, table, padding)
        if departure > COARSE_NOTE_LIMIT:
            changes["local_filter_note"] = (
                f"the grid is coarse for this species: its local potential filtered to the grid departs from the file's "
                f"by {departure:.3f} Ry inside {float(potential.projector_support_radius):.2f} bohr (r^2-weighted rms); "
                "its energies on this grid can carry errors that the aliasing of the file's tables partly cancelled"
            )
        if filter_projectors:
            # (a species whose input filters its projectors, Alpha_Filter of a periodic system, keeps those; the
            # double grid treats the projectors its own way)
            projectors, slopes, end = filtered_projector_tables(potential, local_l, step, padding)
            if projectors:
                changes.update(
                    filtered_projectors=projectors, filtered_projector_slopes=slopes, filtered_projector_radius=end
                )
        result[symbol] = replace(potential, **changes)
    return result


__all__ = [
    "LOCAL_FILTER_DEFAULT",
    "LOCAL_FILTER_ENV",
    "apply_local_filters",
    "core_departure",
    "nonlocal_channels",
    "projector_support_fits",
    "filter_mask",
    "filter_step",
    "filtered_local_reach",
    "filtered_local_table",
    "filtered_projector_tables",
    "projector_mask_function",
    "local_filter_setting",
]
