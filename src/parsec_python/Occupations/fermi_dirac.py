"""Fermi occupations and real-space density construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expit

from ..fortran_numerics import switch_value

BOLTZMANN_RYDBERG_PER_KELVIN = 6.33327186e-6

#: The limits of the Fermi-level search of PARSEC 1.5 (``utilities/flevel.f90:43-51``):
#: ``difmax``, beyond which ``|(e - mu)/kT|`` gives an occupation of exactly one or zero
#: (36.7: ``exp(-36.7) = 1.2e-16``, below half the double-precision epsilon; the previous
#: tree had 35), ``small``, the tolerance of the electron count (1e-10, before 1e-12), and
#: ``iterlim``, the most bisection steps (1000, before 100).
FERMI_EXPONENT_CUTOFF = 36.7
FERMI_COUNT_TOLERANCE = 1.0e-10
FERMI_MAX_BISECTIONS = 1000

#: ``PARSEC_FERMI_GAP_ROOT``: where the Fermi level falls in a gap.  The bisection of
#: ``flevel.f90`` (and of every search of this package) stops at the first level whose
#: electron count is within :data:`FERMI_COUNT_TOLERANCE` of the target.  In a gap the count
#: differs from the target only by the thermal tails, the electrons in the empty levels and
#: the holes in the filled ones, and wherever both are below the tolerance the count is met:
#: the level found can lie anywhere in that part of the gap, set by the bracket and the
#: levels the search was given.  ``on`` (default) places it at a root instead
#: (:func:`gap_fermi_level`): where the levels below the level found hold the target count
#: within the tolerance and the tails there are so small that the tolerance leaves the level
#: undetermined by more than :data:`FERMI_GAP_RESOLUTION` kT, the level becomes the one root
#: of "electrons above = holes below", the count of that filling of whole levels (a target
#: within the tolerance of such a filling but not equal to it, which no physical input has,
#: is taken as the filling); that root is ``midgap + (kT/2) ln(g_h/g_l)`` for a highest filled
#: level of degeneracy (weight) ``g_h`` and a lowest empty one of ``g_l`` far from the others,
#: and the occupations are those of that level.  Metals and gaps of a few kT, whose count fixes
#: the level, keep the level of the bisection bit for bit.  ``off``: the level of the
#: bisection everywhere, as before.
FERMI_GAP_ROOT_ENV = "PARSEC_FERMI_GAP_ROOT"
#: The uncertainty (in kT) of a level found by the count tolerance from which the level is
#: placed at the root: the count changes by ``tails/kT`` per Ry in a gap, so the tolerance
#: leaves ``tolerance/tails`` kT open, more than ``FERMI_GAP_RESOLUTION`` kT where the tails
#: are below ``tolerance/FERMI_GAP_RESOLUTION`` (1e-4 of a level at the tolerance 1e-10).
FERMI_GAP_RESOLUTION = 1.0e-6
#: Levels farther than this (in kT) from the edge of their side of the gap hold less than
#: ``exp(-60) = 9e-27`` of the edge level's tail each (times the ratio of their weights) and
#: are left out of the root: a million of them move it by 1e-20 kT.
_GAP_ROOT_WINDOW = 60.0


def fermi_gap_root_setting() -> str:
    """``on`` (default) or ``off``, from ``PARSEC_FERMI_GAP_ROOT`` (``off`` under ``PARSEC_FORTRAN_NUMERICS=on`` where
    the switch is not set, :func:`~parsec_python.fortran_numerics.switch_value`); another value is an error."""

    value = switch_value(FERMI_GAP_ROOT_ENV, "on")
    if value not in ("on", "off"):
        raise ValueError(f"{FERMI_GAP_ROOT_ENV} must be on or off, got {value!r}")
    return value


def _log_sum_exp(values: np.ndarray) -> float:
    top = float(np.max(values))
    return top + math.log(float(np.sum(np.exp(values - top))))


def gap_fermi_level(
    eigenvalues: np.ndarray,
    weights: np.ndarray | None,
    target: float,
    kbt: float,
    level: float,
    count_tolerance: float = FERMI_COUNT_TOLERANCE,
) -> float | None:
    """The root of the electron count in the gap that holds ``level``, or ``None``.

    ``eigenvalues`` (any order) with ``weights`` (one per level, ``None``: one each) are the
    levels of a Fermi-level search whose weighted occupations add up to ``target``;
    ``level`` (Ry) is the level its bisection found, ``kbt`` the temperature (Ry).  The
    levels below ``level`` must hold ``target`` within ``count_tolerance`` and the thermal
    tails at ``level``, ``A = sum_above w f`` and ``B = sum_below w (1 - f)``, must be below
    ``count_tolerance / FERMI_GAP_RESOLUTION``: the count ``target + A - B`` then leaves the
    level open by more than ``FERMI_GAP_RESOLUTION`` kT.  Otherwise (a metal, a partly
    filled level, a gap of a few kT, a level on an eigenvalue) the result is ``None``.  The
    root solves ``ln A(mu) = ln B(mu)`` with the exact Fermi function (no cut-off at
    ``FERMI_EXPONENT_CUTOFF``, sums of logarithms, so that a gap of any width has its root)
    by Newton steps kept inside a bracket, to a few units in the last place.  It is the root
    of the count of the filling of the levels below ``level`` (``sum_below w``), which a target
    within the tolerance of it but not equal to it is taken for.  The same levels and the same
    ``level`` give the same root on every process.
    """

    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    held = (
        np.ones_like(values)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    level = float(level)
    kbt = float(kbt)
    if held.shape != values.shape or not (kbt > 0.0) or not np.isfinite(level):
        return None
    counted = held > 0.0
    values, held = values[counted], held[counted]
    below = values < level
    above = values > level
    if not below.any() or not above.any() or np.any(values == level):
        return None
    if abs(float(np.sum(held[below])) - float(target)) > count_tolerance:
        return None
    scaled = (values - level) / kbt
    near = np.abs(scaled) < 100.0  # a level farther holds less than exp(-100) of a tail
    tails = float(
        np.dot(held[near & above], expit(-scaled[near & above]))
        + np.dot(held[near & below], expit(scaled[near & below]))
    )
    if not tails * FERMI_GAP_RESOLUTION < count_tolerance:
        return None

    lowest_empty = float(np.min(values[above]))
    highest_filled = float(np.max(values[below]))
    upper = above & (values - lowest_empty <= _GAP_ROOT_WINDOW * kbt)
    lower = below & (highest_filled - values <= _GAP_ROOT_WINDOW * kbt)
    empty, empty_log_weights = values[upper], np.log(held[upper])
    filled, filled_log_weights = values[lower], np.log(held[lower])

    def balance(mu: float) -> tuple[float, float]:
        """``ln A - ln B`` at ``mu`` and its derivative (1/Ry, positive)."""

        x_empty = (empty - mu) / kbt
        x_filled = (filled - mu) / kbt
        terms_empty = empty_log_weights - np.logaddexp(0.0, x_empty)  # ln(w f)
        terms_filled = filled_log_weights - np.logaddexp(0.0, -x_filled)  # ln(w (1 - f))
        log_empty = _log_sum_exp(terms_empty)
        log_filled = _log_sum_exp(terms_filled)
        slope = (
            float(np.dot(np.exp(terms_empty - log_empty), expit(x_empty)))
            + float(np.dot(np.exp(terms_filled - log_filled), expit(-x_filled)))
        ) / kbt
        return log_empty - log_filled, slope

    low, high = highest_filled, lowest_empty
    step = kbt
    for _ in range(64):
        if balance(low)[0] < 0.0:
            break
        low -= step
        step *= 2.0
    else:
        return None
    step = kbt
    for _ in range(64):
        if balance(high)[0] > 0.0:
            break
        high += step
        step *= 2.0
    else:
        return None
    mu = level if low < level < high else 0.5 * (low + high)
    for _ in range(200):
        value, slope = balance(mu)
        if value == 0.0:
            break
        if value < 0.0:
            low = mu
        else:
            high = mu
        trial = mu - value / slope if slope > 0.0 and np.isfinite(slope) else float("nan")
        # converged before the bracket is consulted: a step of a few units in the last place lands on
        # the end of the bracket that ``mu`` has just become
        if np.isfinite(trial) and abs(trial - mu) <= 4.0 * float(np.spacing(max(abs(mu), abs(trial)))):
            mu = trial
            break
        if not low < trial < high:
            trial = 0.5 * (low + high)
            if trial in (low, high):
                break
        mu = trial
    return float(mu) if np.isfinite(mu) else None


@dataclass(frozen=True)
class OccupationResult:
    fermi_level: float
    occupations: np.ndarray
    electron_count: float


def _fermi_function(eigenvalues: np.ndarray, chemical_potential: float, kbt: float) -> np.ndarray:
    exponent = (eigenvalues - chemical_potential) / kbt
    result = np.empty_like(exponent)
    result[exponent >= FERMI_EXPONENT_CUTOFF] = 0.0
    result[exponent <= -FERMI_EXPONENT_CUTOFF] = 1.0
    middle = np.abs(exponent) < FERMI_EXPONENT_CUTOFF
    result[middle] = 1.0 / (np.exp(exponent[middle]) + 1.0)
    return result


def fermi_occupations(
    eigenvalues: np.ndarray,
    electron_count: float,
    temperature_kelvin: float = 80.0,
    *,
    count_tolerance: float = FERMI_COUNT_TOLERANCE,
    max_iterations: int = FERMI_MAX_BISECTIONS,
    degeneracy_tolerance: float = 1.0e-12,
    degeneracy: float = 2.0,
    weights: np.ndarray | None = None,
) -> OccupationResult:
    """Occupy states with PARSEC's ``f_i in [0,1]`` convention.

    ``degeneracy`` is the maximum electron count per orbital: 2 for the
    default spin-unpolarized case (two spin-degenerate electrons per spatial
    orbital), or 1 when occupying a single spin channel's non-degenerate
    orbitals -- e.g. by pooling both channels' eigenvalues into one call for
    a spin-polarized calculation's single shared Fermi level (``flevel.f90``),
    then splitting the returned ``occupations`` back by channel.

    ``weights`` (one per eigenvalue, in the same sorted order) are per-state
    weights, the k-point weight of each pooled level as in ``flevel.f90``
    (``weight(jj,isp) = kpwt(kplp)``): the electron count is
    ``degeneracy * sum_i w_i f_i``.  ``None`` (the default) is the unweighted
    count of every existing caller, computed exactly as before; see
    :func:`weighted_fermi_occupations`.

    At a finite temperature the level is that of the bisection, or, in a gap whose
    count does not fix it, the root of the count (``PARSEC_FERMI_GAP_ROOT``,
    :func:`gap_fermi_level`).
    """
    if weights is not None:
        return weighted_fermi_occupations(
            eigenvalues,
            weights,
            electron_count,
            temperature_kelvin,
            count_tolerance=count_tolerance,
            max_iterations=max_iterations,
            degeneracy_tolerance=degeneracy_tolerance,
            degeneracy=degeneracy,
        )
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must be finite")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    if not np.isfinite(degeneracy) or degeneracy <= 0:
        raise ValueError("degeneracy must be a positive finite number")
    if (
        not np.isfinite(electron_count)
        or electron_count < 0
        or electron_count > degeneracy * eigenvalues.size
    ):
        raise ValueError("electron count exceeds the available degenerate states")
    if (
        not np.isfinite(count_tolerance)
        or count_tolerance <= 0
        or not np.isfinite(degeneracy_tolerance)
        or degeneracy_tolerance < 0
    ):
        raise ValueError("occupation tolerances must be finite and nonnegative")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(temperature_kelvin):
        raise ValueError("temperature must be finite")
    target = float(electron_count) / degeneracy

    if temperature_kelvin == 0:
        occupations = np.zeros_like(eigenvalues)
        remaining = target
        frontier_start = 0
        frontier_stop = 0
        index = 0
        while index < eigenvalues.size and remaining > count_tolerance:
            scale = max(1.0, abs(float(eigenvalues[index])))
            stop = index + 1
            while (
                stop < eigenvalues.size
                and abs(float(eigenvalues[stop] - eigenvalues[index]))
                <= degeneracy_tolerance * scale
            ):
                stop += 1
            group_size = stop - index
            group_occupation = min(1.0, remaining / group_size)
            occupations[index:stop] = group_occupation
            remaining -= group_occupation * group_size
            frontier_start, frontier_stop = index, stop
            index = stop
        if target <= 0:
            fermi_level = float(eigenvalues[0])
        elif np.all(occupations >= 1.0 - count_tolerance):
            fermi_level = float(eigenvalues[-1])
        elif np.any(
            (occupations[frontier_start:frontier_stop] > count_tolerance)
            & (occupations[frontier_start:frontier_stop] < 1.0 - count_tolerance)
        ):
            fermi_level = float(eigenvalues[frontier_start])
        else:
            fermi_level = float(
                0.5
                * (
                    eigenvalues[frontier_stop - 1]
                    + eigenvalues[frontier_stop]
                )
            )
        return OccupationResult(
            fermi_level=fermi_level,
            occupations=occupations,
            electron_count=float(degeneracy * np.sum(occupations)),
        )
    if temperature_kelvin < 0:
        raise ValueError("negative-temperature file occupations are not supported")
    if target == 0:
        occupations = np.zeros_like(eigenvalues)
        return OccupationResult(float(eigenvalues[0]), occupations, 0.0)
    if target >= eigenvalues.size:
        raise ValueError(
            "finite-temperature occupations require at least one buffered unoccupied state"
        )

    kbt = BOLTZMANN_RYDBERG_PER_KELVIN * temperature_kelvin
    lower = float(eigenvalues[0] - 36.0 * kbt)
    upper = float(eigenvalues[-1] + 36.0 * kbt)
    occupations = _fermi_function(eigenvalues, 0.5 * (lower + upper), kbt)
    previous_chemical_potential: float | None = None
    for _ in range(max_iterations):
        chemical_potential = 0.5 * (lower + upper)
        if chemical_potential == previous_chemical_potential:
            print(f"Warning:")
            print(f"\tFermi-level bisection has stopped advancing (old chemical potential is equal to new chemical potential")
            print(f"\tTarget electron count:\t{target}")
            print(f"\tCurrent electron count:\t{count}")
            print(f"\tDifference =           \t{abs(count - target)}")
            break
        previous_chemical_potential = chemical_potential
        occupations = _fermi_function(eigenvalues, chemical_potential, kbt)
        count = float(np.sum(occupations))
        if abs(count - target) <= count_tolerance:
            break
        if count < target:
            lower = chemical_potential
        else:
            upper = chemical_potential
    else:
        raise RuntimeError("Fermi-level bisection did not reach the electron-count tolerance")

    fermi_level = float(0.5 * (lower + upper))
    if fermi_gap_root_setting() == "on":
        root = gap_fermi_level(eigenvalues, None, target, kbt, fermi_level, count_tolerance)
        if root is not None:
            at_root = _fermi_function(eigenvalues, root, kbt)
            if abs(float(np.sum(at_root)) - target) <= count_tolerance:
                fermi_level, occupations = root, at_root
    return OccupationResult(
        fermi_level=fermi_level,
        occupations=occupations,
        electron_count=float(degeneracy * np.sum(occupations)),
    )


def weighted_fermi_occupations(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    electron_count: float,
    temperature_kelvin: float = 80.0,
    *,
    count_tolerance: float = FERMI_COUNT_TOLERANCE,
    max_iterations: int = FERMI_MAX_BISECTIONS,
    degeneracy_tolerance: float = 1.0e-12,
    degeneracy: float = 2.0,
) -> OccupationResult:
    """:func:`fermi_occupations` with a weight per state (Fortran ``flevel.f90``).

    Each level ``i`` of the sorted ``eigenvalues`` holds at most
    ``degeneracy * w_i`` electrons; the Fermi level ``mu`` solves
    ``sum_i w_i f((e_i - mu)/kT) = N / degeneracy`` (``flevel.f90:225-262``:
    ``enum = sum f * weight``; the limits of PARSEC 1.5, tolerance ``1e-10``,
    ``|beta (e - mu)| > 36.7`` cut-offs, at most 1000 bisection steps,
    :data:`FERMI_COUNT_TOLERANCE`).  The bisection is the one of
    :func:`fermi_occupations` (bracket ``[e_min - 36 kT, e_max + 36 kT]``,
    midpoints), so equal weights ``w`` give the levels of
    ``fermi_occupations(..., degeneracy=degeneracy*w)`` to round-off, and the
    level in a gap is the root of the count as there (``PARSEC_FERMI_GAP_ROOT``).  At zero
    temperature the levels are filled in order, a group of degenerate levels
    sharing one fractional occupation.  The returned occupations are the
    ``f_i`` in ``[0, 1]``, without the weights.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if weights.shape != eigenvalues.shape:
        raise ValueError("weights must hold one value per eigenvalue")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("state weights must be finite and non-negative")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    if not np.isfinite(degeneracy) or degeneracy <= 0:
        raise ValueError("degeneracy must be a positive finite number")
    capacity = float(np.sum(weights))
    if (
        not np.isfinite(electron_count)
        or electron_count < 0
        or electron_count > degeneracy * capacity * (1.0 + 1.0e-12)
    ):
        raise ValueError("electron count exceeds the available degenerate states")
    if not np.isfinite(count_tolerance) or count_tolerance <= 0:
        raise ValueError("occupation tolerances must be finite and nonnegative")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(temperature_kelvin):
        raise ValueError("temperature must be finite")
    target = float(electron_count) / degeneracy

    if temperature_kelvin == 0:
        occupations = np.zeros_like(eigenvalues)
        remaining = target
        index = 0
        frontier_start = frontier_stop = 0
        while index < eigenvalues.size and remaining > count_tolerance:
            scale = max(1.0, abs(float(eigenvalues[index])))
            stop = index + 1
            while (
                stop < eigenvalues.size
                and abs(float(eigenvalues[stop] - eigenvalues[index]))
                <= degeneracy_tolerance * scale
            ):
                stop += 1
            group_weight = float(np.sum(weights[index:stop]))
            group_occupation = 1.0 if group_weight <= 0.0 else min(1.0, remaining / group_weight)
            occupations[index:stop] = group_occupation
            remaining -= group_occupation * group_weight
            frontier_start, frontier_stop = index, stop
            index = stop
        if target <= 0:
            fermi_level = float(eigenvalues[0])
        elif np.all(occupations >= 1.0 - count_tolerance):
            fermi_level = float(eigenvalues[-1])
        elif frontier_stop >= eigenvalues.size or np.any(
            (occupations[frontier_start:frontier_stop] > count_tolerance)
            & (occupations[frontier_start:frontier_stop] < 1.0 - count_tolerance)
        ):
            fermi_level = float(eigenvalues[frontier_start])
        else:
            fermi_level = float(
                0.5 * (eigenvalues[frontier_stop - 1] + eigenvalues[frontier_stop])
            )
        return OccupationResult(
            fermi_level=fermi_level,
            occupations=occupations,
            electron_count=float(degeneracy * np.dot(weights, occupations)),
        )
    if temperature_kelvin < 0:
        raise ValueError("negative-temperature file occupations are not supported")
    if target == 0:
        return OccupationResult(float(eigenvalues[0]), np.zeros_like(eigenvalues), 0.0)
    if target >= capacity:
        raise ValueError(
            "finite-temperature occupations require at least one buffered unoccupied state"
        )

    kbt = BOLTZMANN_RYDBERG_PER_KELVIN * temperature_kelvin
    lower = float(eigenvalues[0] - 36.0 * kbt)
    upper = float(eigenvalues[-1] + 36.0 * kbt)
    occupations = _fermi_function(eigenvalues, 0.5 * (lower + upper), kbt)
    previous_chemical_potential: float | None = None
    count = float(np.dot(weights, occupations))
    for _ in range(max_iterations):
        chemical_potential = 0.5 * (lower + upper)
        if chemical_potential == previous_chemical_potential:
            # The bisection cannot narrow further (the count is as close as
            # floating point allows); keep the last occupations.
            break
        previous_chemical_potential = chemical_potential
        occupations = _fermi_function(eigenvalues, chemical_potential, kbt)
        count = float(np.dot(weights, occupations))
        if abs(count - target) <= count_tolerance:
            break
        if count < target:
            lower = chemical_potential
        else:
            upper = chemical_potential
    else:
        raise RuntimeError("Fermi-level bisection did not reach the electron-count tolerance")

    fermi_level = float(0.5 * (lower + upper))
    if fermi_gap_root_setting() == "on":
        root = gap_fermi_level(eigenvalues, weights, target, kbt, fermi_level, count_tolerance)
        if root is not None:
            at_root = _fermi_function(eigenvalues, root, kbt)
            if abs(float(np.dot(weights, at_root)) - target) <= count_tolerance:
                fermi_level, occupations = root, at_root
    return OccupationResult(
        fermi_level=fermi_level,
        occupations=occupations,
        electron_count=float(degeneracy * np.dot(weights, occupations)),
    )


def density_from_orbitals(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    *,
    degeneracy: float = 2.0,
) -> np.ndarray:
    """Build ``rho = degeneracy/dV * sum_i f_i |psi_i|**2``.

    Columns of ``wavefunctions`` must have Euclidean norm one, matching PARSEC's
    eigensolver convention.  ``degeneracy`` matches
    :func:`fermi_occupations`: 2 for the default spin-unpolarized case, or 1
    when building one spin channel's density from that channel's own orbitals
    and occupations.
    """
    wavefunctions = np.asarray(wavefunctions, dtype=float)
    occupations = np.asarray(occupations, dtype=float)
    if wavefunctions.ndim != 2:
        raise ValueError("wavefunctions must have shape (grid_points, states)")
    if occupations.shape != (wavefunctions.shape[1],):
        raise ValueError("occupation count does not match the wavefunction columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    if not np.isfinite(degeneracy) or degeneracy <= 0:
        raise ValueError("degeneracy must be a positive finite number")
    return (degeneracy / volume_element) * np.sum(
        wavefunctions * wavefunctions * occupations[None, :], axis=1
    )


__all__ = [
    "BOLTZMANN_RYDBERG_PER_KELVIN",
    "FERMI_GAP_ROOT_ENV",
    "OccupationResult",
    "density_from_orbitals",
    "fermi_gap_root_setting",
    "fermi_occupations",
    "gap_fermi_level",
    "weighted_fermi_occupations",
]
