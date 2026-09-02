"""Allocation-bounded recurrence for isolated Hartree multipoles.

The reference implementation evaluates SciPy spherical harmonics separately
for every ``(l,m)`` and every density update.  On molecular grids that special
function work can dominate the Poisson solve.  This module evaluates the same
normalized complex harmonics through associated-Legendre recurrences using a
few length-N work arrays; it neither stores a dense boundary map nor changes
the multipole truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Hartree import MultipoleExpansion


def _normalization(angular_momentum: int, magnetic: int) -> float:
    """Return the normalized complex ``Y_lm`` prefactor for ``m >= 0``."""

    log_ratio = math.lgamma(angular_momentum - magnetic + 1) - math.lgamma(
        angular_momentum + magnetic + 1
    )
    return math.sqrt(
        (2 * angular_momentum + 1)
        * math.exp(log_ratio)
        / (4.0 * math.pi)
    )


def _angular_coordinates(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points, axis=1)
    cosine = np.ones_like(radius)
    nonzero = radius > 0.0
    cosine[nonzero] = points[nonzero, 2] / radius[nonzero]
    cosine = np.clip(cosine, -1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    xy_radius = np.hypot(points[:, 0], points[:, 1])
    phase_positive = np.ones(points.shape[0], dtype=np.complex128)
    away_from_axis = xy_radius > 0.0
    phase_positive[away_from_axis] = (
        points[away_from_axis, 0] + 1j * points[away_from_axis, 1]
    ) / xy_radius[away_from_axis]
    return radius, cosine, sine, phase_positive


def _positive_m_harmonic_rows(
    points: np.ndarray,
    order: int,
):
    """Yield ``(l,m,Y_lm,radius)`` for every nonnegative ``m``.

    ``P_l^m`` includes the Condon--Shortley phase, matching
    ``scipy.special.sph_harm_y`` and PARSEC's complex-harmonic convention.
    Only three real associated-Legendre arrays are live for one ``m``.
    """

    radius, cosine, sine, phase_unit = _angular_coordinates(points)
    diagonal = np.ones(points.shape[0], dtype=np.float64)  # P_0^0
    phase = np.ones(points.shape[0], dtype=np.complex128)

    for magnetic in range(order + 1):
        if magnetic:
            diagonal = -(2 * magnetic - 1) * sine * diagonal
            phase = phase * phase_unit

        previous = diagonal
        yield (
            magnetic,
            magnetic,
            _normalization(magnetic, magnetic) * previous * phase,
            radius,
        )
        if magnetic == order:
            continue

        current = (2 * magnetic + 1) * cosine * diagonal
        yield (
            magnetic + 1,
            magnetic,
            _normalization(magnetic + 1, magnetic) * current * phase,
            radius,
        )
        for angular_momentum in range(magnetic + 2, order + 1):
            following = (
                (2 * angular_momentum - 1) * cosine * current
                - (angular_momentum + magnetic - 1) * previous
            ) / (angular_momentum - magnetic)
            yield (
                angular_momentum,
                magnetic,
                _normalization(angular_momentum, magnetic)
                * following
                * phase,
                radius,
            )
            previous, current = current, following


@dataclass(frozen=True)
class FastMultipoleExpansion(MultipoleExpansion):
    """Reference-compatible expansion with recurrence-based evaluation."""

    def potential(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("boundary points must have shape (n, 3)")
        radius = np.linalg.norm(points, axis=1)
        if np.any(radius <= 0.0):
            raise ValueError("multipole boundary potential is undefined at the origin")

        result = np.zeros(points.shape[0], dtype=np.complex128)
        for angular_momentum, magnetic, harmonic, _ in _positive_m_harmonic_rows(
            points, self.order
        ):
            factor = (
                4.0
                * np.pi
                / (2 * angular_momentum + 1)
                * radius ** (-(angular_momentum + 1))
            )
            result += (
                factor
                * self.moments[(angular_momentum, magnetic)]
                * harmonic
            )
            if magnetic:
                negative_harmonic = ((-1) ** magnetic) * np.conjugate(harmonic)
                result += (
                    factor
                    * self.moments[(angular_momentum, -magnetic)]
                    * negative_harmonic
                )
        return 2.0 * result.real

    __call__ = potential


def density_multipoles_fast(
    density: np.ndarray,
    grid: RealSpaceGrid,
    order: int = 9,
) -> FastMultipoleExpansion:
    """Compute exactly the reference ``Q_lm`` moments without special calls."""

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    if order < 0:
        raise ValueError("multipole order cannot be negative")

    weighted_density = density * grid.volume_element
    moments: dict[tuple[int, int], complex] = {}
    for angular_momentum, magnetic, harmonic, radius in _positive_m_harmonic_rows(
        grid.coordinates, order
    ):
        moment = complex(
            np.sum(
                weighted_density
                * radius**angular_momentum
                * np.conjugate(harmonic)
            )
        )
        moments[(angular_momentum, magnetic)] = moment
        if magnetic:
            moments[(angular_momentum, -magnetic)] = (
                ((-1) ** magnetic) * np.conjugate(moment)
            )
    return FastMultipoleExpansion(order=order, moments=moments)


__all__ = ["FastMultipoleExpansion", "density_multipoles_fast"]
