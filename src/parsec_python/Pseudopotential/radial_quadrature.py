"""Radial integration weights used by PARSEC pseudopotentials."""

from __future__ import annotations

import numpy as np


def parsec_radial_integral(radii: np.ndarray, integrand: np.ndarray) -> float:
    """Integrate tabulated radial data with PARSEC's endpoint convention.

    PARSEC conceptually inserts ``r=0`` with a zero integrand before the
    strictly positive POTRE samples.  For positive samples ``r_i, f_i``, its
    quadrature is

    ``0.5*r_1*f_0
      + sum_(i=1..N-2) 0.5*(r_(i+1)-r_(i-1))*f_i
      + (r_(N-1)-r_(N-2))*f_(N-1)``.

    The last sample receives a full final-interval weight rather than the
    half weight of the ordinary trapezoidal rule.  POTRE projector integrands
    normally vanish there, but retaining this rule gives source-level parity
    and avoids the removed NumPy ``np.trapz`` API.
    """
    radial_grid = np.asarray(radii, dtype=float)
    values = np.asarray(integrand, dtype=float)
    if radial_grid.ndim != 1 or values.shape != radial_grid.shape:
        raise ValueError(
            "radii and integrand must be one-dimensional arrays of equal length"
        )
    if radial_grid.size < 2:
        raise ValueError("at least two positive radial samples are required")
    if not np.all(np.isfinite(radial_grid)) or not np.all(np.isfinite(values)):
        raise ValueError("radial quadrature data must be finite")
    if radial_grid[0] <= 0.0 or np.any(np.diff(radial_grid) <= 0.0):
        raise ValueError("radial quadrature radii must be strictly increasing and positive")

    result = 0.5 * radial_grid[1] * values[0]
    if radial_grid.size > 2:
        weights = 0.5 * (radial_grid[2:] - radial_grid[:-2])
        result += float(weights @ values[1:-1])
    result += (radial_grid[-1] - radial_grid[-2]) * values[-1]
    return float(result)


KB_QUADRATURE_ENV = "PARSEC_KB_QUADRATURE"
KB_QUADRATURE_SETTINGS = ("trapezoid", "logmesh")
KB_QUADRATURE_DEFAULT = "trapezoid"


def kb_quadrature_setting() -> str:
    """``PARSEC_KB_QUADRATURE``: the quadrature of the Kleinman-Bylander denominators ``integral u_l^2 (V_l - V_local)
    dr`` of the radial tables.

    ``trapezoid`` (default): :func:`parsec_radial_integral`, the weights ``0.5*(r(i+1) - r(i-1))`` of ``pseudo.f90``.
    On the logarithmic mesh ``r(i) = a*exp(b*i) - c`` of a table these weights are ``(r(i) + c)*sinh(b)``, the exact
    Jacobian ``b*(r(i) + c)`` times ``sinh(b)/b = 1 + b^2/6``: every denominator comes out too large by ``b^2/6``
    (2.6e-5 for the common ``b = 0.0125``), every projector too small by ``b^2/12``, and the nonlocal energy of an
    occupied channel too small in magnitude by ``b^2/6`` of itself.  With ``Cubic_Spline`` this is the whole error of
    the tables at their own sampling: +1.4e-3 Ry per atom for Au, +1.9e-3 for Ag, +1.5e-3 for Fe, +7.5e-4 for Cr (the
    large KB energies of full d shells), 3e-5 to 5e-5 for H2O, Si and C, -2.6e-6 for Ge (KB integral of its s channel
    0.09 Ry), against the limit of the same potentials sampled up to eight times more densely (A100, command line,
    ``PARSEC_PROJECTOR_TAIL=auto``); PARSEC 1.5 has the same error.

    ``logmesh`` (opt-in): the trapezoid rule in the mesh index with the exact Jacobian ``b*(r(i) + c)``, which for these
    smooth integrands that vanish at both ends of the table is exact to the digits of the table; with ``Cubic_Spline``
    the energies then equal those of the dense-table limit within 3e-7 Ry per atom on clusters and 2.2e-6 on cells.  A
    table whose radii do not follow the logarithmic mesh of its parameters (radius by radius, to 1e-6 relative) is
    refused under ``logmesh``.  Not the default: it moves the energy of
    every run with an occupied nonlocal channel away from PARSEC 1.5 with the same table.  An unknown value is refused
    when the projectors are prepared, before the SCF.
    """

    import os

    value = os.environ.get(KB_QUADRATURE_ENV, KB_QUADRATURE_DEFAULT).strip().lower() or KB_QUADRATURE_DEFAULT
    if value not in KB_QUADRATURE_SETTINGS:
        raise ValueError(f"{KB_QUADRATURE_ENV} must be one of {', '.join(KB_QUADRATURE_SETTINGS)}, not {value!r}")
    return value


def log_mesh_integral(radii: np.ndarray, integrand: np.ndarray, radial_parameter_b: float, radial_offset: float) -> float:
    """Integrate tabulated radial data on the logarithmic mesh ``r(t) = a*exp(b*t) - c`` by the trapezoid rule in ``t``
    with the exact Jacobian ``dr/dt = b*(r + c)``.

    A mesh with ``c > 0`` (the Martins files, ``c = a``) reaches ``r = 0`` at ``t = 0``, one step before its first
    sample, where the integrand is zero: the first sample takes a full step.  A pure exponential mesh (``c = 0``, FHI
    files) starts at its first sample and adds ``0.5*r_1*f_1`` for the interval from the origin, as PARSEC does.  The
    last sample takes half a step (the integrands of the projector denominators vanish there).
    """

    radial_grid = np.asarray(radii, dtype=float)
    values = np.asarray(integrand, dtype=float)
    if radial_grid.ndim != 1 or values.shape != radial_grid.shape:
        raise ValueError("radii and integrand must be one-dimensional arrays of equal length")
    if radial_grid.size < 3:
        raise ValueError("at least three positive radial samples are required")
    if not np.all(np.isfinite(radial_grid)) or not np.all(np.isfinite(values)):
        raise ValueError("radial quadrature data must be finite")
    if radial_grid[0] <= 0.0 or np.any(np.diff(radial_grid) <= 0.0):
        raise ValueError("radial quadrature radii must be strictly increasing and positive")
    b = float(radial_parameter_b)
    c = float(radial_offset)
    if not (b > 0.0 and np.isfinite(b) and c >= 0.0 and np.isfinite(c)):
        raise ValueError(f"invalid logarithmic mesh parameters b={b!r}, c={c!r}")
    # the mesh must be the one of its parameters: r(i) = (r(1) + c)*exp(b*(i - 1)) - c, compared radius by radius
    # (differences of neighbouring radii would magnify the rounding of the printed radii by about 1/(2b))
    predicted = (radial_grid[0] + c) * np.exp(b * np.arange(radial_grid.size)) - c
    deviation = float(np.max(np.abs(radial_grid / predicted - 1.0)))
    if not deviation <= 1.0e-6:
        raise ValueError(
            f"{KB_QUADRATURE_ENV}=logmesh needs the logarithmic mesh r = a*exp(b*i) - c of the table's parameters "
            f"(b={b:.6g}, c={c:.6g}); its radii deviate from it by {deviation:.1e} (relative)"
        )
    terms = values * (b * (radial_grid + c))
    if c > 0.0:
        result = float(np.sum(terms[:-1])) + 0.5 * float(terms[-1])
    else:
        result = 0.5 * float(terms[0]) + float(np.sum(terms[1:-1])) + 0.5 * float(terms[-1])
        result += 0.5 * float(radial_grid[0] * values[0])
    return result


__all__ = [
    "KB_QUADRATURE_ENV",
    "kb_quadrature_setting",
    "log_mesh_integral",
    "parsec_radial_integral",
]
