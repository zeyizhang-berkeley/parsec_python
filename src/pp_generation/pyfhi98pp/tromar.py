"""Norm-conserving pseudopotentials of Troullier-Martins type.

Port of ``tromarLAPACK.f`` (N. Troullier, J.L. Martins, Phys. Rev. B 43,
1993 (1991)).  The pseudo wave function inside the cutoff radius is
``r**(l+1) * exp(p(r))`` with ``p = c0 + c2 r**2 + ... + c12 r**12``; the
coefficients follow from the continuity of the wave function and its first
four derivatives at ``r(nrc)`` (a 5x5 linear system), norm conservation
(``cc0``, inner iteration) and zero curvature of the screened potential at
the origin (``cc2``, outer iteration).

Behaviour of the Fortran that is kept on purpose:

* ``dgesv`` overwrites the matrix ``ac`` with its LU factors.  Inside the
  norm iteration the matrix is restored from ``acsav`` before every solve,
  but the first solve of each outer (curvature) iteration is not preceded
  by a restore.  From the second outer iteration on it therefore factors
  the *already factored* matrix and returns meaningless coefficients; they
  only serve as the starting point of the following norm iteration (whose
  first step is then ``cc0 = cc0 - 0.25`` up to rounding).  The right-hand
  side of that solve also lacks the ``-cc0`` term.
* The secant steps divide by differences that may vanish; the resulting
  Inf/NaN propagate (no trap) and end in one of the "convergence error"
  stops.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np

from . import libm
from .errors import FortranStop
from .fortran_list import list_record
from .lapack import dgesv
from .radial import _fdiv, _powi, seqsum

__all__ = ["TromarResult", "tromar", "JMX", "KMX"]

JMX: int = 20   # maximum number of iterations, zero curvature at origin
KMX: int = 20   # maximum number of iterations, norm conservation


class TromarResult(NamedTuple):
    """Output arguments of ``tromar`` (arrays are new objects)."""

    nrc: int           # grid index next to the cutoff radius (Fortran index)
    u: np.ndarray      # pseudo wave function up to nrc, all-electron beyond
    up: np.ndarray     # dto. 1st derivative (on the logarithmic grid)
    upp: np.ndarray    # dto. 2nd derivative
    vps: np.ndarray    # screened pseudopotential up to nrc, v beyond
    cc: tuple[float, ...] = ()   # cc0 ... cc6 (diagnostic)
    iterations: tuple[int, int] = (0, 0)   # j, k at exit (diagnostic)
    nsolve: int = 0    # number of dgesv calls (diagnostic)


def _slog(x: float) -> float:
    """``log`` with the IEEE results for zero and negative arguments."""
    if x > 0.0:
        return libm.slog(x)
    if x == 0.0:
        return -math.inf
    return math.nan


def _sexp(x: float) -> float:
    """``exp`` with the IEEE result on overflow."""
    try:
        return libm.sexp(x)
    except OverflowError:
        return math.inf


@np.errstate(all="ignore")
def tromar(l: int, rc: float, ein: float, mmax: int, r: np.ndarray,
           u: np.ndarray, up: np.ndarray, upp: np.ndarray, v: np.ndarray,
           warn: Callable[[str], None] | None = None) -> TromarResult:
    """Troullier-Martins pseudopotential for one channel (``tromarLAPACK.f``).

    Arguments as in ``tromar(l,rc,nrc,ein,mmax,r,u,up,upp,v,vps)``:
    ``l`` angular momentum, ``rc`` cutoff radius, ``ein`` reference energy,
    ``u, up, upp`` the all-electron wave function and its derivatives with
    respect to the logarithmic variable, ``v`` the all-electron potential.
    ``warn`` receives the lines written to the error unit.  The stops of the
    Fortran raise :class:`FortranStop`.
    """
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    u = np.array(u[:mmax], dtype=float)
    up = np.array(up[:mmax], dtype=float)
    upp = np.array(upp[:mmax], dtype=float)
    v = np.asarray(v, dtype=float)
    ein = float(ein)

    ll = l + 1
    al = 0.1e0 * libm.slog(float(r[10]) / float(r[0]))
    nrc = int(_slog(_fdiv(float(rc), float(r[0]))) / al + 1)
    k0 = nrc - 1                      # Python index of the mesh point nrc
    if nrc < 3 or nrc > mmax:
        # the Fortran would index its arrays out of bounds here
        raise FortranStop("tromar - stop: cutoff radius outside the radial mesh")
    if len(v) < mmax + 2:
        # v(nrc+1), v(nrc+2) may lie beyond mmax; fhipp supplies zeros there
        padded = np.zeros(mmax + 2)
        padded[:len(v)] = v
        v = padded

    # we want a positive wavefunction at infinity
    if u[k0] < 0.0:
        u = -u
        up = -up
        upp = -upp

    rsq = r * r
    rll = _powi(r, ll)

    cc0 = 0.e0
    cc2 = 0.e0

    # charge inside, nothing fancy just the trapezoidal rule
    u1 = float(u[0])
    r1 = float(r[0])
    unrc = float(u[k0])
    truecore = 0.5 * (u1 * u1 * r1 + unrc * unrc * float(r[k0]))
    truecore = seqsum(u[1:nrc - 1] * u[1:nrc - 1] * r[1:nrc - 1], truecore)
    truecore = al * truecore + 0.5 * u1 * u1 * r1

    rc1 = float(r[k0])
    rc2 = rc1 * rc1
    rc3 = rc1 * rc2
    rc4 = rc1 * rc3
    rc5 = rc1 * rc4
    rc6 = rc1 * rc5
    rc7 = rc1 * rc6
    rc8 = rc1 * rc7
    rc9 = rc1 * rc8
    rc10 = rc1 * rc9
    rc11 = rc1 * rc10
    rc12 = rc1 * rc11

    # derivatives of effective potential
    vir0 = float(v[k0])
    vm2, vm1, vp1, vp2 = float(v[k0 - 2]), float(v[k0 - 1]), float(v[k0 + 1]), float(v[k0 + 2])
    vir2 = (16 * (vm1 + vp1) - (30 * vir0 + vm2 + vp2)) / 12.0e0
    vir1 = (vm2 + 8 * (vp1 - vm1) - vp2) / 12.0e0

    alr = al * rc1
    vir2 = (vir2 - al * vir1) / (alr * alr)
    vir1 = vir1 / alr

    eq1 = _slog(_fdiv(unrc, float(_powi(rc1, ll))))
    eq2 = _fdiv(float(up[k0]) / alr, unrc) - ll / rc1
    eq3 = 2 * (vir0 - ein - ll * eq2 / rc1) - eq2 * eq2
    eq4 = 2 * (vir1 + ll * (eq2 / rc1 - eq3) / rc1 - eq2 * eq3)
    eq5 = 2 * (vir2 + ll * (2 * (eq3 - eq2 / rc1) / rc1 - eq4) / rc1 - eq3 * eq3 - eq2 * eq4)

    # the matrix ac(m,n) as a list of columns
    acsav = [
        [rc2, 2 * rc1, 2.e0, 0.e0, 0.e0],
        [rc6, 6 * rc5, 30 * rc4, 120 * rc3, 360 * rc2],
        [rc8, 8 * rc7, 56 * rc6, 336 * rc5, 1680 * rc4],
        [rc10, 10 * rc9, 90 * rc8, 720 * rc7, 5040 * rc6],
        [rc12, 12 * rc11, 132 * rc10, 1320 * rc9, 11880 * rc8],
    ]
    ac = [col[:] for col in acsav]

    def solve(bc: list[float], suffix: str) -> None:
        _, info = dgesv(5, ac, bc)
        if info != 0:
            # (the messages name dnlcc7 in the Fortran source)
            text = "dnlcc7 - stop: bad input" if info > 0 else "dnlcc7 - stop: singular matrix"
            if warn:
                warn(list_record("& " + text + suffix))
            raise FortranStop(text + suffix)

    rs = rsq[:nrc]
    rn = r[:nrc]
    rlln = rll[:nrc]
    rnrc = float(r[k0])
    ups = np.zeros(nrc)
    cc1 = cc3 = cc4 = cc5 = cc6 = 0.e0
    eq0 = eq6 = 0.e0
    cc0inc = 0.e0
    nsolve = 0

    # zero curvature at origin condition
    cc2inc = 0.25e0
    eq6old = 0.e0
    j = 1
    k = 1
    while j <= JMX:
        bc = [eq1 - rc4 * cc2,
              eq2 - 4 * rc3 * cc2,
              eq3 - 12 * rc2 * cc2,
              eq4 - 24 * rc1 * cc2,
              eq5 - 24 * cc2]
        solve(bc, "")
        nsolve += 1
        cc1, cc3, cc4, cc5, cc6 = bc

        # norm conservation: find cc0 by newton method
        cc0inc = 0.25e0
        eq0old = 0.e0
        k = 1
        while k <= KMX:
            polynom = rs * (cc1 + rs * (cc2 + rs * (cc3 + rs * (cc4 + rs * (cc5 + rs * cc6)))))
            ups = rlln * libm.exp(polynom)
            pseucore = seqsum(rn * ups * ups)
            upsn = float(ups[k0])
            ups1 = float(ups[0])
            pseucore = al * (pseucore - 0.5 * (upsn * upsn * rnrc + ups1 * ups1 * r1)) \
                + 0.5 * ups1 * ups1 * r1
            eq0 = _slog(_fdiv(truecore, pseucore)) - 2 * cc0

            if abs(eq0) < 1.e-12:
                break

            cc0inc = -_fdiv(eq0, _fdiv(eq0 - eq0old, cc0inc))
            eq0old = eq0
            cc0 = cc0 + cc0inc

            ac = [col[:] for col in acsav]
            bc = [eq1 - rc4 * cc2 - cc0,
                  eq2 - 4 * rc3 * cc2,
                  eq3 - 12 * rc2 * cc2,
                  eq4 - 24 * rc1 * cc2,
                  eq5 - 24 * cc2]
            solve(bc, " 1")
            nsolve += 1
            cc1, cc3, cc4, cc5, cc6 = bc
            k += 1
        if k > KMX:
            _stop(warn, "tromar - stop: convergence error cc0 (norm)",
                  cc0, cc0inc, eq0, cc2, cc2inc, eq6, j, k)

        eq6 = cc2 * (2 * l + 5) + cc1 * cc1
        if abs(eq6) < 1.e-6:
            break

        cc2inc = -_fdiv(eq6, _fdiv(eq6 - eq6old, cc2inc))
        eq6old = eq6
        cc2 = cc2 + cc2inc
        j += 1
    if j > JMX:
        _stop(warn, "tromar - stop: convergence error cc2 (curvature)",
              cc0, cc0inc, eq0, cc2, cc2inc, eq6, j, k)

    # screened pseudopotential
    polynom1 = rn * (2 * cc1 + rs * (4 * cc2 + rs * (6 * cc3 + rs * (8 * cc4 + rs * (
        10 * cc5 + rs * 12 * cc6)))))
    polynom2 = 2 * cc1 + rs * (12 * cc2 + rs * (30 * cc3 + rs * (56 * cc4 + rs * (
        90 * cc5 + rs * 132 * cc6))))
    vps = np.array(v[:mmax], dtype=float)
    vps[:nrc] = ein + ll * polynom1 / rn + 0.5 * (polynom2 + polynom1 * polynom1)
    un = ups * _sexp(cc0)
    upn = polynom1 + ll / rn
    uppn = ((polynom2 - ll / (rn * rn)) + upn * upn) * un
    upn = upn * un
    # transform to logarithmic grid
    alrn = al * rn
    uppn = alrn * (al * upn + alrn * uppn)
    upn = alrn * upn
    u[:nrc] = un
    up[:nrc] = upn
    upp[:nrc] = uppn

    return TromarResult(nrc, u, up, upp, vps, (cc0, cc1, cc2, cc3, cc4, cc5, cc6),
                        (j, k), nsolve)


def _stop(warn: Callable[[str], None] | None, text: str, cc0: float, cc0inc: float,
          eq0: float, cc2: float, cc2inc: float, eq6: float, j: int, k: int) -> None:
    if warn:
        warn(list_record("& " + text))
        warn(list_record("&  cc0 cc0inc eq0      ", cc0, cc0inc, eq0))
        warn(list_record("&  cc2 cc2inc eq6      ", cc2, cc2inc, eq6))
        warn(list_record("&  iteration j k       ", j, k))
    raise FortranStop(text)
