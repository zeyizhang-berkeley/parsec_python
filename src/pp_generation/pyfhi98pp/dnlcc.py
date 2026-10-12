"""Model core density for the nonlinear core correction.

Port of ``dnlcc7LAPACK.f``: inside the matching radius ``rnlc`` the full
core density is replaced by the polynomial ``a0 + a3 x**3 + ... + a6 x**6``
with continuous value and first three derivatives at ``rnlc``; ``a0`` is
raised in steps of a factor 1.8 until the polynomial decays monotonically.

Behaviour of the Fortran that is kept on purpose:

* The start value of the search is ``a0 = 1.8*f0(r(inlc))`` evaluated with
  the polynomial coefficients ``a1 ... a6`` *before* they have been
  assigned.  In the reference executable these variables are stack slots
  that were never written and hold zero, so that the first value is
  ``1.8*dc(inlc)``; this is what the port does.
* ``1.8`` is a single precision constant.
* If the polynomial is not monotonic the next ``a0`` is ``1.8`` times its
  value at the first mesh point (counted from ``rnlc`` inwards) where it
  exceeds ``a0``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from . import libm
from .errors import FortranStop
from .fortran_io import fmt_e
from .fortran_list import list_record
from .lapack import dgesv
from .radial import _fdiv, fmom

__all__ = ["DnlccResult", "dnlcc7", "DNLCC_ITMAX"]

DNLCC_ITMAX: int = 20
_C18: float = float(np.float32(1.8))     # the literal 1.8 of the source


class DnlccResult(NamedTuple):
    """Output of ``dnlcc7``."""

    rnlc: float            # matching radius actually used, r(inlc)
    inlc: int              # its grid index (Fortran index)
    dc: np.ndarray         # model core density inside rnlc, full core outside
    dcp: np.ndarray        # dto. 1st derivative
    dcpp: np.ndarray       # dto. 2nd derivative
    coefficients: tuple[float, ...]   # a0 ... a6
    fort24: list[str]      # monitoring output (unit 24), one string per record
    it: int = 0            # number of passes of the a0 search (diagnostic)


def _f0(a: tuple[float, ...], x):
    a0, a1, a2, a3, a4, a5, a6 = a
    x2 = x * x
    x3 = x * x2
    return a0 + a1 * x + a2 * x * x + a3 * x * x * x + a4 * (x2 * x2) + a5 * (x2 * x3) + a6 * (x3 * x3)


def _f1(a: tuple[float, ...], x):
    _, a1, a2, a3, a4, a5, a6 = a
    x2 = x * x
    x3 = x * x2
    return a1 + 2 * a2 * x + 3 * a3 * x * x + 4 * a4 * x3 + 5 * a5 * (x2 * x2) + 6 * a6 * (x2 * x3)


def _f2(a: tuple[float, ...], x):
    _, _, a2, a3, a4, a5, a6 = a
    x2 = x * x
    x3 = x * x2
    return 2 * a2 + 6 * a3 * x + 12 * a4 * x2 + 20 * a5 * x3 + 30 * a6 * (x2 * x2)


@np.errstate(all="ignore")
def dnlcc7(mmax: int, rnlc: float, r: np.ndarray, dps: np.ndarray, dc: np.ndarray,
           dcp: np.ndarray, dcpp: np.ndarray,
           warn: Callable[[str], None] | None = None) -> DnlccResult:
    """Pseudocore density (``dnlcc7LAPACK.f``).

    ``rnlc`` matching radius, ``dps`` pseudo valence density (monitoring
    only), ``dc, dcp, dcpp`` full core density and its radial derivatives.
    The Fortran overwrites ``rnlc`` with the mesh point used; here it is
    returned in the result.
    """
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    dps = np.asarray(dps, dtype=float)
    dc = np.array(dc[:mmax], dtype=float)
    dcp = np.array(dcp[:mmax], dtype=float)
    dcpp = np.array(dcpp[:mmax], dtype=float)

    # pseudocore cutoff radius
    idx = np.nonzero(r <= rnlc)[0]
    if idx.size == 0:
        if warn:
            warn(list_record("& dnlcc7 --- stop: pseudocore radius not found"))
            warn(list_record("& dnlcc7 --- rnlc", float(rnlc)))
        raise FortranStop("dnlcc7 --- stop: pseudocore radius not found")
    inlc = int(idx[-1]) + 1
    k = inlc - 1
    if inlc < 2:
        # the Fortran would read dcpp(0) and r(0) out of bounds
        raise FortranStop("dnlcc7 - stop: pseudocore radius at the first point of the radial mesh")
    rnlc = float(r[k])
    # r(inlc+1), dcpp(inlc+1) for inlc = mmax: the arrays of the main program
    # hold zeros beyond mmax
    r_next = float(r[k + 1]) if k + 1 < mmax else 0.e0
    dcpp_next = float(dcpp[k + 1]) if k + 1 < mmax else 0.e0

    # pseudocore function coefficients
    # loop: want (i)  monotonously decaying pseudocore for gga
    #            (ii) small core charge
    # start dc(r=0)=a0 [violation of (i)] & gradually increase until (i) & (ii) are met
    dck, dcpk, dcppk = float(dc[k]), float(dcp[k]), float(dcpp[k])
    a = (dck, 0.e0, 0.e0, 0.e0, 0.e0, 0.e0, 0.e0)
    i = inlc                      # Fortran loop variable i
    r2 = rnlc * rnlc
    r3 = rnlc * r2
    it = 1
    while it <= DNLCC_ITMAX:
        a0 = _C18 * float(_f0(a, float(r[i - 1])))

        a1 = 0.e0
        a2 = 0.e0
        a3 = _fdiv(_fdiv(dcpp_next - float(dcpp[k - 1]), r_next - float(r[k - 1])), rnlc)
        mat = [
            [1.e0, 3.e0, 1.e0, 3.e0],
            [4 * rnlc, 6 * rnlc, rnlc, 4 * rnlc],
            [10 * r2, 10 * r2, r2, 5 * r2],
            [20 * r3, 15 * r3, r3, 6 * r3],
        ]
        b = [a3 / 6.e0,
             dcppk / (2 * rnlc) - a2 / rnlc,
             (dck - a0 - a2 * r2) / r3,
             dcpk / r2 - 2 * a2 / rnlc]
        _, info = dgesv(4, mat, b)
        if info != 0:
            text = "dnlcc7 - stop: bad input" if info > 0 else "dnlcc7 - stop: singular matrix"
            if warn:
                warn(list_record("& " + text))
            raise FortranStop(text)
        a = (a0, a1, a2, b[0], b[1], b[2], b[3])

        # do i=inlc,2,-1: if(f0(r(i)) .gt. a0) goto 45
        over = np.nonzero(_f0(a, r[1:inlc]) > a0)[0]
        i = int(over[-1]) + 2 if over.size else 1
        if i == 1:
            break
        it += 1
    if it > DNLCC_ITMAX:
        text = "dnlcc7 - stop: pseudocore convergence error"
        if warn:
            warn(list_record("& " + text, it))
        raise FortranStop(text)

    sc, scp, scpp = dck, dcpk, dcppk
    out = [
        list_record("Improved Polynomial Pseudocore -- dnlcc7"),
        list_record("Pseudocore matching at i", inlc, " r ", rnlc),
        list_record("Model parameters:"),
    ]
    for name, value in zip(("a0 ", "a1 ", "a2 ", "a3 ", "a4 ", "a5 ", "a6 "), a):
        out.append(list_record(name, value))

    # save core density beyond matching radius
    dd = dc[:inlc].copy()
    x = r[:inlc]
    dc[:inlc] = _f0(a, x)
    dcp[:inlc] = _f1(a, x)
    dcpp[:inlc] = _f2(a, x)

    # check matching of density and derivatives
    out.append(list_record("Checking values at pseudocore radius:"))
    out.append(list_record(sc, float(dc[k]), "density"))
    out.append(list_record(scp, float(dcp[k]), "1st derivative"))
    out.append(list_record(scpp, float(dcpp[k]), "2nd derivative"))

    # sum rule checks
    al = 0.1e0 * libm.slog(float(r[10]) / float(r[0]))
    t0 = fmom(0, mmax, al, 1.e0, r, dc, warn)
    out.append(list_record(t0, "total pseudo charge"))
    out.append(list_record(_fdiv(-fmom(1, mmax, al, 1.e0, r, dcp, warn), 3 * t0),
                           "1st derivative test, should be = 1"))
    out.append(list_record(_fdiv(fmom(2, mmax, al, 1.e0, r, dcpp, warn), 12 * t0),
                           "2nd derivative test, should be = 1 (test may be inaccurate)"))

    # dd(i) beyond inlc is not set in the Fortran (stack memory, zero in the
    # reference build); these "p" records are removed from the .dat file by psgen
    for j in range(min(inlc + 10, mmax)):
        ddj = float(dd[j]) if j < inlc else 0.e0
        dpsj = float(dps[j]) if j < len(dps) else 0.e0
        out.append("p" + "".join(" " + fmt_e(v, 12, 6)
                                 for v in (float(r[j]), dpsj, float(dc[j]), ddj)))
    return DnlccResult(rnlc, inlc, dc, dcp, dcpp, a, out, it)
