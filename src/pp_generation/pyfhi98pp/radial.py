"""Radial integrators and quadratures on the logarithmic mesh.

Ports of ``dftseq.f`` (radial Schroedinger / scalar-relativistic equation),
``atoaux.f`` (Adams formulas, inlined), ``vestat.f`` (Poisson equation),
``moment.f`` (Simpson quadratures), ``anderson.f`` (potential mixing) and
the array helpers of ``darraux.f``.

Conventions
-----------
* Arrays are float64, 0-based: Python ``x[i]`` is Fortran ``x(i+1)``.
* Integers that are mesh *indices* in the Fortran (``mch``, ``nin``,
  ``ninu`` ...) keep their Fortran (1-based) values: the mesh point is
  ``r[k-1]`` and the points ``1..k`` are ``x[:k]``.
* The floating-point statement order of the Fortran is preserved; sums are
  accumulated sequentially (never with ``sum``/``np.sum``, which round
  differently).
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np

from . import libm
from .constants import DFTSEQ_ITMAX, EPSAE, FSS_I
from .errors import FortranStop

__all__ = [
    "DftseqResult", "dftseq", "vestat", "fmom", "dmelm", "gltfmv",
    "anderson", "dspv", "dextv", "aeo", "aio", "aei", "aii", "seqsum",
]

# coefficient of the Adams formulas exactly as typed in atoaux.f (not 1/24)
C24 = 4.1666666666667e-2


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _fsqrt(x: float) -> float:
    """``sqrt`` with the IEEE result for negative arguments (NaN, no trap)."""
    return math.sqrt(x) if x >= 0.0 or x != x else float("nan")


def _fdiv(a: float, b: float) -> float:
    """Division with the IEEE results for a zero divisor (Inf/NaN, no trap)."""
    if b != 0.0:
        return a / b
    with np.errstate(all="ignore"):
        return float(np.float64(a) / np.float64(b))


def seqsum(values: np.ndarray, start: float = 0.0) -> float:
    """Left-to-right sum ``start + v0 + v1 + ...`` (Fortran do-loop order)."""
    if len(values) == 0:
        return float(start)
    acc = np.empty(len(values) + 1)
    acc[0] = start
    acc[1:] = values
    return float(np.cumsum(acc)[-1])


def _powi(x: np.ndarray, m: int) -> np.ndarray | float:
    """``x**m`` for a run-time integer ``m`` as libgcc ``__powidf2`` does."""
    n = abs(m)
    y: np.ndarray | float = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def dspv(il: int, ih: int, a: np.ndarray, b: np.ndarray) -> float:
    """``a DOT b`` over the Fortran index range ``il..ih`` (``darraux.f``)."""
    return seqsum(np.asarray(a[il - 1:ih], dtype=float) * np.asarray(b[il - 1:ih], dtype=float))


def _seqmin(start: float, values: np.ndarray) -> float:
    """Running minimum ``acc = min(acc, x)`` of a Fortran loop, with NaNs.

    The reference build evaluates ``min``/``dmin1`` with the instruction
    ``minsd acc, x``: ``acc = acc if acc < x else x``.  A NaN element
    therefore replaces the accumulator and is itself replaced by the next
    element, so that the result is the minimum of the elements after the
    last NaN (NaN if the last element is one).
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(start)
    nan = np.isnan(values)
    if nan.any():
        k = int(np.nonzero(nan)[0][-1])
        if k == values.size - 1:
            return float("nan")
        return float(values[k + 1:].min())
    m = float(values.min())
    return float(start) if start < m else m


def _seqmax(start: float, values: np.ndarray) -> float:
    """Running maximum, ``maxsd acc, x`` (see :func:`_seqmin`)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float(start)
    nan = np.isnan(values)
    if nan.any():
        k = int(np.nonzero(nan)[0][-1])
        if k == values.size - 1:
            return float("nan")
        return float(values[k + 1:].max())
    m = float(values.max())
    return float(start) if start > m else m


def dextv(kl: int, kh: int, b: np.ndarray) -> tuple[float, float]:
    """Minimum and maximum of ``b`` over ``kl..kh`` (``darraux.f``).

    As in the Fortran, a reversed range (``kl > kh``) is scanned with
    stride -1 from the lower index and therefore visits no element; the
    result is then ``(1e20, -1e20)``.  NaN elements are treated as the
    reference build does (see :func:`_seqmin`).
    """
    bl, bh = 1.0e20, -1.0e20
    if kl > kh:
        return bl, bh
    seg = np.asarray(b[kl - 1:kh], dtype=float)
    return _seqmin(bl, seg), _seqmax(bh, seg)


def aeo(y, j: int) -> float:
    """Adams extrapolation, outward (``atoaux.f``); ``j`` is 0-based."""
    return C24 * (55.0 * y[j] - 59.0 * y[j - 1] + 37.0 * y[j - 2] - 9.0 * y[j - 3])


def aio(y, j: int) -> float:
    """Adams interpolation, outward (``atoaux.f``); ``j`` is 0-based."""
    return C24 * (9.0 * y[j + 1] + 19.0 * y[j] - 5.0 * y[j - 1] + y[j - 2])


def aei(y, j: int) -> float:
    """Adams extrapolation, inward (``atoaux.f``); ``j`` is 0-based."""
    return -C24 * (55.0 * y[j] - 59.0 * y[j + 1] + 37.0 * y[j + 2] - 9.0 * y[j + 3])


def aii(y, j: int) -> float:
    """Adams interpolation, inward (``atoaux.f``); ``j`` is 0-based."""
    return -C24 * (9.0 * y[j - 1] + 19.0 * y[j] - 5.0 * y[j + 1] + y[j + 2])


# --------------------------------------------------------------------------
# dftseq
# --------------------------------------------------------------------------

class DftseqResult(NamedTuple):
    """Output arguments of ``dftseq``."""

    nin: int          # outermost grid point used (Fortran index / count)
    mch: int          # matching point: classical turning point (Fortran index)
    e: float          # eigenvalue
    u: np.ndarray     # wave function u(r) = r*R(r), length mmax
    up: np.ndarray    # du/dx on the logarithmic variable x = ln r
    upp: np.ndarray   # d2u/dx2
    nint: int = 0     # number of energy iterations (diagnostic, not in Fortran)
    de: float = 0.0   # last perturbation-theory energy shift (diagnostic)


def _outward(u: list, up: list, upp: list, a: list, b: list, ico: int) -> int:
    """Predictor-corrector outward integration, Fortran ``do 70 i=4,ico-1``.

    Fills the Python indices ``4..ico-1`` and returns the node count.
    """
    c = C24
    node = 0
    u0 = u[3]
    p0 = up[3]; p1 = up[2]; p2 = up[1]; p3 = up[0]
    q0 = upp[3]; q1 = upp[2]; q2 = upp[1]; q3 = upp[0]
    for i in range(4, ico):
        ai = a[i]
        bi = b[i]
        # predictor: u(i+1)=u(i)+aeo(up,i), up(i+1)=up(i)+aeo(upp,i)
        un = u0 + c * (55.0 * p0 - 59.0 * p1 + 37.0 * p2 - 9.0 * p3)
        pn = p0 + c * (55.0 * q0 - 59.0 * q1 + 37.0 * q2 - 9.0 * q3)
        q19 = 19.0 * q0
        q5 = 5.0 * q1
        p19 = 19.0 * p0
        p5 = 5.0 * p1
        # corrector, twice
        qn = ai * pn + bi * un
        pn = p0 + c * (9.0 * qn + q19 - q5 + q2)
        un = u0 + c * (9.0 * pn + p19 - p5 + p2)
        qn = ai * pn + bi * un
        pn = p0 + c * (9.0 * qn + q19 - q5 + q2)
        un = u0 + c * (9.0 * pn + p19 - p5 + p2)
        if un * u0 <= 0.0:
            node += 1
        u[i] = un
        up[i] = pn
        upp[i] = qn
        u0 = un
        p3 = p2; p2 = p1; p1 = p0; p0 = pn
        q3 = q2; q2 = q1; q1 = q0; q0 = qn
    return node


def _inward(u: list, up: list, upp: list, a: list, b: list, nin: int, mch: int) -> None:
    """Inward integration, Fortran ``do 130 i=nin,mch+1,-1``.

    Needs the start values at the Python indices ``nin-1..nin+2`` and fills
    ``nin-2`` down to ``mch-1``.
    """
    c = C24
    j = nin - 1
    u0 = u[j]
    p0 = up[j]; p1 = up[j + 1]; p2 = up[j + 2]; p3 = up[j + 3]
    q0 = upp[j]; q1 = upp[j + 1]; q2 = upp[j + 2]; q3 = upp[j + 3]
    for i in range(nin - 2, mch - 2, -1):
        ai = a[i]
        bi = b[i]
        # predictor: u(i-1)=u(i)+aei(up,i), up(i-1)=up(i)+aei(upp,i)
        un = u0 - c * (55.0 * p0 - 59.0 * p1 + 37.0 * p2 - 9.0 * p3)
        pn = p0 - c * (55.0 * q0 - 59.0 * q1 + 37.0 * q2 - 9.0 * q3)
        q19 = 19.0 * q0
        q5 = 5.0 * q1
        p19 = 19.0 * p0
        p5 = 5.0 * p1
        # corrector, twice: aii(y,i) = -c*(9 y(i-1) + 19 y(i) - 5 y(i+1) + y(i+2))
        qn = ai * pn + bi * un
        pn = p0 - c * (9.0 * qn + q19 - q5 + q2)
        un = u0 - c * (9.0 * pn + p19 - p5 + p2)
        qn = ai * pn + bi * un
        pn = p0 - c * (9.0 * qn + q19 - q5 + q2)
        un = u0 - c * (9.0 * pn + p19 - p5 + p2)
        u[i] = un
        up[i] = pn
        upp[i] = qn
        u0 = un
        p3 = p2; p2 = p1; p1 = p0; p0 = pn
        q3 = q2; q2 = q1; q1 = q0; q0 = qn


@np.errstate(all="ignore")
def dftseq(mode: int, z: float, mmax: int, r: np.ndarray, n: int, l: int,
           efm: float, v: np.ndarray, v1: np.ndarray, v2: np.ndarray,
           mch: int = 0, uld: float = 0.0, e: float = 0.0,
           warn: Callable[[str], None] | None = None, nin: int = 0,
           u: np.ndarray | None = None, up: np.ndarray | None = None,
           upp: np.ndarray | None = None) -> DftseqResult:
    """Integrate the radial equation on the logarithmic mesh (``dftseq.f``).

    Parameters follow the Fortran argument list
    ``dftseq(mode,z,mmax,r,n,l,efm,v,v1,v2,nin,mch,uld,e,u,up,upp)``:

    ``mode``
        1 full-potential bound state, scalar-relativistic; 2 pseudopotential
        bound state, non-relativistic; 3 full potential, fixed energy, find
        logarithmic derivative (scalar-relativistic); 4 pseudopotential, find
        the energy giving the logarithmic derivative ``uld`` at ``mch``;
        5 pseudopotential, fixed energy, wave function out to ``mch``;
        6 like 3 but non-relativistic; 7 like 4; 8 like 2 but returns
        ``e = 0`` instead of stopping if no bound state is found.
    ``v, v1, v2``
        potential ``v*Psi``, ``v1*grad(Psi)`` and ``1/v2*laplace(Psi)``.
        In the relativistic modes the Fortran differentiates ``v`` with a
        five-point stencil that reads ``v(mmax+1)`` and ``v(mmax+2)``; if
        ``v`` is longer than ``mmax`` these elements are used, otherwise
        they are taken as zero (which is what ``fhipp`` supplies).
    ``mch``
        input for modes 3-7 (matching point), output otherwise.
    ``uld``
        input for modes 4 and 7.
    ``e``
        start value of the eigenvalue (modes 1, 2, 4, 7, 8) or fixed energy.
    ``warn``
        receives the messages the Fortran writes to the error unit.
    ``nin, u, up, upp``
        previous contents of these in/out arguments of the Fortran routine
        (default: zero).  They only matter for mode 8 when no bound state is
        found: the routine then returns ``e = 0`` and leaves ``nin`` and the
        wave function arrays in whatever state the search reached, which
        includes parts of the previous contents (``psatom.f`` goes on to use
        them).  That state is reproduced exactly.

    Returns a :class:`DftseqResult`; ``u``, ``up``, ``upp`` are new arrays of
    length ``mmax`` that vanish beyond ``nin`` (except for the ``e = 0``
    return of mode 8 just described).  The Fortran ``stop``
    conditions raise :class:`FortranStop`.  Invalid operations follow IEEE
    arithmetic as in the Fortran (no exceptions): if the inward integration
    has to start inside the classically allowed region (``nin`` clipped to
    ``mmax-4``), ``xkap`` is NaN and the routine returns the current ``e``
    with NaN wave functions, exactly like the original.
    """
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    v_full = np.asarray(v, dtype=float)
    v = np.ascontiguousarray(v_full[:mmax])
    amesh = float(r[1]) / float(r[0])
    al = libm.slog(amesh)
    sls = float(l * (l + 1))
    z = float(z)
    e = float(e)

    if mode == 6 or mode == 7:
        ics = max(1, mch)
        ico = max(1, mch)
    else:
        ics = mmax
        ico = mch

    if mode == 1 or mode == 3:
        trel = True
        fss = (1.0 / FSS_I) * (1.0 / FSS_I)
        if l == 0:
            gamma = _fsqrt(1.0 - fss * (z * z))
        else:
            gamma = (l * _fsqrt(float(l * l) - fss * (z * z))
                     + (l + 1) * _fsqrt(float((l + 1) * (l + 1)) - fss * (z * z))) / float(2 * l + 1)
    else:
        trel = False
        fss = 1.0e-20
        gamma = float(l + 1)
    imod = mode
    if imod == 6:
        imod = 3
    if imod == 7:
        imod = 4
    if imod == 8:
        imod = 2

    emin = emax = 0.0
    if imod == 1 or imod == 2:
        rm = float(r[mmax - 1])
        emax = float(v[mmax - 1]) + 0.5 * sls / (rm * rm)
        emin = _seqmin(0.0, v + 0.5 * sls / (r * r))    # dmin1 loop; NaN as in the Fortran
        if e > emax:
            e = 1.25 * emax
        if e < emin:
            e = 0.75 * emin
        if e > emax:
            e = 0.5 * (emax + emin)
    elif imod == 4:
        emax = e + 10.0
        emin = e - 10.0

    x1 = al * r
    x2 = x1 * x1
    rs = r[:ics]
    vs = v[:ics]
    x1s = x1[:ics]
    x2s = x1s * x1s
    v2s = np.asarray(v2, dtype=float)[:ics]

    if trel:
        # dv/dr for the Darwin correction (reads two points beyond mmax)
        ve = np.zeros(mmax + 2)
        take = min(len(v_full), mmax + 2)
        ve[:take] = v_full[:take]
        dv = np.empty(ics)
        dv[0] = (-50.0 * ve[0] + 96.0 * ve[1] - 72.0 * ve[2] + 32.0 * ve[3]
                 - 6.0 * ve[4]) / (24.0 * x1[0])
        dv[1] = (-6.0 * ve[0] - 20.0 * ve[1] + 36.0 * ve[2] - 12.0 * ve[3]
                 + 2.0 * ve[4]) / (24.0 * x1[1])
        dv[2:] = (2.0 * ve[0:ics - 2] - 16.0 * ve[1:ics - 1] + 16.0 * ve[3:ics + 1]
                  - 2.0 * ve[4:ics + 2]) / (24.0 * x1s[2:])
        fr = frp = None
        hfss = 0.5 * fss
        x2fss = x2s * fss
        x1hfss = x1s * 0.5 * fss
        hdv = 0.5 * dv
    else:
        # non-relativistic coefficient arrays for u (fr) and up (frp)
        v1s = np.asarray(v1, dtype=float)[:ics]
        fr = -(2.0 * x2s * v1s * v2s)
        frp = 2.0 * x1s * v1s * v2s

    aux = al * al * sls
    npad = max(mmax, ics) + 8
    rl = r.tolist()
    x1l = x1.tolist()

    # start values of the series, r(i)**gamma for the first four points
    rg = libm.spow_series(rl[:4], gamma)
    hydrogen_like = (l == 0 and imod == 2) or imod == 1

    # mode 8 can return without a solution: keep the arrays exactly as the
    # Fortran does (scale in place after every pass); in the other modes
    # every return rewrites all points and the scaling is done once at exit
    eager = mode == 8
    u = _work_list(u, mmax, npad)
    up = _work_list(up, mmax, npad)
    upp = _work_list(upp, mmax, npad)
    # initialize arrays and remove leftover garbage
    for i in range(4):
        u[i] = up[i] = upp[i] = 0.0
    nint = 0
    previous = None        # (e, emin, emax) at the start of the previous pass
    while True:
        nint += 1
        if mode == 8 and nint <= DFTSEQ_ITMAX:
            # A pass depends on (e, emin, emax) only.  Once the bisection
            # hands back the triple it was given, every further pass up to
            # the iteration limit repeats this one and leaves the arrays as
            # they are: go to the limit directly.  Same result as the
            # Fortran (e = 0), which spends all 200 passes on a state that
            # is not bound.
            if (e, emin, emax) == previous:
                nint = DFTSEQ_ITMAX + 1
            previous = (e, emin, emax)
        if nint > DFTSEQ_ITMAX:
            if mode == 8:
                if warn:
                    warn(f"& dftseq - no bound state found (iter): e=>0 n l {n} {l}")
                return _unfinished(mmax, nin, mch, u, up, upp, nint)
            if warn:
                warn("& dftseq - stop: convergence error")
                warn(f"&   nint n l                 {nint} {n} {l}")
            raise FortranStop("dftseq - stop: convergence error")

        # coefficient array for u in the differential equation
        vme = vs - e
        cf = aux + 2.0 * vme * v2s * x2s
        if trel:
            den = 1.0 + hfss * (e - vs)
            fr = x2fss * (-(vme * vme) + hdv / (rs * den))
            frp = -(x1hfss * dv / den)
        a = (al + frp).tolist()
        b = (cf + fr).tolist()
        if ics < npad:
            a.extend([0.0] * (npad - ics))
            b.extend([0.0] * (npad - ics))

        # classical turning point for matching
        if imod == 1 or imod == 2:
            idx = np.nonzero((cf[:-1] <= 0.0) & (cf[1:] > 0.0))[0]
            if idx.size == 0:
                if mode == 8:
                    if warn:
                        warn(f"& dftseq - no bound state found (ctp): e=>0 n l {n} {l}")
                    return _unfinished(mmax, nin, mch, u, up, upp, nint)
                if warn:
                    warn("& dftseq - stop: no classical turning point")
                    warn(f"&    nint n l e                       {nint} {n} {l} {e}")
                raise FortranStop("dftseq - stop: no classical turning point")
            mch = int(idx[-1]) + 2
            ico = mch
        else:
            nin = mch

        # start wavefunction with series
        for i in range(4):
            if hydrogen_like:
                zr = z * rl[i]
                poly = 1.0 - zr + 0.5 * (zr * zr)
                u[i] = rg[i] * poly
                up[i] = al * gamma * rg[i] * poly + al * rg[i] * (-zr + zr * zr)
            else:
                u[i] = rg[i]
                up[i] = al * gamma * rg[i]
            upp[i] = a[i] * up[i] + b[i] * u[i]

        # outward integration using predictor once, corrector twice
        node = _outward(u, up, upp, a, b, ico)
        uout = u[mch - 1]
        upout = up[mch - 1]

        nodes = node - n + l + 1
        if nodes == 0 or imod == 3 or imod == 5:
            sc = 1.0
            if imod == 1 or imod == 2:
                # start inward integration at 15*classical turning point
                # with simple exponential
                nin = int(mch + 2.7 / al)
                if nin > mmax - 4:
                    nin = mmax - 4
                rn = rl[nin - 1]
                xkap = _fsqrt(sls / (rn * rn) + 2.0 * (float(v[nin - 1]) - e) * efm)
                for i in range(nin - 1, nin + 4):
                    ui = libm.sexp(-xkap * (rl[i] - rn))
                    u[i] = ui
                    up[i] = -rl[i] * al * xkap * ui
                    upp[i] = a[i] * up[i] + b[i] * ui
                _inward(u, up, upp, a, b, nin, mch)
                # scale outside wf for continuity
                sc = _fdiv(uout, u[mch - 1])
                u[mch - 1:nin] = [sc * x for x in u[mch - 1:nin]]
                if eager:
                    up[mch - 1:nin] = [sc * x for x in up[mch - 1:nin]]
                    upp[mch - 1:nin] = [sc * x for x in upp[mch - 1:nin]]
                    upin = up[mch - 1]
                    sc = 1.0
                else:
                    upin = sc * up[mch - 1]
            else:
                upin = uld * uout

            # perform normalization sum
            ro = rl[0] / _fsqrt(amesh)
            sn = libm.spow(ro, 2.0 * gamma + 1.0) / (2.0 * gamma + 1.0)
            for xi, ui in zip(x1l[:nin - 3], u[:nin - 3]):
                sn += xi * (ui * ui)
            ua, ub, uc = u[nin - 3], u[nin - 2], u[nin - 1]
            sn = sn + al * (23.0 * rl[nin - 3] * (ua * ua)
                            + 28.0 * rl[nin - 2] * (ub * ub)
                            + 9.0 * rl[nin - 1] * (uc * uc)) / 24.0
            cn = _fdiv(1.0, _fsqrt(sn))
            uout = cn * uout
            upout = cn * upout
            upin = cn * upin
            if eager:
                # labels 180, 190: normalize in place, zero the tail
                u[:nin] = [cn * x for x in u[:nin]]
                up[:nin] = [cn * x for x in up[:nin]]
                upp[:nin] = [cn * x for x in upp[:nin]]
                if nin < mmax:
                    u[nin:mmax] = [0.0] * (mmax - nin)
                    up[nin:mmax] = [0.0] * (mmax - nin)
                    upp[nin:mmax] = [0.0] * (mmax - nin)
                cn = 1.0

            # exit for fixed-energy calculation
            if imod == 3 or imod == 5:
                return _finish(mmax, nin, mch, e, u, up, upp, sc, cn, mch, nint)

            # perturbation theory for energy shift
            de = 0.5 * uout * (upout - upin) / (al * rl[mch - 1])

            # convergence test and possible exit
            # max(abs(e),0.2d0) is "maxsd |e|, 0.2": 0.2 if e is NaN
            if abs(de) >= (abs(e) if abs(e) > 0.2 else 0.2) * EPSAE:
                if de > 0.0:
                    emin = e
                else:
                    emax = e
                e = e + de
                if e > emax or e < emin:
                    e = 0.5 * (emax + emin)
                continue
            return _finish(mmax, nin, mch, e, u, up, upp, sc, cn, mch, nint, de)
        elif nodes < 0:
            # too few nodes
            emin = e
            e = 0.5 * (emin + emax)
        else:
            # too many nodes
            emax = e
            e = 0.5 * (emin + emax)


def _work_list(values: np.ndarray | None, mmax: int, npad: int) -> list:
    """Working copy of an in/out array as a list of ``npad`` floats."""
    out = [0.0] * npad
    if values is not None:
        k = min(len(values), mmax)
        out[:k] = np.asarray(values[:k], dtype=float).tolist()
    return out


def _unfinished(mmax: int, nin: int, mch: int, u: list, up: list, upp: list,
                nint: int) -> DftseqResult:
    """Mode 8 without a bound state: ``e = 0``, arrays as they stand."""
    return DftseqResult(nin, mch, 0.0, np.array(u[:mmax]), np.array(up[:mmax]),
                        np.array(upp[:mmax]), nint, 0.0)


def _finish(mmax: int, nin: int, mch: int, e: float, u: list, up: list,
            upp: list, sc: float, cn: float, msc: int, nint: int,
            de: float = 0.0) -> DftseqResult:
    """Apply the pending scale factors and zero the tails (labels 150-190).

    ``u[msc-1:nin]`` already carries the continuity factor ``sc``; ``up`` and
    ``upp`` receive it here, then everything is multiplied by ``cn``.
    """
    ua = np.zeros(mmax)
    upa = np.zeros(mmax)
    uppa = np.zeros(mmax)
    k = max(0, min(nin, mmax))
    if k:
        pa = np.array(up[:k])
        qa = np.array(upp[:k])
        if sc != 1.0:
            pa[msc - 1:k] = sc * pa[msc - 1:k]
            qa[msc - 1:k] = sc * qa[msc - 1:k]
        ua[:k] = cn * np.array(u[:k])
        upa[:k] = cn * pa
        uppa[:k] = cn * qa
    return DftseqResult(nin, mch, e, ua, upa, uppa, nint, de)


# --------------------------------------------------------------------------
# vestat
# --------------------------------------------------------------------------

def vestat(mmax: int, zion: float, r: np.ndarray, rho: np.ndarray,
           flag: bool = False) -> tuple[np.ndarray, float | None]:
    """Electrostatic potential of a spherical density (``vestat.f``).

    ``zion`` is the control parameter of the Fortran: total charge for an
    all-electron atom (``sf-z``), 0 for a neutral atom (Coulomb + Hartree),
    number of electrons (Hartree potential).  Returns ``(vo, eeel)`` where
    ``eeel = dmelm(vo, rho)`` if ``flag`` else ``None``.
    """
    r = np.asarray(r[:mmax], dtype=float)
    rho = np.asarray(rho[:mmax], dtype=float)
    al = libm.slog(float(r[1]) / float(r[0]))
    rhoal = rho * al
    r2 = r * r

    def aii_all(y: np.ndarray) -> np.ndarray:
        # aii(y,i) for the Fortran indices i = 2 .. mmax-2
        return -(C24 * (9.0 * y[0:mmax - 3] + 19.0 * y[1:mmax - 2]
                        - 5.0 * y[2:mmax - 1] + y[3:mmax]))

    rv = np.empty(mmax)
    rv[mmax - 1] = zion
    rv[mmax - 2] = zion
    # rv(i-1) = rv(i) + aii(rvp,i), i = mmax-2 .. 2
    acc = np.empty(mmax - 2)
    acc[0] = zion
    acc[1:] = aii_all(rhoal * (r * r2))[::-1]
    rv[:mmax - 2] = np.cumsum(acc)[::-1]
    # tv = tv + aii(rvp,i); rv(i-1) = rv(i-1) - r(i-1)*tv
    acc[0] = 0.0
    acc[1:] = aii_all(rhoal * r2)[::-1]
    tv = np.cumsum(acc)[:0:-1]
    rv[:mmax - 3] = rv[:mmax - 3] - r[:mmax - 3] * tv
    vo = rv / r
    eeel = dmelm(mmax, al, r, vo, rho) if flag else None
    return vo, eeel


# --------------------------------------------------------------------------
# moment.f
# --------------------------------------------------------------------------

def _odd(mmx: int, name: str, warn: Callable[[str], None] | None) -> int:
    if mmx % 2 == 0:
        if warn:
            warn(f"& {name} - got even no. of mesh points max, assume mmx-1")
        return mmx - 1
    return mmx


def fmom(m: int, mmx: int, al: float, fnorm: float, r: np.ndarray, fr: np.ndarray,
         warn: Callable[[str], None] | None = None) -> float:
    """``m``-th radial moment of ``fr`` by Simpson's rule (``moment.f``).

    ``int fr(r) r**(2+m) dr / fnorm`` on the logarithmic mesh.
    """
    mx_ = _odd(mmx, "fmom", warn)
    r = np.asarray(r[:mx_], dtype=float)
    fr = np.asarray(fr[:mx_], dtype=float)
    fm = fr * r * r * r * _powi(r, m)
    esum = seqsum(fm[1:mx_ - 1:2])
    osum = seqsum(fm[2:mx_ - 2:2])
    val = al * (4.0 * esum + 2.0 * osum + float(fm[0]) + float(fm[mx_ - 1])) / 3.0
    with np.errstate(all="ignore"):
        return float(np.float64(val + 0.5 * float(fm[0])) / np.float64(fnorm))


def dmelm(mmx: int, al: float, r: np.ndarray, v: np.ndarray, usq: np.ndarray,
          warn: Callable[[str], None] | None = None) -> float:
    """Matrix element ``int v(r) usq(r) r**2 dr`` (``moment.f``)."""
    mx_ = _odd(mmx, "dmelm", warn)
    r = np.asarray(r[:mx_], dtype=float)
    fm = np.asarray(v[:mx_], dtype=float) * np.asarray(usq[:mx_], dtype=float) * r * r
    t = r * fm
    esum = seqsum(t[1:mx_ - 1:2])
    osum = seqsum(t[2:mx_ - 2:2])
    val = (al * (4.0 * esum + 2.0 * osum + float(t[mx_ - 1]) + float(t[0]))) / 3.0
    return val + 0.5 * float(fm[0]) * float(r[0])


def gltfmv(mmx: int, al: float, r: np.ndarray, ga: np.ndarray, g: np.ndarray,
           gb: np.ndarray, warn: Callable[[str], None] | None = None) -> float:
    """Matrix element ``int ga(r) g(r) gb(r) dr`` (``moment.f``)."""
    mx_ = _odd(mmx, "gltfmv", warn)
    r = np.asarray(r[:mx_], dtype=float)
    ga = np.asarray(ga[:mx_], dtype=float)
    g = np.asarray(g[:mx_], dtype=float)
    gb = np.asarray(gb[:mx_], dtype=float)
    t = r * ga * g * gb
    esum = seqsum(t[1:mx_ - 1:2])
    osum = seqsum(t[2:mx_ - 2:2])
    g1 = float(ga[0]); h1 = float(g[0]); b1 = float(gb[0]); r1 = float(r[0])
    gm = float(ga[mx_ - 1]); hm = float(g[mx_ - 1]); bm = float(gb[mx_ - 1]); rm = float(r[mx_ - 1])
    return ((al * (4.0 * esum + 2.0 * osum + g1 * h1 * b1 * r1 + gm * hm * bm * rm)) / 3.0
            + 0.5 * g1 * h1 * b1 * r1)


# --------------------------------------------------------------------------
# anderson.f
# --------------------------------------------------------------------------

def anderson(bl: float, it: int, mmax: int, r: np.ndarray, vo1: np.ndarray,
             vo: np.ndarray, vi1: np.ndarray, vi: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anderson mixing of the potential (``anderson.f``).

    Returns the updated ``(vi, vi1, vo1)``: the next input potential and the
    saved input/output potentials of this iteration.  ``vo1`` and ``vi1`` are
    not referenced for ``it == 1`` (the Fortran multiplies its uninitialized
    arrays by ``thl = 0`` there).
    """
    r = np.asarray(r[:mmax], dtype=float)
    vo = np.asarray(vo[:mmax], dtype=float)
    vi = np.asarray(vi[:mmax], dtype=float)
    thl = 0.0
    if it > 1:
        vo1 = np.asarray(vo1[:mmax], dtype=float)
        vi1 = np.asarray(vi1[:mmax], dtype=float)
        rl = vo - vi
        rl1 = vo1 - vi1
        dr = rl - rl1
        r2 = r * r
        sn = seqsum(rl * dr * r2)
        sd = seqsum(dr * dr * r2)
        thl = _fdiv(sn, sd)        # 0/0 for an empty atom: NaN as in the Fortran
        vn = ((1.0 - bl) * ((1.0 - thl) * vi + thl * vi1)
              + bl * ((1.0 - thl) * vo + thl * vo1))
    else:
        vn = ((1.0 - bl) * ((1.0 - thl) * vi + thl * 0.0)
              + bl * ((1.0 - thl) * vo + thl * 0.0))
    return vn, vi.copy(), vo.copy()
