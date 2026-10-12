"""Exchange-correlation potentials and energy densities.

Ports of ``vexcor.f`` and of every routine it reaches: ``ldaxc.f``
(``xlda``, ``wigner``, ``cepal``, ``wigscaled``), ``relxc.f``, ``ggarad.f``
(``ggaxrad``, ``ggacrad``), ``gga91_sr.f`` (``exch``, ``corlsd``, ``gcor``,
``corgga``), ``bpgcxc.f`` (``xbecke``, ``corga86``), ``corlyp.f`` and
``pbe.f`` (``exchpbe``, ``corpbe``, ``gcor2``).

All routines are vectorized over the radial mesh: they take float64 arrays
(or scalars) and evaluate exactly the Fortran expressions, operation by
operation and in the same order, so that the only possible deviations from
the Fortran come from ``pow``/``exp``/``log`` of the math library.  Hartree
atomic units.  Functions keep the lowercase Fortran names.

Exchange-correlation types ``iexc`` (header of ``fhipp.f``)::

     1  LDA   Wigner
     2  LDA   Hedin/Lundqvist
     3  LDA   Ceperley/Alder Perdew/Zunger (1980)
     4  GGA   Perdew/Wang (1991)
     5  GGA   Becke (1988) X, Perdew (1986) C
     6  GGA   Perdew/Burke/Ernzerhof (1996)
     7  LDA   like 8 + relativistic correction
     8  LDA   Ceperley/Alder Perdew/Wang (1991)
     9  GGA   Becke (1988) X, Lee/Yang/Parr (1988) C
    10  GGA   Perdew/Wang (1991) X, Lee/Yang/Parr (1988) C
    11  LDA   exchange only
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from . import libm
from .errors import FortranStop
from .radial import dmelm

__all__ = [
    "XcResult", "vexcor", "xlda", "wigner", "cepal", "wigscaled", "relxc",
    "ggaxrad", "ggacrad", "exch", "corlsd", "gcor", "corgga", "xbecke",
    "corga86", "corlyp", "exchpbe", "corpbe", "gcor2",
]

_sqrt = np.sqrt
_abs = np.abs


def _log(x):
    return libm.log(x)


def _exp(x):
    return libm.exp(x)


def _pow(x, y):
    return libm.pow(x, y)


# --------------------------------------------------------------------------
# ldaxc.f
# --------------------------------------------------------------------------

def xlda(d):
    """LDA exchange: returns ``(vx, ex)`` (``ldaxc.f``); zero for ``d <= 1e-100``."""
    ax = -.738558766382022406e0
    thd = .333333333333333333e0
    thd4 = .133333333333333333e1
    eps = 1.e-100
    d = np.asarray(d, dtype=float)
    ok = d > eps
    ex = np.zeros(d.shape)
    ex[ok] = ax * _pow(d[ok], thd)
    vx = thd4 * ex
    return vx, ex


def wigner(rh):
    """Wigner interpolation: returns ``(ex, fx, exc, fxc)`` (``ldaxc.f``)."""
    thrd = .333333333333333333e0
    w1 = -.44e0
    w2 = .779e1
    ax = -.738558766382022447e0
    pif = .620350490899400087e0
    eps = 1.e-100
    rh = np.asarray(rh, dtype=float)
    ex = np.zeros(rh.shape)
    fx = np.zeros(rh.shape)
    exc = np.zeros(rh.shape)
    fxc = np.zeros(rh.shape)
    ok = ~(rh < eps)
    x = _pow(rh[ok], thrd)
    ex_ = ax * x
    fx_ = 4.e0 * thrd * ex_
    rs = pif / x
    y = 1.e0 / (rs + w2)
    ex[ok] = ex_
    fx[ok] = fx_
    exc[ok] = ex_ + w1 * y
    fxc[ok] = fx_ + w1 * (4.e0 * thrd * rs + w2) * y * y
    return ex, fx, exc, fxc


def cepal(rh):
    """Ceperley-Alder (Perdew-Zunger): returns ``(ex, fx, exc, fxc)`` (``ldaxc.f``)."""
    eps = 1.e-100
    rh = np.asarray(rh, dtype=float)
    ex = np.zeros(rh.shape)
    fx = np.zeros(rh.shape)
    exc = np.zeros(rh.shape)
    fxc = np.zeros(rh.shape)
    low = (rh < 0.23873241e0) & ~(rh <= eps)
    high = ~(rh < 0.23873241e0)
    if low.any():
        rs = 0.62035049e0 * _pow(rh[low], -0.3333333333333333e0)
        sqrs = _sqrt(rs)
        den = 1.0e0 + 1.0529e0 * sqrs + 0.3334e0 * rs
        exc_ = -0.4582e0 / rs - 0.1423e0 / den
        exc[low] = exc_
        fxc[low] = exc_ - rs * (0.15273333e0 / (rs * rs)
                                + (0.02497128e0 / sqrs + 0.01581427e0) / (den * den))
    if high.any():
        rs = 0.62035049e0 * _pow(rh[high], -0.3333333333333333e0)
        rsl = _log(rs)
        exc_ = (-0.4582e0 / rs - 0.0480e0 + 0.0311e0 * rsl - 0.0116e0 * rs
                + 0.002e0 * rs * rsl)
        exc[high] = exc_
        fxc[high] = exc_ - rs * (0.15273333e0 / (rs * rs)
                                 + 0.01036667e0 / rs - 0.003866667e0
                                 + 0.00066667e0 * (1.0e0 + rsl))
    done = low | high
    # exchange-only energy and potential
    ex_ = -0.7385587664e0 * _pow(rh[done], .3333333333333333e0)
    ex[done] = ex_
    fx[done] = 4.e0 / 3.e0 * ex_
    return ex, fx, exc, fxc


def wigscaled(rh):
    """Scaled Wigner xc (Zhao/Parr): returns ``(ex, fx, exc, fxc)`` (``ldaxc.f``).

    Not reachable from ``vexcor``; ported for completeness.
    """
    thrd = .333333333333333333e0
    aa = .93222e0
    bb = .947362e-2
    ax = .738558766382022406e0
    rh = np.asarray(rh, dtype=float)
    ex = np.zeros(rh.shape)
    fx = np.zeros(rh.shape)
    exc = np.zeros(rh.shape)
    fxc = np.zeros(rh.shape)
    ok = ~(rh <= 0.e0)
    x = bb * _pow(rh[ok], thrd)
    y = x / (x + 1.e0)
    z = -aa * x / bb
    xyln = x * _log(y)
    exc[ok] = z * (1.e0 + xyln)
    fxc[ok] = thrd * z * (4.e0 + 5.e0 * xyln + y)
    ex_ = ax * z / aa
    ex[ok] = ex_
    fx[ok] = 4.e0 * thrd * ex_
    return ex, fx, exc, fxc


# --------------------------------------------------------------------------
# relxc.f
# --------------------------------------------------------------------------

def relxc(mode: int, d):
    """MacDonald-Vosko relativistic correction factors ``(dex, dvx)`` (``relxc.f``)."""
    a = 1.e0 / 44.3e0
    thrd = 0.333333333333333e0
    d = np.asarray(d, dtype=float)
    dvx = np.ones(d.shape)
    dex = np.ones(d.shape)
    if mode != 0:
        ok = d > 1e-15
        beta = a * _pow(d[ok], thrd)
        gamma = _sqrt(1.e0 + beta * beta)
        f1 = _log(beta + gamma)
        bg = beta * gamma
        b2 = beta * beta
        dvx[ok] = 0.5e0 * (3.e0 * f1 / bg - 1.e0)
        dex[ok] = 1.e0 - 1.5e0 * ((bg - f1) * (bg - f1)) / (b2 * b2)
    return dex, dvx


# --------------------------------------------------------------------------
# gga91_sr.f
# --------------------------------------------------------------------------

def exch(d, s, u, v):
    """PW91 exchange for a spin-unpolarized system: ``(ex, vx)`` (``gga91_sr.f``)."""
    a1, a2, a3, a4 = 0.19645e0, 0.27430e0, 0.15084e0, 100.e0
    ax, a, b1 = -0.7385588e0, 7.7956e0, 0.004e0
    thrd, thrd4 = 0.333333333333e0, 1.33333333333e0
    fac = ax * _pow(d, thrd)
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    p0 = 1.e0 / _sqrt(1.e0 + a * a * s2)
    p1 = _log(a * s + 1.e0 / p0)
    p2 = _exp(-a4 * s2)
    p3 = 1.e0 / (1.e0 + a1 * s * p1 + b1 * s4)
    p4 = 1.e0 + a1 * s * p1 + (a2 - a3 * p2) * s2
    f = p3 * p4
    ex = fac * f
    # energy done. now the potential:
    p5 = b1 * s2 - (a2 - a3 * p2)
    p6 = a1 * s * (p1 + a * s * p0)
    p7 = 2.e0 * (a2 - a3 * p2) + 2.e0 * a3 * a4 * s2 * p2 - 4.e0 * b1 * s2 * f
    fs = p3 * (p3 * p5 * p6 + p7)
    p8 = 2.e0 * s * (b1 - a3 * a4 * p2)
    p9 = a1 * p1 + a * a1 * s * p0 * (3.e0 - a * a * s2 * p0 * p0)
    p10 = (4.e0 * a3 * a4 * s * p2 * (2.e0 - a4 * s2) - 8.e0 * b1 * s * f
           - 4.e0 * b1 * s3 * fs)
    p11 = -p3 * p3 * (a1 * p1 + a * a1 * s * p0 + 4.e0 * b1 * s3)
    fss = p3 * p3 * (p5 * p9 + p6 * p8) + 2.e0 * p3 * p5 * p6 * p11 + p3 * p10 + p7 * p11
    vx = fac * (thrd4 * f - (u - thrd4 * s3) * fss - v * fs)
    return ex, vx


def gcor(a, a1, b1, b2, b3, b4, p, rs):
    """Called by ``corlsd``: returns ``(gg, ggrs)`` (``gga91_sr.f``)."""
    p1 = p + 1.e0
    q0 = -2.e0 * a * (1.e0 + a1 * rs)
    rs12 = _sqrt(rs)
    rs32 = rs12 * (rs12 * rs12)
    rsp = rs if p == 1.0 else _pow(rs, p)
    q1 = 2.e0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs * rsp)
    q2 = _log(1.e0 + 1.e0 / q1)
    gg = q0 * q2
    q3 = a * (b1 / rs12 + 2.e0 * b2 + 3.e0 * b3 * rs12 + 2.e0 * b4 * p1 * rsp)
    ggrs = -2.e0 * a * a1 * q2 - q0 * q3 / (q1 * q1 + q1)
    return gg, ggrs


def _spin_scalar(zet, x):
    """``(1+zet)**x`` and ``(1-zet)**x`` for scalar or array ``zet``."""
    if isinstance(zet, float):
        return libm.spow(1.e0 + zet, x), libm.spow(1.e0 - zet, x)
    return _pow(1.e0 + zet, x), _pow(1.e0 - zet, x)


def _zpow3(zet):
    return zet * (zet * zet)


def _zpow4(zet):
    return (zet * zet) * (zet * zet)


def corlsd(rs, zet):
    """Uniform-gas correlation of Perdew and Wang 1991 (``gga91_sr.f``).

    Returns ``(ec, vcup, vcdn, ecrs, eczet, alfc)``.
    """
    gam, fzz = 0.5198421e0, 1.709921e0
    thrd, thrd4 = 0.333333333333e0, 1.333333333333e0
    zp, zm = _spin_scalar(zet, thrd4)
    f = (zp + zm - 2.e0) / gam
    eu, eurs = gcor(0.0310907e0, 0.21370e0, 7.5957e0, 3.5876e0, 1.6382e0,
                    0.49294e0, 1.00e0, rs)
    ep, eprs = gcor(0.01554535e0, 0.20548e0, 14.1189e0, 6.1977e0, 3.3662e0,
                    0.62517e0, 1.00e0, rs)
    alfm, alfrsm = gcor(0.0168869e0, 0.11125e0, 10.357e0, 3.6231e0, 0.88026e0,
                        0.49671e0, 1.00e0, rs)
    # alfm is minus the spin stiffness alfc
    alfc = -alfm
    z4 = _zpow4(zet)
    ec = eu * (1.e0 - f * z4) + ep * f * z4 - alfm * f * (1.e0 - z4) / fzz
    # energy done. now the potential:
    ecrs = eurs * (1.e0 - f * z4) + eprs * f * z4 - alfrsm * f * (1.e0 - z4) / fzz
    zp, zm = _spin_scalar(zet, thrd)
    fz = thrd4 * (zp - zm) / gam
    eczet = (4.e0 * _zpow3(zet) * f * (ep - eu + alfm / fzz)
             + fz * (z4 * ep - z4 * eu - (1.e0 - z4) * alfm / fzz))
    comm = ec - rs * ecrs / 3.e0 - zet * eczet
    vcup = comm + eczet
    vcdn = comm - eczet
    return ec, vcup, vcdn, ecrs, eczet, alfc


def corgga(rs, zet, t, uu, vv, ww, fk, sk, g, ec, ecrs, eczet):
    """PW91 gradient correction to correlation (``gga91_sr.f``).

    The members ``fk, sk, g, ec, ecrs, eczet`` of ``common /gas/`` are
    explicit arguments.  Returns ``(h, dvcup, dvcdn)``.
    """
    xnu, cc0, cx, alf = 15.75592e0, 0.004235e0, -0.001667212e0, 0.09e0
    c1, c2, c3, c4 = 0.002568e0, 0.023266e0, 7.389e-6, 8.723e0
    c5, c6, a4 = 0.472e0, 7.389e-2, 100.e0
    thrdm, thrd2 = -0.333333333333e0, 0.666666666667e0
    bet = xnu * cc0
    delt = 2.e0 * alf / bet
    g3 = g * (g * g)
    g4 = g3 * g
    pon = -delt * ec / (g3 * bet)
    b = delt / (_exp(pon) - 1.e0)
    b2 = b * b
    t2 = t * t
    t4 = t2 * t2
    t6 = t4 * t2
    rs2 = rs * rs
    rs3 = rs2 * rs
    q4 = 1.e0 + b * t2
    q5 = 1.e0 + b * t2 + b2 * t4
    q6 = c1 + c2 * rs + c3 * rs2
    q7 = 1.e0 + c4 * rs + c5 * rs2 + c6 * rs3
    cc = -cx + q6 / q7
    skfk = sk / fk
    r0 = skfk * skfk
    r1 = a4 * r0 * g4
    coeff = cc - cc0 - 3.e0 * cx / 7.e0
    r2 = xnu * coeff * g3
    r3 = _exp(-r1 * t2)
    h0 = g3 * (bet / delt) * _log(1.e0 + delt * q4 * t2 / q5)
    h1 = r3 * r2 * t2
    h = h0 + h1
    # energy done. now the potential:
    ccrs = (c2 + 2. * c3 * rs) / q7 - q6 * (c4 + 2. * c5 * rs + 3. * c6 * rs2) / (q7 * q7)
    rsthrd = rs / 3.e0
    r4 = rsthrd * ccrs / coeff
    if isinstance(zet, float):
        if abs(zet) >= 1.e0:
            if zet < 0.e0:
                zet = -1.e0 + 1.e-15
            if zet > 0.e0:
                zet = 1.e0 - 1.e-15
    else:
        zet = np.where(zet <= -1.e0, -1.e0 + 1.e-15, zet)
        zet = np.where(zet >= 1.e0, 1.e0 - 1.e-15, zet)
    zp, zm = _spin_scalar(zet, thrdm)
    gz = (zp - zm) / 3.e0
    fac = delt / b + 1.e0
    bg = -3.e0 * b2 * ec * fac / (bet * g4)
    bec = b2 * fac / (bet * g3)
    q8 = q5 * q5 + delt * q4 * q5 * t2
    q9 = 1.e0 + 2.e0 * b * t2
    h0b = -bet * g3 * b * t6 * (2.e0 + b * t2) / q8
    h0rs = -rsthrd * h0b * bec * ecrs
    fact0 = 2.e0 * delt - 6.e0 * b
    fact1 = q5 * q9 + q4 * q9 * q9
    h0bt = 2.e0 * bet * g3 * t4 * ((q4 * q5 * fact0 - delt * fact1) / q8) / q8
    h0rst = rsthrd * t2 * h0bt * bec * ecrs
    h0z = 3.e0 * gz * h0 / g + h0b * (bg * gz + bec * eczet)
    h0t = 2. * bet * g3 * q9 / q8
    h0zt = 3.e0 * gz * h0t / g + h0bt * (bg * gz + bec * eczet)
    fact2 = q4 * q5 + b * t2 * (q4 * q9 + q5)
    fact3 = 2.e0 * b * q5 * q9 + delt * fact2
    h0tt = 4.e0 * bet * g3 * t * (2.e0 * b / q8 - (q9 * fact3 / q8) / q8)
    h1rs = r3 * r2 * t2 * (-r4 + r1 * t2 / 3.e0)
    fact4 = 2.e0 - r1 * t2
    h1rst = r3 * r2 * t2 * (2.e0 * r4 * (1.e0 - r1 * t2) - thrd2 * r1 * t2 * fact4)
    h1z = gz * r3 * r2 * t2 * (3.e0 - 4.e0 * r1 * t2) / g
    h1t = 2.e0 * r3 * r2 * (1.e0 - r1 * t2)
    h1zt = 2.e0 * gz * r3 * r2 * (3.e0 - 11.e0 * r1 * t2 + 4.e0 * r1 * r1 * t4) / g
    h1tt = 4.e0 * r3 * r2 * r1 * t * (-2.e0 + r1 * t2)
    hrs = h0rs + h1rs
    hrst = h0rst + h1rst
    ht = h0t + h1t
    htt = h0tt + h1tt
    hz = h0z + h1z
    hzt = h0zt + h1zt
    comm = h + hrs + hrst + t2 * ht / 6.e0 + 7.e0 * t2 * t * htt / 6.e0
    pref = hz - gz * t2 * ht / g
    fact5 = gz * (2.e0 * ht + t * htt) / g
    comm = comm - pref * zet - uu * htt - vv * ht - ww * (hzt - fact5)
    dvcup = comm + pref
    dvcdn = comm - pref
    return h, dvcup, dvcdn


# --------------------------------------------------------------------------
# bpgcxc.f
# --------------------------------------------------------------------------

def xbecke(d, s, u, v):
    """Becke exchange for a spin-unpolarized system: ``(ex, vx)`` (``bpgcxc.f``)."""
    c = .779555417944150792e1
    b = .42e-2
    bb = -.451357747124625192e-2
    ax = -.738558766382022406e0
    thrd, thrd4 = .333333333333333333e0, .1333333333333333333e1
    # exchange enhancement factor f
    x = c * s
    y1 = 1.e0 / _sqrt(1.e0 + x * x)
    y0 = _log(x + 1.e0 / y1)
    y2 = -x * y1 * y1 * y1
    ddi = 1.e0 / (1.e0 + 6.e0 * b * x * y0)
    dd1 = 6.e0 * b * (y0 + x * y1)
    g = 1.e0 - 0.5e0 * x * dd1 * ddi
    fs = -2.e0 * bb * c * c * ddi
    g1 = -3.e0 * b * (y0 + x * (3.e0 * y1 + x * y2 - dd1 * dd1 * ddi / (6.e0 * b)))
    fss = fs * c * (g1 - g * dd1) * ddi
    fs = fs * g
    f = 1.e0 - bb * x * x * ddi
    # LDA only
    fac = ax * _pow(d, thrd)
    # energy
    ex = fac * f
    # potential
    vx = fac * (thrd4 * f - (u - thrd4 * s * s * s) * fss - v * fs)
    return ex, vx


def corga86(d1, d2, dp11, dp22, dp12, uu, vv):
    """Perdew 1986 gradient correction to correlation (``bpgcxc.f``).

    Returns ``(dec, dvcup, dvcdn)``.
    """
    a1, a2, a3, a4, a5, a6, a7 = (2.568e-3, 1.443307452e-2, 2.843543831e-6,
                                  5.411317331e0, 1.816419933e-1, 1.763993811e-2,
                                  8.12908e-4)
    t13, t23, t43, t53, t76 = (.33333333333333333e0, .66666666666666667e0,
                               .13333333333333333e1, .16666666666666667e1,
                               .11666666666666667e1)
    crt2 = .1587401052e1
    d = d1 + d2
    dm13 = 1.e0 / _pow(d, t13)
    d43 = _pow(d, t43)
    # gradient expansion coefficient
    c1 = a1 + dm13 * (a2 + dm13 * a3)
    c2 = 1.e0 + dm13 * (a4 + dm13 * (a5 + dm13 * a6))
    c = 1.667e-3 + c1 / c2
    dpnorm = _sqrt(dp11 + dp22 + 2.e0 * dp12)
    dpnorm2 = dpnorm * dpnorm
    fi = a7 * dpnorm / (c * _pow(d, t76))
    # spin interpolation
    zet = (d1 - d2) / d
    dd = _sqrt(.5e0 * (_pow(1.e0 + zet, t53) + _pow(1.e0 - zet, t53)))
    # dC(n)/dn
    cp = (-t13 / (d43 * c2 * c2)
          * (c2 * (a2 + 2.e0 * a3 * dm13) - c1 * (a4 + dm13 * (2.e0 * a5 + dm13 * 3.e0 * a6))))
    # spin-independent terms
    www = ((fi - 1.e0) * (cp / c - t43 / d) + (fi - 2.e0) * fi * (cp / c + t76 / d)) * dpnorm2
    uuu = (3.e0 - fi) * fi * uu / dpnorm
    vvv = (fi - 2.e0) * vv
    # spin dependent term
    ddd43 = dd * d43
    zzz = crt2 * .5e0 * t53 / (ddd43 * ddd43) * (_pow(d1, t23) - _pow(d2, t23))
    zz1 = zzz * ((1.e0 - fi) * d2 * dpnorm2 - (2.e0 - fi) * d * (dp12 + dp22))
    zz2 = -zzz * ((1.e0 - fi) * d1 * dpnorm2 - (2.e0 - fi) * d * (dp12 + dp11))
    dec = c / (dd * _exp(fi) * d43)
    dvcup = dec * (www + uuu + vvv + zz1)
    dvcdn = dec * (www + uuu + vvv + zz2)
    dec = dec * dpnorm2 / d
    return dec, dvcup, dvcdn


# --------------------------------------------------------------------------
# corlyp.f
# --------------------------------------------------------------------------

def corlyp(tpot: bool, dp, dm, dp1, dm1, dp2, dm2):
    """Lee-Yang-Parr correlation (``corlyp.f``): ``(ec, vcp0, vcm0)``.

    ``dp, dm`` spin densities, ``dp1, dm1`` their radial gradients and
    ``dp2, dm2`` their Laplacians (the dummy argument ``x`` is dropped).
    """
    aa, bb, cc, dd = 0.04918e0, 0.132e0, 0.2533e0, 0.349e0
    c1, c2, c3, c4, c5 = -4 * aa, dd, 2 * bb, cc, 4.55779986e0
    c6, c7, c8 = 1.e0 / 72.e0, 1.e0 / 18.e0, 0.125e0
    t13 = 1.e0 / 3.e0
    t53, t43 = 5 * t13, 4 * t13
    t83 = 2 * t43
    t89 = 8.e0 / 9.e0
    c9 = t43 + t89

    d0 = dp + dm
    dxsq = 1.e0 / (d0 * d0)
    d1 = dp1 + dm1
    d2 = dp2 + dm2
    d0xt13 = _pow(d0, -t13)
    d0xt53 = d0xt13 * d0xt13 / d0
    dpt53 = _pow(dp, t53)
    dmt53 = _pow(dm, t53)
    # polarization factor
    z = c1 * (dp * dm) * dxsq
    # scaling function
    sc = 1.e0 / (1.e0 + c2 * d0xt13)
    h = c3 * d0xt53 * _exp(-c4 * d0xt13)
    # kinetic energy density expansion
    ga = c5 * (dp * dpt53 + dm * dmt53)
    gb = (c6 * (dp1 * dp1 - dp * dp2 + dm1 * dm1 - dm * dm2)
          + c7 * (dp * dp2 + dm * dm2)
          + c8 * (d0 * d2 - d1 * d1))
    # calculate potential
    if tpot:
        gafp = t83 * c5 * dpt53
        gafm = t83 * c5 * dmt53
        scf = t13 * c2 * d0xt13 / d0 * sc * sc
        sc2 = scf * (d2 + 2 * (scf / sc - 2 * t13 / d0) * d1 * d1)
        chf = t13 * (c4 * d0xt13 - 5) / d0
        hf = chf * h
        hff = h * (chf * chf + t13 * (5.e0 - 4 * t13 * c4 * d0xt13) * dxsq)
        h2 = (hf * d2 + hff * d1 * d1)
        zfp = (c1 * dm - 2 * z * d0) * dxsq
        zfm = (c1 * dp - 2 * z * d0) * dxsq
        yz = z / c1
        yy1 = dp * dm1 + dm * dp1
        yz1 = (yy1 - 2 * yz * d1 * d0) * dxsq
        yz2 = (2 * yz * d1 * d1 - 2 * (yz1 * d1 + yz * d2) * d0
               - 2 * d1 * yy1 / d0 + (dp * dm2 + 2 * dp1 * dm1 + dm * dp2)) * dxsq
        z1 = c1 * yz1
        z2 = c1 * yz2
        yafp = sc * (d0 * zfp + z) + z * d0 * scf
        yafm = sc * (d0 * zfm + z) + z * d0 * scf
        yb = sc * z * h
        ybfp = sc * (h * zfp + z * hf) + z * h * scf
        ybfm = sc * (h * zfm + z * hf) + z * h * scf
        yb1 = sc * (h * z1 + z * hf * d1) + z * h * scf * d1
        yb2 = ((sc * hf + h * scf) * d1 * z1 + h * sc * z2
               + (sc * z1 + z * scf * d1) * hf * d1 + z * sc * h2
               + (z * hf * d1 + h * z1) * scf * d1 + z * h * sc2)
        # collect contributions
        vcp0 = (yafp + ybfp * (ga + gb)
                + yb * (gafp + 2 * c8 * (c9 * dp2 + 2 * dm2))
                + yb1 * 2 * c8 * (c9 * dp1 + 2 * dm1)
                + yb2 * c8 * (t43 * dp + dm))
        vcm0 = (yafm + ybfm * (ga + gb)
                + yb * (gafm + 2 * c8 * (c9 * dm2 + 2 * dp2))
                + yb1 * 2 * c8 * (c9 * dm1 + 2 * dp1)
                + yb2 * c8 * (t43 * dm + dp))
    else:
        vcp0 = np.zeros(np.shape(d0))
        vcm0 = np.zeros(np.shape(d0))
    # correlation energy per electron
    ec = z * sc * (d0 + h * (ga + gb)) / d0
    return ec, vcp0, vcm0


# --------------------------------------------------------------------------
# pbe.f
# --------------------------------------------------------------------------

def exchpbe(rho, s, u, v, lgga: int = 1, lpot: int = 1):
    """PBE exchange for a spin-unpolarized system: ``(ex, vx)`` (``pbe.f``)."""
    thrd, thrd4 = 1.e0 / 3.e0, 4.e0 / 3.e0
    ax = -0.738558766382022405884230032680836e0
    um, uk = 0.2195149727645171e0, 0.8040e0
    ul = um / uk
    exunif = ax * _pow(rho, thrd)
    if lgga == 0:
        ex = exunif
        return ex, ex * thrd4
    s2 = s * s
    p0 = 1.e0 + ul * s2
    fxpbe = 1e0 + uk - uk / p0
    ex = exunif * fxpbe
    if lpot == 0:
        return ex, None
    # energy done. now the potential:
    fs = 2.e0 * uk * ul / (p0 * p0)
    fss = -4.e0 * ul * s * fs / p0
    vx = exunif * (thrd4 * fxpbe - (u - thrd4 * s2 * s) * fss - v * fs)
    return ex, vx


def gcor2(a, a1, b1, b2, b3, b4, rtrs):
    """Slimmed down version of ``gcor``: ``(gg, ggrs)`` (``pbe.f``)."""
    q0 = -2.e0 * a * (1.e0 + a1 * rtrs * rtrs)
    q1 = 2.e0 * a * rtrs * (b1 + rtrs * (b2 + rtrs * (b3 + b4 * rtrs)))
    q2 = _log(1.e0 + 1.e0 / q1)
    gg = q0 * q2
    q3 = a * (b1 / rtrs + 2.e0 * b2 + rtrs * (3.e0 * b3 + 4.e0 * b4 * rtrs))
    ggrs = -2.e0 * a * a1 * q2 - q0 * q3 / (q1 * (1.e0 + q1))
    return gg, ggrs


def corpbe(rs, zet, t, uu, vv, ww, lgga: int = 1, lpot: int = 1):
    """PBE correlation (``pbe.f``).

    Returns ``(ec, vcup, vcdn, h, dvcup, dvcdn)``; the gradient parts are
    ``None`` if ``lgga == 0`` and the potentials ``None`` if ``lpot == 0``.
    """
    thrd = 1.e0 / 3.e0
    thrdm = -thrd
    thrd2 = 2.e0 * thrd
    sixthm = thrdm / 2.e0
    thrd4 = 4.e0 * thrd
    gam = 0.5198420997897463295344212145565e0
    fzz = 8.e0 / (9.e0 * gam)
    gamma = 0.03109069086965489503494086371273e0
    bet = 0.06672455060314922e0
    delt = bet / gamma
    eta = 1.e-12
    # find LSD energy contributions, using [c](10) and Table I[c]
    rtrs = _sqrt(rs)
    eu, eurs = gcor2(0.0310907e0, 0.21370e0, 7.5957e0, 3.5876e0, 1.6382e0,
                     0.49294e0, rtrs)
    ep, eprs = gcor2(0.01554535e0, 0.20548e0, 14.1189e0, 6.1977e0, 3.3662e0,
                     0.62517e0, rtrs)
    alfm, alfrsm = gcor2(0.0168869e0, 0.11125e0, 10.357e0, 3.6231e0, 0.88026e0,
                         0.49671e0, rtrs)
    z4 = _zpow4(zet)
    zp, zm = _spin_scalar(zet, thrd4)
    f = (zp + zm - 2.e0) / gam
    ec = eu * (1.e0 - f * z4) + ep * f * z4 - alfm * f * (1.e0 - z4) / fzz
    # LSD potential from [c](A1)
    ecrs = eurs * (1.e0 - f * z4) + eprs * f * z4 - alfrsm * f * (1.e0 - z4) / fzz
    zp, zm = _spin_scalar(zet, thrd)
    fz = thrd4 * (zp - zm) / gam
    eczet = (4.e0 * _zpow3(zet) * f * (ep - eu + alfm / fzz)
             + fz * (z4 * ep - z4 * eu - (1.e0 - z4) * alfm / fzz))
    comm = ec - rs * ecrs / 3.e0 - zet * eczet
    vcup = comm + eczet
    vcdn = comm - eczet
    if lgga == 0:
        return ec, vcup, vcdn, None, None, None
    # PBE correlation energy
    zp, zm = _spin_scalar(zet, thrd2)
    g = (zp + zm) / 2.e0
    g3 = g * (g * g)
    pon = -ec / (g3 * gamma)
    b = delt / (_exp(pon) - 1.e0)
    b2 = b * b
    t2 = t * t
    t4 = t2 * t2
    q4 = 1.e0 + b * t2
    q5 = 1.e0 + b * t2 + b2 * t4
    h = g3 * (bet / delt) * _log(1.e0 + delt * q4 * t2 / q5)
    if lpot == 0:
        return ec, vcup, vcdn, h, None, None
    # energy done. now the potential, using appx e of [b].
    g4 = g3 * g
    t6 = t4 * t2
    rsthrd = rs / 3.e0
    op = 1.e0 + zet
    om = 1.e0 - zet
    if isinstance(zet, float):
        gz = (libm.spow(op * op + eta, sixthm) - libm.spow(om * om + eta, sixthm)) / 3.e0
    else:
        gz = (_pow(op * op + eta, sixthm) - _pow(om * om + eta, sixthm)) / 3.e0
    fac = delt / b + 1.e0
    bg = -3.e0 * b2 * ec * fac / (bet * g4)
    bec = b2 * fac / (bet * g3)
    q8 = q5 * q5 + delt * q4 * q5 * t2
    q9 = 1.e0 + 2.e0 * b * t2
    hb = -bet * g3 * b * t6 * (2.e0 + b * t2) / q8
    hrs = -rsthrd * hb * bec * ecrs
    fact0 = 2.e0 * delt - 6.e0 * b
    fact1 = q5 * q9 + q4 * q9 * q9
    hbt = 2.e0 * bet * g3 * t4 * ((q4 * q5 * fact0 - delt * fact1) / q8) / q8
    hrst = rsthrd * t2 * hbt * bec * ecrs
    hz = 3.e0 * gz * h / g + hb * (bg * gz + bec * eczet)
    ht = 2.e0 * bet * g3 * q9 / q8
    hzt = 3.e0 * gz * ht / g + hbt * (bg * gz + bec * eczet)
    fact2 = q4 * q5 + b * t2 * (q4 * q9 + q5)
    fact3 = 2.e0 * b * q5 * q9 + delt * fact2
    htt = 4.e0 * bet * g3 * t * (2.e0 * b / q8 - (q9 * fact3 / q8) / q8)
    comm = h + hrs + hrst + t2 * ht / 6.e0 + 7.e0 * t2 * t * htt / 6.e0
    pref = hz - gz * t2 * ht / g
    fact5 = gz * (2.e0 * ht + t * htt) / g
    comm = comm - pref * zet - uu * htt - vv * ht - ww * (hzt - fact5)
    dvcup = comm + pref
    dvcdn = comm - pref
    return ec, vcup, vcdn, h, dvcup, dvcdn


# --------------------------------------------------------------------------
# ggarad.f
# --------------------------------------------------------------------------

_EPS_GGA = 1.e-15
_PISQ3 = .296088132032680740e2
_THRD = .333333333333333333e0
_THRD2 = .666666666666666666e0
_PI = .314159265358979312e1
_CRS = 1.91915829267751281e0


def ggaxrad(mode: int, r, rh, rhp, rhpp):
    """Exchange potential and energy density, spherical symmetry (``ggarad.f``).

    ``mode``: 1 LSDA, 2 Becke, 3 Perdew-Wang 91, 5 PBE.  ``rh, rhp, rhpp``
    are the densities of ONE spin channel and its radial derivatives; both
    spin channels are taken equal (the program is spin-unpolarized), so the
    Fortran loop over ``i=1,2`` contributes twice the same term.  Returns
    ``(xpot, xen)``: exchange potential (one spin) and energy per electron.
    """
    rh = np.asarray(rh, dtype=float)
    xpot = np.zeros(rh.shape)
    xen = np.zeros(rh.shape)
    ok = ~(rh <= _EPS_GGA)
    rhk = rh[ok]
    d = 2.e0 * rhk
    if mode == 1:
        vx, ex = xlda(d)
    elif mode in (2, 3, 5):
        rk = np.asarray(r, dtype=float)[ok]
        dp = 2.e0 * np.asarray(rhp, dtype=float)[ok]
        dpp = 2.e0 * np.asarray(rhpp, dtype=float)[ok]
        fk = _pow(_PISQ3 * d, _THRD)
        fk2 = 2.e0 * fk
        fk2sq = fk2 * fk2
        s = _abs(dp) / (fk2 * d)
        u = _abs(dp) * dpp / (d * d * fk2 * fk2sq)
        v = (dpp + 2.e0 * dp / rk) / (d * fk2sq)
        if mode == 2:
            ex, vx = xbecke(d, s, u, v)
        elif mode == 3:
            ex, vx = exch(d, s, u, v)
        else:
            ex, vx = exchpbe(d, s, u, v, 1, 1)
    else:
        raise FortranStop("ggaxrad : mode improper")
    xpot[ok] = vx
    # xen = (rh(1)*ex + rh(2)*ex) / max(rh(1)+rh(2),eps)
    term = rhk * ex
    xen[ok] = (term + term) / np.maximum(rhk + rhk, _EPS_GGA)
    return xpot, xen


def ggacrad(mode: int, r, rh, rhp, rhpp):
    """Correlation potential and energy density, spherical symmetry (``ggarad.f``).

    ``mode``: 1 LSDA (PW91 parametrization), 2 Perdew-Wang 91, 3 Perdew 86,
    4 Lee-Yang-Parr, 5 PBE.  Arguments as in :func:`ggaxrad` (one spin
    channel, both channels equal).  Returns ``(cpot, cen)``.
    """
    rh = np.asarray(rh, dtype=float)
    cpot = np.zeros(rh.shape)
    cen = np.zeros(rh.shape)
    dtot = rh + rh
    ok = ~(dtot <= _EPS_GGA)
    rh1 = rh[ok]
    d = dtot[ok]
    if mode != 4:
        zet = (rh1 - rh1) / d
        zet = 0.0 if not zet.any() else zet
        fk = _pow(_PISQ3 * d, _THRD)
        rs = _CRS / fk
        if mode == 1:
            ec, vcup, _, _, _, _ = corlsd(rs, zet)
            dvcup = 0.e0
            h = 0.e0
        elif mode == 2 or mode == 5:
            rk = np.asarray(r, dtype=float)[ok]
            rp1 = np.asarray(rhp, dtype=float)[ok]
            rpp1 = np.asarray(rhpp, dtype=float)[ok]
            if mode == 2:
                ec, vcup, _, ecrs, eczet, _ = corlsd(rs, zet)
            dp = rp1 + rp1
            dpp = rpp1 + rpp1
            ztp = (rp1 - rp1 - zet * dp) / d
            sk = 2.e0 * _sqrt(fk / _PI)
            zp, zm = _spin_scalar(zet, _THRD2)
            g = (zp + zm) / 2.e0
            gks2 = 2.e0 * sk * g
            gks2sq = gks2 * gks2
            t = _abs(dp) / (d * gks2)
            uu = _abs(dp) * dpp / (d * d * gks2sq * gks2)
            vv = (dpp + 2.e0 * dp / rk) / (d * gks2sq)
            ww = dp * ztp / (d * gks2sq)
            if mode == 2:
                h, dvcup, _ = corgga(rs, zet, t, uu, vv, ww, fk, sk, g, ec, ecrs, eczet)
            else:
                ec, vcup, _, h, dvcup, _ = corpbe(rs, zet, t, uu, vv, ww, 1, 1)
        elif mode == 3:
            rk = np.asarray(r, dtype=float)[ok]
            rp1 = np.asarray(rhp, dtype=float)[ok]
            rpp1 = np.asarray(rhpp, dtype=float)[ok]
            ec, vcup, _, _, _, _ = corlsd(rs, zet)
            dp = rp1 + rp1
            dpp = rpp1 + rpp1
            uu = _abs(dp) * dpp
            vv = dpp + 2.e0 * dp / rk
            dp11 = rp1 * rp1
            h = np.zeros(d.shape)
            dvcup = np.zeros(d.shape)
            # corga86 is skipped where both gradients vanish; the Fortran
            # then reuses stale h, dvcup of an earlier call (taken as 0 here)
            nz = rp1 != 0.e0
            if nz.any():
                hh, dv, _ = corga86(rh1[nz], rh1[nz], dp11[nz], dp11[nz], dp11[nz],
                                    uu[nz], vv[nz])
                h[nz] = hh
                dvcup[nz] = dv
        else:
            raise FortranStop("ggacrad : mode improper")
        cpot[ok] = vcup + dvcup
        cen[ok] = ec + h
    else:
        rk = np.asarray(r, dtype=float)[ok]
        rp1 = np.asarray(rhp, dtype=float)[ok]
        rpp1 = np.asarray(rhpp, dtype=float)[ok]
        flp1 = rpp1 + 2.e0 * rp1 / rk
        ec, vcp0, _ = corlyp(True, rh1, rh1, rp1, rp1, flp1, flp1)
        cen[ok] = ec
        cpot[ok] = vcp0
    return cpot, cen


# --------------------------------------------------------------------------
# vexcor.f
# --------------------------------------------------------------------------

class XcResult(NamedTuple):
    """Output of :func:`vexcor`."""

    vpxc: np.ndarray        # xc potential
    evxc: float | None      # xc potential energy  int rho*vxc   (flag only)
    ex: float | None        # exchange energy                    (flag only)
    ec: float | None        # correlation energy                 (flag only)
    dex: np.ndarray         # exchange energy density per electron
    dec: np.ndarray         # correlation energy density per electron
    dexc: np.ndarray | None  # dex + dec, common /xced/           (flag only)


def vexcor(iexc: int, mmax: int, r: np.ndarray, rho: np.ndarray,
           rhop: np.ndarray, rhopp: np.ndarray, flag: bool = False,
           ec_in: float = 0.0) -> XcResult:
    """Exchange-correlation potential and energies (``vexcor.f``).

    ``rho`` is the total density (electrons per unit volume times 4 pi, as
    everywhere in the program), ``rhop`` and ``rhopp`` its first and second
    radial derivatives.  With ``flag`` the energies ``ex``, ``ec`` and
    ``evxc`` are integrated with :func:`dmelm`.

    Two peculiarities of the Fortran are kept and documented here:

    * ``iexc = 7``: the correlation energy density is overwritten by the
      dummy argument ``ec`` (``dec(i)=ec``), i.e. by the value ``ec`` had on
      entry.  ``ec_in`` supplies that value (0 for the call in ``fhipp``, so
      the reported correlation energy is exactly zero).
    * ``iexc = 2``: ``dec`` is never assigned in the Fortran (``dex`` holds
      the full xc energy density and ``ec`` integrates uninitialized
      memory).  Here ``dec = 0``.
    """
    r = np.asarray(r[:mmax], dtype=float)
    rho = np.asarray(rho[:mmax], dtype=float)
    rhop = np.asarray(rhop[:mmax], dtype=float)
    rhopp = np.asarray(rhopp[:mmax], dtype=float)
    pi4 = 16.0e0 * math.atan(1.0e0)
    with np.errstate(all="ignore"):
        if iexc == 11:
            # x only
            vpxc, dex = xlda(rho / pi4)
            dec = np.zeros(mmax)
        elif iexc == 1:
            # C Wigner
            ex_, _, exc_, fxc_ = wigner(rho / pi4)
            vpxc = fxc_
            dex = ex_
            dec = exc_ - ex_
        elif iexc == 2:
            # C Hedin-Lundqvist
            rh = rho / pi4
            vpxc = np.zeros(mmax)
            dex = np.zeros(mmax)
            dec = np.zeros(mmax)
            ok = rh != 0.0e0
            rs = 0.62035049e0 * _pow(rh[ok], -0.3333333333333333e0)
            x = rs / 21.0e0
            aln = _log(1.0e0 + 1.0e0 / x)
            ecp = aln + (x * (x * x) * aln - x * x) + x / 2 - 1.0e0 / 3.0e0
            dex[ok] = -0.458175e0 / rs - 0.0225e0 * ecp
            vpxc[ok] = -0.6109e0 / rs - 0.0225e0 * aln
        elif iexc == 3:
            # XC Ceperley - Alder, Perdew-Zunger parametrization
            ex_, _, exc_, fxc_ = cepal(rho / pi4)
            vpxc = fxc_
            dex = ex_
            dec = exc_ - ex_
        elif iexc in (4, 5, 6, 9, 10):
            d = .5e0 * rho / pi4
            dp = .5e0 * rhop / pi4
            dpp = .5e0 * rhopp / pi4
            xmode, cmode = {4: (3, 2), 5: (2, 3), 6: (5, 5), 9: (2, 4), 10: (3, 4)}[iexc]
            xpot, dex = ggaxrad(xmode, r, d, dp, dpp)
            cpot, dec = ggacrad(cmode, r, d, dp, dpp)
            vpxc = xpot + cpot
        elif iexc == 7:
            # PW91 LDA + MacDonald-Vosko relativistic correction to exchange
            d = 0.5e0 * rho / pi4
            xpot, dex = ggaxrad(1, r, d, None, None)
            cpot, _ = ggacrad(1, r, d, None, None)
            fex, fvx = relxc(1, 2.e0 * d)
            dex = fex * dex
            dec = np.full(mmax, float(ec_in))
            vpxc = fvx * xpot + cpot
        elif iexc == 8:
            # XC Ceperley-Alder in Perdew-Wang parametrization of 1991
            d = 0.5e0 * rho / pi4
            xpot, dex = ggaxrad(1, r, d, None, None)
            cpot, dec = ggacrad(1, r, d, None, None)
            vpxc = xpot + cpot
        else:
            raise FortranStop(f"vexcor - stop: xc option not implemented {iexc}")

        evxc = ex = ec = dexc = None
        if flag:
            # total energy components
            dexc = dex + dec
            al = libm.slog(float(r[1]) / float(r[0]))
            ex = dmelm(mmax, al, r, rho, dex)
            ec = dmelm(mmax, al, r, rho, dec)
            evxc = dmelm(mmax, al, r, rho, vpxc)
    return XcResult(vpxc, evxc, ex, ec, dex, dec, dexc)
