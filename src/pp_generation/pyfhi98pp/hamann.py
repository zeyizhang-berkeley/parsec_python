"""Generalized norm-conserving pseudopotentials of Hamann type.

Port of ``hamann.f`` (D.R. Hamann, Phys. Rev. B 40, 2980 (1989); version
adopted at the Fritz-Haber-Institut).  The all-electron potential is cut off
inside ``rc`` with ``fc = exp(-(r/rc)**alam)`` and an additive term
``cl*fc`` is iterated until the nodeless solution has the reference energy;
the wave function is then corrected by ``del * r**(l+1) * fc`` to restore
the norm and the potential is inverted for the corrected wave function.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from . import libm
from .errors import FortranStop
from .fortran_list import list_record
from .radial import _fdiv, _fsqrt, _powi, dftseq, seqsum

__all__ = ["HamannResult", "hamann", "ALAM", "HAMANN_ITMAX"]

ALAM: float = 3.5e0        # alam: healing parameter for the wave function
HAMANN_ITMAX: int = 30     # maximum number of iterations for the additive term


class HamannResult(NamedTuple):
    """Output arguments of ``hamann`` (arrays are new objects)."""

    u: np.ndarray      # radial pseudo wave function
    up: np.ndarray     # dto. 1st derivative (on the logarithmic grid)
    upp: np.ndarray    # dto. 2nd derivative
    v: np.ndarray      # screened pseudopotential
    nin: int = 0       # outermost point of the pseudo wave function
    mchi: int = 0      # matching point of the last dftseq call
    etest: float = 0.0  # eigenvalue of the final potential
    iter: int = 0      # iterations for the additive term (diagnostic)
    gam: float = 0.0   # norm correction factor (diagnostic)
    delta: float = 0.0  # coefficient of the norm correction, "del" (diagnostic)


@np.errstate(all="ignore")
def hamann(z: float, epspp: float, l: int, rc: float, ein: float, mode: int,
           mch: int, mmax: int, r: np.ndarray, u: np.ndarray, up: np.ndarray,
           upp: np.ndarray, vin: np.ndarray,
           warn: Callable[[str], None] | None = None) -> HamannResult:
    """Hamann pseudopotential for one channel (``hamann.f``).

    Arguments as in ``hamann(io,z,epspp,l,rc,ein,mode,mch,mmax,r,u,up,upp,
    vin,v)`` without the unused unit number: ``z`` nuclear charge passed to
    ``dftseq`` (0 in ``ncpp``), ``epspp`` accuracy of the additive term,
    ``rc`` cutoff radius, ``ein`` reference energy, ``mode`` the ``dftseq``
    mode (2 bound state; 4, 7 energy for a given logarithmic derivative),
    ``mch`` the grid index of the reference radius, ``u, up, upp`` the
    all-electron wave function, ``vin`` the all-electron potential.
    """
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    vin = np.ascontiguousarray(np.asarray(vin, dtype=float)[:mmax])
    u = np.array(u[:mmax], dtype=float)
    up = np.array(up[:mmax], dtype=float)
    upp = np.array(upp[:mmax], dtype=float)
    rc = float(rc)
    ein = float(ein)
    alam = ALAM

    # initialization
    wm = 1.e0
    wa = np.zeros(mmax)
    wb = np.ones(mmax)
    amesh = float(r[1]) / float(r[0])
    al = libm.slog(amesh)

    etest = ein
    upmch = float(up[mch - 1])
    umch = float(u[mch - 1])
    uld = _fdiv(upmch, umch)

    # construct pseudopotential: form cutoff potential
    texp = libm.pow(r / rc, alam)
    small = texp < 700.0e0
    fc = np.zeros(mmax)
    fc[small] = libm.exp(-texp[small])
    vc = (1.0e0 - fc) * vin

    # iteration to find correct additive term
    k2l3 = 2 * l + 3
    r2l3 = _powi(r, k2l3)
    rl2 = _powi(r, l + 2)
    rl1 = _powi(r, l + 1)
    r1 = float(r[0])
    fc1 = float(fc[0])
    ro = r1 / _fsqrt(amesh)
    ro2l3 = float(_powi(ro, k2l3))
    it = 0
    cl = float(vin[int(libm.slog(rc / r1) / al + 1) - 1])       # vin(nrc)
    sf_terms = al * r2l3 * (fc * fc)
    alrl2fc = al * rl2 * fc
    alr = al * r
    while True:
        it += 1
        if it > HAMANN_ITMAX:
            text = "hamann - stop: convergence error l,rc,rmch,ein"
            if warn:
                warn(list_record("& " + text, l, rc, float(r[mch - 1]), ein))
            raise FortranStop(text)

        v = vc + cl * fc

        etest = ein
        mchi = mch
        res = dftseq(mode, z, mmax, r, l + 1, l, wm, v, wa, wb, mch=mchi, uld=uld,
                     e=etest, warn=warn, u=u, up=up, upp=upp)
        nin, mchi, etest = res.nin, res.mch, res.e
        u, up, upp = res.u, res.up, res.upp

        # perform integral times cutoff function
        # also integrals for norm correction
        u1 = float(u[0])
        sv = ro2l3 * (u1 * u1 * fc1 / float(_powi(r1, 2 * l + 2))) / float(k2l3)
        sf = ro2l3 * (fc1 * fc1) / float(k2l3)
        sx = ro2l3 * (u1 * fc1 / float(_powi(r1, l + 1))) / float(k2l3)

        m = nin - 3
        sf = seqsum(sf_terms[:m], sf)
        sx = seqsum(alrl2fc[:m] * u[:m], sx)
        sv = seqsum(alr[:m] * (u[:m] * u[:m]) * fc[:m], sv)

        ia, ib, ic = nin - 3, nin - 2, nin - 1          # Python indices of nin-2, nin-1, nin
        ua, ub, uc = float(u[ia]), float(u[ib]), float(u[ic])
        fa, fb, fcc = float(fc[ia]), float(fc[ib]), float(fc[ic])
        ra, rb, rcc = float(r[ia]), float(r[ib]), float(r[ic])
        sv = sv + al * (23.e0 * ra * (ua * ua) * fa
                        + 28.e0 * rb * (ub * ub) * fb
                        + 9.e0 * rcc * (uc * uc) * fcc) / 24.e0
        sf = sf + al * (23.e0 * float(r2l3[ia]) * fa
                        + 28.e0 * float(r2l3[ib]) * fb
                        + 9.e0 * float(r2l3[ic]) * fcc) / 24.e0
        sx = sx + al * (23.e0 * float(rl2[ia]) * ua * fa
                        + 28.e0 * float(rl2[ib]) * ub * fb
                        + 9.e0 * float(rl2[ic]) * uc * fcc) / 24.e0

        # perturbation theory to improve potential
        dcl = _fdiv(ein - etest, sv)
        cl = cl + dcl
        if not abs(dcl) > epspp:
            break

    # find coefficient for norm correction
    gam = abs(_fdiv(umch, float(u[mch - 1])))
    sx = sx * (gam * gam)
    sf = sf * (gam * gam)
    delta = _fdiv(-sx + _fdiv(sx, abs(sx)) * _fsqrt(sx * sx - sf * (gam * gam - 1.0e0)), sf)

    # construct final wavefunction
    u = u.copy()
    u[:nin] = gam * (u[:nin] + delta * rl1[:nin] * fc[:nin])

    # construct final pseudopotential
    rn = r[:nin]
    x = rn / rc
    vn = v[:nin]
    v = vin.copy()
    v[:nin] = vn + (gam * delta * rl1[:nin] * fc[:nin] / (2.e0 * u[:nin])) * (
        (alam * alam * libm.pow(x, 2 * alam)
         - (2.e0 * alam * l + alam * (alam + 1.e0)) * libm.pow(x, alam)) / (rn * rn)
        + 2.e0 * etest - 2.e0 * vn)

    res = dftseq(mode, z, mmax, r, l + 1, l, wm, v, wa, wb, mch=mchi, uld=uld,
                 e=etest, warn=warn, u=u, up=up, upp=upp)
    return HamannResult(res.u, res.up, res.upp, v, res.nin, res.mch, res.e, it, gam, delta)
