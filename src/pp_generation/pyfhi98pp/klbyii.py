"""Bound-state spectrum in a basis of Laguerre functions.

Ports of ``klbyiiLAPACK.f`` (radial Schroedinger equation with a semilocal
or a fully separable potential as a 30 x 30 matrix eigenvalue problem),
``laguerre.f`` (associated Laguerre polynomials and the basis functions
built from them), ``gaussq.f``/``gauss.h`` (32-point Gauss-Laguerre
quadrature) and ``basis.h`` (scaling of the basis).

The Fortran routine keeps its tables in ``save`` variables that are set up
on the first call; here they live in a :class:`LaguerreBasis` object that
the caller (``ppcheck``) creates once and passes to every :func:`klbyii`
call.  Hartree atomic units.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import libm
from .constants import MS
from .dspevx import dspevx_lowest

__all__ = ["LaguerreBasis", "klbyii", "plgr0", "plgr1", "bas0", "bas1",
           "GAUSS_X", "GAUSS_W", "NMX"]

# --- gauss.h: Gauss-Laguerre quadrature, integrate[exp(-x) f(x)] ------------
IMX = 32
GAUSS_X: tuple[float, ...] = tuple(float(s) for s in """
 .44489365833267018e-01 .23452610951961854e+00 .57688462930188643e+00
 .10724487538178176e+01 .17224087764446454e+01 .25283367064257949e+01
 .34922132730219945e+01 .46164567697497674e+01 .59039585041742439e+01
 .73581267331862411e+01 .89829409242125961e+01 .10783018632539972e+02
 .12763697986742725e+02 .14931139755522557e+02 .17292454336715315e+02
 .19855860940336055e+02 .22630889013196774e+02 .25628636022459248e+02
 .28862101816323475e+02 .32346629153964737e+02 .36100494805751974e+02
 .40145719771539442e+02 .44509207995754938e+02 .49224394987308639e+02
 .54333721333396907e+02 .59892509162134018e+02 .65975377287935053e+02
 .72687628090662709e+02 .80187446977913523e+02 .88735340417892399e+02
 .98829542868283973e+02 .11175139809793770e+03""".split())
GAUSS_W: tuple[float, ...] = tuple(float(s) for s in """
 .10921834195238497e+00 .21044310793881323e+00 .23521322966984801e+00
 .19590333597288104e+00 .12998378628607176e+00 .70578623865717442e-01
 .31760912509175070e-01 .11918214834838557e-01 .37388162946115248e-02
 .98080330661495513e-03 .21486491880136419e-03 .39203419679879472e-04
 .59345416128686329e-05 .74164045786675522e-06 .76045678791207815e-07
 .63506022266258067e-08 .42813829710409289e-09 .23058994918913361e-10
 .97993792887270941e-12 .32378016577292665e-13 .81718234434207195e-15
 .15421338333938233e-16 .21197922901636186e-18 .20544296737880455e-20
 .13469825866373952e-22 .56612941303973593e-25 .14185605454630369e-27
 .19133754944542243e-30 .11922487600982224e-33 .26715112192401370e-37
 .13386169421062563e-41 .45105361938989742e-47""".split())

# --- basis.h: f(x) = b exp(-c x) x**ll L_n^k(a x) ---------------------------
BASIS_A = 8.0
BASIS_B = 22.6274169979695223
BASIS_C = 0.5 * BASIS_A
BASIS_L = 0
BASIS_LL = BASIS_L + 1
BASIS_K = 2 * BASIS_L + 2

NMX = 30          # nmx: basis size (consistent with the quadrature order)


def _factorials(nkmx: int) -> list[float]:
    """``fac(i) = i*fac(i-1)`` in double precision, as in ``laguerre.f``."""
    fac = [1.0]
    for i in range(1, nkmx + 1):
        fac.append(i * fac[i - 1])
    return fac


_FAC = _factorials(34)


def _coefficients(n: int, k: int, first: int) -> tuple[list[float], float]:
    """Series coefficients of ``plgr0`` (``first = 0``) or ``plgr1`` (1).

    ``phase*fac(n+k)/(fac(n-m)*fac(k+m))/fac(m-first)`` for ``m = first..n``
    and the normalization ``sqrt(fac(n+k)/fac(n))``.
    """
    fac = _FAC
    phase = -1.0 if first == 0 else 1.0
    coef = []
    for m in range(first, n + 1):
        phase = -phase
        coef.append(phase * fac[n + k] / (fac[n - m] * fac[k + m]) / fac[m - first])
    return coef, math.sqrt(fac[n + k] / fac[n])


def _series(coef: list[float], norm: float, x: np.ndarray) -> np.ndarray:
    """``sum_m coef(m) * x**j`` accumulated in the Fortran order.

    The powers are built by repeated multiplication (``y = y*x``) and the
    terms are added one after the other, both exactly as in the do-loop.
    """
    acc = np.zeros(x.shape)
    y = np.ones(x.shape)
    for j, cm in enumerate(coef):
        if j > 0:
            y = y * x
        acc = acc + cm * y
    return acc / norm


def plgr0(n: int, k: int, x) -> np.ndarray:
    """Normalized associated Laguerre polynomial ``L_n^k(x)`` (``laguerre.f``)."""
    if n + k > 34:
        raise ValueError("plgr0")
    x = np.asarray(x, dtype=float)
    coef, norm = _coefficients(n, k, 0)
    return _series(coef, norm, x)


def plgr1(n: int, k: int, x) -> np.ndarray:
    """First derivative of ``plgr0`` with respect to ``x`` (``laguerre.f``)."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.zeros(x.shape)
    coef, norm = _coefficients(n, k, 1)
    return _series(coef, norm, x)


def bas0(n: int, x) -> np.ndarray:
    """Basis function ``b exp(-a x/2) x L_n^k(a x)`` (``laguerre.f``)."""
    x = np.asarray(x, dtype=float)
    xx = x * BASIS_A
    return BASIS_B * libm.exp(-.5 * xx) * x * plgr0(n, BASIS_K, xx)


def bas1(n: int, x) -> np.ndarray:
    """First derivative of :func:`bas0` (``laguerre.f``)."""
    x = np.asarray(x, dtype=float)
    xx = x * BASIS_A
    return ((-BASIS_C + BASIS_LL / x) * bas0(n, x)
            + BASIS_A * BASIS_B * libm.exp(-.5 * xx) * x * plgr1(n, BASIS_K, xx))


def _simpson_rows(al: float, r: np.ndarray, t: np.ndarray, first: np.ndarray,
                  last: np.ndarray) -> np.ndarray:
    """Simpson sums of ``gltfmv`` for several integrands at once.

    ``t[k, i] = r(i)*ga(i)*g(i)*gb(i)`` (the summand of the Fortran loops),
    ``first``/``last`` the end-point products ``ga*g*gb*r`` at the first and
    the last mesh point, formed in the order of the Fortran expression.
    """
    mx_ = t.shape[1]
    esum = np.cumsum(t[:, 1:mx_ - 1:2], axis=1)[:, -1]
    if mx_ > 3:
        osum = np.cumsum(t[:, 2:mx_ - 2:2], axis=1)[:, -1]
    else:
        osum = np.zeros(t.shape[0])
    return (al * (4.0 * esum + 2.0 * osum + first + last)) / 3.0


class LaguerreBasis:
    """Tables of ``klbyii`` that the Fortran sets up once (``tfirst``).

    ``g0[n, i]``   basis function ``n`` on the radial mesh (``g0(i,n+1)``)
    ``ht[n1, n2]`` kinetic energy matrix, ``htl`` its centrifugal part (both
                   for ``n1 <= n2`` only, as in the Fortran)
    ``hnl[l]``     projections of the separable potential of channel ``l``,
                   computed on the first call with ``nnl > 0`` for that ``l``
    """

    def __init__(self, mmax: int, r: np.ndarray):
        if mmax % 2 == 0:
            raise ValueError("klbyii: odd number of mesh points expected")
        self.mmax = mmax
        self.r = np.ascontiguousarray(r[:mmax], dtype=float)
        self.al = .1 * libm.slog(float(self.r[10]) / float(self.r[0]))
        a = BASIS_A
        self.g0 = np.array([bas0(n, self.r) for n in range(NMX)])
        x = np.array(GAUSS_X)
        w = np.array(GAUSS_W)
        ex = libm.exp(0.5 * x)
        f0 = np.array([bas0(n, x / a) * ex / x for n in range(NMX)])
        f1 = np.array([bas1(n, x / a) * ex for n in range(NMX)])
        ht = np.zeros((NMX, NMX))
        htl = np.zeros((NMX, NMX))
        for n2 in range(NMX):
            # explicit gauss quadrature, summed in the order of the points
            ht[:n2 + 1, n2] = np.cumsum(w * f1[:n2 + 1] * f1[n2], axis=1)[:, -1]
            htl[:n2 + 1, n2] = np.cumsum(w * f0[:n2 + 1] * f0[n2], axis=1)[:, -1]
        self.ht = 0.5 * ht / a
        self.htl = a * htl
        self.hnl: dict[int, np.ndarray] = {}
        # r(i)*g0(i,n): first factor pair of every matrix element
        self._rg0 = self.r * self.g0

    def projections(self, l: int, vnl: np.ndarray) -> np.ndarray:
        """``hnl(n,l+1) = gltfmv(g0(n), vnl, r)``, kept from the first call."""
        if l not in self.hnl:
            r = self.r
            vnl = np.asarray(vnl[:self.mmax], dtype=float)
            t = self._rg0 * vnl * r
            first = self.g0[:, 0] * vnl[0] * r[0] * r[0]
            last = self.g0[:, -1] * vnl[-1] * r[-1] * r[-1]
            half = 0.5 * self.g0[:, 0] * vnl[0] * r[0] * r[0]
            self.hnl[l] = _simpson_rows(self.al, r, t, first, last) + half
        return self.hnl[l]

    def potential_matrix(self, vloc: np.ndarray) -> np.ndarray:
        """``gltfmv(g0(n1), vloc, g0(n2))`` for ``n1 <= n2`` (upper triangle)."""
        r = self.r
        g0 = self.g0
        a = self._rg0 * vloc                 # r*ga*g
        v1 = float(vloc[0])
        vm = float(vloc[-1])
        out = np.zeros((NMX, NMX))
        for n2 in range(NMX):
            t = a[:n2 + 1] * g0[n2]
            first = g0[:n2 + 1, 0] * v1 * g0[n2, 0] * r[0]
            last = g0[:n2 + 1, -1] * vm * g0[n2, -1] * r[-1]
            half = 0.5 * g0[:n2 + 1, 0] * v1 * g0[n2, 0] * r[0]
            out[:n2 + 1, n2] = _simpson_rows(self.al, r, t, first, last) + half
        return out


def klbyii(l: int, mmax: int, r: np.ndarray, vsc: np.ndarray, vsl: np.ndarray,
           nnl: int, cnl: float, vnl: np.ndarray, enl: np.ndarray,
           basis: LaguerreBasis, warn: Callable[[str], None] | None = None,
           eigensolver: Callable | None = None) -> tuple[np.ndarray, list[float], int]:
    """Bound states of a (semi)local plus separable potential (``klbyii``).

    ``l``    angular momentum
    ``vsc``  local (screening) potential, ``vsl`` semilocal potential
    ``nnl``  number of projectors (0 or 1), ``cnl`` their strength (inverse
             of the average nonlocal potential), ``vnl`` projector function
    ``enl``  spectrum array of the caller (``enl(ms)``): as in the Fortran
             only the leading negative eigenvalues are stored, the other
             elements keep their previous contents
    ``basis`` the tables of this run (:class:`LaguerreBasis`)
    ``eigensolver`` replacement for :func:`dspevx_lowest` (test harness)

    Returns ``(enl, he, info)``: the updated copy of ``enl``, the eigenvalues
    delivered by ``dspevx`` (at most four, ascending) and its ``info``.  The
    real space eigenfunctions ``unl`` of the Fortran are not formed: with
    ``jobz = 'N'`` the eigenvector array is never set there either.
    """
    r = basis.r
    vloc = np.asarray(vsc[:mmax], dtype=float) + np.asarray(vsl[:mmax], dtype=float)
    fll = .5 * l * (l + 1)
    hu = basis.ht + fll * basis.htl + basis.potential_matrix(vloc)
    if nnl > 0:
        hnl = basis.projections(l, vnl)
        for n2 in range(NMX):
            hu[:n2 + 1, n2] = hu[:n2 + 1, n2] + cnl * hnl[:n2 + 1] * hnl[n2]
    # lower triangle packed by columns: hessl(n3) = hu(n2,n1), n2 >= n1
    hessl = [float(hu[n1, n2]) for n1 in range(NMX) for n2 in range(n1, NMX)]
    solver = dspevx_lowest if eigensolver is None else eigensolver
    m, w, info = solver(NMX, hessl, 1, 4, 1.e-6)
    he = [0.0] * NMX
    for k in range(min(m, len(w), NMX)):
        he[k] = float(w[k])
    if info != 0:
        if warn:
            warn(" & klbyii - warning: eigenvalues not converged")
            # the Fortran prints iwk2(1:4) (ifail), which dspevx leaves at zero
            warn(" & klbyii - he(i)" + "".join("%12d" % 0 for _ in range(4)))
    enl = np.array(enl, dtype=float)
    i = 0
    while he[i] < 0.0 and i + 1 < MS:
        enl[i] = he[i]
        i += 1
    return enl, he[:4], info
