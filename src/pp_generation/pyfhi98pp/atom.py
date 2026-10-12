"""Self-consistent all-electron atom.

Ports of ``atomini.f`` (Thomas-Fermi start potential and eigenvalue
guesses) and ``sratom_n.f`` (self-consistent scalar-relativistic or
non-relativistic spherical atom, optional frozen core).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import libm
from .radial import anderson, dftseq, vestat
from .xc import vexcor

__all__ = ["tfapot", "atomini", "SratomResult", "sratom"]


def tfapot(r, z: float):
    """Generalized Thomas-Fermi atomic potential (``atomini.f``)."""
    r = np.asarray(r, dtype=float)
    b = libm.spow(0.69395656e0 / z, .33333333e0)
    x = r / b
    xs = np.sqrt(x)
    t = z / (1.0e0 + xs * (0.02747e0 - x * (0.1486e0 - 0.007298e0 * x))
             + x * (1.243e0 + x * (0.2302e0 + 0.006944e0 * x)))
    t = np.where(t < 1.0e0, 1.0e0, t)
    return -t / r


def atomini(imx: int, mmax: int, z: float, n: Sequence[int], f: Sequence[float],
            r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initial effective potential and eigenvalues (``atomini.f``).

    Returns ``(vi, e)``: the Thomas-Fermi potential on the mesh and the
    hydrogen-like eigenvalue guesses of the ``imx`` states.
    """
    vi = tfapot(np.asarray(r[:mmax], dtype=float), z)
    e = np.zeros(imx)
    sf = 0.e0
    vlast = float(vi[mmax - 1])
    for i in range(imx):
        sf = sf + f[i]
        zz = z - sf + 1.e0
        q = zz / n[i]
        e[i] = -.5e0 * (q * q)
        if e[i] > vlast:
            e[i] = 2.0e0 * vlast
    return vi, e


@dataclass
class SratomResult:
    """Output arguments of ``sratom`` (all arrays are new objects)."""

    it: int               # iterations used; ``itmax + 1`` if not converged
    converged: bool       # tconv at exit
    e: np.ndarray         # e(ms): eigenvalues (Hartree)
    ninu: list[int]       # ninu(ms): outermost mesh point of each state
    de: float             # relative change of the eigenvalue sum
    rpk: np.ndarray       # rpk(ms): outermost peak of each wave function
    vi: np.ndarray        # vi(mx): effective potential after the last mixing
    rho: np.ndarray       # rho(mx): total density (times 4 pi)
    rhop: np.ndarray      # rhop(mx): d rho/dr
    rhopp: np.ndarray     # rhopp(mx): d2 rho/dr2
    dc: np.ndarray        # dc(mx): core density
    dcp: np.ndarray       # dcp(mx)
    dcpp: np.ndarray      # dcpp(mx)
    uu: np.ndarray        # uu(mx,ms) as array [state, mesh point]
    ves: np.ndarray       # electrostatic potential of the last iteration
    vxc: np.ndarray       # xc potential of the last iteration
    nint_total: int = 0   # dftseq energy iterations in total (diagnostic)


def sratom(it: int, irl: int, igr: int, iexc: int, tcore: bool, z: float,
           nc: int, nv: int, n: Sequence[int], l: Sequence[int],
           f: Sequence[float], e: Sequence[float], mmax: int, r: np.ndarray,
           vi: np.ndarray, dc: np.ndarray | None = None,
           dcp: np.ndarray | None = None, dcpp: np.ndarray | None = None,
           uu: np.ndarray | None = None, rpk: np.ndarray | None = None,
           warn: Callable[[str], None] | None = None,
           trace: Callable[[dict], None] | None = None) -> SratomResult:
    """Self-consistent spherical all-electron atom (``sratom_n.f``).

    Arguments follow the Fortran list
    ``sratom(it,irl,igr,iexc,tcore,z,nc,nv,n,l,f,e,ninu,de,mmax,r,rpk,vi,
    rho,rhop,rhopp,dc,dcp,dcpp,uu,tsic,esic,vsic)``; in/out arrays are not
    modified but returned in the result.

    ``it``     maximum number of iterations (``itmx``)
    ``irl``    1 scalar-relativistic, 2 non-relativistic (the ``dftseq`` mode)
    ``igr``    1 to evaluate the density derivatives
    ``tcore``  True: all states are solved and the core density is
               accumulated; False: frozen core, ``dc, dcp, dcpp`` are input
    ``e``      start eigenvalues, ``vi`` start potential
    ``uu``     previous wave functions (``uu(mx,ms)``, zero in ``fhipp``);
               as in the Fortran only the points ``1..nin`` of a state are
               overwritten, so stale values can survive beyond ``nin``.

    ``warn``   receives messages for the error unit; ``trace`` (debugging
               aid, not in the Fortran) is called after every iteration with
               a dict of the current ``it, e, ninu, rho, rhop, rhopp, ves,
               vxc, vi``.

    The self-interaction branch (``tsic``) is compiled out in the Fortran
    (``parameter(tsic=.false.)``) and is not ported.
    """
    nst = nc + nv
    imx = it
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    amesh = float(r[1]) / float(r[0])
    al = libm.slog(amesh)
    wa = np.zeros(mmax)
    wb = np.ones(mmax)
    e = np.array(e[:nst], dtype=float)
    f = [float(x) for x in f[:nst]]
    vi = np.array(vi[:mmax], dtype=float)

    def _arr(a: np.ndarray | None) -> np.ndarray:
        return np.zeros(mmax) if a is None else np.array(a[:mmax], dtype=float)

    dc, dcp, dcpp = _arr(dc), _arr(dcp), _arr(dcpp)
    uu = np.zeros((nst, mmax)) if uu is None else np.array(uu, dtype=float)[:nst, :mmax]
    rpk = np.zeros(nst) if rpk is None else np.array(rpk[:nst], dtype=float)

    # bl is the mixing factor for the potential in the self-consistency loop
    bl = 0.5e0
    als = al * al
    sf = 0.e0
    for i in range(nst):
        sf = sf + f[i]
    ninu = [1] * nst

    r2 = r * r
    r3 = r * r2
    r4 = r2 * r2
    vi1 = np.zeros(mmax)
    vo1 = np.zeros(mmax)
    eeig_old = 0.e0
    de = 0.e0
    nint_total = 0
    nstart = 0 if tcore else nc
    tconv = False
    rho = rhop = rhopp = ves = vxc = np.zeros(mmax)
    itn = 0
    for itn in range(1, imx + 1):
        # initialize for new iteration
        tconv = True
        ninmx = 0
        rho = np.zeros(mmax)
        rhop = np.zeros(mmax)
        rhopp = np.zeros(mmax)

        for i in range(nstart, nst):
            et = e[i]
            res = dftseq(irl, z, mmax, r, n[i], l[i], 1.e0, vi, wa, wb,
                         e=float(et), warn=warn)
            nin = res.nin
            u, up, upp = res.u, res.up, res.upp
            e[i] = res.e
            nint_total += res.nint
            uu[i, :nin] = u[:nin]
            if e[i] != et:
                tconv = False
            ninu[i] = nin
            if nin > ninmx:
                ninmx = nin

            # accumulate charge & radial derivatives; rhop & rhopp are
            # derivatives w.r.t. r, up & upp w.r.t. the logarithmic variable
            fi = f[i]
            rho += fi * u * u
            if igr == 1:
                w = up / al - u
                rhop += fi * 2.e0 * w * u
                rhopp += fi * 2.e0 * (w * (up / al - 3.e0 * u) + (upp - al * up) * u / als)

            if i == nc - 1:
                # core density (zero beyond ninmx since u vanishes there)
                dc = rho / r2
                dcp = rhop / r3
                dcpp = rhopp / r4
                rho = np.zeros(mmax)
                rhop = np.zeros(mmax)
                rhopp = np.zeros(mmax)

            # find outermost peak of wavefunction (needed for pseudopotentials)
            if tconv or itn >= imx:
                idx = np.nonzero(up[:nin - 1] * up[1:nin] < 0.0e0)[0]
                # Fortran falls through to r(0) (out of bounds) if no peak
                rpk[i] = r[int(idx[-1])] if idx.size else float("nan")

        # make total density rho()
        rho = rho / r2 + dc
        rhop = rhop / r3 + dcp
        rhopp = rhopp / r4 + dcpp

        # global effective potential
        ves, _ = vestat(mmax, sf - z, r, rho, False)
        vxc = vexcor(iexc, mmax, r, rho, rhop, rhopp, False).vpxc
        vo = ves + vxc

        # generate next iteration's potential using anderson method
        vi, vi1, vo1 = anderson(bl, itn, mmax, r, vo1, vo, vi1, vi)

        # total energy
        eeig = 0.e0
        for i in range(nstart, nst):
            eeig = eeig + f[i] * float(e[i])
        with np.errstate(all="ignore"):
            de = float(abs(np.float64(eeig - eeig_old) / np.float64(eeig)))
        eeig_old = eeig
        if trace is not None:
            trace(dict(it=itn, e=e.copy(), ninu=list(ninu), rho=rho, rhop=rhop,
                       rhopp=rhopp, ves=ves, vxc=vxc, vi=vi))

        if tconv:
            break
    else:
        itn = imx + 1

    return SratomResult(
        it=itn, converged=tconv, e=e, ninu=ninu, de=de, rpk=rpk, vi=vi, rho=rho,
        rhop=rhop, rhopp=rhopp, dc=dc, dcp=dcp, dcpp=dcpp, uu=uu, ves=ves,
        vxc=vxc, nint_total=nint_total,
    )
