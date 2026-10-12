"""Self-consistent pseudo-atom.  Port of ``psatom.f``.

The routine solves the valence states in their own (semilocal) ionic
pseudopotentials plus a common screening potential, on the logarithmic
mesh, with ``dftseq`` in mode 8: a state without a bound solution gets the
eigenvalue zero and the program goes on.  In that case ``dftseq`` returns
its work arrays in an unfinished state and ``psatom.f`` uses them (and the
stale mesh index ``nin``) as if they were a wave function.  This is
reproduced: the arrays ``u, up, upp`` and ``nin``, ``mch`` are carried from
one ``dftseq`` call to the next exactly like the Fortran local variables.

Uninitialized local variables of the Fortran routine (``vee``, ``vxc``,
``vo1``, ``vi1``, ``u`` ... on first use) are taken as zero, which is what
the reference executable finds on its fresh stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import libm
from .radial import anderson, dftseq, vestat
from .xc import vexcor

__all__ = ["PsatomResult", "psatom"]


@dataclass
class PsatomResult:
    """Output arguments of ``psatom`` (all arrays are new objects)."""

    it: int               # iterations used; ``itmx + 1`` if not converged
    converged: bool       # tconv at exit
    e: np.ndarray         # e(ms): eigenvalues (Hartree); 0 = no bound state
    mup: int              # outermost mesh point used in the last iteration
    de: float             # relative change of the eigenvalue sum
    vi: np.ndarray        # screening potential after the last mixing step
    rho: np.ndarray       # valence density (times 4 pi)
    rhop: np.ndarray      # d rho/dr
    rhopp: np.ndarray     # d2 rho/dr2
    uu: np.ndarray        # wave functions [state, mesh point]
    vi1: np.ndarray       # screening potential the last iteration started from
    nint_total: int = 0   # dftseq energy iterations in total (diagnostic)


def psatom(it: int, igr: int, iexc: int, z: float, nc: int, nv: int,
           n: Sequence[int], l: Sequence[int], f: Sequence[float],
           e: Sequence[float], mmax: int, r: np.ndarray, vi: np.ndarray,
           vps: np.ndarray, dc: np.ndarray, dcp: np.ndarray, dcpp: np.ndarray,
           uu: np.ndarray | None = None,
           warn: Callable[[str], None] | None = None,
           trace: Callable[[dict], None] | None = None) -> PsatomResult:
    """Self-consistent (pseudo-)atom on the logarithmic mesh (``psatom.f``).

    Arguments follow the Fortran list ``psatom(it,igr,iexc,z,nc,nv,n,l,f,e,
    mup,de,mmax,r,vi,vps,rho,rhop,rhopp,dc,dcp,dcpp,uu)``:

    ``it``     maximum number of iterations
    ``igr``    1 to evaluate the density derivatives (gradient corrections)
    ``z``      nuclear charge passed to ``dftseq`` (0 for a pseudo-atom)
    ``e``      start eigenvalues, ``vi`` start screening potential
    ``vps``    ionic pseudopotential of every state, ``vps[i]`` = ``vps(:,i+1)``
    ``dc``..   (pseudo) core density and its derivatives, added to the
               valence density in the exchange-correlation potential
    ``uu``     previous contents of the wave function array (zero in ``pslp``)

    ``warn`` receives the messages ``dftseq`` writes to the error unit;
    ``trace`` (debugging aid) is called after every iteration.
    """
    nst = nc + nv
    imx = it
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    al = 0.1e0 * libm.slog(float(r[10]) / float(r[0]))
    wa = np.zeros(mmax)
    wb = np.ones(mmax)
    e = np.array(e[:nst], dtype=float)
    f = [float(x) for x in f[:nst]]
    vi = np.asarray(vi[:mmax], dtype=float)
    vps = np.asarray(vps, dtype=float)[:nst, :mmax]
    dc = np.asarray(dc[:mmax], dtype=float)
    dcp = np.asarray(dcp[:mmax], dtype=float)
    dcpp = np.asarray(dcpp[:mmax], dtype=float)
    uu = np.zeros((nst, mmax)) if uu is None else np.array(uu, dtype=float)[:nst, :mmax]

    # initial potential
    vpw = vps + vi

    # bl is the mixing factor for the sc-potential
    blmin = 0.40e0
    blmax = 0.95e0
    als = al * al
    sf = 0.e0
    for i in range(nst):
        sf = sf + f[i]

    r2 = r * r
    r3 = r * r2
    r4 = r2 * r2
    # local variables of the Fortran routine that survive from call to call
    # of dftseq / from iteration to iteration (fresh stack: zero)
    nin = 0
    mch = 0
    u = np.zeros(mmax)
    up = np.zeros(mmax)
    upp = np.zeros(mmax)
    vxc = np.zeros(mmax)
    vee = np.zeros(mmax)
    vi1 = np.zeros(mmax)
    vo1 = np.zeros(mmax)
    eeig_old = 0.e0
    de = 0.e0
    mup = 0
    nint_total = 0
    tconv = False
    rho = np.zeros(mmax)
    rhop = np.zeros(mmax)
    rhopp = np.zeros(mmax)
    itn = 0
    for itn in range(1, imx + 1):
        tconv = True
        mup = 0
        # bl=min(blmax,blmin*sqrt(float(it))): the square root is taken in
        # single precision
        bl = min(blmax, blmin * float(np.sqrt(np.float32(itn))))

        # initialize for new iteration
        rho = np.zeros(mmax)
        rhop = np.zeros(mmax)
        rhopp = np.zeros(mmax)

        # solve for pseudo states, rho() is the valence density
        for i in range(nst):
            et = float(e[i])
            res = dftseq(8, z, mmax, r, n[i], l[i], 1.e0, vpw[i], wa, wb,
                         mch=mch, e=et, warn=warn, nin=nin, u=u, up=up, upp=upp)
            nin, mch = res.nin, res.mch
            u, up, upp = res.u, res.up, res.upp
            nint_total += res.nint
            if e[i] != res.e:
                tconv = False
            e[i] = res.e
            mup = max(mup, nin)

            # accumulate charge & radial derivatives; rhop & rhopp are
            # derivatives w.r.t. r, up & upp w.r.t. the logarithmic variable
            k = max(0, min(nin, mmax))
            fi = f[i]
            uk = u[:k]
            uu[i, :k] = uk
            rho[:k] = rho[:k] + fi * uk * uk
            if igr == 1:
                upk = up[:k]
                w = upk / al - uk
                rhop[:k] = rhop[:k] + fi * 2.e0 * w * uk
                rhopp[:k] = rhopp[:k] + fi * 2.e0 * (
                    w * (upk / al - 3.e0 * uk) + (upp[:k] - al * upk) * uk / als)

        # output change of electronic part of the effective potential
        rho = rho / r2
        ves, _ = vestat(mmax, sf, r, rho, False)

        # make total density rho()
        rho = rho + dc
        rhop = rhop / r3 + dcp
        rhopp = rhopp / r4 + dcpp
        # vexcor is called with mup points only: vxc beyond keeps its
        # previous contents
        if mup > 0:
            vxc[:mup] = vexcor(iexc, mup, r, rho, rhop, rhopp, False).vpxc[:mup]
        vo = ves + vxc

        # updating the potentials
        vee, vi1, vo1 = anderson(bl, itn, mmax, r, vo1, vo, vi1, vee)
        vpw = vps + vee

        # total energy
        eeig = 0.e0
        for i in range(nst):
            eeig = eeig + f[i] * float(e[i])
        with np.errstate(all="ignore"):
            de = float(abs(np.float64(eeig - eeig_old) / np.float64(eeig)))
        eeig_old = eeig
        if trace is not None:
            trace(dict(it=itn, e=e.copy(), nin=nin, mup=mup, rho=rho, rhop=rhop,
                       rhopp=rhopp, ves=ves, vxc=vxc.copy(), vee=vee, uu=uu.copy()))

        if tconv:
            break
    else:
        itn = imx + 1

    # output (screening) electronic potential
    return PsatomResult(
        it=itn, converged=tconv, e=e, mup=mup, de=de, vi=vee,
        rho=rho - dc, rhop=rhop - dcp, rhopp=rhopp - dcpp, uu=uu, vi1=vi1,
        nint_total=nint_total,
    )
