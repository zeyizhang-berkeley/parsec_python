"""Kinetic-energy convergence of the pseudo wave functions in momentum space.

Ports of ``kinkon.f`` (convergence test of the orbital kinetic energy with
the plane-wave cutoff), ``dfbt_srt.f`` (Fourier-Bessel transform on the
logarithmic mesh) and ``bessel.f`` (spherical Bessel functions times their
argument).  Hartree atomic units; cutoff energies in Rydberg.

Two places of ``kinkon.f`` read variables that were never assigned:

* If the bracket of 1 meV is not reached below the maximum cutoff, the
  cutoff column of the last ``ck`` row is ``nint(fk(kmx+1)**2)`` with an
  element of ``fk`` beyond the mesh that was set up.  In the reference
  executable this element holds a large leftover number and the ``i4``
  field overflows (``****``); the port prints the same.
* If the eigenvalue passed in is exactly zero (state without a bound
  solution) the transform is skipped and ``et, cnorm, kmx, fk, gk, rho``
  are whatever the stack holds.  On later calls these are the values of the
  previous ``kinkon`` call.  On the first call the reference executable
  finds one of two things there:

  - if ``klbyii`` has not run before (no projector channel): zeros, and for
    ``rho`` the array ``up`` left by the last ``dftseq`` call of
    ``ppcheck``, displaced by six mesh points (if ``ppcheck`` made no such
    call, as with ``pswatch -d``: the array ``vi1`` of ``psatom``, displaced
    by 14 points).  The rows printed are then
    ``ck  l  1.0E+00  0 < 0.000000  0.000000E+00`` and a ``cx`` row
    computed from that array;
  - if ``klbyii`` has run: its quadrature tables.  The loop bound ``kmx``
    is then a number of the order 1e9, the loop runs off the end of the
    stack and the program dies with a segmentation fault (with the
    ``pswatch -q`` redirection the whole report is lost with it).

  :class:`KinkonMemory` carries this state; the first situation is
  reproduced, the second raises :class:`FortranCrash`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import libm
from .errors import FortranStop
from .fortran_io import fmt_e, fmt_es, fmt_f, fmt_i
from .radial import _powi, dftseq, dmelm, fmom

__all__ = ["FortranCrash", "KinkonMemory", "KinkonRow", "KinkonResult", "kinkon", "dfbt",
           "bessj"]


class FortranCrash(FortranStop):
    """The Fortran program terminates abnormally at this point (signal).

    Raised where the reference executable is known to run into a
    segmentation fault.  ``messages`` holds the lines written before; note
    that ``pswatch -q`` loses them, because the output is still buffered
    when the program dies.
    """

MXK = 2000                                   # mxk: maximum number of k points
RY2_SINGLE = float(np.float32(27.2116))      # parameter(ry2=27.2116): real*4 literal
C_CONVC = float(np.float32(3.675e-5))        # c_convc=3.675e-5: 1 meV in Hartree


# --------------------------------------------------------------------------
# bessel.f
# --------------------------------------------------------------------------

def _besrec(l: int, x: np.ndarray, sinx: np.ndarray, cosx: np.ndarray) -> np.ndarray:
    """Recursion for spherical Bessel functions (``besrec``), ``l >= 2``."""
    out = np.empty(x.shape)
    up = x > float(l)
    if up.any():
        xu = x[up]
        downj = sinx[up] / xu
        tempj = (downj - cosx[up]) / xu
        upj = tempj
        for m in range(2, l + 1):
            upj = (2.0 * m - 1.0) / xu * tempj - downj
            downj = tempj
            tempj = upj
        out[up] = upj
    down = ~up
    if down.any():
        xd = x[down]
        mstart = 2 * l + int(1.0e1 * math.sqrt(float(l)))
        upj = np.zeros(xd.shape)
        tempj = np.ones(xd.shape)
        value = np.zeros(xd.shape)
        downj = tempj
        for m in range(mstart - 1, -1, -1):
            downj = (2.0 * m + 3.0) / xd * tempj - upj
            if m == l:
                value = downj
            upj = tempj
            tempj = downj
        out[down] = value / downj * sinx[down] / xd
    return out


def bessj(l: int, x) -> np.ndarray:
    """``x*j_l(x)`` for the spherical Bessel function ``j_l`` (``bessel.f``)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        if l == 0:
            return libm.sin(x)
        out = np.empty(x.shape)
        tiny = x < 1.0e-10
        if tiny.any():
            kdiv = 1
            for k in range(1, 2 * l, 2):
                kdiv = kdiv * k
            xt = x[tiny]
            out[tiny] = xt * (_powi(xt, l) / float(kdiv))
        big = ~tiny
        if big.any():
            xb = x[big]
            sinx = libm.sin(xb)
            cosx = libm.cos(xb)
            if l == 1:
                out[big] = sinx / xb - cosx
            elif l == 2:
                out[big] = (3.0e0 / xb / xb - 1.0e0) * sinx - 3.0e0 * cosx / xb
            else:
                out[big] = xb * _besrec(l, xb, sinx, cosx)
        return out


# --------------------------------------------------------------------------
# dfbt_srt.f
# --------------------------------------------------------------------------

def dfbt(mx: int, r: np.ndarray, gp: np.ndarray, lb: int, ekmx: float, delk: float,
         mxe: int = MXK, inverse: bool = False
         ) -> tuple[int, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Fourier-Bessel transform of a function on the logarithmic mesh.

    ``dfbt(mx,r,gp,lb,ekmx,delk,mxe,kmx,fk,gk,func,iou,cnorm)``: ``gp`` is
    the real space function, ``lb`` the angular momentum, ``ekmx`` the
    cutoff (Ry) and ``delk`` the increment of the linear k mesh.  Returns
    ``(kmx, fk, gk, cnorm, gp_back)``: number of k points, the k mesh, the
    transform, the normalization ``2/pi`` and - only with ``inverse`` - the
    back transform that the Fortran routine stores in ``gp`` (``kinkon``
    does not use it).
    """
    r = np.asarray(r[:mx], dtype=float)
    gp = np.asarray(gp[:mx], dtype=float)
    pi4 = 16.0 * (math.pi / 4.0)
    cnorm = 8.0 / pi4

    # cutoff energy in rydberg
    fkmx = math.sqrt(ekmx)
    fk = [0.0]
    kmx = 1
    for k in range(2, mxe + 1):
        fk.append(delk * (k - 1))
        kmx = k
        if fk[-1] >= fkmx:
            break
    fk = np.array(fk)

    # Fourier-Bessel transform; the radial mesh is assumed logarithmic
    al = libm.slog(float(r[1]) / float(r[0]))
    bes = bessj(lb, r[None, :] * fk[:, None])
    func = gp * bes / r
    with np.errstate(all="ignore"):
        fm = func * r * r * r * _powi(r, 0)
        esum = np.cumsum(fm[:, 1:mx - 1:2], axis=1)[:, -1]
        osum = np.cumsum(fm[:, 2:mx - 2:2], axis=1)[:, -1]
        val = al * (4.0 * esum + 2.0 * osum + fm[:, 0] + fm[:, mx - 1]) / 3.0
        gk = cnorm * ((val + 0.5 * fm[:, 0]) / 1.0)

    gp_back = None
    if inverse:
        # inverse transform (trapezoidal rule on the linear k mesh)
        gint = gk[:, None] * bes / r
        acc = 0.5 * (gint[0] + gint[kmx - 1])
        for k in range(1, kmx - 1):
            acc = acc + gint[k]
        gp_back = acc * delk
    return kmx, fk, gk, cnorm, gp_back


# --------------------------------------------------------------------------
# kinkon.f
# --------------------------------------------------------------------------

@dataclass
class KinkonMemory:
    """Local variables of ``kinkon`` as the next call finds them.

    The Fortran routine does not initialize ``et, cnorm, kmx, fk, gk, rho``
    when it is called with a zero eigenvalue; it then works with the
    previous contents of its stack frame (see the module docstring).

    ``rho``       what the frame holds in place of ``rho``; before the first
                  call this is ``up(7:)`` of ``ppcheck`` (use
                  :meth:`from_ppcheck`)
    ``poisoned``  the frame holds leftovers of ``klbyii``: a first call with
                  a zero eigenvalue then crashes
    """

    et: float = 0.0
    cnorm: float = 0.0
    kmx: int = 0
    fk: np.ndarray = field(default_factory=lambda: np.zeros(MXK))
    gk: np.ndarray = field(default_factory=lambda: np.zeros(MXK))
    rho: np.ndarray | None = None
    poisoned: bool = False

    #: displacement between ``rho`` of ``kinkon`` and ``up`` of ``ppcheck``
    #: (``vi1`` of ``psatom``) on the stack of the reference executable, in
    #: mesh points
    UP_SHIFT = 6
    VI1_SHIFT = 14

    @classmethod
    def from_ppcheck(cls, up_last: np.ndarray | None, klbyii_called: bool,
                     vi1: np.ndarray | None = None) -> "KinkonMemory":
        """State of the frame when ``pslp`` calls ``kinkon`` for the first time.

        ``up_last`` is ``up`` of the last ``dftseq`` call of ``ppcheck``
        (``None``: no call), ``vi1`` the array of that name in ``psatom``.
        """
        rho = None
        left, shift = (up_last, cls.UP_SHIFT) if up_last is not None else (vi1, cls.VI1_SHIFT)
        if left is not None:
            left = np.asarray(left, dtype=float)
            rho = np.zeros(len(left))
            rho[:max(len(left) - shift, 0)] = left[shift:]
        return cls(rho=rho, poisoned=klbyii_called)


@dataclass
class KinkonRow:
    """One ``ck`` row: bracket, cutoff, norm and kinetic energy."""

    bracket_eV: float         # c_conv*ry2: accuracy bracket (eV)
    cutoff_Ry: int | None     # nint(fk**2); None if the bracket was not met
    norm: float               # norm of the wave function up to the cutoff
    ekin: float               # kinetic energy up to the cutoff (Hartree)
    met: bool = True          # False for the '<' row (cutoff > ekmx)


@dataclass
class KinkonResult:
    """Kinetic-energy convergence of one state."""

    l: int
    n: int                    # principal quantum number used for the state
    e: float                  # eigenvalue passed in (Hartree)
    et: float                 # eigenvalue recalculated by dftseq
    rows: list[KinkonRow]
    x_norm: float             # real space norm (the ``cx`` row)
    x_ekin: float             # real space kinetic energy (Hartree)
    converged: bool           # all four brackets met below ekmx
    lines: list[str]          # the report lines of this state
    fk: np.ndarray | None = None      # k mesh (1/bohr)
    gk: np.ndarray | None = None      # Fourier-Bessel coefficients
    valid: bool = True        # False: zero eigenvalue, rows are leftovers


def _nint(x: float) -> int | None:
    """``nint`` into a default integer; ``None`` if it does not fit."""
    if x != x or abs(x) >= 2147483647.5:
        return None
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _row601(l: int, bracket: float, cutoff: int | None, mark: str | None,
            g_norm: float, g_ekin: float) -> str:
    cut = "****" if cutoff is None else fmt_i(cutoff, 4)
    text = "ck  " + fmt_i(l, 2) + "    " + fmt_es(bracket, 8, 1) + "   " + cut
    text += "  " if mark is None else " " + mark
    return text + fmt_f(g_norm, 9, 6) + "  " + fmt_e(g_ekin, 13, 6)


def kinkon(mmax: int, ekmx: float, g_del: float, n: int, l: int, e: float,
           r: np.ndarray, vps: np.ndarray, memory: KinkonMemory | None = None,
           warn: Callable[[str], None] | None = None) -> KinkonResult:
    """Convergence of the kinetic energy of one state (``kinkon.f``).

    ``kinkon(iu,mmax,ekmx,g_del,n,l,e,r,vps)``: the wave function of the
    state ``n, l`` with eigenvalue ``e`` in the effective potential ``vps``
    is recalculated (non-relativistically, ``dftseq`` mode 2), transformed
    to momentum space on a linear mesh of increment ``g_del`` up to the
    cutoff ``ekmx`` (Ry), and the cutoffs at which its kinetic energy is
    converged to within 1 eV ... 1 meV are tabulated.

    ``memory`` holds the leftovers of earlier calls (see
    :class:`KinkonMemory`) and is updated in place.  The lines the routine
    writes itself (table rows and its message for the error unit, which is
    the same stream in ``pslp``) are returned in ``lines``; ``warn``
    receives the messages of ``dftseq``, which precede them.
    """
    if memory is None:
        memory = KinkonMemory()
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    vps = np.asarray(vps[:mmax], dtype=float)
    al = libm.slog(float(r[1]) / float(r[0]))
    wa = np.zeros(mmax)
    wb = np.ones(mmax)
    lines: list[str] = []
    valid = e != 0.0

    if not valid and memory.poisoned:
        raise FortranCrash("kinkon - segmentation fault: state without bound solution "
                           f"(n l {n} {l}), uninitialized loop bound")

    # transform
    if valid:
        res = dftseq(2, 0.e0, mmax, r, n, l, 1.e0, vps, wa, wb, e=float(e), warn=warn)
        with np.errstate(all="ignore"):
            fr = res.u / r
            rho = fr * fr
        kmx, fk, gk, cnorm, _ = dfbt(mmax, r, fr, l, ekmx, g_del, MXK)
        memory.et = res.e
        memory.cnorm = cnorm
        memory.kmx = kmx
        memory.fk = np.zeros(MXK)
        memory.fk[:kmx] = fk
        memory.gk = np.zeros(MXK)
        memory.gk[:kmx] = gk
        memory.rho = rho
        memory.poisoned = False
    et = memory.et
    cnorm = memory.cnorm
    kmx = memory.kmx
    fk = memory.fk
    gk = memory.gk
    rho = np.zeros(mmax) if memory.rho is None else np.asarray(memory.rho, dtype=float)[:mmax]
    if len(rho) < mmax:
        rho = np.concatenate([rho, np.zeros(mmax - len(rho))])

    with np.errstate(all="ignore"):
        # real space expectation values
        x_norm = fmom(0, mmax, al, 1.e0, r, rho)
        x_epot = dmelm(mmax, al, r, vps, rho)
        x_ekin = et - x_epot

        # kinetic energy convergence
        delk = float(np.float64(g_del) / np.float64(2 * cnorm))
        hdelk = 0.5 * delk
        g_norm = 0.e0
        g_ekin = 0.e0
        # convergence threshold 3.675e-5 a.u. = 1 meV
        c_convc = C_CONVC
        c_conv = c_convc * 1000
        rows: list[KinkonRow] = []
        fkl = fk.tolist()
        gkl = gk.tolist()
        done = False
        for i in range(1, kmx):
            fa, ga = fkl[i], gkl[i]
            fb, gb = fkl[i - 1], gkl[i - 1]
            pa = fa * ga
            pb = fb * gb
            g_ekin = g_ekin * g_norm + hdelk * (pa * pa + pb * pb)
            g_norm = g_norm + delk * (ga * ga + gb * gb)
            g_ekin = float(np.float64(g_ekin) / np.float64(g_norm))
            if abs(g_ekin - x_ekin) <= c_conv:
                cutoff = _nint(fa * fa)
                lines.append(_row601(l, c_conv * RY2_SINGLE, cutoff, None, g_norm, g_ekin))
                rows.append(KinkonRow(c_conv * RY2_SINGLE, cutoff, g_norm, g_ekin, True))
                c_conv = c_conv / 10
            if c_conv < c_convc:
                done = True
                break
    if not done:
        if valid or kmx >= 2:
            # nint(fk(kmx+1)**2) with fk(kmx+1) never set: overflows the field
            cutoff = None
        else:
            # zero-trip loop: i = 2 and fk(2) is zero on the untouched stack
            cutoff = _nint(float(fk[1]) * float(fk[1]))
        lines.append(_row601(l, c_conv * RY2_SINGLE, cutoff, "<", g_norm, g_ekin))
        rows.append(KinkonRow(c_conv * RY2_SINGLE, None, g_norm, g_ekin, False))
        lines.append(" & kinkon - bracket not met for cutoff < ekmx =" + " "
                     + str(_nint(ekmx)).rjust(11) + " " + " Ry")
    lines.append("cx  " + fmt_i(l, 2) + " " + fmt_i(n, 2) + " " * 18 + fmt_f(x_norm, 9, 6)
                 + "  " + fmt_e(x_ekin, 13, 6))
    lines.append("")
    return KinkonResult(l=l, n=n, e=float(e), et=float(et), rows=rows, x_norm=x_norm,
                        x_ekin=x_ekin, converged=done, lines=lines,
                        fk=fk[:kmx].copy() if valid else None,
                        gk=gk[:kmx].copy() if valid else None, valid=valid)
