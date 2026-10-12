"""Norm-conserving pseudopotentials: construction and unscreening.

Port of ``ncpp.f``.  For every angular momentum ``l = 0 .. lmax`` the
reference state is solved in the all-electron potential, the screened
pseudopotential is built with the Hamann or the Troullier-Martins scheme,
the pseudo valence density is accumulated, and finally all potentials are
unscreened with the Hartree and exchange-correlation potential of the
pseudo valence density (plus the model core density if the nonlinear core
correction is switched on).

Unit files of the Fortran routine and where they end up here::

    fort.23   monitoring data          -> appended to the list ``out``
    fort.24   pseudocore monitoring    -> NcppResult.fort24 (records)
    fort.25   pseudo valence density   -> NcppResult.rho, rhop, rhopp
    fort.27   model core density       -> NcppResult.core (dc, dcp, dcpp)
    fort.28   unscreening monitor      -> NcppResult.dc, dcp, ves, vxc
    fort.39   pseudo and ae wfcts      -> NcppResult.ups, uu, ninu
    fort.4[0-4]  ionic potentials      -> NcppResult.ups, vps
    fort.4[5-9]  screened potentials   -> NcppResult.vscr

(:meth:`NcppResult.unit_files` renders the texts.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import libm
from .constants import EPSPP, RY2
from .dnlcc import dnlcc7
from .errors import FortranStop
from .fortran_io import fmt_a, fmt_e, fmt_f, fmt_i
from .fortran_list import list_record, unit6_lines
from .hamann import hamann
from .labels import spp
from .outwfct import e20_rows, out39, outcore, outpot
from .radial import _fdiv, dextv, dftseq, dmelm, fmom, vestat
from .tromar import tromar
from .xc import vexcor

__all__ = ["ExcitedState", "NcppChannel", "NcppResult", "ncpp", "MODMAP"]

# modmap: dftseq mode of the all-electron solution -> mode for the pseudopotential
MODMAP: tuple[int, ...] = (2, 2, 4, 4, 5, 7, 7)

# single precision constants of the source
_EPS_VALENCY: float = float(np.float32(1e-12))
_PI4_28: float = 12.56637096405029296875     # pi4=16*atan(1.0), evaluated in single precision
_C016: float = float(np.float32(0.16))
_C133: float = float(np.float32(1.33))

_INT_MIN = -2147483648


def _slog(x: float) -> float:
    if x > 0.0:
        return libm.slog(x)
    if x == 0.0:
        return -math.inf
    return math.nan


def _fint(x: float) -> int:
    """Fortran ``int()`` of a double as the x86 instruction cvttsd2si does it."""
    if x != x or x >= 2147483648.0 or x <= -2147483649.0:
        return _INT_MIN
    return int(x)


def _wrap(i: int) -> int:
    """32-bit wrap-around of integer arithmetic."""
    return (i + 2147483648) % 4294967296 - 2147483648


@dataclass(frozen=True)
class ExcitedState:
    """An occupied valence state that is not the reference state of its ``l``."""

    n: int              # principal quantum number of the all-electron state
    f: float            # occupation
    e_ae: float         # all-electron eigenvalue (Hartree) as read from the atom
    e_ps: float         # eigenvalue of the 1st excited pseudo state (Hartree)

    @property
    def e_ae_ev(self) -> float:
        return self.e_ae * RY2

    @property
    def e_ps_ev(self) -> float:
        return self.e_ps * RY2


@dataclass(frozen=True)
class NcppChannel:
    """One row of the "pseudo atom" table of unit 23."""

    l: int               # angular momentum
    i_pp_type: int       # 1 Hamann, 2 Troullier-Martins
    n: int               # np: principal quantum number of the reference state
    f: float             # occupation of the reference state
    rc_input: float      # cutoff radius requested (default or input)
    nrc: int             # grid index next to the cutoff radius (Fortran index)
    rc: float            # r(nrc): cutoff radius actually used
    mch: int             # mchf: grid index of the matching radius
    rmatch: float        # r(mchf)
    e_ae: float          # all-electron reference energy / eigenvalue (Hartree)
    e_ps: float          # eigenvalue of the pseudo state (Hartree)
    norm: float          # gam: norm test
    slope: float         # gpr: slope test
    ninu: int            # outermost point of the all-electron wave function
    excited: tuple[ExcitedState, ...] = ()

    @property
    def scheme(self) -> str:
        """``'h'`` (Hamann) or ``'t'`` (Troullier-Martins)."""
        return spp(self.i_pp_type)[0]

    @property
    def e_ae_ev(self) -> float:
        """All-electron eigenvalue in eV as printed (``e*ry2``)."""
        return self.e_ae * RY2

    @property
    def e_ps_ev(self) -> float:
        """Pseudo eigenvalue in eV as printed (``etest*ry2``)."""
        return self.e_ps * RY2


@dataclass
class NcppResult:
    """Everything ``ncpp`` computes.  Arrays are float64 of length ``mmax``;
    per-channel arrays have the shape ``(lmax+1, mmax)``."""

    lmax: int
    mmax: int
    r: np.ndarray
    zion: float                 # zin - zc: charge of the ionic pseudopotential
    zv: float                   # number of valence electrons of the atom
    cval: float                 # number of valence electrons of the pseudo atom
    channels: list[NcppChannel]
    ups: np.ndarray             # pseudo wave functions
    vps: np.ndarray             # ionic pseudopotentials (screened if cval <= 0)
    vscr: np.ndarray            # screened pseudopotentials
    uu: np.ndarray              # all-electron wave functions of the reference states
    rho: np.ndarray             # pseudo valence density
    rhop: np.ndarray
    rhopp: np.ndarray
    dc: np.ndarray              # density used for the xc unscreening (dc of the Fortran on return)
    dcp: np.ndarray
    dcpp: np.ndarray
    core: tuple[np.ndarray, np.ndarray, np.ndarray] | None   # unit 27: model core density
    ves: np.ndarray             # Hartree potential of the pseudo valence density
    vxc: np.ndarray             # xc potential of ``dc``
    rnlc: float                 # pseudocore radius (the mesh point used if > r(1))
    iequidensity: int           # grid index of the core-valence equidensity radius (0: n/a)
    etot: float                 # total energy of the pseudo atom (Hartree)
    ekin: float
    evps: float                 # "potential energy"
    eeel: float                 # 2 * Hartree energy
    ex: float
    ec: float
    eeig: float
    tc: float                   # integrated valence density
    t1: float
    t2: float
    xvgr: tuple[int, int, int] = (0, 0, 1)
    fort24: list[str] | None = None        # records of unit 24 (dnlcc7), None if not written
    unit6: list[str] = field(default_factory=list)

    @property
    def amesh(self) -> float:
        return float(self.r[1]) / float(self.r[0])

    # --- unit files -------------------------------------------------------
    def fort25(self) -> str:
        """Pseudo valence density, unit 25."""
        return outcore(self.mmax, 1, self.r, self.rho, self.rhop, self.rhopp)

    def fort27(self) -> str | None:
        """Model core density, unit 27 (``None`` without core correction)."""
        if self.core is None:
            return None
        return "".join(line + "\n" for line in e20_rows(self.r, *self.core))

    def fort28(self) -> str:
        """Monitoring of the unscreening, unit 28."""
        lines = []
        for i in range(0, self.mmax, max(self.mmax // 100, 1)):
            d = float(self.dc[i])
            if d > 0.e0:
                with np.errstate(all="ignore"):
                    sgrad = float(_C016 * abs(float(self.dcp[i]) / _PI4_28)
                                  / np.float64(libm.spow(d / _PI4_28, _C133)))
            else:
                sgrad = 0.e0
            # format 5(e12.6,1x): the trailing 1x writes nothing
            lines.append(" ".join(fmt_e(x, 12, 6) for x in (
                float(self.r[i]), d, sgrad, float(self.ves[i]), float(self.vxc[i]))))
        return "".join(line + "\n" for line in lines)

    def fort39(self) -> str:
        """Pseudo and all-electron valence wave functions, unit 39."""
        ch = self.channels
        return out39(self.lmax, [c.ninu for c in ch], [c.n for c in ch], [c.rc for c in ch],
                     self.r, self.ups, self.uu)

    def fort4x(self) -> tuple[list[str], list[str]]:
        """Texts of the units ``40+l`` (ionic) and ``45+l`` (screened)."""
        return outpot(self.mmax, self.lmax + 1, self.zion, [c.rc for c in self.channels],
                      self.r, self.ups, self.vps, self.vscr)

    def unit_files(self) -> dict[str, str]:
        """All unit files written by ``ncpp`` as ``{'fort.NN': text}``."""
        files: dict[str, str] = {}
        if self.fort24 is not None:
            files["fort.24"] = "".join(line + "\n" for line in self.fort24)
        files["fort.25"] = self.fort25()
        core = self.fort27()
        if core is not None:
            files["fort.27"] = core
        files["fort.28"] = self.fort28()
        files["fort.39"] = self.fort39()
        ionic, screened = self.fort4x()
        for l, text in enumerate(ionic):
            files[f"fort.{40 + l}"] = text
        for l, text in enumerate(screened):
            files[f"fort.{45 + l}"] = text
        return files


def _line608(label: str, value: float) -> str:
    return fmt_a(label, 30) + "  " + fmt_f(value, 16, 5)


@np.errstate(all="ignore")
def ncpp(irl: int, iexc: int, zin: float, zc: float, zv: float, lmax: int,
         i_pp_type: Sequence[int], np_: Sequence[int], e: Sequence[float],
         f: Sequence[float], rc: Sequence[float], v: np.ndarray, mmax: int,
         r: np.ndarray, rnlc: float, dc: np.ndarray, dcp: np.ndarray,
         dcpp: np.ndarray, rho: np.ndarray, rhop: np.ndarray, rhopp: np.ndarray,
         nval: int, nae: Sequence[int], lae: Sequence[int], eae: Sequence[float],
         fae: Sequence[float], out: list[str] | None = None,
         warn: Callable[[str], None] | None = None) -> NcppResult:
    """Construct and unscreen the pseudopotentials (``ncpp.f``).

    Arguments follow ``ncpp(iu,irl,iexc,zin,zc,zv,lmax,i_pp_type,np,e,f,rc,v,
    mmax,r,rnlc,dc,dcp,dcpp,rho,rhop,rhopp,tsic,vorb,ups,nval,nae,lae,eae,
    fae)`` without the unit number and the self-interaction arguments:

    ``irl``        relativity mode of the all-electron atom (1 or 2)
    ``zin``        nuclear charge, ``zc``/``zv`` core/valence electrons
    ``i_pp_type``  scheme per channel (1 Hamann, 2 Troullier-Martins)
    ``np_, e, f, rc``  reference state per channel (index ``l``): principal
                   quantum number, energy, occupation, cutoff radius
    ``v``          all-electron effective potential
    ``rnlc``       pseudocore radius (0: linearized core-valence xc)
    ``dc, dcp, dcpp``     core density and derivatives
    ``rho, rhop, rhopp``  valence density and derivatives
    ``nval, nae, lae, eae, fae``  the valence states of the atom
    ``out``        list receiving the records of unit 23 (kept up to date
                   even if a stop is raised)
    ``warn``       receives the lines written to the error unit

    A stop during the construction of a channel raises :class:`FortranStop`
    with the attribute ``channel`` set to its angular momentum.
    """
    out = [] if out is None else out
    unit6: list[str] = []

    def say(text: str) -> None:
        for line in unit6_lines(text):
            unit6.append(line)
            if warn:
                warn(line)

    r = np.ascontiguousarray(r[:mmax], dtype=float)
    v = np.asarray(v, dtype=float)
    vfull = np.ascontiguousarray(v[:mmax])
    dc = np.array(dc[:mmax], dtype=float)
    dcp = np.array(dcp[:mmax], dtype=float)
    dcpp = np.array(dcpp[:mmax], dtype=float)
    rho = np.array(rho[:mmax], dtype=float)
    rhop = np.array(rhop[:mmax], dtype=float)
    rhopp = np.array(rhopp[:mmax], dtype=float)
    e = [float(x) for x in e]
    f = [float(x) for x in f]
    rc = [float(x) for x in rc]
    eae = [float(x) for x in eae]
    fae = [float(x) for x in fae]
    zin, zc, zv, rnlc = float(zin), float(zc), float(zv), float(rnlc)
    nch = lmax + 1
    r1 = float(r[0])

    al = 0.1e0 * libm.slog(float(r[10]) / r1)
    als = al * al

    wm = 1.e0
    wa = np.zeros(mmax)
    wb = np.ones(mmax)

    # save full potential
    vin = vfull.copy()

    # unscreen full potential (only used by the disabled global tail cutoff
    # and, if the pseudo atom has no electrons, by the unit 28 output)
    ves, _ = vestat(mmax, zv, r, rho, False)
    if rnlc > 0.0:
        rho = rho + dc
        rhop = rhop + dcp
        rhopp = rhopp + dcpp
    vxc = vexcor(iexc, mmax, r, rho, rhop, rhopp, False).vpxc
    # rcut_global = r(mmax) + 1.0: the Coulomb tail cutoff is never applied

    # null eigenvalue energy accumulator
    eeig = 0.0e0
    evps = 0.0e0

    # null charge accumulator & derivatives
    cval = 0.e0
    rho = np.zeros(mmax)
    rhop = np.zeros(mmax)
    rhopp = np.zeros(mmax)

    out.append("")
    out.append("          === pseudo atom ===")
    out.append("")
    out.append("  l  type  rcore       rmatch          eigenvalue(eV)      norm test   slope test")
    out.append("                                 all-electron     pseudo")

    r2 = r * r
    r3 = r * r2
    r4 = r2 * r2
    ups = np.zeros((nch, mmax))
    vps = np.zeros((nch, mmax))
    vscr = np.zeros((nch, mmax))
    uu = np.zeros((nch, mmax))
    drh = np.zeros((nch, mmax))
    channels: list[NcppChannel] = []
    mch = 0
    uldf = 0.e0

    # loop to construct pseudopotential for each state
    for l1 in range(1, nch + 1):
        l = l1 - 1
        k = l1 - 1
        itype = i_pp_type[k]
        try:
            etest = e[k]
            rc_input = rc[k]
            nrc = _fint(_slog(_fdiv(rc[k], r1)) / al + 1)
            if f[k] != 0.0e0:
                mode = irl
            else:
                mode = 3 * irl
                if itype == 1:
                    mch = _fint(_slog(2.5 * rc[k] / r1) / al)
                if itype == 2:
                    mch = nrc + 1
                if mch < 5 or mch > mmax:
                    # the Fortran would index its arrays out of bounds
                    raise FortranStop("ncpp - stop: matching radius outside the radial mesh")

            # full potential solution
            res = dftseq(mode, zin, mmax, r, np_[k], l, wm, v, wa, wb, mch=mch, uld=uldf,
                         e=e[k], warn=say)
            ninu = res.nin
            mch = res.mch
            e[k] = res.e
            uu[k] = res.u
            up, upp = res.up, res.upp

            if itype == 1:
                mchf = mch
            elif itype == 2:
                mchf = nrc + 1
            else:
                raise FortranStop("ncpp - pseudopotential scheme not implemented")
            if mchf < 1 or mchf > mmax:
                raise FortranStop("ncpp - stop: matching radius outside the radial mesh")
            upmch = float(up[mchf - 1])
            umch = float(uu[k, mchf - 1])
            uldf = _fdiv(upmch, umch)

            mode = MODMAP[mode - 1]

            # call potential subroutines
            if itype == 1:
                hres = hamann(0.e0, EPSPP, l, rc[k], e[k], mode, mchf, mmax, r, uu[k], up, upp,
                              vfull, warn=say)
                u, up, upp, vpl = hres.u, hres.up, hres.upp, hres.v
            else:
                tres = tromar(l, rc[k], e[k], mmax, r, uu[k], up, upp, v, warn=say)
                nrc = tres.nrc
                u, up, upp, vpl = tres.u, tres.up, tres.upp, tres.vps
            vps[k] = vpl

            # accumulate charge, derivatives & eigenvalues
            cval = cval + f[k]
            if f[k] > 0.0:
                n_ = ninu
                un, upn, uppn = u[:n_], up[:n_], upp[:n_]
                t = un / r[:n_]
                drh[k, :n_] = t * t
                rho[:n_] = rho[:n_] + f[k] * drh[k, :n_]
                w = upn / al - un
                drhp = 2.e0 * w * un / r3[:n_]
                drhpp = 2.e0 * (w * (upn / al - 3.e0 * un) + (uppn - al * upn) * un / als) / r4[:n_]
                rhop[:n_] = rhop[:n_] + f[k] * drhp
                rhopp[:n_] = rhopp[:n_] + f[k] * drhpp

            mch = mchf
            res = dftseq(mode, 0.e0, mmax, r, l + 1, l, wm, vpl, wa, wb, mch=mch, uld=uldf,
                         e=etest, warn=say)
            mch = res.mch
            etest = res.e
            u, up, upp = res.u, res.up, res.upp
            ups[k] = u
            eeig = eeig + f[k] * e[k]
            evps = evps + f[k] * dmelm(mmax, al, r, vpl, drh[k], say)

            # diagnostic output
            gam = abs(_fdiv(umch, float(u[mchf - 1])))
            gpr = abs(_fdiv(upmch, float(up[mchf - 1])))
            rc[k] = float(r[nrc - 1])
            norm_test, slope_test = gam, gpr

            out.append(fmt_i(l, 3) + "  " + spp(itype)[0] + "  "
                       + "".join(fmt_f(x, 12, 7) for x in (
                           rc[k], float(r[mchf - 1]), e[k] * RY2, etest * RY2, gam, gpr)))
            e_ps = etest

            # excess occupied states
            excited: list[ExcitedState] = []
            for i in range(nval):
                if l == lae[i] and np_[k] != nae[i]:
                    cval = cval + fae[i]
                    etest = eae[i]
                    res = dftseq(2, 0.e0, mmax, r, l + 2, l, wm, vpl, wa, wb, mch=mch,
                                 uld=uldf, e=etest, warn=say)
                    mch = res.mch
                    etest = res.e
                    gam = etest
                    # for consistent unscreening: augment density w/ full potential wavefunction
                    res = dftseq(irl, 0.e0, mmax, r, nae[i], l, wm, v, wa, wb, mch=mch,
                                 uld=uldf, e=gam, warn=say)
                    nin = res.nin
                    mch = res.mch
                    gam = res.e
                    un, upn, uppn = res.u[:nin], res.up[:nin], res.upp[:nin]
                    w = upn / al - un
                    rhopp[:nin] = rhopp[:nin] + fae[i] * 2.e0 * (
                        w * (upn / al - 3.e0 * un) + (uppn - al * upn) * un / als) / r4[:nin]
                    rhop[:nin] = rhop[:nin] + fae[i] * 2.e0 * w * un / r3[:nin]
                    usq = res.u.copy()
                    t = un / r[:nin]
                    usq[:nin] = t * t
                    rho[:nin] = rho[:nin] + fae[i] * usq[:nin]

                    eeig = eeig + fae[i] * etest
                    evps = evps + fae[i] * dmelm(mmax, al, r, vpl, usq, say)

                    out.append(fmt_i(l, 3) + "  " + spp(itype) + " " * 26
                               + fmt_f(eae[i] * RY2, 12, 7) + fmt_f(etest * RY2, 12, 7)
                               + "   " + "(excited state!)")
                    excited.append(ExcitedState(nae[i], fae[i], eae[i], etest))
        except FortranStop as exc:
            exc.channel = l
            exc.messages = list(unit6)
            raise
        except (ArithmeticError, ValueError, IndexError) as exc:
            # invalid numbers the Fortran would have carried on with (or
            # crashed on); reported as a stop of this channel
            stop = FortranStop("ncpp - stop: numerical breakdown in channel construction "
                               f"({type(exc).__name__}: {exc})")
            stop.channel = l
            stop.messages = list(unit6)
            raise stop from exc

        channels.append(NcppChannel(
            l=l, i_pp_type=itype, n=np_[k], f=f[k], rc_input=rc_input, nrc=nrc, rc=rc[k],
            mch=mchf, rmatch=float(r[mchf - 1]), e_ae=e[k], e_ps=e_ps, norm=norm_test,
            slope=slope_test, ninu=ninu, excited=tuple(excited)))

    # pseudopotentials done

    # test: 1 = Integral(rho)/charge
    tc = fmom(0, mmax, al, 1.e0, r, rho, say)
    # test: 1 = -Integral(r*rho')/(3*charge)
    t1 = -fmom(1, mmax, al, cval, r, rhop, say) / 3.e0
    # test: 1 = Integral(r**2 * rho'')/(12*charge)
    t2 = fmom(2, mmax, al, cval, r, rhopp, say) / 12.e0

    if abs(cval - zv) > _EPS_VALENCY:
        say(list_record("& ncpp -- warning: valency mismatch?", zv, cval))

    # potential energy (before unscreening)
    ekin = eeig - evps

    # unscreen pseudopotentials
    eeel = 0.e0     # (not assigned in the Fortran if cval <= 0)
    ex = 0.e0
    ec = 0.e0
    core = None
    fort24 = None
    iequidensity = 0
    if cval > 0.0:
        ves, eeel = vestat(mmax, cval, r, rho, True)
        if rnlc > 0.0:
            # nonlinear core correction
            above = np.nonzero(dc - rho * 0.0 > rho)[0]
            iequidensity = (int(above[-1]) + 1 if above.size else 0) + 1
            if rnlc > r1:
                core_res = dnlcc7(mmax, rnlc, r, rho, dc, dcp, dcpp, warn=say)
                rnlc = core_res.rnlc
                dc, dcp, dcpp = core_res.dc, core_res.dcp, core_res.dcpp
                fort24 = core_res.fort24
            # output pseudo core density
            core = (dc.copy(), dcp.copy(), dcpp.copy())
            dc = dc + rho
            dcp = dcp + rhop
            dcpp = dcpp + rhopp
        else:
            dc = rho.copy()
            dcp = rhop.copy()
            dcpp = rhopp.copy()
        # (iexc 7: vexcor uses the incoming, unset ec as correlation energy density)
        xc = vexcor(iexc, mmax, r, dc, dcp, dcpp, True, ec_in=0.e0)
        vxc, ex, ec = xc.vpxc, xc.ex, xc.ec

        vin = ves + vxc
        vscr = vps.copy()
        vps = vps - vin

    # external potential energy (i.e. unscreened pseudopotentials)
    evps = evps - dmelm(mmax, al, r, vin, rho, say)
    etot = ekin + .5 * eeel + evps + ex + ec

    # pseudoatom output
    out.append("")
    if rnlc > 0.:
        out.append(_line608("-- nlcv XC cutoff radius", rnlc))
        req = float(r[iequidensity - 1]) if 1 <= iequidensity <= mmax else 0.e0
        out.append(_line608("cv equidensity radius", req))
        out.append(_line608("integrated model core density",
                            fmom(0, mmax, al, 1.e0, r, dc, say) - tc))
        out.append(_line608("true core charge", zc))

    out.append("")
    out.append(" " * 34 + "(Hartree a.u.)")
    out.append(_line608("total energy", etot))
    out.append(_line608("kinetic energy", ekin))
    out.append(_line608("potential energy", evps))
    out.append(_line608("hartree energy", 0.5 * eeel))
    out.append(_line608("xc energy", ex + ec))
    out.append(_line608("integrated valence density", tc))
    out.append(_line608("... 1st derivative test 1 =", t1))
    out.append(_line608("... 2nd derivative test 1 =", t2))

    # for xvgr x y
    vl = 1.e20
    vh = -1.e20
    for k in range(nch):
        bl, bh = dextv(1, mmax, vps[k])
        # min(bl,vl), max(bh,vh) as the reference build evaluates them with NaNs
        vl = vl if vl < bl else bl
        vh = vh if vh > bh else bh
    il = _wrap(_fint(vl / 5) * 5)
    ih = _wrap(_fint(vh / 5) * 5)
    i = max(-100, _wrap(_fint(vl - il) - 1 + il))
    j = min(50, _wrap(_fint(vh - ih) + 1 + ih))
    d = _wrap(j - i)
    step = max(1, d // 4 if d >= 0 else -((-d) // 4))
    out.append(fmt_a("y range xvgr", 30) + " " * 6 + fmt_i(i, 4) + fmt_i(j, 4) + fmt_i(step, 4))

    return NcppResult(
        lmax=lmax, mmax=mmax, r=r, zion=zin - zc, zv=zv, cval=cval, channels=channels,
        ups=ups, vps=vps, vscr=vscr, uu=uu, rho=rho, rhop=rhop, rhopp=rhopp,
        dc=dc, dcp=dcp, dcpp=dcpp, core=core, ves=ves, vxc=vxc, rnlc=rnlc,
        iequidensity=iequidensity, etot=etot, ekin=ekin, evps=evps, eeel=eeel, ex=ex, ec=ec,
        eeig=eeig, tc=tc, t1=t1, t2=t2, xvgr=(i, j, step), fort24=fort24, unit6=unit6)
