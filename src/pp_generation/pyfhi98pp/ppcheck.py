"""Ghost-state analysis and logarithmic derivatives.

Ports of ``ppcheck.f`` (spectrum of the Kleinman-Bylander form, ghost-state
criteria of X. Gonze, R. Stumpf, M. Scheffler, Phys. Rev. B 44, 8503 (1991),
logarithmic derivatives of the all-electron, semilocal and separable
potentials), ``derlkb.f`` (radial integration with a separable potential)
and ``rcovalent.f`` (default diagnostic radii).  Hartree atomic units.

Uninitialized variables of ``ppcheck.f``: the nuclear charge ``z`` passed to
``dftseq`` for the all-electron logarithmic derivatives and the step
``deltaold`` are never assigned.  In the reference executable both stack
slots hold zero (they lie in a part of the stack that no earlier routine
writes), so the relativistic exponent of the series start is that of
``z = 0``; this is reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import libm
from .errors import FortranStop
from .fortran_io import fmt_a, fmt_f, fmt_i
from .klbyii import LaguerreBasis, klbyii
from .labels import labels
from .radial import C24, _powi, dftseq, dmelm

__all__ = ["KbChannel", "LogDerivative", "PpcheckResult", "ppcheck", "derlkb",
           "rcovalent", "energy_mesh", "RY2"]

RY2 = 27.2116                       # parameter (ry2=27.2116d0) of ppcheck.f
MXE = 1000

VERDICTS = ("no ghost", "one or more ghosts", "undetermined", "illdefined")

# rcovalent.f: data rdv/.../ are single precision literals
_RDV = tuple(float(np.float32(float(s))) for s in """
 0.60 1.76
 2.33 1.70 1.55 1.46 1.42 1.38 1.36 1.34
 2.91 2.57 2.23 2.10 2.00 1.93 1.87 1.85
 3.84 3.29
 2.72 2.50 2.31 2.23 2.21 2.21 2.19 2.17 2.21 2.36
           2.38 2.31 2.27 2.19 2.16 2.12
 4.08 3.61
 3.06 2.74 2.53 2.46 2.40 2.36 2.36 2.42 2.53 2.80
           2.72 2.67 2.65 2.57 2.51 2.48
 4.44 3.74
 3.19 3.12 3.12 3.10 3.08 3.06 3.50 3.04 3.01 3.01 2.99
 2.97 2.95 2.95 2.95
 2.72 2.53 2.46 2.42 2.38 2.40 2.46 2.53 2.82
           2.80 2.78 2.76 2.76 2.74 2.71
 4.82 3.88
 3.31 3.12 3.00 3.00 3.00 3.00 3.00 3.00 3.00 3.00 3.00 3.00""".split())

# what the reference executable finds behind the 100 elements of rdv when
# the nuclear charge exceeds 100: the constants stored after the table (two
# integers that read as a denormal number, then zeros).  The diagnostic
# radius is then practically zero.
_RDV_BEYOND: dict[int, float] = {101: 17179869184001 * 5e-324, 102: 0.0, 103: 0.0}


def rcovalent(i: int) -> float:
    """Default (covalent) radius of element ``i`` for the logarithmic
    derivatives (``rcovalent.f``); the table ends at ``i = 100``."""
    if 1 <= i <= len(_RDV):
        return _RDV[i - 1]
    if i in _RDV_BEYOND:
        return _RDV_BEYOND[i]
    raise FortranStop(f"rcovalent - no radius tabulated for element {i}")


# --------------------------------------------------------------------------
# derlkb.f
# --------------------------------------------------------------------------

def _simpson_cols(f1: np.ndarray, s1: np.ndarray, s2: np.ndarray, fn: np.ndarray,
                  h: float) -> np.ndarray:
    """``dsimpson``: ``(4*s1 + 2*s2 + f(1) + f(n))*h/3`` with the partial
    sums ``s1`` (even points) and ``s2`` (odd inner points) of ``dssum``."""
    return (4. * s1 + 2. * s2 + f1 + fn) * h / 3.


def derlkb(l: int, e, ich: int, dkb: float, mmax: int, r: np.ndarray,
           vkb: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Logarithmic derivative for a separable (Kleinman-Bylander) potential.

    ``derlkb(l,e,ich,dlkb,dkb,dedl,nodes,mmax,r,vkb,v)`` of ``derlkb.f``
    (original version by Gonze and Stumpf): ``l`` angular momentum, ``e``
    energy, ``ich`` mesh index of the diagnostic radius (odd), ``dkb`` the
    average nonlocal potential, ``vkb = u_l (v_l - v_loc)`` the projector
    and ``v`` the screened local potential.  Returns ``dlkb``, ``r`` times
    the logarithmic derivative at ``r(ich)``.

    ``e`` may be an array: the radial integration is then carried out for
    all energies at once (each operation element-wise, in the order of the
    Fortran statements, so every energy gets exactly the scalar result).
    """
    scalar = np.ndim(e) == 0
    e = np.atleast_1d(np.asarray(e, dtype=float))
    r = np.asarray(r[:mmax], dtype=float)
    vkb = np.asarray(vkb[:mmax], dtype=float)
    v = np.asarray(v[:mmax], dtype=float)
    nm12 = (ich - 1) // 2
    if nm12 * 2 != ich - 1:
        raise FortranStop("simpson - stop: wrong n")
    if ich < 5:
        raise FortranStop("derlkb - fewer than five mesh points up to the diagnostic radius")
    rl = r.tolist()
    vkbl = vkb.tolist()
    vl = v.tolist()
    c = C24

    sls = float(l * (l + 1))
    al = 0.1e0 * libm.slog(rl[10] / rl[0])
    als = al * al
    alsls = als * sls
    als2 = 2.0 * als

    with np.errstate(all="ignore"):
        # coefficient array for u in the differential equation
        def cf(i: int) -> np.ndarray:
            return alsls + als2 * (vl[i] - e) * (rl[i] * rl[i])

        # start wavefunction with series
        ups0 = (vl[1] * rl[1] - vl[0] * rl[0]) / (rl[1] - rl[0])
        zcorr = -(vl[0] - ups0) * rl[0]
        c2 = -zcorr / float(l + 1)
        c3 = zcorr * zcorr / float((2 * l + 3) * (l + 1)) + (ups0 - e) / float(2 * l + 3)
        p1 = [float(_powi(rl[i], l + 1)) for i in range(4)]
        p2 = [float(_powi(rl[i], l + 2)) for i in range(4)]
        p3 = [float(_powi(rl[i], l + 3)) for i in range(4)]

        def integrate(start_u, start_up, inhom) -> tuple:
            """Outward integration, predictor once and corrector twice.

            Returns ``(u(ich), up(ich), simpson(u*vkb*r))``.
            """
            us = [None] * 4
            ps = [None] * 4
            qs = [None] * 4
            for i in range(4):
                us[i] = start_u(i)
                ps[i] = start_up(i)
                qs[i] = al * ps[i] + cf(i) * us[i]
                if inhom is not None:
                    qs[i] = qs[i] + inhom(i)
            # Simpson sums of u*vkb*r: f(1), even points, odd inner points
            fw = [us[i] * vkbl[i] * rl[i] for i in range(4)]
            f1 = fw[0]
            s1 = fw[1]                    # f(2)
            s2 = fw[2]                    # f(3)
            s1 = s1 + fw[3]               # f(4)
            u0 = us[3]
            p0, pm1, pm2, pm3 = ps[3], ps[2], ps[1], ps[0]
            q0, qm1, qm2, qm3 = qs[3], qs[2], qs[1], qs[0]
            fn = fw[3]
            for i in range(4, ich):       # Python index of the new point
                cfi = cf(i)
                un = u0 + c * (55.0 * p0 - 59.0 * pm1 + 37.0 * pm2 - 9.0 * pm3)
                pn = p0 + c * (55.0 * q0 - 59.0 * qm1 + 37.0 * qm2 - 9.0 * qm3)
                q19 = 19.0 * q0
                q5 = 5.0 * qm1
                p19 = 19.0 * p0
                p5 = 5.0 * pm1
                gi = None if inhom is None else inhom(i)
                for _ in range(2):
                    qn = al * pn + cfi * un
                    if gi is not None:
                        qn = qn + gi
                    pn = p0 + c * (9.0 * qn + q19 - q5 + qm2)
                    un = u0 + c * (9.0 * pn + p19 - p5 + pm2)
                u0 = un
                pm3, pm2, pm1, p0 = pm2, pm1, p0, pn
                qm3, qm2, qm1, q0 = qm2, qm1, q0, qn
                fn = un * vkbl[i] * rl[i]
                if i < ich - 1:
                    if i % 2 == 1:
                        s1 = s1 + fn      # even Fortran index
                    else:
                        s2 = s2 + fn      # odd Fortran index
            anorm = _simpson_cols(f1, s1, s2, fn, al)
            return u0, p0, anorm

        # first function: homogeneous solution, normalized
        u_ich, up_ich, anorm = integrate(
            lambda i: p1[i] + c2 * p2[i] + c3 * p3[i],
            lambda i: al * ((l + 1) * p1[i] + c2 * (l + 2) * p2[i] + c3 * (l + 3) * p3[i]),
            None)
        wkb = u_ich * dkb / anorm
        wkbp = up_ich * dkb / anorm

        # the second function: coefficients of the inhomogeneous term
        a0 = vkbl[0] / float(_powi(rl[0], l + 1))

        def cfkb(i: int) -> np.ndarray:
            return 2 * (al * al) * (rl[i] * rl[i]) * dkb / anorm / a0 * vkbl[i]

        c3 = (ups0 - e + zcorr * zcorr / float(l + 1) + 1) / float(2 * l + 3)
        u_ich, up_ich, anorm1 = integrate(
            lambda i: (p1[i] + c2 * p2[i] + c3 * p3[i]) * dkb / anorm,
            lambda i: al * ((l + 1) * p1[i] + c2 * (l + 2) * p2[i]
                            + c3 * (l + 3) * p3[i]) * dkb / anorm,
            cfkb)

        # subtraction of function w to give Vv1=0
        v1kb = u_ich - wkb / dkb * anorm1
        v1kb = v1kb * a0 / dkb * anorm
        v1kbp = (up_ich - wkbp / dkb * anorm1) * a0 / dkb * anorm

        # solution
        uout = v1kb + wkb
        upout = v1kbp + wkbp
        dlkb = upout / al / uout - 1.0e0
    return float(dlkb[0]) if scalar else dlkb


# --------------------------------------------------------------------------
# ppcheck.f
# --------------------------------------------------------------------------

@dataclass
class KbChannel:
    """Ghost-state analysis of one projector channel (``ppcheck.f``).

    Energies are in Hartree (``*_eV`` properties: times 27.2116 as in the
    report).  ``semilocal`` and ``nonlocal_`` are the rows of the table
    "spectrum of bound states" exactly as printed, in eV.  The Fortran does
    not clear its array between the two ``klbyii`` calls of a channel: if
    the separable Hamiltonian has fewer bound states than the semilocal
    one, the surplus entries of the nonlocal row repeat the semilocal
    values.  ``he_semilocal`` and ``he_nonlocal`` are the four lowest
    eigenvalues actually computed (zero where ``dspevx`` delivered none).
    These spectra are ``None`` if they were not computed (fast ghost test).
    """

    l: int
    verdict: str              # 'no ghost', 'one or more ghosts', 'undetermined', 'illdefined'
    message: str              # the line of the report, e.g. '* no ghost (ekb < 0, eref < eloc0)'
    ekb: float                # Kleinman-Bylander energy (Hartree)
    ckb: float                # Kleinman-Bylander cosine
    eloc0: float              # ground state of the local potential (Hartree; 0: none)
    eloc1: float              # first excited state of the local potential (Hartree; 0: none)
    eref: float               # reference energy: state of the semilocal potential (Hartree)
    searched_reference: bool = False      # eref had to be calculated here
    notes: tuple[str, ...] = ()           # 'eref', 'eloc0', 'eloc1': assumed zero
    semilocal: tuple[float, ...] | None = None   # spectrum (eV) as printed: e0, e1, e2
    nonlocal_: tuple[float, ...] | None = None   # spectrum (eV) as printed: e0, e1, e2
    he_semilocal: tuple[float, ...] | None = None   # eigenvalues from dspevx (Hartree)
    he_nonlocal: tuple[float, ...] | None = None
    info_semilocal: int = 0
    info_nonlocal: int = 0

    @property
    def ekb_eV(self) -> float:
        return self.ekb * RY2

    @property
    def eloc0_eV(self) -> float:
        return self.eloc0 * RY2

    @property
    def eloc1_eV(self) -> float:
        return self.eloc1 * RY2

    @property
    def eref_eV(self) -> float:
        return self.eref * RY2

    @property
    def ghost_free(self) -> bool:
        return self.verdict == "no ghost"

    @property
    def margin(self) -> float | None:
        """Distance (Hartree) of ``eref`` from the nearest bound of the
        ghost-free window; ``None`` unless the verdict is 'no ghost'."""
        if self.verdict != "no ghost":
            return None
        if self.ekb > 0.0:
            return min(self.eref - self.eloc0, self.eloc1 - self.eref)
        return self.eloc0 - self.eref


@dataclass
class LogDerivative:
    """Logarithmic derivatives of one angular momentum at the diagnostic
    radius: energy meshes (Hartree) and ``r d/dr ln(psi)`` for the
    all-electron (unit ``5l``), semilocal (``6l``) and separable (``7l``)
    potential.  The separable curve exists only for projector channels."""

    l: int
    e_ae: np.ndarray
    d_ae: np.ndarray
    e_sl: np.ndarray
    d_sl: np.ndarray
    e_nl: np.ndarray | None = None
    d_nl: np.ndarray | None = None


@dataclass
class PpcheckResult:
    """What ``ppcheck`` produced."""

    channels: list[KbChannel] = field(default_factory=list)
    rld: float = 0.0                   # diagnostic radius asked for
    ild: int = 0                       # its mesh index (Fortran value)
    r_diag: float = 0.0                # r(ild), the radius printed
    logder: list[LogDerivative] = field(default_factory=list)
    logder_done: bool = False          # the curves were calculated
    e: np.ndarray | None = None        # reference spectrum after the search
    up_last: np.ndarray | None = None  # ``up`` of the last dftseq call (see kinkon)
    klbyii_called: bool = False        # the Laguerre-basis spectra were computed


def energy_mesh() -> np.ndarray:
    """Fixed energy mesh of the separable-potential curves (``ecc``).

    From -2 Hartree in steps of 0.030, between -1 and 1 in steps of 0.003;
    the step sizes are single precision literals in the Fortran.
    """
    iep = 870
    es1 = float(np.float32(0.030))
    es2 = float(np.float32(0.003))
    ecc = [-2.0]
    eccmx = 2.0
    last = iep - 1
    for i in range(iep - 1):
        if ecc[i] > -1.0 and ecc[i] < 1.0:
            ecc.append(ecc[i] + es2)
        else:
            ecc.append(ecc[i] + es1)
        if ecc[i + 1] > eccmx:
            last = i
            break
    # the Fortran leaves the loop with iep = i (1-based), also when it runs out
    return np.array(ecc[:last + 1])


def _line640(label: str, value: float, unit: str | None = None, name: str | None = None) -> str:
    text = fmt_a(label, 30) + "  " + fmt_f(value, 12, 4)
    if unit is not None:
        text += " " + fmt_a(unit, 2) + "    " + fmt_a(name or "", 5)
    return text


def ppcheck(tnrl: bool, tlgd: bool, tkb: bool, l_loc: int, lmax: int, lbeg: int,
            lend: int, mmax: int, rld: float, e: Sequence[float], r: np.ndarray,
            vae: np.ndarray, vi: np.ndarray, uu: np.ndarray, vbare: np.ndarray,
            out: Callable[[str], None] | None = None,
            warn: Callable[[str], None] | None = None,
            spectra: bool = True, logder: bool | str = True,
            eigensolver: Callable | None = None) -> PpcheckResult:
    """Spectrum and logarithmic derivatives of the pseudopotentials.

    Arguments follow ``ppcheck(iu,tnrl,tlgd,tkb,l_loc,lmax,lbeg,lend,mmax,
    rld,e,r,vae,vi,uu,vbare)``:

    ``tnrl``   reference (all-electron) calculation was non-relativistic
    ``tlgd``   do the logarithmic derivatives
    ``tkb``    test the separable potentials for ghost states
    ``l_loc``  angular momentum of the local potential, ``lmax`` highest
               channel present, ``lbeg..lend`` channels to check
    ``rld``    diagnostic radius
    ``e``      reference spectrum per channel (``e[l]``; >= 0: to be found)
    ``vae``    all-electron potential (may be longer than ``mmax``)
    ``vi``     screening potential
    ``uu``     pseudo wave functions per channel, ``uu[l]``
    ``vbare``  ionic pseudopotentials per channel, ``vbare[l]``

    ``out`` receives every line written to the output unit, ``warn`` the
    messages of ``dftseq`` (the same stream in ``pslp``).  Not in the
    Fortran: ``spectra = False`` skips the two ``klbyii`` calls per channel
    (their table rows are then missing from the output) and ``logder =
    False`` skips the calculation of the curves but not the lines printed
    with them; both exist for the fast ghost test.  ``logder = "dftseq"``
    computes the all-electron and semilocal curves only (not those of the
    separable potentials, which need ``derlkb``).
    """
    if out is None:
        def out(line: str) -> None:
            return None
    r = np.ascontiguousarray(r[:mmax], dtype=float)
    vi = np.asarray(vi[:mmax], dtype=float)
    vae_full = np.asarray(vae, dtype=float)
    uu = np.asarray(uu, dtype=float)
    vbare = np.asarray(vbare, dtype=float)
    e = np.array(e, dtype=float)
    wa = np.zeros(mmax)
    wb = np.ones(mmax)
    al = 0.1e0 * libm.slog(float(r[10]) / float(r[0]))
    ll_loc = abs(l_loc) + 1
    lbeg = abs(lbeg)
    lend = abs(lend)
    llbeg = lbeg + 1
    llend = lend + 1
    llmx = lmax + 1
    result = PpcheckResult(rld=rld)
    projector_channels = [ll for ll in range(llbeg, min(llend, llmx) + 1)
                          if tkb and ll != ll_loc]

    # separable potentials
    vkb: dict[int, np.ndarray] = {}
    vaverage: dict[int, float] = {}
    spec: dict[int, dict] = {}
    if tkb:
        out("")
        out(" --- kb potentials: spectrum of bound states (eV) ---")
        out("")
        out("            l          e0            e1            e2")
    basis = None
    enl = np.zeros(20)
    with np.errstate(all="ignore"):
        for ll in projector_channels:
            vk = (vbare[ll - 1, :mmax] - vbare[ll_loc - 1, :mmax]) * uu[ll - 1, :mmax] / (r * r)
            vaverage[ll] = dmelm(mmax, al, r, vk, uu[ll - 1])
            ckb = float(np.float64(1.e0) / np.float64(vaverage[ll]))
            vk = vk * r
            vkb[ll] = vk
            spec[ll] = {}
            if not spectra:
                continue
            if basis is None:
                basis = LaguerreBasis(mmax, r)
            enl[:3] = 0.e0
            # spectrum semilocal case w/ polynomial representation
            enl, he, info = klbyii(ll - 1, mmax, r, vi, vbare[ll - 1], 0, ckb, vk, enl,
                                   basis, warn=out, eigensolver=eigensolver)
            printed = tuple(RY2 * min(float(enl[i]), 0.e0) for i in range(3))
            out("semilocal " + fmt_i(ll - 1, 3) + "".join("  " + fmt_f(x, 12, 4) for x in printed))
            spec[ll].update(semilocal=printed, he_semilocal=tuple(he), info_semilocal=info)
            # spectrum nonlocal case w/ polynomial representation
            enl, he, info = klbyii(ll - 1, mmax, r, vi, vbare[ll_loc - 1], 1, ckb, vk, enl,
                                   basis, warn=out, eigensolver=eigensolver)
            printed = tuple(RY2 * min(float(enl[i]), 0.e0) for i in range(3))
            out("nonlocal  " + fmt_i(ll - 1, 3) + "".join("  " + fmt_f(x, 12, 4) for x in printed))
            spec[ll].update(nonlocal_=printed, he_nonlocal=tuple(he), info_nonlocal=info)

    # Kleinman Bylander analysis following Gonze et al
    # dftseq work arrays of ppcheck (in/out arguments of every call)
    nin = 0
    mch = 0
    u = np.zeros(mmax)
    up = np.zeros(mmax)
    upp = np.zeros(mmax)

    def bound_state(n_: int, l_: int, v_: np.ndarray, e_: float) -> float:
        nonlocal nin, mch, u, up, upp
        res = dftseq(8, 0.e0, mmax, r, n_, l_, 1.e0, v_, wa, wb, mch=mch, e=e_,
                     warn=warn, nin=nin, u=u, up=up, upp=upp)
        nin, mch, u, up, upp = res.nin, res.mch, res.u, res.up, res.upp
        return res.e

    # screened local potential
    vsl = vi + vbare[ll_loc - 1, :mmax] if tkb else None

    for ll in projector_channels:
        _, sl, _ = labels(1, ll - 1, 1)
        out("")
        out(" --- analysis of kb potentials: " + sl + " waves  ---")
        out("")
        notes: list[str] = []
        searched = False

        # check for input reference spectrum (screened semilocal potential)
        # if eigenvalue comes as zero assume it needs to be evaluated
        if e[ll - 1] >= 0.0:
            searched = True
            out(" --- searching reference state ---")
            wc = vi + vbare[ll - 1, :mmax]
            e[ll - 1] = bound_state(ll, ll - 1, wc, float(e[ll - 1]))
            if e[ll - 1] >= 0.0:
                notes.append("eref")
                out(" note: for the semilocal " + sl + " component no bound " + sl + " state")
                out("       is found, the ghost state analysis will assume")
                out("       a zero reference energy (variable eref)")
                out("")

        with np.errstate(all="ignore"):
            vnorm = dmelm(mmax, al, r, vkb[ll], vkb[ll])
            ekb = float(np.float64(vnorm) / np.float64(vaverage[ll]))
            ckb = float(np.float64(vaverage[ll]) / np.sqrt(np.float64(vnorm)))
        eref = float(e[ll - 1])

        # local potential groundstate
        eloc0 = bound_state(ll, ll - 1, vsl, eref)
        if eloc0 >= 0.0:
            notes.append("eloc0")
            out(" note: for the local potential no bound " + sl + " state is")
            out("       found, the ghost state analysis will assume")
            out("       a zero groundstate energy (variable eloc0)")
            out("")
        # local potential first excited state
        eloc1 = bound_state(ll + 1, ll - 1, vsl, eloc0)
        if eloc1 >= 0.0:
            notes.append("eloc1")
            out(" note: for the local potential no bound excited " + sl + " state")
            out("       is found, the ghost state analysis will assume")
            out("       a zero 1st excited state energy (variable eloc1)")
            out("")

        # ghost state criteria, e(ll) is the bound state (reference) energy
        extra = None
        if ekb > 0.0:
            if eref >= eloc0:
                if eref < eloc1:
                    verdict, message = VERDICTS[0], "* no ghost (ekb > 0, eloc0 < eref < eloc1)"
                elif eloc1 < 0.0:
                    verdict, message = VERDICTS[1], "* one or more ghosts (ekb > 0, eref > eloc1)"
                else:
                    verdict, message = VERDICTS[2], "* undetermined (ekb > 0, eref = eloc1 = 0)"
                    extra = "  note: to decide, inspect log derivatives"
            else:
                verdict, message = VERDICTS[3], "* illdefined (ekb > 0, eref < eloc0)"
        else:
            if eref < eloc0:
                verdict, message = VERDICTS[0], "* no ghost (ekb < 0, eref < eloc0)"
            elif eloc0 < 0.0:
                verdict, message = VERDICTS[1], "* one or more ghosts (ekb < 0, eref > eloc0)"
            else:
                verdict, message = VERDICTS[2], "* undetermined (ekb < 0, eref = eloc0 = 0)"
                extra = "  note: to decide, inspect log derivatives"
        out(" " + message)
        if extra is not None:
            out(" " + extra)

        out("")
        out(_line640("kb cosine", ckb))
        out(_line640("kb energy", ekb * RY2, "eV", "ekb"))
        out(_line640("local potential groundstate", eloc0 * RY2, "eV", "eloc0"))
        out(_line640("dto. 1st excited state", eloc1 * RY2, "eV", "eloc1"))
        out(_line640("reference energy", eref * RY2, "eV", "eref"))
        result.channels.append(KbChannel(
            l=ll - 1, verdict=verdict, message=message, ekb=ekb, ckb=ckb, eloc0=eloc0,
            eloc1=eloc1, eref=eref, searched_reference=searched, notes=tuple(notes),
            **spec[ll]))

    result.e = e
    # ``up`` as the last dftseq call left it; None if there was no call
    result.up_last = up if projector_channels else None
    result.klbyii_called = basis is not None
    if not tlgd:
        return result

    # logarithmic derivatives
    # diagnostic radius rld
    ild = 0
    for i in range(1, mmax + 1, 2):
        if rld <= r[i - 1]:
            ild = i
            break
    if ild == 0:
        raise FortranStop("pslp - radius for log derivative > r(mmax)")
    if 2 % ild == 0:
        ild = ild + 1
    result.ild = ild
    result.r_diag = float(r[ild - 1])
    out("")
    out(" --- logarithmic derivatives: at radius =" + fmt_f(r[ild - 1], 7, 4) + " ---")
    out("")

    # energy mesh (should be extensive and dense for finding ghost states)
    # for full and semilocal potential an adaptive mesh is used (coarser)
    ecc = energy_mesh()
    iep = len(ecc)
    eccmx = 2.0
    nl_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # nonlocal (kleinman bylander) potentials
    if tkb:
        out(" --- nonlocal potentials ---")
    if ild < 5:
        # diagnostic radius at the first mesh points (the default radius of
        # elements beyond the table of rcovalent is zero): with projector
        # channels the Fortran stops in dsimpson; without, its dftseq calls
        # read below the first array element.  The curves are not computed.
        if projector_channels:
            raise FortranStop("simpson - stop: wrong n")
        result.up_last = np.zeros(mmax)
        logder = False
    if logder is True:
        for ll in projector_channels:
            vkb[ll] = vkb[ll] * r
            dl = derlkb(ll - 1, ecc, ild, vaverage[ll], mmax, r, vkb[ll], vsl)
            nl_curves[ll - 1] = (ecc.copy(), dl)

    # all-electron potential (kind=1), semilocal potentials (kind=2)
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    deltaold = 0.0
    z = 0.0
    rl = r.tolist()
    for kind in (1, 2):
        irl = 3
        if tnrl or kind == 2:
            irl = 6
        if kind == 1:
            out(" --- all-electron potential ---")
        else:
            out(" --- semilocal potentials ---")
        if not logder:
            continue
        for ll in range(llbeg, llend + 1):
            if kind == 2:
                if ll <= llmx:
                    v = vi + vbare[ll - 1, :mmax]
                else:
                    v = vi + vbare[ll_loc - 1, :mmax]
                # the Fortran overwrites vae(1:mmax); points beyond keep the
                # all-electron values (not referenced in mode 6)
                vae_full = np.concatenate([v, vae_full[mmax:]])
            en = [float(ecc[0])]
            dl = []
            for i in range(iep):
                res = dftseq(irl, z, mmax, r, ll, ll - 1, 1.e0, vae_full, wa, wb,
                             mch=ild, e=en[i])
                ul = res.u[:ild].tolist()
                up = res.up
                uk = ul[ild - 1]
                with np.errstate(all="ignore"):
                    uld = float(np.float64(up[ild - 1]) / np.float64(uk * al)) - 1.0
                    # adaptive stepsize using the energy derivative of the
                    # log-derivative (friedel sum rule)
                    gsum = 0.
                    for k in range(ild - 1):
                        gsum = gsum + ul[k] * ul[k] * rl[k]
                    rk = rl[ild - 1]
                    gsum = float(np.float64(-2 * rk * al * (gsum + 0.5 * uk * uk * rk))
                                 / np.float64(uk * uk))
                    delta = float(abs(np.float64(0.5) / np.float64(gsum)))
                if abs(uld) > 20.:
                    delta = deltaold
                deltaold = delta
                en.append(en[i] + delta)
                dl.append(uld)
                if en[i] > eccmx:
                    break
            curves[(kind, ll - 1)] = (np.array(en[:len(dl)]), np.array(dl))
            result.up_last = up

    if logder is True:
        result.logder_done = True
        for ll in range(llbeg, llend + 1):
            e_ae, d_ae = curves[(1, ll - 1)]
            e_sl, d_sl = curves[(2, ll - 1)]
            e_nl, d_nl = nl_curves.get(ll - 1, (None, None))
            result.logder.append(LogDerivative(ll - 1, e_ae, d_ae, e_sl, d_sl, e_nl, d_nl))
    return result
