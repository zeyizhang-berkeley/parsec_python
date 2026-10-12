"""Main program ``fhipp`` and the ``psgen`` driver.

Port of ``fhipp.f``: the all-electron atom (everything up to the line
``fhipp - all-electron atom done`` plus the files written right after it,
``fort.37`` and ``fort.19``) and the pseudopotential branch (default cutoff
radii and reference energies, the optional per-channel records, the call of
``ncpp`` and the echo of the input); of ``out38`` from ``outwfct.f``; and of
the input preparation / output assembly of the csh script
``bin/Tools/psgen`` (``.dat``, ``.cpi``, ``.fc``, ``.aep``).

The Fortran unit files map to Python as follows::

    fort.20  control flags           -> arguments tdopsp, tnrl (read_fort20)
    fort.22  atomic input (.ini)     -> FhiInput (parse_ini)
    fort.18  frozen core density     -> argument fort18 (text of a .fc file)
    fort.36  initial potential       -> argument fort36 (text of a .aep file)
    fort.23  monitoring data (.dat)  -> FhippState.fort23 (list of lines)
    fort.38  all-electron wfcts      -> FhippState.fort38 (text)
    fort.37  effective potential     -> FhippState.fort37 (text, .aep)
    fort.19  full core density       -> FhippState.fort19 (text, .fc)
    unit 6   messages                -> FhippState.unit6 (list of lines)
    fort.24, fort.25, fort.27, fort.28, fort.39, fort.4[0-9]
                                     -> FhippState.ncpp (NcppResult; its
                                        method unit_files() gives the texts)

The pseudopotential branch of ``fhipp.f`` (reading ``ltmx``, the default
cutoff radii, ``ncpp`` ...) is :func:`pseudopotentials`, called by
:func:`fhipp` right after the all-electron files have been produced;
:class:`FhippState` keeps every variable of the main program that is live
at that point and receives the results of the branch.
"""

from __future__ import annotations

import fnmatch
import getpass
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .atom import atomini, sratom
from .constants import IGRMAP, ITMX, MS, MX, RY2, SREV, TSIC
from .defrtm import defrtm
from .errors import FortranStop
from .fortran_list import list_record, unit6_lines
from .fortran_io import (
    FortranEndOfFile, FortranReadError, ListReader, fmt_a, fmt_e, fmt_es,
    fmt_f, fmt_i,
)
from .labels import labelmap, labels, spp
from .mesh import logmesh
from .ncpp import NcppChannel, NcppResult, ncpp
from .radial import dmelm, dspv, fmom, vestat
from .xc import vexcor

__all__ = [
    "OptionalLine", "FhiInput", "parse_ini", "read_fort20", "FhippState",
    "fhipp", "all_electron", "pseudopotentials", "format_fort19",
    "format_fort37", "out38", "apply_psgen_options", "CpiData", "PsgenResult",
    "psgen", "format_cpi", "format_dat", "UNUSED_FIT_PARAMETERS",
]

_VERIFY = "fhipp - verify input data file fort.22"


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OptionalLine:
    """One optional record ``lt rct et spptype`` of the ``.ini`` file."""

    lt: int        # angular momentum the changes apply to
    rct: float     # new core radius (bohr) if non-zero
    et: float      # new reference energy (eV) if non-zero
    spptype: str   # pseudopotential type, first letter h/H or t/T


@dataclass(frozen=True)
class FhiInput:
    """Content of ``fort.22`` after the normalizations done by ``fhipp``.

    ``z, nc, nv, rnlc, n, l, f`` are stored as absolute values and ``iexc``
    is mapped to 8 if it is 0 or larger than 12, exactly as in the Fortran.
    The pseudopotential records are parsed eagerly but their read errors are
    only recorded, because the Fortran reads them after the all-electron
    atom is done.
    """

    z: float                       # atomic number
    nc: int                        # number of core states
    nv: int                        # number of valence states
    iexc: int                      # exchange-correlation type
    rnlc: float                    # cutoff radius of the pseudocore
    n: tuple[int, ...]             # principal quantum numbers
    l: tuple[int, ...]             # angular momenta
    f: tuple[float, ...]           # occupation numbers
    ltmx: int | None = None        # maximum l of the pseudopotentials
    spptype: str | None = None     # default scheme as read (h... or t...)
    options: tuple[OptionalLine, ...] = ()
    pp_header_error: bool = False  # ltmx/spptype record missing or bad
    options_error: bool = False    # an optional record could not be read

    @property
    def sf(self) -> float:
        """Number of electrons, summed in input order as in the Fortran."""
        total = 0.0
        for value in self.f:
            total = total + value
        return total


def parse_ini(text: str) -> FhiInput:
    """Read the atomic input the way ``fhipp.f`` reads unit 22.

    Raises :class:`FortranStop` with the original stop messages for bad
    header or state records and for the ``l``/``n`` range checks.
    """
    reader = ListReader(text)
    try:
        z, nc, nv, iexc, rnlc = reader.read("riiir")
    except (FortranReadError, FortranEndOfFile):
        raise FortranStop(_VERIFY) from None
    if None in (z, nc, nv, iexc, rnlc):
        raise FortranStop(_VERIFY)
    z, nc, nv = abs(z), abs(nc), abs(nv)
    if nc + nv > MS:
        raise FortranStop(f"fhipp - data: more than ms = {MS} states")
    ns, ls, fs = [], [], []
    for _ in range(nc + nv):
        try:
            ni, li, fi = reader.read("iir")
        except (FortranReadError, FortranEndOfFile):
            raise FortranStop(_VERIFY) from None
        if None in (ni, li, fi):
            raise FortranStop(_VERIFY)
        ni, li, fi = abs(ni), abs(li), abs(fi)
        if ni <= li:
            raise FortranStop("fhipp - data: l .le. n")
        if li > 4:
            raise FortranStop("fhipp - data: l out of range (0 ... 4)")
        ns.append(ni)
        ls.append(li)
        fs.append(fi)
    iexc = abs(iexc)
    if iexc == 0 or iexc > 12:
        iexc = 8
    rnlc = abs(rnlc)

    # records read by the pseudopotential branch
    ltmx = spptype = None
    pp_header_error = False
    options: list[OptionalLine] = []
    options_error = False
    try:
        ltmx, spptype = reader.read("ia")
        # null values leave the (zero / blank) variables of the Fortran untouched
        ltmx = 0 if ltmx is None else ltmx
        spptype = "" if spptype is None else spptype[:25]          # character*25
    except (FortranReadError, FortranEndOfFile):
        ltmx = spptype = None
        pp_header_error = True
    if not pp_header_error:
        last_type = "-"
        while True:
            try:
                lt, rct, et, sp = reader.read("irra")
            except FortranEndOfFile:
                break
            except FortranReadError:
                options_error = True
                break
            # null values keep the previous content of the variables
            if lt is None:
                lt = options[-1].lt if options else 0
            rct = 0.0 if rct is None and not options else (options[-1].rct if rct is None else rct)
            et = 0.0 if et is None and not options else (options[-1].et if et is None else et)
            sp = last_type if sp is None else sp[:25]
            last_type = sp
            options.append(OptionalLine(lt, rct, et, sp))
    return FhiInput(z, nc, nv, iexc, rnlc, tuple(ns), tuple(ls), tuple(fs),
                    ltmx, spptype, tuple(options), pp_header_error, options_error)


def read_fort20(text: str | None, tdopsp: bool = True, tnrl: bool = False) -> tuple[bool, bool]:
    """``read(20,*,end=10,err=10) tdopsp,tnrl`` with the Fortran defaults."""
    if text is None:
        return tdopsp, tnrl
    try:
        values = ListReader(text).read("ll")
    except (FortranReadError, FortranEndOfFile) as exc:
        values = list(exc.values) + [None, None]
    if values[0] is not None:
        tdopsp = bool(values[0])
    if values[1] is not None:
        tnrl = bool(values[1])
    return tdopsp, tnrl


# --------------------------------------------------------------------------
# state of the main program
# --------------------------------------------------------------------------

@dataclass
class FhippState:
    """Variables of ``program fhipp`` after the all-electron atom.

    Field names are the Fortran variable names.  Arrays are 0-based float64
    of length ``mmax`` (``uu`` has shape ``(nc+nv, mmax)``, i.e. ``uu[i-1]``
    is Fortran ``uu(:,i)``); integers that are mesh indices (``ninu``) keep
    their Fortran values.  Energies are in Hartree.
    """

    inp: FhiInput
    # control
    tdopsp: bool          # generate pseudopotentials
    tnrl: bool            # non-relativistic atom
    tcore: bool           # False in frozen-core mode (fort.18 was read)
    irl: int              # 1 scalar-relativistic, 2 non-relativistic
    igr: int              # 1: density gradients evaluated
    iexc: int
    # atom
    z: float
    nc: int
    nv: int
    rnlc: float
    sf: float             # number of electrons
    n: list[int]
    l: list[int]
    f: np.ndarray
    # mesh
    mmax: int
    amesh: float
    al: float             # log(amesh) as computed in logmesh
    r: np.ndarray
    # self-consistent atom (output of sratom)
    it: int               # number of iterations (itmx+1: not converged)
    de: float             # "convergence" printed in fort.23
    e: np.ndarray         # eigenvalues
    ninu: list[int]       # outermost mesh point of each state
    rpk: np.ndarray       # outermost peak of each wave function
    vi: np.ndarray        # effective potential (after the last mixing step)
    rho: np.ndarray       # total density
    rhop: np.ndarray
    rhopp: np.ndarray
    dc: np.ndarray        # core density
    dcp: np.ndarray
    dcpp: np.ndarray
    uu: np.ndarray        # wave functions [state, mesh point]
    # energies and tests
    ves: np.ndarray       # Hartree potential of rho (vestat with zion = sf)
    vxc: np.ndarray       # xc potential of rho
    dexc: np.ndarray      # xc energy density, common /xced/
    tc: float             # integrated density
    t1: float             # 1st derivative test
    t2: float             # 2nd derivative test
    eeel: float           # electron-electron interaction (2 * Hartree energy)
    evxc: float           # xc potential energy
    ex: float
    ec: float
    ecou: float           # electron-nucleus energy
    epot_core: float
    ekin_core: float
    esic_core: float
    ekin: float
    esic: float
    eeig: float           # orbital energy sum
    etot: float
    symz: str
    symxc: str
    # outputs
    unit6: list[str] = field(default_factory=list)    # messages (unit ie)
    fort23: list[str] = field(default_factory=list)   # lines of unit iu
    fort38: str = ""                                  # wave functions
    fort37: str | None = None                         # potential (.aep)
    fort19: str | None = None                         # core density (.fc)
    stage: str = "all-electron"                       # last completed stage
    timings: dict[str, float] = field(default_factory=dict)
    # --- pseudopotential branch (set by pseudopotentials()) ---------------
    # lists indexed by the angular momentum l = 0 .. lmax (Fortran l1 = l+1)
    ltmx: int | None = None              # maximum l as read from the input
    lmax: int | None = None              # max(ltmx, 0)
    i_pp_type_def: int | None = None     # default scheme: 1 Hamann, 2 Troullier-Martins
    i_pp_type: list[int] = field(default_factory=list)   # scheme per channel
    np_: list[int] = field(default_factory=list)         # np: reference principal quantum numbers
    ep: list[float] = field(default_factory=list)        # reference energies passed to ncpp
    fp: list[float] = field(default_factory=list)        # reference occupations
    rc: list[float] = field(default_factory=list)        # cutoff radii passed to ncpp
    rc_d: list[tuple[float, float]] = field(default_factory=list)   # default radii (Hamann, TM)
    inode: list[int] = field(default_factory=list)       # grid index of the outermost node
    ipeak: list[int] = field(default_factory=list)       # grid index of the outermost peak
    emax: float | None = None            # highest valence eigenvalue: default reference energy
    zv: float | None = None              # number of valence electrons
    zc: float | None = None              # number of core electrons
    ncpp: NcppResult | None = None       # everything computed by ncpp

    @property
    def fort23_text(self) -> str:
        return "".join(line + "\n" for line in self.fort23)


# --------------------------------------------------------------------------
# output files
# --------------------------------------------------------------------------

def out38(nmn: int, nmx: int, ninu: Sequence[int], n: Sequence[int],
          l: Sequence[int], r: np.ndarray, u: np.ndarray) -> str:
    """Wave functions as written to unit 38 (``out38`` in ``outwfct.f``).

    ``nmn, nmx`` are the Fortran state indices (1-based, inclusive);
    ``u[i-1]`` is the wave function of state ``i``.
    """
    lines = []
    step = max(ninu[nmx - 1] // 100, 1)
    for i in range(nmx, nmn - 1, -1):
        sn, sl, _ = labels(n[i - 1], l[i - 1], 1)
        nn = n[i - 1] if 1 <= n[i - 1] <= 6 else 7     # labels() overwrites n
        lines.append("# " + sn + sl + " " + fmt_i(i, 2) + " " + fmt_i(nn, 2) + " "
                     + fmt_i(l[i - 1], 2) + " " + fmt_i(0, 2) + " " + fmt_i(ninu[i - 1], 5))
        ui = u[i - 1]
        for j in range(0, ninu[i - 1], step):
            lines.append(fmt_e(r[j], 12, 6) + " " + fmt_e(ui[j], 12, 6))
    return "".join(line + "\n" for line in lines)


def format_fort37(st: FhippState) -> str:
    """All-electron effective potential, unit 37 (``<prefix>.aep``)."""
    lines = [fmt_f(st.z, 10, 5) + " " + fmt_i(st.mmax, 4) + " " + fmt_i(st.iexc, 4)
             + " " + fmt_i(st.irl, 4)]
    for i in range(st.mmax):
        lines.append(fmt_e(st.r[i], 12, 6) + " " + fmt_e(st.vi[i], 20, 14))
    return "".join(line + "\n" for line in lines)


def format_fort19(st: FhippState) -> str | None:
    """Full core density, unit 19 (``<prefix>.fc``); ``None`` if frozen core.

    For ``nc = 0`` the Fortran reads ``ninu(0)`` (out of bounds); the value
    found there with gfortran is 0, which is reproduced.
    """
    if not st.tcore:
        return None
    npts = st.ninu[st.nc - 1] if st.nc > 0 else 0
    lines = [fmt_i(npts, 6) + " " + fmt_e(st.ekin_core, 20, 14)]
    for i in range(npts):
        lines.append(fmt_e(st.r[i], 20, 14) + " " + fmt_e(st.dc[i], 20, 14) + " "
                     + fmt_e(st.dcp[i], 20, 14) + " " + fmt_e(st.dcpp[i], 20, 14))
    return "".join(line + "\n" for line in lines)


def _line608(label: str, value: float) -> str:
    return fmt_a(label, 30) + "  " + fmt_f(value, 16, 5)


# --------------------------------------------------------------------------
# main program
# --------------------------------------------------------------------------

def _read_fort36(text: str | None, mmax: int) -> np.ndarray | None:
    """Initial potential from unit 36 (a ``.aep`` file) or ``None``.

    The potential is used only if the header and all ``nin`` records can be
    read and ``nin == mmax`` (Fortran: ``i .eq. nin+1 .and. nin .eq. mmax``).
    """
    if not text:
        return None
    reader = ListReader(text)
    try:
        head = reader.read("riii")
        nin = head[1]
        if nin is None or nin != mmax:
            return None
        vae = np.zeros(mmax)
        for i in range(nin):
            row = reader.read("rr")
            if row[1] is not None:
                vae[i] = row[1]
    except (FortranReadError, FortranEndOfFile):
        return None
    return vae


def _read_fort18(text: str | None, mmax: int, r: np.ndarray, unit6: list[str]
                 ) -> tuple[bool, float, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen-core density from unit 18 (a ``.fc`` file).

    Returns ``(tcore, ekin_core, dc, dcp, dcpp)``; ``tcore`` is False if a
    core density with ``nin > 0`` points was found.
    """
    dc = np.zeros(mmax)
    dcp = np.zeros(mmax)
    dcpp = np.zeros(mmax)
    ekin_core = 0.e0
    if not text:
        return True, ekin_core, dc, dcp, dcpp
    reader = ListReader(text)
    nin = 0
    i = 0            # Fortran loop index at label 58 (0: header not read)
    rxx2 = 0.e0
    arrays = (dc, dcp, dcpp)

    def store(row: Sequence[object], k: int) -> None:
        if k <= mmax:
            for arr, value in zip(arrays, list(row[1:4]) + [None] * 3):
                if value is not None:
                    arr[k - 1] = value

    try:
        head = reader.read("ir")
    except (FortranReadError, FortranEndOfFile) as exc:
        head = (list(exc.values) + [None, None])[:2]
        if head[0] is not None:
            nin = head[0]
    else:
        nin = head[0] if head[0] is not None else 0
        if head[1] is not None:
            ekin_core = head[1]
        if nin > MX:
            raise FortranStop("fhipp - fort.18 exceeds mx mesh points")
        i = 1
        while i <= nin:
            try:
                row = reader.read("rrrr")
            except (FortranReadError, FortranEndOfFile) as exc:
                store(list(exc.values) + [None] * 4, i)
                break
            if i == 2 and row[0] is not None:
                rxx2 = row[0]
            store(row, i)
            i += 1
    if nin > 0:
        if i != nin + 1:
            unit6.append(" & fhipp - warning: inconsistent input unit fort.18")
        if abs(float(r[1]) - rxx2) > 1.e-12:
            unit6.append(" & fhipp - warning: grid from unit fort.18 incompatible")
        if mmax > nin:
            dc[nin:] = 0.e0
            dcp[nin:] = 0.e0
            dcpp[nin:] = 0.e0
        return False, ekin_core, dc, dcp, dcpp
    return True, ekin_core, dc, dcp, dcpp


def all_electron(inp: FhiInput, *, tnrl: bool = False, fort18: str | None = None,
                 fort36: str | None = None, itmx: int = ITMX,
                 trace: Callable[[dict], None] | None = None) -> FhippState:
    """All-electron part of ``program fhipp``.

    Runs from the mesh generation to the line ``fhipp - all-electron atom
    done`` and returns the :class:`FhippState` with unit 23 (all-electron
    part) and unit 38 filled.  ``fort18`` and ``fort36`` are the texts of the
    optional frozen-core density and initial potential files; ``trace`` is
    passed to :func:`sratom` (debugging aid).
    """
    t_start = time.perf_counter()
    unit6: list[str] = []
    out: list[str] = []
    z, nc, nv, iexc, rnlc = inp.z, inp.nc, inp.nv, inp.iexc, inp.rnlc
    n = list(inp.n)
    l = list(inp.l)
    f = np.array(inp.f, dtype=float)
    nst = nc + nv
    sf = inp.sf
    if sf > z:
        out.append(" & fhipp - warning: negative ion")
    if TSIC and iexc != 8:
        raise FortranStop("fhipp - data: bad XC for SIC")

    # all-electron atom:    irl = 1 scalar-relativistic, 2 non-relativistic
    # gradient corrections: igr = 1 gradients evaluated
    tcore = True
    irl = 2 if tnrl else 1
    igr = IGRMAP[max(iexc, 1) - 1]
    if tcore:
        igr = 1

    # grid generation and initial potential
    mmax, amesh, al, r = logmesh(z)
    vi, e = atomini(nst, mmax, z, n, f, r)

    # alternate starting potential
    vae = _read_fort36(fort36, mmax)
    if vae is not None:
        unit6.append(" & fhipp - initial potential read from file fort.36")
        vi = vae

    # for frozen core calculation read core density
    tcore, ekin_core, dc, dcp, dcpp = _read_fort18(fort18, mmax, r, unit6)

    # all-electron calculation
    t_scf = time.perf_counter()
    try:
        sr = sratom(itmx, irl, igr, iexc, tcore, z, nc, nv, n, l, f, e, mmax, r, vi,
                    dc, dcp, dcpp, warn=unit6.append, trace=trace)
    except FortranStop as exc:
        exc.messages = list(unit6)      # what the program wrote before the stop
        raise
    t_scf = time.perf_counter() - t_scf
    it, de = sr.it, sr.de
    e, ninu, rpk, vi = sr.e, sr.ninu, sr.rpk, sr.vi
    rho, rhop, rhopp = sr.rho, sr.rhop, sr.rhopp
    dc, dcp, dcpp, uu = sr.dc, sr.dcp, sr.dcpp, sr.uu
    if it > itmx:
        unit6.append(" & fhipp - warning: potential not converged")

    # some tests of the electron density
    # test: 1 = Integral(rho)/charge
    tc = fmom(0, mmax, al, 1.e0, r, rho)
    t1 = t2 = 0.0
    if igr == 1:
        # test: 1 = -Integral(r*rho')/(3*charge)
        t1 = -fmom(1, mmax, al, sf, r, rhop) / 3.e0
        # test: 1 = Integral(r**2 * rho'')/(12*charge)
        t2 = fmom(2, mmax, al, sf, r, rhopp) / 12.e0

    # total energy components of all-electron atom
    # Coulomb energy, xc-potential energy, etc
    ves, eeel = vestat(mmax, sf, r, rho, True)
    xc = vexcor(iexc, mmax, r, rho, rhop, rhopp, True, ec_in=0.0)
    vxc, evxc, ex, ec = xc.vpxc, xc.evxc, xc.ex, xc.ec
    ecou = -z * fmom(-1, mmax, al, 1.e0, r, rho)
    # core
    epot_core = dmelm(mmax, al, r, vi, dc)
    esic_core = 0.e0
    if tcore:
        ekin_core = 0.e0
        for i in range(nc):
            ekin_core = ekin_core + float(f[i]) * float(e[i])
        ekin_core = ekin_core - epot_core
    # valence
    ekin = 0.e0
    esic = 0.e0
    for i in range(nc, nst):
        ekin = ekin + float(f[i]) * float(e[i])
    ekin = ekin_core + ekin - dmelm(mmax, al, r, vi, rho) + epot_core
    esic = esic_core + esic
    eeig = dspv(1, nst, f, e)
    if not tcore:
        eeig = dspv(nc + 1, nst, f, e)
    etot = ekin + 0.5 * eeel + ecou + ex + ec + esic

    # output for all-electron calculation
    symz, symxc = labelmap(int(z), iexc)
    out.append("")
    out.append("fhi pseudopotential tool fhipp - version " + fmt_a(SREV.ljust(10), 10))
    out.append("")
    out.append(fmt_a("chemical symbol", 30) + "  " + fmt_a(symz, 2))
    out.append(fmt_a("nuclear charge", 30) + "  " + fmt_f(z, 5, 2))
    out.append(fmt_a("total charge", 30) + "  " + fmt_f(z - sf, 5, 2))
    out.append(fmt_a("number of core states", 30) + "  " + fmt_i(nc, 2))
    out.append(fmt_a("number of valence states", 30) + "  " + fmt_i(nv, 2))
    out.append(fmt_a("exchange-correlation model", 30) + "  " + fmt_i(iexc, 2)
               + "  " + fmt_a(symxc, 30))
    if TSIC:
        out.append(fmt_a("self-interaction correction", 30))
    if tnrl:
        out.append(fmt_a("non-relativistic mode", 30))
    else:
        out.append(fmt_a("scalar-relativistic mode", 30))
    if not tcore:
        out.append(fmt_a("frozen-core mode", 30))
    out.append(fmt_a("parameters radial mesh", 30) + "  " + fmt_i(mmax, 4)
               + fmt_f(amesh, 12, 6) + "  " + fmt_e(r[0], 12, 6))
    out.append("")
    out.append(fmt_a("=== all-electron atom ===", 32))
    out.append("")
    out.append("<        n     l      occupation  eigenvalue(eV)")
    out.append("")
    nstart = 1 if tcore else nc + 1
    for i in range(nstart, nst + 1):
        out.append("< " + fmt_i(i, 2) + "    " + fmt_i(n[i - 1], 2) + "    "
                   + fmt_i(l[i - 1], 2) + "    " + fmt_f(f[i - 1], 10, 4)
                   + fmt_f(float(e[i - 1]) * RY2, 18, 4))
    out.append("")
    out.append(" " * 34 + "(Hartree a.u.)")
    out.append(_line608("total energy", etot))
    out.append(_line608("kinetic energy", ekin))
    out.append(_line608("orbital energy", eeig))
    out.append(_line608("coulomb energy", ecou))
    out.append(_line608("hartree energy", 0.5 * eeel))
    out.append(_line608("exchange-correlation energy", ex + ec))
    out.append(_line608("xc potential energy", evxc))
    out.append(fmt_a("number of iterations", 30) + "  " + fmt_i(it, 16) + "  "
               + fmt_a("convergence", 12) + fmt_es(de, 9, 1))
    out.append(_line608("integrated density", tc))
    if igr == 1:
        out.append(_line608("... 1st derivative test 1 = ", t1))
        out.append(_line608("... 2nd derivative test 1 = ", t2))

    # all-electron wave functions
    if tcore:
        fort38 = out38(1, nst, ninu, n, l, r, uu)
    else:
        fort38 = out38(nc + 1, nst, ninu, n, l, r, uu)

    out.append("")
    out.append(" fhipp - all-electron atom done")

    state = FhippState(
        inp=inp, tdopsp=True, tnrl=tnrl, tcore=tcore, irl=irl, igr=igr, iexc=iexc,
        z=z, nc=nc, nv=nv, rnlc=rnlc, sf=sf, n=n, l=l, f=f,
        mmax=mmax, amesh=amesh, al=al, r=r,
        it=it, de=de, e=e, ninu=ninu, rpk=rpk, vi=vi, rho=rho, rhop=rhop,
        rhopp=rhopp, dc=dc, dcp=dcp, dcpp=dcpp, uu=uu,
        ves=ves, vxc=vxc, dexc=xc.dexc, tc=tc, t1=t1, t2=t2, eeel=eeel, evxc=evxc,
        ex=ex, ec=ec, ecou=ecou, epot_core=epot_core, ekin_core=ekin_core,
        esic_core=esic_core, ekin=ekin, esic=esic, eeig=eeig, etot=etot,
        symz=symz, symxc=symxc, unit6=unit6, fort23=out, fort38=fort38,
    )
    state.timings["sratom"] = t_scf
    state.timings["nint_total"] = float(sr.nint_total)
    state.timings["all_electron"] = time.perf_counter() - t_start
    return state


def fhipp(inp: FhiInput, *, tdopsp: bool = True, tnrl: bool = False,
          fort18: str | None = None, fort36: str | None = None) -> FhippState:
    """``program fhipp``: all-electron atom and pseudopotentials.

    ``tdopsp`` and ``tnrl`` are the control flags of ``fort.20``.  With
    ``tdopsp = False`` (``psgen -ao``) the program stops after the
    all-electron atom: units 23 and 38 only.  With ``tdopsp = True`` the
    effective potential (unit 37) and the core density (unit 19) are
    written and the pseudopotentials are generated (:func:`pseudopotentials`;
    ``state.stage`` becomes ``"pseudopotential"``).

    A ``stop`` of the Fortran raises :class:`FortranStop`.  If it happens in
    the pseudopotential branch the exception carries the attribute ``state``
    (the :class:`FhippState` with everything written so far) and, for a stop
    inside the construction of one channel, ``channel`` (its angular
    momentum).
    """
    try:
        state = all_electron(inp, tnrl=tnrl, fort18=fort18, fort36=fort36)
    except FortranStop as exc:
        exc.messages = [line for text in (getattr(exc, "messages", None) or [])
                        for line in unit6_lines(text)]
        raise
    # messages of the atomic-core routines in the layout of the Fortran
    state.unit6[:] = [line for text in state.unit6 for line in unit6_lines(text)]
    state.tdopsp = tdopsp
    if not tdopsp:
        return state

    # ==== all-electron atom done ===

    # all-electron effective potential
    state.fort37 = format_fort37(state)
    # output full core charge density
    state.fort19 = format_fort19(state)

    t_start = time.perf_counter()
    try:
        pseudopotentials(state)
    except FortranStop as exc:
        exc.state = state
        exc.messages = list(state.unit6)
        raise
    state.timings["pseudopotentials"] = time.perf_counter() - t_start
    return state


def pseudopotentials(state: FhippState) -> FhippState:
    """Pseudopotential branch of ``program fhipp`` (after ``fort.19``).

    Uses the records ``ltmx spptype`` and ``lt rct et spptype`` (already
    parsed into ``state.inp``), sets up the default cutoff radii and
    reference energies, calls :func:`ncpp` and echoes the input.  ``state``
    is updated in place (unit 23, unit 6, the per-channel lists and
    ``state.ncpp``) and returned.

    Notes on the Fortran that the port keeps:

    * ``rc_d`` (default radii) is never initialized; the reference build
      holds zeros there (static storage).
    * An optional record always resets the reference energy of its channel,
      also for an occupied one, where it then only serves as the start value
      of the eigenvalue search (``et`` below 0.005 eV selects ``emax``).
    * For a Troullier-Martins channel without tabulated radius and with
      ``np > l+1`` the computed default is 0, so the run stops with
      "cutoff radius too small" unless a radius is given.
    """
    inp = state.inp
    out = state.fort23
    unit6 = state.unit6
    z, nc, nv = state.z, state.nc, state.nv
    n, l, f, e = state.n, state.l, state.f, state.e
    mmax, r, rpk, uu = state.mmax, state.r, state.rpk, state.uu
    r1 = float(r[0])
    iz = int(z)

    # prepare default pseudopotential input
    if inp.pp_header_error or inp.ltmx is None or inp.spptype is None:
        raise FortranStop(_VERIFY)
    ltmx, spptype = inp.ltmx, inp.spptype
    spptype25 = spptype.ljust(25)[:25]
    if spptype[:1] in ("h", "H"):
        out.append("")
        out.append("    " + fmt_a("=== HAMANN mode === ", 20) + "  " + spptype25[:6])
        i_pp_type_def = 1
    elif spptype[:1] in ("t", "T"):
        out.append("")
        out.append("    " + fmt_a("=== TROULLIER-MARTINS mode ===", 30) + "  " + spptype25[:6])
        i_pp_type_def = 2
    else:
        text = "& fhipp - stop: bad pseudopotential type" + spptype25
        unit6.append(list_record(text))
        raise FortranStop(text[2:].rstrip())

    np_ = [0] * MS
    ep = [0.0e0] * MS
    fp = [0.0e0] * MS
    rc = [0.0e0] * MS
    inode = [1] * MS
    ipeak = [1] * MS
    i_pp_type = [i_pp_type_def] * MS
    rc_d = [[0.0e0, 0.0e0] for _ in range(MS)]      # static storage, zero in the Fortran

    # defaults for Hamann case (i_pp_type = 1), cf Ref [1]
    csc = 1.9e0
    csc1 = 4.0e0
    # vsc(1), vsc1(1): Hamann; vsc(2), vsc1(2): Troullier-Martins case (i_pp_type = 2);
    # 1.55 and 1.60 are single precision constants in the source
    with np.errstate(all="ignore"):
        vsc = (0.60e0, float(np.float32(1.55)) * float(np.sqrt(np.float64(7.e0) / np.float64(z))))
        vsc1 = (0.40e0, float(np.float32(1.60)) * float(np.sqrt(np.float64(z) / 7.e0)))

    # Hamann's choice if valence state is unoccupied and unbound
    for i in range(nc):
        l1 = l[i] + 1
        np_[l1 - 1] = max(np_[l1 - 1], n[i])
        if n[i] == l[i] + 1:
            rc_d[l1 - 1][0] = csc1 * float(rpk[i])
        else:
            rc_d[l1 - 1][0] = csc * float(rpk[i])
        fp[l1 - 1] = 0.e0

    # for occupied valence states
    rc_dmax = [0.0e0, 0.0e0]
    emax = -100.0e0
    zv = 0.0e0
    if nv == 0:
        emax = 0.0e0
    else:
        for i in range(nc + nv - 1, nc - 1, -1):
            zv = zv + float(f[i])
            emax = max(emax, float(e[i]))
            l1 = l[i] + 1
            k = l1 - 1
            np_[k] = n[i]
            fp[k] = float(f[i])
            ep[k] = float(e[i])

            # compute default radius
            for itype in (1, 2):
                rc_d[k][itype - 1] = vsc[itype - 1] * float(rpk[i])
                if n[i] == l1:
                    rc_d[k][itype - 1] = vsc1[itype - 1] * float(rpk[i])
                if i_pp_type[k] == 2:
                    rc_d[k][i_pp_type[k] - 1] = defrtm(np_[k], l1 - 1, iz)
                rc_dmax[itype - 1] = max(rc_dmax[itype - 1], rc_d[k][itype - 1])
            # find outermost node
            ui = uu[i]
            rpki = float(rpk[i])
            if n[i] > l1:
                neg = np.nonzero(ui[:mmax - 1] * ui[1:mmax] < 0.0)[0]
                jnode = int(neg[-1]) + 1 if neg.size else 0
                cand = np.nonzero(rpki <= r[jnode:mmax - 1])[0]
                if cand.size:
                    ipeak[k] = jnode + int(cand[0]) + 1
                inode[k] = jnode
            else:
                cand = np.nonzero(rpki >= r[:mmax - 1])[0]
                ipeak[k] = int(cand[-1]) + 1 if cand.size else 0
                inode[k] = 1

    # unoccupied valence states & default assignment
    lmax = max(ltmx, 0)
    if lmax < 0 or lmax > 4:
        raise FortranStop("fhipp - lmax out of range (0 ... 4)")
    for l1 in range(1, lmax + 2):
        k = l1 - 1
        i_pp_type[k] = i_pp_type[0]
        if fp[k] == 0.0 and ep[k] >= 0.0:
            ep[k] = emax
            np_[k] = max(np_[k] + 1, l1)
            rc_d[k][0] = max(rc_dmax[0], rc_d[k][0])
            rc_d[k][1] = max(rc_dmax[1], rc_d[k][1])
        if i_pp_type[k] == 2:
            rc_d[k][i_pp_type[k] - 1] = defrtm(np_[k], l1 - 1, iz)

        rc[k] = rc_d[k][i_pp_type[k] - 1]
        # in case the cutoff radius is not tabulated (troullier-martins) it is computed,
        # these radii may need adjustment for transferability/representability reasons
        if rc[k] < r1:
            unit6.append(list_record("& fhipp - warning: no tabulated default core radius l", l1 - 1))
            rc[k] = rc_d[k][i_pp_type[0] - 1] / vsc[0]
            if np_[k] == l1:
                with np.errstate(all="ignore"):
                    rc[k] = float(np.sqrt(np.float64(
                        rc_d[k][0] / vsc1[0] * (l1 * (l1 - 1) + 0.5e0))))

    # print info on valence wavefunctions
    def r_at(index: int) -> float:
        # r(0) is read out of bounds by the Fortran if no node / peak was found
        return float(r[index - 1]) if 1 <= index <= mmax else 0.e0

    out.append("")
    out.append("  l  n     radius:     node      peak       default core")
    for l1 in range(1, lmax + 2):
        k = l1 - 1
        out.append("x" + fmt_i(l1 - 1, 2) + " " + fmt_i(np_[k], 2) + " " * 10
                   + "".join(" " * 5 + fmt_f(x, 6, 3)
                             for x in (r_at(inode[k]), r_at(ipeak[k]), rc[k])))

    # defaults done

    # alternative input overriding default values
    #   lt        l value for following parameter changes
    #   rct       if non-zero, new core radius for angular momentum lt
    #   et        if non-zero, new energy for lt in eV
    #   spptype   pseudopotential type
    tinopt = [False] * MS
    ltopt = [0] * MS
    rctopt = [0.e0] * MS
    ectopt = [0.e0] * MS
    sppopt = [" "] * MS
    for opt in inp.options:
        lt, rct, et = opt.lt, opt.rct, opt.et
        if lt > lmax:
            unit6.append(list_record(
                "& fhipp - warning: optional input not applied: lmax < lt", lt))
            continue
        l1 = lt + 1
        if lt < 0 or lt > 4:
            raise FortranStop("fhipp - optional data: l out of range (0 ... 4)")
        k = l1 - 1
        if opt.spptype[:1] in ("h", "H"):
            i_pp_type[k] = 1
        elif opt.spptype[:1] in ("t", "T"):
            i_pp_type[k] = 2
        else:
            i_pp_type[k] = i_pp_type_def
        if rct > r1:
            rc[k] = abs(rct)
            rctopt[k] = rc[k]
        else:
            rc[k] = rc_d[k][i_pp_type[k] - 1]
            rctopt[k] = 0.0e0
        if abs(et) < 0.5e-2:
            ep[k] = emax
            ectopt[k] = 0.0e0
        else:
            ep[k] = et / RY2
            ectopt[k] = ep[k]
        ltopt[k] = lt
        sppopt[k] = spp(i_pp_type[k])[0]
        tinopt[k] = True
    if inp.options_error:
        unit6.append(list_record("& fhipp - warning: failed reading optional data"))

    nch = lmax + 1
    state.ltmx, state.lmax, state.i_pp_type_def = ltmx, lmax, i_pp_type_def
    state.i_pp_type = i_pp_type[:nch]
    state.np_ = np_[:nch]
    state.ep = ep[:nch]
    state.fp = fp[:nch]
    state.rc = rc[:nch]
    state.rc_d = [tuple(x) for x in rc_d[:nch]]
    state.inode = inode[:nch]
    state.ipeak = ipeak[:nch]
    state.emax = emax

    # verification of cutoff radii
    for l1 in range(1, nch + 1):
        k = l1 - 1
        if rc[k] < r1 or rc[k] < r_at(inode[k]):
            line = list_record("& fhipp - stop: cutoff radius too small l", l1 - 1)
            unit6.append(line)
            exc = FortranStop(line[3:])
            exc.channel = l1 - 1
            raise exc

    # number of core & valence electrons
    zc = state.sf - zv
    state.zv, state.zc = zv, zc

    # valence density
    rho = state.rho - state.dc
    rhop = state.rhop - state.dcp
    rhopp = state.rhopp - state.dcpp

    # pseudopotentials
    res = ncpp(state.irl, state.iexc, z, zc, zv, lmax, i_pp_type[:nch], np_[:nch], ep[:nch],
               fp[:nch], rc[:nch], state.vi, mmax, r, state.rnlc, state.dc, state.dcp,
               state.dcpp, rho, rhop, rhopp, nv, n[nc:], l[nc:], [float(x) for x in e[nc:]],
               [float(x) for x in f[nc:]], out=out, warn=unit6.append)
    state.ncpp = res

    out.append("")
    out.append(" fhipp - done for input @")
    out.append("")

    # output input data (rnlc as modified by dnlcc7)
    out.append("@ " + fmt_f(z, 5, 2) + fmt_i(nc, 3) + fmt_i(nv, 3) + fmt_i(state.iexc, 3)
               + fmt_e(res.rnlc, 8, 2) + " " + fmt_a(": z nc nv iexc rnlc", 19))
    nst = nc + nv
    n1, l1_, f1 = (n[0], l[0], float(f[0])) if nst > 0 else (0, 0, 0.e0)
    out.append("@" + " " * 5 + fmt_i(n1, 1) + "  " + fmt_i(l1_, 1) + "  " + fmt_f(f1, 5, 2)
               + " " * 8 + fmt_a(": n l f", 7))
    for i in range(1, nst):
        out.append("@" + " " * 5 + fmt_i(n[i], 1) + "  " + fmt_i(l[i], 1) + "  "
                   + fmt_f(float(f[i]), 5, 2))
    out.append("@ " + fmt_i(ltmx, 1) + "  " + spp(i_pp_type_def)[0] + " " * 19
               + fmt_a(": lmax s_pp_def", 15))
    first = True
    for l1 in range(1, nch + 1):
        k = l1 - 1
        if tinopt[k]:
            line = ("@ " + fmt_i(ltopt[k], 1) + "  " + fmt_f(rctopt[k], 5, 2) + "  "
                    + fmt_e(ectopt[k], 8, 2) + "  " + sppopt[k])
            if first:
                first = False
                line += "  " + fmt_a(": lt rct et s_pp_type", 21)
            out.append(line)
    state.stage = "pseudopotential"
    return state


# --------------------------------------------------------------------------
# psgen driver script
# --------------------------------------------------------------------------

def _awk_fields(line: str) -> list[str]:
    return line.split()


def _awk_field(fields: Sequence[str], k: int) -> str:
    return fields[k - 1] if 0 < k <= len(fields) else ""


def _awk_equal(text: str, number: int) -> bool:
    """awk comparison ``$1 == l`` of a field with a number."""
    try:
        return float(text) == float(number)
    except ValueError:
        return text == str(number)


def apply_psgen_options(ini_text: str, *, xctyp: str | int | None = None,
                        rcore: str | float | None = None,
                        default_type: str | None = None,
                        channel_types: Mapping[int, str] | None = None,
                        channel_radii: Mapping[int, str | float] | None = None) -> str:
    """Edit the input like the command line options of the ``psgen`` script.

    ``xctyp`` (``-xc``), ``rcore`` (``-rc``), ``default_type`` (``-T``),
    ``channel_types`` (``-ts/-tp/-td/-tf`` as ``{l: 'h'|'t'}``) and
    ``channel_radii`` (``-rs/-rp/-rd/-rf`` as ``{l: radius}``) are applied
    with the awk commands of the script, including their reformatting of the
    edited lines (fields joined by single blanks).
    """
    lines = ini_text.splitlines()
    if not lines:
        return ini_text
    if channel_types:
        # case [-][t][spdf]: anything but t/T/h/H becomes "h"
        channel_types = {ell: (v if v in ("t", "T", "h", "H") else "h")
                         for ell, v in channel_types.items()}
    if channel_radii:
        # case [-][r][spdf]: the value must look like [0-9][.][0-9]*, else "0.0"
        channel_radii = {ell: (str(v) if fnmatch.fnmatchcase(str(v), "[0-9][.][0-9]*") else "0.0")
                         for ell, v in channel_radii.items()}
    first = _awk_fields(lines[0])
    try:
        iwc = int(float(_awk_field(first, 2)) + float(_awk_field(first, 3)) + 2)
    except ValueError:
        iwc = 2
    if default_type is not None:
        value = default_type[:1]
        value = value if value in "tThH" and value else "t"
        if len(lines) >= iwc:
            lines[iwc - 1] = _awk_field(_awk_fields(lines[iwc - 1]), 1) + " " + value
    pp_type = _awk_field(_awk_fields(lines[iwc - 1]), 2) if len(lines) >= iwc else ""
    if xctyp is not None:
        fld = _awk_fields(lines[0])
        lines[0] = " ".join([_awk_field(fld, 1), _awk_field(fld, 2), _awk_field(fld, 3),
                             str(xctyp)] + [_awk_field(fld, k) for k in range(5, 10)])
    if rcore is not None:
        fld = _awk_fields(lines[0])
        lines[0] = " ".join([_awk_field(fld, k) for k in range(1, 5)] + [str(rcore)]
                            + [_awk_field(fld, k) for k in range(6, 10)])
    for ell in range(4):
        if channel_types and ell in channel_types:
            value = channel_types[ell]
            head, tail, found = lines[:iwc], [], False
            for line in lines[iwc:]:
                fld = _awk_fields(line)
                if _awk_equal(_awk_field(fld, 1), ell):
                    tail.append(f"{ell} {_awk_field(fld, 2)} {_awk_field(fld, 3)} {value}")
                    found = True
                else:
                    tail.append(line)
            if not found:
                tail.append(f"{ell} 0.0 0.0 {value}")
            lines = head + tail
        if channel_radii and ell in channel_radii:
            value = channel_radii[ell]
            head, tail, found = lines[:iwc], [], False
            for line in lines[iwc:]:
                fld = _awk_fields(line)
                if _awk_equal(_awk_field(fld, 1), ell):
                    tail.append(f"{ell} {value} {_awk_field(fld, 3)} {_awk_field(fld, 4)}")
                    found = True
                else:
                    tail.append(line)
            if not found:
                tail.append(f"{ell} {value} 0.0 {pp_type}")
            lines = head + tail
    return "".join(line + "\n" for line in lines)


# the "unused_fit_parameters" block the script puts into every .cpi file
UNUSED_FIT_PARAMETERS: str = (
    "  0.0000    0.0000    0.0000   0.0000\n"
    + "  0.0000    .00e+00   .00e+00\n" * 9
)


def format_cpi(res: NcppResult) -> str:
    """The ``.cpi`` table as assembled by the script ("fhi94md output").

    Line 1: ionic charge (last field of the header of ``fort.40``) and the
    number of channels; ten lines of unused fit parameters; per channel the
    mesh size and increment followed by the records ``i r u(r) v(r)`` of
    ``fort.4[0-4]``; finally the model core density (``fort.27``) if the
    nonlinear core correction is on.
    """
    ionic, _ = res.fort4x()
    parts = [fmt_e(res.zion, 20, 14) + "   " + str(len(ionic)) + "\n", UNUSED_FIT_PARAMETERS]
    for text in ionic:
        head, _, body = text.partition("\n")
        fields = head.split()
        parts.append(fields[1] + "  " + fields[2] + "\n")
        parts.append(body)
    core = res.fort27()
    if core is not None:
        parts.append(core)     # append pseudocore
    return "".join(parts)


def format_dat(state: FhippState, info: str = "pyfhi98pp.psgen") -> str:
    """The ``.dat`` file: ``#info`` line, unit 23 and, if the pseudocore was
    constructed, unit 24 without its ``p`` records (``fgrep -v "p "``)."""
    text = "#info " + info + "\n" + state.fort23_text
    if state.ncpp is not None and state.ncpp.fort24 is not None:
        text += "".join(line + "\n" for line in state.ncpp.fort24 if "p " not in line)
    return text


@dataclass(frozen=True)
class CpiData:
    """The content of a ``.cpi`` table as full-precision arrays.

    The field names are the parameter names of
    ``pp_generation.converters.CpiTable.from_arrays``, so that
    ``CpiTable.from_arrays(**cpi.as_kwargs())`` builds the table without
    reading a file.  (The numbers of a ``.cpi`` file are these values
    rounded to 14 significant digits.)
    """

    valence_charge: float                      # ionic charge zin - zc
    radius: np.ndarray                         # radial mesh (bohr)
    wavefunctions: tuple[np.ndarray, ...]      # u_l(r) = r*R_l(r), l = 0 .. lmax
    potentials: tuple[np.ndarray, ...]         # ionic pseudopotentials (Hartree)
    core_density: np.ndarray | None = None     # model core density (4 pi included)
    core_first_derivative: np.ndarray | None = None
    core_second_derivative: np.ndarray | None = None
    mesh_increment: float | None = None        # amesh = r(2)/r(1)

    def as_kwargs(self) -> dict[str, object]:
        """Keyword arguments for ``CpiTable.from_arrays``."""
        return dict(valence_charge=self.valence_charge, radius=self.radius,
                    wavefunctions=self.wavefunctions, potentials=self.potentials,
                    core_density=self.core_density,
                    core_first_derivative=self.core_first_derivative,
                    core_second_derivative=self.core_second_derivative,
                    mesh_increment=self.mesh_increment)


@dataclass
class PsgenResult:
    """What the ``psgen`` driver produced.

    ``state`` holds every variable of the program (see :class:`FhippState`;
    ``state.ncpp`` is the :class:`NcppResult` of the pseudopotential
    construction).  The texts of the output files are available without
    reading the files back, and the properties below give direct access to
    the quantities a caller usually needs.  Energies are in Hartree unless
    the name ends in ``_ev`` (then ``Hartree*27.2116`` as printed).
    """

    state: FhippState
    prefix: str
    ini_text: str                                   # effective input (fort.22)
    files: dict[str, Path] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)   # stdout of fhipp.x
    dat_text: str = ""                              # <prefix>.dat
    cpi_text: str | None = None                     # <prefix>.cpi
    fc_text: str | None = None                      # <prefix>.fc  (unit 19)
    aep_text: str | None = None                     # <prefix>.aep (unit 37)

    # --- energies ---------------------------------------------------------
    @property
    def etot_ae(self) -> float:
        """Total energy of the all-electron atom (Hartree, full precision)."""
        return self.state.etot

    @property
    def etot_ps(self) -> float | None:
        """Total energy of the pseudo atom (Hartree, full precision)."""
        return None if self.state.ncpp is None else self.state.ncpp.etot

    # --- channels ---------------------------------------------------------
    @property
    def channels(self) -> list[NcppChannel]:
        """Rows of the "pseudo atom" table: ``l, scheme, rc, rmatch, e_ae_ev,
        e_ps_ev, norm, slope`` (and more) for every generated channel."""
        return [] if self.state.ncpp is None else self.state.ncpp.channels

    @property
    def lmax(self) -> int | None:
        return self.state.lmax

    @property
    def zion(self) -> float | None:
        """Ionic charge ``z - zc`` (the valence charge of the ``.cpi`` table)."""
        return None if self.state.ncpp is None else self.state.ncpp.zion

    @property
    def rcore(self) -> float:
        """Core radius of the nonlinear core correction actually used (the
        mesh point chosen by ``dnlcc7``); 0 if the correction is off."""
        return self.state.rnlc if self.state.ncpp is None else self.state.ncpp.rnlc

    # --- valence states of the all-electron atom ---------------------------
    @property
    def valence_states(self) -> list[tuple[int, int, float, float]]:
        """``(n, l, occupation, eigenvalue in Hartree)`` of the valence states."""
        st = self.state
        return [(st.n[i], st.l[i], float(st.f[i]), float(st.e[i]))
                for i in range(st.nc, st.nc + st.nv)]

    @property
    def valence_eigenvalues(self) -> list[float]:
        """Valence eigenvalues of the all-electron atom (Hartree)."""
        st = self.state
        return [float(x) for x in st.e[st.nc:st.nc + st.nv]]

    @property
    def valence_eigenvalues_ev(self) -> list[float]:
        return [x * RY2 for x in self.valence_eigenvalues]

    @property
    def emax(self) -> float:
        """Highest valence eigenvalue (Hartree): the default reference energy
        of unoccupied channels (0 without valence states)."""
        if self.state.emax is not None:
            return self.state.emax
        return max([-100.0e0] + self.valence_eigenvalues) if self.state.nv else 0.0e0

    @property
    def emax_ev(self) -> float:
        return self.emax * RY2

    # --- mesh and tables ----------------------------------------------------
    @property
    def mesh(self) -> np.ndarray:
        """The radial mesh ``r(1..mmax)`` (bohr)."""
        return self.state.r

    @property
    def amesh(self) -> float:
        return self.state.amesh

    @property
    def cpi(self) -> CpiData | None:
        """The ``.cpi`` data as arrays (``None`` before the pseudopotentials)."""
        res = self.state.ncpp
        if res is None:
            return None
        core = res.core
        return CpiData(
            valence_charge=res.zion, radius=res.r,
            wavefunctions=tuple(res.ups[k] for k in range(res.lmax + 1)),
            potentials=tuple(res.vps[k] for k in range(res.lmax + 1)),
            core_density=None if core is None else core[0],
            core_first_derivative=None if core is None else core[1],
            core_second_derivative=None if core is None else core[2],
            mesh_increment=res.amesh)

    @property
    def units(self) -> dict[str, str]:
        """Texts of the Fortran unit files: ``fort.23``, ``fort.38`` and, after
        the pseudopotential stage, ``fort.19``, ``fort.37``, ``fort.24``,
        ``fort.25``, ``fort.27``, ``fort.28``, ``fort.39``, ``fort.4[0-9]``
        (those that the Fortran run would have written)."""
        st = self.state
        files = {"fort.22": self.ini_text, "fort.23": st.fort23_text, "fort.38": st.fort38}
        if st.fort19 is not None:
            files["fort.19"] = st.fort19
        if st.fort37 is not None:
            files["fort.37"] = st.fort37
        if st.ncpp is not None:
            files.update(st.ncpp.unit_files())
        return files


def psgen(ini: str | Path, out: str = "ncpp", workdir: str | Path = ".", *,
          only_atom: bool = False, nonrelativistic: bool = False,
          xctyp: str | int | None = None, rcore: str | float | None = None,
          default_type: str | None = None,
          channel_types: Mapping[int, str] | None = None,
          channel_radii: Mapping[int, str | float] | None = None,
          core_file: str | Path | None = None,
          initial_potential: str | Path | None = None,
          keep_scratch: bool = False, wavefunctions: bool = True,
          write: bool = True) -> PsgenResult:
    """Python counterpart of ``psgen [options] -o out ini``.

    ``ini`` is the path of the input file (or its text if it contains a
    newline).  Options: ``only_atom`` (``-ao``), ``nonrelativistic``
    (``-r``), ``xctyp`` (``-xc``), ``rcore`` (``-rc``), ``default_type``
    (``-T``), ``channel_types`` (``-ts`` ...), ``channel_radii`` (``-rs``
    ...), ``core_file`` (``-fc``), ``initial_potential`` (``-ip``) and
    ``keep_scratch`` (``-g``: also write the ``fort.*`` unit files).

    Files written to ``workdir`` (as assembled by the script):
    ``<out>.dat`` = ``#info`` line + unit 23 (+ unit 24 without its ``p``
    records), ``<out>.cpi`` = the pseudopotential table, ``<out>.fc`` = unit
    19 and ``<out>.aep`` = unit 37 (the last three only without
    ``only_atom``).  The xmgr graphics files are not produced; instead the
    raw unit 38 is written to ``<out>.ae_wfct`` when ``wavefunctions`` is
    set.  Unlike the script, the input file is not rewritten in place; the
    edited input is returned as ``ini_text`` (and written to ``fort.22``
    with ``keep_scratch``).  With ``write=False`` nothing is written; the
    texts are on the returned :class:`PsgenResult` either way.

    A ``stop`` of the Fortran program raises :class:`FortranStop` (message =
    the stop text; ``exc.messages`` = the lines written to the terminal;
    ``exc.state`` = the program state reached; ``exc.channel`` = the
    angular momentum whose construction failed, if the stop occurred while a
    channel was built).  No file is written in that case (the script would
    leave a truncated ``.dat``).
    """
    workdir = Path(workdir)
    if isinstance(ini, Path) or "\n" not in str(ini):
        ini_text = Path(ini).read_text(encoding="utf-8")
    else:
        ini_text = str(ini)
    if any(opt is not None for opt in (xctyp, rcore, default_type)) or channel_types or channel_radii:
        ini_text = apply_psgen_options(
            ini_text, xctyp=xctyp, rcore=rcore, default_type=default_type,
            channel_types=channel_types, channel_radii=channel_radii)
    fort18 = Path(core_file).read_text(encoding="utf-8") if core_file is not None else None
    fort36 = (Path(initial_potential).read_text(encoding="utf-8")
              if initial_potential is not None else None)

    line_1 = [".t.", ".f.", ":", "TDOPSP", "TNRL"]
    if only_atom:
        line_1[0] = ".f."
    if nonrelativistic:
        line_1[1] = ".t."
    fort20 = " ".join(line_1) + "\n"
    tdopsp, tnrl = read_fort20(fort20)

    state = fhipp(parse_ini(ini_text), tdopsp=tdopsp, tnrl=tnrl, fort18=fort18, fort36=fort36)

    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - environment dependent
        user = "unknown"
    stamp = f"{out} {time.strftime('%H:%M:%S %b %d %Y')} {user}"

    t_format = time.perf_counter()
    result = PsgenResult(state=state, prefix=out, ini_text=ini_text,
                         messages=list(state.unit6))
    result.dat_text = format_dat(state, f"pyfhi98pp.psgen {stamp}")
    result.fc_text = state.fort19
    result.aep_text = state.fort37
    if state.ncpp is not None:
        result.cpi_text = format_cpi(state.ncpp)
    state.timings["format"] = time.perf_counter() - t_format
    if not write:
        return result
    workdir.mkdir(parents=True, exist_ok=True)

    def put(name: str, text: str) -> None:
        path = workdir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        result.files[name] = path

    put(f"{out}.dat", result.dat_text)
    if result.fc_text is not None:
        put(f"{out}.fc", result.fc_text)
    if result.aep_text is not None:
        put(f"{out}.aep", result.aep_text)
    if result.cpi_text is not None:
        put(f"{out}.cpi", result.cpi_text)
    if wavefunctions:
        put(f"{out}.ae_wfct", state.fort38)
    if keep_scratch:
        put("fort.20", fort20)
        for name, text in result.units.items():
            put(name, text)
        if fort18 is not None:
            put("fort.18", fort18)
        if fort36 is not None:
            put("fort.36", fort36)
    return result
