"""Main program ``pslp`` and the ``pswatch`` driver.

Port of ``pslp.f`` (test program for semilocal pseudopotentials:
self-consistent pseudo-atom, ghost-state analysis of the Kleinman-Bylander
form, logarithmic derivatives, kinetic-energy convergence) and of the input
preparation / output assembly of the csh script ``bin/Tools/pswatch``.

The Fortran unit files map to Python as follows::

    fort.21  control parameters       -> PslpControl (options of pswatch)
    fort.22  atomic parameters (.ini) -> argument ini (text or path)
    fort.31  pseudopotentials (.cpi)  -> argument cpi (text or path)
    fort.37  all-electron potential   -> argument aep (text or path)
    unit 6   report (<prefix>.test)   -> PswatchResult.report
    fort.38  pseudo wave functions    -> PswatchResult.fort38
    fort.4x, fort.27 (pswatch -cpo)   -> PswatchResult.cpo
    fort.5x/6x/7x log. derivatives    -> PswatchResult.logder (arrays)

Entry points: :func:`pswatch` (complete report) and :func:`ghost_check` /
:func:`ghost_scan` (pseudo-atom and ghost-state analysis only).

Variables that ``pslp.f`` reads without having set them (``dc`` without a
pseudocore in the file, ``ep`` of unoccupied states, ``ec`` on entry of
``vexcor``) are zero in the reference executable and here.
"""

from __future__ import annotations

import getpass
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import libm
from .constants import IGRMAP, ITMX, MS, MX, RY2
from .errors import FortranStop
from .fortran_io import (
    FortranEndOfFile, FortranReadError, ListReader, fmt_a, fmt_e, fmt_es, fmt_f,
    fmt_i, parse_real,
)
from .kinkon import FortranCrash, KinkonMemory, KinkonResult, kinkon
from .labels import labelmap, labels
from .ppcheck import KbChannel, LogDerivative, PpcheckResult, ppcheck, rcovalent
from .psatom import psatom
from .psgen import out38
from .radial import dextv, dmelm, dspv, fmom, gltfmv, vestat
from .xc import vexcor

__all__ = [
    "PslpControl", "PseudoAtom", "PswatchResult", "GhostResult", "pslp",
    "pswatch", "ghost_check", "ghost_scan", "read_cpi", "read_aep",
    "flushed_output", "KbChannel", "LogDerivative", "KinkonResult", "FortranCrash",
]

SREV = "rev 1998"         # parameter(srev='rev 1998'), character*10
RMX = 80.e0               # parameter(rmx=80.d0): mesh is cut at this radius


# --------------------------------------------------------------------------
# messages of the error unit (unit 6 as well)
# --------------------------------------------------------------------------

_INT = r"(-?\d+)"
_MSG_BOUND = re.compile(r"^(& dftseq - no bound state found \((?:iter|ctp)\): e=>0 n l) "
                        + _INT + " " + _INT + "$")
_MSG_NINT = re.compile(r"^&   nint n l\s+" + _INT + " " + _INT + " " + _INT + "$")
_MSG_CTP = re.compile(r"^&    nint n l e\s+" + _INT + " " + _INT + " " + _INT + r" (\S+)$")


def _list_int(value: int) -> str:
    """Integer item of a list-directed record: a blank and 11 columns."""
    return " " + str(int(value)).rjust(11)


def _list_real(value: float) -> str:
    """``real*8`` item of a list-directed record: a blank and 25 columns."""
    value = float(value)
    if value != value:
        return " " + "NaN".rjust(25)
    sign = "-" if np.signbit(value) else ""
    if np.isinf(value):
        return " " + (sign + "Infinity").rjust(25)
    mag = abs(value)
    if mag == 0.0:
        return " " + (sign + "0." + "0" * 16).rjust(20) + " " * 5
    mant, _, exptext = f"{mag:.16E}".partition("E")
    exp = int(exptext)
    if -1 <= exp <= 16:
        return " " + (sign + f"{mag:#.{16 - exp}f}").rjust(20) + " " * 5
    return " " + (sign + mant + "E" + ("-" if exp < 0 else "+") + f"{abs(exp):03d}").rjust(25)


def _unit6(message: str) -> list[str]:
    """Render a message of the atomic-core routines as gfortran prints it."""
    m = _MSG_BOUND.match(message)
    if m:
        return [" " + m.group(1) + _list_int(int(m.group(2))) + _list_int(int(m.group(3)))]
    m = _MSG_NINT.match(message)
    if m:
        return [" &   nint n l                " + "".join(_list_int(int(g)) for g in m.groups()),
                " &   eigenvalue log to file err.dftseq_e",
                " &   potential  log to file err.dftseq_v"]
    m = _MSG_CTP.match(message)
    if m:
        return [" &    nint n l e                      "
                + "".join(_list_int(int(g)) for g in m.groups()[:3])
                + _list_real(float(m.group(4))),
                " &    0-order coefficient to file err.dftseq_tp"]
    return [message if message.startswith(" ") else " " + message]


def flushed_output(lines: Sequence[str], blocksize: int = 4096) -> list[str]:
    """What a redirected run leaves in the file if the program dies.

    ``pswatch -q`` redirects the output of ``pslp.x`` into the ``.test``
    file.  The run-time library (libgfortran) then buffers unit 6 and
    writes the buffer out whenever it holds at least ``blocksize`` bytes at
    the end of a record.  If the program is killed by a signal (see
    :class:`FortranCrash`) the records still in the buffer are lost.  Given
    all records written before the crash this returns the ones that reached
    the file, so that a report of the Fortran reference set can be compared
    also in that case.
    """
    done: list[str] = []
    held: list[str] = []
    used = 0
    for line in lines:
        held.append(line)
        used += len(line.encode("utf-8")) + 1
        if used >= blocksize:
            done.extend(held)
            held = []
            used = 0
    return done


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

@dataclass
class PslpControl:
    """Content of ``fort.21`` (the options of the ``pswatch`` script).

    Defaults are those of ``pslp.f``; :func:`pswatch` fills the fields the
    way the script does (``lmax`` from the ``.cpi`` header and so on).
    """

    l_loc: int = 2        # angular momentum of the local potential
    lbeg: int = 0         # lowest l for the logarithmic derivatives
    lend: int = 2         # highest l for the logarithmic derivatives
    lmax: int = 2         # highest l among the pseudopotentials
    rld: float = 0.e0     # diagnostic radius (<= 0: 1.3 * covalent radius)
    tlgd: bool = True     # compute logarithmic derivatives
    tkb: bool = True      # perform the ghost-state analysis
    tiwf: bool = True     # use the input wave functions as projectors
    tnrl: bool = False    # pseudopotentials are non-relativistic


@dataclass
class Fort31Data:
    """Pseudopotentials as ``pslp`` reads them from unit 31."""

    mmax: int
    amesh: float
    r: np.ndarray             # radial mesh as read (all points of the file)
    uu: np.ndarray            # pseudo wave functions [channel, mesh point]
    vbare: np.ndarray         # ionic pseudopotentials [channel, mesh point]
    dc: np.ndarray            # pseudocore density (zero if absent)
    dcp: np.ndarray
    dcpp: np.ndarray
    core_points: int = 0      # number of pseudocore records read


def _text(source: str | Path | None) -> str | None:
    """Content of a file given as path, or the text itself (has a newline).

    The empty string stands for an empty file.
    """
    if source is None:
        return None
    if isinstance(source, str) and source == "":
        return ""
    if isinstance(source, Path) or "\n" not in str(source):
        return Path(source).read_text(encoding="utf-8")
    return str(source)


def cpi_lmax(cpi_text: str) -> int:
    """``awk 'NR == 1 {print $2-1}'``: number of potentials minus one."""
    first = cpi_text.split("\n", 1)[0].split()
    try:
        return int(float(first[1]) - 1)
    except (IndexError, ValueError):
        return -1


def _bad_cpi(lines: list[str], ll: int, mmax: int, i: int) -> FortranStop:
    lines.append(" & pslp - stop: bad pseudopotential file fort.31")
    lines.append(" &        current l,mmax,i " + _list_int(ll - 1) + _list_int(mmax)
                 + _list_int(i))
    exc = FortranStop("pslp - stop: bad pseudopotential file fort.31")
    exc.messages = list(lines)
    return exc


def read_cpi(cpi_text: str, llmx: int, tnlc: bool = False,
             lines: list[str] | None = None) -> tuple[Fort31Data, bool]:
    """Read the pseudopotential file like ``pslp.f`` reads unit 31.

    ``cpi_text`` is the complete ``.cpi`` file; the script passes it on
    without its first 11 lines (``tail -n +12``).  ``llmx`` potentials are
    read, then the pseudocore density if present.  Returns ``(data, tnlc)``;
    ``tnlc`` becomes True if a pseudocore is found.  Bad files raise
    :class:`FortranStop` after the messages of the Fortran were appended to
    ``lines``.
    """
    if lines is None:
        lines = []
    records = cpi_text.split("\n")[11:]
    if records and records[-1] == "":
        records.pop()
    pos = 0
    nrec = len(records)
    mmax = MX
    amesh = 0.e0
    r = np.zeros(0)
    uu_rows: list[np.ndarray] = []
    vb_rows: list[np.ndarray] = []
    i = 0
    for ll in range(1, llmx + 1):
        # read(31,*) mmax,amesh ; then mmax records iunused,r,uu,vbare
        try:
            if pos >= nrec:
                raise ValueError
            head = records[pos].split()
            mmax_new = int(head[0])
            amesh = float(head[1])
            mmax = mmax_new
            pos += 1
            if mmax > MX or mmax < 2 or pos + mmax > nrec:
                raise ValueError
            block = _real_block(records[pos:pos + mmax])
            if block.shape != (mmax, 4):
                raise ValueError
            pos += mmax
            i = mmax + 1
        except (ValueError, IndexError):
            # the fast path failed: repeat the block record by record with
            # the list-directed reader to find where the Fortran stops
            raise _slow_cpi_error(records, ll, llmx, lines) from None
        r = block[:, 1].copy()
        uu_rows.append(block[:, 2].copy())
        vb_rows.append(block[:, 3].copy())
        with np.errstate(all="ignore"):
            if abs(amesh - float(r[1]) / float(r[0])) > float(np.float32(1e-10)):
                raise _bad_cpi(lines, ll, mmax, i)
    # pseudocore density: read(31,*,end=60,err=60) r(i),dc(i),dcp(i),dcpp(i)
    dc = np.zeros(mmax)
    dcp = np.zeros(mmax)
    dcpp = np.zeros(mmax)
    ncore = 0
    for i in range(mmax):
        if pos >= nrec:
            break
        try:
            vals = [float(t) for t in records[pos].split()[:4]]
            if len(vals) < 4:
                raise ValueError
        except ValueError:
            vals = _slow_core_record(records[pos])
            if vals is None:
                break
        pos += 1
        r[i], dc[i], dcp[i], dcpp[i] = vals
        ncore += 1
    if tnlc and ncore == 0:
        lines.append(" & pslp - stop: bad pseudopotential file fort.31")
        lines.append(" &        error reading pseudo core density")
        exc = FortranStop("pslp - stop: bad pseudopotential file fort.31")
        exc.messages = list(lines)
        raise exc
    if not tnlc and ncore > 0:
        lines.append(" & pslp - pseudocore present (fort.31)")
        tnlc = True
    data = Fort31Data(mmax=mmax, amesh=amesh, r=r, uu=np.array(uu_rows), vbare=np.array(vb_rows),
                   dc=dc, dcp=dcp, dcpp=dcpp, core_points=ncore)
    return data, tnlc


def _real_block(records: list[str]) -> np.ndarray:
    """Four numbers per record (``iunused, r, uu, vbare``) as a float array.

    Fortran output of numbers with a three-digit exponent has no exponent
    letter (``0.41091483741069-101``); such records take the slower path
    through :func:`parse_real`.
    """
    try:
        return np.array([[float(t) for t in rec.split()[:4]] for rec in records])
    except ValueError:
        rows = []
        for rec in records:
            tok = rec.split()[:4]
            int(tok[0])
            rows.append([float(tok[0])] + [parse_real(t) for t in tok[1:]])
        return np.array(rows)


def _slow_core_record(record: str) -> list[float] | None:
    try:
        vals = ListReader(record).read("rrrr")
    except (FortranReadError, FortranEndOfFile):
        return None
    if None in vals:
        return None
    return [float(v) for v in vals]


def _slow_cpi_error(records: list[str], ll_fail: int, llmx: int, lines: list[str]) -> FortranStop:
    """Locate the failing record of unit 31 with the list-directed reader."""
    reader = ListReader("\n".join(records) + "\n")
    mmax = MX
    i = 0
    for ll in range(1, llmx + 1):
        try:
            head = reader.read("ir")
            if head[0] is not None:
                mmax = head[0]
            i = 1
            while i <= mmax:
                reader.read("irrr")
                i += 1
        except (FortranReadError, FortranEndOfFile):
            return _bad_cpi(lines, ll, mmax, i)
    return _bad_cpi(lines, ll_fail, mmax, i)


def read_aep(aep_text: str | None, mmax: int, amesh: float, iexc: int, irl: int,
             lines: list[str]) -> np.ndarray:
    """All-electron potential from unit 37 (``<prefix>.aep``).

    Returns ``vae`` with at least ``mmax`` elements (all points of the file
    are kept: the relativistic ``dftseq`` reads two points beyond ``mmax``).
    Mismatch and error messages are appended to ``lines`` as in ``pslp.f``.
    """
    vae = np.zeros(mmax + 2)
    terr = True
    records = [] if aep_text is None else aep_text.split("\n")
    if records and records[-1] == "":
        records.pop()
    try:
        if not records:
            raise FortranEndOfFile()
        head = ListReader(records[0] + "\n").read("riii")
        if None in head:
            raise FortranReadError("null value")
        _, mmaxtmp, iexcae, irlae = head
    except (FortranReadError, FortranEndOfFile):
        lines.append(" & pslp - warning: bad/missing full potential file fort.37")
        return vae
    if iexc != iexcae:
        lines.append(" & pslp - XC-type mismatch - using IEXC" + _list_int(iexc))
    if irlae != irl:
        lines.append(" & pslp - Relativity type mismatch - using IRL" + _list_int(irl))
    nread = 0
    rxx = [0.e0, 0.e0]
    values = []
    for rec in records[1:1 + max(mmaxtmp, 0)]:
        tok = rec.split()
        try:
            rv, vv = float(tok[0]), float(tok[1])
        except (ValueError, IndexError):
            try:
                rv, vv = ListReader(rec + "\n").read("rr")
                if rv is None or vv is None:
                    break
            except (FortranReadError, FortranEndOfFile):
                break
            rv, vv = float(rv), float(vv)
        if nread < 2:
            rxx[nread] = rv
        values.append(vv)
        nread += 1
    if nread == mmaxtmp:
        terr = False
    if len(values) > len(vae):
        vae = np.zeros(len(values))
    vae[:len(values)] = values
    with np.errstate(all="ignore"):
        ratio = float(np.float64(rxx[1]) / np.float64(rxx[0]))
    if abs(amesh - ratio) > float(np.float32(5e-6)):
        lines.append(" & pslp - warning: grid from unit fort.37 incompatible")
    if terr:
        lines.append(" & pslp - warning: bad/missing full potential file fort.37")
    return vae


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class PseudoAtom:
    """Self-consistent pseudo-atom of ``pslp`` (Hartree atomic units).

    State arrays (``n, l, f, e, ep``) run over the ``nv`` valence states in
    input order; ``n`` is the principal quantum number of the pseudo state
    (``l + 1`` for the lowest state of each ``l``).  Channel arrays (``uu``,
    ``vbare``) run over ``l = 0 .. lmax``.
    """

    zfull: float              # nuclear charge
    symz: str                 # chemical symbol
    iexc: int
    symxc: str
    igr: int                  # 1: gradient-corrected functional
    cval: float               # number of valence electrons
    nv: int                   # number of valence states
    tnlc: bool                # nonlinear core-valence XC
    lmax: int                 # highest l of the input pseudopotentials
    mmax: int                 # mesh points used (odd, r < 80 bohr)
    amesh: float
    r: np.ndarray
    n: list[int]
    l: list[int]
    f: list[float]
    e: np.ndarray             # eigenvalues (Hartree); 0: no bound state
    ep: np.ndarray            # potential energy of the occupied states
    it: int                   # number of iterations (itmx+1: not converged)
    converged: bool
    de: float                 # "convergence" printed in the report
    nin: int                  # outermost mesh point of the wave functions
    etot: float               # total energy
    ekin: float               # kinetic energy
    evps: float               # ionic pseudopotential energy
    eeel: float               # electron-electron interaction (2 * Hartree energy)
    ex: float                 # exchange energy
    ec: float                 # correlation energy
    evps_loc: float           # local potential energy (depends on l_loc)
    evxc: float               # xc potential energy
    cexc: float | None        # xc energy core   (nonlinear core correction only)
    vexc: float | None        # xc energy valence
    tc_val: float             # integrated valence density
    tc_core: float | None     # integrated core density
    t1c: float | None         # ... 1st derivative test
    t2c: float | None         # ... 2nd derivative test
    r_equidensity: float | None   # estimated equidensity radius
    vi: np.ndarray            # screening potential
    vi_prev: np.ndarray       # screening potential the last iteration started from
    rho: np.ndarray           # valence density (times 4 pi)
    rhop: np.ndarray
    rhopp: np.ndarray
    ves: np.ndarray           # Hartree potential of the valence density
    vxc: np.ndarray           # xc potential of valence + core density
    ups: np.ndarray           # calculated wave functions [state, mesh point]
    uu: np.ndarray            # input wave functions [channel, mesh point]
    vbare: np.ndarray         # ionic pseudopotentials [channel, mesh point]
    dc: np.ndarray            # pseudocore density
    dcp: np.ndarray
    dcpp: np.ndarray
    ntl: list[int]            # state index (1-based, 0: none) of each channel
    lines: list[str] = field(default_factory=list)     # its part of the report

    @property
    def ehartree(self) -> float:
        return 0.5 * self.eeel

    @property
    def exc(self) -> float:
        return self.ex + self.ec

    @property
    def e_eV(self) -> np.ndarray:
        return np.asarray(self.e) * RY2


@dataclass
class GhostResult:
    """Pseudo-atom and ghost-state analysis for one local channel."""

    l_loc: int
    atom: PseudoAtom
    channels: list[KbChannel]
    lines: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ghost_free(self) -> bool:
        """True if every projector channel has the verdict 'no ghost'."""
        return all(ch.verdict == "no ghost" for ch in self.channels)

    @property
    def verdicts(self) -> dict[int, str]:
        return {ch.l: ch.verdict for ch in self.channels}


@dataclass
class PswatchResult:
    """What ``pswatch`` (``pslp.x``) produced."""

    control: PslpControl
    atom: PseudoAtom
    lines: list[str]                      # unit 6, line by line
    info: str                             # the '#info' line of the script
    stage: str                            # 'pseudoatom' (-ao) or 'tested'
    channels: list[KbChannel] = field(default_factory=list)
    rld: float = 0.0                      # diagnostic radius asked for
    ild: int = 0                          # mesh index of the radius used
    r_diag: float = 0.0                   # radius used (printed in the report)
    logder: list[LogDerivative] = field(default_factory=list)
    kinetic: list[KinkonResult] = field(default_factory=list)
    fort38: str = ""                      # pseudo wave functions (unit 38)
    cpo: str | None = None                # text of the pswatch -cpo file
    stop: str | None = None               # stop message, if the program stopped
    files: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def report(self) -> str:
        """Content of ``<prefix>.test``: ``#info`` line plus unit 6."""
        return self.info + "\n" + self.text

    @property
    def text(self) -> str:
        """Unit 6 (the report without the ``#info`` line)."""
        return "".join(line + "\n" for line in self.lines)

    @property
    def ghost_free(self) -> bool:
        return all(ch.verdict == "no ghost" for ch in self.channels)

    @property
    def verdicts(self) -> dict[int, str]:
        return {ch.l: ch.verdict for ch in self.channels}


# --------------------------------------------------------------------------
# main program
# --------------------------------------------------------------------------

@dataclass
class _Setup:
    """Variables of ``program pslp`` before the pseudo-atom is solved."""

    ctl: PslpControl
    zfull: float
    nc: int
    nv: int
    iexc: int
    rnlc: float
    irl: int
    igr: int
    tnlc: bool
    cval: float
    norb: int
    n: list[int]
    l: list[int]
    f: list[float]
    lmax: int
    l_loc: int
    lbeg: int
    lend: int
    mmax: int
    amesh: float
    al: float
    r: np.ndarray
    uu: np.ndarray
    vbare: np.ndarray
    dc: np.ndarray
    dcp: np.ndarray
    dcpp: np.ndarray
    lines: list[str]


def _setup(ctl: PslpControl, ini_text: str, cpi_text: str) -> _Setup:
    """``pslp.f`` from the first statement to the restricted radial mesh."""
    lines: list[str] = []
    zfull = 0.e0
    nc = nv = iexc = 0
    rnlc = 0.e0
    np_: list[int] = []
    lp: list[int] = []
    fp: list[float] = []
    reader = ListReader(ini_text)
    try:
        head = reader.read("riiir")
        if None in head:
            raise FortranReadError("null value")
        zfull, nc, nv, iexc, rnlc = head
        zfull = abs(zfull)
        nc = abs(nc)
        nv = abs(nv)
        if nc + nv > MS:
            raise FortranStop(f"pslp - data: more than ms = {MS} states")
        for _ in range(nc + nv):
            rec = reader.read("iir")
            if None in rec:
                raise FortranReadError("null value")
            ni, li, fi = rec
            fi = abs(fi)
            np_.append(ni)
            lp.append(li)
            fp.append(fi)
            if ni <= li:
                raise FortranStop("pslp - data: l .le. n")
            if li > 3:
                raise FortranStop("pslp - data: l .gt. 3")
    except (FortranReadError, FortranEndOfFile):
        # label 14: the program goes on with what it has (zeros otherwise)
        np_ += [0] * (nc + nv - len(np_))
        lp += [0] * (nc + nv - len(lp))
        fp += [0.e0] * (nc + nv - len(fp))

    irl = 2 if ctl.tnrl else 1
    tnlc = abs(rnlc) > 0.0
    iexc = abs(iexc)
    if iexc == 0:
        iexc = 8
    igr = IGRMAP[iexc - 1] if iexc <= len(IGRMAP) else 0

    lmax = max(0, ctl.lmax)
    l_loc = max(0, min(ctl.l_loc, lmax))
    lbeg = max(0, ctl.lbeg)
    lend = max(lbeg, min(ctl.lend, 5))
    llmx = lmax + 1

    n = [ll for ll in range(1, llmx + 1)]
    l = [ll - 1 for ll in range(1, llmx + 1)]
    f = [0.e0] * llmx
    cval = 0.e0
    norb = 0
    for i in range(nc, nc + nv):
        norb += 1
        if norb > len(n):
            n.append(0)
            l.append(0)
            f.append(0.e0)
        nn = lp[i] + 1
        for j in range(norb - 1):
            if l[j] == lp[i]:
                nn += 1
        n[norb - 1] = nn
        l[norb - 1] = lp[i]
        f[norb - 1] = fp[i]
        cval = cval + fp[i]

    # pseudopotentials, radial meshs assumed to be the same throughout
    cpi, tnlc = read_cpi(cpi_text, llmx, tnlc, lines)

    # restrict range of radial mesh
    r = cpi.r
    amesh = float(r[1]) / float(r[0])
    al = libm.slog(amesh)
    mmax = cpi.mmax
    below = np.nonzero(r[:mmax] < RMX)[0]
    mmax = int(below[-1]) + 1 if below.size else 0
    if mmax % 2 == 0:
        mmax -= 1
    return _Setup(ctl=ctl, zfull=zfull, nc=nc, nv=nv, iexc=iexc, rnlc=rnlc, irl=irl,
                  igr=igr, tnlc=tnlc, cval=cval, norb=norb, n=n, l=l, f=f, lmax=lmax,
                  l_loc=l_loc, lbeg=lbeg, lend=lend, mmax=mmax, amesh=amesh, al=al,
                  r=r[:mmax].copy(), uu=cpi.uu[:, :mmax].copy(),
                  vbare=cpi.vbare[:, :mmax].copy(), dc=cpi.dc[:mmax].copy(),
                  dcp=cpi.dcp[:mmax].copy(), dcpp=cpi.dcpp[:mmax].copy(), lines=lines)


def _line608(label: str, value: float) -> str:
    return fmt_a(label, 30) + "  " + fmt_f(value, 12, 5)


def _pseudoatom(s: _Setup, l_loc: int, itmx: int = ITMX,
                trace: Callable[[dict], None] | None = None) -> PseudoAtom:
    """Self-consistent pseudo-atom, its energies and its part of the report."""
    lines = list(s.lines)

    def warn(message: str) -> None:
        lines.extend(_unit6(message))

    mmax, r, al = s.mmax, s.r, s.al
    norb, n, l, f = s.norb, s.n, s.l, s.f
    cval, igr, iexc, tnlc = s.cval, s.igr, s.iexc, s.tnlc
    llmx = s.lmax + 1
    ll_loc = l_loc + 1
    vbare = s.vbare
    dc, dcp, dcpp = s.dc, s.dcp, s.dcpp

    # self-consistent pseudoatom: initial potential and eigenvalues
    q = max(1.e0, float(np.sqrt(cval)))
    vi = -q / r
    e0 = [-0.5e0 * q] * norb
    vorb = np.zeros((max(norb, llmx), mmax))
    for i in range(norb):
        if l[i] + 1 <= llmx:
            vorb[i] = vbare[l[i]]
        else:
            vorb[i] = vbare[ll_loc - 1]

    try:
        ps = psatom(itmx, igr, iexc, 0.e0, 0, norb, n, l, f, e0, mmax, r, vi, vorb[:norb],
                    dc, dcp, dcpp, warn=warn, trace=trace)
    except FortranStop as exc:
        exc.messages = list(lines)
        raise
    it, de, nin = ps.it, ps.de, ps.mup
    e = ps.e
    vi = ps.vi
    rho, rhop, rhopp = ps.rho, ps.rhop, ps.rhopp
    ups = ps.uu

    with np.errstate(all="ignore"):
        # total charges
        tc_val = fmom(0, mmax, al, 1.e0, r, rho)
        tc_core = t1c = t2c = None
        if tnlc:
            tc_core = fmom(0, mmax, al, 1.e0, r, dc)
            if igr == 1:
                t1c = -fmom(1, mmax, al, tc_core, r, dcp) / 3.e0
                t2c = fmom(2, mmax, al, tc_core, r, dcpp) / 12.e0

        # total energy components: hartree energy
        ves, eeel = vestat(mmax, cval, r, rho, True)

        # xc energy
        if tnlc:
            rho = rho + dc
            rhop = rhop + dcp
            rhopp = rhopp + dcpp
        xc = vexcor(iexc, mmax, r, rho, rhop, rhopp, True, ec_in=0.0)
        vxc, ex, ec = xc.vpxc, xc.ex, xc.ec
        if tnlc:
            rho = rho - dc
            rhop = rhop - dcp
            rhopp = rhopp - dcpp
        evxc = dmelm(mmax, al, r, vxc, rho)
        vexc = dmelm(mmax, al, r, xc.dexc, rho)
        cexc = ex + ec - vexc

        # ionic pseudopotential energy
        ep = np.zeros(max(norb, llmx))
        evps = 0.e0
        for i in range(norb):
            if f[i] > 0.0:
                ep[i] = gltfmv(mmax, al, r, ups[i], vorb[i], ups[i])
                evps = evps + f[i] * float(ep[i])
        # local pseudopotential energy
        evps_loc = dmelm(mmax, al, r, vbare[ll_loc - 1], rho)
        # screened pseudopotential energy
        epsp = evps + dmelm(mmax, al, r, vi, rho)
        # kinetic energy from eigenvalues
        ekin = dspv(1, norb, f, e) - epsp
        etot = ekin + .5 * eeel + evps + ex + ec

    symz, symxc = labelmap(int(s.zfull), iexc)
    out = lines.append
    out("")
    out("fhi pseudopotential tool pslp - version " + fmt_a(SREV.ljust(10), 10))
    out("")
    out(fmt_a("chemical symbol", 30) + "  " + fmt_a(symz, 2))
    out(fmt_a("nuclear charge", 30) + "  " + fmt_f(s.zfull, 5, 2))
    out(fmt_a("number of valence electrons", 30) + "  " + fmt_f(cval, 5, 2))
    out(fmt_a("number of valence states", 30) + "  " + fmt_i(s.nv, 2))
    out(fmt_a("exchange-correlation model", 30) + "  " + fmt_i(iexc, 2) + "  " + fmt_a(symxc, 30))
    if tnlc:
        out(fmt_a("nonlinear core-valence XC", 30))
    out(fmt_a("parameters radial mesh", 30) + "  " + fmt_i(mmax, 4) + fmt_f(s.amesh, 12, 6)
        + "  " + fmt_e(r[0], 12, 6))
    _, sl, _ = labels(1, llmx - 1, 1)
    out(fmt_a("input pseudopotentials up to", 30) + "   " + sl)
    out("")
    out(" " * 10 + "=== pseudo atom (Hartree a.u.) ===")
    out("")
    out("<        n     l   occupation  eigenvalue(eV)  potential energy")
    for i in range(s.nv):
        out("< " + fmt_i(i + 1, 2) + "    " + fmt_i(n[i], 2) + "    " + fmt_i(l[i], 2) + "    "
            + fmt_f(f[i], 8, 4) + "  " + fmt_f(float(e[i]) * RY2, 12, 4) + "    "
            + fmt_f(float(ep[i]), 12, 5))
    out("")
    out(_line608("total energy", etot))
    out(_line608("kinetic energy", ekin))
    out(_line608("ionic pseudopotential energy", evps))
    out(_line608("hartree energy", 0.5 * eeel))
    out(_line608("xc energy", ex + ec))
    out(_line608("local potential energy", evps_loc))
    out(_line608("xc potential energy", evxc))
    if tnlc:
        out(_line608("xc energy core", cexc))
        out(_line608("xc energy valence", vexc))
    out(_line608("integrated valence density", tc_val))
    out(fmt_a("number of iterations", 30) + "  " + fmt_i(it, 12) + "  " + fmt_a("convergence", 12)
        + fmt_es(de, 9, 1))
    r_equi = None
    if tnlc:
        out(_line608("integrated core density", tc_core))
        if igr == 1:
            out(_line608(" ... 1st derivative test", t1c))
            out(_line608(" ... 2nd derivative test", t2c))
        # do i=nin-10,1,-1: if(rho(i) .le. dc(i)) goto 590 ... prints r(i+1)
        i = nin - 10
        while i >= 1:
            if rho[i - 1] <= dc[i - 1]:
                break
            i -= 1
        r_equi = float(r[i]) if 0 <= i < mmax else float("nan")
        out(_line608("estimated equidensity radius >", r_equi))

    # calculated wave function of each channel
    ntl = []
    for ll in range(1, llmx + 1):
        k = 0
        for i in range(norb):
            if n[i] == ll:
                k = i + 1
        ntl.append(k)

    # for xvgr x y
    vl = 1.e20
    vh = -1.e20
    for ll in range(1, llmx + 1):
        bl, bh = dextv(1, mmax, vbare[ll - 1])
        vl = min(bl, vl)
        vh = max(bh, vh)
    i = _fint(vl - _fint(vl / 5) * 5) - 1 + _fint(vl / 5) * 5
    j = _fint(vh - _fint(vh / 5) * 5) + 1 + _fint(vh / 5) * 5
    ll = max(1, _idiv(j - i, 4))
    out(fmt_a("y range xvgr", 30) + " " * 6 + fmt_i(i, 4) + fmt_i(j, 4) + fmt_i(ll, 4))

    return PseudoAtom(
        zfull=s.zfull, symz=symz, iexc=iexc, symxc=symxc, igr=igr, cval=cval, nv=s.nv,
        tnlc=tnlc, lmax=s.lmax, mmax=mmax, amesh=s.amesh, r=r, n=list(n[:norb]),
        l=list(l[:norb]), f=list(f[:norb]), e=e, ep=ep[:norb].copy(), it=it,
        converged=ps.converged, de=de, nin=nin, etot=etot, ekin=ekin, evps=evps, eeel=eeel,
        ex=ex, ec=ec, evps_loc=evps_loc, evxc=evxc, cexc=cexc if tnlc else None,
        vexc=vexc if tnlc else None, tc_val=tc_val, tc_core=tc_core, t1c=t1c, t2c=t2c,
        r_equidensity=r_equi, vi=vi, vi_prev=ps.vi1, rho=rho, rhop=rhop, rhopp=rhopp,
        ves=ves, vxc=vxc,
        ups=ups, uu=s.uu, vbare=vbare, dc=dc, dcp=dcp, dcpp=dcpp, ntl=ntl, lines=lines)


def _fint(x: float) -> int:
    """Fortran ``int`` of a real (truncation); the x86 result for NaN/overflow."""
    if x != x or abs(x) >= 2147483648.0:
        return -2147483648
    return int(x)


def _idiv(a: int, b: int) -> int:
    """Fortran integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _channel_data(s: _Setup, atom: PseudoAtom, tiwf: bool) -> tuple[np.ndarray, list[float]]:
    """Projector wave functions and reference eigenvalues per channel."""
    llmx = s.lmax + 1
    uu = atom.uu.copy()
    ep = [0.e0] * llmx
    for ll in range(1, llmx + 1):
        k = atom.ntl[ll - 1]
        if k > 0:
            if not tiwf:
                uu[ll - 1] = atom.ups[k - 1]
            ep[ll - 1] = float(atom.e[k - 1])
    return uu, ep


def _cpo_text(s: _Setup, atom: PseudoAtom) -> str:
    """The file the script assembles for ``-cpo`` from ``fort.4x``/``fort.27``."""
    llmx = min(s.lmax + 1, 5)
    mmax = atom.mmax
    r = atom.r
    out = [fmt_e(s.cval, 20, 14) + "   " + str(llmx)]
    out += ["  0.0000    0.0000    0.0000   0.0000"] + ["  0.0000    .00e+00   .00e+00"] * 9
    amesh = float(r[1]) / float(r[0])
    rtext = [fmt_e(x, 20, 14) for x in r.tolist()]
    for ll in range(1, llmx + 1):
        k = atom.ntl[ll - 1]
        u = atom.ups[k - 1] if k > 0 else np.zeros(mmax)
        # awk '{print $2," "$3}': the two fields with two blanks between
        out.append(str(mmax) + "  " + fmt_e(amesh, 20, 14))
        ut = [fmt_e(x, 20, 14) for x in u[:mmax].tolist()]
        vt = [fmt_e(x, 20, 14) for x in atom.vbare[ll - 1][:mmax].tolist()]
        out += [fmt_i(i + 1, 4) + " " + rtext[i] + " " + ut[i] + " " + vt[i] for i in range(mmax)]
    if atom.tnlc:
        dct = [[fmt_e(x, 20, 14) for x in a[:mmax].tolist()] for a in (atom.dc, atom.dcp, atom.dcpp)]
        out += [rtext[i] + " " + dct[0][i] + " " + dct[1][i] + " " + dct[2][i] for i in range(mmax)]
    return "".join(line + "\n" for line in out)


def pslp(ctl: PslpControl, ini_text: str, cpi_text: str, aep_text: str | None = None, *,
         logder: bool = True, kinetic: bool = True, spectra: bool = True,
         eigensolver: Callable | None = None, itmx: int = ITMX,
         trace: Callable[[dict], None] | None = None) -> PswatchResult:
    """``program pslp``.

    ``ctl`` is the content of ``fort.21``, ``ini_text`` of ``fort.22``,
    ``cpi_text`` the complete ``.cpi`` file (``fort.31`` is its tail) and
    ``aep_text`` of ``fort.37``.  Not in the Fortran (all default to the
    Fortran behaviour): ``logder = False`` skips the calculation of the
    logarithmic-derivative curves, which do not enter the report (the row
    of a state without bound solution in the kinetic-energy table is then
    not that of the Fortran, see :mod:`.kinkon`); ``kinetic = False`` skips
    the kinetic-energy table; ``spectra = False`` the bound-state spectra.
    ``eigensolver`` replaces the port of ``dspevx`` (test harness).

    A ``stop`` of the Fortran program raises :class:`FortranStop`;
    ``exc.messages`` holds the report lines written before it and, if the
    pseudo-atom was completed, ``exc.result`` the :class:`PswatchResult` so
    far (pseudo-atom, ghost analysis ...).  Where the Fortran program is
    known to die with a segmentation fault the exception is a
    :class:`FortranCrash` (state without bound solution in the
    kinetic-energy test, see :mod:`.kinkon`; ``kinetic = False`` avoids it).
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    s = _setup(ctl, ini_text, cpi_text)
    timings["read"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    atom = _pseudoatom(s, s.l_loc, itmx=itmx, trace=trace)
    timings["psatom"] = time.perf_counter() - t1
    lines = list(atom.lines)
    out = lines.append

    def warn(message: str) -> None:
        lines.extend(_unit6(message))

    mmax, r = s.mmax, s.r
    llmx = s.lmax + 1
    n, l, f = s.n, s.l, s.f

    # radial pseudo-wavefunctions [the u(r)]
    fort38 = out38(1, s.nv, [atom.nin] * max(s.nv, 1), n, l, r, atom.ups) if s.nv > 0 else ""
    result = PswatchResult(control=ctl, atom=atom, lines=lines, info="", stage="pseudoatom",
                           fort38=fort38, timings=timings)
    if not ctl.tiwf:
        result.cpo = _cpo_text(s, atom)
    if not ctl.tlgd and not ctl.tkb:
        result.stop = "pslp - pseudoatom done"
        timings["total"] = time.perf_counter() - t0
        return result

    out("")
    out(" pslp - pseudoatom done - now testing")

    # logarithmic derivatives and ghost state test: read all-electron potential
    vae = np.zeros(mmax + 2)
    if ctl.tlgd:
        out("")
        if s.irl == 1:
            out(" --- assuming scalar-relativistic all-electron atom ---")
        elif s.irl == 2:
            out(" --- assuming nonrelativistic all-electron atom ---")
        vae = read_aep(aep_text, mmax, s.amesh, s.iexc, s.irl, lines)

    # input or calculated wavefunctions for kb projectors
    _, sl, _ = labels(1, s.l_loc, 1)
    out("")
    out(" --- " + sl + " component taken as local potential ---")
    if ctl.tkb:
        if ctl.tiwf:
            out(" --- input wavefunctions used for kb potentials ---")
        else:
            out(" --- calculated wavefunctions used for kb potentials ---")
    uu, ep = _channel_data(s, atom, ctl.tiwf)

    # diagnostic radius
    rld = ctl.rld
    if rld <= 0.0:
        rld = rcovalent(int(s.zfull)) * float(np.float32(1.3))

    # states of the kinetic-energy table, in the order of the Fortran loops
    kstates = [(ll, i) for ll in range(s.lbeg + 1, s.lend + 2) for i in range(s.nv)
               if l[i] == ll - 1 and f[i] > 0.0] if kinetic else []
    logder_mode: bool | str = bool(logder)
    projectors = ctl.tkb and spectra and any(
        ll != s.l_loc + 1 for ll in range(s.lbeg + 1, min(s.lend, s.lmax) + 2))
    if (not logder and ctl.tlgd and kstates and float(atom.e[kstates[0][1]]) == 0.0
            and not projectors):
        # the row of a state without bound solution is computed from what
        # the last dftseq call of ppcheck leaves on the stack (with
        # projector channels the Fortran program crashes there instead)
        logder_mode = "dftseq"

    t1 = time.perf_counter()
    try:
        pc = ppcheck(ctl.tnrl, ctl.tlgd, ctl.tkb, s.l_loc, s.lmax, s.lbeg, s.lend, mmax, rld,
                     ep, r, vae, atom.vi, uu, s.vbare, out=out, warn=warn, spectra=spectra,
                     logder=logder_mode, eigensolver=eigensolver)
    except FortranStop as exc:
        exc.messages = list(lines)
        exc.result = result
        raise
    timings["ppcheck"] = time.perf_counter() - t1
    result.channels = pc.channels
    result.rld, result.ild, result.r_diag = rld, pc.ild, pc.r_diag
    result.logder = pc.logder
    result.stage = "tested"

    # reciprocal space analysis (occupied states)
    if kinetic:
        t1 = time.perf_counter()
        out("")
        out(" --- kinetic energy convergence in momentum space ---")
        out("")
        out("     l  n  bracket   cutoff    norm   kinet. energy")
        out("            (eV)      (Ry)               (Hartree)")
        memory = KinkonMemory.from_ppcheck(pc.up_last, pc.klbyii_called, atom.vi_prev)
        nvb = s.vbare.shape[0]
        neff: dict[int, int] = {}
        for ll, i in kstates:
            neff.setdefault(ll, ll)
            vsl = atom.vi + (s.vbare[ll - 1] if ll <= nvb else 0.e0)
            try:
                kk = kinkon(mmax, 169.e0, 0.05e0, neff[ll], ll - 1, float(atom.e[i]), r, vsl,
                            memory=memory, warn=warn)
            except FortranStop as exc:
                exc.messages = list(lines)
                exc.result = result
                raise
            lines.extend(kk.lines)
            result.kinetic.append(kk)
            neff[ll] += 1
        timings["kinkon"] = time.perf_counter() - t1
    timings["total"] = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------
# pswatch driver script
# --------------------------------------------------------------------------

def _control(cpi_text: str, lloc: int | None, lrange: Sequence[int] | None,
             rdiag: float | str, only_atom: bool, check_ghost: bool, log_derivative: bool,
             calc_wfct: bool, nonrelativistic: bool) -> PslpControl:
    """``fort.21`` as the script writes it."""
    lmax = cpi_lmax(cpi_text)
    if lloc is not None:
        if lloc > lmax:
            raise ValueError(f"pswatch - local potential L = {lloc} out of range in the cpi file")
    else:
        lloc = lmax
    if lrange is None:
        lbeg, lend = 0, lmax
    else:
        lbeg, lend = int(lrange[0]), int(lrange[1])
    return PslpControl(
        l_loc=int(lloc), lbeg=lbeg, lend=lend, lmax=lmax, rld=float(rdiag),
        tlgd=not (only_atom or not log_derivative),
        tkb=not (only_atom or not check_ghost),
        tiwf=not calc_wfct, tnrl=bool(nonrelativistic))


def pswatch(ini: str | Path, cpi: str | Path | None = None, aep: str | Path | None = None, *,
            prefix: str = "ncpp", lloc: int | None = None,
            lrange: Sequence[int] | None = None, rdiag: float | str = 0.0,
            only_atom: bool = False, check_ghost: bool = True, log_derivative: bool = True,
            nonrelativistic: bool = False, cpo: str | Path | bool | None = None,
            workdir: str | Path = ".", write: bool = False, compute_logder: bool = True,
            kinetic: bool = True, eigensolver: Callable | None = None) -> PswatchResult:
    """Python counterpart of ``pswatch [options] -i prefix ini``.

    ``ini``, ``cpi`` and ``aep`` are paths, or the file contents if the
    string contains a newline.  Without ``cpi``/``aep`` the files
    ``<prefix>.cpi`` and ``<prefix>.aep`` in ``workdir`` are used (a missing
    ``.aep`` is not an error: the program then warns and uses a zero
    all-electron potential; ``aep=""`` says the same explicitly).

    Options of the script: ``lloc`` (``-l``, default: highest channel of the
    file), ``lrange = (lbeg, lend)`` (``-L``), ``rdiag`` (``-rl``),
    ``only_atom`` (``-ao``), ``check_ghost = False`` (``-kb``),
    ``log_derivative = False`` (``-d``), ``nonrelativistic`` (``-r``) and
    ``cpo`` (``-cpo``: build the potential table with the recalculated wave
    functions, which then also serve as projectors; a path writes the file,
    ``True`` only fills ``result.cpo``).

    With ``write`` the report is written to ``<workdir>/<prefix>.test`` like
    the script does.  ``compute_logder = False`` skips the calculation of
    the logarithmic-derivative curves (they are not part of the report);
    ``kinetic = False`` leaves out the kinetic-energy table.  The xmgr
    graphics files of the script are not produced.

    Returns a :class:`PswatchResult`.  A ``stop`` of the Fortran program
    raises :class:`FortranStop` (``exc.messages``: the report so far,
    ``exc.result``: the result so far if the pseudo-atom was completed); see
    :func:`pslp`.
    """
    workdir = Path(workdir)
    out_name = str(prefix).replace("\\", "/").split("/")[-1]
    ini_text = _text(ini)
    cpi_text = _text(cpi if cpi is not None else workdir / f"{prefix}.cpi")
    if aep is None:
        aep_path = workdir / f"{prefix}.aep"
        aep_text = aep_path.read_text(encoding="utf-8") if aep_path.is_file() else None
    else:
        aep_text = _text(aep)
    ctl = _control(cpi_text, lloc, lrange, rdiag, only_atom, check_ghost, log_derivative,
                   bool(cpo), nonrelativistic)
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - environment dependent
        user = "unknown"
    info = f"#info pyfhi98pp.pswatch {out_name} {time.strftime('%H:%M:%S %b %d %Y')} {user}"
    result = pslp(ctl, ini_text, cpi_text, aep_text, logder=compute_logder, kinetic=kinetic,
                  eigensolver=eigensolver)
    result.info = info
    if write:
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / f"{out_name}.test"
        path.write_text(result.report, encoding="utf-8", newline="\n")
        result.files[path.name] = path
    if cpo and not isinstance(cpo, bool) and result.cpo is not None:
        path = Path(cpo)
        if not path.is_absolute():
            path = workdir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.cpo, encoding="utf-8", newline="\n")
        result.files[path.name] = path
    return result


# --------------------------------------------------------------------------
# fast ghost-state test
# --------------------------------------------------------------------------

def _ghost(s: _Setup, atom: PseudoAtom, l_loc: int, spectra: bool,
           eigensolver: Callable | None) -> GhostResult:
    lines: list[str] = []

    def warn(message: str) -> None:
        lines.extend(_unit6(message))

    uu, ep = _channel_data(s, atom, True)
    pc = ppcheck(False, False, True, l_loc, s.lmax, 0, s.lmax, s.mmax, 0.e0, ep, s.r,
                 np.zeros(s.mmax + 2), atom.vi, uu, s.vbare, out=lines.append, warn=warn,
                 spectra=spectra, logder=False, eigensolver=eigensolver)
    return GhostResult(l_loc=l_loc, atom=atom, channels=pc.channels, lines=lines)


def ghost_scan(ini: str | Path, cpi: str | Path, llocs: Iterable[int] | None = None, *,
               spectra: bool = False, eigensolver: Callable | None = None
               ) -> dict[int, GhostResult]:
    """Ghost-state analysis for several local channels.

    Runs the self-consistent pseudo-atom and the Kleinman-Bylander analysis
    of ``ppcheck`` (no logarithmic derivatives, no kinetic-energy test) for
    every local channel in ``llocs`` (default: all channels of the file).
    The numbers are those of the full report for the same local channel.
    The pseudo-atom does not depend on the local channel unless a valence
    state has an angular momentum above the highest pseudopotential, so it
    is solved once and shared in the usual case.  ``spectra = True`` adds
    the bound-state spectra in the Laguerre basis (``klbyii``), which the
    verdicts do not need.

    Returns ``{l_loc: GhostResult}``.
    """
    ini_text = _text(ini)
    cpi_text = _text(cpi)
    lmax = cpi_lmax(cpi_text)
    todo = list(range(lmax + 1)) if llocs is None else [int(x) for x in llocs]
    results: dict[int, GhostResult] = {}
    setup = None
    shared: PseudoAtom | None = None
    for l_loc in todo:
        if l_loc > lmax:
            raise ValueError(f"pswatch - local potential L = {l_loc} out of range in the cpi file")
        t0 = time.perf_counter()
        if setup is None:
            ctl = PslpControl(l_loc=l_loc, lbeg=0, lend=lmax, lmax=lmax, tlgd=False)
            setup = _setup(ctl, ini_text, cpi_text)
        ll_dep = any(setup.l[i] + 1 > setup.lmax + 1 for i in range(setup.norb))
        t1 = time.perf_counter()
        if shared is None or ll_dep:
            atom = _pseudoatom(setup, l_loc)
            if not ll_dep:
                shared = atom
        else:
            # only the local potential energy depends on the local channel
            with np.errstate(all="ignore"):
                evps_loc = dmelm(setup.mmax, setup.al, setup.r, setup.vbare[l_loc], shared.rho)
            atom = PseudoAtom(**{**shared.__dict__, "evps_loc": evps_loc,
                                 "lines": _relabel_local(shared.lines, evps_loc)})
        t2 = time.perf_counter()
        res = _ghost(setup, atom, max(0, min(l_loc, setup.lmax)), spectra, eigensolver)
        res.timings = {"read+psatom": t2 - t0, "psatom": t2 - t1,
                       "ppcheck": time.perf_counter() - t2, "total": time.perf_counter() - t0}
        results[l_loc] = res
    return results


def _relabel_local(lines: list[str], evps_loc: float) -> list[str]:
    label = fmt_a("local potential energy", 30)
    return [_line608("local potential energy", evps_loc) if line.startswith(label) else line
            for line in lines]


def ghost_check(ini: str | Path, cpi: str | Path, lloc: int | None = None, *,
                spectra: bool = False, eigensolver: Callable | None = None) -> GhostResult:
    """Ghost-state analysis for one local channel (default: the highest).

    Pseudo-atom plus Kleinman-Bylander analysis only; see :func:`ghost_scan`.
    """
    cpi_text = _text(cpi)
    if lloc is None:
        lloc = cpi_lmax(cpi_text)
    return ghost_scan(ini, cpi_text, [lloc], spectra=spectra, eigensolver=eigensolver)[lloc]
