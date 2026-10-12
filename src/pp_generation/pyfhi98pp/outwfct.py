"""Output subroutines of the pseudopotential program.

Port of ``outcore``, ``outpot`` and ``out39`` of ``outwfct.f`` (``out38`` is
in :mod:`.psgen`).  Each function returns the text of the unit file(s) the
Fortran routine writes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .fortran_io import fmt_a, fmt_e, fmt_f, fmt_i
from .labels import labels

__all__ = ["outcore", "outpot", "out39", "e20_rows"]


def _e20(values) -> list[str]:
    """``e20.14`` fields of a float array."""
    return [fmt_e(value, 20, 14) for value in np.asarray(values, dtype=float).tolist()]


def e20_rows(*columns) -> list[str]:
    """Records of the format ``e20.14,n(1x,e20.14)`` for equally long arrays."""
    cols = [_e20(col) for col in columns]
    return [" ".join(row) for row in zip(*cols)]


def outcore(mx: int, mstep: int, r: np.ndarray, dc: np.ndarray, dcp: np.ndarray,
            dcpp: np.ndarray) -> str:
    """Density and its derivatives, ``outcore(iu,mx,mstep,r,dc,dcp,dcpp)``.

    Records ``r, dc, dcp, dcpp`` (format ``e20.14,3(1x,e20.14)``) for the
    points ``1, 1+mstep, ...`` up to ``mx``.
    """
    sl = slice(0, mx, mstep)
    return "".join(line + "\n" for line in e20_rows(r[sl], dc[sl], dcp[sl], dcpp[sl]))


def outpot(mmax: int, llmx: int, cval: float, rc: Sequence[float], r: np.ndarray,
           ups: np.ndarray, vps: np.ndarray, vscr: np.ndarray
           ) -> tuple[list[str], list[str]]:
    """Pseudopotentials and wave functions, ``outpot`` of ``outwfct.f``.

    Returns ``(ionic, screened)``: the texts of the units ``40+l`` (index,
    radius, pseudo wave function, ionic pseudopotential) and ``45+l``
    (radius, screened pseudopotential) for ``l = 0 .. llmx-1``.  ``ups``,
    ``vps`` and ``vscr`` are arrays ``[channel, mesh point]``.
    """
    amesh = float(r[1]) / float(r[0])
    ionic, screened = [], []
    rtext = _e20(r[:mmax])
    index = [fmt_i(i, 4) for i in range(1, mmax + 1)]
    for ll in range(1, llmx + 1):
        head = ("# " + fmt_i(mmax, 5) + " " + fmt_e(amesh, 20, 14) + " " + fmt_i(ll - 1, 1)
                + " " + fmt_f(rc[ll - 1], 8, 4) + " " + fmt_e(cval, 20, 14) + "\n")
        utext = _e20(ups[ll - 1][:mmax])
        vtext = _e20(vps[ll - 1][:mmax])
        stext = _e20(vscr[ll - 1][:mmax])
        ionic.append(head + "".join(f"{i} {a} {b} {c}\n"
                                    for i, a, b, c in zip(index, rtext, utext, vtext)))
        screened.append(head + "".join(f"{a} {b}\n" for a, b in zip(rtext, stext)))
    return ionic, screened


def out39(lmax: int, ninu: Sequence[int], np_: Sequence[int], rc: Sequence[float],
          r: np.ndarray, ups: np.ndarray, uae: np.ndarray) -> str:
    """Pseudo and all-electron valence wave functions (unit 39), ``out39``.

    ``ninu, np_, rc`` are indexed by channel (``l``), ``ups`` and ``uae``
    are arrays ``[channel, mesh point]``.  As in the Fortran, a block ends
    after the first printed point whose successor vanishes, and the point
    after the last mesh point counts as zero.
    """
    lines: list[str] = []
    step = max(ninu[lmax] // 100, 1)
    for l1 in range(1, lmax + 2):
        u = np.asarray(ups[l1 - 1], dtype=float)
        ua = np.asarray(uae[l1 - 1], dtype=float)
        nin = ninu[l1 - 1]
        sn, sl, _ = labels(l1, l1 - 1, 1)
        ip = 1
        if float(ua[nin - 5]) * float(u[nin - 5]) <= 0.0:
            ip = -1
        lines.append(fmt_a("##", 3) + fmt_a(sn + sl, 3) + fmt_i(l1, 2) + fmt_i(l1, 2)
                     + fmt_i(l1 - 1, 2) + " " + fmt_f(rc[l1 - 1], 6, 3))
        for i in range(0, nin, step):
            lines.append(fmt_e(r[i], 12, 6) + " " + fmt_e(u[i], 12, 6))
            if i + 1 >= len(u) or u[i + 1] == 0.0:
                break
        npl = np_[l1 - 1]
        sn, sl, _ = labels(npl, l1 - 1, 1)
        npl = npl if 1 <= npl <= 6 else 7           # labels() overwrites n
        lines.append(fmt_a("# ", 3) + fmt_a(sn + sl, 3) + fmt_i(l1, 2) + fmt_i(npl, 2)
                     + fmt_i(l1 - 1, 2) + " " + fmt_a("ae", 2))
        for i in range(0, nin, step):
            lines.append(fmt_e(r[i], 12, 6) + " " + fmt_e(ip * float(ua[i]), 12, 6))
            if i + 1 >= len(ua) or ua[i + 1] == 0.0:
                break
    return "".join(line + "\n" for line in lines)
