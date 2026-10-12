"""List-directed output (``write(unit,*)``) as produced by gfortran.

``fhipp.x`` writes its messages and the monitoring file of the pseudocore
(unit 24, appended to ``<prefix>.dat`` by the ``psgen`` script) with
list-directed ``write`` statements.  gfortran renders them as follows:

* every record starts with one blank;
* the items are separated by one blank, except that a character item
  directly following another character item is not separated;
* ``integer`` items are right-justified in 11 columns, ``logical`` items are
  ``T`` or ``F``, character items are written as they are;
* ``real*8`` items take 25 columns with 17 significant digits: fixed-point
  notation followed by five blanks if ``0.1 <= |x| < 1e17`` (and for zero),
  otherwise ``d.ddddddddddddddddE+ddd``;
* ``real*4`` items take 16 columns with 9 significant digits in the same way
  (fixed-point notation followed by four blanks, or ``d.ddddddddE+dd``).
"""

from __future__ import annotations

import math
import re

import numpy as np

__all__ = ["list_real", "list_real4", "list_int", "list_record", "unit6_lines"]


def _list_float(value: float, digits: int, width: int, expw: int) -> str:
    if value != value:
        return "NaN".rjust(width)
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if math.isinf(value):
        return (sign + "Infinity").rjust(width)
    tail = " " * (expw + 2)
    mag = abs(value)
    if mag == 0.0:
        return (sign + "0." + "0" * (digits - 1)).rjust(width - len(tail)) + tail
    mant, _, exptext = f"{mag:.{digits - 1}E}".partition("E")
    exp = int(exptext)
    if -1 <= exp <= digits - 1:
        text = f"{mag:#.{digits - 1 - exp}f}"
        return (sign + text).rjust(width - len(tail)) + tail
    return (sign + mant + "E" + ("-" if exp < 0 else "+") + f"{abs(exp):0{expw}d}").rjust(width)


def list_real(value: float) -> str:
    """A ``real*8`` item of a list-directed record (25 columns)."""
    return _list_float(float(value), 17, 25, 3)


def list_real4(value: float) -> str:
    """A ``real*4`` item of a list-directed record (16 columns)."""
    return _list_float(float(np.float32(value)), 9, 16, 2)


def list_int(value: int) -> str:
    """A default ``integer`` item of a list-directed record (11 columns)."""
    return str(int(value)).rjust(11)


def list_record(*items: object) -> str:
    """One list-directed output record.

    ``str`` items are character items, ``bool`` logicals, ``int`` default
    integers, ``float`` ``real*8`` and ``numpy.float32`` ``real*4`` values.
    """
    out = [" "]
    previous_char = False
    for k, item in enumerate(items):
        is_char = isinstance(item, str)
        if k and not (is_char and previous_char):
            out.append(" ")
        if is_char:
            out.append(item)
        elif isinstance(item, (bool, np.bool_)):
            out.append("T" if item else "F")
        elif isinstance(item, (int, np.integer)):
            out.append(list_int(int(item)))
        elif isinstance(item, np.float32):
            out.append(list_real4(float(item)))
        else:
            out.append(list_real(float(item)))
        previous_char = is_char
    return "".join(out)


# --------------------------------------------------------------------------
# messages of the atomic-core routines
# --------------------------------------------------------------------------

_NINT = re.compile(r"^&   nint n l\s+(-?\d+) (-?\d+) (-?\d+)$")
_CTP = re.compile(r"^&    nint n l e\s+(-?\d+) (-?\d+) (-?\d+) (\S+)$")
_BOUND = re.compile(r"^(& dftseq - no bound state found \((?:iter|ctp)\): e=>0 n l) (-?\d+) (-?\d+)$")


def unit6_lines(message: str) -> list[str]:
    """Render a message for the error unit the way the Fortran prints it.

    The atomic-core routines (``dftseq``, ``fmom`` ...) pass their
    messages on as plain text; this gives them the list-directed layout of
    gfortran (leading blank, 11-column integers, 25-column reals) and adds
    the lines announcing the ``err.dftseq_*`` dump files that the Fortran
    prints before it stops (the port does not write these files).  Lines
    that already start with a blank are returned unchanged.
    """
    if message.startswith(" "):
        return [message]
    m = _BOUND.match(message)
    if m:
        return [list_record(m.group(1), int(m.group(2)), int(m.group(3)))]
    m = _NINT.match(message)
    if m:
        return [list_record("&   nint n l                ", *(int(g) for g in m.groups())),
                list_record("&   eigenvalue log to file err.dftseq_e"),
                list_record("&   potential  log to file err.dftseq_v")]
    m = _CTP.match(message)
    if m:
        return [list_record("&    nint n l e                      ",
                            *(int(g) for g in m.groups()[:3]), float(m.group(4))),
                list_record("&    0-order coefficient to file err.dftseq_tp")]
    return [" " + message]
