"""Fortran-compatible formatted output and list-directed input.

The output helpers render numbers exactly like the gfortran edit descriptors
used in the FHI98PP unit files (``e20.14``, ``e12.6``, ``f16.5``, ``i5``,
``a30``, ``1pe9.1`` ...).  The input helper :class:`ListReader` mimics the
list-directed ``read(unit,*)`` statements of the Fortran sources: values are
separated by blanks or commas, a statement always starts on a new record,
continues over following records until its list is satisfied and discards
the rest of the last record (so trailing comments such as ``: z nc nv`` are
ignored).  ``d`` exponents are accepted.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

__all__ = [
    "FortranReadError", "FortranEndOfFile", "ListReader",
    "fmt_a", "fmt_i", "fmt_f", "fmt_e", "fmt_es", "parse_real",
]


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def fmt_a(text: str, width: int) -> str:
    """``Aw``: right-justified; the leftmost ``w`` characters if too long."""
    if len(text) >= width:
        return text[:width]
    return " " * (width - len(text)) + text


def fmt_i(value: int, width: int) -> str:
    """``Iw``: right-justified integer, asterisks on overflow."""
    text = str(int(value))
    if len(text) > width:
        return "*" * width
    return " " * (width - len(text)) + text


def _nonfinite(value: float, width: int) -> str:
    if math.isnan(value):
        text = "NaN"
    else:
        text = "Infinity" if width >= 8 else "Inf"
        if value < 0:
            text = "-" + text
            if len(text) > width:
                text = "-Inf"
    if len(text) > width:
        return "*" * width
    return " " * (width - len(text)) + text


def _fit(sign: str, body: str, width: int) -> str:
    """Place ``sign + body``; drop the optional leading zero if necessary."""
    text = sign + body
    if len(text) > width and body.startswith("0."):
        text = sign + body[1:]
    if len(text) > width:
        return "*" * width
    return " " * (width - len(text)) + text


def fmt_f(value: float, width: int, d: int) -> str:
    """``Fw.d`` as written by gfortran."""
    value = float(value)
    if not math.isfinite(value):
        return _nonfinite(value, width)
    body = f"{abs(value):.{d}f}"
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    return _fit(sign, body, width)


def _exponent(exp: int) -> str:
    sign = "-" if exp < 0 else "+"
    mag = abs(exp)
    if mag <= 99:
        return f"E{sign}{mag:02d}"
    if mag <= 999:
        return f"{sign}{mag:03d}"
    return "*" * 4


def fmt_e(value: float, width: int, d: int) -> str:
    """``Ew.d`` (``0.ddddE+ee``) as written by gfortran."""
    value = float(value)
    if not math.isfinite(value):
        return _nonfinite(value, width)
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if value == 0.0:
        digits, exp = "0" * d, 0
    else:
        mant, _, exptext = f"{abs(value):.{d - 1}e}".partition("e")
        digits = mant.replace(".", "")
        exp = int(exptext) + 1
    return _fit(sign, "0." + digits + _exponent(exp), width)


def fmt_es(value: float, width: int, d: int) -> str:
    """``1PEw.d`` (``d.dddE+ee``) as written by gfortran."""
    value = float(value)
    if not math.isfinite(value):
        return _nonfinite(value, width)
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    if value == 0.0:
        mant, exp = "0." + "0" * d, 0
    else:
        mant, _, exptext = f"{abs(value):.{d}e}".partition("e")
        exp = int(exptext)
    text = sign + mant + _exponent(exp)
    if len(text) > width:
        return "*" * width
    return " " * (width - len(text)) + text


# --------------------------------------------------------------------------
# list-directed input
# --------------------------------------------------------------------------

class FortranReadError(ValueError):
    """Conversion error of a list-directed read (the ``err=`` branch)."""

    def __init__(self, message: str, values: Sequence[object] = ()):
        super().__init__(message)
        self.values = list(values)


class FortranEndOfFile(EOFError):
    """End of file during a list-directed read (the ``end=`` branch)."""

    def __init__(self, message: str = "end of file", values: Sequence[object] = ()):
        super().__init__(message)
        self.values = list(values)


_INT_RE = re.compile(r"^[+-]?\d+$")
_REAL_RE = re.compile(
    r"^([+-]?)(\d+\.?\d*|\.\d+)(?:[eEdDqQ]([+-]?\d+)|([+-]\d+))?$"
)
_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$", re.S)


def parse_real(token: str) -> float:
    """Convert a Fortran real literal (``1.5d0``, ``.5``, ``2``, ``1.e-3``)."""
    match = _REAL_RE.match(token)
    if not match:
        low = token.lower().lstrip("+-")
        if low in ("inf", "infinity", "nan"):
            return float(token)
        raise FortranReadError(f"bad real number in list input: {token!r}")
    sign, mant, exp1, exp2 = match.groups()
    exp = exp1 if exp1 is not None else exp2
    return float(f"{sign}{mant}e{exp}" if exp is not None else f"{sign}{mant}")


def _parse_int(token: str) -> int:
    if not _INT_RE.match(token):
        raise FortranReadError(f"bad integer in list input: {token!r}")
    return int(token)


def _parse_logical(token: str) -> bool:
    body = token[1:] if token.startswith(".") else token
    if body[:1] in ("t", "T"):
        return True
    if body[:1] in ("f", "F"):
        return False
    raise FortranReadError(f"bad logical in list input: {token!r}")


_CONVERT = {"r": parse_real, "i": _parse_int, "l": _parse_logical, "a": str}
_END = object()   # a slash: terminates the statement
_NULL = None      # a null value: the list item keeps its value


def _tokenize(record: str) -> list[object]:
    """Split one record into values, null values and the slash terminator."""
    out: list[object] = []
    pos, size = 0, len(record)
    blanks = " \t\r"
    while True:
        while pos < size and record[pos] in blanks:
            pos += 1
        if pos >= size:
            return out
        char = record[pos]
        if char == "/":
            out.append(_END)
            return out
        if char == ",":
            out.append(_NULL)
            pos += 1
            continue
        if char in "'\"":
            end = pos + 1
            chunk = []
            while end < size:
                if record[end] == char:
                    if end + 1 < size and record[end + 1] == char:
                        chunk.append(char)
                        end += 2
                        continue
                    break
                chunk.append(record[end])
                end += 1
            out.append(("q", "".join(chunk)))
            pos = end + 1
        else:
            end = pos
            while end < size and record[end] not in blanks and record[end] not in ",/":
                end += 1
            out.append(record[pos:end])
            pos = end
        while pos < size and record[pos] in blanks:
            pos += 1
        if pos < size and record[pos] == ",":
            pos += 1


class ListReader:
    """Sequential list-directed reader over the records of a text unit."""

    def __init__(self, text: str):
        self._records = text.splitlines()
        self._next = 0

    @property
    def record_number(self) -> int:
        """Number of records consumed so far."""
        return self._next

    def read(self, kinds: str) -> list[object]:
        """Execute one ``read(unit,*)`` for items of the given kinds.

        ``kinds`` is a string of ``r`` (real), ``i`` (integer), ``l``
        (logical) and ``a`` (character).  Null values (``,,`` or after a
        slash) are returned as ``None``.  Raises :class:`FortranEndOfFile`
        or :class:`FortranReadError`; both carry the values converted so far
        in ``.values``.
        """
        values: list[object] = []
        want = len(kinds)
        while len(values) < want:
            if self._next >= len(self._records):
                raise FortranEndOfFile(values=values)
            record = self._records[self._next]
            self._next += 1
            for token in _tokenize(record):
                if token is _END:
                    values.extend([None] * (want - len(values)))
                    break
                if token is _NULL:
                    values.append(None)
                elif isinstance(token, tuple):
                    if kinds[len(values)] != "a":
                        raise FortranReadError(
                            f"unexpected character constant {token[1]!r}", values)
                    values.append(token[1])
                else:
                    count, text = 1, token
                    repeat = _REPEAT_RE.match(token)
                    if repeat and kinds[len(values)] != "a":
                        count, text = int(repeat.group(1)), repeat.group(2)
                    for _ in range(count):
                        if len(values) >= want:
                            break
                        if text == "":
                            values.append(None)
                            continue
                        try:
                            values.append(_CONVERT[kinds[len(values)]](text))
                        except FortranReadError as exc:
                            raise FortranReadError(str(exc), values) from None
                if len(values) >= want:
                    break
        return values
