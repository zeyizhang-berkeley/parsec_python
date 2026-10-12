"""Fortran number editing used by the FHI98PP and Quantum ESPRESSO writers.

The converters reproduce files written by Fortran programs digit for digit,
so the three edit descriptors those programs use are implemented here:
``Ew.d`` (FHI98PP tables), ``ESw.d`` (UPF arrays) and list-directed output of
a double (UPF attributes and ``PP_DIJ``), all as GNU Fortran prints them.
"""

from __future__ import annotations

import decimal
import math
import re

from .potre import ConversionError


# Fortran drops the exponent letter when the exponent needs three digits:
# ``0.12340000000000-104`` is 0.1234e-104.
_BARE_EXPONENT_RE = re.compile(r"(?<=[0-9.])([-+][0-9]{3})$")
_LOG_CONTEXT = decimal.Context(prec=50)


def parse_real(token: str) -> float:
    """Read one real as a Fortran list-directed ``READ`` does."""

    normalized = token.replace("D", "E").replace("d", "e")
    try:
        return float(normalized)
    except ValueError:
        if "e" in normalized.lower():
            raise
        return float(_BARE_EXPONENT_RE.sub(r"E\g<1>", normalized))


def correctly_rounded_log(value: float) -> float:
    """Natural logarithm rounded to the nearest double on every platform.

    ``math.log`` is the C library's and differs between platforms in the last
    bit for a fraction of a percent of arguments.  The correctly rounded
    result agrees with glibc (which the reference Fortran used) for all but
    about one argument in 1500.
    """

    if not value > 0.0 or not math.isfinite(value):
        raise ConversionError(f"logarithm of {value!r} is undefined")
    return float(_LOG_CONTEXT.ln(decimal.Decimal(value)))


def _exponent_field(exponent: int) -> str:
    sign = "-" if exponent < 0 else "+"
    if abs(exponent) < 100:
        return f"E{sign}{abs(exponent):02d}"
    if abs(exponent) < 1000:
        return f"{sign}{abs(exponent):03d}"
    raise ConversionError(f"exponent {exponent} does not fit a Fortran field")


def _require_finite(value: float) -> None:
    if not math.isfinite(value):
        raise ConversionError(f"cannot write the non-finite value {value!r}")


def format_e(value: float, width: int, digits: int) -> str:
    """``Ew.d``: ``0.56818181818182E-03``, ``-.39456428640058E+00``."""

    _require_finite(value)
    mantissa, exponent_text = f"{abs(value):.{digits - 1}E}".split("E")
    exponent = int(exponent_text) + 1 if value != 0.0 else 0
    sign = "-" if math.copysign(1.0, value) < 0.0 else ""
    tail = "." + mantissa.replace(".", "") + _exponent_field(exponent)
    # The optional leading zero is the first thing dropped in a narrow field.
    text = sign + "0" + tail
    if len(text) > width:
        text = sign + tail
    if len(text) > width:
        raise ConversionError(f"{value!r} does not fit the field E{width}.{digits}")
    return text.rjust(width)


def format_es(value: float, width: int, digits: int) -> str:
    """``ESw.d``: ``5.681818181818200E-04`` right-justified in ``width``."""

    _require_finite(value)
    mantissa, exponent_text = f"{value:.{digits}E}".split("E")
    text = mantissa + _exponent_field(int(exponent_text))
    if len(text) > width:
        raise ConversionError(f"{value!r} does not fit the field ES{width}.{digits}")
    return text.rjust(width)


def format_list_directed(value: float) -> str:
    """List-directed output of one double: a 25-character field.

    GNU Fortran prints 17 significant digits, without an exponent when the
    rounded magnitude lies in [0.1, 1e17) (and for zero), followed by five
    blanks, and as ``d.ddddddddddddddddE+ddd`` otherwise.
    """

    _require_finite(value)
    exponent = int(f"{value:.16E}".split("E")[1]) if value != 0.0 else 0
    if -1 <= exponent <= 16:
        return f"{value:.{16 - exponent}f}".rjust(20) + " " * 5
    mantissa, exponent_text = f"{value:.16E}".split("E")
    exponent = int(exponent_text)
    sign = "-" if exponent < 0 else "+"
    return f"{mantissa}E{sign}{abs(exponent):03d}".rjust(25)


def format_r2c(value: float) -> str:
    """Quantum ESPRESSO's ``r2c``: the list-directed form without blanks."""

    return format_list_directed(value).strip()
