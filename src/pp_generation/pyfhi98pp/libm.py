"""Transcendental functions of the platform math library.

This module is the only place where ``exp``, ``log`` and ``pow`` enter the
port, and ``sin`` and ``cos`` for the Bessel functions of the test program
(``sqrt`` is correctly rounded everywhere and used directly).  All
other arithmetic follows the Fortran statement order and is reproducible to
the last bit; the functions below are not: the Fortran reference links
glibc, whose results differ from other math libraries in the last bit for
roughly one argument in a thousand.  That is the entire source of the
round-off level differences between this port and the Fortran program.

Array versions (``exp``, ``log``, ``pow``) act element-wise on float64
arrays; scalar versions (``sexp``, ``slog``, ``spow``) on Python floats.
The functions are looked up through the module at call time, so a test
harness can substitute another implementation (for instance an oracle that
evaluates the glibc functions) by assigning to these names.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = ["exp", "log", "pow", "sin", "cos", "sexp", "slog", "spow", "spow_series"]


def exp(x):
    """Element-wise ``exp`` (Fortran ``exp``/``dexp``)."""
    return np.exp(x)


def sin(x):
    """Element-wise ``sin`` (used by the Bessel functions of ``bessel.f``)."""
    return np.sin(x)


def cos(x):
    """Element-wise ``cos`` (used by the Bessel functions of ``bessel.f``)."""
    return np.cos(x)


def log(x):
    """Element-wise natural logarithm (Fortran ``log``/``dlog``)."""
    return np.log(x)


def pow(x, y: float):  # noqa: A001 - mirrors the C library name
    """Element-wise ``x**y`` for a real exponent (C ``pow``).

    The exponent is expanded to an array so that numpy cannot replace the
    library call by a shortcut (``x*x`` for ``y == 2`` and the like), which
    would not round like ``pow`` does.
    """
    x = np.asarray(x, dtype=float)
    return np.power(x, np.full(x.shape, y))


def sexp(x: float) -> float:
    """Scalar ``exp``."""
    return math.exp(x)


def slog(x: float) -> float:
    """Scalar natural logarithm."""
    return math.log(x)


def spow(x: float, y: float) -> float:
    """Scalar ``x**y`` for a real exponent."""
    return math.pow(x, y)


def spow_series(x: Sequence[float], y: float) -> list[float]:
    """``x[i]**y`` for the four start points of the series in ``dftseq``.

    Kept separate from :func:`spow` because the reference build
    (gfortran 13 -O2) vectorizes this one loop and evaluates it with the
    two-wide vector ``pow`` of libmvec, whose last bit differs from the
    scalar ``pow`` slightly more often.
    """
    return [math.pow(v, y) for v in x]
