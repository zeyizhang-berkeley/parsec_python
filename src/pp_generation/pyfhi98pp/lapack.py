"""Linear solver ``dgesv`` for the small systems of ``tromar`` and ``dnlcc7``.

The Fortran calls LAPACK ``dgesv`` for a 5x5 (``tromarLAPACK.f``) and a 4x4
(``dnlcc7LAPACK.f``) system.  The result of ``dgesv`` is defined only up to
rounding, and ``tromar`` is sensitive to it in two ways: its matrix is badly
conditioned (powers ``rc**2 ... rc**12``), and the routine re-uses the
*factored* matrix as the input of a later call (see ``tromar``), so even the
LU factors are part of the numerical definition of the program.

This module therefore reproduces, operation by operation, the ``dgesv`` of
the library the reference executable is linked with (OpenBLAS 0.3.26,
``interface/lapack/gesv.c`` -> ``lapack/getf2/getf2.c`` and ``trsv`` for a
single right-hand side, x86-64 kernels with fused multiply-add):

* left-looking LU with partial pivoting: for column ``j`` the earlier row
  interchanges are applied, the part above the diagonal is obtained with dot
  products (``b[i] -= dot``), the part from the diagonal down with a matrix
  vector product (``b[j:] -= A[j:, :j] @ b[:j]``), then the pivot is searched
  and the column scaled with the reciprocal of the pivot;
* every ``acc + x*y`` of the dot products, the matrix-vector product and the
  triangular solves is a fused multiply-add (one rounding);
* forward substitution with the unit lower factor, back substitution with a
  division by the diagonal of the upper factor.

The emulation was compared bit for bit (LU factors, pivots, solution) with
all ``dgesv`` calls logged from the unmodified ``fhipp.x``; it is exact for
the matrix orders used here (``n <= 5``).
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["dgesv", "fma"]

_TWO53 = 9007199254740992.0     # 2**53
_SFMIN = 2.2250738585072014e-308  # dlamch('S')


def _fma_exact(a: float, b: float, c: float) -> float:
    """Correctly rounded ``a*b + c`` (round to nearest even) for doubles."""
    if a == 0.0 or b == 0.0 or not (math.isfinite(a) and math.isfinite(b)
                                    and math.isfinite(c)):
        return a * b + c
    ma, ea = math.frexp(a)
    mb, eb = math.frexp(b)
    prod = int(ma * _TWO53) * int(mb * _TWO53)   # exact integer, |prod| < 2**106
    ep = ea + eb - 106
    if c != 0.0:
        mc, ec = math.frexp(c)
        ic = int(mc * _TWO53)
        ec -= 53
        if ep >= ec:
            total = (prod << (ep - ec)) + ic
            et = ec
        else:
            total = prod + (ic << (ec - ep))
            et = ep
    else:
        total, et = prod, ep
    if total == 0:
        return 0.0
    mag = -total if total < 0 else total
    bits = mag.bit_length()
    if et + bits < -1000:
        # result near the subnormal range: exact rational arithmetic
        return float(Fraction(total) * Fraction(2) ** et)
    if bits > 64:
        # keep 64 bits plus a sticky bit: enough to round to 53 bits correctly
        shift = bits - 64
        sticky = 1 if mag & ((1 << shift) - 1) else 0
        mag = (mag >> shift) | sticky
        et += shift
    try:
        # int -> float is correctly rounded; the scaling by 2**et is exact
        res = math.ldexp(float(mag), et)
    except OverflowError:
        res = math.inf
    return -res if total < 0 else res


def _make_fma():
    native = getattr(math, "fma", None)     # Python >= 3.13
    if native is None:
        return _fma_exact

    def fma_(a: float, b: float, c: float) -> float:
        try:
            return native(a, b, c)
        except (OverflowError, ValueError):  # math.fma raises instead of inf/nan
            return _fma_exact(a, b, c)

    return fma_


fma = _make_fma()
fma.__doc__ = "Fused multiply-add ``a*b + c`` with a single rounding."
_safe_fma = fma


def dgesv(n: int, a: list[list[float]], b: list[float]) -> tuple[list[int], int]:
    """Solve ``A x = b`` in place like LAPACK ``dgesv(n,1,a,n,ipiv,b,n,info)``.

    ``a`` is the matrix as a list of *columns* (``a[j][i]`` is Fortran
    ``a(i+1,j+1)``) and is overwritten by its LU factors (unit lower factor
    below the diagonal, rows interchanged), ``b`` is overwritten by the
    solution.  Returns ``(ipiv, info)`` with the 1-based pivot rows and
    ``info`` = 0 or the (1-based) index of the first exactly zero pivot; as
    in LAPACK the right-hand side is left untouched if ``info != 0``.
    """
    f = _safe_fma
    ipiv = [0] * n
    info = 0
    for j in range(n):
        col = a[j]
        # apply the previous row interchanges to column j
        for i in range(j):
            ip = ipiv[i] - 1
            if ip != i:
                col[i], col[ip] = col[ip], col[i]
        # upper part: b(i) -= dot(a(i,1:i-1), b(1:i-1))
        for i in range(1, j):
            t = 0.0
            for k in range(i):
                t = f(col[k], a[k][i], t)
            col[i] = col[i] - t
        # diagonal and below: b(j:n) -= a(j:n,1:j-1) * b(1:j-1)
        if j:
            for i in range(j, n):
                t = 0.0
                for k in range(j):
                    t = f(a[k][i], col[k], t)
                col[i] = col[i] - t
        # pivot search (first largest modulus)
        jp = j
        big = abs(col[j])
        for i in range(j + 1, n):
            if abs(col[i]) > big:
                big = abs(col[i])
                jp = i
        ipiv[j] = jp + 1
        temp1 = col[jp]
        if temp1 != 0.0:
            if jp != j:
                for k in range(j + 1):
                    ak = a[k]
                    ak[j], ak[jp] = ak[jp], ak[j]
            if j + 1 < n:
                if abs(temp1) >= _SFMIN:
                    s = 1.0 / temp1
                    for i in range(j + 1, n):
                        col[i] = col[i] * s
                else:
                    for i in range(j + 1, n):
                        col[i] = col[i] / temp1
        elif not info:
            info = j + 1
    if info:
        return ipiv, info
    # solve: row interchanges, L (unit lower), U
    for i in range(n):
        ip = ipiv[i] - 1
        if ip != i:
            b[i], b[ip] = b[ip], b[i]
    for i in range(n):
        xi = -b[i]
        col = a[i]
        for k in range(i + 1, n):
            b[k] = f(xi, col[k], b[k])
    for i in range(n - 1, -1, -1):
        col = a[i]
        try:
            b[i] = b[i] / col[i]
        except ZeroDivisionError:      # cannot happen for info == 0
            b[i] = math.copysign(math.inf, b[i]) if b[i] == b[i] and b[i] != 0.0 else math.nan
        xi = -b[i]
        for k in range(i):
            b[k] = f(xi, col[k], b[k])
    return ipiv, info
