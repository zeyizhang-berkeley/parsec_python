"""Lowest eigenvalues of a real symmetric matrix in packed storage.

Port of the LAPACK call made by ``klbyiiLAPACK.f``::

    call dspevx('N','I','L',nmx,hessl,-1.d4,1.d1,1,4,1.d-6,i,he,hz,nmx,...)

i.e. eigenvalues ``il..iu`` only, no eigenvectors, lower triangle packed by
columns, absolute tolerance ``abstol``.  For this job ``DSPEVX`` reduces the
matrix to tridiagonal form with ``DSPTRD`` (Householder reflections starting
from the first column) and finds the eigenvalues by bisection with
``DSTEBZ``/``DLAEBZ``.  The bisection stops as soon as an interval is
narrower than ``abstol``; the returned eigenvalue is the midpoint of that
interval and therefore differs from the exact eigenvalue by up to
``abstol/2``.  Because the program prints these numbers (``abstol = 1e-6``
Hartree, printed to ``1e-4`` eV) the bisection is ported step by step
(reference LAPACK 3.x, serial branch of ``DLAEBZ``) instead of being
replaced by a different eigensolver: only then do the same intervals, and
with them the same printed digits, come out.

The level-1/2 BLAS operations inside ``DSPTRD`` are written as the loops of
the reference BLAS.  An optimized BLAS (the Fortran reference build links
OpenBLAS) orders these sums differently, which changes the tridiagonal
matrix at round-off level; see the validation notes of the port for the
(rare) effect on a printed digit.

Fortran ``MAX``/``MIN`` are written out as comparisons where a NaN could
occur so that the control flow matches gfortran (a NaN operand is ignored).
"""

from __future__ import annotations

import math

from . import libm

__all__ = ["dsptrd_lower", "dstebz_index", "dspevx_lowest"]

# DLAMCH values of IEEE double precision
_SAFMIN = 2.2250738585072014e-308      # dlamch('S')
_ULP = 2.220446049250313e-16           # dlamch('P') = eps*base
_EPS = 1.1102230246251565e-16          # dlamch('E')
_HUGE = 1.7976931348623157e308         # dlamch('O')


def _fmax(a: float, b: float) -> float:
    """gfortran ``MAX(a, b)``: ``b`` if ``b > a`` or ``a`` is NaN."""
    return b if (b > a or a != a) else a


def _fmin(a: float, b: float) -> float:
    """gfortran ``MIN(a, b)``: ``b`` if ``b < a`` or ``a`` is NaN."""
    return b if (b < a or a != a) else a


def _dnrm2(x: list) -> float:
    """Euclidean norm with the scaled sum of squares of the reference BLAS."""
    scale = 0.0
    ssq = 1.0
    for value in x:
        if value != 0.0:
            absxi = abs(value)
            if scale < absxi:
                q = scale / absxi
                ssq = 1.0 + ssq * (q * q)
                scale = absxi
            else:
                q = absxi / scale
                ssq = ssq + q * q
    return scale * math.sqrt(ssq)


def _dlapy2(x: float, y: float) -> float:
    """``sqrt(x**2 + y**2)`` without unnecessary overflow (``DLAPY2``)."""
    if x != x:
        result = x
        if y != y:
            result = y
        return result
    if y != y:
        return y
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    if z == 0.0 or w > _HUGE:
        return w
    q = z / w
    return w * math.sqrt(1.0 + q * q)


def _dlarfg(alpha: float, x: list) -> tuple[float, float]:
    """Elementary reflector (``DLARFG``); ``x`` is overwritten by ``v(2:n)``.

    Returns ``(beta, tau)``.
    """
    if not x:
        return alpha, 0.0
    xnorm = _dnrm2(x)
    if xnorm == 0.0:
        return alpha, 0.0
    beta = -math.copysign(_dlapy2(alpha, xnorm), alpha)
    safmin = _SAFMIN / _EPS
    knt = 0
    if abs(beta) < safmin:
        rsafmn = 1.0 / safmin
        while True:
            knt += 1
            for k in range(len(x)):
                x[k] = rsafmn * x[k]
            beta = beta * rsafmn
            alpha = alpha * rsafmn
            if not (abs(beta) < safmin and knt < 20):
                break
        xnorm = _dnrm2(x)
        beta = -math.copysign(_dlapy2(alpha, xnorm), alpha)
    tau = (beta - alpha) / beta
    scal = 1.0 / (alpha - beta)
    for k in range(len(x)):
        x[k] = scal * x[k]
    for _ in range(knt):
        beta = beta * safmin
    return beta, tau


def dsptrd_lower(n: int, ap: list) -> tuple[list, list]:
    """Tridiagonal form of a packed lower symmetric matrix (``DSPTRD``, 'L').

    ``ap`` holds the lower triangle by columns (``n*(n+1)/2`` floats) and is
    overwritten.  Returns ``(d, e)``: diagonal (``n``) and sub-diagonal
    (``n-1``) of ``T = Q**T A Q``.
    """
    d = [0.0] * n
    e = [0.0] * max(n - 1, 0)
    ii = 0                                   # position of A(i,i), 0-based
    for i in range(n - 1):
        m = n - i - 1                        # order of the trailing matrix
        i1i1 = ii + m + 1                    # position of A(i+1,i+1)
        # generate H(i) to annihilate A(i+2:n,i)
        x = ap[ii + 2:ii + 1 + m]
        beta, taui = _dlarfg(ap[ii + 1], x)
        ap[ii + 2:ii + 1 + m] = x
        e[i] = beta
        if taui != 0.0:
            v = [1.0] + x
            # y := tau * A * v   (DSPMV, lower, beta = 0)
            y = [0.0] * m
            kk = i1i1
            for j in range(m):
                temp1 = taui * v[j]
                temp2 = 0.0
                y[j] = y[j] + temp1 * ap[kk]
                k = kk + 1
                for q in range(j + 1, m):
                    a = ap[k]
                    y[q] = y[q] + temp1 * a
                    temp2 = temp2 + a * v[q]
                    k += 1
                y[j] = y[j] + taui * temp2
                kk += m - j
            # w := y - 1/2 * tau * (y**T v) * v
            dot = 0.0
            for j in range(m):
                dot = dot + y[j] * v[j]
            alpha = -0.5 * taui * dot
            for j in range(m):
                y[j] = y[j] + alpha * v[j]
            # A := A - v w**T - w v**T   (DSPR2, lower, alpha = -1)
            kk = i1i1
            for j in range(m):
                vj = v[j]
                yj = y[j]
                if vj != 0.0 or yj != 0.0:
                    temp1 = -yj
                    temp2 = -vj
                    k = kk
                    for q in range(j, m):
                        ap[k] = ap[k] + v[q] * temp1 + y[q] * temp2
                        k += 1
                kk += m - j
        d[i] = ap[ii]
        ii = i1i1
    d[n - 1] = ap[ii]
    return d, e


def _dlaebz_search(ijob: int, nitmax: int, n: int, mmax: int, minp: int,
                   abstol: float, reltol: float, pivmin: float, d: list, e2: list,
                   nval: list, ab: list, c: list, nab: list) -> tuple[int, int]:
    """Serial iteration loop of ``DLAEBZ`` for ``IJOB`` 2 and 3.

    ``ab`` and ``nab`` are lists of ``[lower, upper]`` pairs, one per
    interval (``mmax`` of them allocated, the first ``minp`` in use), ``c``
    the search points and ``nval`` the wanted counts (``IJOB = 3``).
    Returns ``(info, mout)``.
    """
    atoli, rtoli = abstol, reltol
    kf = 0
    kl = minp - 1
    if ijob == 2:
        for ji in range(minp):
            c[ji] = 0.5 * (ab[ji][0] + ab[ji][1])
    for _ in range(nitmax):
        klnew = kl
        for ji in range(kf, kl + 1):
            # N(w): number of eigenvalues less than w
            tmp1 = c[ji]
            tmp2 = d[0] - tmp1
            itmp1 = 0
            if tmp2 <= pivmin:
                itmp1 = 1
                tmp2 = _fmin(tmp2, -pivmin)
            for j in range(1, n):
                tmp2 = d[j] - e2[j - 1] / tmp2 - tmp1
                if tmp2 <= pivmin:
                    itmp1 += 1
                    tmp2 = _fmin(tmp2, -pivmin)
            if ijob <= 2:
                # insure that N(w) is monotone
                itmp1 = min(nab[ji][1], max(nab[ji][0], itmp1))
                if itmp1 == nab[ji][1]:
                    ab[ji][1] = tmp1
                elif itmp1 == nab[ji][0]:
                    ab[ji][0] = tmp1
                elif klnew + 1 < mmax:
                    klnew += 1
                    ab[klnew][1] = ab[ji][1]
                    nab[klnew][1] = nab[ji][1]
                    ab[klnew][0] = tmp1
                    nab[klnew][0] = itmp1
                    ab[ji][1] = tmp1
                    nab[ji][1] = itmp1
                else:
                    return mmax + 1, kl + 1
            else:
                # binary search: keep the interval containing w with N(w) = nval
                if itmp1 <= nval[ji]:
                    ab[ji][0] = tmp1
                    nab[ji][0] = itmp1
                if itmp1 >= nval[ji]:
                    ab[ji][1] = tmp1
                    nab[ji][1] = itmp1
        kl = klnew
        # check for convergence
        kfnew = kf
        for ji in range(kf, kl + 1):
            tmp1 = abs(ab[ji][1] - ab[ji][0])
            tmp2 = max(abs(ab[ji][1]), abs(ab[ji][0]))
            if tmp1 < max(atoli, pivmin, rtoli * tmp2) or nab[ji][0] >= nab[ji][1]:
                # converged: swap with position kfnew, then increment kfnew
                if ji > kfnew:
                    ab[ji], ab[kfnew] = ab[kfnew], ab[ji]
                    nab[ji], nab[kfnew] = nab[kfnew], nab[ji]
                    if ijob == 3:
                        nval[ji], nval[kfnew] = nval[kfnew], nval[ji]
                kfnew += 1
        kf = kfnew
        # choose midpoints
        for ji in range(kf, kl + 1):
            c[ji] = 0.5 * (ab[ji][0] + ab[ji][1])
        if kf > kl:
            break
    return max(kl + 1 - kf, 0), kl + 1


def _sturm_count(n: int, d: list, e2: list, pivmin: float, x: float) -> int:
    """``IJOB = 1`` count of ``DLAEBZ``: eigenvalues not larger than ``x``."""
    tmp1 = d[0] - x
    if abs(tmp1) < pivmin:
        tmp1 = -pivmin
    count = 1 if tmp1 <= 0.0 else 0
    for j in range(1, n):
        tmp1 = d[j] - e2[j - 1] / tmp1 - x
        if abs(tmp1) < pivmin:
            tmp1 = -pivmin
        if tmp1 <= 0.0:
            count += 1
    return count


def _itmax(a: float, pivmin: float) -> int:
    """``INT((LOG(a+pivmin)-LOG(pivmin))/LOG(2)) + 2`` with the x86 result
    of the conversion for NaN/overflow (the most negative integer)."""
    value = (libm.slog(a + pivmin) - libm.slog(pivmin)) / libm.slog(2.0) if a + pivmin > 0.0 \
        else float("nan")
    if value != value or abs(value) >= 2147483648.0:
        return -2147483648 + 2
    return int(value) + 2


def dstebz_index(n: int, il: int, iu: int, abstol: float, d: list, e: list
                 ) -> tuple[int, list, int]:
    """Eigenvalues ``il..iu`` of a symmetric tridiagonal matrix by bisection.

    ``DSTEBZ`` with ``RANGE = 'I'``, ``ORDER = 'E'``.  Returns
    ``(m, w, info)``: number of eigenvalues found, the eigenvalues in
    ascending order and the LAPACK ``info`` (0, or 1-3 for non-convergence /
    wrong count, 4 if the Gershgorin interval was not usable).
    """
    fudge = 2.1
    relfac = 2.0
    irange = 3
    if il == 1 and iu == n:
        raise ValueError("dstebz_index: the full range (RANGE = 'A' path) is not ported")
    rtoli = _ULP * relfac
    ncnvrg = False
    toofew = False
    if n == 1:
        return 1, [d[0]], 0

    # splitting points
    isplit: list[int] = []
    work = [0.0] * n                    # e(j)**2, zero at the splits
    pivmin = 1.0
    for j in range(1, n):
        tmp1 = e[j - 1] * e[j - 1]
        if abs(d[j] * d[j - 1]) * (_ULP * _ULP) + _SAFMIN > tmp1:
            isplit.append(j)
            work[j - 1] = 0.0
        else:
            work[j - 1] = tmp1
            pivmin = _fmax(pivmin, tmp1)
    isplit.append(n)
    nsplit = len(isplit)
    pivmin = pivmin * _SAFMIN

    wl = wu = wlu = wul = 0.0
    nwl = nwu = 0
    if irange == 3:
        # Gershgorin interval of the whole matrix as the initial interval
        gu = d[0]
        gl = d[0]
        tmp1 = 0.0
        for j in range(n - 1):
            tmp2 = math.sqrt(work[j]) if work[j] >= 0.0 else float("nan")
            gu = _fmax(gu, d[j] + tmp1 + tmp2)
            gl = _fmin(gl, d[j] - tmp1 - tmp2)
            tmp1 = tmp2
        gu = _fmax(gu, d[n - 1] + tmp1)
        gl = _fmin(gl, d[n - 1] - tmp1)
        tnorm = _fmax(abs(gl), abs(gu))
        gl = gl - fudge * tnorm * _ULP * n - fudge * 2.0 * pivmin
        gu = gu + fudge * tnorm * _ULP * n + fudge * pivmin
        itmax = _itmax(tnorm, pivmin)
        atoli = _ULP * tnorm if abstol <= 0.0 else abstol
        ab = [[gl, gu], [gl, gu]]
        c = [gl, gu]
        nab = [[-1, n + 1], [-1, n + 1]]
        nval = [il - 1, iu]
        _dlaebz_search(3, itmax, n, 2, 2, atoli, rtoli, pivmin, d, work, nval, ab, c, nab)
        if nval[1] == iu:
            wl, wlu, nwl = ab[0][0], ab[0][1], nab[0][0]
            wu, wul, nwu = ab[1][1], ab[1][0], nab[1][1]
        else:
            wl, wlu, nwl = ab[1][0], ab[1][1], nab[1][0]
            wu, wul, nwu = ab[0][1], ab[0][0], nab[0][1]
        if nwl < 0 or nwl >= n or nwu < 1 or nwu > n:
            return 0, [], 4
    # find eigenvalues: loop over blocks, recompute nwl and nwu
    w: list[float] = []
    iblock: list[int] = []
    iend = 0
    nwl = 0
    nwu = 0
    for jb in range(1, nsplit + 1):
        ioff = iend
        ibegin = ioff                     # 0-based start of the block
        iend = isplit[jb - 1]
        nin = iend - ioff
        if nin == 1:
            db = d[ibegin]
            if irange == 1 or wl >= db - pivmin:
                nwl += 1
            if irange == 1 or wu >= db - pivmin:
                nwu += 1
            if irange == 1 or (wl < db - pivmin and wu >= db - pivmin):
                w.append(db)
                iblock.append(jb)
            continue
        # Gershgorin interval of the block
        gu = d[ibegin]
        gl = d[ibegin]
        tmp1 = 0.0
        for j in range(ibegin, iend - 1):
            tmp2 = abs(e[j])
            gu = _fmax(gu, d[j] + tmp1 + tmp2)
            gl = _fmin(gl, d[j] - tmp1 - tmp2)
            tmp1 = tmp2
        gu = _fmax(gu, d[iend - 1] + tmp1)
        gl = _fmin(gl, d[iend - 1] - tmp1)
        bnorm = _fmax(abs(gl), abs(gu))
        gl = gl - fudge * bnorm * _ULP * nin - fudge * pivmin
        gu = gu + fudge * bnorm * _ULP * nin + fudge * pivmin
        if abstol <= 0.0:
            atoli = _ULP * _fmax(abs(gl), abs(gu))
        else:
            atoli = abstol
        if irange > 1:
            if gu < wl:
                nwl += nin
                nwu += nin
                continue
            gl = _fmax(gl, wl)
            gu = _fmin(gu, wu)
            if gl >= gu:
                continue
        # set up the initial interval and count the eigenvalues in it
        db = d[ibegin:iend]
        e2b = work[ibegin:iend]
        n_lo = _sturm_count(nin, db, e2b, pivmin, gl)
        n_hi = _sturm_count(nin, db, e2b, pivmin, gu)
        im = n_hi - n_lo
        nwl += n_lo
        nwu += n_hi
        iwoff = len(w) - n_lo
        # compute the eigenvalues of the block
        itmax = _itmax(gu - gl, pivmin)
        ab = [[gl, gu]] + [[0.0, 0.0] for _ in range(nin - 1)]
        nab = [[n_lo, n_hi]] + [[0, 0] for _ in range(nin - 1)]
        c = [0.0] * nin
        iinfo, iout = _dlaebz_search(2, itmax, nin, nin, 1, atoli, rtoli, pivmin, db, e2b,
                                     [], ab, c, nab)
        # copy the eigenvalues into w and iblock
        need = len(w) + im
        w.extend([0.0] * im)
        iblock.extend([0] * im)
        for j in range(iout):
            tmp1 = 0.5 * (ab[j][0] + ab[j][1])
            if j + 1 > iout - iinfo:
                ncnvrg = True
                ib = -jb
            else:
                ib = jb
            for je in range(nab[j][0] + iwoff, nab[j][1] + iwoff):
                if 0 <= je < need:
                    w[je] = tmp1
                    iblock[je] = ib
    m = len(w)

    # discard extra eigenvalues if (wl, wu] holds more than il..iu
    if irange == 3:
        idiscl = il - 1 - nwl
        idiscu = nwu - iu
        if idiscl > 0 or idiscu > 0:
            keep_w: list[float] = []
            keep_b: list[int] = []
            for je in range(m):
                if w[je] <= wlu and idiscl > 0:
                    idiscl -= 1
                elif w[je] >= wul and idiscu > 0:
                    idiscu -= 1
                else:
                    keep_w.append(w[je])
                    keep_b.append(iblock[je])
            w, iblock = keep_w, keep_b
            m = len(w)
        if idiscl > 0 or idiscu > 0:
            # bad arithmetic: kill off the smallest idiscl / largest idiscu
            if idiscl > 0:
                wkill = wu
                for _ in range(idiscl):
                    iw = -1
                    for je in range(m):
                        if iblock[je] != 0 and (w[je] < wkill or iw < 0):
                            iw = je
                            wkill = w[je]
                    if iw >= 0:
                        iblock[iw] = 0
            if idiscu > 0:
                wkill = wl
                for _ in range(idiscu):
                    iw = -1
                    for je in range(m):
                        if iblock[je] != 0 and (w[je] > wkill or iw < 0):
                            iw = je
                            wkill = w[je]
                    if iw >= 0:
                        iblock[iw] = 0
            keep = [je for je in range(m) if iblock[je] != 0]
            w = [w[je] for je in keep]
            iblock = [iblock[je] for je in keep]
            m = len(w)
        if idiscl < 0 or idiscu < 0:
            toofew = True

    # ORDER = 'E': sort the eigenvalues from smallest to largest
    if nsplit > 1:
        for je in range(m - 1):
            ie = -1
            tmp1 = w[je]
            for j in range(je + 1, m):
                if w[j] < tmp1:
                    ie = j
                    tmp1 = w[j]
            if ie >= 0:
                iblock[ie], iblock[je] = iblock[je], iblock[ie]
                w[ie] = w[je]
                w[je] = tmp1
    info = 0
    if ncnvrg:
        info += 1
    if toofew:
        info += 2
    return m, w, info


def dspevx_lowest(n: int, ap, il: int, iu: int, abstol: float) -> tuple[int, list, int]:
    """``DSPEVX('N','I','L',n,ap,vl,vu,il,iu,abstol,m,w,...)``.

    ``ap``: lower triangle packed by columns (not modified).  Returns
    ``(m, w, info)`` like the Fortran routine: ``w[:m]`` are the eigenvalues
    ``il..iu`` in ascending order.  A matrix containing NaN makes the
    Fortran routine return ``info = 4`` with no eigenvalue (the Gershgorin
    interval is NaN); that outcome is returned directly.
    """
    ap = [float(v) for v in ap]
    if any(v != v for v in ap):
        return 0, [], 4
    # scale the matrix to the allowable range, if necessary
    smlnum = _SAFMIN / _ULP
    bignum = 1.0 / smlnum
    rmin = math.sqrt(smlnum)
    rmax = min(math.sqrt(bignum), 1.0 / math.sqrt(math.sqrt(_SAFMIN)))
    anrm = 0.0
    for v in ap:
        a = abs(v)
        if anrm < a:
            anrm = a
    iscale = 0
    sigma = 1.0
    abstll = abstol
    if 0.0 < anrm < rmin:
        iscale = 1
        sigma = rmin / anrm
    elif anrm > rmax:
        iscale = 1
        sigma = rmax / anrm
    if iscale:
        ap = [sigma * v for v in ap]
        if abstol > 0.0:
            abstll = abstol * sigma
    d, e = dsptrd_lower(n, ap)
    m, w, info = dstebz_index(n, il, iu, abstll, d, e)
    if iscale:
        imax = m if info == 0 else info - 1
        for k in range(min(imax, len(w))):
            w[k] = (1.0 / sigma) * w[k]
    return m, w, info
