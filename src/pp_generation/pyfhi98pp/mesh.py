"""Logarithmic radial mesh.  Port of ``logmesh.f``."""

from __future__ import annotations

import numpy as np

from . import libm
from .constants import AINC, MX, RMAX, RMIN
from .errors import FortranStop

__all__ = ["logmesh"]


def logmesh(z: float) -> tuple[int, float, float, np.ndarray]:
    """Return ``(mmax, amesh, al, r)`` for nuclear charge ``z``.

    ``logmesh.f``: ``r(1) = rmin/z``, ``r(i) = amesh*r(i-1)``, with the
    number of points ``int(log(rmax/rmin*z)/al)`` made odd.  ``r`` has
    ``mmax`` elements (Python index ``i`` is Fortran index ``i+1``).
    The recurrence is kept (not ``r1*amesh**i``) because it defines the
    rounding of every mesh point.
    """
    amesh = AINC
    al = libm.slog(amesh)
    cmax = RMAX / RMIN
    mmax = int(libm.slog(cmax * z) / al)
    if mmax % 2 == 0:
        mmax += 1
    if mmax > MX:
        raise FortranStop("logmesh - mesh too large")
    r = np.empty(mmax)
    value = RMIN / z
    r[0] = value
    for i in range(1, mmax):
        value = amesh * value
        r[i] = value
    return mmax, amesh, al, r
