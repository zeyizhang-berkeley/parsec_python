"""Default cutoff radii of the Troullier-Martins scheme.  Port of ``defrtm.f``.

The Fortran fills a local table ``r(4,80)`` (radii in units of 0.01 bohr,
first index ``l+1``, second index the nuclear charge) with assignment
statements, some of them conditional on the principal quantum number, and
returns ``0.01*r(min(l+1,4),z)`` with a fall-back to the next lower ``l``
that has an entry.  The table below is the outcome of those statements for
each column, including the slips of the source, which are kept:

* ``r(2,12)`` is assigned twice (240, then 250) and ``r(3,12)`` never;
* ``r(1,17)`` is assigned twice (152, then 160) and ``r(2,17)`` never;
* the Tc block assigns ``r(3,42)`` (244, overriding the 246 of Mo) instead
  of ``r(3,43)``, which stays zero;
* the Sn block has ``if(n .eq. 4) r(3,49) = 234`` (In, not Sn).

A zero means "not tabulated"; the main program then computes a radius.

Deviation from the Fortran (deliberate): the source declares ``r(4,80)``
but writes ``r(j,z)`` for any ``z``, which corrupts memory for ``z > 80``
(Tl crashes).  Here every ``z`` without entries, including ``z > 80``,
returns 0, which is what a build with a larger table does.
"""

from __future__ import annotations

from .errors import FortranStop

__all__ = ["defrtm"]

# z: (r(1,z), r(2,z), r(3,z), r(4,z)) after all unconditional assignments
_R: dict[int, tuple[int, int, int, int]] = {
    1: (130, 0, 0, 0),
    2: (110, 0, 0, 0),
    3: (220, 0, 250, 0),
    4: (210, 0, 240, 0),
    5: (170, 0, 0, 0),
    6: (150, 0, 0, 0),
    7: (145, 0, 0, 0),
    8: (140, 0, 0, 0),
    9: (135, 0, 0, 0),
    10: (130, 0, 0, 0),
    11: (271, 271, 300, 0),
    12: (210, 250, 0, 0),
    13: (180, 200, 215, 0),
    14: (173, 190, 203, 0),
    15: (166, 180, 196, 0),
    16: (159, 170, 189, 0),
    17: (160, 0, 182, 0),
    18: (145, 0, 175, 0),
    19: (330, 320, 0, 0),
    20: (280, 300, 260, 0),
    21: (274, 274, 245, 0),
    22: (266, 266, 240, 0),
    23: (262, 262, 240, 0),
    24: (256, 256, 235, 0),
    25: (242, 250, 230, 0),
    26: (235, 245, 225, 0),
    27: (230, 240, 220, 0),
    28: (225, 235, 215, 0),
    29: (210, 230, 210, 0),
    30: (205, 230, 270, 0),
    31: (210, 230, 250, 0),
    32: (200, 220, 250, 0),
    33: (200, 220, 245, 0),
    34: (190, 210, 240, 0),
    35: (180, 210, 220, 0),
    36: (175, 200, 210, 0),
    37: (400, 425, 350, 0),
    38: (350, 400, 350, 0),
    39: (320, 320, 270, 0),
    40: (295, 310, 260, 0),
    41: (280, 300, 250, 0),
    42: (270, 280, 244, 0),
    43: (260, 265, 0, 0),
    44: (250, 260, 242, 0),
    45: (250, 260, 240, 0),
    46: (248, 260, 238, 0),
    47: (242, 262, 242, 0),
    48: (242, 254, 234, 0),
    49: (244, 254, 280, 0),
    50: (234, 244, 275, 0),
    51: (226, 238, 270, 0),
    52: (226, 230, 270, 0),
    53: (215, 224, 270, 0),
    54: (210, 210, 250, 0),
    55: (400, 450, 400, 0),
    56: (430, 500, 400, 0),
    58: (350, 400, 300, 320),
    72: (280, 284, 246, 0),
    73: (268, 284, 246, 0),
    74: (264, 270, 246, 0),
    75: (260, 266, 246, 0),
    76: (256, 270, 254, 0),
    77: (254, 268, 254, 0),
    78: (254, 268, 254, 0),
    79: (254, 268, 254, 0),
    80: (250, 250, 285, 0),
}

# (z, n): {row index (0-based): value}, the ``if(n .eq. ...)`` statements
_R_N: dict[tuple[int, int], dict[int, int]] = {
    (19, 3): {0: 180},
    (20, 3): {0: 180, 2: 260},
    (30, 3): {2: 205},
    (31, 3): {2: 210},
    (49, 4): {2: 234},          # 242 from the In block, then 234 from the Sn block
    (55, 5): {0: 230, 1: 260},
    (56, 5): {0: 230, 1: 260},
    (80, 5): {2: 240},
}


def defrtm(n: int, l: int, z: int) -> float:
    """Default Troullier-Martins cutoff radius (bohr) or 0 if not tabulated.

    ``n``, ``l`` are the quantum numbers of the reference state, ``z`` the
    (integer) nuclear charge.
    """
    if z < 1 or n < l + 1 or l < 0:
        raise FortranStop("defrtm - illegal argument")
    col = list(_R.get(z, (0, 0, 0, 0)))
    for row, value in _R_N.get((z, n), {}).items():
        col[row] = value
    ra = col[min(l + 1, 4) - 1]
    if ra == 0 and l >= 3:
        ra = col[2]
    if ra == 0 and l >= 2:
        ra = col[1]
    if ra == 0 and l >= 1:
        ra = col[0]
    return 0.01e0 * ra
