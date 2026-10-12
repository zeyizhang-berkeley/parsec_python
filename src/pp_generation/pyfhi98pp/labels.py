"""Element, functional and quantum-number labels.  Port of ``labelmap.f``."""

from __future__ import annotations

__all__ = ["labelmap", "spp", "labels"]

IZMX = 86
IXCMX = 20

# sxc(ixc): character*40 descriptors of the exchange-correlation models
_SXC: tuple[str, ...] = (
    "LDA Wigner",
    "LDA Hedin/Lundqvist",
    "LDA CA Perdew/Zunger 1980",
    "GGA Perdew/Wang 1991",
    "GGA Becke/Perdew",
    "GGA Perdew/Burke/Ernzerhof",
    "LDA CA Perdew/Wang 1991 + X relativistic",
    "LDA CA Perdew/Wang 1991",
    "GGA X Becke C Lee/Yang/Parr",
    "GGA X PW91  C Lee/Yang/Parr",
    "LDA exchange only",
) + ("unknown",) * 9

# sz(iz): character*2 chemical symbols
_SZ: tuple[str, ...] = (
    "H ", "He",
    "Li", "Be", "B ", "C ", "N ", "O ", "F ", "Ne",
    "Na", "Mg", "Al", "Si", "P ", "S ", "Cl", "Ar",
    "K ", "Ca", "Sc", "Ti", "V ", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y ", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I ", "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W ", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
)

_SYMBOLS_PP: tuple[str, ...] = ("hamann", "troullier-martins")


def labelmap(iz: int, ixc: int) -> tuple[str, str]:
    """Return ``(symz, symxc)`` as ``character*2`` and ``character*40``.

    ``labelmap.f``: nuclear charge to element symbol and xc type to its
    descriptor; both strings are blank padded to the Fortran lengths.
    ``iz > 86`` gives ``'no'`` (``'no label'`` cut to two characters).
    """
    if iz <= IZMX:
        if iz < 1:
            raise IndexError("labelmap: nuclear charge below 1")
        symz = _SZ[iz - 1]
    else:
        symz = "no label"[:2]
    if ixc <= IXCMX:
        if ixc < 1:
            raise IndexError("labelmap: xc index below 1")
        symxc = _SXC[ixc - 1]
    else:
        symxc = "no label"
    return symz, symxc.ljust(40)[:40]


def spp(ipp: int) -> str:
    """Pseudopotential type label (``character*20`` function ``spp``)."""
    if ipp <= len(_SYMBOLS_PP):
        return _SYMBOLS_PP[ipp - 1].ljust(20)
    return "what?".ljust(20)


def labels(n: int, l: int, m: int) -> tuple[str, str, str]:
    """Return the one-character tags ``(sn, sl, sm)`` of ``labels``.

    Out-of-range arguments map to ``'.'``.  (The Fortran routine also
    overwrites such arguments with 7, 5 and 3; callers here keep theirs.)
    """
    cn = "123456."
    cl = "spdfg."
    cm = "+-."
    if n < 1 or n > 6:
        n = 7
    if l < 0 or l > 4:
        l = 5
    if m < 1 or m > 2:
        m = 3
    return cn[n - 1], cl[l], cm[m - 1]
