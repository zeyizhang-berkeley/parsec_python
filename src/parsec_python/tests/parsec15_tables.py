"""The radial tables of PARSEC 1.5 for the tests that compare with its numbers or with numbers made by them.

A species without ``Cubic_Spline`` takes the cubic spline by default (``PARSEC_CUBIC_SPLINE_DEFAULT=on``), the
projectors of a converted ``MARTINS_NEW`` table end at its channel cutoffs (``PARSEC_PROJECTOR_TAIL=auto``), the
Kleinman-Bylander denominators take the exact quadrature of the logarithmic mesh (``PARSEC_KB_QUADRATURE=auto``), the
Perdew-Zunger correlation is continuous (``PARSEC_CA_CONTINUITY=on``), a slab with forces takes the Hartree
potential that makes its energy stationary (``PARSEC_HARTREE_VARIATIONAL=slab``), the spline of a local potential
hands over to ``-2 Z/r`` at its Coulomb onset (``PARSEC_SPLINE_COULOMB_TAIL=auto``), the local potential and the
projectors are Fourier-filtered to the grid (``PARSEC_LOCAL_FILTER=all``) and the LDA+U projectors are smooth
(``PARSEC_LDAU_SMOOTH_PROJECTORS=on``).  A test whose reference comes from
PARSEC 1.5 (or from this package before those defaults) runs under their off values: ``setUpModule =
parsec15_tables_module`` for a whole module, :func:`parsec15_tables` around a block, or :func:`start_parsec15_tables`
in ``setUpClass``.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

#: The off values of those defaults: the linear tables, the tabulated tails, the KB quadrature and the correlation of
#: PARSEC 1.5, the Poisson potential of a slab, the spline out to the end of the table, the unfiltered tables and the
#: LDA+U projectors of upot.F90.
PARSEC15_TABLES = {
    "PARSEC_CUBIC_SPLINE_DEFAULT": "off",
    "PARSEC_PROJECTOR_TAIL": "off",
    "PARSEC_KB_QUADRATURE": "trapezoid",
    "PARSEC_CA_CONTINUITY": "off",
    "PARSEC_HARTREE_VARIATIONAL": "off",
    "PARSEC_SPLINE_COULOMB_TAIL": "off",
    "PARSEC_LOCAL_FILTER": "off",
    "PARSEC_LDAU_SMOOTH_PROJECTORS": "off",
}


def parsec15_tables():
    """A patch of the environment with :data:`PARSEC15_TABLES` (a context manager or decorator)."""

    return mock.patch.dict(os.environ, PARSEC15_TABLES)


def start_parsec15_tables(test_class=None) -> None:
    """:data:`PARSEC15_TABLES` until the class (``test_class``) or the module is done."""

    patcher = parsec15_tables()
    patcher.start()
    if test_class is None:
        unittest.addModuleCleanup(patcher.stop)
    else:
        test_class.addClassCleanup(patcher.stop)


def parsec15_tables_module() -> None:
    """``setUpModule`` of a module whose tests all compare with PARSEC 1.5's tables."""

    start_parsec15_tables()


__all__ = ["PARSEC15_TABLES", "parsec15_tables", "parsec15_tables_module", "start_parsec15_tables"]
