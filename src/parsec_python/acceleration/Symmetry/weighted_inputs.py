"""Weighted sectors for the inputs that kept the abelian group before 2026-11-19.

The one-dimensional sectors of a subgroup ``H`` of the point group ``G`` with weights (:class:`.PointGroupReduction`,
"Weighted sectors of a larger subgroup" in ``acceleration/README.md``) solve one partner of some degenerate levels
and count it ``w`` times.  Four kinds of input kept the abelian group of the orbitals until 2026-11-19
(``driver._orbital_group_refusal``); each now takes the weighted sectors where its switch is ``auto`` (the default)
and the plan of the group does (``PARSEC_ORBITAL_GROUP``, ``PARSEC_ORBITAL_GROUP_MAX_COST``):

* ``PARSEC_ORBITAL_GROUP_FIXED_COUNT`` -- ``Fix_Number_Eigenvalues``: the first solve of the run brackets the lowest
  ``States_Num`` levels counted with their weights, and every sector keeps the number of them it holds
  (``CuPySymmetrySCFEigensolver._fix_sector_states``, which counts by weight).
* ``PARSEC_ORBITAL_GROUP_LIBXC`` -- a libxc LDA or GGA (not a hybrid or meta-GGA, which keep their own refusals): the
  device evaluator of libxc takes the orbit fields of ``H`` with the operation table of its stencil
  (``V_xc/device_libxc.py``, the geometry of the code's own GGA kernels), the host one expands them to the full
  grid and reduces its fields again (exact for invariant fields).
* ``PARSEC_ORBITAL_GROUP_OCCUPATIONS`` -- ``occup.in``: the ``n``-th occupation of the file goes to the ``n``-th
  level counted with the partners, so a state of weight ``w`` takes ``w`` consecutive entries, which must be equal
  (:func:`parsec_python.Occupations.weighted_fixed_occupations`), as must those of the degenerate partners of a level
  that several solved states hold (the e of Td in the sectors A1 and B1 of D2d, weight 1 each: the density, averaged
  over ``G``, would give each the mean; ``PointGroupReduction.sector_partner_irreps``, levels within
  ``Occupations.PARTNER_LEVEL_WIDTH``).  Where they are not (a partly filled multiplet that the file fills partner by
  partner, or one that the level order of a later step splits), the calculation starts again in the sectors of the
  abelian group with a NOTE (``driver._SpinSectorFallback``); the check runs at every SCF step, in both channels of a
  spin run.
* ``PARSEC_ORBITAL_GROUP_SPIN`` -- collinear spin (``Spin_Symmetry_Sectors``): both channels in the weighted sectors,
  with the weights in the Fermi level, the occupations and the density of each channel.  The density of a channel
  is averaged over ``G`` like the unpolarized one, so the density test of the spin guard (test 2 of
  ``SCF/spin_sectors.py``) cannot see an operation outside ``H``; the levels can: the guard counts a state of weight
  ``w`` as ``w`` partner levels in distinct sectors (tests 1 and 3), so that a partly filled multiplet held by one
  weighted state is caught as one held by several sectors is.  A run whose guard fires starts again in the abelian
  sectors (whose own guard may still send it to the full grid).

``off`` keeps the abelian group of before for that input, bit for bit.  Every value is read and checked in every
preparation (:func:`weighted_input_settings`), so that a value the code does not know stops the run before the
SCF.  :func:`abelian_orbitals` forces the abelian group while a calculation that left the weighted sectors is
prepared again.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os

#: The switch of each kind of input (see the module).
WEIGHTED_INPUT_SWITCHES = {
    "fixed_count": "PARSEC_ORBITAL_GROUP_FIXED_COUNT",
    "libxc": "PARSEC_ORBITAL_GROUP_LIBXC",
    "occupations": "PARSEC_ORBITAL_GROUP_OCCUPATIONS",
    "spin": "PARSEC_ORBITAL_GROUP_SPIN",
}
_VALUES = ("auto", "off")

# The reason a preparation keeps the abelian group whatever the input (a calculation that left the weighted sectors).
_FORCED_ABELIAN: ContextVar[str | None] = ContextVar("parsec_forced_abelian_orbitals", default=None)


def weighted_input_setting(kind: str) -> str:
    """The value of the switch of ``kind`` (a key of :data:`WEIGHTED_INPUT_SWITCHES`): ``auto`` or ``off``."""

    name = WEIGHTED_INPUT_SWITCHES[kind]
    value = os.environ.get(name, "auto").strip().lower() or "auto"
    if value not in _VALUES:
        raise ValueError(f"{name} must be auto or off, not {value!r}")
    return value


def weighted_input_settings() -> dict[str, str]:
    """Every switch of :data:`WEIGHTED_INPUT_SWITCHES` read and checked, by kind."""

    return {kind: weighted_input_setting(kind) for kind in WEIGHTED_INPUT_SWITCHES}


def forced_abelian_reason() -> str | None:
    """Why the preparation under way keeps the abelian group (:func:`abelian_orbitals`), ``None`` otherwise."""

    return _FORCED_ABELIAN.get()


@contextmanager
def abelian_orbitals(reason: str):
    """Prepare the calculations of this block with the abelian group of the orbitals, saying ``reason``."""

    token = _FORCED_ABELIAN.set(str(reason))
    try:
        yield
    finally:
        _FORCED_ABELIAN.reset(token)


__all__ = [
    "WEIGHTED_INPUT_SWITCHES",
    "abelian_orbitals",
    "forced_abelian_reason",
    "weighted_input_setting",
    "weighted_input_settings",
]
