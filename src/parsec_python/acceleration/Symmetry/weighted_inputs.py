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
  sectors (whose own guard may still send it to the full grid), or goes straight to the full grid where those
  sectors would stop the same levels (``PARSEC_WEIGHTED_GUARD_SHORTCUT`` below).

``off`` keeps the abelian group of before for that input, bit for bit.  Every value is read and checked in every
preparation (:func:`weighted_input_settings`), so that a value the code does not know stops the run before the
SCF.  :func:`abelian_orbitals` forces the abelian group while a calculation that left the weighted sectors is
prepared again.

Three switches since 2026-11-21 decide where the weighted sectors of spin and ``occup.in`` serve (also read and
checked in every preparation):

* ``PARSEC_ORBITAL_GROUP_MPI`` (:func:`mpi_weighted_setting`) -- ``auto``: spin polarization and ``occup.in`` take
  the weighted sectors on several MPI ranks too (the full-SCF runner), as in one process.  Every rank prepares the
  same plan; a run that leaves them is prepared again in the abelian sectors on every rank
  (``MPISectorContext.reprepare``).  ``off``: they keep the abelian group there, as before, with a NOTE.
* ``PARSEC_WEIGHTED_GUARD_SHORTCUT`` (:func:`guard_shortcut_setting`) -- ``auto``: a spin run of a cluster, or of a
  cell whose operations take no translation (whose abelian route keeps the group about the origin), whose guard
  stops the weighted sectors goes straight to the full grid where the sectors of the abelian group would stop the
  same state (the reported multiplet holds a representation of ``G`` whose partners lie in two or more abelian
  sectors, ``PointGroupReduction.sector_abelian_split``; for a cell whose abelian sectors take its commuting
  involutions with their translations those flags are formed against that group where a report asks for them,
  ``PointGroupReduction.sector_abelian_split_source``, under ``PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS=auto``;
  ``off`` of that switch keeps the way through its abelian sectors, as before 2026-11-22).  For a stop at the start
  Hamiltonian (test 1), which both routes share, the run of
  the abelian sectors that would follow is the one ``off`` of ``PARSEC_ORBITAL_GROUP_SPIN`` pays before its own full
  grid, bit for bit, wherever those sectors stop there too, as on the clusters and cells about the origin measured; for
  a cell with translations that is a prediction as well (the neutral vacancy on a B site of a diamond 3^3 cell: its
  abelian sectors passed test 1 and stopped at test 3, the same energy in 26 against 16 iterations); a stop at the
  end of the SCF (test 3) is predicted from the weighted run's own occupations, and
  gives the full-grid result, which is the abelian route's wherever that route stops at its end too.  ``off``:
  always through the abelian sectors, as before.
* ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR`` (:func:`spin_floor_setting`) -- the smallest basis (grid points times states
  of a channel) whose spin run takes the weighted sectors; a smaller one keeps the abelian group with a NOTE.
  ``auto``: :data:`SPIN_FLOOR_ENTRIES`, which is 0: measured on A100 (one node, first runs with empty kernel caches
  and runs with warm ones) from the N atom (3.7e6 entries), Cr2, O2, V- in diamond 2^3, 3^3 and 4^3, nanodiamonds
  k02 to k05 and their Si analogues k02 to k04 (up to 2.4e9), the weighted sectors were as fast or faster at every
  size, at a lower GPU peak, except V- 3^3 with warm caches (SCF 3.8 -> 4.8 s, a cell whose weighted stencils apply
  2.6 times slower per vector; with empty caches 20.3 -> 9.9 s), so there is no floor; ``off``: no floor either;
  an integer: that floor.
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


def weighted_route_settings() -> tuple[str, str, int]:
    """The three switches of the module's end read and checked: ``PARSEC_ORBITAL_GROUP_MPI``,
    ``PARSEC_WEIGHTED_GUARD_SHORTCUT`` and ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR``."""

    # (and PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS, which the guard reads where it stops)
    guard_shortcut_translations_setting()
    return mpi_weighted_setting(), guard_shortcut_setting(), spin_floor_setting()


MPI_SWITCH = "PARSEC_ORBITAL_GROUP_MPI"
GUARD_SHORTCUT_SWITCH = "PARSEC_WEIGHTED_GUARD_SHORTCUT"
GUARD_SHORTCUT_TRANSLATIONS_SWITCH = "PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS"
SPIN_FLOOR_SWITCH = "PARSEC_ORBITAL_GROUP_SPIN_FLOOR"
#: The floor of ``PARSEC_ORBITAL_GROUP_SPIN_FLOOR=auto``: grid points times states of a channel (see the module: the
#: measurements found no size below which the abelian sectors are faster, so none).
SPIN_FLOOR_ENTRIES = 0


def _auto_or_off(name: str) -> str:
    value = os.environ.get(name, "auto").strip().lower() or "auto"
    if value not in _VALUES:
        raise ValueError(f"{name} must be auto or off, not {value!r}")
    return value


def mpi_weighted_setting() -> str:
    """``PARSEC_ORBITAL_GROUP_MPI``: ``auto`` (spin and ``occup.in`` in the weighted sectors on several MPI ranks) or
    ``off`` (the abelian group there, as before)."""

    return _auto_or_off(MPI_SWITCH)


def guard_shortcut_setting() -> str:
    """``PARSEC_WEIGHTED_GUARD_SHORTCUT``: ``auto`` (a guard stop that the abelian sectors would repeat goes straight to
    the full grid) or ``off`` (through the abelian sectors, as before)."""

    return _auto_or_off(GUARD_SHORTCUT_SWITCH)


def guard_shortcut_translations_setting() -> str:
    """``PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS``: ``auto`` (the shortcut of ``PARSEC_WEIGHTED_GUARD_SHORTCUT``
    also for a cell whose abelian route takes its commuting involutions with their translations: the flags against
    that group, ``PointGroupReduction.sector_abelian_split_source``) or ``off`` (such a cell goes through its abelian
    sectors, as before 2026-11-22)."""

    return _auto_or_off(GUARD_SHORTCUT_TRANSLATIONS_SWITCH)


def spin_floor_setting() -> int:
    """``PARSEC_ORBITAL_GROUP_SPIN_FLOOR``: the floor in grid points times states (``auto``: :data:`SPIN_FLOOR_ENTRIES`;
    ``off``: 0, no floor; an integer >= 0)."""

    raw = os.environ.get(SPIN_FLOOR_SWITCH, "auto").strip().lower() or "auto"
    if raw == "auto":
        return int(SPIN_FLOOR_ENTRIES)
    if raw == "off":
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{SPIN_FLOOR_SWITCH} must be auto, off or a number of entries >= 0, not {raw!r}")
    return value


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
    "GUARD_SHORTCUT_SWITCH",
    "MPI_SWITCH",
    "SPIN_FLOOR_ENTRIES",
    "SPIN_FLOOR_SWITCH",
    "WEIGHTED_INPUT_SWITCHES",
    "abelian_orbitals",
    "forced_abelian_reason",
    "guard_shortcut_setting",
    "mpi_weighted_setting",
    "spin_floor_setting",
    "weighted_input_setting",
    "weighted_input_settings",
    "weighted_route_settings",
]
