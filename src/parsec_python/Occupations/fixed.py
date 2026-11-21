"""Occupations fixed by the user: PARSEC's ``occup.in``.

A negative ``Fermi_Temp`` tells PARSEC to search no Fermi level.
``initial.f90`` reads the occupation of every state from the file
``occup.in`` of the run directory, and ``flevel.f90`` assigns them at every
SCF step to the computed states in ascending order of energy: the state with
the n-th lowest eigenvalue, whatever its symmetry representation, receives
the n-th occupation.

The file holds one section per spin channel::

    title line                  skipped
    ifmax                       states 1..ifmax are fully occupied
    m                           number of occupations set by hand
    j  occupation               m records: state index and its occupation

Everything is read with list-directed statements, so text after the numbers
of a record is a comment.  ``initial.f90`` reads ``nspin`` sections, hence
only the first one in a calculation without spin polarization: whatever
follows the first section is not read.  There an occupation of one stands
for the two electrons of a doubly occupied orbital, the convention of
:func:`fermi_occupations`.  A spin-polarized calculation reads two
sections, the up channel first, and an occupation of one stands for one
electron (:mod:`parsec_python.Occupations.spin`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .fermi_dirac import OccupationResult


#: ``small`` of ``initial.f90``: largest accepted difference between the
#: electrons the fixed occupations hold and those of the system.
ELECTRON_COUNT_TOLERANCE = 1.0e-5

_INTEGER_RE = re.compile(r"[-+]?\d+")
_REAL_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?")


@dataclass(frozen=True)
class OccupationFile:
    """Occupations of the first spin section of an ``occup.in``.

    ``occupations`` has one value per state.  ``unread_lines`` counts the
    nonblank lines after that section: further spin sections, which a
    calculation without spin polarization does not read.
    """

    occupations: tuple[float, ...]
    unread_lines: int
    source: Path


@dataclass(frozen=True)
class SpinOccupationFile:
    """Occupations of the two spin sections of an ``occup.in``, up first.

    An occupation of one stands for one electron there.  ``unread_lines``
    counts the nonblank lines after the second section.
    """

    occupations_up: tuple[float, ...]
    occupations_down: tuple[float, ...]
    unread_lines: int
    source: Path


def read_occupation_file(path: str | Path, number_of_states: int) -> OccupationFile:
    """Read the occupations a spin-unpolarized PARSEC run takes from ``path``.

    ``number_of_states`` is ``States_Num``: states beyond ``ifmax`` are empty
    unless set by hand.  An index outside ``1..number_of_states`` writes
    beyond the occupation array in ``initial.f90`` and is refused here.
    """

    sections, unread, source = _read_sections(path, number_of_states, 1)
    return OccupationFile(sections[0], unread, source)


def read_spin_occupation_file(
    path: str | Path, number_of_states: int
) -> SpinOccupationFile:
    """Read the occupations a spin-polarized PARSEC run takes from ``path``.

    ``initial.f90`` reads one section per spin channel, the up channel
    first, each with its title line.  Every channel has ``number_of_states``
    states.
    """

    sections, unread, source = _read_sections(path, number_of_states, 2)
    return SpinOccupationFile(sections[0], sections[1], unread, source)


def _read_sections(
    path: str | Path, number_of_states: int, count: int
) -> tuple[list[tuple[float, ...]], int, Path]:
    """Read the first ``count`` spin sections of an ``occup.in``."""

    source = Path(path)
    states = int(number_of_states)
    if states != number_of_states or states < 1:
        raise ValueError("number_of_states must be a positive integer")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(
            f"a negative Fermi_Temp reads the occupations from {source}, "
            f"which cannot be read: {error}"
        ) from error
    position = 0

    def record(count: int, what: str) -> list[str]:
        # ``read(90,*)`` with ``count`` items: as many records as the list
        # needs, and the rest of the last one is dropped.
        nonlocal position
        found: list[str] = []
        while len(found) < count:
            if position >= len(lines):
                raise ValueError(f"{source} ends before {what}")
            found.extend(lines[position].replace(",", " ").split())
            position += 1
        return found[:count]

    def integer(token: str, what: str) -> int:
        if _INTEGER_RE.fullmatch(token) is None:
            raise ValueError(f"{source}: expected an integer for {what}, found {token!r}")
        return int(token)

    if not lines:
        raise ValueError(f"{source} is empty")
    sections: list[tuple[float, ...]] = []
    for section in range(count):
        # The title line is passed whatever it holds.
        if position >= len(lines):
            raise ValueError(
                f"{source} ends before the section of spin channel "
                f"{section + 1}; a spin-polarized calculation reads one "
                "section per channel"
            )
        position += 1
        highest_full = integer(
            record(1, "the number of fully occupied states")[0],
            "the number of fully occupied states",
        )
        if not 0 <= highest_full <= states:
            raise ValueError(
                f"{source}: {highest_full} fully occupied states do not fit "
                f"States_Num = {states}"
            )
        occupations = [1.0] * highest_full + [0.0] * (states - highest_full)
        overrides = integer(
            record(1, "the number of occupations set by hand")[0],
            "the number of occupations set by hand",
        )
        for _ in range(overrides):
            index_token, value_token = record(2, "an occupation set by hand")
            index = integer(index_token, "a state index")
            if not 1 <= index <= states:
                raise ValueError(
                    f"{source}: state index {index} lies outside 1..{states} "
                    "(States_Num)"
                )
            if _REAL_RE.fullmatch(value_token) is None:
                raise ValueError(
                    f"{source}: expected an occupation for state {index}, "
                    f"found {value_token!r}"
                )
            value = float(value_token.replace("D", "E").replace("d", "e"))
            if not np.isfinite(value):
                raise ValueError(
                    f"{source}: the occupation of state {index} is not finite"
                )
            occupations[index - 1] = value
        sections.append(tuple(occupations))
    unread = sum(1 for line in lines[position:] if line.strip())
    return sections, unread, source


def check_fixed_occupations(
    occupations: Sequence[float], electron_count: float
) -> None:
    """Refuse occupations that do not hold the electrons of the system.

    ``initial.f90`` stops with "fixed occupation inconsistent with stated
    number of electrons" when ``2*sum(occupations)`` differs from the
    electron count by more than ``1e-5``.
    """

    held = 2.0 * float(np.sum(np.asarray(occupations, dtype=float)))
    if abs(held - float(electron_count)) > ELECTRON_COUNT_TOLERANCE:
        raise ValueError(
            f"the fixed occupations hold {held:.6f} electrons, but the system "
            f"has {float(electron_count):.6f}"
        )


def fixed_occupations(
    eigenvalues: np.ndarray,
    electron_count: float,
    occupations: Sequence[float],
) -> OccupationResult:
    """Occupy the states in ascending order of energy with given values.

    ``occupations[n]`` goes to the n-th lowest eigenvalue, as the branch of
    ``flevel.f90`` for a negative Fermi temperature does.  No Fermi level is
    searched, and the result reports ``nan`` for it.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    values = np.array(occupations, dtype=float)
    if values.shape != eigenvalues.shape:
        raise ValueError(
            f"{values.size} fixed occupations were given for "
            f"{eigenvalues.size} computed states"
        )
    check_fixed_occupations(values, electron_count)
    return OccupationResult(
        fermi_level=float("nan"),
        occupations=values,
        electron_count=float(2.0 * np.sum(values)),
    )


#: Largest difference between the ``occup.in`` entries of the partners of one weighted state that counts as equal.
PARTNER_OCCUPATION_TOLERANCE = 1.0e-12

#: Levels (Ry) of two solved states that may be partners of one level (their sectors hold a representation of the
#: full group whose partners several solved states share) closer than this count as one level.  The potential of the
#: weighted sectors keeps the full group, so such partners differ by the error of the eigensolver alone; the width of
#: the exact multiplets of the spin guard (``acceleration/SCF/spin_sectors.MULTIPLET_WIDTH``).
PARTNER_LEVEL_WIDTH = 1.0e-4


class UnequalPartnerOccupations(ValueError):
    """``occup.in`` gives the partners of one state of weighted sectors different occupations.

    The one-dimensional sectors of a subgroup with weights hold one state for ``w`` degenerate partners and give
    them one occupation, and average the density over the full group, which gives the partners of a level that
    several solved states share (the e of Td in two sectors of D2d) the mean of their occupations; the file asks
    for a state that breaks the symmetry of those partners.  ``first`` is the position (from one) of the first
    partner in the list of the file, ``entries`` the entries of the partners (zero past ``States_Num``),
    ``energy`` the level (Ry), ``positions`` the positions of every partner where they are not consecutive.  The
    accelerated driver starts such a run again in the sectors of the abelian group.
    """

    def __init__(self, first: int, entries, energy: float, channel: str | None = None, positions=None) -> None:
        self.first = int(first)
        self.entries = tuple(float(value) for value in entries)
        self.energy = float(energy)
        self.channel = channel
        self.positions = (
            tuple(range(self.first, self.first + len(self.entries)))
            if positions is None
            else tuple(int(position) for position in positions)
        )
        where = "" if channel is None else f" of the {channel} channel"
        consecutive = self.positions == tuple(range(self.first, self.first + len(self.entries)))
        listed = (
            f"{self.first} to {self.first + len(self.entries) - 1}"
            if consecutive
            else ", ".join(str(position) for position in self.positions)
        )
        super().__init__(
            f"occup.in gives the {len(self.entries)} degenerate partners{where} at {self.energy:.6f} Ry (states "
            f"{listed} counted with their partners) the unequal "
            f"occupations {', '.join(f'{value:g}' for value in self.entries)}: the weighted symmetry sectors give "
            "every partner of a level one occupation (PARSEC_ORBITAL_GROUP_OCCUPATIONS=off keeps the abelian sectors)"
        )


def _shared_levels(eigenvalues: np.ndarray, partners, width: float) -> list[list[int]]:
    """Groups of two or more solved states that may be partners of one level (ascending state indices).

    Two states are joined where their levels lie within ``width`` and their partner masks share a bit (a
    representation of the full group that both sectors hold and whose partners several solved states share);
    the groups are the connected components of that relation.
    """

    masks = [int(mask) for mask in partners]
    candidates = [index for index, mask in enumerate(masks) if mask]
    parent = {index: index for index in candidates}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(candidates):
        for second in candidates[position + 1 :]:
            if eigenvalues[second] - eigenvalues[first] > width:
                break
            if masks[first] & masks[second]:
                parent[root(second)] = root(first)
    groups: dict[int, list[int]] = {}
    for index in candidates:
        groups.setdefault(root(index), []).append(index)
    return [members for members in groups.values() if len(members) > 1]


def partner_occupations(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    occupations: Sequence[float],
    *,
    channel: str | None = None,
    partners: Sequence[int] | None = None,
) -> np.ndarray:
    """The ``occup.in`` occupation of every state of weighted sectors, in ``[0, 1]`` per partner.

    ``eigenvalues`` are the sorted levels of the solved states and ``weights`` their weights ``w`` (the degenerate
    partners each stands for).  Counted with their partners they are the levels of the full grid, in order, so the
    ``n``-th entry of ``occupations`` (the list of the file, ``States_Num`` long) goes to the ``n``-th partner: a
    state of weight ``w`` takes ``w`` consecutive entries, zero past the end of the list (the partners of the last
    multiplet beyond ``States_Num``).  Raises :class:`UnequalPartnerOccupations` where those entries differ by more
    than :data:`PARTNER_OCCUPATION_TOLERANCE`.

    ``partners`` (one integer per state, a bit mask: ``CuPySymmetryEigvalResult.state_partners``): the
    representations of the full group held by the sector of each state whose partners several solved states share.
    Degenerate states (within :data:`PARTNER_LEVEL_WIDTH`) whose masks share a bit may be partners of one level,
    whose entries must then be equal as well: the density of the weighted sectors is averaged over the full group,
    which would give each of them the mean.  An accidental degeneracy where the entries change only sends the run
    to the abelian sectors.  A level that the end of the list cuts with its last partners past the solved states
    is not seen (the full grid keeps an arbitrary combination of such partners).
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(weights)
    if eigenvalues.ndim != 1 or eigenvalues.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if np.any(np.diff(eigenvalues) < 0):
        raise ValueError("eigenvalues must be sorted")
    if weights.shape != eigenvalues.shape or np.any(weights < 1) or np.any(weights != np.rint(weights)):
        raise ValueError("one positive integer weight per state is needed")
    repeats = np.rint(weights).astype(np.int64)
    values = np.asarray(occupations, dtype=float)
    stops = np.cumsum(repeats)
    starts = stops - repeats
    if int(stops[-1]) < values.size:
        raise ValueError(
            f"{values.size} fixed occupations were given for the {int(stops[-1])} computed states "
            "(partners counted)"
        )
    padded = np.zeros(int(stops[-1]), dtype=float)
    padded[: values.size] = values
    result = padded[starts].copy()
    for state in np.flatnonzero(repeats > 1):
        block = padded[starts[state] : stops[state]]
        if float(np.ptp(block)) > PARTNER_OCCUPATION_TOLERANCE:
            raise UnequalPartnerOccupations(int(starts[state]) + 1, block, float(eigenvalues[state]), channel)
    if partners is not None:
        if len(partners) != eigenvalues.size:
            raise ValueError("one partner mask per state is needed")
        for group in _shared_levels(eigenvalues, partners, PARTNER_LEVEL_WIDTH):
            block = np.concatenate([padded[starts[state] : stops[state]] for state in group])
            if float(np.ptp(block)) > PARTNER_OCCUPATION_TOLERANCE:
                positions = [int(position) + 1 for state in group for position in range(starts[state], stops[state])]
                raise UnequalPartnerOccupations(
                    positions[0], block, float(eigenvalues[group[0]]), channel, positions=positions
                )
    return result


def weighted_fixed_occupations(
    eigenvalues: np.ndarray,
    weights: np.ndarray,
    electron_count: float,
    occupations: Sequence[float],
    *,
    partners: Sequence[int] | None = None,
) -> OccupationResult:
    """:func:`fixed_occupations` for the states of weighted symmetry sectors (:func:`partner_occupations`).

    The returned occupations are per partner (without the weights), as those of
    :func:`parsec_python.Occupations.weighted_fermi_occupations`; ``electron_count`` of the result counts every
    partner.  The electrons of the file are checked as :func:`fixed_occupations` checks them.  ``partners``: the
    partner masks of the states (:func:`partner_occupations`).
    """

    values = np.array(occupations, dtype=float)
    check_fixed_occupations(values, electron_count)
    per_state = partner_occupations(eigenvalues, weights, values, partners=partners)
    return OccupationResult(
        fermi_level=float("nan"),
        occupations=per_state,
        electron_count=float(2.0 * np.dot(np.asarray(weights, dtype=float), per_state)),
    )


__all__ = [
    "ELECTRON_COUNT_TOLERANCE",
    "OccupationFile",
    "PARTNER_LEVEL_WIDTH",
    "PARTNER_OCCUPATION_TOLERANCE",
    "SpinOccupationFile",
    "UnequalPartnerOccupations",
    "check_fixed_occupations",
    "fixed_occupations",
    "partner_occupations",
    "read_occupation_file",
    "read_spin_occupation_file",
    "weighted_fixed_occupations",
]
