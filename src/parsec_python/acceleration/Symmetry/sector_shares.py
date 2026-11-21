"""Shares of the lowest levels among weighted sectors, from the valence orbitals of the atoms
(``PARSEC_WEIGHTED_FIRST_SHARE``).

The first solve of weighted sectors (:class:`.axis_reflection.PointGroupReduction`) gives every sector the even
share ``requested / sum(weights)`` of the states with a margin of 10 % (``Eigensolvers/symmetry.first_sector_count``).
The sectors do not hold even shares of the lowest levels: atoms on the axes and planes of the group make some
irreducible representations more frequent among them than others.  The Td nanodiamonds in the sectors of D2d
(weights 1, 1, 3, 3) hold 1.094, 0.906, 0.906 and 1.094 times the even share of their occupied levels at k06 and
1.041, 0.958, 0.958, 1.042 at k14, so that two sectors take 10 % more columns than they need, plus the 4-9 % they
hold less, and the other two are close to their margin.

The valence orbitals of the atoms (the channels of each pseudopotential with valence electrons: ``s`` for H, ``s``
and ``p`` for C) span a representation of the group whose character at an operation ``h`` is the sum, over the
atoms that ``h`` leaves in place, of the characters ``chi_l(h)`` of their channels.  Its multiplicity in the
one-dimensional character of a sector ``s`` (:meth:`ReflectionRepresentationDecomposition.characters`),

    n_s = (1 / |H|) sum_h chi_s(h) sum_{a: h a = a} sum_l chi_l(h),

is the number of those orbitals the sector holds.  For the nanodiamonds k02 to k14 ``n_s / sum_s w_s n_s`` equals the
measured share of the occupied levels of every sector to the three digits above (the occupied levels are the bonding
combinations of the valence orbitals), and for bare silicon clusters (Si29, Si47, Si71, Td) to within one level.
:func:`predicted_level_shares` turns the counts into the share of each sector of the requested states: those up to
the count of the valence orbitals by these shares, the rest evenly.
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Sequence

import numpy as np

SWITCH = "PARSEC_WEIGHTED_FIRST_SHARE"


#: The value where none is named (since 2026-11-21; ``even`` before).
DEFAULT = "auto"
#: The smallest basis of the full grid (grid points times the states and the buffer) from which ``auto`` takes the
#: shares of the valence orbitals in one process (2^27).  First runs (empty kernel caches) on one node of 4 A100, even
#: -> orbitals, SCF seconds: nanodiamonds k06 16.3 -> 11.0, k08 39.0 -> 19.5, k10 135.9 -> 56.4 (the counts of the
#: shares let the placement give each sector the device pool); the negative diamond vacancy, spin-polarized, 4^3
#: (3.8e8 entries) 21.5 -> 20.9 and 5^3 (1.5e9) 25.1 -> 13.6; but 3^3 (6.9e7) 9.6 -> 13.4 on a first run (its second
#: run 4.7 -> 3.2), and molecules and clusters on one A100 (SF6, CF4, Na55, Si86, Si113, CH4, H2) the same times.
MINIMUM_ENTRIES = 1 << 27


def weighted_first_share_setting() -> str:
    """``PARSEC_WEIGHTED_FIRST_SHARE``: ``auto`` (the default since 2026-11-21) is ``orbitals`` in one process and
    ``even`` in the ranks of an MPI run (:func:`weighted_first_share_route`); ``even`` (the default before) gives every
    weighted sector the even share of the states at its first solve with a margin of 10 %, as before; ``orbitals``
    (2026-11-19) gives it the share of the valence orbitals of the atoms that it holds
    (:func:`valence_orbital_sector_counts`) with a margin of 3 % (a run without whole counts or with the same count
    in every sector, :func:`attach_valence_orbital_counts`, keeps the even share).  ``auto`` keeps the even share in
    MPI runs: in the MPI runner on two nodes the SCF of the nanodiamond k10 took 14.1 s with ``orbitals`` instead of
    12.8 s and 9.1 instead of 8.1 GiB, since the smaller counts of two sectors fell below the work at which a basis is
    divided over the devices of a group (``sector_bases_divisible``), which changed the device layout of the
    sectors; one process with the GPUs of a node was faster with ``orbitals`` (README, the row of the switch)."""

    value = os.environ.get(SWITCH, DEFAULT).strip().lower() or DEFAULT
    if value not in {"auto", "even", "orbitals"}:
        raise ValueError(f"{SWITCH} must be auto, even or orbitals, not {value!r}")
    return value


def weighted_first_share_route(*, mpi: bool, entries: int | None = None) -> str:
    """The share of the first counts of weighted sectors: ``orbitals`` or ``even``.  ``auto`` is ``orbitals`` in one
    process from :data:`MINIMUM_ENTRIES` of the full-grid basis on (``entries``: grid points times the states and the
    buffer; ``None``, not known, keeps the even share) and ``even`` where the run has ranks (``mpi``: an MPI context,
    the same on every rank) or a smaller basis."""

    value = weighted_first_share_setting()
    if value == "auto":
        return "even" if mpi or entries is None or int(entries) < MINIMUM_ENTRIES else "orbitals"
    return value


def angular_character(l: int, matrix) -> float:
    """The character of the ``2 l + 1`` real spherical harmonics of degree ``l`` at the orthogonal ``matrix``.

    ``matrix = det * R`` with ``R`` a rotation by ``theta`` (``trace R = 1 + 2 cos theta``): ``det^l`` times
    ``sum_{m=-l}^{l} exp(i m theta)`` (``Y_lm(-r) = (-1)^l Y_lm(r)``).
    """

    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    det = 1.0 if np.linalg.det(matrix) > 0.0 else -1.0
    proper = det * matrix
    cosine = min(1.0, max(-1.0, 0.5 * (float(np.trace(proper)) - 1.0)))
    theta = math.acos(cosine)
    value = 1.0 + 2.0 * sum(math.cos(m * theta) for m in range(1, int(l) + 1))
    return float(det ** int(l) * value)


def valence_channels(pseudopotentials: Mapping[str, object]) -> dict[str, tuple[int, ...]] | None:
    """The angular momenta with valence electrons of every species (``channel_occupations`` of its table), or
    ``None`` where a table does not give them."""

    channels: dict[str, tuple[int, ...]] = {}
    for symbol, table in pseudopotentials.items():
        occupations = getattr(table, "channel_occupations", None)
        if not occupations:
            return None
        found = tuple(sorted(int(l) for l, value in dict(occupations).items() if float(value) > 0.0))
        if not found:
            return None
        channels[str(symbol)] = found
    return channels


def valence_orbital_sector_counts(
    decomposition,
    atoms: Sequence[object],
    channels: Mapping[str, Sequence[int]],
    grid,
    *,
    tolerance: float = 1.0e-7,
) -> np.ndarray | None:
    """``n_s``: how many valence orbitals of ``atoms`` every solved sector of ``decomposition`` holds, or ``None``.

    ``None`` for a reduction that is not a :class:`.axis_reflection.PointGroupReduction`, an atom of a species
    without channels, an image of an atom that matches no atom within ``tolerance`` (bohr), or counts that are not
    whole numbers.
    """

    from .axis_reflection import PointGroupReduction, atom_image_table

    reduction = getattr(decomposition, "reduction", None)
    if not isinstance(reduction, PointGroupReduction) or not atoms:
        return None
    if any(str(getattr(atom, "symbol", "")) not in channels for atom in atoms):
        return None
    operations = np.asarray(reduction.operations, dtype=np.int64).reshape(-1, 3, 3)
    images = atom_image_table(atoms, operations, getattr(reduction, "translations", None), grid, tolerance)
    if images is None:
        return None
    characters = np.asarray(decomposition.characters, dtype=np.float64)
    if characters.shape[1] != operations.shape[0]:
        return None
    identity = np.arange(len(atoms))
    symbols = [str(atom.symbol) for atom in atoms]
    trace = np.zeros(operations.shape[0], dtype=np.float64)
    for index, matrix in enumerate(operations):
        cache: dict[int, float] = {}
        for atom in np.flatnonzero(images[index] == identity):
            for l in channels[symbols[atom]]:
                if l not in cache:
                    cache[l] = angular_character(l, matrix)
                trace[index] += cache[l]
    counts = characters @ trace / float(operations.shape[0])
    rounded = np.rint(counts)
    if np.any(np.abs(counts - rounded) > 1.0e-6) or np.any(rounded < 0.0):
        return None
    return rounded.astype(np.int64)


def predicted_level_shares(counts, weights, requested_states: int) -> np.ndarray | None:
    """The share of every sector of the lowest ``requested_states`` levels (a level of sector ``s`` counts once;
    ``sum_s w_s share_s = 1``): those up to the count of valence orbitals ``sum_s w_s n_s`` by the shares of the
    orbitals ``n_s / sum_s w_s n_s``, the rest evenly ``1 / sum_s w_s``.  ``None`` without orbitals."""

    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights * counts))
    if counts.shape != weights.shape or total <= 0.0 or requested_states <= 0:
        return None
    requested = float(requested_states)
    covered = min(requested, total)
    levels = covered * counts / total + (requested - covered) / float(np.sum(weights))
    return levels / requested


def attach_valence_orbital_counts(decomposition, atoms, pseudopotentials, grid, *,
                                  tolerance: float = 1.0e-7, mpi: bool = False,
                                  entries: int | None = None) -> np.ndarray | None:
    """Give a weighted ``decomposition`` the counts of the valence orbitals of ``atoms`` in its sectors
    (``valence_orbital_counts``, :func:`valence_orbital_sector_counts`) where ``PARSEC_WEIGHTED_FIRST_SHARE`` gives
    ``orbitals`` (:func:`weighted_first_share_route`: ``auto`` in one process, ``mpi`` false, from ``entries`` of
    :data:`MINIMUM_ENTRIES` on);
    the first counts of its sectors then take their shares (``Eigensolvers/symmetry.first_level_share``).  The
    counts, or ``None`` where none are set (the route ``even``, no weights, no channels, no whole counts, or the
    same count in every sector: their shares are the even ones, and the even share with its margin of 10 % stays,
    as for the bulk cells of diamond and silicon, whose one fewer margin took an SCF step more on A100).

    Every MPI rank prepares the same decomposition from the same atoms and tables, so all of them place and size
    the sectors alike.
    """

    # a decomposition kept from an earlier preparation (the geometry cache) drops the counts of that run first
    if "valence_orbital_counts" in getattr(decomposition, "__dict__", {}):
        object.__delattr__(decomposition, "valence_orbital_counts")
    if weighted_first_share_route(mpi=mpi, entries=entries) != "orbitals":
        return None
    weights = getattr(decomposition, "sector_weights", None)
    weights = np.zeros(0, dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
    if weights.size == 0 or not np.any(weights != 1):
        return None
    channels = valence_channels(pseudopotentials or {})
    if channels is None:
        return None
    counts = valence_orbital_sector_counts(decomposition, atoms, channels, grid, tolerance=tolerance)
    if counts is None or float(np.sum(weights * counts)) <= 0.0 or bool(np.all(counts == counts[0])):
        return None
    object.__setattr__(decomposition, "valence_orbital_counts", np.asarray(counts, dtype=np.int64))
    return counts


__all__ = [
    "MINIMUM_ENTRIES",
    "SWITCH",
    "angular_character",
    "attach_valence_orbital_counts",
    "predicted_level_shares",
    "valence_channels",
    "valence_orbital_sector_counts",
    "weighted_first_share_route",
    "weighted_first_share_setting",
]
