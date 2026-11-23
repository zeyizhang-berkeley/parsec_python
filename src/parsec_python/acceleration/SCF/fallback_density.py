"""The density of the full grid that a spin cluster leaves its symmetry sectors for (``PARSEC_SPIN_FALLBACK_DENSITY``).

Where the symmetry guard of :mod:`.spin_sectors` stops a spin-polarized cluster in its sectors, the run starts again
on the full grid (``Spin_Symmetry_Sectors: off``), which keeps no symmetry at all.  Atoms and linear molecules with a
partly filled degenerate shell then have a continuous family of states of one energy (the orientation of the
occupied orbitals); the loop drifts along it and does not converge: the Fe atom (PBE, 100 K) keeps its weighted
residual near 5e-4 Ry for 150 iterations with its energy still falling, the broken-symmetry O2 pair whose start
Hamiltonian stops the sectors keeps 3e-4 Ry.  PARSEC, and the reference solver of this port, keep the orbitals of a
cluster in the irreducible representations of the abelian group of the atoms (the operations that only change the
signs of coordinates, ``symmetry_detection`` of PARSEC's ``Found Abelian subgroup``), so the density of each channel
keeps that group, the orientation is pinned to the axes and the same inputs converge (Fe atom 28 iterations, O2 25).

``PARSEC_SPIN_FALLBACK_DENSITY``:

* ``off`` (the default): the full grid keeps no symmetry, as before;
* ``abelian``: the full grid of one process averages the density of each channel over that group in every iteration
  (the path of ``Symmetrize_Charge_Density``, :func:`parsec_python.Grid.symmetrize_density`), so that it reaches the
  state PARSEC's abelian group gives.  That state may lie above a state that breaks the group too (the Fe atom:
  6.9 mRy above the energy the full grid drifts to), which is why this is not the default.  Clusters only; the shared
  layouts of ``PARSEC_SPIN_FALLBACK_SHARED`` and the MPI ranks keep the full grid without symmetry.
"""

from __future__ import annotations

import os

import numpy as np

SWITCH = "PARSEC_SPIN_FALLBACK_DENSITY"

#: The eight operations that change the signs of coordinates (PARSEC's abelian candidates of a cluster).
AXIS_REFLECTIONS = np.array(
    [np.diag([a, b, c]) for a in (1.0, -1.0) for b in (1.0, -1.0) for c in (1.0, -1.0)], dtype=np.float64
)


def fallback_density_setting() -> str:
    """The value of ``PARSEC_SPIN_FALLBACK_DENSITY`` (``off`` or ``abelian``); any other value is an error."""

    value = os.environ.get(SWITCH, "off").strip().lower() or "off"
    if value not in ("off", "abelian"):
        raise ValueError(f"{SWITCH} must be off or abelian, got {value!r}")
    return value


def abelian_density_group(system):
    """The axis-reflection group of the atoms of ``system`` on its grid, ``None`` where it holds only the identity or
    does not map the grid onto itself (a periodic cell, a grid shifted otherwise)."""

    if getattr(system.input, "periodic_cell", None) is not None:
        return None
    from parsec_python.SCF.single_point import density_point_group

    try:
        group = density_point_group(system.grid, system.atoms, candidates=AXIS_REFLECTIONS)
    except ValueError:
        return None
    return group if group.order > 1 else None


def _set_group(system, group) -> None:
    """Put ``group`` (or ``None``) on ``system``; an accelerated system reads that of its (frozen) reference."""

    reference = getattr(system, "reference", None)
    if reference is not None:
        from dataclasses import replace

        system.reference = replace(reference, density_symmetry=group)
    else:
        system.density_symmetry = group


def with_fallback_density(system) -> str | None:
    """Set the density group of the full-grid ``system`` under ``abelian``; the text of the NOTE, ``None`` otherwise.

    A system that symmetrizes its density already (``Symmetrize_Charge_Density``) keeps its own group."""

    if fallback_density_setting() == "off" or getattr(system, "density_symmetry", None) is not None:
        return None
    group = abelian_density_group(system)
    if group is None:
        return None
    _set_group(system, group)
    updater = getattr(system, "geometry_updater", None)
    if callable(updater) and not getattr(updater, "keeps_fallback_density", False):
        # An ionic step returns the system of the moved atoms from the reference the preparation keeps, without the
        # group: the group of the moved atoms goes onto every such system (None where they keep only the identity).
        def pinned(current, atoms, _update=updater):
            moved = _update(current, atoms)
            _set_group(moved, abelian_density_group(moved))
            return moved

        pinned.keeps_fallback_density = True
        system.geometry_updater = pinned
    return (
        f"the density of each channel averaged over the abelian group of the atoms ({group.order} operations, "
        f"{SWITCH}=abelian)"
    )


__all__ = ["AXIS_REFLECTIONS", "SWITCH", "abelian_density_group", "fallback_density_setting", "with_fallback_density"]
