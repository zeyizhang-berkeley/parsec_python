"""Why an SCF loop that did not converge may not have been able to: an occupied highest computed state.

``States_Num`` fixes how many levels the eigensolver returns.  Where the occupation of the highest of them is not
negligible (a small gap or a high ``Fermi_Temp``), and that level belongs to a degenerate multiplet whose partners lie
beyond ``States_Num``, the eigensolver returns another part of the multiplet in every iteration: the density of a part
of a degenerate subspace depends on which part it is, so the density and the residual of the loop do not settle.  An
8-atom silicon cube at the Gamma point with ``States_Num: 20`` and ``Fermi_Temp: 300`` (the six-fold lowest conduction
level, folded from the three X points, cut after four of its states, each occupied 1e-4) keeps its residual near
2e-4 Ry for 200 iterations on both sets of routines, on the reference solver and on the full grid of the accelerated
one, with or without ``Symmetrize_Charge_Density`` (PARSEC's check keeps the identity alone for this conventional
cell, so that symmetrization averages nothing); 21 states (five of the six) stall alike.  With 22 or more states it
converges in 14-15 iterations (A100: 7.7 s against 25 s for the 200), and so do 20 states at 80 K or 1 K (17), where
the cut level holds nothing.  PARSEC 1.5 has no such test.

:func:`truncated_states_note` writes the note that ``parsec.out`` prints after ``Maximum SCF iterations reached
without convergence.``; it changes nothing in the calculation.
"""

from __future__ import annotations

import numpy as np

#: The occupation (electrons per state over the spin degeneracy) of the highest computed state above which an SCF loop
#: that did not converge is told that ``States_Num`` may be too small.
OCCUPIED_TOP = 1.0e-6
#: Levels within this many Ry of the highest computed one count as its multiplet.
MULTIPLET_WIDTH = 1.0e-5


def _blocks(result):
    """``(label, eigenvalues, occupations)`` of every eigenproblem of ``result``: the k-points (k-major) and the spin
    channels (up first) of the pooled arrays, or of the two channels of a spin-polarized k-point result."""

    k_points = getattr(result, "k_points", None)
    kpoints = 1 if k_points is None else max(int(np.asarray(k_points).reshape(-1, 3).shape[0]), 1)
    if getattr(result, "eigenvalues_up", None) is not None:
        blocks = []
        for channel in ("up", "down"):
            values = np.asarray(getattr(result, f"eigenvalues_{channel}"), dtype=np.float64).reshape(-1)
            filled = np.asarray(getattr(result, f"occupations_{channel}"), dtype=np.float64).reshape(-1)
            if values.size == 0 or values.size != filled.size or values.size % kpoints:
                continue
            size = values.size // kpoints
            for index in range(kpoints):
                label = channel + (f", k-point {index + 1}" if kpoints > 1 else "")
                part = slice(index * size, (index + 1) * size)
                blocks.append((label, values[part], filled[part]))
        return blocks
    eigenvalues = np.asarray(getattr(result, "eigenvalues", ()), dtype=np.float64).reshape(-1)
    occupations = np.asarray(getattr(result, "occupations", ()), dtype=np.float64).reshape(-1)
    if eigenvalues.size == 0 or eigenvalues.size != occupations.size:
        return []
    channels = 2 if getattr(result, "spin", None) is not None else 1
    pieces = kpoints * channels
    if eigenvalues.size % pieces:
        return [("", eigenvalues, occupations)]
    size = eigenvalues.size // pieces
    blocks = []
    for index in range(pieces):
        label = []
        if channels == 2:
            label.append("up" if index // kpoints == 0 else "down")
        if kpoints > 1:
            label.append(f"k-point {index % kpoints + 1}")
        part = slice(index * size, (index + 1) * size)
        blocks.append((", ".join(label), eigenvalues[part], occupations[part]))
    return blocks


def truncated_states_note(result) -> list[str]:
    """The lines of the note for an SCF ``result`` that did not converge, or ``[]`` (converged, or the highest computed
    state of every eigenproblem holds no more than :data:`OCCUPIED_TOP`)."""

    if getattr(result, "converged", True):
        return []
    worst = None
    for label, eigenvalues, occupations in _blocks(result):
        order = np.argsort(eigenvalues, kind="stable")
        top = int(order[-1])
        occupation = float(occupations[top])
        if occupation <= OCCUPIED_TOP:
            continue
        multiplet = int(np.count_nonzero(eigenvalues[top] - eigenvalues <= MULTIPLET_WIDTH))
        if worst is None or occupation > worst[1]:
            worst = (label, occupation, multiplet, eigenvalues.size, float(eigenvalues[top]))
    if worst is None:
        return []
    label, occupation, multiplet, states, level = worst
    where = f" ({label})" if label else ""
    group = (
        f", the last of {multiplet} levels within {MULTIPLET_WIDTH:.0e} Ry of each other"
        if multiplet > 1
        else ""
    )
    return [
        f" NOTE: the highest computed state{where}, state {states} at {level:.6f} Ry{group}, holds an occupation of "
        f"{occupation:.1e}.",
        "   Where its level is degenerate with levels beyond States_Num, the eigensolver returns another part of",
        "   that multiplet in every iteration and the density cannot settle; raise States_Num past the multiplet",
        "   (or lower Fermi_Temp).",
    ]


__all__ = ["MULTIPLET_WIDTH", "OCCUPIED_TOP", "truncated_states_note"]
