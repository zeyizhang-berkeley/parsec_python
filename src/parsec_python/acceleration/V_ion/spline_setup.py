"""The ``Cubic_Spline`` core densities of a periodic preparation, bit for bit.

:func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density` evaluates a core density with
``Cubic_Spline`` through :meth:`~parsec_python.Pseudopotential.ParsecPseudopotential.interpolate_density`, which
builds the spline of the whole table (a Python recurrence over its 400 to 1000 samples) for every image of every
atom: about 1 ms per image, and a slab sums its images out to the end of the table (118 bohr for a POTRE file), so a
germanium slab of 8 atoms spent 22 s on its core density and a 64-atom cell 2.3 s (A100 node, before 2026-11-22).
The device sums of the linear tables do not apply: the spline's ``x**3`` is the library's ``pow``.

Here (``PARSEC_SPLINE_SETUP=auto``, the default) the same loop runs with the spline of each species built once
(:meth:`~parsec_python.Pseudopotential.ParsecPseudopotential.density_spline`: the same knots and second derivatives),
the terms of the images formed on the threads of the periodic setup (``PARSEC_PERIODIC_SETUP_THREADS``) in blocks of
consecutive images of one atom: the grid points and distances of each image from the installed support finder (a
wire or slab block, also of Andrew Grieder's slab grid, from one dense scan of its images, the rows and distances of
``dense_support``), the points inside
the support, and the values of the whole block from one call of the readable interpolation (elementwise functions of
the distances: the values of the separate calls).  The calling thread adds them into the density in the order of the
readable loop (atom, then image): the density is the readable one bit for bit.  ``readable`` restores the loop.
Valence densities (always interpolated linearly, as ``initchrg.f90``) and linear core densities keep their own sums.
"""

from __future__ import annotations

import os

import numpy as np

SWITCH = "PARSEC_SPLINE_SETUP"
SETTINGS = ("auto", "readable")
DEFAULT_SETTING = "auto"

#: Host bytes of the image terms in flight: the finished terms the calling thread has not added yet and the
#: temporaries of the workers.
_PENDING_BYTES = 1 << 30
#: Arrays of 8 bytes per point of a block that one worker holds at its peak (the coordinates relative to an image,
#: the distances, rows, the selection, the values and the spline's temporaries), an upper bound.
_WORKER_ARRAYS = 16
#: The images of one atom that a worker takes together: their balls hold about this many points, at most this many
#: images (one call of the spline for all of them: the Python steps of a call are most of the time of a small ball).
_BLOCK_POINTS = 1 << 19
_BLOCK_IMAGES = 64


def spline_setup_setting() -> str:
    """``PARSEC_SPLINE_SETUP``: ``auto`` (default) or ``readable``."""

    value = os.environ.get(SWITCH, DEFAULT_SETTING).strip().lower() or DEFAULT_SETTING
    if value not in SETTINGS:
        raise ValueError(f"{SWITCH} must be one of {', '.join(SETTINGS)}, got {value!r}")
    return value


def _takes_grid(grid) -> bool:
    """Every periodic grid: those whose readable sum takes the images of :func:`atomic_density_images` with the
    default dimensions, and Andrew Grieder's slab, whose caller passes two (:func:`_slab_grid`)."""

    from parsec_python.Grid.pbc import PeriodicRealSpaceGrid

    return isinstance(grid, PeriodicRealSpaceGrid)


def _slab_grid(grid) -> bool:
    """Andrew Grieder's slab grid: its images run along the two in-plane vectors to the end of the table
    (``periodic_dimensions=2`` of the readable sum) and its support finder is the dense scan."""

    from parsec_python.Grid.pbc import SlabRealSpaceGrid

    return isinstance(grid, SlabRealSpaceGrid)


def spline_superposed_density(grid, atoms, potentials, specifications, lattice_vectors, core: bool):
    """The core density of :func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density` with the splines built
    once and the image terms on threads, bit for bit, or ``None``: a valence density, no ``Cubic_Spline`` core among
    the atoms, a grid it does not take, ``PARSEC_SPLINE_SETUP=readable``."""

    if not core or lattice_vectors is None or not _takes_grid(grid):
        return None
    contributing = [atom for atom in atoms if potentials[atom.symbol].has_nonlinear_core_correction]
    if not any(specifications[atom.symbol].use_spline for atom in contributing):
        return None
    if spline_setup_setting() != "auto":
        return None
    from parsec_python.V_ion import ionic_potential as _ionic

    from .periodic_setup_fortran import _ordered_results, setup_threads

    padding = grid.settings.stencil_half_width
    volume_element = float(getattr(grid, "volume_element", 0.0) or 0.0)
    slab = _slab_grid(grid)
    # a wire or slab of the Fortran-derived routines and Andrew Grieder's slab: the dense scan of dense_support
    confined = bool(getattr(grid, "is_confined_periodic", False)) or slab
    columns = [np.asarray(grid.coordinates)[:, axis] for axis in range(3)] if confined else None
    splines = {}
    items = []
    largest = 1
    for atom in contributing:
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        if specification.use_spline and atom.symbol not in splines:
            splines[atom.symbol] = potential.density_spline(padding, core=True)
        position = np.asarray(atom.position, dtype=np.float64)
        support, translations = _ionic.atomic_density_images(
            grid, position, potential, specification, lattice_vectors, core=True, periodic_dimensions=2 if slab else 3
        )
        # the images of one atom in blocks of about _BLOCK_POINTS points of their balls, in the readable order
        ball = int(grid.size)
        if volume_element > 0.0:
            ball = max(1, min(ball, int(4.0 / 3.0 * np.pi * support**3 / volume_element)))
        per_block = max(1, min(_BLOCK_IMAGES, _BLOCK_POINTS // ball))
        largest = max(largest, per_block * ball)
        centres = position[None, :] + np.asarray(translations, dtype=np.float64).reshape(-1, 3)
        for start in range(0, len(centres), per_block):
            items.append((atom.symbol, support, centres[start : start + per_block]))

    def work(item):
        # every image's points and distances from the installed finder, those inside the support, and the values of
        # all of them from one call of the readable interpolation (elementwise functions of the distances)
        symbol, support, centres = item
        specification = specifications[symbol]
        found = []
        if confined:
            # a wire or slab: the dense scan of every image of the block at once (the rows and distances of
            # dense_support: sqrt((dx^2 + dy^2) + dz^2) of the coordinates minus the centre, r <= support)
            square = columns[0][None, :] - centres[:, 0:1]
            square *= square
            offset = columns[1][None, :] - centres[:, 1:2]
            offset *= offset
            square += offset
            offset = columns[2][None, :] - centres[:, 2:3]
            offset *= offset
            square += offset
            del offset
            distances = np.sqrt(square, out=square)
            for distance in distances:
                rows = np.flatnonzero(distance <= support)
                radius = distance[rows]
                inside = radius < support
                if np.any(inside):
                    found.append((rows[inside], radius[inside]))
        else:
            for centre in centres:
                rows, _, radius = _ionic._support_finder(grid, centre, support)
                inside = radius < support
                if np.any(inside):
                    found.append((rows[inside], radius[inside]))
        if not found:
            return ()
        radius = found[0][1] if len(found) == 1 else np.concatenate([image[1] for image in found])
        values = potentials[symbol].interpolate_density(
            radius,
            core=True,
            use_spline=specification.use_spline,
            spline_padding_width=padding,
            spline=splines.get(symbol),
        )
        sizes = np.cumsum([image[0].size for image in found])[:-1]
        return [(image[0], part) for image, part in zip(found, np.split(values, sizes))]

    density = np.zeros(grid.size, dtype=float)
    block = 8 * largest
    threads = max(1, min(setup_threads(), (_PENDING_BYTES - 2 * 2 * block) // (_WORKER_ARRAYS * block)))
    for images in _ordered_results(work, items, threads):
        for rows, values in images:
            density[rows] += values
    return density


__all__ = ["DEFAULT_SETTING", "SETTINGS", "SWITCH", "spline_setup_setting", "spline_superposed_density"]
