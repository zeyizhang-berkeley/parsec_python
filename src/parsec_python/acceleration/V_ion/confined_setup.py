"""Fast preparation of a wire or slab (:class:`parsec_python.Grid.confined.ConfinedPeriodicGrid`), bit for bit.

The readable sums of a wire or slab measure the distance of every grid point to every image of every atom:

* :func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density` (the starting and the core densities) sums the
  images along the periodic directions out to the end of the radial table (``rs(ns-1)``, 75 to 78 bohr for the FHI
  tables), so the ball of nearly every image holds the whole grid, and the sum costs ``atoms x images x points``:
  283 s of the preparation of a 72-atom graphene sheet, 536 s of a 72-atom h-BN sheet (A100 node, 2026-11-22);
* the support finder of the projectors (:func:`parsec_python.V_ion.ionic_potential.dense_support`) scans the whole
  grid for each image of a ball of a few bohr.

Here, under ``PARSEC_CONFINED_SETUP=auto`` (the default):

``confined_ball``
    the points of a ball from the index box around its centre: along lattice direction ``m`` a point within ``R`` of
    the centre lies within ``R |inverse(A_n)[m, :]|`` of it in the lattice coordinate ``u_m = (i_m + s_m) h_m``, so
    the box holds every point of the ball (one index of margin for round-off); its rows are taken in increasing
    order, the order of the dense scan, with the same displacements and distances (the same expressions on the same
    coordinates).

``device_confined_atomic_density``
    the image sum of the densities on the device, one thread per grid point adding the terms of the atoms and their
    images in the order of the readable loop (atom, then translation), each term with the host's operations: the
    offset ``x - c`` from the centre ``c = position + T`` the host forms, ``(dx^2 + dy^2) + dz^2`` (the order of
    ``np.linalg.norm``), the correctly rounded square root, ``r < rs(ns-1)``, and ``np.interp``'s value
    ``slope_j (r - x_j) + y_j`` (``y_j`` at ``r = x_j``, the first value below the first sample), all rounded to
    nearest without fused multiply-adds.  The host adds the terms of one image into the points of its ball in the same
    order, so the density is the readable one bit for bit.  ``Cubic_Spline`` core densities, a run without a usable
    device (``PARSEC_DEVICE=cpu``, no CuPy) and ``PARSEC_CONFINED_SETUP=readable`` keep the readable sum.

``PARSEC_CONFINED_SETUP=readable`` restores the dense scans (the same numbers, for comparison and timing).
"""

from __future__ import annotations

import math
import os

import numpy as np

SWITCH = "PARSEC_CONFINED_SETUP"
SETTINGS = ("auto", "readable")
DEFAULT_SETTING = "auto"


def confined_setup_setting() -> str:
    """``PARSEC_CONFINED_SETUP``: ``auto`` (default, the fast sums of this module) or ``readable``."""

    value = os.environ.get(SWITCH, DEFAULT_SETTING).strip().lower() or DEFAULT_SETTING
    if value not in SETTINGS:
        raise ValueError(f"{SWITCH} must be one of {', '.join(SETTINGS)}, got {value!r}")
    return value


def confined_grid(grid) -> bool:
    """Whether ``grid`` is the grid of a wire or slab of the Fortran-derived routines."""

    return bool(getattr(grid, "is_confined_periodic", False))


def handles(grid) -> bool:
    """Whether the fast sums of this module take ``grid`` (a wire or slab grid under ``auto``)."""

    return confined_grid(grid) and confined_setup_setting() == "auto"


# ---------------------------------------------------------------------------
# Balls from index boxes.


def _box_geometry(grid):
    """``(inverse, widths)`` of the grid: ``u = inverse @ r`` are the lattice coordinates of a Cartesian point and
    ``widths[m] = |inverse[m, :]|`` bounds ``|du_m|`` per bohr of ``|dr|``."""

    normalized = np.asarray(grid.normalized_lattice, dtype=float)
    inverse = np.linalg.inv(normalized)
    return inverse, np.linalg.norm(inverse, axis=1)


def _dense_ball(grid, centre: np.ndarray, radius: float):
    """:func:`parsec_python.V_ion.ionic_potential.dense_support` with the distances from the coordinate columns,
    ``sqrt((dx^2 + dy^2) + dz^2)`` (the operations of ``np.linalg.norm`` without its strided reduction), and the
    displacements of the kept points only: the same rows, displacements and distances."""

    coordinates = grid.coordinates
    square = coordinates[:, 0] - centre[0]
    square *= square
    offset = coordinates[:, 1] - centre[1]
    offset *= offset
    square += offset
    offset = coordinates[:, 2] - centre[2]
    offset *= offset
    square += offset
    del offset
    distances = np.sqrt(square, out=square)
    rows = np.flatnonzero(distances <= radius)
    return rows, coordinates[rows] - centre, distances[rows]


def confined_ball(grid, centre: np.ndarray, radius: float):
    """``(rows, relative, distances)`` of :func:`parsec_python.V_ion.ionic_potential.dense_support` on a wire or slab
    grid, from the index box around ``centre`` (the same rows in the same order, the same values).

    A box of at least half the index box (the 75-bohr balls of the readable density sums, which a run without a
    device takes) is the dense scan itself (:func:`_dense_ball`): the slice, sort and gather of the box would only
    add to it.
    """

    inverse, widths = _box_geometry(grid)
    centre = np.asarray(centre, dtype=float)
    lattice_centre = inverse @ centre
    spacings = np.asarray(grid.spacings, dtype=float)
    shift = np.asarray(grid.shift, dtype=float)
    low_index = np.asarray(grid.index_min, dtype=np.int64)
    high_index = np.asarray(grid.index_max, dtype=np.int64)
    ranges = []
    for axis in range(3):
        half = float(radius) * float(widths[axis])
        centre_index = float(lattice_centre[axis]) / spacings[axis] - shift[axis]
        reach = half / spacings[axis]
        # one index of margin on each side for round-off, the box of the grid's indices
        low = max(math.ceil(centre_index - reach) - 1, int(low_index[axis]))
        high = min(math.floor(centre_index + reach) + 1, int(high_index[axis]))
        if low > high:
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty((0, 3)), np.empty(0)
        ranges.append((low - int(low_index[axis]), high - int(low_index[axis])))
    (i0, i1), (j0, j1), (k0, k1) = ranges
    lookup = np.asarray(grid.lookup)
    if 2 * (i1 - i0 + 1) * (j1 - j0 + 1) * (k1 - k0 + 1) >= lookup.size:
        return _dense_ball(grid, centre, radius)
    rows = lookup[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1].reshape(-1)
    rows = np.sort(rows[rows >= 0])
    relative = grid.coordinates[rows] - centre
    distances = np.linalg.norm(relative, axis=1)
    keep = distances <= radius
    return rows[keep], relative[keep], distances[keep]


# ---------------------------------------------------------------------------
# Superposed atomic densities on the device.

_THREADS = 128

_CONFINED_DENSITY_KERNEL = r"""
__device__ __forceinline__ double numpy_interp(
    const double r, const double* xp, const double* fp, const double* slopes, const int n, const double left)
{
    if (r < xp[0]) return left;
    if (r >= xp[n - 1]) return fp[n - 1];
    int low = 0, high = n - 1;  // xp[low] <= r < xp[high]
    while (high - low > 1) {
        const int middle = (low + high) >> 1;
        if (xp[middle] <= r) low = middle; else high = middle;
    }
    if (xp[low] == r) return fp[low];
    return __dadd_rn(__dmul_rn(slopes[low], __dsub_rn(r, xp[low])), fp[low]);
}

// superpose_atomic_density on a wire or slab grid: every point adds the terms of the image centres in the order of
// the readable loop (atom, then translation), each term r < support with np.interp's value.
extern "C" __global__
void confined_atomic_density(
    const long long point_count, const double* coordinates,
    const long long centre_count, const double* centres, const int* kinds,
    const double* radii, const double* tables, const double* slopes, const int* offsets, const int* counts,
    const double* supports, const double* bounds,
    double* density)
{
    const long long point = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    const double px = coordinates[3 * point + 0];
    const double py = coordinates[3 * point + 1];
    const double pz = coordinates[3 * point + 2];
    double total = 0.0;
    for (long long c = 0; c < centre_count; ++c) {
        const int kind = kinds[c];
        const double bound = bounds[kind];
        const double dx = __dsub_rn(px, centres[3 * c + 0]);
        const double sx = __dmul_rn(dx, dx);
        if (sx > bound) continue;
        const double dy = __dsub_rn(py, centres[3 * c + 1]);
        const double planar = __dadd_rn(sx, __dmul_rn(dy, dy));
        if (planar > bound) continue;
        const double dz = __dsub_rn(pz, centres[3 * c + 2]);
        const double square = __dadd_rn(planar, __dmul_rn(dz, dz));
        if (square > bound) continue;
        const double r = __dsqrt_rn(square);
        if (!(r < supports[kind])) continue;
        const int offset = offsets[kind];
        total = __dadd_rn(total, numpy_interp(r, radii + offset, tables + offset, slopes + offset, counts[kind],
                                              tables[offset]));
    }
    density[point] = total;
}
"""

_COMPILED: dict[int, object] = {}


def _density_kernel(cp):
    device = int(cp.cuda.Device().id)
    if device not in _COMPILED:
        kernel = cp.RawKernel(_CONFINED_DENSITY_KERNEL, "confined_atomic_density", options=("--std=c++11",))
        kernel.compile()
        _COMPILED[device] = kernel
    return _COMPILED[device]


def _density_tables(atoms, potentials, specifications, core: bool):
    """``(contributing atoms, symbols, tables)``: the atoms that add a density, their species and the radial table of
    each (``None`` for a table the device does not interpolate: a ``Cubic_Spline`` core density)."""

    from parsec_python.Pseudopotential.radial_derivatives import valence_density_table

    contributing = [atom for atom in atoms if not core or potentials[atom.symbol].has_nonlinear_core_correction]
    symbols = list(dict.fromkeys(atom.symbol for atom in contributing))
    tables = {}
    for symbol in symbols:
        potential = potentials[symbol]
        specification = specifications[symbol]
        if core:
            if specification.use_spline:
                return contributing, symbols, None
            tables[symbol] = np.asarray(potential.core_density, dtype=np.float64)
        else:
            # the table of the readable loop: the filtered density, the file's, or the one rebuilt from u_l
            tables[symbol] = np.asarray(
                valence_density_table(potential, read_from_file=specification.read_valence_density), dtype=np.float64
            )
    return contributing, symbols, tables


def image_centres(grid, atoms, potentials) -> tuple[np.ndarray, np.ndarray]:
    """``(centres, owners)``: ``position + T`` for every translation ``T`` of
    :meth:`~parsec_python.Grid.confined.ConfinedPeriodicGrid.image_translations` (out to ``rs(ns-1)``) of every atom,
    in the order of the readable loop, and the index of the atom of each."""

    centres, owners = [], []
    for index, atom in enumerate(atoms):
        position = np.asarray(atom.position, dtype=np.float64)
        support = potentials[atom.symbol].interpolation_cutoff
        translations = grid.image_translations(position, support)
        # position + translation, the host's centre of each image
        centres.append(position[None, :] + translations)
        owners.append(np.full(len(translations), index, dtype=np.int64))
    if not centres:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate(centres), np.concatenate(owners)


def device_confined_atomic_density(grid, atoms, potentials, specifications, lattice_vectors, core: bool):
    """The density of :func:`parsec_python.V_ion.ionic_potential.superpose_atomic_density` on a wire or slab grid,
    summed on the device bit for bit, or ``None`` (another grid, ``PARSEC_CONFINED_SETUP=readable``, no usable device,
    a ``Cubic_Spline`` core density): the readable sum then runs."""

    if lattice_vectors is None or not handles(grid):
        return None
    from .periodic_device import _cupy, device_id

    cp = _cupy()
    if cp is None:
        return None
    contributing, symbols, tables = _density_tables(atoms, potentials, specifications, core)
    if tables is None:
        return None
    points = int(grid.size)
    if not contributing:
        return np.zeros(points, dtype=float)
    radii, values, slopes, offsets, counts, supports, bounds = [], [], [], [], [], [], []
    cursor = 0
    for symbol in symbols:
        samples = np.asarray(potentials[symbol].radii, dtype=np.float64)
        table = tables[symbol]
        radii.append(samples)
        values.append(table)
        # np.interp's slopes (y_{j+1} - y_j)/(x_{j+1} - x_j)
        slopes.append(np.append((table[1:] - table[:-1]) / (samples[1:] - samples[:-1]), 0.0))
        offsets.append(cursor)
        counts.append(samples.size)
        cursor += samples.size
        support = float(potentials[symbol].interpolation_cutoff)
        supports.append(support)
        # a square above support^2 (1 + 1e-12) is outside the ball whatever its rounding
        bounds.append(support * support * (1.0 + 1.0e-12))
    centres, owners = image_centres(grid, contributing, potentials)
    kinds = np.array([symbols.index(contributing[owner].symbol) for owner in owners], dtype=np.int32)
    coordinates = np.ascontiguousarray(grid.coordinates, dtype=np.float64)
    with cp.cuda.Device(device_id(cp)):
        field = cp.zeros(points, dtype=cp.float64)
        _density_kernel(cp)(
            ((points + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                np.int64(points),
                cp.asarray(coordinates),
                np.int64(len(centres)),
                cp.asarray(np.ascontiguousarray(centres.reshape(-1))),
                cp.asarray(kinds),
                cp.asarray(np.concatenate(radii)),
                cp.asarray(np.concatenate(values)),
                cp.asarray(np.concatenate(slopes)),
                cp.asarray(np.asarray(offsets, dtype=np.int32)),
                cp.asarray(np.asarray(counts, dtype=np.int32)),
                cp.asarray(np.asarray(supports, dtype=np.float64)),
                cp.asarray(np.asarray(bounds, dtype=np.float64)),
                field,
            ),
        )
        return np.asarray(cp.asnumpy(field), dtype=np.float64)


# ---------------------------------------------------------------------------
# The short-ranged local potential and the Gaussian charges, images on threads.

#: Host bytes of the threaded image terms: the results the workers hold ahead of the calling thread and the
#: temporaries of the workers' own terms.
_PENDING_BYTES = 1 << 30
#: Grid arrays of float64 that one worker holds at its peak while it forms the three terms of an image (distances,
#: the table term and its temporaries, the compensation and the Gaussian with theirs), its three results included.
_WORKER_ARRAYS = 13


def threaded_short_range_local_potential(grid, atoms, potentials, specifications):
    """:func:`parsec_python.V_ion.wire_slab.short_range_local_potential` with the images on worker threads, bit for
    bit, or ``None`` (another grid, ``PARSEC_CONFINED_SETUP=readable``).

    Every image term is a whole-grid array of elementwise functions of the distances (the table, ``erf``, ``exp``):
    the workers (:func:`.periodic_setup_fortran.setup_threads`) evaluate the terms of the images with the readable
    expressions, the distances ``sqrt((dx^2 + dy^2) + dz^2)`` from the coordinate columns (``np.linalg.norm``'s
    operations without its strided reduction), and the calling thread adds them into the potential and the
    Gaussian density in the order of the readable loop (atom, image; the table term, then the compensation).
    The workers are as many as fit their peaks (:data:`_WORKER_ARRAYS` grid arrays each) and the finished terms
    held ahead into :data:`_PENDING_BYTES` (1 GiB), and never fewer than one (the readable loop's memory): eight
    workers up to 1.2 M grid points, five at 1.8 M, two at 3 M (13 arrays is an upper bound: about seven
    were measured).
    """

    if not handles(grid):
        return None
    from scipy.special import erf

    from parsec_python.V_ion.wire_slab import ShortRangeLocalPotential, gaussian_width

    from .periodic_setup_fortran import _ordered_results, setup_threads

    coordinates = np.asarray(grid.coordinates, dtype=np.float64)
    columns = [np.ascontiguousarray(coordinates[:, axis]) for axis in range(3)]
    padding = grid.settings.stencil_half_width
    splines = {}
    items = []
    nominal = 0.0
    for atom in atoms:
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = float(potential.ionic_charge)
        sigma = gaussian_width(potential)
        nominal += charge
        density_factor = charge / sigma**3 / np.sqrt(np.pi) / np.pi
        position = np.asarray(atom.position, dtype=np.float64)
        if specification.use_spline and atom.symbol not in splines:
            splines[atom.symbol] = potential.local_potential_spline(specification.local_angular_momentum, padding)
        for translation in grid.compensation_image_translations(position, sigma):
            items.append((atom.symbol, charge, sigma, density_factor, position + translation))

    def work(item):
        symbol, charge, sigma, density_factor, centre = item
        potential = potentials[symbol]
        specification = specifications[symbol]
        # (dx^2 + dy^2) + dz^2, each square formed in place of its offset
        square = columns[0] - centre[0]
        square *= square
        offset = columns[1] - centre[1]
        offset *= offset
        square += offset
        offset = columns[2] - centre[2]
        offset *= offset
        square += offset
        del offset
        distance = np.sqrt(square, out=square)
        local = potential.local_potential(
            distance,
            specification.local_angular_momentum,
            use_spline=specification.use_spline,
            spline_padding_width=padding,
            spline=splines.get(symbol),
        )
        ratio = distance / sigma
        compensation = np.empty_like(distance)
        nonzero = distance != 0.0
        compensation[nonzero] = 2.0 * charge * erf(ratio[nonzero]) / distance[nonzero]
        compensation[~nonzero] = 2.0 / np.sqrt(np.pi) * 2.0 * charge / sigma
        gaussian = density_factor * np.exp(-ratio * ratio)
        return local, compensation, gaussian

    total = np.zeros(grid.size, dtype=np.float64)
    gaussian = np.zeros(grid.size, dtype=np.float64)
    # within _PENDING_BYTES: every worker's peak (_WORKER_ARRAYS grid arrays) and the two finished images of three
    # arrays that _ordered_results holds beyond the workers; one worker (the readable loop's memory) at the least
    array = 8 * max(int(grid.size), 1)
    threads = max(1, min(setup_threads(), (_PENDING_BYTES - 2 * 3 * array) // (_WORKER_ARRAYS * array)))
    for local, compensation, term in _ordered_results(work, items, threads):
        total += local
        total += compensation
        gaussian += term
    return ShortRangeLocalPotential(
        potential=total,
        gaussian_density=gaussian,
        nominal_charge=nominal,
        grid_charge=float(np.sum(gaussian) * grid.volume_element),
    )


__all__ = [
    "DEFAULT_SETTING",
    "SETTINGS",
    "SWITCH",
    "confined_ball",
    "confined_grid",
    "confined_setup_setting",
    "device_confined_atomic_density",
    "handles",
    "image_centres",
    "threaded_short_range_local_potential",
]
