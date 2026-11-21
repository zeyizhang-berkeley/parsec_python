"""Local-pseudopotential and core-correction forces of a periodic cell, without a scan of the grid.

:func:`parsec_python.Forces.periodic_local.periodic_local_forces` measures the distance of every grid point to every image of
every atom and builds a grid-by-reciprocal-vector phase matrix for each atom, so a 864-atom cell is out of reach.  The same
integrals are formed here the way :func:`..V_ion.periodic_setup.fft_local_ionic_potential` forms the potential they
differentiate:

``F_ion,a = dv [ sum_T sum_(r in ball(R_a + T)) rho(r) s'(d) (r - R_a - T)/d
            + 2 Z_a (4 pi/V) sum_(G != 0) w(G) G Im( exp(-i G.R_a) rho~(G) ) ]``

with ``s(d) = V_a(d) + 2 Z erf(eta d)/d`` (short ranged: a ball around each image of the atom, as wide as the pseudopotential
table or the Gaussian tail), ``w(G) = exp(-G^2/4 eta^2)/G^2`` and ``rho~(G) = sum_r rho(r) exp(iG.r)``, the density's
discrete Fourier transform, one FFT, read at the reciprocal vectors of the series.  As in the potential, ``eta`` is the
largest one whose series ends below the Nyquist frequency of the grid, so ``rho~`` is read at frequencies the FFT holds
exactly.  The series is evaluated for all atoms at once with the separable phases of the structure factor.  The core
correction is the ball sum of the reference.

The sums are written against an array module ``xp`` (NumPy, or CuPy for a calculation on a GPU): the balls are generated
on the host from index ranges, their radial slopes, products and sums are formed with ``xp``, and only the ``(atoms, 3)``
result comes back.  The numbers are those of the reference to the accuracy of the splitting (about 1e-7 Ry/bohr).

The series is streamed (``PARSEC_PERIODIC_FORCE_SERIES``, :func:`periodic_force_series_setting`): the real transform
of the density holds half of the reciprocal box, the term of ``-G`` is that of ``G`` for a real density, and the box
is weighted and contracted slab by slab with the phases of a block of atoms, so no ``(3, M0, M1, M2)`` stack of the
three components is formed (about 1.5 complex fields of the grid at the peak instead of about 6).  A cell whose
density is invariant under a group of operations takes the local forces of one atom per orbit
(:func:`orbit_periodic_local_forces`, :func:`orbit_local_forces`; the k-point wedge and Abelian sectors through
``PARSEC_PERIODIC_LOCAL_ORBITS``).

On a GPU the ball sums of an orthorhombic cell are formed for many atoms per call (``PARSEC_PERIODIC_LOCAL_BALLS``,
:func:`_batched_ball_sums`): the index box around every atom of a batch, the same distances as the per-atom loop
(the coordinates of the grid minus the image of the atom, axis by axis), the same radial slopes, products and masks,
and one reduction per batch; the loop formed the boxes on the host atom by atom (about 8 ms an atom).
"""

from __future__ import annotations

import math
import os
from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf as _host_erf

from parsec_python.Forces import LocalForceTerms, species_force_tables
from parsec_python.Forces.periodic_local import dense_periodic_local_forces as reference_periodic_local_forces
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.Pseudopotential.radial_derivatives import tail_start_radius
from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

from ..V_ion.periodic_setup import _atom_balls, _cell_geometry, fast_splitting

#: Grid points whose radial slopes are evaluated together.
_POINTS_PER_CALL = 1_000_000
#: Complex numbers of one block of the reciprocal contraction.
_CONTRACTION_ELEMENTS = 4_000_000
#: Complex numbers of one slab of the streamed series, and of one product of a slab with the phases of a block of atoms.
_SLAB_ELEMENTS = 4_000_000
#: The most atoms whose phases one contraction of the streamed series takes.
_STREAM_ATOMS = 512
#: Orbits of which one more atom is summed to check the equivariance of the local forces of the orbits.
_SAMPLED_ORBITS = 8

_SERIES_SWITCH = "PARSEC_PERIODIC_FORCE_SERIES"
_SERIES_VALUES = ("stream", "stack", "host")
_SECTOR_LOCAL_SWITCH = "PARSEC_PERIODIC_SECTOR_LOCAL_FORCES"
_SECTOR_LOCAL_VALUES = ("orbits", "all")
_ORBITS_SWITCH = "PARSEC_PERIODIC_LOCAL_ORBITS"
_BALLS_SWITCH = "PARSEC_PERIODIC_LOCAL_BALLS"
_AUTO_OFF = ("auto", "off")
#: Points of the index boxes of one batch of :func:`_batched_ball_sums` (two doubles each on the device with the fused
#: terms of CuPy, about 13 with the array operations); fewer where the device has less than 16 doubles a point free.
_BATCH_POINTS = 1 << 23
_MIN_BATCH_POINTS = 1 << 16


def periodic_force_series_setting() -> str:
    """``PARSEC_PERIODIC_FORCE_SERIES``: how the reciprocal series of the local forces of a cell is summed.

    ``stream`` (default): half of the reciprocal box from the real transform of the density, weighted and contracted
    slab by slab on the device of the sums, or on the host where that device cannot hold the transform;
    ``host``: the same sums on the host always; ``stack``: the three components as one ``(3, M0, M1, M2)`` stack on
    the device, bit for bit as before (to 2026-11-18).  The values agree to round-off (1e-12 relative).
    """

    value = os.environ.get(_SERIES_SWITCH, "stream").strip().lower() or "stream"
    if value not in _SERIES_VALUES:
        raise ValueError(f"{_SERIES_SWITCH} must be one of {', '.join(_SERIES_VALUES)}, not {value!r}")
    return value


def periodic_sector_local_forces_setting() -> str:
    """``PARSEC_PERIODIC_SECTOR_LOCAL_FORCES``: the local forces of a cell solved in the sectors of a point group.

    ``orbits`` (default): the ball sums and the series of one atom per orbit of the full group ``G`` of the sectors,
    turned onto the other atoms of the orbit (:func:`orbit_periodic_local_forces`; the density and the exchange-
    correlation potential of such a run are averaged over ``G``); ``all``: the sums of every atom, as before
    (to 2026-11-18).  The values agree to round-off.
    """

    value = os.environ.get(_SECTOR_LOCAL_SWITCH, "orbits").strip().lower() or "orbits"
    if value not in _SECTOR_LOCAL_VALUES:
        raise ValueError(f"{_SECTOR_LOCAL_SWITCH} must be one of {', '.join(_SECTOR_LOCAL_VALUES)}, not {value!r}")
    return value


def _auto_off(name: str) -> str:
    value = os.environ.get(name, "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        # as the other auto/off switches of the cell routes read them (PARSEC_GAMMA_SPIN_SECTORS)
        value = "off"
    if value not in _AUTO_OFF:
        raise ValueError(f"{name} must be auto or off, not {value!r}")
    return value


def validate_local_force_settings() -> None:
    """Read every switch of the local forces of a cell, so that a value they do not know stops a run before its SCF.

    ``PARSEC_PERIODIC_FORCE_SERIES``, ``PARSEC_PERIODIC_SECTOR_LOCAL_FORCES``, ``PARSEC_PERIODIC_LOCAL_ORBITS``,
    ``PARSEC_PERIODIC_LOCAL_BALLS`` and ``PARSEC_SPLINE_FORCE_CENTER`` (since 2026-11-21) are read where the forces are
    formed, after the whole first SCF; the preparation of a run (``acceleration.driver.prepare_single_point``, the
    command line and the MPI runner) calls this first (since 2026-11-19).  ``PARSEC_PROJECTOR_CENTER`` (2026-11-21) is
    read where the projectors are built, before the SCF, and here too.  Nothing else changes: the values are read
    again where they are used.
    """

    from parsec_python.Forces.periodic import spline_force_center_setting
    from parsec_python.V_ion.ionic_potential import projector_center_setting

    periodic_force_series_setting()
    periodic_sector_local_forces_setting()
    periodic_local_orbits_setting()
    periodic_local_balls_setting()
    spline_force_center_setting()
    projector_center_setting()


def periodic_local_orbits_setting() -> str:
    """``PARSEC_PERIODIC_LOCAL_ORBITS``: the local forces of one atom per orbit on the routes without weighted sectors.

    ``auto`` (default): a k-point run on the irreducible wedge of its mesh, whose density is averaged over the
    operations of the wedge (:func:`parsec_python.Forces.periodic.reduced_kpoint_operations` with the
    ``density_symmetry`` of the system, ``PARSEC_KPOINT_FORCE_WEDGE``), and a Gamma cell in the sectors of an Abelian
    group, whose density is that of the orbits of its group, take the sums of one atom per orbit of that group,
    turned onto the others (:func:`orbit_local_forces`); ``off``: the sums of every atom, as before (to 2026-11-19).
    The values agree to round-off.  The weighted sectors of a point group keep ``PARSEC_PERIODIC_SECTOR_LOCAL_FORCES``.
    """

    return _auto_off(_ORBITS_SWITCH)


def periodic_local_balls_setting() -> str:
    """``PARSEC_PERIODIC_LOCAL_BALLS``: how the ball sums of the local forces of a cell are formed on a GPU.

    ``auto`` (default): for an orthorhombic cell the balls of many atoms per call on the device
    (:func:`_batched_ball_sums`); ``off``: the loop over the atoms, whose balls the host forms one by one, as before
    (to 2026-11-19).  A calculation on the host and an oblique cell keep the loop.  The values agree to round-off
    (the same terms, summed in another order).
    """

    return _auto_off(_BALLS_SWITCH)


class _Slope:
    """The radial slope of one tabulated function with the rules of ``forcloc.f90``, on the arrays of ``xp``.

    Zero closer than the first radial sample; between it and ``rs(npoint-6)`` the derivative of the padded cubic spline of
    the table, or the linear interpolation of its derivative table; from there outwards zero, or the Coulomb slope
    ``2 Z/r**2`` for the local potential (``charge``).  The same rules as
    :func:`parsec_python.Pseudopotential.radial_derivatives.evaluate_local_potential_derivative` and ``evaluate_density_derivative``.
    """

    def __init__(self, xp, potential: ParsecPseudopotential, values, *, spline_padding: int | None, table, charge=None):
        self.xp = xp
        self.first = float(potential.radii[0])
        self.tail = tail_start_radius(potential)
        self.charge = charge
        if spline_padding is not None:
            spline = ParsecRadialSpline.from_positive_grid(potential.radii, values, spline_padding)
            self.knots = xp.asarray(spline.knots)
            self.values = xp.asarray(spline.values)
            self.second = xp.asarray(spline.second_derivatives)
            self.radii = self.table = None
        else:
            self.knots = self.values = self.second = None
            self.radii = xp.asarray(np.asarray(potential.radii, dtype=float))
            self.table = xp.asarray(np.asarray(table, dtype=float))

    def __call__(self, distances):
        xp = self.xp
        result = xp.zeros_like(distances)
        inner = (distances >= self.first) & (distances < self.tail)
        sample = distances[inner]
        if self.knots is not None:
            lower = xp.clip(xp.searchsorted(self.knots, sample, side="right") - 1, 0, int(self.knots.size) - 2)
            upper = lower + 1
            step = self.knots[upper] - self.knots[lower]
            left = (self.knots[upper] - sample) / step
            right = (sample - self.knots[lower]) / step
            result[inner] = (
                (self.values[upper] - self.values[lower]) / step
                - (3.0 * left * left - 1.0) / 6.0 * step * self.second[lower]
                + (3.0 * right * right - 1.0) / 6.0 * step * self.second[upper]
            )
        else:
            result[inner] = xp.interp(sample, self.radii, self.table)
        if self.charge is not None:
            tail = distances >= self.tail
            result[tail] = 2.0 * self.charge / (distances[tail] * distances[tail])
        return result

    def dense(self, distances):
        """:meth:`__call__` without the selections: every expression on every distance and the rules by ``where``,
        so the values of the kept distances are those of :meth:`__call__` (the batched sums, no compaction)."""

        xp = self.xp
        inner = (distances >= self.first) & (distances < self.tail)
        if self.knots is not None:
            lower = xp.clip(xp.searchsorted(self.knots, distances, side="right") - 1, 0, int(self.knots.size) - 2)
            upper = lower + 1
            step = self.knots[upper] - self.knots[lower]
            left = (self.knots[upper] - distances) / step
            right = (distances - self.knots[lower]) / step
            value = (
                (self.values[upper] - self.values[lower]) / step
                - (3.0 * left * left - 1.0) / 6.0 * step * self.second[lower]
                + (3.0 * right * right - 1.0) / 6.0 * step * self.second[upper]
            )
            del lower, upper, step, left, right
        else:
            value = xp.interp(distances, self.radii, self.table)
        result = xp.where(inner, value, 0.0)
        del value, inner
        if self.charge is not None:
            tail = distances >= self.tail
            square = xp.where(tail, distances * distances, 1.0)
            result = xp.where(tail, 2.0 * self.charge / square, result)
        return result


def _erf_over_r_slope(xp, erf, charge: float, radius, eta: float):
    """``d/dr [2 Z erf(eta r)/r]``, smooth through ``r = 0`` (where it vanishes)."""

    far = radius > 1.0e-8
    r = xp.where(far, radius, 1.0)
    value = 2.0 * charge * (2.0 * eta * xp.exp(-((eta * r) ** 2)) / (math.sqrt(math.pi) * r) - erf(eta * r) / (r * r))
    return xp.where(far, value, 0.0)


def _inverse(xp, radius):
    positive = radius > 0.0
    return xp.where(positive, 1.0 / xp.where(positive, radius, 1.0), 0.0)


def _reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice_vectors):
    """``2 Z_a (4 pi/V) sum_G w(G) G Im(exp(-iG.R_a) rho~(G))`` for every atom, ``(atoms, 3)`` (no ``dv``).

    Summed as :func:`periodic_force_series_setting` says: streamed (:func:`_streamed_reciprocal_forces`) on the
    device of ``xp`` or, where that device cannot hold the transform or an allocation fails, on the host; or as the
    stack of the three components (:func:`_stacked_reciprocal_forces`).
    """

    setting = periodic_force_series_setting()
    if setting == "stack":
        return _stacked_reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice_vectors)
    arguments = (grid, positions, charges, density, eta, g_cutoff, lattice_vectors)
    if setting == "host" or xp is np:
        return _streamed_reciprocal_forces(np, *arguments)
    if not _device_holds_series(xp, grid, g_cutoff, lattice_vectors, len(positions)):
        return _streamed_reciprocal_forces(np, *arguments)
    from ..Hartree.gpu_memory import allocation_failure

    try:
        return _streamed_reciprocal_forces(xp, *arguments)
    except Exception as error:  # noqa: BLE001 - only a failed allocation falls back
        if not allocation_failure(error):
            raise
    xp.get_default_memory_pool().free_all_blocks()
    return _streamed_reciprocal_forces(np, *arguments)


def _series_box(grid, g_cutoff, lattice_vectors):
    """``(n, inverse, volume, half, metric)``: the grid counts, the inverse lattice, the cell volume, the largest
    frequency of the series along every axis and the metric ``(2 pi)^2 inverse^T inverse`` of the frequencies."""

    n, lattice, inverse = _cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    half = np.minimum(np.floor(g_cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64), (n - 1) // 2)
    metric = (2.0 * np.pi) ** 2 * (inverse.T @ inverse)
    return n, inverse, volume, half, metric


def _streamed_series_bytes(grid, g_cutoff, lattice_vectors, atoms: int) -> int:
    """The device bytes of :func:`_streamed_reciprocal_forces`: the density, its half transform (twice, for the
    work area of the FFT), the weighted half box and the products of one slab."""

    n, _, _, half, _ = _series_box(grid, g_cutoff, lattice_vectors)
    points = int(np.prod(n))
    transform = int(n[0]) * int(n[1]) * (int(n[2]) // 2 + 1)
    box = int(2 * half[0] + 1) * int(2 * half[1] + 1) * int(half[2] + 1)
    block = min(max(int(atoms), 1), _STREAM_ATOMS)
    return 8 * points + 16 * 2 * transform + 16 * box + 16 * 4 * _SLAB_ELEMENTS + 16 * 6 * block * int(n.max())


def _device_holds_series(xp, grid, g_cutoff, lattice_vectors, atoms: int) -> bool:
    """Whether the device of ``xp`` has the bytes of :func:`_streamed_series_bytes` free (a module that names no
    device always has)."""

    if getattr(xp, "cuda", None) is None:
        return True
    from ..Hartree.gpu_memory import device_free_bytes

    free = device_free_bytes()
    return free is None or free >= _streamed_series_bytes(grid, g_cutoff, lattice_vectors, atoms)


def _host_fft_workers() -> int:
    """The threads of a host transform: ``OMP_NUM_THREADS`` where it is set, else every core."""

    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "")))
    except ValueError:
        return -1


def _half_spectrum(xp, density, shape):
    """``R(m) = sum_r rho(r) exp(-2 pi i sum_j m_j r_j/n_j)`` for ``m_2`` in ``[0, n_2//2]``, ``(n0, n1, n2//2 + 1)``:
    the forward transform of the real density along its last axis halved, so ``rho~(G) = conj(R)``."""

    if xp is np:
        import scipy.fft

        return scipy.fft.rfftn(np.asarray(density, dtype=np.float64).reshape(shape), workers=_host_fft_workers())
    field = xp.asarray(density).reshape(shape)
    rfftn = getattr(xp.fft, "rfftn", None)
    if rfftn is not None:
        return rfftn(field)
    # a module without the real transform: the half of the inverse transform, conjugated
    return (xp.fft.ifftn(field) * float(np.prod(shape)))[:, :, : shape[2] // 2 + 1].conj()


def _streamed_reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice_vectors):
    """The series of :func:`_stacked_reciprocal_forces`, streamed over slabs of the first frequency.

    For a real density ``rho~(-G) = conj(rho~(G))``, so the term of ``-G`` equals that of ``G``: the box keeps
    ``m_2 >= 0``, with twice the weight beyond the plane ``m_2 = 0``, and reads ``rho~`` from the real transform
    (:func:`_half_spectrum`).  With ``G = 2 pi sum_j m_j b_j`` (``b_j`` the columns of the inverse lattice) and
    ``A(m) = c(m_2) w(G) rho~(G)`` the force is ``2 pi sum_j b_j Im S_j`` with
    ``S_j = sum_m m_j A(m) exp(-2 pi i m.f_a)``: the separable phases of the fractional coordinates ``f_a`` of a
    block of atoms are contracted with each slab of ``A`` by two products along ``m_2`` (one with ``m_2`` in its
    phases), then along ``m_1`` and ``m_0``.  Only the weighted half box, one slab of products and the phases of
    one block are held besides the transform.
    """

    n, inverse, volume, half, metric = _series_box(grid, g_cutoff, lattice_vectors)
    count = len(positions)
    if count == 0:
        return np.zeros((0, 3))
    shape = tuple(int(v) for v in n)
    first = np.arange(-half[0], half[0] + 1, dtype=np.int64)
    second = np.arange(-half[1], half[1] + 1, dtype=np.int64)
    third = np.arange(0, half[2] + 1, dtype=np.int64)
    sizes = (first.size, second.size, third.size)
    shifts = np.asarray(grid.shift, dtype=float)
    shifted = bool(np.any(shifts != 0.0))

    spectrum = _half_spectrum(xp, density, shape)
    second_rows = xp.asarray(np.mod(second, n[1]))[None, :, None]
    third_rows = xp.asarray(third)[None, None, :]
    rows_per_slab = max(1, min(sizes[0], _SLAB_ELEMENTS // (sizes[1] * sizes[2])))
    slabs = []
    m1, m2 = second[None, :, None], third[None, None, :]
    for start in range(0, sizes[0], rows_per_slab):
        m0 = first[start : start + rows_per_slab][:, None, None]
        g_squared = (
            metric[0, 0] * m0**2 + metric[1, 1] * m1**2 + metric[2, 2] * m2**2
            + 2.0 * (metric[0, 1] * m0 * m1 + metric[0, 2] * m0 * m2 + metric[1, 2] * m1 * m2)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
        weight[g_squared == 0.0] = 0.0
        weight[g_squared > g_cutoff**2] = 0.0
        weight[:, :, 1:] *= 2.0  # the terms of -G
        if shifted:
            weight = weight * np.exp(2.0j * np.pi * (m0 * shifts[0] / n[0] + m1 * shifts[1] / n[1] + m2 * shifts[2] / n[2]))
        first_rows = xp.asarray(np.mod(m0, n[0]))
        slabs.append(spectrum[first_rows, second_rows, third_rows].conj() * xp.asarray(weight))
    del spectrum

    fractional = np.asarray(positions, dtype=np.float64) @ inverse
    rows = slabs[0].shape[0]
    block = max(1, min(count, _STREAM_ATOMS, _SLAB_ELEMENTS // (rows * sizes[1])))
    sums = np.zeros((count, 3), dtype=np.complex128)
    for start in range(0, count, block):
        stop = min(start + block, count)
        width = stop - start
        phases = [
            np.exp(-2.0j * np.pi * np.outer(frequencies, fractional[start:stop, axis]))
            for axis, frequencies in enumerate((first, second, third))
        ]
        along_third = xp.asarray(phases[2])
        along_third_weighted = xp.asarray(third[:, None] * phases[2])
        along_second = xp.asarray(phases[1])
        along_second_weighted = xp.asarray(second[:, None] * phases[1])
        totals = [None, None, None]
        offset = 0
        for slab in slabs:
            height = slab.shape[0]
            along_first = xp.asarray(phases[0][offset : offset + height])
            along_first_weighted = xp.asarray(first[offset : offset + height, None] * phases[0][offset : offset + height])
            flat = slab.reshape(height * sizes[1], sizes[2])
            plain = xp.matmul(flat, along_third).reshape(height, sizes[1], width)
            weighted = xp.matmul(flat, along_third_weighted).reshape(height, sizes[1], width)
            parts = (
                xp.einsum("ia,ia->a", xp.einsum("ija,ja->ia", plain, along_second), along_first_weighted),
                xp.einsum("ia,ia->a", xp.einsum("ija,ja->ia", plain, along_second_weighted), along_first),
                xp.einsum("ia,ia->a", xp.einsum("ija,ja->ia", weighted, along_second), along_first),
            )
            totals = [part if total is None else total + part for total, part in zip(totals, parts)]
            offset += height
        sums[start:stop] = _to_host(xp, xp.stack(totals)).T
    forces = 2.0 * np.pi * (sums.imag @ inverse.T)
    return forces * (2.0 * np.asarray(charges, dtype=np.float64) * 4.0 * np.pi / volume)[:, None]


def _stacked_reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice_vectors):
    """The series of :func:`_reciprocal_forces` with the three components as one ``(3, M0, M1, M2)`` stack on the
    device of ``xp`` (``PARSEC_PERIODIC_FORCE_SERIES=stack``, the sums to 2026-11-18)."""

    n, lattice, inverse = _cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    half = np.minimum(np.floor(g_cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64), (n - 1) // 2)
    frequencies = [np.arange(-m, m + 1, dtype=np.int64) for m in half]
    m0, m1, m2 = (frequencies[0][:, None, None], frequencies[1][None, :, None], frequencies[2][None, None, :])
    metric = (2.0 * np.pi) ** 2 * (inverse.T @ inverse)
    g_squared = (
        metric[0, 0] * m0**2 + metric[1, 1] * m1**2 + metric[2, 2] * m2**2
        + 2.0 * (metric[0, 1] * m0 * m1 + metric[0, 2] * m0 * m2 + metric[1, 2] * m1 * m2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    weight[g_squared == 0.0] = 0.0
    weight[g_squared > g_cutoff**2] = 0.0
    # the components of G = sum_j m_j b_j, b_j = 2 pi inverse[:, j]
    components = [
        2.0 * np.pi * (inverse[c, 0] * m0 + inverse[c, 1] * m1 + inverse[c, 2] * m2) * weight for c in range(3)
    ]

    # rho~(G) = sum_r rho(r) exp(iG.r), at the points (i + s)/n of the lattice vectors: one FFT
    shape = tuple(int(v) for v in n)
    spectrum = xp.fft.ifftn(xp.asarray(density).reshape(shape)) * int(np.prod(shape))
    index = [np.mod(frequencies[axis], n[axis]) for axis in range(3)]
    transform = spectrum[xp.asarray(index[0])[:, None, None], xp.asarray(index[1])[None, :, None], xp.asarray(index[2])[None, None, :]]
    shifts = np.asarray(grid.shift, dtype=float)
    if np.any(shifts != 0.0):
        transform = transform * xp.asarray(
            np.exp(2.0j * np.pi * (m0 * shifts[0] / n[0] + m1 * shifts[1] / n[1] + m2 * shifts[2] / n[2]))
        )
    sizes = tuple(int(2 * m + 1) for m in half)
    series = xp.stack([transform * xp.asarray(component) for component in components])  # (3, M0, M1, M2)
    series = series.reshape(3, sizes[0] * sizes[1], sizes[2])

    fractional = positions @ inverse
    forces = np.zeros((len(positions), 3))
    block = max(1, _CONTRACTION_ELEMENTS // (3 * sizes[0] * sizes[1]))
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        phases = [
            xp.asarray(np.exp(-2.0j * np.pi * np.outer(frequencies[axis], fractional[start:stop, axis]))) for axis in range(3)
        ]
        along_z = xp.matmul(series, phases[2]).reshape(3, sizes[0], sizes[1], stop - start)
        along_y = xp.einsum("cija,ja->cia", along_z, phases[1])
        total = xp.einsum("cia,ia->ca", along_y, phases[0])
        forces[start:stop] = _to_host(xp, total.imag.T)
    return forces * (2.0 * charges * 4.0 * np.pi / volume)[:, None]


def _to_host(xp, array) -> np.ndarray:
    return np.asarray(xp.asnumpy(array) if hasattr(xp, "asnumpy") else array)


def _uses_batched_balls(xp) -> bool:
    """Whether the ball sums run batched (:func:`_batched_ball_sums`): on a device module (CuPy) under
    ``PARSEC_PERIODIC_LOCAL_BALLS=auto``."""

    return periodic_local_balls_setting() == "auto" and getattr(xp, "cuda", None) is not None


def _orthorhombic_geometry(grid):
    """``(axes, sides, n, shift)`` of a grid whose point ``(i, j, k)`` lies at ``(x_i, y_j, z_k)``: the coordinates
    along the three axes (taken from the grid's own), the side lengths, the points per axis and the shift; ``None``
    for a cell whose lattice vectors are not the axes or a grid whose coordinates are not those of its axes."""

    lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    if lattice.shape != (3, 3) or np.any(lattice != np.diag(np.diag(lattice))) or np.any(np.diag(lattice) <= 0.0):
        return None
    n = np.asarray(grid.points_per_axis, dtype=np.int64)
    coordinates = getattr(grid, "coordinates", None)
    total = int(np.prod(n))
    if coordinates is None or tuple(np.shape(coordinates)) != (total, 3):
        return None
    strides = (int(n[1] * n[2]), int(n[2]), 1)
    axes = tuple(
        np.ascontiguousarray(coordinates[np.arange(int(n[m]), dtype=np.int64) * strides[m], m], dtype=np.float64)
        for m in range(3)
    )
    rows = np.unique(np.linspace(0, total - 1, 33).astype(np.int64))
    indices = (rows // strides[0], (rows // strides[1]) % n[1], rows % n[2])
    if not all(np.array_equal(coordinates[rows, m], axes[m][indices[m]]) for m in range(3)):
        return None
    return axes, np.diag(lattice).copy(), n, np.asarray(grid.shift, dtype=np.float64)


def _batch_points(xp) -> int:
    """The points of the index boxes of one batch: :data:`_BATCH_POINTS`, fewer on a device with little free memory."""

    if getattr(xp, "cuda", None) is None:
        return _BATCH_POINTS
    from ..Hartree.gpu_memory import device_free_bytes

    free = device_free_bytes()
    if free is None:
        return _BATCH_POINTS
    return int(min(_BATCH_POINTS, max(_MIN_BATCH_POINTS, int(free) // (16 * 8))))


_FUSED_PREAMBLE = r"""
__device__ __forceinline__ long long first_above(const double* a, const long long n, const double x)
{
    // numpy.searchsorted(a, x, side="right"): the first index whose value is above x
    long long low = 0, high = n;
    while (low < high) {
        const long long middle = (low + high) >> 1;
        if (a[middle] > x) high = middle; else low = middle + 1;
    }
    return low;
}
"""

_FUSED_OPERATION = r"""
const double d = sqrt((xx + yy) + zz);
const bool keep = strict ? (d < radius) : (d <= radius);
double c = 0.0;
if (keep) {
    double s = 0.0;
    if (d >= first && d < tail) {
        if (spline) {
            long long lower = first_above(&knots[0], count, d) - 1;
            if (lower < 0) lower = 0;
            if (lower > count - 2) lower = count - 2;
            const long long upper = lower + 1;
            const double step = knots[upper] - knots[lower];
            const double left = (knots[upper] - d) / step;
            const double right = (d - knots[lower]) / step;
            s = (values[upper] - values[lower]) / step
                - (3.0 * left * left - 1.0) / 6.0 * step * second[lower]
                + (3.0 * right * right - 1.0) / 6.0 * step * second[upper];
        } else {
            long long j = first_above(&radii[0], samples, d) - 1;
            if (j < 0) j = 0;
            if (j > samples - 2) j = samples - 2;
            const double slope = (table[j + 1] - table[j]) / (radii[j + 1] - radii[j]);
            s = slope * (d - radii[j]) + table[j];
        }
    } else if (coulomb && d >= tail) {
        s = 2.0 * charge / (d * d);
    }
    if (gaussian && d > 1.0e-8) {
        const double a = eta * d;
        s += 2.0 * charge * (2.0 * eta * exp(-(a * a)) / (1.7724538509055159 * d) - erf(a) / (d * d));
    }
    c = s * field[rx + ry + rz] * (d > 0.0 ? 1.0 / d : 0.0);
}
out = c;
"""

_FUSED_KERNELS: dict = {}


def _fused_kernel(xp):
    """The elementwise kernel of the terms of :func:`_batched_ball_sums` on CuPy (``None`` for another module): the
    distance, the radial slope of :class:`_Slope` (the spline of its knots, or the linear interpolation of its table,
    and the Coulomb tail), the slope of ``2 Z erf(eta r)/r``, the field at the row and ``1/d``, in one pass."""

    factory = getattr(xp, "ElementwiseKernel", None)
    if factory is None or getattr(xp, "cuda", None) is None:
        return None
    key = int(xp.cuda.Device().id)
    if key not in _FUSED_KERNELS:
        _FUSED_KERNELS[key] = factory(
            "float64 xx, float64 yy, float64 zz, int64 rx, int64 ry, int64 rz, raw float64 field, "
            "raw float64 knots, raw float64 values, raw float64 second, int64 count, raw float64 radii, "
            "raw float64 table, int64 samples, float64 first, float64 tail, int32 spline, int32 coulomb, "
            "int32 gaussian, float64 charge, float64 eta, float64 radius, int32 strict",
            "float64 out",
            _FUSED_OPERATION,
            "parsec_periodic_local_ball_terms",
            preamble=_FUSED_PREAMBLE,
            options=("--fmad=false",),
        )
    return _FUSED_KERNELS[key]


class _FusedSlope:
    """The arguments of :func:`_fused_kernel` for the slope of a :class:`_Slope` (with ``2 Z erf(eta r)/r`` where
    ``eta`` is given): the device arrays of its spline or table, its radii and its Coulomb tail."""

    def __init__(self, xp, slope: _Slope, *, eta: float | None = None, charge: float = 0.0):
        empty = xp.zeros(2, dtype=xp.float64)
        spline = slope.knots is not None
        self.arrays = (
            slope.knots if spline else empty,
            slope.values if spline else empty,
            slope.second if spline else empty,
            np.int64(slope.knots.size if spline else 0),
            empty if spline else slope.radii,
            empty if spline else slope.table,
            np.int64(0 if spline else slope.radii.size),
        )
        self.scalars = (
            np.float64(slope.first),
            np.float64(slope.tail),
            np.int32(1 if spline else 0),
            np.int32(0 if slope.charge is None else 1),
            np.int32(0 if eta is None else 1),
            np.float64(charge if slope.charge is None else slope.charge),
            np.float64(0.0 if eta is None else eta),
        )


def _batched_ball_sums(xp, geometry, positions, radius: float, field, slope, *, strict: bool, points: int | None = None):
    """``sum_T sum_(r: |r - p - T| <= radius) slope(d) field(r) (r - p - T)/d`` with ``d = |r - p - T|`` for every
    position ``p`` (``< radius`` where ``strict``), ``(atoms, 3)`` on the host, without ``dv``.

    The sums of the loop of :func:`fast_periodic_local_forces` (the image balls of :func:`..V_ion.periodic_setup.
    _atom_balls` for the local potential, those of the core density's images): the grid point ``u`` of an index box
    around ``p`` that may lie outside the cell stands for the point ``i = u mod n`` of the cell and the image
    ``T = -((u - i)/n) L`` of the atom, so ``r - p - T`` is formed as the loop forms it, ``x_i - (p + T)`` axis by axis,
    and its length as ``sqrt((x^2 + y^2) + z^2)``; ``slope`` gets the same distances and the points beyond the radius
    are masked.  A batch takes the boxes of several atoms (``points`` grid points at most, the boxes cut along their
    first axis where one is larger), the field gathered at their rows, and is reduced as
    ``sum_i x_i sum_(j,k) c_ijk`` (and alike for ``y``, ``z``).  The terms are those of the loop, summed in another order.
    ``slope`` is a function of the distances (any array module), or a :class:`_FusedSlope`, whose terms one
    elementwise kernel of CuPy forms (:func:`_fused_kernel`, the same expressions without fused multiply-adds).
    """

    fused = _fused_kernel(xp) if isinstance(slope, _FusedSlope) else None
    if isinstance(slope, _FusedSlope) and fused is None:
        raise ValueError("the fused terms of the ball sums need CuPy")
    axes, sides, n, shift = geometry
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    count = positions.shape[0]
    sums = np.zeros((count, 3))
    if count == 0:
        return sums
    half = float(radius) * n.astype(np.float64) / sides  # half widths of the balls in grid steps
    widths = (np.ceil(2.0 * half).astype(np.int64) + 4)
    starts = np.floor(positions / sides * n - shift - half).astype(np.int64) - 1
    budget = int(points) if points is not None else _batch_points(xp)
    plane = int(widths[1] * widths[2])
    rows_per_piece = max(1, min(int(widths[0]), budget // plane))
    atoms_per_batch = max(1, budget // (rows_per_piece * plane)) if rows_per_piece == int(widths[0]) else 1
    offsets = [np.arange(int(width), dtype=np.int64) for width in widths]
    for first in range(0, count, atoms_per_batch):
        last = min(first + atoms_per_batch, count)
        block = positions[first:last]
        relative, wrapped = [], []
        for m in range(3):
            unwrapped = starts[first:last, m, None] + offsets[m][None, :]
            inside = np.mod(unwrapped, n[m])
            image = -((unwrapped - inside) // n[m])
            relative.append(axes[m][inside] - (block[:, m, None] + image * sides[m]))
            wrapped.append(inside)
        y, z = xp.asarray(relative[1]), xp.asarray(relative[2])
        y_rows, z_rows = xp.asarray(wrapped[1] * n[2]), xp.asarray(wrapped[2])
        y_square, z_square = (y * y)[:, None, :, None], (z * z)[:, None, None, :]
        totals = xp.zeros((last - first, 3), dtype=xp.float64)
        for low in range(0, int(widths[0]), rows_per_piece):
            high = min(low + rows_per_piece, int(widths[0]))
            x = xp.asarray(relative[0][:, low:high])
            x_rows = xp.asarray(wrapped[0][:, low:high] * (n[1] * n[2]))[:, :, None, None]
            if fused is not None:
                coefficient = fused(
                    (x * x)[:, :, None, None], y_square, z_square, x_rows, y_rows[:, None, :, None],
                    z_rows[:, None, None, :], field, *slope.arrays, *slope.scalars, np.float64(radius),
                    np.int32(1 if strict else 0),
                )
            else:
                distances = xp.sqrt(((x * x)[:, :, None, None] + y_square) + z_square)
                keep = distances < radius if strict else distances <= radius
                rows = (x_rows + y_rows[:, None, :, None]) + z_rows[:, None, None, :]
                values = field[rows]
                del rows
                coefficient = slope(distances) * values * _inverse(xp, distances)
                del values
                coefficient = xp.where(keep, coefficient, 0.0)
                del keep, distances
            totals += xp.stack(
                (
                    xp.sum(xp.sum(coefficient, axis=(2, 3)) * x, axis=1),
                    xp.sum(xp.sum(coefficient, axis=(1, 3)) * y, axis=1),
                    xp.sum(xp.sum(coefficient, axis=(1, 2)) * z, axis=1),
                ),
                axis=1,
            )
            del coefficient, x
        sums[first:last] = _to_host(xp, totals)
    return sums


def _batched_local_sums(xp, erf, grid, geometry, atoms, potentials, specifications, positions, density_x, xc_x, eta, erfc_cutoff):
    """``(ionic, core_correction)`` ball sums of every atom by :func:`_batched_ball_sums`, species by species, with the
    slopes and reaches of the loop of :func:`fast_periodic_local_forces` (without ``dv``)."""

    count = len(atoms)
    ionic = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    half_width = grid.settings.stencil_half_width
    symbols = np.asarray([atom.symbol for atom in atoms])
    for symbol in dict.fromkeys(symbols.tolist()):
        members = np.flatnonzero(symbols == symbol)
        potential = potentials[symbol]
        specification = specifications[symbol]
        charge = potential.ionic_charge
        tables = species_force_tables(potential, specification)
        local = _Slope(
            xp, potential, potential.channel_potentials[specification.local_angular_momentum],
            spline_padding=half_width, table=None, charge=charge,
        )
        reach = max(erfc_cutoff, _ionic.coulomb_onset_radius(potential, specification.local_angular_momentum))

        def short_range(distances, _slope=local, _charge=charge):
            return _slope.dense(distances) + _erf_over_r_slope(xp, erf, _charge, distances, eta)

        fused = _fused_kernel(xp) is not None
        terms = _FusedSlope(xp, local, eta=eta) if fused else short_range
        with np.errstate(divide="ignore", invalid="ignore"):
            ionic[members] = _batched_ball_sums(xp, geometry, positions[members], reach, density_x, terms, strict=False)
        if tables.core_density_derivative is not None:
            core = _Slope(
                xp, potential, potential.core_density,
                spline_padding=half_width if specification.use_spline else None,
                table=tables.core_density_derivative,
            )
            support = _ionic.atomic_density_support_radius(potential, specification, core=True)
            terms = _FusedSlope(xp, core) if fused else core.dense
            with np.errstate(divide="ignore", invalid="ignore"):
                core_correction[members] = _batched_ball_sums(
                    xp, geometry, positions[members], support, xc_x, terms, strict=True
                )
    return ionic, core_correction


def fast_periodic_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
    xc_potential: np.ndarray,
    *,
    xp=np,
    erf=None,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms:
    """The result of :func:`parsec_python.Forces.periodic_local.periodic_local_forces`, ``(atoms, 3)`` in Ry/bohr.

    ``atoms`` lie inside the cell.  ``xp`` is the array module that forms the sums (CuPy on a GPU, with
    ``erf`` from ``cupyx.scipy.special``); the periodic local potential must use the cubic spline of its table
    (:data:`parsec_python.V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`), otherwise the readable routine runs.
    """

    if not _ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE:
        return reference_periodic_local_forces(
            grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, tolerance=tolerance
        )
    erf = _host_erf if erf is None and xp is np else erf
    if erf is None:
        raise ValueError("an array module other than NumPy needs its erf")
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    xc = np.asarray(xc_potential, dtype=np.float64)
    for name, field in (("density", density), ("xc_potential", xc)):
        if field.shape != (grid.size,):
            raise ValueError(f"{name} must hold one value per grid point")
    count = len(atoms)
    ionic = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    if count == 0:
        return LocalForceTerms(ionic, np.zeros((0, 3)), core_correction)

    eta, erfc_cutoff, g_cutoff = fast_splitting(grid, lattice, tolerance)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    half_width = grid.settings.stencil_half_width

    # the series of the long-range part: all atoms from one FFT
    ionic += _reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice)

    density_x, xc_x = xp.asarray(density), xp.asarray(xc)
    # PARSEC_PERIODIC_LOCAL_BALLS=auto: the balls of many atoms per call on the device (an orthorhombic cell)
    geometry = _orthorhombic_geometry(grid) if _uses_batched_balls(xp) else None
    if geometry is not None:
        balls, cores = _batched_local_sums(
            xp, erf, grid, geometry, atoms, potentials, specifications, positions, density_x, xc_x, eta, erfc_cutoff
        )
        ionic += balls
        core_correction += cores
        volume_element = grid.volume_element
        return LocalForceTerms(
            ionic=ionic * volume_element,
            self_consistency=np.zeros((count, 3)),
            core_correction=core_correction * volume_element,
        )
    tables: dict = {}
    slopes: dict = {}

    def sums(sample_rows, sample_relative, sample_distances, field, slope) -> np.ndarray:
        rows = xp.asarray(np.concatenate(sample_rows))
        relative = xp.asarray(np.concatenate(sample_relative))
        distances = xp.asarray(np.concatenate(sample_distances))
        coefficient = slope(distances) * field[rows] * _inverse(xp, distances)
        return _to_host(xp, xp.sum(coefficient[:, None] * relative, axis=0))

    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = potential.ionic_charge
        position = positions[index]
        if atom.symbol not in tables:
            tables[atom.symbol] = species_force_tables(potential, specification)
            local = _Slope(
                xp, potential, potential.channel_potentials[specification.local_angular_momentum],
                spline_padding=half_width, table=None, charge=charge,
            )
            core = None
            if tables[atom.symbol].core_density_derivative is not None:
                core = _Slope(
                    xp, potential, potential.core_density,
                    spline_padding=half_width if specification.use_spline else None,
                    table=tables[atom.symbol].core_density_derivative,
                )
            slopes[atom.symbol] = (local, core)
        local_slope, core_slope = slopes[atom.symbol]

        # short-ranged part: V_a + 2 Z erf(eta r)/r over a ball around every image; beyond the Coulomb onset of the table
        # it is -2 Z erfc(eta r)/r, so the ball reaches the larger of the onset and the Gaussian tail
        reach = max(erfc_cutoff, _ionic.coulomb_onset_radius(potential, specification.local_angular_momentum))

        def short_range(distances, _slope=local_slope, _charge=charge):
            return _slope(distances) + _erf_over_r_slope(xp, erf, _charge, distances, eta)

        pending: tuple[list, list, list] = ([], [], [])
        points = 0
        for rows, distances, relative in _atom_balls(grid, position, reach, vectors=True):
            pending[0].append(rows), pending[1].append(relative), pending[2].append(distances)
            points += rows.size
            if points >= _POINTS_PER_CALL:
                ionic[index] += sums(*pending, density_x, short_range)
                pending, points = ([], [], []), 0
        if points:
            ionic[index] += sums(*pending, density_x, short_range)

        if core_slope is not None:
            support = _ionic.atomic_density_support_radius(potential, specification, core=True)
            pending, points = ([], [], []), 0
            for translation in _ionic._periodic_image_translations(lattice, support):
                rows, relative, radius = _ionic._support_finder(grid, position + translation, support)
                inside = radius < support
                if not np.any(inside):
                    continue
                pending[0].append(rows[inside]), pending[1].append(relative[inside]), pending[2].append(radius[inside])
                points += int(np.count_nonzero(inside))
                if points >= _POINTS_PER_CALL:
                    core_correction[index] += sums(*pending, xc_x, core_slope)
                    pending, points = ([], [], []), 0
            if points:
                core_correction[index] += sums(*pending, xc_x, core_slope)

    volume_element = grid.volume_element
    return LocalForceTerms(
        ionic=ionic * volume_element,
        self_consistency=np.zeros((count, 3)),
        core_correction=core_correction * volume_element,
    )


def atom_orbits(operations, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(images, representatives, sizes)`` of the orbits of ``count`` atoms under a group of atom operations.

    ``images[g, a]`` is the atom that operation ``g`` carries atom ``a`` onto; the orbit of an atom is the column of
    its images (the operations form a group), its representative the smallest index in it, ``representatives`` the
    atoms that are their own representative (ascending) and ``sizes`` the number of atoms in each of their orbits.
    """

    images = np.asarray([np.asarray(operation.image, dtype=np.int64) for operation in operations], dtype=np.int64)
    if images.ndim != 2 or images.shape[1] != count:
        raise ValueError(f"the operations act on {images.shape[-1] if images.ndim == 2 else 0} atoms, not {count}")
    smallest = images.min(axis=0)
    representatives = np.flatnonzero(smallest == np.arange(count, dtype=np.int64))
    sizes = np.bincount(smallest, minlength=count)[representatives]
    return images, representatives, sizes


def turn_onto_orbits(values: np.ndarray, operations, images: np.ndarray, representatives: np.ndarray, sizes, count: int):
    """``(count, 3)``: the average over the operations of the vectors ``values`` of the representatives, each
    counted ``sizes`` times (the other atoms zero).

    ``S_G(x)_b = |G|^-1 sum_g R_g x_(g^-1 b)`` (:func:`parsec_python.Forces.symmetry.symmetrize_forces`) of that
    ``x``: an atom ``b = g a`` of the orbit of representative ``a`` gets ``R_g F_a`` averaged over the
    ``|G|/|orbit|`` operations that carry ``a`` onto ``b``, so an equivariant term (``F_(g a) = R_g F_a``) is
    turned onto every atom and the representative keeps its own value averaged over its site group.  Only the rows of
    the representatives are moved.
    """

    total = np.zeros((count, 3), dtype=np.float64)
    weighted = np.asarray(values, dtype=np.float64) * np.asarray(sizes, dtype=np.float64)[:, None]
    for index, operation in enumerate(operations):
        total[images[index, representatives]] += weighted @ np.asarray(operation.matrix, dtype=np.float64).T
    return total / float(len(operations))


def orbit_periodic_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
    xc_potential: np.ndarray,
    operations,
    *,
    xp=np,
    erf=None,
    tolerance: float = 1.0e-14,
) -> tuple[LocalForceTerms, dict]:
    """:func:`fast_periodic_local_forces` of a cell whose density and exchange-correlation potential are invariant
    under the group ``operations`` (atom operations with ``image`` and ``matrix``, the identity among them).

    The local ionic and core-correction forces are equivariant then (``F_(g a) = R_g F_a``): they are summed for one
    atom per orbit (:func:`atom_orbits`) and turned onto the others (:func:`turn_onto_orbits`); a diamond cell of
    64,000 atoms with the 192 operations of its sectors sums 446 atoms.  The sums of a few other atoms (the last atom of
    up to :data:`_SAMPLED_ORBITS` orbits, in the same call: no other transform of the density) check the
    equivariance.  ``details``: ``representatives``, the number of orbits; ``sampled``, the number of atoms checked;
    ``site_asymmetry`` (Ry/bohr), the largest change of the force of a representative by the average over its site
    group, and of the force of a checked atom by its turned value: round-off where the density is invariant.
    """

    def summed_terms(chosen):
        return fast_periodic_local_forces(
            grid, chosen, potentials, specifications, lattice_vectors, density, xc_potential,
            xp=xp, erf=erf, tolerance=tolerance,
        )

    return orbit_local_forces(summed_terms, atoms, operations)


def orbit_local_forces(
    local_terms, atoms: Sequence[Atom], operations, *, self_consistency: bool = False
) -> tuple[LocalForceTerms, dict]:
    """The local forces of :func:`orbit_periodic_local_forces` from any routine of them: ``local_terms(chosen)``
    returns the :class:`LocalForceTerms` of the atoms ``chosen`` (whose sums need no other atom), here those of one atom
    per orbit of ``operations`` and of up to :data:`_SAMPLED_ORBITS` checked atoms, in one call.  Returns the terms of
    every atom and ``details`` as :func:`orbit_periodic_local_forces` does; the density and exchange-correlation
    potential that ``local_terms`` reads must be invariant under ``operations``.  ``self_consistency`` turns the
    self-consistency term of the Fortran-derived routines (``(V_out - V_in)`` times the slope of the atomic valence
    density, ``forcloc.f90``) onto the orbits as well, whose residual potential must then be invariant too; without
    it that term is zero, as the default routines have none.
    """

    count = len(atoms)
    images, representatives, sizes = atom_orbits(operations, count)
    larger = representatives[sizes > 1]
    chosen = larger[np.unique(np.linspace(0, larger.size - 1, min(larger.size, _SAMPLED_ORBITS)).astype(np.int64))]
    sampled = images[:, chosen].max(axis=0) if chosen.size else np.zeros(0, dtype=np.int64)
    summed = np.concatenate([representatives, sampled])
    terms = local_terms([atoms[index] for index in summed])
    own = slice(0, representatives.size)
    ionic = turn_onto_orbits(terms.ionic[own], operations, images, representatives, sizes, count)
    core_correction = turn_onto_orbits(terms.core_correction[own], operations, images, representatives, sizes, count)
    residual = (
        turn_onto_orbits(terms.self_consistency[own], operations, images, representatives, sizes, count)
        if self_consistency
        else np.zeros((count, 3))
    )
    site = 0.0
    if summed.size:
        site = float(
            max(
                np.max(np.abs(ionic[summed] - terms.ionic)),
                np.max(np.abs(core_correction[summed] - terms.core_correction)),
                np.max(np.abs(residual[summed] - terms.self_consistency)) if self_consistency else 0.0,
            )
        )
    details = {"representatives": int(representatives.size), "sampled": int(sampled.size), "site_asymmetry": site}
    return LocalForceTerms(ionic=ionic, self_consistency=residual, core_correction=core_correction), details


def _device_module():
    """``(xp, erf, device)`` for the forces of a calculation run through the reference drivers: CuPy on the first device of
    the k-point plan when that plan uses a GPU, else NumPy."""

    from ..SCF.kpoint_parallel import resolve_backend

    plan = resolve_backend()
    if plan is None or plan.name != "cupy":
        return np, None, None
    from cupyx.scipy.special import erf

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    return cp, erf, int(plan.device_ids[0])


def local_force_provider(
    grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, *, tolerance: float = 1.0e-14
) -> LocalForceTerms:
    """What :func:`parsec_python.Forces.periodic_local.periodic_local_forces` hands its sums to once the acceleration
    package is imported: :func:`fast_periodic_local_forces` on the device the k-point solver uses, or on the host."""

    xp, erf, device = _device_module()
    arguments = (grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential)
    if device is None:
        return fast_periodic_local_forces(*arguments, tolerance=tolerance)
    with xp.cuda.Device(device):
        return fast_periodic_local_forces(*arguments, xp=xp, erf=erf, tolerance=tolerance)


__all__ = [
    "atom_orbits",
    "fast_periodic_local_forces",
    "local_force_provider",
    "orbit_local_forces",
    "orbit_periodic_local_forces",
    "periodic_force_series_setting",
    "periodic_local_balls_setting",
    "periodic_local_orbits_setting",
    "periodic_sector_local_forces_setting",
    "turn_onto_orbits",
    "validate_local_force_settings",
]
