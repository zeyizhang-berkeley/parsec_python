"""Exact exchange of a periodic cell's sector orbitals in the classes of its half-cell translations.

``PARSEC_CUPY_EXCHANGE_TRANSLATIONS`` (``auto`` by default, ``off``).  The conventions are those of
:mod:`.exchange_device` (orbitals ``q = sqrt(dV) psi``, ``W = V_x Psi``, ``E_x = (w/2) sum_i f_i <q_i|W_i>``) and of
the periodic pair solver of the Gamma point (:class:`.bloch_exchange_device.DevicePeriodicPairSolvers`:
``v = ifftn(K(G) fftn(rho))`` on the box of the cell, numpy's conventions).

What it uses.  The real sectors of an unpolarized Gamma-point hybrid of a bulk orthorhombic cell
(``PARSEC_GAMMA_HYBRID_SECTORS=auto``) are the one-dimensional representations of an abelian group ``G`` that holds the
translations by half the cell that map the atoms onto themselves (``Symmetry.axis_reflection._half_cell_translations``):
a subgroup ``T`` of the eight half translations ``a`` (bit ``2 - d`` of ``a`` set: half the cell along axis ``d``, the
indexing of :data:`parsec_python.Hartree.kronecker.SIGN_OPERATIONS`).  An orbital of sector ``A`` satisfies
``q(r + a) = chi_A(a) q(r)`` on ``T``, so the pair density of sectors ``A`` and ``B`` has the character
``chi_A chi_B`` there.  Written as a Fourier series ``sum_m c_m exp(2 pi i m.x/n)`` on the box (``x`` the integer
coordinates of the grid), such a density has only the frequencies whose parities ``m_d mod 2`` along the halved axes
``H`` (the axes ``T`` moves) form a class ``t`` with ``(-1)^(t.a) = chi_A(a) chi_B(a)`` for every ``a`` of ``T``:
``2^|H| / |T|`` of the ``2^|H|`` parity classes, a sublattice of ``1/|T|`` of the ``G`` vectors.

The class solve.  A function of parity class ``t`` (``f(r + a) = (-1)^(t.a) f(r)`` for every half translation ``a``
along ``H``) is fixed by its values on the reduced box ``E`` (``n_d/2`` points along a halved axis, ``n_d`` along the
others), and ``f(x) exp(-2 pi i t.x/n)`` is periodic on ``E``: its FFT on ``E`` holds the coefficients ``c_(2k+t)``
(times ``|E|``).  So the potential of a class density is ``exp(2 pi i t.x/n) ifftn_E(K(2k + t) fftn_E(f exp(-2 pi i
t.x/n)))``, exactly, with the kernel on the sublattice of the class.  ``K`` is real and even (the half spectrum of the
real-to-complex path, extended evenly), so the operator maps real densities to real potentials, and two densities of
one class go through one complex transform as ``rho_1 + i rho_2`` (the potentials are the real and imaginary parts).

The coset form.  ``E`` is a fundamental domain of the half translations along ``H``, and the cosets ``h_c`` of ``T``
among them give ``S = {h_c E}``, a fundamental domain of ``T`` of ``N/|T|`` points: a sector orbital is fixed by its
values on ``S`` (its *coset form*), a class component of a pair density is ``rho_t(r) = (1/m) sum_c chi_t(h_c)
rho(h_c r)`` and its potential ``v(h_c r) = sum_t chi_t(h_c) v_t(r)`` (``m = |S|/|E|`` cosets; the tables of
:class:`.exchange_device.ParityGroup`, whose sign operations become half translations here).  The orbitals of every
sector are gathered once into the coset form, grouped by their character on ``T`` (one block per character: the pooled
block of :mod:`.exchange_device` on ``S`` instead of the grid, ``1/|T|`` of its memory).  The tiles of those blocks
(each pair of two occupied states once, the empty states against the occupied ones) form the class densities of up to
``densities_per_solve`` pairs at a time with one CUDA kernel, straight into the twisted complex boxes, transform them,
multiply them by the kernel of their class, transform them back, and accumulate ``W`` on ``S`` with a second kernel
that untwists the potentials (one thread per coset row and column of ``W`` where a pair has at most two classes,
``PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE``, else one per coset row: the same bits).  ``W`` is then folded into every sector from one point of each orbit, as the parity
classes of a cluster's sectors are.  Every pair density costs ``N/|T|`` points instead of ``N``: for a diamond
supercell of even repeats (``|T| = 8``) the transforms are eight times smaller.  The energies are those of the pooled
full-grid block (``off``) to round-off.

``off``, a cell whose sectors carry no half translation and a slab's padded box take the pooled full-grid block
(:func:`.exchange_device._apply_pooled_exchange`) as before.  A build whose coset blocks do not fit the free memory
of the device takes the pooled block where that fits, else the tiles of the sectors (``pool-fallback``); as the block
is ``|T|`` times larger than the coset blocks, such a build usually takes the tiles (the solver's description names
those builds, ``translations-unfit``).  A cluster's sector orbitals never get here: they keep their parity classes or
tiles (:func:`.exchange_device.pooled_exchange_applies` needs the solver of a periodic cell, ``small_build``).

Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from . import exchange_device as _exchange
from .exchange_device import (
    POOL_MEMORY_MARGIN,
    ChannelOrbitals,
    DeviceExchangeApplication,
    OrbitalSector,
    ParityGroup,
    _parity_sector,
    _tile_pairs,
    assign_tiles,
    plan_exchange,
    run_exchange_tiles,
    sector_exchange_energies,
)

SETTING = "PARSEC_CUPY_EXCHANGE_TRANSLATIONS"
#: Bytes of the complex class boxes of one batch of pair densities at most (at least ``Exx_Batch_Size`` pairs), as
#: the chunks of the periodic k-point loop (``bloch_exchange_device.CHUNK_BYTES``): few and large transforms keep the
#: threads of several devices from waiting on each other's Python calls.
BATCH_BYTES = 1024 * 1024 * 1024
#: Pairs of one batch at most.
BATCH_PAIRS = 4096
#: A build of fewer pair-density points than this (pairs times the points of the coset form) runs on the device of
#: the blocks alone, as a small build of the full grid (``bloch_exchange_device.SMALL_BUILD_POINTS``).
SMALL_BUILD_POINTS = 1.0e9


def exchange_translation_setting() -> str:
    """``PARSEC_CUPY_EXCHANGE_TRANSLATIONS``: ``auto`` (default) or ``off``.

    ``auto``: the sector orbitals that take the pooled block (``PARSEC_CUPY_EXCHANGE_POOL=auto``) of a bulk cell whose
    sectors carry half-cell translations solve their pair densities in the classes of those translations, on the
    sublattice of each class (this module).  ``off``: the pooled full-grid block, bit for bit as before.  A wrong value
    is refused when the device exchange is set up (:func:`parsec_python.acceleration.SCF.hybrid.device_exchange_for`).
    """

    value = os.environ.get(SETTING, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{SETTING} must be auto or off, not {value!r}")
    return value


def _axes(mask: int) -> tuple[int, ...]:
    """The axes ``d`` whose bit ``2 - d`` is set in ``mask``."""

    return tuple(d for d in range(3) if (int(mask) >> (2 - d)) & 1)


def _character(parity: int, translation: int) -> int:
    """``(-1)^(t.a)``: the sign of a class-``t`` function under the half translation ``a``."""

    return -1 if bin(int(parity) & int(translation)).count("1") % 2 else 1


# ---------------------------------------------------------------------------
# The reduced box and the translations of the sectors.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslationLayout:
    """The reduced box ``E`` of a grid box ``shape`` (row-major) halved along the axes of ``halved`` (a bit mask).

    ``images[a]`` is the grid index of every point of ``E`` moved by the half translation ``a`` (``-1`` for an ``a``
    that moves an axis not halved), in the row-major order of ``E``.  The attributes the parity classes of a cluster
    read (``images``, ``size``) make it a layout of :func:`.exchange_device._parity_sector`.
    """

    shape: tuple[int, int, int]
    halved: int
    reduced_shape: tuple[int, int, int]
    images: np.ndarray

    @property
    def size(self) -> int:
        return int(np.prod(self.reduced_shape))

    @property
    def label(self) -> str:
        return "x".join(str(value) for value in self.reduced_shape)

    def twist(self, parity: int) -> np.ndarray:
        """``exp(-2 pi i t.x/n)`` on ``E`` (row-major) for the class ``parity``."""

        factors = []
        for d in range(3):
            count = int(self.reduced_shape[d])
            if d in _axes(parity):
                factors.append(np.exp(-2.0j * np.pi * np.arange(count, dtype=np.float64) / float(self.shape[d])))
            else:
                factors.append(np.ones(count, dtype=np.complex128))
        values = factors[0][:, None, None] * factors[1][None, :, None] * factors[2][None, None, :]
        return np.ascontiguousarray(values.reshape(-1))

    def kernel_slices(self, parity: int) -> tuple[slice, slice, slice]:
        """The frequencies ``2k + t`` of the class ``parity`` in the full box: every other one along a halved axis."""

        halved = _axes(self.halved)
        return tuple(
            slice(1 if d in _axes(parity) else 0, None, 2) if d in halved else slice(None) for d in range(3)
        )


_LAYOUTS: dict = {}
_LAYOUTS_LOCK = Lock()


def translation_layout(shape, halved: int) -> TranslationLayout:
    """:class:`TranslationLayout` of ``shape`` halved along ``halved`` (each halved axis has an even count), cached."""

    shape = tuple(int(value) for value in shape)
    halved = int(halved)
    key = (shape, halved)
    with _LAYOUTS_LOCK:
        found = _LAYOUTS.get(key)
    if found is not None:
        return found
    axes = _axes(halved)
    if any(shape[d] % 2 for d in axes):
        raise ValueError("a halved axis needs an even number of points")
    reduced = tuple(shape[d] // 2 if d in axes else shape[d] for d in range(3))
    coordinates = [np.arange(reduced[d], dtype=np.int64) for d in range(3)]
    images = np.full((8, int(np.prod(reduced))), -1, dtype=np.int64)
    for mask in range(8):
        if mask & ~halved:
            continue
        moved = [coordinates[d] + (shape[d] // 2 if d in _axes(mask) else 0) for d in range(3)]
        index = (moved[0][:, None, None] * shape[1] + moved[1][None, :, None]) * shape[2] + moved[2][None, None, :]
        images[mask] = index.reshape(-1)
    images.setflags(write=False)
    found = TranslationLayout(shape, halved, reduced, images)
    with _LAYOUTS_LOCK:
        _LAYOUTS[key] = found
    return found


def _shift(shape, mask: int) -> np.ndarray:
    """Grid index of ``p + a`` for every grid point ``p`` (row-major), ``a`` the half translation ``mask``."""

    steps = tuple(-(int(shape[d]) // 2) if d in _axes(mask) else 0 for d in range(3))
    grid = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(tuple(int(value) for value in shape))
    return np.roll(grid, steps, axis=(0, 1, 2)).reshape(-1)


def _translation_character(expansion, shift: np.ndarray) -> int | None:
    """``chi(a)`` of a sector whose orbits the half translation ``a`` (``shift``) keeps, else ``None``.

    Decided on every grid point: ``a`` belongs to the group of the sectors when every point and its image lie in the
    same orbit (sector row), admitted by the sector both or neither; the character is the sign of the ratio of their
    phases, one sign for all.
    """

    weights = expansion.point_weights
    admitted = weights != 0.0
    moved = weights[shift]
    if not np.array_equal(moved != 0.0, admitted):
        return None
    rows = expansion.point_rows
    if not np.array_equal(rows[shift][admitted], rows[admitted]):
        return None
    signs = np.sign(moved[admitted]) * np.sign(weights[admitted])
    if not signs.size or not np.all(signs == signs[0]):
        return None
    return int(signs[0])


def _span(generators) -> dict[int, tuple[int, ...]]:
    """Every sum (exclusive or) of ``generators``, with the generators it takes."""

    found = {0: ()}
    for generator in generators:
        for mask, parts in list(found.items()):
            found.setdefault(mask ^ generator, parts + (generator,))
    return found


def _translation_group(orbitals: ChannelOrbitals, shape) -> tuple[tuple[int, ...], list[dict]] | None:
    """The half translations ``T`` the sectors of ``orbitals`` keep and every sector's character on them.

    ``T`` is found on the first sector (every half translation along axes of an even count), its generators are
    checked on every other sector; both are cached on the expansions.  ``None`` where ``T`` is the identity alone or a
    sector does not carry it.
    """

    shape = tuple(int(value) for value in shape)
    candidates = [mask for mask in range(1, 8) if not any(shape[d] % 2 for d in _axes(mask))]
    if not candidates:
        return None
    key = ("translations", shape)
    shifts: dict[int, np.ndarray] = {}

    def shift(mask: int) -> np.ndarray:
        found = shifts.get(mask)
        if found is None:
            found = shifts[mask] = _shift(shape, mask)
        return found

    first = orbitals.sectors[0].expansion
    found = first._parity.get(key)
    if found is None:
        found = {0: 1}
        for mask in candidates:
            value = _translation_character(first, shift(mask))
            if value is not None:
                found[mask] = value
        first._parity[key] = found
    if not found:
        return None
    elements = tuple(sorted(found))
    if len(elements) < 2:
        return None
    generators: list[int] = []
    for mask in elements:
        if mask not in _span(generators):
            generators.append(mask)
    span = _span(generators)
    if set(span) != set(elements):
        return None
    characters = []
    for sector in orbitals.sectors:
        expansion = sector.expansion
        values = expansion._parity.get(key)
        if values is None or (values and set(values) != set(elements)):
            values = {0: 1}
            for mask in generators:
                value = _translation_character(expansion, shift(mask))
                if value is None:
                    values = False
                    break
                values[mask] = value
            if values:
                for mask, parts in span.items():
                    values[mask] = int(np.prod([values[part] for part in parts])) if parts else 1
            expansion._parity[key] = values
        if not values:
            return None
        characters.append({mask: int(values[mask]) for mask in elements})
    return elements, characters


def _coset_group(elements: tuple[int, ...], halved: int) -> ParityGroup:
    """The half translations ``T`` and one translation of every coset of ``T`` among those along ``halved``."""

    covered: set[int] = set()
    cosets = []
    for mask in range(8):
        if mask & ~int(halved) or mask in covered:
            continue
        cosets.append(mask)
        covered.update(mask ^ element for element in elements)
    return ParityGroup(tuple(elements), tuple(cosets))


# ---------------------------------------------------------------------------
# The context of one channel.
# ---------------------------------------------------------------------------


@dataclass
class TranslationContext:
    """What the translation classes of one channel need: the reduced box, ``T`` and its cosets, the sectors' coset
    maps (:class:`.exchange_device.ParitySector` on ``S``) and their characters on ``T``."""

    layout: TranslationLayout
    group: ParityGroup
    characters: tuple
    sectors: tuple
    densities_per_solve: int
    _tables: dict = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def cosets(self) -> int:
        return len(self.group.cosets)

    @property
    def coset_rows(self) -> int:
        return self.cosets * int(self.layout.size)

    @property
    def order(self) -> int:
        """``|T|``, the identity included."""

        return len(self.group.operations)

    @property
    def label(self) -> str:
        """What the solver's description names (:meth:`.bloch_exchange_device.DevicePeriodicPairSolvers.describe`):
        the half-cell translations without the identity and the reduced box, e.g. ``7 half-cell translations, the
        36x36x36 box of each class`` (a body-centred cell: ``half-cell translation, the 8x8x8 box of each class``)."""

        count = self.order - 1
        if count == 1:
            return f"half-cell translation, the {self.layout.label} box of each class"
        return f"{count} half-cell translations, the {self.layout.label} box of each class"

    def classes(self, character: dict) -> tuple[int, ...]:
        """The parity classes along the halved axes whose character on ``T`` is ``character`` (mask -> +-1)."""

        halved = int(self.layout.halved)
        return tuple(
            t
            for t in range(8)
            if not t & ~halved and all(_character(t, a) == character[a] for a in self.group.operations)
        )

    def pair_classes(self, first: dict, second: dict) -> tuple[int, ...]:
        return self.classes({a: first[a] * second[a] for a in self.group.operations})

    def updates_by_columns(self) -> bool:
        """Whether the builds update ``W`` by coset row and column (``PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE=auto`` and a
        pair density of at most :data:`COLUMN_UPDATE_CLASSES` classes, the cosets of ``T``): the same bits as by
        rows."""

        return exchange_column_update_setting() == "auto" and self.cosets <= COLUMN_UPDATE_CLASSES

    def tables(self, cp, classes: tuple[int, ...], inverse_volume: float):
        """Device ``chi_t(h_c)/(m h^3)`` (the class densities), ``chi_t(h_c)`` (``W``) and the twists of ``classes``."""

        key = (int(cp.cuda.Device().id), tuple(classes), float(inverse_volume))
        with self._lock:
            found = self._tables.get(key)
        if found is None:
            table = self.group.table(classes)
            twists = np.stack([self.layout.twist(parity) for parity in classes])
            found = (
                cp.asarray(table * (float(inverse_volume) / self.cosets)),
                cp.asarray(table),
                cp.asarray(twists),
            )
            with self._lock:
                self._tables[key] = found
        return found

    def run_tiles(self, cp, orbitals, solvers, blocks, work, device: int, deliver, timing: dict, ready=None) -> None:
        """The tiles ``work`` of ``device`` in translation classes (the module text), for
        :func:`.exchange_device.run_exchange_tiles`: ``orbitals`` is the :class:`TranslationChannel` of the coset
        blocks, ``deliver`` adds a block's ``W`` (coset form) to its block of every column."""

        inverse_volume = 1.0 / float(orbitals.volume_element)
        pending: list = []
        step = int(self.densities_per_solve)
        by_columns = self.updates_by_columns()

        def settle() -> None:
            for begin, finish in pending:
                finish.synchronize()
                timing[device] += cp.cuda.get_elapsed_time(begin, finish) * 1.0e-3
            pending.clear()

        def weights_of(block):
            sector = orbitals.sectors[block.sector]
            return cp.asarray(orbitals.occupations[sector.states[block.positions]])

        with cp.cuda.Device(device), solvers.stream(device):
            first_index = None
            first = first_acc = first_weights = None
            for tile in work:
                if ready is not None:
                    ready(tile)
                if tile.first != first_index:
                    if first_acc is not None:
                        deliver(blocks[first_index], first_acc)
                    first_index = tile.first
                    block = blocks[first_index]
                    first = orbitals.sectors[block.sector].columns(cp, block.positions)
                    first_acc = cp.zeros(first.shape, dtype=cp.float64, order="F")
                    first_weights = weights_of(block)
                second_block = blocks[tile.second]
                if tile.kind == "diagonal":
                    second, second_acc, second_weights = first, first_acc, first_weights
                else:
                    second = orbitals.sectors[second_block.sector].columns(cp, second_block.positions)
                    second_acc = (
                        cp.zeros(second.shape, dtype=cp.float64, order="F") if tile.kind == "pair" else None
                    )
                    second_weights = weights_of(second_block)
                classes = self.pair_classes(
                    orbitals.characters[blocks[first_index].sector], orbitals.characters[second_block.sector]
                )
                density_table, table, twists = self.tables(cp, classes, inverse_volume)
                pairs = _tile_pairs(tile.kind, first.shape[1], second.shape[1])
                left, right, update = (cp.asarray(values) for values in pairs)
                shape = (tile.kind, int(first.shape[1]), int(second.shape[1]))
                aliased = second_acc is None or second_acc is first_acc
                for begin in range(0, int(left.size), step):
                    i, j, u = left[begin : begin + step], right[begin : begin + step], update[begin : begin + step]
                    boxes = translation_densities(cp, self, first, second, i, j, density_table, twists)
                    start, finish = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    solvers.solve_translation_classes(cp, boxes, classes, self.layout)
                    finish.record()
                    pending.append((start, finish))
                    events = (
                        _device_events(
                            cp, shape, pairs, begin, min(begin + step, int(left.size)), int(first.shape[1]), aliased
                        )
                        if by_columns
                        else None
                    )
                    translation_update(
                        cp, self, boxes, twists, table, first, second, i, j, first_weights, second_weights, u,
                        first_acc, second_acc, events=events,
                    )
                    del boxes
                settle()
                if tile.kind == "pair":
                    deliver(second_block, second_acc)
                del second, second_acc
            if first_acc is not None:
                deliver(blocks[first_index], first_acc)


def translation_context(orbitals: ChannelOrbitals, solvers) -> TranslationContext | None:
    """The :class:`TranslationContext` of a channel's sector orbitals, or ``None`` where the pooled full-grid block
    stays: ``PARSEC_CUPY_EXCHANGE_TRANSLATIONS=off``, a solver without class solves or whose FFT box is not the grid's
    box (a slab's padded box, a cluster's Kronecker solver), orbitals on the full grid, sectors without half-cell
    translations."""

    if exchange_translation_setting() == "off":
        return None
    box = getattr(solvers, "box", None)
    if box is None or not callable(getattr(solvers, "solve_translation_classes", None)):
        return None
    if not getattr(box, "identity", False) or tuple(box.fft_shape) != tuple(box.box_shape):
        return None
    sectors = orbitals.sectors
    if len(sectors) < 2 or any(sector.expansion is None for sector in sectors):
        return None
    shape = tuple(int(value) for value in box.box_shape)
    if int(np.prod(shape)) != int(orbitals.full_size):
        return None
    found = _translation_group(orbitals, shape)
    if found is None:
        return None
    elements, characters = found
    halved = 0
    for mask in elements:
        halved |= int(mask)
    layout = translation_layout(shape, halved)
    group = _coset_group(elements, halved)
    coset_sectors = []
    for sector, character in zip(sectors, characters):
        key = ("translation-sector", shape, halved, elements)
        cached = sector.expansion._parity.get(key)
        if cached is None:
            cached = _parity_sector(sector.expansion, layout, group, character)
            sector.expansion._parity[key] = cached
        coset_sectors.append(cached)
    rows = len(group.cosets) * layout.size
    step = max(int(solvers.batch_size), min(BATCH_PAIRS, int(BATCH_BYTES // (8 * rows))))
    return TranslationContext(layout, group, tuple(characters), tuple(coset_sectors), step)


# ---------------------------------------------------------------------------
# CUDA kernels of the class densities and of W.
# ---------------------------------------------------------------------------


_KERNEL_SOURCE = r"""
#include <cupy/complex.cuh>

// The class densities of the pairs (left[k], right[k]) of two coset blocks, two pairs to one complex value and
// twisted: out[t, h, r] = (rho_t,2h(r) + i rho_t,2h+1(r)) twist_t(r), rho_t(r) = sum_c table[t, c] q_i(h_c r) q_j(h_c r)
// (table holds chi_t(h_c) / (m h^3)); the rows h past the pairs (and the odd pair's imaginary part) are zero.
extern "C" __global__ void translation_pair_densities(
    const double* __restrict__ first,            // coset form of block 1 (m n_r x columns)
    const double* __restrict__ second,           // coset form of block 2
    const long long* __restrict__ left,
    const long long* __restrict__ right,
    const double* __restrict__ table,            // classes x m
    const complex<double>* __restrict__ twist,   // classes x n_r
    complex<double>* __restrict__ out,           // classes x halves x n_r
    const long long n_r, const int cosets, const int classes, const int count, const int halves,
    const int per_thread)
{
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_r) return;
    const long long ld = (long long)cosets * n_r;
    const int h0 = blockIdx.y * per_thread;
    const int h1 = min(halves, h0 + per_thread);
    double products[8];
    double values[2][8];
    for (int h = h0; h < h1; ++h) {
        for (int part = 0; part < 2; ++part) {
            const int k = 2 * h + part;
            if (k >= count) {
                for (int t = 0; t < classes; ++t) values[part][t] = 0.0;
                continue;
            }
            const double* a = first + left[k] * ld + r;
            const double* b = second + right[k] * ld + r;
            for (int c = 0; c < cosets; ++c) products[c] = a[c * n_r] * b[c * n_r];
            for (int t = 0; t < classes; ++t) {
                double sum = 0.0;
                for (int c = 0; c < cosets; ++c) sum += table[t * cosets + c] * products[c];
                values[part][t] = sum;
            }
        }
        for (int t = 0; t < classes; ++t) {
            out[((long long)t * halves + h) * n_r + r] =
                complex<double>(values[0][t], values[1][t]) * twist[(long long)t * n_r + r];
        }
    }
}

// W_i -= sum_j f_j q_j v_ij and, where second_update, W_j -= f_i q_i v_ij, in coset form, with
// v_ij(h_c r) = sum_t chi_t(h_c) v_t,ij(r) and v_t,2h + i v_t,2h+1 = boxes[t, h, r] conj(twist_t(r)).  One thread per
// coset row runs the pairs in their order (fixed-order sums; second_acc may be first_acc).
extern "C" __global__ void translation_pair_update(
    const complex<double>* __restrict__ boxes,   // classes x halves x n_r
    const complex<double>* __restrict__ twist,   // classes x n_r
    const double* __restrict__ table,            // chi_t(h_c), classes x m
    const double* first, const double* second,
    const long long* __restrict__ left, const long long* __restrict__ right,
    const double* __restrict__ first_weights, const double* __restrict__ second_weights,
    const signed char* __restrict__ second_update,
    double* first_acc, double* second_acc,
    const long long n_r, const int cosets, const int classes, const int count, const int halves)
{
    const long long ld = (long long)cosets * n_r;
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= ld) return;
    const int c = (int)(p / n_r);
    const long long r = p - (long long)c * n_r;
    double coefficient[8];
    complex<double> untwist[8];
    complex<double> potential[8];
    for (int t = 0; t < classes; ++t) {
        coefficient[t] = table[t * cosets + c];
        untwist[t] = conj(twist[(long long)t * n_r + r]);
    }
    double sum = 0.0;
    long long current = -1;
    for (int k = 0; k < count; ++k) {
        if ((k & 1) == 0) {
            for (int t = 0; t < classes; ++t) {
                potential[t] = boxes[((long long)t * halves + (k >> 1)) * n_r + r] * untwist[t];
            }
        }
        double v = 0.0;
        for (int t = 0; t < classes; ++t) v += coefficient[t] * ((k & 1) ? potential[t].imag() : potential[t].real());
        const long long i = left[k];
        const long long j = right[k];
        if (i != current) {
            if (current >= 0) first_acc[p + current * ld] -= sum;
            sum = 0.0;
            current = i;
        }
        sum += second_weights[j] * second[p + j * ld] * v;
        if (second_update[k]) second_acc[p + j * ld] -= first_weights[i] * first[p + i * ld] * v;
    }
    if (current >= 0) first_acc[p + current * ld] -= sum;
}

// The update above with one thread per coset row and column of W (PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE=auto): column
// `column` (of first_acc, or of second_acc from `split` on where `aliased` is 0) takes the events of the pairs that
// write it, events[offsets[column] .. offsets[column + 1]) in the order of the pairs, with the operations of the
// kernel above in the same order: role bit 1, the pair (i = column, j) whose sum of a run of consecutive pairs of
// the same i is subtracted where the run ends; bit 2, the pair (i, j = column) subtracted at once.  The same values
// in the same order: the same bits.  blockIdx.x is the column, so the blocks that run together hold every column of
// the same rows (their potentials and orbitals shared in the caches).
extern "C" __global__ void translation_pair_update_columns(
    const complex<double>* __restrict__ boxes,   // classes x halves x n_r
    const complex<double>* __restrict__ twist,   // classes x n_r
    const double* __restrict__ table,            // chi_t(h_c), classes x m
    const double* first, const double* second,
    const long long* __restrict__ left, const long long* __restrict__ right,
    const double* __restrict__ first_weights, const double* __restrict__ second_weights,
    const int* __restrict__ offsets, const int* __restrict__ events, const signed char* __restrict__ roles,
    double* first_acc, double* second_acc,
    const long long n_r, const int cosets, const int classes, const int halves, const int split, const int aliased)
{
    const int column = blockIdx.x;
    const int begin = offsets[column];
    const int end = offsets[column + 1];
    if (begin == end) return;
    const long long ld = (long long)cosets * n_r;
    double* target = (aliased || column < split) ? first_acc + (long long)column * ld
                                                 : second_acc + (long long)(column - split) * ld;
    for (long long p = (long long)blockIdx.y * blockDim.x + threadIdx.x; p < ld;
         p += (long long)gridDim.y * blockDim.x) {
        const int c = (int)(p / n_r);
        const long long r = p - (long long)c * n_r;
        double coefficient[8];
        complex<double> untwist[8];
        complex<double> potential[8];
        for (int t = 0; t < classes; ++t) {
            coefficient[t] = table[t * cosets + c];
            untwist[t] = conj(twist[(long long)t * n_r + r]);
        }
        double value = target[p];
        double sum = 0.0;
        int running = 0;
        int last = -2;
        int loaded = -1;
        for (int e = begin; e < end; ++e) {
            const int k = events[e];
            const int role = roles[e];
            if (running && !((role & 1) && k == last + 1)) {
                value -= sum;
                running = 0;
            }
            if ((k >> 1) != loaded) {
                loaded = k >> 1;
                for (int t = 0; t < classes; ++t) {
                    potential[t] = boxes[((long long)t * halves + loaded) * n_r + r] * untwist[t];
                }
            }
            double v = 0.0;
            for (int t = 0; t < classes; ++t) v += coefficient[t] * ((k & 1) ? potential[t].imag() : potential[t].real());
            if (role & 1) {
                if (!running) {
                    sum = 0.0;
                    running = 1;
                }
                const long long j = right[k];
                sum += second_weights[j] * second[p + j * ld] * v;
                last = k;
            }
            if (role & 2) {
                const long long i = left[k];
                value -= first_weights[i] * first[p + i * ld] * v;
            }
        }
        if (running) value -= sum;
        target[p] = value;
    }
}
"""
_KERNELS: dict[int, Any] = {}
_KERNELS_LOCK = Lock()
_HALVES_PER_THREAD = 4
_THREADS = 256
#: The rows of the complex boxes of one batch are rounded up to a multiple of this (fewer cuFFT plans).
_HALVES_STEP = 8


def _kernels(cp):
    device = int(cp.cuda.Device().id)
    with _KERNELS_LOCK:
        found = _KERNELS.get(device)
        if found is None:
            module = cp.RawModule(code=_KERNEL_SOURCE, options=("--fmad=false",))
            found = (
                module.get_function("translation_pair_densities"),
                module.get_function("translation_pair_update"),
                module.get_function("translation_pair_update_columns"),
            )
            _KERNELS[device] = found
    return found


COLUMN_UPDATE_SETTING = "PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE"
#: The update by columns is taken where the pair densities have at most this many classes (the cosets of ``T``):
#: it reads the class potentials of a pair twice (for ``W_i`` and for ``W_j``), which four classes make slower than
#: the update by rows (a body-centred cube, measured on A100), while one or two classes gain from the threads.
COLUMN_UPDATE_CLASSES = 2


def exchange_column_update_setting() -> str:
    """``PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE``: ``auto`` (default, since 2026-11-21) or ``off``.

    ``auto``: the update of ``W`` from the class potentials of a batch of pairs (:func:`translation_update`) runs one
    thread per coset row and column of ``W`` (``translation_pair_update_columns``), each column taking the pairs that
    write it in the order of the pairs, so the sums are those of ``off`` to the last bit, where a pair density has
    at most :data:`COLUMN_UPDATE_CLASSES` classes (one for a supercell of even repeats, two for the face-centring
    translations of a cube; a body-centred cell, four, keeps the update by rows).  ``off``: one thread per coset
    row, which runs every pair of the batch, as before (a few hundred blocks for the coset form of a 512-atom
    diamond cell: about a third of the GPU time of a build).  A wrong value is refused when the device exchange is
    set up (:func:`parsec_python.acceleration.SCF.hybrid.device_exchange_for`).
    """

    value = os.environ.get(COLUMN_UPDATE_SETTING, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{COLUMN_UPDATE_SETTING} must be auto or off, not {value!r}")
    return value


def column_events(left, right, update, first_columns: int, *, aliased: bool):
    """The events of :func:`translation_update` by column of ``W`` (``translation_pair_update_columns``).

    ``(offsets, events, roles)``: column ``c`` (of the first accumulator; with ``aliased`` false the columns from
    ``first_columns`` on are those of the second) takes the pairs ``events[offsets[c]:offsets[c + 1]]`` in their
    order, with ``roles`` bit 1 where it is the ``W_i`` of the pair (``left``) and bit 2 where it is the ``W_j`` the
    pair updates (``right`` where ``update``).  A pair whose ``i`` and ``j`` are the same column of one accumulator
    is one event with both bits (the kernel runs bit 1 first, as the update by rows does).
    """

    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    count = int(left.size)
    pairs = np.arange(count, dtype=np.int64)
    updated = np.flatnonzero(np.asarray(update) != 0)
    shift = 0 if aliased else int(first_columns)
    column = np.concatenate([left, right[updated] + shift])
    pair = np.concatenate([pairs, updated])
    role = np.concatenate([np.ones(count, dtype=np.int8), np.full(updated.size, 2, dtype=np.int8)])
    order = np.lexsort((role, pair, column))
    column, pair, role = column[order], pair[order], role[order]
    if column.size > 1:
        same = np.flatnonzero((column[1:] == column[:-1]) & (pair[1:] == pair[:-1]))
        if same.size:
            role[same] |= role[same + 1]
            keep = np.ones(column.size, dtype=bool)
            keep[same + 1] = False
            column, pair, role = column[keep], pair[keep], role[keep]
    columns = int(first_columns)
    if column.size:
        columns = max(columns, int(column[-1]) + 1)
    offsets = np.searchsorted(column, np.arange(columns + 1, dtype=np.int64)).astype(np.int32)
    return offsets, pair.astype(np.int32), role


_EVENTS: dict = {}
_EVENTS_LOCK = Lock()
#: Event tables kept on the devices at most (a few kB each: the batches of the tile shapes of a run).
_EVENTS_LIMIT = 1024


def _device_events(cp, key, pairs, begin: int, finish: int, first_columns: int, aliased: bool):
    """:func:`column_events` of the pairs ``[begin, finish)`` of a tile on the current device, cached by ``key``."""

    key = (int(cp.cuda.Device().id),) + tuple(key) + (int(begin), int(finish), bool(aliased))
    with _EVENTS_LOCK:
        found = _EVENTS.get(key)
    if found is None:
        left, right, update = (values[begin:finish] for values in pairs)
        offsets, events, roles = column_events(left, right, update, first_columns, aliased=aliased)
        found = (cp.asarray(offsets), cp.asarray(events), cp.asarray(roles), int(offsets.size) - 1)
        with _EVENTS_LOCK:
            if len(_EVENTS) >= _EVENTS_LIMIT:
                _EVENTS.clear()
            _EVENTS[key] = found
    return found


def batch_halves(count: int) -> int:
    """Rows of the complex boxes of ``count`` pairs: two pairs a row, rounded up to :data:`_HALVES_STEP`."""

    halves = (int(count) + 1) // 2
    return ((halves + _HALVES_STEP - 1) // _HALVES_STEP) * _HALVES_STEP


def translation_densities(cp, context: TranslationContext, first, second, left, right, table, twists):
    """The twisted complex class boxes ``(classes, halves, n_r)`` of the pairs ``(left[k], right[k])`` of two coset
    blocks (F order, ``coset_rows`` rows): ``(rho_t,2h + i rho_t,2h+1) exp(-2 pi i t.x/n)``."""

    for array in (first, second):
        if not array.flags.f_contiguous or int(array.shape[0]) != context.coset_rows:
            raise ValueError("the class densities need F-contiguous coset blocks")
    kernel = _kernels(cp)[0]
    size = int(context.layout.size)
    classes = int(table.shape[0])
    count = int(left.size)
    halves = batch_halves(count)
    out = cp.empty((classes, halves, size), dtype=cp.complex128)
    grid = ((size + _THREADS - 1) // _THREADS, (halves + _HALVES_PER_THREAD - 1) // _HALVES_PER_THREAD)
    kernel(
        grid,
        (_THREADS,),
        (first, second, left, right, table, twists, out, np.int64(size), np.int32(context.cosets),
         np.int32(classes), np.int32(count), np.int32(halves), np.int32(_HALVES_PER_THREAD)),
    )
    return out


def translation_update(cp, context: TranslationContext, boxes, twists, table, first, second, left, right,
                       first_weights, second_weights, second_update, first_acc, second_acc, *, events=None) -> None:
    """``W_i -= sum_j f_j q_j v_ij`` and, where ``second_update``, ``W_j -= f_i q_i v_ij`` in coset form, from the
    class potentials of :meth:`DevicePeriodicPairSolvers.solve_translation_classes` (twisted, two pairs a row).

    ``events``: the device tables of :func:`column_events` for these pairs (``offsets, events, roles, columns``;
    ``aliased`` where ``second_acc`` is ``None`` or ``first_acc``), which run the update by row and column of ``W``
    (``PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE=auto``, the same bits); ``None``: by row, as before."""

    rows = context.coset_rows
    for array in (first, second, first_acc) + (() if second_acc is None else (second_acc,)):
        if not array.flags.f_contiguous or int(array.shape[0]) != rows:
            raise ValueError("the update of W needs F-contiguous coset blocks")
    classes = int(boxes.shape[0])
    if events is not None:
        offsets, pairs, roles, columns = events
        aliased = second_acc is None or second_acc is first_acc
        split = int(first_acc.shape[1])
        if columns <= 0:
            return
        blocks = min((rows + _THREADS - 1) // _THREADS, 65535)
        _kernels(cp)[2](
            (int(columns), int(blocks)),
            (_THREADS,),
            (
                boxes, twists, table, first, second, left, right, first_weights, second_weights, offsets, pairs, roles,
                first_acc, first_acc if second_acc is None else second_acc,
                np.int64(context.layout.size), np.int32(context.cosets), np.int32(classes),
                np.int32(int(boxes.shape[1])), np.int32(split), np.int32(1 if aliased else 0),
            ),
        )
        return
    kernel = _kernels(cp)[1]
    kernel(
        ((rows + _THREADS - 1) // _THREADS,),
        (_THREADS,),
        (
            boxes, twists, table, first, second, left, right, first_weights, second_weights, second_update,
            first_acc, first_acc if second_acc is None else second_acc,
            np.int64(context.layout.size), np.int32(context.cosets), np.int32(classes), np.int32(int(left.size)),
            np.int32(int(boxes.shape[1])),
        ),
    )


# ---------------------------------------------------------------------------
# The blocks of the coset form and a build.
# ---------------------------------------------------------------------------


@dataclass
class TranslationChannel(ChannelOrbitals):
    """The coset blocks of one channel: one block (an :class:`.exchange_device.OrbitalSector` without expansion, the
    coset form of its columns) per character on ``T``.

    ``characters[b]`` is the character of block ``b`` on ``T``; ``members[s]`` the block of the channel's sector ``s``
    and the first column of its states there.
    """

    context: Any = None
    characters: tuple = ()
    members: tuple = ()


def _coset_channel(cp, orbitals: ChannelOrbitals, context: TranslationContext, home: int) -> TranslationChannel:
    """The sector orbitals of ``orbitals`` gathered into the coset blocks of their characters on ``home``."""

    groups: dict[tuple, list[int]] = {}
    for index, character in enumerate(context.characters):
        groups.setdefault(tuple(character[a] for a in context.group.operations), []).append(index)
    rows = context.coset_rows
    blocks, characters, members = [], [], [None] * len(orbitals.sectors)
    with cp.cuda.Device(int(home)):
        for number, (_key, indices) in enumerate(groups.items()):
            width = int(sum(int(orbitals.sectors[index].local.size) for index in indices))
            values = cp.empty((rows, width), dtype=cp.float64, order="F")
            states = []
            offset = 0
            for index in indices:
                sector = orbitals.sectors[index]
                count = int(sector.local.size)
                members[index] = (number, offset)
                if count:
                    gather, weights, _entries, _fold = context.sectors[index].arrays(cp)
                    columns = sector.columns(cp, np.arange(count, dtype=np.int64))
                    part = columns[gather, :]
                    part *= weights[:, None]
                    values[:, offset : offset + count] = part
                    del columns, part
                states.append(np.asarray(sector.states, dtype=np.int64))
                offset += count
            blocks.append(
                OrbitalSector(
                    representation=None,
                    device=int(home),
                    vectors=values,
                    local=np.arange(width, dtype=np.int64),
                    states=np.concatenate(states) if states else np.zeros(0, dtype=np.int64),
                    expansion=None,
                )
            )
            characters.append(dict(context.characters[indices[0]]))
        cp.cuda.get_current_stream().synchronize()
    return TranslationChannel(
        tuple(blocks),
        orbitals.occupations,
        float(orbitals.spin_weight),
        int(rows),
        float(orbitals.volume_element),
        context=context,
        characters=tuple(characters),
        members=tuple(members),
    )


def _build_pairs(orbitals: ChannelOrbitals, columns: str, known) -> int:
    occupied = int(orbitals.occupied_states().size)
    empty = int(orbitals.state_count) - occupied
    pairs = 0 if known is not None else occupied * (occupied + 1) // 2
    if columns == "all" or known is not None:
        pairs += empty * occupied
    return pairs


def translation_devices(orbitals: ChannelOrbitals, context: TranslationContext, devices, columns: str,
                        known=None) -> tuple[int, ...]:
    """The devices of a build: ``devices``, or the home of the blocks alone for a small build
    (:data:`SMALL_BUILD_POINTS`)."""

    home = int(orbitals.sectors[0].device)
    devices = tuple(int(device) for device in devices) or (home,)
    if len(devices) > 1 and _build_pairs(orbitals, columns, known) * float(context.coset_rows) < SMALL_BUILD_POINTS:
        devices = (home,)
    return devices


def translation_exchange_bytes(orbitals: ChannelOrbitals, context: TranslationContext, solvers, devices, *,
                               columns: str = "all") -> dict[int, int]:
    """Device bytes a build in translation classes takes, by device (what the devices hold already not counted).

    On the home of the blocks (the device of the first sector): the coset blocks and their ``W`` (``8 S M`` each,
    ``S`` the coset rows), the coset maps of the sectors not there yet, and the larger of the gather of the widest
    sector (its columns and their coset form), the work of one tile (two blocks and two accumulators of
    ``Exx_Batch_Size`` columns, the complex boxes of one batch and as much again for cuFFT, a received part where
    other devices take tiles) and the fold (the folded ``W_G`` of the home's sectors and one coset part); on every
    other device of the build the work of one tile, and on the device of every sector its folded ``W_G``.
    """

    home = int(orbitals.sectors[0].device)
    rows = int(context.coset_rows)
    block = 8 * rows * int(orbitals.state_count)
    batch = int(solvers.batch_size)
    builders = translation_devices(orbitals, context, devices, columns)
    tile = 8 * rows * 4 * batch + 2 * 8 * rows * int(context.densities_per_solve)
    received = 8 * rows * batch if any(device != home for device in builders) else 0
    parts: dict[int, int] = {}
    gather = fold = maps = 0
    for sector, coset in zip(orbitals.sectors, context.sectors):
        count = int(sector.local.size)
        parts[int(sector.device)] = parts.get(int(sector.device), 0) + 8 * int(sector.rows) * count
        gather = max(gather, 8 * count * (int(sector.rows) + 2 * rows))
        fold = max(fold, 8 * count * (int(sector.rows) + rows))
        if int(home) not in coset._device:
            maps += int(coset.device_bytes())
    needed = {device: count for device, count in parts.items() if device != home}
    for device in builders:
        if device != home:
            needed[device] = needed.get(device, 0) + tile
    needed[home] = 2 * block + maps + max(gather, tile + received, fold + parts.get(home, 0))
    return needed


def translation_exchange_fits(orbitals: ChannelOrbitals, context: TranslationContext, solvers, devices, *,
                              columns: str = "all") -> bool:
    """Whether :func:`translation_exchange_bytes` and 2 % of every device fit what the devices have free now."""

    for device, count in sorted(translation_exchange_bytes(orbitals, context, solvers, devices,
                                                           columns=columns).items()):
        # through the module (the free memory the pooled block reads too)
        available, total = _exchange._device_memory(device)
        if count + POOL_MEMORY_MARGIN * float(total) > float(available):
            return False
    return True


def apply_translation_exchange(
    orbitals: ChannelOrbitals,
    solvers,
    devices: Sequence[int],
    *,
    columns: str,
    known: DeviceExchangeApplication | None,
    context: TranslationContext | None = None,
) -> DeviceExchangeApplication:
    """:func:`.exchange_device.apply_device_exchange` of sector orbitals in the classes of their half translations.

    The sector orbitals are gathered into coset blocks (:class:`TranslationChannel`), whose tiles form ``W`` on
    ``S`` (:meth:`TranslationContext.run_tiles`); ``W`` is folded into every sector and moved to its device.  An
    application of the occupied columns keeps the coset blocks and their ``W`` (``pooled``), which the build of every
    column completes and consumes, as the pooled full-grid block of :mod:`.exchange_device` is.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    started = perf_counter()
    if known is not None:
        inner_known = known.pooled
        channel = inner_known.orbitals
        context = channel.context
        # Consumed: the folded parts of the occupied build are folded again from the completed W.
        known.applied[:] = [None] * len(known.applied)
        known.pooled = None
    else:
        if context is None:
            raise ValueError("a build in translation classes needs its context")
        inner_known = None
        channel = _coset_channel(cp, orbitals, context, int(orbitals.sectors[0].device))
    home = int(channel.sectors[0].device)
    builders = translation_devices(channel, context, devices, columns, known=inner_known)
    plan = plan_exchange(channel, solvers.batch_size, columns=columns, known=inner_known)
    if inner_known is not None:
        applied = inner_known.applied
    else:
        applied = []
        with cp.cuda.Device(home):
            for block in channel.sectors:
                applied.append(cp.zeros((int(block.rows), int(block.local.size)), dtype=cp.float64, order="F"))
    label = getattr(solvers, "note_translations", None)
    if callable(label):
        label(context.label)
    timing = run_exchange_tiles(channel, solvers, plan, assign_tiles(plan.tiles, builders), applied, parity=context)
    inner = DeviceExchangeApplication(
        orbitals=channel,
        applied=applied,
        columns=plan.done(),
        occupied=plan.occupied,
        energy=0.0,
        pair_solves=int(plan.pair_solves + (0 if inner_known is None else inner_known.pair_solves)),
        poisson_seconds=float(
            max(timing.values(), default=0.0) + (0.0 if inner_known is None else inner_known.poisson_seconds)
        ),
        seconds=0.0,
    )
    # W of every sector from one coset point of each of its orbits, on the sector's device.
    folded = []
    with cp.cuda.Device(home):
        for index, sector in enumerate(orbitals.sectors):
            number, offset = channel.members[index]
            count = int(sector.local.size)
            _gather, _weights, entries, weights = context.sectors[index].arrays(cp)
            part = cp.asfortranarray(applied[number][:, offset : offset + count][entries, :])
            part *= weights[:, None]
            cp.cuda.get_current_stream().synchronize()
            if int(sector.device) != home:
                with cp.cuda.Device(int(sector.device)):
                    moved = cp.empty(part.shape, dtype=cp.float64, order="F")
                    moved.data.copy_from_device(part.data, part.nbytes)
                part = moved
            folded.append(part)
    if known is None:
        energy = 0.0
        for value in sector_exchange_energies(orbitals, folded).values():
            energy += value
        energy *= 0.5 * orbitals.spin_weight
    else:
        energy = float(known.energy)
    keep = columns == "occupied"
    result = DeviceExchangeApplication(
        orbitals=orbitals,
        applied=folded,
        columns=inner.columns,
        occupied=inner.occupied,
        energy=float(energy),
        pair_solves=int(inner.pair_solves),
        poisson_seconds=float(inner.poisson_seconds),
        seconds=float(perf_counter() - started + (0.0 if known is None else known.seconds)),
        pooled=inner if keep else None,
    )
    if not keep:
        inner.applied[:] = [None] * len(inner.applied)
        for block in channel.sectors:
            block.vectors = None
    return result


def reference_class_potentials(densities, layout: TranslationLayout, group: ParityGroup, classes, kernel):
    """NumPy reference of a class solve: the potentials ``(coset rows, nb)`` of real pair densities ``(coset rows,
    nb)`` in coset form whose character on ``T`` admits ``classes``, with the full kernel ``K(G)`` of the box
    (``v = ifftn(K fftn(rho))`` on the grid restricted to the coset form, for the tests)."""

    densities = np.asarray(densities, dtype=np.float64)
    size = layout.size
    cosets = len(group.cosets)
    table = group.table(tuple(classes))
    values = densities.reshape(cosets, size, -1)
    result = np.zeros_like(values)
    kernel = np.asarray(kernel, dtype=np.float64)
    for index, parity in enumerate(classes):
        component = np.einsum("c,cpb->bp", table[index], values) / cosets
        twist = layout.twist(parity)
        boxed = (component * twist[None, :]).reshape((-1,) + tuple(layout.reduced_shape))
        spectrum = np.fft.fftn(boxed, axes=(1, 2, 3))
        spectrum *= kernel[layout.kernel_slices(parity)][None]
        solved = np.fft.ifftn(spectrum, axes=(1, 2, 3)).reshape(-1, size) * np.conj(twist)[None, :]
        result += table[index][:, None, None] * np.real(solved).T[None, :, :]
    return result.reshape(cosets * size, -1)


__all__ = [
    "COLUMN_UPDATE_SETTING",
    "SETTING",
    "TranslationChannel",
    "TranslationContext",
    "TranslationLayout",
    "apply_translation_exchange",
    "column_events",
    "exchange_column_update_setting",
    "exchange_translation_setting",
    "reference_class_potentials",
    "translation_context",
    "translation_exchange_bytes",
    "translation_exchange_fits",
    "translation_layout",
]
