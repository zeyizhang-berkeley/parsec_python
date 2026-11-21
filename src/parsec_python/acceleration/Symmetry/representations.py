"""Real one-dimensional representations of a commuting reflection group.

For an orbit ``O_w`` with stabilizer ``S_w``, a real character ``chi_Gamma``
contributes one normalized orbital degree of freedom exactly when

``chi_Gamma(s) = 1  for every s in S_w``.

On an admitted orbit the expansion is

``U_Gamma[i,w] = chi_Gamma(g_i) / sqrt(|O_w|)``,

where ``g_i`` maps the selected representative to full-grid row ``i``.  An
orbit rejected by the stabilizer rule contributes no column to that sector.
This is the standard finite-group projector, not a boundary approximation:

``P_Gamma = |G|^-1 sum_g chi_Gamma(g) R(g)``.

Free half-shifted grids are the special case ``S_w={e}``; every sector then
retains the former common wedge dimension and byte-for-byte phase convention.
Zero-shift planes and axes instead give different, mathematically required
sector dimensions (for example, odd reflection states vanish on a mirror
plane).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
import os
import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.V_ion import NonlocalProjectorOperator

from .axis_reflection import (
    AxisReflectionReduction,
    PointGroupReduction,
    SignedPermutationReduction,
    _RowOrderTable,
    _fast_maps_requested,
    _grid_row_mapping_operation,
)


# PARSEC's D2h order is Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u.  For diagonal
# Cartesian signs these are respectively the scalar monomials
# 1, yz, xz, xy, xyz, x, y, z.  Deduplicating this order after restriction to
# a detected subgroup also gives deterministic labels for smaller groups.
_PARSEC_REFLECTION_PARITIES = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


def operator_build_workers(representation_count: int) -> int:
    """Return the measured cold-setup worker count.

    A/B measurements show that a four-worker pool pays for itself when there
    are several independent representations, while creating a pool for only
    two representations does not improve complete first-calculation time.
    An explicit environment setting remains authoritative for profiling.
    """

    count = int(representation_count)
    if count < 1:
        raise ValueError("representation_count must be positive")
    raw = os.environ.get("PARSEC_SYMMETRY_OPERATOR_WORKERS")
    if raw is None:
        return 1 if count <= 2 else min(count, 4)
    raw = raw.strip()
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_SYMMETRY_OPERATOR_WORKERS must be a positive integer"
        ) from error
    if requested < 1:
        raise ValueError(
            "PARSEC_SYMMETRY_OPERATOR_WORKERS must be a positive integer"
        )
    return min(count, requested)


def _representative_images(
    grid: RealSpaceGrid,
    reduction: AxisReflectionReduction,
) -> np.ndarray:
    """Return the full row of every operation applied to every representative.

    The decomposition reads a row map only at the orbit representatives, and
    the detector has already accepted every operation on the whole grid, so
    only those rows are mapped and no grid-sized temporary is built.

    With ``PARSEC_SYMMETRY_FAST_MAPS`` the whole-grid map of an operation is
    read out of the lookup table in the order of the rows, which costs less
    than the coordinates of the representatives alone, and its entries at
    the representatives are kept.  An operation without a whole-grid map
    is still decided on the representatives, and so is every operation of a
    grid whose rows are not numbered along its table.
    """

    operations = (
        reduction.operations
        if isinstance(reduction, SignedPermutationReduction)
        else [np.diag(np.asarray(signs, dtype=np.int8)) for signs in reduction.signs]
    )
    images = np.empty(
        (reduction.group_order, reduction.wedge_size), dtype=np.int64
    )
    table = _RowOrderTable(grid) if _fast_maps_requested() else None
    if table is not None and not table.in_row_order:
        # Nothing to read off: a whole-grid map would be gathered through the
        # coordinates of every point instead of the representatives.
        table = None
    translations = getattr(reduction, "translations", None)
    for index, operation in enumerate(operations):
        mapping = None
        # r -> R r + t in a cell that repeats a smaller one (PointGroupReduction.translations)
        translation = None if translations is None else translations[index]
        # The identity is its own shortcut: the representatives themselves.
        if table is not None and not np.array_equal(operation, np.eye(3)):
            whole = _grid_row_mapping_operation(
                grid, operation, 1.0e-10, table=table, translation=translation
            )
            if whole is not None:
                mapping = whole[reduction.representative_rows]
        if mapping is None:
            mapping = _grid_row_mapping_operation(
                grid, operation, 1.0e-10, reduction.representative_rows, translation=translation
            )
        if mapping is None:
            raise RuntimeError("accepted symmetry operation no longer maps the grid")
        images[index] = mapping
    return images


@dataclass(frozen=True)
class PartnerRows:
    """The partner rows of a unit that is a sector of a larger subgroup ``K`` (``PARSEC_ORBITAL_IRREP_ROWS=partner``).

    ``cosets``: the positions of the right cosets of ``H`` that make ``K``; ``first[k]``: the first row of the unit's
    sector of ``H`` in partner row ``k`` (ascending); ``row_of_sector[s]``: the partner row of sector row ``s``
    (``-1`` where the unit is zero there: an orbit of ``K`` that its character does not admit); ``values[s]``: the
    coefficient ``e_s`` of the isometry ``E`` (``x_sector = E x_partner``: ``x_sector[s] = values[s] *
    x_partner[row_of_sector[s]]``), ``+-sqrt(|O_s| / |O_K|)``, positive at ``first``.
    """

    representation: int
    cosets: tuple[int, ...]
    first: np.ndarray
    row_of_sector: np.ndarray
    values: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.first.size)

    def expand(self, vectors: np.ndarray) -> np.ndarray:
        """``E x``: partner rows (rows x columns, or one vector) to the coefficients of the sector of ``H``."""

        array = np.asarray(vectors, dtype=np.float64)
        held = self.row_of_sector >= 0
        out = np.zeros((self.row_of_sector.size,) + array.shape[1:], dtype=np.float64)
        scale = self.values[held] if array.ndim == 1 else self.values[held][:, None]
        out[held] = scale * array[self.row_of_sector[held]]
        return out

    def restrict(self, vectors: np.ndarray) -> np.ndarray:
        """``E^T y``: coefficients of the sector of ``H`` to partner rows."""

        array = np.asarray(vectors, dtype=np.float64)
        held = np.flatnonzero(self.row_of_sector >= 0)
        out = np.zeros((self.rows,) + array.shape[1:], dtype=np.float64)
        scale = self.values[held] if array.ndim == 1 else self.values[held][:, None]
        np.add.at(out, self.row_of_sector[held], scale * array[held])
        return out


@dataclass(frozen=True)
class ReflectionRepresentationDecomposition:
    """Character table, row phases, and stabilizer-filtered sector maps."""

    reduction: AxisReflectionReduction
    characters: np.ndarray
    phases: np.ndarray
    orbit_to_sector: np.ndarray
    # Orbit index list of each representation, kept after its first request.
    _sector_orbit_lists: dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # The projector tables of the units of irreducible representations (projection_tables), kept likewise.
    _projection_tables: dict[int, tuple] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # The partner rows of the units that are sectors of a larger subgroup (partner_rows, partner_decomposition,
    # solved_size; PARSEC_ORBITAL_IRREP_ROWS), kept likewise.
    _partner_cache: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def representation_count(self) -> int:
        return int(self.characters.shape[0])

    @property
    def full_size(self) -> int:
        return self.reduction.full_size

    @property
    def sector_weights(self) -> np.ndarray:
        """How many states each state of every sector counts for: 1 for the groups of before.

        A :class:`.axis_reflection.PointGroupReduction` solves the one-dimensional sectors of a subgroup of
        the full point group only; a state of one counts for the partners of its level that no solved
        sector holds (:mod:`.orbital_groups`).
        """

        weights = getattr(self.reduction, "sector_weights", None)
        if weights is None:
            return np.ones(self.representation_count, dtype=np.int64)
        return np.asarray(weights, dtype=np.int64)

    @property
    def sector_partner_masks(self) -> tuple[int, ...] | None:
        """Per sector, a bit mask of the representations of the full group whose partners several solved states share.

        Bit ``g`` is set where the sector holds representation ``g`` of the full point group and that level's
        partners lie in more than one solved state (:attr:`.axis_reflection.PointGroupReduction.sector_partner_irreps`);
        ``None`` for the groups of before.  Two degenerate states whose masks share a bit may be partners of one
        level (``Occupations.partner_occupations``).
        """

        held = getattr(self.reduction, "sector_partner_irreps", None)
        if held is None:
            return None
        return tuple(sum(1 << int(g) for g in irreps) for irreps in held)

    @property
    def sector_shares(self) -> np.ndarray | None:
        """The expected share of the states of every unit of irreducible representations (``PARSEC_ORBITAL_IRREPS``,
        :func:`.orbital_groups.irrep_units`), ``None`` for the sectors of before (even shares)."""

        shares = getattr(self.reduction, "sector_shares", None)
        return None if shares is None else np.asarray(shares, dtype=np.float64)

    def sector_projected(self, representation: int) -> bool:
        """Whether sector ``representation`` is a unit that keeps only some representations of its character."""

        projected = getattr(self.reduction, "sector_projected", None)
        return bool(projected is not None and projected[int(representation)])

    @property
    def projected_sectors(self) -> tuple[int, ...]:
        """The units with a projector (:meth:`sector_projected`)."""

        projected = getattr(self.reduction, "sector_projected", None)
        return () if projected is None else tuple(int(index) for index in np.flatnonzero(projected))

    def projection_tables(self, representation: int) -> tuple[np.ndarray, np.ndarray] | None:
        """The projector of a unit onto its irreducible representations of ``G``, on its sector's coefficients.

        ``None`` for a sector solved whole.  Otherwise ``(indices, scales)``, both ``terms x sector_size``: the
        projection of a sector vector ``x`` is ``(P x)[s] = sum_t scales[t, s] * x[indices[t, s]]``.  For the right
        coset representatives ``k`` of ``G = union H k`` (the identity first) with ``b_k != 0``
        (``reduction.sector_projections``), and ``g^-1 = h k``:

        ``P_Gamma f = (levels / |G|) sum_g chi(g) f(g^-1 r)``, so ``(P x)_w = sum_k b_k lambda(h_k) sqrt(|O_w| / |O_w'|)
        x_w'`` where ``k r_w = h_k r_w'`` (``r_w`` the representative of orbit ``w`` of ``H``; ``lambda(h_k)`` is the
        phase of the row ``k r_w`` in the sector, zero where that orbit is not in it).  Built once per unit.
        """

        index = int(representation)
        if not self.sector_projected(index):
            return None
        cached = self._projection_tables.get(index)
        if cached is not None:
            return cached
        reduction = self.reduction
        grid = getattr(reduction, "grid", None)
        if grid is None or getattr(reduction, "coset_operations", None) is None:
            raise RuntimeError("the projector of a unit needs the grid and the cosets of the full group")
        coefficients = np.asarray(reduction.sector_projections[index], dtype=np.float64)
        orbits = self.sector_orbit_indices(index)
        representatives = reduction.representative_rows[orbits]
        mapping = self.orbit_to_sector[index]
        multiplicities = reduction.multiplicities.astype(np.float64)
        translations = getattr(reduction, "coset_translations", None)
        size = orbits.size
        indices: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for coset, coefficient in enumerate(coefficients):
            if abs(coefficient) < 1.0e-14:
                continue
            if coset == 0:
                # the identity coset (orbital_groups: the identity first)
                indices.append(np.arange(size, dtype=np.int64))
                scales.append(np.full(size, coefficient, dtype=np.float64))
                continue
            rows = _grid_row_mapping_operation(
                grid, reduction.coset_operations[coset], 1.0e-10, representatives,
                translation=None if translations is None else translations[coset],
            )
            if rows is None:
                raise RuntimeError("a coset representative of the full group no longer maps the grid")
            images = reduction.full_to_wedge[rows]
            columns = mapping[images]
            phase = self.phases[index, rows].astype(np.float64)
            admitted = columns >= 0
            scale = np.where(
                admitted, coefficient * phase * np.sqrt(multiplicities[orbits] / multiplicities[images]), 0.0
            )
            indices.append(np.where(admitted, columns, 0).astype(np.int64))
            scales.append(scale)
        tables = (np.ascontiguousarray(np.vstack(indices)), np.ascontiguousarray(np.vstack(scales)))
        for array in tables:
            array.setflags(write=False)
        self._projection_tables[index] = tables
        return tables

    def project_vectors(self, values: np.ndarray, representation: int) -> np.ndarray:
        """``P x`` of sector vectors on the host (rows x columns, or one vector): :meth:`projection_tables`."""

        tables = self.projection_tables(representation)
        array = np.asarray(values, dtype=np.float64)
        if tables is None:
            return array.copy()
        indices, scales = tables
        if array.ndim == 1:
            return np.sum(scales * array[indices], axis=0)
        out = np.zeros_like(array)
        for row_indices, row_scales in zip(indices, scales):
            out += row_scales[:, None] * array[row_indices]
        return out

    def sector_rank(self, representation: int) -> int:
        """The dimension of the space a sector solves: its rows, or the trace of a unit's projector."""

        index = int(representation)
        tables = self.projection_tables(index)
        if tables is None:
            return self.sector_size(index)
        indices, scales = tables
        own = np.arange(indices.shape[1], dtype=np.int64)
        return int(round(float(np.sum(np.where(indices == own[None, :], scales, 0.0)))))

    def partner_cosets(self, representation: int) -> tuple[int, ...] | None:
        """The coset positions of the larger subgroup ``K`` on whose orbits unit ``representation`` is solved.

        ``None`` (the rows of its sector of ``H``) unless ``PARSEC_ORBITAL_IRREP_ROWS=partner`` (read at the first
        question and kept for the decomposition's life) and the unit is a
        projected one whose states are a whole one-dimensional sector of a subgroup ``K`` of the full group
        (:attr:`.axis_reflection.PointGroupReduction.sector_partner_cosets`), and unless this decomposition was told
        to keep the sector rows (:meth:`keep_sector_rows`).
        """

        index = int(representation)
        cosets = getattr(self.reduction, "sector_partner_cosets", None)
        if cosets is None or not self.sector_projected(index) or cosets[index] is None:
            return None
        if self._partner_cache.get("kept") is not None:
            return None
        setting = self._partner_cache.get("setting")
        if setting is None:
            # read once: every later question of the run gets the answer of its preparation
            from .axis_reflection import orbital_irrep_rows_setting

            setting = self._partner_cache.setdefault("setting", orbital_irrep_rows_setting())
        if setting != "partner":
            return None
        return tuple(int(c) for c in cosets[index])

    def keep_sector_rows(self, reason: str) -> None:
        """Solve every unit on the rows of its sector of ``H`` from now on (``reason`` is kept for the record)."""

        if self._partner_cache.get("kept") is None:
            self._partner_cache["kept"] = str(reason)

    @property
    def sector_rows_kept(self) -> str | None:
        """Why :meth:`keep_sector_rows` was called, or ``None``."""

        return self._partner_cache.get("kept")

    @property
    def partner_units(self) -> tuple[int, ...]:
        """The units solved on the orbits of a larger subgroup (:meth:`partner_cosets`)."""

        return tuple(index for index in range(self.representation_count) if self.partner_cosets(index) is not None)

    def solved_size(self, representation: int) -> int:
        """The rows on which sector ``representation`` is solved: those of its sector, or for a unit on partner rows
        (:meth:`partner_cosets`) the admitted orbits of ``K``.

        For a unit on partner rows this is the trace of its projector, computed from the orbits that a coset of
        ``K`` maps onto themselves (one row map of their representatives per coset; :meth:`partner_rows` builds the
        rows themselves and gives the same count), so that every MPI rank counts every unit alike without the
        tables of the units it does not solve.
        """

        index = int(representation)
        if self.partner_cosets(index) is None:
            return self.sector_size(index)
        # (zero for a unit whose range is empty on this grid, which irrep_units_refusal refuses)
        return self.projector_trace(index)

    def projector_trace(self, representation: int) -> int:
        """The trace of the projector of unit ``representation`` (:meth:`projection_tables`), the dimension of the
        space it solves; its rows for a sector solved whole.

        Computed from the orbits that a coset with a nonzero coefficient maps onto themselves (one row map of their
        representatives per coset), without the tables, so that every MPI rank counts every unit alike; it equals
        :meth:`sector_rank`.  Zero where the unit's representations have no function on this grid (a coarse grid of
        a large group); an error for a trace that is no integer or negative.
        """

        index = int(representation)
        if not self.sector_projected(index):
            return self.sector_size(index)
        key = ("size", index)
        cached = self._partner_cache.get(key)
        if cached is not None:
            return int(cached)
        reduction = self.reduction
        coefficients = np.asarray(reduction.sector_projections[index], dtype=np.float64)
        orbits = self.sector_orbit_indices(index)
        trace = float(coefficients[0]) * float(orbits.size)
        translations = getattr(reduction, "coset_translations", None)
        for coset in np.flatnonzero(np.abs(coefficients) > 1.0e-12):
            coset = int(coset)
            if coset == 0:
                continue
            fixed = np.flatnonzero(np.asarray(reduction.coset_images[coset])[orbits] == orbits)
            if not fixed.size:
                continue
            rows = _grid_row_mapping_operation(
                reduction.grid, reduction.coset_operations[coset], 1.0e-10,
                reduction.representative_rows[orbits[fixed]],
                translation=None if translations is None else translations[coset],
            )
            if rows is None:
                raise RuntimeError("a coset representative of the full group no longer maps the grid")
            trace += float(coefficients[coset]) * float(np.sum(self.phases[index, rows].astype(np.float64)))
        size = int(round(trace))
        if abs(trace - size) > 1.0e-6 or size < 0:
            raise RuntimeError(f"the projector of unit {index} has no integer trace ({trace!r})")
        self._partner_cache[key] = size
        return size

    @property
    def empty_units(self) -> tuple[int, ...]:
        """The units whose projector has an empty range on this grid (:meth:`projector_trace` zero)."""

        return tuple(index for index in self.projected_sectors if self.projector_trace(index) == 0)

    @property
    def solved_sizes(self) -> tuple[int, ...]:
        """:meth:`solved_size` of every sector."""

        return tuple(self.solved_size(index) for index in range(self.representation_count))

    def partner_rows(self, representation: int) -> "PartnerRows | None":
        """The partner rows of unit ``representation`` (:meth:`partner_cosets`), or ``None``.

        The projector ``P`` of such a unit is ``E E^T``, ``E`` the isometry from the functions of the sector ``mu`` of
        ``K`` to the coefficients of the sector of ``H``: an orbit of ``K`` is the union of the orbits of ``H`` that
        its cosets carry onto each other, and the normalized function of ``mu`` on it has the coefficient ``e_s =
        +-sqrt(|O_s| / |O_K|)`` on each of them (zero on an orbit of ``K`` whose stabilizer ``mu`` does not admit).
        ``E`` is read off ``P`` itself (:meth:`projection_tables`): the blocks of ``P`` on the orbits of ``K`` are
        ``e e^T``, of rank one, which is checked entry by entry, with ``e`` positive at the first row of each.
        """

        index = int(representation)
        cosets = self.partner_cosets(index)
        if cosets is None:
            return None
        key = ("rows", index)
        cached = self._partner_cache.get(key)
        if cached is not None:
            return cached
        from scipy.sparse.csgraph import connected_components

        indices, scales = self.projection_tables(index)
        size = int(indices.shape[1])
        own = np.broadcast_to(np.arange(size, dtype=np.int64)[None, :], indices.shape).reshape(-1)
        matrix = sp.coo_matrix(
            (np.asarray(scales, dtype=np.float64).reshape(-1), (own, np.asarray(indices).reshape(-1))),
            shape=(size, size),
        ).tocsr()
        matrix.sum_duplicates()
        largest = float(np.max(np.abs(matrix.data), initial=0.0))
        matrix.data[np.abs(matrix.data) <= 1.0e-12 * max(largest, 1.0)] = 0.0
        matrix.eliminate_zeros()
        count, labels = connected_components(matrix, directed=True, connection="weak")
        labels = np.asarray(labels, dtype=np.int64)
        first = np.full(count, size, dtype=np.int64)
        np.minimum.at(first, labels, np.arange(size, dtype=np.int64))
        diagonal = matrix.diagonal()
        pivot = diagonal[first]
        admitted = pivot > 1.0e-12
        coo = matrix.tocoo()
        values = np.zeros(size, dtype=np.float64)
        at_first = coo.col == first[labels[coo.row]]
        values[coo.row[at_first]] = coo.data[at_first] / np.sqrt(pivot[labels[coo.row[at_first]]])
        widths = np.bincount(labels, minlength=count)
        entries = np.bincount(coo.row, minlength=size)
        held = admitted[labels]
        if (
            np.any(held & (values == 0.0))
            or np.any(~held & (entries > 0))
            or np.any(held & (entries != widths[labels]))
            or np.max(np.abs(coo.data - values[coo.row] * values[coo.col]), initial=0.0) > 1.0e-11
            or np.any(values[first[admitted]] <= 0.0)
        ):
            raise RuntimeError(
                f"the projector of unit {index} is not that of a sector of a larger subgroup "
                "(PARSEC_ORBITAL_IRREP_ROWS=sector solves it on the rows of its sector)"
            )
        number = np.full(count, -1, dtype=np.int64)
        kept = np.flatnonzero(admitted)
        kept = kept[np.argsort(first[kept], kind="stable")]
        number[kept] = np.arange(kept.size, dtype=np.int64)
        rows = PartnerRows(
            representation=index,
            cosets=cosets,
            first=np.ascontiguousarray(first[kept]),
            row_of_sector=np.ascontiguousarray(number[labels]),
            values=np.ascontiguousarray(values),
        )
        if rows.rows != self.solved_size(index):
            raise RuntimeError(
                f"unit {index} has {rows.rows} partner rows, its projector a trace of {self.solved_size(index)}"
            )
        for array in (rows.first, rows.row_of_sector, rows.values):
            array.setflags(write=False)
        self._partner_cache[key] = rows
        return rows

    def partner_decomposition(self, representation: int) -> "ReflectionRepresentationDecomposition | None":
        """The one sector of ``K`` on which unit ``representation`` is solved, as a decomposition of its own.

        Its reduction is the orbit map of ``K`` (:class:`.axis_reflection.AxisReflectionReduction`: the union of
        the orbits of ``H`` that the cosets of ``K`` carry onto each other, its smallest row the representative,
        the orbits in ascending order of it), and its one sector the partner rows (:meth:`partner_rows`, in their
        order), with the phase of every grid row that of the sector of ``H`` times the sign of ``e`` on its orbit
        of ``H``.  The sector stencil, the projectors and their gradients of the unit are reduced with it as those
        of any sector (``Symmetry/operator_cache.build_partner_operators``); its expansion of a vector is that of
        the sector of ``H`` of ``E x`` (checked by the tests).  Grid-sized arrays: built for the units a process
        solves only.
        """

        index = int(representation)
        rows = self.partner_rows(index)
        if rows is None:
            return None
        key = ("decomposition", index)
        cached = self._partner_cache.get(key)
        if cached is not None:
            return cached
        from scipy.sparse.csgraph import connected_components

        from .axis_reflection import AxisReflectionReduction

        reduction = self.reduction
        wedge = int(reduction.wedge_size)
        images = np.asarray(reduction.coset_images, dtype=np.int64)[list(rows.cosets)]
        sources = np.repeat(np.arange(wedge, dtype=np.int64)[None, :], images.shape[0], axis=0).reshape(-1)
        graph = sp.coo_matrix(
            (np.ones(sources.size, dtype=np.int8), (sources, images.reshape(-1))), shape=(wedge, wedge)
        ).tocsr()
        count, labels = connected_components(graph, directed=True, connection="weak")
        labels = np.asarray(labels, dtype=np.int64)
        smallest = np.full(count, wedge, dtype=np.int64)
        np.minimum.at(smallest, labels, np.arange(wedge, dtype=np.int64))
        order = np.argsort(smallest, kind="stable")
        number = np.empty(count, dtype=np.int64)
        number[order] = np.arange(count, dtype=np.int64)
        orbit_class = number[labels]
        multiplicities = np.bincount(
            orbit_class, weights=np.asarray(reduction.multiplicities, dtype=np.float64), minlength=count
        )
        multiplicities = np.rint(multiplicities).astype(np.int64)
        representative_rows = np.asarray(reduction.representative_rows, dtype=np.int64)[smallest[order]]
        orbits = self.sector_orbit_indices(index)
        held = rows.row_of_sector >= 0
        orbit_to_sector = np.full(count, -1, dtype=np.int64)
        orbit_to_sector[orbit_class[orbits[rows.first]]] = np.arange(rows.rows, dtype=np.int64)
        # Every orbit of H of an admitted orbit of K is a row of the unit's sector, with that partner row.
        members = np.bincount(orbit_class, minlength=count)
        if (
            not np.array_equal(orbit_to_sector[orbit_class[orbits[held]]], rows.row_of_sector[held])
            or not np.array_equal(
                np.bincount(rows.row_of_sector[held], minlength=rows.rows), members[orbit_class[orbits[rows.first]]]
            )
        ):
            raise RuntimeError(f"the partner rows of unit {index} are not the orbits of its subgroup")
        expected = np.sqrt(
            np.asarray(reduction.multiplicities, dtype=np.float64)[orbits[held]]
            / multiplicities[orbit_class[orbits[held]]]
        )
        if np.max(np.abs(np.abs(rows.values[held]) - expected), initial=0.0) > 1.0e-12:
            raise RuntimeError(f"the partner rows of unit {index} do not have the weights of their orbits")
        signs = np.zeros(wedge, dtype=np.int8)
        signs[orbits[held]] = np.where(rows.values[held] > 0.0, 1, -1).astype(np.int8)
        full_to_wedge = np.asarray(reduction.full_to_wedge)
        phases = (np.asarray(self.phases[index], dtype=np.int8) * signs[full_to_wedge]).astype(np.int8)
        subgroup = AxisReflectionReduction(
            signs=np.ones((1, 3), dtype=np.int8),
            representative_rows=np.ascontiguousarray(representative_rows),
            full_to_wedge=np.ascontiguousarray(orbit_class[full_to_wedge]),
            multiplicities=np.ascontiguousarray(multiplicities),
        )
        decomposition = ReflectionRepresentationDecomposition(
            reduction=subgroup,
            characters=np.ones((1, 1), dtype=np.int8),
            phases=np.ascontiguousarray(phases[None, :]),
            orbit_to_sector=np.ascontiguousarray(orbit_to_sector[None, :]),
        )
        admitted_rows = representative_rows[orbit_to_sector >= 0]
        if not np.all(phases[admitted_rows] == 1):
            raise RuntimeError(f"the partner rows of unit {index} have no positive phase at their representatives")
        self._partner_cache[key] = decomposition
        return decomposition

    def partner_expansion(self, representation: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(orbits, rows, values)`` of a unit on partner rows, ``None`` for any other sector: the orbit of ``H``
        of every coefficient of its sector that the unit fills, the partner row that gives it and the value per
        grid point there (``e_s / sqrt(|O_s|)``), so that the orbit values of a partner vector ``x`` are
        ``wedge[orbits] = values * x[rows]`` (the orbitals' export, ``CuPySymmetryOrbitals.to_full_host``)."""

        rows = self.partner_rows(representation)
        if rows is None:
            return None
        key = ("expansion", int(representation))
        cached = self._partner_cache.get(key)
        if cached is not None:
            return cached
        orbits = self.sector_orbit_indices(representation)
        held = np.flatnonzero(rows.row_of_sector >= 0)
        expansion = (
            np.ascontiguousarray(orbits[held]),
            np.ascontiguousarray(rows.row_of_sector[held]),
            np.ascontiguousarray(
                rows.values[held] / np.sqrt(np.asarray(self.reduction.multiplicities, dtype=np.float64)[orbits[held]])
            ),
        )
        for array in expansion:
            array.setflags(write=False)
        self._partner_cache[key] = expansion
        return expansion

    @property
    def wedge_size(self) -> int:
        """Return the scalar-field orbit count shared with Hartree and SCF."""

        return self.reduction.wedge_size

    @property
    def sector_sizes(self) -> tuple[int, ...]:
        """Return the exact orbital dimension of every representation."""

        return tuple(
            int(np.count_nonzero(mapping >= 0))
            for mapping in self.orbit_to_sector
        )

    def sector_size(self, representation: int) -> int:
        """Return one stabilizer-filtered representation dimension."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        return int(np.count_nonzero(self.orbit_to_sector[index] >= 0))

    def sector_orbit_indices(self, representation: int) -> np.ndarray:
        """Map sector columns to the underlying scalar-field orbit indices.

        The list is a function of the immutable sector map alone, and the
        row selection of the operator assembly, the vector transforms and
        the eigensolver ask for it repeatedly.  It is therefore built once
        per representation, on first request; every caller receives the
        same read-only array.
        """

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        orbits = self._sector_orbit_lists.get(index)
        if orbits is None:
            mapping = self.orbit_to_sector[index]
            admitted = np.flatnonzero(mapping >= 0)
            order = np.argsort(mapping[admitted], kind="stable")
            orbits = np.ascontiguousarray(admitted[order], dtype=np.int64)
            orbits.setflags(write=False)
            self._sector_orbit_lists[index] = orbits
        return orbits

    @classmethod
    def build(
        cls,
        grid: RealSpaceGrid,
        reduction: AxisReflectionReduction,
    ) -> "ReflectionRepresentationDecomposition":
        """Build all characters and apply the exact orbit-stabilizer rule."""

        order = reduction.group_order
        if order < 1:
            raise ValueError("orbital decomposition requires a group")
        # A group of the identity alone gives one sector: the whole grid
        # (AxisReflectionReduction.identity, PARSEC_CUPY_SINGLE_SECTOR).
        character_rows: list[np.ndarray] = []
        seen: set[bytes] = set()
        if isinstance(reduction, PointGroupReduction):
            # The real one-dimensional characters of the chosen subgroup that are solved
            # (orbital_groups.orbital_group_plan): fewer than its order where it is not abelian.
            character_rows = list(np.asarray(reduction.sector_characters, dtype=np.int8))
        elif isinstance(reduction, SignedPermutationReduction):
            # For C2^k, chi_p(g_b)=(-1)^(p dot b).  These are all |G|
            # mutually orthogonal real one-dimensional characters.
            rank = int(reduction.generator_bits.shape[1])
            for parity in product((0, 1), repeat=rank):
                exponents = np.asarray(parity, dtype=np.int8)
                character = np.where(
                    (reduction.generator_bits @ exponents) % 2,
                    -1,
                    1,
                ).astype(np.int8)
                character_rows.append(character)
        else:
            # Diagonal sign groups are subgroups of C2^3.  Restricting the
            # PARSEC-ordered product characters preserves D2h labels.
            signs = reduction.signs.astype(np.int8, copy=False)
            for parity in _PARSEC_REFLECTION_PARITIES:
                character = np.ones(order, dtype=np.int8)
                for axis, exponent in enumerate(parity):
                    if exponent:
                        character *= signs[:, axis]
                key = character.tobytes()
                if key not in seen:
                    seen.add(key)
                    character_rows.append(character)
        characters = np.ascontiguousarray(np.vstack(character_rows), dtype=np.int8)
        rows = order if not isinstance(reduction, PointGroupReduction) else characters.shape[0]
        if characters.shape != (rows, order):
            raise RuntimeError("failed to construct the reflection character table")
        # The units of the irreducible representations of the full group (PARSEC_ORBITAL_IRREPS) may share a
        # character of the subgroup: its distinct rows are the ones that must be orthogonal.
        distinct = (
            np.unique(characters, axis=0)
            if getattr(reduction, "sector_projected", None) is not None
            else characters
        )
        gram = distinct.astype(np.int64) @ distinct.astype(np.int64).T
        if not np.array_equal(gram, order * np.eye(distinct.shape[0], dtype=np.int64)):
            raise RuntimeError("reflection characters are not orthogonal")

        representatives = reduction.representative_rows
        # ``images[g,w]`` is the full row obtained by applying operation g to
        # orbit representative w.  Duplicate images expose a stabilizer.
        images = _representative_images(grid, reduction)

        phases = np.zeros(
            (characters.shape[0], reduction.full_size), dtype=np.int8
        )
        orbit_to_sector = np.full(
            (characters.shape[0], reduction.wedge_size), -1, dtype=np.int64
        )
        fast = _fast_maps_requested()
        if fast:
            # An image counts where no earlier operation gave the same row:
            # the number of different images, without sorting every column.
            unique_image_counts = np.ones(reduction.wedge_size, dtype=np.int64)
            for operation in range(1, order):
                repeated = images[operation] == images[0]
                for earlier in range(1, operation):
                    repeated |= images[operation] == images[earlier]
                unique_image_counts += ~repeated
        else:
            sorted_images = np.sort(images, axis=0)
            unique_image_counts = 1 + np.count_nonzero(
                np.diff(sorted_images, axis=0), axis=0
            )
        if not np.array_equal(
            unique_image_counts, reduction.multiplicities
        ):
            raise RuntimeError("reflection operations do not reproduce every orbit")
        stabilizer = images == representatives[None, :]
        # Where every image lies in the orbit of its representative, a row
        # is reached from one orbit only, and the last operation that
        # reaches it is the same for every representation: one label per row
        # replaces a phase scatter per representation and operation.  The
        # images of an orbit are then as many different rows as its
        # multiplicity, so multiplicities that add up to the grid leave no
        # row without a label.
        row_operation = None
        orbit_numbers = np.arange(reduction.wedge_size, dtype=np.int64)
        if (
            fast
            and int(np.sum(reduction.multiplicities)) == reduction.full_size
            and all(
                np.array_equal(
                    reduction.full_to_wedge[images[operation]], orbit_numbers
                )
                for operation in range(order)
            )
        ):
            row_operation = np.empty(reduction.full_size, dtype=np.intp)
            for operation in range(order):
                row_operation[images[operation]] = operation
        # Only the identity has every character +1.  Where no other operation
        # fixes a representative, every representation admits every orbit.
        free = (
            row_operation is not None
            and reduction.wedge_size > 0
            and not np.any(stabilizer[np.any(characters != 1, axis=0)])
        )
        for representation, character in enumerate(characters):
            if free:
                orbit_to_sector[representation] = orbit_numbers
                phases[representation] = character[row_operation]
                continue
            # chi is trivial on S_w iff every operation fixing the selected
            # representative has character +1.  This is equivalent to the
            # former duplicate-image test, evaluated for every orbit at once.
            admitted = np.all(
                (~stabilizer) | (character[:, None] == 1), axis=0
            )
            admitted_orbits = np.flatnonzero(admitted)
            if admitted_orbits.size < 1:
                raise RuntimeError("a reflection representation has zero dimension")
            orbit_to_sector[representation, admitted_orbits] = np.arange(
                admitted_orbits.size, dtype=np.int64
            )
            if row_operation is not None:
                phases[representation] = character[row_operation]
                if admitted_orbits.size != admitted.size:
                    # Rows of a rejected orbit keep phase zero.
                    phases[representation, ~admitted[reduction.full_to_wedge]] = 0
                continue
            for operation in range(order):
                phases[
                    representation,
                    images[operation, admitted_orbits],
                ] = character[operation]

        # Identity maps every admitted representative to itself.  Rejected
        # representative rows deliberately carry phase zero.
        for representation in range(characters.shape[0]):
            admitted = orbit_to_sector[representation] >= 0
            if not np.all(phases[representation, representatives[admitted]] == 1):
                raise RuntimeError("invalid representation phase convention")
        return cls(
            reduction=reduction,
            characters=characters,
            phases=np.ascontiguousarray(phases),
            orbit_to_sector=np.ascontiguousarray(orbit_to_sector),
        )

    def reduce_operator(
        self,
        operator: sp.spmatrix,
        representation: int,
    ) -> sp.csr_matrix:
        """Construct ``U_Gamma.T A U_Gamma`` from representative rows."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        matrix = sp.csr_matrix(operator, dtype=np.float64)
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError("operator does not match the representation grid")
        matrix.sum_duplicates()
        matrix.sort_indices()
        selected = matrix[
            self.reduction.representative_rows[
                self.sector_orbit_indices(index)
            ],
            :,
        ].tocoo(copy=False)
        return self._assemble_reduced_operator(selected, index)

    def _selected_representations(
        self,
        representations: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if representations is None:
            return tuple(range(self.representation_count))
        selected = tuple(sorted({int(index) for index in representations}))
        if not selected or selected[0] < 0 or selected[-1] >= self.representation_count:
            raise IndexError("representation index is outside the character table")
        return selected

    def reduce_operators(
        self,
        operator: sp.spmatrix,
        representations: Sequence[int] | None = None,
    ) -> tuple[sp.csr_matrix | None, ...]:
        """Construct every ``U_Gamma.T A U_Gamma`` in one sparse pass.

        The full operator is canonicalized and its representative rows are
        gathered only once.  Each representation differs solely by the
        character phase multiplying the gathered column.  This is exactly
        the same projection used by :meth:`reduce_operator`; the batched
        route only avoids repeating the expensive CSR slicing and COO
        conversion once per representation.

        ``representations`` restricts assembly to those sectors; the result
        keeps one slot per representation and leaves the others ``None``.
        A process that owns only some sectors need not assemble the rest.
        """

        wanted = self._selected_representations(representations)
        matrix = sp.csr_matrix(operator, dtype=np.float64)
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError("operator does not match the representation grid")
        matrix.sum_duplicates()
        matrix.sort_indices()
        # Gather every scalar-orbit representative row exactly once.  The
        # earlier implementation performed this large CSR slice and COO
        # conversion independently for every representation even though the
        # source rows differ only by stabilizer admission.  ``selected.row``
        # is now the scalar-orbit index; the exact representation row map is
        # applied inside ``_assemble_reduced_operator``.
        if os.environ.get("PARSEC_NATIVE_SECTOR_ASSEMBLY", "0") == "1":
            from ..backends.native import _load_native

            native = _load_native()
            if not hasattr(native, "reduce_sector_csr"):
                raise RuntimeError("PARSEC_NATIVE_SECTOR_ASSEMBLY requires rebuilding the native extension")
            sectors: list[sp.csr_matrix | None] = [None] * self.representation_count
            for index in wanted:
                payload = native.reduce_sector_csr(
                    matrix.indptr, matrix.indices, matrix.data,
                    self.reduction.representative_rows[self.sector_orbit_indices(index)],
                    self.reduction.full_to_wedge, self.orbit_to_sector[index],
                    self.reduction.multiplicities, self.phases[index],
                )
                size = self.sector_size(index)
                reduced = sp.csr_matrix((payload["data"], payload["indices"], payload["indptr"]),
                                        shape=(size, size))
                asymmetry = reduced - reduced.T
                maximum = float(np.max(np.abs(asymmetry.data), initial=0.0))
                scale = float(np.max(np.abs(reduced.data), initial=1.0))
                if maximum > 5.0e-13 * scale:
                    raise ValueError(f"representation operator is not symmetric: maximum asymmetry {maximum:.3e}")
                sectors[index] = reduced
            return tuple(sectors)

        selected = matrix[
            self.reduction.representative_rows,
            :,
        ].tocoo(copy=False)
        workers = operator_build_workers(len(wanted))

        def assemble(index: int) -> sp.csr_matrix:
            return self._assemble_reduced_operator(
                selected,
                index,
                rows_are_scalar_orbits=True,
            )

        assembled: list[sp.csr_matrix | None] = [None] * self.representation_count
        if workers == 1:
            for index in wanted:
                assembled[index] = assemble(index)
            return tuple(assembled)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="parsec-representation-build",
        ) as executor:
            for index, reduced in zip(wanted, executor.map(assemble, wanted)):
                assembled[index] = reduced
        return tuple(assembled)

    def reduce_hermitian_operators(
        self,
        operator: sp.spmatrix,
        representations: Sequence[int] | None = None,
    ) -> tuple[sp.csr_matrix | None, ...]:
        """``U_Gamma.T A U_Gamma`` of a complex Hermitian ``A`` that commutes with the operations.

        The basis ``U_Gamma`` of a sector is real (the characters are +-1), so the projection of a complex operator
        is that of :meth:`reduce_operators` on its complex entries: the same rows, phases and normalization, a
        Hermitian sector matrix (checked).  The operators of a k-point (:mod:`..SCF.kpoint_sectors`).
        """

        wanted = self._selected_representations(representations)
        matrix = sp.csr_matrix(operator, dtype=np.complex128)
        if matrix.shape != (self.full_size, self.full_size):
            raise ValueError("operator does not match the representation grid")
        matrix.sum_duplicates()
        matrix.sort_indices()
        selected = matrix[self.reduction.representative_rows, :].tocoo(copy=False)
        sectors: list[sp.csr_matrix | None] = [None] * self.representation_count
        for index in wanted:
            row_orbits = selected.row
            sector_rows = self.orbit_to_sector[index, row_orbits]
            column_orbits = self.reduction.full_to_wedge[selected.col]
            sector_columns = self.orbit_to_sector[index, column_orbits]
            keep = (sector_rows >= 0) & (sector_columns >= 0)
            normalization = np.sqrt(
                self.reduction.multiplicities[row_orbits[keep]] / self.reduction.multiplicities[column_orbits[keep]]
            )
            data = selected.data[keep] * self.phases[index, selected.col[keep]] * normalization
            size = self.sector_size(index)
            reduced = sp.coo_matrix((data, (sector_rows[keep], sector_columns[keep])), shape=(size, size)).tocsr()
            reduced.sum_duplicates()
            reduced.eliminate_zeros()
            reduced.sort_indices()
            difference = reduced - reduced.conj().T
            maximum = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
            scale = float(np.max(np.abs(reduced.data), initial=1.0))
            if maximum > 5.0e-13 * scale:
                raise ValueError(f"representation operator is not Hermitian: maximum asymmetry {maximum:.3e}")
            sectors[index] = reduced
        return tuple(sectors)

    def _assemble_reduced_operator(
        self,
        selected: sp.coo_matrix,
        representation: int,
        *,
        rows_are_scalar_orbits: bool = False,
    ) -> sp.csr_matrix:
        """Assemble one reduced matrix from already-selected wedge rows."""

        index = int(representation)
        if rows_are_scalar_orbits:
            # The sector map answers this route by itself, so a process that
            # assembles a sector it does not solve builds no orbit list for it.
            row_orbits = selected.row
            sector_rows = self.orbit_to_sector[index, row_orbits]
        else:
            row_orbits = self.sector_orbit_indices(index)[selected.row]
            sector_rows = selected.row
        column_orbits = self.reduction.full_to_wedge[selected.col]
        sector_columns = self.orbit_to_sector[index, column_orbits]
        keep = (sector_rows >= 0) & (sector_columns >= 0)
        if not np.any(keep):
            raise RuntimeError("representation projection produced an empty operator")
        kept_row_orbits = row_orbits[keep]
        kept_column_orbits = column_orbits[keep]
        normalization = np.sqrt(
            self.reduction.multiplicities[kept_row_orbits]
            / self.reduction.multiplicities[kept_column_orbits]
        )
        data = (
            selected.data[keep]
            * self.phases[index, selected.col[keep]]
            * normalization
        )
        sector_size = self.sector_size(index)
        reduced = sp.coo_matrix(
            (
                data,
                (sector_rows[keep], sector_columns[keep]),
            ),
            shape=(sector_size, sector_size),
        ).tocsr()
        reduced.sum_duplicates()
        reduced.eliminate_zeros()
        reduced.sort_indices()
        asymmetry = reduced - reduced.T
        maximum = float(np.max(np.abs(asymmetry.data))) if asymmetry.nnz else 0.0
        scale = float(np.max(np.abs(reduced.data), initial=1.0))
        if maximum > 5.0e-13 * scale:
            raise ValueError(
                "representation operator is not symmetric: "
                f"maximum asymmetry {maximum:.3e}"
            )
        return reduced

    def reduce_nonlocal_operator(
        self,
        operator: NonlocalProjectorOperator,
        representation: int,
    ) -> NonlocalProjectorOperator:
        """Project KB factors as ``B_Gamma=U_Gamma.T B``."""

        index = int(representation)
        if not 0 <= index < self.representation_count:
            raise IndexError("representation index is outside the character table")
        projectors = sp.coo_matrix(operator.projectors, dtype=np.float64)
        return self._assemble_reduced_nonlocal(operator, projectors, index)

    def reduce_nonlocal_operators(
        self,
        operator: NonlocalProjectorOperator,
        representations: Sequence[int] | None = None,
    ) -> tuple[NonlocalProjectorOperator | None, ...]:
        """Construct every ``U_Gamma.T B`` from one canonical COO buffer.

        PARSEC's Kleinman--Bylander operator remains ``B D B.T`` in each
        representation.  Only the basis transformation of ``B`` is batched;
        projector coefficients, signs, labels, and summation order are kept.
        ``representations`` restricts assembly as in :meth:`reduce_operators`.
        """

        wanted = self._selected_representations(representations)
        # (complex where the projectors are those of a k-point: the real basis of a sector keeps their phases)
        projectors = sp.coo_matrix(
            operator.projectors, dtype=np.result_type(operator.projectors.dtype, np.float64)
        )
        projectors.sum_duplicates()
        workers = operator_build_workers(len(wanted))

        def assemble(index: int) -> NonlocalProjectorOperator:
            return self._assemble_reduced_nonlocal(operator, projectors, index)

        assembled: list[NonlocalProjectorOperator | None] = [None] * self.representation_count
        if workers == 1:
            for index in wanted:
                assembled[index] = assemble(index)
            return tuple(assembled)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="parsec-projector-build",
        ) as executor:
            for index, reduced in zip(wanted, executor.map(assemble, wanted)):
                assembled[index] = reduced
        return tuple(assembled)

    def _assemble_reduced_nonlocal(
        self,
        operator: NonlocalProjectorOperator,
        projectors: sp.coo_matrix,
        representation: int,
    ) -> NonlocalProjectorOperator:
        """Assemble one reduced projector matrix from shared COO buffers."""

        index = int(representation)
        row_orbits = self.reduction.full_to_wedge[projectors.row]
        sector_rows = self.orbit_to_sector[index, row_orbits]
        keep = sector_rows >= 0
        scale = 1.0 / np.sqrt(self.reduction.multiplicities[row_orbits[keep]])
        reduced = sp.coo_matrix(
            (
                projectors.data[keep]
                * self.phases[index, projectors.row[keep]]
                * scale,
                (
                    sector_rows[keep],
                    projectors.col[keep],
                ),
            ),
            shape=(self.sector_size(index), projectors.shape[1]),
        ).tocsc()
        reduced.sum_duplicates()
        reduced.eliminate_zeros()
        reduced.sort_indices()
        return NonlocalProjectorOperator(
            projectors=reduced,
            signs=np.asarray(operator.signs, dtype=np.float64).copy(),
            labels=tuple(operator.labels),
        )

    def invariant_wedge_values(self, values: np.ndarray) -> np.ndarray:
        """Orbit-average a symmetry-invariant local scalar field."""

        reduced = self.reduction.reduce_vector(values)
        return reduced / np.sqrt(self.reduction.multiplicities)

    def reduce_vector(
        self,
        values: np.ndarray,
        representation: int,
    ) -> np.ndarray:
        """Apply ``U_Gamma.T`` to one full-grid host vector."""

        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.full_size,):
            raise ValueError("full vector does not match the representation grid")
        weights = self.phases[int(representation)] * array
        sums = np.bincount(
            self.reduction.full_to_wedge,
            weights=weights,
            minlength=self.wedge_size,
        )
        sector_orbits = self.sector_orbit_indices(representation)
        return sums[sector_orbits] / np.sqrt(
            self.reduction.multiplicities[sector_orbits]
        )

    def expand_vector(
        self,
        values: np.ndarray,
        representation: int,
    ) -> np.ndarray:
        """Apply ``U_Gamma`` to one wedge vector on the host."""

        array = np.asarray(values, dtype=np.float64)
        index = int(representation)
        sector_orbits = self.sector_orbit_indices(index)
        if array.shape != (sector_orbits.size,):
            raise ValueError("sector vector does not match the representation")
        base_wedge = np.zeros(self.wedge_size, dtype=np.float64)
        base_wedge[sector_orbits] = array / np.sqrt(
            self.reduction.multiplicities[sector_orbits]
        )
        return np.ascontiguousarray(
            self.phases[index] * base_wedge[self.reduction.full_to_wedge]
        )


__all__ = ["PartnerRows", "ReflectionRepresentationDecomposition", "operator_build_workers"]
