"""The scalar SCF fields of weighted sectors on the orbits of the full point group.

Weighted sectors (:class:`.axis_reflection.PointGroupReduction`) solve the one-dimensional sectors of a subgroup
``H`` of the point group ``G`` of the structure and count the partners of the levels of ``G`` with weights.  Their
density is averaged over ``G`` every step, so the density and every potential built from it are invariant under
``G``; until 2026-11-19 they were nevertheless kept with one value per orbit of ``H``.  A ``G``-orbit is the union
of the ``H``-orbits ``coset_images[c, w]`` over the right coset representatives ``c``, so its value is that of any of
them, and up to ``[G:H]`` (Td over D2d: 3; the 192 operations of a diamond supercell over 32: 6) times fewer
values carry the same field (fewer where orbits on axes and planes are smaller: about 2.9-3 for the nanodiamonds,
4.4-5.4 for diamond supercells).

:class:`FieldOrbitReduction` is the orbit map of ``G`` in the form of the other reductions (``representative_rows``,
``full_to_wedge``, ``multiplicities``, ``operations``), so the orbit algebra of the SCF loop
(:class:`..SCF.symmetry_fields.SymmetrySCFReducer`), the Hartree solve on the wedge, the boundary builders and the XC
evaluators on orbits take it as they take ``H``.  It keeps the subgroup and the map from the orbits of ``H`` to its
own, which the sector eigensolver (its local potential on the orbits of ``H``) and the sector density (summed on
the orbits of ``H``) cross with :meth:`FieldOrbitReduction.to_subgroup` and :meth:`FieldOrbitReduction.from_subgroup`.

``PARSEC_SECTOR_FIELD_GROUP`` (:func:`sector_field_group_setting`) chooses the group of the fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import os

import numpy as np
import scipy.sparse as sp

from .axis_reflection import SignedPermutationReduction

_SWITCH = "PARSEC_SECTOR_FIELD_GROUP"
_CACHE_ATTRIBUTE = "_field_orbit_reduction"


def sector_field_group_setting() -> str:
    """``PARSEC_SECTOR_FIELD_GROUP``: ``full`` (the default since 2026-11-19) or ``subgroup``.

    ``full``: the scalar fields of weighted sectors (density, Hartree and XC potentials, the mixer, the residual
    norms and the energy integrals) keep one value per orbit of the full point group ``G``
    (:class:`FieldOrbitReduction`), and the Hartree conjugate gradients run on that wedge.  ``subgroup``: one value
    per orbit of the solved subgroup ``H``, as before.  The former bits need ``PARSEC_SECTOR_INPUT_GROUP_AVERAGE=off``
    as well (its default ``auto`` averages the full-grid inputs over ``G``, a round-off change), and where they apply
    ``PARSEC_IONIC_BACKEND=native``, ``PARSEC_IONIC_SUM_GROUP=first``, ``PARSEC_IONIC_MOVED_SUMS=full`` and
    ``PARSEC_HARTREE_BOUNDARY_ROWS=diagonal``.  Only weighted sectors with more than one coset
    (Td over D2d, Oh over D4h, the space groups of supercells) are concerned; abelian sectors, k-point sectors,
    spin channels and the full grid keep their orbits.
    """

    value = os.environ.get(_SWITCH, "full").strip().lower() or "full"
    if value not in {"full", "subgroup"}:
        raise ValueError(f"{_SWITCH} must be full or subgroup, not {value!r}")
    return value


@dataclass(frozen=True)
class FieldOrbitReduction(SignedPermutationReduction):
    """One value per orbit of the full group ``G`` of a :class:`.axis_reflection.PointGroupReduction`.

    ``operations`` are the signed permutations of ``G`` (``full_operations`` of the subgroup reduction) and
    ``translations`` theirs (``None`` for the point group about the origin); ``signs`` and ``generator_bits`` are
    placeholders of the order of ``G``.  ``subgroup`` is the reduction of the solved subgroup ``H`` and
    ``subgroup_to_field[w]`` the orbit of ``G`` that holds orbit ``w`` of ``H``.  Representatives are the smallest
    rows of the orbits, in ascending order, as for every reduction.
    """

    subgroup: object = field(default=None, compare=False, repr=False)
    subgroup_to_field: np.ndarray = field(default=None, compare=False, repr=False)
    translations: np.ndarray = None

    @cached_property
    def subgroup_weights(self) -> np.ndarray:
        """The multiplicities of the orbits of ``H`` as float64, read-only."""

        weights = np.array(self.subgroup.multiplicities, dtype=np.float64)
        weights.setflags(write=False)
        return weights

    def from_subgroup(self, values) -> np.ndarray:
        """The values per orbit of ``G`` of a field given per orbit of ``H``: the average over each orbit of ``G``.

        For a field invariant under ``H`` this is its average over ``G`` (each orbit of ``H`` weighted by its points).
        """

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (int(self.subgroup.wedge_size),):
            raise ValueError("the field does not match the orbits of the subgroup")
        sums = np.bincount(
            self.subgroup_to_field, weights=self.subgroup_weights * values, minlength=self.wedge_size
        )
        return sums / self.multiplicities

    def to_subgroup(self, values) -> np.ndarray:
        """The values per orbit of ``H`` of a field given per orbit of ``G`` (every orbit of ``H`` takes that of its
        orbit of ``G``)."""

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.wedge_size,):
            raise ValueError("the field does not match the orbits of the full group")
        return np.ascontiguousarray(values[self.subgroup_to_field])


def _classes(images: np.ndarray, size: int) -> np.ndarray:
    """The class of every orbit of ``H`` under the relation ``w ~ images[c, w]``: the connected components."""

    from scipy.sparse.csgraph import connected_components

    count = int(images.shape[0])
    rows = np.repeat(np.arange(size, dtype=np.int64)[None, :], count, axis=0).reshape(-1)
    graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, np.asarray(images, dtype=np.int64).reshape(-1))),
        shape=(size, size),
    ).tocsr()
    _, labels = connected_components(graph, directed=True, connection="weak")
    return np.asarray(labels, dtype=np.int64)


def build_field_orbit_reduction(reduction) -> FieldOrbitReduction | None:
    """The :class:`FieldOrbitReduction` of a :class:`.axis_reflection.PointGroupReduction`, ``None`` where it has
    no coset beside ``H`` (the orbits of ``G`` are those of ``H``)."""

    images = getattr(reduction, "coset_images", None)
    operations = getattr(reduction, "full_operations", None)
    if images is None or operations is None or int(np.asarray(images).shape[0]) <= 1:
        return None
    images = np.asarray(images, dtype=np.int64)
    size = int(reduction.wedge_size)
    if images.ndim != 2 or images.shape[1] != size:
        raise ValueError("the coset images do not match the orbits of the subgroup")
    labels = _classes(images, size)
    # Each class by its smallest orbit of H, whose representative is the smallest row of the class (the
    # representatives of H are the smallest rows of their orbits, in ascending order).
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(first, kind="stable")
    number = np.empty(order.size, dtype=np.int64)
    number[order] = np.arange(order.size, dtype=np.int64)
    subgroup_to_field = np.ascontiguousarray(number[labels])
    representatives = np.ascontiguousarray(
        np.asarray(reduction.representative_rows, dtype=np.int64)[np.sort(first)]
    )
    multiplicities = np.bincount(
        subgroup_to_field, weights=np.asarray(reduction.multiplicities, dtype=np.float64), minlength=order.size
    )
    multiplicities = np.rint(multiplicities).astype(np.int64)
    group_order = int(np.asarray(operations).shape[0])
    if np.any(multiplicities <= 0) or np.any(group_order % multiplicities):
        raise RuntimeError("an orbit of the full group does not divide its order")
    full_to_wedge = np.ascontiguousarray(subgroup_to_field[np.asarray(reduction.full_to_wedge, dtype=np.int64)])
    translations = getattr(reduction, "full_translations", None)
    return FieldOrbitReduction(
        signs=np.ones((group_order, 3), dtype=np.int8),
        representative_rows=representatives,
        full_to_wedge=full_to_wedge,
        multiplicities=multiplicities,
        operations=np.ascontiguousarray(operations, dtype=np.int8),
        generator_bits=np.zeros((group_order, 0), dtype=np.int8),
        subgroup=reduction,
        subgroup_to_field=subgroup_to_field,
        translations=None if translations is None else np.ascontiguousarray(translations, dtype=np.int64),
    )


def field_orbit_reduction(reduction) -> FieldOrbitReduction | None:
    """:func:`build_field_orbit_reduction`, built once per reduction (kept on the reduction itself)."""

    cached = getattr(reduction, _CACHE_ATTRIBUTE, None)
    if cached is not None:
        return cached
    built = build_field_orbit_reduction(reduction)
    if built is not None:
        object.__setattr__(reduction, _CACHE_ATTRIBUTE, built)
    return built


def _phases_of_one(size: int) -> np.ndarray:
    return np.ones(size, dtype=np.int8)


def field_negative_laplacian(grid, reduction: FieldOrbitReduction):
    """The normalized ``U^T (-nabla_FD^2) U`` on the orbits of ``reduction``, from the grid.

    ``U`` has one column ``e_O / sqrt(m_O)`` per orbit ``O``; row ``w`` is the stencil of the representative point
    of orbit ``w`` with every neighbour mapped to its orbit ``v`` and multiplied by ``sqrt(m_w / m_v)``
    (:meth:`.axis_reflection.AxisReflectionReduction.reduce_operator` of the full-grid matrix, without building
    it).  A cluster grid of one spacing takes the C++ kernel of the sector stencils where the extension has it and
    the palette of the stencil holds its coefficients (a :class:`..backends.cupy_stencil_major.StencilMajorHostMetadata`);
    otherwise the CSR matrix is built here with NumPy.
    """

    from .sector_stencil import (
        SectorStencilUnavailable,
        _check_symmetric,
        _padded_lookup,
        _stencil_columns,
        _unequal_steps,
        axis_stencil_values,
        native_kernel_available,
    )

    from .axis_reflection import _lattice_period

    if int(reduction.full_size) != int(grid.size):
        raise ValueError("the field orbits do not match the grid")
    if _lattice_period(grid) is not None:
        # (the Hartree problem of a periodic cell is solved on its full grid by its own solver)
        raise ValueError("the wedge operator of the field orbits is built for a cluster grid")
    if not _unequal_steps(grid) and native_kernel_available():
        from ..backends.cupy_stencil_major import PaletteOverflow, StencilMajorHostMetadata
        from ..backends.native import _load_native

        size = int(reduction.wedge_size)
        try:
            payload = _load_native().build_sector_stencil(
                grid.integer_coordinates,
                grid.index_min,
                grid.lookup,
                int(grid.settings.expansion_order),
                float(grid.spacing),
                np.ascontiguousarray(reduction.representative_rows, dtype=np.int64),
                np.ascontiguousarray(reduction.full_to_wedge, dtype=np.int64),
                np.arange(size, dtype=np.int64),
                np.ascontiguousarray(reduction.multiplicities, dtype=np.int64),
                _phases_of_one(int(grid.size)),
            )
            _check_symmetric(float(payload["maximum_asymmetry"]), float(payload["scale"]))
            return StencilMajorHostMetadata(
                shape=(size, size),
                neighbors=payload["neighbors"],
                coefficient_codes=payload["codes"],
                coefficient_palette=payload["palette"],
            )
        except (PaletteOverflow, SectorStencilUnavailable, RuntimeError, ValueError):
            # More coefficients than the byte codes of the packed stencil hold: the CSR below, which the GPU CG
            # packs with two-byte codes.
            pass
    width = int(grid.settings.expansion_order) // 2
    diagonal, axis_values = axis_stencil_values(grid)
    slot_values = [diagonal]
    for axis in range(3):
        for shell in range(-width, width + 1):
            if shell:
                slot_values.append(axis_values[axis][abs(shell) - 1])
    slot_values = np.asarray(slot_values, dtype=np.float64)
    columns_of = _stencil_columns(grid, width, _padded_lookup(grid, width))
    rows = np.asarray(reduction.representative_rows, dtype=np.int64)
    full_to_wedge = np.asarray(reduction.full_to_wedge, dtype=np.int64)
    roots = np.sqrt(np.asarray(reduction.multiplicities, dtype=np.float64))
    size = int(rows.size)
    pieces_rows, pieces_columns, pieces_values = [], [], []
    block = 1 << 15
    for start in range(0, size, block):
        stop = min(start + block, size)
        columns = columns_of(rows[start:stop])
        valid = columns >= 0
        targets = full_to_wedge[np.where(valid, columns, 0)]
        local = np.broadcast_to(np.arange(start, stop, dtype=np.int64)[:, None], columns.shape)
        values = np.broadcast_to(slot_values[None, :], columns.shape)
        pieces_rows.append(local[valid])
        pieces_columns.append(targets[valid])
        pieces_values.append(values[valid])
    row_index = np.concatenate(pieces_rows)
    column_index = np.concatenate(pieces_columns)
    del pieces_rows, pieces_columns
    reduced = sp.csr_matrix(
        (np.concatenate(pieces_values), (row_index, column_index)), shape=(size, size)
    )
    del row_index, column_index, pieces_values
    # The entries of one orbit added first, then normalized: the order of ``reduce_operator``.
    reduced.sum_duplicates()
    row_of_entry = np.repeat(np.arange(size, dtype=np.int64), np.diff(reduced.indptr))
    reduced.data *= roots[row_of_entry] / roots[reduced.indices]
    del row_of_entry
    reduced.eliminate_zeros()
    reduced.sort_indices()
    # The audit of a reduced operator (symmetric for an operator that commutes with the group), by two random
    # vectors instead of the transpose: |x.Ay - y.Ax| against the scale of the products.
    generator = np.random.default_rng(20261119)
    left, right = generator.standard_normal(size), generator.standard_normal(size)
    forward, backward = float(left @ (reduced @ right)), float(right @ (reduced @ left))
    scale = float(np.max(np.abs(reduced.data), initial=1.0)) * float(np.linalg.norm(left) * np.linalg.norm(right))
    _check_symmetric(abs(forward - backward), scale)
    return reduced


__all__ = [
    "FieldOrbitReduction",
    "build_field_orbit_reduction",
    "field_negative_laplacian",
    "field_orbit_reduction",
    "sector_field_group_setting",
]
