"""``Symmetrize_Charge_Density`` on the orbits of the symmetry sectors.

``symm_scalar`` of ``newrho.F90`` (lines 195-351) averages the density of
every SCF iteration over the full point group ``G`` of the atoms
(:mod:`parsec_python.Grid.point_group`):

``f_sym(r) = (1/|G|) sum_g f(g r)``.

An accelerated SCF loop with symmetry sectors keeps one value per orbit of
a subgroup ``H`` of ``G``, the reflections or commuting signed permutations
of its sectors (:mod:`parsec_python.acceleration.Symmetry`), and a field
that ``H`` leaves invariant has the value ``f_w`` at every point of orbit
``w``.  At the representative row ``r_w`` of the orbit the average is

``f_sym(w) = (1/|G|) sum_g f[orbit(g r_w)]``,

the value the reference loop computes at that row from the expanded field,
term for term and in the same order of the operations.  The average is
invariant under ``G`` and so under ``H``: its orbit values are the whole
field, and no full-grid array is formed in the loop.  The operations of
``G`` that are not in ``H`` (the threefold axes of a tetrahedral cluster,
whose sectors use its ``D_2`` subgroup) are those the average adds.

The report is that of :func:`parsec_python.Grid.symmetrize_density`: the
largest change of a point (the largest over the orbits), relative to the
sum of the field over PARSEC's wedge, and the charges before and after,
whose full-grid sums are taken with the orbit multiplicities (equal to the
reference sums but for their rounding).

A periodic cell (``PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto``, since
2026-11-19) averages over the space group of the parser
(:class:`parsec_python.Grid.space_group.PeriodicGridSymmetry`: ``symm_scalar``
with the fractional translations of the Fortran-derived routines, the
identity where PARSEC keeps none), whose operations need not contain those
of the sectors (the translations of a supercell, a centre off the origin):
the average of a field the sectors leave invariant is constant on their
orbits wherever the sectors' operations map the operations of the space
group onto operations of it, which the build checks on the rows of a sample
(:func:`build_wedge_density_symmetry`) and refuses otherwise.  The spin
channels of a cluster in sectors are averaged alike, each on its orbits.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock

import numpy as np

from parsec_python.Grid.point_group import (
    GridPointGroup,
    ScalarSymmetrization,
    symmetrize_density,
)

from .symmetry_fields import SymmetryScalarField

SYMMETRIZED_DENSITY_SECTORS_SWITCH = "PARSEC_SYMMETRIZED_DENSITY_SECTORS"

#: Rows on which the average of a periodic group is checked to be constant on the orbits of the sectors.
_PERIODIC_CHECK_ROWS = 4096


def symmetrized_density_sectors_setting() -> str:
    """``PARSEC_SYMMETRIZED_DENSITY_SECTORS``: ``auto`` (the default) or ``off``.

    ``Symmetrize_Charge_Density: .true.`` on the accelerated driver: ``auto`` keeps the symmetry sectors of a bulk
    cell at the Gamma point (the density of every iteration averaged over the space group of the parser on the orbits
    of the sectors, this module) and of a spin-polarized cluster (each channel averaged on the orbits); ``off`` runs
    such a cell with the reference solver on the full grid and the two channels of such a cluster on the full grid,
    as before 2026-11-19.
    """

    value = os.environ.get(SYMMETRIZED_DENSITY_SECTORS_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{SYMMETRIZED_DENSITY_SECTORS_SWITCH} must be auto or off, not {value!r}")
    return value


# Groups built for the reduction of a prepared system, by identity of both:
# the SCF loop asks for its group once per iteration.
_CACHE: dict[tuple[int, int], tuple[object, object, "WedgeDensitySymmetry"]] = {}
_CACHE_LOCK = Lock()
_CACHE_SIZE = 4


def _subgroup_matrices(reduction) -> np.ndarray:
    """The ``(|H|, 3, 3)`` matrices of the operations of a sector reduction."""

    operations = getattr(reduction, "operations", None)
    if operations is not None:
        return np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)
    signs = np.asarray(reduction.signs, dtype=np.float64).reshape(-1, 3)
    return np.einsum("gi,ij->gij", signs, np.eye(3))


@dataclass(frozen=True)
class WedgeDensitySymmetry:
    """The point group of ``Symmetrize_Charge_Density`` on sector orbits.

    ``group`` is the reference group (its matrices and full-grid images);
    ``reduction`` the orbit map of the sectors; ``orbit_images[g, w]`` the
    orbit of the image of the representative row of orbit ``w`` under
    operation ``g`` of ``group``.
    """

    group: GridPointGroup
    reduction: object
    orbit_images: np.ndarray

    @property
    def order(self) -> int:
        return self.group.order

    @property
    def matrices(self) -> np.ndarray:
        return self.group.matrices

    @property
    def reduced_order(self) -> int:
        return self.group.reduced_order

    def symmetrize_values(self, values: np.ndarray) -> np.ndarray:
        """``(1/|G|) sum_g f[orbit(g r_w)]`` of the orbit values ``values``."""

        values = np.asarray(values, dtype=np.float64)
        if self.order <= 1:
            return values.copy()
        total = np.zeros_like(values)
        for images in self.orbit_images:
            total += values[images]
        return total / float(self.order)

    def symmetrize_density(self, density, volume_element: float):
        """Symmetrize one density of the SCF loop and report it as ``symm_scalar`` does.

        ``density`` is a :class:`SymmetryScalarField` of this reduction; a
        full-grid array is symmetrized by the reference function.
        """

        if not isinstance(density, SymmetryScalarField):
            return symmetrize_density(density, self.group, volume_element)
        if density.reduction is not self.reduction:
            raise ValueError("the density uses another symmetry map than its group")
        values = density.values
        symmetric = self.symmetrize_values(values)
        weights = np.asarray(self.reduction.multiplicities, dtype=np.float64)
        before = float(np.dot(weights, values))
        wedge_sum = before / self.reduced_order
        deviation = float(np.max(np.abs(symmetric - values), initial=0.0))
        return SymmetryScalarField(self.reduction, symmetric), ScalarSymmetrization(
            maximum_deviation_percent=(deviation / wedge_sum * 100.0 if before else 0.0),
            charge_before=before * float(volume_element),
            charge_after=float(np.dot(weights, symmetric)) * float(volume_element),
        )


def _periodic_group(group) -> bool:
    from parsec_python.Grid.space_group import PeriodicGridSymmetry

    return isinstance(group, PeriodicGridSymmetry)


def build_wedge_density_symmetry(group: GridPointGroup, reduction) -> WedgeDensitySymmetry:
    """Map the operations of ``group`` onto the orbits of ``reduction``.

    Raises ``ValueError`` where an operation of the sectors is not one of
    ``group``: the average would then not be constant on their orbits.  The
    space group of a periodic cell (a :class:`~parsec_python.Grid.space_group.
    PeriodicGridSymmetry`) is checked on the rows instead: its average of a
    field constant on the orbits must be constant on them, which holds where
    the operations of the sectors map the group onto itself (a group of the
    identity alone always), and is tested at the rows of a fixed sample.
    """

    if _periodic_group(group):
        return _periodic_wedge_density_symmetry(group, reduction)
    matrices = np.asarray(group.matrices, dtype=np.float64)
    for operation in _subgroup_matrices(reduction):
        if not np.any(np.all(np.abs(matrices - operation[None]) < 1.0e-9, axis=(1, 2))):
            raise ValueError(
                "Symmetrize_Charge_Density: an operation of the symmetry sectors is "
                "not in the point group of the atoms; run with --symmetry off"
            )
    images = np.asarray(group.images)
    if images.shape[1] != int(reduction.full_size):
        raise ValueError("the point group and the symmetry sectors use different grids")
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    representatives = np.asarray(reduction.representative_rows)
    index_type = np.int32 if reduction.wedge_size < 2**31 else np.int64
    orbit_images = np.empty((images.shape[0], representatives.size), dtype=index_type)
    for index, rows in enumerate(images):
        orbit_images[index] = full_to_wedge[rows[representatives]]
    return WedgeDensitySymmetry(group=group, reduction=reduction, orbit_images=orbit_images)


def _periodic_wedge_density_symmetry(group, reduction) -> WedgeDensitySymmetry:
    """:func:`build_wedge_density_symmetry` of the space group of a periodic cell.

    The rows of the group are read at the points that they are needed at (``group.image_rows``: the representatives
    of the orbits, the sample of the check), from the table or from the affine maps of a group that holds none
    (``PARSEC_PERIODIC_IMAGE_ROWS``), the same integers."""

    operations, points = group.images.shape
    if points != int(reduction.full_size):
        raise ValueError("the space group and the symmetry sectors use different grids")
    full_to_wedge = np.asarray(reduction.full_to_wedge)
    representatives = np.asarray(reduction.representative_rows)
    index_type = np.int32 if reduction.wedge_size < 2**31 else np.int64
    orbit_images = np.empty((operations, representatives.size), dtype=index_type)
    for index in range(operations):
        orbit_images[index] = full_to_wedge[group.image_rows(index, representatives)]
    if operations > 1:
        # The average at a row must be the average at the representative of its orbit: a field of random orbit
        # values, at the rows of a fixed sample (operations of the sectors that do not map the group onto itself
        # make most rows differ).
        draw = np.random.default_rng(20261119)
        values = draw.random(int(reduction.wedge_size))
        at_representatives = np.zeros(representatives.size)
        for row_images in orbit_images:
            at_representatives += values[row_images]
        rows = draw.choice(
            int(reduction.full_size), size=min(_PERIODIC_CHECK_ROWS, int(reduction.full_size)), replace=False
        )
        at_rows = np.zeros(rows.size)
        for operation in range(operations):
            at_rows += values[full_to_wedge[group.image_rows(operation, rows)]]
        if np.max(np.abs(at_rows - at_representatives[full_to_wedge[rows]]), initial=0.0) > 1.0e-9 * operations:
            raise ValueError(
                "Symmetrize_Charge_Density: the operations of the symmetry sectors do not map the space group of "
                "the cell (its operations that map the grid onto itself) onto itself, so its average would not be "
                f"constant on their orbits; run with {SYMMETRIZED_DENSITY_SECTORS_SWITCH}=off (the reference solver "
                "on the full grid) or --symmetry off"
            )
    return WedgeDensitySymmetry(group=group, reduction=reduction, orbit_images=orbit_images)


def wedge_density_symmetry(group: GridPointGroup, reduction) -> WedgeDensitySymmetry:
    """:func:`build_wedge_density_symmetry`, built once per group and reduction."""

    key = (id(group), id(reduction))
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is not None and entry[0] is group and entry[1] is reduction:
            return entry[2]
    built = build_wedge_density_symmetry(group, reduction)
    with _CACHE_LOCK:
        _CACHE[key] = (group, reduction, built)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
    return built


__all__ = [
    "SYMMETRIZED_DENSITY_SECTORS_SWITCH",
    "WedgeDensitySymmetry",
    "build_wedge_density_symmetry",
    "symmetrized_density_sectors_setting",
    "wedge_density_symmetry",
]
