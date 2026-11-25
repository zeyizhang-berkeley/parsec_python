"""PARSEC-compatible uniform cluster grids.

This module contains the native Python implementation of PARSEC's isolated
real-space domain construction.  It can be imported independently of the SCF
driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import GridSettings


@dataclass(frozen=True)
class RealSpaceGrid:
    """Active points of a uniform, nonperiodic PARSEC cluster grid.

    Integer points are ordered like ``grid_partition.f90``: x, then y, then z,
    all traversed from their maximum index to their minimum index.
    """

    settings: GridSettings
    integer_coordinates: np.ndarray
    coordinates: np.ndarray
    index_min: np.ndarray
    index_max: np.ndarray
    lookup: np.ndarray

    @property
    def spacing(self) -> float:
        return self.settings.spacing

    @property
    def volume_element(self) -> float:
        return self.spacing**3

    @property
    def size(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in (self.index_max - self.index_min + 1))

    def rows_for_integer_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Return active row numbers, or ``-1`` for points outside the domain."""
        points = np.asarray(points, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("integer grid points must have shape (n, 3)")
        local = points - self.index_min
        valid = np.all((local >= 0) & (local < np.asarray(self.lookup.shape)), axis=1)
        rows = np.full(points.shape[0], -1, dtype=np.int64)
        if np.any(valid):
            loc = local[valid]
            rows[valid] = self.lookup[loc[:, 0], loc[:, 1], loc[:, 2]]
        return rows

    def physical_coordinates(self, integer_points: np.ndarray) -> np.ndarray:
        integer_points = np.asarray(integer_points, dtype=float)
        return (integer_points + np.asarray(self.settings.shift)) * self.spacing

    def integrate(self, values: np.ndarray) -> float:
        values = np.asarray(values)
        if values.shape[0] != self.size:
            raise ValueError("values do not match the active grid")
        return float(np.sum(values) * self.volume_element)


SPHERE_TIES_SWITCH = "PARSEC_SPHERE_TIES"
_SPHERE_TIES = ("dot", "symmetric")


def sphere_ties_setting() -> str:
    """``PARSEC_SPHERE_TIES``: how a point exactly on the sphere is decided.

    ``dot`` (the default): ``x^2 + y^2 + z^2 <= R^2`` with the squares summed as ``np.einsum`` sums them, as before.
    A point on the sphere (on a grid through the origin, ``Grid_Shift`` zero, whose ``(R/h)^2`` is an integer; on the
    shifted grid ``(R/h)^2 = 2 k + 3/4``) is then decided by the round-off of that sum, which depends on the order of
    its coordinates: of ``(12, 20, 9) h`` and ``(20, 12, 9) h`` at ``R = 25 h`` one is in and the other out, and the
    domain loses the operations that permute the axes (C4z, S4z, the threefold axes).  ``symmetric`` (opt-in): the
    three squares summed in ascending order, a value every signed permutation of the axes leaves unchanged, so a point
    and its images under those operations are in or out together (the points off the sphere are decided as before).
    Another value is an error when the grid is built.
    """

    import os

    text = os.environ.get(SPHERE_TIES_SWITCH, "dot").strip().lower() or "dot"
    if text not in _SPHERE_TIES:
        raise ValueError(f"{SPHERE_TIES_SWITCH} must be one of {', '.join(_SPHERE_TIES)}, not {text!r}")
    return text


def _squared_radii(coordinates: np.ndarray) -> np.ndarray:
    """``|r|^2`` of every point as the sphere test takes it (:func:`sphere_ties_setting`)."""

    if sphere_ties_setting() == "symmetric":
        values = np.asarray(coordinates, dtype=np.float64)
        squares = np.sort(values * values, axis=1)
        return (squares[:, 0] + squares[:, 1]) + squares[:, 2]
    return np.einsum("ij,ij->i", coordinates, coordinates)


def _inside_domain(settings: GridSettings, coordinates: np.ndarray) -> np.ndarray:
    if settings.domain_shape == "sphere":
        return _squared_radii(coordinates) <= settings.radius**2
    if settings.domain_shape == "ellipsoid":
        # grid_partition.f90 outside_domain, case 1.
        radii = np.asarray(settings.domain_parameters, dtype=float)
        return np.sum((coordinates / radii) ** 2, axis=1) <= 1.0
    if settings.domain_shape == "cylinder":
        # outside_domain, case 2: the axis coordinate and the two others.
        radius, length, axis = settings.domain_parameters
        along = int(axis) - 1
        across = [index for index in range(3) if index != along]
        polar = np.sum(coordinates[:, across] ** 2, axis=1)
        return (polar <= radius**2) & (coordinates[:, along] ** 2 <= (length / 2.0) ** 2)
    lengths = np.asarray(settings.box_lengths, dtype=float)
    return np.all(np.abs(coordinates) <= 0.5 * lengths + 1.0e-14, axis=1)


def _fold_block_points(integer_coordinates: np.ndarray, side: int) -> np.ndarray:
    """Every point of the blocks of ``side**3`` points that hold one of ``integer_coordinates``.

    PARSEC 1.5 (``grid_partition.f90:531-607``): a block enters the domain whole when any of its points lies
    inside the shape.  Its blocks start at the lower corner of the padded box ``bbox_pad`` and run in steps of
    ``side``; for a cluster that corner is made even (``grid_partition.f90:258-320``), so the blocks of side 2
    are ``[2m, 2m + 1]`` along every axis.  The points come in the order of :func:`build_cluster_grid`: x, then
    y, then z, each from its largest index to its smallest.
    """

    blocks = np.unique(np.floor_divide(integer_coordinates, side), axis=0)
    corner = np.arange(side)
    offsets = np.stack(np.meshgrid(corner, corner, corner, indexing="ij"), axis=-1).reshape(-1, 3)
    points = (blocks[:, None, :] * side + offsets[None, :, :]).reshape(-1, 3)
    order = np.lexsort((-points[:, 2], -points[:, 1], -points[:, 0]))
    return points[order]


def build_cluster_grid(settings: GridSettings) -> RealSpaceGrid:
    """Build the full active cluster domain without symmetry reduction.

    PARSEC uses ``n = floor(2*radius/h) + 2`` for a confined system, integer
    bounds ``[-n/2, n-n/2-1]``, and positions ``h*(integer + shift)``.

    ``settings.foldblock_side_length`` 2 (PARSEC 1.5's ``Foldblock_Side_Length``) fills the domain up to whole
    blocks of 2 x 2 x 2 points (:func:`_fold_block_points`); a block may reach one index beyond the bounds, which
    then grow by one on each side.
    """
    n = int(np.floor(2.0 * settings.enclosing_radius / settings.spacing)) + 2
    index_min = np.full(3, -(n // 2), dtype=int)
    index_max = np.full(3, n + index_min[0] - 1, dtype=int)

    axes = [np.arange(index_max[d], index_min[d] - 1, -1, dtype=int) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    integer_coordinates = np.column_stack([component.reshape(-1) for component in mesh])
    coordinates = (integer_coordinates + np.asarray(settings.shift)) * settings.spacing
    active = _inside_domain(settings, coordinates)
    integer_coordinates = integer_coordinates[active]
    coordinates = coordinates[active]

    side = int(getattr(settings, "foldblock_side_length", 1))
    if side > 1:
        integer_coordinates = _fold_block_points(integer_coordinates, side)
        coordinates = (integer_coordinates + np.asarray(settings.shift)) * settings.spacing
        index_min = index_min - (side - 1)
        index_max = index_max + (side - 1)
    extent = int(index_max[0] - index_min[0] + 1)

    lookup = np.full((extent, extent, extent), -1, dtype=np.int64)
    local = integer_coordinates - index_min
    lookup[local[:, 0], local[:, 1], local[:, 2]] = np.arange(integer_coordinates.shape[0])

    return RealSpaceGrid(
        settings=settings,
        integer_coordinates=integer_coordinates,
        coordinates=coordinates,
        index_min=index_min,
        index_max=index_max,
        lookup=lookup,
    )


__all__ = ["RealSpaceGrid", "build_cluster_grid"]
