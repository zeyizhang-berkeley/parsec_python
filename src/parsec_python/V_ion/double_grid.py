"""Ono-Hirose double grid for the nonlocal projectors (``nonloc.F90``).

With ``Double_Grid_Order`` ``N > 1`` PARSEC does not sample a
Kleinman-Bylander projector at a grid point ``r_i`` but averages it over the
points of a grid ``N`` times finer around it (Ono and Hirose, Phys. Rev.
Lett. 82, 5016 (1999)):

``beta(r_i) = sum_(dx,dy,dz) w(dx) w(dy) w(dz) beta(r_i + (dx,dy,dz)*h/N)``,

``d = -(N-1), ..., N-1``, ``w(d) = (1 - |d|/N)/N``.  The weights of one
direction add up to one; they are those of linear interpolation from the
coarse to the fine grid, so the sum is the adjoint of that interpolation.
``nonloc.F90:116-121`` sets them up, ``205-259`` and ``317-406`` choose the
coarse points (one is kept when any of its fine points lies within the
projector radius ``rc``), and ``505-569`` sums the values and the gradients
(the gradients feed ``forcnloc.f90`` unchanged).  A fine point beyond ``rc``
contributes nothing; the radial table is evaluated at the distance clamped
below to the first radial sample.  ``N = 1`` is the ordinary sampling.

The harmonics of a fine point closer to the atom than the first radial
sample divide the vector by that sample, as ``ylm_real`` of ``nonloc.F90``
does with its clamped distance and as the projectors of
:func:`parsec_python.V_ion.build_nonlocal_projectors` do
(``PARSEC_PROJECTOR_CENTER=auto`` and ``tip``, since 2026-11-21); under ``off``
they take the true direction, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential, ParsecRadialSpline


def double_grid_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Offsets ``d = -(N-1), ..., N-1`` and weights ``(1 - |d|/N)/N``."""

    order = int(order)
    if order < 1:
        raise ValueError("the double grid order must be at least one")
    offsets = np.arange(-(order - 1), order)
    inverse = 1.0 / float(order)
    weights = (1.0 - np.abs(offsets).astype(np.float64) * inverse) * inverse
    return offsets, weights


def double_grid_points(order: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Displacements of the ``(2N-1)**3`` fine points and their weights.

    The order is that of the loops of ``nonloc.F90`` (x slowest, z
    fastest); displacements are ``d*h/N`` in bohr.
    """

    offsets, weights = double_grid_weights(order)
    inverse = 1.0 / float(order)
    shifts = offsets.astype(np.float64) * float(spacing) * inverse
    dx, dy, dz = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    wx, wy, wz = np.meshgrid(weights, weights, weights, indexing="ij")
    displacements = np.column_stack((dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)))
    return displacements, (wx * wy * wz).reshape(-1)


@dataclass(frozen=True)
class DoubleGridSupport:
    """The coarse points of one projector image and their fine points.

    ``rows`` are ascending rows of the grid; ``relative[p, m]`` is fine point
    ``m`` of row ``p`` minus the atom (image) position, ``radius`` its
    length, and ``weights`` the double-grid weight of the fine point, zero
    beyond the projector radius.
    """

    rows: np.ndarray
    relative: np.ndarray
    radius: np.ndarray
    weights: np.ndarray


def double_grid_support(
    coordinates: np.ndarray,
    center: np.ndarray,
    support_radius: float,
    order: int,
    spacing: float,
) -> DoubleGridSupport:
    """Coarse grid points with a fine point inside ``support_radius``."""

    displacements, weights = double_grid_points(order, spacing)
    reach = float(np.max(np.linalg.norm(displacements, axis=1)))
    coarse = np.asarray(coordinates, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    near = np.linalg.norm(coarse, axis=1) <= support_radius + reach + 1.0e-12 * (1.0 + support_radius)
    candidates = np.flatnonzero(near)
    relative = coarse[candidates][:, None, :] + displacements[None, :, :]
    radius = np.linalg.norm(relative, axis=2)
    inside = radius <= support_radius
    kept = np.any(inside, axis=1)
    return DoubleGridSupport(
        rows=candidates[kept],
        relative=relative[kept],
        radius=radius[kept],
        weights=np.where(inside[kept], weights[None, :], 0.0),
    )


def interpolate_projector(
    potential: ParsecPseudopotential,
    table: np.ndarray,
    radius: np.ndarray,
    spline: ParsecRadialSpline | None,
) -> np.ndarray:
    """The radial projector at ``radius`` as the projector builders take it."""

    if spline is not None:
        return spline(np.maximum(radius, potential.radii[0]))
    return np.interp(radius, potential.radii, table, left=table[0], right=0.0)


def build_double_grid_projectors(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
    *,
    order: int | None = None,
):
    """Kleinman-Bylander projectors averaged over the double grid.

    Arguments, result, column order and periodic images are those of
    :func:`parsec_python.V_ion.build_nonlocal_projectors`; ``order``
    defaults to ``grid.settings.double_grid_order``.  The Bloch phase of a
    coarse point is that of the point itself, as in ``nonloc.F90``.
    """

    from .ionic_potential import (
        NonlocalProjectorOperator,
        _periodic_image_translations,
        _projector_support_radius,
        projector_center_radius,
        projector_center_setting,
        real_spherical_harmonics,
    )

    # the centre radius of the harmonics: rs(2) under PARSEC_PROJECTOR_CENTER=auto and tip, the true direction under off
    center_setting = projector_center_setting()
    if order is None:
        order = int(getattr(grid.settings, "double_grid_order", 1))
    complex_valued = k_point is not None
    if complex_valued and lattice_vectors is None:
        raise ValueError("k_point requires lattice_vectors")
    value_dtype = complex if complex_valued else float
    k_point = None if k_point is None else np.asarray(k_point, dtype=float)
    sqrt_dv = np.sqrt(grid.volume_element)
    spacing = float(grid.spacing)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)
        support_radius = _projector_support_radius(potential)
        translations = (
            np.zeros((1, 3))
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support_radius)
        )
        images = []
        for translation in translations:
            support = double_grid_support(
                grid.coordinates, position + translation, support_radius, order, spacing
            )
            if support.rows.size:
                phase = (
                    1.0
                    if k_point is None
                    else np.exp(
                        -1j * ((grid.coordinates[support.rows] - translation) @ k_point)
                    )
                )
                images.append((support, phase))
        if not images:
            continue

        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            table, sign = potential.radial_projector(angular_momentum, local_l)
            spline = (
                ParsecRadialSpline.from_positive_grid(
                    potential.radii, table, grid.settings.stencil_half_width
                )
                if specification.use_spline
                else None
            )
            width = 2 * angular_momentum + 1
            channel_columns = list(range(column, column + width))
            column += width
            for harmonic_index in range(width):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
            for support, phase in images:
                points, fine = support.radius.shape
                radial = interpolate_projector(
                    potential, table, support.radius.reshape(-1), spline
                )
                harmonics = real_spherical_harmonics(
                    angular_momentum,
                    support.relative.reshape(-1, 3),
                    center_radius=projector_center_radius(potential, center_setting, off_radius=0.0),
                )
                scaled = (sqrt_dv * radial)[:, None] * harmonics
                summed = np.einsum(
                    "pm,pmh->ph", support.weights, scaled.reshape(points, fine, width)
                )
                for harmonic_index, target in enumerate(channel_columns):
                    projector = phase * summed[:, harmonic_index]
                    keep = np.abs(projector) > 1.0e-16
                    if not np.any(keep):
                        continue
                    rows.append(support.rows[keep])
                    columns.append(np.full(np.count_nonzero(keep), target, dtype=np.int64))
                    values.append(projector[keep].astype(value_dtype, copy=False))

    if column == 0:
        matrix = sp.csc_matrix((grid.size, 0), dtype=value_dtype)
    else:
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()
    return NonlocalProjectorOperator(
        projectors=matrix,
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


#: A channel is named in the note of :func:`double_grid_note_lines` when its projection falls below this share.
NOTE_RATIO = 0.99
#: Channels whose Kleinman-Bylander energy in the atomic orbital is below this (Ry) carry no energy to lose.
NOTE_ENERGY_FLOOR = 1.0e-6


@dataclass(frozen=True)
class ChannelProjection:
    """The projection of a species' pseudo-orbital onto its own projector on a grid.

    ``ratio`` is ``<beta_lm|phi_lm>`` on the double grid of the run, averaged over ``m``, divided by the radial value
    ``int beta_l(r) u_l(r) r dr``; ``sampled_ratio`` the same for plain sampling (order one) on the same grid.
    ``kb_energy`` is ``sign * <beta|phi>^2`` of the radial value, the nonlocal energy of one electron in the atomic
    orbital (Ry), and ``occupation`` that orbital's electrons in the reference configuration of the file (``None``
    where the format gives none).
    """

    angular_momentum: int
    ratio: float
    sampled_ratio: float
    kb_energy: float
    occupation: float | None

    @property
    def energy_change(self) -> float:
        """First-order change of the channel's nonlocal energy per electron in the atomic orbital (Ry)."""

        return self.kb_energy * (self.ratio * self.ratio - 1.0)


def _local_cube(spacing: float, reach: float) -> np.ndarray:
    """The points of a cubic grid of ``spacing`` with the origin on a point, out to ``reach`` along each axis."""

    count = int(np.ceil(reach / spacing)) + 1
    axis = np.arange(-count, count + 1, dtype=np.float64) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))


def orbital_projection_ratios(
    potential: ParsecPseudopotential,
    specification: SpeciesPotential,
    order: int,
    spacing: float,
    stencil_half_width: int,
    *,
    chunk: int = 4096,
) -> list[ChannelProjection]:
    """:class:`ChannelProjection` of every nonlocal channel of a species with ``Double_Grid_Order`` ``order``.

    The atom sits on a point of a cubic grid of ``spacing``.  The projector values are those of
    :func:`build_double_grid_projectors` (the support, fine-point weights, radial table or spline, harmonics and
    centre radius of the run); the orbital ``u_l(r)/r Y_lm`` is sampled at the coarse points.  Channels whose
    Kleinman-Bylander energy is below :data:`NOTE_ENERGY_FLOOR` are left out.  The tables are taken as read (no
    Fourier filter).  Nothing here enters a calculation.
    """

    from ..Pseudopotential.radial_quadrature import parsec_radial_integral
    from .ionic_potential import (
        _projector_support_radius,
        projector_center_radius,
        projector_center_setting,
        real_spherical_harmonics,
    )

    radii = np.asarray(potential.radii, dtype=np.float64)
    center_radius = projector_center_radius(potential, projector_center_setting(), off_radius=0.0)
    support_radius = _projector_support_radius(potential)
    displacements, _ = double_grid_points(order, spacing)
    cube = _local_cube(spacing, support_radius + float(np.max(np.linalg.norm(displacements, axis=1))))
    origin = np.zeros(3)
    volume = spacing**3
    local_l = specification.local_angular_momentum
    occupations = getattr(potential, "channel_occupations", None) or {}
    result: list[ChannelProjection] = []
    for angular_momentum in sorted(potential.radial_wavefunctions):
        if angular_momentum == local_l:
            continue
        table, sign = potential.radial_projector(angular_momentum, local_l)
        wave = np.asarray(potential.radial_wavefunctions[angular_momentum], dtype=np.float64)
        radial = parsec_radial_integral(radii, table * wave * radii)
        if radial * radial < NOTE_ENERGY_FLOOR:
            continue
        spline = (
            ParsecRadialSpline.from_positive_grid(radii, table, stencil_half_width)
            if specification.use_spline
            else None
        )
        width = 2 * angular_momentum + 1
        ratios = []
        for grid_order in (order, 1):
            projections = np.zeros(width)
            for start in range(0, cube.shape[0], chunk):
                block = cube[start : start + chunk]
                support = double_grid_support(block, origin, support_radius, grid_order, spacing)
                if support.rows.size == 0:
                    continue
                points, fine = support.radius.shape
                values = interpolate_projector(potential, table, support.radius.reshape(-1), spline)
                harmonics = real_spherical_harmonics(
                    angular_momentum, support.relative.reshape(-1, 3), center_radius=center_radius
                )
                beta = np.einsum(
                    "pm,pmh->ph", support.weights, (values[:, None] * harmonics).reshape(points, fine, width)
                )
                coarse = block[support.rows]
                distance = np.linalg.norm(coarse, axis=1)
                orbital = np.interp(distance, radii, wave / radii, right=0.0)[:, None] * real_spherical_harmonics(
                    angular_momentum, coarse, center_radius=float(radii[0])
                )
                projections += np.einsum("ph,ph->h", beta, orbital) * volume
            ratios.append(float(np.mean(projections)) / radial)
        occupation = occupations.get(angular_momentum)
        result.append(
            ChannelProjection(
                angular_momentum=angular_momentum,
                ratio=ratios[0],
                sampled_ratio=ratios[1],
                kb_energy=float(sign) * radial * radial,
                occupation=None if occupation is None else float(occupation),
            )
        )
    return result


def double_grid_note_lines(problem) -> tuple[str, ...]:
    """The NOTE of the report on the projections the double grid of ``problem`` loses (empty for order one).

    One line per species and nonlocal channel whose :func:`orbital_projection_ratios` ratio falls below
    :data:`NOTE_RATIO`: the ratio, that of plain sampling on the same spacing, and the first-order change of the
    nonlocal energy per electron in the atomic orbital and for the atoms of the species with the electrons of the
    file's reference configuration.  Output only: a failure of the estimate prints nothing.
    """

    grid = problem.grid
    order = int(getattr(grid, "double_grid_order", 1) or 1)
    if order <= 1:
        return ()
    try:
        from .ionic_potential import load_pseudopotentials

        specifications = problem.pseudopotentials
        potentials = load_pseudopotentials(specifications, check_correlation=False)
        spacing = float(grid.spacing)
        rows = []
        for label, specification in specifications.items():
            count = sum(1 for atom in problem.atoms if atom.symbol == label)
            if count == 0:
                continue
            for channel in orbital_projection_ratios(
                potentials[label], specification, order, spacing, int(grid.stencil_half_width)
            ):
                if channel.ratio >= NOTE_RATIO:
                    continue
                change = channel.energy_change
                text = (
                    f"       {label} l={channel.angular_momentum}: {channel.ratio:.3f} of the radial value "
                    f"(order 1: {channel.sampled_ratio:.3f}), {change:+.4f} Ry per electron"
                )
                if channel.occupation:
                    text += (
                        f", {change * channel.occupation * count:+.3f} Ry for {count} atom"
                        f"{'s' if count > 1 else ''} with {channel.occupation:.2f} electrons"
                        f"{' each' if count > 1 else ''}"
                    )
                rows.append(text)
    except Exception:  # noqa: BLE001 - the note never stops or changes a run
        return ()
    if not rows:
        return ()
    return (
        f" NOTE: Double_Grid_Order {order} shrinks the projections of these pseudo-orbitals onto their own",
        f"       projectors (atom on a point of a cubic grid of spacing {spacing:.4f} bohr); first-order change of",
        "       the nonlocal energy, which the SCF partly relaxes:",
        *rows,
        "",
    )


__all__ = [
    "ChannelProjection",
    "DoubleGridSupport",
    "build_double_grid_projectors",
    "double_grid_note_lines",
    "orbital_projection_ratios",
    "double_grid_points",
    "double_grid_support",
    "double_grid_weights",
    "interpolate_projector",
]
