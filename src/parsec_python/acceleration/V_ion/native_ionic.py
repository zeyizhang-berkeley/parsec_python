"""C++/OpenMP construction of PARSEC local and KB ionic grid fields.

The public objects returned here are the same NumPy arrays and
``NonlocalProjectorOperator`` used by :mod:`parsec_python`.  Only the
atom-by-grid loops, radial interpolation, and real-harmonic evaluation move to
native code; POTRE parsing, KB denominators, support rules, column ordering,
and sparse assembly stay visible in Python.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.V_ion import NonlocalProjectorOperator

from ..backends import native as native_backend


_EMPTY = np.empty(0, dtype=np.float64)


def _values64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _spline_payload(
    potential: ParsecPseudopotential,
    values: np.ndarray,
    grid: RealSpaceGrid,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        return _EMPTY, _EMPTY, _EMPTY
    spline = ParsecRadialSpline.from_positive_grid(
        potential.radii,
        values,
        grid.settings.stencil_half_width,
    )
    return (
        _values64(spline.knots),
        _values64(spline.values),
        _values64(spline.second_derivatives),
    )


def _projector_support_radius(potential: ParsecPseudopotential) -> float:
    requested = max(
        potential.channel_cutoffs.values(), default=potential.radii[0]
    )
    next_index = int(np.searchsorted(potential.radii, requested, side="right"))
    next_index = min(next_index, potential.radii.size - 2)
    return float(potential.radii[next_index])


class NativeIonicBuilders:
    """Cache grid coordinates and expose reference-compatible setup hooks."""

    def __init__(self) -> None:
        self._grid_identity: int | None = None
        self._grid_size = 0
        self._evaluator = None

    def _for_grid(self, grid: RealSpaceGrid):
        identity = id(grid)
        if self._evaluator is None or self._grid_identity != identity:
            self._evaluator = native_backend._load_native().RadialGridEvaluator(
                _values64(grid.coordinates)
            )
            self._grid_identity = identity
            self._grid_size = grid.size
        return self._evaluator

    def build_local_ionic_potential(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
    ) -> np.ndarray:
        """Build ``sum_a V_local,a`` with PARSEC rV/spline interpolation."""

        evaluator = self._for_grid(grid)
        total = np.zeros(grid.size, dtype=np.float64)
        for atom in atoms:
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            values = _values64(
                potential.channel_potentials[
                    specification.local_angular_momentum
                ]
            )
            spline = _spline_payload(
                potential, values, grid, specification.use_spline
            )
            total += np.asarray(
                evaluator.local_potential(
                    _values64(atom.position),
                    _values64(potential.radii),
                    values,
                    float(potential.ionic_charge),
                    *spline,
                ),
                dtype=np.float64,
            )
        return total

    def superpose_atomic_density(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
        *,
        core: bool = False,
    ) -> np.ndarray:
        """Build initial valence or frozen NLCC density with native loops."""

        evaluator = self._for_grid(grid)
        total = np.zeros(grid.size, dtype=np.float64)
        for atom in atoms:
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            if core and not potential.has_nonlinear_core_correction:
                continue
            if core:
                radial_density = _values64(potential.core_density)
                use_spline = specification.use_spline
            elif specification.read_valence_density:
                radial_density = _values64(potential.valence_density)
                # initchrg.f90 always uses linear interpolation for VCD.
                use_spline = False
            else:
                radial_density = np.zeros_like(potential.radii, dtype=np.float64)
                for angular_momentum, wavefunction in (
                    potential.radial_wavefunctions.items()
                ):
                    radial_density += (
                        potential.channel_occupations.get(angular_momentum, 0.0)
                        * wavefunction
                        * wavefunction
                        / (4.0 * np.pi * potential.radii * potential.radii)
                    )
                radial_density = _values64(radial_density)
                use_spline = False
            spline = _spline_payload(
                potential, radial_density, grid, use_spline
            )
            total += np.asarray(
                evaluator.density(
                    _values64(atom.position),
                    _values64(potential.radii),
                    radial_density,
                    *spline,
                ),
                dtype=np.float64,
            )
        return total

    def build_nonlocal_projectors(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
    ) -> NonlocalProjectorOperator:
        """Build PARSEC-order KB sparse columns with native radial kernels."""

        evaluator = self._for_grid(grid)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        signs: list[float] = []
        labels: list[tuple[int, int, int]] = []
        column = 0
        square_root_volume = float(np.sqrt(grid.volume_element))

        for atom_index, atom in enumerate(atoms):
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            local_l = specification.local_angular_momentum
            for angular_momentum in sorted(potential.radial_wavefunctions):
                if angular_momentum == local_l:
                    continue
                radial, denominator_sign = potential.radial_projector(
                    angular_momentum, local_l
                )
                radial = _values64(radial)
                spline = _spline_payload(
                    potential, radial, grid, specification.use_spline
                )
                payload = evaluator.projector_channel(
                    _values64(atom.position),
                    _values64(potential.radii),
                    radial,
                    _projector_support_radius(potential),
                    int(angular_momentum),
                    square_root_volume,
                    *spline,
                )
                support_rows = np.asarray(payload["rows"], dtype=np.int64)
                channel_values = np.asarray(payload["values"], dtype=np.float64)
                for harmonic_index in range(channel_values.shape[1]):
                    projector = channel_values[:, harmonic_index]
                    keep = np.abs(projector) > 1.0e-16
                    kept_rows = support_rows[keep]
                    rows.append(kept_rows)
                    columns.append(
                        np.full(kept_rows.size, column, dtype=np.int64)
                    )
                    values.append(projector[keep])
                    signs.append(denominator_sign)
                    labels.append(
                        (atom_index, angular_momentum, harmonic_index)
                    )
                    column += 1

        if column == 0:
            matrix = sp.csc_matrix((grid.size, 0), dtype=np.float64)
        else:
            matrix = sp.coo_matrix(
                (
                    np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=(grid.size, column),
            ).tocsc()
        return NonlocalProjectorOperator(
            projectors=matrix,
            signs=np.asarray(signs, dtype=np.float64),
            labels=tuple(labels),
        )


__all__ = ["NativeIonicBuilders"]
