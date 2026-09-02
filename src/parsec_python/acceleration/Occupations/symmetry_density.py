"""Density construction that keeps representation orbitals on the wedge."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..Eigensolvers.symmetry import CuPySymmetryOrbitals
from ..SCF.symmetry_fields import SymmetrySCFReducer


class CuPySymmetryDensityBuilder:
    """Build the full scalar density without expanding every wavefunction.

    A real one-dimensional representation differs between symmetry images
    only by a sign.  Since the density contains ``|psi|**2``, all images have
    the same value.  The wrapped ordinary CuPy builder therefore evaluates
    the globally ordered, already-normalized wedge columns once, downloads a
    wedge-length density, and expands that scalar field by orbit lookup.
    """

    def __init__(
        self,
        device_builder: Any,
        timing_stats: Any | None = None,
        reducer: SymmetrySCFReducer | None = None,
    ) -> None:
        self.device_builder = device_builder
        self.timing_stats = timing_stats
        self.reducer = reducer

    def __call__(
        self,
        wavefunctions: CuPySymmetryOrbitals,
        occupations: np.ndarray,
        volume_element: float,
    ) -> np.ndarray:
        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            return self.device_builder(
                wavefunctions, occupations, volume_element
            )

        wedge_density = self.device_builder(
            wavefunctions.scaled_wedge_vectors,
            occupations,
            volume_element,
        )
        if self.reducer is not None:
            return self.reducer.field(wedge_density)
        expansion_started = perf_counter()
        full_to_wedge = np.asarray(
            wavefunctions.full_to_wedge, dtype=np.int64
        )
        density = np.ascontiguousarray(wedge_density[full_to_wedge])
        if self.timing_stats is not None:
            self.timing_stats.density_seconds += (
                perf_counter() - expansion_started
            )
        return density


__all__ = ["CuPySymmetryDensityBuilder"]
