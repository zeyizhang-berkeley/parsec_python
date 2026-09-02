"""Reusable C++/OpenMP CA/PZ-LDA evaluator."""

from __future__ import annotations

import numpy as np

from parsec_python.V_xc import XCResult

from ..backends.native import _load_native
from ..SCF.symmetry_fields import SymmetrySCFReducer, SymmetryScalarField


class NativeCALDAEvaluator:
    """Cache frozen NLCC density and evaluate CA-LDA in native float64 loops."""

    def __init__(
        self,
        core_density: np.ndarray,
        volume_element: float,
        reducer: SymmetrySCFReducer | None = None,
    ) -> None:
        self.reducer = reducer
        core = (
            np.ascontiguousarray(core_density, dtype=np.float64)
            if reducer is None
            else np.ascontiguousarray(
                reducer.wedge_values(core_density), dtype=np.float64
            )
        )
        if core.ndim != 1:
            raise ValueError("core_density must be one-dimensional")
        self._evaluator = (
            _load_native().CALDAEvaluator(core, float(volume_element))
            if reducer is None
            else _load_native().CALDAEvaluator(
                core,
                float(volume_element),
                np.ascontiguousarray(
                    reducer.reduction.multiplicities, dtype=np.int64
                ),
            )
        )
        self.size = int(core.size)

    def __call__(self, valence_density) -> XCResult:
        valence = np.ascontiguousarray(
            valence_density
            if self.reducer is None
            else self.reducer.wedge_values(valence_density),
            dtype=np.float64,
        )
        if valence.shape != (self.size,):
            raise ValueError("valence density does not match the XC grid")
        payload = self._evaluator.evaluate(valence)
        potential = np.asarray(payload["potential"], dtype=np.float64)
        epsilon = np.asarray(payload["energy_per_electron"], dtype=np.float64)
        energy_density = np.asarray(payload["energy_density"], dtype=np.float64)
        if self.reducer is not None:
            potential = self.reducer.field(potential)
            epsilon = self.reducer.field(epsilon)
            energy_density = self.reducer.field(energy_density)
        return XCResult(
            potential=potential,
            energy_per_electron=epsilon,
            energy_density=energy_density,
            total_energy=float(payload["total_energy"]),
        )


__all__ = ["NativeCALDAEvaluator"]
