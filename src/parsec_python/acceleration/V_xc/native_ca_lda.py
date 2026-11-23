"""Reusable C++/OpenMP CA/PZ-LDA evaluator."""

from __future__ import annotations

import numpy as np

from parsec_python.V_xc import XCResult

from ..backends.native import _load_native
from ..SCF.symmetry_fields import SymmetrySCFReducer, SymmetryScalarField


class NativeCALDAEvaluator:
    """Cache frozen NLCC density and evaluate CA-LDA in native float64 loops.

    The C++ loops hold PARSEC's coefficients.  Under ``PARSEC_CA_CONTINUITY=on`` (the default; read when the
    evaluator is made) the points of ``r_s < 1`` take the change of the high-density coefficients on top of them
    (:meth:`_continuous`), the energy weighted by the multiplicities of the orbits as the C++ loops weight it.
    """

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
        from parsec_python.V_xc.ca_lda import ca_continuity_setting

        self.continuity = ca_continuity_setting()
        self._core = core
        self._volume_element = float(volume_element)
        self._weights = (
            None
            if reducer is None
            else np.asarray(reducer.reduction.multiplicities, dtype=np.float64)
        )
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
        if self.continuity == "on":
            payload = self._continuous(valence)
        else:
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

    def _continuous(self, valence: np.ndarray) -> dict:
        """The payload of the C++ ``evaluate`` under ``PARSEC_CA_CONTINUITY=on``.

        The C++ loops evaluate PARSEC's coefficients; where ``r_s < 1`` the high-density form then gains the change
        of its coefficients, ``(dc3 ln r_s - dc4) r_s`` in the energy per electron and that less
        ``(dc3 ln r_s - dc5) r_s / 3`` in the potential.  Points within 1e-10 of ``r_s = 1``, where the C++ loops
        and NumPy could take different forms, are evaluated by :func:`ca_lda` itself.
        """

        from parsec_python.V_xc.ca_lda import ca_high_density_coefficients, ca_lda

        payload = self._evaluator.evaluate(valence)
        # the C++ loops return new arrays on every call: the change is made in place
        potential = np.asarray(payload["potential"], dtype=np.float64)
        epsilon = np.asarray(payload["energy_per_electron"], dtype=np.float64)
        energy_density = np.asarray(payload["energy_density"], dtype=np.float64)
        total = float(payload["total_energy"])
        density = valence + self._core
        critical = 0.75 / np.pi
        # the points that can lie at r_s <= 1 + 1e-10 (a superset), then the tests of every point on them alone
        candidates = np.flatnonzero(density > critical * (1.0 - 2.0e-10))
        if candidates.size == 0:
            return {"potential": potential, "energy_per_electron": epsilon, "energy_density": energy_density,
                    "total_energy": total}
        ratio = density[candidates] / critical
        high_mask = ratio > 1.0 + 1.0e-10
        near_mask = np.abs(ratio - 1.0) <= 1.0e-10
        changed = candidates[high_mask | near_mask]
        before = energy_density[changed]
        high = candidates[high_mask]
        near = candidates[near_mask]
        if high.size:
            on = ca_high_density_coefficients("on")
            off = ca_high_density_coefficients("off")
            dc3, dc4, dc5 = (on[i] - off[i] for i in range(3))
            rho = density[high]
            rs = np.cbrt(critical / rho)
            log_r = np.log(rs)
            change = (dc3 * log_r - dc4) * rs
            epsilon[high] += change
            potential[high] += change - (dc3 * log_r - dc5) * rs / 3.0
            energy_density[high] = rho * epsilon[high]
        if near.size:
            exact = ca_lda(valence[near], self._volume_element, self._core[near])
            potential[near] = exact.potential
            epsilon[near] = exact.energy_per_electron
            energy_density[near] = exact.energy_density
        if changed.size:
            weights = 1.0 if self._weights is None else self._weights[changed]
            total += float(np.sum(weights * (energy_density[changed] - before))) * self._volume_element
        return {
            "potential": potential,
            "energy_per_electron": epsilon,
            "energy_density": energy_density,
            "total_energy": total,
        }


__all__ = ["NativeCALDAEvaluator"]
