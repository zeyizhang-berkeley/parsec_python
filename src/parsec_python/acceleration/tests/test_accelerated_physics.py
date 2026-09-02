"""Focused parity tests for newly accelerated physical kernels."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.Occupations import CuPyDeviceDensityBuilder
from parsec_python.acceleration.SCF import SymmetrySCFReducer, SymmetryScalarField
from parsec_python.acceleration.Symmetry import AxisReflectionReduction
from parsec_python.acceleration.V_ion import NativeIonicBuilders
from parsec_python.acceleration.V_xc import NativeCALDAEvaluator
from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    cupy_available,
    require_cupy,
)
from parsec_python.acceleration.backends.cupy_compact import CuPyCompactFiniteDifference
from parsec_python.acceleration.backends.cupy_stencil_major import (
    CuPyStencilMajorFiniteDifference,
)
from parsec_python.acceleration.backends.native import (
    native_available,
    native_build_info,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Occupations import density_from_orbitals
from parsec_python.V_ion import (
    build_local_ionic_potential,
    build_nonlocal_projectors,
    load_pseudopotentials,
    superpose_atomic_density,
)
from parsec_python.V_xc import ca_lda
from parsec_python.models import Atom, GridSettings, SpeciesPotential


def _native_kernels() -> tuple[str, ...]:
    if not native_available():
        return ()
    return tuple(native_build_info().get("implemented_kernels", ()))


NATIVE_KERNELS = _native_kernels()
GPU_AVAILABLE = cupy_available()


class _NumPyCuPyShim:
    """Minimum no-RawKernel CuPy surface for testing the safe fallback."""

    float64 = np.float64

    @staticmethod
    def asarray(values, dtype=None):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def asnumpy(values):
        return np.asarray(values)


class DeviceDensityTests(unittest.TestCase):
    def test_density_builder_fallback_matches_reference(self) -> None:
        rng = np.random.default_rng(811)
        vectors = rng.normal(size=(317, 13))
        occupations = rng.random(13)
        volume = 0.125
        builder = CuPyDeviceDensityBuilder(_NumPyCuPyShim())
        expected = density_from_orbitals(vectors, occupations, volume)
        actual = builder(vectors, occupations, volume)
        np.testing.assert_array_equal(actual, expected)


@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class CompactCUDAFiniteDifferenceTests(unittest.TestCase):
    def test_compact_stencil_and_fused_local_match_csr(self) -> None:
        cp, cpsparse = require_cupy()
        dimension = 4097
        matrix = sp.diags(
            (
                -0.25 * np.ones(dimension - 2),
                -1.5 * np.ones(dimension - 1),
                4.0 * np.ones(dimension),
                -1.5 * np.ones(dimension - 1),
                -0.25 * np.ones(dimension - 2),
            ),
            (-2, -1, 0, 1, 2),
            format="csr",
        )
        compact = CuPyCompactFiniteDifference(cp, matrix)
        device_matrix = cpsparse.csr_matrix(matrix)
        local = cp.linspace(-0.7, 0.3, dimension, dtype=cp.float64)
        rng = np.random.default_rng(55)
        for width in (1, 6, 7, 13):
            with self.subTest(width=width):
                vectors = cp.asarray(
                    rng.normal(size=(dimension, width)), dtype=cp.float64
                )
                expected = device_matrix @ vectors + local[:, None] * vectors
                actual = compact.apply(vectors, local)
                cp.cuda.get_current_stream().synchronize()
                np.testing.assert_allclose(
                    cp.asnumpy(actual),
                    cp.asnumpy(expected),
                    rtol=2e-14,
                    atol=2e-14,
                )
        self.assertEqual(compact.palette_size, 3)

    def test_stencil_major_and_fused_recurrence_match_literal_action(self) -> None:
        cp, cpsparse = require_cupy()
        dimension = 4097
        matrix = sp.diags(
            (
                -0.25 * np.ones(dimension - 2),
                -1.5 * np.ones(dimension - 1),
                4.0 * np.ones(dimension),
                -1.5 * np.ones(dimension - 1),
                -0.25 * np.ones(dimension - 2),
            ),
            (-2, -1, 0, 1, 2),
            format="csr",
        )
        stencil = CuPyStencilMajorFiniteDifference(cp, matrix)
        device_matrix = cpsparse.csr_matrix(matrix)
        local = cp.linspace(-0.7, 0.3, dimension, dtype=cp.float64)
        rng = np.random.default_rng(91)
        for width in (1, 6, 7, 13):
            with self.subTest(width=width):
                current = cp.asarray(
                    rng.normal(size=(dimension, width)), dtype=cp.float64
                )
                previous = cp.asarray(
                    rng.normal(size=(dimension, width)), dtype=cp.float64
                )
                ordinary = device_matrix @ current + local[:, None] * current
                actual_action = stencil.apply(current, local)
                actual_recurrence = stencil.chebyshev_recurrence(
                    current,
                    local,
                    center=0.37,
                    scale=0.81,
                    sigma_next=-0.63,
                    previous=previous,
                    sigma=0.22,
                )
                expected_recurrence = (
                    (ordinary - 0.37 * current) * 0.81 - 0.22 * previous
                ) * -0.63
                cp.cuda.get_current_stream().synchronize()
                np.testing.assert_allclose(
                    cp.asnumpy(actual_action),
                    cp.asnumpy(ordinary),
                    rtol=2e-14,
                    atol=2e-14,
                )
                np.testing.assert_allclose(
                    cp.asnumpy(actual_recurrence),
                    cp.asnumpy(expected_recurrence),
                    rtol=2e-14,
                    atol=2e-14,
                )
        self.assertEqual(stencil.slot_count, 5)
        self.assertEqual(stencil.palette_size, 3)

    def test_hamiltonian_fused_recurrence_includes_nonlocal_action(self) -> None:
        cp, _ = require_cupy()
        dimension = 4097
        matrix = sp.diags(
            (-np.ones(dimension - 1), 2.0 * np.ones(dimension), -np.ones(dimension - 1)),
            (-1, 0, 1),
            format="csr",
        )
        local = np.linspace(-0.4, 0.6, dimension)
        rows = np.array([3, 4, 5, 2000, 2001, 2002])
        columns = np.array([0, 0, 0, 1, 1, 1])
        values = np.array([0.3, -0.2, 0.1, -0.4, 0.5, 0.2])
        projectors = sp.csr_matrix(
            (values, (rows, columns)), shape=(dimension, 2)
        )
        hamiltonian = CuPyHamiltonian(
            matrix,
            local,
            (projectors, np.array([1.0, -1.0])),
        )
        self.assertIsInstance(
            hamiltonian.compact_finite_difference,
            CuPyStencilMajorFiniteDifference,
        )
        rng = np.random.default_rng(117)
        current = cp.asarray(rng.normal(size=(dimension, 6)))
        previous = cp.asarray(rng.normal(size=(dimension, 6)))
        ordinary = hamiltonian.apply(current)
        expected = ((ordinary - 0.2 * current) * 0.75 - 0.31 * previous) * 0.67
        actual = hamiltonian.chebyshev_recurrence(
            current,
            center=0.2,
            scale=0.75,
            sigma_next=0.67,
            previous=previous,
            sigma=0.31,
        )
        cp.cuda.get_current_stream().synchronize()
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            rtol=2e-14,
            atol=2e-14,
        )


@unittest.skipUnless(
    "CALDAEvaluator" in NATIVE_KERNELS,
    "native extension does not provide CALDAEvaluator",
)
class NativeCALDATests(unittest.TestCase):
    def test_native_ca_lda_matches_vectorized_reference(self) -> None:
        rng = np.random.default_rng(227)
        valence = np.abs(rng.normal(size=200_003))
        valence[::997] = 0.0
        core = 0.02 * np.abs(rng.normal(size=valence.size))
        volume = 0.008
        expected = ca_lda(valence, volume, core)
        actual = NativeCALDAEvaluator(core, volume)(valence)
        np.testing.assert_allclose(actual.potential, expected.potential, rtol=0, atol=3e-15)
        np.testing.assert_allclose(
            actual.energy_per_electron,
            expected.energy_per_electron,
            rtol=0,
            atol=3e-15,
        )
        np.testing.assert_allclose(
            actual.energy_density,
            expected.energy_density,
            rtol=0,
            atol=2e-14,
        )
        self.assertAlmostEqual(actual.total_energy, expected.total_energy, places=11)

    def test_native_ca_lda_weighted_wedge_matches_full_grid(self) -> None:
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.8,
                radius=2.5,
                expansion_order=4,
                shift=(0.5, 0.5, 0.5),
            )
        )
        reduction = AxisReflectionReduction.detect(
            grid, (Atom("H", (0.0, 0.0, 0.0)),)
        )
        reducer = SymmetrySCFReducer(reduction)
        rng = np.random.default_rng(229)
        valence = reducer.field(
            np.abs(rng.normal(size=reduction.wedge_size))
        )
        core = reducer.field(
            0.02 * np.abs(rng.normal(size=reduction.wedge_size))
        )
        expected = ca_lda(
            reducer.to_full(valence), grid.volume_element, reducer.to_full(core)
        )
        actual = NativeCALDAEvaluator(
            core, grid.volume_element, reducer=reducer
        )(valence)
        self.assertIsInstance(actual.potential, SymmetryScalarField)
        np.testing.assert_allclose(
            reducer.to_full(actual.potential),
            expected.potential,
            rtol=0.0,
            atol=3.0e-15,
        )
        np.testing.assert_allclose(
            reducer.to_full(actual.energy_per_electron),
            expected.energy_per_electron,
            rtol=0.0,
            atol=3.0e-15,
        )
        np.testing.assert_allclose(
            reducer.to_full(actual.energy_density),
            expected.energy_density,
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertAlmostEqual(actual.total_energy, expected.total_energy, places=12)


@unittest.skipUnless(
    "RadialGridEvaluator" in NATIVE_KERNELS,
    "native extension does not provide RadialGridEvaluator",
)
class NativeIonicSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package_root = Path(__file__).resolve().parents[2]
        potential_path = (
            package_root
            / "benchmarks"
            / "0d_naphthalene"
            / "C_POTRE.DAT"
        )
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.55, radius=3.4, expansion_order=8)
        )
        cls.atoms = (Atom("C", (0.1, -0.2, 0.05)),)
        cls.potentials = load_pseudopotentials(
            {"C": SpeciesPotential(potential_path, 1)}
        )
        cls.native = NativeIonicBuilders()

    def _compare(self, *, use_spline: bool) -> None:
        specifications = {
            "C": SpeciesPotential(
                self.potentials["C"].source,
                1,
                read_valence_density=False,
                use_spline=use_spline,
            )
        }
        expected_local = build_local_ionic_potential(
            self.grid, self.atoms, self.potentials, specifications
        )
        actual_local = self.native.build_local_ionic_potential(
            self.grid, self.atoms, self.potentials, specifications
        )
        np.testing.assert_allclose(actual_local, expected_local, rtol=0, atol=5e-15)

        for core in (False, True):
            expected_density = superpose_atomic_density(
                self.grid,
                self.atoms,
                self.potentials,
                specifications,
                core=core,
            )
            actual_density = self.native.superpose_atomic_density(
                self.grid,
                self.atoms,
                self.potentials,
                specifications,
                core=core,
            )
            np.testing.assert_allclose(
                actual_density, expected_density, rtol=0, atol=5e-15
            )

        expected_projectors = build_nonlocal_projectors(
            self.grid, self.atoms, self.potentials, specifications
        )
        actual_projectors = self.native.build_nonlocal_projectors(
            self.grid, self.atoms, self.potentials, specifications
        )
        self.assertEqual(actual_projectors.labels, expected_projectors.labels)
        np.testing.assert_array_equal(
            actual_projectors.signs, expected_projectors.signs
        )
        self.assertEqual(
            actual_projectors.projectors.nnz,
            expected_projectors.projectors.nnz,
        )
        difference = (
            actual_projectors.projectors - expected_projectors.projectors
        )
        if difference.nnz:
            self.assertLessEqual(np.max(np.abs(difference.data)), 5e-15)

    def test_linear_radial_strategy_matches_reference(self) -> None:
        self._compare(use_spline=False)

    def test_parsec_spline_strategy_matches_reference(self) -> None:
        self._compare(use_spline=True)


if __name__ == "__main__":
    unittest.main()
