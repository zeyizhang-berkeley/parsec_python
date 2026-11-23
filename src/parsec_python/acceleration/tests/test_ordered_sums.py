"""Products in a fixed order of summation on the device (``PARSEC_ORDERED_SUMS``).

cuSPARSE's ``csr @ dense`` gives other last bits from call to call, which a slowly converging SCF amplifies into
other iteration counts.  :class:`parsec_python.acceleration.Hamiltonian.complex_csr_device.OrderedCSR` adds the terms
of every entry in a fixed order; the device operators of the k-point and band solvers take it under ``auto``.  The
tests check its products against SciPy, that repeated products are equal bit for bit, the switch, and that a device
Bloch operator repeats its result bit for bit and equals the host operator.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.Hamiltonian.complex_csr_device import ordered_sums_setting

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_ORDERED_SUMS", None)
            self.assertEqual(ordered_sums_setting(), "auto")
        for value in ("auto", "off", " OFF "):
            with mock.patch.dict(os.environ, {"PARSEC_ORDERED_SUMS": value}):
                self.assertEqual(ordered_sums_setting(), value.strip().lower())
        with mock.patch.dict(os.environ, {"PARSEC_ORDERED_SUMS": "on"}):
            with self.assertRaises(ValueError):
                ordered_sums_setting()


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceTests(unittest.TestCase):
    def test_products_equal_scipy_and_repeat_bit_for_bit(self) -> None:
        import cupy as cp
        import cupyx.scipy.sparse as cps
        import scipy.sparse as sp

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import OrderedCSR

        rng = np.random.default_rng(3)
        # rows of a stencil's length (square: a diagonal and an accumulated product), long rows (the adjoint of the
        # projectors), short rows (the projectors)
        for rows, columns, density in ((4000, 4000, 49 / 4000), (300, 20000, 0.1), (20000, 300, 0.01)):
            for complex_matrix in (False, True):
                host = sp.random(rows, columns, density=density, format="csr", random_state=1)
                if complex_matrix:
                    host = (host + 1j * sp.random(rows, columns, density=density, format="csr", random_state=2)).tocsr()
                matrix = OrderedCSR(cps.csr_matrix(host))
                self.assertEqual(matrix.long_rows, rows < columns)
                self.assertEqual(matrix.nnz, host.nnz)
                for complex_block in (False, True):
                    block = rng.normal(size=(columns, 7))
                    if complex_block:
                        block = block + 1j * rng.normal(size=(columns, 7))
                    expected = host @ block
                    scale = float(np.max(np.abs(expected)))
                    label = f"{rows}x{columns} complex matrix {complex_matrix} complex block {complex_block}"
                    device = cp.asarray(block)
                    first = matrix @ device
                    self.assertTrue(first.flags.c_contiguous, label)
                    np.testing.assert_allclose(cp.asnumpy(first), expected, rtol=0, atol=1.0e-14 * scale, err_msg=label)
                    for _ in range(20):
                        self.assertTrue(bool(cp.array_equal(matrix @ device, first)), label)
                    # a column-major block and a vector
                    np.testing.assert_allclose(
                        cp.asnumpy(matrix @ cp.asfortranarray(device)), expected, rtol=0, atol=1.0e-14 * scale,
                        err_msg=label,
                    )
                    vector = block[:, 0]
                    np.testing.assert_allclose(
                        cp.asnumpy(matrix @ cp.asarray(vector)), host @ vector, rtol=0, atol=1.0e-14 * scale,
                        err_msg=label,
                    )
                    if rows == columns:
                        diagonal = rng.normal(size=rows)
                        got = matrix.product(device, diagonal=cp.asarray(diagonal))
                        np.testing.assert_allclose(
                            cp.asnumpy(got), expected + diagonal[:, None] * block, rtol=0, atol=1.0e-14 * scale,
                            err_msg=label,
                        )
                        target = cp.array(device, dtype=first.dtype)
                        matrix.product(device, target, accumulate=True)
                        np.testing.assert_allclose(
                            cp.asnumpy(target), block + expected, rtol=0, atol=1.0e-14 * scale, err_msg=label
                        )

    def test_the_backend_follows_the_switch(self) -> None:
        import cupyx.scipy.sparse as cps
        import scipy.sparse as sp

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import OrderedCSR
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        host = sp.random(50, 60, density=0.2, format="csr", random_state=4)
        backend = CupyArrayBackend(0)
        with mock.patch.dict(os.environ, {"PARSEC_ORDERED_SUMS": "auto"}):
            self.assertIsInstance(backend.matrix(host, dtype=np.float64), OrderedCSR)
        with mock.patch.dict(os.environ, {"PARSEC_ORDERED_SUMS": "off"}):
            self.assertIsInstance(backend.matrix(host, dtype=np.float64), cps.csr_matrix)

    def test_a_device_bloch_operator_repeats_itself(self) -> None:
        import cupy as cp

        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = silicon("pbe", spacing=0.855)
            system = prepare_periodic_single_point(problem)
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        rng = np.random.default_rng(5)
        for k_point in (np.zeros(3), np.array([0.11, 0.07, 0.05])):
            projectors = build_nonlocal_projectors(
                system.grid, system.cell_atoms, system.pseudopotentials, problem.pseudopotentials, lattice,
                k_point=k_point, periodic_dimensions=problem.periodic_cell.dimensions,
            )
            host = KPointKohnShamHamiltonian(
                system.negative_laplacian, system.gradient, system.ionic_potential, projectors, k_point
            )
            block = rng.normal(size=(host.shape[0], 6)) + 1j * rng.normal(size=(host.shape[0], 6))
            expected = host.apply(block)
            for fused in (False, True):  # the sparse Laplacian and k.grad, or the stencil kernel
                with mock.patch.dict(os.environ, {"PARSEC_ORDERED_SUMS": "auto"}):
                    converter = DeviceBlochConverter(CupyArrayBackend(0))
                    converter.fused_stencil = fused
                    operator = converter.operator(host, real=False)
                    first = operator.apply(cp.asarray(block))
                    label = f"k {k_point} fused {fused}"
                    np.testing.assert_allclose(
                        cp.asnumpy(first), expected, rtol=0, atol=1.0e-12 * float(np.max(np.abs(expected))),
                        err_msg=label,
                    )
                    for _ in range(10):
                        self.assertTrue(bool(cp.array_equal(operator.apply(cp.asarray(block)), first)), label)


if __name__ == "__main__":
    unittest.main()
