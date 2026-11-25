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

    def test_the_long_row_switch(self) -> None:
        from parsec_python.acceleration.Hamiltonian.complex_csr_device import long_row_split_setting

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_LONG_ROW_SPLIT", None)
            self.assertEqual(long_row_split_setting(), "auto")
        for value in ("auto", "off", " OFF "):
            with mock.patch.dict(os.environ, {"PARSEC_LONG_ROW_SPLIT": value}):
                self.assertEqual(long_row_split_setting(), value.strip().lower())
        with mock.patch.dict(os.environ, {"PARSEC_LONG_ROW_SPLIT": "on"}):
            with self.assertRaises(ValueError):
                long_row_split_setting()

    def test_the_chunks_of_the_rows(self) -> None:
        """Every entry of a row in exactly one chunk, in order; a row of at most ``chunk`` entries is one chunk (an
        empty one too); no chunks where no row is longer than ``chunk``."""

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import row_chunks

        lengths = np.array([0, 5, 8, 9, 23, 1, 16, 0])
        indptr = np.concatenate(([0], np.cumsum(lengths)))
        begin, end, starts = row_chunks(indptr, chunk=8)
        self.assertEqual(starts.dtype, np.int32)
        np.testing.assert_array_equal(starts, [0, 1, 2, 3, 5, 8, 9, 11, 12])
        for row in range(lengths.size):
            pieces = list(zip(begin[starts[row]:starts[row + 1]], end[starts[row]:starts[row + 1]]))
            covered = [p for b, e in pieces for p in range(b, e)]
            self.assertEqual(covered, list(range(indptr[row], indptr[row + 1])))
            self.assertTrue(all(e - b <= 8 for b, e in pieces))
            if lengths[row] <= 8:
                self.assertEqual(len(pieces), 1)
        self.assertIsNone(row_chunks(np.array([0, 8, 16, 16]), chunk=8))
        self.assertIsNone(row_chunks(np.array([0]), chunk=8))


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

    def test_long_rows_in_chunks(self) -> None:
        """``PARSEC_LONG_ROW_SPLIT=auto`` against ``off`` and SciPy, for :class:`OrderedCSR` and
        :class:`DeviceComplexCSR`: the same product within round-off, the same bits in every repeat, the bits of
        ``off`` for rows of at most one chunk and for a matrix of no longer row, and accumulation."""

        import cupy as cp
        import cupyx.scipy.sparse as cps
        import scipy.sparse as sp

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import (
            LONG_ROW_CHUNK,
            DeviceComplexCSR,
            OrderedCSR,
        )

        rng = np.random.default_rng(11)
        columns = 40000
        # rows of no, short, one chunk exactly, one entry more and many chunks (the adjoint of filtered projectors)
        lengths = [0, 300, LONG_ROW_CHUNK, LONG_ROW_CHUNK + 1, 5000, 23000, 31000]
        pieces = []
        for length in lengths:
            row = np.zeros(columns)
            row[np.sort(rng.choice(columns, size=length, replace=False))] = rng.normal(size=length)
            pieces.append(row)
        dense = np.array(pieces)
        short = np.flatnonzero(np.array(lengths) <= LONG_ROW_CHUNK)
        for complex_matrix in (False, True):
            values = dense + (1j * dense * rng.normal(size=dense.shape) if complex_matrix else 0.0)
            host = sp.csr_matrix(values)
            for kind in ("ordered", "device"):
                if kind == "device" and not complex_matrix:
                    continue
                matrices = {}
                for split in ("off", "auto"):
                    with mock.patch.dict(os.environ, {"PARSEC_LONG_ROW_SPLIT": split}):
                        matrices[split] = (
                            OrderedCSR(cps.csr_matrix(host)) if kind == "ordered" else DeviceComplexCSR(cp, host)
                        )
                for count in (1, 9, 17):
                    block = rng.normal(size=(columns, count))
                    if complex_matrix or kind == "ordered":
                        block = block + 1j * rng.normal(size=(columns, count))
                    device = cp.asarray(block)
                    expected = host @ block
                    # the round-off of a sum of many terms scales with the sum of their moduli, not with the result
                    scale = float(np.max(abs(host) @ np.abs(block)))
                    label = f"{kind} complex {complex_matrix} columns {count}"
                    off = matrices["off"].product(device)
                    auto = matrices["auto"].product(device)
                    np.testing.assert_allclose(cp.asnumpy(auto), expected, rtol=0, atol=1.0e-14 * scale, err_msg=label)
                    np.testing.assert_allclose(cp.asnumpy(off), expected, rtol=0, atol=1.0e-14 * scale, err_msg=label)
                    self.assertTrue(bool(cp.array_equal(auto[short], off[short])), label)
                    for _ in range(5):
                        self.assertTrue(bool(cp.array_equal(matrices["auto"].product(device), auto)), label)
                    target = cp.array(auto)
                    matrices["auto"].product(device, target, accumulate=True)
                    np.testing.assert_allclose(cp.asnumpy(target), 2.0 * expected, rtol=0, atol=2.0e-14 * scale,
                                               err_msg=label)
        # no row longer than a chunk: the kernel of off and its bits
        host = sp.random(20, 30000, density=0.01, format="csr", random_state=6)
        host = (host + 1j * sp.random(20, 30000, density=0.01, format="csr", random_state=7)).tocsr()
        self.assertLessEqual(int(np.diff(host.indptr).max()), LONG_ROW_CHUNK)
        block = cp.asarray(rng.normal(size=(30000, 9)) + 1j * rng.normal(size=(30000, 9)))
        products = []
        for split in ("off", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_LONG_ROW_SPLIT": split}):
                products.append(OrderedCSR(cps.csr_matrix(host)).product(block))
                products.append(DeviceComplexCSR(cp, host).product(block))
        self.assertTrue(bool(cp.array_equal(products[0], products[2])))
        self.assertTrue(bool(cp.array_equal(products[1], products[3])))

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
