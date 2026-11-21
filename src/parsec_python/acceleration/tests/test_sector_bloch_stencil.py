"""The kinetic term of a sector of a k-point as a gathered stencil (``acceleration/Hamiltonian/sector_bloch_stencil.py``).

The stencils read off the matrices of the whole grid are those of ``AxialStencils.from_matrices`` (and nothing for a
matrix that is not one), the table of every sector is the reduced complex CSR of before to round-off (silicon with
its 192 operations, diamond with orbits of several sizes, both kinetic forms, general and zone-boundary k), the
host stencil takes a step of the recurrence, ``PARSEC_KPOINT_SECTOR_STENCIL=off`` keeps the CSR and a bad value is
refused.  On the device the kernel gives the host's numbers, the operator of a sector its stencil and the fused
recurrence, and the filters of one device the same subspace as the CSR route.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.Hamiltonian import sector_bloch_stencil as sbs
from parsec_python.acceleration.Hamiltonian.bloch_stencil import AxialStencils, grid_shape_of
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.tests.test_kpoint_sectors import (
    SectorOperatorTests,
    _decomposition,
    _silicon_system,
)

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

_ROOT = Path(__file__).resolve().parents[4]
_DIAMOND = _ROOT / "examples" / "3d_diamond_benchmark"


def _helper():
    problem, system = _silicon_system()
    helper = SectorOperatorTests()
    helper.problem, helper.system = problem, system
    helper.cell = sectors.CellGroup(system.grid, system.cell_atoms)
    helper.side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
    return helper


def _both_routes(k_point, decomposition, full):
    """The sector Hamiltonians of both settings of the switch."""
    found = {}
    for setting in ("off", "auto"):
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": setting}):
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
            found[setting] = [split.hamiltonian(sector, full) for sector in range(decomposition.representation_count)]
    return found


def _difference(csr, table) -> float:
    csr = sp.csr_matrix(csr)
    difference = csr - table.to_csr()
    largest = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    return largest / float(np.max(np.abs(csr.data)))


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_KPOINT_SECTOR_STENCIL", None)
            self.assertEqual(sbs.sector_stencil_setting(), "auto")
        for value, expected in (("off", "off"), (" AUTO ", "auto"), ("", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": value}):
                self.assertEqual(sbs.sector_stencil_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": "csr"}):
            with self.assertRaises(ValueError):
                sbs.sector_stencil_setting()

    def test_a_bad_value_stops_the_solver_set_up(self) -> None:
        # read before the early returns of sector_solver: also a run whose k-points are not split stops on it
        from types import SimpleNamespace

        job = SimpleNamespace(geometry=None, extras=None, extras_by_columns=False, states_per_task=None)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": "csr", "PARSEC_KPOINT_SECTORS": "off"}):
            with self.assertRaises(ValueError):
                sectors.sector_solver(job, None)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": "off", "PARSEC_KPOINT_SECTORS": "off"}):
            self.assertIsNone(sectors.sector_solver(job, None))


class TableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _helper()

    def test_the_stencils_of_the_whole_grid(self) -> None:
        system = self.helper.system
        gradient = tuple(system.gradient)
        shape = grid_shape_of(gradient, system.grid.size)
        expected = AxialStencils.from_matrices(shape, system.negative_laplacian, gradient)
        found = sbs.checked_axial_stencils(system.negative_laplacian, gradient)
        self.assertEqual((found.shape, found.half_width, found.centre), (expected.shape, expected.half_width, expected.centre))
        np.testing.assert_array_equal(found.laplacian, expected.laplacian)
        np.testing.assert_array_equal(found.gradient, expected.gradient)
        # one entry of one row changed, or an entry off the axes: no stencil
        changed = sp.csr_matrix(system.negative_laplacian, copy=True)
        changed.data[changed.indptr[700] + 2] *= 1.0 + 1.0e-15
        self.assertIsNone(sbs.checked_axial_stencils(changed, gradient))
        bent = sp.csr_matrix(gradient[0], copy=True)
        bent = (bent + sp.csr_matrix(([1.0e-3], ([5], [5 + 13])), shape=bent.shape)).tocsr()
        self.assertIsNone(sbs.checked_axial_stencils(system.negative_laplacian, (bent, gradient[1], gradient[2])))

    def test_the_tables_are_the_reduced_matrices(self) -> None:
        helper = self.helper
        side = helper.side
        for k_point in (np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0]), np.array([0.11, 0.07, 0.05])):
            _, _, decomposition = _decomposition(helper.cell, k_point)
            for kinetic in ("gradient", "bloch"):
                full = helper._hamiltonian(k_point, kinetic)
                found = _both_routes(k_point, decomposition, full)
                rng = np.random.default_rng(4)
                for sector, (old, new) in enumerate(zip(found["off"], found["auto"])):
                    self.assertIsInstance(old.bloch_laplacian, sp.csr_matrix)
                    self.assertIsInstance(new.bloch_laplacian, sbs.SectorStencilTable)
                    self.assertLess(_difference(old.bloch_laplacian, new.bloch_laplacian), 1.0e-14)
                    block = rng.normal(size=(old.shape[0], 3)) + 1j * rng.normal(size=(old.shape[0], 3))
                    expected = old.apply(block)
                    np.testing.assert_allclose(
                        new.apply(block), expected, rtol=0, atol=1.0e-13 * float(np.max(np.abs(expected))),
                        err_msg=f"{k_point} {kinetic} sector {sector}",
                    )

    def test_the_host_product(self) -> None:
        # table @ block on the host: one scipy product of the matrix of the table, made on the first product and kept
        helper = self.helper
        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(helper.cell, k_point)
        full = helper._hamiltonian(k_point, "gradient")
        table = _both_routes(k_point, decomposition, full)["auto"][1].bloch_laplacian
        self.assertIsInstance(table, sbs.SectorStencilTable)
        self.assertIsNone(table._csr)
        rng = np.random.default_rng(9)
        block = rng.normal(size=(table.shape[0], 4)) + 1j * rng.normal(size=(table.shape[0], 4))
        got = table @ block
        matrix = table._csr
        self.assertIsInstance(matrix, sp.csr_matrix)
        np.testing.assert_array_equal(got, matrix @ block)
        np.testing.assert_array_equal(table @ block[:, 0], matrix @ block[:, 0])
        self.assertIs(table._csr, matrix)
        # the same term as the slot sums of the kernel
        slots = table.centre * block
        for slot in range(table.slots):
            slots = slots + table.factors(slot)[:, None] * block[table.columns[slot]]
        np.testing.assert_allclose(got, slots, rtol=0, atol=1.0e-13 * float(np.max(np.abs(slots))))

    @unittest.skipUnless((_DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
    def test_diamond_orbits_of_several_sizes(self) -> None:
        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors

        directory = Path(tempfile.mkdtemp(prefix="parsec_sstencil_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        subprocess.run(
            [sys.executable, str(_DIAMOND / "make_diamond.py"), "1", "1", "1", str(directory / "parsec.in"),
             "--kmesh", "3", "3", "3"],
            check=True, capture_output=True,
        )
        (directory / "C_POTRE.DAT").write_bytes((_DIAMOND / "C_POTRE.DAT").read_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(directory / "parsec.in").problem
            system = prepare_periodic_single_point(problem)
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        palettes = set()
        for fraction in ((0.0, 0.0, 0.0), (1.0 / 3.0, 0.0, 0.0), (0.5, 0.0, 0.0)):
            k_point = 2.0 * np.pi * np.asarray(fraction) / lattice[0, 0]
            _, _, decomposition = _decomposition(cell, k_point)
            projectors = build_nonlocal_projectors(
                system.grid, system.cell_atoms, system.pseudopotentials, problem.pseudopotentials, lattice,
                k_point=k_point, periodic_dimensions=problem.periodic_cell.dimensions,
            )
            for kinetic in ("gradient", "bloch"):
                bloch = build_bloch_negative_laplacian(system.grid, k_point, "u") if kinetic == "bloch" else None
                full = KPointKohnShamHamiltonian(
                    system.negative_laplacian, system.gradient, system.ionic_potential, projectors, k_point,
                    bloch_laplacian=bloch,
                )
                found = _both_routes(k_point, decomposition, full)
                for old, new in zip(found["off"], found["auto"]):
                    palettes.add(len(new.bloch_laplacian.palette))
                    self.assertLess(_difference(old.bloch_laplacian, new.bloch_laplacian), 1.0e-14)
        # factors sqrt(m_r / m_j) other than one: orbits of several sizes
        self.assertGreater(max(palettes), 2)

    def test_a_cell_of_unequal_steps(self) -> None:
        # a stretched diamond cube (0.641 / 0.641 / 0.667 bohr): one set of stencil values per axis; its sectors
        # (the involutions of D2d about the origin) against the reduced matrices of the gradient form
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.Laplacian.finite_difference import build_gradient
        from parsec_python.acceleration.tests.test_periodic_unequal_steps import _grid, _stretched_diamond

        sides = (6.41, 6.41, 6.67)
        grid = _grid(sides, 0.65)
        atoms = _stretched_diamond(sides)
        laplacian, gradient = build_negative_laplacian(grid), build_gradient(grid)
        stencils = sbs.checked_axial_stencils(laplacian, gradient)
        self.assertIsNotNone(stencils)
        self.assertNotEqual(stencils.laplacian[0, 0], stencils.laplacian[2, 0])
        cell = sectors.CellGroup(grid, atoms)
        k_point = np.array([0.0, 0.0, 0.21])
        _, _, decomposition = _decomposition(cell, k_point)
        self.assertGreater(decomposition.representation_count, 1)
        centre, weights = sbs.kinetic_weights(stencils, k_point)
        matrix = sp.csr_matrix(laplacian, dtype=np.complex128) - 2.0j * float(k_point[2]) * sp.csr_matrix(gradient[2])
        matrix = matrix + float(k_point @ k_point) * sp.identity(grid.size, dtype=np.complex128, format="csr")
        reduced = decomposition.reduce_hermitian_operators(matrix)
        for sector in range(decomposition.representation_count):
            table = sbs.build_sector_table(decomposition, sector, stencils.shape, stencils.half_width, centre, weights)
            self.assertLess(_difference(reduced[sector], table), 1.0e-14)

    def test_the_recurrence_step_on_the_host(self) -> None:
        helper = self.helper
        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(helper.cell, k_point)
        full = helper._hamiltonian(k_point, "gradient")
        host = _both_routes(k_point, decomposition, full)["auto"][1]
        stencil = sbs.HostSectorStencil(host.bloch_laplacian)
        rng = np.random.default_rng(2)
        block = rng.normal(size=(host.shape[0], 3)) + 1j * rng.normal(size=(host.shape[0], 3))
        previous = rng.normal(size=block.shape) + 0j
        added = 0.01 * block
        before, after = np.array([0.1, 0.2, 0.3]), np.array([1.1, 0.9, 0.7])
        got = stencil.apply(
            block, stencil.diagonal(host._diagonal), added=added, previous=previous, center=1.3, scale=0.7,
            before=before, after=after,
        )
        local = host.bloch_laplacian @ block + host._diagonal[:, None] * block
        expected = (((local + added) - 1.3 * block) * 0.7 - before[None, :] * previous) * after[None, :]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1.0e-13 * float(np.max(np.abs(expected))))

    def test_the_numpy_device_operator(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend

        helper = self.helper
        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(helper.cell, k_point)
        full = helper._hamiltonian(k_point, "bloch")
        rng = np.random.default_rng(8)
        for host in _both_routes(k_point, decomposition, full)["auto"]:
            operator = DeviceBlochConverter(NumpyArrayBackend()).operator(host)
            self.assertIsInstance(operator.stencil, sbs.HostSectorStencil)
            self.assertTrue(operator.fused_sector_recurrence)
            block = rng.normal(size=(host.shape[0], 4)) + 1j * rng.normal(size=(host.shape[0], 4))
            expected = host.apply(block)
            np.testing.assert_allclose(
                operator.apply(block), expected, rtol=0, atol=1.0e-13 * float(np.max(np.abs(expected)))
            )


class CPURouteTests(unittest.TestCase):
    """The solvers on the CPU keep the reduced CSR of the kinetic term (the host applies it in one scipy product, 3-4
    times faster than the slots of a table); the device code (CuPy or its host double numpy-device) takes the table."""

    def test_the_backend_decides(self) -> None:
        self.assertFalse(sectors.sector_operators_on_device("cpu"))
        self.assertTrue(sectors.sector_operators_on_device("numpy-device"))
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu", "PARSEC_KPOINT_BACKEND": ""}):
            self.assertFalse(sectors.sector_operators_on_device())
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "numpy-device"}):
            self.assertTrue(sectors.sector_operators_on_device())

    def test_the_route_follows_the_backend(self) -> None:
        from parsec_python.SCF.kpoint_solver import KPointJob, SerialKPointSolver

        helper = _helper()
        problem, system = helper.problem, helper.system
        k_point = np.pi / helper.side * np.array([1.0, 0.0, 0.0])
        geometry = {
            "grid": system.grid,
            "atoms": tuple(system.cell_atoms),
            "k_points": k_point[None, :],
            "weights": np.ones(1),
            "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
        }
        job = KPointJob(
            n_tasks=1, number_of_states=22, eigval_settings=lambda *args: None,
            make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {}, geometry=geometry,
        )

        def plain(inner, **options):
            return SerialKPointSolver(inner)

        full = helper._hamiltonian(k_point, "gradient")
        found = {}
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0):
            for backend, setting in (("cpu", "auto"), ("numpy-device", "auto"), ("numpy-device", "off")):
                with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_STENCIL": setting}):
                    solver = sectors.sector_solver(job, plain, problem_size=system.grid.size, backend=backend)
                    self.assertIsInstance(solver, sectors.SectorKPointSolver)
                    plan = solver._plans[0]
                    self.assertGreater(plan.order, 1)
                    self.assertEqual(plan.device_operators, (backend, setting) == ("numpy-device", "auto"))
                    found[backend, setting] = [
                        plan.hamiltonian(sector, full).bloch_laplacian for sector in range(plan.order)
                    ]
        for table in found["numpy-device", "auto"]:
            self.assertIsInstance(table, sbs.SectorStencilTable)
        # on the CPU the matrices of PARSEC_KPOINT_SECTOR_STENCIL=off, bit for bit
        for kept, csr in zip(found["cpu", "auto"], found["numpy-device", "off"]):
            self.assertIsInstance(kept, sp.csr_matrix)
            for part in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(kept, part), getattr(csr, part))


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _helper()

    def test_the_kernel_and_the_operator_of_a_sector(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import DeviceComplexCSR
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter

        helper = self.helper
        rng = np.random.default_rng(5)
        for k_point in (np.zeros(3), np.array([0.11, 0.07, 0.05])):
            _, _, decomposition = _decomposition(helper.cell, k_point)
            for kinetic in ("gradient", "bloch"):
                full = helper._hamiltonian(k_point, kinetic)
                found = _both_routes(k_point, decomposition, full)
                for sector in range(decomposition.representation_count):
                    old, new = found["off"][sector], found["auto"][sector]
                    converter = DeviceBlochConverter(CupyArrayBackend(0))
                    off, auto = converter.operator(old), converter.operator(new)
                    self.assertIsInstance(off.laplacian, DeviceComplexCSR)
                    self.assertIsNone(off.stencil)
                    self.assertIsInstance(auto.stencil, sbs.DeviceSectorStencil)
                    self.assertFalse(off.fused_recurrence)
                    self.assertTrue(auto.fused_sector_recurrence)
                    block = rng.normal(size=(old.shape[0], 37)) + 1j * rng.normal(size=(old.shape[0], 37))
                    expected = old.apply(block)
                    scale = float(np.max(np.abs(expected)))
                    np.testing.assert_allclose(
                        cp.asnumpy(auto.apply(cp.asarray(block))), expected, rtol=0, atol=1.0e-13 * scale,
                        err_msg=f"{k_point} {kinetic} sector {sector}",
                    )
                    # one step of the recurrence in the kernel against the operator's products and passes
                    current, previous = cp.asarray(block), cp.asarray(0.5 * block[::-1].copy())
                    before = cp.asarray(rng.uniform(0.1, 0.9, 37))
                    after = cp.asarray(rng.uniform(0.5, 1.5, 37))
                    step = auto.chebyshev_step(current, previous, 1.7, 0.3, before, after)
                    reference = ((off.apply(current) - 1.7 * current) * 0.3 - before[None, :] * previous) * after[None, :]
                    np.testing.assert_allclose(
                        cp.asnumpy(step), cp.asnumpy(reference), rtol=0,
                        atol=1.0e-13 * float(cp.max(cp.abs(reference))),
                    )

    def test_the_filter_of_one_device(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers import chebff_spinor as cs
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter

        helper = self.helper
        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(helper.cell, k_point)
        full = helper._hamiltonian(k_point, "gradient")
        found = _both_routes(k_point, decomposition, full)
        converter = DeviceBlochConverter(CupyArrayBackend(0))
        operators = {setting: converter.operator(found[setting][0]) for setting in ("off", "auto")}
        rows = int(operators["off"].shape[0])
        # (the counting wrapper takes the single Lanczos vectors as one-column blocks, as in the solver)
        bounding = cs._CountingOperator(operators["off"], cs._Stats(None))
        upper, smallest = cs._lanczos_upper_bound(cp, bounding, rows, 20, np.random.default_rng(3))
        lower = (2.0 * smallest + upper) / 3.0
        half, center = 0.5 * (upper - lower), 0.5 * (upper + lower)
        sigma_one = half / (smallest - center)
        degree = 12
        entering, produced, _ = cs._chain(sigma_one, sigma_one, degree)
        columns = 40
        shape = (len(entering), columns)
        rng = np.random.default_rng(6)
        block = cp.asarray(rng.uniform(-1, 1, (rows, columns)) + 1j * rng.uniform(-1, 1, (rows, columns)))
        results = {}
        for setting, operator in operators.items():
            stats = cs._Stats(None)
            counted = cs._CountingOperator(operator, stats)
            out = cp.empty_like(block)
            cs._filter_group(
                cp, counted, block, out, slice(0, columns), degree, center, half, sigma_one,
                np.broadcast_to(np.asarray(entering)[:, None], shape),
                np.broadcast_to(np.asarray(produced)[:, None], shape), 16,
            )
            results[setting] = (cp.asnumpy(out), dict(stats.values))
        (old, old_stats), (new, new_stats) = results["off"], results["auto"]
        np.testing.assert_allclose(new, old, rtol=0, atol=1.0e-11 * float(np.max(np.abs(old))))
        self.assertEqual(old_stats, new_stats)  # the same applications and columns counted


if __name__ == "__main__":
    unittest.main()
