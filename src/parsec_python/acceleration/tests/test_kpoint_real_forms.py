"""Real forms of the sectors of a k-point kept by an antiunitary ``P_g K`` (``acceleration/SCF/kpoint_real_forms.py``,
``PARSEC_KPOINT_SECTOR_REAL``).

CPU: ``T`` is unitary and ``T^H H_s T`` real with the spectrum of the sector, the real projector parts and kinetic
matrix are its terms, the paired stencil of the host is ``T^H K T``; the SCF of the benchmark's reduced diamond mesh on
the device code of the host (``numpy-device``) keeps its energy, density and nonlocal forces, with and without the
classes of ``PARSEC_KPOINT_SECTOR_CLASSES``.  GPU: the paired stencil kernel gives the host's numbers, recurrence
included, and the SCF on CuPy keeps the energy and forces.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.Hamiltonian.sector_bloch_stencil import (
    build_sector_table,
    checked_axial_stencils,
    kinetic_weights,
)
from parsec_python.acceleration.Hamiltonian.sector_pair_stencil import HostPairStencil, PairStencilTable
from parsec_python.acceleration.SCF import kpoint_real_forms as forms
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.tests import test_kpoint_sectors as base

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


class _Admitted:
    def __call__(self, index) -> bool:
        return True

    def in_density_group(self, index) -> bool:
        return True


def setUpModule() -> None:
    patcher = mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "off"), ("OFF", "off"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_REAL": value}):
                self.assertEqual(forms.sector_real_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_REAL": "on"}):
            with self.assertRaises(ValueError):
                forms.sector_real_setting()
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches()


class RealFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = base._silicon_system()
        cls.cell = sectors.CellGroup(cls.system.grid, cls.system.cell_atoms)
        cls.side = float(np.asarray(cls.problem.periodic_cell.lattice_vectors)[0, 0])
        cls.helper = base.SectorOperatorTests()
        cls.helper.system, cls.helper.problem = cls.system, cls.problem
        cls.stencils = checked_axial_stencils(cls.system.negative_laplacian, cls.system.gradient)

    def test_the_real_form_of_a_sector(self) -> None:
        rng = np.random.default_rng(5)
        seen = 0
        for k_point in (
            2.0 * np.pi / self.side * np.array([0.0, 0.0, 1.0 / 3.0]),
            2.0 * np.pi / self.side * np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            2.0 * np.pi / self.side * np.array([0.25, 0.25, 0.5]),
            np.array([0.11, 0.07, 0.05]),
        ):
            _, elements, decomposition = base._decomposition(self.cell, k_point)
            found = forms.sector_real_forms(
                self.cell, elements, decomposition, k_point, _Admitted(), range(decomposition.representation_count)
            )
            self.assertIsNotNone(found, k_point)
            full = self.helper._hamiltonian(k_point, "gradient")
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
            centre, weights = kinetic_weights(self.stencils, k_point)
            for sector, form in found.items():
                seen += 1
                dense = base._dense(split.hamiltonian(sector, full))
                unitary = form.adjoint_matrix().conj().T.toarray()
                np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(form.size), atol=1.0e-14)
                real = unitary.conj().T @ dense @ unitary
                self.assertLess(float(np.max(np.abs(real.imag))), 1.0e-12)
                np.testing.assert_allclose(
                    np.linalg.eigvalsh(real.real), np.linalg.eigvalsh(dense), rtol=0, atol=1.0e-11
                )
                block = rng.normal(size=(form.size, 3))
                np.testing.assert_allclose(form.to_complex(block), unitary @ block, atol=1.0e-14)
                image = dense @ (unitary @ block)
                np.testing.assert_allclose(form.to_real(image, real=False), unitary.conj().T @ image, atol=1.0e-12)
                # the projector term and the kinetic term of the real form
                _, projectors, _ = split._static[sector]
                forward, adjoint, signs = forms.real_projector_parts(projectors.projectors, projectors.signs, form)
                matrix = projectors.projectors.toarray()
                expected = unitary.conj().T @ (matrix * projectors.signs) @ matrix.conj().T @ unitary
                np.testing.assert_allclose((forward.toarray() * signs) @ adjoint.toarray(), expected.real, atol=1.0e-13)
                table = build_sector_table(
                    decomposition, sector, self.stencils.shape, self.stencils.half_width, centre, weights
                )
                kinetic = unitary.conj().T @ table.to_csr().toarray() @ unitary
                pair = HostPairStencil(PairStencilTable(table, form))
                np.testing.assert_allclose(
                    pair.apply(block, pair.diagonal(np.zeros(form.size))), kinetic.real @ block, rtol=0, atol=1.0e-12
                )
                np.testing.assert_allclose(
                    forms.real_kinetic_matrix(table.to_csr(), form).toarray(), kinetic.real, rtol=0, atol=1.0e-13
                )
                # the Hamiltonian of the real form on the host
                split.real_forms = {sector: form}
                hamiltonian = split.hamiltonian(sector, full)
                split.real_forms = None
                np.testing.assert_allclose(hamiltonian.apply(block), real.real @ block, rtol=0, atol=1.0e-12)
        self.assertGreater(seen, 0)


def _diamond_mesh(test: unittest.TestCase):
    from parsec_python.Input.parsec_input import parse_parsec_input

    directory = Path(tempfile.mkdtemp(prefix="parsec_kreal_"))
    test.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
    subprocess.run(
        [sys.executable, str(base._DIAMOND / "make_diamond.py"), "1", "1", "1", str(directory / "parsec.in"),
         "--kmesh", "3", "3", "3"],
        check=True, capture_output=True,
    )
    (directory / "C_POTRE.DAT").write_bytes((base._DIAMOND / "C_POTRE.DAT").read_bytes())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = parse_parsec_input(directory / "parsec.in").problem
    return replace(
        problem,
        scf=replace(problem.scf, convergence_criterion=1.0e-9),
        eigensolver=replace(problem.eigensolver, tolerance=1.0e-10),
    )


def _both(problem, environment):
    from parsec_python.Grid.kpoints import kpoint_sampling
    from parsec_python.SCF.kpoints import run_scf_kpoints
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    results = {}
    for setting in ("off", "auto"):
        with mock.patch.dict(os.environ, dict(environment, PARSEC_KPOINT_SECTOR_REAL=setting)), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
            points, shares = kpoint_sampling(problem)
            results[setting] = run_scf_kpoints(system, points, shares, compute_forces=True)
    return results["off"], results["auto"]


@unittest.skipUnless((base._DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
class SCFTests(unittest.TestCase):
    def assert_same(self, off, auto) -> None:
        self.assertTrue(off.converged and auto.converged)
        self.assertNotIn("kpoint_sector_real_forms", off.timings.detail)
        self.assertGreater(auto.timings.detail["kpoint_sector_real_forms"], 0.0)
        self.assertAlmostEqual(float(auto.energies.total), float(off.energies.total), delta=1.0e-9)
        np.testing.assert_allclose(np.asarray(auto.density), np.asarray(off.density), rtol=0, atol=1.0e-8)
        self.assertGreater(float(np.max(np.abs(off.nonlocal_forces))), 1.0e-3)
        np.testing.assert_allclose(auto.nonlocal_forces, off.nonlocal_forces, rtol=0, atol=1.0e-7)

    def test_the_benchmark_mesh_in_real_forms(self) -> None:
        problem = _diamond_mesh(self)
        for classes in ("off", "auto"):
            environment = {
                "PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1",
                "PARSEC_KPOINT_SECTOR_CLASSES": classes,
            }
            self.assert_same(*_both(problem, environment))

    def test_the_cpu_route_stays_complex(self) -> None:
        problem = _diamond_mesh(self)
        off, auto = _both(problem, {"PARSEC_KPOINT_BACKEND": "cpu"})
        self.assertNotIn("kpoint_sector_real_forms", auto.timings.detail)
        self.assertEqual(float(auto.energies.total), float(off.energies.total))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceTests(unittest.TestCase):
    def test_the_kernel_gives_the_numbers_of_the_host(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.sector_pair_stencil import DevicePairStencil

        problem, system = base._silicon_system()
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        stencils = checked_axial_stencils(system.negative_laplacian, system.gradient)
        rng = np.random.default_rng(3)
        for k_point in (2.0 * np.pi / side * np.array([0.0, 0.0, 1.0 / 3.0]), np.array([0.11, 0.07, 0.05])):
            _, elements, decomposition = base._decomposition(cell, k_point)
            found = forms.sector_real_forms(
                cell, elements, decomposition, k_point, _Admitted(), range(decomposition.representation_count)
            )
            centre, weights = kinetic_weights(stencils, k_point)
            for sector, form in found.items():
                table = build_sector_table(decomposition, sector, stencils.shape, stencils.half_width, centre, weights)
                pairs = PairStencilTable(table, form)
                host, device = HostPairStencil(pairs), DevicePairStencil(cp, pairs)
                block = rng.normal(size=(form.size, 5))
                diagonal = form.symmetric_rows(rng.normal(size=form.size))
                added, previous = rng.normal(size=block.shape), rng.normal(size=block.shape)
                before, after = rng.normal(size=5), rng.normal(size=5)
                expected = host.apply(block, host.diagonal(diagonal))
                found_ = cp.asnumpy(device.apply(cp.asarray(block), device.diagonal(cp.asarray(diagonal))))
                np.testing.assert_allclose(found_, expected, rtol=0, atol=1.0e-12)
                expected = host.apply(
                    block, host.diagonal(diagonal), added=added, previous=previous, center=0.3, scale=1.7,
                    before=before, after=after,
                )
                found_ = cp.asnumpy(
                    device.apply(
                        cp.asarray(block), device.diagonal(cp.asarray(diagonal)), added=cp.asarray(added),
                        previous=cp.asarray(previous), center=0.3, scale=1.7, before=cp.asarray(before),
                        after=cp.asarray(after),
                    )
                )
                np.testing.assert_allclose(found_, expected, rtol=0, atol=1.0e-12)

    def test_the_real_csr_kernels(self) -> None:
        import cupy as cp
        import scipy.sparse as sp

        from parsec_python.acceleration.Hamiltonian.sector_pair_stencil import DeviceRealCSR

        rng = np.random.default_rng(11)
        for shape, density in (((300, 40), 0.1), ((40, 300), 0.9)):  # short rows and the long rows of an adjoint
            matrix = sp.random(*shape, density=density, random_state=3, format="csr")
            block = rng.normal(size=(shape[1], 7))
            found = cp.asnumpy(DeviceRealCSR(cp, matrix) @ cp.asarray(block))
            np.testing.assert_allclose(found, matrix @ block, rtol=0, atol=1.0e-12)

    @unittest.skipUnless((base._DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
    def test_the_scf_on_the_gpu(self) -> None:
        problem = _diamond_mesh(self)
        off, auto = _both(problem, {"PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_KPOINT_SECTOR_CLASSES": "auto"})
        SCFTests.assert_same(self, off, auto)


if __name__ == "__main__":
    unittest.main()
