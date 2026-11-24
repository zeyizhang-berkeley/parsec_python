"""Sectors of the complex characters of an abelian little group of a k-point (``acceleration/Symmetry/complex_sectors.py``,
``PARSEC_KPOINT_SECTOR_CHARACTERS=complex``).

CPU: the plans of the default (the complex characters on the threefold axis of a vacancy cube, the involutions at
Gamma and on the z axis, none with the Hubbard term of LDA+U; ``real`` as before), the characters of an abelian
group, the union of the spectra of the sectors (the threefold axis of a vacancy
cube without the translations of the face-centred cell, a fourfold axis, the silicon cube) is the spectrum of the
k-point, the orbitals of a sector and its reduction are adjoint, the gathered stencil with a complex palette is the
reduced kinetic term, and the classes of the complex sectors are images of each other; the SCF of the diamond vacancy
cube on the reduced 3 x 3 x 3 mesh keeps its energy and forces on the device code of the host and on the CPU route.
GPU: the kernel of the complex palette gives the numbers of the host, and the SCF on CuPy keeps the energy.
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
import scipy.sparse as sp

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.Hamiltonian.sector_bloch_stencil import (
    HostSectorStencil,
    build_sector_table,
    checked_axial_stencils,
    kinetic_weights,
)
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.Symmetry import complex_sectors
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


def _vacancy_system():
    """The silicon cube of the tests with the atom at the origin taken out: Td about the vacancy, no translations."""

    from parsec_python.SCF.pbc import prepare_periodic_single_point
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = silicon("pbe", spacing=0.855)
        atoms = [atom for atom in problem.atoms if float(np.max(np.abs(np.asarray(atom.position)))) > 1.0e-8]
        problem = replace(problem, atoms=atoms)
        return problem, prepare_periodic_single_point(problem)


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "complex"), ("COMPLEX", "complex"), ("real", "real")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_CHARACTERS": value}):
                self.assertEqual(complex_sectors.sector_characters_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_CHARACTERS": "auto"}):
            with self.assertRaises(ValueError):
                complex_sectors.sector_characters_setting()
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches()


class DefaultPlanTests(unittest.TestCase):
    """The plans of the vacancy cube at Gamma, on the threefold axis and on the z axis: the default takes the complex
    characters of C3 on the threefold axis only, ``real`` the involutions everywhere (the axis then stays whole: its
    halves of Cs would not halve the work), and a loop with the Hubbard term of LDA+U takes no complex sectors."""

    def plans(self, value: str, **extra):
        from parsec_python.SCF.kpoint_solver import KPointJob

        problem, system = _vacancy_system()
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])

        class Invariance:
            def __init__(self, *arguments) -> None:
                pass

            def __call__(self, index) -> bool:
                return True

            def in_density_group(self, index) -> bool:
                return True

        geometry = {
            "grid": system.grid,
            "atoms": tuple(system.cell_atoms),
            "k_points": 2.0 * np.pi / side * np.array([[0.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.0, 0.0, 0.25]]),
            "weights": np.ones(3),
            "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
            "invariance": Invariance,
            **extra,
        }
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_CHARACTERS": value}):
            return sectors.plan_sectors(KPointJob(3, 22, None, None, None, geometry=geometry))

    @staticmethod
    def kinds(plans):
        return [None if plan is None else bool(getattr(plan.decomposition, "is_complex", False)) for plan in plans]

    def test_the_threefold_axis_by_default(self) -> None:
        default, real = self.plans(""), self.plans("real")
        self.assertEqual(self.kinds(default), [False, True, False])
        self.assertEqual(default[1].order, 3)
        self.assertEqual(self.kinds(real), [False, None, False])
        for found, before in ((default[0], real[0]), (default[2], real[2])):
            self.assertEqual(found.sizes, before.sizes)

    def test_no_complex_characters_with_the_hubbard_term(self) -> None:
        marked = self.plans("", hubbard_part=lambda *arguments, **keywords: {})
        self.assertEqual(self.kinds(marked), [False, None, False])


class DecompositionTests(unittest.TestCase):
    def check(self, problem, system, k_points, expected_moduli) -> None:
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        helper = base.SectorOperatorTests()
        helper.system, helper.problem = system, problem
        stencils = checked_axial_stencils(system.negative_laplacian, system.gradient)
        for fractional, modulus in zip(k_points, expected_moduli):
            k_point = 2.0 * np.pi / side * np.asarray(fractional, dtype=float)
            members = sectors.generated_subgroup(cell.table, cell.little_group(k_point, lambda index: True))
            abelian = complex_sectors.largest_abelian_subgroup(cell.table, members)
            decomposition = sectors._complex_candidate(cell, abelian)
            self.assertEqual(decomposition.modulus, modulus, fractional)
            self.assertEqual(sum(decomposition.sector_sizes), system.grid.size)
            full = helper._hamiltonian(k_point, "gradient")
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
            dense = {s: base._dense(split.hamiltonian(s, full)) for s in range(decomposition.representation_count)}
            levels = np.sort(np.concatenate([np.linalg.eigvalsh(matrix) for matrix in dense.values()]))[:30]
            np.testing.assert_allclose(levels, np.linalg.eigvalsh(base._dense(full))[:30], rtol=0, atol=1.0e-11)
            centre, weights = kinetic_weights(stencils, k_point)
            gradient = sum(float(k_point[axis]) * system.gradient[axis] for axis in range(3))
            kinetic = (
                sp.csr_matrix(sectors._materialized(system.negative_laplacian), dtype=np.complex128)
                - 2.0j * gradient + float(k_point @ k_point) * sp.identity(system.grid.size)
            )
            for sector, matrix in dense.items():
                values, vectors = np.linalg.eigh(matrix)
                orbitals = split.orbitals(sector, vectors[:, :3])
                residual = full.apply(orbitals) - orbitals * values[:3]
                self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)
                np.testing.assert_allclose(split.reduce(sector, orbitals), vectors[:, :3], rtol=0, atol=1.0e-12)
                table = build_sector_table(decomposition, sector, stencils.shape, stencils.half_width, centre, weights)
                self.assertTrue(np.iscomplexobj(table.palette) or modulus <= 2)
                reduced = decomposition.reduce_hermitian_operators(kinetic, [sector])[sector]
                np.testing.assert_allclose(table.to_csr().toarray(), reduced.toarray(), rtol=0, atol=1.0e-12)
                block = np.random.default_rng(sector).normal(size=(table.shape[0], 3)) + 0j
                host = HostSectorStencil(table)
                np.testing.assert_allclose(
                    host.apply(block, host.diagonal(np.zeros(table.shape[0]))), reduced @ block, rtol=0, atol=1.0e-12
                )
            classes = sectors.sector_classes(cell, members, abelian, decomposition, k_point, _Admitted())
            if modulus > 2:
                self.assertIsNotNone(classes, fractional)
            for sector, member in (classes or {}).items():
                source = dense[member.source]
                size = source.shape[0]
                image = np.zeros((size, size), dtype=np.complex128)
                image[member.rows, np.arange(size)] = member.signs
                expected = image @ (np.conj(source) if member.antiunitary else source) @ image.conj().T
                np.testing.assert_allclose(dense[sector], expected, rtol=0, atol=1.0e-11)

    def test_the_characters_of_a_cyclic_group(self) -> None:
        # Z3 x Z2 as products of a 3-cycle and a 2-cycle on five points
        def compose(a, b):
            return tuple(a[i] for i in b)

        identity = (0, 1, 2, 3, 4)
        r, s = (1, 2, 0, 3, 4), (0, 1, 2, 4, 3)
        group = [identity]
        frontier = [identity]
        while frontier:
            fresh = []
            for element in frontier:
                for generator in (r, s):
                    product = compose(element, generator)
                    if product not in group:
                        group.append(product)
                        fresh.append(product)
            frontier = fresh
        table = np.array([[group.index(compose(a, b)) for b in group] for a in group])
        exponents, modulus = complex_sectors.character_exponents(table, range(len(group)))
        self.assertEqual((exponents.shape, modulus), ((6, 6), 6))
        phases = np.exp(2j * np.pi * exponents / modulus)
        for a in range(6):
            for b in range(6):
                np.testing.assert_allclose(phases[:, table[a, b]], phases[:, a] * phases[:, b], atol=1.0e-12)

    def test_the_sectors_of_the_vacancy_and_of_the_cube(self) -> None:
        problem, system = _vacancy_system()
        # Td about the vacancy: C3 on the threefold axis, C2v (real) on the z axis (S4 reverses k)
        self.check(problem, system, [(1 / 3, 1 / 3, 1 / 3), (0.0, 0.0, 0.25)], [3, 2])
        problem, system = base._silicon_system()
        self.check(problem, system, [(1 / 3, 1 / 3, 1 / 3)], [4])


def _diamond_vacancy(test: unittest.TestCase):
    from parsec_python.Input.parsec_input import parse_parsec_input

    directory = Path(tempfile.mkdtemp(prefix="parsec_kcx_"))
    test.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
    subprocess.run(
        [sys.executable, str(base._DIAMOND / "make_diamond.py"), "1", "1", "1", str(directory / "parsec.in"),
         "--kmesh", "3", "3", "3"],
        check=True, capture_output=True,
    )
    lines = (directory / "parsec.in").read_text().splitlines()
    start = lines.index("begin Atom_Coord")
    del lines[start + 1]  # the atom at the origin
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
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
    for setting in ("real", "complex"):
        with mock.patch.dict(os.environ, dict(environment, PARSEC_KPOINT_SECTOR_CHARACTERS=setting)), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
            points, shares = kpoint_sampling(problem)
            results[setting] = run_scf_kpoints(system, points, shares, compute_forces=True)
    return results["real"], results["complex"]


@unittest.skipUnless((base._DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
class SCFTests(unittest.TestCase):
    def assert_same(self, real, complex_) -> None:
        self.assertTrue(real.converged and complex_.converged)
        self.assertIn("PARSEC_KPOINT_SECTOR_CHARACTERS=complex", complex_.timings.notes["kpoint_symmetry_detection"])
        self.assertLess(
            complex_.timings.detail["kpoint_sector_work_fraction"], real.timings.detail["kpoint_sector_work_fraction"]
        )
        self.assertAlmostEqual(float(complex_.energies.total), float(real.energies.total), delta=1.0e-9)
        np.testing.assert_allclose(np.asarray(complex_.density), np.asarray(real.density), rtol=0, atol=1.0e-8)
        self.assertGreater(float(np.max(np.abs(real.nonlocal_forces))), 1.0e-3)
        np.testing.assert_allclose(complex_.nonlocal_forces, real.nonlocal_forces, rtol=0, atol=1.0e-7)

    def test_the_vacancy_on_the_device_code_and_the_cpu(self) -> None:
        problem = _diamond_vacancy(self)
        for environment in (
            {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1"},
            {"PARSEC_KPOINT_BACKEND": "cpu"},
        ):
            self.assert_same(*_both(problem, environment))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceTests(unittest.TestCase):
    def test_the_kernel_of_the_complex_palette(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.sector_bloch_stencil import DeviceSectorStencil

        problem, system = _vacancy_system()
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        stencils = checked_axial_stencils(system.negative_laplacian, system.gradient)
        k_point = 2.0 * np.pi / side * np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        members = sectors.generated_subgroup(cell.table, cell.little_group(k_point, lambda index: True))
        decomposition = sectors._complex_candidate(cell, complex_sectors.largest_abelian_subgroup(cell.table, members))
        centre, weights = kinetic_weights(stencils, k_point)
        rng = np.random.default_rng(7)
        for sector in range(decomposition.representation_count):
            table = build_sector_table(decomposition, sector, stencils.shape, stencils.half_width, centre, weights)
            host, device = HostSectorStencil(table), DeviceSectorStencil(cp, table)
            rows = table.shape[0]
            block = rng.normal(size=(rows, 4)) + 1j * rng.normal(size=(rows, 4))
            diagonal = rng.normal(size=rows)
            added = rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape)
            previous = rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape)
            before, after = rng.normal(size=4), rng.normal(size=4)
            expected = host.apply(
                block, host.diagonal(diagonal), added=added, previous=previous, center=0.2, scale=1.3,
                before=before, after=after,
            )
            found = cp.asnumpy(
                device.apply(
                    cp.asarray(block), device.diagonal(cp.asarray(diagonal)), added=cp.asarray(added),
                    previous=cp.asarray(previous), center=0.2, scale=1.3, before=cp.asarray(before),
                    after=cp.asarray(after),
                )
            )
            np.testing.assert_allclose(found, expected, rtol=0, atol=1.0e-12)

    @unittest.skipUnless((base._DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
    def test_the_scf_on_the_gpu(self) -> None:
        SCFTests.assert_same(self, *_both(_diamond_vacancy(self), {"PARSEC_KPOINT_BACKEND": "cupy"}))


if __name__ == "__main__":
    unittest.main()
