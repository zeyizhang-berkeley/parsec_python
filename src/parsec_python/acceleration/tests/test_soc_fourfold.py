"""The real sector of C2z split by a four-fold operation about z (``PARSEC_SOC_SECTORS=fourfold``), on the host.

A four-fold rotation C4z or rotoinversion S4z of the atoms and the grid squares to C2z and turns spinors as the
rotation by +pi/2 about z, so in the ``+i`` sector of C2z its eigenvalues are ``+-exp(i pi/4)``; with an operation
``h`` that makes the sector real (``T h`` inverts it) both eigenspaces stay real.  Checked here with NumPy as the array
backend: the plan of an atom (C4z), a square (D4h, C4z) and a tetrahedron (Td, S4z), the spectrum of the two sectors
against that of the unsplit sector and of the dense whole-grid operator, their unfolded states, a cluster without a
four-fold axis about z, and the SCF of the split sectors against the unsplit one.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python import Atom
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.Hamiltonian.device_bloch import NumpyArrayBackend
from parsec_python.acceleration.SCF.self_consistent_soc import (
    DeviceSpinorHamiltonianBuilder,
    run_self_consistent_soc,
)
from parsec_python.acceleration.SCF.soc_sectors import (
    SWITCH,
    SectorRoute,
    SectorSpinors,
    detect_sector_plan,
    soc_orbital_symmetry,
    soc_sectors_setting,
)
from parsec_python.acceleration.tests.test_soc_sectors import _prepared

_D = 1.6


class FourFoldOperatorTests(unittest.TestCase):
    CASES = {
        "atom": ([Atom("Au", [0.0, 0.0, 0.0])], "C4z"),
        "square": ([Atom("Au", [2.2, 0.0, 0.0]), Atom("Au", [-2.2, 0.0, 0.0]), Atom("Au", [0.0, 2.2, 0.0]),
                    Atom("Au", [0.0, -2.2, 0.0])], "C4z"),
        "tetrahedron": ([Atom("Au", [_D, _D, _D]), Atom("Au", [_D, -_D, -_D]), Atom("Au", [-_D, _D, -_D]),
                         Atom("Au", [-_D, -_D, _D])], "S4z"),
    }

    def check(self, atoms, name: str) -> None:
        system, projectors = _prepared(atoms, spacing=0.75, radius=4.0)
        backend = NumpyArrayBackend()
        scalar = system.hamiltonian(system.ionic_potential)
        full = DeviceSpinorHamiltonianBuilder(backend)(scalar, projectors)
        dense = full.apply(np.eye(full.shape[0], dtype=complex))
        full_values = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
        kramers, _ = detect_sector_plan(system.grid, system.atoms, parity=False)
        plan, reason = detect_sector_plan(system.grid, system.atoms, parity=False, fourfold=True)
        self.assertIsNotNone(plan, reason)
        self.assertTrue(plan.real)
        self.assertEqual(plan.split_name, name)
        self.assertEqual(plan.sector_count, 2)
        self.assertEqual(plan.sector_dimension(0) + plan.sector_dimension(1), system.grid.size)
        self.assertIn(f"split by the eigenvalues of {name}", plan.description())
        unsplit = SectorRoute(kramers, backend, projectors).hamiltonian(scalar, projectors)
        size = unsplit.shape[0]
        reduced = unsplit.apply(np.eye(size))
        scale = float(np.abs(reduced).max())
        values = np.linalg.eigvalsh(0.5 * (reduced + reduced.T))
        operators = SectorRoute(plan, backend, projectors).hamiltonian(scalar, projectors).sectors
        both = []
        for operator in operators:
            self.assertTrue(operator.real_arithmetic)
            rows = operator.shape[0]
            matrix = operator.apply(np.eye(rows))
            self.assertEqual(matrix.dtype, np.float64)
            self.assertLess(float(np.abs(matrix - matrix.T).max()), 1.0e-13 * scale)
            sector_values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
            both.append(sector_values)
            # the unfolded states of the sector are eigenvectors of the whole-grid operator
            spinors = SectorSpinors.single(operator.parts, vectors[:, :3], 6).full_spinors(chunk=2)
            np.testing.assert_allclose(spinors.conj().T @ spinors, np.eye(6), atol=1.0e-12)
            residual = full.apply(spinors) - spinors * np.repeat(sector_values[:3], 2)[None, :]
            self.assertLess(float(np.abs(residual).max()), 1.0e-11 * scale)
        merged = np.sort(np.concatenate(both))
        np.testing.assert_allclose(merged, values, atol=1.0e-11 * scale)
        np.testing.assert_allclose(np.repeat(merged, 2), full_values, atol=1.0e-11 * scale)
        # neither sector is empty, and the lowest pairs lie in both
        self.assertTrue(all(part.size > 0 for part in both))

    def test_cases(self) -> None:
        for case, (atoms, name) in self.CASES.items():
            with self.subTest(case):
                self.check(atoms, name)

    def test_with_the_parity_of_a_centre_of_inversion(self) -> None:
        # the square (D4h): the even and odd sectors of i, each split by C4z (which commutes with i)
        atoms = self.CASES["square"][0]
        system, projectors = _prepared(atoms, spacing=0.75, radius=4.0)
        backend = NumpyArrayBackend()
        scalar = system.hamiltonian(system.ionic_potential)
        kramers, _ = detect_sector_plan(system.grid, system.atoms, parity=False)
        plan, _ = detect_sector_plan(system.grid, system.atoms, parity=True, fourfold=True)
        self.assertEqual(plan.sector_count, 4)
        self.assertEqual([base for base, _ in plan.splits], [0, 0, 1, 1])
        self.assertEqual(sum(plan.sector_dimension(index) for index in range(4)), system.grid.size)
        self.assertIn("split by the parity of i", plan.description())
        unsplit = SectorRoute(kramers, backend, projectors).hamiltonian(scalar, projectors)
        reduced = unsplit.apply(np.eye(unsplit.shape[0]))
        scale = float(np.abs(reduced).max())
        values = np.linalg.eigvalsh(0.5 * (reduced + reduced.T))
        parts = []
        for operator in SectorRoute(plan, backend, projectors).hamiltonian(scalar, projectors).sectors:
            matrix = operator.apply(np.eye(operator.shape[0]))
            self.assertLess(float(np.abs(matrix - matrix.T).max()), 1.0e-13 * scale)
            parts.append(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)))
        np.testing.assert_allclose(np.sort(np.concatenate(parts)), values, atol=1.0e-11 * scale)

    def test_without_a_four_fold_axis_about_z_the_sector_stays_whole(self) -> None:
        system, _ = _prepared([Atom("Au", [2.4, 0.0, 0.0]), Atom("Au", [-2.4, 0.0, 0.0])], spacing=0.75, radius=4.0)
        plan, _ = detect_sector_plan(system.grid, system.atoms, parity=False, fourfold=True)
        self.assertIsNotNone(plan)
        self.assertIsNone(plan.splits)
        self.assertEqual(plan.sector_count, 1)
        text, reason = soc_orbital_symmetry(system.grid, system.atoms, "fourfold")
        self.assertEqual(reason, "")
        self.assertNotIn("split by the eigenvalues", text)
        atom, _ = _prepared([Atom("Au", [0.0, 0.0, 0.0])], spacing=0.75, radius=4.0)
        text, reason = soc_orbital_symmetry(atom.grid, atom.atoms, "fourfold")
        self.assertEqual(reason, "")
        self.assertIn("split by the eigenvalues of C4z", text)

    def test_the_switch_takes_the_value(self) -> None:
        with mock.patch.dict(os.environ, {SWITCH: "fourfold"}):
            self.assertEqual(soc_sectors_setting(), "fourfold")


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class FourFoldDeviceTests(unittest.TestCase):
    """The operators, density and unfolded states of the split sectors on the GPU against the same code with NumPy."""

    def test_cupy_matches_numpy(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        rng = np.random.default_rng(5)
        for case in ("square", "tetrahedron"):
            with self.subTest(case):
                system, projectors = _prepared(FourFoldOperatorTests.CASES[case][0], spacing=0.75, radius=4.0)
                scalar = system.hamiltonian(system.ionic_potential)
                plan, _ = detect_sector_plan(system.grid, system.atoms, parity=False, fourfold=True)
                self.assertEqual(plan.sector_count, 2)
                host = SectorRoute(plan, NumpyArrayBackend(), projectors).hamiltonian(scalar, projectors).sectors
                device = SectorRoute(plan, CupyArrayBackend(0), projectors).hamiltonian(scalar, projectors).sectors
                for host_operator, device_operator in zip(host, device):
                    block = rng.normal(size=(host_operator.shape[0], 7))
                    expected = host_operator.apply(block)
                    found = cp.asnumpy(device_operator.apply(cp.asarray(block)))
                    np.testing.assert_allclose(found, expected, atol=1.0e-12 * float(np.abs(expected).max()))
                    occupations = np.linspace(1.0, 0.1, 13)
                    host_states = SectorSpinors.single(host_operator.parts, block, 13)
                    device_states = SectorSpinors.single(device_operator.parts, cp.asarray(block), 13)
                    np.testing.assert_allclose(
                        device_states.density(occupations, 0.3, chunk=3),
                        host_states.density(occupations, 0.3, chunk=3),
                        rtol=1.0e-12,
                        atol=1.0e-14,
                    )
                    np.testing.assert_allclose(
                        device_states.full_spinors(chunk=3), host_states.full_spinors(chunk=3), atol=1.0e-13
                    )


class FourFoldSCFTests(unittest.TestCase):
    """The SCF of the split sectors against the unsplit one (Au atom: C4z, NumPy backend)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system, cls.projectors = _prepared(
            [Atom("Au", [0.0, 0.0, 0.0])], spacing=0.6, radius=6.0, states=20, iterations=40
        )
        cls.results, cls.notes = {}, {}
        for setting in ("kramers", "fourfold"):
            cls.notes[setting] = []
            with mock.patch.dict(os.environ, {SWITCH: setting}):
                cls.results[setting] = run_self_consistent_soc(
                    cls.system, cls.projectors, backend=NumpyArrayBackend(), note=cls.notes[setting].append
                )

    def test_same_energy_states_and_density(self) -> None:
        kramers, fourfold = self.results["kramers"], self.results["fourfold"]
        self.assertTrue(kramers.converged and fourfold.converged)
        self.assertAlmostEqual(fourfold.energies.total, kramers.energies.total, delta=1.0e-7)
        np.testing.assert_allclose(fourfold.eigenvalues, kramers.eigenvalues, atol=1.0e-4)
        np.testing.assert_allclose(fourfold.density, kramers.density, atol=1.0e-4)
        np.testing.assert_allclose(fourfold.occupations, kramers.occupations, atol=1.0e-6)
        spinors = fourfold.spinors
        self.assertEqual(spinors.shape, kramers.spinors.shape)
        np.testing.assert_allclose(spinors.conj().T @ spinors, np.eye(spinors.shape[1]), atol=1.0e-10)
        self.assertIn("split by the eigenvalues of C4z", self.notes["fourfold"][0])
        self.assertIn("eig_sector_pairs", fourfold.timings)


if __name__ == "__main__":
    unittest.main()
