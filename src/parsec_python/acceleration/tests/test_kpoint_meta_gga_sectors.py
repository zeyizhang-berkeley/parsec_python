"""A meta-GGA in the sectors of the k-points (``PARSEC_KPOINT_SECTOR_META_GGA``,
``acceleration/Hamiltonian/sector_bloch_meta_gga.py``).

The orbital term of a sector (``(B_a + i k'_a)^H vtau (B_a + i k'_a)`` and the first-order term of the orbital form of
the density gradient) is the term of the whole grid reduced to the sector: the union of the spectra of the sectors is
the spectrum of the k-point, on the Cartesian axes and on the rotated directions of a little group with a diagonal
mirror; ``tau`` and the orbital density gradient of sector states are those of their orbitals on the whole grid, with
the images of a class; the Gershgorin bound of the term bounds its eigenvalues.  On the device (CuPy, and the NumPy
double of the device code) the operator is the host operator.  The SCF of a reduced mesh (the density symmetrized)
gives the energy of one task per k-point, unpolarized and spin-polarized, in both forms of the density gradient.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.SCF.meta_gga_periodic import _host_bloch_fields
from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors
from parsec_python.acceleration.Hamiltonian import sector_bloch_meta_gga as meta_sectors
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.tests.test_kpoint_sectors import _decomposition, _dense, _silicon_system

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


def setUpModule() -> None:
    # the cells of these checks are small: split them all the same
    for name in ("_MINIMUM_ENTRIES", "_META_MINIMUM_ENTRIES"):
        patcher = mock.patch.object(sectors, name, 0)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_META_GGA": value}):
                self.assertEqual(meta_sectors.kpoint_sector_meta_gga_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_META_GGA": "on"}):
            with self.assertRaises(ValueError):
                meta_sectors.kpoint_sector_meta_gga_setting()
            # refused at the preparation of a run too
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches()

    def test_the_bloch_kinetic_term_is_refused_for_a_meta_gga(self) -> None:
        # the orbital term keeps the gradient form of the kinetic term: PARSEC_KPOINT_KINETIC=bloch stops a meta-GGA at
        # the preparation of its run (it raised after the first iteration of its k-point loop before)
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scan, pbe = silicon("scan", spacing=0.855), silicon("pbe", spacing=0.855)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "bloch"}):
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches(scan)
            sectors.check_kpoint_sector_switches(pbe)
            sectors.check_kpoint_sector_switches()
        for value in ("gradient", "auto", "trim"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": value}):
                sectors.check_kpoint_sector_switches(scan)


class PlanTests(unittest.TestCase):
    def test_a_meta_gga_takes_neither_pairs_nor_real_forms_nor_complex_characters(self) -> None:
        # the gradient form of the kinetic term (no conjugate pairs of a time-reversal invariant point), and the
        # orbital term reduced on the real characters of commuting involutions alone
        from parsec_python.SCF.kpoint_solver import KPointJob

        problem, system = _silicon_system()
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.pi / side * np.array([1.0, 0.0, 0.0]), np.array([0.11, 0.07, 0.05])])
        switches = {
            "PARSEC_KPOINT_KINETIC": "auto",
            "PARSEC_KPOINT_SECTOR_REAL": "auto",
            "PARSEC_KPOINT_SECTOR_CHARACTERS": "complex",
        }
        found = {}
        for meta in (False, True):
            geometry = {
                "grid": system.grid,
                "atoms": tuple(system.cell_atoms),
                "k_points": k_points,
                "weights": np.full(2, 0.5),
                "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
                "density_symmetry": None,
            }
            if meta:
                geometry["meta_fields"] = lambda task: (1.0, "tau", None)
                geometry["meta_derivatives"] = tuple(system.gradient)
            job = KPointJob(
                n_tasks=2, number_of_states=22, eigval_settings=lambda *args: None,
                make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {}, geometry=geometry,
            )
            with mock.patch.dict(os.environ, switches):
                found[meta] = sectors.plan_sectors(job)
        self.assertIsNotNone(found[False][0].partners)
        for plan in found[True]:
            self.assertIsNotNone(plan)
            self.assertIsNone(plan.partners)
            self.assertFalse(plan.real_forms)
            self.assertFalse(getattr(plan.decomposition, "is_complex", False))

    def test_the_floor_of_a_meta_gga_on_several_devices(self) -> None:
        # a basis between the two floors: split on one device (and for another functional), whole where the solver of
        # one task per k-point would deal the k-points over several devices; above the floor of a meta-GGA, split
        from parsec_python.SCF.kpoint_solver import KPointJob

        problem, system = _silicon_system()
        entries = system.grid.size * 22

        def job(meta: bool, k_count: int = 2):
            geometry = {
                "grid": system.grid,
                "atoms": tuple(system.cell_atoms),
                "k_points": np.array([[0.11, 0.07, 0.05], [0.05, 0.11, 0.07]])[:k_count],
                "weights": np.full(k_count, 1.0 / k_count),
                "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
                "density_symmetry": None,
            }
            if meta:
                geometry["meta_fields"] = lambda task: (1.0, "tau", None)
                geometry["meta_derivatives"] = tuple(system.gradient)
            return KPointJob(
                n_tasks=k_count, number_of_states=22, eigval_settings=lambda *args: None,
                make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {}, geometry=geometry,
            )

        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", entries), mock.patch.object(
            sectors, "_META_MINIMUM_ENTRIES", 2 * entries
        ):
            self.assertIsNotNone(sectors.plan_sectors(job(True), devices=1))
            self.assertIsNotNone(sectors.plan_sectors(job(False), devices=4))
            notes: list = []
            self.assertIsNone(sectors.plan_sectors(job(True), notes, devices=4))
            self.assertIn("of a meta-GGA on 2 devices", notes[-1])
            # one k-point: the solver of one task per k-point uses one device whatever the count
            self.assertIsNotNone(sectors.plan_sectors(job(True, 1), devices=4))
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", entries // 2), mock.patch.object(
            sectors, "_META_MINIMUM_ENTRIES", entries
        ):
            self.assertIsNotNone(sectors.plan_sectors(job(True), devices=4))


def _swap_decomposition(cell, k_point):
    """The commuting involutions generated by the diagonal mirror ``x <-> y`` and a translation that commutes with it
    (rotated derivative directions), or ``None``."""

    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    members = sectors.generated_subgroup(cell.table, cell.little_group(k_point, lambda index: True))
    mirrors = [index for index in members if np.array_equal(cell.matrices[index], swap)]
    pure = [index for index in members if index and np.array_equal(cell.matrices[index], np.eye(3))]
    for mirror in mirrors:
        for shift in pure:
            if int(cell.table[mirror, shift]) == int(cell.table[shift, mirror]) and int(cell.table[mirror, mirror]) == 0:
                subgroup = sectors.generated_subgroup(cell.table, [mirror, shift])
                elements, generators = sectors.largest_commuting_involutions(cell.table, subgroup)
                reduction = sectors.little_group_reduction(cell, elements, generators)
                return ReflectionRepresentationDecomposition.build(cell.grid, reduction)
    return None


class SectorOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = _silicon_system()
        cls.cell = sectors.CellGroup(cls.system.grid, cls.system.cell_atoms)
        cls.side = float(np.asarray(cls.problem.periodic_cell.lattice_vectors)[0, 0])

    def _full(self, k_point, decomposition, rng, *, flux: bool):
        """A Bloch Hamiltonian whose vtau and W are invariant under the group of ``decomposition``."""

        system, problem = self.system, self.problem
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        projectors = build_nonlocal_projectors(
            system.grid, system.cell_atoms, system.pseudopotentials, problem.pseudopotentials, lattice,
            k_point=k_point, periodic_dimensions=problem.periodic_cell.dimensions,
        )
        full_to_wedge = decomposition.reduction.full_to_wedge
        vtau = decomposition.invariant_wedge_values(0.5 + rng.random(system.grid.size))[full_to_wedge]
        fluxes = None
        if flux:
            # the gradient of an invariant field turns with the operations
            scalar = decomposition.invariant_wedge_values(rng.standard_normal(system.grid.size))[full_to_wedge]
            fluxes = tuple(0.05 * (matrix @ scalar) for matrix in system.gradient)
        return KPointKohnShamHamiltonian(
            system.negative_laplacian, system.gradient, system.ionic_potential, projectors, k_point,
            kinetic_potential=vtau, gradient_flux=fluxes,
        )

    def _check(self, k_point, decomposition, label) -> None:
        rng = np.random.default_rng(11)
        system = self.system
        for flux in (False, True):
            full = self._full(k_point, decomposition, rng, flux=flux)
            expected = np.linalg.eigvalsh(_dense(full))[:30]
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
            split.device_operators = False
            levels, tau, gradient = [], np.zeros(system.grid.size), np.zeros((3, system.grid.size))
            reference_tau, reference_gradient = np.zeros_like(tau), np.zeros_like(gradient)
            occupations = np.array([2.0, 1.0, 0.5])
            for sector in range(decomposition.representation_count):
                hamiltonian = split.hamiltonian(sector, full)
                self.assertIsInstance(hamiltonian, sectors.SectorMetaBlochHamiltonian)
                matrix = hamiltonian.apply(np.eye(hamiltonian.shape[0], dtype=np.complex128))
                self.assertLess(float(np.max(np.abs(matrix - matrix.conj().T))), 1.0e-12)
                values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
                levels.append(values)
                # Gershgorin's bound of the orbital term bounds its eigenvalues
                bare = sectors.SectorBlochHamiltonian(
                    hamiltonian.negative_laplacian, hamiltonian.gradient, hamiltonian.effective_potential,
                    hamiltonian.nonlocal_operator, hamiltonian.k_point, bloch_laplacian=hamiltonian.bloch_laplacian,
                )
                term = 0.5 * (matrix + matrix.conj().T) - _dense(bare)
                self.assertLessEqual(float(np.max(np.linalg.eigvalsh(term))), hamiltonian.spectral_margin + 1.0e-9)
                found_tau, found_gradient = split.kinetic_density(
                    sector, vectors[:, :3], occupations, 0.37, system.gradient, gradient=True
                )
                tau += found_tau
                gradient += found_gradient
                fields = _host_bloch_fields(
                    system.gradient, split.orbitals(sector, vectors[:, :3]), occupations, k_point, 0.37,
                    with_gradient=True,
                )
                reference_tau += fields[0]
                reference_gradient += fields[1]
            levels = np.sort(np.concatenate(levels))[:30]
            message = f"{label} flux={flux}"
            np.testing.assert_allclose(levels, expected, rtol=0, atol=1.0e-11, err_msg=message)
            np.testing.assert_allclose(tau, reference_tau, rtol=0, atol=1.0e-15, err_msg=message)
            np.testing.assert_allclose(gradient, reference_gradient, rtol=0, atol=1.0e-15, err_msg=message)

    def test_the_sectors_of_the_little_groups(self) -> None:
        for k_point in (np.zeros(3), np.pi / self.side * np.array([1.0, 0.0, 0.0]), np.array([0.11, 0.07, 0.05])):
            _, _, decomposition = _decomposition(self.cell, k_point)
            self._check(k_point, decomposition, str(k_point))

    def test_the_static_terms_follow_the_kinetic_form(self) -> None:
        # a meta-GGA's first iteration may take the Bloch form (PARSEC_KPOINT_KINETIC=auto, trim: no orbital term yet),
        # its later ones the gradient form: the static terms of the sectors (shared with the twin of the other spin
        # channel) are made again from the form of the Hamiltonian at hand
        from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian

        k_point = np.pi / self.side * np.array([1.0, 0.0, 0.0])
        _, _, decomposition = _decomposition(self.cell, k_point)
        rng = np.random.default_rng(19)
        meta = self._full(k_point, decomposition, rng, flux=True)
        bloch = KPointKohnShamHamiltonian(
            meta.negative_laplacian, meta.gradient, meta.effective_potential, meta.nonlocal_operator, k_point,
            bloch_laplacian=build_bloch_negative_laplacian(self.system.grid, k_point, "u"),
        )
        count = decomposition.representation_count
        for device_operators in (False, True):
            fresh = sectors.KPointSectors(0, k_point, decomposition, [8] * count)
            fresh.device_operators = device_operators
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * count)
            split.device_operators = device_operators
            twin = split.twin(1)
            for sector in range(count):
                split.hamiltonian(sector, bloch)
            for owner in (twin, split):
                for sector in range(count):
                    block = rng.standard_normal((decomposition.sector_size(sector), 3)) + 0j
                    expected = fresh.hamiltonian(sector, meta).apply(block)
                    found = owner.hamiltonian(sector, meta).apply(block)
                    np.testing.assert_allclose(found, expected, rtol=0, atol=1.0e-12 * np.abs(expected).max())
            # and back: the Bloch form of the first Hamiltonians again
            fresh_bloch = sectors.KPointSectors(0, k_point, decomposition, [8] * count)
            fresh_bloch.device_operators = device_operators
            block = rng.standard_normal((decomposition.sector_size(0), 3)) + 0j
            np.testing.assert_allclose(
                split.hamiltonian(0, bloch).apply(block), fresh_bloch.hamiltonian(0, bloch).apply(block), rtol=0,
                atol=1.0e-12,
            )

    def test_rotated_derivative_directions(self) -> None:
        from parsec_python.acceleration.Hamiltonian.meta_gga_device import derivative_directions

        k_point = np.array([0.13, 0.13, 0.07])
        decomposition = _swap_decomposition(self.cell, k_point)
        self.assertIsNotNone(decomposition)
        directions = derivative_directions(decomposition.reduction)[0]
        self.assertFalse(np.allclose(np.abs(directions), np.eye(3)))
        geometry = meta_sectors.DecompositionDerivatives(decomposition, self.system.gradient).sector(0)
        # k' = q . k on the directions of the trivial character (B maps the sector into itself), zero on the others
        shifts = geometry.k_shifts(k_point)
        projected = directions.T @ k_point
        for axis, target in enumerate(geometry.targets):
            if target == geometry.sector:
                self.assertAlmostEqual(shifts[axis], projected[axis], delta=1.0e-15)
            else:
                self.assertEqual(shifts[axis], 0.0)
                self.assertLess(abs(projected[axis]), 1.0e-12)
        self.assertGreater(float(np.max(np.abs(shifts))), 0.1)
        self._check(k_point, decomposition, "swap")

    def test_the_images_of_a_class(self) -> None:
        # the sectors not solved (PARSEC_KPOINT_SECTOR_CLASSES): tau and g of the images P_g u (P_g conj(u) under an
        # antiunitary operation) of the states solved, the mean over the class
        from parsec_python.acceleration.tests.test_kpoint_sectors import _Admitted

        seen = {False: 0, True: 0}
        for k_point in (
            np.pi / self.side * np.array([1.0, 0.0, 0.0]),
            2.0 * np.pi / self.side * np.array([0.0, 0.0, 1.0 / 3.0]),
        ):
            members, elements, decomposition = _decomposition(self.cell, k_point)
            for chosen in (members, elements):  # the unitary operations as well, then the antiunitary ones alone
                classes = sectors.sector_classes(self.cell, chosen, elements, decomposition, k_point, _Admitted())
                if not classes:
                    continue
                rng = np.random.default_rng(2)
                full = self._full(k_point, decomposition, rng, flux=True)
                split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
                split.device_operators = False
                split.classes = classes
                for source in sorted({entry.source for entry in classes.values()}):
                    others = [m for m, entry in classes.items() if entry.source == source]
                    seen[False] += sum(1 for m in others if not classes[m].antiunitary)
                    seen[True] += sum(1 for m in others if classes[m].antiunitary)
                    _, vectors = np.linalg.eigh(_dense(split.hamiltonian(source, full)))
                    occupations = np.array([2.0, 2.0, 1.0])
                    tau, gradient = split.kinetic_density(
                        source, vectors[:, :3], occupations, 0.5, self.system.gradient, members=others, gradient=True
                    )
                    reference_tau = np.zeros_like(tau)
                    reference_gradient = np.zeros_like(gradient)
                    for current in [source] + others:
                        block = vectors[:, :3] if current == source else split.member_block(current, vectors[:, :3])
                        fields = _host_bloch_fields(
                            self.system.gradient, split.orbitals(current, block), occupations, k_point, 0.5,
                            with_gradient=True,
                        )
                        reference_tau += fields[0] / (1 + len(others))
                        reference_gradient += fields[1] / (1 + len(others))
                    np.testing.assert_allclose(tau, reference_tau, rtol=0, atol=1.0e-15)
                    np.testing.assert_allclose(gradient, reference_gradient, rtol=0, atol=1.0e-15)
        self.assertGreater(seen[False], 0)
        self.assertGreater(seen[True], 0)

    def test_the_device_operator_of_a_sector(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import (
            CupyArrayBackend,
            DeviceBlochConverter,
            NumpyArrayBackend,
        )

        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(self.cell, k_point)
        rng = np.random.default_rng(7)
        backends = [NumpyArrayBackend()] + ([CupyArrayBackend(0)] if cupy_available() else [])
        for flux in (False, True):
            full = self._full(k_point, decomposition, rng, flux=flux)
            split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
            for device_operators in (False, True):
                split.device_operators = device_operators
                split._static.clear()
                for sector in (0, decomposition.representation_count - 1):
                    host = split.hamiltonian(sector, full)
                    block = rng.standard_normal((host.shape[0], 5)) + 1j * rng.standard_normal((host.shape[0], 5))
                    expected = host.apply(block)
                    for backend in backends:
                        with self.subTest(flux=flux, stencil=device_operators, sector=sector, backend=backend.name):
                            operator = DeviceBlochConverter(backend).operator(host)
                            self.assertIsNotNone(operator.kinetic)
                            self.assertFalse(operator.real_arithmetic)
                            self.assertAlmostEqual(operator.spectral_margin, host.spectral_margin, delta=1.0e-9)
                            result = backend.asnumpy(operator.apply(backend.asarray(np.ascontiguousarray(block))))
                            np.testing.assert_allclose(result, expected, rtol=0, atol=1.0e-11 * np.abs(expected).max())

    @unittest.skipUnless(cupy_available(), "CuPy with a CUDA device is needed")
    def test_device_sums_are_the_host_sums(self) -> None:
        import cupy as cp

        for k_point, decomposition in (
            (np.array([0.11, 0.07, 0.05]), _decomposition(self.cell, np.array([0.11, 0.07, 0.05]))[2]),
            (np.array([0.13, 0.13, 0.07]), _swap_decomposition(self.cell, np.array([0.13, 0.13, 0.07]))),
        ):
            rng = np.random.default_rng(9)
            shared = meta_sectors.DecompositionDerivatives(decomposition, self.system.gradient)
            for sector in range(decomposition.representation_count):
                geometry = shared.sector(sector)
                shifts = geometry.k_shifts(k_point)
                vectors = rng.standard_normal((geometry.rows, 4)) + 1j * rng.standard_normal((geometry.rows, 4))
                occupations = np.array([1.0, 0.0, 0.6, 0.2])
                host = meta_sectors.sector_orbital_sums(geometry, shifts, vectors, occupations, gradient=True)
                device = meta_sectors.sector_orbital_sums(
                    geometry, shifts, cp.asarray(vectors), occupations, cp, gradient=True
                )
                for left, right in zip(host[0] + host[1], device[0] + device[1]):
                    np.testing.assert_allclose(right, left, rtol=1.0e-13, atol=1.0e-15)


class UnequalSectorTests(unittest.TestCase):
    """A grid whose points lie on the symmetry elements (14^3 points of the cube): orbits with stabilizers, sectors of
    unequal sizes, derivatives between sectors of different rows.  The operator of a sector is ``U^H H U`` of the
    whole grid, ``tau`` and ``g`` of sector vectors those of ``U v``, and the device term and sums are the host ones."""

    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.problem = silicon("pbe", spacing=0.684)
            cls.system = prepare_periodic_single_point(cls.problem)
        cls.cell = sectors.CellGroup(cls.system.grid, cls.system.cell_atoms)
        cls.side = float(np.asarray(cls.problem.periodic_cell.lattice_vectors)[0, 0])

    def _cases(self):
        for k_point in (np.zeros(3), np.pi / self.side * np.array([1.0, 0.0, 0.0])):
            _, _, decomposition = _decomposition(self.cell, k_point)
            self.assertGreater(len(set(decomposition.sector_sizes)), 1)
            yield k_point, decomposition

    def test_the_operator_and_fields_of_unequal_sectors(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import (
            CupyArrayBackend,
            DeviceBlochConverter,
            NumpyArrayBackend,
        )

        helper = SectorOperatorTests()
        helper.system, helper.problem = self.system, self.problem
        backends = [NumpyArrayBackend()] + ([CupyArrayBackend(0)] if cupy_available() else [])
        rng = np.random.default_rng(13)
        for k_point, decomposition in self._cases():
            for flux in (False, True):
                full = helper._full(k_point, decomposition, rng, flux=flux)
                split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
                split.device_operators = False
                for sector in range(decomposition.representation_count):
                    label = f"{k_point} flux={flux} sector {sector}"
                    host = split.hamiltonian(sector, full)
                    block = rng.standard_normal((host.shape[0], 3)) + 1j * rng.standard_normal((host.shape[0], 3))
                    expected = split.reduce(sector, full.apply(split.orbitals(sector, block)))
                    found = host.apply(block)
                    np.testing.assert_allclose(found, expected, rtol=0, atol=1.0e-10 * np.abs(expected).max(), err_msg=label)
                    occupations = np.array([1.0, 0.5, 0.25])
                    tau, gradient = split.kinetic_density(
                        sector, block, occupations, 0.3, self.system.gradient, gradient=True
                    )
                    fields = _host_bloch_fields(
                        self.system.gradient, split.orbitals(sector, block), occupations, k_point, 0.3,
                        with_gradient=True,
                    )
                    np.testing.assert_allclose(tau, fields[0], rtol=0, atol=1.0e-12 * np.abs(fields[0]).max(), err_msg=label)
                    np.testing.assert_allclose(
                        gradient, fields[1], rtol=0, atol=1.0e-12 * np.abs(fields[1]).max(), err_msg=label
                    )
                    for backend in backends:
                        operator = DeviceBlochConverter(backend).operator(host)
                        result = backend.asnumpy(operator.apply(backend.asarray(np.ascontiguousarray(block))))
                        np.testing.assert_allclose(
                            result, found, rtol=0, atol=1.0e-11 * np.abs(found).max(), err_msg=f"{label} {backend.name}"
                        )

    @unittest.skipUnless(cupy_available(), "CuPy with a CUDA device is needed")
    def test_device_sums_of_unequal_sectors(self) -> None:
        import cupy as cp

        rng = np.random.default_rng(17)
        for k_point, decomposition in self._cases():
            shared = meta_sectors.DecompositionDerivatives(decomposition, self.system.gradient)
            for sector in range(decomposition.representation_count):
                geometry = shared.sector(sector)
                shifts = geometry.k_shifts(k_point)
                vectors = rng.standard_normal((geometry.rows, 4)) + 1j * rng.standard_normal((geometry.rows, 4))
                occupations = np.array([1.0, 0.0, 0.6, 0.2])
                host = meta_sectors.sector_orbital_sums(geometry, shifts, vectors, occupations, gradient=True)
                device = meta_sectors.sector_orbital_sums(
                    geometry, shifts, cp.asarray(vectors), occupations, cp, gradient=True
                )
                for left, right in zip(host[0] + host[1], device[0] + device[1]):
                    np.testing.assert_allclose(right, left, rtol=1.0e-13, atol=1.0e-15)


class SectorSCFTests(unittest.TestCase):
    """The SCF of a meta-GGA in the sectors = one task per k-point: the Gamma-centred 3 x 3 x 3 mesh of the silicon
    cube (14 points after time reversal), which every operation of the cell maps onto its points or their time-reversal
    partners exactly, so that the loop of one task per k-point keeps the symmetry the sectors hold.  (On an even mesh
    an operation carries some points onto ``k - G``, which the gradient form of the kinetic term, that of every
    meta-GGA, does not make an exact image of ``k``: the loop without sectors then breaks the cell group by the
    discretization, 1.7e-6 in the density of this cube under SCAN on its 2 x 2 x 2 mesh, and the sectors hold the
    symmetric solution, 2e-9 Ry apart.)"""

    def _problem(self, *, spin: bool = False):
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("scan", spacing=0.855)
        problem = replace(
            base,
            scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
            eigensolver=replace(base.eigensolver, tolerance=1.0e-10),
            monkhorst_pack_dimensions=(3, 3, 3),
            monkhorst_pack_shift=(0.0, 0.0, 0.0),
        )
        if spin:
            problem = replace(problem, scf=replace(problem.scf, spin_polarized=True))
        return problem

    def _both(self, problem, backend: str, form: str, *, spin: bool = False, environment=None):
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        results = {}
        for setting in ("off", "auto"):
            switches = {
                "PARSEC_KPOINT_SECTOR_META_GGA": setting,
                "PARSEC_KPOINT_BACKEND": backend,
                "PARSEC_META_GGA_GRADIENT": form,
                "PARSEC_KPOINT_SECTOR_GUARD": "off",
                **(environment or {}),
            }
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
                points, shares = kpoint_sampling(problem)
                run = run_scf_kpoints_spin_polarized if spin else run_scf_kpoints
                results[setting] = run(system, points, shares)
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertNotIn("kpoint_sector_tasks", off.timings.detail)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], len(points) * (2 if spin else 1))
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-10)
        if spin:
            for name in ("eigenvalues_up", "eigenvalues_down"):
                np.testing.assert_allclose(getattr(auto, name), getattr(off, name), rtol=0, atol=1.0e-7)
        else:
            occupied = np.asarray(off.occupations) > 1.0e-8
            np.testing.assert_allclose(
                np.asarray(auto.eigenvalues)[occupied], np.asarray(off.eigenvalues)[occupied], rtol=0, atol=1.0e-8
            )
        return off, auto

    def test_both_forms_of_the_gradient(self) -> None:
        backends = ["numpy-device"] + (["cupy"] if cupy_available() else [])
        problem = self._problem()
        for backend in backends:
            for form in ("orbital", "density"):
                with self.subTest(backend=backend, form=form):
                    self._both(problem, backend, form)

    def test_the_kinetic_form_of_the_first_iteration(self) -> None:
        # PARSEC_KPOINT_KINETIC=auto: the first iteration (no orbital term) in the Bloch form, the others in the
        # gradient form, in the sectors as in the loop of one task per k-point (4.8e-3 Ry apart before the static
        # terms followed the form)
        self._both(self._problem(), "numpy-device", "orbital", environment={"PARSEC_KPOINT_KINETIC": "auto"})

    def test_spin_channels(self) -> None:
        backend = "cupy" if cupy_available() else "numpy-device"
        self._both(self._problem(spin=True), backend, "orbital", spin=True)

    def test_forces_from_the_orbitals_of_the_sectors(self) -> None:
        # one atom moved along x (C2x and the mirrors through x keep the cell), the Gamma and X points: the nonlocal
        # forces from the sector orbitals expanded to the whole grid, the total forces of the loop
        from parsec_python.Forces.total import compute_periodic_forces
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = self._problem()
        atoms = list(problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
        problem = replace(problem, atoms=atoms)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        backend = "cupy" if cupy_available() else "numpy-device"
        found = {}
        for setting in ("off", "auto"):
            switches = {"PARSEC_KPOINT_SECTOR_META_GGA": setting, "PARSEC_KPOINT_BACKEND": backend}
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
                result = run_scf_kpoints(system, k_points, np.array([0.5, 0.5]), compute_forces=True)
                found[setting] = (result, compute_periodic_forces(system, result))
        (off, off_forces), (auto, auto_forces) = found["off"], found["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], 2.0)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-10)
        self.assertGreater(float(np.max(np.abs(off.nonlocal_forces))), 1.0e-3)
        np.testing.assert_allclose(auto.nonlocal_forces, off.nonlocal_forces, rtol=0.0, atol=1.0e-8)
        np.testing.assert_allclose(auto_forces.total, off_forces.total, rtol=0.0, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
