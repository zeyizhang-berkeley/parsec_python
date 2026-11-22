"""Spin-flip (magnetic) symmetry of collinear spin-polarized runs (``PARSEC_SPIN_FLIP_SYMMETRY``).

An antiferromagnetic or broken-symmetry start of PARSEC 1.5 names two labels of one species with opposite
``Initial_Spin_Polarization``.  An operation of the atoms that exchanges the labels carries the up channel onto the
down channel (:mod:`parsec_python.acceleration.SCF.spin_flip`): with ``auto`` the spin loop solves the up channel
alone.  The CPU tests check the detection (partner labels, the operation of a cluster and of a cell, its orbit and
sector maps, the refusals), the maps of fields, orbitals and forces, and a complete loop of a stretched H2 on the
scipy backend against both channels solved; the GPU tests compare complete runs in the symmetry sectors (energies,
levels, densities, forces) with ``off`` and with the full grid.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.acceleration.SCF.spin_flip import (
    IONIC_POTENTIAL_TOLERANCE,
    FlippedOrbitals,
    START_DENSITY_TOLERANCE,
    SpinFlipChannelState,
    WARM_START_TOLERANCE,
    detect_spin_flip,
    spin_flip_setting,
    spin_partner_labels,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, GridSettings, SpeciesPotential

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
H_POTENTIAL = DATA / "spin_ca_H_POTRE.DAT"


def _species(polarizations: dict, path=H_POTENTIAL) -> dict:
    return {
        label: SpeciesPotential(path, 0, element_symbol="H", initial_spin_polarization=value)
        for label, value in polarizations.items()
    }


def _densities(grid, atoms, polarizations):
    """Start densities of the form of initchrg.f90: Gaussians of the atoms times (1 +- p)."""

    coordinates = np.asarray(grid.coordinates)
    up = np.zeros(coordinates.shape[0])
    down = np.zeros(coordinates.shape[0])
    for atom in atoms:
        weight = np.exp(-np.sum((coordinates - np.asarray(atom.position)) ** 2, axis=1))
        up += (1.0 + polarizations[atom.symbol]) * weight
        down += (1.0 - polarizations[atom.symbol]) * weight
    return up, down


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "off"), ("OFF", "off"), ("0", "off"), ("auto", "auto"), (" on ", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": value}):
                self.assertEqual(spin_flip_setting(), expected)
        # off by default: a flip loop can settle in another state with integer occupations (Ni2, A100)
        with mock.patch.dict(os.environ):
            os.environ.pop("PARSEC_SPIN_FLIP_SYMMETRY", None)
            self.assertEqual(spin_flip_setting(), "off")

    def test_fractional_occupations_of_the_up_channel(self) -> None:
        from parsec_python.SCF.spin_polarized import SPIN_FLIP_OCCUPATION_TOLERANCE, _flip_fractional_occupations

        self.assertEqual(SPIN_FLIP_OCCUPATION_TOLERANCE, 0.05)
        largest, lines = _flip_fractional_occupations([-0.5, -0.2, 0.1], [1.0, 1.0 - 1e-9, 0.0])
        self.assertLess(largest, SPIN_FLIP_OCCUPATION_TOLERANCE)
        self.assertEqual(lines, ())
        largest, lines = _flip_fractional_occupations([-0.5, -0.2, -0.19], [1.0, 0.6, 0.4])
        self.assertAlmostEqual(largest, 0.4)
        self.assertEqual(len(lines), 2)
        self.assertIn("up level 2: -0.200000 Ry, occupation 0.6000", lines[0])
        self.assertEqual(_flip_fractional_occupations([], []), (0.0, ()))
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": "always"}), self.assertRaises(ValueError):
            spin_flip_setting()

    def test_unknown_value_is_refused_at_preparation(self) -> None:
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": "maybe"}):
            with self.assertRaises(ValueError):
                driver.prepare_single_point(object(), backend="scipy")


class PartnerTests(unittest.TestCase):
    def test_partners(self) -> None:
        atoms = [Atom("H_a", (0.0, 0.0, 1.0)), Atom("H_b", (0.0, 0.0, -1.0)), Atom("H", (0.0, 0.0, 3.0))]
        partners, reason = spin_partner_labels(atoms, _species({"H_a": 0.5, "H_b": -0.5, "H": 0.0}))
        self.assertEqual(partners, {"H_a": "H_b", "H_b": "H_a", "H": "H"}, reason)

    def test_refusals(self) -> None:
        pair = [Atom("H_a", (0.0, 0.0, 1.0)), Atom("H_b", (0.0, 0.0, -1.0))]
        cases = {
            "same sign": (pair, _species({"H_a": 0.5, "H_b": 0.5})),
            "another polarization": (pair, _species({"H_a": 0.5, "H_b": -0.4})),
            "nothing polarized": (pair, _species({"H_a": 0.0, "H_b": 0.0})),
            "another file": (
                pair,
                {
                    "H_a": SpeciesPotential(H_POTENTIAL, 0, element_symbol="H", initial_spin_polarization=0.5),
                    "H_b": SpeciesPotential(
                        DATA / "spin_pbe_H_POTRE.DAT", 0, element_symbol="H", initial_spin_polarization=-0.5
                    ),
                },
            ),
            "unequal counts": (pair + [Atom("H_a", (0.0, 0.0, 3.0))], _species({"H_a": 0.5, "H_b": -0.5})),
        }
        for name, (atoms, species) in cases.items():
            partners, reason = spin_partner_labels(atoms, species)
            self.assertIsNone(partners, name)
            self.assertTrue(reason, name)

    def test_the_file_counts_by_its_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "H_b_POTRE.DAT"
            copy.write_bytes(H_POTENTIAL.read_bytes())
            species = {
                "H_a": SpeciesPotential(H_POTENTIAL, 0, element_symbol="H", initial_spin_polarization=0.5),
                "H_b": SpeciesPotential(copy, 0, element_symbol="H", initial_spin_polarization=-0.5),
            }
            partners, _ = spin_partner_labels([Atom("H_a", (0, 0, 1)), Atom("H_b", (0, 0, -1))], species)
        self.assertEqual(partners, {"H_a": "H_b", "H_b": "H_a"})


class ClusterDetectionTests(unittest.TestCase):
    """The operation of a cluster, its orbit and sector maps, and the refusals."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=4, shift=(0.5, 0.5, 0.5)))

    def detect(self, atoms, polarizations, *, sectors=True, densities=None, **options):
        from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction
        from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition

        reduction = decomposition = None
        if sectors:
            reduction = SignedPermutationReduction.detect(self.grid, atoms)
            decomposition = ReflectionRepresentationDecomposition.build(self.grid, reduction)
        options.setdefault("electron_count", float(len(atoms)))
        return (
            detect_spin_flip(
                self.grid,
                atoms,
                _species(polarizations),
                start_densities=_densities(self.grid, atoms, polarizations) if densities is None else densities,
                field_reduction=reduction,
                decomposition=decomposition,
                **options,
            ),
            reduction,
            decomposition,
        )

    def test_stretched_dimer_takes_the_inversion(self) -> None:
        atoms = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.0))]
        (operation, reason), reduction, decomposition = self.detect(atoms, {"H_a": 0.8, "H_b": -0.8})
        self.assertIsNotNone(operation, reason)
        np.testing.assert_array_equal(operation.matrix, -np.eye(3, dtype=np.int64))
        np.testing.assert_array_equal(operation.atom_image, [1, 0])
        # the inversion commutes with the sign changes of the sectors: every sector is its own image
        np.testing.assert_array_equal(operation.sector_map, np.arange(decomposition.representation_count))
        self.assertTrue(operation.holds_for(atoms))
        self.assertFalse(operation.holds_for([Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.1))]))

    def test_square_takes_a_swap_of_axes_and_permutes_the_sectors(self) -> None:
        atoms = [
            Atom("H_a", (2.0, 0.0, 0.0)),
            Atom("H_a", (-2.0, 0.0, 0.0)),
            Atom("H_b", (0.0, 2.0, 0.0)),
            Atom("H_b", (0.0, -2.0, 0.0)),
        ]
        polarizations = {"H_a": 0.5, "H_b": -0.5}
        (operation, reason), reduction, decomposition = self.detect(atoms, polarizations)
        self.assertIsNotNone(operation, reason)
        self.assertEqual(int(np.count_nonzero(operation.matrix - np.diag(np.diag(operation.matrix)))), 2)
        self.assertEqual(reduction.group_order, 8)
        mapping = operation.sector_map
        self.assertEqual(sorted(mapping.tolist()), list(range(8)))
        self.assertTrue(np.any(mapping != np.arange(8)))
        # A state of sector chi carried by the operation is a state of sector sector_map[chi].
        rng = np.random.default_rng(3)
        phases = np.asarray(decomposition.phases, dtype=np.float64)
        orbits = np.asarray(reduction.full_to_wedge)
        for chi in range(8):
            state = phases[chi] * rng.standard_normal(reduction.wedge_size)[orbits]
            image = state[operation.inverse_rows]
            target = mapping[chi]
            amplitude = image[np.asarray(reduction.representative_rows)][orbits]
            np.testing.assert_allclose(image, amplitude * phases[target], atol=0, rtol=0)
        # fields: the orbit map gives the full-grid image
        up, down = _densities(self.grid, atoms, polarizations)
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer

        reducer = SymmetrySCFReducer(reduction)
        mapped = operation.down_field(reducer.from_full(up))
        np.testing.assert_allclose(reducer.to_full(mapped), down, atol=1e-12)
        np.testing.assert_allclose(operation.down_field(up), down, atol=1e-12)
        # labels of the down channel
        np.testing.assert_array_equal(operation.down_representations(np.arange(1, 9)), mapping + 1)

    def test_full_grid_route_without_sectors(self) -> None:
        atoms = [Atom("H_a", (0.3, 0.1, 1.5)), Atom("H_b", (-0.3, -0.1, -1.5))]
        (operation, reason), _, _ = self.detect(atoms, {"H_a": 0.5, "H_b": -0.5}, sectors=False)
        self.assertIsNotNone(operation, reason)
        self.assertIsNone(operation.orbit_map)
        self.assertIsNone(operation.sector_map)

    def test_refusals(self) -> None:
        atoms = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.0))]
        polarizations = {"H_a": 0.8, "H_b": -0.8}
        (operation, reason), _, _ = self.detect(atoms, polarizations, electron_count=3.0)
        self.assertIsNone(operation)
        self.assertIn("not even", reason)
        (operation, reason), _, _ = self.detect(
            atoms, polarizations, fixed_occupations=((1.0, 0.0), (0.0, 1.0))
        )
        self.assertIsNone(operation)
        self.assertIn("occup.in", reason)
        (operation, reason), _, _ = self.detect(atoms, polarizations, fixed_occupations=((1.0, 0.0), (1.0, 0.0)))
        self.assertIsNotNone(operation, reason)
        up, down = _densities(self.grid, atoms, polarizations)
        (operation, reason), _, _ = self.detect(atoms, polarizations, densities=(up, down * (1.0 + 1e-3)))
        self.assertIsNone(operation)
        self.assertIn("start density", reason)
        # no operation: the atoms of the partner labels are not images of each other
        crooked = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.4, -2.0))]
        (operation, reason), _, _ = self.detect(crooked, polarizations)
        self.assertIsNone(operation)
        self.assertIn("partner", reason)
        self.assertGreater(START_DENSITY_TOLERANCE, 0.0)

    def test_orbitals_and_forces(self) -> None:
        atoms = [
            Atom("H_a", (2.0, 0.0, 0.0)),
            Atom("H_a", (-2.0, 0.0, 0.0)),
            Atom("H_b", (0.0, 2.0, 0.0)),
            Atom("H_b", (0.0, -2.0, 0.0)),
        ]
        (operation, reason), _, _ = self.detect(atoms, {"H_a": 0.5, "H_b": -0.5}, sectors=False)
        self.assertIsNotNone(operation, reason)
        rng = np.random.default_rng(7)
        up = rng.standard_normal((self.grid.size, 3))
        down = operation.down_orbitals(up)
        np.testing.assert_array_equal(down[operation.rows], up)
        lazy = FlippedOrbitals(up, operation)
        np.testing.assert_array_equal(lazy.to_full_host(), down)
        np.testing.assert_array_equal(np.asarray(lazy), down)
        self.assertEqual(lazy.shape, up.shape)
        # the joined host matrix of both channels, the down columns from the up ones
        from parsec_python.SCF.spin_polarized import SpinOrbitalBlocks

        class Source:
            shape = up.shape

            def to_full_host(self):
                return up

        source = Source()
        joined = SpinOrbitalBlocks(source, FlippedOrbitals(source, operation)).to_full_host()
        np.testing.assert_array_equal(joined, np.hstack((up, down)))
        # forces: F_down(g a) = R F_up(a)
        forces = rng.standard_normal((4, 3))
        mapped = operation.down_forces(forces)
        for index in range(4):
            np.testing.assert_allclose(
                mapped[operation.atom_image[index]], operation.matrix @ forces[index], atol=1e-15
            )
        self.assertTrue(SpinFlipChannelState(operation).spin_flip_channel)

    def test_nonlocal_forces_of_a_flipped_channel(self) -> None:
        """The nonlocal forces of a result whose down channel is the image of the up channel: the up channel's
        contraction alone (one channel), plus its image under the operation."""

        from types import SimpleNamespace

        from parsec_python.acceleration.Forces import compute

        atoms = [
            Atom("H_a", (2.0, 0.0, 0.0)),
            Atom("H_a", (-2.0, 0.0, 0.0)),
            Atom("H_b", (0.0, 2.0, 0.0)),
            Atom("H_b", (0.0, -2.0, 0.0)),
        ]
        (operation, reason), _, _ = self.detect(atoms, {"H_a": 0.5, "H_b": -0.5}, sectors=False)
        self.assertIsNotNone(operation, reason)
        up_forces = np.arange(12, dtype=np.float64).reshape(4, 3)
        calls = []

        def contraction(operator, gradients, wavefunctions, occupations, atom_count):
            calls.append(np.array(occupations))
            return up_forces

        system = SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=None, device_operator=None),
            eigenproblem_solver=SimpleNamespace(),
            projector_gradients=lambda: None,
        )
        channels = (
            SimpleNamespace(occupations=np.array([1.0, 1.0, 0.0]), wavefunctions=None, eigensolver_state=None),
            SimpleNamespace(
                occupations=np.array([1.0, 1.0, 0.0]),
                wavefunctions=FlippedOrbitals(np.zeros((self.grid.size, 3)), operation),
                eigensolver_state=SpinFlipChannelState(operation),
            ),
        )
        with mock.patch.object(compute, "nonlocal_forces", contraction):
            forces = compute._spin_nonlocal_forces(system, channels, None, 4)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(forces, up_forces + operation.down_forces(up_forces), atol=1e-15)
        # both channels solved: each contracted, as before
        calls.clear()
        solved = (channels[0], SimpleNamespace(occupations=channels[1].occupations, wavefunctions=None,
                                               eigensolver_state=None))
        with mock.patch.object(compute, "nonlocal_forces", contraction):
            forces = compute._spin_nonlocal_forces(system, solved, None, 4)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(forces, 2.0 * up_forces, atol=1e-15)


def _coulomb(grid, positions, charges, softening=0.5):
    """A smooth attractive potential of point charges (Ry) on the grid: ``-2 q / sqrt(r^2 + s^2)``."""

    coordinates = np.asarray(grid.coordinates)
    values = np.zeros(coordinates.shape[0])
    for position, charge in zip(positions, charges):
        distance = np.sqrt(np.sum((coordinates - np.asarray(position)) ** 2, axis=1) + softening**2)
        values -= 2.0 * charge / distance
    return values


class ExternalTermTests(unittest.TestCase):
    """The local ionic potential decides between the operations: point charges or a field that break one."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=4, shift=(0.5, 0.5, 0.5)))
        cls.atoms = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.0))]
        cls.polarizations = {"H_a": 0.8, "H_b": -0.8}
        cls.ionic = _coulomb(cls.grid, [atom.position for atom in cls.atoms], [1.0, 1.0])

    def detect(self, ionic):
        return detect_spin_flip(
            self.grid,
            self.atoms,
            _species(self.polarizations),
            electron_count=2.0,
            start_densities=_densities(self.grid, self.atoms, self.polarizations),
            ionic_potential=ionic,
        )

    def test_the_atoms_alone_keep_the_first_operation(self) -> None:
        plain, reason = self.detect(None)
        self.assertIsNotNone(plain, reason)
        operation, reason = self.detect(self.ionic)
        self.assertIsNotNone(operation, reason)
        np.testing.assert_array_equal(operation.matrix, plain.matrix)
        deviation, bound = operation.ionic_potential_check(self.ionic)
        self.assertLessEqual(deviation, bound)
        self.assertEqual(bound, IONIC_POTENTIAL_TOLERANCE * max(1.0, float(np.max(np.abs(self.ionic)))))

    def test_a_point_charge_that_breaks_every_operation(self) -> None:
        charge = _coulomb(self.grid, [(0.1, 0.2, 3.7)], [0.5])
        operation, reason = self.detect(self.ionic + charge)
        self.assertIsNone(operation)
        self.assertIn("local ionic potential", reason)
        # a very weak charge still breaks it: the bound is the rounding, not a physical scale
        operation, reason = self.detect(self.ionic + 1.0e-6 * charge)
        self.assertIsNone(operation)

    def test_a_field_leaves_the_operations_that_keep_it(self) -> None:
        coordinates = np.asarray(self.grid.coordinates)
        for axis in range(2):
            field = 2.0 * 0.001 * coordinates[:, axis]
            operation, reason = self.detect(self.ionic + field)
            self.assertIsNotNone(operation, reason)
            # the operation carries H_a onto H_b and keeps the field: x (or y) is not reversed
            self.assertEqual(int(operation.matrix[axis, axis]), 1)
            self.assertEqual(int(operation.matrix[2, 2]), -1)
            deviation, bound = operation.ionic_potential_check(self.ionic + field)
            self.assertLessEqual(deviation, bound)
        # a field along the bond breaks every operation that exchanges the atoms
        operation, reason = self.detect(self.ionic + 2.0 * 0.001 * coordinates[:, 2])
        self.assertIsNone(operation)
        self.assertIn("local ionic potential", reason)

    def test_warm_start_and_atom_checks(self) -> None:
        operation, reason = self.detect(self.ionic)
        self.assertIsNotNone(operation, reason)
        up, _ = _densities(self.grid, self.atoms, self.polarizations)
        potential = self.ionic - 0.1 * up
        deviation, bound = operation.warm_start_check(potential, operation.down_field(potential))
        self.assertEqual(deviation, 0.0)
        self.assertEqual(bound, WARM_START_TOLERANCE)
        deviation, _ = operation.warm_start_check(potential, potential)
        self.assertGreater(deviation, 1.0e-3)
        self.assertEqual(operation.atom_deviation(self.atoms), 0.0)
        moved = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.0 - 3.0e-8))]
        self.assertAlmostEqual(operation.atom_deviation(moved), 3.0e-8, delta=1e-12)
        relabelled = [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_a", (0.0, 0.0, -2.0))]
        self.assertEqual(operation.atom_deviation(relabelled), float("inf"))


class DetectionCostTests(unittest.TestCase):
    """The candidates of a cell are filtered by a few probe atoms before the full map of the atoms is formed."""

    @staticmethod
    def cell(n, labels):
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        side = 4.0 * n
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=np.diag([side] * 3)),
            PeriodicGridSettings(spacing=1.0, expansion_order=2, shift=(0.0, 0.0, 0.0)),
        )
        sites = np.stack(np.meshgrid(*(np.arange(n) * 4.0,) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        atoms = [Atom(str(label), tuple(site)) for label, site in zip(labels(sites), sites)]
        return grid, atoms

    def test_random_labels_have_no_operation_and_few_full_maps(self) -> None:
        from parsec_python.acceleration.SCF import spin_flip

        rng = np.random.default_rng(1)

        def labels(sites):
            names = np.array(["H_a"] * (len(sites) // 2) + ["H_b"] * (len(sites) - len(sites) // 2))
            rng.shuffle(names)
            return names

        grid, atoms = self.cell(6, labels)
        calls = []
        original = spin_flip._PartnerAtoms.image

        def counted(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        with mock.patch.object(spin_flip._PartnerAtoms, "image", counted):
            operation, reason = detect_spin_flip(grid, atoms, _species({"H_a": 0.5, "H_b": -0.5}), electron_count=216.0)
        self.assertIsNone(operation)
        self.assertIn("no operation maps every atom", reason)
        # 48 matrices with about 108 translations each: the probes leave (almost) none for the full map
        self.assertLessEqual(len(calls), 2)

    def test_checkerboard_takes_a_translation_by_one_site(self) -> None:
        def labels(sites):
            parity = np.rint(np.sum(sites, axis=1) / 4.0).astype(int) % 2
            return np.where(parity == 0, "H_a", "H_b")

        grid, atoms = self.cell(4, labels)
        operation, reason = detect_spin_flip(grid, atoms, _species({"H_a": 0.5, "H_b": -0.5}), electron_count=64.0)
        self.assertIsNotNone(operation, reason)
        np.testing.assert_array_equal(operation.matrix, np.eye(3, dtype=np.int64))
        self.assertEqual(int(np.sum(np.asarray(operation.translation) % 16 != 0)), 1)
        self.assertTrue(operation.holds_for(atoms))
        positions = np.asarray([atom.position for atom in atoms])
        for index, image in enumerate(operation.atom_image):
            self.assertNotEqual(atoms[index].symbol, atoms[int(image)].symbol)
            difference = positions[int(image)] - positions[index]
            self.assertAlmostEqual(float(np.linalg.norm(difference - 16.0 * np.rint(difference / 16.0))), 4.0)


class GuardHintTests(unittest.TestCase):
    """Test 2 of the spin guard names an operation that carries one channel onto the other (a same-label start)."""

    def test_broken_symmetry_state_of_one_label(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import density_symmetry_breaking

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=4, shift=(0.5, 0.5, 0.5)))
        atoms = [Atom("H", (0.0, 0.0, 2.0)), Atom("H", (0.0, 0.0, -2.0))]
        up, down = _densities(grid, [Atom("H_a", (0.0, 0.0, 2.0)), Atom("H_b", (0.0, 0.0, -2.0))],
                              {"H_a": 0.8, "H_b": -0.8})
        report = density_symmetry_breaking([up, down], grid, atoms)
        self.assertIsNotNone(report)
        self.assertTrue(any("carries one channel onto the other" in line for line in report.lines))
        # a state that breaks the operation in one channel alone gets no such line
        report = density_symmetry_breaking([up, 0.5 * (up + down)], grid, atoms)
        self.assertIsNotNone(report)
        self.assertFalse(any("carries one channel onto the other" in line for line in report.lines))


class CellDetectionTests(unittest.TestCase):
    """A bulk cell: the antiferromagnetic CsCl arrangement takes the translation by half the body diagonal."""

    def test_half_cell_translation(self) -> None:
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.acceleration.Symmetry.axis_reflection import (
            SignedPermutationReduction,
            periodic_abelian_reduction,
        )
        from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        side = 6.0
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=np.diag([side] * 3)),
            PeriodicGridSettings(spacing=side / 12, expansion_order=4, shift=(0.0, 0.0, 0.0)),
        )
        atoms = [Atom("H_a", (0.0, 0.0, 0.0)), Atom("H_b", (3.0, 3.0, 3.0))]
        polarizations = {"H_a": 0.6, "H_b": -0.6}
        reduction = periodic_abelian_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        coordinates = np.asarray(grid.coordinates)
        up = np.zeros(grid.size)
        down = np.zeros(grid.size)
        for atom in atoms:
            difference = coordinates - np.asarray(atom.position)
            difference -= side * np.rint(difference / side)
            weight = np.exp(-np.sum(difference * difference, axis=1))
            up += (1.0 + polarizations[atom.symbol]) * weight
            down += (1.0 - polarizations[atom.symbol]) * weight
        operation, reason = detect_spin_flip(
            grid,
            atoms,
            _species(polarizations),
            electron_count=2.0,
            start_densities=(up, down),
            field_reduction=reduction,
            decomposition=decomposition,
        )
        self.assertIsNotNone(operation, reason)
        np.testing.assert_array_equal(operation.matrix, np.eye(3, dtype=np.int64))
        np.testing.assert_array_equal(operation.translation, [6, 6, 6])
        np.testing.assert_allclose(operation.down_field(up), down, atol=1e-12)
        self.assertIn("grid steps", operation.description)
        self.assertTrue(operation.holds_for([Atom("H_a", (6.0, 0.0, 0.0)), Atom("H_b", (3.0, 3.0, -3.0))]))


def _write_h2(
    directory: Path, *, distance=4.0, polarization=0.8, extra=(), radius=7.0, spacing=0.5, signs=(1.0, -1.0)
) -> Path:
    for label in ("H_a", "H_b"):
        (directory / f"{label}_POTRE.DAT").write_bytes(H_POTENTIAL.read_bytes())
    lines = [
        f"Boundary_Sphere_Radius: {radius}",
        f"Grid_Spacing: {spacing}",
        "States_Num: 4",
        "Max_Iter: 200",
        "Convergence_Criterion: 1.0e-8",
        "Eigensolver: chebff",
        "Correlation_Type: ca",
        "Spin_Polarization: .true.",
        "Atom_Types_Num: 2",
    ]
    for label, sign, polarization_sign in (("H_a", 1.0, signs[0]), ("H_b", -1.0, signs[1])):
        lines += [
            f"Atom_Type: {label}",
            "Element_Symbol: H",
            "Local_Component: s",
            f"Initial_Spin_Polarization: {polarization_sign * polarization}",
            "begin Atom_Coord",
            f"   0.0 0.0 {sign * distance / 2}",
            "end Atom_Coord",
        ]
    (directory / "parsec.in").write_text("\n".join(lines + list(extra)) + "\n")
    return directory / "parsec.in"


def _run(path: Path, mode: str, *, backend: str, symmetry: str = "auto", forces: bool = False, extra_env=None):
    from parsec_python.Input import parse_parsec_input
    from parsec_python.acceleration import driver

    environment = {"PARSEC_SPIN_FLIP_SYMMETRY": mode, **(extra_env or {})}
    with mock.patch.dict(os.environ, environment):
        translation = parse_parsec_input(path)
        system = driver.prepare_single_point(translation.problem, backend=backend, symmetry=symmetry)
        system.materialize_final_wavefunctions = True
        result = driver.run_scf(system)
        force = driver.compute_forces(system, result).total if forces else None
    return system, result, force


class ScipyLoopTests(unittest.TestCase):
    """A complete loop of a stretched H2 (two labels) on the scipy backend: the full grid, one channel solved."""

    def test_stretched_dimer(self) -> None:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = _write_h2(directory, radius=6.0, spacing=0.55)
        _, both, both_forces = _run(path, "off", backend="scipy", forces=True)
        system, flip, flip_forces = _run(path, "auto", backend="scipy", forces=True)
        details = dict(flip.backend.details)
        self.assertIn("the up channel is solved", details["spin_flip_symmetry"])
        self.assertNotIn("spin_flip_symmetry", dict(both.backend.details))
        self.assertTrue(both.converged and flip.converged)
        self.assertAlmostEqual(flip.energies.total, both.energies.total, delta=1e-9)
        for index in range(2):
            np.testing.assert_allclose(flip.spin[index].eigenvalues, both.spin[index].eigenvalues, atol=1e-7)
            np.testing.assert_allclose(flip.spin[index].density, both.spin[index].density, atol=1e-6)
        self.assertIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
        # the broken-symmetry state: opposite moments on the two atoms
        moment = flip.spin[0].density - flip.spin[1].density
        self.assertGreater(float(np.sum(np.abs(moment)) * flip.grid.volume_element), 0.5)
        np.testing.assert_allclose(flip_forces, both_forces, atol=1e-6)


class ScipyFractionalTests(unittest.TestCase):
    """A stretched H2 (two labels) at a high Fermi temperature: the up channel of the flip loop ends with fractional
    occupations, so the loop is not kept and runs again with both channels; the energy of off, the note in
    parsec.out."""

    def test_a_fractional_up_level_solves_both_channels(self) -> None:
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF.spin_polarized import SPIN_FLIP_OCCUPATION_TOLERANCE
        from parsec_python.acceleration import driver
        from parsec_python.models import SpinSectorFallbackRecord

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        hot = ("Fermi_Temp: 40000",)
        path = _write_h2(directory, radius=6.0, spacing=0.55, extra=hot)
        _, both, _ = _run(path, "off", backend="scipy")
        records = []
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": "auto"}):
            translation = parse_parsec_input(path)
            system = driver.prepare_single_point(translation.problem, backend="scipy")
            self.assertIsNotNone(system.spin_flip)
            flip = driver.run_scf(system, callback=records.append)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].route, "spin_flip")
        self.assertIn("fractional occupation", notes[0].reason)
        self.assertGreaterEqual(notes[0].largest_fractional_occupation, SPIN_FLIP_OCCUPATION_TOLERANCE)
        # the second loop solved both channels: the down channel has a subspace of its own, the energy of off
        self.assertNotIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
        self.assertTrue(both.converged and flip.converged)
        self.assertAlmostEqual(flip.energies.total, both.energies.total, delta=1e-9)
        # parsec.out of the command line says so
        cli_directory = directory / "cli"
        cli_directory.mkdir()
        _write_h2(cli_directory, radius=6.0, spacing=0.55, extra=hot)
        _, text = _cli(cli_directory, "auto")
        self.assertIn("NOTE: PARSEC_SPIN_FLIP_SYMMETRY: the up channel of the converged loop holds a fractional", text)
        self.assertIn("this SCF loop solves both spin channels", text)
        self.assertAlmostEqual(_last_energy(text), both.energies.total, delta=1e-8)

    def test_the_default_solves_both_channels(self) -> None:
        # off by default: an antiferromagnetic start of two labels has no operation attached and no detail
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration import driver

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = _write_h2(directory, radius=5.0, spacing=0.6)
        with mock.patch.dict(os.environ):
            os.environ.pop("PARSEC_SPIN_FLIP_SYMMETRY", None)
            translation = parse_parsec_input(path)
            system = driver.prepare_single_point(translation.problem, backend="scipy")
        self.assertIsNone(system.spin_flip)
        self.assertNotIn("spin_flip_symmetry", dict(system.backend_info.details))


class _OtherError(RuntimeError):
    pass


class LeaveRouteTests(unittest.TestCase):
    """How run_spin_scf leaves the route: a failed test of the spin sectors in a flip loop, the mark of the system
    for the rest of the run, the warm start of the second loop (the loop itself mocked)."""

    @staticmethod
    def _abandoned():
        from parsec_python.acceleration.SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        report = FractionalOccupations(
            largest=0.5, tolerance=0.05, lines=("up pi*: 0.5000",), reason="a partly filled multiplet",
            measure="largest min(f, 1-f)",
        )
        return SpinSectorsAbandoned("the start Hamiltonian", report)

    @staticmethod
    def _system():
        from types import SimpleNamespace

        from parsec_python.acceleration.models import BackendInfo

        info = BackendInfo(
            requested="auto", selected="cupy",
            details=(("spin_flip_symmetry", "auto: the up channel is solved"), ("other", "x")),
        )
        return SimpleNamespace(
            spin_flip=SimpleNamespace(description="inversion"), backend_info=info, backend=SimpleNamespace(info=info)
        )

    @staticmethod
    def _loops(first_error, *, take=True):
        calls = []

        def loop(system, *, flip_taken=None, **keywords):
            calls.append((flip_taken, keywords))
            if len(calls) == 1:
                if take and flip_taken is not None:
                    flip_taken.append("inversion")
                raise first_error
            return "second loop"

        return calls, loop

    def test_a_failed_sector_test_in_a_flip_loop_solves_both_channels_in_the_same_sectors(self) -> None:
        from parsec_python.SCF import spin_polarized
        from parsec_python.models import SpinSectorFallbackRecord

        system = self._system()
        calls, loop = self._loops(self._abandoned())
        records = []
        with mock.patch.object(spin_polarized, "_run_spin_loop", loop):
            self.assertEqual(spin_polarized.run_spin_scf(system, callback=records.append), "second loop")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[1][0])
        (record,) = records
        self.assertIsInstance(record, SpinSectorFallbackRecord)
        self.assertEqual(record.route, "spin_flip")
        self.assertEqual(record.stage, "the start Hamiltonian")
        self.assertIn("a test of the spin sectors failed in a loop of the route (a partly filled multiplet", record.reason)
        self.assertEqual(record.largest_fractional_occupation, 0.5)
        self.assertEqual(record.multiplets, ("up pi*: 0.5000",))
        # the mark: no operation for the rest of the run, and the detail says so
        self.assertIn("the start Hamiltonian", getattr(system, spin_polarized.SPIN_FLIP_LEFT))
        flip, refusal = spin_polarized._spin_flip_of(
            system, exchange_term=None, meta=None, hubbard_projectors=None, ionic_potential=None
        )
        self.assertIsNone(flip)
        self.assertIsNone(refusal)
        details = dict(system.backend_info.details)
        self.assertTrue(details["spin_flip_symmetry"].startswith("auto: both channels solved (the run left the route"))
        self.assertEqual(details["other"], "x")
        self.assertIs(system.backend.info, system.backend_info)
        # parsec.out
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines

        text = "\n".join(spin_sector_fallback_lines(record))
        self.assertIn("NOTE: PARSEC_SPIN_FLIP_SYMMETRY: a test of the spin sectors failed", text)
        self.assertIn("this SCF loop solves both spin channels", text)

    def test_other_errors_and_loops_without_the_flip_are_not_caught(self) -> None:
        from parsec_python.SCF import spin_polarized
        from parsec_python.acceleration.SCF.spin_sectors import SpinSectorsAbandoned

        system = self._system()
        # a loop that solved both channels: the guard of the spin sectors decides, as before
        calls, loop = self._loops(self._abandoned(), take=False)
        with mock.patch.object(spin_polarized, "_run_spin_loop", loop), self.assertRaises(SpinSectorsAbandoned):
            spin_polarized.run_spin_scf(system)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(getattr(system, spin_polarized.SPIN_FLIP_LEFT, None))
        # another error of a flip loop is not a reason to leave the route
        calls, loop = self._loops(_OtherError("no"))
        with mock.patch.object(spin_polarized, "_run_spin_loop", loop), self.assertRaises(_OtherError):
            spin_polarized.run_spin_scf(system)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(getattr(system, spin_polarized.SPIN_FLIP_LEFT, None))

    def test_a_fractional_up_channel_marks_the_system_and_drops_the_saved_subspaces(self) -> None:
        from parsec_python.SCF import SCFWarmStart
        from parsec_python.SCF import spin_polarized

        system = self._system()
        calls, loop = self._loops(spin_polarized._SpinFlipFractional(0.37, ("up level 1",)))
        warm = SCFWarmStart(
            input_potential=(np.zeros(2), np.zeros(2)), hartree_potential=np.zeros(2),
            eigensolver_state=("up", "down"), first_iteration_degree=12,
        )
        records = []
        with mock.patch.object(spin_polarized, "_run_spin_loop", loop):
            spin_polarized.run_spin_scf(system, callback=records.append, warm_start=warm)
        self.assertIs(calls[0][1]["warm_start"], warm)
        second = calls[1][1]["warm_start"]
        self.assertIsNone(second.eigensolver_state)
        self.assertIsNone(second.first_iteration_degree)
        self.assertIs(second.input_potential, warm.input_potential)
        self.assertIn("fractional occupation of the up channel", getattr(system, spin_polarized.SPIN_FLIP_LEFT))
        self.assertIn("fractional occupation", records[0].reason)

    def test_a_system_prepared_again_keeps_both_channels(self) -> None:
        # driver._SpinSectorFallback fills the same system object again: the mark stays and corrects the new detail
        from dataclasses import replace

        from parsec_python.SCF import spin_polarized
        from parsec_python.acceleration import driver

        system = self._system()
        driver._keep_spin_flip_left(system)
        self.assertEqual(dict(system.backend_info.details)["spin_flip_symmetry"], "auto: the up channel is solved")
        spin_polarized.leave_spin_flip(system, "a reason")
        fresh = replace(system.backend_info, details=(("spin_flip_symmetry", "auto: the up channel is solved"),))
        system.backend_info = system.backend.info = fresh
        driver._keep_spin_flip_left(system)
        self.assertEqual(
            dict(system.backend_info.details)["spin_flip_symmetry"],
            "auto: both channels solved (the run left the route: a reason)",
        )
        self.assertIs(system.backend.info, system.backend_info)


def _point_charge(position, charge=0.5) -> tuple[str, ...]:
    return (
        "Add_Point_Charges: .true.",
        "Point_Typ_Num: 1",
        f"Pt_Chg: {charge}",
        "begin Point_Coord",
        "  {:.6f} {:.6f} {:.6f}".format(*position),
        "end Point_Coord",
    )


def _cli(directory: Path, mode: str) -> tuple[int, str]:
    """The accelerated CLI on the scipy backend in ``directory``: its return code and ``parsec.out``."""

    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    from parsec_python.acceleration import cli as cli_module

    previous = Path.cwd()
    os.chdir(directory)
    try:
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": mode}), redirect_stdout(
            StringIO()
        ), redirect_stderr(StringIO()) as error:
            code = cli_module.main(["parsec.in", "--device", "cpu", "--backend", "scipy"])
    finally:
        os.chdir(previous)
    if code:
        raise AssertionError(f"the CLI returned {code}: {error.getvalue()[-3000:]}")
    return code, (directory / "parsec.out").read_text(errors="replace")


def _last_energy(text: str) -> float:
    return float(re.findall(r"Total Energy =\s+(-?\d+\.\d+)", text)[-1])


class ScipyExternalTermTests(unittest.TestCase):
    """Point charges, Polarizability and Restart_Run of a stretched H2 (two labels) on the scipy backend: the route
    where an operation keeps the ionic potential and the warm start, both channels elsewhere, and off and auto
    agree; ``parsec.out`` says which."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def folder(self, name: str) -> Path:
        path = self.directory / name
        path.mkdir()
        return path

    def test_point_charges(self) -> None:
        # off every plane of the dimer: no operation keeps the charge, both channels are solved, as with off
        path = _write_h2(self.folder("off_planes"), radius=6.0, spacing=0.55, extra=_point_charge((0.1, 0.2, 3.7)))
        _, both, _ = _run(path, "off", backend="scipy")
        _, flip, _ = _run(path, "auto", backend="scipy")
        detail = dict(flip.backend.details)["spin_flip_symmetry"]
        self.assertIn("both channels solved", detail)
        self.assertIn("local ionic potential", detail)
        self.assertNotIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
        self.assertAlmostEqual(flip.energies.total, both.energies.total, delta=1e-10)
        self.assertGreater(float(np.max(np.abs(flip.spin[0].eigenvalues - flip.spin[1].eigenvalues))), 1.0e-3)
        # on the mirror plane z = 0: the mirror exchanges the atoms and keeps the charge
        path = _write_h2(self.folder("on_plane"), radius=6.0, spacing=0.55, extra=_point_charge((3.7, 0.0, 0.0)))
        _, both, _ = _run(path, "off", backend="scipy")
        system, flip, _ = _run(path, "auto", backend="scipy")
        self.assertIn("the up channel is solved", dict(flip.backend.details)["spin_flip_symmetry"])
        self.assertEqual(int(system.spin_flip.matrix[0, 0]), 1)
        self.assertEqual(int(system.spin_flip.matrix[2, 2]), -1)
        self.assertIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
        self.assertTrue(both.converged and flip.converged)
        self.assertAlmostEqual(flip.energies.total, both.energies.total, delta=1e-9)
        for index in range(2):
            np.testing.assert_allclose(flip.spin[index].eigenvalues, both.spin[index].eigenvalues, atol=1e-7)

    def test_polarizability(self) -> None:
        records = {}
        for mode in ("off", "auto"):
            directory = self.folder(mode)
            _write_h2(directory, radius=6.0, spacing=0.55, extra=("Polarizability: .true.",))
            _, text = _cli(directory, mode)
            rows = []
            for line in (directory / "polar.dat").read_text().splitlines():
                # format 93 of polar.f90: direction (F4.0, F3.0, F3.0, so "0.-1. 0."), energy, dipole
                if not re.match(r"^\s*-?\d+\.", line):
                    continue
                values = [float(item) for item in re.findall(r"-?\d+\.\d*", line)]
                if len(values) == 7:
                    rows.append(values)
            self.assertEqual(len(rows), 7, mode)
            records[mode] = (np.asarray(rows), text)
        both, flip = records["off"][0], records["auto"][0]
        np.testing.assert_allclose(flip[:, 3], both[:, 3], atol=1e-7)
        np.testing.assert_allclose(flip[:, 4:], both[:, 4:], atol=1e-4)
        # the field loops have dipoles along their fields
        self.assertGreater(float(np.max(np.abs(flip[1:, 4:]))), 1.0e-2)
        text = records["auto"][1]
        self.assertIn("NOTE: PARSEC_SPIN_FLIP_SYMMETRY=auto: the up channel is solved", text)
        # the zero-field loop takes the route, the six loops with a field solve both channels
        self.assertEqual(text.count("this SCF loop solves both spin channels"), 6)
        self.assertEqual(text.count("does not map the local ionic potential of this loop onto itself"), 6)
        self.assertNotIn("PARSEC_SPIN_FLIP_SYMMETRY", records["off"][1])

    def test_restart_run(self) -> None:
        # a ferromagnetic first run: its parsec.dat holds a down potential that is no image of the up potential
        first = self.folder("first")
        _write_h2(first, radius=6.0, spacing=0.55, signs=(1.0, 1.0))
        _cli(first, "off")
        energies = {}
        for mode in ("off", "auto"):
            directory = self.directory / f"restart_{mode}"
            shutil.copytree(first, directory)
            _write_h2(directory, radius=6.0, spacing=0.55, extra=("Restart_Run: .true.",))
            _, text = _cli(directory, mode)
            energies[mode] = _last_energy(text)
            if mode == "auto":
                self.assertIn("NOTE: PARSEC_SPIN_FLIP_SYMMETRY=auto: the up channel is solved", text)
                self.assertIn("the down potential of the warm start", text)
                self.assertIn("this SCF loop solves both spin channels", text)
        # the restart continues the state of parsec.dat in both modes
        self.assertAlmostEqual(energies["auto"], energies["off"], delta=1e-9)
        # a restart of a state that has the operation keeps the route
        start = self.folder("afm")
        _write_h2(start, radius=6.0, spacing=0.55)
        _, text = _cli(start, "auto")
        reference = _last_energy(text)
        restart = self.directory / "afm_restart"
        shutil.copytree(start, restart)
        _write_h2(restart, radius=6.0, spacing=0.55, extra=("Restart_Run: .true.",))
        _, text = _cli(restart, "auto")
        self.assertIn("the up channel is solved", text)
        self.assertNotIn("this SCF loop solves both spin channels", text)
        self.assertAlmostEqual(_last_energy(text), reference, delta=1e-7)


class OutputLineTests(unittest.TestCase):
    def test_note_of_a_loop_that_solves_both_channels(self) -> None:
        from parsec_python.SCF.spin_polarized import _spin_flip_record
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines

        record = _spin_flip_record(("the operation [[1]] does not map it", "largest deviation [Ry]", 0.25, 1e-9))
        self.assertEqual(record.route, "spin_flip")
        text = "\n".join(spin_sector_fallback_lines(record))
        self.assertIn("NOTE: PARSEC_SPIN_FLIP_SYMMETRY: the operation [[1]] does not map it;", text)
        self.assertIn("this SCF loop solves both spin channels", text)
        self.assertIn("largest deviation [Ry] = 2.500E-01, allowed 1.0E-09", text)
        text = "\n".join(spin_sector_fallback_lines(_spin_flip_record(("LDA+U", "", float("nan"), float("nan")))))
        self.assertNotIn("allowed", text)


class WeightedChannelTests(unittest.TestCase):
    def test_the_flipped_channel_takes_the_weights_and_partners_of_the_up_channel(self) -> None:
        # (with PARSEC_ORBITAL_GROUP_SPIN) In weighted sectors (H2 along z: C4v) the loop reads the state
        # weights of both channels; the down channel taken as the image of the up one has those of the up channel.
        from types import SimpleNamespace

        from parsec_python.SCF.spin_polarized import _channel_state_weights, _take_flipped_channel

        flip = SimpleNamespace(
            down_orbitals=lambda values: values, down_representations=lambda labels: labels[::-1],
            channel_state=lambda: "flipped",
        )
        eigenvalues = [np.array([-1.0, -0.5]), None]
        wavefunctions, representations, states = [np.eye(2), None], [np.array([1, 2]), None], ["up", None]
        weights, partners = [np.array([1.0, 2.0]), None], [(0, 4), None]
        _take_flipped_channel(flip, eigenvalues, wavefunctions, representations, states, weights, partners)
        np.testing.assert_array_equal(weights[1], [1.0, 2.0])
        self.assertIsNot(weights[1], weights[0])
        self.assertEqual(partners[1], (0, 4))
        self.assertEqual(states[1], "flipped")
        np.testing.assert_array_equal(_channel_state_weights(weights)[1], [1.0, 2.0])
        # abelian sectors or the full grid: no weights in either channel, as before
        weights = [None, None]
        _take_flipped_channel(flip, eigenvalues, wavefunctions, representations, states, weights, [None, None])
        self.assertIsNone(_channel_state_weights(weights))


class PublicChannelTests(unittest.TestCase):
    """The down channel of a result of weighted sectors lists the images of every partner of the up channel."""

    def test_the_columns_follow_the_listed_levels(self) -> None:
        from dataclasses import dataclass
        from types import SimpleNamespace

        from parsec_python.SCF.spin_polarized import SpinOrbitalBlocks, _public_channels

        @dataclass(frozen=True)
        class Orbitals:
            values: np.ndarray
            state_weights: np.ndarray | None = None

            @property
            def shape(self):
                return (self.values.shape[0], int(np.sum(self.state_weights)) if self.state_weights is not None
                        else self.values.shape[1])

            def to_full_host(self, block_states: int = 8) -> np.ndarray:
                if self.state_weights is None:
                    return self.values.copy()
                return np.repeat(self.values, self.state_weights, axis=1)

        up = Orbitals(np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
        flip = SimpleNamespace(inverse_rows=np.array([2, 0, 1]))
        down = FlippedOrbitals(up, flip)
        weights = np.array([1, 3])
        levels = np.array([-1.0, -0.5])
        filled = np.array([1.0, 1.5])
        sectors = np.array([1, 2])
        public = _public_channels([levels, levels], [filled, filled], [sectors, sectors], [up, down], [weights, weights])
        self.assertEqual(public[1][3].shape, (3, 4))
        self.assertIs(public[1][3].source, public[0][3])
        np.testing.assert_array_equal(public[1][0], [-1.0, -0.5, -0.5, -0.5])
        joined = SpinOrbitalBlocks(public[0][3], public[1][3]).to_full_host()
        self.assertEqual(joined.shape, (3, 8))
        np.testing.assert_array_equal(joined[:, 4:], joined[:, :4][flip.inverse_rows])
        # without weights the channels are those of the loop
        plain = _public_channels([levels, levels], [filled, filled], [sectors, sectors], [up, down], [None, None])
        self.assertIs(plain[0][3], up)
        self.assertIs(plain[1][3], down)


_IRON_PAIR = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.42000000 5.42000000 5.42000000
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.22
Expansion_Order: 12
States_Num: 16
Fermi_Temp: 1000.0
Max_Iter: 100
Convergence_Criterion: 1.0e-6
Diag_Tolerance: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.2
Correlation_Type: pbe
Spin_Polarization: true
Atom_Types_Num: 2
Atom_Type: Fe_a
Element_Symbol: Fe
Local_Component: s
Initial_Spin_Polarization: 0.5
begin Atom_Coord
  0.0 0.0 0.0
end Atom_Coord
Atom_Type: Fe_b
Element_Symbol: Fe
Local_Component: s
Initial_Spin_Polarization: -0.5
begin Atom_Coord
  2.71 2.71 2.71
end Atom_Coord
"""


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class FlippedChannelFileTests(unittest.TestCase):
    """``parsec.dat`` of antiferromagnetic bcc Fe2 in the weighted sectors with the down channel as the image."""

    @staticmethod
    def _run(directory: Path, setting: str):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        import parsec_python.acceleration.cli as cli_module
        from parsec_python.Output.parsec_dat import read_parsec_dat

        directory.mkdir()
        (directory / "parsec.in").write_text(_IRON_PAIR)
        for label in ("Fe_a", "Fe_b"):
            shutil.copyfile(DATA / "spin_pbe_Fe_POTRE.DAT", directory / f"{label}_POTRE.DAT")
        errors = StringIO()
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": setting}), redirect_stdout(StringIO()), \
                redirect_stderr(errors):
            status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy"])
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        if status != 0:
            raise AssertionError(f"{setting}: {errors.getvalue()}\n{log[-3000:]}")
        return read_parsec_dat(directory / "parsec.dat"), log

    def test_the_file_holds_the_images_of_every_partner(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            flipped, log = self._run(Path(name) / "auto", "auto")
            solved, _ = self._run(Path(name) / "off", "off")
        self.assertIn("weights", log)
        volume = float(np.prod(flipped.step))
        for spin in range(2):
            levels = flipped.levels[spin][0]
            block = flipped.wavefunctions[spin][0]
            count = int(np.flatnonzero(levels.occupations > 0.0)[-1]) + 1
            self.assertEqual(block.vectors.shape[1], count)
            np.testing.assert_array_equal(block.state_numbers, np.arange(1, count + 1))
            # orthonormal columns, whose occupations give the density of the channel in the file
            np.testing.assert_allclose(block.vectors.T @ block.vectors, np.eye(count), atol=2e-6)
            density = np.sum(levels.occupations[:count] * block.vectors**2, axis=1) / volume
            np.testing.assert_allclose(density, flipped.densities[spin], atol=2e-6)
            # the levels and densities of both channels solved
            np.testing.assert_allclose(levels.energies, solved.levels[spin][0].energies, atol=2e-6)
            np.testing.assert_allclose(flipped.densities[spin], solved.densities[spin], atol=2e-5)
        # the down orbitals are the images of the up ones: the same levels, the density moved to the other atom
        np.testing.assert_array_equal(flipped.levels[0][0].energies, flipped.levels[1][0].energies)
        self.assertGreater(float(np.max(np.abs(flipped.densities[0] - flipped.densities[1]))), 1.0e-3)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class SectorRunTests(unittest.TestCase):
    """Complete CuPy runs in the symmetry sectors: one channel solved against both, and against the full grid."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def assert_same(self, reference, result, *, energy=1e-8, levels=2e-7, density=2e-6) -> None:
        self.assertTrue(reference.converged and result.converged)
        self.assertAlmostEqual(result.energies.total, reference.energies.total, delta=energy)
        for index in range(2):
            np.testing.assert_allclose(result.spin[index].eigenvalues, reference.spin[index].eigenvalues, atol=levels)
            np.testing.assert_allclose(result.spin[index].occupations, reference.spin[index].occupations, atol=1e-8)
            np.testing.assert_allclose(result.spin[index].density, reference.spin[index].density, atol=density)

    def test_stretched_dimer_in_the_sectors(self) -> None:
        path = _write_h2(self.directory)
        _, both, both_forces = _run(path, "off", backend="auto", forces=True)
        system, flip, flip_forces = _run(path, "auto", backend="auto", forces=True)
        details = dict(flip.backend.details)
        self.assertTrue(details["orbital_symmetry"].startswith("CuPy real"))
        self.assertIn("the up channel is solved", details["spin_flip_symmetry"])
        self.assert_same(both, flip)
        np.testing.assert_allclose(flip_forces, both_forces, atol=1e-6)
        # only the up channel kept subspaces
        self.assertIn("up:", details["orbital_sector_final_state_counts"])
        self.assertNotIn("down:", details["orbital_sector_final_state_counts"])
        np.testing.assert_array_equal(flip.spin[1].representations, flip.spin[0].representations)
        # the full grid of both channels
        with mock.patch.dict(os.environ, {"PARSEC_SPIN_FLIP_SYMMETRY": "off"}):
            from parsec_python.Input import parse_parsec_input
            from parsec_python.acceleration import driver

            translation = parse_parsec_input(_write_h2(self.directory, extra=("Spin_Symmetry_Sectors: off",)))
            full_system = driver.prepare_single_point(translation.problem, backend="auto", symmetry="auto")
            full = driver.run_scf(full_system)
        self.assert_same(full, flip, energy=1e-7, levels=1e-6, density=1e-5)

    def test_square_permutes_the_sectors_of_the_down_channel(self) -> None:
        for label in ("H_a", "H_b"):
            (self.directory / f"{label}_POTRE.DAT").write_bytes(H_POTENTIAL.read_bytes())
        lines = [
            "Boundary_Sphere_Radius: 8.0",
            "Grid_Spacing: 0.5",
            "States_Num: 6",
            "Max_Iter: 200",
            "Convergence_Criterion: 1.0e-8",
            "Eigensolver: chebff",
            "Correlation_Type: ca",
            "Spin_Polarization: .true.",
            "Atom_Types_Num: 2",
            "Atom_Type: H_a",
            "Element_Symbol: H",
            "Local_Component: s",
            "Initial_Spin_Polarization: 0.8",
            "begin Atom_Coord",
            "   2.0 0.0 0.0",
            "  -2.0 0.0 0.0",
            "end Atom_Coord",
            "Atom_Type: H_b",
            "Element_Symbol: H",
            "Local_Component: s",
            "Initial_Spin_Polarization: -0.8",
            "begin Atom_Coord",
            "   0.0 2.0 0.0",
            "   0.0 -2.0 0.0",
            "end Atom_Coord",
        ]
        path = self.directory / "parsec.in"
        path.write_text("\n".join(lines) + "\n")
        _, both, both_forces = _run(path, "off", backend="auto", forces=True)
        _, flip, flip_forces = _run(path, "auto", backend="auto", forces=True)
        self.assertIn("the up channel is solved", dict(flip.backend.details)["spin_flip_symmetry"])
        self.assert_same(both, flip)
        np.testing.assert_allclose(flip_forces, both_forces, atol=1e-6)
        # the down channel's sector labels are the images of the up channel's, as both channels solved found them
        self.assertFalse(np.array_equal(flip.spin[1].representations, flip.spin[0].representations))
        for index in range(2):
            np.testing.assert_array_equal(
                np.sort(flip.spin[index].representations), np.sort(both.spin[index].representations)
            )

    def test_nonlocal_forces_of_a_stretched_sodium_dimer(self) -> None:
        """Na2 stretched to 8 bohr, a broken-symmetry singlet (p and d projectors): the nonlocal forces of the
        down channel taken from the up channel, in the sectors and on the full grid."""

        potential = DATA / "Na_POTRE.DAT"
        for label in ("Na_a", "Na_b"):
            (self.directory / f"{label}_POTRE.DAT").write_bytes(potential.read_bytes())

        def write(extra=()):
            lines = [
                "Boundary_Sphere_Radius: 12.0",
                "Grid_Spacing: 0.6",
                "States_Num: 6",
                "Max_Iter: 200",
                "Convergence_Criterion: 1.0e-8",
                "Eigensolver: chebff",
                "Correlation_Type: ca",
                "Spin_Polarization: .true.",
                *extra,
                "Atom_Types_Num: 2",
            ]
            for label, sign in (("Na_a", 1.0), ("Na_b", -1.0)):
                lines += [
                    f"Atom_Type: {label}",
                    "Element_Symbol: Na",
                    "Local_Component: s",
                    f"Initial_Spin_Polarization: {0.8 * sign}",
                    "begin Atom_Coord",
                    f"   0.0 0.0 {4.0 * sign}",
                    "end Atom_Coord",
                ]
            path = self.directory / "parsec.in"
            path.write_text("\n".join(lines) + "\n")
            return path

        for extra in ((), ("Spin_Symmetry_Sectors: off",)):
            with self.subTest(extra=extra):
                path = write(extra)
                system, both, both_forces = _run(path, "off", backend="auto", forces=True)
                self.assertGreater(system.nonlocal_operator.projectors.shape[1], 0)
                _, flip, flip_forces = _run(path, "auto", backend="auto", forces=True)
                self.assertIn("the up channel is solved", dict(flip.backend.details)["spin_flip_symmetry"])
                self.assertIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
                if extra:
                    # the full grid of both channels: the levels as the 1e-8 criterion leaves them (3.5e-7 seen)
                    self.assert_same(both, flip, energy=1e-7, levels=1e-6, density=1e-5)
                else:
                    self.assert_same(both, flip)
                # a weakly polarized broken-symmetry state at 8 bohr (0.15 electrons of |m| seen), not a closed shell
                moment = flip.spin[0].density - flip.spin[1].density
                self.assertGreater(float(np.sum(np.abs(moment)) * flip.grid.volume_element), 0.1)
                # opposite forces on the two atoms along the axis, as both channels solved give them
                np.testing.assert_allclose(flip_forces, both_forces, atol=1e-6)
                np.testing.assert_allclose(flip_forces[0], -flip_forces[1], atol=1e-8)

    def test_forces_of_a_stretched_sodium_dimer_in_a_cell(self) -> None:
        """(r7 merge of spinflip and spinforce) Na2 stretched to 8 bohr in a periodic cube at the Gamma point on the
        default routines: the forces of the cell in the sectors (PARSEC_SPIN_SECTOR_FORCES) with the down channel
        taken from the up channel against both channels solved."""

        potential = DATA / "Na_POTRE.DAT"
        for label in ("Na_a", "Na_b"):
            (self.directory / f"{label}_POTRE.DAT").write_bytes(potential.read_bytes())
        lines = [
            "Periodic_System: .true.",
            "Boundary_Conditions: bulk",
            "begin Cell_Shape",
            "  19.2 19.2 19.2",
            "end Cell_Shape",
            "Lattice_Vector_Scale: 1.0 bohr",
            "Coordinate_Unit: Cartesian_Bohr",
            "Use_Fortran_Routines: .false.",
            "Grid_Spacing: 0.6",
            "States_Num: 6",
            "Max_Iter: 200",
            "Convergence_Criterion: 1.0e-8",
            "Eigensolver: chebff",
            "Correlation_Type: ca",
            "Spin_Polarization: .true.",
            "Atom_Types_Num: 2",
        ]
        for label, sign in (("Na_a", 1.0), ("Na_b", -1.0)):
            lines += [
                f"Atom_Type: {label}",
                "Element_Symbol: Na",
                "Local_Component: s",
                f"Initial_Spin_Polarization: {0.8 * sign}",
                "begin Atom_Coord",
                f"   0.0 0.0 {4.0 * sign}",
                "end Atom_Coord",
            ]
        path = self.directory / "parsec.in"
        path.write_text("\n".join(lines) + "\n")
        system, both, both_forces = _run(path, "off", backend="auto", forces=True)
        self.assertIsNotNone(system.grid.cell)
        self.assertGreater(system.nonlocal_operator.projectors.shape[1], 0)
        _, flip, flip_forces = _run(path, "auto", backend="auto", forces=True)
        details = dict(flip.backend.details)
        self.assertIn("the up channel is solved", details["spin_flip_symmetry"])
        self.assertIn("sectors", details["spin_symmetry_sectors"])
        self.assertIsInstance(flip.spin[1].eigensolver_state, SpinFlipChannelState)
        self.assert_same(both, flip, energy=1e-7, levels=1e-6, density=1e-5)
        np.testing.assert_allclose(flip_forces, both_forces, atol=1e-6)
        np.testing.assert_allclose(flip_forces[0], -flip_forces[1], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
