"""Projector orbits of LDA+U in the symmetry sectors (``PARSEC_LDAU_PROJECTOR_ORBITS``, ``Hamiltonian/hubbard_device.py``).

The sectors of a reduction that knows the orbits of its atoms carry the KB and LDA+U columns of one atom per orbit,
scaled by the square root of the orbit size (``compress_projectors_for_sectors``); the coefficients of the
representatives are written there, and the occupation matrices of the other atoms are those of the representatives
turned with the d harmonics of an operation of the sectors (``HubbardOrbits``).

Checked on the host, for a D2h cluster (abelian orbits attached) and the fcc Ag cube (the commuting involutions with
the face-centring translations, whose reduction knows its orbits): the columns that the sector operators keep; that
random states of every sector give the occupation matrices of every atom from those of the representatives; that the
compressed sector operators with the coefficients of the representatives are the sector operators of every atom with
the coefficients of every atom; the switch values.  On a GPU: complete LDA+U runs of a cluster and of the fcc cube in
the sectors under ``auto`` and ``off`` and on the full grid.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.Hamiltonian.hubbard import build_hubbard_projectors
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import build_nonlocal_projectors
from parsec_python.models import Atom, GridSettings, PeriodicCell, PeriodicGridSettings, SpeciesPotential

from ..Hamiltonian.hubbard_device import (
    EXPANSION_COLUMNS,
    DeviceHubbard,
    HubbardOrbits,
    augmented_nonlocal_operator,
    ldau_projector_orbits_setting,
    sector_group_average,
    sector_hubbard_orbits,
)
from ..Symmetry.axis_reflection import (
    SignedPermutationReduction,
    attach_abelian_atom_orbits,
    compress_projectors_for_sectors,
    periodic_abelian_reduction,
    sector_orbits_exact,
    sector_orbits_exact_setting,
)
from ..Symmetry.representations import ReflectionRepresentationDecomposition
from ..backends.cupy import cupy_available
from ..backends.native import native_available

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

# Ag in D2h: one atom at the origin, a rectangle in the xy plane and two atoms on the z axis.
D2H_AG = (
    Atom("Ag", (0.0, 0.0, 0.0)),
    Atom("Ag", (3.0, 2.6, 0.0)),
    Atom("Ag", (-3.0, 2.6, 0.0)),
    Atom("Ag", (3.0, -2.6, 0.0)),
    Atom("Ag", (-3.0, -2.6, 0.0)),
    Atom("Ag", (0.0, 0.0, 3.6)),
    Atom("Ag", (0.0, 0.0, -3.6)),
)
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
_AG_SIDE = 7.729


class _Case:
    """A grid, atoms, a reduction with atom orbits, the KB factor and the LDA+U projectors (U = 4 eV)."""

    def __init__(self, grid, atoms, reduction, lattice=None) -> None:
        path = DATA / "Ag_POTRE.DAT"
        potentials = {"Ag": read_parsec_pseudopotential(path)}
        species = {"Ag": SpeciesPotential(path, 0, hubbard_u=4.0 / 13.605693122994)}
        self.grid = grid
        self.atoms = tuple(atoms)
        self.reduction = reduction
        self.kb = build_nonlocal_projectors(grid, self.atoms, potentials, species, lattice_vectors=lattice)
        self.hubbard = build_hubbard_projectors(grid, self.atoms, potentials, species, lattice_vectors=lattice)
        self.decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)


def _cluster() -> _Case:
    grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.5, expansion_order=6))
    reduction = SignedPermutationReduction.detect(grid, D2H_AG)
    with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "auto"}):
        assert attach_abelian_atom_orbits(reduction, D2H_AG, grid)
    return _Case(grid, D2H_AG, reduction)


def _cell() -> _Case:
    lattice = _AG_SIDE * np.eye(3)
    grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=_AG_SIDE / 16, expansion_order=6))
    atoms = tuple(Atom("Ag", tuple(position)) for position in _FCC * _AG_SIDE)
    abelian = SignedPermutationReduction.detect(grid, atoms)
    reduction = periodic_abelian_reduction(grid, atoms, abelian)
    assert reduction is not abelian and reduction.atom_representatives is not None
    return _Case(grid, atoms, reduction, lattice)


def _mirror_cell() -> _Case:
    """A cube of 10 bohr with a pair of Ag atoms that the swap of x and y carries onto each other (no translation
    does: a group of four, the swap and the mirror z -> -z) and an atom on the diagonal, its own orbit."""

    side = 10.0
    lattice = side * np.eye(3)
    grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=side / 16, expansion_order=6))
    atoms = (
        Atom("Ag", (2.5, 0.0, 0.0)),
        Atom("Ag", (0.0, 2.5, 0.0)),
        Atom("Ag", (3.125, 3.125, 5.0)),
    )
    abelian = SignedPermutationReduction.detect(grid, atoms)
    reduction = periodic_abelian_reduction(grid, atoms, abelian)
    if getattr(reduction, "atom_representatives", None) is None:
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "auto"}):
            assert attach_abelian_atom_orbits(reduction, atoms, grid)
    return _Case(grid, atoms, reduction, lattice)


def _random_states(case: _Case, generator):
    """Random states of every sector and positive occupations (any such density matrix commutes with the group)."""

    vectors, weights = [], []
    for sector in range(case.decomposition.representation_count):
        size = case.decomposition.sector_size(sector)
        count = min(3, size)
        vectors.append(generator.standard_normal((size, count)))
        weights.append(generator.uniform(0.2, 1.0, count))
    return vectors, weights


def _occupations(case: _Case, matrix: sp.spmatrix, vectors, weights, atoms: int) -> np.ndarray:
    """``sum_s sum_n f_n (B_s^T c_n)(B_s^T c_n)^T`` per atom from the projectors ``matrix`` reduced to each sector."""

    reduced = case.decomposition.reduce_nonlocal_operators(
        _Operator(matrix, tuple((index, 0, 0) for index in range(matrix.shape[1])))
    )
    total = np.zeros((atoms, 5, 5))
    for sector, (block, occupation) in enumerate(zip(vectors, weights)):
        overlaps = (reduced[sector].projectors.T @ block).reshape(atoms, 5, -1)
        total += np.einsum("amn,akn->amk", overlaps * occupation[None, None, :], overlaps)
    return total


class _Operator:
    def __init__(self, projectors, labels) -> None:
        self.projectors = sp.csc_matrix(projectors)
        self.signs = np.ones(self.projectors.shape[1])
        self.labels = labels


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("off", "off"), ("0", "off"), ("No", "off")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PARSEC_LDAU_PROJECTOR_ORBITS": value}):
                self.assertEqual(ldau_projector_orbits_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_LDAU_PROJECTOR_ORBITS": "on"}):
            with self.assertRaises(ValueError):
                ldau_projector_orbits_setting()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_LDAU_PROJECTOR_ORBITS", None)
            self.assertEqual(ldau_projector_orbits_setting(), "auto")


class SpinSwitchTests(unittest.TestCase):
    """``PARSEC_LDAU_SPIN_SECTORS``: neither under off (the default), a Gamma cell under auto, a cluster too under on."""

    def test_values_and_refusals(self) -> None:
        from types import SimpleNamespace

        from ..SCF.gamma_features import ldau_spin_sector_setting
        from ..driver import _spin_sector_refusal

        for value, expected in (("", "off"), ("auto", "auto"), ("ON", "on"), ("off", "off"), ("0", "off")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PARSEC_LDAU_SPIN_SECTORS": value}):
                self.assertEqual(ldau_spin_sector_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_LDAU_SPIN_SECTORS": "cells"}):
            with self.assertRaises(ValueError):
                ldau_spin_sector_setting()

        def problem(periodic):
            return SimpleNamespace(
                scf=SimpleNamespace(spin_symmetry_sector_mode="auto", symmetrize_density=False),
                pseudopotentials={"Ag": SimpleNamespace(has_hubbard=True)},
                periodic_cell=object() if periodic else None,
            )

        expected = {
            ("auto", True): None,
            ("auto", False): "LDA+U",
            ("on", True): None,
            ("on", False): None,
            ("off", True): "LDA+U",
            ("off", False): "LDA+U",
        }
        # without the switch: the routes of before for both (the cell on the 1x1x1 k-point driver, the cluster on
        # the full grid of the driver)
        with mock.patch.dict(os.environ):
            os.environ.pop("PARSEC_LDAU_SPIN_SECTORS", None)
            self.assertEqual(ldau_spin_sector_setting(), "off")
            for periodic in (True, False):
                self.assertIn("PARSEC_LDAU_SPIN_SECTORS=off", _spin_sector_refusal(problem(periodic)))
        for (value, periodic), reason in expected.items():
            with self.subTest(value=value, periodic=periodic), mock.patch.dict(
                os.environ, {"PARSEC_LDAU_SPIN_SECTORS": value}
            ):
                found = _spin_sector_refusal(problem(periodic))
                if reason is None:
                    self.assertIsNone(found)
                else:
                    self.assertIn(reason, found)


class OrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {"cluster": _cluster(), "cell": _cell(), "mirror cell": _mirror_cell()}

    def test_kept_columns_are_those_of_the_compressed_factor(self) -> None:
        for name, case in self.cases.items():
            with self.subTest(case=name):
                orbits = sector_hubbard_orbits(case.hubbard, case.kb.labels, case.reduction, case.atoms, case.grid)
                self.assertIsNotNone(orbits)
                augmented = augmented_nonlocal_operator(case.kb, case.hubbard)
                compressed = compress_projectors_for_sectors(augmented, case.reduction)
                kb_kept = [label for label in compressed.labels if label[1] >= 0]
                hubbard_kept = [label for label in compressed.labels if label[1] == -2]
                self.assertEqual(orbits.kb_columns, len(kb_kept))
                # the KB columns first, then the fifteen columns of every kept LDA+U atom in order
                self.assertTrue(all(label[1] >= 0 for label in compressed.labels[: orbits.kb_columns]))
                kept_atoms = [int(case.hubbard.atom_indices[k]) for k in orbits.kept]
                self.assertEqual(
                    hubbard_kept, [(atom, -2, column) for atom in kept_atoms for column in range(EXPANSION_COLUMNS)]
                )
                representatives = np.asarray(case.reduction.atom_representatives)
                np.testing.assert_array_equal(
                    np.asarray(case.hubbard.atom_indices)[orbits.sources],
                    representatives[np.asarray(case.hubbard.atom_indices)],
                )
                for rotation in orbits.rotations:
                    np.testing.assert_allclose(rotation @ rotation.T, np.eye(5), atol=1e-12)
                with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "off"}):
                    self.assertIsNone(
                        sector_hubbard_orbits(case.hubbard, case.kb.labels, case.reduction, case.atoms, case.grid)
                    )
                self.assertIsNone(sector_hubbard_orbits(case.hubbard, case.kb.labels, None, case.atoms, case.grid))

    def test_occupations_of_every_atom_from_the_representatives(self) -> None:
        generator = np.random.default_rng(20261121)
        for name, case in self.cases.items():
            with self.subTest(case=name):
                orbits = sector_hubbard_orbits(case.hubbard, case.kb.labels, case.reduction, case.atoms, case.grid)
                vectors, weights = _random_states(case, generator)
                every = _occupations(case, case.hubbard.projectors, vectors, weights, case.hubbard.atom_count)
                columns = (5 * orbits.kept[:, None] + np.arange(5)[None, :]).reshape(-1)
                kept = _occupations(
                    case, sp.csc_matrix(case.hubbard.projectors)[:, columns], vectors, weights, orbits.kept.size
                )
                scale = float(np.abs(every).max())
                self.assertGreater(scale, 1.0e-6)
                np.testing.assert_allclose(orbits.expand(kept), every, rtol=0, atol=1e-12 * scale)
                if name != "cell":
                    # atoms of one orbit differ (a mirror or a swap of axes, not only the face-centring translations)
                    self.assertGreater(float(np.abs(every - every[orbits.sources]).max()), 1.0e-6 * scale)

    def test_compressed_sector_operators_with_the_representatives_coefficients(self) -> None:
        generator = np.random.default_rng(1121)
        for name, case in self.cases.items():
            with self.subTest(case=name):
                orbits = sector_hubbard_orbits(case.hubbard, case.kb.labels, case.reduction, case.atoms, case.grid)
                # potential matrices of an invariant state: those of the representatives, turned onto their orbits
                representatives = generator.standard_normal((orbits.kept.size, 5, 5))
                representatives = representatives + np.swapaxes(representatives, 1, 2)
                potentials = orbits.expand(representatives)
                device = DeviceHubbard(case.hubbard, len(case.kb.signs), None, None, orbits)
                layouts = device.coefficient_layouts(potentials)
                augmented = augmented_nonlocal_operator(case.kb, case.hubbard)
                compressed = compress_projectors_for_sectors(augmented, case.reduction)
                full_signs = np.array(augmented.signs, dtype=np.float64)
                first, values = layouts[full_signs.size]
                full_signs[first : first + values.size] = values
                kept_signs = np.array(compressed.signs, dtype=np.float64)
                first, values = layouts[kept_signs.size]
                kept_signs[first : first + values.size] = values
                self.assertLess(kept_signs.size, full_signs.size)
                whole = case.decomposition.reduce_nonlocal_operators(augmented)
                part = case.decomposition.reduce_nonlocal_operators(compressed)
                for sector in range(case.decomposition.representation_count):
                    left = whole[sector].projectors
                    right = part[sector].projectors
                    every = (left @ sp.diags(full_signs) @ left.T).toarray()
                    kept = (right @ sp.diags(kept_signs) @ right.T).toarray()
                    scale = max(1.0, float(np.abs(every).max()))
                    np.testing.assert_allclose(kept, every, rtol=0, atol=1e-11 * scale, err_msg=f"sector {sector}")
                # zero coefficients for None (the first iteration)
                for first, values in device.coefficient_layouts(None).values():
                    self.assertFalse(np.any(values))
                # without orbits: the layout of every atom only
                plain = DeviceHubbard(case.hubbard, len(case.kb.signs), None, None, None)
                self.assertEqual(set(plain.coefficient_layouts(potentials)), {full_signs.size})

    def test_expand_checks_its_shape(self) -> None:
        orbits = HubbardOrbits(
            kept=np.array([0]), sources=np.array([0, 0]), rotations=np.stack([np.eye(5), -np.eye(5)]), kb_columns=0
        )
        matrix = np.arange(25.0).reshape(1, 5, 5)
        np.testing.assert_allclose(orbits.expand(matrix), np.concatenate([matrix, matrix]))
        with self.assertRaises(ValueError):
            orbits.expand(np.zeros((2, 5, 5)))


# Ag6 in D4h: a square in the xy plane and two apices on the z axis (weighted sectors of a subgroup of D4h).
D4H_AG = (
    Atom("Ag", (3.3, 0.0, 0.0)),
    Atom("Ag", (-3.3, 0.0, 0.0)),
    Atom("Ag", (0.0, 3.3, 0.0)),
    Atom("Ag", (0.0, -3.3, 0.0)),
    Atom("Ag", (0.0, 0.0, 2.8)),
    Atom("Ag", (0.0, 0.0, -2.8)),
)


class GroupAverageTests(unittest.TestCase):
    """The average over the full group of weighted sectors (``HubbardGroupAverage``)."""

    @classmethod
    def setUpClass(cls) -> None:
        from ..Symmetry.axis_reflection import PointGroupReduction, detect_point_group_reduction

        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        abelian = SignedPermutationReduction.detect(grid, D4H_AG)
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.75"}):
            weighted = detect_point_group_reduction(grid, D4H_AG, abelian)
        assert isinstance(weighted, PointGroupReduction)
        cls.case = _Case(grid, D4H_AG, weighted)
        cls.abelian = abelian

    def test_abelian_groups_take_no_average(self) -> None:
        case = self.case
        self.assertIsNone(sector_group_average(case.hubbard, self.abelian, case.atoms, case.grid))
        self.assertIsNone(sector_group_average(case.hubbard, None, case.atoms, case.grid))

    def test_the_average_is_covariant_and_idempotent(self) -> None:
        case = self.case
        average = sector_group_average(case.hubbard, case.reduction, case.atoms, case.grid)
        self.assertEqual(len(average.operations), 16)
        generator = np.random.default_rng(7)
        matrices = generator.standard_normal((case.hubbard.atom_count, 5, 5))
        matrices = matrices + np.swapaxes(matrices, 1, 2)
        averaged = average.occupations(matrices)
        np.testing.assert_allclose(average.occupations(averaged), averaged, atol=1e-13)
        for harmonics, image in average.operations:
            np.testing.assert_allclose(
                averaged[image], np.einsum("mk,akl,nl->amn", harmonics, averaged, harmonics), atol=1e-12
            )
        forces = generator.standard_normal((case.hubbard.atom_count, 3))
        symmetric = average.forces(forces)
        np.testing.assert_allclose(average.forces(symmetric), symmetric, atol=1e-13)
        for (_harmonics, image), rotation in zip(average.operations, average.rotations):
            np.testing.assert_allclose(symmetric[image], symmetric @ rotation.T, atol=1e-12)
        # the apices are fixed by the four-fold axis: their forces lie along z
        np.testing.assert_allclose(symmetric[4:, :2], 0.0, atol=1e-13)

    def test_the_loose_atom_tolerance_of_the_detection(self) -> None:
        """An Oh Ag13 cuboctahedron and the fcc Ag cube with one atom 5e-4 bohr off, their groups detected within
        ``PARSEC_SYMMETRY_ATOM_TOLERANCE=1e-3``: the orbits and the group average match the images at that tolerance
        and equal those of the exact structure (before, both raised at 1e-4 bohr)."""

        from types import SimpleNamespace

        from ..Hamiltonian.hubbard_device import ldau_orbits_exact
        from ..Symmetry.axis_reflection import detect_point_group_reduction
        from ..driver import _sectors_keep_orbits

        side = _AG_SIDE
        cube = [(0.0, 0.0, 0.0)]
        cube += [(sx * side / 2, sy * side / 2, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
        cube += [(sx * side / 2, 0.0, sz * side / 2) for sx in (-1, 1) for sz in (-1, 1)]
        cube += [(0.0, sy * side / 2, sz * side / 2) for sy in (-1, 1) for sz in (-1, 1)]
        cluster_grid = build_cluster_grid(GridSettings(spacing=0.6, radius=11.0, expansion_order=6))
        lattice = side * np.eye(3)
        cell_grid = build_periodic_grid(
            PeriodicCell(lattice), PeriodicGridSettings(spacing=side / 16, expansion_order=6)
        )
        fcc = [tuple(position) for position in _FCC * side]

        def case(positions, grid, tolerance, periodic):
            atoms = tuple(Atom("Ag", tuple(position)) for position in positions)
            abelian = SignedPermutationReduction.detect(grid, atoms, atom_tolerance=tolerance)
            if periodic:
                reduction = periodic_abelian_reduction(grid, atoms, abelian, atom_tolerance=tolerance)
            else:
                reduction = detect_point_group_reduction(grid, atoms, abelian, atom_tolerance=tolerance)
            self.assertIsNotNone(getattr(reduction, "atom_representatives", None))
            return _Case(grid, atoms, reduction, lattice if periodic else None)

        for name, positions, grid, periodic in (
            ("cluster", cube, cluster_grid, False),
            ("cell", fcc, cell_grid, True),
        ):
            with self.subTest(case=name):
                moved = np.array(positions, dtype=np.float64)
                moved[1, 0] += 5.0e-4
                exact = case(positions, grid, 1.0e-7, periodic)
                loose = case(moved, grid, 1.0e-3, periodic)
                np.testing.assert_array_equal(loose.reduction.atom_representatives, exact.reduction.atom_representatives)
                with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-3"}):
                    orbits = sector_hubbard_orbits(
                        loose.hubbard, loose.kb.labels, loose.reduction, loose.atoms, loose.grid
                    )
                    average = sector_group_average(loose.hubbard, loose.reduction, loose.atoms, loose.grid)
                reference = sector_hubbard_orbits(exact.hubbard, exact.kb.labels, exact.reduction, exact.atoms, exact.grid)
                self.assertIsNotNone(orbits)
                np.testing.assert_array_equal(orbits.kept, reference.kept)
                np.testing.assert_array_equal(orbits.sources, reference.sources)
                np.testing.assert_allclose(orbits.rotations, reference.rotations, atol=1e-12)
                self.assertEqual(orbits.kb_columns, reference.kb_columns)
                expected = sector_group_average(exact.hubbard, exact.reduction, exact.atoms, exact.grid)
                if expected is None:
                    self.assertIsNone(average)
                else:
                    self.assertEqual(len(average.operations), len(expected.operations))
                    for (turned, image), (turned_exact, image_exact) in zip(average.operations, expected.operations):
                        np.testing.assert_array_equal(image, image_exact)
                        np.testing.assert_allclose(turned, turned_exact, atol=1e-12)
                # at the default tolerance the off atom matches no image: a clear error, not a wrong orbit
                with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": ""}):
                    with self.assertRaisesRegex(RuntimeError, "does not map the LDA\\+U atoms"):
                        sector_hubbard_orbits(loose.hubbard, loose.kb.labels, loose.reduction, loose.atoms, loose.grid)
                # the driver keeps the orbits of LDA+U only where the atoms keep the group within 1e-6 bohr
                with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-3"}):
                    self.assertFalse(ldau_orbits_exact(loose.reduction, loose.atoms, loose.grid))
                    self.assertTrue(ldau_orbits_exact(exact.reduction, exact.atoms, exact.grid))
                    self.assertTrue(ldau_orbits_exact(None, loose.atoms, loose.grid))
                    with_u = SimpleNamespace(hubbard=loose.hubbard, grid=loose.grid)
                    without_u = SimpleNamespace(hubbard=None, grid=loose.grid)
                    self.assertFalse(_sectors_keep_orbits(with_u, "auto", loose.reduction, loose.atoms))
                    # the KB columns of a run without LDA+U follow the same distance (PARSEC_SECTOR_ORBITS_EXACT)
                    self.assertFalse(_sectors_keep_orbits(without_u, "auto", loose.reduction, loose.atoms))
                    with mock.patch.dict(os.environ, {"PARSEC_SECTOR_ORBITS_EXACT": "off"}):
                        self.assertTrue(_sectors_keep_orbits(without_u, "auto", loose.reduction, loose.atoms))
                        # (LDA+U has its own switch)
                        self.assertFalse(_sectors_keep_orbits(with_u, "auto", loose.reduction, loose.atoms))
                    exact_u = SimpleNamespace(hubbard=exact.hubbard, grid=exact.grid)
                    exact_plain = SimpleNamespace(hubbard=None, grid=exact.grid)
                    self.assertTrue(_sectors_keep_orbits(exact_u, "auto", exact.reduction, exact.atoms))
                    self.assertTrue(_sectors_keep_orbits(exact_plain, "auto", exact.reduction, exact.atoms))
                    self.assertFalse(_sectors_keep_orbits(exact_u, "off", exact.reduction, exact.atoms))
                # at a tolerance of 1e-6 bohr or less nothing is checked (every operation found maps the atoms)
                with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-6"}):
                    self.assertTrue(ldau_orbits_exact(loose.reduction, loose.atoms, loose.grid))
                    self.assertTrue(sector_orbits_exact(loose.reduction, loose.atoms, loose.grid))

    def test_the_switch_of_the_exact_orbits(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PARSEC_SECTOR_ORBITS_EXACT": value}):
                self.assertEqual(sector_orbits_exact_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_ORBITS_EXACT": "on"}):
            with self.assertRaises(ValueError):
                sector_orbits_exact_setting()

    def test_the_weighted_switch(self) -> None:
        from ..Hamiltonian.hubbard_device import ldau_weighted_sectors_setting

        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off"), ("false", "off")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PARSEC_LDAU_WEIGHTED_SECTORS": value}):
                self.assertEqual(ldau_weighted_sectors_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_LDAU_WEIGHTED_SECTORS": "on"}):
            with self.assertRaises(ValueError):
                ldau_weighted_sectors_setting()


_CELL = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side:.6f} {side:.6f} {side:.6f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing:.10f}
Expansion_Order: 8
States_Num: 34
Fermi_Temp: 1000
Max_Iter: 150
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-10
Eigensolver: chebff
Output_Level: 2
Correlation_Type: ca
{extra}Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
ldaplusu_u 4.0 eV
begin Atom_Coord
{coordinates}end Atom_Coord
"""

_CLUSTER = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 9.0
Hartree_Boundary_Tolerance: off
Grid_Spacing: 0.6
Coordinate_Unit: Cartesian_Bohr
States_Num: 50
Fermi_Temp: 1000.0
Correlation_Type: ca
Convergence_Criterion: 1.0e-8
Max_Iter: 80
Eigensolver: chebff
Mixing_Param: 0.2
{extra}Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
ldaplusu_u 4.0 eV
begin Atom_Coord
{coordinates}end Atom_Coord
"""


def _coordinates(positions) -> str:
    return "".join(f"  {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in positions)


def _forces(path: Path) -> np.ndarray:
    """The last table of total forces (after the net force is removed) of ``parsec.out``."""

    text = path.read_text(errors="replace")
    marker = "coordinates and tot. forces (after setting net force to zero)"
    rows = []
    for line in text[text.rindex(marker):].splitlines()[1:]:
        if re.match(r"\s*\d+\s+-?\d", line):
            rows.append([float(value) for value in line.split()[4:7]])
        elif rows and line.strip():
            break
    return np.array(rows)


def _energy(path: Path) -> float:
    values = re.findall(r"Total Energy[^=\n]*=\s*(-?\d+\.\d+)", path.read_text(errors="replace"))
    return float(values[-1])


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class DeviceTests(unittest.TestCase):
    """``main.py`` on a GPU: the sectors with the orbits (auto) against every atom (off) and the full grid."""

    def run_case(self, text: str, settings: dict[str, dict[str, str]]):
        import parsec_python.acceleration.cli as cli_module

        folder = Path(tempfile.mkdtemp(prefix="parsec_ldau_orbits_gpu_"))
        self.addCleanup(shutil.rmtree, folder, True)
        logs = {}
        for name, environment in settings.items():
            path = folder / name / "parsec.in"
            path.parent.mkdir(parents=True)
            path.write_text(text)
            shutil.copyfile(DATA / "Ag_POTRE.DAT", path.parent / "Ag_POTRE.DAT")
            environment = dict(environment)
            arguments = [str(path), "--no-archive"]
            if environment.pop("symmetry", None) == "off":
                arguments += ["--symmetry", "off"]
            with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(
                StringIO()
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(cli_module.main(arguments), 0, name)
            logs[name] = path.parent / "parsec.out"
        return logs

    def test_cell_in_the_sectors(self) -> None:
        text = _CELL.format(
            side=_AG_SIDE, spacing=_AG_SIDE / 16, extra="", coordinates=_coordinates(_FCC * _AG_SIDE)
        )
        logs = self.run_case(
            text,
            {
                "full": {"PARSEC_GAMMA_LDAU_SECTORS": "off"},
                "every": {
                    "PARSEC_GAMMA_LDAU_SECTORS": "on",
                    "PARSEC_LDAU_PROJECTOR_ORBITS": "off",
                    "PARSEC_LDAU_WEIGHTED_SECTORS": "off",
                },
                "orbits": {
                    "PARSEC_GAMMA_LDAU_SECTORS": "on",
                    "PARSEC_LDAU_PROJECTOR_ORBITS": "auto",
                    "PARSEC_LDAU_WEIGHTED_SECTORS": "off",
                },
            },
        )
        for name in ("every", "orbits"):
            self.assertIn("exact commuting group of 32 operations", logs[name].read_text())
            self.assertAlmostEqual(_energy(logs[name]), _energy(logs["full"]), delta=2.0e-7)
        self.assertAlmostEqual(_energy(logs["orbits"]), _energy(logs["every"]), delta=1.0e-8)

    def test_cell_forces_in_the_sectors(self) -> None:
        # the cube with one atom moved along z (a smaller group) and forces: PARSEC_LDAU_SECTOR_FORCES=auto keeps
        # the sectors for them, off reads the full grid, PARSEC_GAMMA_LDAU_SECTORS=off solves it all there
        positions = _FCC * _AG_SIDE
        positions[1, 2] += 0.12
        text = _CELL.format(
            side=_AG_SIDE, spacing=_AG_SIDE / 16, extra="Skip_force: .false.\n", coordinates=_coordinates(positions)
        )
        logs = self.run_case(
            text,
            {
                "full": {"PARSEC_GAMMA_LDAU_SECTORS": "off"},
                "scf": {"PARSEC_GAMMA_LDAU_SECTORS": "on", "PARSEC_LDAU_SECTOR_FORCES": "off"},
                "sectors": {"PARSEC_GAMMA_LDAU_SECTORS": "on", "PARSEC_LDAU_SECTOR_FORCES": "auto"},
            },
        )
        expected = _forces(logs["full"])
        self.assertGreater(float(np.abs(expected).max()), 1.0e-3)
        for name in ("scf", "sectors"):
            self.assertAlmostEqual(_energy(logs[name]), _energy(logs["full"]), delta=2.0e-7)
            np.testing.assert_allclose(_forces(logs[name]), expected, atol=1.0e-5)
        self.assertIn("in the sectors of its point group", logs["sectors"].read_text())
        self.assertNotIn("in the sectors of its point group", logs["scf"].read_text())

    def test_weighted_sectors(self) -> None:
        # the fcc cube in the weighted sectors of its point group with the face-centring translations, and the
        # D4h cluster: the occupation matrices averaged over the full group (PARSEC_LDAU_WEIGHTED_SECTORS=auto)
        cell = _CELL.format(side=_AG_SIDE, spacing=_AG_SIDE / 16, extra="", coordinates=_coordinates(_FCC * _AG_SIDE))
        logs = self.run_case(
            cell,
            {
                "full": {"PARSEC_GAMMA_LDAU_SECTORS": "off"},
                "weighted": {"PARSEC_GAMMA_LDAU_SECTORS": "on", "PARSEC_LDAU_WEIGHTED_SECTORS": "auto"},
            },
        )
        text = logs["weighted"].read_text()
        self.assertNotIn("exact commuting group", text)
        self.assertRegex(text, r"exact group of \d+ operations")
        self.assertAlmostEqual(_energy(logs["weighted"]), _energy(logs["full"]), delta=2.0e-7)
        cluster = _CLUSTER.format(extra="", coordinates=_coordinates([atom.position for atom in D4H_AG]))
        logs = self.run_case(
            cluster,
            {
                "full": {"symmetry": "off"},
                "abelian": {"PARSEC_LDAU_WEIGHTED_SECTORS": "off"},
                "weighted": {"PARSEC_LDAU_WEIGHTED_SECTORS": "auto"},
            },
        )
        self.assertAlmostEqual(_energy(logs["weighted"]), _energy(logs["full"]), delta=2.0e-7)
        self.assertAlmostEqual(_energy(logs["abelian"]), _energy(logs["full"]), delta=2.0e-7)

    def test_cluster_in_the_sectors(self) -> None:
        text = _CLUSTER.format(extra="", coordinates=_coordinates([atom.position for atom in D2H_AG]))
        logs = self.run_case(
            text,
            {
                "full": {"PARSEC_LDAU_PROJECTOR_ORBITS": "off", "symmetry": "off"},
                "every": {"PARSEC_LDAU_PROJECTOR_ORBITS": "off"},
                "orbits": {"PARSEC_LDAU_PROJECTOR_ORBITS": "auto"},
            },
        )
        self.assertAlmostEqual(_energy(logs["orbits"]), _energy(logs["every"]), delta=1.0e-8)
        self.assertAlmostEqual(_energy(logs["orbits"]), _energy(logs["full"]), delta=2.0e-7)


if __name__ == "__main__":
    unittest.main()
