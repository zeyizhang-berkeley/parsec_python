"""Gamma-point sectors of wires, slabs and oblique cells (``PARSEC_SLAB_WIRE_SECTORS``, ``PARSEC_OBLIQUE_SECTORS``).

A wire or slab of the Fortran-derived routines (:mod:`parsec_python.Grid.confined`) whose periodic vectors lie along
the axes takes the signed permutations about the origin that map its atoms modulo the periodic lattice and its
cylinder or slab of points onto themselves (the mirror ``z -> -z`` of a slab among them).  A cell whose grid axes are
not the Cartesian axes (a hexagonal or primitive cubic cell, a hexagonal slab) is detected in the frame of its grid
indices (:mod:`parsec_python.acceleration.Symmetry.oblique_frame`): the signed permutations of the grid axes that are
isometries of the cell and keep its finite-difference operators.

The model tests (CPU) compare the weighted spectra of the sectors with the full grid; the SCF tests (A100) the
energies and levels of the sectors with the full grid of the same input (both switches off).
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.confined import build_confined_periodic_grid, confined_lattice_vectors
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    _lattice_period,
    detect_point_group_reduction,
    slab_wire_sectors_setting,
)
from parsec_python.acceleration.Symmetry.oblique_frame import (
    admissible_index_operations,
    index_frame,
    needs_index_frame,
    oblique_sectors_setting,
    sector_negative_laplacian,
    symmetric_stencil_lines,
)
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_SLAB_WIRE = "PARSEC_SLAB_WIRE_SECTORS"
_OBLIQUE = "PARSEC_OBLIQUE_SECTORS"
_ON = {_SLAB_WIRE: "auto", _OBLIQUE: "auto"}


def _confined(periodicity, vectors, *, radius=4.0, spacing=0.5, order=4):
    cell = PeriodicCell(confined_lattice_vectors(vectors))
    settings = PeriodicGridSettings(
        spacing=spacing, expansion_order=order, shift=(0.5, 0.5, 0.5), periodicity=periodicity,
        confinement_radius=radius, fortran_routines=True,
    )
    return build_confined_periodic_grid(cell, settings)


def _hydrogen_problem(periodicity, atoms, *, radius=4.0, spacing=0.5):
    vectors = [[3.0, 0.0, 0.0]] if periodicity == 1 else [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]]
    return SinglePointInput(
        atoms=atoms,
        pseudopotentials={"H": SpeciesPotential(path=DATA / "H_POTRE.DAT", local_angular_momentum=0)},
        grid=PeriodicGridSettings(
            spacing=spacing, expansion_order=8, shift=(0.5, 0.5, 0.5), periodicity=periodicity,
            confinement_radius=radius, fortran_routines=True,
        ),
        periodic_cell=PeriodicCell(confined_lattice_vectors(vectors)),
        scf=SCFSettings(number_of_states=2, convergence_criterion=1e-9, max_iterations=80),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1e-9),
        hartree=HartreeSettings(relative_tolerance=1e-12),
    )


def _images(grid):
    """Cartesian lattice translations around the cell (periodic directions only)."""

    periodicity = int(getattr(grid, "periodicity", 3))
    vectors = np.asarray(grid.cell.lattice_vectors, dtype=float)[:periodicity]
    ranges = [range(-3, 4)] * periodicity
    mesh = np.meshgrid(*ranges, indexing="ij")
    integers = np.column_stack([axis.reshape(-1) for axis in mesh]).astype(float)
    return integers @ vectors


def _wells(grid, atoms, depth=2.0, width=0.9):
    coordinates = np.asarray(grid.coordinates, dtype=float)
    potential = np.zeros(grid.size)
    for atom in atoms:
        for image in _images(grid):
            difference = coordinates - (np.asarray(atom.position) + image)
            potential -= depth * np.exp(-np.sum(difference * difference, axis=1) / width)
    return potential


def _weighted_spectrum(decomposition, hamiltonian, count):
    weighted = []
    for index, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
        if matrix is None or matrix.shape[0] == 0:
            continue
        values = np.linalg.eigvalsh(matrix.toarray())[:count]
        weights = getattr(decomposition, "sector_weights", None)
        weight = 1 if weights is None else int(weights[index])
        weighted += [(value, weight) for value in values]
    return np.array([value for value, weight in sorted(weighted) for _ in range(weight)][:count])


def _check_spectrum(test, grid, frame_grid, reduction, laplacian, atoms, count=30):
    hamiltonian = (0.5 * laplacian + sp.diags(_wells(grid, atoms))).tocsr()
    decomposition = ReflectionRepresentationDecomposition.build(frame_grid, reduction)
    np.testing.assert_allclose(
        _weighted_spectrum(decomposition, hamiltonian, count),
        np.linalg.eigvalsh(hamiltonian.toarray())[:count],
        rtol=0.0,
        atol=1.0e-10,
    )


def _detect(grid, atoms, tolerance=1.0e-7):
    abelian = SignedPermutationReduction.detect(grid, atoms, atom_tolerance=tolerance)
    return detect_point_group_reduction(grid, atoms, abelian, atom_tolerance=tolerance)


def _operations(reduction):
    full = getattr(reduction, "full_operations", None)
    if full is not None:
        return np.asarray(full)
    operations = getattr(reduction, "operations", None)
    if operations is not None:
        return np.asarray(operations)
    return np.asarray([np.diag(signs) for signs in reduction.signs])


class SwitchTests(unittest.TestCase):
    def test_the_switches(self) -> None:
        for name, setting in ((_SLAB_WIRE, slab_wire_sectors_setting), (_OBLIQUE, oblique_sectors_setting)):
            with self.subTest(name=name):
                for value, expected in (("", "off"), ("auto", "auto"), (" OFF ", "off"), ("Auto", "auto")):
                    with mock.patch.dict(os.environ, {name: value}):
                        self.assertEqual(setting(), expected)
                with mock.patch.dict(os.environ, {name: "on"}), self.assertRaises(ValueError):
                    setting()

    def test_an_unknown_value_stops_the_preparation(self) -> None:
        from parsec_python.acceleration import driver

        problem = _hydrogen_problem(2, [Atom("H", (0.0, 0.0, 0.7)), Atom("H", (0.0, 0.0, -0.7))])
        for name in (_SLAB_WIRE, _OBLIQUE):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: "yes"}):
                with self.assertRaisesRegex(ValueError, name):
                    driver.prepare_single_point(problem, backend="scipy")


class WireSlabDetectionTests(unittest.TestCase):
    def test_the_period_of_a_slab(self) -> None:
        grid = _confined(2, [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]])
        with mock.patch.dict(os.environ, {_SLAB_WIRE: "off"}), self.assertRaisesRegex(ValueError, _SLAB_WIRE):
            _lattice_period(grid)
        with mock.patch.dict(os.environ, _ON):
            period = _lattice_period(grid)
        np.testing.assert_allclose(period[:2], [3.0, 3.2])
        # no image wraps along the confined axis: far beyond the slab
        self.assertGreater(period[2], 20.0 * grid.confinement_radius)

    def test_a_slab_keeps_its_mirror(self) -> None:
        atoms = [Atom("H", (0.0, 0.0, 0.7)), Atom("H", (0.0, 0.0, -0.7))]
        grid = _confined(2, [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]])
        with mock.patch.dict(os.environ, _ON):
            reduction = _detect(grid, atoms)
            operations = _operations(reduction)
            self.assertEqual(int(reduction.group_order), 8)
            self.assertTrue(any(np.array_equal(op, np.diag([1, 1, -1])) for op in operations))
            _check_spectrum(self, grid, grid, reduction, build_negative_laplacian(grid), atoms)

    def test_a_tilted_molecule_and_atoms_on_the_cell_edge(self) -> None:
        grid = _confined(2, [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]])
        for atoms, order in (
            ([Atom("H", (0.3, 0.0, 0.65)), Atom("H", (-0.3, 0.0, -0.65))], 4),
            # x = 1.5 = a/2: the mirror of x maps the atom onto its image
            ([Atom("H", (1.5, 0.0, 0.7)), Atom("H", (1.5, 0.0, -0.7))], 8),
        ):
            with self.subTest(atoms=atoms), mock.patch.dict(os.environ, _ON):
                reduction = _detect(grid, atoms)
                self.assertEqual(int(reduction.group_order), order)
                _check_spectrum(self, grid, grid, reduction, build_negative_laplacian(grid), atoms)

    def test_a_wire_takes_its_point_group(self) -> None:
        atoms = [Atom("H", (0.0, 0.8, 0.0)), Atom("H", (0.0, -0.8, 0.0)), Atom("H", (0.0, 0.0, 0.8)),
                 Atom("H", (0.0, 0.0, -0.8))]
        grid = _confined(1, [[3.0, 0.0, 0.0]], radius=3.5)
        with mock.patch.dict(os.environ, _ON):
            reduction = _detect(grid, atoms)
            self.assertIsInstance(reduction, PointGroupReduction)
            operations = _operations(reduction)
            # D4h about the axis: the exchange of y and z, never of x with a confined axis
            self.assertEqual(operations.shape[0], 16)
            self.assertTrue(np.all(np.abs(operations[:, 0, 0]) == 1))
            self.assertTrue(any(abs(op[1, 2]) == 1 for op in operations))
            _check_spectrum(self, grid, grid, reduction, build_negative_laplacian(grid), atoms)

    def test_a_slab_takes_translations_along_its_periodic_axes(self) -> None:
        # h-BN in its centred rectangular cell (four atoms): the centring translation (a/2, b/2, 0) joins the group
        a, b = 4.73, 4.73 * 3**0.5
        atoms = [Atom("B", (0.0, 0.0, 0.0)), Atom("B", (a / 2, b / 2, 0.0)), Atom("N", (0.0, b / 3, 0.0)),
                 Atom("N", (a / 2, 5 * b / 6, 0.0))]
        grid = _confined(2, [[a, 0.0, 0.0], [0.0, b, 0.0]], radius=2.5, spacing=0.6)
        from parsec_python.acceleration.Symmetry.axis_reflection import periodic_abelian_reduction

        with mock.patch.dict(os.environ, _ON):
            abelian = SignedPermutationReduction.detect(grid, atoms)
            reduction = detect_point_group_reduction(grid, atoms, abelian)
            self.assertIsNotNone(getattr(reduction, "full_translations", None))
            translations = np.asarray(reduction.full_translations)
            # translations along the periodic axes only
            self.assertTrue(np.all(translations[:, 2] == 0))
            self.assertTrue(np.any(translations[:, :2] != 0))
            self.assertEqual(int(reduction.group_order), 2 * int(abelian.group_order))
            _check_spectrum(self, grid, grid, reduction, build_negative_laplacian(grid), atoms)
            commuting = periodic_abelian_reduction(grid, atoms, abelian)
            self.assertEqual(int(commuting.group_order), 2 * int(abelian.group_order))
            _check_spectrum(self, grid, grid, commuting, build_negative_laplacian(grid), atoms)
        with mock.patch.dict(os.environ, {**_ON, "PARSEC_PERIODIC_TRANSLATIONS": "off"}):
            self.assertIsNone(getattr(_detect(grid, atoms), "full_translations", None))

    def test_a_periodic_axis_goes_onto_a_periodic_axis_of_its_step(self) -> None:
        # a square slab of equal steps: x and y exchange, z stays
        atoms = [Atom("H", (0.0, 0.0, 0.7)), Atom("H", (0.0, 0.0, -0.7))]
        grid = _confined(2, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], radius=3.0)
        with mock.patch.dict(os.environ, _ON):
            operations = _operations(_detect(grid, atoms))
        self.assertTrue(np.all(np.abs(operations[:, 2, 2]) == 1))
        self.assertTrue(any(abs(op[0, 1]) == 1 for op in operations))

    def test_the_route(self) -> None:
        from parsec_python.acceleration.driver import periodic_sector_refusal

        problem = _hydrogen_problem(2, [Atom("H", (0.0, 0.0, 0.7)), Atom("H", (0.0, 0.0, -0.7))])
        with mock.patch.dict(os.environ, {_SLAB_WIRE: "off"}):
            self.assertIn("wire or slab", periodic_sector_refusal(problem, features=True))
        with mock.patch.dict(os.environ, _ON):
            self.assertIsNone(periodic_sector_refusal(problem, features=True))
            # an MPI run or a folded mesh (features false) keeps the route of before
            self.assertIn("wire or slab", periodic_sector_refusal(problem, features=False))
            for scf, reason in (
                (replace(problem.scf, spin_polarized=True), "spin"),
                (replace(problem.scf, xc_functional="pbe0"), "hybrid"),
                (replace(problem.scf, symmetrize_density=True), "Symmetrize_Charge_Density"),
            ):
                with self.subTest(reason=reason):
                    try:
                        changed = replace(problem, scf=scf)
                    except (TypeError, ValueError):
                        continue
                    self.assertIn(reason, periodic_sector_refusal(changed, features=True))


def _hexagonal_cell(a=6.06, c=9.83, digits=10):
    vectors = np.array([[a, 0.0, 0.0], [-a / 2, a * 3**0.5 / 2, 0.0], [0.0, 0.0, c]])
    return np.round(vectors, digits)


class ObliqueFrameTests(unittest.TestCase):
    def test_which_grids_need_the_frame(self) -> None:
        cubic = build_periodic_grid(PeriodicCell(np.diag([6.0, 6.0, 6.0])), PeriodicGridSettings(spacing=0.75, expansion_order=4))
        self.assertFalse(needs_index_frame(cubic))
        hexagonal = build_periodic_grid(PeriodicCell(_hexagonal_cell()), PeriodicGridSettings(spacing=0.6, expansion_order=4))
        self.assertTrue(needs_index_frame(hexagonal))
        self.assertFalse(needs_index_frame(_confined(2, [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]])))
        self.assertTrue(needs_index_frame(_confined(2, [[3.0, 0.0, 0.0], [-1.5, 1.5 * 3**0.5, 0.0]])))
        # the slab grid of the default routines (zero beyond its planes) takes no frame
        from parsec_python.Grid.pbc import build_slab_grid

        andrew = build_slab_grid(
            PeriodicCell(np.array([[4.0, 0.0, 0.0], [-2.0, 2.0 * 3**0.5, 0.0], [0.0, 0.0, 8.0]]), dimensions=2),
            PeriodicGridSettings(spacing=0.5, expansion_order=4),
        )
        self.assertFalse(needs_index_frame(andrew))

    def test_the_admissible_operations(self) -> None:
        # hexagonal: E, C2z, the mirror through a1 + a2 and its product, each with z -> +-z (8 of the 24); the
        # primitive fcc cell 12 of 48
        hexagonal = build_periodic_grid(PeriodicCell(_hexagonal_cell()), PeriodicGridSettings(spacing=0.6, expansion_order=4))
        self.assertEqual(len(admissible_index_operations(hexagonal)), 8)
        a = 10.26
        fcc = build_periodic_grid(
            PeriodicCell(np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])),
            PeriodicGridSettings(spacing=0.75, expansion_order=4),
        )
        operations = admissible_index_operations(fcc)
        self.assertEqual(len(operations), 12)
        frame = index_frame(fcc)
        for operation in operations:
            rotation = frame.cartesian(operation)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)

    def test_a_typed_hexagonal_cell_takes_the_mean_weights(self) -> None:
        grid = build_periodic_grid(PeriodicCell(_hexagonal_cell(digits=10)), PeriodicGridSettings(spacing=0.6, expansion_order=4))
        frame = index_frame(grid)
        lines = symmetric_stencil_lines(grid, frame.operations)
        self.assertIsNotNone(lines)
        laplacian = build_negative_laplacian(grid)
        symmetric = sector_negative_laplacian(grid, frame.operations)
        change = abs(symmetric - laplacian).max()
        self.assertGreater(change, 0.0)
        self.assertLess(change, 1.0e-9 * abs(laplacian).max())
        from parsec_python.acceleration.Symmetry.axis_reflection import _grid_row_mapping_operation

        for operation in frame.operations:
            mapping = _grid_row_mapping_operation(frame.frame_grid, operation, 1.0e-10)
            self.assertIsNotNone(mapping)
            self.assertEqual(abs(symmetric[mapping][:, mapping] - symmetric).max(), 0.0)
        # an exact primitive fcc cell: its equivalent weights differ by round-off at most
        a = 10.0
        fcc = build_periodic_grid(
            PeriodicCell(np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])),
            PeriodicGridSettings(spacing=0.75, expansion_order=4),
        )
        symmetric = sector_negative_laplacian(fcc, index_frame(fcc).operations)
        if symmetric is not None:
            own = build_negative_laplacian(fcc)
            self.assertLess(abs(symmetric - own).max(), 1.0e-14 * abs(own).max())

    def test_spectra_in_the_frame(self) -> None:
        a = 10.26
        cases = []
        fcc_vectors = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
        cases.append((
            "fcc", build_periodic_grid(PeriodicCell(fcc_vectors), PeriodicGridSettings(spacing=0.75, expansion_order=4)),
            [Atom("Si", tuple(np.array([1.0, 1.0, 1.0]) * a / 8)), Atom("Si", tuple(-np.array([1.0, 1.0, 1.0]) * a / 8))],
        ))
        hexagonal = _hexagonal_cell()
        middle = (np.array([1 / 3, 2 / 3, 1 / 4]) + np.array([2 / 3, 1 / 3, 3 / 4])) / 2
        frac = [np.array([1 / 3, 2 / 3, 1 / 4]) - middle, np.array([2 / 3, 1 / 3, 3 / 4]) - middle]
        cases.append((
            "hcp", build_periodic_grid(PeriodicCell(hexagonal), PeriodicGridSettings(spacing=0.7, expansion_order=4)),
            [Atom("Mg", tuple(f @ hexagonal)) for f in frac],
        ))
        b = 4.73
        slab_vectors = [[b, 0.0, 0.0], [-b / 2, round(b * 3**0.5 / 2, 10), 0.0]]
        cases.append((
            "hexagonal slab", _confined(2, slab_vectors, radius=3.0, spacing=0.5),
            [Atom("B", (0.0, 0.0, 0.0)), Atom("N", tuple(np.array([1 / 3, 2 / 3, 0.0]) @ np.array(slab_vectors + [[0, 0, 1.0]])))],
        ))
        for name, grid, atoms in cases:
            with self.subTest(name=name), mock.patch.dict(os.environ, _ON):
                self.assertTrue(needs_index_frame(grid))
                frame = index_frame(grid)
                reduction = _detect(frame.frame_grid, frame.atoms(atoms), frame.tolerance(1.0e-7))
                self.assertGreater(int(reduction.group_order), 1)
                laplacian = sector_negative_laplacian(grid, frame.operations)
                if laplacian is None:
                    laplacian = build_negative_laplacian(grid)
                _check_spectrum(self, grid, frame.frame_grid, reduction, laplacian, atoms, count=24)

    def test_the_route(self) -> None:
        from parsec_python.acceleration.driver import periodic_sector_refusal

        a = 10.26
        problem = SinglePointInput(
            atoms=[Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (a / 4, a / 4, a / 4))],
            pseudopotentials={"Si": SpeciesPotential(path=DATA / "Si_POTRE.DAT", local_angular_momentum=0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=8),
            periodic_cell=PeriodicCell(np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])),
            scf=SCFSettings(number_of_states=8),
        )
        with mock.patch.dict(os.environ, {_OBLIQUE: "off"}):
            self.assertIn("oblique", periodic_sector_refusal(problem, features=True))
        with mock.patch.dict(os.environ, _ON):
            self.assertIsNone(periodic_sector_refusal(problem, features=True))
            self.assertIn("oblique", periodic_sector_refusal(problem, features=False))
            self.assertIn("spin", periodic_sector_refusal(replace(problem, scf=replace(problem.scf, spin_polarized=True)), features=True))

    def test_the_force_operations_of_the_frame_are_cartesian(self) -> None:
        # The operations of sectors detected in the frame act on the Cartesian atoms as R = S W S^-1 with
        # tau = -S W^T shift: R (r_a - tau) is the image atom modulo the lattice, and the moved-atom check runs in the
        # frame (a structure on the group passes, an atom moved off it is refused).
        from types import SimpleNamespace

        from parsec_python.acceleration.Forces.sector_symmetry import (
            abelian_cell_operations,
            check_sector_atoms,
            sector_group_operations,
        )
        from parsec_python.acceleration.Symmetry.axis_reflection import periodic_abelian_reduction

        hexagonal = _hexagonal_cell()
        middle = (np.array([1 / 3, 2 / 3, 1 / 4]) + np.array([2 / 3, 1 / 3, 3 / 4])) / 2
        cell = PeriodicCell(hexagonal)
        frac = [np.array([1 / 3, 2 / 3, 1 / 4]) - middle, np.array([2 / 3, 1 / 3, 3 / 4]) - middle]
        base = [Atom("Mg", tuple(f @ hexagonal)) for f in frac]
        # a 2 x 2 x 1 supercell (translations by half the cell)
        vectors = hexagonal * np.array([2.0, 2.0, 1.0])[:, None]
        atoms = [
            Atom("Mg", tuple(np.asarray(atom.position) + i * hexagonal[0] + j * hexagonal[1]))
            for atom in base for i in range(2) for j in range(2)
        ]
        grid = build_periodic_grid(PeriodicCell(vectors), PeriodicGridSettings(spacing=0.7, expansion_order=4))
        with mock.patch.dict(os.environ, _ON):
            frame = index_frame(grid)
            frame_atoms = frame.atoms(atoms)
            tolerance = frame.tolerance(1.0e-7)
            point = _detect(frame.frame_grid, frame_atoms, tolerance)
            abelian = periodic_abelian_reduction(
                frame.frame_grid, frame_atoms, SignedPermutationReduction.detect(frame.frame_grid, frame_atoms,
                                                                                atom_tolerance=tolerance),
                atom_tolerance=tolerance,
            )
        positions = np.asarray([atom.position for atom in atoms])
        inverse = np.linalg.inv(vectors)

        def check(operations):
            self.assertGreater(len(operations), 1)
            for operation in operations:
                rotation = np.asarray(operation.matrix)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
                tau = np.zeros(3) if operation.translation is None else np.asarray(operation.translation)
                moved = (positions - tau) @ rotation.T
                difference = (moved - positions[np.asarray(operation.image)]) @ inverse
                np.testing.assert_allclose(difference - np.rint(difference), 0.0, atol=1.0e-9)

        self.assertIsNotNone(getattr(point, "full_translations", None))
        object.__setattr__(point, "index_frame", frame)
        check(sector_group_operations(point, atoms, point.grid, vectors))
        object.__setattr__(abelian, "index_frame", frame)
        system = SimpleNamespace(
            input=SimpleNamespace(periodic_cell=PeriodicCell(vectors)),
            backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=SimpleNamespace(reduction=abelian))),
            atoms=tuple(atoms),
            grid=grid,
        )
        check(abelian_cell_operations(system))
        check_sector_atoms(point, atoms, grid, vectors)
        moved = list(atoms)
        moved[0] = Atom("Mg", tuple(np.asarray(atoms[0].position) + np.array([1.0e-3, 0.0, 0.0])))
        with self.assertRaisesRegex(ValueError, "leave the group") as caught:
            check_sector_atoms(point, moved, grid, vectors)
        # the message in bohr: the Cartesian operation, W and its shift named as acting on the grid indices, and the
        # distance of an image moved by 1e-3 bohr (|R d - d| at most twice that)
        message = str(caught.exception)
        self.assertIn("(on the grid indices [", message)
        self.assertIn("(tolerance 1e-06 bohr)", message)
        import re

        distance = float(re.search(r"\(Mg\) ([0-9.e+-]+) bohr from the nearest", message).group(1))
        self.assertGreaterEqual(distance, 0.999e-3)
        self.assertLessEqual(distance, 2.001e-3)

    def test_the_frame_only_where_the_route_takes_the_cell(self) -> None:
        # a tetragonal cell on the general grid of the Fortran-derived routines (diagonal lattice, unequal steps):
        # the frame for an unpolarized run in one process, none for spin or an MPI context (the full grid of off)
        from types import SimpleNamespace

        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.acceleration.driver import _symmetry_index_frame

        a, c = 10.26, 11.2
        problem = SinglePointInput(
            atoms=[Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (a / 4, a / 4, c / 4))],
            pseudopotentials={"Si": SpeciesPotential(path=DATA / "Si_POTRE.DAT", local_angular_momentum=0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=8, fortran_routines=True),
            periodic_cell=PeriodicCell(np.diag([a, a, c])),
            scf=SCFSettings(number_of_states=8),
        )
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        self.assertTrue(getattr(grid, "is_general_periodic", False))
        self.assertTrue(needs_index_frame(grid))
        with mock.patch.dict(os.environ, _ON):
            self.assertIsNotNone(_symmetry_index_frame(grid, problem, None))
            spin = replace(problem, scf=replace(problem.scf, spin_polarized=True))
            self.assertIsNone(_symmetry_index_frame(grid, spin, None))
            self.assertIsNone(_symmetry_index_frame(grid, problem, SimpleNamespace(rank=0, root=0, size=2)))
        with mock.patch.dict(os.environ, {_OBLIQUE: "off"}):
            self.assertIsNone(_symmetry_index_frame(grid, problem, None))

    def test_molecular_dynamics_of_a_frame_cell_plan_the_sectors(self) -> None:
        # one movable atom (no random velocities): the plan of a cell; its sectors, found in the frame, give the
        # Cartesian operations the moves are checked and averaged with (Forces/sector_symmetry.py)
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.moving_sectors import plan_moving_sectors

        a = 10.26
        problem = SinglePointInput(
            atoms=[Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (a / 4, a / 4, a / 4))],
            pseudopotentials={"Si": SpeciesPotential(path=DATA / "Si_POTRE.DAT", local_angular_momentum=0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=8),
            periodic_cell=PeriodicCell(np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])),
            scf=SCFSettings(number_of_states=8),
        )
        translation = SimpleNamespace(
            problem=problem, molecular_dynamics=SimpleNamespace(friction_coefficient=0.0, time_step=40.0),
            relaxation=None, compute_forces=True, symmetry_off_reasons=("molecular_dynamics",), ignore_symmetry=False,
            afm_scan=None, restart_run=False, polarizability_field=None, movable=(True, False),
        )
        environment = {**_ON, "PARSEC_MOVING_SECTORS": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}
        with mock.patch.dict(os.environ, environment):
            plan, reason = plan_moving_sectors(translation)
        self.assertIsNone(reason)
        self.assertTrue(plan.periodic)
        self.assertEqual(plan.fixed, (False, True))

    def test_the_lost_operations_are_said_in_bohr(self) -> None:
        # an hcp cell whose second atom is 1e-5 bohr off: the line names the tolerances and the distance in bohr,
        # not in the units of the frame
        from parsec_python.acceleration.Symmetry.axis_reflection import LOOSE_ATOM_TOLERANCE, lost_operations_note

        hexagonal = _hexagonal_cell()
        grid = build_periodic_grid(PeriodicCell(hexagonal), PeriodicGridSettings(spacing=0.7, expansion_order=4))
        middle = (np.array([1 / 3, 2 / 3, 1 / 4]) + np.array([2 / 3, 1 / 3, 3 / 4])) / 2
        frac = [np.array([1 / 3, 2 / 3, 1 / 4]) - middle, np.array([2 / 3, 1 / 3, 3 / 4]) - middle]
        atoms = [Atom("Mg", tuple(f @ hexagonal)) for f in frac]
        atoms[1] = Atom("Mg", tuple(np.asarray(atoms[1].position) + np.array([0.0, 0.0, 1.0e-5])))
        with mock.patch.dict(os.environ, _ON):
            frame = index_frame(grid)
            note = lost_operations_note(
                frame.frame_grid, frame.atoms(atoms), None, atom_tolerance=frame.tolerance(1.0e-7),
                loose_tolerance=frame.tolerance(LOOSE_ATOM_TOLERANCE), frame=frame,
                reported_tolerances=(1.0e-7, LOOSE_ATOM_TOLERANCE),
            )
        self.assertIsNotNone(note)
        self.assertIn("within 0.001 bohr but not within the atom tolerance 1e-07 bohr", note)
        self.assertIn("symmetric only to 2e-05 bohr", note)

    def test_the_command_line_refuses_an_unknown_value_before_the_hand_over(self) -> None:
        # a cell on the general grid of the Fortran-derived routines goes to the reference solver: the switches are
        # read before that
        import contextlib
        import io

        from parsec_python.acceleration.cli import main

        text = (
            "Boundary_Conditions: bulk\nPeriodic_System: .true.\nbegin Cell_Shape\n  6.06 0.0 0.0\n"
            "  -3.03 5.2481139279 0.0\n  0.0 0.0 9.83\nend Cell_Shape\nLattice_Vector_Scale: 1.0 bohr\n"
            "Grid_Spacing: 0.6\nExpansion_Order: 8\nStates_Num: 4\nCoordinate_Unit: Cartesian_Bohr\n"
            "Atom_Types_Num: 1\nAtom_Type: Mg\nPseudopotential_Format: fhipp\nPotential_Num: 3\nLocal_Component: d\n"
            "Core_Cutoff_Radius: 2.6\nbegin Electron_Per_Orbital\n2 0 0\nend Electron_Per_Orbital\n"
            "begin Atom_Coord\n  0.0 3.4987 2.4575\n  3.03 1.7494 7.3725\nend Atom_Coord\n"
        )
        for name in (_OBLIQUE, _SLAB_WIRE):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                path = _input(Path(folder), text, [("Mg_FHIPP.DAT", "Mg_FHIPP.DAT")])
                errors = io.StringIO()
                with mock.patch.dict(os.environ, {name: "bogus"}), contextlib.redirect_stderr(errors), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = main([str(path)])
                self.assertEqual(code, 2)
                self.assertIn(name, errors.getvalue())


_SLAB_INPUT = """\
Boundary_Conditions: slab
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: {radius}
Grid_Spacing: {spacing}
Expansion_Order: 8
States_Num: {states}
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-9
Fermi_Temp: 300.0
Correlation_Type: {xc}
Eigensolver: chebdav
Coordinate_Unit: Cartesian_Bohr
{extra}
Atom_Types_Num: 1
Atom_Type: {symbol}
Local_Component: {local}
begin Atom_Coord
{atoms}
end Atom_Coord
"""


def _input(folder: Path, text: str, files) -> Path:
    for source, target in files:
        shutil.copy(DATA / source, folder / target)
    path = folder / "parsec.in"
    path.write_text(text, encoding="utf-8")
    return path


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SectorSCFTests(unittest.TestCase):
    """The SCF of the sectors against the full grid of the same input (A100)."""

    def run_both(self, text: str, files):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration import driver

        results = {}
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(_input(Path(folder), text, files)).problem
            for route in ("off", "auto"):
                with mock.patch.dict(os.environ, {_SLAB_WIRE: route, _OBLIQUE: route, "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}):
                    system = driver.prepare_single_point(problem, backend="cupy")
                    results[route] = driver.run_scf(system)
        return results

    def assert_same(self, results, *, group=True):
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-8)
        np.testing.assert_allclose(np.asarray(auto.eigenvalues), np.asarray(off.eigenvalues), rtol=0.0, atol=1.0e-6)
        details = dict(auto.backend.details)
        if group:
            self.assertNotIn("full grid", details["orbital_symmetry"])
        self.assertIn("full grid", dict(off.backend.details)["orbital_symmetry"])
        return details

    def test_a_slab_and_a_wire(self) -> None:
        slab = _SLAB_INPUT.format(
            cell="  3.0 0.0\n  0.0 3.2", radius=5.0, spacing=0.3, states=2, xc="ca", extra="", symbol="H", local="s",
            atoms="  0.0 0.0 0.7\n  0.0 0.0 -0.7",
        )
        details = self.assert_same(self.run_both(slab, [("H_POTRE.DAT", "H_POTRE.DAT")]))
        self.assertIn("axis-reflection", details["symmetry_detection"])
        wire = _SLAB_INPUT.format(
            cell="  3.0", radius=5.0, spacing=0.3, states=4, xc="pb", extra="", symbol="H", local="s",
            atoms="  0.0 0.8 0.0\n  0.0 -0.8 0.0\n  0.0 0.0 0.8\n  0.0 0.0 -0.8",
        ).replace("Boundary_Conditions: slab", "Boundary_Conditions: wire")
        self.assert_same(self.run_both(wire, [("spin_pbe_H_POTRE.DAT", "H_POTRE.DAT")]))

    def test_a_hexagonal_slab_and_an_oblique_cell(self) -> None:
        a = 4.648
        cell = f"  {a:.10f} 0.0\n  {-a / 2:.10f} {a * 3**0.5 / 2:.10f}"
        vectors = np.array([[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, 1.0]])
        atoms = "\n".join(
            "  " + " ".join(f"{v:.10f}" for v in np.asarray(f) @ vectors) for f in ((1 / 3, 2 / 3, 0), (2 / 3, 1 / 3, 0))
        )
        graphene = _SLAB_INPUT.format(cell=cell, radius=6.0, spacing=0.3, states=6, xc="ca", extra="", symbol="C",
                                      local="p", atoms=atoms)
        details = self.assert_same(self.run_both(graphene, [("spin_ca_C_POTRE.DAT", "C_POTRE.DAT")]))
        self.assertIn("frame of the grid indices", details["symmetry_detection"])
        s = 10.26
        silicon = (
            "Boundary_Conditions: bulk\nPeriodic_System: .true.\nbegin Cell_Shape\n"
            f"  0.0 {s / 2} {s / 2}\n  {s / 2} 0.0 {s / 2}\n  {s / 2} {s / 2} 0.0\nend Cell_Shape\n"
            "Lattice_Vector_Scale: 1.0 bohr\nGrid_Spacing: 0.5\nExpansion_Order: 8\nStates_Num: 8\n"
            "Convergence_Criterion: 1.0e-9\nDiag_Tolerance: 1.0e-9\nFermi_Temp: 300.0\nEigensolver: chebdav\n"
            "Coordinate_Unit: Cartesian_Bohr\nAtom_Types_Num: 1\nAtom_Type: Si\nLocal_Component: s\nbegin Atom_Coord\n"
            f"  0.0 0.0 0.0\n  {s / 4} {s / 4} {s / 4}\nend Atom_Coord\n"
        )
        details = self.assert_same(self.run_both(silicon, [("Si_POTRE.DAT", "Si_POTRE.DAT")]))
        self.assertIn("frame of the grid indices", details["symmetry_detection"])

    def test_the_forces_of_an_oblique_cell(self) -> None:
        # primitive fcc Si with its second atom moved along [111] (C3v): the forces of the sectors of the frame
        # (sector_forces, averaged over S W S^-1) are those of the full grid
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Forces.sector_symmetry import sector_symmetry_operations

        s = 10.26
        moved = s / 4 + 0.03
        silicon = (
            "Boundary_Conditions: bulk\nPeriodic_System: .true.\nbegin Cell_Shape\n"
            f"  0.0 {s / 2} {s / 2}\n  {s / 2} 0.0 {s / 2}\n  {s / 2} {s / 2} 0.0\nend Cell_Shape\n"
            "Lattice_Vector_Scale: 1.0 bohr\nGrid_Spacing: 0.5\nExpansion_Order: 8\nStates_Num: 8\n"
            "Convergence_Criterion: 1.0e-10\nDiag_Tolerance: 1.0e-10\nFermi_Temp: 300.0\nEigensolver: chebdav\n"
            "Coordinate_Unit: Cartesian_Bohr\nAtom_Types_Num: 1\nAtom_Type: Si\nLocal_Component: s\nbegin Atom_Coord\n"
            f"  0.0 0.0 0.0\n  {moved} {moved} {moved}\nend Atom_Coord\n"
        )
        forces = {}
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(_input(Path(folder), silicon, [("Si_POTRE.DAT", "Si_POTRE.DAT")])).problem
            for route in ("off", "auto"):
                with mock.patch.dict(os.environ, {_OBLIQUE: route}):
                    system = driver.prepare_single_point(problem, backend="cupy", sector_forces=True)
                    result = driver.run_scf(system)
                    operations = sector_symmetry_operations(system)
                    if route == "auto":
                        self.assertIsNotNone(operations)
                        self.assertIn("frame of the grid indices", dict(result.backend.details)["symmetry_detection"])
                    forces[route] = driver.compute_forces(system, result, symmetrize=operations is not None).symmetrized
        self.assertGreater(float(np.max(np.abs(forces["off"]))), 1.0e-3)
        np.testing.assert_allclose(forces["auto"], forces["off"], rtol=0.0, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
