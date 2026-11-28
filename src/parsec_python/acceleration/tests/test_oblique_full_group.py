"""The full hexagonal and cubic groups of an oblique cell (``PARSEC_OBLIQUE_FULL_GROUP``).

The operations whose action on the grid indices is an integer matrix but no signed permutation
(:func:`parsec_python.acceleration.Symmetry.oblique_frame.general_index_operations`) keep the Laplacian of both routine
sets; the sectors that take them solve the static Hamiltonian of the full grid, a hexagonal cell keeps the sectors of
its signed permutations (no fewer rows times states), and only a functional without a density gradient takes them.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.acceleration.Symmetry.axis_reflection import (
    SignedPermutationReduction,
    _grid_row_mapping_operation,
    detect_point_group_reduction,
    holds_general_operations,
)
from parsec_python.acceleration.Symmetry.oblique_frame import (
    general_index_operations,
    index_frame,
    oblique_full_group_setting,
    sector_negative_laplacian,
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
_SWITCH = "PARSEC_OBLIQUE_FULL_GROUP"
_ON = {"PARSEC_OBLIQUE_SECTORS": "auto", _SWITCH: "auto"}
_A_SI = 10.26
_FCC = np.array([[0.0, _A_SI / 2, _A_SI / 2], [_A_SI / 2, 0.0, _A_SI / 2], [_A_SI / 2, _A_SI / 2, 0.0]])
_A_NA = 7.98
_BCC = np.array([[-_A_NA / 2, _A_NA / 2, _A_NA / 2], [_A_NA / 2, -_A_NA / 2, _A_NA / 2], [_A_NA / 2, _A_NA / 2, -_A_NA / 2]])


def _hexagonal(a=6.06, c=9.83):
    return np.round(np.array([[a, 0.0, 0.0], [-a / 2, a * 3**0.5 / 2, 0.0], [0.0, 0.0, c]]), 10)


def _grid(vectors, spacing, *, fortran=False):
    return build_periodic_grid(
        PeriodicCell(np.asarray(vectors, dtype=np.float64)),
        PeriodicGridSettings(spacing=spacing, expansion_order=6, fortran_routines=fortran),
    )


def _wells(grid, atoms, depth=2.0, width=0.9):
    coordinates = np.asarray(grid.coordinates, dtype=float)
    vectors = np.asarray(grid.cell.lattice_vectors, dtype=float)
    shifts = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)], dtype=float)
    potential = np.zeros(grid.size)
    for atom in atoms:
        for image in shifts @ vectors:
            difference = coordinates - (np.asarray(atom.position) + image)
            potential -= depth * np.exp(-np.sum(difference * difference, axis=1) / width)
    return potential


def _detect(frame, atoms):
    frame_atoms = frame.atoms(atoms)
    tolerance = frame.tolerance(1.0e-7)
    abelian = SignedPermutationReduction.detect(frame.frame_grid, frame_atoms, atom_tolerance=tolerance)
    return detect_point_group_reduction(frame.frame_grid, frame_atoms, abelian, atom_tolerance=tolerance)


def _weighted_levels(decomposition, hamiltonian, count):
    weighted = []
    for index, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
        if matrix is None or matrix.shape[0] == 0:
            continue
        weight = int(decomposition.sector_weights[index])
        weighted += [value for value in np.linalg.eigvalsh(matrix.toarray())[:count] for _ in range(weight)]
    return np.sort(np.asarray(weighted))[:count]


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_SWITCH, None)
            self.assertEqual(oblique_full_group_setting(), "auto")
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            self.assertEqual(oblique_full_group_setting(), "off")
        with mock.patch.dict(os.environ, {_SWITCH: "on"}):
            with self.assertRaises(ValueError):
                oblique_full_group_setting()

    def test_only_a_functional_without_a_gradient_takes_them(self) -> None:
        from parsec_python.acceleration.driver import _oblique_full_group

        problem = SinglePointInput(
            atoms=[Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (_A_SI / 4, _A_SI / 4, _A_SI / 4))],
            pseudopotentials={"Si": SpeciesPotential(path=DATA / "Si_POTRE.DAT", local_angular_momentum=0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=8),
            periodic_cell=PeriodicCell(_FCC),
            scf=SCFSettings(number_of_states=8, xc_functional="ca"),
        )
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            self.assertFalse(_oblique_full_group(problem))
        with mock.patch.dict(os.environ, {_SWITCH: "auto"}):
            self.assertTrue(_oblique_full_group(problem))
            self.assertTrue(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="pw92"))))
            self.assertFalse(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="pbe"))))
            self.assertFalse(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="lb94"))))
        # a GGA with the gradient averaged over the group of the lattice (PARSEC_SYMMETRIC_GRADIENT=on)
        with mock.patch.dict(os.environ, {_SWITCH: "auto", "PARSEC_SYMMETRIC_GRADIENT": "on"}):
            self.assertTrue(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="pbe"))))
            self.assertTrue(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="pw91"))))
        with mock.patch.dict(os.environ, {_SWITCH: "off", "PARSEC_SYMMETRIC_GRADIENT": "on"}):
            self.assertFalse(_oblique_full_group(replace(problem, scf=replace(problem.scf, xc_functional="pbe"))))


class GeneralOperationTests(unittest.TestCase):
    def test_counts_isometries_and_the_laplacian(self) -> None:
        # D6h has 24 operations, 8 of them signed permutations of the hexagonal grid axes; Oh 48, 12 of them of the
        # primitive fcc and bcc grids.  Each of the others keeps the Laplacian of the grid (both routine sets).
        cases = (
            ("hexagonal", _grid(_hexagonal(), 0.6), 16),
            ("hexagonal general grid", _grid(_hexagonal(), 0.6, fortran=True), 16),
            ("fcc", _grid(_FCC, 0.7), 36),
            ("fcc general grid", _grid(_FCC, 0.7, fortran=True), 36),
            ("bcc", _grid(_BCC, 0.7), 36),
            ("bcc general grid", _grid(_BCC, 0.7, fortran=True), 36),
        )
        for name, grid, count in cases:
            with self.subTest(name=name):
                operations = general_index_operations(grid)
                self.assertEqual(len(operations), count)
                frame = index_frame(grid, general=True)
                self.assertEqual(len(frame.general), count)
                self.assertEqual(frame.order, count + len(frame.signed().operations))
                laplacian = sector_negative_laplacian(grid, frame.operations)
                if laplacian is None:
                    laplacian = build_negative_laplacian(grid)
                laplacian = sp.csr_matrix(laplacian)
                scale = abs(laplacian).max()
                for operation in operations:
                    rotation = frame.cartesian(operation)
                    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
                    rows = _grid_row_mapping_operation(frame.frame_grid, operation, 1.0e-10)
                    self.assertIsNotNone(rows)
                    self.assertEqual(np.unique(rows).size, grid.size)
                    self.assertLess(abs(laplacian[rows][:, rows] - laplacian).max(), 1.0e-12 * scale)
                # without the switch's operations the frame is the one of before
                plain = index_frame(grid)
                self.assertEqual(plain.general, ())
                self.assertEqual(
                    {np.asarray(op).tobytes() for op in plain.operations},
                    {np.asarray(op).tobytes() for op in frame.signed().operations},
                )

    def test_the_sectors_solve_the_full_grid(self) -> None:
        cases = (
            # one Na atom of bcc: Oh in sectors of a subgroup of 16
            ("bcc", _grid(_BCC, 0.65), [Atom("Na", (0.0, 0.0, 0.0))], 48),
            ("bcc general grid", _grid(_BCC, 0.65, fortran=True), [Atom("Na", (0.0, 0.0, 0.0))], 48),
            # diamond about its bond centre: Oh with translations of half a lattice vector
            ("fcc", _grid(_FCC, 0.75), [Atom("Si", tuple(np.ones(3) * _A_SI / 8)), Atom("Si", tuple(-np.ones(3) * _A_SI / 8))], 48),
            # diamond with an atom at the origin: Td about it (24), and the inversion about the bond centre where it
            # is a translation of whole steps (counts a multiple of 4: 48)
            ("fcc Td", _grid(_FCC, 0.75, fortran=True), [Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", tuple(np.ones(3) * _A_SI / 4))], 0),
            ("fcc Td 14", _grid(_FCC, 0.52), [Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", tuple(np.ones(3) * _A_SI / 4))], 0),
        )
        for name, grid, atoms, order in cases:
            if order == 0:
                order = 48 if int(grid.points_per_axis[0]) % 4 == 0 else 24
            with self.subTest(name=name, counts=tuple(int(v) for v in grid.points_per_axis)):
                frame = index_frame(grid, general=True)
                reduction = _detect(frame, atoms)
                self.assertTrue(holds_general_operations(reduction))
                self.assertEqual(len(reduction.full_operations), order)
                signed = _detect(index_frame(grid), atoms)
                self.assertLess(int(getattr(signed, "full_operations", signed.operations).shape[0]), order)
                laplacian = sector_negative_laplacian(grid, frame.operations)
                if laplacian is None:
                    laplacian = build_negative_laplacian(grid)
                hamiltonian = sp.csr_matrix(0.5 * laplacian + sp.diags(_wells(grid, atoms)))
                decomposition = ReflectionRepresentationDecomposition.build(frame.frame_grid, reduction)
                self.assertLess(max(decomposition.sector_sizes), grid.size // 4)
                count = 30
                np.testing.assert_allclose(
                    _weighted_levels(decomposition, hamiltonian, count),
                    np.linalg.eigvalsh(hamiltonian.toarray())[:count],
                    rtol=0.0,
                    atol=1.0e-10,
                )

    def test_a_hexagonal_cell_keeps_the_sectors_of_its_signed_permutations(self) -> None:
        # D6h solves no fewer rows times states in real one-dimensional sectors than its 8 signed permutations: the
        # detection with the switch gives the reduction of before
        hexagonal = _hexagonal()
        middle = (np.array([1 / 3, 2 / 3, 1 / 4]) + np.array([2 / 3, 1 / 3, 3 / 4])) / 2
        atoms = [Atom("Mg", tuple((f - middle) @ hexagonal)) for f in (np.array([1 / 3, 2 / 3, 1 / 4]), np.array([2 / 3, 1 / 3, 3 / 4]))]
        grid = _grid(hexagonal, 0.7)
        extended = _detect(index_frame(grid, general=True), atoms)
        plain = _detect(index_frame(grid), atoms)
        self.assertFalse(holds_general_operations(extended))
        self.assertEqual(type(extended), type(plain))
        np.testing.assert_array_equal(extended.representative_rows, plain.representative_rows)
        np.testing.assert_array_equal(extended.full_to_wedge, plain.full_to_wedge)
        np.testing.assert_array_equal(np.asarray(extended.operations), np.asarray(plain.operations))

    def test_the_force_operations_are_cartesian(self) -> None:
        from parsec_python.acceleration.Forces.sector_symmetry import sector_group_operations

        atoms = [Atom("Si", tuple(np.ones(3) * _A_SI / 8)), Atom("Si", tuple(-np.ones(3) * _A_SI / 8))]
        grid = _grid(_FCC, 0.75)
        frame = index_frame(grid, general=True)
        reduction = _detect(frame, atoms)
        object.__setattr__(reduction, "index_frame", frame)
        operations = sector_group_operations(reduction, atoms, reduction.grid, _FCC)
        self.assertEqual(len(operations), 48)
        positions = np.asarray([atom.position for atom in atoms])
        inverse = np.linalg.inv(_FCC)
        for operation in operations:
            rotation = np.asarray(operation.matrix)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9)
            tau = np.zeros(3) if operation.translation is None else np.asarray(operation.translation)
            difference = ((positions - tau) @ rotation.T - positions[np.asarray(operation.image)]) @ inverse
            np.testing.assert_allclose(difference - np.rint(difference), 0.0, atol=1.0e-9)


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class GammaSectorTests(unittest.TestCase):
    def test_energies_and_forces_against_the_full_grid(self) -> None:
        from parsec_python.acceleration import driver

        problem = _fcc_problem(moved=0.05)
        results = {}
        for label, environment in (
            ("full", {"PARSEC_OBLIQUE_SECTORS": "off", _SWITCH: "off"}),
            ("signed", {"PARSEC_OBLIQUE_SECTORS": "auto", _SWITCH: "off"}),
            ("general", _ON),
        ):
            with mock.patch.dict(os.environ, environment):
                system = driver.prepare_single_point(problem, backend="cupy", sector_forces=label != "full")
                result = driver.run_scf(system)
                self.assertTrue(result.converged)
                detection = str(dict(result.backend.details).get("symmetry_detection", ""))
                forces = driver.compute_forces(system, result, symmetrize=label != "full")
                results[label] = (float(result.energies.total), np.asarray(forces.symmetrized), detection)
        self.assertIn("other integer maps", results["general"][2])
        self.assertNotIn("other integer maps", results["signed"][2])
        for label in ("signed", "general"):
            self.assertLess(abs(results[label][0] - results["full"][0]), 1.0e-9)
            np.testing.assert_allclose(results[label][1], results["full"][1], rtol=0.0, atol=1.0e-7)

    def test_pbe_with_the_symmetric_gradient_against_the_full_grid(self) -> None:
        # PBE takes the full cubic group with the gradient averaged over the group of the lattice
        # (PARSEC_SYMMETRIC_GRADIENT=on); without it a GGA keeps the signed permutations
        from parsec_python.acceleration import driver

        problem = _fcc_problem(moved=0.05, xc="pbe")
        results = {}
        for label, environment in (
            ("full", {"PARSEC_OBLIQUE_SECTORS": "off", "PARSEC_SYMMETRIC_GRADIENT": "on"}),
            ("general", {**_ON, "PARSEC_SYMMETRIC_GRADIENT": "on"}),
            ("axis", {**_ON, "PARSEC_SYMMETRIC_GRADIENT": "off"}),
        ):
            with mock.patch.dict(os.environ, environment):
                system = driver.prepare_single_point(problem, backend="cupy", sector_forces=label != "full")
                result = driver.run_scf(system)
                self.assertTrue(result.converged)
                detection = str(dict(result.backend.details).get("symmetry_detection", ""))
                forces = driver.compute_forces(system, result, symmetrize=label != "full")
                results[label] = (float(result.energies.total), np.asarray(forces.symmetrized), detection)
        self.assertIn("other integer maps", results["general"][2])
        self.assertNotIn("other integer maps", results["axis"][2])
        self.assertLess(abs(results["general"][0] - results["full"][0]), 1.0e-9)
        np.testing.assert_allclose(results["general"][1], results["full"][1], rtol=0.0, atol=1.0e-7)
        # another discretization of the gradient: a different energy, within the order of the stencil
        self.assertGreater(abs(results["axis"][0] - results["full"][0]), 1.0e-8)
        self.assertLess(abs(results["axis"][0] - results["full"][0]), 1.0e-2)


def _fcc_problem(moved=0.0, xc="ca"):
    # diamond Si in a 2 x 2 x 2 supercell of the primitive fcc cell, atom 1 moved along z: C2v about z (its C2z is no
    # signed permutation of the grid axes)
    vectors = 2.0 * _FCC
    basis = [np.zeros(3), np.ones(3) * _A_SI / 4]
    atoms = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                shift = i * _FCC[0] + j * _FCC[1] + k * _FCC[2]
                for position in basis:
                    atoms.append(position + shift)
    atoms[1] = atoms[1] + np.array([0.0, 0.0, moved])
    return SinglePointInput(
        atoms=[Atom("Si", tuple(position)) for position in atoms],
        pseudopotentials={"Si": SpeciesPotential(path=DATA / "Si_POTRE.DAT", local_angular_momentum=0)},
        grid=PeriodicGridSettings(spacing=0.75, expansion_order=8),
        periodic_cell=PeriodicCell(vectors),
        scf=SCFSettings(number_of_states=40, convergence_criterion=1e-10, max_iterations=80, xc_functional=xc),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1e-10),
        hartree=HartreeSettings(relative_tolerance=1e-12),
    )


if __name__ == "__main__":
    unittest.main()
