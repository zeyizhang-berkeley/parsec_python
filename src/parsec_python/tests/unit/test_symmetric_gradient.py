"""The gradient of an oblique periodic grid averaged over the group of its lattice (``PARSEC_SYMMETRIC_GRADIENT``)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_gradient
from parsec_python.Laplacian.symmetric_gradient import (
    SWITCH,
    apply_gradient,
    apply_gradient_transpose,
    full_group_wedge,
    lattice_index_operations,
    step_matrix,
    symmetric_gradient_lines,
    symmetric_gradient_setting,
)
from parsec_python.V_xc.pbe import density_gradient, first_derivative_coefficients, gradient_transpose, pbe
from parsec_python.models import PeriodicCell, PeriodicGridSettings

_A_SI = 10.26
_FCC = np.array([[0.0, _A_SI / 2, _A_SI / 2], [_A_SI / 2, 0.0, _A_SI / 2], [_A_SI / 2, _A_SI / 2, 0.0]])
_A_NA = 7.98
_BCC = np.array([[-_A_NA / 2, _A_NA / 2, _A_NA / 2], [_A_NA / 2, -_A_NA / 2, _A_NA / 2], [_A_NA / 2, _A_NA / 2, -_A_NA / 2]])
_ON = {SWITCH: "on"}
_OFF = {SWITCH: "off"}


def _hexagonal(a=6.06, c=9.83):
    return np.round(np.array([[a, 0.0, 0.0], [-a / 2, a * 3**0.5 / 2, 0.0], [0.0, 0.0, c]]), 10)


def _grid(vectors, spacing, *, fortran=False, order=6):
    return build_periodic_grid(
        PeriodicCell(np.asarray(vectors, dtype=np.float64)),
        PeriodicGridSettings(spacing=spacing, expansion_order=order, fortran_routines=fortran),
    )


def _cases():
    return (
        ("hexagonal", _grid(_hexagonal(), 0.6), 24),
        ("hexagonal general grid", _grid(_hexagonal(), 0.6, fortran=True), 24),
        ("fcc", _grid(_FCC, 0.7), 48),
        ("fcc general grid", _grid(_FCC, 0.7, fortran=True), 48),
        ("bcc", _grid(_BCC, 0.7), 48),
        ("bcc general grid", _grid(_BCC, 0.7, fortran=True), 48),
    )


def _signed(matrix) -> bool:
    absolute = np.abs(np.asarray(matrix))
    return bool(np.all(absolute.sum(axis=0) == 1) and np.all(absolute.sum(axis=1) == 1))


def _row_map(grid, matrix):
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    images = (np.asarray(grid.integer_coordinates, dtype=np.int64) @ np.asarray(matrix, dtype=np.int64).T) % counts
    return np.asarray(grid.rows_for_integer_coordinates(images), dtype=np.int64)


def _field(grid, seed=3, shells=2):
    """A smooth periodic field (reciprocal vectors of the cell up to ``shells``) and its exact gradient."""

    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(grid.cell.lattice_vectors, dtype=np.float64)).T
    rng = np.random.default_rng(seed)
    r = np.asarray(grid.coordinates, dtype=np.float64)
    values = np.zeros(grid.size)
    gradient = np.zeros((3, grid.size))
    for m in np.ndindex(2 * shells + 1, 2 * shells + 1, 2 * shells + 1):
        m = np.asarray(m) - shells
        if not np.any(m) or np.sum(np.abs(m)) > shells:
            continue
        g = m @ reciprocal
        amplitude = rng.standard_normal() / (1.0 + float(g @ g))
        phase = r @ g + rng.uniform(0.0, 2.0 * np.pi)
        values += amplitude * np.cos(phase)
        gradient -= amplitude * np.outer(g, np.sin(phase))
    return values, gradient


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SWITCH, None)
            self.assertEqual(symmetric_gradient_setting(), "off")
        with mock.patch.dict(os.environ, _ON):
            self.assertEqual(symmetric_gradient_setting(), "on")
        with mock.patch.dict(os.environ, {SWITCH: "auto"}):
            with self.assertRaises(ValueError):
                symmetric_gradient_setting()
        with mock.patch.dict(os.environ, {SWITCH: "on", "PARSEC_OBLIQUE_FULL_GROUP": "yes"}):
            with self.assertRaises(ValueError):
                full_group_wedge()
        with mock.patch.dict(os.environ, {SWITCH: "on", "PARSEC_OBLIQUE_FULL_GROUP": "off"}):
            self.assertFalse(full_group_wedge())
        with mock.patch.dict(os.environ, {SWITCH: "on", "PARSEC_OBLIQUE_FULL_GROUP": "auto"}):
            self.assertTrue(full_group_wedge())
        with mock.patch.dict(os.environ, _OFF):
            self.assertFalse(full_group_wedge())

    def test_off_and_grids_of_signed_permutations_keep_the_axis_gradient(self) -> None:
        hexagonal = _grid(_hexagonal(), 0.6)
        with mock.patch.dict(os.environ, _OFF):
            self.assertIsNone(symmetric_gradient_lines(hexagonal))
        rng = np.random.default_rng(1)
        with mock.patch.dict(os.environ, _ON):
            for name, grid in (
                ("cube", _grid(np.diag([10.26, 10.26, 10.26]), 0.6)),
                ("tetragonal general grid", _grid(np.diag([10.26, 10.26, 11.2]), 0.6, fortran=True)),
                ("monoclinic general grid", _grid(np.array([[10.26, 0, 0], [0, 10.26, 0], [2.0, 0, 10.26]]), 0.7, fortran=True)),
            ):
                with self.subTest(name=name):
                    self.assertIsNone(symmetric_gradient_lines(grid))
                    values = rng.standard_normal(grid.size)
                    with mock.patch.dict(os.environ, _OFF):
                        expected = density_gradient(values, grid)
                    np.testing.assert_array_equal(density_gradient(values, grid), expected)


class SymmetricGradientTests(unittest.TestCase):
    def test_it_commutes_with_every_map_of_the_lattice(self) -> None:
        rng = np.random.default_rng(7)
        for name, grid, order in _cases():
            with self.subTest(name=name):
                operations = lattice_index_operations(grid)
                self.assertEqual(len(operations), order)
                steps = step_matrix(grid)
                inverse = np.linalg.inv(steps)
                values = rng.standard_normal(grid.size)
                for setting, exact in (("on", True), ("off", False)):
                    with mock.patch.dict(os.environ, {SWITCH: setting}):
                        gradient = density_gradient(values, grid)
                        scale = float(np.max(np.abs(gradient)))
                        worst = 0.0
                        for matrix in operations:
                            rows = _row_map(grid, matrix)
                            moved = np.empty_like(values)
                            moved[rows] = values
                            rotation = steps @ np.asarray(matrix, dtype=np.float64) @ inverse
                            error = float(np.max(np.abs(density_gradient(moved, grid)[:, rows] - rotation @ gradient)))
                            if _signed(matrix) or exact:
                                self.assertLess(error, 1.0e-10 * scale)
                            worst = max(worst, error)
                        if not exact:
                            # the axis gradient does not commute with the three-, four- and six-fold axes
                            self.assertGreater(worst, 0.1 * scale)

    def test_transpose_matrices_and_linear_fields(self) -> None:
        rng = np.random.default_rng(11)
        with mock.patch.dict(os.environ, _ON):
            for name, grid, _ in _cases():
                with self.subTest(name=name):
                    lines = symmetric_gradient_lines(grid)
                    self.assertIsNotNone(lines)
                    weights = first_derivative_coefficients(grid.settings.expansion_order)
                    values = rng.standard_normal(grid.size)
                    fluxes = rng.standard_normal((3, grid.size))
                    gradient = apply_gradient(values, grid, lines, weights)
                    left = float(np.sum(gradient * fluxes))
                    right = float(values @ gradient_transpose(fluxes, grid))
                    self.assertLess(abs(left - right), 1.0e-12 * abs(left))
                    np.testing.assert_allclose(
                        gradient_transpose(fluxes, grid), apply_gradient_transpose(fluxes, grid, lines, weights)
                    )
                    matrices = build_gradient(grid)
                    np.testing.assert_allclose(
                        np.vstack([sp.csr_matrix(m) @ values for m in matrices]), gradient, rtol=0.0,
                        atol=1.0e-12 * float(np.max(np.abs(gradient))),
                    )
                    # consistent: a field linear along a lattice direction within the stencil (each difference is
                    # exact for it), so the coefficients sum to the identity
                    total = np.zeros((3, 3))
                    for offset, vector in zip(lines.offsets, lines.vectors):
                        total += np.outer(vector, steps_of(grid) @ offset)
                    np.testing.assert_allclose(total, np.eye(3), atol=1.0e-12)

    def test_the_order_of_accuracy(self) -> None:
        # the error of both gradients falls as h^(Expansion_Order) on a smooth field (order 4: about 16x per halving)
        for vectors in (_hexagonal(), _FCC, _BCC):
            errors = {"on": [], "off": []}
            for spacing in (0.6, 0.3):
                grid = _grid(vectors, spacing, order=4)
                values, exact = _field(grid)
                for setting in errors:
                    with mock.patch.dict(os.environ, {SWITCH: setting}):
                        errors[setting].append(float(np.max(np.abs(density_gradient(values, grid) - exact))))
            for setting, (coarse, fine) in errors.items():
                with self.subTest(cell=vectors.tolist()[0], setting=setting):
                    self.assertGreater(coarse / fine, 10.0)
            # the same order: their ratio stays within a factor of two
            self.assertLess(errors["on"][1] / errors["off"][1], 2.0)
            self.assertGreater(errors["on"][1] / errors["off"][1], 0.5)

    def test_the_pbe_potential_of_a_symmetric_density_is_symmetric(self) -> None:
        for name, grid, _ in _cases():
            with self.subTest(name=name):
                operations = lattice_index_operations(grid)
                maps = [_row_map(grid, matrix) for matrix in operations]
                values, _ = _field(grid, seed=5, shells=3)
                density = 0.02 + 0.015 * values / float(np.max(np.abs(values)))
                density = np.mean([density[rows] for rows in maps], axis=0)
                worst = {}
                for setting in ("on", "off"):
                    with mock.patch.dict(os.environ, {SWITCH: setting}):
                        potential = np.asarray(pbe(density, grid, None).potential)
                    worst[setting] = max(float(np.max(np.abs(potential[rows] - potential))) for rows in maps)
                self.assertLess(worst["on"], 1.0e-12)
                self.assertGreater(worst["off"], 1.0e-8)


class WedgeTests(unittest.TestCase):
    def test_the_stencil_exact_wedge_keeps_the_maps_of_the_lattice(self) -> None:
        from parsec_python.Grid.kpoints import default_stencil_lines, stencil_preserving_matrices

        vectors = _hexagonal()
        counts = np.array([12, 12, 16])
        lines = default_stencil_lines(vectors, counts)
        # 60-degree rotation about c (lattice coordinates), a mirror and the identity
        matrices = np.array([[[1, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]], np.eye(3, dtype=int)])
        with mock.patch.dict(os.environ, _OFF):
            self.assertEqual(stencil_preserving_matrices(matrices, counts, lines), [False, True, True])
        with mock.patch.dict(os.environ, {SWITCH: "on", "PARSEC_OBLIQUE_FULL_GROUP": "off"}):
            self.assertEqual(stencil_preserving_matrices(matrices, counts, lines), [False, True, True])
        with mock.patch.dict(os.environ, _ON):
            self.assertEqual(stencil_preserving_matrices(matrices, counts, lines), [True, True, True])

    def test_the_fortran_mesh_takes_the_shortest_representatives(self) -> None:
        # fcc Ag with a shifted 5x5x5 mesh (the Fortran-derived routines, not requested): Fortran's representatives
        # in (-1/2, 1/2] include two that are no shortest image; under on the mesh takes the shortest ones (no free
        # level below zero calls the guard), unless Use_Fortran_Routines: .true. or PARSEC_KPOINT_MESH_GUARD=off
        import itertools
        import shutil
        import tempfile
        import warnings
        from pathlib import Path

        from parsec_python.Grid import kpoint_sampling
        from parsec_python.Input import parse_parsec_input

        half = 7.72 / 2
        text = (
            "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n"
            f"  0.0 {half} {half}\n  {half} 0.0 {half}\n  {half} {half} 0.0\nend Cell_Shape\n"
            "Lattice_Vector_Scale: 1.0 bohr\nCoordinate_Unit: Cartesian_Bohr\nGrid_Spacing: 0.6\nExpansion_Order: 8\n"
            "Correlation_Type: pbe\nStates_Num: 12\nFermi_Temp: 500.0\nKpoint_Method: mp\n"
            "begin Monkhorst_Pack_Grid\n  5 5 5\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n  0.0 0.0 0.0\nend Monkhorst_Pack_Shift\n{extra}"
            "Atom_Types_Num: 1\nAtom_Type: Ag\nLocal_Component: s\nbegin Atom_Coord\n  0.0 0.0 0.0\nend Atom_Coord\n"
        )
        data = Path(__file__).resolve().parents[1] / "data"

        def mesh(extra, environment):
            folder = Path(tempfile.mkdtemp(prefix="symgrad_mesh_"))
            try:
                shutil.copy(data / "Ag_POTRE.DAT", folder / "Ag_POTRE.DAT")
                (folder / "parsec.in").write_text(text.format(extra=extra))
                cwd = os.getcwd()
                try:
                    os.chdir(folder)
                    with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        problem = parse_parsec_input(folder / "parsec.in").problem
                        points, weights = kpoint_sampling(problem)
                finally:
                    os.chdir(cwd)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            reciprocal = 2.0 * np.pi * np.linalg.inv(problem.periodic_cell.lattice_vectors).T
            longer = 0
            for point in np.asarray(points):
                fraction = point @ np.linalg.inv(reciprocal)
                shortest = min(
                    float(np.linalg.norm((fraction + np.array(shift)) @ reciprocal))
                    for shift in itertools.product((-1, 0, 1), repeat=3)
                )
                longer += int(float(np.linalg.norm(point)) > shortest + 1.0e-9)
            self.assertTrue(problem.grid.fortran_routines)
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
            return problem, longer

        problem, longer = mesh("", _OFF)
        self.assertEqual(longer, 2)
        self.assertFalse(getattr(problem, "kpoint_shortest_representatives", False))
        problem, longer = mesh("", _ON)
        self.assertEqual(longer, 0)
        self.assertTrue(problem.kpoint_shortest_representatives)
        self.assertTrue(problem.kpoint_tie_images)
        self.assertEqual(int(problem.space_group.order), 48)
        for extra, environment in (
            ("Use_Fortran_Routines: .true.\n", _ON),
            ("", {**_ON, "PARSEC_KPOINT_MESH_GUARD": "off"}),
        ):
            with self.subTest(extra=extra, environment=environment):
                problem, longer = mesh(extra, environment)
                self.assertEqual(longer, 2)
                self.assertFalse(getattr(problem, "kpoint_shortest_representatives", False))

    def test_the_free_levels_follow_the_gradient(self) -> None:
        # the lowest level of the free Bloch operator -L - 2ik.G + k^2 of the grid: the guard's plane-wave symbols
        # against the dense operator, with each gradient.  The axis gradient of the oblique grid has a level below
        # zero at this k; the symmetric one, a difference along each line of the Laplacian, has none (each second
        # difference bounds the square of the first difference along its line, as on an orthogonal grid)
        from parsec_python.Grid.kpoints import lowest_free_levels
        from parsec_python.Laplacian import build_negative_laplacian

        cell = PeriodicCell(_BCC)
        settings = PeriodicGridSettings(spacing=0.9, expansion_order=4, fortran_routines=True)
        grid = build_periodic_grid(cell, settings)
        reciprocal = 2.0 * np.pi * np.linalg.inv(_BCC).T
        laplacian = sp.csr_matrix(build_negative_laplacian(grid)).toarray()
        for fraction in ([0.75, 0.75, 0.0], [0.6, 0.6, 0.6]):
            k_point = np.array(fraction) @ reciprocal
            for setting in ("off", "on"):
                with self.subTest(k=fraction, setting=setting), mock.patch.dict(os.environ, {SWITCH: setting}):
                    gradient = [sp.csr_matrix(m).toarray() for m in build_gradient(grid)]
                    operator = laplacian - 2j * sum(k_point[c] * gradient[c] for c in range(3))
                    operator = operator + float(k_point @ k_point) * np.eye(grid.size)
                    lowest = float(np.min(np.linalg.eigvalsh(0.5 * (operator + operator.conj().T))))
                    found = lowest_free_levels(k_point[None, :], cell, settings)
                    self.assertIsNotNone(found)
                    if setting == "off":
                        self.assertLess(lowest, -0.5)
                        self.assertAlmostEqual(float(found[0]), lowest, delta=1.0e-8 * abs(lowest))
                    else:
                        self.assertGreater(lowest, -1.0e-10)
                        self.assertGreaterEqual(float(found[0]), 0.0)


def steps_of(grid):
    return step_matrix(grid)


if __name__ == "__main__":
    unittest.main()
