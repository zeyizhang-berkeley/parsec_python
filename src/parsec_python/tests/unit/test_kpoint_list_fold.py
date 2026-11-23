"""The points of a ``Kpoint_List`` on oblique cells (``PARSEC_KPOINT_LIST_FOLD``).

PARSEC 1.5 runs the vectors of a manual list as written.  The finite-difference Bloch operator
``-L - 2i k.G + k^2`` is not invariant under ``k -> k + G``, and on an oblique grid it is not bounded below in ``k``:
some plane waves have ``-L < |G|^2``, so a point far from the first Brillouin zone can have levels below zero that no
continuum state has.

* ``auto`` (default): on the default routines the list runs at the shortest images of its points, a point of the zone
  boundary at all of them with equal shares of its weight.  The Fortran-derived routines keep the written vectors
  unless one has a free level below zero; then a list on routines that were not requested is treated as on the default
  routines, and one under ``Use_Fortran_Routines: .true.`` keeps its vectors with a warning.
* ``shared``: the treatment of the default routines on every route.
* ``literal``: the written vectors on every route.

``PARSEC_KPOINT_MESH_GUARD`` does the same for a Monkhorst-Pack mesh with such a level on Fortran-derived routines that
the input did not request.  Band paths and ``nscf_kpoints`` get a warning and keep their points.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.Grid.cell import reciprocal_lattice
from parsec_python.Grid.kpoints import (
    FREE_LEVEL_TOLERANCE,
    fold_to_centered_zone,
    fold_to_shortest,
    kpoint_list_fold_setting,
    kpoint_mesh_guard_setting,
    kpoint_sampling,
    lowest_free_levels,
    negation_closed_shortest,
    negation_closed_written,
    shortest_images,
    shortest_list_points,
    tie_shared_points,
    time_reversal_reduced,
    written_tie_images,
)
from parsec_python.Input.parsec_input import read_symmetry_of_structure
from parsec_python.Input.restarted_symmetry import symmetry_signature
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.tests.unit.test_oblique_kpoint_wedge import _BCC, _brute_force, _lattices, _mesh, _same_rows
from parsec_python.tests.unit.test_soc_kpoint_wedge import _FCC, _Directory, _text

_SCALAR = dict(extra="", so_psp=".false.")
_BLOCK = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n{mesh}\nend Monkhorst_Pack_Grid\n"
_NOTE = "Kpoint_List:"
_FAR = (5 / 6, 5 / 6, 5 / 6)


def _listed(text: str, mesh: str, fractions, weights=None) -> str:
    """``text`` with its Monkhorst-Pack block replaced by a list of ``fractions`` along the reciprocal vectors."""

    block = _BLOCK.format(mesh=mesh)
    assert block in text
    rows = []
    for index, row in enumerate(np.asarray(fractions, dtype=float)):
        weight = "" if weights is None else f" {float(weights[index]):.16f}"
        rows.append(" ".join(f"{value:.16f}" for value in row) + weight)
    return text.replace(
        block,
        "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
        + "\n".join(rows)
        + "\nend Kpoint_List\n",
    )


def _bcc(extra: str = "", mesh: str = "6 6 6", spacing: float = 0.6) -> str:
    """A primitive bcc Au cell at order 12 on a grid of at most 12 points per axis (the Fortran-derived routines)."""

    text = _text(cell="fcc", mesh=mesh, spacing=spacing, extra=extra, so_psp=".false.").replace(_FCC, _BCC)
    return text.replace("Expansion_Order: 6", "Expansion_Order: 12")


def _exact_levels(lattice, spacing, order, k_points, fortran=False):
    """The lowest level of the free operator at each point, from the FFT of one row of the grid's matrices."""

    cell = PeriodicCell(lattice_vectors=np.asarray(lattice, dtype=float))
    settings = PeriodicGridSettings(spacing=spacing, expansion_order=order, fortran_routines=fortran)
    grid = build_periodic_grid(cell, settings)
    points = np.asarray(grid.integer_coordinates)
    low = points.min(axis=0)
    counts = points.max(axis=0) - low + 1
    origin = int(np.flatnonzero(np.all(points == low, axis=1))[0])

    def symbol(matrix):
        row = matrix.tocsr().getrow(origin)
        kernel = np.zeros(tuple(counts), dtype=np.complex128)
        for column, value in zip(row.indices, row.data):
            kernel[tuple(points[column] - low)] += value
        return np.fft.fftn(kernel)

    laplacian = symbol(build_negative_laplacian(grid))
    gradient = [symbol(matrix) for matrix in build_gradient(grid)]
    levels = []
    for k in np.atleast_2d(k_points):
        drift = sum(value * part for value, part in zip(k, gradient))
        levels.append(float((laplacian - 2j * drift + k @ k).real.min()))
    return cell, settings, np.asarray(levels)


class PointTests(unittest.TestCase):
    def test_only_the_longer_points_move(self) -> None:
        generator = np.random.default_rng(11)
        fractions = np.vstack(
            [generator.uniform(-1.5, 1.5, size=(120, 3)), _mesh((4, 4, 4)), fold_to_centered_zone(_mesh((4, 4, 4)))]
        )
        for name, lattice in _lattices().items():
            with self.subTest(lattice=name):
                reciprocal = reciprocal_lattice(lattice)
                written = fractions @ reciprocal
                points, moved = shortest_list_points(written, lattice)
                best = _brute_force(fractions, reciprocal)[0]
                np.testing.assert_allclose(np.sum(points**2, axis=1), best, rtol=1.0e-9, atol=1.0e-12)
                difference = (points - written) @ np.linalg.inv(reciprocal)
                np.testing.assert_allclose(difference, np.rint(difference), atol=1.0e-9)
                np.testing.assert_array_equal(points[~moved], written[~moved])
                self.assertTrue(np.all(np.sum(written[moved] ** 2, axis=1) > best[moved]))
                again, none = shortest_list_points(points, lattice)
                self.assertFalse(np.any(none))
                np.testing.assert_array_equal(again, points)

    def test_setting(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("PARSEC_KPOINT_LIST_FOLD", None)
            self.assertEqual(kpoint_list_fold_setting(), "auto")
        for value in ("auto", "shared", "literal"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_LIST_FOLD": value.upper()}):
                self.assertEqual(kpoint_list_fold_setting(), value)
        for value in ("shortest", "centred"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_LIST_FOLD": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_LIST_FOLD"):
                    kpoint_list_fold_setting()

    def test_the_free_levels_are_those_of_the_grid(self) -> None:
        for name, spacing, fortran in (
            ("bcc", 0.6, True),
            ("bcc", 0.4, False),
            ("fcc", 0.35, False),
            ("hexagonal", 0.5, True),
            ("monoclinic", 0.6, True),
        ):
            with self.subTest(lattice=name, spacing=spacing):
                lattice = _lattices()[name]
                reciprocal = reciprocal_lattice(lattice)
                raw = _mesh((4, 4, 4)) + 1.0 / 12.0
                fractions = np.vstack([raw, fold_to_centered_zone(raw), fold_to_shortest(raw, reciprocal)])
                k_points = fractions @ reciprocal
                cell, settings, exact = _exact_levels(lattice, spacing, 12, k_points, fortran)
                levels = lowest_free_levels(k_points, cell, settings)
                below = levels < 0.0
                np.testing.assert_allclose(levels[below], exact[below], atol=1.0e-9)
                # a value that is not below zero says that no level is
                self.assertFalse(np.any(exact[~below] < -FREE_LEVEL_TOLERANCE))
                # and none in the first zone
                self.assertFalse(np.any(levels[2 * len(raw):] < -FREE_LEVEL_TOLERANCE))

    def test_a_point_far_from_the_first_zone_and_its_shortest_image(self) -> None:
        bcc = np.array([[-3.995, 3.995, 3.995], [3.995, -3.995, 3.995], [3.995, 3.995, -3.995]])
        reciprocal = reciprocal_lattice(bcc)
        far = np.array([_FAR]) @ reciprocal
        cell = PeriodicCell(lattice_vectors=bcc)
        settings = PeriodicGridSettings(spacing=0.4, expansion_order=12, fortran_routines=True)
        self.assertLess(float(lowest_free_levels(far, cell, settings)[0]), -1.0)
        near = fold_to_shortest(np.array([_FAR]), reciprocal) @ reciprocal
        self.assertFalse(np.any(lowest_free_levels(near, cell, settings) < -FREE_LEVEL_TOLERANCE))

    def test_no_level_below_zero_on_an_orthogonal_grid(self) -> None:
        for order in (2, 6, 12, 20):
            cube = PeriodicCell(lattice_vectors=8.0 * np.eye(3))
            settings = PeriodicGridSettings(spacing=0.5, expansion_order=order, fortran_routines=True)
            k_points = np.array([_FAR, (3.0, -2.0, 1.5)]) * 2.0 * np.pi / 8.0
            self.assertIsNone(lowest_free_levels(k_points, cube, settings))
            levels = _exact_levels(8.0 * np.eye(3), 0.5, order, k_points, True)[2]
            self.assertTrue(np.all(levels > -1.0e-10))


class ParserTests(_Directory):
    def notes(self, translation):
        return [line for line in translation.warnings if line.startswith(_NOTE)]

    def test_the_default_routines_run_the_points_of_the_mesh_however_written(self) -> None:
        for cell, mesh in (("fcc", "4 4 4"), ("hcp", "4 4 2")):
            with self.subTest(cell=cell):
                dimensions = tuple(int(v) for v in mesh.split())
                base = _text(cell=cell, mesh=mesh, **_SCALAR)
                reference = None
                for written in (_mesh(dimensions), fold_to_centered_zone(_mesh(dimensions))):
                    translation = self.parse(_listed(base, mesh, written))
                    problem = translation.problem
                    self.assertFalse(translation.routines.fortran)
                    self.assertTrue(problem.kpoint_tie_images)
                    k_points, weights = kpoint_sampling(problem)
                    self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
                    lattice = problem.periodic_cell.lattice_vectors
                    reciprocal = reciprocal_lattice(lattice)
                    closed = negation_closed_shortest(_mesh(dimensions), reciprocal) @ reciprocal
                    expected, shares = tie_shared_points(closed, np.full(len(closed), 1.0 / len(closed)), lattice)
                    table = np.column_stack([k_points, weights])
                    self.assertTrue(_same_rows(table, np.column_stack([expected, shares]), atol=1.0e-9))
                    if reference is not None:
                        self.assertTrue(_same_rows(table, reference, atol=1.0e-9))
                    reference = table

    def test_a_list_inside_the_zone_keeps_its_bits(self) -> None:
        text = _text(cell="hcp", **_SCALAR)
        written = np.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.2], [-0.15, 0.1, -0.3]])
        default = self.parse(_listed(text, "4 4 2", written))
        literal = self.parse(_listed(text, "4 4 2", written), {"PARSEC_KPOINT_LIST_FOLD": "literal"})
        np.testing.assert_array_equal(default.problem.kpoints, literal.problem.kpoints)
        self.assertFalse(default.problem.kpoint_tie_images)
        self.assertEqual(self.notes(default), [])

    def test_literal_keeps_the_written_vectors(self) -> None:
        written = _mesh((4, 4, 2))
        translation = self.parse(_listed(_text(cell="hcp", **_SCALAR), "4 4 2", written),
                                 {"PARSEC_KPOINT_LIST_FOLD": "literal"})
        problem = translation.problem
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(problem.kpoints, written @ reciprocal, atol=1.0e-14)
        self.assertFalse(problem.kpoint_tie_images)
        # (no free level below zero on this grid: nothing to say)
        self.assertEqual(self.notes(translation), [])

    def test_orthogonal_cells_move_the_points_outside_the_zone(self) -> None:
        """PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD=auto: a point of an orthogonal cell outside the first zone runs at its
        shortest image with its weight, a point of the boundary (a component 1/2) and every point of a list inside the
        zone keep their written vectors; literal and off keep every written vector."""

        written = _mesh((4, 4, 4))
        text = _listed(_text(cell="cube", **_SCALAR), "4 4 4", written)
        reference = self.parse(text, {"PARSEC_KPOINT_LIST_FOLD": "literal"}).problem
        reciprocal = reciprocal_lattice(reference.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(reference.kpoints, written @ reciprocal, atol=1.0e-14)
        expected = np.where(written > 0.5 + 1.0e-12, written - 1.0, written) @ reciprocal
        for value in ("auto", "shared"):
            with self.subTest(value=value):
                translation = self.parse(text, {"PARSEC_KPOINT_LIST_FOLD": value})
                np.testing.assert_allclose(translation.problem.kpoints, expected, atol=1.0e-14)
                np.testing.assert_array_equal(translation.problem.kpoint_weights, reference.kpoint_weights)
                self.assertFalse(translation.problem.kpoint_tie_images)
                self.assertTrue(any("orthogonal cell" in note for note in translation.warnings))
        off = self.parse(text, {"PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD": "off"}).problem
        np.testing.assert_array_equal(off.kpoints, reference.kpoints)
        # a list inside the first zone keeps its bits
        inside = _listed(_text(cell="cube", **_SCALAR), "4 4 4", np.where(written > 0.5 + 1.0e-12, written - 1.0, written))
        kept = self.parse(inside, {"PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD": "off"}).problem
        np.testing.assert_array_equal(self.parse(inside).problem.kpoints, kept.kpoints)
        with self.assertRaisesRegex(Exception, "PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD"):
            self.parse(text, {"PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD": "on"})
        # Fortran-derived routines the run chose itself (a grid of 6 points per axis at order 6) fold too; requested
        # ones keep the written vectors
        coarse = text.replace("Grid_Spacing: 0.31", "Grid_Spacing: 0.8")
        chosen = self.parse(coarse)
        self.assertTrue(chosen.routines.fortran)
        self.assertIsNot(chosen.routines.requested, True)
        np.testing.assert_allclose(chosen.problem.kpoints, expected, atol=1.0e-14)
        requested = self.parse(coarse.replace("Correlation_Type: CA", "Correlation_Type: CA\nUse_Fortran_Routines: .true."))
        self.assertIs(requested.routines.requested, True)
        np.testing.assert_allclose(requested.problem.kpoints, written @ reciprocal, atol=1.0e-14)

    def test_unrequested_fortran_routines_keep_a_list_without_a_level_below_zero(self) -> None:
        written = _mesh((4, 4, 4))
        text = _listed(_text(cell="fcc", spacing=0.6, **_SCALAR), "4 4 4", written)
        translation = self.parse(text.replace("Expansion_Order: 6", "Expansion_Order: 12"))
        self.assertTrue(translation.routines.fortran)
        self.assertIsNot(translation.routines.requested, True)
        reciprocal = reciprocal_lattice(translation.problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(translation.problem.kpoints, written @ reciprocal, atol=1.0e-14)
        self.assertFalse(translation.problem.kpoint_tie_images)
        self.assertEqual(self.notes(translation), [])

    def test_unrequested_fortran_routines_move_a_list_with_a_level_below_zero(self) -> None:
        written = _mesh((6, 6, 6))
        translation = self.parse(_listed(_bcc(), "6 6 6", written))
        problem = translation.problem
        self.assertTrue(translation.routines.fortran)
        self.assertIsNot(translation.routines.requested, True)
        self.assertTrue(problem.kpoint_tie_images)
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(
            np.sum(problem.kpoints**2, axis=1), _brute_force(written, reciprocal)[0], rtol=1.0e-9, atol=1.0e-12
        )
        k_points, weights = kpoint_sampling(problem)
        self.assertGreater(len(k_points), len(written))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
        self.assertFalse(np.any(lowest_free_levels(k_points, problem.periodic_cell, problem.grid) < -1.0e-8))
        self.assertIn("were not requested", self.notes(translation)[0])
        self.assertIn("|k| = ", self.notes(translation)[0])

    def test_requested_fortran_routines_keep_the_vectors_and_warn(self) -> None:
        written = _mesh((6, 6, 6))
        text = _listed(_bcc("Use_Fortran_Routines: .true."), "6 6 6", written)
        translation = self.parse(text)
        problem = translation.problem
        self.assertTrue(translation.routines.fortran)
        self.assertIs(translation.routines.requested, True)
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(problem.kpoints, written @ reciprocal, atol=1.0e-14)
        self.assertFalse(problem.kpoint_tie_images)
        note = self.notes(translation)[0]
        for piece in ("|k| = ", "Use_Fortran_Routines: .true.", "PARSEC_KPOINT_LIST_FOLD=shared"):
            self.assertIn(piece, note)
        # shared takes the shortest images there too
        moved = self.parse(text, {"PARSEC_KPOINT_LIST_FOLD": "shared"}).problem
        self.assertTrue(moved.kpoint_tie_images)
        np.testing.assert_allclose(
            np.sum(moved.kpoints**2, axis=1), _brute_force(written, reciprocal)[0], rtol=1.0e-9, atol=1.0e-12
        )

    def test_literal_warns_on_a_level_below_zero_and_keeps_the_vectors(self) -> None:
        written = _mesh((6, 6, 6))
        translation = self.parse(_listed(_bcc(), "6 6 6", written), {"PARSEC_KPOINT_LIST_FOLD": "literal"})
        reciprocal = reciprocal_lattice(translation.problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(translation.problem.kpoints, written @ reciprocal, atol=1.0e-14)
        self.assertFalse(translation.problem.kpoint_tie_images)
        self.assertIn("PARSEC_KPOINT_LIST_FOLD=literal", self.notes(translation)[0])

    def test_a_weighted_list_shares_its_weights_among_the_images(self) -> None:
        written = _mesh((2, 2, 2))
        weights = np.arange(1.0, 9.0)
        text = _listed(_text(cell="fcc", mesh="2 2 2", extra="Use_Fortran_Routines: .false.", so_psp=".false."),
                       "2 2 2", written, weights)
        problem = self.parse(text).problem
        self.assertTrue(problem.kpoint_tie_images)
        k_points, shares = kpoint_sampling(problem)
        self.assertGreater(len(k_points), 8)
        self.assertAlmostEqual(float(np.sum(shares)), 1.0, places=14)
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        classes = np.rint((k_points @ np.linalg.inv(reciprocal)) * 2.0).astype(int) % 2
        for row, weight in zip(written, weights / weights.sum()):
            key = np.rint(row * 2.0).astype(int) % 2
            self.assertAlmostEqual(float(np.sum(shares[np.all(classes == key, axis=1)])), float(weight), places=12)

    def test_a_hybrid_list_takes_the_images_of_its_mesh(self) -> None:
        written = _mesh((4, 4, 2))
        text = _listed(_text(cell="hcp", xc="PBE0", **_SCALAR), "4 4 2", written)
        problem = self.parse(text).problem
        self.assertTrue(problem.scf.is_hybrid)
        self.assertFalse(problem.kpoint_tie_images)
        reciprocal = reciprocal_lattice(problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(
            problem.kpoints, negation_closed_shortest(written, reciprocal) @ reciprocal, atol=1.0e-12
        )

    def test_an_ordinary_mesh_does_not_warn(self) -> None:
        for cell in ("fcc", "hcp", "cube"):
            with self.subTest(cell=cell):
                translation = self.parse(_text(cell=cell, **_SCALAR))
                self.assertEqual([n for n in translation.warnings if n.startswith("Monkhorst_Pack_Grid: k = ")], [])

    def test_an_unknown_value_stops_every_input(self) -> None:
        for text in (_text(cell="cube", **_SCALAR), _text(cell="hcp", **_SCALAR)):
            with self.assertRaisesRegex(Exception, "PARSEC_KPOINT_LIST_FOLD"):
                self.parse(text, {"PARSEC_KPOINT_LIST_FOLD": "shortest"})


class MeshGuardTests(_Directory):
    def notes(self, translation):
        return [line for line in translation.warnings if line.startswith("Monkhorst_Pack_Grid: k = ")]

    def test_unrequested_fortran_routines_take_the_shortest_images(self) -> None:
        translation = self.parse(_bcc())
        problem = translation.problem
        self.assertTrue(translation.routines.fortran)
        self.assertIsNot(translation.routines.requested, True)
        self.assertTrue(problem.kpoint_shortest_representatives)
        self.assertTrue(problem.kpoint_tie_images)
        self.assertFalse(problem.kpoint_written_ties)
        k_points, weights = kpoint_sampling(problem)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
        self.assertFalse(np.any(lowest_free_levels(k_points, problem.periodic_cell, problem.grid) < -1.0e-8))
        self.assertIn("were not requested", self.notes(translation)[0])
        self.assertIn("PARSEC_KPOINT_MESH_GUARD=off", self.notes(translation)[0])

    def test_off_restores_the_mesh(self) -> None:
        off = self.parse(_bcc(), {"PARSEC_KPOINT_MESH_GUARD": "off"})
        self.assertFalse(off.problem.kpoint_shortest_representatives)
        self.assertFalse(off.problem.kpoint_tie_images)
        self.assertIn("The mesh keeps its points (PARSEC_KPOINT_MESH_GUARD=off)", self.notes(off)[0])
        requested = self.parse(_bcc("Use_Fortran_Routines: .true."))
        np.testing.assert_array_equal(kpoint_sampling(off.problem)[0], kpoint_sampling(requested.problem)[0])
        np.testing.assert_array_equal(kpoint_sampling(off.problem)[1], kpoint_sampling(requested.problem)[1])

    def test_requested_fortran_routines_keep_the_mesh(self) -> None:
        for value in ("auto", "off"):
            with self.subTest(value=value):
                translation = self.parse(_bcc("Use_Fortran_Routines: .true."), {"PARSEC_KPOINT_MESH_GUARD": value})
                self.assertIs(translation.routines.requested, True)
                self.assertFalse(translation.problem.kpoint_shortest_representatives)
                self.assertFalse(translation.problem.kpoint_tie_images)
                self.assertIn("The mesh keeps its points (Use_Fortran_Routines: .true.)", self.notes(translation)[0])

    def test_a_hybrid_takes_one_image(self) -> None:
        text = _bcc().replace("Correlation_Type: CA", "Correlation_Type: PBE0")
        problem = self.parse(text).problem
        self.assertTrue(problem.scf.is_hybrid)
        self.assertTrue(problem.kpoint_shortest_representatives)
        self.assertFalse(problem.kpoint_tie_images)

    def test_dos_and_gw_keep_the_mesh(self) -> None:
        translation = self.parse(_bcc("Create_DOS: .true."))
        self.assertFalse(translation.problem.kpoint_shortest_representatives)
        self.assertFalse(translation.problem.kpoint_tie_images)
        self.assertIn("The mesh keeps its points", self.notes(translation)[0])

    def test_a_mesh_without_such_a_level_is_untouched(self) -> None:
        text = _text(cell="fcc", spacing=0.6, **_SCALAR).replace("Expansion_Order: 6", "Expansion_Order: 12")
        translation = self.parse(text)
        off = self.parse(text, {"PARSEC_KPOINT_MESH_GUARD": "off"})
        self.assertTrue(translation.routines.fortran)
        self.assertFalse(translation.problem.kpoint_shortest_representatives)
        self.assertEqual(self.notes(translation), [])
        np.testing.assert_array_equal(kpoint_sampling(translation.problem)[0], kpoint_sampling(off.problem)[0])

    def test_a_restart_reads_the_guard_again(self) -> None:
        translation = self.parse(_bcc())
        positions = np.asarray([atom.position for atom in translation.problem.atoms], dtype=float)
        fresh = read_symmetry_of_structure(translation, positions)
        self.assertTrue(fresh.problem.kpoint_shortest_representatives)
        self.assertEqual(symmetry_signature(fresh), symmetry_signature(translation))

    def test_an_unknown_value_stops_every_input(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_MESH_GUARD": "on"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_MESH_GUARD"):
                kpoint_mesh_guard_setting()
        with self.assertRaisesRegex(Exception, "PARSEC_KPOINT_MESH_GUARD"):
            self.parse(_text(cell="cube", **_SCALAR), {"PARSEC_KPOINT_MESH_GUARD": "on"})

    def test_the_wedge_keeps_the_operations_of_the_stencil(self) -> None:
        """PARSEC_KPOINT_GUARD_WEDGE=auto: the guarded mesh of a primitive bcc cell on Fortran's general grid (axes and
        the fourth body diagonal, first differences along the axes) is reduced by the 12 of 48 operations that keep the
        grid's operators, the forces averaged over all 48; crystal keeps the 48 operations, as before."""

        from parsec_python.Grid.kpoints import grid_stencil_lines, stencil_preserving_operations

        auto = self.parse(_bcc(mesh="4 4 4"))
        # a restart reads the same subgroup again (before another input replaces the file it reads)
        positions = np.asarray([atom.position for atom in auto.problem.atoms], dtype=float)
        fresh = read_symmetry_of_structure(auto, positions)
        self.assertEqual(symmetry_signature(fresh), symmetry_signature(auto))
        crystal = self.parse(_bcc(mesh="4 4 4"), {"PARSEC_KPOINT_GUARD_WEDGE": "crystal"})
        whole = crystal.problem.space_group
        self.assertEqual(whole.order, 48)
        self.assertIsNone(crystal.problem.force_space_group)
        problem = auto.problem
        self.assertEqual(problem.space_group.order, 12)
        self.assertEqual(problem.force_space_group.order, 48)
        self.assertTrue(problem.scf.symmetrize_density)
        counts, lines = grid_stencil_lines(problem.periodic_cell, problem.grid)
        kept = stencil_preserving_operations(whole, problem.periodic_cell.lattice_vectors, counts, lines=lines)
        self.assertEqual(sorted(map(str, kept.rotations)), sorted(map(str, problem.space_group.rotations)))
        self.assertTrue(any("reduced by 12 of the 48 operations" in note for note in auto.warnings))
        self.assertFalse(any("moved mesh" in note for note in crystal.warnings))
        # the subgroup's wedge holds every point of the mesh at its shortest images: more stars, the same total weight
        k_auto, w_auto = kpoint_sampling(problem)
        k_crystal, w_crystal = kpoint_sampling(crystal.problem)
        self.assertGreater(len(k_auto), len(k_crystal))
        self.assertAlmostEqual(float(np.sum(w_auto)), 1.0, places=13)
        self.assertFalse(np.any(lowest_free_levels(k_auto, problem.periodic_cell, problem.grid) < -1.0e-8))
        # a hybrid keeps its k-points (the time-reversal pairs here, the whole mesh) and does not average their density
        hybrid_text = _bcc(mesh="4 4 4").replace("Correlation_Type: CA", "Correlation_Type: PBE0")
        hybrid = self.parse(hybrid_text).problem
        hybrid_whole = self.parse(hybrid_text, {"PARSEC_KPOINT_GUARD_WEDGE": "crystal"}).problem
        self.assertFalse(hybrid.scf.symmetrize_density)
        self.assertTrue(hybrid_whole.scf.symmetrize_density)
        self.assertEqual(hybrid.space_group.order, 12)
        self.assertEqual(hybrid.force_space_group.order, 48)
        self.assertEqual(hybrid_whole.space_group.order, 48)
        self.assertIsNone(hybrid_whole.force_space_group)
        np.testing.assert_array_equal(kpoint_sampling(hybrid)[0], kpoint_sampling(hybrid_whole)[0])
        # the guard of PARSEC_KPOINT_MESH_GUARD=off and a requested routine set keep the whole group
        for text, environment in (
            (_bcc(mesh="4 4 4"), {"PARSEC_KPOINT_MESH_GUARD": "off"}),
            (_bcc("Use_Fortran_Routines: .true.", mesh="4 4 4"), {}),
        ):
            with self.subTest(text=text[:40], environment=environment):
                found = self.parse(text, environment).problem
                self.assertIsNone(found.force_space_group)
                if found.space_group is not None:
                    self.assertEqual(found.space_group.order, 48)

    def test_an_unknown_wedge_value_stops_every_input(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_guard_wedge_setting

        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_GUARD_WEDGE": "off"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_GUARD_WEDGE"):
                kpoint_guard_wedge_setting()
        with self.assertRaisesRegex(Exception, "PARSEC_KPOINT_GUARD_WEDGE"):
            self.parse(_text(cell="cube", **_SCALAR), {"PARSEC_KPOINT_GUARD_WEDGE": "off"})


class WrittenTieTests(unittest.TestCase):
    def test_the_ties_survive_the_rounding_of_the_input(self) -> None:
        for name in ("fcc", "bcc", "hexagonal"):
            lattice = _lattices()[name]
            reciprocal = reciprocal_lattice(lattice)
            for dimensions in ((6, 6, 6), (4, 4, 2)):
                exact = [group.shape[0] for group in shortest_images(_mesh(dimensions), reciprocal)]
                for digits in (16, 8, 6, 4):
                    with self.subTest(lattice=name, mesh=dimensions, digits=digits):
                        rounded = np.round(_mesh(dimensions), digits)
                        found = [group.shape[0] for group in written_tie_images(rounded, reciprocal)]
                        self.assertEqual(found, exact)

    def test_points_off_the_boundary_are_not_split(self) -> None:
        for name, lattice in _lattices().items():
            reciprocal = reciprocal_lattice(lattice)
            for n in (5, 7, 9, 11, 13, 24):
                with self.subTest(lattice=name, n=n):
                    points = _mesh((n, n, n))
                    exact = [group.shape[0] for group in shortest_images(points, reciprocal)]
                    found = [group.shape[0] for group in written_tie_images(points, reciprocal)]
                    self.assertEqual(found, exact)

    def test_a_long_cell_does_not_join_its_close_planes(self) -> None:
        a, c = 5.0, 400.0
        lattice = np.array([[a, 0.0, 0.0], [-a / 2, a * 3**0.5 / 2, 0.0], [0.0, 0.0, c]])
        reciprocal = reciprocal_lattice(lattice)
        near = np.array([[0.0, 0.0, 0.49], [0.1, 0.05, 0.4975]])
        self.assertEqual([g.shape[0] for g in written_tie_images(near, reciprocal)], [1, 1])
        # a test against the largest |b|^2 would call them ties
        metric = reciprocal @ reciprocal.T
        step = np.array([0.0, 0.0, 1.0])
        excess = float(step @ metric @ step) - 2.0 * float(near[0] @ metric @ step)
        self.assertLess(excess, 1.0e-3 * float(np.max(np.diag(metric))))
        on_the_plane = np.array([[0.1, 0.05, 0.5]])
        self.assertEqual(written_tie_images(on_the_plane, reciprocal)[0].shape[0], 2)

    def test_weights_are_conserved(self) -> None:
        lattice = _lattices()["fcc"]
        reciprocal = reciprocal_lattice(lattice)
        rounded = np.round(_mesh((4, 4, 4)), 6) @ reciprocal
        weights = np.full(len(rounded), 1.0 / len(rounded))
        k_points, shares = tie_shared_points(rounded, weights, lattice, written=True)
        self.assertGreater(len(k_points), len(rounded))
        self.assertAlmostEqual(float(np.sum(shares)), 1.0, places=14)


class HybridOrderTests(unittest.TestCase):
    def test_the_images_do_not_depend_on_the_order_or_the_round_off(self) -> None:
        generator = np.random.default_rng(5)
        for name, dimensions in (("fcc", (4, 4, 4)), ("hexagonal", (4, 4, 2)), ("bcc", (6, 6, 6))):
            lattice = _lattices()[name]
            reciprocal = reciprocal_lattice(lattice)
            written = _mesh(dimensions) @ reciprocal
            base = negation_closed_written(written, lattice)
            # the images of the mesh itself
            np.testing.assert_allclose(base, negation_closed_shortest(_mesh(dimensions), reciprocal) @ reciprocal,
                                       atol=1.0e-12)
            for label, order in (("reversed", np.arange(len(written))[::-1]),
                                 ("shuffled", generator.permutation(len(written)))):
                with self.subTest(lattice=name, order=label):
                    moved = negation_closed_written(written[order], lattice)
                    self.assertTrue(_same_rows(moved, base, atol=1.0e-12))
                    # the time-reversal pairs keep the same members whatever the order of the lines
                    equal = np.full(len(moved), 1.0 / len(moved))
                    self.assertTrue(
                        _same_rows(
                            time_reversal_reduced(moved, equal, lattice)[0],
                            time_reversal_reduced(base, equal, lattice)[0],
                            atol=1.0e-12,
                        )
                    )
            with self.subTest(lattice=name, order="round-off"):
                noise = generator.uniform(-2.0e-15, 2.0e-15, size=written.shape)
                np.testing.assert_allclose(negation_closed_written(written + noise, lattice), base, atol=1.0e-12)
            with self.subTest(lattice=name, order="other images"):
                other = (fold_to_centered_zone(_mesh(dimensions)) - 1.0) @ reciprocal
                np.testing.assert_allclose(negation_closed_written(other, lattice), base, atol=1.0e-12)

    def test_the_set_is_closed_under_negation(self) -> None:
        lattice = _lattices()["hexagonal"]
        reciprocal = reciprocal_lattice(lattice)
        points = negation_closed_written(_mesh((4, 4, 2))[::-1] @ reciprocal, lattice) @ np.linalg.inv(reciprocal)
        for row in points:
            twice = 2.0 * row
            own = np.max(np.abs(twice - np.rint(twice))) < 1.0e-9
            self.assertTrue(own or float(np.min(np.max(np.abs(points + row), axis=1))) < 1.0e-9)


class PathTests(_Directory):
    def test_a_band_path_warns_and_keeps_its_points(self) -> None:
        extra = "begin bandstruc\n1 0.0 0.0 0.0 0.8333333333 0.8333333333 0.8333333333 G-far\nend bandstruc"
        translation = self.parse(_bcc(extra))
        notes = [line for line in translation.warnings if line.startswith("bandstruc: k = ")]
        self.assertEqual(len(notes), 1)
        self.assertIn("|k| = ", notes[0])
        self.assertEqual(translation.problem.bands.segments[0].end, (0.8333333333, 0.8333333333, 0.8333333333))

    def test_nscf_points_warn_and_stay(self) -> None:
        extra = (
            "Restart_Run: .true.\nnscf_kpoint_unit: reciprocal_lattice_vectors\nnscf_states_num: 4\n"
            "begin nscf_kpoints\n0.8333333333 0.8333333333 0.8333333333 1.0\nend nscf_kpoints"
        )
        translation = self.parse(_bcc(extra))
        notes = [line for line in translation.warnings if line.startswith("nscf_kpoints: k = ")]
        self.assertEqual(len(notes), 1)
        reciprocal = reciprocal_lattice(translation.problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(translation.nscf.k_points @ np.linalg.inv(reciprocal), [[0.8333333333] * 3],
                                   atol=1.0e-12)

    def test_a_short_path_says_nothing(self) -> None:
        extra = "begin bandstruc\n1 0.0 0.0 0.0 0.1 0.1 0.1 G-near\nend bandstruc"
        translation = self.parse(_bcc(extra, mesh="2 2 2"))
        self.assertEqual([line for line in translation.warnings if line.startswith("bandstruc: k = ")], [])


class EdgeTests(_Directory):
    def test_shared_with_a_restarted_dos_of_its_own_mesh(self) -> None:
        extra = (
            "Create_DOS: .true.\nRestart_Run: .true.\nbegin DOS_MP_Grid\n4 4 4\nend DOS_MP_Grid"
        )
        text = _listed(_text(cell="fcc", extra=extra, so_psp=".false."), "4 4 4", _mesh((4, 4, 4)))
        for value in ("auto", "shared", "literal"):
            with self.subTest(value=value):
                problem = self.parse(text, {"PARSEC_KPOINT_LIST_FOLD": value}).problem
                self.assertIsNone(problem.kpoints)
                self.assertEqual(tuple(problem.monkhorst_pack_dimensions), (4, 4, 4))

    def test_every_wave_of_a_large_grid_is_visited(self) -> None:
        bcc = np.array([[-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, -4.0]])
        reciprocal = reciprocal_lattice(bcc)
        far = np.array([[17.3, 16.9, 17.1], [23.2, 22.8, 23.05]]) @ reciprocal
        cell, settings, exact = _exact_levels(bcc, 0.1, 12, far)
        self.assertGreater(int(np.max(np.asarray(build_periodic_grid(cell, settings).points_per_axis))), 64)
        levels = lowest_free_levels(far, cell, settings)
        self.assertTrue(np.all(exact < -1.0))
        np.testing.assert_allclose(levels, exact, rtol=1.0e-10, atol=1.0e-8)

    def test_a_first_zone_point_of_a_very_coarse_grid_warns(self) -> None:
        half = 2.75
        cell_text = f"begin Cell_Shape\n{-half} {half} {half}\n{half} {-half} {half}\n{half} {half} {-half}\nend Cell_Shape"
        base = _bcc(mesh="4 4 4", spacing=0.794).replace(_BCC.split("\nLattice")[0], cell_text)
        reciprocal = reciprocal_lattice(np.array([[-half, half, half], [half, -half, half], [half, half, -half]]))
        written = fold_to_shortest(_mesh((4, 4, 4)), reciprocal)
        translation = self.parse(_listed(base, "4 4 4", written))
        notes = [line for line in translation.warnings if line.startswith(_NOTE)]
        self.assertTrue(any("too coarse" in line for line in notes), notes)
        points, _ = kpoint_sampling(translation.problem)
        problem = translation.problem
        self.assertTrue(np.any(lowest_free_levels(points, problem.periodic_cell, problem.grid) < -FREE_LEVEL_TOLERANCE))


class RunTests(_Directory):
    """A coarse hcp Au2 cell on the CPU."""

    def energy(self, text: str, environment=None) -> float:
        from parsec_python.cli import main as cli_main
        from parsec_python.tests.archives import read_archive

        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        with mock.patch.dict(os.environ, environment or {}):
            code = cli_main([str(path), "--quiet"])
        self.assertEqual(code, 0)
        archive = read_archive(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        return float(archive["energy_total_ry"])

    def test_two_writings_of_one_list_give_one_energy(self) -> None:
        base = _text(cell="hcp", mesh="3 3 1", spacing=0.45, extra="Use_Fortran_Routines: .false.", so_psp=".false.")
        base = base.replace("States_Num: 32", "States_Num: 20")
        lattice = self.parse(base).problem.periodic_cell.lattice_vectors
        reciprocal = reciprocal_lattice(lattice)
        written = _mesh((3, 3, 1))
        shortest = fold_to_shortest(written, reciprocal)
        self.assertFalse(np.allclose(written, shortest))
        reference = self.energy(_listed(base, "3 3 1", shortest))
        self.assertAlmostEqual(self.energy(_listed(base, "3 3 1", written)), reference, delta=1.0e-9)
        literal = self.energy(_listed(base, "3 3 1", written), {"PARSEC_KPOINT_LIST_FOLD": "literal"})
        self.assertGreater(abs(literal - reference), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
