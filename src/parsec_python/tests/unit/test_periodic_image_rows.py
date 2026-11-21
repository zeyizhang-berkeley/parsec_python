"""The images of a periodic density group without their table (``PARSEC_PERIODIC_IMAGE_ROWS``, 2026-11-21).

:func:`parsec_python.Grid.space_group.periodic_grid_symmetry` keeps no table of operations x points where every
operation is an integer affine map of a row-major grid (:class:`~parsec_python.Grid.space_group.AffineImageRows`):
its rows, entries, gathers and the averages of :mod:`parsec_python.Grid.point_group` (scalar and vector fields, the
orbital density gradient of a periodic meta-GGA) are those of the table bit for bit, on cubic cells and on an oblique
cell whose matrices mix the axes, also by slabs on threads; ``off`` builds the table as before.
"""

from __future__ import annotations

import os
import pickle
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid import space_group
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Grid.point_group import gathered_images, symmetrize_density, symmetrize_vector
from parsec_python.Grid.space_group import (
    AffineImageRows,
    find_grid_space_group,
    find_space_group,
    periodic_grid_symmetry,
    periodic_image_rows_setting,
)
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings
from parsec_python.SCF.meta_gga_periodic import symmetrized_gradient

_A = 3.567 / 0.529177210903
_BASIS = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]


def _diamond():
    atoms = [
        Atom("C", [round((b + shift) * _A, 10) for b in basis]) for shift in (0.0, 0.25) for basis in _BASIS
    ]
    return np.diag([round(_A, 10)] * 3), atoms


def _cases():
    """``(label, lattice, grid, group)``: the diamond cube on the benchmark's shifted grid (Td about an atom, with the
    face-centring translations), the cube of the Fortran-derived routines with its whole space group (48), a
    tetragonal box of unequal counts, and diamond in the primitive fcc cell (matrices that mix the axes)."""

    lattice, atoms = _diamond()
    cases = []
    settings = PeriodicGridSettings(
        spacing=_A / 18 * (1.0 - 1.0e-9), expansion_order=8, shift=(0.5, 0.5, 0.5), points_rule="fortran"
    )
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)
    cases.append(("cube 18", lattice, grid, find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)))
    group = find_space_group(lattice, atoms)
    settings = PeriodicGridSettings(
        fortran_routines=True, spacing=_A / 16, expansion_order=6, shift=(0.5, 0.5, 0.5),
        fractional_translations=group.commensurate_translations(),
    )
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)
    cases.append(("fortran cube", lattice, grid, group))
    tetragonal = np.diag([round(_A, 10), round(_A, 10), round(1.5 * _A, 10)])
    rods = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", (0.5 * _A, 0.5 * _A, 0.75 * _A))]
    settings = PeriodicGridSettings(spacing=_A / 12 * (1.0 - 1.0e-9), expansion_order=4, shift=(0.0, 0.0, 0.0),
                                    points_rule="fortran")
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=tetragonal), settings)
    cases.append(("tetragonal", tetragonal, grid, find_grid_space_group(tetragonal, rods, grid.points_per_axis,
                                                                         grid.shift)))
    fcc = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * _A
    pair = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", tuple(np.full(3, 0.25 * _A)))]
    settings = PeriodicGridSettings(spacing=0.4, expansion_order=4, shift=(0.0, 0.0, 0.0), points_rule="fortran")
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=fcc), settings)
    cases.append(("oblique fcc", fcc, grid, find_grid_space_group(fcc, pair, grid.points_per_axis, grid.shift)))
    return cases


def _both(grid, group):
    """The group without its table (``on``: at any size and thread count) and with it (``off``)."""

    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "on"}):
        rows = periodic_grid_symmetry(grid, group)
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "off"}):
        table = periodic_grid_symmetry(grid, group)
    return rows, table


class ImageRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = _cases()

    def test_a_small_grid_keeps_its_table(self) -> None:
        # below 2^17 points the gathers of the table are faster: auto keeps it, the route of off
        label, _, grid, group = self.cases[0]
        self.assertLess(grid.size, space_group._IMAGE_ROWS_POINTS)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "auto"}):
            kept = periodic_grid_symmetry(grid, group)
        self.assertTrue(kept.holds_table)
        self.assertIsNotNone(kept.affine_maps)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "auto", "PARSEC_PERIODIC_SETUP_THREADS": "2"}), \
                mock.patch.object(space_group, "_IMAGE_ROWS_POINTS", grid.size):
            self.assertFalse(periodic_grid_symmetry(grid, group).holds_table)
        # on one thread the gathers of the table are faster: auto keeps it there, on takes the rows all the same
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "auto", "PARSEC_PERIODIC_SETUP_THREADS": "1"}), \
                mock.patch.object(space_group, "_IMAGE_ROWS_POINTS", grid.size):
            self.assertTrue(periodic_grid_symmetry(grid, group).holds_table)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "on", "PARSEC_PERIODIC_SETUP_THREADS": "1"}):
            self.assertFalse(periodic_grid_symmetry(grid, group).holds_table)

    def test_the_switch(self) -> None:
        for raw, expected in (("", "auto"), ("auto", "auto"), (" On", "on"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": raw}):
                self.assertEqual(periodic_image_rows_setting(), expected)
        label, _, grid, group = self.cases[0]
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": "table"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_IMAGE_ROWS must be auto, on or off"):
                periodic_grid_symmetry(grid, group)

    def test_what_the_rows_keep_is_less_than_the_table(self) -> None:
        # the broadcast index vectors of a box are kilobytes; an oblique operation (a row of M that reads all three
        # axes, the mirrors of the primitive fcc cell) keeps the row of the table (int32), not an int64 index array
        # of every point: never more than the table
        for label, _, grid, group in self.cases:
            with self.subTest(case=label):
                rows, table = _both(grid, group)
                self.assertEqual(rows.held_bytes, 0)
                self.assertEqual(table.held_bytes, table.table_bytes)
                symmetrize_density(np.ones(grid.size), rows, grid.volume_element)
                symmetrize_vector(np.ones((3, grid.size)), rows)
                oblique = [operation for operation in range(group.order) if rows.images.oblique(operation)]
                matrices = [matrix for matrix, _ in rows.affine_maps]
                self.assertEqual(
                    oblique, [g for g, matrix in enumerate(matrices) if (np.count_nonzero(matrix, axis=1) == 3).any()]
                )
                self.assertLessEqual(rows.held_bytes, table.table_bytes)
                vectors = rows.held_bytes - len(oblique) * grid.size * 4
                self.assertLess(vectors, table.table_bytes // 20)
                if label == "oblique fcc":
                    self.assertEqual((group.order, len(oblique)), (24, 18))  # (the grid's group of this case)
                    self.assertLess(rows.held_bytes, table.table_bytes)
                else:
                    self.assertEqual(oblique, [])
                    self.assertLess(rows.held_bytes, table.table_bytes // 20)
                for operation in oblique:
                    np.testing.assert_array_equal(rows.image_row(operation), table.images[operation])
                    self.assertFalse(rows.image_row(operation).flags.writeable)

    def test_rows_entries_and_table_are_those_of_the_table(self) -> None:
        for label, _, grid, group in self.cases:
            with self.subTest(case=label, order=group.order):
                rows, table = _both(grid, group)
                self.assertIsInstance(rows.images, AffineImageRows)
                self.assertFalse(rows.holds_table)
                self.assertIsInstance(table.images, np.ndarray)
                self.assertTrue(table.holds_table)
                self.assertEqual(rows.images.shape, table.images.shape)
                self.assertEqual(rows.images.dtype, table.images.dtype)
                self.assertEqual(rows.table_bytes, table.images.nbytes)
                self.assertEqual(rows.size, grid.size)
                self.assertEqual(len(rows.images), group.order)
                np.testing.assert_array_equal(np.asarray(rows.images), table.images)
                sample = np.random.default_rng(5).integers(0, grid.size, size=97)
                for operation in range(group.order):
                    np.testing.assert_array_equal(rows.image_row(operation), table.images[operation])
                    np.testing.assert_array_equal(rows.image_rows(operation, sample), table.images[operation][sample])
                    np.testing.assert_array_equal(rows.images[operation, sample], table.images[operation, sample])
                np.testing.assert_array_equal(rows.images[:, sample], table.images[:, sample])
                np.testing.assert_array_equal(rows.images[-1], table.images[-1])
                for got, expected in zip(rows.images, table.images):
                    np.testing.assert_array_equal(got, expected)
                self.assertEqual(rows.affine_maps is not None, True)

    def test_averages_are_those_of_the_table_bit_for_bit(self) -> None:
        draw = np.random.default_rng(11)
        for label, lattice, grid, group in self.cases:
            with self.subTest(case=label):
                rows, table = _both(grid, group)
                density = draw.random(grid.size)
                ours, our_report = symmetrize_density(density, rows, grid.volume_element)
                theirs, their_report = symmetrize_density(density, table, grid.volume_element)
                np.testing.assert_array_equal(ours, theirs)
                self.assertEqual(our_report, their_report)
                vectors = draw.standard_normal((3, grid.size))
                np.testing.assert_array_equal(symmetrize_vector(vectors, rows), symmetrize_vector(vectors, table))
                for operation in (0, group.order - 1):
                    gathered = gathered_images(vectors, rows, operation)
                    expected = vectors[:, table.images[operation]]
                    np.testing.assert_array_equal(gathered, expected)
                    self.assertEqual(gathered.strides, expected.strides)
                if np.allclose(lattice, np.diag(np.diag(lattice))):
                    # (the meta-GGA gradient average takes Cartesian rotations: the boxes)
                    np.testing.assert_array_equal(
                        symmetrized_gradient(vectors, rows, lattice), symmetrized_gradient(vectors, table, lattice)
                    )
                total = np.zeros(grid.size)
                for operation in range(group.order):
                    rows.accumulate(total, density, operation)
                np.testing.assert_array_equal(total / float(group.order), theirs)

    def test_slabs_on_threads_add_the_same(self) -> None:
        draw = np.random.default_rng(2)
        for label, _, grid, group in (self.cases[1], self.cases[3]):
            rows, table = _both(grid, group)
            density = draw.random(grid.size)
            vectors = draw.standard_normal((3, grid.size))
            stacked = draw.standard_normal((2, 3, grid.size))
            expected, _ = symmetrize_density(density, table, grid.volume_element)
            expected_vectors = symmetrize_vector(vectors, table)
            for threads in ("1", "3", "8"):
                with self.subTest(case=label, threads=threads), mock.patch.object(space_group, "_THREADED_SUMS", 64), \
                        mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}):
                    ours, _ = symmetrize_density(density, rows, grid.volume_element)
                    np.testing.assert_array_equal(ours, expected)
                    np.testing.assert_array_equal(symmetrize_vector(vectors, rows), expected_vectors)
                    for operation in (1, group.order - 2):
                        for field in (density, vectors, stacked, np.asfortranarray(vectors)):
                            gathered = rows.gather(field, operation)
                            reference = field[..., table.images[operation]]
                            np.testing.assert_array_equal(gathered, reference)
                            self.assertEqual(gathered.strides, reference.strides)

    def test_a_slab_of_the_kept_pool_runs_its_own_slabs_inline(self) -> None:
        # the threads of the slabs are kept between calls; a call from one of them must not wait on the pool
        seen = []

        def inner(first, last):
            seen.append((first, last))

        def outer(first, last):
            space_group._run_by_slabs(inner, 4)

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": "2"}):
            space_group._run_by_slabs(outer, 2)
            pool = space_group._slab_pool(2)
            self.assertIs(space_group._slab_pool(2), pool)
        self.assertEqual(sorted(seen), [(0, 4), (0, 4)])

    def test_the_rows_pickle_and_the_row_major_test_by_blocks(self) -> None:
        label, _, grid, group = self.cases[0]
        rows, table = _both(grid, group)
        again = pickle.loads(pickle.dumps(rows))
        np.testing.assert_array_equal(np.asarray(again.images), table.images)
        points, counts, _ = space_group.grid_fractional_points(grid)
        with mock.patch.object(space_group, "_ROW_MAJOR_BLOCK", 7):
            self.assertTrue(space_group._row_major_points(grid, points, counts))
            moved = points.copy()
            moved[grid.size - 1, 2] += 1
            self.assertFalse(space_group._row_major_points(grid, moved, counts))
        with self.assertRaises(IndexError):
            rows.images.at(0, np.array([grid.size]))
        with self.assertRaises(IndexError):
            rows.images.row(group.order)


if __name__ == "__main__":
    unittest.main()
