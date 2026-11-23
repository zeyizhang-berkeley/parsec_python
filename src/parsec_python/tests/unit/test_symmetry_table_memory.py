"""Density groups and k-point stars without their tables of images (2026-11-22).

* A slab whose points fill a box takes the rows formed where they are read (``PARSEC_PERIODIC_IMAGE_ROWS``,
  :func:`parsec_python.Grid.space_group.confined_grid_symmetry`): the same integers as its table, the same sums; a
  wire keeps its table.
* ``PARSEC_PERIODIC_IMAGE_ROWS=on`` keeps no row of an oblique operation either (the primitive fcc cell): its gathers
  form the indices of a few planes at a time (``_ObliqueFactor``), the same integers and values.
* The gathers of the stars of a hybrid's k-point wedge (``V_xc.bloch_exchange.MeshStars``) are int32, the inverse
  permutations of the images as before.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Grid.space_group import (
    AffineImageRows,
    confined_grid_symmetry,
    find_grid_space_group,
    find_wire_slab_space_group,
    periodic_grid_symmetry,
)


def _with_rows(setting: str):
    return mock.patch.dict(os.environ, {"PARSEC_PERIODIC_IMAGE_ROWS": setting})


def _same_symmetry(test: unittest.TestCase, table, rows, points: int) -> None:
    np.testing.assert_array_equal(np.asarray(table.images), np.asarray(rows.images))
    generator = np.random.default_rng(7)
    field = generator.standard_normal(points)
    np.testing.assert_array_equal(table.image_sum(field), rows.image_sum(field))
    vectors = generator.standard_normal((3, points))
    for operation in range(table.order):
        np.testing.assert_array_equal(table.gather(vectors, operation), rows.gather(vectors, operation))
        chosen = generator.integers(0, points, 17)
        np.testing.assert_array_equal(table.image_rows(operation, chosen), rows.image_rows(operation, chosen))
        total_a, total_b = np.zeros(points), np.zeros(points)
        table.accumulate(total_a, field, operation)
        rows.accumulate(total_b, field, operation)
        np.testing.assert_array_equal(total_a, total_b)


class SlabRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        from parsec_python.tests.unit import test_slab_mesh_symmetry as slabs

        self.slabs = slabs
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        for name in ("B_FHIPP.DAT", "N_FHIPP.DAT", "C_FHIPP.DAT"):
            shutil.copy(slabs._HBN / name, self.directory / name)
        shutil.copy(slabs._C_POTRE, self.directory / "C_POTRE.DAT")

    def slab(self, text: str):
        from parsec_python.Grid.confined import build_confined_periodic_grid
        from parsec_python.Input.parsec_input import parse_parsec_input

        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        problem = parse_parsec_input(path).problem
        grid = build_confined_periodic_grid(problem.periodic_cell, problem.grid)
        group, _ = find_wire_slab_space_group(
            problem.periodic_cell.lattice_vectors, 2, problem.grid.confinement_radius, problem.atoms
        )
        return grid, group

    def test_slab_rows_are_the_table(self) -> None:
        slabs = self.slabs
        cases = {
            # hexagonal: matrices that mix the two periodic axes (12 operations), z -> -z among them
            "hbn": slabs._HBN_INPUT.format(mesh="6 6 1", extra="", xc="pbe", z_n="0.0", flag_b="", flag_n=""),
            # S4 about z (4 operations, z -> -z with a quarter turn)
            "s4": slabs._SQUARE_INPUT.format(extra="", coords=slabs._S4_COORDS),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                grid, group = self.slab(text)
                self.assertGreater(group.order, 1)
                with _with_rows("off"):
                    table = confined_grid_symmetry(grid, group)
                for setting in ("on",):
                    with _with_rows(setting):
                        rows = confined_grid_symmetry(grid, group)
                    self.assertTrue(table.holds_table)
                    self.assertIsInstance(rows.images, AffineImageRows)
                    self.assertEqual(rows.counts[:2], tuple(int(value) for value in grid.points_per_axis[:2]))
                    _same_symmetry(self, table, rows, grid.size)
                    self.assertLess(rows.held_bytes, table.table_bytes)
                # auto: below 2^17 points the table, as a small bulk grid
                with _with_rows("auto"):
                    self.assertTrue(confined_grid_symmetry(grid, group).holds_table)

    def test_the_rows_average_a_slab_density_as_the_table(self) -> None:
        from parsec_python.Grid.point_group import symmetrize_scalar

        slabs = self.slabs
        grid, group = self.slab(
            slabs._HBN_INPUT.format(mesh="6 6 1", extra="", xc="pbe", z_n="0.0", flag_b="", flag_n="")
        )
        field = np.random.default_rng(3).random(grid.size)
        with _with_rows("off"):
            reference = symmetrize_scalar(field, confined_grid_symmetry(grid, group))
        with _with_rows("on"):
            averaged = symmetrize_scalar(field, confined_grid_symmetry(grid, group))
        np.testing.assert_array_equal(reference, averaged)

    def test_a_wire_keeps_its_table(self) -> None:
        from parsec_python.Grid.confined import build_confined_periodic_grid
        from parsec_python.models import PeriodicCell, PeriodicGridSettings
        from parsec_python.tests.unit.test_wire_slab_ionic import NA_CHAIN, confined_lattice_vectors

        cell = PeriodicCell(confined_lattice_vectors([[10.0, 0.0, 0.0]]))
        settings = PeriodicGridSettings(
            spacing=0.9, shift=(0.5, 0.5, 0.5), expansion_order=8, periodicity=1, confinement_radius=6.0
        )
        grid = build_confined_periodic_grid(cell, settings)
        group, _ = find_wire_slab_space_group(cell.lattice_vectors, 1, 6.0, NA_CHAIN)
        with _with_rows("on"):
            symmetry = confined_grid_symmetry(grid, group)
        self.assertTrue(symmetry.holds_table)
        self.assertIsNone(symmetry.affine_maps)


class LeanObliqueTests(unittest.TestCase):
    def fcc(self, spacing: float):
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

        side = 3.567 / 0.529177210903
        lattice = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * side
        atoms = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", tuple(np.full(3, 0.25 * side)))]
        settings = PeriodicGridSettings(spacing=spacing, expansion_order=4, shift=(0.0, 0.0, 0.0), points_rule="fortran")
        grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), settings)
        return grid, find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)

    def test_lean_oblique_gathers_are_the_table(self) -> None:
        grid, group = self.fcc(0.2)
        with _with_rows("off"):
            table = periodic_grid_symmetry(grid, group)
        with _with_rows("on"):
            lean = periodic_grid_symmetry(grid, group)
        self.assertTrue(lean.images.lean)
        oblique = [operation for operation in range(lean.order) if lean.images.oblique(operation)]
        self.assertTrue(oblique)
        _same_symmetry(self, table, lean, grid.size)
        # the planes of the components alone: no row of an oblique operation is held
        plane = int(grid.points_per_axis[1]) * int(grid.points_per_axis[2])
        self.assertLess(lean.held_bytes, 3 * 4 * plane * lean.order + 64 * 1024)
        self.assertLess(lean.held_bytes, len(oblique) * grid.size * 4)

    def test_blocks_of_planes_give_the_same_values(self) -> None:
        from parsec_python.Grid import space_group

        grid, group = self.fcc(0.2)
        with _with_rows("off"):
            table = periodic_grid_symmetry(grid, group)
        field = np.random.default_rng(5).standard_normal(grid.size)
        for block in (1, 7, 1 << 20):
            with self.subTest(block=block), mock.patch.object(space_group, "_FACTOR_BLOCK_POINTS", block):
                with _with_rows("on"):
                    lean = periodic_grid_symmetry(grid, group)
                np.testing.assert_array_equal(table.image_sum(field), lean.image_sum(field))

    def test_auto_keeps_the_rows_of_oblique_operations(self) -> None:
        grid, group = self.fcc(0.2)
        with _with_rows("auto"), mock.patch("parsec_python.Grid.space_group._IMAGE_ROWS_POINTS", 0), mock.patch(
            "parsec_python.Laplacian.finite_difference._build_threads", return_value=4
        ):
            rows = periodic_grid_symmetry(grid, group)
        self.assertFalse(rows.images.lean)
        rows.image_sum(np.ones(grid.size))
        oblique = sum(1 for operation in range(rows.order) if rows.images.oblique(operation))
        self.assertGreaterEqual(rows.held_bytes, oblique * grid.size * 4)


class MeshStarGatherTests(unittest.TestCase):
    def test_gathers_are_int32_inverse_permutations(self) -> None:
        from parsec_python.tests.unit.test_hybrid_image_pairs import _cube_wedge

        system, _k_points, _weights, _box, stars = _cube_wedge("2 2 2")
        symmetry = system.density_symmetry
        self.assertTrue(stars.gathers)
        for operation, gather in stars.gathers.items():
            self.assertEqual(gather.dtype, np.int32)
            mapped = np.asarray(symmetry.images[operation], dtype=np.int64)
            reference = np.empty_like(mapped)
            reference[mapped] = np.arange(mapped.size, dtype=np.int64)
            np.testing.assert_array_equal(gather, reference)


if __name__ == "__main__":
    unittest.main()
