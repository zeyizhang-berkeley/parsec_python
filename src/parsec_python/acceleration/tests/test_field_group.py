"""Scalar fields of weighted symmetry sectors and the full group of the structure.

Weighted sectors (``PointGroupReduction``) solve the one-dimensional sectors of a subgroup ``H`` of the point group
``G`` and count the partners of the levels of ``G`` with weights.  The tests here cover what the SCF fields take
from ``G``:

* the full-grid inputs averaged over ``G`` (``PARSEC_SECTOR_INPUT_GROUP_AVERAGE``);
* the scalar fields on the orbits of ``G`` (``PARSEC_SECTOR_FIELD_GROUP``), with the gradients of the device XC
  evaluators on them for a group of more than 255 operations (a doubled cubic cell: Oh with 8 half-cell translations);
* the GPU ionic sums on the orbits of ``G``, of ``H`` or of the abelian group, whichever first maps the atoms exactly
  (``PARSEC_IONIC_SUM_GROUP``), and those of moved structures (``PARSEC_IONIC_MOVED_SUMS``);
* the Hartree boundary on the rows of its wedge (``PARSEC_HARTREE_BOUNDARY_ROWS``).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings


def sphere_grid(shift=(0.5, 0.5, 0.5), spacing=0.5, radius=3.8, order=8):
    return build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=order, shift=shift))


def tetrahedral_problem(shift=(0.5, 0.5, 0.5), *, offset=0.0, atom_tolerance=1.0e-7):
    """A CH4-like Td structure on a sphere grid and its weighted reduction (D2d sectors of Td).

    ``offset`` moves the first hydrogen along x (bohr), a structure that is Td only within a looser
    ``atom_tolerance``.
    """

    from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction

    grid = sphere_grid(shift)
    a = 1.2
    positions = np.array([[0.0, 0.0, 0.0]] + [
        [a * sx, a * sy, a * sz] for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])
    positions[1, 0] += offset
    atoms = (Atom("C", positions[0]),) + tuple(Atom("H", p) for p in positions[1:])
    abelian = SignedPermutationReduction.detect(grid, atoms, atom_tolerance=atom_tolerance)
    reduction = detect_point_group_reduction(grid, atoms, abelian, atom_tolerance=atom_tolerance)
    return grid, atoms, positions, reduction


def group_images(grid, reduction):
    """The row map of every operation of the full group of ``reduction``."""

    from parsec_python.acceleration.Symmetry.axis_reflection import _grid_row_mapping_operation

    maps = []
    for operation in np.asarray(reduction.full_operations):
        mapping = _grid_row_mapping_operation(grid, operation, 1.0e-10)
        assert mapping is not None
        maps.append(mapping)
    return maps


class InputGroupAverageTests(unittest.TestCase):
    """``PARSEC_SECTOR_INPUT_GROUP_AVERAGE``: full-grid inputs of weighted sectors averaged over ``G``."""

    def test_switch_values(self) -> None:
        from parsec_python.acceleration.SCF.symmetry_fields import sector_input_group_average_setting

        for value, expected in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with patch.dict(os.environ, {"PARSEC_SECTOR_INPUT_GROUP_AVERAGE": value}):
                self.assertEqual(sector_input_group_average_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_SECTOR_INPUT_GROUP_AVERAGE": "on"}):
            with self.assertRaises(ValueError):
                sector_input_group_average_setting()

    def test_an_input_invariant_under_h_alone_is_averaged_over_g(self) -> None:
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer, group_average_applies
        from parsec_python.acceleration.Symmetry import PointGroupReduction

        grid, atoms, positions, reduction = tetrahedral_problem()
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertEqual(len(reduction.full_operations), 24)
        self.assertEqual(reduction.group_order, 8)
        self.assertTrue(group_average_applies(reduction))
        # A field invariant under H but not under G: the H-orbit average of x^2 + 2 y^2 + 3 z^2 plus a ripple.
        xyz = grid.coordinates
        raw = xyz[:, 0] ** 2 + 2.0 * xyz[:, 1] ** 2 + 3.0 * xyz[:, 2] ** 2 + 0.1 * np.sin(3.0 * xyz[:, 0])
        plain = SymmetrySCFReducer(reduction)
        averaged = SymmetrySCFReducer(reduction, group_average_inputs=True)
        h_values = plain.wedge_values(raw)
        # off: the orbit average of before, bit for bit
        sums = np.bincount(reduction.full_to_wedge, weights=raw, minlength=reduction.wedge_size)
        np.testing.assert_array_equal(h_values, sums / reduction.multiplicities)
        self.assertGreater(np.max(np.abs(reduction.symmetrize_values(h_values) - h_values)), 0.1)
        g_values = averaged.from_full(raw).values
        np.testing.assert_array_equal(g_values, reduction.symmetrize_values(h_values))
        # Invariant under every operation of G, and the average of the full field over G.
        full = averaged.expand_values(g_values)
        group_average = np.zeros(grid.size)
        for mapping in group_images(grid, reduction):
            image = np.empty(grid.size)
            image[mapping] = full
            np.testing.assert_allclose(image, full, rtol=0.0, atol=1.0e-12)
            moved = np.empty(grid.size)
            moved[mapping] = raw
            group_average += moved
        group_average /= len(reduction.full_operations)
        np.testing.assert_allclose(full, group_average, rtol=1.0e-12, atol=1.0e-12)
        # A field that is invariant under G keeps its values to round-off.
        invariant = np.sum(xyz * xyz, axis=1)
        np.testing.assert_allclose(
            averaged.from_full(invariant).values, plain.from_full(invariant).values, rtol=1.0e-14, atol=1.0e-14)

    def test_reductions_without_cosets_are_not_averaged(self) -> None:
        from parsec_python.acceleration.SCF.symmetry_fields import group_average_applies
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction

        grid, atoms, _, _ = tetrahedral_problem()
        self.assertFalse(group_average_applies(SignedPermutationReduction.detect(grid, atoms)))
        self.assertFalse(group_average_applies(None))

    def test_loose_tolerance_ionic_field_is_g_invariant(self) -> None:
        """A structure Td only within a loose tolerance: its ionic-like field becomes invariant under G."""

        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry import PointGroupReduction

        grid, atoms, positions, reduction = tetrahedral_problem(offset=1.0e-3, atom_tolerance=1.0e-2)
        self.assertIsInstance(reduction, PointGroupReduction)
        xyz = grid.coordinates
        field = np.zeros(grid.size)
        for position in positions:
            field -= 1.0 / np.sqrt(np.sum((xyz - position) ** 2, axis=1) + 0.5)
        plain = SymmetrySCFReducer(reduction).from_full(field).values
        averaged = SymmetrySCFReducer(reduction, group_average_inputs=True).from_full(field).values
        # Averaged over H alone the field keeps a part that G does not: of the order of the offset.
        self.assertGreater(np.max(np.abs(reduction.symmetrize_values(plain) - plain)), 1.0e-6)
        np.testing.assert_allclose(reduction.symmetrize_values(averaged), averaged, rtol=0.0, atol=1.0e-14)


class FieldOrbitReductionTests(unittest.TestCase):
    """``PARSEC_SECTOR_FIELD_GROUP``: the orbits of the full group as the wedge of the SCF fields."""

    def test_switch_values(self) -> None:
        from parsec_python.acceleration.Symmetry.field_orbits import sector_field_group_setting

        for value, expected in (("", "full"), ("full", "full"), (" Subgroup ", "subgroup")):
            with patch.dict(os.environ, {"PARSEC_SECTOR_FIELD_GROUP": value}):
                self.assertEqual(sector_field_group_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_SECTOR_FIELD_GROUP": "on"}):
            with self.assertRaises(ValueError):
                sector_field_group_setting()

    def test_orbits_of_the_full_group(self) -> None:
        from parsec_python.acceleration.Symmetry.field_orbits import FieldOrbitReduction, field_orbit_reduction

        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            with self.subTest(shift=shift):
                grid, atoms, positions, reduction = tetrahedral_problem(shift)
                field = field_orbit_reduction(reduction)
                self.assertIsInstance(field, FieldOrbitReduction)
                self.assertIs(field_orbit_reduction(reduction), field)
                self.assertEqual(field.group_order, 24)
                self.assertIs(field.subgroup, reduction)
                labels = np.asarray(field.full_to_wedge)
                # Constant on the image of every operation of G, one label per orbit, the smallest row first.
                for mapping in group_images(grid, reduction):
                    np.testing.assert_array_equal(labels[mapping], labels)
                np.testing.assert_array_equal(
                    field.multiplicities, np.bincount(labels, minlength=field.wedge_size))
                np.testing.assert_array_equal(labels[field.representative_rows], np.arange(field.wedge_size))
                smallest = np.full(field.wedge_size, grid.size)
                np.minimum.at(smallest, labels, np.arange(grid.size))
                np.testing.assert_array_equal(field.representative_rows, smallest)
                self.assertTrue(np.all(np.diff(field.representative_rows) > 0))
                # The orbits of G hold whole orbits of H: about three of D2d each.
                np.testing.assert_array_equal(
                    field.subgroup_to_field[reduction.full_to_wedge], labels)
                self.assertLess(field.wedge_size, 0.45 * reduction.wedge_size)
                # A field invariant under G crosses between the two wedges unchanged; one invariant under H alone
                # is averaged over G.
                xyz = grid.coordinates
                invariant = np.exp(-np.sum(xyz * xyz, axis=1)) + np.prod(xyz, axis=1) ** 2
                g_values = invariant[field.representative_rows]
                h_values = invariant[reduction.representative_rows]
                np.testing.assert_allclose(field.to_subgroup(g_values), h_values, rtol=1e-15, atol=0)
                np.testing.assert_allclose(field.from_subgroup(h_values), g_values, rtol=1e-14, atol=1e-16)
                h_only = xyz[:, 0] ** 2 + 2.0 * xyz[:, 1] ** 2 + 3.0 * xyz[:, 2] ** 2
                h_orbit = np.bincount(reduction.full_to_wedge, weights=h_only) / reduction.multiplicities
                np.testing.assert_allclose(
                    field.to_subgroup(field.from_subgroup(h_orbit)), reduction.symmetrize_values(h_orbit),
                    rtol=1e-13, atol=1e-13)

    def test_wedge_laplacian_is_the_reduced_full_grid_matrix(self) -> None:
        from parsec_python.acceleration.Symmetry.field_orbits import field_negative_laplacian, field_orbit_reduction
        from parsec_python.acceleration.backends.cupy_stencil_major import StencilMajorHostMetadata
        from parsec_python.Laplacian.finite_difference import build_negative_laplacian

        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            with self.subTest(shift=shift):
                grid, atoms, positions, reduction = tetrahedral_problem(shift)
                field = field_orbit_reduction(reduction)
                expected = field.reduce_operator(build_negative_laplacian(grid))
                for native in (True, False):
                    with patch(
                        "parsec_python.acceleration.Symmetry.sector_stencil.native_kernel_available",
                        return_value=native and _native_stencil_kernel(),
                    ):
                        built = field_negative_laplacian(grid, field)
                    if isinstance(built, StencilMajorHostMetadata):
                        built = built.to_csr()
                    difference = abs(built - expected)
                    self.assertLess(difference.max(), 1e-12 * abs(expected).max())

    def test_symmetrized_density_on_the_orbits_of_the_full_group(self) -> None:
        """``Symmetrize_Charge_Density`` on the field orbits: the reference average of the expanded density."""

        from parsec_python.acceleration.SCF.density_symmetry import build_wedge_density_symmetry
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.field_orbits import field_orbit_reduction
        from parsec_python.Grid.point_group import symmetrize_density
        from parsec_python.SCF.single_point import density_point_group

        grid, atoms, positions, reduction = tetrahedral_problem()
        field = field_orbit_reduction(reduction)
        group = density_point_group(grid, atoms)
        self.assertEqual(group.order, 24)
        reducer = SymmetrySCFReducer(field)
        density = reducer.from_full(np.random.default_rng(5).random(grid.size))
        symmetric, report = build_wedge_density_symmetry(group, field).symmetrize_density(
            density, grid.volume_element)
        expected, expected_report = symmetrize_density(reducer.to_full(density), group, grid.volume_element)
        np.testing.assert_allclose(reducer.to_full(symmetric), expected, rtol=1e-14, atol=1e-15)
        self.assertAlmostEqual(report.charge_before, expected_report.charge_before, places=10)
        self.assertAlmostEqual(report.charge_after, expected_report.charge_after, places=10)


_CUBE_SIDE = 10.26


def doubled_cubic_cell(points: int = 12):
    """Eight atoms on the simple-cubic sites of a cube of side 10.26 bohr: a simple-cubic cell repeated twice along
    each axis, whose group is Oh about an atom with the 8 half-cell translations (384 operations; the weighted
    sectors solve a subgroup of 64)."""

    from parsec_python.Grid.pbc import build_periodic_grid
    from parsec_python.models import PeriodicCell, PeriodicGridSettings

    half = _CUBE_SIDE / 2.0
    atoms = [Atom("Si", (i * half, j * half, k * half)) for i in range(2) for j in range(2) for k in range(2)]
    grid = build_periodic_grid(
        PeriodicCell(_CUBE_SIDE * np.eye(3)), PeriodicGridSettings(spacing=_CUBE_SIDE / points, expansion_order=8)
    )
    return grid, atoms


class OrbitOperationTableTests(unittest.TestCase):
    """The rotation per grid row the device XC evaluators turn gradients with, for a group of any order."""

    @staticmethod
    def _one_entry_per_operation(grid, reduction) -> np.ndarray:
        """The rotation per row as it was taken before 2026-11-19 (one table entry per operation, the first
        operation that maps the representative onto the row, the identity first)."""

        from parsec_python.acceleration.Symmetry.axis_reflection import _grid_row_mapping_operation

        operations = np.asarray(reduction.operations, dtype=np.int64)
        translations = getattr(reduction, "translations", None)
        order = sorted(range(operations.shape[0]), key=lambda g: not np.array_equal(operations[g], np.eye(3)))
        rotation = np.full((reduction.full_size, 6), 99, dtype=np.int64)
        for g in order:
            images = _grid_row_mapping_operation(
                grid, operations[g].astype(np.int8), 1.0e-10, selected_rows=np.asarray(reduction.representative_rows),
                translation=None if translations is None else translations[g],
            )
            permutation = np.argmax(np.abs(operations[g]), axis=1)
            fresh = images[rotation[images, 0] == 99]
            rotation[fresh] = np.concatenate([permutation, operations[g][np.arange(3), permutation]])
        return rotation

    def test_a_group_of_384_operations_takes_a_byte_per_row(self) -> None:
        from parsec_python.acceleration.Symmetry.axis_reflection import (
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.acceleration.Symmetry.field_orbits import field_orbit_reduction
        from parsec_python.acceleration.V_xc.device_xc import _orbit_operations

        grid, atoms = doubled_cubic_cell()
        subgroup = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        field = field_orbit_reduction(subgroup)
        self.assertEqual(field.group_order, 384)
        self.assertEqual(subgroup.group_order, 64)
        for reduction, entries in ((field, 48), (subgroup, 8)):
            with self.subTest(order=reduction.group_order):
                codes, table = _orbit_operations(grid, reduction)
                self.assertEqual(codes.dtype, np.uint8)
                # each distinct rotation once (the translations turn no vector)
                self.assertEqual(table.shape, (entries, 6))
                self.assertEqual(len({tuple(row) for row in table}), entries)
                # every row reads the rotation it read with one entry per operation: the same bits
                np.testing.assert_array_equal(
                    table[codes].astype(np.int64), self._one_entry_per_operation(grid, reduction)
                )


class IonicSumGroupTests(unittest.TestCase):
    """``PARSEC_IONIC_SUM_GROUP`` and ``PARSEC_IONIC_MOVED_SUMS``: the orbits the GPU ionic sums run on."""

    def test_switch_values(self) -> None:
        from parsec_python.acceleration.driver import ionic_sum_group_setting, moved_ionic_sum_setting
        from parsec_python.acceleration.Hartree.cupy_boundary import boundary_rows_setting

        cases = (
            (ionic_sum_group_setting, "PARSEC_IONIC_SUM_GROUP", ("full", "first"), "abelian"),
            (moved_ionic_sum_setting, "PARSEC_IONIC_MOVED_SUMS", ("orbits", "full"), "wedge"),
            (boundary_rows_setting, "PARSEC_HARTREE_BOUNDARY_ROWS", ("wedge", "diagonal"), "full"),
        )
        for setting, name, values, wrong in cases:
            with patch.dict(os.environ, {name: ""}):
                self.assertEqual(setting(), values[0])
            for value in values:
                with patch.dict(os.environ, {name: value.upper()}):
                    self.assertEqual(setting(), value)
            with patch.dict(os.environ, {name: wrong}):
                with self.assertRaises(ValueError):
                    setting()

    def test_ionic_backend_values(self) -> None:
        from parsec_python.acceleration import driver

        with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "CuPy"}):
            self.assertEqual(driver.ionic_backend_setting(), "cupy")
            self.assertTrue(driver._gpu_ionic_requested())
        with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "native"}):
            self.assertFalse(driver._gpu_ionic_requested())
        with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "gpu"}):
            with self.assertRaises(ValueError):
                driver.ionic_backend_setting()
        with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "auto"}):
            # A periodic cell and a CPU run keep the host builders.
            self.assertFalse(driver._gpu_ionic_requested(periodic=True))
            with patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
                self.assertFalse(driver._gpu_ionic_requested())
            with patch("parsec_python.acceleration.backends.cupy.cupy_available", return_value=False):
                self.assertFalse(driver._gpu_ionic_requested())

    def test_the_sums_take_the_orbits_of_the_full_group(self) -> None:
        from parsec_python.acceleration.driver import _ionic_sum_reduction
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction
        from parsec_python.acceleration.Symmetry.exact_images import orbit_sum_points
        from parsec_python.acceleration.Symmetry.field_orbits import FieldOrbitReduction

        grid, atoms, positions, reduction = tetrahedral_problem()
        chosen = _ionic_sum_reduction(reduction)
        self.assertIsInstance(chosen, FieldOrbitReduction)
        abelian = SignedPermutationReduction.detect(grid, atoms)
        self.assertIs(_ionic_sum_reduction(abelian), abelian)
        self.assertIsNone(_ionic_sum_reduction(None))
        points, obstacle = orbit_sum_points(grid, atoms, chosen)
        self.assertIsNone(obstacle)
        np.testing.assert_array_equal(points, grid.coordinates[chosen.representative_rows])
        # The images under every operation of Td are exact here: a sum at the representative stands for its orbit.
        self.assertLess(chosen.wedge_size, abelian.wedge_size / 4)
        # Atoms that are images only within a tolerance: every point is summed.
        moved = (atoms[0],) + tuple(
            type(atom)(atom.symbol, tuple(np.asarray(atom.position) + (1e-12 if index == 1 else 0.0)))
            for index, atom in enumerate(atoms[1:], start=1)
        )
        self.assertIsNone(orbit_sum_points(grid, moved, chosen)[0])
        # ... and on the orbits for a moved structure, whose images may be off by round-off.
        self.assertIsNotNone(orbit_sum_points(grid, moved, chosen, atom_tolerance=1e-10)[0])
        self.assertIsNone(orbit_sum_points(grid, moved, chosen, atom_tolerance=1e-13)[0])

    def test_a_structure_exact_under_the_abelian_group_alone_keeps_its_orbits(self) -> None:
        """Td within the detection tolerance, its y coordinates a bit off x and z: the sign flips of D2 map the atoms
        exactly, the axis permutations of Td and of the sectors' D2d do not.  The sums take the orbits of D2, as they
        did before the full group, instead of every grid point."""

        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Symmetry import (
            PointGroupReduction,
            SignedPermutationReduction,
            detect_point_group_reduction,
        )
        from parsec_python.acceleration.Symmetry.field_orbits import FieldOrbitReduction

        grid = sphere_grid()
        a, b = 1.2, float(np.nextafter(1.2, 2.0))
        atoms = (Atom("C", (0.0, 0.0, 0.0)),) + tuple(
            Atom("H", (a * sx, b * sy, a * sz)) for sx, sy, sz in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        )
        abelian = SignedPermutationReduction.detect(grid, atoms, atom_tolerance=1e-7)
        reduction = detect_point_group_reduction(grid, atoms, abelian, atom_tolerance=1e-7)
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertEqual(abelian.group_order, 4)
        published = driver._ionic_sum_candidates(reduction, first=abelian)
        self.assertIsInstance(published, tuple)
        self.assertIsInstance(published[0], FieldOrbitReduction)
        self.assertIs(published[1], reduction)
        self.assertIs(published[2], abelian)
        # one reduction alone is published as itself, as before
        self.assertIs(driver._ionic_sum_candidates(abelian, first=abelian), abelian)
        self.assertIsNone(driver._ionic_sum_candidates(None))
        chosen, points, obstacles = driver._orbit_sum_choice(grid, atoms, published)
        self.assertIs(chosen, abelian)
        self.assertEqual(len(obstacles), 2)
        np.testing.assert_array_equal(points, grid.coordinates[abelian.representative_rows])

        class Builders:
            told = []

            def sum_on_orbits(self, grid, reduction, points):
                self.told.append(reduction)

        builders = Builders()
        reference = type("Reference", (), {"grid": grid, "atoms": atoms})()
        preparation = driver._Preparation()
        preparation.symmetry_reduction.publish(published)
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_IONIC_SUM_GRID", None)
            text = driver._choose_ionic_sum_grid(builders, reference, preparation)
        self.assertEqual(builders.told, [abelian])
        self.assertTrue(text.startswith(f"symmetry wedge ({abelian.wedge_size} of {grid.size} points)"), text)
        self.assertIn("the larger one failed", text)
        # Exact under Td: the orbits of Td, the report as before.
        exact_grid, exact_atoms, _, exact_reduction = tetrahedral_problem()
        exact_published = driver._ionic_sum_candidates(
            exact_reduction, first=SignedPermutationReduction.detect(exact_grid, exact_atoms)
        )
        chosen, _, obstacles = driver._orbit_sum_choice(exact_grid, exact_atoms, exact_published)
        self.assertIs(chosen, exact_published[0])
        self.assertEqual(obstacles, [])
        self.assertEqual(
            driver._orbit_sum_wedge_text(chosen, obstacles),
            f"symmetry wedge ({chosen.wedge_size} of {exact_grid.size} points)",
        )


def _native_stencil_kernel() -> bool:
    from parsec_python.acceleration.Symmetry.sector_stencil import native_kernel_available

    return native_kernel_available()


def _device_ready() -> bool:
    from parsec_python.acceleration.backends.cupy import cupy_available
    from parsec_python.acceleration.backends.native import native_available

    return cupy_available() and native_available()


def _boundary_ready() -> bool:
    if not _device_ready():
        return False
    from parsec_python.acceleration.tests.test_hartree_boundary import native_order_ready

    return native_order_ready(9)


@unittest.skipUnless(_device_ready(), "CUDA runtime or native extension is unavailable")
class MovedIonicSumTests(unittest.TestCase):
    """``PARSEC_IONIC_MOVED_SUMS``: the GPU ionic sums of moved atoms on the orbits of the full group."""

    def test_moved_atoms_that_are_exact_images_are_summed_on_the_orbits(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.backends.selection import BackendSelection
        from parsec_python.acceleration.Symmetry import SignedPermutationReduction, detect_point_group_reduction
        from parsec_python.models import SpeciesPotential
        from parsec_python.tests.force_reference import DATA
        from parsec_python.V_ion import load_pseudopotentials

        specifications = {
            "Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 1),
            "H": SpeciesPotential(DATA / "H_POTRE.DAT", 0),
        }
        potentials = load_pseudopotentials(specifications)
        grid = sphere_grid((0.5, 0.5, 0.5), spacing=0.4, radius=5.0)
        a = 1.7
        corners = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        atoms = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(Atom("H", (a * x, a * y, a * z)) for x, y, z in corners)
        reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
        chosen = driver._ionic_sum_reduction(reduction)
        # The moved structure: the tetrahedron scaled by 1.01, exact images again.
        moved = (atoms[0],) + tuple(Atom("H", (1.01 * a * x, 1.01 * a * y, 1.01 * a * z)) for x, y, z in corners)
        selection = BackendSelection(
            requested="auto", selected="cupy", finite_difference_builder="native", hartree_backend="native")
        values = {}
        for label, setting in (("orbits", "orbits"), ("full", "full")):
            with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "cupy", "PARSEC_IONIC_MOVED_SUMS": setting}):
                hooks = driver._geometry_builder_hooks(
                    selection, ionic_sector_rank=False, orbits=(grid, moved, chosen))
                builder = hooks["local_ionic_builder"]
                potential = builder(grid, moved, potentials, specifications)
                density = hooks["atomic_density_builder"](grid, moved, potentials, specifications)
                values[label] = (potential, density, builder.__self__.summed_points)
        self.assertEqual(values["orbits"][2], chosen.wedge_size)
        self.assertEqual(values["full"][2], grid.size)
        rows = chosen.representative_rows
        for index in (0, 1):
            orbit, full = values["orbits"][index], values["full"][index]
            np.testing.assert_array_equal(orbit[rows], full[rows])
            np.testing.assert_allclose(orbit, full, rtol=1.0e-13, atol=1.0e-13 * np.abs(full).max())
        # Atoms that are images to round-off (a position averaged over the group): the orbits; atoms off by more
        # than MOVED_IMAGE_TOLERANCE: every point, and the report says why.
        for factor, points in ((1.0 + 1e-15, chosen.wedge_size), (1.0 + 1e-8, grid.size)):
            shaken = (moved[0], Atom("H", tuple(np.asarray(moved[1].position) * factor))) + moved[2:]
            report = {}
            with patch.dict(os.environ, {"PARSEC_IONIC_BACKEND": "cupy"}):
                hooks = driver._geometry_builder_hooks(
                    selection, ionic_sector_rank=False, orbits=(grid, shaken, chosen), report=report)
                hooks["local_ionic_builder"](grid, shaken, potentials, specifications)
            self.assertEqual(hooks["local_ionic_builder"].__self__.summed_points, points)
            self.assertTrue(report["ionic_sum_grid"].startswith(
                "symmetry wedge" if points == chosen.wedge_size else "full grid (an image of an atom is farther"),
                report)


@unittest.skipUnless(_boundary_ready(), "needs a CUDA device and the native boundary geometry of this tree")
class BoundaryRowsTests(unittest.TestCase):
    """``PARSEC_HARTREE_BOUNDARY_ROWS``: the boundary of a permuting group on the rows of its own wedge."""

    def test_switch_values(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import boundary_rows_setting

        for value, expected in (("", "wedge"), ("wedge", "wedge"), ("Diagonal", "diagonal")):
            with patch.dict(os.environ, {"PARSEC_HARTREE_BOUNDARY_ROWS": value}):
                self.assertEqual(boundary_rows_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_HARTREE_BOUNDARY_ROWS": "full"}):
            with self.assertRaises(ValueError):
                boundary_rows_setting()

    def test_rows_of_the_wedge_give_the_projected_rhs(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_boundary import (
            CuPyPointMultipoleBoundaryBuilder,
            CuPySubgroupMultipoleBoundaryBuilder,
        )
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.acceleration.Symmetry import diagonal_subgroup_reduction
        from parsec_python.acceleration.Symmetry.field_orbits import field_orbit_reduction

        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            grid, atoms, positions, reduction = tetrahedral_problem(shift)
            field = field_orbit_reduction(reduction)
            diagonal = diagonal_subgroup_reduction(grid, reduction.operations)
            xyz = grid.coordinates
            density = np.zeros(grid.size)
            for position, depth in zip(positions, (3.0, 1.0, 1.0, 1.0, 1.0)):
                density += depth * np.exp(-np.sum((xyz - position) ** 2, axis=1) / 0.8)
            density *= 8.0 / grid.integrate(density)
            charges = np.array([4.0, 1.0, 1.0, 1.0, 1.0])
            for tail in (False, True):
                point = CuPyPointMultipoleBoundaryBuilder(grid, 9)
                if tail:
                    point.set_atomic_tail(positions, charges)
                full_rhs, full_boundary = point.build(density)
                for target in (reduction, field):
                    built = {}
                    for rows in ("wedge", "diagonal"):
                        with self.subTest(shift=shift, tail=tail, group=target.group_order, rows=rows):
                            builder = CuPySubgroupMultipoleBoundaryBuilder(grid, target, diagonal, 9, rows=rows)
                            if tail:
                                builder.set_atomic_tail(positions, charges)
                            rhs, boundary = builder.build_reduced(density)
                            np.testing.assert_allclose(
                                rhs, target.reduce_vector(full_rhs), rtol=3.0e-12, atol=3.0e-12)
                            for key, value in full_boundary.moments.items():
                                self.assertLess(abs(boundary.moments[key] - value), 1.0e-9 * (1.0 + abs(value)))
                            wedge = density[target.representative_rows]
                            from_field = builder.build_reduced(SymmetryScalarField(target, wedge))[0]
                            np.testing.assert_allclose(from_field, rhs, rtol=1.0e-12, atol=1.0e-12)
                            check = builder.boundary_check(wedge, 10, minimum_order=4, positions=positions)
                            self.assertEqual(check["points"].shape, (10, 3))
                            built[rows] = (rhs.copy(), builder.exterior_point_count, builder.device_storage_bytes)
                    np.testing.assert_allclose(built["wedge"][0], built["diagonal"][0], rtol=1.0e-13, atol=1.0e-13)
                    # The exterior points of the rows of the wedge alone: fewer, and less device memory.
                    self.assertLess(built["wedge"][1], built["diagonal"][1])
                    self.assertLess(built["wedge"][2], built["diagonal"][2])


def _ge5_problem(directory, *, eps=0.0, outer="Ge"):
    """Ge in the centre of a tetrahedron of ``outer`` atoms (Ge: the case of the Td sector force test; H: GeH4, whose
    hydrogens have no projectors), the outer z scaled by ``1 + eps``."""

    from dataclasses import replace
    from pathlib import Path

    from parsec_python.acceleration.tests.test_forces_geometry import _translation

    problem = _translation("ge2h2_nlcc_tiny", Path(directory)).problem
    bond = (4.9 if outer == "Ge" else 2.9) / np.sqrt(3.0)
    atoms = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
        Atom(outer, (bond * x, bond * y, bond * z * (1.0 + eps)))
        for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    )
    return replace(
        problem,
        atoms=atoms,
        recenter_geometry=False,
        grid=replace(problem.grid, radius=9.0),
        scf=replace(problem.scf, number_of_states=16 if outer == "Ge" else 8),
    )


@unittest.skipUnless(_device_ready(), "CUDA runtime or native extension is unavailable")
class FieldGroupSCFTests(unittest.TestCase):
    """The SCF of a Td cluster with its fields on the orbits of Td and on those of the sectors' D2d."""

    def _run(self, problem, environment, *, forces=True):
        from parsec_python.acceleration import driver

        with patch.dict(os.environ, environment):
            system = driver.prepare_single_point(problem, symmetry="on")
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system)
            table = driver.compute_forces(system, result) if forces else None
        return system, result, table

    def test_full_group_fields_give_the_scf_of_the_subgroup_fields(self) -> None:
        import tempfile

        from parsec_python.acceleration.Symmetry import PointGroupReduction

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        problem = _ge5_problem(directory.name)
        runs = {}
        for label, environment in (
            ("full", {"PARSEC_SECTOR_FIELD_GROUP": "full"}),
            ("subgroup", {"PARSEC_SECTOR_FIELD_GROUP": "subgroup"}),
            ("former", {
                "PARSEC_SECTOR_FIELD_GROUP": "subgroup",
                "PARSEC_SECTOR_INPUT_GROUP_AVERAGE": "off",
                "PARSEC_HARTREE_BOUNDARY_ROWS": "diagonal",
                "PARSEC_IONIC_SUM_GROUP": "first",
            }),
        ):
            runs[label] = self._run(problem, environment)
        system, result, table = runs["full"]
        sectors = system.backend.symmetry_eigensolver
        self.assertIsInstance(sectors.decomposition.reduction, PointGroupReduction)
        details = dict(system.backend_info.details)
        self.assertTrue(details["scf_scalar_field_orbits"].startswith("full group (24 operations)"), details)
        self.assertIn("full-grid inputs averaged", dict(runs["subgroup"][0].backend_info.details)[
            "scf_scalar_field_orbits"])
        for label in ("subgroup", "former"):
            other_system, other, other_table = runs[label]
            self.assertAlmostEqual(result.energies.total, other.energies.total, delta=2.0e-8)
            self.assertEqual(result.iterations, other.iterations)
            np.testing.assert_allclose(table.total, other_table.total, rtol=0, atol=2.0e-6)

    def test_loose_tolerance_energy_is_second_order(self) -> None:
        """A structure Td within a loose tolerance: averaged over G, the weighted sectors are off by O(eps^2)."""

        import tempfile

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # GeH4: the hydrogens, which have no projectors, are off Td by up to 1.7e-2 bohr; the projectors of the
        # central Ge keep Td exactly, so what the weighted sectors see beyond the group is in the local fields.
        loose = {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "5e-2"}
        errors = {}
        for eps in (1.0e-2, 5.0e-3):
            problem = _ge5_problem(directory.name, eps=eps, outer="H")
            # The reference: the exact group of the structure (D2d about z) in abelian sectors.
            reference = self._run(problem, {"PARSEC_ORBITAL_GROUP": "abelian"}, forces=False)[1].energies.total
            for label, environment in (
                ("averaged", dict(loose, PARSEC_SECTOR_FIELD_GROUP="subgroup")),
                ("full", dict(loose, PARSEC_SECTOR_FIELD_GROUP="full")),
                ("former", dict(loose, PARSEC_SECTOR_FIELD_GROUP="subgroup", PARSEC_SECTOR_INPUT_GROUP_AVERAGE="off")),
            ):
                energy = self._run(problem, environment, forces=False)[1].energies.total
                errors.setdefault(label, []).append(abs(energy - reference))
        print("loose-tolerance energy errors (eps 1e-2, 5e-3):", errors)
        for label in ("averaged", "full"):
            # quadratic: halving eps divides the error by about four (the former first-order one by about two)
            self.assertLess(errors[label][1], 0.4 * errors[label][0] + 1.0e-9)

    def test_ionic_sums_of_a_structure_exact_under_the_abelian_group_alone(self) -> None:
        """Ge5 with the y coordinates of its outer atoms one bit larger: Td (and D2d) within the detection tolerance,
        the abelian D2 exactly.  The GPU ionic sums take the orbits of D2 with PARSEC_IONIC_SUM_GROUP=full, as with
        first, instead of every grid point."""

        import tempfile
        from dataclasses import replace

        from parsec_python.acceleration import driver

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        problem = _ge5_problem(directory.name)

        def one_bit_out(value: float) -> float:
            return float(np.nextafter(value, np.copysign(np.inf, value)))

        atoms = (problem.atoms[0],) + tuple(
            Atom(atom.symbol, (atom.position[0], one_bit_out(atom.position[1]), atom.position[2]))
            for atom in problem.atoms[1:]
        )
        problem = replace(problem, atoms=atoms)
        grids, energies = {}, {}
        for label in ("full", "first"):
            with patch.dict(os.environ, {"PARSEC_IONIC_SUM_GROUP": label, "PARSEC_IONIC_BACKEND": "cupy"}):
                system = driver.prepare_single_point(problem, symmetry="on")
                system.materialize_final_wavefunctions = False
                details = dict(system.backend_info.details)
                energies[label] = driver.run_scf(system).energies.total
            grids[label] = details.get("ionic_sum_grid")
        print("ionic sum grids:", grids)
        self.assertTrue(str(grids["first"]).startswith("symmetry wedge ("), grids)
        wedge = str(grids["first"]).split(";")[0]
        # the same orbits as with the abelian group published first, after the larger groups failed
        self.assertTrue(str(grids["full"]).startswith(wedge), grids)
        self.assertIn("the larger one failed", str(grids["full"]))
        self.assertAlmostEqual(energies["full"], energies["first"], delta=2.0e-8)


@unittest.skipUnless(_device_ready(), "CUDA runtime or native extension is unavailable")
class ManyOperationCellSCFTests(unittest.TestCase):
    """A PBE cell whose full group has 384 operations: its fields on the orbits of that group (the device PBE
    evaluator turns the gradients by a byte per row) give the SCF of the subgroup fields and of the full grid."""

    def test_pbe_of_a_doubled_cubic_cell(self) -> None:
        from dataclasses import replace
        import warnings

        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        half = _CUBE_SIDE / 2.0
        problem = replace(
            silicon("pbe"),
            atoms=tuple(
                Atom("Si", (i * half, j * half, k * half)) for i in range(2) for j in range(2) for k in range(2)
            ),
        )
        runs = {}
        for label, environment in (
            ("full", {"PARSEC_SECTOR_FIELD_GROUP": "full"}),
            ("subgroup", {"PARSEC_SECTOR_FIELD_GROUP": "subgroup"}),
            ("grid", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
        ):
            with patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy")
                system.materialize_final_wavefunctions = False
                details = dict(system.backend_info.details)
                result = driver.run_scf(system)
                self.assertTrue(result.converged, label)
                table = driver.compute_forces(system, result, symmetrize=False)
            runs[label] = (details, result, table)
        details = runs["full"][0]
        self.assertTrue(details["scf_scalar_field_orbits"].startswith("full group (384 operations)"), details)
        energy = runs["full"][1].energies.total
        self.assertAlmostEqual(energy, runs["subgroup"][1].energies.total, delta=1.0e-9)
        self.assertEqual(runs["full"][1].iterations, runs["subgroup"][1].iterations)
        self.assertAlmostEqual(energy, runs["grid"][1].energies.total, delta=1.0e-8)
        for label in ("subgroup", "grid"):
            np.testing.assert_allclose(runs["full"][2].total, runs[label][2].total, rtol=0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
