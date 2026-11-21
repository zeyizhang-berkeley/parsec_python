"""Symmetry kept where inputs used to drop it (2026-11-19).

* ``PARSEC_MOVE_FLAG_SUBGROUP``: a relaxation whose ``Move_Flag`` fixes atoms keeps the subgroup of the operations that
  map the fixed atoms onto fixed atoms, for a cluster and for a bulk cell at the Gamma point; the detection of the
  driver takes the fixed atoms as a species of their own.
* ``PARSEC_SYMMETRIZED_DENSITY_SECTORS``: ``Symmetrize_Charge_Density`` keeps the sectors of a bulk cell at the Gamma
  point (its space group averaged on the orbits of the sectors) and of a spin-polarized cluster.
* ``PARSEC_SYMMETRIZE_POSITIONS``: the input positions averaged over the group found within
  ``PARSEC_SYMMETRY_ATOM_TOLERANCE``.

The CPU tests check the groups and the routing; the GPU tests compare the runs in sectors with those of the full
grid (the switches off).
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_BOND = 4.9 / np.sqrt(3.0)
#: Ge5 of Td about the origin, the bonds stretched beyond 4.6 bohr (forces along them).
GE5 = np.array([(0.0, 0.0, 0.0)] + [(_BOND * x, _BOND * y, _BOND * z)
                for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))])


def _cupy_ready() -> bool:
    from parsec_python.acceleration.backends.cupy import cupy_available
    from parsec_python.acceleration.backends.native import native_available

    return cupy_available() and native_available()


def _ge5_input(directory: Path, *, positions=GE5, flags=None, extra: str = "") -> Path:
    """``parsec.in`` of the Ge5 cluster (``flags``: the movement index of every atom, Move_Flag -1)."""

    lines = [
        "Boundary_Sphere_Radius: 9.0",
        "Grid_Spacing: 0.5",
        "Eigensolver: chebdav",
        "Max_Iter: 200",
        "Convergence_Criterion: 1.0e-8",
        "Output_Level: 2",
        "States_Num: 16",
        "Atom_Types_Num: 1",
        "Atom_Type: Ge",
        "Local_Component: p",
    ]
    if flags is not None:
        lines.append("Move_Flag: -1")
    lines.append("begin Atom_Coord")
    for index, row in enumerate(np.asarray(positions)):
        tail = "" if flags is None else f"  {int(flags[index])}"
        lines.append("  " + "  ".join(repr(float(value)) for value in row) + tail)
    lines += ["end Atom_Coord", extra]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "parsec.in").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy(DATA / "Ge_POTRE.DAT", directory / "Ge_POTRE.DAT")
    return directory / "parsec.in"


def _run_cli(path: Path, environment: dict, options=("--backend", "cupy")):
    """The accelerated command line: ``(status, archive, parsec.out, stderr)``."""

    from parsec_python.acceleration import cli as cli_module

    errors = StringIO()
    with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", *options])
    directory = path.parent
    archive = {}
    if (directory / "parsec_python_results.npz").is_file():
        with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
            archive = {key: data[key] for key in data.files}
    log = (directory / "parsec.out").read_text(encoding="utf-8") if (directory / "parsec.out").is_file() else ""
    return status, archive, log, errors.getvalue()


class FixedAtomDetectionTests(unittest.TestCase):
    """The detection with the fixed atoms as a species of their own (CPU)."""

    def test_the_labels_are_read_by_the_detection_alone(self) -> None:
        from parsec_python.acceleration.driver import FIXED_ATOM_LABEL, _detection_atoms

        atoms = tuple(Atom("Ge", position) for position in GE5)
        self.assertIs(_detection_atoms(atoms, None), atoms)
        labelled = _detection_atoms(atoms, (False, True, False, False, False))
        self.assertEqual([atom.symbol for atom in labelled], ["Ge", "Ge" + FIXED_ATOM_LABEL, "Ge", "Ge", "Ge"])
        np.testing.assert_array_equal([atom.position for atom in labelled], GE5)
        with self.assertRaisesRegex(ValueError, "one Move_Flag per atom"):
            _detection_atoms(atoms, (True,))

    def test_the_detection_finds_the_subgroup_that_keeps_the_fixed_atoms(self) -> None:
        from parsec_python.acceleration.Symmetry import (
            detect_point_group_reduction,
            load_or_detect_reflection_reduction,
        )
        from parsec_python.acceleration.driver import _detection_atoms
        from parsec_python.Forces import atom_symmetry_operations, operations_preserving

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=9.0))
        atoms = tuple(Atom("Ge", position) for position in GE5)
        for fixed, order in (
            (None, 24),  # Td: the weighted sectors of D2d
            ((False, True, False, False, False), 6),  # C3v about the fixed atom: its mirror x <-> y
            ((False, True, True, False, False), 4),  # C2v
        ):
            with self.subTest(fixed=fixed):
                detected = _detection_atoms(atoms, fixed)
                origin, _ = load_or_detect_reflection_reduction(grid, detected, cache_directory=None)
                reduction = detect_point_group_reduction(grid, detected, origin) if origin.group_order > 1 else origin
                full = getattr(reduction, "full_operations", None)
                found = len(full) if full is not None else reduction.group_order
                expected = operations_preserving(
                    atom_symmetry_operations(atoms), [True] * 5 if fixed is None else [not flag for flag in fixed]
                )
                self.assertEqual(len(expected), order)
                # the sectors use a subgroup of the operations that respect the flags (all of them where weighted)
                self.assertEqual(order % found, 0)
                if fixed is not None:
                    self.assertLess(found, 24)
                    # every operation of the sectors maps the fixed atoms onto fixed atoms
                    from parsec_python.acceleration.Symmetry.axis_reflection import abelian_sector_operations

                    matrices = (
                        np.asarray(full).reshape(-1, 3, 3)
                        if full is not None
                        else np.asarray(abelian_sector_operations(reduction)).reshape(-1, 3, 3)
                    )
                    kept = {np.asarray(operation.matrix).round().astype(int).tobytes() for operation in expected}
                    for matrix in matrices:
                        self.assertIn(np.asarray(matrix).round().astype(int).tobytes(), kept)


class PositionSymmetryTests(unittest.TestCase):
    """``PARSEC_SYMMETRIZE_POSITIONS`` on clusters and cells (CPU)."""

    def test_the_switch(self) -> None:
        from parsec_python.Forces.position_symmetry import symmetrize_positions_setting

        for value, expected in (("", "off"), ("off", "off"), ("ON", "on")):
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZE_POSITIONS": value}):
                self.assertEqual(symmetrize_positions_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZE_POSITIONS": "yes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SYMMETRIZE_POSITIONS must be off or on"):
                symmetrize_positions_setting()

    def test_a_rounded_cluster_gets_its_group(self) -> None:
        from parsec_python.Forces import atom_symmetry_operations
        from parsec_python.Forces.position_symmetry import symmetrized_cluster_positions

        noisy = GE5.copy()
        noisy[1:, 2] *= 1.0 + 1.0e-5  # D2d exactly, Td within 4e-5 bohr
        same, order = symmetrized_cluster_positions(noisy, ["Ge"] * 5, 1.0e-7)
        self.assertEqual(order, 8)
        np.testing.assert_allclose(same, noisy, rtol=0, atol=1e-14)
        symmetric, order = symmetrized_cluster_positions(noisy, ["Ge"] * 5, 1.0e-4)
        self.assertEqual(order, 24)
        self.assertLess(float(np.abs(symmetric - noisy).max()), 4.0e-5)
        atoms = [Atom("Ge", position) for position in symmetric]
        self.assertEqual(len(atom_symmetry_operations(atoms, tolerance=1.0e-12)), 24)

    def test_a_rounded_supercell_gets_its_space_group(self) -> None:
        from parsec_python.Forces.position_symmetry import _symmetrized_cell, symmetrized_cell_positions

        side = 10.26
        base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75],
                         [.75, .25, .75], [.75, .75, .25]]) * side
        cells = np.array([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=float) * side
        perfect = (base[None] + cells[:, None]).reshape(-1, 3)
        noisy = perfect + np.random.default_rng(7).uniform(-1e-5, 1e-5, perfect.shape)
        vectors = np.eye(3) * 2 * side
        same, order = symmetrized_cell_positions(noisy, ["Si"] * 64, vectors, 1.0e-7)
        self.assertEqual(order, 1)
        np.testing.assert_array_equal(same, noisy)
        symmetric, order = symmetrized_cell_positions(noisy, ["Si"] * 64, vectors, 1.0e-4)
        # Fd-3m of the conventional cube (192) times the 8 cubes of the supercell
        self.assertEqual(order, 192 * 8)
        self.assertLess(float(np.abs(symmetric - noisy).max()), 3.0e-5)
        # without a grid: the perfect crystal moved as a whole by the mean of the noise (every difference one vector)
        difference = symmetric - perfect
        difference -= np.round(difference / (2 * side)) * (2 * side)
        np.testing.assert_allclose(difference, np.broadcast_to(difference[0], difference.shape), rtol=0, atol=1e-11)
        self.assertGreater(float(np.abs(difference).max()), 1.0e-7)
        # with the grid: shifted back onto it, the perfect crystal itself (Td about the origin, the operations with a
        # quarter of a cube as translation where the grid has whole steps for it)
        for counts, pinned in ((64, 47), (66, 23), (68, 23)):
            with self.subTest(counts=counts):
                grid = (np.array([counts] * 3), np.zeros(3))
                symmetric, order, found, shift = _symmetrized_cell(noisy, ["Si"] * 64, vectors, 1.0e-4, grid)
                self.assertEqual((order, found), (192 * 8, pinned))
                self.assertGreater(shift, 1.0e-7)
                difference = symmetric - perfect
                difference -= np.round(difference / (2 * side)) * (2 * side)
                np.testing.assert_allclose(difference, 0.0, rtol=0, atol=1e-12)
        # a substitution keeps Td about it
        symbols = ["Si"] * 63 + ["C"]
        _, order = symmetrized_cell_positions(perfect, symbols, vectors, 1.0e-6)
        self.assertEqual(order, 24)

    def test_cells_are_shifted_onto_the_operations_of_their_grid(self) -> None:
        from parsec_python.Forces.position_symmetry import _symmetrized_cell

        noise = np.random.default_rng(3)
        # diamond 2x2x2 with a bond centre (D3d) at the origin
        side = 6.74
        base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75],
                         [.75, .25, .75], [.75, .75, .25]]) * side
        cells = np.array([(i, j, k) for i in range(2) for j in range(2) for k in range(2)], dtype=float) * side
        perfect = (base[None] + cells[:, None]).reshape(-1, 3) - side / 8.0
        perfect -= np.floor(perfect / (2 * side)) * (2 * side)
        noisy = perfect + noise.uniform(-1e-5, 1e-5, perfect.shape)
        for counts, pinned in ((48, 47), (50, 11)):
            symmetric, order, found, _ = _symmetrized_cell(
                noisy, ["C"] * 64, np.eye(3) * 2 * side, 1.0e-4, (np.array([counts] * 3), np.zeros(3))
            )
            self.assertEqual((order, found), (1536, pinned))
            difference = symmetric - perfect
            difference -= np.round(difference / (2 * side)) * (2 * side)
            np.testing.assert_allclose(difference, 0.0, rtol=0, atol=1e-12)
        # hcp in its oblique cell (inversion centre at the origin); a grid shifted by half a step along z keeps every
        # operation, one shifted along the oblique axes too (which Fortran PARSEC zeroes) only those that map it
        a, c = 6.06, 9.83
        vectors = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
        perfect = np.array([[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]]) @ vectors
        noisy = perfect + noise.uniform(-1e-5, 1e-5, perfect.shape)
        for grid_shift, pinned in (((0.0, 0.0, 0.0), 23), ((0.0, 0.0, 0.5), 23), ((0.5, 0.5, 0.5), 7)):
            symmetric, order, found, _ = _symmetrized_cell(
                noisy, ["Mg"] * 2, vectors, 1.0e-4, (np.array([20, 20, 32]), np.asarray(grid_shift))
            )
            self.assertEqual((order, found), (24, pinned))
            difference = (symmetric - perfect) @ np.linalg.inv(vectors)
            np.testing.assert_allclose((difference - np.round(difference)) @ vectors, 0.0, rtol=0, atol=1e-12)
        # no operation near one of the grid: nothing to pin, the average kept
        symmetric, order, found, shift = _symmetrized_cell(
            noisy + np.array([0.0123, 0.0371, 0.0517]), ["Mg"] * 2, vectors, 1.0e-4, (np.array([20, 20, 32]), np.zeros(3))
        )
        self.assertEqual((order, found, shift), (24, 0, 0.0))
        # the horizontal mirror 1.5e-4 bohr off a plane of the grid (tolerance 1e-4): the operations that flip z would
        # need a shift beyond the tolerance and keep their translations; the 11 that keep z are pinned
        lifted = perfect + np.array([0.0, 0.0, 1.5e-4])
        symmetric, order, found, shift = _symmetrized_cell(
            lifted, ["Mg"] * 2, vectors, 1.0e-4, (np.array([20, 20, 32]), np.zeros(3))
        )
        self.assertEqual((order, found), (24, 11))
        self.assertLess(shift, 1e-12)
        np.testing.assert_allclose(symmetric, lifted, rtol=0, atol=1e-12)

    def test_the_translations_of_a_supercell_are_averaged_in_linear_memory(self) -> None:
        import tracemalloc

        from parsec_python.Forces.position_symmetry import _symmetrized_cell

        # diamond 8^3 (4,096 atoms, 2,048 pure translations): a table of every translation took 165 MiB
        side = 6.74
        base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75],
                         [.75, .25, .75], [.75, .75, .25]]) * side
        cells = np.array([(i, j, k) for i in range(8) for j in range(8) for k in range(8)], dtype=float) * side
        perfect = (base[None] + cells[:, None]).reshape(-1, 3)
        noisy = perfect + np.random.default_rng(1).uniform(-1e-6, 1e-6, perfect.shape)
        tracemalloc.start()
        try:
            symmetric, order, found, _ = _symmetrized_cell(
                noisy, ["C"] * len(noisy), np.eye(3) * 8 * side, 1.0e-5, (np.array([192] * 3), np.zeros(3))
            )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual((order, found), (192 * 512, 47))
        self.assertLess(peak, 24 * 2**20)
        difference = symmetric - perfect
        difference -= np.round(difference / (8 * side)) * (8 * side)
        np.testing.assert_allclose(difference, 0.0, rtol=0, atol=1e-11)

    def test_operations_that_form_no_group_are_refused(self) -> None:
        from parsec_python.Forces.position_symmetry import symmetrized_cluster_positions

        # an isosceles triangle in the plane z = 0 whose legs differ by 0.06 bohr: within 0.1 bohr the mirror
        # x -> -x and the swap x <-> y are taken, but they generate more than the structure keeps
        positions = np.array([[2.0, 0.0, 0.0], [-2.03, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]])
        try:
            symmetric, order = symmetrized_cluster_positions(positions, ["H"] * 4, 0.1)
        except ValueError as error:
            self.assertIn("form no group", str(error))
        else:
            # where they do form a group, the result keeps every one of them
            from parsec_python.Forces import atom_symmetry_operations

            atoms = [Atom("H", position) for position in symmetric]
            self.assertEqual(len(atom_symmetry_operations(atoms, tolerance=1.0e-10)), order)

    def test_the_translation_is_changed_only_under_on(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Forces.position_symmetry import symmetrize_input_positions

        noisy = GE5 + np.array([3.0, 3.0, 3.0])
        noisy[1:, 2] += 1.0e-5 * np.sign(noisy[1:, 2] - 3.0)
        problem = SimpleNamespace(
            atoms=tuple(Atom("Ge", position) for position in noisy), periodic_cell=None, recenter_geometry=True
        )
        translation = SimpleNamespace(problem=problem)
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZE_POSITIONS": "off"}):
            self.assertEqual(symmetrize_input_positions(translation), (translation, []))
        with mock.patch.dict(
            os.environ, {"PARSEC_SYMMETRIZE_POSITIONS": "on", "PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4"}
        ), mock.patch("parsec_python.Forces.position_symmetry.replace", side_effect=_namespace_replace):
            changed, notes = symmetrize_input_positions(translation)
        self.assertEqual(len(notes), 1)
        self.assertIn("averaged over the 24 operations", notes[0])
        moved = np.array([atom.position for atom in changed.problem.atoms])
        self.assertLess(float(np.abs(moved - noisy).max()), 2.0e-5)
        # symmetric about the centre of its box, which the preparation recentres on
        from parsec_python.Forces import atom_symmetry_operations
        from parsec_python.V_ion import center_cluster_geometry

        centred = center_cluster_geometry(changed.problem.atoms)
        self.assertEqual(len(atom_symmetry_operations(centred, tolerance=1.0e-10)), 24)

    def test_a_cell_is_shifted_onto_the_operations_of_its_grid(self) -> None:
        from itertools import permutations, product
        from types import SimpleNamespace

        from parsec_python.Forces.position_symmetry import symmetrize_input_positions
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        # the displaced silicon cube (Td about the origin, 18 points per side) with 1e-5 bohr of noise
        noisy = displaced_positions() + np.random.default_rng(5).uniform(-1e-5, 1e-5, (8, 3))
        noisy -= np.floor(noisy / 10.26) * 10.26
        problem = SimpleNamespace(
            atoms=tuple(Atom("Si", position) for position in noisy),
            periodic_cell=PeriodicCell(np.eye(3) * 10.26),
            grid=PeriodicGridSettings(spacing=0.57),
        )
        with mock.patch.dict(
            os.environ, {"PARSEC_SYMMETRIZE_POSITIONS": "on", "PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4"}
        ), mock.patch("parsec_python.Forces.position_symmetry.replace", side_effect=_namespace_replace):
            changed, notes = symmetrize_input_positions(SimpleNamespace(problem=problem))
        self.assertEqual(len(notes), 1)
        self.assertIn("averaged over the 24 operations of the cell", notes[0])
        self.assertIn("which gives 23 point operations a translation that maps the grid onto itself", notes[0])
        moved = np.array([atom.position for atom in changed.problem.atoms])
        # every operation of Td about the origin exactly (the neighbours keep the mean of their noise along the bonds)
        held = 0
        for order in permutations(range(3)):
            for signs in product((1.0, -1.0), repeat=3):
                rotation = np.diag(signs) @ np.eye(3)[list(order)]
                distance = (moved @ rotation.T)[:, None, :] - moved[None, :, :]
                distance -= np.round(distance / 10.26) * 10.26
                held += bool(np.all(np.min(np.abs(distance).max(axis=2), axis=1) < 1e-12))
        self.assertEqual(held, 24)
        difference = moved - displaced_positions()
        difference -= np.round(difference / 10.26) * 10.26
        self.assertLess(float(np.abs(difference).max()), 2e-5)
        np.testing.assert_allclose(difference[:4], 0.0, rtol=0, atol=1e-12)


def _namespace_replace(item, **changes):
    from dataclasses import is_dataclass
    from types import SimpleNamespace

    if is_dataclass(item):
        return replace(item, **changes)
    return SimpleNamespace(**{**vars(item), **changes})


class PeriodicDensitySymmetryTests(unittest.TestCase):
    """The average of the space group of a cell on the orbits of its sectors (CPU)."""

    @staticmethod
    def _cell():
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        cell = PeriodicCell(lattice_vectors=np.eye(3) * 6.0)
        return build_periodic_grid(cell, PeriodicGridSettings(spacing=0.5, expansion_order=4)), cell

    def test_a_group_the_sectors_keep_is_averaged_on_the_orbits(self) -> None:
        from parsec_python.Grid.point_group import symmetrize_density
        from parsec_python.Grid.space_group import identity_space_group, periodic_grid_symmetry
        from parsec_python.acceleration.SCF.density_symmetry import build_wedge_density_symmetry
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.acceleration.Symmetry import detect_point_group_reduction, load_or_detect_reflection_reduction

        grid, cell = self._cell()
        atoms = (Atom("Si", (0.0, 0.0, 0.0)),)
        origin, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
        reduction = detect_point_group_reduction(grid, atoms, origin)
        # the identity alone (PARSEC keeps no group): the field as it is
        identity = periodic_grid_symmetry(grid, identity_space_group())
        wedge = build_wedge_density_symmetry(identity, reduction)
        values = np.random.default_rng(1).random(reduction.wedge_size)
        np.testing.assert_array_equal(wedge.symmetrize_values(values), values)
        # the full group of the cube about the atom: the reference average of the expanded field, on the orbits
        from parsec_python.Grid.space_group import find_space_group

        group = periodic_grid_symmetry(grid, find_space_group(cell.lattice_vectors, atoms))
        self.assertEqual(group.order, 48)
        wedge = build_wedge_density_symmetry(group, reduction)
        field = SymmetryScalarField(reduction, values)
        averaged, _ = wedge.symmetrize_density(field, grid.volume_element)
        # (the field on the full grid: the value of its orbit at every point)
        expected, _ = symmetrize_density(values[reduction.full_to_wedge], group, grid.volume_element)
        np.testing.assert_allclose(averaged.values[reduction.full_to_wedge], expected, rtol=0, atol=1e-13)

    def test_a_group_the_sectors_do_not_map_onto_itself_is_refused(self) -> None:
        from parsec_python.Grid.space_group import PeriodicGridSymmetry, affine_grid_images
        from parsec_python.acceleration.SCF.density_symmetry import build_wedge_density_symmetry
        from parsec_python.acceleration.Symmetry import detect_point_group_reduction, load_or_detect_reflection_reduction

        grid, _ = self._cell()
        atoms = (Atom("Si", (0.0, 0.0, 0.0)),)
        origin, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
        reduction = detect_point_group_reduction(grid, atoms, origin)
        counts = tuple(int(value) for value in grid.points_per_axis)
        # the identity and the mirror y -> -y through the plane y = 3 steps: the fourfold axis of the sectors (D4h
        # about x) maps it onto the mirror z -> -z through z = 3 steps, which is not in the group
        identity = np.arange(grid.size)
        mirror = affine_grid_images(np.diag([1, -1, 1]), np.array([0, 6, 0]), counts, np.int64)
        group = PeriodicGridSymmetry(
            matrices=np.stack([np.eye(3), np.diag([1.0, -1.0, 1.0])]), images=np.stack([identity, mirror])
        )
        with self.assertRaisesRegex(ValueError, "PARSEC_SYMMETRIZED_DENSITY_SECTORS=off"):
            build_wedge_density_symmetry(group, reduction)

    def test_the_switch_and_the_route_of_a_cell(self) -> None:
        from parsec_python.acceleration import cli as cli_module
        from parsec_python.acceleration.SCF.density_symmetry import symmetrized_density_sectors_setting
        from parsec_python.Input.parsec_input import parse_parsec_input

        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "maybe"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SYMMETRIZED_DENSITY_SECTORS must be auto or off"):
                symmetrized_density_sectors_setting()
        with tempfile.TemporaryDirectory() as folder:
            path = _cell_input(Path(folder), "Symmetrize_Charge_Density: .true.")
            translation = parse_parsec_input(path)
        name = "charge-density symmetrization of a periodic cell (Symmetrize_Charge_Density)"
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "auto"}):
            self.assertNotIn(name, cli_module._reference_only_features(translation, accelerated_backend_selected=True))
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "off"}):
            self.assertIn(name, cli_module._reference_only_features(translation, accelerated_backend_selected=True))
        # a spin-polarized cluster keeps its sectors under auto
        from parsec_python.acceleration.driver import _spin_sector_refusal

        scf = SimpleNamespaceSCF(spin_symmetry_sector_mode="auto", symmetrize_density=True)
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "auto"}):
            self.assertIsNone(_spin_sector_refusal(_Problem(scf)))
        with mock.patch.dict(os.environ, {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "off"}):
            self.assertIn("Symmetrize_Charge_Density", _spin_sector_refusal(_Problem(scf)))


class SimpleNamespaceSCF:
    def __init__(self, **values):
        self.__dict__.update(values)


class _Problem:
    def __init__(self, scf):
        self.scf = scf
        self.pseudopotentials = {}


def _cell_input(directory: Path, extra: str = "", *, species=None, side: float = 5.4, spacing: float = 0.5,
                states: int = 8) -> Path:
    """``parsec.in`` of a cubic cell (default: CsCl of Si and Ge, group Oh with the parser) with ``extra``."""

    species = species or [("Si", "p", [(0.0, 0.0, 0.0)]), ("Ge", "p", [(side / 2, side / 2, side / 2)])]
    lines = [
        "Periodic_System: .true.", "Boundary_Conditions: bulk", "begin Cell_Shape", f" {side} {side} {side}",
        "end Cell_Shape", "Lattice_Vector_Scale: 1.0 bohr", "Coordinate_Unit: Cartesian_Bohr",
        f"Grid_Spacing: {spacing}", "Expansion_Order: 8", f"States_Num: {states}", "Fermi_Temp: 300.0",
        "Convergence_Criterion: 1.0e-8", "Max_Iter: 200", "Output_Level: 2", f"Atom_Types_Num: {len(species)}",
    ]
    for symbol, local, coordinates in species:
        lines += [f"Atom_Type: {symbol}", f"Local_Component: {local}", "begin Atom_Coord"]
        lines += ["  " + "  ".join(repr(float(value)) for value in row) for row in coordinates]
        lines += ["end Atom_Coord"]
    lines.append(extra)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "parsec.in").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for symbol, _, _ in species:
        shutil.copy(DATA / f"{symbol}_POTRE.DAT", directory / f"{symbol}_POTRE.DAT")
    return directory / "parsec.in"


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class MoveFlagRelaxationTests(unittest.TestCase):
    """Relaxations with fixed atoms in the sectors of their subgroup equal those of the full grid (GPU)."""

    RELAXATION = "Minimization: bfgs\nMovement_Num: 2\nForce_Min: 1.0e-5\n"

    def _compare(self, sectors, full, *, fixed) -> None:
        for label, (status, _, log, errors) in (("sectors", sectors), ("full", full)):
            self.assertIn(status, (0, 4), f"{label}: {errors}\n{log[-4000:]}")
        positions = np.asarray(sectors[1]["relaxation_positions"])
        self.assertEqual(positions.shape[0], 3)
        self.assertGreater(float(np.abs(positions[-1] - positions[0]).max()), 1.0e-3)
        for index in np.flatnonzero(fixed):
            np.testing.assert_array_equal(positions[:, index], np.broadcast_to(positions[0, index], positions[:, index].shape))
        for key in ("relaxation_positions", "relaxation_energies"):
            np.testing.assert_allclose(sectors[1][key], full[1][key], rtol=0, atol=1e-6, err_msg=key)
        # the move after the last structure: its forces differ by about 1e-6 Ry/bohr (the full grid does not average
        # them), which the optimizer turns into a few 1e-6 bohr
        np.testing.assert_allclose(
            sectors[1]["relaxation_final_positions"], full[1]["relaxation_final_positions"], rtol=0, atol=5e-6
        )
        np.testing.assert_allclose(sectors[1]["relaxation_forces"], full[1]["relaxation_forces"], rtol=0, atol=2e-6)

    def test_a_cluster_keeps_the_subgroup_of_its_fixed_atom(self) -> None:
        flags = (1, 0, 1, 1, 1)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            runs = {
                setting: _run_cli(_ge5_input(root / setting, flags=flags, extra=self.RELAXATION),
                                  {"PARSEC_MOVE_FLAG_SUBGROUP": setting})
                for setting in ("auto", "off")
            }
        sectors, full = runs["auto"], runs["off"]
        self._compare(sectors, full, fixed=[flag == 0 for flag in flags])
        self.assertIn("keeps the subgroup of the 6 operations that map the fixed atoms onto fixed atoms", sectors[2])
        self.assertIn("count as a species of their own", sectors[2])
        self.assertIn("orbital_symmetry = CuPy real", sectors[2])
        self.assertIn("Move_Flag fixes atoms that a symmetry operation", full[2])
        self.assertNotIn("orbital_symmetry = CuPy real", full[2])

    def test_a_cell_keeps_the_sectors_of_the_subgroup(self) -> None:
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            for count, positions, group in (
                # Td about the origin; three of its four displaced neighbours fixed: C3v about the fourth
                (5, displaced_positions(), None),
                # C2v about the origin (the atom there moved along z), every atom but that one fixed: the fixed atoms
                # are a union of orbits, the whole group is kept (4 commuting operations)
                (1, displaced_positions(origin_shift=0.05), "an Abelian group of 4 operations"),
            ):
                with self.subTest(move_flag=count):
                    # (the input of the cube has output_level 4: every detail of the backend)
                    extra = f"Move_Flag: {count}\n" + self.RELAXATION
                    runs = {}
                    for setting in ("auto", "off"):
                        path = write_silicon(root / f"{count}_{setting}", extra, positions=positions)
                        runs[setting] = _run_cli(path, {"PARSEC_MOVE_FLAG_SUBGROUP": setting,
                                                       "PARSEC_PERIODIC_SECTOR_FORCES": "auto"})
                    sectors, full = runs["auto"], runs["off"]
                    self._compare(sectors, full, fixed=[index >= count for index in range(8)])
                    self.assertIn("Forces of the periodic cell in the sectors of its point group", sectors[2])
                    self.assertIn("count as a species of their own", sectors[2])
                    self.assertNotIn("Forces of the periodic cell in the sectors", full[2])
                    if group is not None:
                        self.assertIn(group, sectors[2])


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class SymmetrizedDensitySectorTests(unittest.TestCase):
    """Symmetrize_Charge_Density in sectors equals the full grid (GPU)."""

    def test_a_cell_at_the_gamma_point(self) -> None:
        si8 = [("Si", "p", (np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25],
                                      [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]]) * 10.26).tolist())]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            # (Si8: 24 states complete the six-fold lowest conduction level, which Fermi_Temp 300 K occupies at 1e-4 per
            # state; 20 states cut it after four, and the reference solver then kept its residual near 2e-4 Ry for all
            # of its 200 iterations, on every route, with or without the symmetrization: Output/state_truncation.py)
            for label, keywords in (("CsCl", {}), ("Si8", {"species": si8, "side": 10.26, "states": 24})):
                with self.subTest(cell=label):
                    sectors = _run_cli(
                        _cell_input(root / f"{label}_auto", "Symmetrize_Charge_Density: .true.", **keywords),
                        {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "auto"},
                    )
                    full = _run_cli(
                        _cell_input(root / f"{label}_off", "Symmetrize_Charge_Density: .true.", **keywords),
                        {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "off"},
                        options=("--backend", "auto"),
                    )
                    self.assertEqual(sectors[0], 0, f"sectors: {sectors[3]}\n{sectors[2][-4000:]}")
                    self.assertEqual(full[0], 0, f"full: {full[3]}\n{full[2][-4000:]}")
                    self.assertIn("orbital_symmetry = CuPy real", sectors[2])
                    # the density of every iteration averaged and reported, as by the reference solver
                    self.assertIn("electron charge after symmetrization", sectors[2])
                    self.assertAlmostEqual(
                        float(sectors[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=2e-6
                    )

    def test_forces_of_a_cell_on_the_accelerated_full_grid(self) -> None:
        # The forces of the Fortran-derived routines keep the full grid of the accelerated driver (no sectors), now
        # with the density averaged there over the space group (C4v about the Si atom: the Ge atom moved along z)
        species = [("Si", "p", [(0.0, 0.0, 0.0)]), ("Ge", "p", [(2.7, 2.7, 2.95)])]
        extra = "Symmetrize_Charge_Density: .true.\nSkip_force: .false."
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            accelerated = _run_cli(_cell_input(root / "auto", extra, species=species),
                                   {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "auto"})
            reference = _run_cli(_cell_input(root / "off", extra, species=species),
                                 {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": "off"}, options=("--backend", "auto"))
        for label, (status, _, log, errors) in (("accelerated", accelerated), ("reference", reference)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-4000:]}")
        self.assertIn("Acceleration backend", accelerated[2])
        self.assertNotIn("Acceleration backend", reference[2])
        self.assertGreater(float(np.abs(reference[1]["forces_total"]).max()), 1.0e-3)
        np.testing.assert_allclose(accelerated[1]["forces_total"], reference[1]["forces_total"], rtol=0, atol=2e-6)
        self.assertAlmostEqual(
            float(accelerated[1]["energy_total_ry"]), float(reference[1]["energy_total_ry"]), delta=2e-6
        )

    def test_a_spin_polarized_cluster(self) -> None:
        lines = [
            "Boundary_Sphere_Radius: 6.5", "Grid_Spacing: 0.5", "States_Num: 8", "Max_Iter: 300",
            "Convergence_Criterion: 1.0e-8", "Eigensolver: chebff", "Correlation_Type: ca", "Spin_Polarization: .true.",
            "Output_Level: 2", "Atom_Types_Num: 1", "Atom_Type: N", "Local_Component: p", "begin Atom_Coord",
            "  0.0 0.0 0.0", "end Atom_Coord", "Symmetrize_Charge_Density: .true.",
        ]
        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for setting in ("auto", "off"):
                directory = Path(name) / setting
                directory.mkdir(parents=True)
                (directory / "parsec.in").write_text("\n".join(lines) + "\n", encoding="utf-8")
                shutil.copy(DATA / "spin_ca_N_POTRE.DAT", directory / "N_POTRE.DAT")
                runs[setting] = _run_cli(directory / "parsec.in", {"PARSEC_SYMMETRIZED_DENSITY_SECTORS": setting})
        for label, (status, _, log, errors) in runs.items():
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-4000:]}")
        self.assertIn("orbital_symmetry = CuPy real", runs["auto"][2])
        self.assertNotIn("orbital_symmetry = CuPy real", runs["off"][2])
        self.assertIn("electron charge after symmetrization", runs["auto"][2])
        self.assertAlmostEqual(
            float(runs["auto"][1]["energy_total_ry"]), float(runs["off"][1]["energy_total_ry"]), delta=2e-6
        )


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class SymmetrizedPositionForceTests(unittest.TestCase):
    """Rounded inputs with PARSEC_SYMMETRIZE_POSITIONS=on: forces in the sectors of Td, a cluster and a cell (GPU)."""

    def test_forces_of_the_symmetrized_cluster(self) -> None:
        noisy = GE5.copy()
        noisy[1:, 2] *= 1.0 + 1.0e-5
        environment = {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4", "PARSEC_SYMMETRIZE_POSITIONS": "on"}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            sectors = _run_cli(_ge5_input(root / "sectors", positions=noisy, extra="Skip_force: .false."), environment)
            full = _run_cli(
                _ge5_input(root / "full", positions=noisy, extra="Skip_force: .false."),
                environment,
                options=("--backend", "cupy", "--symmetry", "off"),
            )
        for label, (status, _, log, errors) in (("sectors", sectors), ("full", full)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-4000:]}")
        self.assertIn("PARSEC_SYMMETRIZE_POSITIONS=on: the positions are averaged over the 24 operations", sectors[2])
        self.assertIn("weights 1, 1, 3, 3", sectors[2])
        self.assertGreater(float(np.abs(full[1]["forces_total"]).max()), 1.0e-3)
        np.testing.assert_allclose(sectors[1]["forces_total"], full[1]["forces_total"], rtol=0, atol=2e-6)
        self.assertAlmostEqual(float(sectors[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=1e-7)

    def test_forces_of_a_symmetrized_cell(self) -> None:
        # The displaced silicon cube (Td about the origin) with 1e-5 bohr of noise: the average over the group floats
        # by the mean of the noise, which the operations about the origin of the sectors and their 1e-6 bohr check of
        # the forces refused; shifted back onto the grid, the forces in the sectors are those of the full grid.
        from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon

        noisy = displaced_positions() + np.random.default_rng(5).uniform(-1e-5, 1e-5, (8, 3))
        noisy -= np.floor(noisy / 10.26) * 10.26
        environment = {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4", "PARSEC_SYMMETRIZE_POSITIONS": "on"}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            sectors = _run_cli(write_silicon(root / "sectors", "Skip_force: .false.\n", positions=noisy),
                               {**environment, "PARSEC_PERIODIC_SECTOR_FORCES": "auto"})
            full = _run_cli(write_silicon(root / "full", "Skip_force: .false.\n", positions=noisy),
                            {**environment, "PARSEC_PERIODIC_SECTOR_FORCES": "off"})
        for label, (status, _, log, errors) in (("sectors", sectors), ("full", full)):
            self.assertEqual(status, 0, f"{label}: {errors}\n{log[-4000:]}")
        for log in (sectors[2], full[2]):
            self.assertIn("PARSEC_SYMMETRIZE_POSITIONS=on: the positions are averaged over the 24 operations of the cell",
                          log)
            self.assertIn("which gives 23 point operations a translation that maps the grid onto itself", log)
        self.assertIn("Forces of the periodic cell in the sectors of its point group: averaged over 24 operations",
                      sectors[2])
        self.assertNotIn("Forces of the periodic cell in the sectors", full[2])
        self.assertGreater(float(np.abs(full[1]["forces_total"]).max()), 1.0e-3)
        np.testing.assert_allclose(sectors[1]["forces_total"], full[1]["forces_total"], rtol=0, atol=2e-6)
        self.assertAlmostEqual(float(sectors[1]["energy_total_ry"]), float(full[1]["energy_total_ry"]), delta=1e-7)


if __name__ == "__main__":
    unittest.main()
