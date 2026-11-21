"""A repeating Gamma-point cell as a smaller cell with k-points (``SCF/supercell_mesh.py``): the switches of the mesh
and of the wedge, the cells they build and the k-points of those cells."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Grid.kpoints import kpoint_sampling_details
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.SCF import kpoint_supercell, supercell_mesh

DIAMOND = Path(__file__).resolve().parents[4] / "examples" / "3d_diamond_benchmark"


def _make_diamond():
    spec = importlib.util.spec_from_file_location("_make_diamond_for_tests", DIAMOND / "make_diamond.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diamond_translation(n1: int, n2: int, n3: int, *, settings=(), directory: Path):
    """The Gamma-point input of make_diamond.py (cubes of diamond, 18 points per edge), parsed."""
    path = directory / f"d{n1}{n2}{n3}" / "parsec.in"
    argv = [str(n1), str(n2), str(n3), str(path)]
    for setting in settings:
        argv += ["--set", setting]
    with mock.patch("sys.argv", ["make_diamond.py", *argv]):
        _make_diamond().main()
    shutil.copy(DIAMOND / "C_POTRE.DAT", path.parent / "C_POTRE.DAT")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


def _switches(mesh: str | None = None, wedge: str | None = None, route: str = "auto") -> dict:
    values = {"PARSEC_SUPERCELL_KPOINTS": route}
    if mesh is not None:
        values[supercell_mesh.MESH_SWITCH] = mesh
    if wedge is not None:
        values[supercell_mesh.WEDGE_SWITCH] = wedge
    return values


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {supercell_mesh.MESH_SWITCH: "", supercell_mesh.WEDGE_SWITCH: ""}):
            self.assertEqual(supercell_mesh.supercell_mesh_setting(), "all")
            self.assertEqual(supercell_mesh.supercell_wedge_setting(), "auto")
        for value, expected in (("ALL", "all"), (" auto ", "auto"), ("3x3x3", (3, 3, 3)), ("2X1x5", (2, 1, 5))):
            with mock.patch.dict(os.environ, {supercell_mesh.MESH_SWITCH: value}):
                self.assertEqual(supercell_mesh.supercell_mesh_setting(), expected)
        for value in ("1x1x1", "3x3", "0x2x2", "-1x2x2", "axbxc", "on", "3"):
            with mock.patch.dict(os.environ, {supercell_mesh.MESH_SWITCH: value}):
                with self.assertRaises(ValueError):
                    supercell_mesh.supercell_mesh_setting()
        with mock.patch.dict(os.environ, {supercell_mesh.WEDGE_SWITCH: "on"}):
            with self.assertRaises(ValueError):
                supercell_mesh.validate_settings()
        with mock.patch.dict(os.environ, {"PARSEC_SUPERCELL_KPOINTS": "yes"}):
            with self.assertRaises(ValueError):
                supercell_mesh.validate_settings()


class AutoMeshTests(unittest.TestCase):
    """The mesh of ``PARSEC_SUPERCELL_KPOINTS_MESH=auto`` for cubic diamond supercells of ``n^3`` cubes (18 points per
    cube edge, ``8 n^3`` atoms, ``16 n^3 + 6`` states): the 64-atom cell holds 6.3e6 entries (above ``2^22``), the cube
    1.6e5."""

    @staticmethod
    def _mesh(n: int, sides=None):
        from types import SimpleNamespace

        side = 6.7406530865
        lattice = np.diag(np.asarray(sides if sides is not None else (n * side,) * 3, dtype=float))
        problem = SimpleNamespace(scf=SimpleNamespace(number_of_states=16 * n ** 3 + 6),
                                  periodic_cell=SimpleNamespace(lattice_vectors=lattice))
        return supercell_mesh._auto_mesh(problem, (n, n, n), np.array([18 * n] * 3))

    def test_the_largest_mesh_of_cells_large_enough(self) -> None:
        # the cell of 2 x 2 x 2 cubes where 2 divides n and leaves a mesh above two, else of 3 x 3 x 3
        for n, mesh in ((6, (3, 3, 3)), (10, (5, 5, 5)), (12, (6, 6, 6)), (9, (3, 3, 3)), (15, (5, 5, 5)),
                        (21, (7, 7, 7)), (18, (9, 9, 9))):
            self.assertEqual(self._mesh(n), mesh, n)

    def test_the_cube_with_all_repeats_or_the_gamma_point(self) -> None:
        # no larger cell: the cube with the mesh of all the repeats; a mesh of twos alone is the Gamma route's
        for n, mesh in ((5, (5, 5, 5)), (7, (7, 7, 7)), (3, (3, 3, 3)), (4, (4, 4, 4)), (2, None), (1, None)):
            self.assertEqual(self._mesh(n), mesh, n)

    def test_the_order_in_which_the_meshes_are_tried(self) -> None:
        # the large cells (most points first), the mesh of all the repeats, then the smaller cells (most points first),
        # which a smallest cell too small for the stencil falls back to
        from types import SimpleNamespace

        def order(n):
            lattice = np.diag([n * 6.7406530865] * 3)
            problem = SimpleNamespace(scf=SimpleNamespace(number_of_states=16 * n ** 3 + 6),
                                      periodic_cell=SimpleNamespace(lattice_vectors=lattice))
            return supercell_mesh._auto_meshes(problem, (n, n, n), np.array([18 * n] * 3))

        self.assertEqual(order(12), [(6, 6, 6), (4, 4, 4), (3, 3, 3), (12, 12, 12)])
        self.assertEqual(order(6), [(3, 3, 3), (6, 6, 6)])
        self.assertEqual(order(4), [(4, 4, 4)])
        self.assertEqual(order(2), [])

        # 12 points per cube edge: the cell of 2 x 2 x 2 cubes is below the floor, after the cube
        lattice = np.diag([6 * 6.7406530865] * 3)
        problem = SimpleNamespace(scf=SimpleNamespace(number_of_states=16 * 216 + 6),
                                  periodic_cell=SimpleNamespace(lattice_vectors=lattice))
        self.assertEqual(supercell_mesh._auto_meshes(problem, (6, 6, 6), np.array([72] * 3)), [(6, 6, 6), (3, 3, 3)])


class SupercellMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="parsec_supercell_mesh_"))
        cls.d222 = diamond_translation(2, 2, 2, directory=cls.directory)
        cls.d333 = diamond_translation(3, 3, 3, directory=cls.directory)
        cls.d621 = diamond_translation(6, 2, 1, directory=cls.directory)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, _switches(route="off")):
            self.assertIsNone(supercell_mesh.supercell_mesh_route(self.d333))

    def test_the_cell_of_an_odd_repeat_takes_its_grid(self) -> None:
        # 54 points along the side of three cubes: the cube's 18 under PARSEC's rule of the points, which a spacing
        # a round-off above the step gave 16 (the route did not apply before 2026-11-19)
        with mock.patch.dict(os.environ, _switches(wedge="off")):
            cell, repeats = kpoint_supercell.kpoints_of_supercell(self.d333.problem)
        self.assertEqual(repeats, (3, 3, 3))
        np.testing.assert_array_equal(kpoint_supercell._grid_counts(cell), [18, 18, 18])
        self.assertEqual(len(cell.atoms), 8)

    def test_the_mesh_of_all_repeats_keeps_the_former_cell(self) -> None:
        with mock.patch.dict(os.environ, _switches(wedge="off")):
            route = supercell_mesh.supercell_mesh_route(self.d333)
            former, _ = kpoint_supercell.kpoints_of_supercell(self.d333.problem)
        self.assertEqual((route.mesh, route.repeats, route.wedge), ((3, 3, 3), (3, 3, 3), None))
        self.assertIs(route.problem.space_group, None)
        self.assertEqual(repr(route.problem), repr(former))
        # time reversal alone: (27 + 1) / 2 points
        self.assertEqual(len(kpoint_sampling_details(route.problem).k_points), 14)

    def test_the_wedge_of_the_cell(self) -> None:
        for translation, points in ((self.d333, 4), (self.d222, 4)):
            with mock.patch.dict(os.environ, _switches()):
                route = supercell_mesh.supercell_mesh_route(translation)
            # Td about an atom of the cube with 18 points per edge, with time reversal the wedge of Oh
            self.assertEqual(route.wedge, 24)
            self.assertTrue(route.problem.scf.symmetrize_density)
            sampling = kpoint_sampling_details(route.problem)
            self.assertEqual(len(sampling.k_points), points)
            self.assertAlmostEqual(float(np.sum(sampling.weights)), 1.0, places=12)
            self.assertIn("irreducible wedge of 24 operations", route.describe())

    def test_a_mesh_dividing_the_repeats(self) -> None:
        with mock.patch.dict(os.environ, _switches(mesh="3x1x1")):
            route = supercell_mesh.supercell_mesh_route(self.d621)
        self.assertEqual((route.mesh, route.repeats), ((3, 1, 1), (6, 2, 1)))
        lattice = np.asarray(route.problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(np.diag(lattice), np.diag(np.asarray(self.d621.problem.periodic_cell.lattice_vectors)) / [3, 1, 1])
        np.testing.assert_array_equal(kpoint_supercell._grid_counts(route.problem), [36, 36, 18])
        self.assertEqual(len(route.problem.atoms), len(self.d621.problem.atoms) // 3)
        states = int(self.d621.problem.scf.number_of_states)
        self.assertEqual(route.problem.scf.number_of_states, -(-states // 3) + 6)
        self.assertEqual(route.problem.monkhorst_pack_dimensions, (3, 1, 1))
        self.assertIn("the cell of 2 x 2 x 1 of them", route.describe())
        with mock.patch.dict(os.environ, _switches(mesh="4x1x1")):
            with self.assertRaisesRegex(ValueError, "must divide these repeats"):
                supercell_mesh.supercell_mesh_route(self.d621)

    def test_the_full_scf_runner_refuses_the_route_by_name(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        with mock.patch.dict(os.environ, _switches()):
            with self.assertRaisesRegex(ValueError, r"PARSEC_SUPERCELL_KPOINTS=auto: .*main\.py --mpi"):
                mpi_full_scf._refuse_unsupported_periodic(self.d333)
            # the supercell of a k-point mesh (PARSEC_KPOINT_SUPERCELL) is the runner's own
            mpi_full_scf._refuse_unsupported_periodic(self.d333, supercell=True)
        with mock.patch.dict(os.environ, _switches(route="off")):
            mpi_full_scf._refuse_unsupported_periodic(self.d333)
        with mock.patch.dict(os.environ, {**_switches(route="off"), supercell_mesh.MESH_SWITCH: "2x2"}):
            with self.assertRaisesRegex(ValueError, supercell_mesh.MESH_SWITCH):
                mpi_full_scf._refuse_unsupported_periodic(self.d333)

    def test_moving_atoms_and_ignore_symmetry_keep_time_reversal(self) -> None:
        from dataclasses import replace

        for translation in (replace(self.d333, compute_forces=True), replace(self.d333, ignore_symmetry=True)):
            with mock.patch.dict(os.environ, _switches()):
                route = supercell_mesh.supercell_mesh_route(translation)
            self.assertIsNone(route.wedge)
            self.assertIsNone(route.problem.space_group)

    def test_the_supercell_grid_is_not_built(self) -> None:
        # the counts of the supercell come from the rule of the points; only the grids of the cells are built
        from parsec_python.Grid import pbc

        built = []
        original = pbc.build_periodic_grid

        def recording(cell, settings, **options):
            grid = original(cell, settings, **options)
            built.append(int(np.prod(grid.points_per_axis)))
            return grid

        with mock.patch.object(pbc, "build_periodic_grid", recording):
            with mock.patch.dict(os.environ, _switches(mesh="auto")):
                route = supercell_mesh.supercell_mesh_route(self.d333)
        self.assertIsNotNone(route)
        self.assertTrue(built)
        self.assertLess(max(built), 54 ** 3)

    def test_without_a_state_count_the_cell_takes_its_default(self) -> None:
        # States_Num is required by the reader, but a translation made in code may leave it to the SCF's default
        from dataclasses import replace

        problem = self.d333.problem
        unset = replace(self.d333, problem=replace(problem, scf=replace(problem.scf, number_of_states=None)))
        with mock.patch.dict(os.environ, _switches()):
            route = supercell_mesh.supercell_mesh_route(unset)
        self.assertIsNone(route.problem.scf.number_of_states)
        # the default ceil(864/2) + 6 of the supercell, from the valence charges of its pseudopotentials
        self.assertEqual(supercell_mesh._supercell_states(unset.problem), 438)
        counts = np.array([54, 54, 54])
        self.assertEqual(supercell_mesh._auto_meshes(unset.problem, (3, 3, 3), counts),
                         supercell_mesh._auto_meshes(problem, (3, 3, 3), counts))


A_BOHR = 3.567 / 0.529177210903


class StencilLimitTests(unittest.TestCase):
    """Cells at the limit of the stencil of ``Expansion_Order`` 12 (more than 12 points per axis): the cube of diamond
    with 14 points per edge, whose spacing first rounds to 12 points, and with 12 points, too small."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="parsec_supercell_stencil_"))

        def make(n, per_edge, label):
            spacing = A_BOHR / per_edge * (1.0 - 1.0e-9)
            settings = (f"Grid_Spacing: {spacing:.14f}", "Expansion_Order: 12")
            return diamond_translation(n, n, n, settings=settings, directory=cls.directory / label)

        cls.d333_14 = make(3, 14, "p14")
        cls.d333_12 = make(3, 12, "p12")
        cls.d666_12 = make(6, 12, "p12")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_a_first_spacing_below_the_stencil_tries_the_next(self) -> None:
        with mock.patch.dict(os.environ, _switches()):
            route = supercell_mesh.supercell_mesh_route(self.d333_14)
            cell, repeats = kpoint_supercell.kpoints_of_supercell(self.d333_14.problem)
        self.assertEqual((route.mesh, repeats), ((3, 3, 3), (3, 3, 3)))
        np.testing.assert_array_equal(kpoint_supercell._grid_counts(route.problem), [14, 14, 14])
        np.testing.assert_array_equal(kpoint_supercell._grid_counts(cell), [14, 14, 14])

    def test_a_cell_too_small_for_the_stencil_leaves_the_gamma_point(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        notes: list[str] = []
        with mock.patch.dict(os.environ, _switches()):
            self.assertIsNone(supercell_mesh.supercell_mesh_route(self.d333_12, notes=notes))
            self.assertIsNone(kpoint_supercell.kpoints_of_supercell(self.d333_12.problem))
            # the runner runs it at the Gamma point, as without the switch
            mpi_full_scf._refuse_unsupported_periodic(self.d333_12)
        self.assertEqual(len(notes), 1)
        self.assertIn("too small for this stencil", notes[0])
        self.assertIn("Gamma point of the supercell", notes[0])
        # the mesh 3x3x3 of three cubes has no larger cell than the cube: an explicit one is an error
        with mock.patch.dict(os.environ, _switches(mesh="3x3x3")):
            with self.assertRaisesRegex(ValueError, "too small for this stencil"):
                supercell_mesh.supercell_mesh_route(self.d333_12)
        with mock.patch.dict(os.environ, _switches(mesh="auto")):
            notes = []
            self.assertIsNone(supercell_mesh.supercell_mesh_route(self.d333_12, notes=notes))
            self.assertIn("too small for this stencil", notes[0])

    def test_the_automatic_mesh_takes_a_larger_cell(self) -> None:
        with mock.patch.dict(os.environ, _switches(mesh="auto")):
            route = supercell_mesh.supercell_mesh_route(self.d666_12)
        # the cube (12 points) is too small: the cell of 2 x 2 x 2 cubes with the 3 x 3 x 3 mesh
        self.assertEqual((route.mesh, route.repeats), ((3, 3, 3), (6, 6, 6)))
        np.testing.assert_array_equal(kpoint_supercell._grid_counts(route.problem), [24, 24, 24])
        with mock.patch.dict(os.environ, _switches(mesh="6x6x6")):
            with self.assertRaisesRegex(ValueError, "too small for this stencil"):
                supercell_mesh.supercell_mesh_route(self.d666_12)
        with mock.patch.dict(os.environ, _switches()):
            self.assertIsNone(supercell_mesh.supercell_mesh_route(self.d666_12))


class ChargedSupercellTests(unittest.TestCase):
    """``Net_Charges`` is that of the supercell: each cell of ``K`` takes ``1/K`` of it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(tempfile.mkdtemp(prefix="parsec_supercell_charge_"))
        cls.d333 = diamond_translation(3, 3, 3, settings=("Net_Charges: -1",), directory=cls.directory)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_the_charge_of_each_cell(self) -> None:
        self.assertEqual(self.d333.problem.scf.net_charge, -1.0)
        for mesh, cells in (("all", 27), ("3x1x1", 3), ("3x3x1", 9)):
            with mock.patch.dict(os.environ, _switches(mesh=mesh)):
                route = supercell_mesh.supercell_mesh_route(self.d333)
            self.assertEqual(route.cells, cells)
            self.assertAlmostEqual(route.problem.scf.net_charge, -1.0 / cells, places=15)
            self.assertIn(f"Net_Charges {-1.0 / cells:.10g} in the cell, 1/{cells} of the supercell's",
                          route.describe())
        with mock.patch.dict(os.environ, _switches()):
            cell, _ = kpoint_supercell.kpoints_of_supercell(self.d333.problem)
        self.assertAlmostEqual(cell.scf.net_charge, -1.0 / 27.0, places=15)


if __name__ == "__main__":
    unittest.main()
