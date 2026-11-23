"""The fast preparation of wires and slabs (:mod:`parsec_python.acceleration.V_ion.confined_setup`) is bit for bit.

* ``confined_ball`` returns the rows, displacements and distances of the dense scan, in its order;
* the device sum of the superposed valence and core densities equals the readable sum bit for bit;
* the short-ranged local potential with its images on threads equals the readable loop bit for bit;
* the Bessel tables of the wire Hartree boundary, once per distinct radius, equal the direct ones;
* a whole preparation under ``PARSEC_CONFINED_SETUP=auto`` equals the one under ``readable``;
* the switch refuses an unknown value.

Cells: an oblique (hexagonal) slab and a wire, a FHI silicon table (75 bohr) with an H table carrying a core
correction, atoms off the grid points and one on a point.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the providers)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.V_ion import confined_setup
from parsec_python.Grid.confined import build_confined_periodic_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import ionic_potential as _ionic
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.V_ion import wire_slab as _wire_slab

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_SPECIES = """
Atom_Type: Si
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 2.1
begin Electron_Per_Orbital
2 2 0
end Electron_Per_Orbital
begin Atom_Coord
{si}
end Atom_Coord

Atom_Type: H
Local_Component: s
begin Atom_Coord
{h}
end Atom_Coord
"""


def _text(kind: str) -> str:
    if kind == "slab":
        # a hexagonal slab: the general (oblique) grid of a slab
        head = (
            "Boundary_Conditions: slab\nbegin Cell_Shape\n  4.2 0.0\n  -2.1 3.6373066958946424\nend Cell_Shape\n"
            "Boundary_Sphere_Radius: 4.5\n"
        )
        si = "  0.31 0.27 0.40\n  -1.13 1.95 -0.52"
        h = "  1.07 0.62 1.85\n  0.0 0.0 0.0"
    else:
        head = "Boundary_Conditions: wire\nbegin Cell_Shape\n  4.4\nend Cell_Shape\nBoundary_Sphere_Radius: 4.0\n"
        si = "  0.31 0.27 0.40\n  -1.43 0.15 -0.52"
        h = "  1.07 -1.62 1.05\n  0.0 0.0 0.0"
    return (
        head
        + "Grid_Spacing: 0.45\nExpansion_Order: 4\nStates_Num: 8\nCorrelation_Type: ca\nCoordinate_Unit: Cartesian_Bohr\n"
        + "Atom_Types_Num: 2\n"
        + _SPECIES.format(si=si, h=h)
    )


def _problem(kind: str):
    folder = Path(tempfile.mkdtemp(prefix=f"confined_{kind}_"))
    shutil.copy(DATA / "Si_FHIPP.DAT", folder / "Si_FHIPP.DAT")
    shutil.copy(DATA / "H_CORE_POTRE.DAT", folder / "H_POTRE.DAT")
    (folder / "parsec.in").write_text(_text(kind), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = parse_parsec_input(folder / "parsec.in").problem
    _FOLDERS.append(folder)  # the tables are read when the potentials are loaded
    return problem


_FOLDERS: list[Path] = []


def tearDownModule() -> None:
    for folder in _FOLDERS:
        shutil.rmtree(folder, ignore_errors=True)


class _Cases:
    cache: dict = {}

    @classmethod
    def get(cls, kind: str):
        if kind not in cls.cache:
            problem = _problem(kind)
            grid = build_confined_periodic_grid(problem.periodic_cell, problem.grid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                potentials = load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
            cls.cache[kind] = (problem, grid, potentials)
        return cls.cache[kind]


def _readable():
    return mock.patch.dict(os.environ, {confined_setup.SWITCH: "readable"})


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), (" Readable ", "readable")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {confined_setup.SWITCH: value}):
                self.assertEqual(confined_setup.confined_setup_setting(), expected)
        with mock.patch.dict(os.environ, {confined_setup.SWITCH: "fast"}):
            with self.assertRaises(ValueError):
                confined_setup.confined_setup_setting()

    def test_readable_turns_the_sums_off(self) -> None:
        _, grid, _ = _Cases.get("slab")
        self.assertTrue(confined_setup.handles(grid))
        with _readable():
            self.assertFalse(confined_setup.handles(grid))

    def test_the_providers_are_installed(self) -> None:
        self.assertIs(_wire_slab._short_range_provider, confined_setup.threaded_short_range_local_potential)


class BallTests(unittest.TestCase):
    def test_the_balls_are_those_of_the_dense_scan(self) -> None:
        rng = np.random.default_rng(7)
        for kind in ("slab", "wire"):
            _, grid, _ = _Cases.get(kind)
            lattice = np.asarray(grid.cell.lattice_vectors, dtype=float)
            periodic = grid.periodicity
            centres = [np.zeros(3), grid.coordinates[17].copy(), grid.coordinates[grid.size // 2].copy()]
            for _ in range(12):
                centre = rng.uniform(-5.0, 5.0, size=3)
                # an image a few cells away along the periodic vectors
                shift = rng.integers(-2, 3, size=periodic).astype(float) @ lattice[:periodic]
                centres.append(centre + shift)
            for centre in centres:
                for radius in (0.45, 1.3, 2.7, 6.5):
                    with self.subTest(kind=kind, centre=centre.tolist(), radius=radius):
                        rows, relative, distances = confined_setup.confined_ball(grid, centre, radius)
                        dense_rows, dense_relative, dense_distances = _ionic.dense_support(grid, centre, radius)
                        np.testing.assert_array_equal(rows, dense_rows)
                        np.testing.assert_array_equal(relative, dense_relative)
                        np.testing.assert_array_equal(distances, dense_distances)

    def test_a_ball_of_most_of_the_grid_is_the_dense_scan(self) -> None:
        _, grid, _ = _Cases.get("slab")
        centre = np.array([0.2, 0.1, 0.3])
        for radius, dense in ((75.0, 1), (0.9, 0)):
            with self.subTest(radius=radius), mock.patch.object(
                confined_setup, "_dense_ball", wraps=confined_setup._dense_ball
            ) as scan:
                found = confined_setup.confined_ball(grid, centre, radius)
                self.assertEqual(scan.call_count, dense)
            for value, reference in zip(found, _ionic.dense_support(grid, centre, radius)):
                np.testing.assert_array_equal(value, reference)

    def test_the_installed_finder_takes_the_box(self) -> None:
        _, grid, _ = _Cases.get("slab")
        with mock.patch.object(confined_setup, "confined_ball", wraps=confined_setup.confined_ball) as ball:
            _ionic._support_finder(grid, np.array([0.2, 0.1, 0.3]), 2.0)
        self.assertEqual(ball.call_count, 1)
        with _readable(), mock.patch.object(confined_setup, "confined_ball") as ball:
            _ionic._support_finder(grid, np.array([0.2, 0.1, 0.3]), 2.0)
        self.assertEqual(ball.call_count, 0)


class ShortRangeTests(unittest.TestCase):
    def test_the_workers_fit_the_memory_bound(self) -> None:
        from parsec_python.acceleration.V_ion import periodic_setup_fortran

        problem, grid, potentials = _Cases.get("slab")
        array = 8 * grid.size
        seen = []
        original = periodic_setup_fortran._ordered_results

        def record(work, items, threads):
            seen.append(threads)
            return original(work, items, threads)

        readable = None
        with _readable():
            readable = _wire_slab.short_range_local_potential(grid, problem.atoms, potentials, problem.pseudopotentials)
        for budget, expected in ((6 * array + 2 * 13 * array, 2), (6 * array, 1), (1 << 40, 8)):
            with self.subTest(budget=budget), mock.patch.object(confined_setup, "_PENDING_BYTES", budget), \
                    mock.patch.object(periodic_setup_fortran, "_ordered_results", record), \
                    mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": "8"}):
                found = confined_setup.threaded_short_range_local_potential(
                    grid, problem.atoms, potentials, problem.pseudopotentials
                )
                self.assertEqual(seen[-1], expected)
                np.testing.assert_array_equal(found.potential, readable.potential)


    def test_the_threaded_sum_is_the_readable_loop(self) -> None:
        for kind in ("slab", "wire"):
            problem, grid, potentials = _Cases.get(kind)
            for threads in ("1", "3"):
                with self.subTest(kind=kind, threads=threads), mock.patch.dict(
                    os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}
                ):
                    fast = confined_setup.threaded_short_range_local_potential(
                        grid, problem.atoms, potentials, problem.pseudopotentials
                    )
                    with _readable():
                        self.assertIsNone(
                            confined_setup.threaded_short_range_local_potential(
                                grid, problem.atoms, potentials, problem.pseudopotentials
                            )
                        )
                        readable = _wire_slab.short_range_local_potential(
                            grid, problem.atoms, potentials, problem.pseudopotentials
                        )
                    np.testing.assert_array_equal(fast.potential, readable.potential)
                    np.testing.assert_array_equal(fast.gaussian_density, readable.gaussian_density)
                    self.assertEqual(fast.grid_charge, readable.grid_charge)
                    self.assertEqual(fast.nominal_charge, readable.nominal_charge)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceDensityTests(unittest.TestCase):
    def test_the_device_densities_are_the_readable_sums(self) -> None:
        from dataclasses import replace

        for kind in ("slab", "wire"):
            problem, grid, potentials = _Cases.get(kind)
            lattice = problem.periodic_cell.lattice_vectors
            # linear tables (a species without Cubic_Spline takes the spline by default, whose core density keeps
            # the host sum: the test below)
            specifications = {
                symbol: replace(specification, use_spline=False)
                for symbol, specification in problem.pseudopotentials.items()
            }
            for core in (False, True):
                with self.subTest(kind=kind, core=core):
                    device = confined_setup.device_confined_atomic_density(
                        grid, problem.atoms, potentials, specifications, lattice, core
                    )
                    self.assertIsNotNone(device)
                    with _readable():
                        readable = _ionic.superpose_atomic_density(
                            grid, problem.atoms, potentials, specifications, lattice, core=core
                        )
                    self.assertGreater(float(np.max(readable)), 0.0)
                    np.testing.assert_array_equal(device, readable)

    def test_a_spline_core_density_keeps_the_readable_sum(self) -> None:
        from dataclasses import replace

        problem, grid, potentials = _Cases.get("slab")
        specifications = dict(problem.pseudopotentials)
        specifications["H"] = replace(specifications["H"], use_spline=True)
        self.assertIsNone(
            confined_setup.device_confined_atomic_density(
                grid, problem.atoms, potentials, specifications, problem.periodic_cell.lattice_vectors, True
            )
        )


class WireBoundaryTests(unittest.TestCase):
    def test_the_bessel_rows_are_the_direct_tables(self) -> None:
        from scipy.special import ive, kve

        from parsec_python.Hartree import wire as _wire
        from parsec_python.Hartree.confined import exterior_stencil

        problem, grid, _ = _Cases.get("wire")
        order = 5
        plan = _wire.build_wire_boundary_plan(grid, exterior_stencil(grid).coordinates, order)
        wave = 2.0 * np.pi / grid.period
        n = np.arange(1, order + 1)
        rho = np.sqrt(grid.coordinates[:, 1] ** 2 + grid.coordinates[:, 2] ** 2)
        argument = wave * n[None, :] * rho[:, None]
        scale = np.exp(argument - wave * n[None, :] * grid.confinement_radius)
        np.testing.assert_array_equal(plan.source_bessel, _wire._bessel_table(ive, order, argument) * scale[:, :, None])
        radius = plan.boundary_radius
        boundary_argument = wave * n[None, :] * radius[:, None]
        boundary_scale = np.exp(-(boundary_argument - wave * n[None, :] * grid.confinement_radius))
        np.testing.assert_array_equal(
            plan.boundary_bessel, _wire._bessel_table(kve, order, boundary_argument) * boundary_scale[:, :, None]
        )
        # the grid has lines along the axis: far fewer distinct radii than points
        self.assertLess(np.unique(rho).size * 4, rho.size)


class SlabBoundaryTests(unittest.TestCase):
    def test_the_phases_per_column_are_the_direct_ones(self) -> None:
        from parsec_python.Hartree.confined import exterior_stencil
        from parsec_python.Hartree.slab_fortran import build_slab_boundary_plan, slab_reciprocal_half_plane

        _, grid, _ = _Cases.get("slab")
        order = 4
        stencil = exterior_stencil(grid)
        plan = build_slab_boundary_plan(grid, stencil.integer_points, order)
        vectors = slab_reciprocal_half_plane(order)
        wave = 2.0 * np.pi / np.sqrt(np.sum(grid.cell.lattice_vectors[:2] ** 2, axis=1))
        wrapped = grid.wrap_periodic(np.asarray(stencil.integer_points, dtype=np.int64))
        eu = (wrapped[:, 0] + grid.shift[0]) * grid.spacings[0]
        ev = (wrapped[:, 1] + grid.shift[1]) * grid.spacings[1]
        phase = eu[:, None] * vectors[None, :, 0] * wave[0] + ev[:, None] * vectors[None, :, 1] * wave[1]
        np.testing.assert_array_equal(plan.boundary_cos, np.cos(phase))
        np.testing.assert_array_equal(plan.boundary_sin, np.sin(phase))


class GridTests(unittest.TestCase):
    def test_rows_by_columns_are_those_of_the_wrapped_points(self) -> None:
        rng = np.random.default_rng(11)
        for kind in ("slab", "wire"):
            _, grid, _ = _Cases.get(kind)
            for _ in range(20):
                points = grid.integer_coordinates + rng.integers(-40, 41, size=3)
                local = grid.wrap_periodic(points) - grid.index_min
                valid = np.all((local >= 0) & (local < grid.points_per_axis), axis=1)
                expected = np.full(points.shape[0], -1, dtype=np.int64)
                inside = local[valid]
                expected[valid] = grid.lookup[inside[:, 0], inside[:, 1], inside[:, 2]]
                np.testing.assert_array_equal(grid.rows_for_integer_coordinates(points), expected)

    def test_the_gradient_is_built_when_first_read(self) -> None:
        from parsec_python.Laplacian import build_gradient
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        problem, grid, _ = _Cases.get("wire")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_wire_slab_single_point(problem)
        self.assertFalse(system.gradient.built)
        for lazy, direct in zip(system.gradient, build_gradient(grid)):
            self.assertEqual((lazy != direct).nnz, 0)
            np.testing.assert_array_equal(lazy.indptr, direct.indptr)
            np.testing.assert_array_equal(lazy.indices, direct.indices)
            np.testing.assert_array_equal(lazy.data, direct.data)


class PreparationTests(unittest.TestCase):
    def test_auto_prepares_the_readable_fields(self) -> None:
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        for kind in ("slab", "wire"):
            problem, _, _ = _Cases.get(kind)
            with self.subTest(kind=kind):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fast = prepare_wire_slab_single_point(problem)
                    with _readable():
                        readable = prepare_wire_slab_single_point(problem)
                for name in ("ionic_potential", "initial_density", "core_density", "gaussian_potential"):
                    np.testing.assert_array_equal(getattr(fast, name), getattr(readable, name), err_msg=name)
                self.assertEqual(fast.ion_ion_energy, readable.ion_ion_energy)
                a = fast.nonlocal_operator.projectors.tocsr()
                b = readable.nonlocal_operator.projectors.tocsr()
                a.sort_indices()
                b.sort_indices()
                np.testing.assert_array_equal(a.indptr, b.indptr)
                np.testing.assert_array_equal(a.indices, b.indices)
                np.testing.assert_array_equal(a.data, b.data)
                np.testing.assert_array_equal(fast.nonlocal_operator.signs, readable.nonlocal_operator.signs)


if __name__ == "__main__":
    unittest.main()
