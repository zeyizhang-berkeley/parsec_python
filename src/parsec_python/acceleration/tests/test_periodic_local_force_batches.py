"""The local forces of periodic cells without weighted sectors: batched ball sums and the orbits of other groups.

``PARSEC_PERIODIC_LOCAL_BALLS=auto`` forms the ball sums of many atoms per call on a GPU (an orthorhombic cell) where
the loop formed the balls of one atom at a time on the host: the same terms summed in another order, so the forces
agree to round-off.  On the host the batched sums are run by forcing the route (the default keeps the loop there).
``PARSEC_PERIODIC_LOCAL_ORBITS=auto`` sums one atom per orbit where the density of the run is averaged over a group
that is not that of weighted sectors: the irreducible wedge of a k-point mesh (``Forces.total.compute_periodic_forces``)
and the Abelian sectors of a Gamma cell (``acceleration.Forces.compute.compute_periodic_forces``).
"""

from __future__ import annotations

from dataclasses import replace
import os
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.acceleration.Forces import periodic_local
from parsec_python.acceleration.Forces.periodic_local import (
    fast_periodic_local_forces,
    orbit_local_forces,
    periodic_local_balls_setting,
    periodic_local_orbits_setting,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.Grid import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_NA, _GE = _DATA / "Na_POTRE.DAT", _DATA / "Ge_POTRE.DAT"
_CARBON = Path(__file__).resolve().parents[4] / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_RECTANGULAR = np.diag([7.7, 8.4, 9.1])
_SHIFTS = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.3, 0.8))
_SPECIES = (("Na", _NA, 0, False), ("Ge", _GE, 1, True), ("Ge", _GE, 1, False))


def _fields(grid, lattice):
    """Smooth, arbitrary density and XC fields without any symmetry."""
    phase = 2.0 * np.pi * (grid.coordinates @ np.linalg.inv(lattice))
    density = 0.3 + 0.2 * np.cos(phase[:, 0] + 0.4) * np.sin(phase[:, 1]) + 0.1 * np.sin(phase[:, 2] - phase[:, 0])
    xc = -0.5 + 0.3 * np.sin(phase[:, 0]) * np.cos(phase[:, 2] + 0.3) + 0.2 * np.cos(phase[:, 1] - 2 * phase[:, 2])
    return density, xc


def _arguments(lattice, spacing, shift, species, count, seed=3):
    symbol, path, local, spline = species
    potential = read_parsec_pseudopotential(path)
    grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=spacing, expansion_order=6, shift=shift))
    density, xc = _fields(grid, lattice)
    atoms = [Atom(symbol, tuple(position)) for position in np.random.default_rng(seed).random((count, 3)) @ lattice]
    return grid, atoms, {symbol: potential}, {symbol: SpeciesPotential(path, local, use_spline=spline)}, lattice, density, xc


def _six_sample_arguments(lattice, shift):
    """Four atoms with the six-sample test table with a core density (``H_CORE_POTRE.DAT``, splines), the first on a
    grid point."""
    grid, atoms, *rest = _arguments(lattice, 0.7, shift, ("H", _DATA / "H_CORE_POTRE.DAT", 0, True), 4)
    atoms[0] = Atom("H", tuple(grid.coordinates[17]))
    return (grid, atoms, *rest)


def _batched(arguments, *, xp=np, erf=None, points=None):
    """fast_periodic_local_forces with the batched ball sums forced (on any module)."""
    patches = [mock.patch.object(periodic_local, "_uses_batched_balls", return_value=True)]
    if points is not None:
        patches.append(mock.patch.object(periodic_local, "_BATCH_POINTS", points))
    for patch in patches:
        patch.start()
    try:
        return fast_periodic_local_forces(*arguments, xp=xp, erf=erf)
    finally:
        for patch in reversed(patches):
            patch.stop()


def _assert_round_off(test, value, reference, label, *, relative=1.0e-13):
    scale = max(float(np.max(np.abs(reference.ionic))), 1.0)
    test.assertGreater(float(np.max(np.abs(reference.ionic))), 1.0e-3, label)
    test.assertLessEqual(float(np.max(np.abs(value.ionic - reference.ionic))), relative * 10.0 * scale, label)
    test.assertLessEqual(float(np.max(np.abs(value.core_correction - reference.core_correction))), relative * 10.0 * scale, label)
    np.testing.assert_array_equal(value.self_consistency, 0.0)


class BatchedBallTests(unittest.TestCase):
    """The batched ball sums (forced on the host) against the per-atom loop."""

    def test_the_batched_sums_are_those_of_the_loop(self) -> None:
        # a cell smaller than the balls (overlapping images), grids off the origin, a species without and one with a
        # partial core (its density by spline and by the derivative table)
        for shift in _SHIFTS:
            for species in _SPECIES:
                arguments = _arguments(_RECTANGULAR, 0.7, shift, species, 5)
                label = f"{species[0]} spline={species[3]} shift={shift}"
                with self.subTest(label):
                    loop = fast_periodic_local_forces(*arguments)
                    _assert_round_off(self, _batched(arguments), loop, label)
                    if species[0] == "Ge":
                        self.assertGreater(float(np.max(np.abs(loop.core_correction))), 1.0e-2, label)

    def test_batches_of_any_size_give_the_same_sums(self) -> None:
        arguments = _arguments(_RECTANGULAR, 0.7, _SHIFTS[2], _SPECIES[1], 7)
        loop = fast_periodic_local_forces(*arguments)
        whole = _batched(arguments)
        # one atom per batch with its box cut along the first axis; a few atoms per batch
        for points in (1, 3_000, 1 << 16):
            with self.subTest(points=points):
                _assert_round_off(self, _batched(arguments, points=points), loop, f"{points} points")
                _assert_round_off(self, _batched(arguments, points=points), whole, f"{points} points")

    def test_a_diamond_cell_and_two_species(self) -> None:
        a = 3.567 / 0.529177210903
        lattice = np.diag([2.0 * a] * 3)
        arguments = _arguments(lattice, a / 18.0 * (1.0 - 1.0e-9), (0.0, 0.0, 0.0), ("C", _CARBON, 2, True), 6)
        grid, atoms, potentials, specifications, _, density, xc = arguments
        potentials = {**potentials, "Ge": read_parsec_pseudopotential(_GE)}
        specifications = {**specifications, "Ge": SpeciesPotential(_GE, 1, use_spline=True)}
        atoms = atoms[:3] + [Atom("Ge", atoms[3].position)] + atoms[4:] + [Atom("Ge", (1.0, 2.0, 3.0))]
        arguments = (grid, atoms, potentials, specifications, lattice, density, xc)
        loop = fast_periodic_local_forces(*arguments)
        _assert_round_off(self, _batched(arguments), loop, "C + Ge")
        self.assertEqual(float(np.max(np.abs(loop.core_correction[[0, 1, 2, 4, 5]]))), 0.0)

    def test_a_table_of_six_samples_and_an_atom_on_a_grid_point(self) -> None:
        # forcloc.f90's tail radius rs(npoint-6) lies at the origin of this table: the slopes switch to the Coulomb one
        # (and the core slope to zero) at rs(ns-1), where the sampled tables do, so an atom on a grid point is finite
        from parsec_python.Forces.periodic_local import dense_periodic_local_forces

        for shift in _SHIFTS[:2]:
            for lattice in (_RECTANGULAR, 6.6 * (np.ones((3, 3)) - np.eye(3))):
                arguments = _six_sample_arguments(lattice, shift)
                with self.subTest(shift=shift, oblique=bool(lattice[0, 0] == 0.0)):
                    loop = fast_periodic_local_forces(*arguments)
                    self.assertTrue(np.all(np.isfinite(loop.ionic)) and np.all(np.isfinite(loop.core_correction)))
                    self.assertGreater(float(np.max(np.abs(loop.core_correction))), 1.0e-3)
                    _assert_round_off(self, _batched(arguments), loop, f"shift={shift}")
                    reference = dense_periodic_local_forces(*arguments)
                    np.testing.assert_allclose(loop.ionic, reference.ionic, atol=1.0e-6)
                    np.testing.assert_allclose(loop.core_correction, reference.core_correction, atol=1.0e-10)

    def test_oblique_cells_and_the_host_keep_the_loop(self) -> None:
        fcc = 6.6 * (np.ones((3, 3)) - np.eye(3))
        arguments = _arguments(fcc, 0.7, _SHIFTS[2], _SPECIES[1], 3)
        self.assertIsNone(periodic_local._orthorhombic_geometry(arguments[0]))
        loop = fast_periodic_local_forces(*arguments)
        np.testing.assert_array_equal(_batched(arguments).ionic, loop.ionic)
        # NumPy (and a module without a device) under auto, any module under off: the loop
        self.assertFalse(periodic_local._uses_batched_balls(np))
        device = types.SimpleNamespace(cuda=object())
        self.assertTrue(periodic_local._uses_batched_balls(device))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_BALLS": "off"}):
            self.assertFalse(periodic_local._uses_batched_balls(device))

    def test_a_grid_whose_coordinates_are_not_those_of_its_axes_keeps_the_loop(self) -> None:
        grid = _arguments(_RECTANGULAR, 0.7, _SHIFTS[0], _SPECIES[0], 1)[0]
        self.assertIsNotNone(periodic_local._orthorhombic_geometry(grid))
        moved = types.SimpleNamespace(
            cell=grid.cell, points_per_axis=grid.points_per_axis, shift=grid.shift,
            coordinates=grid.coordinates + np.array([1.0e-3, 0.0, 0.0]) * (np.arange(grid.size)[:, None] % 2),
        )
        self.assertIsNone(periodic_local._orthorhombic_geometry(moved))

    def test_settings(self) -> None:
        for name, reader in (
            ("PARSEC_PERIODIC_LOCAL_BALLS", periodic_local_balls_setting),
            ("PARSEC_PERIODIC_LOCAL_ORBITS", periodic_local_orbits_setting),
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(name, None)
                self.assertEqual(reader(), "auto")
            for value in ("auto", "off"):
                with mock.patch.dict(os.environ, {name: f" {value.upper()} "}):
                    self.assertEqual(reader(), value)
            with mock.patch.dict(os.environ, {name: "on"}), self.assertRaises(ValueError):
                reader()


def _wedge_cell():
    """``(system, result)``: the displaced Si8 cube (Td about the origin) prepared on the default routines with the
    operations of its grid as the space group of a reduced mesh (``density_symmetry``), and the fields of a k-point
    result on that wedge: the superposed atomic density averaged over the group, an XC field of it, zero nonlocal
    forces and unequal weights (a wedge)."""

    from parsec_python.Grid.point_group import symmetrize_density
    from parsec_python.Grid.space_group import find_grid_space_group
    from parsec_python.SCF import prepare_periodic_single_point
    from parsec_python.SCF import pbc as periodic
    from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon

    problem = displaced_silicon()
    problem = replace(problem, grid=replace(problem.grid, fortran_routines=False))
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    grid = periodic.build_periodic_grid(problem.periodic_cell, problem.grid)
    group = find_grid_space_group(lattice, problem.atoms, np.asarray(grid.points_per_axis), grid.shift)
    problem = replace(problem, space_group=group, scf=replace(problem.scf, symmetrize_density=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = prepare_periodic_single_point(problem)
    density = periodic.superpose_atomic_density(
        system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials, lattice, periodic_dimensions=3
    )
    density, _ = symmetrize_density(density, system.density_symmetry, system.grid.volume_element)
    xc = -np.cbrt(np.abs(density)) - 0.1 * density
    result = types.SimpleNamespace(
        k_points=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        k_weights=np.array([0.25, 0.75]),
        density=density,
        xc_potential=xc,
        nonlocal_forces=np.zeros((len(system.atoms), 3)),
    )
    return system, result


class KPointWedgeOrbitTests(unittest.TestCase):
    """The local forces of a k-point wedge from one atom per orbit of its group equal those of every atom."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system, cls.result = _wedge_cell()

    def _forces(self, setting: str, system=None):
        from parsec_python.Forces import compute_periodic_forces

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_ORBITS": setting}):
            return compute_periodic_forces(self.system if system is None else system, self.result)

    def test_the_orbits_of_the_wedge_give_the_forces_of_every_atom(self) -> None:
        from parsec_python.Forces.periodic import reduced_kpoint_operations

        operations = reduced_kpoint_operations(self.system, self.result)
        self.assertEqual(len(operations), 24)
        with mock.patch(
            "parsec_python.acceleration.Forces.periodic_local.fast_periodic_local_forces", wraps=fast_periodic_local_forces
        ) as fast:
            orbits = self._forces("auto")
        # the three orbits of Td about the origin and the checked atoms in one call
        self.assertEqual(fast.call_count, 1)
        self.assertLess(len(fast.call_args.args[1]), len(self.system.atoms))
        everything = self._forces("off")
        self.assertGreater(float(np.abs(everything.local.ionic).max()), 1.0e-3)
        np.testing.assert_allclose(orbits.local.ionic, everything.local.ionic, rtol=0, atol=1.0e-12)
        np.testing.assert_allclose(orbits.local.core_correction, everything.local.core_correction, rtol=0, atol=1.0e-12)
        np.testing.assert_allclose(orbits.total, everything.total, rtol=0, atol=1.0e-12)
        self.assertEqual(orbits.phase_seconds["local_representatives"], 3.0)
        self.assertLess(orbits.phase_seconds["local_site_asymmetry"], 1.0e-12)
        self.assertIn("local", everything.phase_seconds)
        self.assertNotIn("local_representatives", everything.phase_seconds)

    def test_a_density_that_is_not_averaged_sums_every_atom(self) -> None:
        unaveraged = replace(self.system, density_symmetry=None)
        with mock.patch(
            "parsec_python.acceleration.Forces.periodic_local.orbit_local_forces", wraps=orbit_local_forces
        ) as orbits:
            forces = self._forces("auto", unaveraged)
        orbits.assert_not_called()
        self.assertNotIn("local_representatives", forces.phase_seconds)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceBatchedBallTests(unittest.TestCase):
    """The batched ball sums on the GPU against the loop on the GPU (PARSEC_PERIODIC_LOCAL_BALLS=off) and the host."""

    def _both(self, arguments, label, **options):
        import cupy as cp
        from cupyx.scipy.special import erf

        self.assertTrue(periodic_local._uses_batched_balls(cp))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_BALLS": "off"}):
            loop = fast_periodic_local_forces(*arguments, xp=cp, erf=erf)
        with mock.patch.object(periodic_local, "_batched_local_sums", wraps=periodic_local._batched_local_sums) as spy:
            if options:
                with mock.patch.object(periodic_local, "_BATCH_POINTS", options["points"]):
                    batched = fast_periodic_local_forces(*arguments, xp=cp, erf=erf)
            else:
                batched = fast_periodic_local_forces(*arguments, xp=cp, erf=erf)
        spy.assert_called_once()
        _assert_round_off(self, batched, loop, label)
        _assert_round_off(self, batched, fast_periodic_local_forces(*arguments), label + " (host)")

    def test_the_device_batches_are_the_loop(self) -> None:
        for shift in _SHIFTS:
            for species in _SPECIES:
                with self.subTest(species=species[0], spline=species[3], shift=shift):
                    self._both(_arguments(_RECTANGULAR, 0.7, shift, species, 5), f"{species[0]} {shift}")
        self._both(_arguments(_RECTANGULAR, 0.7, _SHIFTS[2], _SPECIES[1], 9), "small batches", points=5_000)

    def test_a_table_of_six_samples_and_an_atom_on_a_grid_point(self) -> None:
        # the fused terms and the loop on the device: no 2 Z/0 at the atom (NaN before 2026-11-23)
        import cupy as cp
        from cupyx.scipy.special import erf

        for shift in _SHIFTS[:2]:
            arguments = _six_sample_arguments(_RECTANGULAR, shift)
            with self.subTest(shift=shift):
                self._both(arguments, f"H six samples {shift}")
                oblique = _six_sample_arguments(6.6 * (np.ones((3, 3)) - np.eye(3)), shift)
                device = fast_periodic_local_forces(*oblique, xp=cp, erf=erf)
                self.assertTrue(np.all(np.isfinite(device.ionic)))
                _assert_round_off(self, device, fast_periodic_local_forces(*oblique), f"fcc {shift}")

    def test_a_diamond_cell(self) -> None:
        import time

        import cupy as cp
        from cupyx.scipy.special import erf

        a = 3.567 / 0.529177210903
        lattice = np.diag([3.0 * a] * 3)
        arguments = _arguments(lattice, a / 18.0 * (1.0 - 1.0e-9), (0.0, 0.0, 0.0), ("C", _CARBON, 2, True), 216)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_BALLS": "off"}):
            started = time.perf_counter()
            loop = fast_periodic_local_forces(*arguments, xp=cp, erf=erf)
            loop_seconds = time.perf_counter() - started
        started = time.perf_counter()
        batched = fast_periodic_local_forces(*arguments, xp=cp, erf=erf)
        batched_seconds = time.perf_counter() - started
        _assert_round_off(self, batched, loop, "diamond 3^3")
        print(f"\n216 C atoms on 54^3: loop {loop_seconds:.2f} s, batched {batched_seconds:.2f} s, "
              f"max |dF| {float(np.max(np.abs(batched.ionic - loop.ionic))):.2e} Ry/bohr")


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class AbelianSectorOrbitTests(unittest.TestCase):
    """compute_forces of a cell in Abelian sectors: the local terms of the orbits of H against those of every atom."""

    def test_c2v_silicon(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Forces.sector_symmetry import abelian_cell_operations
        from parsec_python.acceleration.tests.test_periodic_local_force_orbits import _silicon_cell

        problem = _silicon_cell()
        atoms = list(problem.atoms)
        atoms[0] = Atom(atoms[0].symbol, tuple(np.asarray(atoms[0].position) + np.array([0.0, 0.0, 0.05])))
        problem = replace(problem, atoms=tuple(atoms))
        tables = {}
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": "auto"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system)
            self.assertTrue(result.converged)
            operations = abelian_cell_operations(system)
            self.assertIsNotNone(operations)
            self.assertGreater(len(operations), 1)
            for setting in ("off", "auto"):
                with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_LOCAL_ORBITS": setting}):
                    tables[setting] = driver.compute_forces(system, result, symmetrize=False)
        everything, orbits = tables["off"], tables["auto"]
        self.assertGreater(float(np.abs(everything.total).max()), 1.0e-3)
        np.testing.assert_allclose(orbits.local.ionic, everything.local.ionic, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(orbits.local.core_correction, everything.local.core_correction, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(orbits.total, everything.total, rtol=0, atol=1.0e-10)
        self.assertLess(orbits.phase_seconds["local_representatives"], len(atoms))
        self.assertLess(orbits.phase_seconds["local_site_asymmetry"], 1.0e-10)
        self.assertNotIn("local_representatives", everything.phase_seconds)
        print(
            f"\nSi8 C2v in Abelian sectors ({len(operations)} operations): local {everything.phase_seconds['local']:.3f} s "
            f"(every atom) vs {orbits.phase_seconds['local']:.3f} s ({orbits.phase_seconds['local_representatives']:.0f} "
            f"atoms), asymmetry {orbits.phase_seconds['local_site_asymmetry']:.2e}"
        )


if __name__ == "__main__":
    unittest.main()
