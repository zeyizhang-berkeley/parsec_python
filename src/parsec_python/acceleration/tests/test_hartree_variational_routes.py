"""The potential of the Hartree energy (``PARSEC_HARTREE_VARIATIONAL``) on the Hartree routes of the accelerated driver.

Every route gives the energies and forces of the readable path.  The classes
named ``CuPy...`` need a CUDA device and are skipped without one; the
``Native...`` class needs the C++ extension.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import parsec_python.acceleration.driver as driver_module
from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.Forces import compute_forces as reference_forces
from parsec_python.Hartree.variational import (
    HARTREE_VARIATIONAL_SWITCH,
    FieldLayout,
    HartreeAdjoint,
    VariationalHartree,
    ZeroBoundaryHostSolver,
)
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Grid import build_cluster_grid
from parsec_python.SCF import run_scf as reference_scf
from parsec_python.SCF.single_point import prepare_single_point as prepare_reference

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
KT = 6.33327186e-06
# Switches of the Hartree routes that a launch environment may name.
ROUTE_SWITCHES = (
    "PARSEC_HARTREE_LINEAR_BACKEND",
    "PARSEC_HARTREE_BOUNDARY_BACKEND",
    "PARSEC_HARTREE_BOUNDARY_KERNEL",
    "PARSEC_CUPY_RESIDENT_HARTREE",
    "PARSEC_HARTREE_ATOMIC_TAIL_VALUES",
    HARTREE_VARIATIONAL_SWITCH,
)


def _na2(symmetric=False):
    """Na2 in a small sphere, Solver_Lpole 2 (raised to 6 with the atomic tail), forces wanted."""

    positions = [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]] if symmetric else [[-3.03, 0.11, 0.07], [3.02, -0.05, 0.04]]
    return SinglePointInput(
        atoms=[Atom("Na", position) for position in positions],
        pseudopotentials={"Na": SpeciesPotential(DATA / "Na_POTRE.DAT", 0, use_spline=True)},
        grid=GridSettings(spacing=0.7, radius=7.5, expansion_order=6),
        scf=SCFSettings(
            max_iterations=300, number_of_states=6, convergence_criterion=1.0e-10, fermi_temperature_kelvin=500.0
        ),
        hartree=HartreeSettings(multipole_order=2, max_iterations=4000, relative_tolerance=1.0e-11, absolute_tolerance=1.0e-15),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-11),
        forces_wanted=True,
    )


def _environment(**names):
    environment = {name: value for name, value in os.environ.items() if name not in ROUTE_SWITCHES}
    environment.update(names)
    return patch.dict(os.environ, environment, clear=True)


def _reference(problem, value="on"):
    with _environment(**{HARTREE_VARIATIONAL_SWITCH: value}):
        system = prepare_reference(problem)
        result = reference_scf(system)
        return result, reference_forces(system, result, symmetrize=False)


def _accelerated(problem, **options):
    environment = {HARTREE_VARIATIONAL_SWITCH: "on", **options.pop("environment", {})}
    with _environment(**environment):
        system = driver_module.prepare_single_point(problem, **options)
        result = driver_module.run_scf(system)
        forces = driver_module.compute_forces(system, result, symmetrize=False)
        return system, result, forces


class RouteAgreement:
    """Mix-in: a route against the readable path."""

    def assert_route(self, problem, *, energy_tolerance=1.0e-9, force_tolerance=1.0e-7, **options):
        result, forces = _reference(problem)
        system, accelerated, accelerated_forces = _accelerated(problem, **options)
        details = dict(system.backend_info.details)
        self.assertIsNotNone(system.hartree_variational, details.get("hartree_variational"))
        self.assertIn("derivative of the Hartree energy", details["hartree_variational"])
        self.assertAlmostEqual(accelerated.energies.total, result.energies.total, delta=energy_tolerance)
        self.assertIsNotNone(accelerated_forces.hartree_boundary)
        np.testing.assert_allclose(
            accelerated_forces.unsymmetrized, forces.unsymmetrized, rtol=0, atol=force_tolerance
        )
        np.testing.assert_allclose(
            accelerated_forces.hartree_boundary, forces.hartree_boundary, rtol=0, atol=1.0e-9
        )
        self.assertIn("hartree_variational_solves", dict(accelerated.backend.details))
        # both loops ran with V' and keep V_H of the Poisson solve in their results
        poisson = np.asarray(result.hartree_potential, dtype=float)
        ran_with = np.asarray(result.output_effective_potential - result.ionic_potential - result.xc_potential)
        gap = float(np.abs(ran_with - poisson).max())
        self.assertGreater(gap, 1.0e-6)
        np.testing.assert_allclose(np.asarray(accelerated.hartree_potential, dtype=float), poisson, rtol=0, atol=0.05 * gap)
        return system, accelerated, accelerated_forces


class HostRouteTests(RouteAgreement, unittest.TestCase):
    def test_the_scipy_route(self) -> None:
        self.assert_route(_na2(), backend="scipy", symmetry="off")

    def test_off_restores_the_route_before(self) -> None:
        problem = _na2()
        system, result, forces = _accelerated(
            problem, backend="scipy", symmetry="off", environment={HARTREE_VARIATIONAL_SWITCH: "off"}
        )
        self.assertIsNone(system.hartree_variational)
        self.assertIsNone(forces.hartree_boundary)
        self.assertTrue(dict(system.backend_info.details)["hartree_variational"].startswith("off"))
        with _environment(**{HARTREE_VARIATIONAL_SWITCH: "off"}):
            reference = prepare_reference(problem)
            self.assertIsNone(reference.hartree_variational)
            plain = reference_scf(reference)
        self.assertAlmostEqual(result.energies.total, plain.energies.total, delta=1.0e-12)


@unittest.skipUnless(native_available(), "the C++ extension is not built")
class NativeRouteTests(RouteAgreement, unittest.TestCase):
    def test_the_host_cg_on_the_full_grid_and_on_the_wedge(self) -> None:
        for symmetric, symmetry in ((False, "off"), (True, "auto")):
            with self.subTest(symmetric=symmetric):
                self.assert_route(
                    _na2(symmetric), backend="native", symmetry=symmetry, energy_tolerance=2.0e-9, force_tolerance=2.0e-7
                )


@unittest.skipUnless(cupy_available(), "CuPy needs a CUDA device")
class CuPyRouteTests(RouteAgreement, unittest.TestCase):
    def test_the_device_sums_are_those_of_the_host(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=5.0, expansion_order=6))
        helper = VariationalHartree(grid, HartreeSettings(multipole_order=12), ZeroBoundaryHostSolver(build_negative_laplacian(grid), grid))
        y = np.random.default_rng(3).standard_normal(helper.adjoint.pairs.exterior_count)
        host = helper._transpose(y, FieldLayout(grid))
        helper.device = 0
        device = helper._transpose(y, FieldLayout(grid))
        np.testing.assert_allclose(device, host, rtol=0, atol=1.0e-12 * float(np.abs(host).max()))

    def test_the_device_tail_is_that_of_the_host(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=5.0, expansion_order=6))
        rng = np.random.default_rng(7)
        positions = rng.uniform(-2.0, 2.0, (9, 3))
        charges = rng.uniform(1.0, 4.0, 9)
        u = rng.standard_normal(grid.size)
        layout = FieldLayout(grid)
        host = HartreeAdjoint(grid, 12)
        device = HartreeAdjoint(grid, 12, device=0)
        # C_L is the small difference of the Coulomb sums (about 10 Ry here) and their series: rounding of those
        expected = host.tail_values(positions, charges)
        np.testing.assert_allclose(device.tail_values(positions, charges), expected, rtol=0, atol=1.0e-13)
        self.assertGreater(float(np.abs(expected).max()), 1.0e-6)
        # the multipole part by its fourth-order difference at a step of 1e-3 bohr, exact to about 1e-10
        expected = host.tail_forces(layout, u, positions, charges)
        np.testing.assert_allclose(
            device.tail_forces(layout, u, positions, charges), expected, rtol=0, atol=1.0e-9 * float(np.abs(expected).max())
        )

    def test_the_gpu_routes(self) -> None:
        cases = (
            ("full grid", _na2(), {"symmetry": "off"}),
            ("resident chain on the wedge", _na2(True), {"symmetry": "auto"}),
            ("wedge through the host interface", _na2(True), {"symmetry": "auto", "environment": {"PARSEC_CUPY_RESIDENT_HARTREE": "0"}}),
        )
        for name, problem, options in cases:
            with self.subTest(route=name):
                system, _, _ = self.assert_route(
                    problem, backend="auto", energy_tolerance=2.0e-9, force_tolerance=2.0e-7, **options
                )
                self.assertIn("device", dict(system.backend_info.details)["hartree_variational"])

    def test_auto_decides_as_the_readable_path(self) -> None:
        for radius, active in ((7.5, True), (14.0, False)):
            problem = replace(_na2(), grid=GridSettings(spacing=0.7, radius=radius, expansion_order=6))
            with self.subTest(radius=radius):
                system, result, forces = _accelerated(
                    problem, backend="auto", symmetry="off", environment={HARTREE_VARIATIONAL_SWITCH: "auto"}
                )
                self.assertEqual(system.hartree_variational.active, active)
                self.assertEqual(forces.hartree_boundary is not None, active)
                reference, _ = _reference(problem, "auto" if active else "off")
                self.assertAlmostEqual(result.energies.total, reference.energies.total, delta=2.0e-9)

    def test_the_force_follows_the_energy_on_the_gpu(self) -> None:
        problem = _na2()
        with _environment(**{HARTREE_VARIATIONAL_SWITCH: "on"}):
            system = driver_module.prepare_single_point(problem, backend="auto", symmetry="off")
            positions = np.array([atom.position for atom in system.atoms], dtype=float)

            def free_energy(result):
                f = np.clip(np.asarray(result.occupations, dtype=float).ravel(), 1.0e-300, 1.0 - 1.0e-16)
                return float(result.energies.total) + 2.0 * KT * 500.0 * float(
                    np.sum(f * np.log(f) + (1.0 - f) * np.log(1.0 - f))
                )

            result = driver_module.run_scf(system)
            force = driver_module.compute_forces(system, result, symmetrize=False).unsymmetrized[0, 0]
            energies = []
            current = system
            step = 1.0e-3
            for sign in (1.0, -1.0):
                moved = positions.copy()
                moved[0, 0] += sign * step
                current = driver_module.update_geometry(
                    current, [replace(atom, position=p) for atom, p in zip(system.atoms, moved)]
                )
                energies.append(free_energy(driver_module.run_scf(current)))
        slope = -(energies[0] - energies[1]) / (2.0 * step)
        self.assertLess(abs(force - slope), 3.0e-7)


if __name__ == "__main__":
    unittest.main()
