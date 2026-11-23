"""PARSEC_LINEAR_TABLE_SLOPES: forces of species without Cubic_Spline with the slopes of the linear interpolants.

Under ``interpolant`` every force term of a linearly interpolated table is the derivative of the energy the run
evaluates (central differences at steps far below the spacing of the radial samples, so that no grid point crosses a
sample); under ``tabulated`` (the default) the slope tables of ``pseudo.f90`` give PARSEC 1.5's forces bit for bit.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces.local import local_forces
from parsec_python.Forces.nonlocal_force import (
    build_nonlocal_projector_gradients,
    build_periodic_projector_gradients,
    nonlocal_forces,
)
from parsec_python.Forces.periodic_local import dense_periodic_local_forces
from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.models import GridSettings
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential import radial_derivatives as radial
from parsec_python.V_ion import load_pseudopotentials, superpose_atomic_density
from parsec_python.V_ion.ionic_potential import build_local_ionic_potential, build_nonlocal_projectors

_DATA = Path(__file__).resolve().parents[1] / "data"
_GE = _DATA / "Ge_POTRE.DAT"  # nonlinear core correction, s p d channels
_SI = _DATA / "Si_POTRE.DAT"
_ENV = radial.LINEAR_SLOPES_ENV
_STEP = 1.0e-7


def _setting(value):
    return mock.patch.dict(os.environ, {_ENV: value})


class SettingTests(unittest.TestCase):
    def test_default_and_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            self.assertEqual(radial.linear_slope_setting(), "tabulated")
        for value in ("tabulated", "interpolant", " Interpolant "):
            with _setting(value):
                self.assertEqual(radial.linear_slope_setting(), value.strip().lower())

    def test_unknown_value_stops_the_preparation(self):
        with _setting("smooth"):
            with self.assertRaises(ValueError):
                radial.linear_slope_setting()
        try:
            from parsec_python.acceleration.Forces.periodic_local import validate_local_force_settings
        except ImportError:  # pragma: no cover - the acceleration package is part of the tree
            self.skipTest("no acceleration package")
        with _setting("smooth"):
            with self.assertRaises(ValueError):
                validate_local_force_settings()


class RadialSlopeTests(unittest.TestCase):
    """The interpolant slopes against differences of the radial functions the energy samples, taken within one
    segment of the radial grid, where ``r V`` (or ``V`` itself, or a density) is linear."""

    @classmethod
    def setUpClass(cls):
        cls.potential = read_parsec_pseudopotential(_GE)
        radii = cls.potential.radii
        lower = np.arange(5, radii.size - 3, 7)
        cls.width = radii[lower + 1] - radii[lower]
        cls.points = radii[lower] + 0.37 * cls.width

    def _chord(self, function):
        delta = 0.2 * self.width
        return (function(self.points + delta) - function(self.points - delta)) / (2 * delta)

    def test_local_potential(self):
        potential, points = self.potential, self.points
        inside = points < potential.interpolation_cutoff
        # r V linear in a segment: V' = ((r V)' - V)/r
        weighted = self._chord(lambda r: r * potential.local_potential(r, 1))
        expected = np.where(inside, (weighted - potential.local_potential(points, 1)) / points,
                            2.0 * potential.ionic_charge / points**2)
        slope = radial.evaluate_local_potential_derivative(potential, points, 1, slopes="interpolant")
        np.testing.assert_allclose(slope[inside], expected[inside], rtol=1e-6, atol=1e-8)
        direct = radial.evaluate_local_potential_derivative(potential, points, 1, direct=True, slopes="interpolant")
        expected = self._chord(lambda r: potential.local_potential(r, 1, direct=True))
        np.testing.assert_allclose(direct[inside], expected[inside], rtol=1e-6, atol=1e-8)
        # below the first sample the value is constant
        self.assertEqual(float(radial.evaluate_local_potential_derivative(
            potential, np.array([0.5 * potential.radii[0]]), 1, slopes="interpolant")[0]), 0.0)

    def test_density(self):
        for density, core in ((self.potential.core_density, True), (self.potential.valence_density, False)):
            expected = self._chord(lambda r: self.potential.interpolate_density(r, core=core))
            slope = radial.evaluate_density_derivative(self.potential, self.points, density, slopes="interpolant")
            np.testing.assert_allclose(slope, expected, rtol=1e-6, atol=1e-8)

    def test_tabulated_is_the_former_value(self):
        table = radial.local_potential_derivative_table(self.potential, 1)
        former = radial.evaluate_local_potential_derivative(self.potential, self.points, 1, table=table)
        explicit = radial.evaluate_local_potential_derivative(
            self.potential, self.points, 1, table=table, slopes="tabulated"
        )
        np.testing.assert_array_equal(former, explicit)


class _Cluster:
    def __init__(self):
        self.spec = SpeciesPotential(_GE, 1, use_spline=False)
        self.potentials = load_pseudopotentials({"Ge": self.spec}, xc_functional="ca", check_correlation=False)
        self.specs = {"Ge": self.spec}
        self.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=5.0, expansion_order=6))
        self.positions = np.array([[0.137, -0.081, 0.213], [2.31, 1.27, -1.06]])
        coordinates = np.asarray(self.grid.coordinates)
        self.density = np.asarray(superpose_atomic_density(self.grid, self.atoms(), self.potentials, self.specs))
        self.xc = -0.9 * np.cbrt(self.density) + 0.01 * np.cos(coordinates[:, 1])
        relative = coordinates - self.positions[0]
        envelope = np.sqrt(self.density)
        orbitals = np.stack([envelope, envelope * relative[:, 0], envelope * relative[:, 2]], axis=1)
        self.orbitals = orbitals / np.linalg.norm(orbitals, axis=0)
        self.occupations = np.array([1.0, 0.6, 0.3])

    def atoms(self, positions=None):
        positions = self.positions if positions is None else positions
        return [Atom("Ge", np.array(p)) for p in positions]

    def energies(self, positions):
        atoms = self.atoms(positions)
        v = np.asarray(build_local_ionic_potential(self.grid, atoms, self.potentials, self.specs))
        core = np.asarray(superpose_atomic_density(self.grid, atoms, self.potentials, self.specs, core=True))
        operator = build_nonlocal_projectors(self.grid, atoms, self.potentials, self.specs)
        overlaps = np.asarray(operator.projectors.T @ self.orbitals)
        nonlocal_energy = 2.0 * float(np.sum(np.asarray(operator.signs)[:, None] * overlaps**2 * self.occupations))
        dv = self.grid.volume_element
        return np.array([dv * float(self.density @ v), dv * float(self.xc @ core), nonlocal_energy])

    def forces(self):
        terms = local_forces(self.grid, self.atoms(), self.potentials, self.specs, self.density,
                             np.zeros(self.grid.size), self.xc)
        operator = build_nonlocal_projectors(self.grid, self.atoms(), self.potentials, self.specs)
        gradients = build_nonlocal_projector_gradients(self.grid, self.atoms(), self.potentials, self.specs)
        separable = nonlocal_forces(operator, gradients, self.orbitals, self.occupations, 2)
        return np.stack([terms.ionic[0], terms.core_correction[0], separable[0]])


class ClusterForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = _Cluster()
        differences = np.zeros((3, 3))
        for axis in range(3):
            plus, minus = cls.case.positions.copy(), cls.case.positions.copy()
            plus[0, axis] += _STEP
            minus[0, axis] -= _STEP
            differences[:, axis] = -(cls.case.energies(plus) - cls.case.energies(minus)) / (2 * _STEP)
        cls.differences = differences

    def test_interpolant_terms_are_the_derivatives_of_the_energy(self):
        with _setting("interpolant"):
            forces = self.case.forces()
        for term, name in enumerate(("local", "core", "nonlocal")):
            error = float(np.max(np.abs(forces[term] - self.differences[term])))
            self.assertLess(error, 2e-6 * max(1.0, float(np.max(np.abs(self.differences[term])))), name)

    def test_tabulated_is_the_default(self):
        os.environ.pop(_ENV, None)
        default = self.case.forces()
        with _setting("tabulated"):
            tabulated = self.case.forces()
        np.testing.assert_array_equal(default, tabulated)
        with _setting("interpolant"):
            interpolant = self.case.forces()
        # the two slopes differ (the interpolation of the tables is not the derivative of the energy)
        self.assertGreater(float(np.max(np.abs(interpolant - tabulated))), 1e-6)


def _ge_cell_tables():
    """The Ge table cut at 14 bohr (its core density is 1e-14 there), so that the image sums stay small."""

    full = read_parsec_pseudopotential(_GE)
    keep = int(np.searchsorted(full.radii, 14.0, side="right"))
    take = lambda values: values[:keep]  # noqa: E731
    return {"Ge": dataclasses.replace(
        full, radii=take(full.radii), channel_potentials={l: take(v) for l, v in full.channel_potentials.items()},
        valence_density=take(full.valence_density), core_density=take(full.core_density),
        radial_wavefunctions={l: take(v) for l, v in full.radial_wavefunctions.items()},
        spin_orbit_channel_potentials={l: take(v) for l, v in full.spin_orbit_channel_potentials.items()},
    )}


class PeriodicTests(unittest.TestCase):
    """The core term of the default routines of a cell and the projectors of the k-point routes."""

    fortran_routines = False

    @classmethod
    def setUpClass(cls):
        cls.potentials = _ge_cell_tables()
        cls.specs = {"Ge": SpeciesPotential(_GE, 1, use_spline=False)}
        cls.lattice = np.diag([7.7, 8.4, 9.1])
        cls.grid = build_periodic_grid(
            PeriodicCell(cls.lattice),
            PeriodicGridSettings(spacing=0.7, expansion_order=6, fortran_routines=cls.fortran_routines),
        )
        cls.positions = np.array([[0.13, 0.27, 0.31], [0.62, 0.45, 0.78]]) @ cls.lattice
        phase = 2.0 * np.pi * (cls.grid.coordinates @ np.linalg.inv(cls.lattice))
        cls.density = 0.3 + 0.2 * np.cos(phase[:, 0] + 0.4) * np.sin(phase[:, 1])
        cls.xc = -0.5 + 0.3 * np.sin(phase[:, 0]) * np.cos(phase[:, 2] + 0.3)
        cls.residual = 0.01 * np.cos(phase[:, 1])

    def atoms(self, positions):
        return [Atom("Ge", np.array(p)) for p in positions]

    def test_core_term_is_the_derivative_of_the_core_energy(self):
        def energy(positions):
            core = superpose_atomic_density(self.grid, self.atoms(positions), self.potentials, self.specs,
                                            self.lattice, core=True)
            return self.grid.volume_element * float(self.xc @ np.asarray(core))

        differences = np.zeros(3)
        for axis in range(3):
            plus, minus = self.positions.copy(), self.positions.copy()
            plus[0, axis] += _STEP
            minus[0, axis] -= _STEP
            differences[axis] = -(energy(plus) - energy(minus)) / (2 * _STEP)
        with _setting("interpolant"):
            terms = dense_periodic_local_forces(self.grid, self.atoms(self.positions), self.potentials, self.specs,
                                                self.lattice, self.density, self.xc)
        self.assertLess(float(np.max(np.abs(terms.core_correction[0] - differences))), 2e-6)
        with _setting("tabulated"):
            former = dense_periodic_local_forces(self.grid, self.atoms(self.positions), self.potentials, self.specs,
                                                 self.lattice, self.density, self.xc)
        self.assertGreater(float(np.max(np.abs(former.core_correction[0] - differences))), 1e-6)

    def test_fast_core_sums_follow_the_switch(self):
        """The ball sums of the acceleration package (NumPy here, CuPy where present) give the readable core term."""

        from parsec_python.acceleration.Forces.periodic_local import fast_periodic_local_forces

        arguments = (self.grid, self.atoms(self.positions), self.potentials, self.specs, self.lattice, self.density,
                     self.xc)
        modules = [(np, None)]
        try:
            import cupy
            from cupyx.scipy.special import erf

            cupy.zeros(1)
            modules.append((cupy, erf))
        except Exception:  # noqa: BLE001 - no device
            pass
        for setting in ("tabulated", "interpolant"):
            with _setting(setting):
                reference = dense_periodic_local_forces(*arguments)
                for xp, erf in modules:
                    fast = fast_periodic_local_forces(*arguments, xp=xp, erf=erf)
                    np.testing.assert_allclose(fast.core_correction, reference.core_correction, rtol=0, atol=1e-11,
                                               err_msg=f"{setting} {xp.__name__}")

    def test_bloch_projector_gradients(self):
        k_point = np.array([0.11, -0.07, 0.05])
        rng = np.random.default_rng(3)
        orbitals = rng.standard_normal((self.grid.size, 2)) + 1j * rng.standard_normal((self.grid.size, 2))
        orbitals /= np.linalg.norm(orbitals, axis=0)
        occupations = np.array([1.0, 0.4])

        def energy(positions):
            operator = build_nonlocal_projectors(self.grid, self.atoms(positions), self.potentials, self.specs,
                                                 self.lattice, k_point)
            overlaps = np.asarray(operator.projectors.conj().T @ orbitals)
            return 2.0 * float(np.sum(np.asarray(operator.signs)[:, None] * np.abs(overlaps) ** 2 * occupations))

        differences = np.zeros(3)
        for axis in range(3):
            plus, minus = self.positions.copy(), self.positions.copy()
            plus[0, axis] += _STEP
            minus[0, axis] -= _STEP
            differences[axis] = -(energy(plus) - energy(minus)) / (2 * _STEP)
        operator = build_nonlocal_projectors(self.grid, self.atoms(self.positions), self.potentials, self.specs,
                                             self.lattice, k_point)
        with _setting("interpolant"):
            gradients = build_periodic_projector_gradients(self.grid, self.atoms(self.positions), self.potentials,
                                                           self.specs, self.lattice, k_point)
        forces = nonlocal_forces(operator, gradients, orbitals, occupations, 2)
        scale = max(1.0, float(np.max(np.abs(differences))))
        self.assertLess(float(np.max(np.abs(forces[0] - differences))), 2e-6 * scale)


class FortranRoutineTableTests(unittest.TestCase):
    """``PARSEC_PERIODIC_FORCE_SLOPES=tables`` on the Fortran-derived routines: the slopes of ``pseudo.f90``."""

    fortran_routines = True
    setUpClass = classmethod(PeriodicTests.setUpClass.__func__)
    atoms = PeriodicTests.atoms

    def test_core_term_is_that_of_the_tables(self):
        from parsec_python.Forces import periodic as fortran_routines

        previous = fortran_routines._local_force_provider
        fortran_routines.set_local_force_provider(None)
        try:
            terms = {}
            for setting in ("interpolant", "tables"):
                with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_FORCE_SLOPES": setting}):
                    terms[setting] = fortran_routines.periodic_local_forces(
                        self.grid, self.atoms(self.positions), self.potentials, self.specs, self.lattice,
                        self.density, self.residual, self.xc,
                    )
        finally:
            fortran_routines.set_local_force_provider(previous)
        with _setting("tabulated"):
            default_routines = dense_periodic_local_forces(
                self.grid, self.atoms(self.positions), self.potentials, self.specs, self.lattice, self.density,
                self.xc,
            )
        np.testing.assert_allclose(terms["tables"].core_correction, default_routines.core_correction, rtol=0,
                                   atol=1e-12)
        for name in ("ionic", "self_consistency", "core_correction"):
            self.assertGreater(
                float(np.max(np.abs(getattr(terms["tables"], name) - getattr(terms["interpolant"], name)))), 1e-7, name
            )

    def test_projectors_take_dvw(self):
        from parsec_python.Forces.periodic import build_periodic_projector_gradients as fortran_gradients

        arguments = (self.grid, self.atoms(self.positions), self.potentials, self.specs, self.lattice)
        tables = fortran_gradients(*arguments, radial_slopes="tables")
        tabulated = fortran_gradients(*arguments, radial_slopes="tabulated")
        for a, b in zip(tables.components, tabulated.components):
            self.assertEqual(abs(a - b).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
