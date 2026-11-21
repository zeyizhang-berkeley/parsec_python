"""The spline slopes of the local forces of a cell at the grid points next to an atom (``PARSEC_SPLINE_FORCE_CENTER``).

The padded cubic spline of a ``Cubic_Spline`` local potential has a slope of its own at ``r = 0+`` (3.6e-3 Ry/bohr for
the carbon POTRE table): the potential is a cone at the atom.  Up to 2026-11-20 a grid point within round-off of an
atom took that slope along the direction of the round-off in the local forces of the Fortran-derived routines of a
cell (the readable sums of :func:`parsec_python.Forces.periodic.periodic_local_forces` and the ball sums of
:mod:`parsec_python.acceleration.Forces.periodic_spline`), so an atom moved by 1e-11 bohr off a grid point changed its
force by the slope at the tip.  Under ``auto`` (the default) the points within ``min(rs(2), 1e-6 bohr)`` of the atom
are left out of the spline slopes (the symmetric value at the tip); ``fortran`` leaves out every point closer than the
first radial sample, the rule of ``forcloc.f90``.  Checked here (CPU):

* the switch and its values, also through ``validate_local_force_settings``;
* an atom on a grid point and the same atom 1e-11 bohr off it get the same local forces under ``auto``, different
  ones under ``off`` (the artifact), on both routes; both routes agree with each other under ``auto``;
* the force of the atom on the grid point is the central difference of the local energy of the fast potential;
* with no grid point within the first radial sample of an atom the forces of ``auto`` are those of ``off`` bit for
  bit, and a species without ``Cubic_Spline`` is not touched;
* an atom between 1e-6 bohr and ``rs(2)`` from a grid point keeps the exact slope under ``auto`` (the bits of
  ``off``), and loses it under ``fortran``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces import periodic as force_sums
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import superpose_atomic_density
from parsec_python.V_ion.ewald import ewald_local_ionic_potential

_ROOT = Path(__file__).resolve().parents[4]
_C_POTRE = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_CUBE = 6.0 * np.eye(3)
_ENV = force_sums.SPLINE_CENTER_ENV


def _case(spline: bool = True, offset=(0.0, 0.0, 0.0), on_grid: bool = True):
    potential = read_parsec_pseudopotential(_C_POTRE)
    specification = SpeciesPotential(_C_POTRE, 1, use_spline=spline)
    grid = build_periodic_grid(
        PeriodicCell(lattice_vectors=_CUBE), PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=True)
    )
    coordinates = np.asarray(grid.coordinates)
    point = coordinates[np.argmin(np.linalg.norm(coordinates - np.array([2.0, 2.2, 2.4]), axis=1))]
    first = point if on_grid else point + np.array([0.13, 0.07, 0.11])
    atoms = (
        Atom("C", (first + np.asarray(offset, dtype=float)).tolist()),
        Atom("C", (first + np.array([1.21, 1.43, 1.37])).tolist()),
    )
    potentials, specifications = {"C": potential}, {"C": specification}
    density = superpose_atomic_density(grid, atoms, potentials, specifications, _CUBE)
    residual = 1.0e-3 * np.sin(coordinates[:, 0]) * density
    xc = -0.9 * np.cbrt(density)
    return grid, atoms, potentials, specifications, density, residual, xc


def _forces(provider, case, setting: str):
    grid, atoms, potentials, specifications, density, residual, xc = case
    previous = force_sums._local_force_provider
    force_sums.set_local_force_provider(provider)
    try:
        with mock.patch.dict(os.environ, {_ENV: setting}):
            return force_sums.periodic_local_forces(
                grid, atoms, potentials, specifications, _CUBE, density, residual, xc
            )
    finally:
        force_sums.set_local_force_provider(previous)


def _fast_provider():
    from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

    return spline_periodic_local_forces


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            self.assertEqual(force_sums.spline_force_center_setting(), "auto")
        for value, expected in (("auto", "auto"), (" OFF ", "off"), ("", "auto"), ("Fortran", "fortran")):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertEqual(force_sums.spline_force_center_setting(), expected)
        with mock.patch.dict(os.environ, {_ENV: "rs2"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_SPLINE_FORCE_CENTER must be auto, fortran or off"):
                force_sums.spline_force_center_setting()
            from parsec_python.acceleration.Forces.periodic_local import validate_local_force_settings

            with self.assertRaisesRegex(ValueError, "PARSEC_SPLINE_FORCE_CENTER"):
                validate_local_force_settings()

    def test_the_slope_at_the_tip(self) -> None:
        potential = read_parsec_pseudopotential(_C_POTRE)
        first = float(potential.radii[0])
        self.assertGreater(first, force_sums.ROUND_OFF_DISTANCE)
        distances = np.array([1.0e-11, 0.5e-6, 0.5 * first, first, 0.1])
        slope = force_sums.local_potential_slope(potential, distances, 1, use_spline=True, spline_padding_width=3)
        # the cone of the padded spline: a slope of its own at r = 0+
        self.assertGreater(abs(slope[0]), 1.0e-3)
        self.assertEqual(force_sums.spline_center_radius(potential, "auto"), force_sums.ROUND_OFF_DISTANCE)
        self.assertEqual(force_sums.spline_center_radius(potential, "fortran"), first)
        self.assertEqual(force_sums.spline_center_radius(potential, "off"), 0.0)
        kept = force_sums.without_spline_center(slope, distances, potential, "auto")
        np.testing.assert_array_equal(kept, [0.0, 0.0, slope[2], slope[3], slope[4]])
        kept = force_sums.without_spline_center(slope, distances, potential, "fortran")
        np.testing.assert_array_equal(kept, [0.0, 0.0, 0.0, slope[3], slope[4]])
        self.assertIs(force_sums.without_spline_center(slope, distances, potential, "off"), slope)


class CenterTests(unittest.TestCase):
    """The two routes of the Fortran-derived routines (readable sums and ball sums) on an atom placed on the grid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.on = _case()
        cls.near = _case(offset=(1.0e-11, -0.7e-11, 0.4e-11))
        cls.routes = {"readable": None, "fast": _fast_provider()}

    def test_an_atom_on_a_grid_point_takes_no_slope_of_the_tip(self) -> None:
        results = {}
        for name, provider in self.routes.items():
            for setting in ("auto", "off"):
                results[name, setting] = (
                    _forces(provider, self.on, setting).total,
                    _forces(provider, self.near, setting).total,
                )
        for name in self.routes:
            with self.subTest(route=name):
                on, near = results[name, "auto"]
                np.testing.assert_allclose(near, on, rtol=0.0, atol=1.0e-10)
                old_on, old_near = results[name, "off"]
                # the former sums: the slope of the cone along the direction of the 1e-11 bohr
                self.assertGreater(float(np.max(np.abs(old_near[0] - old_on[0]))), 1.0e-6)
                # the other atom is not near a grid point: its force is the same
                np.testing.assert_allclose(on[1], old_on[1], rtol=0.0, atol=1.0e-12)
        # the two routes under auto (the readable sum adds the spline error of -2Z/r out to its longer reach)
        np.testing.assert_allclose(results["fast", "auto"][0], results["readable", "auto"][0], rtol=0.0, atol=2.0e-6)

    def test_the_force_is_the_central_difference_of_the_energy(self) -> None:
        # the atom 1e-11 bohr off the grid point: the difference over +-2e-4 bohr sees the cone symmetrically
        grid, atoms, potentials, specifications, density, _, _ = self.near
        forces = _forces(self.routes["fast"], self.near, "auto").ionic
        former = _forces(self.routes["fast"], self.near, "off").ionic
        step = 2.0e-4
        for axis in range(3):
            energies = []
            for shift in (step, -step, 2.0 * step, -2.0 * step):
                position = np.array(atoms[0].position, dtype=float)
                position[axis] += shift
                moved = (Atom("C", position.tolist()), atoms[1])
                potential = ewald_local_ionic_potential(grid, moved, potentials, specifications, _CUBE)
                energies.append(grid.volume_element * float(density @ potential))
            derivative = (8.0 * (energies[0] - energies[1]) - (energies[2] - energies[3])) / (12.0 * step)
            with self.subTest(axis=axis):
                # (laptop: 1.5e-10 Ry/bohr; the former sums 1.4e-5 to 3.6e-5 off)
                self.assertAlmostEqual(forces[0, axis], -derivative, delta=1.0e-7)
                self.assertGreater(abs(former[0, axis] + derivative), 1.0e-6)

    def test_between_round_off_and_the_first_sample(self) -> None:
        # the atom 5e-6 bohr off the grid point (round-off radius 1e-6, first sample 1.3e-5): auto keeps the slope
        # of the spline, the derivative of the energy, bit for bit as off; fortran (forcloc.f90) leaves it out
        case = _case(offset=(3.0e-6, -2.5e-6, 3.0e-6))
        for name, provider in self.routes.items():
            with self.subTest(route=name):
                auto = _forces(provider, case, "auto")
                off = _forces(provider, case, "off")
                fortran = _forces(provider, case, "fortran")
                for term in ("ionic", "self_consistency", "core_correction"):
                    np.testing.assert_array_equal(getattr(auto, term), getattr(off, term))
                self.assertGreater(float(np.max(np.abs(fortran.ionic[0] - off.ionic[0]))), 1.0e-7)

    def test_atoms_off_the_grid_and_linear_tables_are_not_changed(self) -> None:
        cases = {"off the grid": _case(on_grid=False), "linear table": _case(spline=False)}
        for label, case in cases.items():
            for name, provider in self.routes.items():
                for value in ("auto", "fortran"):
                    with self.subTest(case=label, route=name, value=value):
                        changed = _forces(provider, case, value)
                        off = _forces(provider, case, "off")
                        for term in ("ionic", "self_consistency", "core_correction"):
                            np.testing.assert_array_equal(getattr(changed, term), getattr(off, term))


if __name__ == "__main__":
    unittest.main()
