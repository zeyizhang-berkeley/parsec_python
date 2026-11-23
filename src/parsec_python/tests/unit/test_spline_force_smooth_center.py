"""The slopes of the smooth behaviour of the local potential and the core density next to an atom
(``PARSEC_SPLINE_FORCE_CENTER=auto``, since 2026-11-22).

Within ``c = 2 rs(2)`` of an atom no table is the smooth function it stands for (below its first sample the padded
spline has a slope of its own at ``r = 0+`` and relaxes to the table over the first intervals), while the energy over a
step of 1e-3 bohr or more averages the padded interval out where it is narrower than the step and follows the smooth
behaviour ``V(0) + a r^2``.  ``auto`` gives every grid point closer than ``c`` the slope ``s(c) d/c`` of its route;
``tip`` keeps the forces of 2026-11-21.  Checked here (CPU; the device sums in
:mod:`parsec_python.acceleration.tests.test_spline_force_smooth_center_device`):

* the radius and the helpers (continuity at ``c``, coarse tables and the other values untouched);
* every route at fixed density, an atom on a grid point (1e-11 bohr off it) and 5e-4 bohr from one: with the carbon
  FHI table (first sample 1.04e-3 bohr, ``c`` = 2.08e-3) the local force of ``auto`` is the central difference of the
  local energy over 2e-3 bohr and ``tip`` is not (cluster, both routine sets of a cell, the core density of a
  core-corrected FHI table); the carbon POTRE table is sampled down to 1.28e-5 bohr (``c`` = 2.56e-5): 5e-4 bohr from
  a grid point ``auto`` keeps the bits of ``tip``, 2e-5 bohr from one its change is the rule itself (the near point's
  slope ``s(c) d/c`` in place of ``s(d)``);
* an atom farther than ``c`` from every grid point gets the bits of ``tip``; the routes agree with each other.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces import periodic as fortran_sums
from parsec_python.Forces import periodic_local as default_sums
from parsec_python.Forces.local import local_forces
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import GridSettings
from parsec_python.Pseudopotential import radial_derivatives as radial
from parsec_python.V_ion import load_pseudopotentials, superpose_atomic_density
from parsec_python.V_ion.ewald import ewald_local_ionic_potential
from parsec_python.V_ion.ionic_potential import build_local_ionic_potential

_ROOT = Path(__file__).resolve().parents[4]
_C_POTRE = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_C_FHI = _ROOT / "examples" / "3d_hybrid_functionals" / "C_FHIPP.DAT"
_B_FHI = _ROOT / "examples" / "3d_hybrid_functionals" / "B_FHIPP.DAT"
_H_CORE = Path(__file__).resolve().parents[1] / "data" / "H_CORE_POTRE.DAT"
_ENV = radial.SPLINE_FORCE_CENTER_ENV
_CUBE = 6.0 * np.eye(3)
_DIRECTION = np.array([1.0, 0.7, 0.4]) / np.linalg.norm([1.0, 0.7, 0.4])
_STEP = 2.0e-3


def _species(table: str, *, spline: bool = True) -> SpeciesPotential:
    if table == "potre":
        return SpeciesPotential(_C_POTRE, 2, use_spline=spline)
    if table == "boron":
        # an FHI table with a core density (first sample 1.25e-3 bohr)
        return SpeciesPotential(
            _B_FHI, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
            electrons_per_orbital=(2.0, 1.0, 0.0), core_cutoff_radius=1.8, nonlinear_core_correction=True,
        )
    return SpeciesPotential(
        _C_FHI, 2, use_spline=spline, pseudopotential_format="fhipp", number_of_channels=3,
        electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=1.6,
    )


def _load(specifications):
    return load_pseudopotentials(specifications, xc_functional="pbe", check_correlation=False)


def _setting(value: str):
    return mock.patch.dict(os.environ, {_ENV: value})


def _near_point(grid, guess, offset):
    coordinates = np.asarray(grid.coordinates)
    point = coordinates[int(np.argmin(np.linalg.norm(coordinates - np.asarray(guess), axis=1)))]
    return point + offset * _DIRECTION


class HelperTests(unittest.TestCase):
    def test_radius(self) -> None:
        specifications = {"C": _species("potre"), "F": _species("fhi")}
        potentials = _load({"C": specifications["C"]})
        potre = potentials["C"]
        fhi = _load({"C": specifications["F"]})["C"]
        self.assertEqual(radial.smooth_center_radius(potre, "auto"), 2.0 * float(potre.radii[0]))
        self.assertAlmostEqual(radial.smooth_center_radius(potre, "auto"), 2.56e-5, delta=1e-7)
        self.assertAlmostEqual(radial.smooth_center_radius(fhi, "auto"), 2.0 * float(fhi.radii[0]), places=15)
        for value in ("tip", "fortran", "off"):
            self.assertEqual(radial.smooth_center_radius(potre, value), 0.0)
        coarse = _load({"H": SpeciesPotential(_H_CORE, 0)})["H"]
        self.assertGreaterEqual(float(coarse.radii[0]), radial.SMOOTH_CENTER_TABLE_LIMIT)
        self.assertEqual(radial.smooth_center_radius(coarse, "auto"), 0.0)

    def test_the_slope_inside_joins_the_slope_outside(self) -> None:
        potential = _load({"C": _species("fhi")})["C"]
        radius = radial.smooth_center_radius(potential, "auto")

        def slope_of(points):
            return radial.evaluate_local_potential_derivative(
                potential, points, 2, use_spline=True, spline_padding_width=4
            )

        distances = np.array([0.0, 1.0e-11, 3.0e-4, 0.5 * radius, radius * (1.0 - 1e-12), radius, 0.1])
        smooth = radial.with_smooth_center(slope_of, distances, radius)
        at = float(slope_of(np.array([radius]))[0])
        np.testing.assert_allclose(smooth[:4], at * distances[:4] / radius, rtol=1e-14, atol=0.0)
        self.assertAlmostEqual(smooth[4], at, delta=1e-9)
        np.testing.assert_array_equal(smooth[5:], slope_of(distances[5:]))
        # the readable rule of forcloc.f90 (tip) leaves the points closer than rs(2) out
        self.assertEqual(float(slope_of(np.array([3.0e-4]))[0]), 0.0)
        np.testing.assert_array_equal(radial.with_smooth_center(slope_of, distances, 0.0), slope_of(distances))
        np.testing.assert_array_equal(
            radial.evaluate_local_potential_derivative(
                potential, distances, 2, use_spline=True, spline_padding_width=4, center=radius
            ),
            smooth,
        )

    def test_values(self) -> None:
        for value, expected in (("Tip", "tip"), ("", "auto"), ("auto", "auto")):
            with _setting(value):
                self.assertEqual(radial.spline_force_center_setting(), expected)
                self.assertEqual(fortran_sums.spline_force_center_setting(), expected)
        with _setting("smooth"):
            with self.assertRaisesRegex(ValueError, "must be auto, tip, fortran or off"):
                radial.spline_force_center_setting()


def _rule_change(grid, position, density, potential, lattice=None):
    """The change of the local force of an atom from ``tip`` to ``auto`` by the rule at the grid points within ``c``:
    ``dV rho(r) [s(c) d/c - s(d)] (r - R)/d`` with ``s`` the slope of the padded spline (POTRE: the first sample lies
    below the points, so both values take the spline there)."""

    from parsec_python.Pseudopotential import ParsecRadialSpline
    from parsec_python.Pseudopotential.radial_derivatives import spline_derivative

    spline = ParsecRadialSpline.from_positive_grid(
        potential.radii, potential.channel_potentials[2], grid.settings.stencil_half_width
    )
    radius = radial.smooth_center_radius(potential, "auto")
    at = float(spline_derivative(spline, np.array([radius]))[0])
    coordinates = np.asarray(grid.coordinates)
    images = [np.zeros(3)] if lattice is None else [np.asarray(t) @ lattice for t in np.ndindex(3, 3, 3)]
    change = np.zeros(3)
    for image in images:
        shift = image - (0.0 if lattice is None else np.ones(3) @ lattice)
        relative = coordinates - (np.asarray(position) + shift)
        distance = np.linalg.norm(relative, axis=1)
        for row in np.flatnonzero((distance < radius) & (distance > 0.0)):
            d = distance[row]
            slope = float(spline_derivative(spline, np.array([d]))[0])
            change += grid.volume_element * density[row] * (at * d / radius - slope) * relative[row] / d
    return change


def _central_difference(energy, atoms, index):
    """-dE/dR of atom ``index`` along the three axes, central differences over ``_STEP``."""

    result = np.zeros(3)
    for axis in range(3):
        values = []
        for sign in (1.0, -1.0):
            position = np.array(atoms[index].position, dtype=float)
            position[axis] += sign * _STEP
            moved = list(atoms)
            moved[index] = Atom(atoms[index].symbol, position.tolist())
            values.append(energy(tuple(moved)))
        result[axis] = -(values[0] - values[1]) / (2.0 * _STEP)
    return result


class ClusterTests(unittest.TestCase):
    """:func:`parsec_python.Forces.local.local_forces` (clusters) at fixed density."""

    def _check(
        self, table: str, offset: float, *, auto_tolerance: float | None, tip_error: float | None, beyond: bool = False
    ) -> None:
        specifications = {"C": _species(table)}
        potentials = _load(specifications)
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=4.0, expansion_order=8))
        first = _near_point(grid, (0.3, -0.2, 0.1), offset)
        atoms = (Atom("C", first.tolist()), Atom("C", (first + np.array([1.3, 1.1, -1.2])).tolist()))
        density = superpose_atomic_density(grid, atoms, potentials, specifications)
        zero = np.zeros(grid.size)

        def energy(moved):
            return grid.volume_element * float(density @ build_local_ionic_potential(grid, moved, potentials, specifications))

        reference = _central_difference(energy, atoms, 0)
        with _setting("auto"):
            auto = local_forces(grid, atoms, potentials, specifications, density, zero, zero).ionic
        with _setting("tip"):
            tip = local_forces(grid, atoms, potentials, specifications, density, zero, zero).ionic
        if beyond:
            # farther than c from every grid point: the table's own slope, the bits of tip
            np.testing.assert_array_equal(auto, tip)
        elif auto_tolerance is not None:
            self.assertLess(float(np.max(np.abs(auto[0] - reference))), auto_tolerance)
        else:
            # POTRE within 2 rs(2) of the point: the rule itself at the near point, and closer to the central
            # difference than tip
            change = _rule_change(grid, atoms[0].position, density, potentials["C"])
            self.assertGreater(float(np.max(np.abs(change))), 1.0e-8)
            np.testing.assert_allclose(auto[0] - tip[0], change, rtol=0.0, atol=1.0e-12)
            self.assertLess(float(np.max(np.abs(auto[0] - reference))), float(np.max(np.abs(tip[0] - reference))))
        if tip_error is not None:
            self.assertGreater(float(np.max(np.abs(tip[0] - reference))), tip_error)
        # the other atom lies far from every grid point: the bits of tip
        np.testing.assert_array_equal(auto[1], tip[1])

    # The central difference over 2e-3 bohr has a truncation of about 3e-6 Ry/bohr here also on the grid point, where
    # auto and tip agree.

    def test_fhi_table_next_to_a_grid_point(self) -> None:
        # forcloc.f90 leaves the point 5e-4 bohr away out (first sample 1.04e-3): off by 2 a d rho dV
        self._check("fhi", 5.0e-4, auto_tolerance=1.0e-5, tip_error=1.0e-3)

    def test_fhi_table_on_a_grid_point(self) -> None:
        self._check("fhi", 1.0e-11, auto_tolerance=1.0e-5, tip_error=None)

    def test_potre_table_within_its_first_intervals(self) -> None:
        # 2e-5 bohr from the point: between rs(2) = 1.28e-5 and c = 2.56e-5 bohr
        self._check("potre", 2.0e-5, auto_tolerance=None, tip_error=None)

    def test_potre_table_beyond_its_first_intervals(self) -> None:
        self._check("potre", 5.0e-4, auto_tolerance=None, tip_error=None, beyond=True)


class CellTests(unittest.TestCase):
    """Both routine sets of a cell at fixed density: the Fortran-derived sums (readable and the host ball sums of
    :mod:`parsec_python.acceleration.Forces.periodic_spline`) and the default routines (readable and
    :func:`parsec_python.acceleration.Forces.periodic_local.fast_periodic_local_forces` on the host)."""

    @staticmethod
    def _case(table: str, offset: float, *, fortran: bool, on_grid: bool = True):
        specifications = {"C" if table != "boron" else "B": _species(table)}
        potentials = _load(specifications)
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=_CUBE),
            PeriodicGridSettings(spacing=0.5, expansion_order=6, fortran_routines=fortran),
        )
        guess = (2.0, 2.2, 2.4)
        first = _near_point(grid, guess, offset) if on_grid else np.asarray(guess)
        symbol = next(iter(specifications))
        atoms = (Atom(symbol, first.tolist()), Atom(symbol, (first + np.array([1.21, 1.43, 1.37])).tolist()))
        density = superpose_atomic_density(grid, atoms, potentials, specifications, _CUBE)
        return grid, atoms, potentials, specifications, density

    @staticmethod
    def _fortran_forces(case, setting, *, fast: bool, xc=None):
        grid, atoms, potentials, specifications, density = case
        zero = np.zeros(grid.size)
        previous = fortran_sums._local_force_provider
        if fast:
            from parsec_python.acceleration.Forces.periodic_spline import spline_periodic_local_forces

            fortran_sums.set_local_force_provider(spline_periodic_local_forces)
        else:
            fortran_sums.set_local_force_provider(None)
        try:
            with _setting(setting):
                return fortran_sums.periodic_local_forces(
                    grid, atoms, potentials, specifications, _CUBE, density, zero, zero if xc is None else xc
                )
        finally:
            fortran_sums.set_local_force_provider(previous)

    @staticmethod
    def _default_forces(case, setting, *, fast: bool):
        grid, atoms, potentials, specifications, density = case
        zero = np.zeros(grid.size)
        with _setting(setting):
            if fast:
                from parsec_python.acceleration.Forces.periodic_local import fast_periodic_local_forces

                return fast_periodic_local_forces(grid, atoms, potentials, specifications, _CUBE, density, zero)
            return default_sums.dense_periodic_local_forces(grid, atoms, potentials, specifications, _CUBE, density, zero)

    @staticmethod
    def _energy(case):
        grid, atoms, potentials, specifications, density = case

        def energy(moved):
            return grid.volume_element * float(
                density @ ewald_local_ionic_potential(grid, moved, potentials, specifications, _CUBE)
            )

        return energy

    def test_fortran_routines_next_to_a_grid_point(self) -> None:
        for table, offset, tip_error in (
            ("fhi", 5.0e-4, 5.0e-4), ("fhi", 1.0e-11, None), ("potre", 2.0e-5, None), ("potre", 5.0e-4, None)
        ):
            case = self._case(table, offset, fortran=True)
            reference = _central_difference(self._energy(case), case[1], 0)
            for fast in (False, True):
                with self.subTest(table=table, offset=offset, fast=fast):
                    auto = self._fortran_forces(case, "auto", fast=fast).ionic
                    tip = self._fortran_forces(case, "tip", fast=fast).ionic
                    if table == "potre" and offset > 2.56e-5:
                        # beyond 2 rs(2) of every grid point: the table's own slope, the bits of tip
                        np.testing.assert_array_equal(auto, tip)
                    elif table == "potre":
                        # the rule itself at the near point, and closer to the central difference than tip
                        change = _rule_change(case[0], case[1][0].position, case[4], case[2]["C"], _CUBE)
                        self.assertGreater(float(np.max(np.abs(change))), 1.0e-8)
                        np.testing.assert_allclose(auto[0] - tip[0], change, rtol=0.0, atol=1.0e-12)
                        self.assertLess(
                            float(np.max(np.abs(auto[0] - reference))), float(np.max(np.abs(tip[0] - reference)))
                        )
                    else:
                        # (the readable sum adds the spline error of -2Z/r out to its longer reach: 2e-6)
                        self.assertLess(float(np.max(np.abs(auto[0] - reference))), 4.0e-6)
                    if tip_error is not None:
                        self.assertGreater(float(np.max(np.abs(tip[0] - reference))), tip_error)
                    np.testing.assert_array_equal(auto[1], tip[1])

    def test_default_routines_next_to_a_grid_point(self) -> None:
        # the default routines take POTRE tables (an FHI cell runs on the Fortran-derived routines); 2e-5 bohr from the
        # grid point, within c = 2 rs(2) = 2.56e-5 bohr
        case = self._case("potre", 2.0e-5, fortran=False)
        reference = _central_difference(self._energy(case), case[1], 0)
        results = {}
        for fast in (False, True):
            for setting in ("auto", "tip"):
                results[fast, setting] = self._default_forces(case, setting, fast=fast).ionic
            with self.subTest(fast=fast):
                change = _rule_change(case[0], case[1][0].position, case[4], case[2]["C"], _CUBE)
                np.testing.assert_allclose(results[fast, "auto"][0] - results[fast, "tip"][0], change, rtol=0.0,
                                           atol=1.0e-12)
                self.assertLess(
                    float(np.max(np.abs(results[fast, "auto"][0] - reference))),
                    float(np.max(np.abs(results[fast, "tip"][0] - reference))),
                )
                np.testing.assert_array_equal(results[fast, "auto"][1], results[fast, "tip"][1])
        np.testing.assert_allclose(results[True, "auto"], results[False, "auto"], rtol=0.0, atol=1.0e-7)

    def test_core_density_of_a_core_corrected_fhi_table(self) -> None:
        # the core term at fixed V_xc: d/dR of dV sum V_xc rho_core(R)
        case = self._case("boron", 5.0e-4, fortran=True)
        grid, atoms, potentials, specifications, density = case
        xc = -0.9 * np.cbrt(np.maximum(density, 0.0)) + 0.01 * np.sin(np.asarray(grid.coordinates)[:, 0])

        def energy(moved):
            core = superpose_atomic_density(grid, moved, potentials, specifications, _CUBE, core=True)
            return grid.volume_element * float(xc @ core)

        reference = _central_difference(energy, atoms, 0)
        for fast in (False, True):
            with self.subTest(fast=fast):
                auto = self._fortran_forces(case, "auto", fast=fast, xc=xc).core_correction
                tip = self._fortran_forces(case, "tip", fast=fast, xc=xc).core_correction
                self.assertLess(float(np.max(np.abs(auto[0] - reference))), 1.0e-6)
                self.assertGreater(float(np.max(np.abs(auto[0] - tip[0]))), 0.0)
                np.testing.assert_array_equal(auto[1], tip[1])

    def test_far_from_the_grid_the_bits_of_tip(self) -> None:
        for fortran in (True, False):
            case = self._case("potre", 0.0, fortran=fortran, on_grid=False)
            for fast in (False, True):
                with self.subTest(fortran=fortran, fast=fast):
                    run = self._fortran_forces if fortran else self._default_forces
                    auto, tip = run(case, "auto", fast=fast), run(case, "tip", fast=fast)
                    for term in ("ionic", "self_consistency", "core_correction"):
                        np.testing.assert_array_equal(getattr(auto, term), getattr(tip, term))


if __name__ == "__main__":
    unittest.main()
