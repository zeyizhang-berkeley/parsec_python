"""The centre radius of the projector harmonics and of their gradients (``PARSEC_PROJECTOR_CENTER``, 2026-11-21).

``nonloc.F90`` clamps the distance of a grid point to its atom below to the first radial sample ``rs(2)`` in the radial
interpolation and in ``ylm_real`` / ``ylm_cplx`` (``rnorm = max(|d|, rrmin)``), for the projector values and their
gradients alike.  Up to 2026-11-20 the projectors of this port divided the harmonics by ``min(rs(2), 1e-6 bohr)``
(the double grid and the spin-orbit projectors by ``|d|``), so between 1e-6 bohr and ``rs(2)`` a projector had the full
``w(rs(2))`` along the direction of the atom, and the gradient of a cell's nonlocal forces grew as ``w(rs(2))/|d|`` up
to ``w(rs(2))/1e-6``: an atom within round-off of a grid point got 0.1 to 0.75 Ry/bohr from that one point.  Under
``auto`` (the default) every builder divides by ``rs(2)``.  Checked here (CPU):

* the switch, its values and its validation before the SCF;
* the projector of a point within ``rs(2)`` is ``sqrt(dV) w(rs(2)) Y(d/rs(2))``, continuous down to the atom;
* the gradients of a cell are the derivative of its projectors with respect to the atom, also within ``rs(2)``, and
  an atom 1e-11 bohr off a grid point gets the gradient of the atom on it; under ``off`` the former bits;
* the double-grid and spin-orbit projectors take the same centre radius; the harmonic gradients with ``inside``.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from parsec_python.models import Atom, SpeciesPotential
from parsec_python.tests.unit.test_default_routines_kpoint_wedge import _C_POTRE, _diamond, _grid
from parsec_python.V_ion import ionic_potential as ionic

_ENV = ionic.PROJECTOR_CENTER_ENV
#: the columns of atom 4 (s and p; the local channel is d): four per atom
_ATOM4 = slice(16, 20)


def _potential():
    from parsec_python.Pseudopotential import read_parsec_pseudopotential

    return read_parsec_pseudopotential(_C_POTRE)


def _cell(offset=(0.0, 0.0, 0.0), spline: bool = True):
    """The 8-atom diamond cube on the half-step grid of 18 points per edge: the atoms at a/4 sit on grid points."""

    lattice, atoms = _diamond(1, decimals=16)
    atoms = list(atoms)
    atoms[4] = Atom("C", (np.asarray(atoms[4].position) + np.asarray(offset, dtype=float)).tolist())
    grid = _grid(lattice, 18)
    specification = SpeciesPotential(path=_C_POTRE, local_angular_momentum=2, use_spline=spline)
    return lattice, atoms, grid, {"C": _potential()}, {"C": specification}


def _projectors(case, setting: str):
    from parsec_python.V_ion import build_nonlocal_projectors

    lattice, atoms, grid, potentials, specifications = case
    with mock.patch.dict(os.environ, {_ENV: setting}):
        return build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice).projectors.toarray()


def _gradients(case, setting: str):
    from parsec_python.Forces.periodic import build_periodic_projector_gradients

    lattice, atoms, grid, potentials, specifications = case
    with mock.patch.dict(os.environ, {_ENV: setting}):
        result = build_periodic_projector_gradients(grid, atoms, potentials, specifications, lattice)
    return [component.toarray() for component in result.components]


def _nearest(case, atom_index: int = 4):
    _, atoms, grid, _, _ = case
    coordinates = np.asarray(grid.coordinates)
    distances = np.linalg.norm(coordinates - np.asarray(atoms[atom_index].position), axis=1)
    return int(np.argmin(distances)), float(np.min(distances))


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            self.assertEqual(ionic.projector_center_setting(), "auto")
        for value, expected in (("auto", "auto"), (" OFF ", "off"), ("", "auto")):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertEqual(ionic.projector_center_setting(), expected)
        with mock.patch.dict(os.environ, {_ENV: "rs2"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_PROJECTOR_CENTER must be auto or off"):
                ionic.projector_center_setting()
            from parsec_python.acceleration.Forces.periodic_local import validate_local_force_settings

            with self.assertRaisesRegex(ValueError, "PARSEC_PROJECTOR_CENTER"):
                validate_local_force_settings()

    def test_radius(self) -> None:
        potential = _potential()
        first = float(potential.radii[0])
        self.assertGreater(first, ionic.PROJECTOR_CENTER_ROUND_OFF)
        self.assertEqual(ionic.projector_center_radius(potential, "auto"), first)
        self.assertEqual(ionic.projector_center_radius(potential, "off"), ionic.PROJECTOR_CENTER_ROUND_OFF)
        self.assertEqual(ionic.projector_center_radius(potential, "off", off_radius=0.0), 0.0)
        self.assertEqual(ionic.projector_center_radius(potential, "auto", off_radius=0.0), first)


class ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = float(_potential().radii[0])
        cls.on = _cell()
        cls.near = _cell(offset=(1.0e-11, -0.7e-11, 0.4e-11))
        # between 1e-6 bohr and rs(2) (1.3e-5 bohr for this table)
        cls.inside = _cell(offset=(0.3 * cls.first, 0.2 * cls.first, -0.4 * cls.first))
        cls.outside = _cell(offset=(0.013, -0.021, 0.017))

    def test_the_projector_within_the_first_sample(self) -> None:
        row, distance = _nearest(self.inside)
        self.assertLess(distance, self.first)
        self.assertGreater(distance, ionic.PROJECTOR_CENTER_ROUND_OFF)
        lattice, atoms, grid, potentials, _ = self.inside
        potential = potentials["C"]
        relative = np.asarray(grid.coordinates[row]) - np.asarray(atoms[4].position)
        auto = _projectors(self.inside, "auto")
        off = _projectors(self.inside, "off")
        # atom 4 owns columns 4*(3+1)..: the s channel (1 column) then the p channel (3 columns), local d
        p_columns = slice(17, 20)
        radial, _ = potential.radial_projector(1, 2)
        expected = np.sqrt(grid.volume_element) * float(radial[0]) * 0.4886025119029199 * relative / self.first
        np.testing.assert_allclose(auto[row, p_columns], expected, rtol=1.0e-12, atol=0.0)
        # off: the full w(rs(2)) along the direction of the atom
        np.testing.assert_allclose(
            np.linalg.norm(off[row, p_columns]), np.sqrt(grid.volume_element) * abs(float(radial[0])) * 0.4886025119029199,
            rtol=1.0e-9,
        )

    def test_continuous_down_to_the_atom_and_unchanged_outside(self) -> None:
        on = _projectors(self.on, "auto")
        near = _projectors(self.near, "auto")
        self.assertLess(float(np.max(np.abs(near - on))), 1.0e-9 * float(np.max(np.abs(on))))
        # atom 4 far from every grid point (the atoms at a/4 + a/2 lattice sites stay on theirs): the same bits
        np.testing.assert_array_equal(_projectors(self.outside, "auto")[:, _ATOM4], _projectors(self.outside, "off")[:, _ATOM4])

    def test_gradients_are_the_derivative_of_the_projectors(self) -> None:
        # the atom 1e-11 bohr off the grid point and the atom within rs(2): central differences of the projector
        # columns of atom 4 with respect to its position, steps far below rs(2) (the projector is a polynomial there)
        for label, case in (("near", self.near), ("inside", self.inside)):
            row, _ = _nearest(case)
            gradients = _gradients(case, "auto")
            step = 0.05 * self.first
            lattice, atoms, grid, potentials, specifications = case
            for axis in range(3):
                shifted = []
                for sign in (1.0, -1.0):
                    moved = list(atoms)
                    position = np.asarray(atoms[4].position, dtype=float).copy()
                    position[axis] += sign * step
                    moved[4] = Atom("C", position.tolist())
                    shifted.append(_projectors((lattice, moved, grid, potentials, specifications), "auto"))
                derivative = (shifted[0] - shifted[1]) / (2.0 * step)
                columns = _ATOM4
                with self.subTest(case=label, axis=axis):
                    # the gradient is with respect to the grid point: minus the derivative with respect to the atom
                    np.testing.assert_allclose(
                        gradients[axis][row, columns], -derivative[row, columns], rtol=0.0,
                        atol=1.0e-9 * float(np.max(np.abs(gradients[axis][row, columns]))) + 1.0e-14,
                    )

    def test_round_off_takes_the_gradient_of_the_atom_on_the_point(self) -> None:
        row, _ = _nearest(self.near)
        on = _gradients(self.on, "auto")
        near = _gradients(self.near, "auto")
        former = _gradients(self.near, "off")
        for axis in range(3):
            np.testing.assert_allclose(near[axis][row], on[axis][row], rtol=0.0, atol=1.0e-9 * np.max(np.abs(on[axis])))
        # the former bits: w(rs(2))/1e-6 at the point, rs(2)/1e-6 = 13 times the slope of the p projector
        ratio = np.linalg.norm(np.concatenate([g[row] for g in former])) / np.linalg.norm(
            np.concatenate([g[row] for g in near])
        )
        self.assertGreater(ratio, 5.0)
        # atom 4 far from every grid point: the same bits under both values
        for auto, off in zip(_gradients(self.outside, "auto"), _gradients(self.outside, "off")):
            np.testing.assert_array_equal(auto[:, _ATOM4], off[:, _ATOM4])


def _default_gradients(case, setting: str):
    """The gradients of Andrew Grieder's routines (``Forces.nonlocal_force.build_periodic_projector_gradients``)."""

    from parsec_python.Forces.nonlocal_force import build_periodic_projector_gradients

    lattice, atoms, grid, potentials, specifications = case
    with mock.patch.dict(os.environ, {_ENV: setting}):
        result = build_periodic_projector_gradients(grid, atoms, potentials, specifications, lattice)
    return [component.toarray() for component in result.components]


class DefaultRoutineGradientTests(unittest.TestCase):
    """The default routines' gradients take the same rule within ``rs(2)`` (since 2026-11-21): an atom on a grid point
    of a cell of these routines got a nonlocal force 1.3e-3 Ry/bohr off the central difference of the energy (64-atom
    diamond, 2x2x2 mesh), because the gradient of the point on the atom was evaluated at 1e-3 bohr."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.first = float(_potential().radii[0])
        cls.on = _cell()
        cls.near = _cell(offset=(1.0e-11, -0.7e-11, 0.4e-11))
        cls.inside = _cell(offset=(0.3 * cls.first, 0.2 * cls.first, -0.4 * cls.first))
        cls.outside = _cell(offset=(0.013, -0.021, 0.017))

    def test_within_the_first_sample_they_are_those_of_the_fortran_derived_routines(self) -> None:
        for label, case in (("on", self.on), ("near", self.near), ("inside", self.inside)):
            row, distance = _nearest(case)
            self.assertLess(distance, self.first)
            default = _default_gradients(case, "auto")
            fortran = _gradients(case, "auto")
            for axis in range(3):
                with self.subTest(case=label, axis=axis):
                    scale = float(np.max(np.abs(fortran[axis][row, _ATOM4])))
                    self.assertGreater(scale, 0.0)
                    np.testing.assert_allclose(
                        default[axis][row, _ATOM4], fortran[axis][row, _ATOM4], rtol=0.0, atol=1.0e-12 * scale
                    )

    def test_off_and_the_other_points_keep_the_former_bits(self) -> None:
        row, _ = _nearest(self.on)
        auto = _default_gradients(self.on, "auto")
        off = _default_gradients(self.on, "off")
        others = np.ones(auto[0].shape[0], dtype=bool)
        others[row] = False
        for axis in range(3):
            # of the columns of atom 4 only its point within rs(2) changes; it changes (the floor took 1e-3 bohr there)
            # (the other atoms at a/4 sites sit on grid points too, so their columns change at their own points)
            np.testing.assert_array_equal(auto[axis][others][:, _ATOM4], off[axis][others][:, _ATOM4])
            self.assertGreater(float(np.max(np.abs(auto[axis][row, _ATOM4] - off[axis][row, _ATOM4]))), 0.0)
        # atom 4 far from every grid point: the same bits under both values
        for first, second in zip(_default_gradients(self.outside, "auto"), _default_gradients(self.outside, "off")):
            np.testing.assert_array_equal(first[:, _ATOM4], second[:, _ATOM4])


class OtherBuilderTests(unittest.TestCase):
    def test_spin_orbit_and_double_grid_harmonics(self) -> None:
        from parsec_python.Eigensolvers.perturbative_soc import complex_spherical_harmonics

        center = 2.0e-5
        points = np.array([[1.0e-12, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5 * center, 0.0, 0.0], [1.0, 0.5, 0.0]])
        old = complex_spherical_harmonics(1, points)
        new = complex_spherical_harmonics(1, points, center_radius=center)
        np.testing.assert_array_equal(new[1], 0.0)
        np.testing.assert_array_equal(new[3], old[3])
        np.testing.assert_allclose(np.abs(new[2]), 0.5 * np.abs(old[2]), rtol=1.0e-14)
        np.testing.assert_allclose(np.abs(new[0]), 1.0e-12 / center * np.abs(old[0]), rtol=1.0e-9)

    def test_harmonic_gradients_inside(self) -> None:
        from parsec_python.Forces.nonlocal_force import real_harmonics_with_gradients
        from parsec_python.Forces.spin_orbit_fortran import complex_harmonics_with_gradients
        from parsec_python.V_ion.ionic_potential import real_spherical_harmonics
        from parsec_python.Eigensolvers.perturbative_soc import complex_spherical_harmonics

        center = 1.0e-3
        rng = np.random.default_rng(3)
        relative = 0.6 * center * rng.standard_normal((5, 3)) / np.sqrt(3.0)
        relative = relative[np.linalg.norm(relative, axis=1) < center]
        inside = np.ones(len(relative), dtype=bool)
        inverse = np.full(len(relative), 1.0 / center)
        unit = relative / center
        step = 1.0e-7
        for l in (1, 2, 3):
            _, gradients = real_harmonics_with_gradients(l, unit, inverse, inside=inside)
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                derivative = (
                    real_spherical_harmonics(l, relative + shift, center_radius=center)
                    - real_spherical_harmonics(l, relative - shift, center_radius=center)
                ) / (2.0 * step)
                with self.subTest(kind="real", l=l, axis=axis):
                    np.testing.assert_allclose(gradients[:, :, axis], derivative, rtol=1.0e-6, atol=1.0e-6)
        for l in (1, 2):
            _, gradients = complex_harmonics_with_gradients(l, unit, inverse, inside=inside)
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                derivative = (
                    complex_spherical_harmonics(l, relative + shift, center_radius=center)
                    - complex_spherical_harmonics(l, relative - shift, center_radius=center)
                ) / (2.0 * step)
                with self.subTest(kind="complex", l=l, axis=axis):
                    np.testing.assert_allclose(gradients[:, :, axis], derivative, rtol=1.0e-6, atol=1.0e-6)

    def test_double_grid_projectors_take_the_centre(self) -> None:
        from parsec_python.V_ion.double_grid import build_double_grid_projectors

        lattice, atoms, grid, potentials, specifications = _cell(offset=(2.0e-6, -3.0e-6, 1.0e-6))
        values = {}
        for setting in ("auto", "off"):
            with mock.patch.dict(os.environ, {_ENV: setting}):
                values[setting] = build_double_grid_projectors(
                    grid, atoms, potentials, specifications, lattice_vectors=lattice, order=2
                ).projectors.toarray()
        row, _ = _nearest((lattice, atoms, grid, potentials, specifications))
        # the coarse point on the atom sums fine points at 2e-6 .. h/2: only its own fine point lies within rs(2)
        self.assertGreater(float(np.max(np.abs(values["auto"][row, _ATOM4] - values["off"][row, _ATOM4]))), 0.0)
        others = np.ones(values["auto"].shape[0], dtype=bool)
        others[row] = False
        np.testing.assert_array_equal(values["auto"][others][:, _ATOM4], values["off"][others][:, _ATOM4])


class NativeTests(unittest.TestCase):
    def test_native_builder_takes_the_reference_rule(self) -> None:
        try:
            from parsec_python.acceleration.backends import native as native_backend

            module = native_backend._load_native()
        except Exception:  # noqa: BLE001 - the extension is optional
            self.skipTest("the native extension is not built")
        if not hasattr(module, "RadialGridEvaluator"):
            self.skipTest("the native extension has no radial kernel")
        from parsec_python.acceleration.V_ion.native_ionic import NativeIonicBuilders
        from parsec_python.Grid.cluster import build_cluster_grid
        from parsec_python.models import GridSettings
        from parsec_python.V_ion import build_nonlocal_projectors

        potential = _potential()
        first = float(potential.radii[0])
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        coordinates = np.asarray(grid.coordinates)
        point = coordinates[np.argmin(np.linalg.norm(coordinates - np.array([0.3, 0.2, 0.1]), axis=1))]
        atoms = [
            Atom("C", (point + np.array([0.3, -0.2, 0.4]) * first).tolist()),
            Atom("C", (point + np.array([1.21, 1.43, 1.37])).tolist()),
        ]
        potentials = {"C": potential}
        for spline in (True, False):
            specifications = {"C": SpeciesPotential(path=_C_POTRE, local_angular_momentum=2, use_spline=spline)}
            for setting in ("auto", "off"):
                with self.subTest(spline=spline, setting=setting), mock.patch.dict(os.environ, {_ENV: setting}):
                    native = NativeIonicBuilders().build_nonlocal_projectors(grid, atoms, potentials, specifications)
                    reference = build_nonlocal_projectors(grid, atoms, potentials, specifications)
                    np.testing.assert_allclose(
                        native.projectors.toarray(), reference.projectors.toarray(), rtol=0.0, atol=1.0e-13
                    )


if __name__ == "__main__":
    unittest.main()
