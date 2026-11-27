"""Tests for non-collinear magnetism: XC, Hamiltonian, energy, SCF, input."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    ca_lda_spin_polarized,
)
from parsec_python.cli import main as cli_main
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Energy import (
    total_energy_no_degeneracy_spin_polarized,
    total_energy_noncollinear,
)
from parsec_python.Grid import build_cluster_grid, monkhorst_pack_grid
from parsec_python.Hamiltonian.spinor_operator import (
    SpinorKohnShamHamiltonian,
    apply_zeeman_field,
)
from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints_spin_polarized
from parsec_python.SCF.noncollinear import (
    run_self_consistent_noncollinear,
    run_self_consistent_noncollinear_kpoints,
    spinor_density_and_magnetization,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized
from parsec_python.V_xc import noncollinear_xc, pbe_spin_polarized
from parsec_python.tests.archives import read_archive

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class NoncollinearXCTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.n = 10 ** rng.uniform(-2, 0, 200)
        direction = rng.normal(size=(3, 200))
        direction /= np.linalg.norm(direction, axis=0)
        self.m = direction * self.n * rng.uniform(0.05, 0.95, 200)

    def _lsda(self, up, down):
        return ca_lda_spin_polarized(up, down, 1.0)

    def test_collinear_limit_matches_lsda(self) -> None:
        m = np.zeros((3, self.n.size))
        m[2] = 0.6 * self.n
        result = noncollinear_xc(self.n, m, self._lsda)
        direct = ca_lda_spin_polarized(0.8 * self.n, 0.2 * self.n, 1.0)
        np.testing.assert_allclose(
            result.potential, 0.5 * (direct.potential_up + direct.potential_down)
        )
        np.testing.assert_allclose(result.field[2], 0.5 * (direct.potential_up - direct.potential_down))
        np.testing.assert_allclose(result.field[:2], 0.0, atol=1e-14)
        self.assertAlmostEqual(result.total_energy, direct.total_energy, places=12)

    def test_rotation_covariance(self) -> None:
        rotation = _rotation([1.0, 2.0, -0.5], 0.9)
        base = noncollinear_xc(self.n, self.m, self._lsda)
        rotated = noncollinear_xc(self.n, rotation @ self.m, self._lsda)
        np.testing.assert_allclose(rotated.potential, base.potential, rtol=1e-12)
        np.testing.assert_allclose(rotated.field, rotation @ base.field, rtol=1e-10, atol=1e-13)
        self.assertAlmostEqual(rotated.total_energy, base.total_energy, places=10)

    def test_field_is_parallel_to_magnetization(self) -> None:
        result = noncollinear_xc(self.n, self.m, self._lsda)
        cross = np.cross(result.field.T, self.m.T)
        self.assertLess(np.max(np.abs(cross)), 1e-12)

    def test_zero_magnetization_gives_zero_field(self) -> None:
        zero = np.zeros((3, self.n.size))
        result = noncollinear_xc(self.n, zero, self._lsda)
        np.testing.assert_array_equal(result.field, 0.0)
        self.assertTrue(np.all(np.isfinite(result.potential)))

    def test_magnetization_above_density_is_clipped(self) -> None:
        m = np.zeros((3, self.n.size))
        m[0] = self.n * (1.0 + 1e-13)
        result = noncollinear_xc(self.n, m, self._lsda)
        self.assertTrue(np.all(np.isfinite(result.potential)))

    def test_pbe_rotation_covariance_on_a_grid(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.7, radius=3.2, expansion_order=8, shift=(0.5, 0.5, 0.5))
        )
        r2 = np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)
        n = 0.2 * np.exp(-0.5 * r2) + 0.01
        direction = np.stack([np.cos(0.3 * grid.coordinates[:, 0]), np.sin(0.3 * grid.coordinates[:, 0]), 0 * r2])
        m = 0.5 * n * direction
        rotation = _rotation([0.3, 1.0, 0.2], 1.1)

        def functional(up, down):
            return pbe_spin_polarized(up, down, grid)

        base = noncollinear_xc(n, m, functional)
        rotated = noncollinear_xc(n, rotation @ m, functional)
        self.assertAlmostEqual(rotated.total_energy, base.total_energy, places=9)
        np.testing.assert_allclose(rotated.field, rotation @ base.field, rtol=1e-8, atol=1e-10)


class NoncollinearAxisFrameTests(unittest.TestCase):
    """The frames of a gradient functional (PARSEC_NONCOLLINEAR_GGA_FRAME, V_xc.noncollinear)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.6, radius=3.6, expansion_order=8, shift=(0.5, 0.5, 0.5))
        )
        x = cls.grid.coordinates
        r2 = np.einsum("ij,ij->i", x, x)
        cls.n = 0.3 * np.exp(-0.6 * r2) + 0.02
        # a polarization with a nodal plane through grid points (the plane of x nearest 0.4)
        plane = x[np.argmin(np.abs(x[:, 0] - 0.4)), 0]
        cls.signed = 0.4 * cls.n * np.tanh(2.0 * (x[:, 0] - plane))
        cls.core = 0.05 * np.exp(-2.0 * r2)

    def _pbe(self, up, down):
        return pbe_spin_polarized(up, down, self.grid, self.core)

    def test_collinear_magnetization_is_the_collinear_functional(self) -> None:
        # along z, nodes included: the collinear PBE of (n +- m_z)/2, bit for bit
        from parsec_python.V_xc.noncollinear import axis_frame_xc

        m = np.zeros((3, self.n.size))
        m[2] = self.signed
        result = axis_frame_xc(self.n, m, self._pbe, np.array([0.0, 0.0, 1.0]), self.core)
        direct = self._pbe(0.5 * (self.n + self.signed), 0.5 * (self.n - self.signed))
        self.assertEqual(result.total_energy, direct.total_energy)
        np.testing.assert_array_equal(result.potential, 0.5 * (direct.potential_up + direct.potential_down))
        np.testing.assert_array_equal(result.field[2], 0.5 * (direct.potential_up - direct.potential_down))
        np.testing.assert_array_equal(result.field[:2], 0.0)
        # the local frame of |m| differs at the nodes on this grid
        local = noncollinear_xc(self.n, m, self._pbe)
        self.assertGreater(abs(local.total_energy - direct.total_energy), 1.0e-8)

    def test_rotation_covariance(self) -> None:
        from parsec_python.V_xc.noncollinear import axis_frame_xc

        x = self.grid.coordinates
        m = np.stack([0.05 * self.n * np.sin(x[:, 1]), 0.02 * self.n * np.cos(x[:, 2]), self.signed])
        rotation = _rotation([0.3, 1.0, 0.2], 1.1)
        z = np.array([0.0, 0.0, 1.0])
        base = axis_frame_xc(self.n, m, self._pbe, z, self.core)
        rotated = axis_frame_xc(self.n, rotation @ m, self._pbe, rotation @ z, self.core)
        self.assertAlmostEqual(rotated.total_energy, base.total_energy, places=11)
        np.testing.assert_allclose(rotated.potential, base.potential, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(rotated.field, rotation @ base.field, rtol=1e-8, atol=1e-10)

    def test_potential_and_field_are_the_derivatives_of_the_energy(self) -> None:
        # central differences of the energy in m (every component) and n at points beside the node and elsewhere,
        # with transverse parts (the term of the axis' dependence on m included)
        from parsec_python.V_xc.noncollinear import axis_frame_xc

        x = self.grid.coordinates
        m = np.stack([0.05 * self.n * np.sin(x[:, 1]), 0.02 * self.n * np.cos(x[:, 2]), self.signed])
        z = np.array([0.0, 0.0, 1.0])
        base = axis_frame_xc(self.n, m, self._pbe, z, self.core)
        dv = self.grid.volume_element
        near = np.argsort(np.abs(self.signed))[:3]
        points = list(near) + [0, self.n.size // 3, int(np.argmax(self.n))]
        step = 1.0e-6
        for i in points:
            for k in range(3):
                plus, minus = m.copy(), m.copy()
                plus[k, i] += step
                minus[k, i] -= step
                slope = (
                    axis_frame_xc(self.n, plus, self._pbe, z, self.core).total_energy
                    - axis_frame_xc(self.n, minus, self._pbe, z, self.core).total_energy
                ) / (2.0 * step * dv)
                self.assertAlmostEqual(slope, base.field[k, i], delta=2.0e-5 * (1.0 + abs(slope)))
            plus, minus = self.n.copy(), self.n.copy()
            plus[i] += step
            minus[i] -= step
            slope = (
                axis_frame_xc(plus, m, self._pbe, z, self.core).total_energy
                - axis_frame_xc(minus, m, self._pbe, z, self.core).total_energy
            ) / (2.0 * step * dv)
            self.assertAlmostEqual(slope, base.potential[i], delta=2.0e-5 * (1.0 + abs(slope)))

    def test_field_is_continuous_where_m_vanishes(self) -> None:
        # transverse noise at the nodes of m_z turns the field of the local frame there, not that of the axis frame
        from parsec_python.V_xc.noncollinear import axis_frame_xc

        m = np.zeros((3, self.n.size))
        m[2] = self.signed
        noisy = m.copy()
        rng = np.random.default_rng(3)
        noisy[:2] = 1.0e-7 * rng.normal(size=(2, self.n.size))
        z = np.array([0.0, 0.0, 1.0])
        axis_change = np.max(np.abs(axis_frame_xc(self.n, noisy, self._pbe, z).field
                                    - axis_frame_xc(self.n, m, self._pbe, z).field))
        local_change = np.max(np.abs(noncollinear_xc(self.n, noisy, self._pbe).field
                                     - noncollinear_xc(self.n, m, self._pbe).field))
        self.assertLess(axis_change, 1.0e-4)
        self.assertGreater(local_change, 1.0e-3)

    def test_collinear_frame(self) -> None:
        # the collinear functional of m.u, the field along u: transverse parts get no field and no energy; the field
        # and potential are the derivatives of that energy
        from parsec_python.V_xc.noncollinear import collinear_frame_xc

        x = self.grid.coordinates
        u = np.array([0.0, 0.6, 0.8])
        m = np.outer(u, self.signed) + np.stack([0.05 * self.n * np.sin(x[:, 1]), 0.0 * self.n, 0.0 * self.n])
        result = collinear_frame_xc(self.n, m, self._pbe, u)
        a = u @ m
        direct = self._pbe(0.5 * (self.n + a), 0.5 * (self.n - a))
        self.assertEqual(result.total_energy, direct.total_energy)
        np.testing.assert_array_equal(result.field, np.outer(u, 0.5 * (direct.potential_up - direct.potential_down)))
        np.testing.assert_array_equal(result.potential, 0.5 * (direct.potential_up + direct.potential_down))
        dv = self.grid.volume_element
        step = 1.0e-6
        for i in (0, int(np.argmin(np.abs(self.signed))), int(np.argmax(self.n))):
            for k in range(3):
                plus, minus = m.copy(), m.copy()
                plus[k, i] += step
                minus[k, i] -= step
                slope = (
                    collinear_frame_xc(self.n, plus, self._pbe, u).total_energy
                    - collinear_frame_xc(self.n, minus, self._pbe, u).total_energy
                ) / (2.0 * step * dv)
                self.assertAlmostEqual(slope, result.field[k, i], delta=2.0e-5 * (1.0 + abs(slope)))

    def test_guard_of_the_axis_frame(self) -> None:
        # a magnetic state far from one axis (two halves with orthogonal moments) leaves the axis frame after
        # GUARD_EVALUATIONS evaluations in a row; a collinear one keeps it
        from parsec_python.V_xc.noncollinear import (
            FRAME_AXIS, FRAME_LOCAL, GUARD_EVALUATIONS, NoncollinearFrame, axis_frame_xc,
        )

        x = self.grid.coordinates
        z = np.array([0.0, 0.0, 1.0])
        canted = np.zeros((3, self.n.size))
        left = x[:, 0] < 0.0
        canted[2, left] = 0.3 * self.n[left]
        canted[0, ~left] = 0.3 * self.n[~left]
        frame = NoncollinearFrame(FRAME_AXIS, z)
        for count in range(1, GUARD_EVALUATIONS):
            result = frame.evaluate(self.n, canted, self._pbe, None, 1.0)
            self.assertEqual(frame.mode, FRAME_AXIS)
        self.assertEqual(result.total_energy, axis_frame_xc(self.n, canted, self._pbe, z).total_energy)
        result = frame.evaluate(self.n, canted, self._pbe, None, 1.0)
        self.assertEqual((frame.mode, frame.left_axis_at), (FRAME_LOCAL, GUARD_EVALUATIONS))
        self.assertEqual(result.total_energy, noncollinear_xc(self.n, canted, self._pbe).total_energy)
        collinear = np.zeros((3, self.n.size))
        collinear[2] = self.signed
        frame = NoncollinearFrame(FRAME_AXIS, z)
        for _ in range(2 * GUARD_EVALUATIONS):
            frame.evaluate(self.n, collinear, self._pbe, None, 1.0)
        self.assertEqual((frame.mode, frame.left_axis_at), (FRAME_AXIS, None))

    def test_which_runs_take_which_frame(self) -> None:
        from parsec_python.V_xc.noncollinear import frame_decision

        species = {"Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)}

        def atoms(*moments):
            return [Atom("Au", [float(i), 0.0, 0.0], moment) for i, moment in enumerate(moments)]

        def decide(functional, positions, spin_orbit=False, setting="auto"):
            frame = frame_decision(functional, positions, species, spin_orbit, setting)
            return frame.mode, None if frame.axis is None else tuple(np.round(frame.axis, 12))

        parallel = atoms([0, 0, 1], [0, 0, 2])
        antiparallel = atoms([1, 1, 0], [-1, -1, 0])
        canted = atoms([0, 0, 1], [1, 0, 1])
        self.assertEqual(decide("pbe", parallel), ("collinear", (0.0, 0.0, 1.0)))
        self.assertEqual(decide("pbe", parallel, spin_orbit=True), ("axis", (0.0, 0.0, 1.0)))
        half = round(np.sqrt(0.5), 12)
        self.assertEqual(decide("pbe", antiparallel), ("collinear", (half, half, 0.0)))
        # no initial moment (Quantum ESPRESSO's compute_ux), non-collinear moments, off, the local functionals
        self.assertEqual(decide("pbe", atoms(None, None)), ("local", None))
        self.assertEqual(decide("pbe", canted, spin_orbit=True), ("local", None))
        self.assertEqual(decide("pbe", parallel, setting="off"), ("local", None))
        self.assertEqual(decide("ca", parallel), ("local", None))
        self.assertEqual(decide("pw92", parallel, spin_orbit=True), ("local", None))
        self.assertEqual(decide("libxc:lda_x+lda_c_pw", parallel), ("local", None))

    def test_the_frame_is_decided_once_per_run(self) -> None:
        # the moved structures keep the frame of the first one, also with atoms rebuilt without their moments (the
        # Fortran-derived ionic loops)
        from dataclasses import replace

        from parsec_python.V_xc.noncollinear import system_frame

        problem = _au_cluster([0, 0, 1], xc="pbe")
        system = prepare_single_point(problem)
        # decided when the system is built: a copy made before the first XC evaluation (the loosened SCF of an
        # upscaled relaxation) shares the object
        frame = system.noncollinear_frame
        self.assertIsNotNone(frame)
        scf = replace(system.input.scf, convergence_criterion=1e-3)
        loosened = replace(system, input=replace(system.input, scf=scf))
        self.assertIs(loosened.noncollinear_frame, frame)
        self.assertIs(system_frame(loosened), frame)
        self.assertIs(system_frame(system), frame)
        self.assertEqual(frame.mode, "collinear")
        self.assertIsNone(prepare_single_point(_au_cluster(None, nc=False, xc="pbe")).noncollinear_frame)
        bare = tuple(Atom(atom.symbol, atom.position) for atom in system.atoms)
        moved = replace(system, atoms=bare, input=replace(system.input, atoms=bare))
        self.assertIs(system_frame(moved), frame)
        fresh = prepare_single_point(replace(problem, atoms=list(bare)))
        self.assertEqual(system_frame(fresh).mode, "local")

    def test_setting(self) -> None:
        from unittest import mock

        from parsec_python.V_xc.noncollinear import NONCOLLINEAR_GGA_FRAME_ENV, noncollinear_gga_frame_setting

        with mock.patch.dict("os.environ", {NONCOLLINEAR_GGA_FRAME_ENV: "", "PARSEC_FORTRAN_NUMERICS": "off"}):
            self.assertEqual(noncollinear_gga_frame_setting(), "auto")
        with mock.patch.dict("os.environ", {NONCOLLINEAR_GGA_FRAME_ENV: "", "PARSEC_FORTRAN_NUMERICS": "on"}):
            self.assertEqual(noncollinear_gga_frame_setting(), "off")
        for value in ("yes", "on"):
            with mock.patch.dict("os.environ", {NONCOLLINEAR_GGA_FRAME_ENV: value}):
                with self.assertRaisesRegex(ValueError, NONCOLLINEAR_GGA_FRAME_ENV):
                    noncollinear_gga_frame_setting()


class ZeemanFieldTests(unittest.TestCase):
    def test_hermitian_and_pauli_structure(self) -> None:
        rng = np.random.default_rng(1)
        n = 30
        field = rng.normal(size=(3, n))
        up = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        down = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        a_up, a_down = apply_zeeman_field(field, up, down)
        up2 = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        down2 = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        b_up, b_down = apply_zeeman_field(field, up2, down2)
        lhs = np.vdot(up2, a_up) + np.vdot(down2, a_down)
        rhs = np.vdot(b_up, up) + np.vdot(b_down, down)
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)
        # B along z -> diagonal +/-Bz; B along x -> pure swap.
        z_only = np.zeros((3, n))
        z_only[2] = field[2]
        zu, zd = apply_zeeman_field(z_only, up, down)
        np.testing.assert_allclose(zu, field[2][:, None] * up)
        np.testing.assert_allclose(zd, -field[2][:, None] * down)
        x_only = np.zeros((3, n))
        x_only[0] = field[0]
        xu, xd = apply_zeeman_field(x_only, up, down)
        np.testing.assert_allclose(xu, field[0][:, None] * down)
        np.testing.assert_allclose(xd, field[0][:, None] * up)


class SpinorMagnetizationTests(unittest.TestCase):
    def _single(self, up, down):
        spinor = np.array([[up], [down]], dtype=complex)  # one grid point, one state
        return spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0)

    def test_known_spin_directions(self) -> None:
        s = 1.0 / np.sqrt(2.0)
        for up, down, expected in (
            (1.0, 0.0, (0, 0, 1)),
            (0.0, 1.0, (0, 0, -1)),
            (s, s, (1, 0, 0)),
            (s, -s, (-1, 0, 0)),
            (s, 1j * s, (0, 1, 0)),
            (s, -1j * s, (0, -1, 0)),
        ):
            n, m = self._single(up, down)
            self.assertAlmostEqual(float(n[0]), 1.0)
            np.testing.assert_allclose(m[:, 0], expected, atol=1e-12)

    def test_pure_state_has_unit_polarization(self) -> None:
        rng = np.random.default_rng(2)
        spinor = rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))
        spinor /= np.linalg.norm(spinor)
        n, m = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(m[:, 0])), float(n[0]), places=12)

    def test_k_weight_scales_linearly(self) -> None:
        spinor = np.array([[0.6], [0.8j]], dtype=complex)
        n1, m1 = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0, 1.0)
        n2, m2 = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0, 0.25)
        np.testing.assert_allclose(n2, 0.25 * n1)
        np.testing.assert_allclose(m2, 0.25 * m1)


class NoncollinearEnergyTests(unittest.TestCase):
    def test_z_aligned_matches_collinear_energy(self) -> None:
        rng = np.random.default_rng(3)
        n_grid = 40
        up = rng.uniform(0.01, 0.2, n_grid)
        down = rng.uniform(0.01, 0.2, n_grid)
        v_in_up, v_in_down = rng.normal(size=(2, n_grid))
        ionic, hartree = rng.normal(size=(2, n_grid))
        vxc_up, vxc_down = rng.normal(size=(2, n_grid))
        eig = rng.normal(size=12)
        occ = rng.uniform(0, 1, 12)
        collinear = total_energy_no_degeneracy_spin_polarized(
            eig, occ, up, down, v_in_up, v_in_down, ionic, hartree, vxc_up, vxc_down,
            -1.3, 0.7, 0.05, alpha_z_energy=0.1, band_energy_weight=0.5,
        )
        magnetization = np.zeros((3, n_grid))
        magnetization[2] = up - down
        input_field = np.zeros((3, n_grid))
        input_field[2] = 0.5 * (v_in_up - v_in_down)
        xc_field = np.zeros((3, n_grid))
        xc_field[2] = 0.5 * (vxc_up - vxc_down)
        general = total_energy_noncollinear(
            eig, occ, up + down, magnetization,
            0.5 * (v_in_up + v_in_down), input_field, ionic, hartree,
            0.5 * (vxc_up + vxc_down), xc_field, -1.3, 0.7, 0.05,
            alpha_z_energy=0.1, band_energy_weight=0.5,
        )
        for name in ("eigenvalue", "hartree", "integral_vxc_rho", "electron_ion", "electronic", "total"):
            self.assertAlmostEqual(getattr(general, name), getattr(collinear, name), places=10, msg=name)


def _au_cluster(direction, *, soc=False, nc=True, polarization=0.3, convergence=1e-4, xc="ca"):
    return SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0], direction)],
        pseudopotentials={
            "Au": SpeciesPotential(
                _AU_POTRE, 0, spin_orbit=soc, initial_spin_polarization=polarization
            )
        },
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=20,
            convergence_criterion=convergence,
            noncollinear=nc,
            spin_polarized=not nc,
            self_consistent_spin_orbit=soc,
            xc_functional=xc,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _cos(vector, direction):
    direction = np.asarray(direction, dtype=float)
    return float(np.dot(vector, direction) / np.linalg.norm(vector) / np.linalg.norm(direction))


class NoncollinearClusterSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = run_scf_spin_polarized(
            prepare_single_point(_au_cluster(None, nc=False))
        )

    def test_z_aligned_reproduces_the_collinear_driver(self) -> None:
        result = run_self_consistent_noncollinear(
            prepare_single_point(_au_cluster([0, 0, 1]))
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energies.total, self.reference.energies.total, delta=1e-4)
        self.assertAlmostEqual(result.magnetic_moment, self.reference.magnetic_moment, delta=1e-2)

    def test_energy_is_invariant_to_the_moment_direction_without_soc(self) -> None:
        """No spin-orbit coupling: a global spin rotation cannot change the
        energy, and the converged moment must follow the seed direction."""
        for direction in ([1, 0, 0], [1, 2, 2], [0, -1, 0]):
            with self.subTest(direction=direction):
                result = run_self_consistent_noncollinear(
                    prepare_single_point(_au_cluster(direction))
                )
                self.assertTrue(result.converged)
                self.assertAlmostEqual(
                    result.energies.total, self.reference.energies.total, delta=1e-4
                )
                self.assertAlmostEqual(result.magnetic_moment, 1.0, delta=1e-2)
                self.assertGreater(_cos(result.magnetic_moment_vector, direction), 0.999)

    def test_with_soc_a_spherical_atom_is_still_rotation_invariant(self) -> None:
        energies = []
        for direction in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
            system = prepare_single_point(_au_cluster(direction, soc=True))
            projectors = tuple(
                build_spin_orbit_projectors(
                    system.grid,
                    system.atoms,
                    system.pseudopotentials,
                    system.input.pseudopotentials,
                )
            )
            result = run_self_consistent_noncollinear(system, projectors)
            self.assertTrue(result.converged)
            self.assertGreater(result.magnetic_moment, 0.9)
            energies.append(result.energies.total)
        # A cubic grid breaks continuous rotation symmetry slightly.
        self.assertLess(max(energies) - min(energies), 5e-3)

    def test_requires_the_noncollinear_setting(self) -> None:
        system = prepare_single_point(_au_cluster(None, nc=False))
        with self.assertRaises(ValueError):
            run_self_consistent_noncollinear(system)


class NoncollinearTwoAtomTests(unittest.TestCase):
    def test_perpendicular_moments_are_preserved(self) -> None:
        problem = SinglePointInput(
            atoms=[
                Atom("Au", [-4.0, 0.0, 0.0], [1, 0, 0]),
                Atom("Au", [4.0, 0.0, 0.0], [0, 1, 0]),
            ],
            pseudopotentials={
                "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.4)
            },
            grid=GridSettings(spacing=0.7, radius=9.0, expansion_order=6),
            scf=SCFSettings(
                max_iterations=100,
                number_of_states=30,
                convergence_criterion=1e-3,
                noncollinear=True,
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        result = run_self_consistent_noncollinear(system)
        self.assertTrue(result.converged)
        m = result.magnetization
        dv = system.grid.volume_element
        left = system.grid.coordinates[:, 0] < 0
        left_moment = dv * m[:, left].sum(axis=1)
        right_moment = dv * m[:, ~left].sum(axis=1)
        self.assertGreater(_cos(left_moment, [1, 0, 0]), 0.99)
        self.assertGreater(_cos(right_moment, [0, 1, 0]), 0.99)
        self.assertAlmostEqual(float(np.linalg.norm(left_moment)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.linalg.norm(right_moment)), 1.0, delta=0.05)
        # not collinear: the two atomic moments are orthogonal
        self.assertLess(abs(float(np.dot(left_moment, right_moment))), 0.05)


def _au_periodic(direction, *, convergence=1e-4):
    box = 8.0
    return SinglePointInput(
        atoms=[Atom("Au", [box / 2] * 3, direction)],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=box * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=20,
            convergence_criterion=convergence,
            noncollinear=True,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class NoncollinearPeriodicTests(unittest.TestCase):
    def test_rotation_invariance_with_kpoints(self) -> None:
        energies = []
        for direction in ([0, 0, 1], [1, 1, 0]):
            system = prepare_periodic_single_point(_au_periodic(direction))
            k_points, weights = monkhorst_pack_grid(system.input.periodic_cell, (2, 1, 1))
            result = run_self_consistent_noncollinear_kpoints(system, k_points, weights)
            self.assertTrue(result.converged)
            # Without SOC the global moment direction is a zero-energy soft
            # mode the SCF residual does not pin, so only require that it
            # stays near the seed; energy and |m| are the sharp checks.
            self.assertGreater(_cos(result.magnetic_moment_vector, direction), 0.99)
            self.assertAlmostEqual(result.magnetic_moment, 1.0, delta=0.05)
            energies.append(result.energies.total)
        self.assertAlmostEqual(energies[0], energies[1], delta=2e-4)

    def test_z_aligned_matches_the_collinear_kpoint_driver_without_soc(self) -> None:
        """With no spin-orbit coupling a z-aligned non-collinear run is exactly
        the collinear problem.  (With SOC they legitimately differ: the spinors
        then carry local transverse magnetization that the collinear driver's
        z-projected spin densities cannot see, so the non-collinear XC acts on
        a larger local |m| -- measured as -1.5 mRy and |m_z| 0.895 vs 0.953
        for Au at Gamma, independent of SCF tolerance.)"""
        species = {
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)
        }
        base = _au_periodic([0, 0, 1])

        def build(**scf_flags):
            return SinglePointInput(
                atoms=base.atoms,
                pseudopotentials=species,
                grid=base.grid,
                periodic_cell=base.periodic_cell,
                # 1000 K for both runs, not the default 80 K: at 80 K this Gamma-point cell needs
                # 36 to 78 of the 80 steps depending on the route (2026-11-05, collinear / general:
                # A100 CPU 58 / 56, CuPy 55-56 / 36, the device solver on the host 57 / 55; RTX 5070
                # CPU 68 / 56, CuPy 74-78 (not repeatable) / 69, and once more than 80), so the pass
                # was chance.  At 1000 K every route needs 9 steps and the two runs agree to 2e-8 Ry.
                scf=SCFSettings(
                    max_iterations=80,
                    number_of_states=20,
                    convergence_criterion=1e-4,
                    fermi_temperature_kelvin=1000.0,
                    **scf_flags,
                ),
                hartree=base.hartree,
                eigensolver=base.eigensolver,
            )

        gamma = (np.zeros((1, 3)), np.ones(1))
        collinear = run_self_consistent_soc_kpoints_spin_polarized(
            prepare_periodic_single_point(
                build(spin_polarized=True, self_consistent_spin_orbit=True)
            ),
            *gamma,
        )
        general = run_self_consistent_noncollinear_kpoints(
            prepare_periodic_single_point(build(noncollinear=True)), *gamma
        )
        self.assertTrue(collinear.converged and general.converged)
        self.assertAlmostEqual(general.energies.total, collinear.energies.total, delta=2e-4)
        self.assertAlmostEqual(general.magnetic_moment, abs(collinear.magnetic_moment), delta=1e-2)


# O2 in an 8 bohr cell at Gamma with PBE (O FHI), Spin_Polarization or Non_Collinear_magnetism in {extra}
_O2_PBE_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  8.0 0.0 0.0
  0.0 8.0 0.0
  0.0 0.0 8.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Grid_Spacing: 0.4999999995
Initial_Density: sad
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 8
Correlation_Type: pbe
States_Num: 16
Fermi_Temp: 500 K
Max_Iter: 100
Convergence_Criterion: 1e-09 Ry
Eigensolver: chebff
Diag_Tolerance: 1e-10
Mixing_Method: Anderson
Mixing_Param: 0.2
Memory_Param: 4
{extra}Atom_Types_Num: 1
Atom_Type: O
Pseudopotential_Format: fhipp
Potential_Num: 3
Local_Component: d
Core_Cutoff_Radius: 1.6
begin Electron_Per_Orbital
2 4 0
end Electron_Per_Orbital
Cubic_Spline: .true.
Initial_Spin_Polarization: 0.3
begin Atom_Coord
  4.0 4.0 2.83
  4.0 4.0 5.14
end Atom_Coord
"""

_CLUSTER_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 1.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Non_Collinear_magnetism: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .false.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   0.0 0.0 0.0
End atom_coord
{extra}
Correlation_Type: CA
"""


class NoncollinearInputAndCliTests(unittest.TestCase):
    def _directory(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
        (directory / "Au_POTRE.DAT").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def _write(self, directory, extra=""):
        path = directory / "parsec.in"
        path.write_text(_CLUSTER_INPUT.format(extra=extra), encoding="utf-8")
        return path

    def test_cluster_run_reports_the_moment_vector(self) -> None:
        directory = self._directory()
        path = self._write(
            directory, "begin Initial_NCL_Moment\n0 3 4\nend Initial_NCL_Moment\n"
        )
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Converged magnetic moment vector", log)
        archive = read_archive(directory / "parsec_python_results.npz")
        vector = archive["magnetic_moment_vector"]
        self.assertGreater(_cos(vector, [0, 3, 4]), 0.999)
        self.assertEqual(archive["magnetization_mu_b_per_bohr3"].shape[0], 3)

    def test_periodic_run_with_kpoints(self) -> None:
        directory = self._directory()
        text = _CLUSTER_INPUT.format(
            extra=(
                "begin Initial_NCL_Moment\n1 0 0\nend Initial_NCL_Moment\n"
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\n"
                "end Monkhorst_Pack_Grid\n"
            )
        )
        text = text.replace("Boundary_Sphere_Radius: 6.0\n", "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n")
        text = text.replace("0.0 0.0 0.0\nEnd atom_coord", "4.0 4.0 4.0\nEnd atom_coord")
        path = directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        archive = read_archive(directory / "parsec_python_results.npz")
        self.assertGreater(_cos(archive["magnetic_moment_vector"], [1, 0, 0]), 0.999)

    def test_gradient_functional_reaches_the_collinear_driver(self) -> None:
        # O2 with PBE from parallel moments (the default collinear frame): converged, the energy of the collinear spin
        # driver; along z and along x alike (no spin-orbit coupling)
        o_fhi = Path(__file__).resolve().parents[4] / "examples" / "3d_hybrid_functionals" / "O_FHIPP.DAT"
        energies = {}
        for label, extra in (
            ("spin", "Spin_Polarization: .true.\n"),
            ("z", "Non_Collinear_magnetism: .true.\n"),
            ("x", "Non_Collinear_magnetism: .true.\n"),
        ):
            directory = self._directory()
            (directory / "O_FHIPP.DAT").write_bytes(o_fhi.read_bytes())
            moments = {"z": "0 0 1", "x": "1 0 0"}.get(label)
            text = _O2_PBE_INPUT.format(extra=extra) + (
                f"begin Initial_NCL_Moment\n{moments}\n{moments}\nend Initial_NCL_Moment\n" if moments else ""
            )
            path = directory / "parsec.in"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(cli_main([str(path), "--quiet"]), 0, label)
            archive = read_archive(directory / "parsec_python_results.npz")
            energies[label] = float(archive["energy_total_ry"])
            if moments:
                self.assertGreater(_cos(archive["magnetic_moment_vector"], [float(v) for v in moments.split()]), 0.999)
        self.assertAlmostEqual(energies["z"], energies["spin"], delta=1.0e-8)
        self.assertAlmostEqual(energies["x"], energies["spin"], delta=1.0e-8)

    def test_moment_block_without_the_flag_is_rejected(self) -> None:
        directory = self._directory()
        text = _CLUSTER_INPUT.format(
            extra="begin Initial_NCL_Moment\n0 0 1\nend Initial_NCL_Moment\n"
        ).replace("Non_Collinear_magnetism: .true.\n", "")
        path = directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--dry-run"]), 2)

    def test_wrong_number_of_moment_rows_is_rejected(self) -> None:
        directory = self._directory()
        path = self._write(
            directory, "begin Initial_NCL_Moment\n0 0 1\n1 0 0\nend Initial_NCL_Moment\n"
        )
        self.assertEqual(cli_main([str(path), "--dry-run"]), 2)

    def test_atom_normalizes_its_moment_direction(self) -> None:
        atom = Atom("Au", [0, 0, 0], [0, 0, 5])
        np.testing.assert_allclose(atom.initial_moment, [0, 0, 1])
        self.assertIsNone(Atom("Au", [0, 0, 0], [0, 0, 0]).initial_moment)
        with self.assertRaises(ValueError):
            Atom("Au", [0, 0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
