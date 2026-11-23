"""``PARSEC_CA_CONTINUITY``: the high-density coefficients of the Perdew-Zunger correlation of CA-LDA.

PARSEC 1.5 (``exc.f90``, ``corca_nspn`` and ``corlsd_ca``) takes the four digits of Perdew and Zunger's paper, with
which the two forms of the correlation energy per electron differ by 6.41e-5 Ry at ``r_s = 1``: the total energy steps
by ``(3/(4 pi)) 6.41e-5 h^3`` Ry whenever the density of a grid point crosses ``3/(4 pi)`` e/bohr^3.  ``on`` joins the
forms with continuous values and slopes (libxc's ``LDA_C_PZ_MOD``; the default since 2026-11-23); ``off`` keeps
PARSEC's numbers bit for bit.  The NumPy functionals, the C++ evaluator of the clusters and the CUDA kernels follow the
switch alike.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.V_xc.ca_lda import (
    CA_CONTINUITY_ENV,
    ca_continuity_setting,
    ca_high_density_coefficients,
    ca_lda,
    ca_lda_spin_polarized,
)
from parsec_python.V_xc.ca_lda_spin import ca_lda_spin

CRITICAL = 3.0 / (4.0 * np.pi)  # the density of r_s = 1
SPLIT = 1.0e-12


def _parsec_ca(rho):
    """The CA-LDA of corca_nspn with PARSEC's literal coefficients (the code before the switch)."""

    rs = (0.75 / (np.pi * rho)) ** (1.0 / 3.0)
    a0 = (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)
    vx = -2.0 / (np.pi * a0 * rs)
    ex = 0.75 * vx
    ec = np.empty_like(rs)
    vc = np.empty_like(rs)
    low = rs >= 1.0
    r = rs[low]
    s = np.sqrt(r)
    g, b1, b2 = -0.2846, 1.0529, 0.3334
    ec[low] = g / (1.0 + b1 * s + b2 * r)
    vc[low] = (ec[low] * ec[low] / g) * (1.0 + (7.0 / 6.0) * b1 * s + (4.0 / 3.0) * b2 * r)
    r = rs[~low]
    lr = np.log(r)
    c1, c2, c3, c4, c5 = 0.0622, 0.096, 0.004, 0.0232, 0.0192
    ec[~low] = c1 * lr - c2 + (c3 * lr - c4) * r
    vc[~low] = ec[~low] - (c1 + (c3 * lr - c5) * r) / 3.0
    return vx + vc, ex + ec


def _densities(count=400, seed=5):
    rng = np.random.default_rng(seed)
    values = CRITICAL * np.exp(rng.uniform(-6.0, 2.5, count))
    values[:4] = (CRITICAL * (1.0 - SPLIT), CRITICAL * (1.0 + SPLIT), 1.0e-9, 4.0)
    return values


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: ""}):
            self.assertEqual(ca_continuity_setting(), "on")
            self.assertEqual(ca_high_density_coefficients(), ca_high_density_coefficients("on"))
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "off"}):
            self.assertEqual(ca_high_density_coefficients(), (0.004, 0.0232, 0.0192, 0.0014, -0.0096))
        for value in ("on", " ON "):
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: value}):
                self.assertEqual(ca_continuity_setting(), "on")
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "auto"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_CA_CONTINUITY must be off or on"):
                ca_continuity_setting()
            with self.assertRaises(ValueError):
                ca_lda(np.array([0.3]), 1.0)

    def test_continuous_coefficients(self) -> None:
        c3, c4, c5, cs, ds = ca_high_density_coefficients("on")
        # libxc's LDA_C_PZ_MOD (Hartree): c = 0.0020191519406228, d = -0.0116320663789130 of the unpolarized gas,
        # d = -0.00480126353790614 of the fully polarized one
        self.assertAlmostEqual(c3 / 2.0, 0.0020191519406228, delta=1.0e-16)
        self.assertAlmostEqual(-c4 / 2.0, -0.0116320663789130, delta=1.0e-16)
        self.assertAlmostEqual(ds / 2.0, -0.00480126353790614, delta=1.0e-16)
        self.assertAlmostEqual(c5, c4 - c3, delta=1.0e-18)
        # the joined forms: equal values and slopes at r_s = 1
        for a, b, c, d, g, b1, b2 in ((0.0622, -0.096, c3, -c4, -0.2846, 1.0529, 0.3334),
                                      (0.0311, -0.0538, cs, ds, -0.1686, 1.3981, 0.2611)):
            self.assertAlmostEqual(b + d, g / (1.0 + b1 + b2), delta=1.0e-16)
            self.assertAlmostEqual(a + c + d, -g * (0.5 * b1 + b2) / (1.0 + b1 + b2) ** 2, delta=1.0e-16)

    def test_the_input_refuses_an_unknown_value(self) -> None:
        from parsec_python.Input import parse_parsec_input

        import shutil

        data = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(data / "water_H_POTRE.DAT", Path(folder) / "H_POTRE.DAT")
            path = Path(folder) / "parsec.in"
            path.write_text(
                "Boundary_Sphere_Radius: 6.0\nGrid_Spacing: 0.5\nStates_Num: 4\nAtom_Types_Num: 1\nAtom_Type: H\n"
                "Local_Component: s\nbegin Atom_Coord\n 0.0 0.0 0.0\n 1.4 0.0 0.0\nend Atom_Coord\n"
            )
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "smooth"}):
                with self.assertRaisesRegex(ValueError, "PARSEC_CA_CONTINUITY"):
                    parse_parsec_input(path)


class FunctionalTests(unittest.TestCase):
    def test_off_keeps_the_bits_of_parsec(self) -> None:
        rho = _densities()
        core = 0.02 * _densities(seed=9)
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "off"}):
            result = ca_lda(rho, 0.3)
            potential, epsilon = _parsec_ca(rho)
            self.assertTrue(np.array_equal(result.potential, potential))
            self.assertTrue(np.array_equal(result.energy_per_electron, epsilon))
            with_core = ca_lda(rho, 0.3, core)
            self.assertTrue(np.array_equal(with_core.potential, _parsec_ca(rho + core)[0]))

    def test_the_parsec_steps_at_rs_1(self) -> None:
        rho = np.array([CRITICAL * (1.0 - SPLIT), CRITICAL * (1.0 + SPLIT)])
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "off"}):
            result = ca_lda(rho, 1.0)
        self.assertAlmostEqual(np.diff(result.energy_per_electron)[0], 6.4133e-5, delta=1.0e-8)
        self.assertAlmostEqual(np.diff(result.potential)[0], 5.5523e-5, delta=1.0e-8)
        self.assertAlmostEqual(np.diff(result.energy_density)[0], CRITICAL * 6.41328e-5, delta=1.0e-9)

    def test_on_is_continuous_at_rs_1(self) -> None:
        rho = np.array([CRITICAL * (1.0 - SPLIT), CRITICAL * (1.0 + SPLIT)])
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
            unpolarized = ca_lda(rho, 1.0)
            for name in ("potential", "energy_per_electron", "energy_density"):
                self.assertLess(abs(np.diff(getattr(unpolarized, name))[0]), 1.0e-11, name)
            for fraction in (0.5, 0.8, 1.0):
                up, down = fraction * rho, (1.0 - fraction) * rho
                for result in (ca_lda_spin(up, down, 1.0), ca_lda_spin_polarized(up, down, 1.0)):
                    for name in ("potential_up", "potential_down", "energy_density"):
                        self.assertLess(abs(np.diff(getattr(result, name))[0]), 1.0e-11, (fraction, name))

    def test_on_changes_only_the_high_density_points(self) -> None:
        rho = _densities()
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "off"}):
            off = ca_lda(rho, 1.0)
            off_spin = ca_lda_spin(0.7 * rho, 0.3 * rho, 1.0)
            off_polarized = ca_lda_spin_polarized(0.7 * rho, 0.3 * rho, 1.0)
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
            on = ca_lda(rho, 1.0)
            on_spin = ca_lda_spin(0.7 * rho, 0.3 * rho, 1.0)
            on_polarized = ca_lda_spin_polarized(0.7 * rho, 0.3 * rho, 1.0)
        low = (0.75 / (np.pi * rho)) ** (1.0 / 3.0) >= 1.0
        self.assertTrue(np.any(low) and np.any(~low))
        self.assertTrue(np.array_equal(on.potential[low], off.potential[low]))
        self.assertTrue(np.all(on.potential[~low] != off.potential[~low]))
        self.assertTrue(np.array_equal(on_spin.potential_up[low], off_spin.potential_up[low]))
        self.assertTrue(np.array_equal(on_polarized.potential_down[low], off_polarized.potential_down[low]))
        # the two spin forms and the unpolarized one agree under on as under off
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
            np.testing.assert_allclose(ca_lda_spin(rho / 2, rho / 2, 1.0).potential_up, on.potential, rtol=1.0e-12)
            np.testing.assert_allclose(ca_lda_spin_polarized(rho / 2, rho / 2, 1.0).potential_up, on.potential,
                                       rtol=1.0e-12)
            np.testing.assert_allclose(on_spin.potential_up, on_polarized.potential_up, rtol=1.0e-12)

    def test_the_potential_is_the_derivative_of_the_energy_density(self) -> None:
        rho = np.array([0.3, 0.6, 2.0, 7.0])  # r_s < 1
        step = 1.0e-6
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
            center = ca_lda(rho, 1.0)
            slope = (ca_lda(rho * (1 + step), 1.0).energy_density
                     - ca_lda(rho * (1 - step), 1.0).energy_density) / (2.0 * step * rho)
        np.testing.assert_allclose(center.potential, slope, rtol=1.0e-8)


class _Grid:
    def __init__(self):
        from parsec_python.Grid.cluster import build_cluster_grid
        from parsec_python.models import GridSettings

        self.grid = build_cluster_grid(GridSettings(spacing=0.45, radius=5.0, expansion_order=12))
        x = self.grid.coordinates
        self.density = 1.4 * np.exp(-np.sum((x - 0.2) ** 2, axis=1) / 1.3) + 1.0e-6
        self.core = 0.05 * np.exp(-np.sum(x * x, axis=1) / 0.4)


class _ParsecLoops:
    """Stands in for the C++ ``CALDAEvaluator``: PARSEC's coefficients, the energy weighted by multiplicities."""

    def __init__(self, core, volume, weights=None):
        self.core, self.volume, self.weights = core, volume, weights

    def evaluate(self, valence):
        with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "off"}):
            result = ca_lda(valence, self.volume, self.core)
        weights = np.ones_like(valence) if self.weights is None else self.weights
        return {
            "potential": result.potential,
            "energy_per_electron": result.energy_per_electron,
            "energy_density": result.energy_density,
            "total_energy": float(np.dot(weights, result.energy_density)) * self.volume,
        }


class ClusterEvaluatorCorrectionTests(unittest.TestCase):
    """The change that the cluster evaluator adds to its C++ loops under ``on`` (no native extension needed)."""

    def test_the_corrected_loops_are_the_continuous_functional(self) -> None:
        from parsec_python.acceleration.V_xc.native_ca_lda import NativeCALDAEvaluator

        rho = _densities(seed=21)
        rho[4:8] = CRITICAL * np.array([1.0 - 3.0e-11, 1.0 + 3.0e-11, 1.0 + 2.0e-10, 1.0 - 2.0e-10])
        core = 0.01 * _densities(seed=22)
        weights = np.random.default_rng(3).integers(1, 9, rho.size).astype(float)
        volume = 0.21
        for multiplicities in (None, weights):
            evaluator = object.__new__(NativeCALDAEvaluator)
            evaluator._core = core
            evaluator._volume_element = volume
            evaluator._weights = multiplicities
            evaluator._evaluator = _ParsecLoops(core, volume, multiplicities)
            payload = evaluator._continuous(rho)
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
                reference = ca_lda(rho, volume, core)
            for name in ("potential", "energy_per_electron", "energy_density"):
                np.testing.assert_allclose(payload[name], getattr(reference, name), rtol=1.0e-13, atol=1.0e-300)
            expected = float(np.dot(np.ones_like(rho) if multiplicities is None else multiplicities,
                                    reference.energy_density)) * volume
            self.assertAlmostEqual(payload["total_energy"], expected, delta=1.0e-12 * abs(expected))

    def test_the_candidate_points_give_the_bits_of_a_scan_of_every_point(self) -> None:
        """The correction finds its points among those above (1 - 2e-10) 3/(4 pi): the same bits as the tests of every
        point (the evaluator before 2026-11-23)."""

        from parsec_python.acceleration.V_xc.native_ca_lda import NativeCALDAEvaluator

        def every_point(evaluator, valence):
            payload = evaluator._evaluator.evaluate(valence)
            potential, epsilon, energy = (np.array(payload[name]) for name in
                                          ("potential", "energy_per_electron", "energy_density"))
            density = valence + evaluator._core
            ratio = density / CRITICAL
            high, near = ratio > 1.0 + 1.0e-10, np.abs(ratio - 1.0) <= 1.0e-10
            before = energy.copy()
            on, off = ca_high_density_coefficients("on"), ca_high_density_coefficients("off")
            dc3, dc4, dc5 = (on[i] - off[i] for i in range(3))
            rs = np.cbrt(CRITICAL / density[high])
            change = (dc3 * np.log(rs) - dc4) * rs
            epsilon[high] += change
            potential[high] += change - (dc3 * np.log(rs) - dc5) * rs / 3.0
            energy[high] = density[high] * epsilon[high]
            if np.any(near):
                exact = ca_lda(valence[near], evaluator._volume_element, evaluator._core[near])
                potential[near], epsilon[near], energy[near] = (exact.potential, exact.energy_per_electron,
                                                                 exact.energy_density)
            changed = high | near
            weights = 1.0 if evaluator._weights is None else evaluator._weights[changed]
            total = float(payload["total_energy"]) + float(
                np.sum(weights * (energy[changed] - before[changed]))) * evaluator._volume_element
            return potential, epsilon, energy, total

        random = np.random.default_rng(29)
        for trial in range(6):
            rho = CRITICAL * np.exp(random.uniform(-8.0, 2.5, 5000))
            rho[:12] = CRITICAL * (1.0 + random.uniform(-3.0e-10, 3.0e-10, 12))
            core = (trial % 2) * 0.01 * CRITICAL * np.exp(random.uniform(-8.0, 2.0, rho.size))
            multiplicities = random.integers(1, 9, rho.size).astype(float) if trial % 3 else None
            evaluator = object.__new__(NativeCALDAEvaluator)
            evaluator._core, evaluator._volume_element, evaluator._weights = core, 0.17, multiplicities
            evaluator._evaluator = _ParsecLoops(core, 0.17, multiplicities)
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: "on"}):
                payload = evaluator._continuous(rho)
                reference = every_point(evaluator, rho)
            for value, expected in zip(
                (payload["potential"], payload["energy_per_electron"], payload["energy_density"]), reference[:3]
            ):
                self.assertTrue(np.array_equal(value, expected))
            self.assertEqual(payload["total_energy"], reference[3])


def _native():
    try:
        from parsec_python.acceleration.backends.native import native_available
    except ImportError:  # pragma: no cover
        return False
    return native_available()


@unittest.skipUnless(_native(), "the native extension is unavailable")
class NativeEvaluatorTests(unittest.TestCase):
    def test_the_cluster_evaluator_follows_the_switch(self) -> None:
        from parsec_python.acceleration.V_xc import NativeCALDAEvaluator

        case = _Grid()
        volume = case.grid.volume_element
        for setting in ("off", "on"):
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: setting}):
                evaluator = NativeCALDAEvaluator(case.core, volume)
                value = evaluator(case.density)
                reference = ca_lda(case.density, volume, case.core)
            self.assertEqual(evaluator.continuity, setting)
            for name in ("potential", "energy_per_electron", "energy_density"):
                np.testing.assert_allclose(getattr(value, name), getattr(reference, name), rtol=1.0e-13, atol=0.0)
            self.assertAlmostEqual(value.total_energy, reference.total_energy, delta=1.0e-12)


def _cupy():
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available
    except ImportError:  # pragma: no cover
        return False
    return cupy_available()


@unittest.skipUnless(_cupy(), "CUDA runtime is unavailable")
class DeviceKernelTests(unittest.TestCase):
    def test_the_kernels_follow_the_switch(self) -> None:
        # the device evaluates CA-LDA in periodic cells (a cluster has the C++ evaluator): the cell of
        # test_periodic_ca_device with densities on both sides of r_s = 1
        from parsec_python.acceleration.tests.test_periodic_ca_device import _cell_system
        from parsec_python.acceleration.V_xc import device_xc

        case = _Grid()
        _problem, system = _cell_system("ca")
        case.grid = system.grid
        size = int(case.grid.size)
        random = np.random.default_rng(17)
        case.density = 10.0 ** random.uniform(-7.0, 1.5, size=size)
        case.core = 0.01 * 10.0 ** random.uniform(-7.0, 1.5, size=size)
        self.assertTrue(np.any(case.density > CRITICAL) and np.any(case.density < CRITICAL))
        volume = case.grid.volume_element
        up, down = 0.65 * case.density, 0.35 * case.density
        totals = {}
        for setting in ("off", "on", "off"):
            with mock.patch.dict(os.environ, {CA_CONTINUITY_ENV: setting}):
                device = device_xc.DeviceXCEvaluator("ca", case.grid, core_density=case.core)(case.density)
                host = ca_lda(case.density, volume, case.core)
                spin = device_xc.DeviceSpinXCEvaluator("ca", case.grid, core_density=case.core)
                spin_device = spin(up, down)
                spin_host = ca_lda_spin(up, down, volume, case.core)
                spin.form = "spin_polarized"
                polarized_device = spin(up, down)
                polarized_host = ca_lda_spin_polarized(up, down, volume, case.core)
            for name in ("potential", "energy_per_electron", "energy_density"):
                np.testing.assert_allclose(np.asarray(getattr(device, name)), getattr(host, name), rtol=2.0e-14,
                                           atol=1.0e-300)
            for d, h in ((spin_device, spin_host), (polarized_device, polarized_host)):
                for name in ("potential_up", "potential_down", "energy_density"):
                    np.testing.assert_allclose(np.asarray(getattr(d, name)), getattr(h, name), rtol=2.0e-14,
                                               atol=1.0e-300)
            totals.setdefault(setting, []).append(device.total_energy)
        self.assertEqual(totals["off"][0], totals["off"][1])
        self.assertNotEqual(totals["off"][0], totals["on"][0])


if __name__ == "__main__":
    unittest.main()
