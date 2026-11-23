"""The quadrature of the Kleinman-Bylander denominators (``PARSEC_KB_QUADRATURE``).

``pseudo.f90`` integrates ``u_l^2 (V_l - V_local)`` with the weights ``0.5*(r(i+1) - r(i-1))``.  On the logarithmic
mesh ``r(i) = a*exp(b*i) - c`` of a table they are ``sinh(b)/b`` times the exact Jacobian ``b*(r(i) + c)``, so every
denominator is too large by ``b^2/6``.  ``logmesh`` (opt-in) takes the trapezoid rule in the mesh index with the exact
Jacobian.  Checked here (CPU):

* the values of the switch (an unknown value is refused);
* ``log_mesh_integral`` integrates ``r^2 exp(-r)`` to the last digits on a Martins mesh and on a pure
  exponential (FHI) mesh, where ``parsec_radial_integral`` is ``b^2/6`` too large;
* ``trapezoid`` (default) gives the projectors of ``pseudo.f90`` bit for bit; ``logmesh`` scales every projector by
  ``sqrt(D_trapezoid/D_logmesh) = 1 + b^2/12`` (the denominators of the tests' Ag table move by ``b^2/6``), the
  spin-orbit pair and the derivative table with them;
* a table whose radii are not the mesh of its parameters is refused under ``logmesh``, radius by radius: the
  ``MARTINS_WANG`` table of the tests, whose eight printed digits follow its mesh within 3.1e-7, is accepted;
* the resident set-up key changes only under ``logmesh``.
"""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Pseudopotential import read_martins_wang_pseudopotential, read_parsec_pseudopotential
from parsec_python.Pseudopotential.radial_derivatives import radial_projector_with_derivative
from parsec_python.Pseudopotential.radial_quadrature import (
    KB_QUADRATURE_ENV,
    kb_quadrature_setting,
    log_mesh_integral,
    parsec_radial_integral,
)

_DATA = Path(__file__).resolve().parents[1] / "data"
_AG = _DATA / "Ag_POTRE.DAT"
_FE = _DATA / "spin_pbe_Fe_POTRE.DAT"
_H_SYNTHETIC = _DATA / "H_POTRE.DAT"
_NA_WANG = _DATA / "Na_POTRW.DAT"


def _env(value):
    return mock.patch.dict(os.environ, {KB_QUADRATURE_ENV: value})


class KbQuadratureSettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop(KB_QUADRATURE_ENV, None)
            self.assertEqual(kb_quadrature_setting(), "trapezoid")
        for value in ("trapezoid", "logmesh", " LogMesh "):
            with _env(value):
                self.assertEqual(kb_quadrature_setting(), value.strip().lower())
        with _env("exact"):
            with self.assertRaises(ValueError):
                kb_quadrature_setting()


class LogMeshIntegralTests(unittest.TestCase):
    def test_martins_mesh(self) -> None:
        a, b = 1.0e-4, 0.0125
        radii = a * np.expm1(b * np.arange(1, 1400))
        integrand = radii**2 * np.exp(-radii)
        self.assertAlmostEqual(log_mesh_integral(radii, integrand, b, a), 2.0, delta=2.0e-11)
        trapezoid = parsec_radial_integral(radii, integrand)
        self.assertAlmostEqual(trapezoid / 2.0 - 1.0, np.sinh(b) / b - 1.0, delta=1.0e-9)

    def test_exponential_mesh(self) -> None:
        a, b = 1.0e-5, 0.0247
        radii = a * np.exp(b * np.arange(0, 720))
        integrand = radii**2 * np.exp(-radii)
        self.assertAlmostEqual(log_mesh_integral(radii, integrand, b, 0.0), 2.0, delta=1.0e-11)

    def test_mesh_must_follow_its_parameters(self) -> None:
        a, b = 1.0e-4, 0.0125
        radii = a * np.expm1(b * np.arange(1, 1400))
        with self.assertRaises(ValueError):
            log_mesh_integral(radii, radii**2 * np.exp(-radii), 1.1 * b, a)


class KbQuadratureProjectorTests(unittest.TestCase):
    def test_default_is_pseudo_f90(self) -> None:
        potential = read_parsec_pseudopotential(_AG)
        with mock.patch.dict(os.environ):
            os.environ.pop(KB_QUADRATURE_ENV, None)
            projector, sign = potential.radial_projector(2, 0)
        delta_v = potential.channel_potentials[2] - potential.channel_potentials[0]
        wave = potential.radial_wavefunctions[2]
        denominator = parsec_radial_integral(potential.radii, wave * wave * delta_v)
        expected = delta_v * wave / potential.radii / np.sqrt(abs(denominator))
        expected[:8] = expected[16]
        np.testing.assert_array_equal(projector, expected)
        self.assertEqual(sign, float(np.sign(denominator)))
        with _env("trapezoid"):
            np.testing.assert_array_equal(potential.radial_projector(2, 0)[0], projector)

    def test_logmesh_scales_projectors(self) -> None:
        potential = read_parsec_pseudopotential(_AG)
        b = potential.radial_parameter_b
        with _env("trapezoid"):
            base = {l: potential.radial_projector(l, 0) for l in (1, 2)}
            base_so = potential.spin_orbit_projectors(2, 0)
            base_slope = radial_projector_with_derivative(potential, 2, 0)
        with _env("logmesh"):
            for l in (1, 2):
                projector, sign = potential.radial_projector(l, 0)
                self.assertEqual(sign, base[l][1])
                ratio = projector[200:600] / base[l][0][200:600]
                np.testing.assert_allclose(ratio, ratio[0], rtol=1.0e-13)
                # denominators b^2/6 smaller: projectors 1 + b^2/12 larger (to the quadrature of the table)
                self.assertAlmostEqual(ratio[0] - 1.0, b * b / 12.0, delta=0.05 * b * b / 12.0)
            so = potential.spin_orbit_projectors(2, 0)
            self.assertAlmostEqual(float(so[0][300] / base_so[0][300]) - 1.0, b * b / 12.0, delta=0.05 * b * b / 12.0)
            slope = radial_projector_with_derivative(potential, 2, 0)
            np.testing.assert_allclose(slope[1][200:600] / base_slope[1][200:600], ratio[0], rtol=1.0e-12)

    def test_fe_table(self) -> None:
        potential = read_parsec_pseudopotential(_FE)
        b = potential.radial_parameter_b
        with _env("trapezoid"):
            trapezoid = potential.radial_projector(2, 0)[0]
        with _env("logmesh"):
            logmesh = potential.radial_projector(2, 0)[0]
        self.assertAlmostEqual(float(logmesh[300] / trapezoid[300]) - 1.0, b * b / 12.0, delta=0.05 * b * b / 12.0)

    def test_martins_wang_table_accepted(self) -> None:
        potential = read_martins_wang_pseudopotential(_NA_WANG, symbol="Na", core_cutoff_radius=2.5)
        radii, b, c = potential.radii, potential.radial_parameter_b, potential.radial_offset
        predicted = (radii[0] + c) * np.exp(b * np.arange(radii.size)) - c
        self.assertLess(float(np.max(np.abs(radii / predicted - 1.0))), 1.0e-6)
        local_l = potential.local_channel
        channels = [l for l in sorted(potential.channel_potentials) if l != local_l]
        self.assertTrue(channels)
        with _env("trapezoid"):
            trapezoid = {l: potential.radial_projector(l, local_l)[0] for l in channels}
        with _env("logmesh"):
            for l in channels:
                projector = potential.radial_projector(l, local_l)[0]
                ratio = float(projector[300] / trapezoid[l][300])
                self.assertAlmostEqual(ratio - 1.0, b * b / 12.0, delta=0.05 * b * b / 12.0)

    def test_synthetic_mesh_refused(self) -> None:
        potential = read_parsec_pseudopotential(_H_SYNTHETIC)
        with _env("logmesh"):
            with self.assertRaises(ValueError):
                potential.kb_denominator(np.ones_like(potential.radii))
        with _env("trapezoid"):
            potential.kb_denominator(np.ones_like(potential.radii))


class KbQuadratureSetupKeyTests(unittest.TestCase):
    def test_key_changes_only_under_logmesh(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.V_ion.setup_options import hash_pseudopotential_setup

        specification = SimpleNamespace(old_interpolation=False, has_fourier_filter=False)

        def key(value):
            digest = hashlib.sha256()
            with _env(value):
                hash_pseudopotential_setup(digest, specification)
            return digest.hexdigest()

        self.assertEqual(key("trapezoid"), hashlib.sha256().hexdigest())
        self.assertNotEqual(key("logmesh"), hashlib.sha256().hexdigest())


if __name__ == "__main__":
    unittest.main()
