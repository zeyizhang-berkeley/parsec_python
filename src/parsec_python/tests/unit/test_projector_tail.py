"""The projectors of a table beyond its channel cutoffs (``PARSEC_PROJECTOR_TAIL``).

``nonloc.F90`` keeps the grid points up to ``rcore``, the first radial sample beyond the largest channel cutoff ``rc``
of a ``MARTINS_NEW`` table.  The tables converted from UPF files hold a small ``V_l - V_local`` beyond ``rc`` (the
carbon table of the diamond examples: its projectors are 1.6e-4 and -1.3e-4 at ``rcore``), so the nonlocal energy
jumps whenever a grid point crosses that sphere.  ``auto`` (the default) sets the difference to zero beyond ``rc``;
``off`` keeps the table (PARSEC 1.5).  Checked here (CPU):

* the values of the switch, refused before any projector is built;
* ``off`` is the formula of ``pseudo.f90`` bit for bit, and a table whose difference is zero beyond ``rc`` is not
  changed by ``auto`` (projectors and their derivative tables, bit for bit);
* under ``auto`` the carbon projectors vanish beyond ``rc`` and keep their shape inside (one normalization factor);
* the nonlocal energy at fixed orbitals (``Cubic_Spline``) jumps by 3.5e-6 Ry when a grid point crosses ``rcore``
  under ``off`` and is continuous under ``auto``; away from the crossing the force is its derivative under both.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Forces.nonlocal_force import build_nonlocal_projector_gradients, nonlocal_forces
from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings, SpeciesPotential
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential.potre import PROJECTOR_TAIL_ENV, projector_tail_setting
from parsec_python.Pseudopotential.radial_derivatives import radial_projector_with_derivative
from parsec_python.Pseudopotential.radial_quadrature import parsec_radial_integral
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors

_ROOT = Path(__file__).resolve().parents[4]
_C_CONVERTED = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_AG = _ROOT / "src" / "parsec_python" / "tests" / "data" / "Ag_POTRE.DAT"
_GE = _ROOT / "src" / "parsec_python" / "tests" / "data" / "Ge_POTRE.DAT"


def _setting(value):
    return mock.patch.dict(os.environ, {PROJECTOR_TAIL_ENV: value})


class ProjectorTailSwitchTests(unittest.TestCase):
    def test_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PROJECTOR_TAIL_ENV, None)
            self.assertEqual(projector_tail_setting(), "auto")
        for value in ("auto", "off", " AUTO "):
            with _setting(value):
                self.assertEqual(projector_tail_setting(), value.strip().lower())
        with _setting("cutoff"):
            with self.assertRaisesRegex(ValueError, PROJECTOR_TAIL_ENV):
                projector_tail_setting()
            # refused where every route reads its tables, before a projector is built
            with self.assertRaisesRegex(ValueError, PROJECTOR_TAIL_ENV):
                load_pseudopotentials({"C": SpeciesPotential(_C_CONVERTED, 2)}, xc_functional="pbe")


class ProjectorTailTableTests(unittest.TestCase):
    def test_off_is_the_formula_of_pseudo_f90(self):
        potential = read_parsec_pseudopotential(_C_CONVERTED)
        # with the denominators of pseudo.f90 (PARSEC_KB_QUADRATURE=trapezoid)
        with _setting("off"), mock.patch.dict(os.environ, {"PARSEC_KB_QUADRATURE": "trapezoid"}):
            for channel in (0, 1):
                projector, sign = potential.radial_projector(channel, 2)
                delta = potential.channel_potentials[channel] - potential.channel_potentials[2]
                wave = potential.radial_wavefunctions[channel]
                denominator = parsec_radial_integral(potential.radii, wave * wave * delta)
                expected = delta * wave / potential.radii / np.sqrt(abs(denominator))
                expected[:8] = expected[16]
                np.testing.assert_array_equal(projector, expected)
                self.assertEqual(sign, float(np.sign(denominator)))

    def test_tables_without_a_tail_keep_their_bits(self):
        for path, local in ((_AG, 0), (_GE, 2)):
            potential = read_parsec_pseudopotential(path)
            self.assertIsNotNone(potential.projector_tail_radius)
            for channel in potential.channel_potentials:
                if channel == local:
                    continue
                with _setting("off"):
                    before = radial_projector_with_derivative(potential, channel, local)
                with _setting("auto"):
                    after = radial_projector_with_derivative(potential, channel, local)
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                self.assertEqual(before[2], after[2])

    def test_auto_zeroes_the_converted_tail(self):
        potential = read_parsec_pseudopotential(_C_CONVERTED)
        radius = potential.projector_tail_radius
        self.assertAlmostEqual(radius, 1.498152959386, places=10)
        support = potential.projector_support_radius
        beyond = potential.radii > radius * (1.0 + 1.0e-9)
        inside = ~beyond
        for channel in (0, 1):
            with _setting("off"):
                old, _ = potential.radial_projector(channel, 2)
            with _setting("auto"):
                new, _ = potential.radial_projector(channel, 2)
            at_support = int(np.argmin(np.abs(potential.radii - support)))
            self.assertGreater(abs(old[at_support]), 1.0e-4)
            self.assertEqual(new[at_support], 0.0)
            self.assertFalse(np.any(new[beyond]))
            # one normalization factor inside: the denominator loses the tail's share
            mask = inside & (np.abs(old) > 1.0e-8)
            ratio = new[mask] / old[mask]
            self.assertLess(float(np.ptp(ratio)), 1.0e-12)
            self.assertLess(abs(float(ratio[0]) - 1.0), 1.0e-4)


class ProjectorTailEnergyTests(unittest.TestCase):
    """A carbon atom with a grid point at ``rcore`` +- 1e-7 bohr: the nonlocal energy at fixed orbitals."""

    @classmethod
    def setUpClass(cls):
        cls.grid = build_cluster_grid(GridSettings(spacing=0.35, radius=4.0))
        cls.potentials = {"C": read_parsec_pseudopotential(_C_CONVERTED)}
        cls.specifications = {"C": SpeciesPotential(_C_CONVERTED, 2, use_spline=True)}
        coordinates = np.asarray(cls.grid.coordinates)
        point = coordinates[int(np.argmin(np.linalg.norm(coordinates - np.array([1.0, 0.2, 0.1]), axis=1)))]
        cls.direction = np.array([-1.0, 0.0, 0.0])
        cls.support = cls.potentials["C"].projector_support_radius
        # the atom on the line through the point; at the reference position the point lies exactly at rcore
        cls.reference = point - cls.support * cls.direction
        distance = np.linalg.norm(coordinates - point - 0.37 * np.array([1.0, 0.6, -0.4]), axis=1)
        orbitals = np.stack([np.exp(-0.5 * distance ** 2), coordinates[:, 0] * np.exp(-0.5 * distance ** 2)], axis=1)
        cls.orbitals = orbitals / np.linalg.norm(orbitals, axis=0)
        cls.occupations = np.ones(2)

    def energy(self, shift):
        atoms = [Atom("C", (self.reference + shift * self.direction).tolist())]
        operator = build_nonlocal_projectors(self.grid, atoms, self.potentials, self.specifications)
        overlaps = operator.projectors.T @ self.orbitals
        return 2.0 * float(np.sum(self.occupations * np.sum(np.asarray(operator.signs)[:, None] * overlaps ** 2, axis=0)))

    def force(self, shift):
        atoms = [Atom("C", (self.reference + shift * self.direction).tolist())]
        operator = build_nonlocal_projectors(self.grid, atoms, self.potentials, self.specifications)
        gradients = build_nonlocal_projector_gradients(self.grid, atoms, self.potentials, self.specifications)
        return float(nonlocal_forces(operator, gradients, self.orbitals, self.occupations, 1)[0] @ self.direction)

    def test_jump_under_off_and_continuity_under_auto(self):
        epsilon = 1.0e-9
        step = 1.0e-4
        jumps = {}
        for value in ("off", "auto"):
            with _setting(value):
                jumps[value] = self.energy(-epsilon) - self.energy(epsilon)
                # away from the crossing the force is the derivative of the energy (Cubic_Spline) under both values
                for side in (1.0, -1.0):
                    slope = -(self.energy(side * 3.0 * step) - self.energy(side * step)) / (2.0 * side * step)
                    self.assertLess(abs(self.force(side * 2.0 * step) - slope), 1.0e-6)
        self.assertGreater(abs(jumps["off"]), 1.0e-6)
        self.assertLess(abs(jumps["auto"]), 1.0e-7)


if __name__ == "__main__":
    unittest.main()
