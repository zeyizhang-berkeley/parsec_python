"""Where the cubic spline of a local potential hands over to ``-2 Z/r`` (``PARSEC_SPLINE_COULOMB_TAIL``).

PARSEC's spline is clamped with zero slope at the end of the table, where ``-2 Z/r`` has the slope ``2 Z/r^2``: over
the last ~10 bohr of a table the spline leaves ``-2 Z/r`` by up to 1e-4 Ry.  ``auto`` (the default) evaluates
``-2 Z/r`` from twelve samples past the table's Coulomb onset on (``ParsecPseudopotential.local_spline_cutoff``),
``off`` the spline out to ``rs(ns-1)`` as before.  Checked here (CPU): the values of the switch (and the umbrella),
the cutoff and the sample count the kernels are given, the values and slopes of the host evaluators under both
values, the slope at the hand-over of a table that is ``-2 Z/r`` abruptly, the linear tables unchanged, and a table
without a Coulomb tail keeping its spline to the end.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Forces.periodic import local_potential_slope
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential.fhipp import read_fhipp_pseudopotential
from parsec_python.Pseudopotential.potre import (
    SPLINE_COULOMB_TAIL_ADVANCE,
    SPLINE_COULOMB_TAIL_ENV,
    spline_coulomb_tail_setting,
)
from parsec_python.Pseudopotential.radial_derivatives import evaluate_local_potential_derivative
from parsec_python.V_ion.ionic_potential import coulomb_onset_radius

_DATA = Path(__file__).resolve().parents[1] / "data"
_WIDTH = 6


def _environment(value=None, **extra):
    names = (SPLINE_COULOMB_TAIL_ENV, "PARSEC_FORTRAN_NUMERICS")
    environment = {name: value for name, value in os.environ.items() if name not in names}
    if value is not None:
        environment[SPLINE_COULOMB_TAIL_ENV] = value
    environment.update(extra)
    return mock.patch.dict(os.environ, environment, clear=True)


def _tables():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fhi = read_fhipp_pseudopotential(
            _DATA / "Si_FHIPP.DAT", symbol="Si", number_of_channels=3, electrons_per_orbital=(2.0, 2.0, 0.0),
            core_cutoff_radius=2.1,
        )
    return (("Si POTRE", read_parsec_pseudopotential(_DATA / "Si_POTRE.DAT"), 0), ("Si FHIPP", fhi, 2))


def _fresh(potential):
    # a new object: no cutoff of an earlier value cached
    return dataclasses.replace(potential)


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with _environment():
            self.assertEqual(spline_coulomb_tail_setting(), "auto")
        for value in ("auto", " OFF ", "off"):
            with _environment(value):
                self.assertEqual(spline_coulomb_tail_setting(), value.strip().lower())
        with _environment("on"):
            with self.assertRaisesRegex(ValueError, SPLINE_COULOMB_TAIL_ENV):
                spline_coulomb_tail_setting()

    def test_the_umbrella_restores_parsec(self) -> None:
        with _environment(None, PARSEC_FORTRAN_NUMERICS="on"):
            self.assertEqual(spline_coulomb_tail_setting(), "off")
        with _environment("auto", PARSEC_FORTRAN_NUMERICS="on"):
            self.assertEqual(spline_coulomb_tail_setting(), "auto")


class CutoffTests(unittest.TestCase):
    def test_the_cutoff_follows_the_coulomb_onset(self) -> None:
        for name, table, local_l in _tables():
            with self.subTest(table=name):
                with _environment():
                    potential = _fresh(table)
                    cutoff = potential.local_spline_cutoff(local_l)
                    onset = int(np.searchsorted(potential.radii, coulomb_onset_radius(potential, local_l)))
                    self.assertEqual(cutoff, float(potential.radii[onset + SPLINE_COULOMB_TAIL_ADVANCE]))
                    self.assertLess(cutoff, 10.0)
                    samples = potential.local_spline_samples(local_l)
                    self.assertEqual(float(potential.radii[samples - 2]), cutoff)
                    # every sample from the onset on is -2 Z/r to 1e-9 Ry
                    beyond = potential.radii >= potential.radii[onset]
                    deviation = potential.channel_potentials[local_l][beyond] + 2.0 * potential.ionic_charge / (
                        potential.radii[beyond])
                    self.assertLess(float(np.max(np.abs(deviation))), 1.0e-9)
                with _environment("off"):
                    potential = _fresh(table)
                    self.assertEqual(potential.local_spline_cutoff(local_l), potential.interpolation_cutoff)
                    self.assertEqual(potential.local_spline_samples(local_l), potential.radii.size)

    def test_a_sharp_onset_hands_over_with_a_continuous_slope(self) -> None:
        # a table converted from a UPF file is -2 Z/r abruptly from its generation radius: at the onset the spline's
        # slope is still 7.3e-3 Ry/bohr off 2 Z/r^2; SPLINE_COULOMB_TAIL_ADVANCE samples on it has rung down
        table = read_parsec_pseudopotential(_DATA / "spin_ca_H_POTRE.DAT")
        two_z = 2.0 * table.ionic_charge
        spline = {"use_spline": True, "spline_padding_width": _WIDTH}
        with _environment():
            potential = _fresh(table)
            onset = coulomb_onset_radius(potential, 0)
            cutoff = potential.local_spline_cutoff(0)
            for radius, bound in ((onset, None), (cutoff, 1.0e-6)):
                below = np.array([radius * (1.0 - 1.0e-9)])
                with _environment("off"):
                    slope = local_potential_slope(_fresh(table), below, 0, **spline)[0]
                jump = abs(slope - two_z / below[0] ** 2)
                if bound is None:
                    self.assertGreater(jump, 1.0e-3)
                else:
                    self.assertLess(jump, bound)
            # the evaluators of auto take that spline below the cutoff and the Coulomb slope from it on
            pair = np.array([cutoff * (1.0 - 1.0e-9), cutoff])
            slopes = evaluate_local_potential_derivative(potential, pair, 0, **spline)
            self.assertLess(abs(slopes[1] - slopes[0]), 1.0e-6)

    def test_a_table_without_coulomb_tail_keeps_its_end(self) -> None:
        _, table, local_l = _tables()[0]
        values = np.array(table.channel_potentials[local_l], dtype=float)
        outer = table.radii >= 0.5 * table.radii[-1]
        values[outer] *= 1.0 + 1.0e-4
        channels = dict(table.channel_potentials)
        channels[local_l] = values
        broken = dataclasses.replace(table, channel_potentials=channels)
        with _environment():
            self.assertEqual(broken.local_spline_cutoff(local_l), broken.interpolation_cutoff)
            self.assertEqual(broken.local_spline_samples(local_l), broken.radii.size)


class ValueTests(unittest.TestCase):
    def _values(self, table, local_l, value, points, **options):
        with _environment(value):
            potential = _fresh(table)
            return (
                potential.local_spline_cutoff(local_l),
                potential.local_potential(points, local_l, **options),
                evaluate_local_potential_derivative(potential, points, local_l, **options),
                local_potential_slope(potential, points, local_l, **options),
            )

    def test_the_spline_takes_the_coulomb_tail(self) -> None:
        points = np.concatenate((np.linspace(0.0, 10.0, 20001), np.linspace(10.0, 130.0, 24001)))
        spline = {"use_spline": True, "spline_padding_width": _WIDTH}
        for name, table, local_l in _tables():
            with self.subTest(table=name):
                cutoff, value, slope, exact_slope = self._values(table, local_l, None, points, **spline)
                _, old_value, old_slope, old_exact = self._values(table, local_l, "off", points, **spline)
                two_z = 2.0 * table.ionic_charge
                inside, beyond = points < cutoff, points >= cutoff
                np.testing.assert_array_equal(value[inside], old_value[inside])
                np.testing.assert_array_equal(value[beyond], -two_z / points[beyond])
                np.testing.assert_array_equal(exact_slope[inside], old_exact[inside])
                np.testing.assert_array_equal(exact_slope[beyond], two_z / points[beyond] ** 2)
                np.testing.assert_array_equal(slope[beyond], two_z / points[beyond] ** 2)
                # the old spline left -2 Z/r near the end of the table
                end = (points > table.interpolation_cutoff - 15.0) & (points < table.interpolation_cutoff)
                self.assertGreater(float(np.max(np.abs(old_value[end] + two_z / points[end]))), 1.0e-6)
                # continuous at the cutoff: the spline meets the sample there
                with _environment():
                    potential = _fresh(table)
                    pair = potential.local_potential(np.array([cutoff * (1 - 1e-12), cutoff]), local_l, **spline)
                self.assertLess(abs(pair[1] - pair[0]), 1.0e-9)

    def test_the_slope_is_the_derivative_of_the_value(self) -> None:
        spline = {"use_spline": True, "spline_padding_width": _WIDTH}
        for name, table, local_l in _tables():
            with self.subTest(table=name):
                with _environment():
                    potential = _fresh(table)
                    cutoff = potential.local_spline_cutoff(local_l)
                    points = np.array([0.5 * cutoff, 0.9 * cutoff, 1.1 * cutoff, 20.0, 60.0, 100.0])
                    step = 1.0e-5
                    difference = (
                        potential.local_potential(points + step, local_l, **spline)
                        - potential.local_potential(points - step, local_l, **spline)
                    ) / (2.0 * step)
                    slopes = local_potential_slope(potential, points, local_l, **spline)
                np.testing.assert_allclose(slopes, difference, rtol=1.0e-6, atol=1.0e-9)

    def test_linear_tables_are_unchanged(self) -> None:
        points = np.linspace(0.0, 130.0, 50001)
        for name, table, local_l in _tables():
            with self.subTest(table=name):
                for options in ({}, {"direct": True}):
                    new = self._values(table, local_l, None, points, **options)
                    old = self._values(table, local_l, "off", points, **options)
                    for a, b in zip(new[1:], old[1:]):
                        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
