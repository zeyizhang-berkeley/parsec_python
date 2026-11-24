"""The local potential Fourier-filtered to the grid (``PARSEC_LOCAL_FILTER``, ``Pseudopotential/local_filter.py``).

Checked here (CPU): the values of the switch (an unknown value is refused when the tables are read); the filtered table
keeps the components of the file's potential below the grid's cutoff, removes those that alias, is ``-2 Z/r`` to
round-off beyond its onset and does not depend on the width of the Gaussian split; every consumer of the local
potential reads it and no projector does; the grid step of the filter on cluster and oblique grids; the first-order
egg-box sum ``h^3 sum_i rho(r_i - R) V(r_i - R)`` of an atom moved over one grid step; and a prepared cluster, whose
local potential alone changes under ``on`` and which ``off`` leaves as before.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential import local_filter
from parsec_python.Pseudopotential.fhipp import read_fhipp_pseudopotential
from parsec_python.Pseudopotential.local_filter import (
    LOCAL_FILTER_ENV,
    apply_local_filters,
    filter_mask,
    filter_step,
    filtered_local_table,
    local_filter_setting,
)
from parsec_python.Pseudopotential.radial_derivatives import local_potential_derivative_table
from parsec_python.V_ion.ionic_potential import (
    _coulomb_to_round_off,
    coulomb_onset_radius,
)

_DATA = Path(__file__).resolve().parents[1] / "data"


def _setting(value):
    return mock.patch.dict(os.environ, {LOCAL_FILTER_ENV: value})


def _oxygen():
    return read_parsec_pseudopotential(_DATA / "water_O_POTRE.DAT")


def _silicon():
    return read_fhipp_pseudopotential(
        _DATA / "Si_FHIPP.DAT", symbol="Si", number_of_channels=3, electrons_per_orbital=[2, 2, 0],
        core_cutoff_radius=2.0,
    )


def _radial_transform(potential, table, q, charge):
    """``4 pi int r^2 (V + 2 Z erf(2 r)/r) j0(q r) dr`` on a fine uniform mesh (the cubic spline of ``table``): the
    transform of ``V`` but for the Coulomb part, whose Gaussian-smeared rest has no components at the ``q`` here."""

    from scipy.special import erf

    from parsec_python.Pseudopotential.radial_spline import ParsecRadialSpline

    r = np.arange(1, 30001) * 5.0e-4
    spline = ParsecRadialSpline.from_positive_grid(potential.radii, table, 6)
    values = np.where(r < potential.radii[-2], spline(np.minimum(r, potential.radii[-2])), -2.0 * charge / r)
    short = values + 2.0 * charge * erf(2.0 * r) / r
    return np.array([4.0 * np.pi * np.sum(5.0e-4 * r * short * np.sin(k * r) / k) for k in np.atleast_1d(q)])


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOCAL_FILTER_ENV, None)
            self.assertEqual(local_filter_setting(), local_filter.LOCAL_FILTER_DEFAULT)
        for value in ("on", " ON ", "off", "all"):
            with _setting(value):
                self.assertEqual(local_filter_setting(), value.strip().lower())
        with _setting("auto"):
            with self.assertRaises(ValueError):
                local_filter_setting()

    def test_unknown_value_refused_when_the_tables_are_read(self) -> None:
        from parsec_python.V_ion.ionic_potential import load_pseudopotentials

        with _setting("maybe"):
            with self.assertRaises(ValueError):
                load_pseudopotentials({})

    def test_off_returns_the_tables_of_the_file(self) -> None:
        potential = _oxygen()

        class Grid:
            spacing = 0.2

        with _setting("off"):
            result = apply_local_filters({"O": potential}, {"O": mock.Mock(local_angular_momentum=1)}, Grid())
        self.assertIs(result["O"], potential)


class TableTests(unittest.TestCase):
    def test_band_limit_and_coulomb_tail(self) -> None:
        for potential, local_l in ((_oxygen(), 1), (_silicon(), 2)):
            charge = float(potential.ionic_charge)
            for step in (0.16, 0.3):
                table = filtered_local_table(potential, local_l, step, 6)
                q0 = np.pi / step
                raw = potential.channel_potentials[local_l]
                low = np.array([0.2, 0.4, 0.6]) * q0
                kept = _radial_transform(potential, table, low, charge)
                original = _radial_transform(potential, raw, low, charge)
                # F(0.6 q0) = 1 - 3e-6: the components the grid represents are kept
                np.testing.assert_allclose(kept, original, rtol=2e-5, atol=1e-6 * np.abs(original).max())
                high = np.array([2.5, 3.0]) * q0
                removed = _radial_transform(potential, table, high, charge)
                before = _radial_transform(potential, raw, high, charge)
                self.assertLess(np.abs(removed).max(), 1e-3 * np.abs(before).max() + 1e-7)
                # -2 Z/r to round-off from the onset on (the rule the periodic sums read)
                filtered = replace(potential, filtered_local_potentials={local_l: table})
                onset = coulomb_onset_radius(filtered, local_l)
                self.assertTrue(_coulomb_to_round_off(filtered, local_l))
                beyond = potential.radii > onset
                np.testing.assert_allclose(table[beyond] * potential.radii[beyond], -2.0 * charge, rtol=0, atol=1e-13)
                self.assertTrue(np.all(np.isfinite(table)))

    def test_independent_of_the_gaussian_split(self) -> None:
        potential = _oxygen()
        table = filtered_local_table(potential, 1, 0.2, 6)
        with mock.patch.object(local_filter, "SPLIT_STEPS", 5.0), mock.patch.object(local_filter, "_TABLES", {}):
            wider = filtered_local_table(potential, 1, 0.2, 6)
        np.testing.assert_allclose(wider, table, rtol=0, atol=1e-9)

    def test_mask(self) -> None:
        q0 = np.pi / 0.2
        values = filter_mask(np.array([0.0, 1.0, 1.5, 2.0]) * q0, 0.2)
        self.assertGreater(values[0], 1.0 - 1e-15)
        self.assertGreater(values[1], 1.0 - 2.1e-4)
        self.assertAlmostEqual(values[2], 0.5, places=14)
        self.assertLess(values[3], 2.1e-4)


class ConsumerTests(unittest.TestCase):
    def test_every_local_reader_takes_the_filtered_table(self) -> None:
        potential = _oxygen()
        table = filtered_local_table(potential, 1, 0.2, 6)
        filtered = replace(potential, filtered_local_potentials={1: table})
        np.testing.assert_array_equal(filtered.local_channel_potential(1), table)
        np.testing.assert_array_equal(potential.local_channel_potential(1), potential.channel_potentials[1])
        r = np.linspace(0.0, 6.0, 301)
        for keywords in ({}, {"use_spline": True, "spline_padding_width": 6}, {"direct": True}):
            self.assertGreater(
                np.abs(filtered.local_potential(r, 1, **keywords) - potential.local_potential(r, 1, **keywords)).max(),
                1e-3,
            )
        spline = filtered.local_potential_spline(1, 6)
        values = filtered.local_potential(r[1:], 1, use_spline=True, spline_padding_width=6)
        # the spline local_potential evaluates, up to its hand-over to -2 Z/r (PARSEC_SPLINE_COULOMB_TAIL=auto), which
        # follows the Coulomb onset of the filtered table, not of the file's
        cutoff = filtered.local_spline_cutoff(1)
        self.assertGreaterEqual(cutoff, coulomb_onset_radius(filtered, 1))
        self.assertGreater(coulomb_onset_radius(filtered, 1), coulomb_onset_radius(potential, 1))
        inside = r[1:] < cutoff
        self.assertTrue(np.any(inside) and np.any(~inside))
        np.testing.assert_array_equal(spline(r[1:])[inside], values[inside])
        np.testing.assert_allclose(
            values[~inside], -2.0 * potential.ionic_charge / r[1:][~inside], rtol=0.0, atol=1.0e-14
        )
        with mock.patch.dict(os.environ, {"PARSEC_SPLINE_COULOMB_TAIL": "off"}):
            fresh = replace(potential, filtered_local_potentials={1: table})
            np.testing.assert_array_equal(
                spline(r[1:]), fresh.local_potential(r[1:], 1, use_spline=True, spline_padding_width=6)
            )
        self.assertGreater(
            np.abs(
                local_potential_derivative_table(filtered, 1) - local_potential_derivative_table(potential, 1)
            ).max(),
            1e-3,
        )
        # the projectors keep V_l - V_local of the file
        for channel in potential.radial_wavefunctions:
            if channel == 1:
                continue
            np.testing.assert_array_equal(
                filtered.radial_projector(channel, 1)[0], potential.radial_projector(channel, 1)[0]
            )

    def test_filter_step(self) -> None:
        class Cluster:
            spacing = 0.25

        class Oblique:
            step_vectors = np.array([[0.3, 0.0, 0.0], [0.15, 0.26, 0.0], [0.0, 0.0, 0.35]])

        self.assertEqual(filter_step(Cluster()), 0.25)
        self.assertAlmostEqual(filter_step(Oblique()), 0.35, places=15)


class ProjectorTests(unittest.TestCase):
    def test_mask_function_filter(self) -> None:
        from scipy.special import spherical_jn

        from parsec_python.Pseudopotential.local_filter import (
            PROJECTOR_SUPPORT_FACTOR,
            filtered_projector_tables,
        )
        from parsec_python.Pseudopotential.radial_derivatives import spline_derivative
        from parsec_python.Pseudopotential.radial_spline import ParsecRadialSpline

        potential = _oxygen()
        step = 0.16
        projectors, slopes, end = filtered_projector_tables(potential, 1, step, 6)
        support = potential.projector_support_radius
        self.assertAlmostEqual(end, PROJECTOR_SUPPORT_FACTOR * support, places=12)
        self.assertEqual(sorted(projectors), [0])
        filtered = projectors[0]
        raw, _ = potential.radial_projector(0, 1)
        self.assertTrue(np.all(filtered[potential.radii >= end] == 0.0))
        spline = ParsecRadialSpline.from_positive_grid(potential.radii, filtered, 6)
        for radius in (0.3, 0.8, 1.5, 1.8):
            index = int(np.searchsorted(potential.radii, radius))
            exact = float(spline_derivative(spline, np.array([potential.radii[index]]))[0])
            self.assertAlmostEqual(slopes[0][index], exact, delta=1e-4 * np.abs(slopes[0]).max())
        # the components beyond q0 = pi/h, and the overlap with the pseudo-wavefunction of the channel
        dr = 0.002
        r = dr * np.arange(1, int(end / dr) + 10)
        raw_spline = ParsecRadialSpline.from_positive_grid(potential.radii, raw, 6)
        old = np.where(r <= support, raw_spline(np.minimum(r, support)), 0.0)
        new = np.where(r < end, spline(r), 0.0)
        q0 = np.pi / step
        q = 0.02 * np.arange(1, int(4 * q0 / 0.02))
        bessel = spherical_jn(0, np.outer(q, r))
        old_q, new_q = bessel @ (r * r * old), bessel @ (r * r * new)

        def beyond(values, start):
            return np.sqrt(np.sum((values[q > start * q0] * q[q > start * q0]) ** 2))

        # the mask of the local potential: the components beyond 2 q0 go, those below q0 stay
        self.assertGreater(beyond(old_q, 2.0), 30.0 * beyond(new_q, 2.0))
        low = q < 0.8 * q0
        np.testing.assert_allclose(new_q[low], old_q[low], rtol=0, atol=1e-4 * np.abs(old_q).max())
        wave = ParsecRadialSpline.from_positive_grid(potential.radii, potential.radial_wavefunctions[0], 6)(r) / r
        self.assertLess(abs(np.sum(r * r * new * wave) / np.sum(r * r * old * wave) - 1.0), 1e-4)


class ImageReachTests(unittest.TestCase):
    def test_wire_and_slab_images_reach_the_filtered_onset(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Grid.confined import ConfinedPeriodicGrid
        from parsec_python.Pseudopotential.local_filter import filtered_local_reach

        vectors = np.array([[4.73, 0.0, 0.0], [-2.365, 4.0963001599, 0.0], [0.0, 0.0, 1.0]])
        slab = SimpleNamespace(
            periodicity=2,
            cell=SimpleNamespace(lattice_vectors=vectors),
            periodic_lattice_vectors=vectors[:2],
            fractional_coordinates=lambda p: (np.atleast_2d(p) @ np.linalg.inv(vectors))[:, :2],
        )
        images = ConfinedPeriodicGrid.compensation_image_translations
        position = np.array([0.1, 0.2, 0.0])
        base = images(slab, position, 0.7)
        np.testing.assert_array_equal(images(slab, position, 0.7, 0.0), base)  # the file's table: Fortran's set
        wide = images(slab, position, 0.7, 12.0)
        self.assertGreater(len(wide), len(base))
        # every image within the reach of a point of the cell is in the set
        reach = np.linalg.norm(wide[:, :2], axis=1)
        self.assertGreaterEqual(reach.max(), 12.0)
        potential = _oxygen()
        self.assertEqual(filtered_local_reach(potential, 1), 0.0)
        filtered = replace(potential, filtered_local_potentials={1: filtered_local_table(potential, 1, 0.3, 6)})
        self.assertAlmostEqual(filtered_local_reach(filtered, 1), coulomb_onset_radius(filtered, 1))


class EggBoxTests(unittest.TestCase):
    def test_first_order_sum_over_one_step(self) -> None:
        from parsec_python.Pseudopotential.radial_spline import ParsecRadialSpline

        # (at coarser steps the atomic density sampled here has components beyond the cutoff itself)
        potential = _oxygen()
        step = 0.2
        table = filtered_local_table(potential, 1, step, 6)
        density = ParsecRadialSpline.from_positive_grid(potential.radii, potential.valence_density, 6)
        n = int(6.0 / step) + 2
        axis = (np.arange(-n, n) + 0.5) * step
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

        def energies(values):
            spline = ParsecRadialSpline.from_positive_grid(potential.radii, values, 6)
            result = []
            for t in np.arange(8) / 8.0:
                d = np.linalg.norm(grid - np.array([t, 0.5, 0.5]) * step, axis=1)
                inside = d < 6.0
                result.append(step**3 * np.sum(density(d[inside]) * spline(d[inside])))
            return np.array(result)

        raw = energies(potential.channel_potentials[1])
        filtered = energies(table)
        self.assertGreater(np.ptp(raw), 20.0 * np.ptp(filtered))


class PreparedClusterTests(unittest.TestCase):
    _INPUT = """Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.35
Correlation_Type: ca
States_Num: 6
Atom_Types_Num: 1
Atom_Type: O
Local_Component: p
Cubic_Spline: .true.
begin Atom_Coord
0.05 0.11 0.6
0.05 0.11 -0.6
end Atom_Coord
"""

    def _prepare(self, value):
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.SCF.single_point import prepare_single_point

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "parsec.in").write_text(self._INPUT)
            (Path(folder) / "O_POTRE.DAT").write_bytes((_DATA / "water_O_POTRE.DAT").read_bytes())
            with _setting(value):
                translation = parse_parsec_input(Path(folder) / "parsec.in")
                return prepare_single_point(translation.problem)

    def test_all_filters_the_projectors_too(self) -> None:
        on = self._prepare("on")
        both = self._prepare("all")
        np.testing.assert_array_equal(both.ionic_potential, on.ionic_potential)
        oxygen = both.pseudopotentials["O"]
        self.assertIn(0, oxygen.filtered_projectors)
        self.assertGreater(oxygen.projector_support_radius, on.pseudopotentials["O"].projector_support_radius)
        self.assertGreater(both.nonlocal_operator.projectors.nnz, on.nonlocal_operator.projectors.nnz)

    def test_only_the_local_potential_changes(self) -> None:
        off = self._prepare("off")
        on = self._prepare("on")
        self.assertFalse(off.pseudopotentials["O"].filtered_local_potentials)
        self.assertIn(1, on.pseudopotentials["O"].filtered_local_potentials)
        self.assertGreater(np.abs(on.ionic_potential - off.ionic_potential).max(), 1e-3)
        np.testing.assert_array_equal(on.initial_density, off.initial_density)
        for name in ("projector_values",):
            if hasattr(off.nonlocal_operator, name):
                np.testing.assert_array_equal(getattr(on.nonlocal_operator, name), getattr(off.nonlocal_operator, name))
        np.testing.assert_array_equal(
            on.nonlocal_operator.as_sparse().toarray(), off.nonlocal_operator.as_sparse().toarray()
        )


if __name__ == "__main__":
    unittest.main()
