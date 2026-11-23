"""Martins-Wang and FHIPP pseudopotentials and the per-species input labels.

The readers are tested on small synthetic files for the layout rules of
``pseudo.f90`` and on the files of the Fortran examples against what Fortran
PARSEC prints for them (grid parameters, normalization, Kleinman-Bylander
integrals, sizes of the nonlocal blocks).  The input tests cover the ESDF
cursors of the per-species labels (``usrinputfile.F90:1756-1907``): the n-th
occurrence of a label goes to the n-th species that reads it.
"""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Pseudopotential import (
    parsec_radial_integral,
    read_fhipp_pseudopotential,
    read_martins_wang_header,
    read_martins_wang_pseudopotential,
    read_parsec_pseudopotential,
)
from parsec_python.Pseudopotential.radial_derivatives import (
    local_potential_derivative_table,
    log_grid_derivative,
)
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials
from parsec_python.models import Atom, GridSettings, SpeciesPotential

DATA = Path(__file__).resolve().parents[1] / "data"
EXAMPLE_SILICON = Path(__file__).resolve().parents[4] / "examples" / "3d_hybrid_functionals" / "Si_FHIPP.DAT"


def assert_printed(test, value: float, printed: float, digits: int) -> None:
    """``value`` rounds to ``printed``, which has ``digits`` significant digits."""

    if printed == 0.0:
        test.assertAlmostEqual(value, 0.0, delta=0.5 * 10.0**-digits)
        return
    unit = 10.0 ** (math.floor(math.log10(abs(printed))) - digits + 1)
    test.assertAlmostEqual(value, printed, delta=0.5001 * unit)


def kb_integral(potential, angular_momentum: int, local_l: int) -> float:
    wave = potential.radial_wavefunctions[angular_momentum]
    return parsec_radial_integral(
        potential.radii,
        wave
        * wave
        * (
            potential.channel_potentials[angular_momentum]
            - potential.channel_potentials[local_l]
        ),
    )


def martins_wang_grid(count: int = 240) -> np.ndarray:
    """Radii ``a*(exp(b*(i-1)) - 1)`` at Fortran indices ``i = 2..count+1``."""

    a, b = 2.5e-4, 0.05
    return a * (np.exp(b * np.arange(1, count + 1)) - 1.0)


def write_martins_wang(
    path: Path,
    *,
    references=(1, 1, 0),
    local_slot: int = 2,
    core: bool = False,
    count: int = 240,
    occupations=(1.0, 0.5, 0.0),
) -> np.ndarray:
    radii = martins_wang_grid(count)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            f"{count}, {int(core)}, 11,  1.5, {local_slot},  {occupations[0]},  "
            f"{occupations[1]},  {occupations[2]}, 0  |nrr,icor,iatom,z,loc\n"
        )
        stream.write(" ".join(str(value) for value in references) + "   1 1 0   |<-iref\n")
        for r in radii:
            values = [
                -1.0 / np.sqrt(r * r + 1.0),
                -0.8 / np.sqrt(r * r + 0.5),
                -0.6 / np.sqrt(r * r + 0.2),
                np.exp(-r),
                r * np.exp(-r),
                r * r * np.exp(-r),
            ]
            if core:
                values.append(0.3 * np.exp(-2.0 * r))
            # Two further columns (spin-orbit) are not read.
            values += [9.0, 9.0]
            stream.write(" ".join(f"{value:.12E}" for value in [r, *values]) + "\n")
        stream.write(" #r, vs(Hart.),vp,vd,phi_s,phi_p,phi_d\n")
    return radii


def write_fhipp(
    path: Path,
    *,
    channels: int = 3,
    mesh: int = 300,
    amesh: float = 1.0247,
    core: bool = True,
    file_channels: int | None = None,
    second_mesh: int | None = None,
    second_amesh: float | None = None,
    start: float = 4.0e-4,
    coulomb: float = 0.0,
) -> np.ndarray:
    radii = start * amesh ** np.arange(1, mesh + 1)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"0.30000000000000E+01   {file_channels or channels}\n")
        for _ in range(10):
            stream.write("  0.0000    .00e+00   .00e+00\n")
        for l in range(channels):
            size = second_mesh if (l == 1 and second_mesh) else mesh
            ratio = second_amesh if (l == 1 and second_amesh) else amesh
            stream.write(f"{size}  {ratio:.14E}\n")
            for index, r in enumerate(radii[:size], start=1):
                u = r ** (l + 1) * np.exp(-r)
                v = -3.0 / np.sqrt(r * r + 0.3 * (l + 1)) + coulomb / r
                stream.write(f"{index:4d} {r:.14E} {u:.14E} {v:.14E}\n")
        if core:
            for r in radii:
                rho = 4.0 * np.pi * 0.2 * np.exp(-3.0 * r)
                stream.write(f"{r:.14E} {rho:.14E} {-rho:.14E} {rho:.14E}\n")
    return radii


class MartinsWangReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_layout_of_a_synthetic_file(self) -> None:
        path = self.directory / "X_POTRW.DAT"
        radii = write_martins_wang(path, core=True)
        header = read_martins_wang_header(path)
        self.assertEqual(header.present, (0, 1))
        self.assertEqual(header.local_angular_momentum, 1)
        potential = read_martins_wang_pseudopotential(path, symbol="X", core_cutoff_radius=2.0)
        np.testing.assert_allclose(potential.radii, radii, rtol=1e-11)
        self.assertEqual(sorted(potential.channel_potentials), [0, 1])
        # Hartree -> Rydberg, R_l -> u_l = r R_l, core as read.
        np.testing.assert_allclose(
            potential.channel_potentials[0], -2.0 / np.sqrt(radii**2 + 1.0), rtol=1e-11
        )
        np.testing.assert_allclose(
            potential.radial_wavefunctions[1], radii * radii * np.exp(-radii), rtol=1e-11
        )
        np.testing.assert_allclose(potential.core_density, 0.3 * np.exp(-2.0 * radii), rtol=1e-11)
        self.assertTrue(potential.has_nonlinear_core_correction)
        self.assertEqual(potential.ionic_charge, 1.5)
        self.assertEqual(potential.channel_occupations, {0: 1.0, 1: 0.5})
        self.assertEqual(potential.local_channel, 1)
        self.assertEqual(potential.core_radius, 2.0)
        self.assertEqual(potential.projector_support_radius, 2.0)
        self.assertEqual(potential.file_format, "martins_wang")
        # The grid parameters of pseudo.f90 (c = a).
        self.assertAlmostEqual(potential.radial_parameter_a / 2.5e-4, 1.0, delta=1e-10)
        self.assertAlmostEqual(potential.radial_parameter_b / 0.05, 1.0, delta=1e-10)
        self.assertEqual(potential.radial_offset, potential.radial_parameter_a)
        expected = (
            1.0 * (radii * np.exp(-radii)) ** 2 + 0.5 * (radii * radii * np.exp(-radii)) ** 2
        ) / (4.0 * np.pi * radii * radii)
        np.testing.assert_allclose(potential.valence_density, expected, rtol=1e-11)

    def test_refusals(self) -> None:
        path = self.directory / "X_POTRW.DAT"
        # Channels that are not s, s+p or s+p+d are mislabeled by pseudo.f90.
        write_martins_wang(path, references=(1, 0, 1))
        with self.assertRaisesRegex(ValueError, "must be s, s and p"):
            read_martins_wang_header(path)
        write_martins_wang(path, local_slot=0)
        with self.assertRaisesRegex(ValueError, "loc=0"):
            read_martins_wang_header(path)
        write_martins_wang(path, local_slot=3)
        with self.assertRaisesRegex(ValueError, "loc=3"):
            read_martins_wang_header(path)
        # pseudo.f90 error 210.
        write_martins_wang(path, count=150)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            read_martins_wang_header(path)
        write_martins_wang(path)
        with self.assertRaisesRegex(ValueError, "positive"):
            read_martins_wang_pseudopotential(path, symbol="X", core_cutoff_radius=0.0)

    def test_sodium_file_against_fortran(self) -> None:
        # parsec.out of the NaH parity run (Fortran PARSEC, 2026-10-27).
        potential = read_martins_wang_pseudopotential(
            DATA / "Na_POTRW.DAT", symbol="Na", core_cutoff_radius=3.1
        )
        assert_printed(self, potential.radial_parameter_a, 0.225341e-03, 6)
        assert_printed(self, potential.radial_parameter_b, 0.100000e-01, 6)
        assert_printed(self, potential.radial_offset, 0.225341e-03, 6)
        self.assertFalse(potential.has_nonlinear_core_correction)
        self.assertEqual(potential.local_channel, 1)
        self.assertEqual(potential.number_of_channels, 3)
        self.assertEqual(potential.ionic_charge, 1.0)
        self.assertEqual(potential.channel_occupations, {0: 1.0, 1: 0.0, 2: 0.0})
        for l, norm in ((0, 1.00001667), (1, 1.00001667), (2, 1.01036503)):
            wave = potential.radial_wavefunctions[l]
            self.assertAlmostEqual(
                parsec_radial_integral(potential.radii, wave * wave), norm, delta=5.01e-9
            )
        for l, value in ((0, 0.01109253), (1, 0.0), (2, -0.0)):
            self.assertAlmostEqual(kb_integral(potential, l, 1), value, delta=5.01e-9)
        # "Sizes of all non-local blocks": 676 points for Na of the NaH case.
        grid = build_cluster_grid(GridSettings(spacing=0.3 / 0.529177, radius=6.0 / 0.529177))
        species = {
            "Na": SpeciesPotential(
                DATA / "Na_POTRW.DAT",
                1,
                pseudopotential_format="martins_wang",
                core_cutoff_radius=3.1,
            )
        }
        potentials = load_pseudopotentials(species)
        # The recentred position of Na in that run.  Fortran counts the grid
        # points within the projector radius; a column drops the points where
        # its value is zero (beyond the cutoffs of both channels).
        position = np.array([0.0, 0.0, -1.782957])
        radius = potentials["Na"].projector_support_radius
        inside = np.linalg.norm(grid.coordinates - position, axis=1) <= radius
        self.assertEqual(int(np.count_nonzero(inside)), 676)
        operator = build_nonlocal_projectors(grid, [Atom("Na", position)], potentials, species)
        rows = np.unique(operator.projectors.indices)
        self.assertTrue(set(rows.tolist()) <= set(np.flatnonzero(inside).tolist()))

    def test_loader_checks_the_local_channel(self) -> None:
        species = SpeciesPotential(
            DATA / "Na_POTRW.DAT", 0, pseudopotential_format="martins_wang", core_cutoff_radius=3.1
        )
        with self.assertRaisesRegex(ValueError, "names the local channel l=1"):
            load_pseudopotentials({"Na": species})
        with self.assertRaisesRegex(ValueError, "SO_PSP is not supported"):
            load_pseudopotentials({"Na": replace(species, local_angular_momentum=1, spin_orbit=True)})
        with self.assertRaisesRegex(ValueError, "only core_cutoff_radius"):
            SpeciesPotential(
                DATA / "Na_POTRW.DAT",
                1,
                pseudopotential_format="martins_wang",
                core_cutoff_radius=3.1,
                number_of_channels=3,
            )


class FHIPPReaderTests(unittest.TestCase):
    """The file as ``pseudo.f90`` reads it (``PARSEC_FHI_ORIGIN=off``; :class:`FHIOriginTests` covers ``auto``)."""

    def setUp(self) -> None:
        environment = mock.patch.dict(os.environ, {"PARSEC_FHI_ORIGIN": "off"})
        environment.start()
        self.addCleanup(environment.stop)
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def read(self, path: Path, **changes):
        options = dict(
            symbol="X",
            number_of_channels=3,
            electrons_per_orbital=(2.0, 1.0, 0.0, 0.0),
            core_cutoff_radius=2.2,
            nonlinear_core_correction=True,
        )
        options.update(changes)
        return read_fhipp_pseudopotential(path, **options)

    def test_layout_of_a_synthetic_file(self) -> None:
        path = self.directory / "X_FHIPP.DAT"
        radii = write_fhipp(path)
        potential = self.read(path)
        np.testing.assert_allclose(potential.radii, radii, rtol=1e-13)
        self.assertEqual(potential.ionic_charge, 3.0)
        self.assertEqual(potential.radial_parameter_c, 0.0)
        self.assertEqual(potential.radial_offset, 0.0)
        self.assertAlmostEqual(potential.radial_parameter_a, radii[0] / 1.0247, delta=1e-18)
        self.assertAlmostEqual(potential.radial_parameter_b, math.log(1.0247), delta=1e-15)
        np.testing.assert_allclose(
            potential.channel_potentials[2], -6.0 / np.sqrt(radii**2 + 0.9), rtol=1e-12
        )
        np.testing.assert_allclose(potential.radial_wavefunctions[1], radii**2 * np.exp(-radii), rtol=1e-12)
        # The core column is 4*pi times the volume density.
        np.testing.assert_allclose(potential.core_density, 0.2 * np.exp(-3.0 * radii), rtol=1e-12)
        self.assertEqual(potential.channel_occupations, {0: 2.0, 1: 1.0, 2: 0.0})
        self.assertEqual(potential.core_radius, 2.2)
        self.assertEqual(potential.file_format, "fhipp")
        without = self.read(path, nonlinear_core_correction=False)
        self.assertFalse(without.has_nonlinear_core_correction)
        np.testing.assert_array_equal(without.core_density, 0.0)

    def test_refusals(self) -> None:
        path = self.directory / "X_FHIPP.DAT"
        # pseudo.f90 error 203.
        write_fhipp(path, file_channels=4)
        with self.assertRaisesRegex(ValueError, "Potential_Num is 3"):
            self.read(path)
        # pseudo.f90 error 211.
        write_fhipp(path, second_mesh=299)
        with self.assertRaisesRegex(ValueError, "grid size"):
            self.read(path)
        # The check pseudo.f90 error 212 means.
        write_fhipp(path, second_amesh=1.0248)
        with self.assertRaisesRegex(ValueError, "radial increment"):
            self.read(path)
        write_fhipp(path, core=False)
        with self.assertRaisesRegex(ValueError, "Nonlinear_Core_Correction"):
            self.read(path)
        with self.assertRaisesRegex(ValueError, "Electron_Per_Orbital"):
            self.read(path, electrons_per_orbital=(2.0,), nonlinear_core_correction=False)

    def test_files_against_fortran(self) -> None:
        # parsec.out of the SiH4 and Mg2 parity runs (Fortran PARSEC, 2026-10-27).
        silicon = read_fhipp_pseudopotential(
            DATA / "Si_FHIPP.DAT",
            symbol="Si",
            number_of_channels=3,
            electrons_per_orbital=(2, 2, 0),
            core_cutoff_radius=2.5,
        )
        assert_printed(self, silicon.radial_parameter_a, 0.435668e-03, 6)
        assert_printed(self, silicon.radial_parameter_b, 0.243999e-01, 6)
        self.assertEqual(silicon.ionic_charge, 4.0)
        for l, norm, kb in ((0, 1.00009923, 0.73656875), (1, 1.00009923, 0.26759002), (2, 1.05091999, 0.0)):
            wave = silicon.radial_wavefunctions[l]
            self.assertAlmostEqual(parsec_radial_integral(silicon.radii, wave * wave), norm, delta=5.01e-9)
            self.assertAlmostEqual(kb_integral(silicon, l, 2), kb, delta=5.01e-9)
        magnesium = read_fhipp_pseudopotential(
            DATA / "Mg_FHIPP.DAT",
            symbol="Mg",
            number_of_channels=4,
            electrons_per_orbital=(2, 0, 0, 0),
            core_cutoff_radius=2.5,
            nonlinear_core_correction=True,
        )
        assert_printed(self, magnesium.radial_parameter_a, 0.508279e-03, 6)
        self.assertTrue(magnesium.has_nonlinear_core_correction)
        for l, norm, kb in (
            (0, 1.00009923, 0.25153067),
            (1, 1.04106403, 0.48964078),
            (2, 1.00009923, 0.0),
            (3, 1.09397018, -0.00289893),
        ):
            wave = magnesium.radial_wavefunctions[l]
            self.assertAlmostEqual(parsec_radial_integral(magnesium.radii, wave * wave), norm, delta=5.01e-9)
            self.assertAlmostEqual(kb_integral(magnesium, l, 2), kb, delta=5.01e-9)
        # "Sizes of all non-local blocks": 360 points for Si of SiH4 and each Mg.
        grid = build_cluster_grid(GridSettings(spacing=0.3 / 0.529177, radius=6.0 / 0.529177))
        species = {
            "Si": SpeciesPotential(
                DATA / "Si_FHIPP.DAT",
                2,
                pseudopotential_format="fhipp",
                number_of_channels=3,
                electrons_per_orbital=(2, 2, 0),
                core_cutoff_radius=2.5,
            )
        }
        radius = load_pseudopotentials(species)["Si"].projector_support_radius
        inside = np.linalg.norm(grid.coordinates, axis=1) <= radius
        self.assertEqual(int(np.count_nonzero(inside)), 360)


class RadialOffsetTests(unittest.TestCase):
    """``c`` of ``r(i) = a*exp(b*(i-1)) - c`` in the radial derivatives."""

    def test_martins_files_keep_c_equal_to_a(self) -> None:
        potential = read_parsec_pseudopotential(DATA / "spin_ca_O_POTRE.DAT")
        self.assertIsNone(potential.radial_parameter_c)
        self.assertEqual(potential.radial_offset, potential.radial_parameter_a)

    @mock.patch.dict(os.environ, {"PARSEC_FHI_ORIGIN": "off"})
    def test_derivative_on_an_fhipp_grid(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "X_FHIPP.DAT"
            write_fhipp(path, mesh=500, core=False)
            potential = read_fhipp_pseudopotential(
                path,
                symbol="X",
                number_of_channels=3,
                electrons_per_orbital=(2, 1, 0),
                core_cutoff_radius=2.0,
            )
        radii = potential.radii
        derivative = log_grid_derivative(potential, np.exp(-radii))
        inner = (radii > 1e-2) & (radii < 20.0)
        np.testing.assert_allclose(derivative[inner], -np.exp(-radii[inner]), rtol=1e-8)
        # The Martins assumption c = a would be wrong on this grid.
        wrong = log_grid_derivative(replace(potential, radial_parameter_c=None), np.exp(-radii))
        self.assertGreater(np.max(np.abs(wrong[inner] / derivative[inner] - 1.0)), 1e-3)
        # The local-potential table ends with the Coulomb slope.
        table = local_potential_derivative_table(potential, 2)
        self.assertAlmostEqual(table[-1], 2.0 * 3.0 / radii[-1] ** 2, delta=1e-12)


class FHIOriginTests(unittest.TestCase):
    """``PARSEC_FHI_ORIGIN``: the samples ``auto`` adds below the first sample of an FHIPP table."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def read(self, path: Path, setting: str, **changes):
        options = dict(
            symbol="X",
            number_of_channels=3,
            electrons_per_orbital=(2.0, 1.0, 0.0, 0.0),
            core_cutoff_radius=2.2,
            nonlinear_core_correction=True,
        )
        options.update(changes)
        with mock.patch.dict(os.environ, {"PARSEC_FHI_ORIGIN": setting}):
            return read_fhipp_pseudopotential(path, **options)

    def silicon(self, setting: str, path: Path = EXAMPLE_SILICON):
        """The PBE Si table of the examples (refitted under auto); ``DATA / "Si_FHIPP.DAT"`` is the CA table, whose
        samples auto keeps."""

        return self.read(
            path, setting, symbol="Si", electrons_per_orbital=(2, 2, 0),
            core_cutoff_radius=2.5, nonlinear_core_correction=False,
        )

    def test_tables_with_a_coulomb_term_keep_their_samples(self) -> None:
        from parsec_python.Pseudopotential.fhipp import near_origin_ratio

        path = self.directory / "X_FHIPP.DAT"
        # a C/r term of 4e-8 Ha bohr dominates the near-origin structure (ratio about 30) and would move the first
        # sample by about 1e-4 Ry under a refit: the samples of the file are kept, the continuation passes through the
        # first one
        radii = write_fhipp(path, coulomb=4.0e-8)
        off, auto = self.read(path, "off"), self.read(path, "auto")
        self.assertGreater(near_origin_ratio(off.radii, off.channel_potentials[0]), 10.0)
        self.assertEqual((auto.origin_samples > 0, auto.origin_refit), (True, 0))
        added = auto.origin_samples
        for l in range(3):
            np.testing.assert_array_equal(auto.channel_potentials[l][added:], off.channel_potentials[l])
            np.testing.assert_array_equal(auto.radial_wavefunctions[l][added:], off.radial_wavefunctions[l])
            np.testing.assert_allclose(
                auto.radial_wavefunctions[l][:added],
                off.radial_wavefunctions[l][0] * (auto.radii[:added] / radii[0]) ** (l + 1), rtol=1e-14,
            )
        self.assertEqual(float(auto.channel_potentials[1][added]), float(off.channel_potentials[1][0]))
        # without the term the samples within 4 first radii are refitted; so is the PBE Si table of the examples,
        # while the CA table of the tests keeps its samples (its start moves the first sample by 5.1e-6 Ry)
        write_fhipp(path)
        self.assertGreater(self.read(path, "auto").origin_refit, 0)
        self.assertGreater(self.silicon("auto").origin_refit, 0)
        self.assertEqual(self.silicon("auto", DATA / "Si_FHIPP.DAT").origin_refit, 0)

    def test_auto_continues_the_mesh_and_keeps_the_file(self) -> None:
        from parsec_python.Pseudopotential.fhipp import FHI_ORIGIN_RADIUS, origin_sample_count, refit_window
        from parsec_python.Pseudopotential.radial_quadrature import log_mesh_integral

        path = self.directory / "X_FHIPP.DAT"
        radii = write_fhipp(path)
        off = self.read(path, "off")
        auto = self.read(path, "auto")
        added = origin_sample_count(radii[0], 1.0247)
        anchor, stop = refit_window(off.radii)
        self.assertEqual((off.origin_samples, off.origin_refit), (0, 0))
        self.assertEqual((auto.origin_samples, auto.origin_refit), (added, anchor))
        self.assertEqual(added, math.ceil(math.log(radii[0] / FHI_ORIGIN_RADIUS) / math.log(1.0247)))
        # the samples within 4 first radii are replaced; the fit takes the samples from there to 10 first radii
        self.assertGreaterEqual(off.radii[anchor], 4.0 * off.radii[0] * (1.0 - 1e-12))
        self.assertLess(off.radii[anchor - 1], 4.0 * off.radii[0])
        self.assertLessEqual(off.radii[stop - 1], 10.0 * off.radii[0] * (1.0 + 1e-12))
        self.assertLessEqual(auto.radii[0], FHI_ORIGIN_RADIUS)
        self.assertGreater(auto.radii[0] * 1.0247, FHI_ORIGIN_RADIUS)
        # every radius of the file, and its samples from the junction on bit for bit
        np.testing.assert_array_equal(auto.radii[added:], off.radii)
        for l in range(3):
            np.testing.assert_array_equal(auto.channel_potentials[l][added + anchor:], off.channel_potentials[l][anchor:])
            np.testing.assert_array_equal(
                auto.radial_wavefunctions[l][added + anchor:], off.radial_wavefunctions[l][anchor:]
            )
        # the core density of the file is kept at every sample
        np.testing.assert_array_equal(auto.core_density[added:], off.core_density)
        self.assertEqual(auto.radial_parameter_b, off.radial_parameter_b)
        self.assertAlmostEqual(auto.radial_parameter_a, auto.radii[0] / 1.0247, delta=1e-20)
        # one logarithmic mesh: the exact quadrature accepts it
        np.testing.assert_allclose(np.diff(np.log(auto.radii)), math.log(1.0247), rtol=1e-9)
        log_mesh_integral(auto.radii, auto.radial_wavefunctions[0] ** 2, auto.radial_parameter_b, 0.0)

    def test_even_fit_below_the_junction(self) -> None:
        from parsec_python.Pseudopotential.fhipp import _anchored_even_fit

        path = self.directory / "X_FHIPP.DAT"
        write_fhipp(path)
        auto = self.read(path, "auto")
        off = self.read(path, "off")
        added, anchor = auto.origin_samples, auto.origin_refit
        below = auto.radii[: added + anchor]
        for l in range(3):
            # the potentials of the synthetic file are even: the fit is exact
            exact = 2.0 * (-3.0 / np.sqrt(below * below + 0.3 * (l + 1)))
            np.testing.assert_allclose(auto.channel_potentials[l][: added + anchor], exact, rtol=0.0, atol=1e-11)
            # u = r^(l+1) exp(-r) is not A r^(l+1) (1 + beta r^2): within its first-order term over the window
            np.testing.assert_allclose(
                auto.radial_wavefunctions[l][: added + anchor], below ** (l + 1) * np.exp(-below), rtol=1e-2
            )
        # the fits pass through the junction sample (no step there), the core continuation through the first sample
        for values in (off.channel_potentials[1], off.radial_wavefunctions[1] / off.radii**2):
            fit = _anchored_even_fit(off.radii, values, anchor, anchor + 30, 3)
            self.assertEqual(float(fit(off.radii[anchor:anchor + 1])[0]), float(values[anchor]))
        core = _anchored_even_fit(off.radii, off.core_density, 0, 12, 2)
        self.assertEqual(float(core(off.radii[:1])[0]), float(off.core_density[0]))
        # 0.2 exp(-3 r) is not even: its even continuation stays within 3 r(1) of it
        inner = auto.radii[:added]
        np.testing.assert_allclose(auto.core_density[:added], 0.2 * np.exp(-3.0 * inner), rtol=5e-3)
        # the valence density of the new columns
        expected = 2.0 * auto.radial_wavefunctions[0] ** 2 + 1.0 * auto.radial_wavefunctions[1] ** 2
        np.testing.assert_allclose(
            auto.valence_density, expected / (4.0 * np.pi * auto.radii**2), rtol=1e-14
        )

    def test_splines_follow_the_table_to_the_atom(self) -> None:
        from parsec_python.Pseudopotential.radial_derivatives import spline_derivative
        from parsec_python.Pseudopotential.radial_spline import ParsecRadialSpline

        off, auto = self.silicon("off"), self.silicon("auto")
        first = float(off.radii[0])
        # V'(r)/r of the local potential is smooth from the atom to beyond the junction (Si: -123.66 to -123.42
        # Ry/bohr^2; the finer generation of FHI98PP -122.9); the padded spline of the file alone has a crater
        distances = first * np.array([0.05, 0.2, 0.5, 0.99, 1.0, 1.01, 2.0, 3.9, 4.0, 4.1, 6.0])
        auto_slope = spline_derivative(auto.local_potential_spline(2, 4), distances) / distances
        off_slope = spline_derivative(off.local_potential_spline(2, 4), distances) / distances
        self.assertLess(float(np.max(auto_slope) - np.min(auto_slope)), 1.0)
        self.assertGreater(float(np.max(off_slope) - np.min(off_slope)), 100.0)
        # the p projector follows c r down to the atom (the file's first eight samples hold its 17th value)
        distances = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3])
        for potential, low, high in ((auto, 0.9999, 1.0001), (off, 1.4, None)):
            projector, _ = potential.radial_projector(1, 2)
            spline = ParsecRadialSpline.from_positive_grid(potential.radii, projector, 4)
            ratio = spline(distances) / distances / (spline(np.array([3e-3]))[0] / 3e-3)
            if high is None:
                self.assertGreater(ratio[1], low)
            else:
                self.assertTrue(np.all((ratio > low) & (ratio < high)), ratio)

    def test_rules_next_to_the_atom(self) -> None:
        from parsec_python.Hamiltonian.hubbard import interpolate_hubbard_radial
        from parsec_python.Pseudopotential.radial_derivatives import local_potential_derivative_table
        from parsec_python.V_ion.ionic_potential import PROJECTOR_GRADIENT_FLOOR, projector_gradient_distance

        off, auto = self.silicon("off"), self.silicon("auto")
        # the projector gradients of a continued table are exact from twice its 17th sample on (about 3e-5 bohr)
        distances = np.array([0.0, 1e-5, 1e-4, 1e-3, 5e-3])
        np.testing.assert_array_equal(
            projector_gradient_distance(distances, auto), np.maximum(distances, 2.0 * auto.radii[16])
        )
        self.assertLess(2.0 * auto.radii[16], 4e-5)
        np.testing.assert_array_equal(
            projector_gradient_distance(distances, off), np.maximum(distances, PROJECTOR_GRADIENT_FLOOR)
        )
        # dV/dr of pseudo.f90's slope table: zero up to 3e-4 bohr as the file gives it, kept on the continuation
        table = local_potential_derivative_table(auto, 2)
        inner = (auto.radii > 1e-4) & (auto.radii < 3e-4)
        self.assertTrue(np.all(table[inner] != 0.0))
        np.testing.assert_array_equal(local_potential_derivative_table(off, 2)[off.radii <= 3e-4], 0.0)
        # the LDA+U lookup takes the interval of the radii of a continued table (the index of a Martins mesh picks
        # the one below for a share a/(b r) of the distances: 9.8 % in 0.05 to 1.5 bohr for the file's carbon)
        distances = np.linspace(0.05, 1.5, 2001)
        for potential, exact in ((auto, True), (off, False)):
            values, _ = interpolate_hubbard_radial(potential, distances)
            radii = np.concatenate(([0.0], potential.radii))
            radial = np.concatenate(([0.0], potential.radial_wavefunctions[2] / potential.radii))
            reference = np.interp(distances, radii, radial)
            if exact:
                np.testing.assert_allclose(values, reference, rtol=0.0, atol=1e-14)
            else:
                self.assertGreater(float(np.max(np.abs(values - reference))), 1e-9)

    def test_integrals_of_the_file(self) -> None:
        off, auto = self.silicon("off"), self.silicon("auto")
        for l in range(3):
            for name in ("norm", "kb"):
                values = []
                for potential in (off, auto):
                    wave = potential.radial_wavefunctions[l]
                    factor = 1.0 if name == "norm" else potential.channel_potentials[l] - potential.channel_potentials[2]
                    values.append(parsec_radial_integral(potential.radii, wave * wave * factor))
                self.assertAlmostEqual(values[0], values[1], delta=1e-8)

    def test_tables_that_start_at_the_origin_and_refusals(self) -> None:
        path = self.directory / "X_FHIPP.DAT"
        write_fhipp(path, start=5.0e-6)
        off, auto = self.read(path, "off"), self.read(path, "auto")
        self.assertEqual(auto.origin_samples, 0)
        np.testing.assert_array_equal(auto.radii, off.radii)
        np.testing.assert_array_equal(auto.channel_potentials[2], off.channel_potentials[2])
        np.testing.assert_array_equal(auto.valence_density, off.valence_density)
        with self.assertRaisesRegex(ValueError, "PARSEC_FHI_ORIGIN must be auto or off"):
            self.read(path, "on")
        species = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 1)}
        with mock.patch.dict(os.environ, {"PARSEC_FHI_ORIGIN": "yes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_FHI_ORIGIN"):
                load_pseudopotentials(species)


SPECIES_INPUT = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.6
States_Num: 8
Atom_Types_Num: {count}
{species}
"""


class SpeciesCursorTests(unittest.TestCase):
    """The n-th occurrence of a label goes to the n-th species that reads it."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        for name in ("Na_POTRW.DAT", "Si_FHIPP.DAT", "Mg_FHIPP.DAT", "Si_POTR.DAT"):
            (self.directory / name).write_bytes((DATA / name).read_bytes())
        (self.directory / "H_POTRE.DAT").write_bytes((DATA / "spin_ca_H_POTRE.DAT").read_bytes())

    def tearDown(self) -> None:
        self._directory.cleanup()

    def parse(self, species: str, count: int):
        path = self.directory / "parsec.in"
        path.write_text(SPECIES_INPUT.format(species=species, count=count), encoding="utf-8")
        return parse_parsec_input(path)

    @staticmethod
    def block(symbol: str, lines: str, position: str = "0 0 0") -> str:
        return f"Atom_Type: {symbol}\n{lines}\nbegin Atom_Coord\n{position}\nend Atom_Coord\n"

    def test_martins_wang_reads_no_local_component(self) -> None:
        # tests/triclinic: Na (martins_wang) and a second species with the
        # only Local_Component line.
        translation = self.parse(
            self.block("Na", "Pseudopotential_Format: Martins_Wang\nCore_Cutoff_Radius: 3.1")
            + self.block("H", "Local_Component: s", "0 0 3"),
            2,
        )
        species = translation.problem.pseudopotentials
        self.assertEqual(species["Na"].pseudopotential_format, "martins_wang")
        self.assertEqual(species["Na"].local_angular_momentum, 1)
        self.assertEqual(species["Na"].core_cutoff_radius, 3.1)
        self.assertIsNone(species["Na"].number_of_channels)
        self.assertEqual(species["H"].local_angular_momentum, 0)
        self.assertEqual(species["Na"].path.name, "Na_POTRW.DAT")
        # A Local_Component under the Martins-Wang species would go to H.
        with self.assertRaisesRegex(ParsecInputError, "remove the lines of the martins_wang"):
            self.parse(
                self.block("Na", "Pseudopotential_Format: martins_wang\nLocal_Component: d\nCore_Cutoff_Radius: 3.1")
                + self.block("H", "Local_Component: s", "0 0 3"),
                2,
            )
        with self.assertRaisesRegex(ParsecInputError, "needs Core_Cutoff_Radius"):
            self.parse(self.block("Na", "Pseudopotential_Format: martins_wang"), 1)

    def test_cursors_of_core_radius_orbitals_and_core_correction(self) -> None:
        # Core_Cutoff_Radius is read by old, wang and fhipp; Electron_Per_Orbital
        # by old and fhipp; Nonlinear_Core_Correction by fhipp only.
        species = (
            # Pseudopotential_Format is read for every species: without its
            # own line H would take the first one, fhipp, as in PARSEC.
            # Potential_Num is read by fhipp and old (PARSEC 1.5,
            # usrinputfile.F90:3244-3248): H and Na take none.
            self.block(
                "H",
                "Pseudopotential_Format: martins_new\nLocal_Component: s",
                "0 0 4",
            )
            + self.block(
                "Mg",
                "Pseudopotential_Format: fhipp\nLocal_Component: d\nPotential_Num: 4\n"
                "Core_Cutoff_Radius: 2.4\nNonlinear_Core_Correction: .true.\n"
                "begin Electron_Per_Orbital\n2 0 0 0\nend Electron_Per_Orbital",
            )
            + self.block("Na", "Pseudopotential_Format: martins_wang\nCore_Cutoff_Radius: 3.1", "0 0 -4")
            + self.block(
                "Si",
                "Pseudopotential_Format: martins_old\nLocal_Component: p\nPotential_Num: 2\n"
                "Core_Cutoff_Radius: 2.5\n"
                "begin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital",
                "4 0 0",
            )
        )
        translation = self.parse(species, 4)
        result = translation.problem.pseudopotentials
        self.assertEqual(result["Mg"].core_cutoff_radius, 2.4)
        self.assertTrue(result["Mg"].nonlinear_core_correction)
        self.assertEqual(result["Mg"].electrons_per_orbital, (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(result["Mg"].local_angular_momentum, 2)
        self.assertEqual(result["Na"].core_cutoff_radius, 3.1)
        self.assertEqual(result["Si"].core_cutoff_radius, 2.5)
        self.assertEqual(result["Si"].electrons_per_orbital, (2.0, 2.0))
        self.assertEqual(result["Si"].local_angular_momentum, 1)
        self.assertFalse(result["Si"].nonlinear_core_correction)
        loaded = load_pseudopotentials(result)
        self.assertTrue(loaded["Mg"].has_nonlinear_core_correction)
        self.assertEqual(loaded["Mg"].ionic_charge, 2.0)
        # A second Nonlinear_Core_Correction has no fhipp species to read it.
        with self.assertRaisesRegex(ParsecInputError, "nonlinear_core_correction occurs 2 times"):
            self.parse(
                species.replace("Potential_Num: 2\n", "Potential_Num: 2\nNonlinear_Core_Correction: true\n"),
                4,
            )
        # The layout of the PARSEC before 1.5 (a Potential_Num for every species) is refused.
        with self.assertRaisesRegex(ParsecInputError, "Potential_Num occurs 3 times"):
            self.parse(species.replace("Local_Component: s", "Local_Component: s\nPotential_Num: 1"), 4)

    def test_refusals(self) -> None:
        fhipp = (
            "Pseudopotential_Format: fhipp\nLocal_Component: d\nPotential_Num: 3\n"
            "Core_Cutoff_Radius: 2.5\nbegin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital"
        )
        # usrinputfile.F90 error 135.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "SO_PSP is not supported"):
            self.parse(self.block("Si", fhipp + "\nSO_PSP: .true."), 1)
        # usrinputfile.F90 error 234.
        with self.assertRaisesRegex(ParsecInputError, "a fhipp file holds no valence"):
            self.parse(self.block("Si", fhipp + "\nRead_VCD: .true."), 1)
        with self.assertRaisesRegex(ParsecInputError, "needs an Electron_Per_Orbital block"):
            self.parse(
                self.block("Si", fhipp.replace("begin Electron_Per_Orbital\n2 2 0\nend Electron_Per_Orbital", "")),
                1,
            )
        # usrinputfile.F90 error 134.
        with self.assertRaisesRegex(ParsecInputError, "unknown Pseudopotential_Format"):
            self.parse(self.block("Si", "Pseudopotential_Format: fhi\nLocal_Component: d"), 1)
        translation = self.parse(self.block("Si", fhipp), 1)
        self.assertFalse(translation.problem.pseudopotentials["Si"].nonlinear_core_correction)
        self.assertEqual(translation.problem.pseudopotentials["Si"].path.name, "Si_FHIPP.DAT")


if __name__ == "__main__":
    unittest.main()
