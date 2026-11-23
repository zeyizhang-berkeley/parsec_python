"""Double grid (``Double_Grid_Order``), ``Old_Interpolation_Format`` and the
core-density check of ``corecd.f90``.

Fortran PARSEC gives the same energies and forces as the readable solver for
H2O with ``Double_Grid_Order`` 2 and 3 and with ``Old_Interpolation_Format``
(the report of the parity round); these tests pin
the pieces: the weights, the reduction to the ordinary projectors for
``N = 1``, the coarse points of the double grid (the sizes Fortran reports)
and the inputs and output lines.
"""

from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Forces.nonlocal_force import (
    build_nonlocal_projector_gradients,
    double_grid_projector_gradients,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import ParsecInputError, UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Pseudopotential.radial_quadrature import parsec_radial_integral
from parsec_python.Pseudopotential.radial_derivatives import (
    evaluate_local_potential_derivative,
    local_potential_derivative_table,
)
from parsec_python.V_ion import build_nonlocal_projectors, check_core_density, load_pseudopotentials
from parsec_python.V_ion.double_grid import (
    NOTE_RATIO,
    build_double_grid_projectors,
    double_grid_note_lines,
    double_grid_points,
    double_grid_support,
    double_grid_weights,
    orbital_projection_ratios,
)
from parsec_python.V_ion.ionic_potential import real_spherical_harmonics
from parsec_python.acceleration.cli import _reference_only_features
from parsec_python.models import Atom, GridSettings, SpeciesPotential

DATA = Path(__file__).resolve().parents[1] / "data"
ANGSTROM = 1.0 / 0.529177


def water():
    species = {
        "O": SpeciesPotential(DATA / "spin_ca_O_POTRE.DAT", 1),
        "H": SpeciesPotential(DATA / "spin_ca_H_POTRE.DAT", 0),
    }
    atoms = [
        Atom("O", np.array([0.0, 0.11630, 0.0])),
        Atom("H", np.array([0.0, -0.92304, 1.79524])),
        Atom("H", np.array([0.0, -0.92304, -1.79524])),
    ]
    return species, atoms, load_pseudopotentials(species)


class WeightTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        for order in range(1, 6):
            offsets, weights = double_grid_weights(order)
            np.testing.assert_array_equal(offsets, np.arange(-(order - 1), order))
            np.testing.assert_allclose(weights, (1.0 - np.abs(offsets) / order) / order, rtol=1e-15)
            self.assertAlmostEqual(weights.sum(), 1.0, delta=1e-15)
            displacements, weights3 = double_grid_points(order, 0.6)
            self.assertEqual(displacements.shape, ((2 * order - 1) ** 3, 3))
            self.assertAlmostEqual(weights3.sum(), 1.0, delta=1e-14)
            self.assertAlmostEqual(np.abs(displacements).max(), 0.6 * (order - 1) / order, delta=1e-15)
        offsets, weights = double_grid_weights(1)
        np.testing.assert_array_equal(weights, [1.0])
        with self.assertRaises(ValueError):
            double_grid_weights(0)

    def test_weights_interpolate_linear_functions_exactly(self) -> None:
        # The weights are those of linear interpolation from the coarse grid,
        # so the average of a linear function is its value at the centre.
        displacements, weights = double_grid_points(3, 0.5)
        values = 2.0 + displacements @ np.array([0.3, -1.1, 0.7])
        self.assertAlmostEqual(float(weights @ values), 2.0, delta=1e-14)


class ProjectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.species, cls.atoms, cls.potentials = water()
        cls.settings = GridSettings(spacing=0.3 * ANGSTROM, radius=4.0 * ANGSTROM, expansion_order=8)

    def test_order_one_reproduces_the_projectors_exactly(self) -> None:
        grid = build_cluster_grid(self.settings)
        reference = build_nonlocal_projectors(grid, self.atoms, self.potentials, self.species)
        one = build_double_grid_projectors(grid, self.atoms, self.potentials, self.species, order=1)
        self.assertEqual(reference.labels, one.labels)
        np.testing.assert_array_equal(reference.signs, one.signs)
        self.assertEqual((reference.projectors != one.projectors).nnz, 0)
        gradients = build_nonlocal_projector_gradients(grid, self.atoms, self.potentials, self.species)
        gradients_one = double_grid_projector_gradients(
            grid, self.atoms, self.potentials, self.species, order=1
        )
        for left, right in zip(gradients.components, gradients_one.components):
            self.assertEqual((left != right).nnz, 0)

    def test_coarse_points_of_the_double_grid(self) -> None:
        # A coarse point carries the projector when any of its fine points
        # lies within the projector radius (nonloc.F90:317-406).  With this
        # rule the readable solver gives the block sizes Fortran reports for
        # the O of the H2O parity run (72, 160 and 168 points for N = 1, 2, 3
        # at h = 0.3 A; that run uses another O file).
        grid = build_cluster_grid(replace(self.settings, radius=5.5 * ANGSTROM))
        oxygen = read_parsec_pseudopotential(DATA / "spin_ca_O_POTRE.DAT")
        radius = oxygen.projector_support_radius
        position = self.atoms[0].position
        for order in (1, 2, 3):
            support = double_grid_support(grid.coordinates, position, radius, order, grid.spacing)
            displacements, _ = double_grid_points(order, grid.spacing)
            brute = [
                row
                for row in range(grid.size)
                if np.any(
                    np.linalg.norm(grid.coordinates[row] + displacements - position, axis=1) <= radius
                )
            ]
            np.testing.assert_array_equal(support.rows, brute)
            # Coarse points whose fine points all lie inside keep weight one.
            interior = (
                np.linalg.norm(grid.coordinates[support.rows] - position, axis=1)
                < radius - np.sqrt(3.0) * grid.spacing
            )
            self.assertTrue(np.any(interior))
            np.testing.assert_allclose(support.weights.sum(axis=1)[interior], 1.0, rtol=1e-12)

    def test_double_grid_is_the_fine_grid_average(self) -> None:
        grid = build_cluster_grid(replace(self.settings, double_grid_order=3))
        operator = build_nonlocal_projectors(grid, self.atoms[:1], self.potentials, self.species)
        oxygen = self.potentials["O"]
        table, _ = oxygen.radial_projector(0, 1)
        # Column 0 is the s projector of O.
        column = operator.projectors[:, 0].toarray().ravel()
        row = int(np.argmax(np.abs(column)))
        displacements, weights = double_grid_points(3, grid.spacing)
        fine = grid.coordinates[row] + displacements - self.atoms[0].position
        distance = np.linalg.norm(fine, axis=1)
        inside = distance <= oxygen.projector_support_radius
        value = np.sum(
            weights[inside]
            * np.interp(distance[inside], oxygen.radii, table, left=table[0])
            * 0.28209479177387814
        ) * np.sqrt(grid.volume_element)
        self.assertAlmostEqual(column[row], value, delta=1e-13 * abs(value))
        gradients = build_nonlocal_projector_gradients(grid, self.atoms[:1], self.potentials, self.species)
        self.assertEqual(gradients.labels, operator.labels)


class InputAndOutputTests(unittest.TestCase):
    INPUT = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 4.0 ang
Grid_Spacing: 0.3 ang
Expansion_Order: 8
States_Num: 6
{extra}
Atom_Types_Num: 1
Atom_Type: O
Local_Component: p
begin Atom_Coord
0 0 0
end Atom_Coord
"""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        (self.directory / "O_POTRE.DAT").write_bytes((DATA / "spin_ca_O_POTRE.DAT").read_bytes())

    def tearDown(self) -> None:
        self._directory.cleanup()

    def parse(self, extra: str):
        path = self.directory / "parsec.in"
        path.write_text(self.INPUT.format(extra=extra), encoding="utf-8")
        return parse_parsec_input(path)

    def header(self, translation) -> str:
        from parsec_python.Output.parsec_output import ParsecTextReporter

        stream = io.StringIO()
        ParsecTextReporter(stream.write, translation).header()
        return stream.getvalue()

    def test_double_grid_order(self) -> None:
        default = self.parse("")
        self.assertEqual(default.problem.grid.double_grid_order, 1)
        self.assertNotIn("double_grid_order", repr(default.problem.grid))
        self.assertEqual(_reference_only_features(default, accelerated_backend_selected=False), [])
        text = self.header(default)
        self.assertIn(" Order of double grid is    1\n", text)
        self.assertNotIn("DOUBLE GRID WILL", text)
        three = self.parse("Double_Grid_Order: 3")
        self.assertEqual(three.problem.grid.double_grid_order, 3)
        self.assertIn("double_grid_order=3", repr(three.problem.grid))
        text = self.header(three)
        self.assertIn(
            " Order of double grid is    3\n WARNING! DOUBLE GRID WILL MODIFY FORCES AND \n\nTOTAL ENERGY.\n\n",
            text,
        )
        # The note on the projections the double grid loses (h = 0.567 bohr):
        # the s channel of O, after the warning; none for order one.
        self.assertIn(
            "TOTAL ENERGY.\n\n NOTE: Double_Grid_Order 3 shrinks the projections of these pseudo-orbitals",
            text,
        )
        self.assertRegex(text, r"\n       O l=0: 0\.9[0-8]\d of the radial value \(order 1: ")
        self.assertNotIn("NOTE: Double_Grid_Order", self.header(default))
        # The accelerated builders average the projectors of a cluster too
        # (acceleration/tests/test_pseudopotential_setup.py).
        self.assertEqual(_reference_only_features(three, accelerated_backend_selected=False), [])
        # usrinputfile.F90 error 113.
        with self.assertRaisesRegex(ParsecInputError, "greater than zero"):
            self.parse("Double_Grid_Order: 0")
        # SO_PSP with the double grid is implemented (test_misc_parity).

    def test_old_interpolation_format(self) -> None:
        default = self.parse("")
        self.assertFalse(default.problem.pseudopotentials["O"].old_interpolation)
        self.assertNotIn("old_interpolation", repr(default.problem.pseudopotentials["O"]))
        self.assertNotIn("interpolation of pseudopot", self.header(default))
        old = self.parse("Old_Interpolation_Format: .true.")
        self.assertTrue(old.problem.pseudopotentials["O"].old_interpolation)
        self.assertIn("\n\n Using old (direct) interpolation of pseudopot. !\n", self.header(old))
        # The native and CuPy local builders interpolate V directly too.
        self.assertEqual(_reference_only_features(old, accelerated_backend_selected=False), [])


class ProjectionNoteTests(unittest.TestCase):
    """The projections of a species' pseudo-orbitals behind the double-grid NOTE of the report."""

    def test_ratios_use_the_projectors_of_the_run(self) -> None:
        # The projection ratio equals that of the double-grid projectors of
        # a cluster grid with the atom on a grid point.
        species, _, potentials = water()
        oxygen, specification = potentials["O"], species["O"]
        spacing = 0.3 * ANGSTROM
        for order in (1, 3):
            channels = orbital_projection_ratios(
                oxygen, specification, order, spacing, GridSettings(spacing, 1.0).stencil_half_width
            )
            self.assertEqual([channel.angular_momentum for channel in channels], [0, 2])
            grid = build_cluster_grid(
                GridSettings(spacing=spacing, radius=5.5, expansion_order=12, shift=(0.0, 0.0, 0.0),
                             double_grid_order=order)
            )
            operator = build_nonlocal_projectors(grid, [Atom("O", np.zeros(3))], potentials, species)
            columns = operator.projectors.toarray()
            distance = np.linalg.norm(grid.coordinates, axis=1)
            for channel in channels:
                l = channel.angular_momentum
                table, sign = oxygen.radial_projector(l, 1)
                wave = oxygen.radial_wavefunctions[l]
                radial = parsec_radial_integral(oxygen.radii, table * wave * oxygen.radii)
                orbital = np.interp(distance, oxygen.radii, wave / oxygen.radii, right=0.0)[:, None] * (
                    real_spherical_harmonics(l, grid.coordinates, center_radius=float(oxygen.radii[0]))
                )
                picked = [index for index, label in enumerate(operator.labels) if label[1] == l]
                projections = np.einsum("ph,ph->h", columns[:, picked], orbital) * np.sqrt(grid.volume_element)
                self.assertAlmostEqual(channel.ratio, float(np.mean(projections)) / radial, delta=1e-12)
                self.assertAlmostEqual(channel.kb_energy, sign * radial * radial, delta=1e-14)
                if order == 1:
                    self.assertEqual(channel.ratio, channel.sampled_ratio)

    def test_silver_d_channel(self) -> None:
        silver = {"Ag": SpeciesPotential(DATA / "Ag_POTRE.DAT", 0, use_spline=True)}
        potential = load_pseudopotentials(silver)["Ag"]
        width = GridSettings(0.45, 1.0, expansion_order=8).stencil_half_width
        channels = {
            channel.angular_momentum: channel
            for channel in orbital_projection_ratios(potential, silver["Ag"], 2, 0.45, width)
        }
        self.assertEqual(sorted(channels), [1, 2])
        d = channels[2]
        self.assertAlmostEqual(d.ratio, 0.876, delta=2e-3)
        self.assertAlmostEqual(d.sampled_ratio, 1.0, delta=2e-3)
        self.assertEqual(d.occupation, 10.0)
        self.assertLess(d.kb_energy, -7.0)
        self.assertAlmostEqual(d.energy_change, d.kb_energy * (d.ratio**2 - 1.0), delta=1e-15)
        self.assertGreater(d.energy_change, 1.5)
        self.assertGreater(channels[1].ratio, NOTE_RATIO)

    def test_note_lines(self) -> None:
        silver = {"Ag": SpeciesPotential(DATA / "Ag_POTRE.DAT", 0, use_spline=True)}
        atoms = [Atom("Ag", np.zeros(3)), Atom("Ag", np.array([4.8, 0.0, 0.0]))]
        problem = SimpleNamespace(
            grid=GridSettings(spacing=0.45, radius=10.0, expansion_order=8, double_grid_order=2),
            pseudopotentials=silver,
            atoms=atoms,
        )
        lines = double_grid_note_lines(problem)
        self.assertEqual(lines[0], " NOTE: Double_Grid_Order 2 shrinks the projections of these pseudo-orbitals onto their own")
        self.assertEqual(len(lines), 5)
        self.assertRegex(
            lines[3],
            r"^       Ag l=2: 0\.87\d of the radial value \(order 1: 0\.99\d\), \+1\.\d{4} Ry per electron, "
            r"\+3\d\.\d{3} Ry for 2 atoms with 10\.00 electrons each$",
        )
        self.assertEqual(lines[4], "")
        problem.grid = GridSettings(spacing=0.45, radius=10.0, expansion_order=8)
        self.assertEqual(double_grid_note_lines(problem), ())
        # a failure of the estimate prints nothing
        problem.grid = GridSettings(spacing=0.45, radius=10.0, expansion_order=8, double_grid_order=2)
        problem.pseudopotentials = {"Ag": SpeciesPotential(DATA / "missing_POTRE.DAT", 0)}
        self.assertEqual(double_grid_note_lines(problem), ())


class OldInterpolationTests(unittest.TestCase):
    def test_direct_interpolation_of_the_local_potential(self) -> None:
        potential = read_parsec_pseudopotential(DATA / "spin_ca_O_POTRE.DAT")
        values = potential.channel_potentials[1]
        radii = potential.radii
        samples = np.concatenate(
            ([0.0, 0.5 * radii[0]], 0.5 * (radii[100:110] + radii[101:111]), [radii[-2], 2 * radii[-1]])
        )
        direct = potential.local_potential(samples, 1, direct=True)
        inside = samples < potential.interpolation_cutoff
        np.testing.assert_array_equal(
            direct[inside], np.interp(samples[inside], radii, values, left=values[0])
        )
        np.testing.assert_array_equal(direct[~inside], -2.0 * potential.ionic_charge / samples[~inside])
        default = potential.local_potential(samples, 1)
        np.testing.assert_array_equal(default[:2], values[0])
        self.assertTrue(np.all(np.abs(direct[2:12] - default[2:12]) > 0.0))
        # The spline takes precedence, as in ionpot.f90.
        np.testing.assert_array_equal(
            potential.local_potential(samples, 1, use_spline=True, spline_padding_width=4, direct=True),
            potential.local_potential(samples, 1, use_spline=True, spline_padding_width=4),
        )

    def test_direct_interpolation_of_the_slope(self) -> None:
        potential = read_parsec_pseudopotential(DATA / "spin_ca_O_POTRE.DAT")
        table = local_potential_derivative_table(potential, 1)
        radii = potential.radii
        samples = 0.5 * (radii[200:210] + radii[201:211])
        direct = evaluate_local_potential_derivative(potential, samples, 1, table=table, direct=True)
        np.testing.assert_array_equal(direct, np.interp(samples, radii, table))
        default = evaluate_local_potential_derivative(potential, samples, 1, table=table)
        np.testing.assert_array_equal(
            default, np.interp(samples, radii, table * radii * radii) / (samples * samples)
        )


class CoreRadiusGuardTests(unittest.TestCase):
    """The check of a substituted projector builder against ``Core_Cutoff_Radius`` (FHI and old Martins species) with
    the double grid: a coarse point carries the projector when one of its fine points lies within the radius."""

    def test_double_grid_reach(self) -> None:
        from parsec_python.SCF.single_point import _require_core_radius_support

        table = Path(__file__).resolve().parents[4] / "examples" / "3d_hybrid_functionals" / "C_FHIPP.DAT"
        species = {
            "C": SpeciesPotential(
                table, 2, use_spline=True, pseudopotential_format="fhipp", number_of_channels=3,
                electrons_per_orbital=(2.0, 2.0, 0.0), core_cutoff_radius=1.6,
            )
        }
        potentials = load_pseudopotentials(species, xc_functional="pbe", check_correlation=False)
        atoms = [Atom("C", np.array([0.013, -0.021, 0.007]))]
        settings = GridSettings(spacing=0.25, radius=4.0, expansion_order=8)
        fine = build_cluster_grid(replace(settings, double_grid_order=3))
        operator = build_nonlocal_projectors(fine, atoms, potentials, species)
        rows = operator.projectors.indices
        reach = float(np.max(np.linalg.norm(fine.coordinates[rows] - atoms[0].position, axis=1)))
        self.assertGreater(reach, potentials["C"].projector_support_radius)
        _require_core_radius_support(operator, fine, atoms, potentials)
        # the same operator on a grid without the double grid reaches too far
        coarse = build_cluster_grid(settings)
        with self.assertRaisesRegex(RuntimeError, "beyond the core radius"):
            _require_core_radius_support(operator, coarse, atoms, potentials)


class CoreDensityCheckTests(unittest.TestCase):
    def test_corecd_sign_check(self) -> None:
        values = np.array([0.0, 0.2, 1.0])
        self.assertIs(check_core_density(values), values)
        clamped = check_core_density(np.array([-0.05, 0.2, -1e-9]))
        np.testing.assert_array_equal(clamped, [0.0, 0.2, 0.0])
        # corecd.f90 error 381.
        with self.assertRaisesRegex(ValueError, "error 381"):
            check_core_density(np.array([-0.2, 0.2]))


if __name__ == "__main__":
    unittest.main()
