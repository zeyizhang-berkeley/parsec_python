"""The route of a mesh of time-reversal invariant k-points (``SCF/kpoint_supercell.py``, ``PARSEC_KPOINT_SUPERCELL``).

The default ``auto`` solves a single-point SCF of such a mesh as the Gamma point of its supercell where the
estimated difference of the two kinetic forms (the gradient form of the k-points, the Bloch form of the supercell)
is small.  These tests check the symbols the estimate is made of against the operators of the code, the estimate
against SCF energies of the two forms, the decisions and their notes, the grid of a cell with unequal steps, and
the ``parsec.dat`` of the cell that the supercell writes.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian, build_gradient
from parsec_python.SCF import kpoint_supercell as route
from parsec_python.models import PeriodicCell, PeriodicGridSettings

SILICON = Path(__file__).resolve().parents[4] / "examples" / "3d_Si"


def _silicon_input(directory: Path, *, spacing: float, order: int, sides=(10.26, 10.26, 10.26), extra=()) -> Path:
    """``examples/3d_Si`` with the Gamma-centred 2 x 2 x 2 mesh (no shift block)."""
    changes = {"Grid_Spacing": f"{spacing}", "Expansion_Order": f"{order}", "Convergence_Criterion": "1.0e-9",
               "Diag_Tolerance": "1.0e-10"}
    lines = []
    for line in (SILICON / "parsec.in").read_text().splitlines():
        keyword = line.split(":", 1)[0].strip()
        if ":" in line and keyword in changes:
            line = f" {keyword}: {changes[keyword]}"
        elif line.strip().startswith("10.26  10.26  10.26"):
            line = "    " + "  ".join(f"{value}" for value in sides)
        elif line.strip().lower().startswith("output_level"):
            line = " Output_Level: 1"
        lines.append(line)
        if keyword == "Correlation_Type":
            lines += [" Kpoint_Method: mp", " begin Monkhorst_Pack_Grid", " 2 2 2", " end Monkhorst_Pack_Grid",
                      *extra]
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(SILICON / "Si_POTRE.DAT", directory)
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    return directory / "parsec.in"


def _translation(path: Path):
    from parsec_python.Input import parse_parsec_input

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path)


class _Folder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp(prefix="parsec_trim_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.folder, ignore_errors=True)


class KineticSymbolTests(unittest.TestCase):
    """The symbols of the estimate are those of the operators of the k-points and of the supercell: a plane wave
    ``e^{i(q+k)x}`` (the periodic factor ``u = e^{iqx}``) has ``s(q) + 2k d(q) + k^2`` under the gradient form and
    ``s(q + k)`` under the Bloch form."""

    def _check(self, sides, counts, order):
        cell = PeriodicCell(np.diag(sides))
        settings = PeriodicGridSettings(spacing=float(min(np.asarray(sides) / counts)) * (1.0 - 1e-9),
                                        expansion_order=order, shift=(0.5,) * 3, points_rule="fortran")
        grid = build_periodic_grid(cell, settings)
        np.testing.assert_array_equal(grid.points_per_axis, counts)
        steps = np.asarray(sides) / np.asarray(counts)
        laplacian = build_negative_laplacian(grid)
        gradient = build_gradient(grid)
        coordinates = np.asarray(grid.coordinates)
        for axis in range(3):
            k = np.zeros(3)
            k[axis] = np.pi / sides[axis]
            s, d = route._symbols(order, float(steps[axis]))
            for n in (0, 1, -2, 3):
                q = 2.0 * np.pi * n / sides[axis]
                u = np.exp(1j * q * coordinates[:, axis])
                norm = (u.conj() @ u).real
                gradient_form = laplacian @ u - 2j * sum(k[c] * (gradient[c] @ u) for c in range(3)) + (k @ k) * u
                expected = float(s(np.array(q)) + 2 * k[axis] * d(np.array(q)) + k[axis] ** 2)
                self.assertAlmostEqual((u.conj() @ gradient_form).real / norm, expected,
                                       delta=1.0e-10 * max(1.0, abs(expected)), msg=f"gradient axis {axis} n {n}")
                if grid.is_orthorhombic:
                    bloch = build_bloch_negative_laplacian(grid, k, "u") @ u
                    expected = float(s(np.array(q + k[axis])))
                    self.assertAlmostEqual((u.conj() @ bloch).real / norm, expected,
                                           delta=1.0e-10 * max(1.0, abs(expected)), msg=f"bloch axis {axis} n {n}")

    def test_a_cube(self) -> None:
        self._check((6.0, 6.0, 6.0), (10, 10, 10), 8)

    def test_unequal_steps(self) -> None:
        self._check((6.0, 6.0, 7.0), (10, 10, 12), 6)


class EstimateTests(_Folder):
    def test_the_estimate_follows_the_scf_of_the_two_forms(self) -> None:
        # silicon of examples/3d_Si, Expansion_Order 8 at 0.855 bohr (12 points per edge), the reduced 2 x 2 x 2 mesh:
        # E(gradient) - E(Bloch) per atom against the estimate (1.18e-3 measured, 9.9e-4 estimated)
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        translation = _translation(_silicon_input(self.folder / "scf", spacing=0.855, order=8))
        problem = translation.problem
        estimate = route.kinetic_form_difference_estimate(problem)
        k_points, weights = kpoint_sampling(problem)[:2]
        energies = {}
        for form in ("gradient", "bloch"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": form}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_scf_kpoints(prepare_periodic_single_point(problem), k_points, weights)
            self.assertTrue(result.converged, form)
            energies[form] = result.energies.total / len(problem.atoms)
        measured = energies["gradient"] - energies["bloch"]
        self.assertGreater(measured, 0.0)
        self.assertGreater(estimate, 0.0)
        self.assertLess(abs(measured / estimate - 1.0), 0.5, f"measured {measured}, estimate {estimate}")
        self.assertLessEqual(measured, route.ESTIMATE_SAFETY * estimate)

    def test_zero_where_the_kpoints_take_the_bloch_form(self) -> None:
        problem = _translation(_silicon_input(self.folder / "bloch", spacing=0.64125, order=12)).problem
        self.assertGreater(route.kinetic_form_difference_estimate(problem), 0.0)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "bloch"}):
            self.assertEqual(route.kinetic_form_difference_estimate(problem), 0.0)

    def test_it_falls_with_the_step_and_the_order(self) -> None:
        values = [
            route.kinetic_form_difference_estimate(
                _translation(_silicon_input(self.folder / f"s{points}_{order}", spacing=10.26 / points * (1 - 1e-9),
                                            order=order)).problem)
            for points, order in ((14, 8), (16, 8), (20, 8), (16, 12))
        ]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertGreater(values[1], values[3])
        self.assertGreater(values[2], 0.0)


class RouteTests(_Folder):
    def setUp(self) -> None:
        self.patch = mock.patch.dict(os.environ)
        self.patch.start()
        for name in (route.TRIM_SWITCH, route.TOLERANCE_SWITCH, route.MINIMUM_SWITCH, route.GPUS_SWITCH,
                     route.BLOCH_SWITCH, "PARSEC_KPOINT_KINETIC"):
            os.environ.pop(name, None)
        # the 8-atom cubes here are below the floor of the default route: every size, except where a test says
        os.environ[route.MINIMUM_SWITCH] = "0"

    def tearDown(self) -> None:
        self.patch.stop()

    def test_the_switches(self) -> None:
        os.environ.pop(route.MINIMUM_SWITCH)
        self.assertEqual(route.kpoint_supercell_setting(), "auto")
        self.assertAlmostEqual(route.kpoint_supercell_tolerance(), 0.01 / 13.605693122994)
        self.assertEqual(route.kpoint_supercell_minimum(), 2 ** 24)
        self.assertEqual(route.kpoint_supercell_gpus(), 1)
        for name, value in ((route.TRIM_SWITCH, "yes"), (route.TOLERANCE_SWITCH, "-1"),
                            (route.TOLERANCE_SWITCH, "x"), (route.TOLERANCE_SWITCH, "nan"),
                            (route.MINIMUM_SWITCH, "-1"), (route.MINIMUM_SWITCH, "1e6"),
                            (route.GPUS_SWITCH, "0"), (route.GPUS_SWITCH, "two")):
            with mock.patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    route.validate_settings()
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "ON", route.TOLERANCE_SWITCH: "1e-3"}):
            self.assertEqual(route.kpoint_supercell_setting(), "on")
            self.assertEqual(route.kpoint_supercell_tolerance(), 1.0e-3)

    def test_the_decisions(self) -> None:
        translation = _translation(_silicon_input(self.folder / "route", spacing=0.64125, order=12))
        taken = route.trim_supercell_route(translation, device="gpu")
        self.assertTrue(taken.taken)
        self.assertTrue(taken.cell_files)
        self.assertEqual(taken.repeats, (2, 2, 2))
        self.assertEqual(len(taken.problem.atoms), 64)
        self.assertIn("supercell of 8 cells", taken.note)
        self.assertGreater(taken.estimate, 0.0)
        # what keeps the k-points under auto, each with its note
        for changed, word in (
            (replace(translation, compute_forces=True), "forces"),
            (replace(translation, restart_run=True), "Restart_Run"),
            (replace(translation, problem=replace(translation.problem, scf=replace(translation.problem.scf,
                                                                                     spin_polarized=True))),
             "spin polarization"),
        ):
            decision = route.trim_supercell_route(changed, device="gpu")
            self.assertFalse(decision.taken, word)
            self.assertIn(word, decision.note)
        # the floor of the default route: 8 k-points x 4096 points x 24 states = 786,432 entries
        with mock.patch.dict(os.environ, {route.MINIMUM_SWITCH: ""}):
            small = route.trim_supercell_route(translation, device="gpu")
            self.assertFalse(small.taken)
            self.assertIn("a small cell", small.note)
        with mock.patch.dict(os.environ, {route.MINIMUM_SWITCH: "786432"}):
            self.assertTrue(route.trim_supercell_route(translation, device="gpu").taken)
        with mock.patch.dict(os.environ, {route.MINIMUM_SWITCH: "786433"}):
            self.assertFalse(route.trim_supercell_route(translation, device="gpu").taken)
        ranks = route.trim_supercell_route(translation, device="gpu", kpoint_ranks=True)
        self.assertFalse(ranks.taken)
        self.assertIn("--mpi", ranks.note)
        cpu = route.trim_supercell_route(translation, device="cpu")
        self.assertFalse(cpu.taken)
        self.assertIn("CPU", cpu.note)
        with mock.patch.dict(os.environ, {route.TOLERANCE_SWITCH: "1e-7"}):
            small = route.trim_supercell_route(translation, device="gpu")
            self.assertFalse(small.taken)
            self.assertIn("estimated difference", small.note)
        # on: wherever the supercell can be solved, the kind of run and the estimate aside (its parsec.dat its own)
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "on", route.TOLERANCE_SWITCH: "1e-7"}):
            forced = route.trim_supercell_route(replace(translation, compute_forces=True), device="cpu",
                                                kpoint_ranks=True)
            self.assertTrue(forced.taken)
            self.assertFalse(forced.cell_files)
            self.assertIsNotNone(route.supercell_of_trim_mesh(translation.problem))
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "off"}):
            off = route.trim_supercell_route(translation, device="gpu")
            self.assertFalse(off.taken)
            self.assertIsNone(off.note)
            self.assertIsNone(route.supercell_of_trim_mesh(translation.problem))
        # a mesh that is not of time-reversal invariant points: nothing to say
        shifted = replace(translation, problem=replace(translation.problem, monkhorst_pack_dimensions=(3, 3, 3)))
        self.assertIsNone(route.trim_supercell_route(shifted, device="gpu").note)

    def test_a_charged_cell(self) -> None:
        # Net_Charges is the charge of one cell: the supercell of 8 cells carries 8 times as much, 8 x 31 electrons
        from parsec_python.V_ion.ionic_potential import ionic_charge

        translation = _translation(_silicon_input(self.folder / "charged", spacing=0.64125, order=12))
        problem = replace(translation.problem, scf=replace(translation.problem.scf, net_charge=1.0,
                                                           number_of_states=None))
        decision = route.trim_supercell_route(replace(translation, problem=problem), device="gpu")
        self.assertTrue(decision.taken, decision.note)
        tables = route._load_pseudopotentials(problem)
        cell_electrons = ionic_charge(tuple(problem.atoms), tables) - 1.0
        self.assertAlmostEqual(cell_electrons, 31.0)
        self.assertEqual(decision.problem.scf.net_charge, 8.0)
        self.assertAlmostEqual(
            ionic_charge(tuple(decision.problem.atoms), tables) - decision.problem.scf.net_charge, 8 * cell_electrons
        )
        # ceil(31/2) + 6 states per k-point, eight times as many in the supercell
        self.assertEqual(decision.problem.scf.number_of_states, 8 * 22)
        # the other way (PARSEC_SUPERCELL_KPOINTS=auto): the cell of the supercell carries 1/8 of its charge
        with mock.patch.dict(os.environ, {route.BLOCH_SWITCH: "auto"}):
            back = route.kpoints_of_supercell(decision.problem)
        self.assertIsNotNone(back)
        cell, repeats = back
        self.assertEqual(repeats, (2, 2, 2))
        self.assertEqual(len(cell.atoms), 8)
        self.assertEqual(cell.scf.net_charge, 1.0)
        # a neutral cell keeps a neutral supercell
        neutral = route.trim_supercell_route(translation, device="gpu")
        self.assertEqual(neutral.problem.scf.net_charge, 0.0)

    def test_the_gpus_of_the_process_and_saved_orbitals(self) -> None:
        translation = _translation(_silicon_input(self.folder / "gpus", spacing=0.64125, order=12))
        self.assertTrue(route.trim_supercell_route(translation, device="gpu", gpus=1).taken)
        four = route.trim_supercell_route(translation, device="gpu", gpus=4)
        self.assertFalse(four.taken)
        self.assertIn("4 GPUs in one process", four.note)
        self.assertFalse(route.trim_supercell_route(translation, device="gpu", gpus=lambda: 2).taken)

        def broken():
            raise RuntimeError("no CUDA here")

        self.assertIn("could not be counted", route.trim_supercell_route(translation, device="gpu", gpus=broken).note)
        # the count is not asked where something else decides
        self.assertIn("CPU", route.trim_supercell_route(translation, device="cpu", gpus=broken).note)
        with mock.patch.dict(os.environ, {route.GPUS_SWITCH: "4"}):
            self.assertTrue(route.trim_supercell_route(translation, device="gpu", gpus=4).taken)
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "on"}):
            self.assertTrue(route.trim_supercell_route(translation, device="gpu", gpus=4).taken)
        saved = route.trim_supercell_route(translation, device="gpu", orbitals=True)
        self.assertFalse(saved.taken)
        self.assertIn("--save-wavefunctions", saved.note)
        all_states = route.trim_supercell_route(replace(translation, output_all_states=True), device="gpu")
        self.assertFalse(all_states.taken)
        self.assertIn("Output_All_States", all_states.note)
        for value, count in (("0,1", 2), ("current", 1), ("3", 1), ("0, 2, 3", 3)):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_DEVICES": value}):
                self.assertEqual(route.process_gpu_count(), count, value)

    def test_the_restarted_structure_makes_the_supercell(self) -> None:
        # the restart files stand for the atoms of the input: the supercell is made of the moved atoms
        from parsec_python.Relaxation.run import restart_structure

        folder = self.folder / "restart"
        translation = _translation(_silicon_input(folder, spacing=0.64125, order=12))
        moved = [np.asarray(atom.position, dtype=float) + (0.05 if index == 0 else 0.0)
                 for index, atom in enumerate(translation.problem.atoms)]
        # XYZ in angstrom
        lines = [str(len(moved)), "moved"] + [f"Si {x:.12f} {y:.12f} {z:.12f}"
                                              for x, y, z in np.asarray(moved) * 0.529177210903]
        (folder / "cell.xyz").write_text("\n".join(lines) + "\n")
        restarted, _, _ = restart_structure(translation, folder, folder / "cell.xyz")
        np.testing.assert_allclose(restarted.problem.atoms[0].position, moved[0], atol=1.0e-6)
        decision = route.trim_supercell_route(restarted, device="gpu")
        self.assertTrue(decision.taken, decision.note)
        self.assertEqual(len(decision.problem.atoms), 64)
        np.testing.assert_allclose(decision.problem.atoms[0].position, moved[0], atol=1.0e-6)
        self.assertNotEqual(tuple(decision.problem.atoms[0].position), tuple(translation.problem.atoms[0].position))

    def test_the_states_of_an_input_without_states_num(self) -> None:
        translation = _translation(_silicon_input(self.folder / "states", spacing=0.64125, order=12))
        problem = replace(translation.problem, scf=replace(translation.problem.scf, number_of_states=None))
        # 32 valence electrons: 16 + 6 states per k-point, eight times as many in the supercell
        self.assertEqual(route._cell_states(problem), 22)
        decision = route.trim_supercell_route(replace(translation, problem=problem), device="gpu")
        self.assertTrue(decision.taken)
        self.assertEqual(decision.problem.scf.number_of_states, 8 * 22)

    def test_the_decision_refused_by_the_gamma_route(self) -> None:
        translation = _translation(_silicon_input(self.folder / "refused", spacing=0.64125, order=12))
        decision = route.trim_supercell_route(translation, refusal=lambda supercell: "something", device="gpu")
        self.assertFalse(decision.taken)
        self.assertIn("does not solve something", decision.note)

    def test_unequal_steps(self) -> None:
        translation = _translation(_silicon_input(self.folder / "tetragonal", spacing=0.62, order=12,
                                                  sides=(10.26, 10.26, 11.286)))
        counts = route._grid_counts(translation.problem)
        np.testing.assert_array_equal(counts, (16, 16, 18))
        decision = route.trim_supercell_route(translation, device="gpu")
        self.assertTrue(decision.taken, decision.note)
        np.testing.assert_array_equal(route._grid_counts(decision.problem), 2 * counts)
        grid = build_periodic_grid(decision.problem.periodic_cell, decision.problem.grid)
        cell_grid = build_periodic_grid(translation.problem.periodic_cell, translation.problem.grid)
        np.testing.assert_allclose(np.diag(grid.step_vectors), np.diag(cell_grid.step_vectors), rtol=1e-14)


class EnergyLineTests(unittest.TestCase):
    def test_the_energies_of_one_cell(self) -> None:
        from parsec_python.models import CorrectedEnergyBreakdown, EnergyBreakdown

        energies = EnergyBreakdown(eigenvalue=8.0, hartree=16.0, integral_vxc_rho=-8.0, exchange_correlation=-24.0,
                                   electron_ion=-80.0, ion_ion=-40.0, electronic=-100.0, total=-140.0)
        cell = route.energies_per_cell(energies, 8)
        self.assertIsInstance(cell, EnergyBreakdown)
        self.assertEqual((cell.eigenvalue, cell.hartree, cell.ion_ion, cell.electronic, cell.total),
                         (1.0, 2.0, -5.0, -12.5, -17.5))
        self.assertIs(route.energies_per_cell(energies, 1), energies)
        corrected = CorrectedEnergyBreakdown(**vars(energies), dispersion=-0.8)
        self.assertAlmostEqual(route.energies_per_cell(corrected, 8).dispersion, -0.1)
        lines = route.per_cell_lines(-140.0, (2, 2, 2), 8, per_cell_report=True)
        self.assertIn("-17.50000000", lines[0])
        self.assertIn("supercell of 8 cells", lines[1])
        self.assertIn("-140.00000000", lines[1])
        # no line in the format the parsers of "Total Energy =" read: the energy lines above are the cell's
        self.assertFalse(any("Total Energy =" in line for line in lines))


class CellFileTests(_Folder):
    def test_the_parsec_dat_of_the_cell(self) -> None:
        from parsec_python.Output.parsec_dat import parsec_dat_from_result
        from parsec_python.Output.output_files import WavefunctionSelection

        translation = _translation(_silicon_input(self.folder / "file", spacing=0.64125, order=12))
        problem = translation.problem
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "auto", route.MINIMUM_SWITCH: "0"}):
            decision = route.trim_supercell_route(translation, device="gpu")
        counts = route._grid_counts(problem)
        rng = np.random.default_rng(3)
        cell_density = rng.random(tuple(counts))
        supercell_density = np.tile(cell_density, (2, 2, 2)).reshape(-1)
        levels = np.sort(rng.normal(size=8 * 24))
        result = SimpleNamespace(eigenvalues=levels, occupations=np.where(levels < 0, 2.0, 0.0),
                                 next_effective_potential=-supercell_density, density=supercell_density)
        view = route.cell_parsec_dat_view(problem, decision.repeats)
        stand_in, grid, cell_problem = view(result, None, decision.problem)
        np.testing.assert_array_equal(stand_in.density, cell_density.reshape(-1))
        np.testing.assert_array_equal(grid.points_per_axis, counts)
        self.assertIs(cell_problem, problem)
        self.assertEqual(stand_in.k_points.shape, (8, 3))
        np.testing.assert_allclose(stand_in.k_points[-1], np.pi / 10.26)
        np.testing.assert_array_equal(np.sort(stand_in.eigenvalues), levels)
        # every k-point a slice of the whole spectrum
        blocks = stand_in.eigenvalues.reshape(8, 24)
        self.assertTrue(np.all(blocks.min(axis=1) <= levels[7]))
        data, _ = parsec_dat_from_result(stand_in, grid, WavefunctionSelection(mode="list", indices=()),
                                         periodic_cell=problem.periodic_cell, monkhorst_pack_grid=(2, 2, 2))
        self.assertEqual(data.ndim, int(np.prod(counts)))
        self.assertEqual(len(data.levels[0]), 8)
        np.testing.assert_array_equal(data.densities[0], cell_density.reshape(-1))

    def test_the_result_archive_of_the_cell(self) -> None:
        from parsec_python.cli import save_result_archive
        from parsec_python.models import EnergyBreakdown
        from parsec_python.SCF.bands import load_scf_for_bands

        translation = _translation(_silicon_input(self.folder / "archive", spacing=0.64125, order=12))
        problem = translation.problem
        with mock.patch.dict(os.environ, {route.TRIM_SWITCH: "auto", route.MINIMUM_SWITCH: "0"}):
            decision = route.trim_supercell_route(translation, device="gpu")
        counts = route._grid_counts(problem)
        cell_grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        supergrid = build_periodic_grid(decision.problem.periodic_cell, decision.problem.grid)
        rng = np.random.default_rng(5)
        fields = {name: rng.random(tuple(counts)) for name in (
            "density", "core_density", "ionic_potential", "hartree_potential", "xc_potential",
            "input_effective_potential", "output_effective_potential", "next_effective_potential")}
        levels = np.sort(rng.normal(size=8 * 24))
        energies = EnergyBreakdown(eigenvalue=8.0, hartree=16.0, integral_vxc_rho=-8.0, exchange_correlation=-24.0,
                                   electron_ion=-80.0, ion_ion=-40.0, electronic=-100.0, total=-140.0)
        result = SimpleNamespace(
            converged=True, iterations=7, atoms=tuple(decision.problem.atoms), electron_count=256.0, grid=supergrid,
            eigenvalues=levels, occupations=np.where(levels < 0.0, 2.0, 0.0), fermi_level=0.0, energies=energies,
            history=[], wavefunctions=None,
            **{name: np.tile(value, (2, 2, 2)).reshape(-1) for name, value in fields.items()},
        )
        path = self.folder / "archive" / "parsec_python_results.npz"
        save_result_archive(path, route.cell_result_view(result, problem, decision.repeats))
        with np.load(path) as archive:
            np.testing.assert_allclose(archive["atom_coordinates_bohr"],
                                       [atom.position for atom in problem.atoms])
            np.testing.assert_array_equal(archive["output_effective_potential_ry"],
                                          fields["output_effective_potential"].reshape(-1))
            np.testing.assert_array_equal(archive["coordinates_bohr"], cell_grid.coordinates)
            self.assertEqual(int(archive["n_kpoints"]), 8)
            self.assertEqual(int(archive["n_states_per_kpoint"]), 24)
            self.assertEqual(float(archive["energy_total_ry"]), -17.5)
            self.assertEqual(float(archive["electron_count"]), 32.0)
            np.testing.assert_array_equal(np.sort(archive["eigenvalues_ry"]), levels)
        # --bands-from of the input of the cell reads it
        potentials, fermi, states = load_scf_for_bands(
            path, problem, SimpleNamespace(atoms=problem.atoms, grid=cell_grid), "scalar")
        np.testing.assert_array_equal(potentials["effective_potential"],
                                      fields["output_effective_potential"].reshape(-1))
        self.assertEqual(states, 24)


if __name__ == "__main__":
    unittest.main()
