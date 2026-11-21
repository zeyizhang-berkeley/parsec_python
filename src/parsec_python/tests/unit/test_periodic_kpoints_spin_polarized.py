"""Tests for k-point-sampled periodic collinear spin polarization (no
spin-orbit), LDA and PBE."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Grid import monkhorst_pack_grid
from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from parsec_python.SCF.pbc import periodic_spin_start, prepare_periodic_single_point
from parsec_python.SCF.spin_polarized import (
    SPIN_START_ENV,
    _weighted_initial_polarization,
    initchrg_spin_start,
    initchrg_start_needed,
    opposite_spin_starts,
    run_scf_spin_polarized,
    spin_start_setting,
)
from parsec_python.cli import main as cli_main
from parsec_python.tests.archives import read_archive

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
_AU_POTRE = _EXAMPLES / "Au_POTRE.DAT"
_H_POTRE = _EXAMPLES / "H_POTRE.DAT"


def _au_problem(
    *,
    potre: Path = _AU_POTRE,
    spin_polarized: bool = True,
    xc_functional: str = "ca",
    initial_polarization: float = 0.3,
    convergence: float = 5.0e-3,
    max_iterations: int = 80,
) -> SinglePointInput:
    box = 8.0
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
        pseudopotentials={
            "Au": SpeciesPotential(
                potre, 0, initial_spin_polarization=initial_polarization
            )
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=max_iterations,
            number_of_states=16,
            convergence_criterion=convergence,
            spin_polarized=spin_polarized,
            xc_functional=xc_functional,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _auh_problem(*, spin_polarized: bool) -> SinglePointInput:
    """Closed-shell AuH: its paramagnetic state is stable."""
    box = 9.0
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[
            Atom("Au", [6.0, box / 2, box / 2]),
            Atom("H", [6.0 - 2.89, box / 2, box / 2]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.0),
            "H": SpeciesPotential(_H_POTRE, 0),
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=120,
            number_of_states=16 if spin_polarized else 10,
            convergence_criterion=1.0e-4,
            spin_polarized=spin_polarized,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _relabelled_potre(directory: Path, label: str) -> Path:
    lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
    tokens = lines[0].split()
    tokens[1] = label
    lines[0] = " " + " ".join(tokens) + "  "
    path = directory / "Au_POTRE.DAT"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class KPointSpinPolarizedSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system = prepare_periodic_single_point(_au_problem())
        cls.cell = cls.system.input.periodic_cell

    def test_gamma_point_matches_the_gamma_only_spin_driver(self) -> None:
        """One k-point at Gamma must reproduce run_scf_spin_polarized, an
        independent implementation of the same physics."""
        reference = run_scf_spin_polarized(self.system)
        k_result = run_scf_kpoints_spin_polarized(
            self.system, np.zeros((1, 3)), np.ones(1)
        )
        self.assertTrue(reference.converged and k_result.converged)
        self.assertAlmostEqual(k_result.magnetic_moment, reference.magnetic_moment, delta=5e-3)
        self.assertAlmostEqual(k_result.energies.total, reference.energies.total, delta=5e-3)
        self.assertGreater(k_result.magnetic_moment, 0.5)

    def test_monkhorst_pack_grid_conserves_electrons_and_moment(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        result = run_scf_kpoints_spin_polarized(self.system, k_points, weights)
        self.assertTrue(result.converged)
        total = np.sum(result.occupations_up) + np.sum(result.occupations_down)
        self.assertAlmostEqual(
            float(total) * result.occupation_weight, result.electron_count, places=4
        )
        self.assertAlmostEqual(result.occupation_weight, 0.5)
        # Pooled over 2 k-points x 16 states per spin.
        self.assertEqual(result.eigenvalues_up.size, 32)
        self.assertEqual(result.eigenvalues_down.size, 32)
        self.assertGreater(result.magnetic_moment, 0.5)
        self.assertLess(result.magnetic_moment, 1.5)
        self.assertTrue(np.isfinite(result.energies.total))

    def test_moment_equals_integrated_spin_density(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        result = run_scf_kpoints_spin_polarized(self.system, k_points, weights)
        integrated = self.system.grid.volume_element * float(
            np.sum(result.density_up - result.density_down)
        )
        self.assertAlmostEqual(integrated, result.magnetic_moment, delta=1e-3)

    def test_closed_shell_reproduces_the_unpolarized_kpoint_driver(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        polarized = run_scf_kpoints_spin_polarized(
            prepare_periodic_single_point(_auh_problem(spin_polarized=True)),
            k_points,
            weights,
        )
        unpolarized = run_scf_kpoints(
            prepare_periodic_single_point(_auh_problem(spin_polarized=False)),
            k_points,
            weights,
        )
        self.assertTrue(polarized.converged and unpolarized.converged)
        self.assertAlmostEqual(polarized.magnetic_moment, 0.0, delta=1e-3)
        self.assertAlmostEqual(
            polarized.energies.total, unpolarized.energies.total, delta=2e-3
        )

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        # Unequal weights are accepted (Fortran's irreducible wedge,
        # test_space_group_kpoints); weights must still be normalized.
        with self.assertRaises(ValueError):
            run_scf_kpoints_spin_polarized(
                self.system,
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                np.array([0.7, 0.4]),
            )

    def test_requires_spin_polarized_settings(self) -> None:
        system = prepare_periodic_single_point(_au_problem(spin_polarized=False))
        with self.assertRaises(ValueError):
            run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1))


class KPointSpinPolarizedPBETests(unittest.TestCase):
    def test_pbe_converges_with_a_net_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            potre = _relabelled_potre(Path(tmp), "pb")
            system = prepare_periodic_single_point(
                _au_problem(potre=potre, xc_functional="pbe")
            )
            k_points, weights = monkhorst_pack_grid(system.input.periodic_cell, (2, 1, 1))
            result = run_scf_kpoints_spin_polarized(system, k_points, weights)
        self.assertTrue(result.converged)
        self.assertGreater(result.magnetic_moment, 0.5)


_CLI_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 16
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Spin_Polarization: .true.

Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 1 1
end Monkhorst_Pack_Grid

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .false.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
"""


class KPointSpinPolarizedCliTests(unittest.TestCase):
    def test_cli_runs_and_reports_the_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _relabelled_potre(directory, "ca")
            (directory / "parsec.in").write_text(_CLI_INPUT, encoding="utf-8")
            exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
            self.assertEqual(exit_code, 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            archive = read_archive(directory / "parsec_python_results.npz")
            moment = float(archive["magnetic_moment"])
            self.assertTrue(bool(archive["converged"]))
        self.assertIn("Converged magnetic moment N_up - N_down", log)
        # Au 6s^1: one unpaired electron per cell, correctly k-weighted.
        self.assertGreater(moment, 0.5)
        self.assertLess(moment, 1.5)


class _Stop(Exception):
    pass


def _first_spin_fields(test, run, system, stop):
    """The density of the first Hartree solve and the channel densities of the first XC evaluation of ``run``."""
    captured = {}
    cls = type(system)
    hartree = cls.solve_hartree

    def solve_hartree(this, density, *arguments, **keywords):
        captured.setdefault("hartree", np.array(density, dtype=float))
        return hartree(this, density, *arguments, **keywords)

    def evaluate_xc_spin_polarized(this, up, down, *arguments, **keywords):
        captured["xc"] = (np.array(up, dtype=float), np.array(down, dtype=float))
        raise stop

    with mock.patch.object(cls, "solve_hartree", solve_hartree), mock.patch.object(
        cls, "evaluate_xc_spin_polarized", evaluate_xc_spin_polarized
    ):
        with test.assertRaises(stop):
            run(system)
    return captured


def _check_first_fields(test, found, system, initchrg: bool) -> None:
    """``found`` (:func:`_first_spin_fields`) is PARSEC 1.5's start (``initchrg``) or the former one, bit for bit."""
    rho0 = system.initial_density
    if initchrg:
        np.testing.assert_array_equal(found["hartree"], rho0)
        expected = initchrg_spin_start(system)
    else:
        polarization = _weighted_initial_polarization(system)
        expected = (0.5 * (1.0 + polarization) * rho0, 0.5 * (1.0 - polarization) * rho0)
        np.testing.assert_array_equal(found["hartree"], expected[0] + expected[1])
    np.testing.assert_array_equal(found["xc"][0], expected[0])
    np.testing.assert_array_equal(found["xc"][1], expected[1])


class SpinStartTests(unittest.TestCase):
    """``PARSEC_SPIN_START``: the per-species start of initchrg.f90 (``initchrg``; under ``auto``, the default since
    2026-11-21, where the species start with values of both signs) or one polarization fraction of the whole cell
    (``weighted``, the former start) of the collinear k-point loop and of ``run_scf_spin_polarized``."""

    @classmethod
    def setUpClass(cls) -> None:
        problem = _auh_problem(spin_polarized=True)

        def with_starts(au: float, h: float):
            species = dict(problem.pseudopotentials)
            species["Au"] = SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=au)
            species["H"] = SpeciesPotential(_H_POTRE, 0, initial_spin_polarization=h)
            return prepare_periodic_single_point(replace(problem, pseudopotentials=species))

        # opposite starts on the two species (an antiferromagnetic start of two labels has the same shape)
        cls.mixed = with_starts(0.5, -0.5)
        # different values of one sign (one species written, the other at the default 0.1; or at 0)
        cls.same_sign = with_starts(0.5, 0.1)
        cls.with_zero = with_starts(0.5, 0.0)
        cls.uniform = prepare_periodic_single_point(_au_problem())

    def test_auto_takes_initchrg_only_for_values_of_both_signs(self) -> None:
        self.assertTrue(opposite_spin_starts(self.mixed))
        for system in (self.same_sign, self.with_zero, self.uniform):
            self.assertFalse(opposite_spin_starts(system))
        for setting, expected in (("auto", (True, False, False, False)), ("initchrg", (True,) * 4),
                                  ("weighted", (False,) * 4)):
            with mock.patch.dict(os.environ, {SPIN_START_ENV: setting}):
                self.assertEqual(
                    tuple(initchrg_start_needed(system)
                          for system in (self.mixed, self.same_sign, self.with_zero, self.uniform)),
                    expected,
                )

    def test_the_setting_reads_its_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("initchrg", "initchrg"), (" Weighted ", "weighted")):
            with mock.patch.dict(os.environ, {SPIN_START_ENV: value}):
                self.assertEqual(spin_start_setting(), expected)
        with mock.patch.dict(os.environ, {SPIN_START_ENV: "bogus"}), self.assertRaisesRegex(ValueError, SPIN_START_ENV):
            spin_start_setting()

    def test_the_start_of_initchrg_is_per_species(self) -> None:
        up, down = initchrg_spin_start(self.mixed)
        expected = periodic_spin_start(self.mixed)
        np.testing.assert_allclose(up, expected[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(down, expected[1], rtol=0, atol=1e-14)
        grid = self.mixed.grid
        # the channels add up to twice the total starting density (initchrg.f90), the moment is that of the species
        np.testing.assert_allclose(up + down, 2.0 * self.mixed.initial_density, rtol=1e-10, atol=1e-12)
        coordinates = np.asarray(grid.coordinates)
        nearest = [int(np.argmin(np.linalg.norm(coordinates - np.asarray(atom.position), axis=1)))
                   for atom in self.mixed.atoms]
        magnetization = up - down
        self.assertGreater(magnetization[nearest[0]], 0.0)  # Au starts up
        self.assertLess(magnetization[nearest[1]], 0.0)  # H starts down
        # one species: (1 +/- p) rho0
        up, down = initchrg_spin_start(self.uniform)
        np.testing.assert_allclose(up, 1.3 * self.uniform.initial_density, rtol=1e-14, atol=0)
        np.testing.assert_allclose(down, 0.7 * self.uniform.initial_density, rtol=1e-14, atol=0)

    def test_both_loops_start_as_initchrg_and_weighted_keeps_the_former_start(self) -> None:
        loops = {
            "kpoints": lambda system: run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1)),
            "gamma": run_scf_spin_polarized,
        }
        names = {id(self.mixed): "mixed", id(self.same_sign): "same_sign", id(self.uniform): "uniform"}
        for name, run in loops.items():
            for system, setting, initchrg in (
                (self.mixed, "auto", True),  # values of both signs: PARSEC 1.5's start
                (self.mixed, "initchrg", True),
                (self.mixed, "weighted", False),
                (self.same_sign, "auto", False),  # values of one sign: the former start, bit for bit
                (self.same_sign, "initchrg", True),
                (self.uniform, "auto", False),  # one value: the former start, bit for bit
                (self.uniform, "initchrg", True),
            ):
                with self.subTest(loop=name, system=names[id(system)], setting=setting):
                    with mock.patch.dict(os.environ, {SPIN_START_ENV: setting}):
                        found = _first_spin_fields(self, run, system, _Stop)
                    _check_first_fields(self, found, system, initchrg)


class SpinOrbitSpinStartTests(unittest.TestCase):
    """``PARSEC_SPIN_START`` in the two self-consistent spin-orbit loops with collinear spin (isolated and periodic):
    AuH with spin-orbit Au, the two species at opposite starts (PARSEC 1.5's start under ``auto``) and at starts of
    one sign (the former start, bit for bit)."""

    @staticmethod
    def _species(au: float, h: float) -> dict:
        return {
            "Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True, initial_spin_polarization=au),
            "H": SpeciesPotential(_H_POTRE, 0, initial_spin_polarization=h),
        }

    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python import GridSettings
        from parsec_python.SCF.single_point import prepare_single_point

        periodic = _auh_problem(spin_polarized=True)
        periodic = replace(periodic, scf=replace(periodic.scf, self_consistent_spin_orbit=True))
        isolated = SinglePointInput(
            atoms=[Atom("Au", [1.445, 0.0, 0.0]), Atom("H", [-1.445, 0.0, 0.0])],
            pseudopotentials=cls._species(0.5, -0.5),
            grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
            scf=replace(periodic.scf, max_iterations=40, number_of_states=20),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        cls.systems = {}
        for name, starts in (("opposite", (0.5, -0.5)), ("same_sign", (0.5, 0.1))):
            cls.systems[("periodic", name)] = prepare_periodic_single_point(
                replace(periodic, pseudopotentials=cls._species(*starts))
            )
            cls.systems[("isolated", name)] = prepare_single_point(
                replace(isolated, pseudopotentials=cls._species(*starts))
            )

    def test_both_spin_orbit_loops_follow_the_setting(self) -> None:
        from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
        from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints_spin_polarized
        from parsec_python.SCF.self_consistent_soc_spin_polarized import run_self_consistent_soc_spin_polarized

        def isolated(system):
            projectors = tuple(
                build_spin_orbit_projectors(
                    system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
                )
            )
            return run_self_consistent_soc_spin_polarized(system, projectors)

        loops = {
            "isolated": isolated,
            "periodic": lambda system: run_self_consistent_soc_kpoints_spin_polarized(
                system, np.zeros((1, 3)), np.ones(1)
            ),
        }
        for (kind, name), system in self.systems.items():
            for setting in ("auto", "initchrg", "weighted"):
                initchrg = setting == "initchrg" or (setting == "auto" and name == "opposite")
                with self.subTest(loop=kind, starts=name, setting=setting):
                    with mock.patch.dict(os.environ, {SPIN_START_ENV: setting}):
                        found = _first_spin_fields(self, loops[kind], system, _Stop)
                    _check_first_fields(self, found, system, initchrg)
                    if initchrg and name == "opposite":
                        # the moments of the two species have opposite signs (the former start has none here)
                        coordinates = np.asarray(system.grid.coordinates)
                        nearest = [
                            int(np.argmin(np.linalg.norm(coordinates - np.asarray(atom.position), axis=1)))
                            for atom in system.atoms
                        ]
                        magnetization = found["xc"][0] - found["xc"][1]
                        self.assertGreater(magnetization[nearest[0]], 0.0)
                        self.assertLess(magnetization[nearest[1]], 0.0)


if __name__ == "__main__":
    unittest.main()
