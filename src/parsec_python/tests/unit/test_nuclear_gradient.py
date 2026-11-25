"""PARSEC_GGA_NUCLEAR_GRADIENT: the correction of the finite-difference density gradient near the nuclei.

The setting, the model, the off value (the former bits), and the force term of the correction against central
differences of E_xc in the atomic positions at a fixed density, on a cluster and on a periodic cell, unpolarized and
spin (both spin forms), with an Au table whose 5d density has the hole at the nucleus that the correction is for.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python import GridSettings, PeriodicCell, PeriodicGridSettings
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import Atom
from parsec_python.Pseudopotential.potre import read_parsec_pseudopotential
from parsec_python.V_xc import nuclear_gradient as ng
from parsec_python.V_xc.pbe import pbe, pbe_spin_polarized
from parsec_python.V_xc.pbe_spin import pbe_spin

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_ENV = ng.NUCLEAR_GRADIENT_ENV


def _on():
    return mock.patch.dict(os.environ, {_ENV: "on"})


def _system(grid, atoms, lattice=None):
    potential = read_parsec_pseudopotential(_AU_POTRE)
    cell = None if lattice is None else SimpleNamespace(lattice_vectors=lattice, dimensions=3)
    return SimpleNamespace(
        grid=grid,
        atoms=tuple(atoms),
        pseudopotentials={"Au": potential},
        input=SimpleNamespace(periodic_cell=cell),
    )


def _density(system):
    """A smooth density with the atoms' d shells (the model of the table, deformed), fixed while atoms move."""

    model = ng.atomic_model(system.pseudopotentials["Au"])
    density = np.zeros(system.grid.size)
    lattice, dimensions = ng._geometry(system)
    for atom in system.atoms:
        for rows, relative in ng._balls(system.grid, atom.position, model.support, lattice, dimensions):
            s = np.einsum("ij,ij->i", relative, relative)
            inside = s < model.support ** 2
            value = model.parts(s[inside])[0]
            density[rows[inside]] += 0.9 * value * (1.0 + 0.2 * np.sin(relative[inside, 0]))
    return density + 1e-4


def _cluster():
    grid = build_cluster_grid(GridSettings(spacing=0.4, radius=6.0, expansion_order=8, shift=(0.0, 0.0, 0.0)))
    atoms = [Atom("Au", np.array([0.13, -0.07, 0.05])), Atom("Au", np.array([2.6, 2.1, -1.4]))]
    return _system(grid, atoms)


def _cell():
    lattice = 6.4 * np.eye(3)
    grid = build_periodic_grid(PeriodicCell(lattice_vectors=lattice), PeriodicGridSettings(spacing=0.4, expansion_order=8))
    atoms = [Atom("Au", np.array([0.11, 0.06, -0.04])), Atom("Au", np.array([3.3, 3.1, 2.9]))]
    return _system(grid, atoms, lattice)


def _energy(system, atoms, evaluate):
    lattice, dimensions = ng._geometry(system)
    moved = ng.build_correction(system.grid, atoms, system.pseudopotentials, lattice, dimensions)
    with ng._Applied(system.grid, lambda: moved):
        return evaluate().total_energy


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with mock.patch.dict(os.environ, {_ENV: ""}):
            self.assertEqual(ng.nuclear_gradient_setting(), "off")
        for value in ("on", " ON ", "off", "auto"):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertEqual(ng.nuclear_gradient_setting(), value.strip().lower())
        with mock.patch.dict(os.environ, {_ENV: "global"}):
            with self.assertRaisesRegex(ValueError, _ENV):
                ng.nuclear_gradient_setting()

    def test_off_builds_nothing_and_keeps_the_bits(self) -> None:
        system = _cluster()
        density = _density(system)
        with mock.patch.dict(os.environ, {_ENV: "off"}):
            self.assertIsNone(ng.system_correction(system))
            with ng.applied(system):
                inside = pbe(density, system.grid)
        plain = pbe(density, system.grid)
        np.testing.assert_array_equal(inside.potential, plain.potential)
        self.assertEqual(inside.total_energy, plain.total_energy)


class AutoTests(unittest.TestCase):
    def test_auto_takes_the_species_with_a_hole_at_the_nucleus(self) -> None:
        au = read_parsec_pseudopotential(_AU_POTRE)
        h = read_parsec_pseudopotential(_AU_POTRE.with_name("H_POTRE.DAT"))
        self.assertLess(ng.hole_depth(au), ng.HOLE_DEPTH)
        self.assertGreater(ng.hole_depth(h), 0.5)
        tables = {"Au": au, "H": h}
        self.assertEqual(ng.corrected_species(tables, "auto"), ("Au",))
        self.assertEqual(ng.corrected_species(tables, "on"), ("Au", "H"))
        self.assertEqual(ng.corrected_species(tables, "off"), ())
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=5.0, expansion_order=8, shift=(0.0, 0.0, 0.0)))
        atoms = (Atom("Au", np.array([0.1, 0.0, 0.0])), Atom("H", np.array([2.9, 0.1, 0.0])))
        system = SimpleNamespace(grid=grid, atoms=atoms, pseudopotentials=tables,
                                 input=SimpleNamespace(periodic_cell=None))
        hydrogen = SimpleNamespace(grid=grid, atoms=atoms[1:], pseudopotentials=tables,
                                   input=SimpleNamespace(periodic_cell=None))
        with mock.patch.dict(os.environ, {_ENV: "auto"}):
            # a structure without such an atom takes no correction (its bits)
            self.assertIsNone(ng.system_correction(hydrogen))
            auto = ng.system_correction(system)
            self.assertEqual(tuple(auto.models), ("Au",))
        with mock.patch.dict(os.environ, {_ENV: "on"}):
            self.assertIsNotNone(ng.system_correction(hydrogen))
            every = ng.system_correction(system)
        self.assertEqual(sorted(every.models), ["Au", "H"])
        only_au = ng.build_correction(grid, atoms[:1], tables)
        np.testing.assert_allclose(auto.field, only_au.field, atol=0.0)


class RunTests(unittest.TestCase):
    def test_functionals_and_species_of_a_run(self) -> None:
        from parsec_python.models import SCFSettings

        for functional, takes in (("ca", False), ("pw92", False), ("lb94", False), ("hf", False), ("pbe", True),
                                  ("pw91", True), ("pbe0", True), ("b3lyp", True), ("scan", False)):
            with self.subTest(functional=functional):
                self.assertEqual(ng.takes_correction(SCFSettings(xc_functional=functional)), takes)
        with mock.patch.dict(os.environ, {"PARSEC_META_GGA_GRADIENT": "density"}):
            self.assertTrue(ng.takes_correction(SCFSettings(xc_functional="scan")))
        system = _cluster()
        system.input.scf = SCFSettings(xc_functional="pbe")
        with mock.patch.dict(os.environ, {_ENV: "off"}):
            self.assertEqual(ng.corrected_atoms(system), ())
        with mock.patch.dict(os.environ, {_ENV: "auto"}):
            self.assertEqual(ng.corrected_atoms(system), ("Au",))
            system.input.scf = SCFSettings(xc_functional="ca")
            self.assertEqual(ng.corrected_atoms(system), ())
            # auto leaves spin runs to the finite-difference gradient (their bits); on takes them
            system.input.scf = SCFSettings(xc_functional="pbe", spin_polarized=True)
            self.assertEqual(ng.corrected_atoms(system), ())
            self.assertIsNone(ng.system_correction(system))
        with mock.patch.dict(os.environ, {_ENV: "on"}):
            self.assertEqual(ng.corrected_atoms(system), ("Au",))
            self.assertIsNotNone(ng.system_correction(system))

    def test_force_term_of_every_result_type(self) -> None:
        # a SpinPolarizedSinglePointResult holds density_up and density_down (no density, no spin channels): the
        # periodic spin drivers and the k-point loops; a spin loop's SinglePointResult holds its channels
        from parsec_python.models import SCFSettings

        system = _cluster()
        grid = system.grid
        density = _density(system)
        up, down = 0.7 * density, 0.3 * density
        host = SimpleNamespace(
            grid=grid, atoms=system.atoms, pseudopotentials=system.pseudopotentials,
            input=SimpleNamespace(periodic_cell=None, scf=SCFSettings(xc_functional="pbe", spin_polarized=True)),
            evaluate_xc=lambda n, **_: pbe(n, grid),
            evaluate_spin_xc=lambda a, b, *_: pbe_spin(a, b, grid),
            evaluate_xc_spin_polarized=lambda a, b: pbe_spin_polarized(a, b, grid),
        )
        with _on():
            correction = ng.system_correction(host)
            expected = {}
            for name, evaluate in (("spin_polarized", host.evaluate_xc_spin_polarized),
                                   ("spin", host.evaluate_spin_xc)):
                record = ng.FluxRecord()
                with ng.applied(host, record):
                    evaluate(up, down)
                expected[name] = correction.forces(record.flux, record.divergence)
            record = ng.FluxRecord()
            with ng.applied(host, record):
                pbe(density, grid)
            expected["unpolarized"] = correction.forces(record.flux, record.divergence)
            cases = {
                "spin_polarized": SimpleNamespace(density_up=up, density_down=down),
                "spin": SimpleNamespace(density=density, spin=(SimpleNamespace(density=up),
                                                               SimpleNamespace(density=down))),
                "unpolarized": SimpleNamespace(density=density),
            }
            for name, result in cases.items():
                with self.subTest(result=name):
                    forces = ng.correction_forces(host, result)
                    np.testing.assert_allclose(forces, expected[name], rtol=0.0, atol=1e-14)
                    self.assertGreater(float(np.max(np.abs(forces))), 1e-6)

    def test_spin_orbit_forces_with_collinear_spin_are_refused_before_the_scf(self) -> None:
        from parsec_python.models import SCFSettings

        scf = SCFSettings(xc_functional="pbe", spin_polarized=True, self_consistent_spin_orbit=True)
        translation = SimpleNamespace(problem=SimpleNamespace(scf=scf), compute_forces=True)
        with mock.patch.dict(os.environ, {_ENV: "off"}):
            self.assertIsNone(ng.forces_refusal(translation))
        with mock.patch.dict(os.environ, {_ENV: "auto"}):
            self.assertIsNone(ng.forces_refusal(translation))
        for value in ("on",):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertIn(_ENV, ng.forces_refusal(translation))
                self.assertIsNone(ng.forces_refusal(SimpleNamespace(problem=translation.problem,
                                                                    compute_forces=False)))


class ModelTests(unittest.TestCase):
    def test_model_is_smooth_and_tapered(self) -> None:
        model = ng.atomic_model(read_parsec_pseudopotential(_AU_POTRE))
        r = np.linspace(0.0, ng.TAPER_OUTER, 2001)
        value, first, second = model.parts(r * r)
        # the 5d hole: low at the nucleus, the shell near 1 bohr; zero (with its derivatives) at the outer radius
        self.assertLess(value[0], 0.01)
        self.assertGreater(value.max(), 0.3)
        self.assertAlmostEqual(float(value[-1]), 0.0, places=14)
        self.assertAlmostEqual(float(first[-1]), 0.0, places=14)
        # no wiggle of the table's first samples (their spline in r^2 has d2m/ds2 of 1e7 there): the model's is that
        # of the shell
        near = r < 0.1
        self.assertLess(float(np.max(np.abs(second[near]))), 2.0 * float(np.max(np.abs(second[(r > 0.1) & (r < 1.0)]))))


    def test_model_of_tables_on_sparse_meshes(self) -> None:
        # FHI tables: logarithmic meshes 0.1 bohr apart at 4 bohr, sparser than the knots (the fit is made on the
        # table resampled every RESAMPLE_SPACING); the model follows the table and the core correction enters it
        from parsec_python.Pseudopotential.fhipp import read_fhipp_pseudopotential

        folder = Path(__file__).resolve().parents[4] / "examples" / "3d_hybrid_functionals"
        for symbol, electrons, core in (("O", (2.0, 4.0, 0.0), False), ("C", (2.0, 2.0, 0.0), False),
                                        ("Mg", (2.0, 0.0, 0.0), True)):
            potential = read_fhipp_pseudopotential(
                folder / f"{symbol}_FHIPP.DAT", symbol=symbol, number_of_channels=3, electrons_per_orbital=electrons,
                core_cutoff_radius=1.6, nonlinear_core_correction=core,
            )
            radii, density = ng._radial_density(potential)
            model = ng.atomic_model(potential)
            inside = (radii > 0.05) & (radii < ng.TAPER_INNER)
            value = model.parts(radii[inside] ** 2)[0]
            with self.subTest(species=symbol):
                self.assertTrue(np.all(np.isfinite(value)))
                self.assertLess(float(np.max(np.abs(value - density[inside]))), 1e-3 * float(np.max(density)))


class CorrectionTests(unittest.TestCase):
    def _check_forces(self, system, kind: str) -> None:
        density = _density(system)
        grid = system.grid
        with _on():
            correction = ng.system_correction(system)
            self.assertIsNotNone(correction)
            # the correction changes the energy
            off = pbe(density, grid).total_energy
            with ng.applied(system):
                on = pbe(density, grid).total_energy
            self.assertGreater(abs(on - off), 1e-6)

            def evaluate():
                if kind == "spin":
                    return pbe_spin(0.5 * density, 0.5 * density, grid)
                if kind == "spin_unequal":
                    return pbe_spin(0.7 * density, 0.3 * density * np.exp(-0.2 * grid.coordinates[:, 0] ** 2), grid)
                if kind == "spin_polarized":
                    return pbe_spin_polarized(0.6 * density, 0.4 * density, grid)
                return pbe(density, grid)

            record = ng.FluxRecord()
            with ng.applied(system, record):
                evaluate()
            forces = correction.forces(record.flux, record.divergence)
            step = 2e-4
            for index in range(len(system.atoms)):
                numeric = np.zeros(3)
                for axis in range(3):
                    slopes = []
                    for h in (step, 2 * step):
                        moved = []
                        for sign in (1.0, -1.0):
                            atoms = list(system.atoms)
                            position = np.array(atoms[index].position, dtype=float)
                            position[axis] += sign * h
                            atoms[index] = replace(atoms[index], position=position)
                            moved.append(_energy(system, atoms, evaluate))
                        slopes.append(-(moved[0] - moved[1]) / (2 * h))
                    numeric[axis] = (4 * slopes[0] - slopes[1]) / 3
                with self.subTest(kind=kind, atom=index):
                    np.testing.assert_allclose(forces[index], numeric, rtol=0.0, atol=2e-8)
                    self.assertGreater(float(np.max(np.abs(forces[index]))), 1e-6)

    def test_cluster_forces(self) -> None:
        system = _cluster()
        for kind in ("unpolarized", "spin", "spin_unequal", "spin_polarized"):
            self._check_forces(system, kind)

    def test_spin_potentials_are_the_derivative(self) -> None:
        # the channels take the share n_s / n of the correction: the potentials hold the local terms of its
        # density dependence (V_xc.nuclear_gradient.share_potentials); both spin forms, a channel nearly empty
        system = _cluster()
        density = _density(system)
        grid = system.grid
        up = 0.8 * density
        down = 0.2 * density * np.exp(-0.3 * np.sum(grid.coordinates ** 2, axis=1)) + 1e-9
        rng = np.random.default_rng(5)
        with _on(), ng.applied(system):
            for name, evaluate in (("pbe_spin", pbe_spin), ("s_ab", pbe_spin_polarized)):
                result = evaluate(up, down, grid)
                for channel in ("up", "down"):
                    direction = rng.standard_normal(grid.size) * (up if channel == "up" else down) * 1e-2
                    values = []
                    for step in (1e-3, -1e-3):
                        if channel == "up":
                            values.append(evaluate(up + step * direction, down, grid).total_energy)
                        else:
                            values.append(evaluate(up, down + step * direction, grid).total_energy)
                    numeric = (values[0] - values[1]) / 2e-3
                    potential = result.potential_up if channel == "up" else result.potential_down
                    analytic = grid.volume_element * float(np.dot(potential, direction))
                    with self.subTest(form=name, channel=channel):
                        self.assertAlmostEqual(numeric, analytic, delta=1e-6 * abs(analytic) + 1e-12)

    def test_cell_forces(self) -> None:
        self._check_forces(_cell(), "unpolarized")
        self._check_forces(_cell(), "spin_polarized")

    def test_spin_of_equal_channels_is_the_unpolarized_energy(self) -> None:
        system = _cell()
        density = _density(system)
        with _on(), ng.applied(system):
            unpolarized = pbe(density, system.grid).total_energy
            spin = pbe_spin(0.5 * density, 0.5 * density, system.grid).total_energy
            ab = pbe_spin_polarized(0.5 * density, 0.5 * density, system.grid).total_energy
        self.assertAlmostEqual(spin, unpolarized, places=10)
        self.assertAlmostEqual(ab, unpolarized, places=10)

    def test_correction_of_the_atoms_is_their_exact_gradient(self) -> None:
        # n = M: D n + c = grad M at every point (c = grad M - D M)
        system = _cell()
        with _on():
            correction = ng.system_correction(system)
        from parsec_python.V_xc.pbe import density_gradient

        model = ng.atomic_model(system.pseudopotentials["Au"])
        total = np.zeros(system.grid.size)
        gradient = np.zeros((3, system.grid.size))
        lattice, dimensions = ng._geometry(system)
        for atom in system.atoms:
            for rows, relative in ng._balls(system.grid, atom.position, model.support, lattice, dimensions):
                s = np.einsum("ij,ij->i", relative, relative)
                inside = s < model.support ** 2
                value, first, _ = model.parts(s[inside])
                total[rows[inside]] += value
                gradient[:, rows[inside]] += 2.0 * first * relative[inside].T
        np.testing.assert_allclose(density_gradient(total, system.grid) + correction.field, gradient, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
