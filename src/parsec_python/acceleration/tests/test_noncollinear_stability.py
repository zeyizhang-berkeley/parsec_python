"""The stability test of the state of a non-collinear loop in the sectors of its k-points
(``PARSEC_NONCOLLINEAR_SECTOR_STABILITY``), on the host.

Checked: the average over the group of the sectors (a projector; time reversal turns the field over); the Arnoldi
estimate of the largest eigenvalue on a linear response of known spectrum, with and without the rotations of the moments
left out; the Ritz states of the sectors unfolded to the whole grid (orthonormal, the levels of the whole spinor
Hamiltonian); the loop of O2 from antiparallel moments (unstable: the loop starts again with one task per k-point and
reaches the triplet, bit for bit as off; ``off`` keeps test 1 of the guard), a magnetic Au atom kept in its sectors (the
energy of the loop without sectors), a loop in sectors that does not converge started again; the switch values.  The
device code runs with NumPy as the array backend (``PARSEC_KPOINT_BACKEND=numpy-device``).
"""

from __future__ import annotations

import functools
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
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
import parsec_python.acceleration  # noqa: F401  (installs the parallel k-point backend)
from parsec_python.SCF.noncollinear import run_self_consistent_noncollinear_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.acceleration.SCF import noncollinear_stability as stability
from parsec_python.acceleration.SCF.kpoint_spinor_sectors import NONCOLLINEAR_SWITCH, SpinorSectorKPointSolver
from parsec_python.models import SpinSectorFallbackRecord

_TESTS = Path(__file__).resolve().parents[2] / "tests" / "data"
_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_DEVICE = {"PARSEC_KPOINT_BACKEND": "numpy-device"}
# Gamma and the star of (0, 0, 1/3) of an 8 bohr cube
_FRACTIONS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) / 3.0


def _fake_cell(points: int, turn: bool = False):
    """A cell group of the identity and an involution that reverses a line of ``points`` points: the inversion (``R' =
    1``) or, with ``turn``, the twofold rotation about z (``R' = diag(-1, -1, 1)``)."""

    rows = {0: np.arange(points), 1: np.arange(points)[::-1].copy()}
    second = np.diag([-1, -1, 1]) if turn else -np.eye(3, dtype=np.int64)
    return SimpleNamespace(
        table=np.array([[0, 1], [1, 0]]),
        matrices=[np.eye(3, dtype=np.int64), np.asarray(second, dtype=np.int64)],
        row_map=lambda index: rows[index],
    )


class SectorSymmetryTests(unittest.TestCase):
    def test_average_is_a_projector(self) -> None:
        rng = np.random.default_rng(1)
        cell = _fake_cell(7)
        symmetry = stability.SectorSymmetry(cell, [(1, 1)])
        self.assertEqual(symmetry.elements, [(0, 1), (1, 1)])
        vector = rng.standard_normal(28)
        once = symmetry.symmetric(vector)
        np.testing.assert_allclose(symmetry.symmetric(once), once, atol=1.0e-14)
        self.assertLess(abs(float(np.dot(once, symmetry.breaking(vector)))), 1.0e-12)
        # the inversion keeps V(-r) = V(r) and B(-r) = B(r)
        np.testing.assert_allclose(once[:7], once[:7][::-1], atol=1.0e-14)

    def test_time_reversal_turns_the_field_over(self) -> None:
        rng = np.random.default_rng(2)
        cell = _fake_cell(5)
        symmetry = stability.SectorSymmetry(cell, [(1, -1)])
        self.assertEqual(symmetry.elements, [(0, 1), (1, -1)])
        once = symmetry.symmetric(rng.standard_normal(20))
        field = once[5:].reshape(3, 5)
        np.testing.assert_allclose(field[:, ::-1], -field, atol=1.0e-14)
        np.testing.assert_allclose(once[:5][::-1], once[:5], atol=1.0e-14)
        # the inversion with and without time reversal generate time reversal itself: no symmetric field at all
        both = stability.SectorSymmetry(cell, [(1, -1), (1, 1)])
        self.assertIn((0, -1), both.elements)
        np.testing.assert_allclose(both.symmetric(rng.standard_normal(20))[5:], 0.0, atol=1.0e-14)


class ResponseTests(unittest.TestCase):
    @staticmethod
    def linear(symmetry, centre, eigenvalue, rotations=(), rotation_value=1.0):
        """``F(x) = x0 + J (x - x0)``: ``eigenvalue`` on the part that breaks the group, ``rotation_value`` on the
        given modes, 0.3 on the symmetric part."""

        modes = np.asarray(rotations, dtype=float).reshape(len(rotations), centre.size)

        def output(potential, field):
            delta = np.concatenate([potential, field.reshape(-1)]) - centre
            along = modes.T @ (modes @ delta) if len(modes) else 0.0 * delta
            breaking = symmetry.breaking(delta) - along
            return centre + eigenvalue * breaking + rotation_value * along + 0.3 * symmetry.symmetric(delta)

        return output

    def test_largest_eigenvalue_of_a_known_response(self) -> None:
        points = 9
        symmetry = stability.SectorSymmetry(_fake_cell(points), [(1, 1)])
        rng = np.random.default_rng(3)
        centre = symmetry.symmetric(rng.standard_normal(4 * points))
        density = 1.0 + rng.random(points)
        for value, unstable in ((2.5, True), (0.7, False)):
            report = stability.sector_state_stability(self.linear(symmetry, centre, value), centre, symmetry, density)
            self.assertAlmostEqual(report.largest, value, places=8)
            self.assertEqual(report.unstable, unstable)
            self.assertIn(f"{value:.4f}", report.text())

    def test_rotations_of_the_moments_are_left_out(self) -> None:
        points = 9
        rng = np.random.default_rng(4)
        # the inversion with or without time reversal commutes with a rotation of the spins: no rotation breaks it
        inversion = stability.SectorSymmetry(_fake_cell(points), [(1, -1)])
        self.assertEqual(stability.spin_rotation_modes(inversion, inversion.symmetric(rng.standard_normal(36))), [])
        # the twofold rotation about z keeps B_z and turns B_x, B_y over: the rotations about x and y break it
        symmetry = stability.SectorSymmetry(_fake_cell(points, turn=True), [(1, 1)])
        centre = symmetry.symmetric(rng.standard_normal(4 * points))
        modes = stability.spin_rotation_modes(symmetry, centre)
        self.assertEqual(len(modes), 2)
        output = self.linear(symmetry, centre, 0.4, modes, rotation_value=1.2)
        density = 1.0 + rng.random(points)
        kept = stability.sector_state_stability(output, centre, symmetry, density)
        self.assertAlmostEqual(kept.largest, 1.2, places=8)
        left = stability.sector_state_stability(output, centre, symmetry, density, rotations=True)
        self.assertAlmostEqual(left.largest, 0.4, places=8)
        self.assertEqual(left.rotations, 2)
        self.assertIn("2 rotations of the moments left out", left.text())

    def test_the_estimate_settles_before_it_stops(self) -> None:
        # a response with a spread spectrum on the perturbations that break the group (eigenvalues from -3 to 0.9 on
        # a random orthonormal basis of that subspace): the estimate stops after the minimum of probes, once the
        # largest Ritz value has settled, close to the largest eigenvalue
        points = 40
        symmetry = stability.SectorSymmetry(_fake_cell(points), [(1, 1)])
        rng = np.random.default_rng(7)
        centre = symmetry.symmetric(rng.standard_normal(4 * points))
        basis = np.linalg.qr(np.stack([symmetry.breaking(rng.standard_normal(4 * points)) for _ in range(80)], 1))[0]
        basis = basis[:, np.linalg.norm(basis, axis=0) > 0.5][:, :80]
        values = np.linspace(-3.0, 0.9, basis.shape[1])

        def output(potential, field):
            delta = np.concatenate([potential, field.reshape(-1)]) - centre
            return centre + basis @ (values * (basis.T @ symmetry.breaking(delta)))

        report = stability.sector_state_stability(output, centre, symmetry, 1.0 + rng.random(points))
        self.assertGreaterEqual(report.probes, stability.MINIMUM_STEPS)
        self.assertLessEqual(report.probes, stability.STEPS)
        self.assertFalse(report.unstable)
        self.assertLess(abs(report.largest - 0.9), 5.0e-2)
        self.assertIsNotNone(report.vector)

    def test_switch_values(self) -> None:
        from parsec_python.acceleration.SCF.kpoint_sectors import check_kpoint_sector_switches

        for value in ("auto", "off"):
            with mock.patch.dict(os.environ, {stability.SWITCH: value}):
                self.assertEqual(stability.noncollinear_sector_stability_setting(), value)
        with mock.patch.dict(os.environ, {stability.SWITCH: "on"}):
            with self.assertRaises(ValueError):
                stability.noncollinear_sector_stability_setting()
            with self.assertRaises(ValueError):
                check_kpoint_sector_switches()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(stability.SWITCH, None)
            self.assertEqual(stability.noncollinear_sector_stability_setting(), stability.DEFAULT)
            self.assertEqual(stability.DEFAULT, "auto")
            # the sectors of the non-collinear loop, with this test, on request (off by default)
            from parsec_python.acceleration.SCF.kpoint_spinor_sectors import noncollinear_kpoint_sectors_setting

            os.environ.pop(NONCOLLINEAR_SWITCH, None)
            self.assertEqual(noncollinear_kpoint_sectors_setting(), "off")


@functools.lru_cache(maxsize=None)
def _oxygen():
    atoms = [Atom("O", [4.0, 4.0, 2.86], [0, 0, 1]), Atom("O", [4.0, 4.0, 5.14], [0, 0, -1])]
    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials={"O": SpeciesPotential(_TESTS / "spin_ca_O_POTRE.DAT", 1, initial_spin_polarization=0.3)},
        grid=PeriodicGridSettings(spacing=0.4, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=8.0 * np.eye(3)),
        scf=SCFSettings(max_iterations=100, number_of_states=16, convergence_criterion=1.0e-6, noncollinear=True,
                        fermi_temperature_kelvin=500.0),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-7),
    )
    return prepare_periodic_single_point(problem)


@functools.lru_cache(maxsize=None)
def _gold(max_iterations: int = 80, pair: bool = False):
    # one Au atom with its moment along z, or two 8 bohr apart along z with antiparallel moments (a cell twice as long)
    atoms = [Atom("Au", [4.0, 4.0, 4.0], [0, 0, 1])]
    cell = 8.0 * np.eye(3)
    if pair:
        atoms.append(Atom("Au", [4.0, 4.0, 12.0], [0, 0, -1]))
        cell = np.diag([8.0, 8.0, 16.0])
    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True, initial_spin_polarization=0.3)},
        grid=PeriodicGridSettings(spacing=0.89, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=cell),
        scf=SCFSettings(max_iterations=max_iterations, number_of_states=20 * len(atoms), convergence_criterion=1.0e-9,
                        noncollinear=True, fermi_temperature_kelvin=1000.0),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-10),
    )
    return prepare_periodic_single_point(problem)


def _run(system, k_points, value: str, environment=None, steps=None):
    k_points = np.asarray(k_points, dtype=float).reshape(-1, 3)
    weights = np.full(len(k_points), 1.0 / len(k_points))
    with mock.patch.dict(os.environ, {**_DEVICE, NONCOLLINEAR_SWITCH: value, **(environment or {})}):
        return run_self_consistent_noncollinear_kpoints(
            system, k_points, weights, callback=None if steps is None else steps.append
        )


def _notes(steps):
    return [step for step in steps if isinstance(step, SpinSectorFallbackRecord)]


_GOLD_K = 2.0 * np.pi / 8.0 * _FRACTIONS


class StabilitySCFTests(unittest.TestCase):
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _off(cls, case: str, max_iterations: int = 80):
        if case == "oxygen":
            return _run(_oxygen(), np.zeros(3), "off")
        return _run(_gold(max_iterations, case == "pair"), _GOLD_K, "off")

    def test_an_unstable_symmetric_state_leaves_for_the_lower_state(self) -> None:
        # O2 from antiparallel moments at Gamma: the sectors (the inversion times time reversal pairs them) reach the
        # symmetric ensemble of the four pi* levels, whose response to a moment that breaks the pairs is far above 1;
        # the loop without sectors continues from that state displaced along the top Ritz vector and reaches the
        # triplet that the loop without sectors reaches from the start
        steps: list = []
        result = _run(_oxygen(), np.zeros(3), "on", steps=steps)
        off = self._off("oxygen")
        notes = _notes(steps)
        self.assertEqual([note.route for note in notes], ["kpoint", "kpoint_continued"])
        self.assertEqual(notes[0].stage, "the stability test after the SCF loop")
        self.assertIn("unstable", notes[0].reason)
        self.assertIn("continues from the state of the sectors displaced", notes[0].multiplets[1])
        self.assertIn("loop without sectors is kept", notes[1].reason)
        self.assertGreater(result.timings["noncollinear_stability_largest"], 5.0)
        self.assertEqual(result.timings["noncollinear_sector_abandoned"], 1.0)
        self.assertGreater(result.timings["noncollinear_continuation_lower"], 1.0e-2)
        self.assertNotIn("noncollinear_guard_start_share", result.timings)
        self.assertNotIn("eig_spinor_sector_kpoints", result.timings)
        self.assertTrue(result.converged)
        self.assertGreater(off.magnetic_moment, 1.5)
        self.assertLess(abs(result.magnetic_moment - off.magnetic_moment), 1.0e-6)
        self.assertLess(abs(result.energies.total - off.energies.total), 1.0e-8)

    def test_off_keeps_test_one_of_the_guard(self) -> None:
        steps: list = []
        result = _run(_oxygen(), np.zeros(3), "on", {stability.SWITCH: "off"}, steps)
        notes = _notes(steps)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].stage, "the start Hamiltonian")
        self.assertGreater(result.timings["noncollinear_guard_start_share"], 0.5)
        self.assertNotIn("noncollinear_stability_largest", result.timings)
        self.assertEqual(result.energies.total, self._off("oxygen").energies.total)

    def test_a_stable_state_keeps_its_sectors(self) -> None:
        # an Au pair with antiparallel moments (spin-orbit coupling) on a star of k-points, whose moments decay: the 6s
        # levels of the start are partly filled across the sectors (test 1 of the guard would leave them); the largest
        # eigenvalue of the response converges below 1 (about 0.64 after 15 probes): kept, with the energy of off
        steps: list = []
        result = _run(_gold(pair=True), _GOLD_K, "on", steps=steps)
        off = self._off("pair")
        notes = _notes(steps)
        self.assertEqual([note.route for note in notes], ["kpoint_stable"])
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines

        lines = spin_sector_fallback_lines(notes[0])
        self.assertIn("stable against perturbations that break their operations (kept)", lines[1])
        self.assertIn("largest eigenvalue", lines[2])
        self.assertGreater(result.timings["eig_spinor_sector_kpoints"], 0)
        self.assertNotIn("noncollinear_sector_abandoned", result.timings)
        largest = result.timings["noncollinear_stability_largest"]
        self.assertGreater(largest, 0.55)
        self.assertLess(largest, 1.0)
        # (the adaptive stop: at least the minimum of probes, and the largest Ritz value settled)
        self.assertGreaterEqual(result.timings["noncollinear_stability_probes"], float(stability.MINIMUM_STEPS))
        self.assertTrue(result.converged and off.converged)
        self.assertLess(abs(result.energies.total - off.energies.total), 1.0e-9)
        self.assertLess(np.max(np.abs(result.magnetization - off.magnetization)), 1.0e-7)

    def test_a_soft_unstable_moment_ends_below_the_sectors(self) -> None:
        # a magnetic Au atom with its moment along z (spin-orbit coupling): the response to a turn of the moment is
        # just above 1 (about 1.03, the anisotropy of the grid), so its free energy goes down by turning; the loop
        # without sectors from the displaced state ends below the state of the sectors, which the loop without sectors
        # from the start keeps (no noise turns it)
        steps: list = []
        result = _run(_gold(), _GOLD_K, "on", steps=steps)
        off = self._off("gold")
        notes = _notes(steps)
        self.assertEqual([note.route for note in notes], ["kpoint", "kpoint_continued"])
        largest = result.timings["noncollinear_stability_largest"]
        self.assertGreater(largest, stability.BOUND)
        self.assertLess(largest, 1.1)
        self.assertGreater(result.timings["noncollinear_continuation_lower"], 1.0e-6)
        self.assertTrue(result.converged)
        temperature = float(_gold().input.scf.fermi_temperature_kelvin)
        from parsec_python.SCF.noncollinear import _free_energy

        self.assertLess(_free_energy(result, temperature), _free_energy(off, temperature) - 1.0e-6)

    def test_a_loop_in_sectors_that_does_not_converge_is_kept_untested(self) -> None:
        steps: list = []
        result = _run(_gold(4), _GOLD_K, "on", steps=steps)
        notes = _notes(steps)
        self.assertEqual([note.route for note in notes], ["kpoint_stable"])
        self.assertIn("did not converge", notes[0].reason)
        self.assertFalse(result.converged)
        self.assertGreater(result.timings["eig_spinor_sector_kpoints"], 0)
        self.assertEqual(result.timings["noncollinear_stability_untested"], 1.0)
        self.assertNotIn("noncollinear_stability_largest", result.timings)


class WholeStateTests(unittest.TestCase):
    def test_ritz_states_of_the_sectors_on_the_whole_grid(self) -> None:
        # the states the solver of the test starts from: orthonormal whole-grid spinors whose Rayleigh quotients with the
        # whole spinor Hamiltonian at the last potential are their Ritz values
        captured = {}
        original = SpinorSectorKPointSolver.whole_states

        def capture(self, working):
            found = original(self, working)
            captured["states"] = found
            captured["working"] = working
            return found

        system = _gold(pair=True)
        with mock.patch.object(SpinorSectorKPointSolver, "whole_states", capture):
            result = _run(system, _GOLD_K, "on")
        self.assertGreater(result.timings["eig_spinor_sector_kpoints"], 0)
        states = captured["states"]
        self.assertEqual(sorted(states), list(range(len(_GOLD_K))))
        from parsec_python.SCF.kpoints_soc import _k_point_operators, _spinor_hamiltonian

        operators = _k_point_operators(system, _GOLD_K)
        potential = result.input_effective_potential
        for index in (0, 1):
            state = states[index]
            vectors = state.vectors
            self.assertEqual(vectors.shape, (2 * system.grid.size, captured["working"]))
            np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(vectors.shape[1]), atol=1.0e-10)
            self.assertTrue(np.all(np.diff(state.eigenvalues) >= -1.0e-12))
            nonlocal_k, soc_k = operators[index]
            field = result.scf_potentials["xc_field"]
            whole = _spinor_hamiltonian(system, _GOLD_K[index], nonlocal_k, soc_k, potential, xc_field=field)
            quotients = np.real(np.sum(vectors.conj() * whole.apply(vectors), axis=0))
            # (the field of the last step and that of the result differ by the residual of the loop)
            np.testing.assert_allclose(quotients[:12], state.eigenvalues[:12], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
