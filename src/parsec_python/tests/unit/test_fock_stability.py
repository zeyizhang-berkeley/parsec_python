"""The stop of the Fock loop: the bias of dexx and the stability test of a spin-symmetric state.

Independent references: the dense algebra of the ACE of a non-symmetric matrix for the bias of dexx; occupations
written by hand for the test of a spin-symmetric state; and the host Fock loop of an H3 chain with Hartree-Fock at
6000 K from a start without a moment, whose spin-symmetric state at the default Exx_Tolerance lies 0.22 Ry above
the doublet that the loop of the test reaches.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.SCF.fock_diis import OccupiedBlock
from parsec_python.SCF.fock_stability import (
    FOCK_DEXX_FLOOR_ENV,
    FOCK_SPIN_STABILITY_ENV,
    KELVIN_RY,
    PROBE_SHARE,
    SpinStabilityTest,
    dexx_floor_setting,
    displaced_occupations,
    entropy,
    free_energy,
    operator_bias,
    spin_stability_setting,
    symmetric_state,
)
from parsec_python.SCF.hybrid import FockOperator, OrbitalSet, natural_orbital_sets
from parsec_python.V_xc.exchange_operator import ExchangeApplication, compress_exchange, exchange_band_energy
from parsec_python.models import ExchangeLoopIteration


@contextmanager
def _environment(**values):
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        for name, reader in ((FOCK_DEXX_FLOOR_ENV, dexx_floor_setting), (FOCK_SPIN_STABILITY_ENV, spin_stability_setting)):
            with self.subTest(name=name):
                with _environment(**{name: None}):
                    self.assertEqual(reader(), "auto")
                with _environment(**{name: " OFF "}):
                    self.assertEqual(reader(), "off")
                with _environment(**{name: "on"}), self.assertRaisesRegex(ValueError, name):
                    reader()


class _Engine:
    """The host pieces the bias needs: the cross energy of an operator with orbital sets."""

    @staticmethod
    def cross_energy(operator, sets):
        return operator.cross_energy([s.orbitals for s in sets], [s.occupations for s in sets])


class DexxBiasTests(unittest.TestCase):
    def test_bias_of_a_non_symmetric_matrix_is_its_second_order_term(self) -> None:
        """The ACE of the symmetric part of M = -Phi^T W: at P = P0 the estimate is -a_x w sum f_j (A^T M_s^-1 A)_jj,
        0 for a symmetric W."""

        generator = np.random.default_rng(5)
        rows, states = 40, 5
        phi, _ = np.linalg.qr(generator.standard_normal((rows, states)))
        occupations = np.array([1.0, 1.0, 0.6, 0.0, 0.0])
        kernel = generator.standard_normal((rows, rows))
        kernel = kernel @ kernel.T / rows + np.eye(rows)
        for skew in (0.0, 1.0e-3):
            with self.subTest(skew=skew):
                noise = generator.standard_normal((rows, rows))
                operator_matrix = kernel + skew * (noise - noise.T)
                applied = -(operator_matrix @ phi)
                energy = 0.5 * float(np.dot(occupations, np.einsum("ij,ij->j", phi, applied)))
                application = ExchangeApplication(applied, tuple(range(states)), (0, 1, 2), energy, 0, 0.0, 0.0)
                channel = compress_exchange(phi, application, fraction=0.25, spin_weight=1.0)
                operator = FockOperator(channels=(channel,), fraction=0.25, reference_energy=energy)
                bias = operator_bias(_Engine(), operator, [OrbitalSet(phi, occupations, 1.0)], 0.25)
                m = -(phi.T @ applied)
                a = 0.5 * (m - m.T)
                symmetric = 0.5 * (m + m.T)
                expected = -0.25 * 1.0 * float(np.dot(occupations, np.einsum("ij,ij->j", a, np.linalg.solve(symmetric, a))))
                self.assertAlmostEqual(bias, expected, delta=1.0e-14 + 1.0e-9 * abs(expected))
                if skew == 0.0:
                    self.assertLess(abs(bias), 1.0e-14)
                else:
                    self.assertLess(bias, 0.0)
                # The bias is the estimate dexx of the loop at P = P0: 2 E_x^ACE(P0, P0) - 2 E_x(P0).
                cross = 0.5 * exchange_band_energy(channel, phi, occupations, scaled=False)
                self.assertAlmostEqual(bias, 0.25 * (2.0 * cross - 2.0 * energy), delta=1.0e-15)


def _set(occupations, orbitals=None):
    occupations = np.asarray(occupations, dtype=np.float64)
    if orbitals is None:
        orbitals = np.eye(8)[:, : occupations.size]
    return OrbitalSet(orbitals, occupations, 1.0)


class _BlockEngine:
    """Host blocks of the occupied orbitals and the natural sets of a channel (the host engine's)."""

    @staticmethod
    def occupied_blocks(sets):
        return [{None: OccupiedBlock(np.array(s.orbitals[:, s.occupations > 1.0e-3]),
                                     np.array(s.occupations[s.occupations > 1.0e-3]))} for s in sets]

    @staticmethod
    def natural_sets(sets, natural):
        return natural_orbital_sets(sets, natural)


class SymmetricStateTests(unittest.TestCase):
    def test_which_states_are_tested(self) -> None:
        half = [1.0, 0.5, 0.0, 0.0]
        state = symmetric_state((_set(half), _set([1.0, 0.505, 0.0, 0.0])))
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.spread, 0.005)
        self.assertAlmostEqual(state.fractional, 0.5)
        # a magnetic state, a closed shell with integer occupations, one channel
        self.assertIsNone(symmetric_state((_set([1.0, 1.0, 0.0]), _set([1.0, 0.0, 0.0]))))
        self.assertIsNone(symmetric_state((_set([1.0, 1.0, 0.004]), _set([1.0, 1.0, 0.004]))))
        self.assertIsNone(symmetric_state((_set(half),)))

    def test_displacement_moves_a_share_of_the_partly_filled_levels(self) -> None:
        up = {None: OccupiedBlock(np.eye(4)[:, :3], np.array([1.0, 0.5, 0.2]))}
        down = {None: OccupiedBlock(np.eye(4)[:, :3], np.array([1.0, 0.5, 0.2]))}
        (moved_up, moved_down), moved = displaced_occupations([up, down])
        np.testing.assert_allclose(moved_up[None], [1.0, 0.5 + PROBE_SHARE * 0.5, 0.2 + PROBE_SHARE * 0.2])
        np.testing.assert_allclose(moved_down[None], [1.0, 0.5 - PROBE_SHARE * 0.5, 0.2 - PROBE_SHARE * 0.2])
        self.assertAlmostEqual(moved, PROBE_SHARE * 0.7)
        # nothing partly filled: nothing moves
        full = {None: OccupiedBlock(np.eye(4)[:, :2], np.array([1.0, 1.0]))}
        self.assertIsNone(displaced_occupations([full, dict(full)]))

    def test_a_clearly_decaying_displacement_is_stable_at_once(self) -> None:
        """Stable from the second step on, where the last ratio of the displaced moment is at most STEP_RATIO and its
        linear continuation at most PREDICTED_RATIO; ratios near 1 stay undecided (a slowly unstable saddle shows
        them: the H3 chain of the test data at 33000 K, 0.103 -> 0.090 -> 0.090 -> 0.092), as does a slow decay. The
        ratio of the verdict is kept for the next structure, which does not test a state found stable with a ratio up
        to CARRY_RATIO again."""

        from parsec_python.SCF.fock_stability import CARRY_RATIO, PREDICTED_RATIO, STEP_RATIO

        class Scf:
            spin_polarized = True
            fixed_occupations = None
            fermi_temperature_kelvin = 6000.0

        sets = (_set([1.0, 0.5, 0.0, 0.0]), _set([1.0, 0.5, 0.0, 0.0]))

        def moved(moment):
            return (_set([1.0, 0.5 + moment / 2.0, 0.0, 0.0]), _set([1.0, 0.5 - moment / 2.0, 0.0, 0.0]))

        def steps(*moments):
            test = SpinStabilityTest(Scf(), enabled=True)
            test.start(_BlockEngine(), sets, -2.0, 4)
            self.assertAlmostEqual(test.moments[0], 0.1)
            return test, [test.observe(moved(value), 5 + index) for index, value in enumerate(moments)]

        # a first step alone never decides, however strong the decay
        fast, verdicts = steps(0.01)
        self.assertEqual(verdicts, [None])
        # Na4 at 3000 K: 0.40, 0.42
        clear, verdicts = steps(0.040, 0.0168)
        self.assertEqual(verdicts, [None, "stable"])
        self.assertAlmostEqual(clear.ratio, 0.42)
        self.assertIn("stable against a spin transfer", clear.text)
        self.assertIn("decay ratios 0.40 and 0.42", clear.text)
        self.assertIn("went back to it from the copies", clear.text)
        # the marginal saddle: 0.87 then 1.00 and a rise -- never stable, growing at the end
        marginal, verdicts = steps(0.087, 0.087, 0.089)
        self.assertEqual(verdicts, [None, None, None])
        self.assertTrue(marginal.growing)
        self.assertIsNone(marginal.ratio)
        # a slow decay (the chain at 35000 K: 0.81, 0.94, 0.96): undecided, left to the end of the continuation
        slow, verdicts = steps(0.081, 0.0761, 0.0731)
        self.assertEqual(verdicts, [None, None, None])
        self.assertFalse(slow.growing)
        # ratios that rise towards 1 (0.66, 0.84) wait for one that levels off (0.89: continued 0.94)
        rising, verdicts = steps(0.066, 0.0554, 0.0493)
        self.assertEqual(verdicts, [None, None, "stable"])
        self.assertLessEqual(rising.ratio, STEP_RATIO)
        self.assertLessEqual(2.0 * rising.ratio - 0.0554 / 0.066, PREDICTED_RATIO)
        # the next structure of a relaxation
        carried = SpinStabilityTest(Scf(), enabled=True)
        carried.skip(CARRY_RATIO)
        self.assertTrue(carried.waiting)
        self.assertIsNone(carried.start(_BlockEngine(), sets, -2.0, 3))
        self.assertIn("not tested again", carried.text)
        self.assertIn("taken over from such a structure", carried.text)
        self.assertEqual(carried.ratio, CARRY_RATIO)
        self.assertFalse(carried.waiting)
        # a magnetic state there carries nothing on
        magnetic = SpinStabilityTest(Scf(), enabled=True)
        magnetic.skip(0.1)
        self.assertIsNone(magnetic.start(_BlockEngine(), (_set([1.0, 1.0]), _set([1.0, 0.0])), -2.0, 3))
        self.assertIsNone(magnetic.ratio)
        self.assertEqual(magnetic.text, "")

    def test_bloch_sets_and_the_results_tested(self) -> None:
        """The Bloch factors of a cell hold an occupation array per k-point and the weights of the k-points; the
        test takes one k-point (the Gamma point of a spin-polarized cell runs as a mesh of one point), no mesh."""

        from types import SimpleNamespace

        from parsec_python.SCF.fock_stability import channel_electrons, single_point

        def bloch(*rows, weights):
            return SimpleNamespace(occupations=tuple(np.asarray(row, dtype=np.float64) for row in rows),
                                   weights=np.asarray(weights, dtype=np.float64))

        gamma = (bloch([1.0, 0.5, 0.0], weights=[1.0]), bloch([1.0, 0.5, 0.0], weights=[1.0]))
        self.assertIsNotNone(symmetric_state(gamma))
        self.assertAlmostEqual(entropy(gamma), 2.0 * np.log(2.0), places=14)
        mesh = (bloch([1.0, 0.5], [1.0, 0.0], weights=[0.25, 0.75]), bloch([1.0, 0.0], [1.0, 0.0], weights=[0.25, 0.75]))
        np.testing.assert_allclose(channel_electrons(mesh), [1.125, 1.0])
        self.assertAlmostEqual(entropy(mesh), 0.25 * np.log(2.0), places=14)
        self.assertTrue(single_point(SimpleNamespace()))
        self.assertTrue(single_point(SimpleNamespace(k_points=np.zeros((1, 3)))))
        self.assertFalse(single_point(SimpleNamespace(k_points=np.zeros((2, 3)))))

    def test_entropy_and_free_energy(self) -> None:
        sets = (_set([1.0, 0.5, 0.0]), _set([0.25, 0.0, 0.0]))
        expected = 2.0 * np.log(2.0) - (0.25 * np.log(0.25) + 0.75 * np.log(0.75)) - np.log(2.0)
        self.assertAlmostEqual(entropy(sets), expected, places=14)
        self.assertAlmostEqual(free_energy(-1.0, sets, 6000.0), -1.0 - KELVIN_RY * 6000.0 * expected, places=14)
        self.assertEqual(free_energy(-1.0, sets, 0.0), -1.0)

    def test_start_builds_the_displaced_sets_and_the_decision(self) -> None:
        class Scf:
            spin_polarized = True
            fixed_occupations = None
            fermi_temperature_kelvin = 6000.0

        sets = (_set([1.0, 0.5, 0.0, 0.0]), _set([1.0, 0.5, 0.0, 0.0]))
        test = SpinStabilityTest(Scf(), enabled=True)
        displaced = test.start(_BlockEngine(), sets, -2.0, 4)
        self.assertTrue(test.probing)
        np.testing.assert_allclose(displaced[0].occupations[:2], [1.0, 0.55])
        np.testing.assert_allclose(displaced[1].occupations[:2], [1.0, 0.45])
        self.assertAlmostEqual(test.moments[0], 0.1)
        self.assertTrue(test.return_plain)
        magnetic = (_set([1.0, 1.0, 0.0, 0.0]), _set([1.0, 0.0, 0.0, 0.0]))
        # a growing displacement is followed to the end, where a lower state ends the test
        self.assertIsNone(test.observe((_set([1.0, 0.8, 0.0, 0.0]), _set([1.0, 0.2, 0.0, 0.0])), 5))
        self.assertIsNone(test.observe(magnetic, 6))
        self.assertTrue(test.growing)
        self.assertEqual(test.decide(magnetic, -2.2, True, 9), "keep")
        self.assertIn("unstable", test.text)
        self.assertFalse(test.probing)
        self.assertIsNone(test.ratio)
        # a growing one that ends higher goes back to the copies of the symmetric state, and ends there
        again = SpinStabilityTest(Scf(), enabled=True)
        again.start(_BlockEngine(), sets, -2.0, 4)
        again.observe(magnetic, 5)
        self.assertEqual(again.decide(magnetic, -1.9, True, 9), "return")
        back = again.return_sets(_BlockEngine(), sets)
        np.testing.assert_allclose(back[0].occupations[:2], [1.0, 0.5])
        np.testing.assert_allclose(back[1].occupations[:2], [1.0, 0.5])
        again.observe(sets, 10)
        self.assertEqual(again.decide(sets, -2.0, True, 10), "keep")
        self.assertIn("found no lower state", again.text)
        self.assertIn("it ends there after outer iteration 10", again.text)
        # not armed: occup.in, no temperature, the switch off
        Scf.fixed_occupations = (1.0, 0.5)
        self.assertFalse(SpinStabilityTest(Scf(), enabled=True).armed)
        Scf.fixed_occupations = None
        self.assertFalse(SpinStabilityTest(Scf(), enabled=False).armed)


class RunnerEngineTests(unittest.TestCase):
    def test_the_engine_of_the_runner_carries_what_its_guard_reads(self) -> None:
        """The Fock engine of the MPI runner (a DeviceFockEngine that does not call its constructor) has the attributes
        that the weighted guard and the start of a meta-GGA hybrid read: without them every cluster hybrid on the
        runner stopped at its semilocal start (AttributeError)."""

        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.hybrid_mpi import MPIFockEngine

        class Exchange:
            @staticmethod
            def pair_solvers(system, definition):
                return SimpleNamespace(description="stub")

        for occupations, expected in ((None, False), ((1.0, 1.0), True)):
            system = SimpleNamespace(input=SimpleNamespace(scf=SimpleNamespace(fixed_occupations=occupations)))
            engine = MPIFockEngine(Exchange(), system, "definition")
            self.assertIs(engine.fixed_occupations, expected)
            self.assertEqual(engine.definition, "definition")
        # occup.in: the guard returns before it reads the result
        self.assertIsNone(engine.weighted_guard(None, "the semilocal start of the Fock loop"))


class SpinSaddleLoopTests(unittest.TestCase):
    """The host Fock loop of a spin-polarized H3 chain with Hartree-Fock at 6000 K from a start without a moment."""

    TEXT = """Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 6
Solver_Lpole: 4
Hartree_Atomic_Tail: off
States_Num: 6
Fermi_Temp: {temperature}
Max_Iter: 80
Convergence_Criterion: 1.0e-6 Ry
Eigensolver: chebff
Mixing_Method: Anderson
Mixing_Param: 0.3
Spin_Polarization: true
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
Initial_Spin_Polarization: 0.0
begin Atom_Coord
  0.0 0.0 -1.8
  0.0 0.0  0.0
  0.0 0.0  1.8
end Atom_Coord
Correlation_Type: hf
{extra}"""

    def run_case(self, temperature=6000, extra="", **switches):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF import prepare_single_point, run_scf

        directory = Path(tempfile.mkdtemp(prefix="parsec_fock_stability_"))
        try:
            library = Path(__file__).resolve().parents[1] / "data"
            shutil.copy(library / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            (directory / "parsec.in").write_text(self.TEXT.format(temperature=temperature, extra=extra))
            records = []
            with _environment(**switches):
                result = run_scf(prepare_single_point(parse_parsec_input(directory / "parsec.in").problem),
                                 callback=records.append)
            return result, [record for record in records if isinstance(record, ExchangeLoopIteration)]
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def moment(result) -> float:
        return float(np.sum(result.spin[0].occupations) - np.sum(result.spin[1].occupations))

    def test_the_saddle_is_left_for_the_doublet(self) -> None:
        tested, tested_outer = self.run_case(**{FOCK_SPIN_STABILITY_ENV: None, FOCK_DEXX_FLOOR_ENV: None})
        before, before_outer = self.run_case(**{FOCK_SPIN_STABILITY_ENV: "off", FOCK_DEXX_FLOOR_ENV: "off"})
        self.assertTrue(tested.converged and before.converged)
        # off: the loop stops at its spin-symmetric state, half of an electron in each channel's second level
        self.assertLess(abs(self.moment(before)), 1.0e-2)
        self.assertEqual(before.exact_exchange.spin_stability, "")
        # auto: the test leaves it for the doublet, 0.22 Ry lower
        self.assertGreater(abs(self.moment(tested)), 0.99)
        self.assertLess(tested.energies.total, before.energies.total - 0.2)
        self.assertIn("was unstable", tested.exact_exchange.spin_stability)
        start = next(record.iteration for record in tested_outer if "stability test" in record.outer_note)
        self.assertTrue(tested_outer[start].converged)
        self.assertLess(start, tested_outer[-1].iteration)
        # the steps up to the state of the test are those of the loop without it (the bias of dexx is far below
        # Exx_Tolerance here)
        self.assertEqual(start, before_outer[-1].iteration)
        self.assertAlmostEqual(tested_outer[start].total_energy, before.energies.total, delta=1.0e-7)
        # the loop of the test is plain: no extrapolation after its start
        self.assertFalse(any(record.extrapolated for record in tested_outer[start:]))
        self.assertIsNone(tested.exact_exchange.instability)

    def test_a_stable_state_is_kept(self) -> None:
        """At 35000 K the spin-symmetric state is stable: the displacement decays slowly, the continuation finds no
        lower state, and the loop goes back to the copies of the symmetric state's orbitals and ends there, without a
        moment."""

        tested, outer = self.run_case(temperature=35000)
        before, before_outer = self.run_case(
            temperature=35000, **{FOCK_SPIN_STABILITY_ENV: "off", FOCK_DEXX_FLOOR_ENV: "off"}
        )
        self.assertTrue(tested.converged and before.converged)
        self.assertLess(abs(self.moment(tested)), 1.0e-8)
        text = tested.exact_exchange.spin_stability
        # its ratios (0.81, 0.94, 0.96) leave the verdict to the end of the continuation, which finds no lower state
        self.assertIn("found no lower state", text)
        self.assertIn("went back to the symmetric state", text)
        self.assertIn("it ends there", text)
        self.assertIsNone(tested.exact_exchange.spin_stability_ratio)
        start = next(record.iteration for record in outer if "stability test" in record.outer_note)
        self.assertEqual(start, before_outer[-1].iteration)
        self.assertAlmostEqual(tested.energies.total, before.energies.total, delta=5.0e-4)

    def test_a_slowly_unstable_saddle_is_left(self) -> None:
        """At 33000 K the ratios of the displaced moment stay near 1 (0.87, 1.00, 1.02): no early verdict; the
        continuation grows to a state 8.2e-4 Ry lower in E - TS, which is kept."""

        tested, _outer = self.run_case(temperature=33000)
        self.assertTrue(tested.converged)
        text = tested.exact_exchange.spin_stability
        self.assertIn("was unstable", text)
        self.assertNotIn("stable against a spin transfer", text)
        self.assertGreater(abs(self.moment(tested)), 0.2)

    def test_an_unknown_value_is_refused_before_the_run(self) -> None:
        from parsec_python import cli as cli_module

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            shutil.copy(Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            path = directory / "parsec.in"
            path.write_text(self.TEXT.format(temperature=6000, extra=""))
            for name in (FOCK_SPIN_STABILITY_ENV, FOCK_DEXX_FLOOR_ENV):
                with self.subTest(name=name):
                    errors = StringIO()
                    with _environment(**{name: "maybe"}), redirect_stdout(StringIO()), redirect_stderr(errors):
                        code = cli_module.main([str(path), "--quiet"])
                    self.assertEqual(code, 2)
                    self.assertIn(name, errors.getvalue())
                    self.assertFalse((directory / "parsec.out").exists())


if __name__ == "__main__":
    unittest.main()
