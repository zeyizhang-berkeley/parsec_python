"""The stability test of the state of a collinear spin run in the sectors (``PARSEC_SPIN_SECTOR_STABILITY``).

CPU tests of the pieces of :mod:`parsec_python.acceleration.SCF.sector_stability` (the average over the orbits of a
group, the Arnoldi steps on a known response, the collinear flip pattern, the free energy, the switch and the notes);
the route on a CUDA device.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np


class OrbitSymmetryTests(unittest.TestCase):
    def test_the_average_over_the_orbits_of_a_generated_group(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import OrbitSymmetry

        # a ring of 12 points: the reflection i -> 11 - i and the rotation by 3 generate a group of order 8
        size = 12
        reflection = np.array([size - 1 - i for i in range(size)])
        rotation = np.array([(i + 3) % size for i in range(size)])
        symmetry = OrbitSymmetry(size, [reflection, rotation])
        group = {tuple(range(size))}
        frontier = list(group)
        while frontier:
            fresh = []
            for element in frontier:
                for generator in (reflection, rotation):
                    product = tuple(np.asarray(element)[generator])
                    if product not in group:
                        group.add(product)
                        fresh.append(product)
            frontier = fresh
        values = np.random.default_rng(3).standard_normal(2 * size)
        expected = np.concatenate([
            np.mean([row[np.asarray(element)] for element in group], axis=0) for row in values.reshape(2, size)
        ])
        np.testing.assert_allclose(symmetry.symmetric(values), expected, atol=1e-14)
        broken = symmetry.breaking(values)
        np.testing.assert_allclose(symmetry.symmetric(broken), 0.0, atol=1e-14)
        self.assertEqual(symmetry.generators, 2)

    def test_no_generator_keeps_every_point(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import OrbitSymmetry

        symmetry = OrbitSymmetry(5, [])
        values = np.arange(10.0)
        np.testing.assert_array_equal(symmetry.symmetric(values), values)
        with self.assertRaises(ValueError):
            OrbitSymmetry(5, [np.arange(4)])


class ResponseTests(unittest.TestCase):
    def _linear(self, top: float, seed: int = 0):
        from parsec_python.acceleration.SCF.sector_stability import OrbitSymmetry

        size = 10
        symmetry = OrbitSymmetry(size, [np.array([size - 1 - i for i in range(size)])])
        rng = np.random.default_rng(seed)
        dimension = 2 * size
        # a response on the perturbations that break the group (an orthonormal basis of them), largest eigenvalue top
        projector = np.eye(dimension) - np.array([symmetry.symmetric(row) for row in np.eye(dimension)]).T
        values, vectors = np.linalg.eigh(projector)
        breaking = vectors[:, values > 0.5]
        mixed, _ = np.linalg.qr(rng.standard_normal((breaking.shape[1], breaking.shape[1])))
        breaking = breaking @ mixed
        response = breaking @ np.diag(np.linspace(-2.0, top, breaking.shape[1])) @ breaking.T
        largest = float(np.max(np.linalg.eigvals(response).real))
        self.assertAlmostEqual(largest, top, places=10)
        base = symmetry.symmetric(rng.standard_normal(dimension))
        return symmetry, response, base, largest, rng

    def test_the_largest_eigenvalue_of_a_known_response(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import response_stability

        for top, unstable in ((1.4, True), (0.6, False)):
            symmetry, response, base, largest, rng = self._linear(top)
            reference = base + 0.0

            def output(point):
                return reference + response @ (point - base)

            report = response_stability(output, base, symmetry, rng.standard_normal(base.size), reference=reference)
            self.assertAlmostEqual(report.largest, largest, delta=1e-8)
            self.assertEqual(report.unstable, unstable)
            self.assertTrue(report.converged)
            # the top Ritz vector breaks the group and is an eigenvector of the response
            vector = report.vector
            np.testing.assert_allclose(symmetry.symmetric(vector), 0.0, atol=1e-10)
            self.assertAlmostEqual(float(np.max(np.abs(vector))), 1.0)
            np.testing.assert_allclose(response @ vector, report.largest * vector, atol=1e-7)

    def test_the_noise_at_the_centre_is_taken_out(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import response_stability

        symmetry, response, base, largest, rng = self._linear(0.9, seed=4)
        noise = symmetry.breaking(1.0e-6 * rng.standard_normal(base.size))
        reference = base + noise

        def output(point):
            return reference + response @ (point - base)

        report = response_stability(output, base, symmetry, rng.standard_normal(base.size), reference=reference)
        self.assertAlmostEqual(report.largest, largest, delta=1e-8)

    def test_a_symmetric_start_is_no_test(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import OrbitSymmetry, response_stability

        symmetry = OrbitSymmetry(4, [np.array([1, 0, 3, 2])])
        start = symmetry.symmetric(np.arange(8.0))
        report = response_stability(lambda point: point, start, symmetry, start)
        self.assertEqual(report.probes, 0)
        self.assertFalse(report.unstable)


class PatternTests(unittest.TestCase):
    def test_the_moments_of_the_smaller_group_turn_over(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import flip_pattern

        # four atoms of two points each, all moments up (V_up = 0, V_down = 2: field -1)
        up = np.zeros(8)
        down = np.full(8, 2.0)
        centre = np.concatenate([up, down])
        labels = np.repeat(np.arange(4), 2)
        # a mode that lowers the field of atom 0 and raises it on atoms 1, 2, 3
        change = np.array([1.0, 1.0, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4])
        vector = np.concatenate([change, -change])
        start, flipped = flip_pattern(centre, vector, labels, 4)
        np.testing.assert_array_equal(flipped, [True, False, False, False])
        np.testing.assert_allclose(start[:8], [2.0, 2.0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(start[8:], [0.0, 0.0, 2, 2, 2, 2, 2, 2])
        # a change of one sign alone (a global one) splits nothing
        self.assertIsNone(flip_pattern(centre, np.concatenate([np.ones(8), -np.ones(8)]), labels, 4))


class FreeEnergyTests(unittest.TestCase):
    def test_the_entropy_term(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import KELVIN_RY, spin_free_energy

        half = 0.5 * np.log(0.5) * 2.0
        value = spin_free_energy(-1.0, [np.array([1.0, 0.5, 0.0]), np.array([0.5])], 300.0)
        self.assertAlmostEqual(value, -1.0 + KELVIN_RY * 300.0 * 2.0 * half, places=14)
        self.assertEqual(spin_free_energy(-1.0, [np.array([0.5])], 0.0), -1.0)


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import SWITCH, spin_sector_stability_setting

        with mock.patch.dict(os.environ, {SWITCH: ""}):
            self.assertEqual(spin_sector_stability_setting(), "off")
        with mock.patch.dict(os.environ, {SWITCH: "AUTO"}):
            self.assertEqual(spin_sector_stability_setting(), "auto")
        with mock.patch.dict(os.environ, {SWITCH: "on"}), self.assertRaises(ValueError):
            spin_sector_stability_setting()

    def test_the_notes(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        for route, ending in (("spin_stable", "(kept):"), ("spin_unstable", ":"), ("spin_continued", ":")):
            record = SpinSectorFallbackRecord(
                stage="the stability test after the SCF loop",
                largest_fractional_occupation=0.4,
                tolerance=1.02,
                multiplets=("largest eigenvalue 0.4000 of the SCF response",),
                reason="the state of the sectors is stable",
                route=route,
            )
            lines = spin_sector_fallback_lines(record)
            self.assertTrue(lines[1].startswith(" NOTE: Spin_Symmetry_Sectors: the state of the sectors is stable"))
            self.assertTrue(lines[1].endswith(ending))
            self.assertIn("largest eigenvalue 0.4000", "\n".join(lines))
            self.assertNotIn("starts again", "\n".join(lines))


class DeferralTests(unittest.TestCase):
    def test_which_runs_take_the_test(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.sector_stability import SWITCH
        from parsec_python.acceleration.driver import _SpinSectorFallback

        fallback = _SpinSectorFallback.__new__(_SpinSectorFallback)
        fallback.spin = True
        fallback.mpi_context = None
        fallback.plan = None

        def system(**changes):
            scf = SimpleNamespace(is_hybrid=False, is_meta_gga=False, fixed_occupations=None, van_der_waals=False)
            for key in ("is_hybrid", "is_meta_gga", "fixed_occupations"):
                if key in changes:
                    setattr(scf, key, changes.pop(key))
            reference = SimpleNamespace(
                input=SimpleNamespace(scf=scf, eigensolver=SimpleNamespace(method=changes.pop("method", "chebff"))),
                hubbard=changes.pop("hubbard", None),
            )
            fields = dict(
                reference=reference, backend_info=SimpleNamespace(selected="cupy"),
                scalar_field_adapter=SimpleNamespace(guard=True), exchange_device=None, meta_gga_device=None,
                hubbard_device=None, spin_flip=None,
            )
            fields.update(changes)
            return SimpleNamespace(**fields)

        with mock.patch.dict(os.environ, {SWITCH: "off"}):
            self.assertFalse(fallback._stability_wanted(system(), None))
        with mock.patch.dict(os.environ, {SWITCH: "auto"}):
            self.assertTrue(fallback._stability_wanted(system(), None))
            self.assertFalse(fallback._stability_wanted(system(), object()))
            for changes in (
                {"is_hybrid": True}, {"is_meta_gga": True}, {"fixed_occupations": (1.0,)}, {"hubbard": object()},
                {"spin_flip": object()}, {"method": "arpack"}, {"backend_info": SimpleNamespace(selected="numpy")},
                {"scalar_field_adapter": SimpleNamespace(guard=False)},
            ):
                self.assertFalse(fallback._stability_wanted(system(**changes), None), changes)
            fallback.mpi_context = object()
            self.assertFalse(fallback._stability_wanted(system(), None))
            fallback.mpi_context = None
            # a run under auto that does not take the test says why, once
            from parsec_python.Output.parsec_output import spin_sector_fallback_lines

            notes = []
            self.assertFalse(fallback._stability_wanted(system(hubbard=object()), None, notes.append))
            self.assertFalse(fallback._stability_wanted(system(hubbard=object()), None, notes.append))
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].route, "spin_stability_skipped")
            self.assertIn("is not applied to this run: LDA+U", " ".join(spin_sector_fallback_lines(notes[0])))
            fallback._stability_declined = True
            self.assertFalse(fallback._stability_wanted(system(), None))

    def test_tests_1_and_3_put_their_reports_aside(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.spin_sectors import (
            FractionalOccupations,
            SpinSectorReducer,
            SpinSectorsAbandoned,
        )

        adapter = SpinSectorReducer.__new__(SpinSectorReducer)
        adapter.deferred = None
        adapter.defer_start = True
        report = FractionalOccupations(largest=0.5, tolerance=0.05, lines=("up channel",))
        with self.assertRaises(SpinSectorsAbandoned):
            adapter.abandon_start(report)
        adapter.deferred = []
        adapter.abandon_start(report, passes=2)
        self.assertEqual(adapter.deferred[0][0], "the start Hamiltonian")
        self.assertIn("after 2 further solve(s)", adapter.deferred[0][1].lines[-1])
        adapter.defer_start = False
        with self.assertRaises(SpinSectorsAbandoned):
            adapter.abandon_start(report)
        # test 3: a fractional multiplet of the end of the loop
        adapter.defer_start = True
        adapter.deferred = []
        adapter.guard = True
        adapter.structure = None
        adapter.fixed_occupations = False
        adapter.tolerance = 0.05
        adapter.hybrid = False
        adapter.reducer = SimpleNamespace(reduction=SimpleNamespace(sector_weights=None))
        channel = SimpleNamespace(
            eigenvalues=np.array([-0.5, -0.3, -0.3, 0.2]),
            occupations=np.array([1.0, 0.5, 0.5, 0.0]),
            representations=np.array([1, 2, 3, 1]),
            density=np.zeros(4),
        )
        result = SimpleNamespace(spin=(channel, channel))
        adapter.check_spin_result(result, final=True)
        adapter.check_spin_result(result, final=False)
        self.assertEqual(len(adapter.deferred), 1)
        self.assertEqual(adapter.deferred[0][0], "the SCF loop")
        adapter.deferred = None
        with self.assertRaises(SpinSectorsAbandoned):
            adapter.check_spin_result(result, final=True)


class MomentTests(unittest.TestCase):
    def test_the_order_of_the_labels(self) -> None:
        from parsec_python.acceleration.SCF.sector_stability import labelled_moments

        # four atoms on a line, four points each; labels a (+), b (-) and an unpolarized c
        coordinates = np.array([[float(x), 0.0, 0.0] for x in range(16)])
        positions = [[1.5, 0, 0], [5.5, 0, 0], [9.5, 0, 0], [13.5, 0, 0]]
        labels = ["A", "A", "B", "C"]
        starts = [0.5, 0.5, -0.5, 0.0]

        def densities(moments):
            up = np.concatenate([np.full(4, 1.0 + m / 8.0) for m in moments])
            down = np.concatenate([np.full(4, 1.0 - m / 8.0) for m in moments])
            return up, down

        text, kept = labelled_moments(coordinates, positions, labels, starts, densities([2.0, 2.0, -2.0, 0.0]), 1.0)
        self.assertTrue(kept)
        self.assertIn("A (start +0.5) +2.00 +2.00", text)
        self.assertIn("B (start -0.5) -2.00", text)
        # the order turned as a whole is the same order
        self.assertTrue(labelled_moments(coordinates, positions, labels, starts, densities([-2, -2, 2, 0]), 1.0)[1])
        # B turned (ferromagnetic), a vanished moment, a split label: not kept
        for moments in ([2, 2, 2, 0], [2, 2, 0.0, 0], [2, -2, -2, 0]):
            self.assertFalse(labelled_moments(coordinates, positions, labels, starts, densities(moments), 1.0)[1])


class CandidateTests(unittest.TestCase):
    """PARSEC_SPIN_SECTOR_STABILITY=auto where a guard fired: the result of the guard's route (that of off) against a
    converged continuation that waits for it, never above off and never unconverged where off converged."""

    def _fallback(self, free):
        from types import SimpleNamespace

        from parsec_python.acceleration.driver import _SpinSectorFallback

        fallback = _SpinSectorFallback.__new__(_SpinSectorFallback)
        candidate = SimpleNamespace(name="continued")
        fallback._candidate = dict(
            result=candidate, free=free, temperature=0.0, deferred=("the SCF loop", None), why="test 3",
            lines=("moments",), values={}, gain=0.0,
        )
        fallback.replaced = []
        fallback.replace_with_full_grid = lambda system, stage, report: fallback.replaced.append(stage)
        return fallback, candidate

    @staticmethod
    def _result(total, converged):
        from types import SimpleNamespace

        channel = SimpleNamespace(occupations=np.array([1.0, 0.0]))
        return SimpleNamespace(energies=SimpleNamespace(total=total), spin=(channel, channel), converged=converged)

    @staticmethod
    def _system(sectors=False):
        from types import SimpleNamespace

        return SimpleNamespace(
            scalar_field_adapter=object() if sectors else None, backend=SimpleNamespace(symmetry_eigensolver=None)
        )

    def test_the_lower_converged_state(self) -> None:
        notes = []
        fallback, candidate = self._fallback(-1.0)
        off = self._result(-0.9, True)
        self.assertIs(fallback._with_candidate(self._system(), off, notes.append), candidate)
        self.assertEqual(fallback.replaced, [])
        self.assertIn("below the converged result of the guard's route", notes[-1].reason)
        self.assertIsNone(fallback._candidate)
        # the route of off lower (or within the margin): its result
        for total in (-1.1, -1.0 + 5.0e-7):
            fallback, _ = self._fallback(-1.0)
            off = self._result(total, True)
            self.assertIs(fallback._with_candidate(self._system(), off, notes.append), off)
            self.assertIn("is kept: the continued state is not lower", notes[-1].reason)
        # the route of off unconverged: the converged continuation
        fallback, candidate = self._fallback(-1.0)
        self.assertIs(fallback._with_candidate(self._system(), self._result(-1.1, False), notes.append), candidate)
        # the route ended in sectors: the full grid prepared for the continued state
        fallback, candidate = self._fallback(-1.0)
        self.assertIs(fallback._with_candidate(self._system(sectors=True), self._result(-0.9, True)), candidate)
        self.assertEqual(fallback.replaced, ["the SCF loop"])
        # no candidate: the result as it is
        fallback, _ = self._fallback(-1.0)
        fallback._candidate = None
        off = self._result(-0.9, True)
        self.assertIs(fallback._with_candidate(self._system(), off), off)


class DetachTests(unittest.TestCase):
    def test_no_eigensolver_state_stays_with_a_candidate(self) -> None:
        from dataclasses import dataclass
        from types import SimpleNamespace

        from parsec_python.acceleration.driver import _detached_result

        @dataclass(frozen=True)
        class Channel:
            occupations: np.ndarray
            eigensolver_state: object

        @dataclass
        class Inner:
            spin: tuple
            eigensolver_state: object
            wavefunctions: np.ndarray

        state = object()
        inner = Inner(spin=(Channel(np.ones(2), state), Channel(np.ones(2), state)), eigensolver_state=(state, state),
                      wavefunctions=np.zeros((4, 4)))
        wrapper = SimpleNamespace(result=inner, wavefunctions=inner.wavefunctions)
        _detached_result(None, wrapper)
        self.assertIsNone(inner.eigensolver_state)
        self.assertTrue(all(channel.eigensolver_state is None for channel in inner.spin))
        self.assertIsNone(wrapper.eigensolver_state)


class ContinuationTests(unittest.TestCase):
    """_tested_result with the full grid and the test replaced by stand-ins: an unconverged continuation never replaces
    anything, a continuation that runs out of device memory falls back to the result of off, and where a guard fired
    the guard's route follows (raised)."""

    def _setup(self, deferred, run):
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF import sector_stability as stab
        from parsec_python.acceleration.SCF.noncollinear_stability import LineReport
        from parsec_python.acceleration.driver import _SpinSectorFallback

        points = 8
        grid = SimpleNamespace(size=points, coordinates=np.zeros((points, 3)), volume_element=1.0)
        scf = SimpleNamespace(fermi_temperature_kelvin=0.0, convergence_criterion=1.0e-6, use_plain_residual=False)
        reference = SimpleNamespace(
            grid=grid, atoms=(SimpleNamespace(position=np.zeros(3), symbol="X"),),
            input=SimpleNamespace(scf=scf, periodic_cell=None),
        )
        adapter = SimpleNamespace(doublet_report=None, reducer=SimpleNamespace(reduction=None))
        system = SimpleNamespace(scalar_field_adapter=adapter, reference=reference)
        channel = SimpleNamespace(
            wavefunctions=np.zeros((points, 2)), density=np.ones(points), input_effective_potential=np.zeros(points),
            eigenvalues=np.array([-1.0, 0.0]), occupations=np.array([1.0, 0.0]),
        )
        sectors = SimpleNamespace(
            converged=True, spin=(channel, channel), energies=SimpleNamespace(total=-1.0), hartree_potential=None,
            fermi_level=-0.5,
        )
        report = stab.SpinStabilityReport(largest=5.0, history=(5.0,), ritz=(5.0,), probes=2, seconds=0.1,
                                          vector=np.ones(2 * points), centre=np.zeros(2 * points))
        line = LineReport(reference=-1.0, points=((0.0, -1.0), (0.1, -1.1)), shift=0.1, lowest=-1.1)
        outcome = stab.StabilityOutcome(report=report, line=line, start=(np.zeros(points), np.zeros(points)),
                                        states=(object(), object()), seconds=0.2, setup=0.1)
        step = SimpleNamespace(adopt=lambda states: tuple(states), _centre=object())
        fallback = _SpinSectorFallback.__new__(_SpinSectorFallback)
        fallback.problem = SimpleNamespace(pseudopotentials={}, atoms=())
        patches = [
            mock.patch.object(_SpinSectorFallback, "_full_grid_apart", lambda self, system: object()),
            mock.patch.object(stab, "FullGridSpinStep", lambda *args, **kwargs: step),
            mock.patch.object(stab, "test_sector_state", lambda *args, **kwargs: outcome),
            mock.patch("parsec_python.acceleration.SCF.spin_sectors.atom_group_images", lambda *a, **k: []),
            mock.patch("parsec_python.acceleration.SCF.moving_sectors.antiferromagnetic_labels", lambda p: True),
            mock.patch("parsec_python.SCF.single_point._number_of_states", lambda system: 2),
            mock.patch("parsec_python.acceleration.driver._free_device_pools", lambda: None),
        ]
        return fallback, system, sectors, deferred, patches

    def _call(self, deferred, run):
        from contextlib import ExitStack

        fallback, system, sectors, deferred, patches = self._setup(deferred, run)
        notes = []
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            outcome = fallback._tested_result(system, run, sectors, deferred, callback=notes.append)
        return fallback, sectors, outcome, notes

    def test_an_unconverged_continuation_replaces_nothing(self) -> None:
        from types import SimpleNamespace

        def run(system, **keywords):
            channel = SimpleNamespace(occupations=np.array([1.0, 0.0]))
            return SimpleNamespace(converged=False, energies=SimpleNamespace(total=-2.0), spin=(channel, channel),
                                   iterations=200)

        fallback, sectors, outcome, notes = self._call([], run)
        self.assertIs(outcome, sectors)
        self.assertIn("did not converge", notes[-1].reason)
        self.assertIsNone(fallback._candidate)

    def test_device_memory_in_the_continuation(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        def run(system, **keywords):
            raise MemoryError("out of memory allocating")

        fallback, sectors, outcome, notes = self._call([], run)
        self.assertIs(outcome, sectors)
        self.assertIn("ran out of device memory", notes[-1].reason)
        # where a guard fired: the guard's route (raised), nothing waits for it
        report = FractionalOccupations(largest=0.5, tolerance=0.05, lines=("up",))
        with self.assertRaises(SpinSectorsAbandoned):
            self._call([("the SCF loop", report)], run)

    def test_where_a_guard_fired_a_converged_continuation_waits(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.spin_sectors import FractionalOccupations, SpinSectorsAbandoned

        def run(system, **keywords):
            channel = SimpleNamespace(occupations=np.array([1.0, 0.0]), density=np.ones(8), eigensolver_state=None)
            return SimpleNamespace(converged=True, energies=SimpleNamespace(total=-2.0), spin=(channel, channel),
                                   iterations=30, wavefunctions=np.zeros((8, 4)))

        report = FractionalOccupations(largest=0.5, tolerance=0.05, lines=("up",))
        fallback, system, sectors, deferred, patches = self._setup([("the SCF loop", report)], run)
        from contextlib import ExitStack

        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(mock.patch(
                "parsec_python.acceleration.driver._SpinSectorFallback._moments_text", lambda self, s, r: ("m", True)
            ))
            with self.assertRaises(SpinSectorsAbandoned):
                fallback._tested_result(system, run, sectors, deferred)
        self.assertAlmostEqual(fallback._candidate["free"], -2.0)
        self.assertTrue(fallback._stability_declined)


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        return bool(cupy_available())
    except Exception:  # pragma: no cover
        return False


def _manganese_input() -> str:
    return """Spin_Polarization: .true.
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.40000000 5.40000000 5.40000000
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.22
Expansion_Order: 12
States_Num: 15
Fermi_Temp: 300.0
Max_Iter: 500
Convergence_Criterion: 1.0e-6
Diag_Tolerance: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.2
Correlation_Type: pbe
Atom_Types_Num: 2
Atom_Type: Mn_a
Element_Symbol: Mn
Local_Component: s
Initial_Spin_Polarization: 0.5
begin Atom_Coord
  0.0 0.0 0.0
end Atom_Coord
Atom_Type: Mn_b
Element_Symbol: Mn
Local_Component: s
Initial_Spin_Polarization: -0.5
begin Atom_Coord
  2.7 2.7 2.7
end Atom_Coord
"""


@unittest.skipUnless(_cupy_ready(), "needs a CUDA device")
class RouteTests(unittest.TestCase):
    def test_the_symmetric_state_of_bcc_mn2_is_left_for_a_lower_one(self) -> None:
        """bcc Mn2 with start polarizations +0.5 and -0.5 (PBE, 300 K): the sectors and the full grid keep a symmetric
        state with a half-filled pair of each channel; under auto its test finds it unstable and the loop on the full
        grid continues to a lower converged state."""

        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for label, environment in (
                ("off", {}),
                ("auto", {"PARSEC_SPIN_SECTOR_STABILITY": "auto"}),
                ("full", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
            ):
                directory = Path(name) / label
                directory.mkdir()
                (directory / "parsec.in").write_text(_manganese_input())
                for target in ("Mn_a_POTRE.DAT", "Mn_b_POTRE.DAT", "Mn_POTRE.DAT"):
                    shutil.copyfile(pseudopotential_file("tests/data/spin_pbe_Mn_POTRE.DAT"), directory / target)
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy", "--gpus", "1"])
                log = (directory / "parsec.out").read_text(encoding="utf-8")
                self.assertEqual(status, 0, f"{label}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    runs[label] = (float(data["energy_total_ry"]), log)
        self.assertNotIn("point group of the sectors", runs["off"][1])
        self.assertIn("unstable against perturbations that break their point group", runs["auto"][1])
        self.assertIn("continued from the sectors is kept", runs["auto"][1])
        self.assertIn("moments of the atoms by label", runs["auto"][1])
        self.assertLess(runs["auto"][0], min(runs["off"][0], runs["full"][0]) - 5e-3)
        # parsec.out ends with the block of the kept state
        self.assertAlmostEqual(_last_total_energy(runs["auto"][1]), runs["auto"][0], delta=1e-7)

    def test_the_forces_of_a_kept_state_are_those_of_a_restart_from_it(self) -> None:
        """The NO radical (PBE; a partly filled pi* doublet stops the sectors at the start Hamiltonian, and the full
        grid from the start does not converge within its 200 iterations): under auto the state the loop on the full grid
        converges to is kept, with the forces of its own orbitals, those of a restart of the full grid from its
        parsec.dat."""

        import parsec_python.acceleration.cli as cli_module
        from parsec_python.tests.spin_reference import pseudopotential_file

        runs = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            for label, environment, extra in (
                ("auto", {"PARSEC_SPIN_SECTOR_STABILITY": "auto"}, ""),
                ("restart", {}, "Restart_Run: .true.\nSpin_Symmetry_Sectors: off\n"),
            ):
                directory = Path(name) / label
                directory.mkdir()
                (directory / "parsec.in").write_text(_nitric_oxide_input(extra))
                for target, source in (("N_POTRE.DAT", "spin_pbe_N_POTRE.DAT"), ("O_POTRE.DAT", "spin_pbe_O_POTRE.DAT")):
                    shutil.copyfile(pseudopotential_file(f"tests/data/{source}"), directory / target)
                if label == "restart":
                    shutil.copyfile(Path(name) / "auto" / "parsec.dat", directory / "parsec.dat")
                errors = StringIO()
                with mock.patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(errors):
                    status = cli_module.main([str(directory / "parsec.in"), "--quiet", "--backend", "cupy", "--gpus", "1"])
                log = (directory / "parsec.out").read_text(encoding="utf-8")
                self.assertEqual(status, 0, f"{label}: {errors.getvalue()}\n{log[-3000:]}")
                with np.load(directory / "parsec_python_results.npz", allow_pickle=True) as data:
                    runs[label] = (float(data["energy_total_ry"]), np.asarray(data["forces_total"], float), log)
        self.assertIn("continued from the sectors is kept", runs["auto"][2])
        self.assertAlmostEqual(runs["auto"][0], runs["restart"][0], delta=1e-7)
        np.testing.assert_allclose(runs["auto"][1], runs["restart"][1], atol=1e-5)
        self.assertAlmostEqual(_last_total_energy(runs["auto"][2]), runs["auto"][0], delta=1e-7)


def _last_total_energy(log: str) -> float:
    import re

    return float(re.findall(r"Total Energy =\s+(-?[0-9.]+)", log)[-1])


def _nitric_oxide_input(extra: str = "") -> str:
    return (
        "Boundary_Sphere_Radius: 10.0\nGrid_Spacing: 0.3\nStates_Num: 12\nMax_Iter: 200\nEigensolver: chebff\n"
        "Correlation_Type: pbe\nSpin_Polarization: .true.\nCalculate_Force: .true.\nConvergence_Criterion: 1.0e-8\n"
        + extra
        + "Atom_Types_Num: 2\nAtom_Type: N\nLocal_Component: p\nInitial_Spin_Polarization: 0.3\nbegin Atom_Coord\n"
        "  0.0 0.0 -1.085\nend Atom_Coord\nAtom_Type: O\nLocal_Component: d\nInitial_Spin_Polarization: 0.3\n"
        "begin Atom_Coord\n  0.0 0.0 1.085\nend Atom_Coord\n"
    )


if __name__ == "__main__":
    unittest.main()
