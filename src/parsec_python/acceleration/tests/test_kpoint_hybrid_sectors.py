"""The Fock loop of a hybrid with k-points in the sectors of its k-points (``acceleration/SCF/kpoint_sectors.py``).

* ``PARSEC_KPOINT_SECTOR_LEVELS``: the last inner loop of the Fock loop with refined levels (the adaptive inner
  criterion after a loose start) is solved in the sectors, started from the Ritz states of the loop before, and gives
  the energy of the loop with one task per k-point (``off``, as before).
* ``PARSEC_KPOINT_SECTOR_GUARD_FOCK``: the inner loops of a spin-polarized hybrid in the sectors run tests 2 and 3 of
  the guard at their end.  A passing guard changes no number (it only measures, ``parsec.out`` prints what it
  measured); an inner loop at the final criterion that fails one starts the Fock loop again from its semilocal start
  with one task per k-point (the run without sectors); a looser one is measured only; ``off`` tests the semilocal
  start alone.
* ``PARSEC_HYBRID_KPOINT_WEDGE``: the irreducible wedge of a hybrid's mesh, its other points the images of the
  wedge's Bloch factors under the stencil-exact operations that keep the mesh (also on the Fortran-derived routines,
  whose density average takes the whole space group, and on oblique grids), with and without the k-point sectors
  (``on``); ``auto``: the semilocal start on the time-reversal pairs, the Fock loop on the wedge where the start has
  no partly filled state, the pairs bit for bit as ``off`` where it has one.
* The note of a ``Monkhorst_Pack_Grid`` without a shift block says what a hybrid's mesh is: its wedge, or its
  time-reversal pairs and why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.SCF import kpoint_sectors as sectors

_ROOT = Path(__file__).resolve().parents[4]
_HYBRID_EXAMPLES = _ROOT / "examples" / "3d_hybrid_functionals"


def setUpModule() -> None:
    # the cells of these checks are small: split them all the same
    patcher = mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def _silicon(**scf):
    from parsec_python.models import ExactExchangeSettings
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = silicon("pbe0", spacing=0.855)
    changes = dict(
        exact_exchange=ExactExchangeSettings(tolerance=1.0e-8), convergence_criterion=1.0e-8, max_iterations=120
    )
    changes.update(scf)
    return replace(
        base, scf=replace(base.scf, **changes), eigensolver=replace(base.eigensolver, tolerance=1.0e-10)
    )


def _run(problem, k_points, weights, environment, callback=None):
    from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf
    from parsec_python.SCF.pbc import prepare_periodic_single_point

    with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = prepare_periodic_single_point(problem)
        return run_periodic_hybrid_scf(system, k_points, weights, callback=callback)


class SwitchTests(unittest.TestCase):
    def test_the_switches(self) -> None:
        for name, function in (
            ("PARSEC_KPOINT_SECTOR_LEVELS", sectors.sector_levels_setting),
            ("PARSEC_KPOINT_SECTOR_GUARD_FOCK", sectors.sector_guard_fock_setting),
        ):
            for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
                with mock.patch.dict(os.environ, {name: value}):
                    self.assertEqual(function(), expected)
            with mock.patch.dict(os.environ, {name: "on"}):
                with self.assertRaises(ValueError):
                    function()

    def test_a_bad_value_stops_the_hybrid_before_its_first_loop(self) -> None:
        from parsec_python.SCF import hybrid_periodic

        problem = _silicon()
        for name in ("PARSEC_KPOINT_SECTOR_LEVELS", "PARSEC_KPOINT_SECTOR_GUARD_FOCK", "PARSEC_HYBRID_KPOINT_WEDGE"):
            with self.subTest(name=name), mock.patch.object(hybrid_periodic, "BlochExchange") as engine:
                with self.assertRaises(ValueError):
                    _run(problem, np.zeros((1, 3)), np.ones(1), {name: "yes"})
                engine.assert_not_called()


class LevelPassTests(unittest.TestCase):
    def test_the_level_passes_in_the_sectors(self) -> None:
        # Gamma and X (a 2 x 1 x 1 mesh, both their own partners): the unpolarized start stops early, so the last
        # inner loop refines its levels; under auto its solver is that of the sectors, under off the plain one
        from parsec_python.SCF import kpoints

        problem = _silicon()
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        weights = np.array([0.5, 0.5])
        original = kpoints._refined_levels
        results, solvers = {}, {}
        for setting in ("off", "auto"):
            seen: list[str] = []

            def spy(solver, *args, **kwargs):
                seen.append(type(solver).__name__)
                return original(solver, *args, **kwargs)

            with mock.patch.object(kpoints, "_refined_levels", spy):
                results[setting] = _run(
                    problem, k_points, weights,
                    {"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_SECTOR_LEVELS": setting},
                )
            solvers[setting] = seen
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertTrue(solvers["auto"] and solvers["off"], solvers)
        self.assertEqual(set(solvers["auto"]), {"SectorKPointSolver"})
        self.assertNotIn("SectorKPointSolver", solvers["off"])
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=2.0e-8)
        self.assertAlmostEqual(
            auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6
        )
        # The levels: the loop on whole k-points splits two partners at X by 3.4e-5 Ry (A100, 2026-11-19), which the
        # sectors hold degenerate (their orbitals are symmetry adapted); the sums of the levels agree.
        np.testing.assert_allclose(auto.eigenvalues, off.eigenvalues, rtol=0.0, atol=5.0e-5)
        self.assertAlmostEqual(float(np.sum(auto.eigenvalues)), float(np.sum(off.eigenvalues)), delta=1.0e-6)
        # the loops of before took the sectors under either value
        self.assertGreater(off.timings.detail["kpoint_sector_tasks"], 2.0)


class FockGuardTests(unittest.TestCase):
    """H2 in a periodic box of 6 bohr at the Gamma point, spin-polarized PBE0 (the 1x1x1 Bloch loop), 40 states per
    channel so that the sectors of the 8 operations about the bond centre halve the work: every operation of the
    sectors is assumed (no symmetrized density), so the guard tests the loops; the closed shell passes every test."""

    def problem(self):
        from parsec_python.acceleration.tests.test_periodic_hybrid_device import _h2_problem

        problem = _h2_problem("pbe0", spin=True, outer_mixing="diis")
        return replace(problem, scf=replace(problem.scf, number_of_states=40))

    def run_h2(self, environment, callback=None):
        return _run(self.problem(), np.zeros((1, 3)), np.ones(1), environment, callback=callback)

    @staticmethod
    def _made_up_failure():
        """Patches that make test 2 fail in every inner loop of the Fock loop (the job of a loop with the exchange
        sums the exchange energy on the sectors, ``exchange_cross_scale``) and leave the semilocal start alone."""

        from parsec_python.acceleration.SCF.spin_sectors import DENSITY_SYMMETRY_TOLERANCE, FractionalOccupations

        measure = sectors.SectorKPointSolver.density_symmetry_measure
        report = sectors.SectorKPointSolver.density_symmetry_report

        def in_fock_loop(solver) -> bool:
            return solver.job.geometry.get("exchange_cross_scale") is not None

        def made_up_measure(self, densities, volume_element):
            return 0.5 if in_fock_loop(self) else measure(self, densities, volume_element)

        def made_up_report(self, densities, volume_element, tolerance=None):
            if not in_fock_loop(self):
                return report(self, densities, volume_element, tolerance)
            return FractionalOccupations(
                largest=0.5,
                tolerance=DENSITY_SYMMETRY_TOLERANCE,
                lines=("channel 1: made up",),
                reason="a channel density that breaks the operations the sectors assume",
                measure="largest int |rho(g r) - rho(r)| [e]",
            )

        return (
            mock.patch.object(sectors.SectorKPointSolver, "density_symmetry_measure", made_up_measure),
            mock.patch.object(sectors.SectorKPointSolver, "density_symmetry_report", made_up_report),
        )

    def test_a_guard_that_passes_changes_nothing(self) -> None:
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.acceleration.SCF.spin_sectors import DENSITY_SYMMETRY_TOLERANCE

        on = self.run_h2({"PARSEC_KPOINT_SECTORS": "auto"}, callback=lambda record: None)
        off = self.run_h2(
            {"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_SECTOR_GUARD_FOCK": "off"}, callback=lambda record: None
        )
        self.assertTrue(on.converged and off.converged)
        self.assertNotIn("kpoint_sector_abandoned", on.timings.detail)
        self.assertGreater(on.timings.detail["kpoint_sector_tasks"], 2.0)
        # tests that only measure: the same loop
        self.assertAlmostEqual(on.energies.total, off.energies.total, delta=1.0e-12)
        self.assertEqual(on.iterations, off.iterations)
        detail = on.timings.detail
        self.assertGreaterEqual(detail["kpoint_guard_fock_loops"], 1.0)
        self.assertEqual(detail["kpoint_guard_fock_exact"], 1.0)  # the last inner loop is at the final criterion
        self.assertLess(detail["kpoint_guard_fock_density_asymmetry"], DENSITY_SYMMETRY_TOLERANCE)
        self.assertEqual(detail["kpoint_guard_fock_tightenings"], 0.0)
        self.assertIn("kpoint_guard_fock_fractional_share", detail)
        self.assertFalse([key for key in off.timings.detail if key.startswith("kpoint_guard_fock")])
        lines = ParsecTextReporter._kpoint_detail_lines(on.timings)
        for label in ("Test 2, last Fock loop", "Test 3, last Fock loop"):
            self.assertTrue(any(line.lstrip().startswith(label) for line in lines), (label, lines))

    def test_a_failing_inner_loop_starts_the_fock_loop_again_without_sectors(self) -> None:
        from parsec_python.Output.parsec_output import spin_sector_fallback_lines
        from parsec_python.models import SpinSectorFallbackRecord

        whole = self.run_h2({"PARSEC_KPOINT_SECTORS": "off"}, callback=lambda record: None)
        records: list = []
        first, second = self._made_up_failure()
        with first, second:
            result = self.run_h2({"PARSEC_KPOINT_SECTORS": "auto"}, callback=records.append)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].route, "kpoint")
        self.assertEqual(notes[0].stage, "an inner loop of the Fock loop at the final criterion")
        self.assertIn("semilocal start", notes[0].multiplets[-1])
        self.assertIn("NOTE: k-point sectors", "\n".join(spin_sector_fallback_lines(notes[0])))
        detail = result.timings.detail
        self.assertTrue(result.converged)
        self.assertEqual(detail.get("kpoint_sector_abandoned"), 1.0)
        self.assertNotIn("kpoint_sector_tasks", detail)
        self.assertEqual(detail["kpoint_guard_fock_density_asymmetry"], 0.5)
        self.assertGreaterEqual(detail["kpoint_guard_fock_loops"], 1.0)
        # the Fock loop with one task per k-point from its start: the run without sectors (thread sums to round-off)
        self.assertAlmostEqual(result.energies.total, whole.energies.total, delta=1.0e-9)
        self.assertEqual(
            [record.inner_iterations for record in result.exact_exchange.iterations],
            [record.inner_iterations for record in whole.exact_exchange.iterations],
        )
        self.assertEqual(result.iterations, whole.iterations)

    def test_off_tests_the_semilocal_start_alone(self) -> None:
        from parsec_python.models import SpinSectorFallbackRecord

        records: list = []
        first, second = self._made_up_failure()
        with first, second:
            result = self.run_h2(
                {"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_SECTOR_GUARD_FOCK": "off"}, callback=records.append
            )
        self.assertFalse([record for record in records if isinstance(record, SpinSectorFallbackRecord)])
        self.assertNotIn("kpoint_sector_abandoned", result.timings.detail)
        self.assertGreater(result.timings.detail["kpoint_sector_tasks"], 2.0)
        self.assertFalse([key for key in result.timings.detail if key.startswith("kpoint_guard_fock")])

    def test_a_looser_inner_loop_is_measured_only(self) -> None:
        # The same made-up failure on the loops of a relative criterion only: no inner loop at the final criterion
        # fails, so the Fock loop keeps its sectors, and the loops it measured are counted.
        from parsec_python.SCF import kpoints
        from parsec_python.models import SpinSectorFallbackRecord

        records: list = []
        first, second = self._made_up_failure()
        exact = []
        original = kpoints._spin_kpoint_loop

        def loose_only(*args, **kwargs):
            warm_start = kwargs.get("warm_start")
            if kwargs.get("exchange_term") is not None and getattr(warm_start, "relative_criterion", None) is None:
                exact.append(True)
                with mock.patch.object(sectors.SectorKPointSolver, "density_symmetry_measure", lambda *a: 0.0), \
                        mock.patch.object(sectors.SectorKPointSolver, "density_symmetry_report", lambda *a, **k: None):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        with first, second, mock.patch.object(kpoints, "_spin_kpoint_loop", loose_only):
            result = self.run_h2({"PARSEC_KPOINT_SECTORS": "auto"}, callback=records.append)
        self.assertFalse([record for record in records if isinstance(record, SpinSectorFallbackRecord)])
        self.assertTrue(exact)
        detail = result.timings.detail
        self.assertNotIn("kpoint_sector_abandoned", detail)
        loose = len(result.exact_exchange.iterations) - 1 - len(exact)
        if loose > 0:
            # the loops of a relative criterion were measured too (the last tested one is exact and passed)
            self.assertGreater(detail["kpoint_guard_fock_loops"], float(len(exact)))
        self.assertEqual(detail["kpoint_guard_fock_exact"], 1.0)


def _silicon_mesh(
    functional: str = "pbe0", wedge: str = "auto", extra: dict | None = None, mesh: str = "2 2 2",
    shift_block: bool = False, added: tuple = (),
):
    """The silicon cube of 8 atoms (default routines) with the Gamma-centred 2 x 2 x 2 mesh (no shift block);
    ``extra``: other keywords of the template set to other values; ``mesh``: another Monkhorst_Pack_Grid;
    ``shift_block``: a zero Monkhorst_Pack_Shift block (the Fortran-derived routines, whose density average takes the
    whole space group for any mesh); ``added``: lines added to the input."""

    from parsec_python.Input.parsec_input import parse_parsec_input
    from parsec_python.acceleration.tests.test_periodic_point_group import PERIODIC_SILICON

    changes = {
        "Grid_Spacing": "0.855",
        "Expansion_Order": "8",
        "States_Num": "22",
        "Convergence_Criterion": "1.0e-9",
        "Diag_Tolerance": "1.0e-10",
        "Correlation_Type": functional,
        "Fermi_Temp": "1.0",
        **(extra or {}),
    }
    lines = []
    for line in PERIODIC_SILICON.read_text().splitlines():
        keyword = line.split(":", 1)[0].strip()
        lines.append(f" {keyword}: {changes[keyword]}" if ":" in line and keyword in changes else line)
    lines += ["Kpoint_Method: mp", "begin Monkhorst_Pack_Grid", mesh, "end Monkhorst_Pack_Grid", *added]
    if shift_block:
        lines += ["begin Monkhorst_Pack_Shift", "0.0 0.0 0.0", "end Monkhorst_Pack_Shift"]
    directory = Path(tempfile.mkdtemp(prefix="parsec_hybwedge_"))
    shutil.copy(PERIODIC_SILICON.parent / "Si_POTRE.DAT", directory)
    (directory / "parsec.in").write_text("\n".join(lines) + "\n")
    with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        translation = parse_parsec_input(directory / "parsec.in")
    return directory, translation


class HybridWedgeTests(unittest.TestCase):
    """``PARSEC_HYBRID_KPOINT_WEDGE``: the irreducible wedge of a hybrid's mesh, the rest of the mesh reached through
    the images of its Bloch factors (``V_xc/bloch_exchange.py``, ``MeshStars``)."""

    def setUp(self) -> None:
        # the cube is small: no k-point sectors here (they are tested above), the plain solver of the k-points
        patcher = mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def mesh_note(translation) -> str:
        found = [note for note in translation.warnings if note.startswith("Monkhorst_Pack_Grid without")]
        assert len(found) == 1, found
        return found[0]

    def problems(self, values=("on", "off")):
        found = {}
        for wedge in values:
            directory, translation = _silicon_mesh("pbe0", wedge)
            self.addCleanup(shutil.rmtree, directory, True)
            found[wedge] = translation
        return found

    def test_the_wedge_of_the_mesh(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        found = self.problems(("on", "auto", "off"))
        on, auto, off = found["on"].problem, found["auto"].problem, found["off"].problem
        self.assertTrue(on.hybrid_kpoint_wedge)
        self.assertTrue(on.scf.symmetrize_density)
        self.assertIsNone(on.hybrid_kpoint_wedge_deferred)
        self.assertFalse(off.hybrid_kpoint_wedge)
        self.assertIsNone(off.hybrid_kpoint_wedge_deferred)
        # auto: the input of off, the wedge kept aside for the Fock loop of an insulating start
        self.assertFalse(auto.hybrid_kpoint_wedge)
        self.assertEqual(auto.scf.symmetrize_density, off.scf.symmetrize_density)
        self.assertIsNone(auto.space_group)
        self.assertEqual(auto.hybrid_kpoint_wedge_deferred.group.order, on.space_group.order)
        self.assertEqual(auto.hybrid_kpoint_wedge_deferred.dimensions, (2, 2, 2))
        k_on, w_on = kpoint_sampling(on)
        k_auto, w_auto = kpoint_sampling(auto)
        k_off, w_off = kpoint_sampling(off)
        self.assertEqual(len(k_off), 8)
        self.assertEqual(len(k_on), 4)
        np.testing.assert_array_equal(k_auto, k_off)
        np.testing.assert_array_equal(w_auto, w_off)
        np.testing.assert_allclose(np.sort(w_on * 8), [1.0, 1.0, 3.0, 3.0])
        notes = [note for note in found["on"].warnings if note.startswith("Monkhorst_Pack_Grid without")]
        self.assertTrue(notes and "through the images of the wedge" in notes[0], notes)
        notes = [note for note in found["auto"].warnings if note.startswith("Monkhorst_Pack_Grid without")]
        self.assertTrue(notes and "for the semilocal start of the hybrid" in notes[0], notes)
        notes = [note for note in found["off"].warnings if note.startswith("Monkhorst_Pack_Grid without")]
        self.assertTrue(notes and "time reversal only" in notes[0], notes)
        self.assertIn("PARSEC_HYBRID_KPOINT_WEDGE=off keeps the time-reversal pairs", notes[0])
        # forces under PARSEC_HYBRID_KPOINT_FORCE_WEDGE=off keep the pairs of the mesh, and the note says so; under
        # auto (the default) they take the route of the single point (tests/unit/test_hybrid_force_wedge.py)
        for wedge in ("on", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_FORCE_WEDGE": "off"}):
                directory, forces = _silicon_mesh("pbe0", wedge, {"Skip_force": ".false."})
            self.addCleanup(shutil.rmtree, directory, True)
            self.assertFalse(forces.problem.hybrid_kpoint_wedge)
            self.assertIsNone(forces.problem.hybrid_kpoint_wedge_deferred)
            self.assertEqual(len(kpoint_sampling(forces.problem)[0]), 8)
            self.assertEqual(self.mesh_note(forces).count("forces and moving atoms keep the time-reversal pairs"), 1)
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_FORCE_WEDGE": "auto"}):
                directory, forces = _silicon_mesh("pbe0", wedge, {"Skip_force": ".false."})
            self.addCleanup(shutil.rmtree, directory, True)
            self.assertEqual(forces.problem.hybrid_kpoint_wedge, wedge == "on")
            self.assertEqual(forces.problem.hybrid_kpoint_wedge_deferred is not None, wedge == "auto")
            self.assertEqual(len(kpoint_sampling(forces.problem)[0]), 4 if wedge == "on" else 8)
        # no symmetry: the pairs, for that reason
        directory, alone = _silicon_mesh("pbe0", "auto", added=("Ignore_Symmetry: .true.",))
        self.addCleanup(shutil.rmtree, directory, True)
        self.assertFalse(alone.problem.hybrid_kpoint_wedge)
        self.assertIsNone(alone.problem.hybrid_kpoint_wedge_deferred)
        self.assertEqual(len(kpoint_sampling(alone.problem)[0]), 8)
        self.assertIn("or the symmetry is off (Ignore_Symmetry", self.mesh_note(alone))
        from parsec_python.Input.parsec_input import _UNSHIFTED_HYBRID_PAIRS_NOTE

        for translation in (*found.values(), forces, alone):
            self.assertNotIn(_UNSHIFTED_HYBRID_PAIRS_NOTE, translation.warnings)
        # a semilocal run is not touched by the switch
        for wedge in ("on", "auto", "off"):
            directory, semilocal = _silicon_mesh("pbe", wedge)
            self.addCleanup(shutil.rmtree, directory, True)
            self.assertFalse(semilocal.problem.hybrid_kpoint_wedge)
            self.assertIsNone(semilocal.problem.hybrid_kpoint_wedge_deferred)
            self.assertEqual(len(kpoint_sampling(semilocal.problem)[0]), 4)
        for value in ("bad", "grid"):
            with self.assertRaises(ValueError):
                _silicon_mesh("pbe0", value)

    def test_the_images_give_the_exchange_of_the_mesh(self) -> None:
        # W = V_x U of the wedge's k-points from the images of their orbitals equals W from the orbitals solved at
        # every point of the mesh (a semilocal SCF on the eight points, labelled as the images are): the operator
        # depends on the occupied subspace of each point, which the images span to the eigensolver's tolerance
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.SCF.hybrid_periodic import hybrid_mesh_stars, periodic_pair_potentials
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.V_xc.bloch_exchange import GridBox, apply_bloch_exchange, bloch_orbital_set

        problem = self.problems()["on"].problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
        k_points, weights = kpoint_sampling(problem)
        box = GridBox(system.grid)
        stars = hybrid_mesh_stars(system, k_points, weights, box)
        self.assertEqual(stars.mesh_size, 8)
        self.assertEqual(len(stars.sources), 8)
        self.assertGreater(len(stars.gathers), 0)
        np.testing.assert_array_equal(np.sort(stars.point_counts), [1, 1, 3, 3])
        labels = np.array([source[3] for source in stars.sources])
        semilocal = replace(problem, scf=replace(problem.scf, xc_functional="pbe", exact_exchange=None, symmetrize_density=False),
                            hybrid_kpoint_wedge=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = run_scf_kpoints(
                prepare_periodic_single_point(semilocal), labels, np.full(8, 0.125), keep_wavefunctions=True
            )
        self.assertTrue(reference.converged)
        blocks = list(reference.k_wavefunctions)
        occupations = np.split(np.asarray(reference.occupations), np.cumsum([b.shape[1] for b in blocks])[:-1])
        full = bloch_orbital_set(labels, np.full(8, 0.125), blocks, occupations, 2.0, box)
        own = [index for index, source in enumerate(stars.sources) if source[1] < 0 and not source[2]]
        self.assertEqual([stars.sources[index][0] for index in own], list(range(len(k_points))))
        np.testing.assert_allclose(labels[own], k_points, atol=1.0e-12)
        wedge = bloch_orbital_set(
            k_points, weights, [blocks[index] for index in own], [occupations[index] for index in own], 2.0, box, stars
        )
        potentials = periodic_pair_potentials(system, hybrid_definition(problem.scf), k_mesh=(2, 2, 2))
        whole = apply_bloch_exchange(full, potentials, columns="all")
        reduced = apply_bloch_exchange(wedge, potentials, columns="all")
        for rep, index in enumerate(own):
            scale = float(np.max(np.abs(whole.applied[index])))
            np.testing.assert_allclose(reduced.applied[rep], whole.applied[index], rtol=0.0, atol=1.0e-6 * scale)
        self.assertAlmostEqual(reduced.energy, whole.energy, delta=1.0e-8 * abs(whole.energy))
        # every point of the mesh once: the pair solves of the wedge are those of its targets with the whole mesh
        self.assertLess(reduced.pair_solves, whole.pair_solves)

    def test_the_hybrid_scf_on_the_wedge(self) -> None:
        self.assert_same_hybrid(spin_polarized=False)

    def test_the_spin_polarized_hybrid_scf_on_the_wedge(self) -> None:
        self.assert_same_hybrid(spin_polarized=True)

    def assert_same_hybrid(self, *, spin_polarized: bool) -> None:
        # on: the wedge from the start; auto: the start on the pairs, then the wedge where the start has no partly
        # filled state (the unpolarized cube at 1 K), else the pairs bit for bit as off (whichever the polarized start
        # gives: its initial moment may leave a multiplet partly filled)
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.models import HybridMeshRouteRecord

        results, counts, routes = {}, {}, {}
        for wedge, translation in self.problems(("on", "auto", "off")).items():
            problem = replace(
                translation.problem,
                scf=replace(translation.problem.scf, convergence_criterion=1.0e-8, spin_polarized=spin_polarized),
            )
            k_points, weights = kpoint_sampling(problem)
            records: list = []
            results[wedge] = _run(problem, k_points, weights, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}, records.append)
            counts[wedge] = len(k_points)
            routes[wedge] = [record for record in records if isinstance(record, HybridMeshRouteRecord)]
        on, auto, off = results["on"], results["auto"], results["off"]
        self.assertEqual((counts["on"], counts["auto"], counts["off"]), (4, 8, 8))
        self.assertTrue(on.converged and auto.converged and off.converged)
        self.assertEqual((routes["on"], routes["off"]), ([], []))
        self.assertEqual(len(routes["auto"]), 1)
        self.assertIn("irreducible wedge", on.exact_exchange.pair_solver)
        self.assertAlmostEqual(on.energies.total, off.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(on.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6)
        route = routes["auto"][0]
        if not spin_polarized:
            self.assertEqual(route.route, "wedge")
        if route.route == "wedge":
            self.assertEqual((route.points, route.pairs_points, route.mesh_points), (4, 8, 8))
            self.assertEqual(len(auto.k_points), 4)
            self.assertIn("irreducible wedge", auto.exact_exchange.pair_solver)
            self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-7)
            self.assertAlmostEqual(
                auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6
            )
        else:
            self.assertGreater(route.partly_filled, 0)
            self.assertEqual(len(auto.k_points), 8)
            # the run of off (the same objects handed on, AutoRouteTests), to the run-to-run spread of the device
            self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-7)
            self.assertAlmostEqual(
                auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6
            )


def _cubic_matrices() -> np.ndarray:
    """The 48 signed permutation matrices (the point group of a cube, fractional = Cartesian), the identity first."""

    import itertools

    matrices = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            matrices.append(matrix)
    return np.array(matrices)


class MeshStarsTests(unittest.TestCase):
    """``mesh_stars`` takes only the operations that keep the mesh: the exchange operator of the mesh commutes with
    those alone, and the density average of the Fortran-derived routines hands it the whole space group for any
    mesh (a 2 x 2 x n mesh of a cube, n != 2, keeps the 16 operations that keep the z axis)."""

    def test_only_the_operations_that_keep_the_mesh(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Grid.kpoints import irreducible_monkhorst_pack
        from parsec_python.V_xc.bloch_exchange import mesh_stars

        lattice = np.eye(3) * 10.26
        reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
        matrices = _cubic_matrices()
        symmetry = SimpleNamespace(matrices=matrices, images=np.tile(np.arange(8), (len(matrices), 1)))
        shift = (0.0, 0.0, 0.0)
        for mesh in ((2, 2, 1), (2, 2, 3), (2, 2, 4), (2, 2, 2)):
            keeps = [
                index for index, matrix in enumerate(matrices) if mesh == (2, 2, 2) or abs(matrix[2, 2]) == 1.0
            ]
            self.assertEqual(len(keeps), 48 if mesh == (2, 2, 2) else 16)
            rotations = np.array([matrices[index].T for index in keeps]).astype(np.int64)
            sampling = irreducible_monkhorst_pack(lattice, mesh, shift, rotations, time_reversal=True)
            stars = mesh_stars(sampling.k_points, sampling.weights, mesh, shift, reciprocal, symmetry)
            self.assertEqual(stars.mesh_size, int(np.prod(mesh)))
            self.assertEqual(len(stars.sources), stars.mesh_size)
            used = {source[1] for source in stars.sources if source[1] >= 0}
            self.assertTrue(used <= set(keeps), (mesh, sorted(used - set(keeps))))
            np.testing.assert_allclose(stars.point_counts / float(stars.mesh_size), sampling.weights, atol=1.0e-12)
            if mesh != (2, 2, 2):
                # the wedge of the whole group is no wedge of the mesh's exchange: refused, with the counts
                whole = irreducible_monkhorst_pack(
                    lattice, mesh, shift, np.array([m.T for m in matrices]).astype(np.int64), time_reversal=True
                )
                if len(whole.weights) != len(sampling.weights):
                    with self.assertRaisesRegex(NotImplementedError, "points per k-point"):
                        mesh_stars(whole.k_points, whole.weights, mesh, shift, reciprocal, symmetry)

    def test_a_mesh_time_reversal_does_not_keep(self) -> None:
        from parsec_python.Grid.kpoints import mesh_keeps_time_reversal

        self.assertTrue(mesh_keeps_time_reversal((0.0, 0.0, 0.0)))
        self.assertTrue(mesh_keeps_time_reversal((0.5, 0.0, 0.5)))
        self.assertTrue(mesh_keeps_time_reversal((1.0, 0.5, 0.0)))
        self.assertFalse(mesh_keeps_time_reversal((0.25, 0.0, 0.0)))
        # a hybrid on such a mesh keeps the pairs (the exchange of the pairs sums over the mesh and its mirror image)
        directory, translation = _silicon_mesh("pbe0", "on", shift_block=True)
        self.addCleanup(shutil.rmtree, directory, True)
        self.assertTrue(translation.problem.hybrid_kpoint_wedge)
        shifted = (Path(directory) / "parsec.in").read_text().replace("0.0 0.0 0.0\nend Monkhorst_Pack_Shift",
                                                                       "0.25 0.25 0.25\nend Monkhorst_Pack_Shift")
        (Path(directory) / "parsec.in").write_text(shifted)
        from parsec_python.Input.parsec_input import parse_parsec_input

        for wedge in ("on", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                quarter = parse_parsec_input(Path(directory) / "parsec.in")
            self.assertEqual(tuple(quarter.problem.monkhorst_pack_shift), (0.25, 0.25, 0.25))
            self.assertFalse(quarter.problem.hybrid_kpoint_wedge)
            self.assertIsNone(quarter.problem.hybrid_kpoint_wedge_deferred)


class FortranRoutinesWedgeTests(unittest.TestCase):
    """The wedge of a hybrid on the Fortran-derived routines for meshes the whole space group does not keep (the
    cube with 2 x 2 x 1 and 2 x 2 x 4 meshes, a zero Monkhorst_Pack_Shift block): the density average takes the 48
    operations, the wedge and the images of the exchange the 16 that keep the mesh, and the energies are those of
    the time-reversal pairs (before 2026-11-19 evening the images took any operation: the 2 x 2 x 1 total was off by
    6.1e-5 Ry at 0.57 bohr, the 2 x 2 x 4 run stopped).

    These meshes put some images on the other sign of a zone-boundary label (-1/2 where the pairs take +1/2): the
    finite-difference Bloch operator is not invariant under k -> k + G, so the two routes differ by the
    discretization there (A100, Exx_Tolerance 1e-8 Ry: 2 x 2 x 1 7.2e-7 Ry at 0.855 bohr and order 8, 4e-9 Ry at
    0.57 bohr and order 12; 2 x 2 x 4 5.5e-8 Ry at 0.57 bohr, unchanged at tighter tolerances).  The energies are
    compared on the finer grid."""

    MESHES = ("2 2 1", "2 2 4")
    FINE = {"Grid_Spacing": "0.57", "Expansion_Order": "12", "States_Num": "24"}

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def problem(self, mesh: str, wedge: str, extra: dict | None = None):
        directory, translation = _silicon_mesh("pbe0", wedge, extra=extra, mesh=mesh, shift_block=True)
        self.addCleanup(shutil.rmtree, directory, True)
        return translation.problem

    def test_the_images_keep_the_mesh(self) -> None:
        from parsec_python.Grid.kpoints import hybrid_wedge_group, kpoint_sampling
        from parsec_python.SCF.hybrid_periodic import hybrid_mesh_stars
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.V_xc.bloch_exchange import GridBox

        for mesh in self.MESHES:
            problem = self.problem(mesh, "on")
            self.assertTrue(problem.hybrid_kpoint_wedge)
            self.assertEqual(problem.space_group.order, 48)
            kept = hybrid_wedge_group(problem)
            self.assertEqual(kept.order, 16)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
            self.assertEqual(len(system.density_symmetry.matrices), 48)
            k_points, weights = kpoint_sampling(problem)
            stars = hybrid_mesh_stars(system, k_points, weights, GridBox(system.grid))
            self.assertEqual(stars.mesh_size, int(np.prod([int(value) for value in mesh.split()])))
            # the operations of the images (indices of the density average = of the space group) are the wedge's
            keeps = {matrix.tobytes() for matrix in np.asarray(kept.rotation_matrices, dtype=np.int64)}
            whole = np.asarray(problem.space_group.rotation_matrices, dtype=np.int64)
            used = {int(source[1]) for source in stars.sources if source[1] >= 0}
            self.assertTrue(used)
            self.assertTrue(all(whole[index].tobytes() in keeps for index in used), (mesh, sorted(used)))

    def test_the_energies_of_the_pairs(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.models import ExactExchangeSettings

        for mesh in self.MESHES:
            results, counts = {}, {}
            for wedge in ("on", "off"):
                problem = self.problem(mesh, wedge, self.FINE)
                # the Fock loops to 1e-8 Ry (the default Exx_Tolerance of 1e-6 Ry left 7.2e-7 Ry between the two
                # 2 x 2 x 1 totals on A100)
                exchange = problem.scf.exact_exchange
                exchange = (
                    ExactExchangeSettings(tolerance=1.0e-8) if exchange is None else replace(exchange, tolerance=1.0e-8)
                )
                problem = replace(
                    problem, scf=replace(problem.scf, convergence_criterion=1.0e-8, exact_exchange=exchange)
                )
                k_points, weights = kpoint_sampling(problem)
                counts[wedge] = len(k_points)
                results[wedge] = _run(problem, k_points, weights, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge})
            auto, off = results["on"], results["off"]
            self.assertLess(counts["on"], counts["off"], mesh)
            self.assertTrue(auto.converged and off.converged, mesh)
            self.assertIn("irreducible wedge", auto.exact_exchange.pair_solver)
            self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-7, msg=mesh)
            # E_x is first order in the orbitals, the total second order: the 2 x 2 x 4 mesh has a flat direction
            # along which runs of the pairs alone end with E_x 5.4e-5 Ry apart and the same total to 1e-9 Ry (A100,
            # 2026-11-19); the total is the test, E_x only within that spread
            self.assertAlmostEqual(
                auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=2.0e-4, msg=mesh
            )


class WedgeSectorTests(unittest.TestCase):
    """The default route of a large cell: the wedge of the mesh solved in the k-point sectors (stars, the weights of
    the stars in the outer DIIS, ACE of orbitals expanded from the sectors, the level passes in the sectors) against
    the time-reversal pairs without sectors, on the cube with the 2 x 2 x 2 mesh (the sector floor is zero here)."""

    def compare(self, *, spin_polarized: bool) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling

        environments = {
            "on": {"PARSEC_HYBRID_KPOINT_WEDGE": "on", "PARSEC_KPOINT_SECTORS": "auto"},
            "off": {"PARSEC_HYBRID_KPOINT_WEDGE": "off", "PARSEC_KPOINT_SECTORS": "off"},
        }
        if spin_polarized:
            # the start of the spin-polarized cube fails test 1 of the guard (a partly filled multiplet of the
            # polarized start), which would drop the sectors: off keeps them, as the combination is what is tested
            environments["on"]["PARSEC_KPOINT_SECTOR_GUARD"] = "off"
        results, counts = {}, {}
        for wedge, environment in environments.items():
            directory, translation = _silicon_mesh("pbe0", wedge)
            self.addCleanup(shutil.rmtree, directory, True)
            problem = replace(
                translation.problem,
                scf=replace(translation.problem.scf, convergence_criterion=1.0e-8, spin_polarized=spin_polarized),
            )
            k_points, weights = kpoint_sampling(problem)
            counts[wedge] = len(k_points)
            results[wedge] = _run(problem, k_points, weights, environment)
        auto, off = results["on"], results["off"]
        self.assertEqual((counts["on"], counts["off"]), (4, 8))
        self.assertTrue(auto.converged and off.converged)
        self.assertIn("irreducible wedge", auto.exact_exchange.pair_solver)
        self.assertGreater(auto.timings.detail.get("kpoint_sector_tasks", 0.0), 4.0)
        self.assertNotIn("kpoint_sector_tasks", off.timings.detail)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(
            auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6
        )

    def test_unpolarized(self) -> None:
        self.compare(spin_polarized=False)

    def test_spin_polarized(self) -> None:
        self.compare(spin_polarized=True)


def _cupy_available() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available
    except ImportError:  # pragma: no cover
        return False
    return bool(cupy_available())


@unittest.skipUnless(_cupy_available(), "CuPy and a CUDA device are required")
class DeviceWedgeTests(unittest.TestCase):
    """The device engine of the Fock loop on the wedge (the images of the representatives' rows gathered on every
    device): the host engine's loop on the same device workers, unpolarized and spin-polarized."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def compare(self, *, spin_polarized: bool) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF import hybrid_periodic
        from parsec_python.SCF.hybrid import hybrid_definition, run_hybrid_scf
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.SCF.hybrid_periodic import DeviceBlochExchange, device_bloch_engine

        directory, translation = _silicon_mesh("pbe0", "on")
        self.addCleanup(shutil.rmtree, directory, True)
        problem = replace(
            translation.problem,
            scf=replace(translation.problem.scf, convergence_criterion=1.0e-8, spin_polarized=spin_polarized),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
        k_points, k_weights = kpoint_sampling(problem)
        self.assertEqual(len(k_points), 4)
        definition = hybrid_definition(system.input.scf)
        engine = device_bloch_engine(system, definition, k_points, k_weights)
        self.assertIsInstance(engine, DeviceBlochExchange)
        self.assertIsNotNone(engine.stars)
        device = engine.finish(run_hybrid_scf(system, engine=engine))

        class HostOnDeviceWorkers(hybrid_periodic.BlochExchange):
            kpoint_backend = None

        host = run_hybrid_scf(system, engine=HostOnDeviceWorkers(system, definition, k_points, k_weights))
        self.assertTrue(device.converged and host.converged)
        self.assertAlmostEqual(device.energies.total, host.energies.total, delta=1.0e-8)
        # E_x is first order in the orbitals: the device solves the pairs of an image one way and its k-points on
        # pool threads, and the Fock loops stop at the default Exx_Tolerance (1.1e-8 Ry apart once on A100)
        self.assertAlmostEqual(
            device.exact_exchange.exchange_energy, host.exact_exchange.exchange_energy, delta=1.0e-7
        )
        self.assertIn("irreducible wedge", device.exact_exchange.pair_solver)

    def test_unpolarized(self) -> None:
        self.compare(spin_polarized=False)

    def test_spin_polarized(self) -> None:
        self.compare(spin_polarized=True)


class AnyGridWedgeTests(unittest.TestCase):
    """``PARSEC_HYBRID_KPOINT_WEDGE`` on oblique and orthogonal grids (since 2026-11-20): ``on`` takes the wedge of the
    operations that keep the grid, its finite-difference operators and the mesh on any grid (hcp Au2: 8 of its 24
    operations, 14 points; primitive fcc Au: 12 of 48, 13 points), ``auto`` the input of ``off`` (the same k-points and
    weights) with that wedge kept aside for the Fock loop of an insulating start, on the default routines and on
    Fortran's general grid."""

    def parse(self, cell, routines, value):
        from parsec_python.Grid.kpoints import hybrid_wedge_group, kpoint_sampling
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.tests.unit import test_soc_kpoint_wedge as soc

        directory = Path(tempfile.mkdtemp(prefix="parsec_hybwedge_"))
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copy(soc._AU, directory / "Au_POTRE.DAT")
        extra = "Use_Fortran_Routines: .true." if routines == "fortran" else "Use_Fortran_Routines: .false."
        (directory / "parsec.in").write_text(soc._text(cell=cell, extra=extra, so_psp=".false.", xc="PBE0"))
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": value}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(directory / "parsec.in")
            problem = translation.problem
            k_points, weights = kpoint_sampling(problem)
            group = hybrid_wedge_group(problem) if problem.hybrid_kpoint_wedge else None
        return translation, problem, k_points, weights, group

    def test_the_wedge_of_hexagonal_and_fcc_cells(self) -> None:
        from parsec_python.Grid.kpoints import grid_is_oblique

        for cell, operations, wedge, pairs in (("hcp", 8, 14, 20), ("fcc", 12, 13, 36)):
            for routines in ("default", "fortran"):
                with self.subTest(cell=cell, routines=routines):
                    _, auto, k_auto, w_auto, group_auto = self.parse(cell, routines, "auto")
                    self.assertTrue(grid_is_oblique(auto.periodic_cell, auto.grid))
                    self.assertFalse(auto.hybrid_kpoint_wedge)
                    self.assertIsNone(group_auto)
                    self.assertEqual(auto.hybrid_kpoint_wedge_deferred.group.order, operations)
                    _, off, k_off, w_off, _ = self.parse(cell, routines, "off")
                    self.assertFalse(off.hybrid_kpoint_wedge)
                    self.assertIsNone(off.hybrid_kpoint_wedge_deferred)
                    self.assertEqual(len(k_off), pairs)
                    np.testing.assert_array_equal(k_auto, k_off)
                    np.testing.assert_array_equal(w_auto, w_off)
                    self.assertEqual(auto.space_group is None, off.space_group is None)
                    _, on, k_on, w_on, group_on = self.parse(cell, routines, "on")
                    self.assertTrue(on.hybrid_kpoint_wedge)
                    self.assertIsNone(on.hybrid_kpoint_wedge_deferred)
                    self.assertEqual(group_on.order, operations)
                    self.assertEqual(len(k_on), wedge)
                    self.assertAlmostEqual(float(np.sum(w_on)), 1.0, places=12)

    def test_an_orthogonal_grid(self) -> None:
        from parsec_python.Grid.kpoints import grid_is_oblique

        for routines in ("default", "fortran"):
            with self.subTest(routines=routines):
                _, auto, k_auto, w_auto, _ = self.parse("cube", routines, "auto")
                _, off, k_off, w_off, _ = self.parse("cube", routines, "off")
                _, on, k_on, _, group_on = self.parse("cube", routines, "on")
                self.assertFalse(grid_is_oblique(auto.periodic_cell, auto.grid))
                np.testing.assert_array_equal(k_auto, k_off)
                np.testing.assert_array_equal(w_auto, w_off)
                deferred = auto.hybrid_kpoint_wedge_deferred
                if on.hybrid_kpoint_wedge:
                    self.assertEqual(deferred.group.order, group_on.order)
                    self.assertLess(len(k_on), len(k_off))
                else:
                    self.assertIsNone(deferred)

    def test_the_operations_that_keep_the_stencil(self) -> None:
        from parsec_python.Grid.kpoints import grid_stencil_lines, stencil_preserving_matrices
        from parsec_python.Grid.pbc import default_periodic_grid_counts
        from parsec_python.Grid.space_group import find_grid_space_group

        for routines in ("default", "fortran"):
            with self.subTest(routines=routines):
                _, problem, _, _, _ = self.parse("hcp", routines, "off")
                group = problem.space_group
                if routines == "default":
                    # the default routines average nothing for the pairs: the group of the grid itself
                    counts = default_periodic_grid_counts(problem.periodic_cell, problem.grid)
                    group = find_grid_space_group(
                        np.asarray(problem.periodic_cell.lattice_vectors), problem.atoms, counts, problem.grid.shift
                    )
                self.assertEqual(group.order, 24)
                counts, lines = grid_stencil_lines(problem.periodic_cell, problem.grid)
                flags = stencil_preserving_matrices(group.rotation_matrices, counts, lines)
                self.assertEqual(sum(flags), 8)
                # the identity, and the threefold axis turns a grid axis onto the extra direction of the stencil
                matrices = [tuple(np.asarray(m).ravel()) for m in group.rotation_matrices]
                self.assertTrue(flags[matrices.index((1, 0, 0, 0, 1, 0, 0, 0, 1))])
                threefold = [i for i, m in enumerate(group.rotation_matrices)
                             if np.allclose(np.linalg.matrix_power(np.asarray(m), 3), np.eye(3))
                             and not np.allclose(np.asarray(m), np.eye(3))]
                self.assertTrue(threefold)
                self.assertFalse(any(flags[i] for i in threefold))

    def test_the_switch(self) -> None:
        from parsec_python.Grid.kpoints import hybrid_kpoint_wedge_setting

        for value, expected in (("", "auto"), ("ON", "on"), ("auto", "auto"), (" off ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": value}):
                self.assertEqual(hybrid_kpoint_wedge_setting(), expected)
        # grid (the operations of scalar orbitals of 2026-11-19, which turn an oblique grid's stencil) is gone
        for value in ("grid", "yes", "1"):
            with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": value}), self.assertRaises(ValueError):
                hybrid_kpoint_wedge_setting()


class PartlyFilledTests(unittest.TestCase):
    """The test of ``PARSEC_HYBRID_KPOINT_WEDGE=auto`` on the semilocal start: a state is partly filled where
    ``tol <= f <= 1 - tol``, ``tol`` 1e-6."""

    def test_the_threshold(self) -> None:
        from parsec_python.Grid.kpoints import HYBRID_WEDGE_FRACTIONAL_TOLERANCE as tolerance
        from parsec_python.Grid.kpoints import partly_filled_states

        self.assertEqual(tolerance, 1.0e-6)
        self.assertEqual(partly_filled_states(np.array([1.0, 1.0, 0.0, 0.0])), (0, 0.0))
        below = np.nextafter(tolerance, 0.0)
        self.assertEqual(partly_filled_states(np.array([1.0, below, 0.0]))[0], 0)
        self.assertEqual(partly_filled_states(np.array([1.0, tolerance, 0.0]))[0], 1)
        self.assertEqual(partly_filled_states(np.array([1.0 - tolerance, 0.0]))[0], 1)
        self.assertEqual(partly_filled_states(np.array([np.nextafter(1.0 - tolerance, 2.0), 0.0]))[0], 0)
        count, largest = partly_filled_states(np.array([1.0, 0.7, 0.3, 0.0]))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(largest, 0.3, places=15)
        # one array per channel (or per k-point), and an empty one
        self.assertEqual(partly_filled_states([np.array([1.0, 0.0]), np.array([0.5]), np.zeros(0)]), (1, 0.5))
        self.assertEqual(partly_filled_states([np.array([1.0, 0.0]), np.array([0.0, 0.0])]), (0, 0.0))
        self.assertEqual(partly_filled_states(np.array([1.0, 0.5]), tolerance=0.6 / 2)[0], 1)
        for value in (0.0, 0.5, -1.0e-6):
            with self.assertRaises(ValueError):
                partly_filled_states(np.array([1.0]), tolerance=value)

    def test_fermi_dirac_occupations(self) -> None:
        from parsec_python.Grid.kpoints import partly_filled_states
        from parsec_python.Occupations.fermi_dirac import fermi_occupations

        # an insulator at Fermi_Temp 1 K: exactly 0 and 1
        insulator = fermi_occupations(np.array([-0.2, -0.1, 0.1, 0.2]), 4.0, 1.0).occupations
        np.testing.assert_array_equal(insulator, [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(partly_filled_states(insulator), (0, 0.0))
        # at the default 80 K a gap of 0.02 Ry (39 kT) leaves f ~ 3e-9: still an insulator; 0.01 Ry (20 kT) does not
        wide = fermi_occupations(np.array([-0.3, -0.01, 0.01, 0.3]), 4.0, 80.0).occupations
        self.assertEqual(partly_filled_states(wide)[0], 0)
        narrow = fermi_occupations(np.array([-0.3, -0.005, 0.005, 0.3]), 4.0, 80.0).occupations
        self.assertEqual(partly_filled_states(narrow)[0], 2)
        # a metal at the default smearing: a degenerate pair at the Fermi level holding one electron pair
        metal = fermi_occupations(np.array([-0.1, 0.0, 0.0, 0.1]), 4.0, 80.0).occupations
        count, largest = partly_filled_states(metal)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(largest, 0.5, places=8)


@dataclass(frozen=True)
class _Start:
    """The fields of a k-point loop's result that :meth:`BlochExchange.after_start` reads."""

    k_points: np.ndarray
    k_weights: np.ndarray
    k_wavefunctions: tuple
    occupations: np.ndarray
    eigenvalues: np.ndarray
    eigensolver_state: object = None


@dataclass(frozen=True)
class _SpinStart:
    k_points: np.ndarray
    k_weights: np.ndarray
    k_wavefunctions: tuple
    occupations_up: np.ndarray
    occupations_down: np.ndarray
    eigenvalues_up: np.ndarray
    eigenvalues_down: np.ndarray
    eigensolver_state: object = None


class AfterStartTests(unittest.TestCase):
    """:meth:`BlochExchange.after_start` on made-up starts of the silicon cube under auto (CPU): a partly filled
    state keeps everything as it is (the very objects: the run of off), none moves the engine to the wedge, whose
    representatives are points of the pairs with their labels, orbitals and Ritz states."""

    def setUp(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        directory, translation = _silicon_mesh("pbe0", "auto")
        self.addCleanup(shutil.rmtree, directory, True)
        self.problem = translation.problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.system = prepare_periodic_single_point(self.problem)
        self.k_points, self.weights = kpoint_sampling(self.problem)
        self.definition = hybrid_definition(self.problem.scf)

    def engine(self):
        from parsec_python.SCF.hybrid_periodic import BlochExchange

        return BlochExchange(self.system, self.definition, self.k_points, self.weights)

    def start(self, occupied, *, spin: bool = False, states=None):
        rng = np.random.default_rng(5)
        size = self.system.grid.size
        count = len(self.k_points)
        blocks = tuple(rng.normal(size=(size, 3)) + 0j for _ in range(count))
        values = np.tile(np.asarray(occupied, dtype=np.float64), count)
        levels = np.tile([-0.3, -0.2, 0.4], count)
        if spin:
            return _SpinStart(self.k_points, self.weights, (blocks, blocks), values, values.copy(), levels,
                              levels.copy(), states)
        return _Start(self.k_points, self.weights, blocks, values, levels, states)

    def test_a_partly_filled_start_changes_nothing(self) -> None:
        from parsec_python.models import HybridMeshRouteRecord

        engine = self.engine()
        description = engine.description
        start = self.start([1.0, 0.5, 0.0])
        records: list = []
        system, kept = engine.after_start(self.system, start, callback=records.append)
        self.assertIs(system, self.system)
        self.assertIs(kept, start)
        self.assertIsNone(engine.stars)
        self.assertEqual(len(engine.k_points), 8)
        self.assertEqual(engine.description, description)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, HybridMeshRouteRecord)
        self.assertEqual((record.route, record.points, record.pairs_points, record.mesh_points), ("pairs", 4, 8, 8))
        self.assertEqual(record.partly_filled, 8)
        self.assertEqual(record.largest_fractional_occupation, 0.5)
        from parsec_python.Output.parsec_output import hybrid_mesh_route_lines

        text = "\n".join(hybrid_mesh_route_lines(record))
        self.assertIn("8 partly filled state(s)", text)
        self.assertIn("PARSEC_HYBRID_KPOINT_WEDGE=on runs the irreducible wedge of 4 k-points", text)

    def test_an_insulating_start_takes_the_wedge(self) -> None:
        from parsec_python.models import HybridMeshRouteRecord

        engine = self.engine()
        states = {task: f"state {task}" for task in range(8)}
        start = self.start([1.0, 1.0, 0.0], states=states)
        records: list = []
        system, moved = engine.after_start(self.system, start, callback=records.append)
        deferred = self.problem.hybrid_kpoint_wedge_deferred
        self.assertTrue(system.input.hybrid_kpoint_wedge)
        self.assertIsNone(system.input.hybrid_kpoint_wedge_deferred)
        self.assertEqual(system.input.monkhorst_pack_dimensions, (2, 2, 2))
        self.assertIsNone(system.input.kpoints)
        self.assertTrue(system.input.scf.symmetrize_density)
        self.assertEqual(len(system.density_symmetry.matrices), deferred.group.order)
        self.assertIsNone(self.system.density_symmetry)  # the pairs averaged nothing
        self.assertIsNotNone(engine.stars)
        self.assertEqual(len(engine.k_points), 4)
        self.assertIn("irreducible wedge", engine.description)
        np.testing.assert_allclose(np.sort(engine.k_weights * 8), [1.0, 1.0, 3.0, 3.0])
        # the representatives: points of the pairs, their labels, orbitals, levels, occupations and Ritz states
        indices = []
        for k in moved.k_points:
            matches = [i for i, point in enumerate(self.k_points) if np.array_equal(point, k)]
            self.assertEqual(len(matches), 1)
            indices.append(matches[0])
        np.testing.assert_array_equal(moved.k_points, engine.k_points)
        for position, index in enumerate(indices):
            self.assertIs(moved.k_wavefunctions[position], start.k_wavefunctions[index])
        self.assertEqual(
            moved.eigensolver_state, {position: f"state {index}" for position, index in enumerate(indices)}
        )
        np.testing.assert_array_equal(moved.occupations, np.tile([1.0, 1.0, 0.0], 4))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertIsInstance(record, HybridMeshRouteRecord)
        self.assertEqual((record.route, record.points, record.pairs_points, record.operations),
                         ("wedge", 4, 8, deferred.group.order))
        self.assertEqual(record.partly_filled, 0)
        from parsec_python.Output.parsec_output import hybrid_mesh_route_lines

        lines = "\n".join(hybrid_mesh_route_lines(record))
        self.assertIn("The Fock loop runs on the irreducible wedge of 4 k-points", lines)
        # a second call (the engine is on the wedge already) changes nothing
        again, same = engine.after_start(system, moved, callback=records.append)
        self.assertIs(again, system)
        self.assertIs(same, moved)
        self.assertEqual(len(records), 1)

    def test_the_sectors_at_any_size_go_with_the_wedge(self) -> None:
        # PARSEC_HYBRID_KPOINT_SECTORS=on: the key of the loops' cache stays through the move to the wedge (which drops
        # the projectors and plans of the pairs), and the engine's description names it before and after
        from parsec_python.acceleration.SCF.kpoint_sectors import ANY_SIZE_KEY

        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_SECTORS": "on", "PARSEC_KPOINT_SECTORS": ""}):
            engine = self.engine()
        self.assertTrue(engine.projector_cache[ANY_SIZE_KEY])
        self.assertIn("PARSEC_HYBRID_KPOINT_SECTORS=on", engine.description)
        engine.projector_cache["projectors of the pairs"] = object()
        engine.after_start(self.system, self.start([1.0, 1.0, 0.0]))
        self.assertIsNotNone(engine.stars)
        self.assertEqual(set(engine.projector_cache), {ANY_SIZE_KEY})
        self.assertIn("irreducible wedge", engine.description)
        self.assertIn("PARSEC_HYBRID_KPOINT_SECTORS=on", engine.description)
        # off (the default): no key, the description as before
        with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_SECTORS": ""}):
            plain = self.engine()
        self.assertNotIn(ANY_SIZE_KEY, plain.projector_cache)
        self.assertNotIn("PARSEC_HYBRID_KPOINT_SECTORS", plain.description)

    def test_the_states_of_the_sectors_and_of_two_channels(self) -> None:
        engine = self.engine()
        sectors_state = {"kpoint_sectors": {"layout": [], "counts": [], "states": {}}}
        _, moved = engine.after_start(self.system, self.start([1.0, 1.0, 0.0], states=sectors_state))
        self.assertIsNone(moved.eigensolver_state)  # the sectors of the wedge start afresh
        engine = self.engine()
        states = {task: task for task in range(16)}  # (channel, k) = channel * 8 + k
        _, moved = engine.after_start(self.system, self.start([1.0, 1.0, 0.0], spin=True, states=states))
        self.assertEqual(len(moved.k_wavefunctions), 2)
        self.assertEqual([len(channel) for channel in moved.k_wavefunctions], [4, 4])
        indices = [int(np.flatnonzero([np.array_equal(p, k) for p in self.k_points])[0]) for k in moved.k_points]
        expected = {channel * 4 + position: channel * 8 + index
                    for channel in range(2) for position, index in enumerate(indices)}
        self.assertEqual(moved.eigensolver_state, expected)
        self.assertEqual(len(moved.occupations_up), 12)
        # a spin channel with a partly filled state keeps the pairs
        engine = self.engine()
        start = self.start([1.0, 1.0, 0.0], spin=True)
        start = replace(start, occupations_down=np.tile([1.0, 0.25, 0.0], 8))
        _, kept = engine.after_start(self.system, start)
        self.assertIs(kept, start)
        self.assertIsNone(engine.stars)

    def test_off_and_on_take_no_decision(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.hybrid_periodic import BlochExchange
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        for wedge in ("off", "on"):
            directory, translation = _silicon_mesh("pbe0", wedge)
            self.addCleanup(shutil.rmtree, directory, True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(translation.problem)
            k_points, weights = kpoint_sampling(translation.problem)
            engine = BlochExchange(system, self.definition, k_points, weights)
            records: list = []
            blocks = tuple(np.zeros((system.grid.size, 3), dtype=np.complex128) for _ in k_points)
            start = _Start(k_points, weights, blocks, np.tile([1.0, 1.0, 0.0], len(k_points)),
                           np.tile([-0.3, -0.2, 0.4], len(k_points)))
            same_system, same = engine.after_start(system, start, callback=records.append)
            self.assertIs(same_system, system)
            self.assertIs(same, start)
            self.assertEqual(records, [])


class AutoRouteTests(unittest.TestCase):
    """``auto`` on the silicon cube with a smeared start (Fermi_Temp 3000 K: the levels at the gap partly filled): the
    pairs of ``off``, the very objects of the start handed on (the run of off), the NOTE of the route.  The totals
    agree within the run-to-run spread of the device at the default Exx_Tolerance (off against off 2.5e-11 Ry, auto
    against off 6.5e-9 Ry once on A100, 2026-11-20; at 30000 K, where the Fock loop of the smeared cube is
    ill-conditioned, 1.8e-5 Ry)."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_partly_filled_start_keeps_the_pairs_of_off(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.hybrid_periodic import BlochExchange
        from parsec_python.models import HybridMeshRouteRecord

        original = BlochExchange.after_start
        handed: list = []

        def spy(self, system, start, **keywords):
            found = original(self, system, start, **keywords)
            handed.append((found[0] is system, found[1] is start))
            return found

        results, routes = {}, {}
        for wedge in ("auto", "off"):
            directory, translation = _silicon_mesh("pbe0", wedge, {"Fermi_Temp": "3000.0"})
            self.addCleanup(shutil.rmtree, directory, True)
            problem = translation.problem
            k_points, weights = kpoint_sampling(problem)
            records: list = []
            with mock.patch.object(BlochExchange, "after_start", spy):
                results[wedge] = _run(
                    problem, k_points, weights, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}, records.append
                )
            routes[wedge] = [record for record in records if isinstance(record, HybridMeshRouteRecord)]
        auto, off = results["auto"], results["off"]
        self.assertEqual(handed, [(True, True), (True, True)])
        self.assertEqual(routes["off"], [])
        self.assertEqual([record.route for record in routes["auto"]], ["pairs"])
        self.assertGreater(routes["auto"][0].partly_filled, 0)
        self.assertEqual(routes["auto"][0].points, 4)
        np.testing.assert_array_equal(auto.k_points, off.k_points)
        self.assertEqual(len(auto.exact_exchange.iterations), len(off.exact_exchange.iterations))
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6)
        # the levels of the occupied states (the top empty ones are filtered loosely, 4e-6 Ry apart run to run).  The
        # levels converge to first order in the density error of the Fock loop, the totals (variational) to second
        # order, so the levels are compared on the scale of the run-to-run spread of the device, not of the totals:
        # off against off in a fresh process 2.0e-7 / 2.7e-7 Ry, auto against off there 3.1e-7 / 2.6e-7 Ry (totals
        # 1e-11 Ry, A100, 2026-11-20); after the other tests of a suite process the spread grows: 1.7e-6 Ry
        # level by level in the full acceleration suite (two degenerate pairs split 0.3952541/0.3952564 against
        # 0.3952552/0.3952553 Ry) and 7e-7 Ry in a cluster mean in a module run, with the very objects
        # of off handed on (``handed`` above) and the totals within 1e-7 Ry.  The route itself is checked by the spy
        # and the route record above; the levels only bound the run.
        occupied = np.asarray(off.occupations) > 1.0e-3
        np.testing.assert_allclose(
            np.asarray(auto.eigenvalues)[occupied], np.asarray(off.eigenvalues)[occupied], rtol=0.0, atol=1.0e-5
        )


class AnySizeSectorTests(unittest.TestCase):
    """PARSEC_HYBRID_KPOINT_SECTORS=on: the silicon cube (12^3 points, 22 states: below the floor of the k-point
    sectors) solves the start on the pairs and, insulating, the Fock loop on the wedge (auto) in the sectors of its
    k-points, with the energy of the run without them; off keeps every k-point whole, the run as before."""

    def test_the_wedge_of_auto_in_the_sectors(self) -> None:
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.models import HybridMeshRouteRecord

        results, routes = {}, {}
        for setting in ("off", "on"):
            directory, translation = _silicon_mesh("pbe0", "auto")
            self.addCleanup(shutil.rmtree, directory, True)
            problem = translation.problem
            k_points, weights = kpoint_sampling(problem)
            records: list = []
            # the floor of the sectors as in a run (this module lowers it to split its small cells)
            with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 1 << 24):
                results[setting] = _run(
                    problem, k_points, weights,
                    {"PARSEC_HYBRID_KPOINT_WEDGE": "auto", "PARSEC_HYBRID_KPOINT_SECTORS": setting,
                     "PARSEC_KPOINT_SECTORS": ""},
                    records.append,
                )
            routes[setting] = [record.route for record in records if isinstance(record, HybridMeshRouteRecord)]
        off, on = results["off"], results["on"]
        self.assertTrue(off.converged and on.converged)
        self.assertEqual(routes, {"off": ["wedge"], "on": ["wedge"]})
        self.assertEqual(len(on.k_points), 4)
        self.assertNotIn("kpoint_sector_tasks", off.timings.detail)
        self.assertGreater(on.timings.detail["kpoint_sector_tasks"], 4.0)
        self.assertIn("PARSEC_HYBRID_KPOINT_SECTORS=on", on.exact_exchange.pair_solver)
        self.assertNotIn("PARSEC_HYBRID_KPOINT_SECTORS", off.exact_exchange.pair_solver)
        self.assertAlmostEqual(on.energies.total, off.energies.total, delta=5.0e-8)


class MeshNoteTests(unittest.TestCase):
    @unittest.skipUnless((_HYBRID_EXAMPLES / "Si64_k222").exists(), "the hybrid examples are not in this checkout")
    def test_the_note_of_a_mesh_without_a_shift_block(self) -> None:
        from parsec_python.Input.parsec_input import parse_parsec_input

        # The Fortran-derived routines (fhipp): the wedge of the hybrid single point (PARSEC_HYBRID_KPOINT_WEDGE=on,
        # 4 of the 8 points), its time-reversal pairs under off and for the start under auto, the wedge of PBE
        # under any value
        from parsec_python.Grid.kpoints import kpoint_sampling

        notes, counts = {}, {}
        for wedge in ("on", "auto", "off"):
            for name in ("pbe0", "pbe"):
                directory = Path(tempfile.mkdtemp(prefix="parsec_hybmesh_"))
                self.addCleanup(shutil.rmtree, directory, True)
                shutil.copy(_HYBRID_EXAMPLES / "Si64_k222" / name / "parsec.in", directory / "parsec.in")
                shutil.copy(_HYBRID_EXAMPLES / "Si_FHIPP.DAT", directory / "Si_FHIPP.DAT")
                with mock.patch.dict(os.environ, {"PARSEC_HYBRID_KPOINT_WEDGE": wedge}), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    translation = parse_parsec_input(directory / "parsec.in")
                found = [note for note in translation.warnings if note.startswith("Monkhorst_Pack_Grid without")]
                self.assertEqual(len(found), 1, found)
                notes[wedge, name] = found[0]
                counts[wedge, name] = len(kpoint_sampling(translation.problem)[0])
        self.assertIn("reduced by time reversal only", notes["off", "pbe0"])
        self.assertNotIn("irreducible wedge", notes["off", "pbe0"])
        self.assertIn("through the images of the wedge", notes["on", "pbe0"])
        self.assertIn("for the semilocal start of the hybrid", notes["auto", "pbe0"])
        for wedge in ("on", "auto", "off"):
            self.assertIn("reduced to its irreducible wedge", notes[wedge, "pbe"])
            self.assertNotIn("hybrid", notes[wedge, "pbe"])
        self.assertEqual(
            counts,
            {("on", "pbe0"): 4, ("on", "pbe"): 4, ("auto", "pbe0"): 8, ("auto", "pbe"): 4, ("off", "pbe0"): 8,
             ("off", "pbe"): 4},
        )


if __name__ == "__main__":
    unittest.main()
