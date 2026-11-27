"""The sectors of the k-points of a periodic non-collinear SCF (``PARSEC_NONCOLLINEAR_KPOINT_SECTORS``), on the host.

The device code runs with NumPy as the array backend (``PARSEC_KPOINT_BACKEND=numpy-device``).  Checked: the
operations the magnetic group of the initial moments keeps (an atom with its moment along z, x or [111], a pair along z
with parallel and antiparallel moments); the union of the spectra of the sectors against the dense spinor Bloch
Hamiltonian of the whole grid with the exchange-correlation field of the initial moments (the frames of ``|+-z>`` and
``|+-x>``), and the unfolded eigenvectors; the SCF with forces against the loop without sectors (energy, density,
magnetization, moment and forces); an operation that the start field breaks left out of the sectors (test 1 of the
guard); the loop started again with one task per k-point where the final state breaks an operation of the sectors
(test 2) and kept under ``PARSEC_KPOINT_SECTOR_GUARD=off``; the switch values and their off value.
"""

from __future__ import annotations

import functools
import os
import unittest
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
import parsec_python.acceleration  # noqa: F401  (installs the parallel k-point backend)
from parsec_python.SCF.kpoint_solver import KPointJob
from parsec_python.SCF.kpoint_weights import validated_kpoint_weights
from parsec_python.SCF.kpoints_soc import _k_point_operators, _sector_geometry, _spinor_hamiltonian
from parsec_python.SCF.noncollinear import _initial_magnetization, run_self_consistent_noncollinear_kpoints
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter, NumpyArrayBackend
from parsec_python.acceleration.SCF import kpoint_spinor_sectors as module
from parsec_python.acceleration.SCF.kpoint_spinor_sectors import (
    NONCOLLINEAR_SWITCH,
    SpinorSectorKPointSolver,
    noncollinear_kpoint_sectors_setting,
    plan_spinor_sectors,
)

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_BOX = 8.0
_C = _BOX / 2
_CASES = {
    "atom_z": [Atom("Au", [_C] * 3, [0, 0, 1])],
    "atom_x": [Atom("Au", [_C] * 3, [1, 0, 0])],
    "atom_111": [Atom("Au", [_C] * 3, [1, 1, 1])],
    "pair_fm": [Atom("Au", [_C, _C, _C - 1.6], [0, 0, 1]), Atom("Au", [_C, _C, _C + 1.6], [0, 0, 1])],
    # 8 bohr apart along z in a cell twice as long: each atom keeps its moment (the pair above ends non-magnetic)
    "far_afm": [Atom("Au", [_C, _C, _C], [0, 0, 1]), Atom("Au", [_C, _C, _C + _BOX], [0, 0, -1])],
}
# the cells of the cases (rows: lattice vectors)
_CELLS = {"far_afm": np.diag([_BOX, _BOX, 2.0 * _BOX])}
# Gamma and the star of (0, 0, 1/3): a set every operation of the cubic group maps onto itself without time reversal
_FRACTIONS = np.array(
    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
) / 3.0
_DEVICE = {"PARSEC_KPOINT_BACKEND": "numpy-device"}


@functools.lru_cache(maxsize=None)
def _system(case: str, criterion: float = 1.0e-11):
    atoms = _CASES[case]
    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True, initial_spin_polarization=0.3)},
        grid=PeriodicGridSettings(spacing=0.89, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=_CELLS.get(case, _BOX * np.eye(3))),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=20 * len(atoms),
            convergence_criterion=criterion,
            noncollinear=True,
            fermi_temperature_kelvin=1000.0,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=criterion / 10),
    )
    return prepare_periodic_single_point(problem)


def _k_points():
    k_points = 2.0 * np.pi / _BOX * _FRACTIONS
    weights = np.full(len(k_points), 1.0 / len(k_points))
    return validated_kpoint_weights(k_points, weights, "test")


def _start(system):
    """The potential and the field of the first eigenproblem of the loop (the initial density and moments)."""

    xc = system.evaluate_xc_noncollinear(system.initial_density, _initial_magnetization(system))
    return system.ionic_potential + xc.potential, np.asarray(xc.field, dtype=np.float64)


def _plans(system, field):
    k_points, weighting = _k_points()
    with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "on"}):
        geometry = _sector_geometry(system, k_points, weighting, False, noncollinear=True, field=field)
        job = KPointJob(len(k_points), int(system.input.scf.number_of_states), None, None, None, geometry=geometry)
        notes: list = []
        return plan_spinor_sectors(job, notes), notes


class NoncollinearSectorOperatorTests(unittest.TestCase):
    def check(self, case: str, axis: int, gamma_order: int, paired: bool = False):
        system = _system(case)
        potential, field = _start(system)
        self.assertGreater(float(np.max(np.abs(field))), 1.0e-3)
        plans, notes = _plans(system, field)
        self.assertIsNotNone(plans, notes)
        self.assertIn("the magnetic group of the initial moments", notes[-1])
        self.assertEqual(plans[0].axis, axis)
        self.assertEqual(plans[0].order, gamma_order)
        k_points, _ = _k_points()
        converter = DeviceBlochConverter(NumpyArrayBackend())
        operators = _k_point_operators(system, k_points)
        for index, k_point in enumerate(k_points):
            plan = plans[index]
            if plan is None:
                continue
            # Kramers pairs only through an operation that time reversal makes keep the moments
            self.assertEqual(plan.antiunitary is not None, paired)
            nonlocal_k, soc_k = operators[index]
            full = _spinor_hamiltonian(system, k_point, nonlocal_k, soc_k, potential, xc_field=field)
            dense = full.apply(np.eye(full.shape[0], dtype=complex))
            whole = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
            values = []
            for sector in plan.solved:
                operator = plan.hamiltonian(sector, full).device_operator(converter)
                matrix = operator.apply(np.eye(operator.shape[0], dtype=complex))
                self.assertLess(np.max(np.abs(matrix - matrix.conj().T)), 1.0e-12)
                levels, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
                values.append(levels)
                spinors = plan.unfold(sector, vectors[:, :6])
                self.assertLess(np.max(np.abs(full.apply(spinors) - spinors * levels[:6])), 1.0e-10)
                if plan.partner[sector] is not None:
                    values.append(levels)
                    partners = plan.partner_spinors(spinors)
                    self.assertLess(np.max(np.abs(full.apply(partners) - partners * levels[:6])), 1.0e-10)
                    self.assertLess(np.max(np.abs(spinors.conj().T @ partners)), 1.0e-12)
            union = np.sort(np.concatenate(values))
            self.assertEqual(union.size, whole.size)
            self.assertLess(np.max(np.abs(union - whole)), 1.0e-10)
        return plans

    def test_moment_along_z(self) -> None:
        # {E, i, C2z, sigma_z}: C2x and the mirrors normal to x and y turn the moment over
        self.check("atom_z", 2, 4)

    def test_moment_along_x(self) -> None:
        # {E, i, C2x, sigma_x}: the frame of |+-x>, the field B_x on the diagonal and B_y, B_z coupling the components
        self.check("atom_x", 0, 4)

    def test_antiparallel_pair(self) -> None:
        # the inversion about either atom keeps the moments: {E, i, C2z, sigma_z} at Gamma; the half-cell translation
        # and the inversion about the midpoint exchange the atoms and turn their moments over, and times time reversal
        # they keep them: the sectors of every k-point pair up
        plans = self.check("far_afm", 2, 4, paired=True)
        self.assertTrue(all(len(plan.solved) == plan.order // 2 for plan in plans if plan is not None))

    def test_moments_select_the_operations(self) -> None:
        system = _system("atom_111")
        plans, notes = _plans(system, _start(system)[1])
        # a moment along [111]: only the inversion of the involutions keeps it
        self.assertEqual(plans[0].order, 2)
        self.assertEqual(sum(plan is not None for plan in plans), 1, notes)
        system = _system("pair_fm")
        plans, _ = _plans(system, _start(system)[1])
        self.assertEqual(plans[0].order, 4)

    def test_operations_the_start_field_breaks_are_left_out(self) -> None:
        system = _system("atom_z")
        _, field = _start(system)
        # the field of a moment along x for the moments along z: C2z and sigma_z turn it over
        turned = field[[2, 1, 0]].copy()
        plans, notes = _plans(system, turned)
        self.assertIn("the start field breaks", notes[-1])
        for plan in plans:
            if plan is not None:
                for element in plan.elements:
                    if element:
                        self.assertTrue(np.array_equal(np.diag(module._axial(plan.cell.matrices[element]))[:2], [1, 1]))


class NoncollinearSectorSCFTests(unittest.TestCase):
    @staticmethod
    def _run(case: str, value: str, *, steps=None, environment=None):
        # (the sectors of these states, whose 6s levels of Au are partly filled at the start and fractional at 1000 K,
        # under the guard's tests 1 and 3 of PARSEC_KPOINT_SECTOR_GUARD=auto would go to one task per k-point: off here,
        # the algebra of the sectors is the subject; the guard has its own tests)
        guard = {"PARSEC_KPOINT_SECTOR_GUARD": "off"}
        k_points, _ = _k_points()
        weights = np.full(len(k_points), 1.0 / len(k_points))
        with mock.patch.dict(os.environ, {**_DEVICE, NONCOLLINEAR_SWITCH: value, **guard, **(environment or {})}):
            return run_self_consistent_noncollinear_kpoints(
                _system(case), k_points, weights, compute_forces=True,
                callback=None if steps is None else steps.append,
            )

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _whole(cls, case: str):
        return cls._run(case, "off")

    def check(self, case: str, magnetization: float = 1.0e-9, moment: float = 1.0e-7, magnetic: bool = True):
        whole = self._whole(case)
        sectors = self._run(case, "on")
        self.assertTrue(whole.converged and sectors.converged)
        self.assertNotIn("eig_spinor_sector_tasks", whole.timings)
        self.assertGreater(sectors.timings["eig_spinor_sector_kpoints"], 0)
        self.assertLess(sectors.timings["noncollinear_guard_asymmetry"], 1.0e-6)
        if magnetic:
            self.assertGreater(float(np.max(np.abs(whole.magnetization))), 1.0e-3)
        self.assertLess(abs(sectors.energies.total - whole.energies.total), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.density - whole.density)), 1.0e-9)
        self.assertLess(np.max(np.abs(sectors.magnetization - whole.magnetization)), magnetization)
        self.assertLess(np.max(np.abs(sectors.magnetic_moment_vector - whole.magnetic_moment_vector)), moment)
        self.assertLess(np.max(np.abs(sectors.nonlocal_forces - whole.nonlocal_forces)), 1.0e-8)
        occupied = whole.occupations > 1.0e-8
        np.testing.assert_allclose(
            np.sort(sectors.eigenvalues[sectors.occupations > 1.0e-8]), np.sort(whole.eigenvalues[occupied]),
            atol=1.0e-9,
        )
        np.testing.assert_allclose(np.linalg.norm(sectors.spinors, axis=0), 1.0, atol=1.0e-12)
        return sectors

    def test_moment_along_z(self) -> None:
        self.check("atom_z")

    def test_moment_along_x(self) -> None:
        self.check("atom_x")

    def test_antiparallel_pair_in_kramers_pairs(self) -> None:
        # (the antiparallel moments decay in the SCF of this cell, the field of the start is that of the moments:
        # the partners hold through that)
        sectors = self.check("far_afm", magnetic=False)
        self.assertGreater(sectors.timings["eig_spinor_sector_kpoints"], 0)

    def test_guard_starts_the_loop_again_without_sectors(self) -> None:
        def broken(self, density, magnetization, volume_element):
            return [("an operation", 0.5)]

        steps: list = []
        with mock.patch.object(SpinorSectorKPointSolver, "noncollinear_asymmetries", broken), mock.patch.object(
            SpinorSectorKPointSolver, "noncollinear_multiplets", lambda self, *args, **kwargs: []
        ):
            result = self._run("atom_z", "on", steps=steps, environment={"PARSEC_KPOINT_SECTOR_GUARD": "auto"})
            kept = self._run("atom_z", "on", environment={"PARSEC_KPOINT_SECTOR_GUARD": "off"})
        from parsec_python.models import SpinSectorFallbackRecord

        notes = [step for step in steps if isinstance(step, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].route, "kpoint")
        self.assertIn("non-collinear k-point sectors", notes[0].reason)
        self.assertEqual(result.timings["noncollinear_sector_abandoned"], 1.0)
        self.assertNotIn("eig_spinor_sector_kpoints", result.timings)
        whole = self._whole("atom_z")
        self.assertEqual(result.energies.total, whole.energies.total)
        # PARSEC_KPOINT_SECTOR_GUARD=off keeps the result of the sectors
        self.assertGreater(kept.timings["eig_spinor_sector_kpoints"], 0)
        self.assertEqual(kept.timings["noncollinear_guard_asymmetry"], 0.5)


class NoncollinearNoteTests(unittest.TestCase):
    def test_the_reporter_of_the_spinor_loops_writes_the_note(self) -> None:
        # the command line reports a non-collinear k-point loop with iteration_self_consistent_soc: the note of the
        # guard has no levels of its own, its lines are those iteration writes
        import shutil
        import tempfile
        import warnings

        from parsec_python.Input import parse_parsec_input
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.models import SpinSectorFallbackRecord
        from parsec_python.tests.unit.test_noncollinear import _CLUSTER_INPUT

        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        shutil.copyfile(_AU_POTRE, directory / "Au_POTRE.DAT")
        text = _CLUSTER_INPUT.format(extra="begin Initial_NCL_Moment\n0 0 1\nend Initial_NCL_Moment\n")
        text = text.replace(
            "Boundary_Sphere_Radius: 6.0\n",
            "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n",
        ).replace("0.0 0.0 0.0\nEnd atom_coord", "4.0 4.0 4.0\nEnd atom_coord")
        (directory / "parsec.in").write_text(text, encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(directory / "parsec.in")
        record = SpinSectorFallbackRecord(
            stage="the SCF loop", largest_fractional_occupation=0.5, tolerance=0.01, multiplets=("an operation",),
            reason="a density or magnetization that breaks the symmetry the non-collinear k-point sectors assume",
            measure="largest asymmetry", route="kpoint",
        )
        written: dict[str, list[str]] = {"iteration": [], "iteration_self_consistent_soc": []}
        for name, out in written.items():
            getattr(ParsecTextReporter(out.append, translation), name)(record)
        self.assertEqual(written["iteration_self_consistent_soc"], written["iteration"])
        self.assertIn("NOTE: k-point sectors: a density or magnetization", written["iteration"][0])
        # the measured value of test 2 in the timing table of the result
        from types import SimpleNamespace

        lines = ParsecTextReporter._soc_timing_lines(SimpleNamespace(timings={"noncollinear_guard_asymmetry": 2.5e-9}))
        self.assertTrue(any("Test 2, asymmetry" in line and "2.5000E-09" in line for line in lines))
        lines = ParsecTextReporter._soc_timing_lines(SimpleNamespace(timings={"diagonalization": 1.0}))
        self.assertFalse(any("Test 2" in line for line in lines))


class NoncollinearSwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value in ("off", "auto", "on"):
            with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: value}):
                self.assertEqual(noncollinear_kpoint_sectors_setting(), value)
        with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "maybe"}):
            with self.assertRaises(ValueError):
                noncollinear_kpoint_sectors_setting()
            from parsec_python.acceleration.SCF.kpoint_sectors import check_kpoint_sector_switches

            with self.assertRaises(ValueError):
                check_kpoint_sector_switches()
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_GUARD": "maybe"}):
            from parsec_python.acceleration.SCF.kpoint_sectors import check_kpoint_sector_switches

            with self.assertRaises(ValueError):
                check_kpoint_sector_switches()

    def test_off_keeps_the_job_of_before(self) -> None:
        system = _system("atom_z")
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "off", "PARSEC_SOC_KPOINT_SECTORS": "on"}):
            self.assertIsNone(_sector_geometry(system, k_points, weighting, False, noncollinear=True))
        # the switch of the spin-orbit loop does not govern the non-collinear one
        with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "on", "PARSEC_SOC_KPOINT_SECTORS": "off"}):
            geometry = _sector_geometry(system, k_points, weighting, False, noncollinear=True)
        self.assertTrue(geometry["noncollinear"])
        np.testing.assert_allclose(geometry["moments"], [[0.0, 0.0, 0.3]])

    def test_auto_keeps_small_bases_whole(self) -> None:
        system = _system("atom_z")
        k_points, weighting = _k_points()
        with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "auto"}):
            geometry = _sector_geometry(system, k_points, weighting, False, noncollinear=True)
            job = KPointJob(len(k_points), 20, None, None, None, geometry=geometry)
            notes: list = []
            self.assertIsNone(plan_spinor_sectors(job, notes))
        self.assertIn("below the floor", notes[-1])
        self.assertIn(NONCOLLINEAR_SWITCH, notes[-1])

    def test_auto_lowers_the_rank_for_the_memory(self) -> None:
        # (above the floor and the states per sector of auto: the planner alone, 400 states)
        system = _system("atom_z")
        k_points, weighting = validated_kpoint_weights(np.zeros((1, 3)), np.ones(1), "test")
        found = {}
        with mock.patch.dict(os.environ, {NONCOLLINEAR_SWITCH: "auto"}), mock.patch.object(module, "MINIMUM_ENTRIES", 0):
            geometry = _sector_geometry(system, k_points, weighting, False, noncollinear=True)
            self.assertGreater(geometry["soc_entries"], 0.0)
            # Gamma, 1458 spinor rows by 400 states: one task 9.3 MB, 53 MB with its working copies; four sectors take
            # 320 bytes per entry of the projectors, two 160
            for label, entries in (("small", 0.0), ("middle", 3.0e5), ("large", 1.0e8)):
                job = KPointJob(1, 400, None, None, None, geometry={**geometry, "soc_entries": entries})
                notes: list = []
                plans = plan_spinor_sectors(job, notes)
                orders = [plan.order for plan in plans or () if plan is not None]
                found[label] = (max(orders, default=1), notes[-1])
        self.assertEqual(found["small"][0], 4)
        self.assertNotIn("device memory", found["small"][1])
        self.assertEqual(found["middle"][0], 2)
        self.assertIn("lowered the rank of their groups to 1", found["middle"][1])
        self.assertEqual(found["large"][0], 1)
        self.assertIn("lowered the rank of their groups to 0", found["large"][1])
        # the rule: 1.5 times one task per k-point, the free memory where the sectors take more than that
        self.assertTrue(module._memory_exceeds(1.6, 1.0, None))
        self.assertFalse(module._memory_exceeds(1.4, 1.0, None))
        self.assertTrue(module._memory_exceeds(1.4, 1.0, 1.5))
        self.assertFalse(module._memory_exceeds(0.9, 1.0, 0.5))


def _cuda_device() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_cuda_device(), "needs a CUDA device")
class DevicePoolTests(unittest.TestCase):
    def test_free_blocks_of_the_pool_count_as_free_and_go_back(self) -> None:
        import cupy as cp

        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_CUPY_DEVICES": "0"}):
            with cp.cuda.Device(0):
                pool = cp.get_default_memory_pool()
                # the pool as the tests before left it: free parts of blocks that live arrays still hold cannot go
                # back, so the test measures from what remains after a release
                pool.free_all_blocks()
                residual = pool.free_bytes()
                # larger than every free part that remains: a block of its own from the device
                size = (256 << 20) + residual
                block = cp.zeros(size, dtype=cp.uint8)
                del block
                self.assertGreaterEqual(pool.free_bytes(), residual + size)
                device_free = float(cp.cuda.runtime.memGetInfo()[0])
                self.assertGreaterEqual(module.free_device_bytes("cupy"), device_free + size - (16 << 20))
                module.release_device_pools("cupy")
                self.assertLessEqual(pool.free_bytes(), residual)
                self.assertGreaterEqual(float(cp.cuda.runtime.memGetInfo()[0]), device_free + size - (16 << 20))


class NoncollinearMultipletGuardTests(unittest.TestCase):
    def test_a_partly_filled_multiplet_at_the_start_leaves_the_sectors(self) -> None:
        # O2 from antiparallel moments at Gamma (no spin-orbit coupling): its four pi* levels, two in a sector and two
        # in its Kramers partner (the inversion times time reversal), hold two electrons; test 1 sends the loop to one
        # task per k-point, which reaches the triplet as the loop without sectors does (test 1 is that of
        # PARSEC_NONCOLLINEAR_SECTOR_STABILITY=off; the stability test of auto: test_noncollinear_stability)
        side = 8.0
        atoms = [Atom("O", [4.0, 4.0, 2.86], [0, 0, 1]), Atom("O", [4.0, 4.0, 5.14], [0, 0, -1])]
        oxygen = Path(__file__).resolve().parents[2] / "tests" / "data" / "spin_ca_O_POTRE.DAT"
        problem = SinglePointInput(
            atoms=atoms,
            pseudopotentials={"O": SpeciesPotential(oxygen, 1, initial_spin_polarization=0.3)},
            grid=PeriodicGridSettings(spacing=0.4, expansion_order=6),
            periodic_cell=PeriodicCell(lattice_vectors=side * np.eye(3)),
            scf=SCFSettings(max_iterations=100, number_of_states=16, convergence_criterion=1.0e-6, noncollinear=True,
                            fermi_temperature_kelvin=500.0),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-7),
        )
        system = prepare_periodic_single_point(problem)
        results, steps = {}, []
        for value in ("off", "on"):
            with mock.patch.dict(
                os.environ, {**_DEVICE, NONCOLLINEAR_SWITCH: value, "PARSEC_NONCOLLINEAR_SECTOR_STABILITY": "off"}
            ):
                results[value] = run_self_consistent_noncollinear_kpoints(
                    system, np.zeros((1, 3)), np.ones(1), callback=steps.append if value == "on" else None
                )
        from parsec_python.models import SpinSectorFallbackRecord

        notes = [step for step in steps if isinstance(step, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].stage, "the start Hamiltonian")
        self.assertIn("partly filled multiplet", notes[0].reason)
        on, off = results["on"], results["off"]
        self.assertEqual(on.timings["noncollinear_sector_abandoned"], 1.0)
        self.assertGreater(on.timings["noncollinear_guard_start_share"], 0.5)
        self.assertGreater(off.magnetic_moment, 1.5)
        self.assertEqual(on.energies.total, off.energies.total)


if __name__ == "__main__":
    unittest.main()
