"""Worker processes for the k-point sectors of one process (``acceleration/SCF/kpoint_processes.py``).

The static objects of the sector Hamiltonians go to a worker process once and are found again by identity; the route
gives the numbers of the threads it replaces, bit for bit (the same partition of the tasks, the same operations, the same
order of the sums): here on the host double of the device code with two workers (``PARSEC_KPOINT_BACKEND=numpy-device``,
``PARSEC_KPOINT_WORKERS=2``, ``PARSEC_KPOINT_SECTOR_PROCESSES=on``) for LDA+U, both spin channels, the forces and a sector
that grows, and on two CUDA devices where there are.
"""

from __future__ import annotations

from dataclasses import replace
import io
import os
import pickle
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.acceleration.SCF import kpoint_processes as processes
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.tests import test_kpoint_sectors as sector_tests

try:
    from parsec_python.acceleration.backends.cupy import cupy_available, cupy_device_count
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

    def cupy_device_count() -> int:
        return 0


def setUpModule() -> None:
    # small cells: split them all the same
    for name in ("_MINIMUM_ENTRIES", "_META_MINIMUM_ENTRIES"):
        patcher = mock.patch.object(sectors, name, 0)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)


def _two_cuda_devices() -> bool:
    try:
        return bool(cupy_available()) and int(cupy_device_count()) >= 2
    except Exception:  # noqa: BLE001
        return False


_HOST = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "2"}


class SettingTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off"), ("on", "on")):
            with mock.patch.dict(os.environ, {processes.SWITCH: value}):
                self.assertEqual(processes.sector_processes_setting(), expected)
        with mock.patch.dict(os.environ, {processes.SWITCH: "threads"}):
            with self.assertRaises(ValueError):
                processes.sector_processes_setting()
            with self.assertRaises(ValueError):
                sectors.check_kpoint_sector_switches()

    def test_where_processes_are_expected(self) -> None:
        # (the host double only under "on"; one device or the threads never)
        for switches, expected in (
            ({**_HOST, processes.SWITCH: "on"}, True),
            ({**_HOST, processes.SWITCH: "auto"}, False),
            ({**_HOST, processes.SWITCH: "off"}, False),
            ({**_HOST, "PARSEC_KPOINT_WORKERS": "1", processes.SWITCH: "on"}, False),
            ({**_HOST, "PARSEC_KPOINT_SOLVER": "host", processes.SWITCH: "on"}, False),
        ):
            with self.subTest(switches=switches), mock.patch.dict(os.environ, switches):
                self.assertEqual(processes.processes_expected(), expected)


class MetaFloorTests(unittest.TestCase):
    """A meta-GGA on several devices takes the floor of one device where worker processes solve its sectors, and
    that of several where a k-point stays whole (the threads then solve them)."""

    def _calls(self, expected: bool, plans):
        from parsec_python.SCF.kpoint_solver import KPointJob

        job = KPointJob(
            n_tasks=2, number_of_states=4, eigval_settings=lambda *args: None,
            make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {},
            geometry={"meta_fields": lambda task: (1.0, "tau", None)},
        )
        calls = []

        def planned(job, notes=None, *, devices=1):
            calls.append(int(devices))
            return plans

        with mock.patch.object(sectors, "plan_sectors", planned), mock.patch.object(
            sectors, "_solver_devices", lambda backend=None: 4
        ), mock.patch.object(processes, "processes_expected", lambda backend=None: expected), mock.patch.object(
            sectors, "SectorKPointSolver", lambda *args: "solver"
        ):
            sectors.sector_solver(job, lambda *args, **kw: None)
        return calls

    def test_the_floor_of_the_devices_that_solve_the_sectors(self) -> None:
        split = mock.Mock(real_forms=None, sizes=(10, 10), counts=(2, 2))
        self.assertEqual(self._calls(True, [split, split]), [1])
        self.assertEqual(self._calls(False, [split, split]), [4])
        # a whole k-point, or sectors above the size of the workers: planned again with the floor of the four devices
        self.assertEqual(self._calls(True, [split, None]), [1, 4])
        with mock.patch.object(processes, "_TASK_ENTRIES", 19), mock.patch.dict(os.environ, {processes.SWITCH: "auto"}):
            self.assertEqual(self._calls(True, [split, split]), [1, 4])


class SizeRuleTests(unittest.TestCase):
    """Under ``auto`` the workers take sectors whose median task holds at most ``_TASK_ENTRIES`` rows times states;
    ``on`` takes them at any size; extras need the parts of the job's force."""

    def test_the_median_task(self) -> None:
        self.assertEqual(processes.task_entries([10, 100, 1000], [1, 2, 3]), 200)
        self.assertEqual(processes.task_entries([], []), 0)

    def test_what_the_plans_take(self) -> None:
        from parsec_python.SCF.kpoint_solver import KPointJob

        def job(extras=None, force_parts=None):
            return KPointJob(
                n_tasks=1, number_of_states=4, eigval_settings=lambda *args: None,
                make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {}, extras=extras,
                geometry={"force_parts": force_parts},
            )

        plan = mock.Mock(real_forms=None, sizes=(100, 100, 100), counts=(5, 5, 5))
        with mock.patch.dict(os.environ, {processes.SWITCH: "auto"}):
            self.assertTrue(processes.plans_take_processes(job(), [plan]))
            with mock.patch.object(processes, "_TASK_ENTRIES", 499):
                self.assertFalse(processes.plans_take_processes(job(), [plan]))
            self.assertFalse(processes.plans_take_processes(job(), [plan, None]))
            self.assertFalse(processes.plans_take_processes(job(extras=lambda *a, **k: 0.0), [plan]))
            self.assertTrue(processes.plans_take_processes(job(extras=lambda *a, **k: 0.0, force_parts=print), [plan]))
            # a loop with forces up to the larger bound
            with mock.patch.object(processes, "_TASK_ENTRIES", 499), mock.patch.object(
                processes, "_FORCE_TASK_ENTRIES", 500
            ):
                self.assertFalse(processes.plans_take_processes(job(), [plan]))
                forces = job(extras=lambda *a, **k: 0.0, force_parts=print)
                self.assertTrue(processes.plans_take_processes(forces, [plan]))
        with mock.patch.dict(os.environ, {processes.SWITCH: "on"}), mock.patch.object(processes, "_TASK_ENTRIES", 499):
            self.assertTrue(processes.plans_take_processes(job(), [plan]))


class ShippingTests(unittest.TestCase):
    def test_static_objects_go_once(self) -> None:
        shipping = processes._Shipping()
        cache: dict = {}
        static = sp.identity(5, format="csr")
        found = []
        for step in range(3):
            potential = np.full(5, float(step))
            data = shipping.dumps({"kinetic": static, "again": static, "potential": potential})
            message = processes._CacheUnpickler(io.BytesIO(data), cache).load()
            found.append(message)
            if step:
                # the static object is a reference after the first message, the potential goes every time
                self.assertLess(len(data), len(pickle.dumps(static)))
        self.assertIs(found[0]["kinetic"], found[0]["again"])
        self.assertIs(found[1]["kinetic"], found[0]["kinetic"])
        self.assertIs(found[2]["kinetic"], found[0]["kinetic"])
        np.testing.assert_array_equal(found[2]["potential"], np.full(5, 2.0))
        self.assertEqual(len(cache), 1)
        shipping.clear()
        data = shipping.dumps({"kinetic": static})
        fresh: dict = {}
        self.assertEqual(processes._CacheUnpickler(io.BytesIO(data), fresh).load()["kinetic"].shape, (5, 5))


class HostDoubleTests(unittest.TestCase):
    """The worker processes on the host double give the numbers of the threads."""

    def _silicon(self, **scf):
        base = sector_tests.SectorSCFTests._silicon(self)
        return replace(base, scf=replace(base.scf, **scf)) if scf else base

    def _gamma_and_x(self, problem):
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        return np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])]), np.array([0.5, 0.5])

    def _both(self, run, problem, environment=None):
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        results = {}
        for setting in ("off", "on"):
            switches = {"PARSEC_KPOINT_SECTORS": "auto", processes.SWITCH: setting, **_HOST, **(environment or {})}
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[setting] = run(prepare_periodic_single_point(problem))
        threads, workers = results["off"], results["on"]
        self.assertTrue(threads.converged and workers.converged)
        self.assertNotIn("kpoint_sector_processes", threads.timings.detail)
        self.assertEqual(workers.timings.detail["kpoint_sector_processes"], 2.0)
        self.assertEqual(workers.energies.total, threads.energies.total)
        return threads, workers

    def test_spin_channels(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized

        problem = self._silicon(spin_polarized=True)
        k_points, weights = self._gamma_and_x(problem)
        threads, workers = self._both(lambda system: run_scf_kpoints_spin_polarized(system, k_points, weights), problem)
        np.testing.assert_array_equal(workers.eigenvalues_up, threads.eigenvalues_up)
        np.testing.assert_array_equal(workers.eigenvalues_down, threads.eigenvalues_down)

    def test_forces(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints

        problem = self._silicon()
        atoms = list(problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
        problem = replace(problem, atoms=atoms)
        k_points, weights = self._gamma_and_x(problem)
        threads, workers = self._both(
            lambda system: run_scf_kpoints(system, k_points, weights, compute_forces=True), problem
        )
        self.assertGreater(float(np.max(np.abs(threads.nonlocal_forces))), 1.0e-3)
        np.testing.assert_array_equal(workers.nonlocal_forces, threads.nonlocal_forces)

    def test_a_sector_that_grows(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints

        problem = self._silicon()
        k_points, weights = self._gamma_and_x(problem)
        with mock.patch.object(sectors, "_EXTRA_STATES", 0):
            threads, workers = self._both(
                lambda system: run_scf_kpoints(system, k_points, weights), problem,
                {"PARSEC_KPOINT_SECTOR_MARGIN": "1.0"},
            )
        self.assertGreaterEqual(workers.timings.detail["kpoint_sector_regrowths"], 1.0)
        self.assertEqual(workers.timings.detail["kpoint_sector_regrowths"], threads.timings.detail["kpoint_sector_regrowths"])

    def test_meta_gga(self) -> None:
        # SCAN in both forms of the density gradient, an atom moved along x (the classes of the sectors of X), forces
        from parsec_python.SCF.kpoints import run_scf_kpoints

        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scan = silicon("scan", spacing=0.855)
        problem = replace(scan, scf=replace(scan.scf, convergence_criterion=1.0e-9, max_iterations=120),
                          eigensolver=replace(scan.eigensolver, tolerance=1.0e-10))
        atoms = list(problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
        problem = replace(problem, atoms=atoms)
        k_points, weights = self._gamma_and_x(problem)
        for form in ("orbital", "density"):
            with self.subTest(form=form):
                threads, workers = self._both(
                    lambda system: run_scf_kpoints(system, k_points, weights, compute_forces=True), problem,
                    {"PARSEC_META_GGA_GRADIENT": form},
                )
                self.assertGreater(workers.timings.detail["kpoint_sector_tasks"], 2.0)
                np.testing.assert_array_equal(workers.nonlocal_forces, threads.nonlocal_forces)

    def test_processes_that_do_not_start_leave_the_threads(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = self._silicon()
        k_points, weights = self._gamma_and_x(problem)
        found = {}
        for setting in ("off", "on"):
            switches = {"PARSEC_KPOINT_SECTORS": "auto", processes.SWITCH: setting, **_HOST}
            refused = mock.patch.object(
                processes._WorkerProcesses, "ensure_ready", side_effect=processes._StartError("refused")
            )
            with mock.patch.dict(os.environ, switches), refused, mock.patch.object(processes, "_FAILED", []):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    found[setting] = run_scf_kpoints(prepare_periodic_single_point(problem), k_points, weights)
                said = [str(item.message) for item in caught if processes.SWITCH in str(item.message)]
                self.assertEqual(bool(said), setting == "on")
        self.assertNotIn("kpoint_sector_processes", found["on"].timings.detail)
        self.assertGreater(found["on"].timings.detail["kpoint_sector_tasks"], 2.0)
        self.assertEqual(found["on"].energies.total, found["off"].energies.total)

    def test_workers_whose_solvers_cannot_be_made_leave_the_threads(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = self._silicon()
        k_points, weights = self._gamma_and_x(problem)
        original = processes.ProcessKPointSolver._all

        def failing(solver, method, arguments, indices=None):
            if method == "init":
                raise processes.WorkerProcessError("worker process 0 failed: RuntimeError: no context")
            return original(solver, method, arguments, indices)

        found = {}
        for setting in ("off", "on"):
            switches = {"PARSEC_KPOINT_SECTORS": "auto", processes.SWITCH: setting, **_HOST}
            failures = mock.patch.object(processes, "_FAILED", [])
            initial = mock.patch.object(processes.ProcessKPointSolver, "_all", failing)
            with mock.patch.dict(os.environ, switches), failures, initial:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    found[setting] = run_scf_kpoints(prepare_periodic_single_point(problem), k_points, weights)
                said = [str(item.message) for item in caught if processes.SWITCH in str(item.message)]
                self.assertEqual(bool(said), setting == "on")
        self.assertNotIn("kpoint_sector_processes", found["on"].timings.detail)
        self.assertEqual(found["on"].energies.total, found["off"].energies.total)

    def test_ldau(self) -> None:
        runner = sector_tests.LDAUSectorTests()
        found = {}
        for setting in ("off", "on"):
            with mock.patch.dict(os.environ, {processes.SWITCH: setting, "PARSEC_KPOINT_WORKERS": "2"}):
                found[setting] = runner._run("on")
        runner.doCleanups()
        threads, workers = found["off"], found["on"]
        self.assertTrue(threads.converged and workers.converged)
        self.assertEqual(workers.timings.detail["kpoint_sector_processes"], 2.0)
        self.assertEqual(workers.energies.total, threads.energies.total)
        self.assertEqual(workers.energies.hubbard, threads.energies.hubbard)
        self.assertGreater(abs(threads.energies.hubbard), 1.0e-4)


@unittest.skipUnless(_two_cuda_devices(), "needs CuPy with two CUDA devices")
class CudaTests(unittest.TestCase):
    """On two devices the worker processes (the default) give the numbers of the device threads."""

    def test_silicon_mesh(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = sector_tests.SectorSCFTests._silicon(self)
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.array(bits, dtype=float) * np.pi / side for bits in np.ndindex(2, 2, 2)])
        weights = np.full(8, 0.125)
        results = {}
        for setting in ("off", "auto"):
            switches = {"PARSEC_KPOINT_SECTORS": "auto", "PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_CUPY_DEVICES": "0,1",
                        processes.SWITCH: setting}
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[setting] = run_scf_kpoints(prepare_periodic_single_point(problem), k_points, weights,
                                                   compute_forces=True)
        threads, workers = results["off"], results["auto"]
        self.assertTrue(threads.converged and workers.converged)
        self.assertEqual(workers.timings.detail.get("kpoint_sector_processes"), 2.0)
        self.assertEqual(workers.energies.total, threads.energies.total)
        np.testing.assert_array_equal(workers.nonlocal_forces, threads.nonlocal_forces)

    def test_ldau(self) -> None:
        runner = sector_tests.LDAUSectorTests()
        found = {}
        for setting in ("off", "auto"):
            switches = {processes.SWITCH: setting, "PARSEC_CUPY_DEVICES": "0,1"}
            with mock.patch.dict(os.environ, switches):
                found[setting] = runner._run("on", backend="cupy")
        runner.doCleanups()
        threads, workers = found["off"], found["auto"]
        self.assertTrue(threads.converged and workers.converged)
        self.assertEqual(workers.timings.detail.get("kpoint_sector_processes"), 2.0)
        self.assertEqual(workers.energies.total, threads.energies.total)
        self.assertEqual(workers.energies.hubbard, threads.energies.hubbard)


if __name__ == "__main__":
    unittest.main()
