"""Placement of the symmetry sectors of one process with several GPUs (``main.py``, 2026-11-19).

``main.py`` dealt its sectors to the GPUs in turn (sector ``s`` on GPU ``s mod`` the GPU count), each basis whole on
its GPU, while the MPI full-SCF runner places the sectors of a rank by their cost, with the device pool where the
first solve divides the bases, and shares the one sector of the identity over the GPUs.
``PARSEC_CUPY_SECTOR_PLACEMENT=size`` (the default) gives one process the placement of a rank that owns every
sector (``Eigensolvers/symmetry.process_sector_device_groups``); ``cyclic`` the former deal.  The CPU tests check
the rule and the driver's choice; the GPU tests (two devices or more) that both placements give the same SCF,
forces and exported orbitals, and that the bases are shared where the rule says so.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.implicit_stencil import FilterPlan
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Eigensolvers.symmetry import (
    _PROCESS_POOL_MIN_BYTES,
    process_placement_record,
    process_pool_min_bytes,
    process_pool_share_bytes,
    process_sector_device_groups,
    process_sector_placement_setting,
    sector_placement_costs,
)
from parsec_python.acceleration.experimental.mpi_scf import sector_device_groups

# Rows of the weighted sectors of an odd diamond cell (5 sectors, relative to the largest) and an even one (10).
_ODD = (1.0, 0.9803, 0.9901, 0.9803, 0.9901)
_EVEN = (1.0, 0.9403, 0.9701, 0.9403, 0.9701, 0.9552, 0.9552, 0.9552, 0.9552, 0.9552)
# A plan whose sectors of two million rows reach the threshold of the generalized Ritz step (bases divisible).
_LARGE = FilterPlan(states=20000, buffer=16, applications=200)


def _decomposition(relative, rows=2_000_000):
    sizes = tuple(int(round(rows * value)) for value in relative)
    return SimpleNamespace(representation_count=len(sizes), sector_sizes=sizes, sector_weights=None)


@contextmanager
def _environment(**values):
    """The switches of the placement unset (their defaults) but ``values``."""

    names = ("PARSEC_CUPY_SECTOR_PLACEMENT", "PARSEC_SECTOR_PLACEMENT", "PARSEC_SECTOR_DEVICE_POOL",
             "PARSEC_CUPY_SECTOR_POOL_MIN_BYTES", "PARSEC_CUPY_DISTRIBUTED_STATE", "PARSEC_CUPY_GENERALIZED_RITZ",
             "PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD", "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION")
    with mock.patch.dict(os.environ, values):
        for name in names:
            if name not in values:
                os.environ.pop(name, None)
        yield


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        with _environment():
            self.assertEqual(process_sector_placement_setting(), "size")
        for value, expected in (("size", "size"), (" Cyclic ", "cyclic"), ("", "size")):
            with _environment(PARSEC_CUPY_SECTOR_PLACEMENT=value):
                self.assertEqual(process_sector_placement_setting(), expected)
        for value in ("auto", "on", "pool", "1"):
            with _environment(PARSEC_CUPY_SECTOR_PLACEMENT=value), self.assertRaises(ValueError):
                process_sector_placement_setting()

    def test_the_least_basis_of_the_device_pool(self) -> None:
        with _environment():
            self.assertEqual(process_pool_min_bytes(), _PROCESS_POOL_MIN_BYTES)
        for value, expected in (("0", 0), (" 1048576 ", 1048576), ("", _PROCESS_POOL_MIN_BYTES)):
            with _environment(PARSEC_CUPY_SECTOR_POOL_MIN_BYTES=value):
                self.assertEqual(process_pool_min_bytes(), expected)
        for value in ("-1", "64MiB", "1e6", "auto", "off"):
            with _environment(PARSEC_CUPY_SECTOR_POOL_MIN_BYTES=value), self.assertRaises(ValueError):
                process_pool_min_bytes()
        # the mean sector's FP64 values over the GPUs
        self.assertEqual(process_pool_share_bytes((100.0, 300.0), 4), 8.0 * 400.0 / 8)


class RuleTests(unittest.TestCase):
    devices = (0, 1, 2, 3)

    def test_cyclic_one_device_and_the_former_deal(self) -> None:
        decomposition = _decomposition(_ODD)
        with _environment(PARSEC_CUPY_SECTOR_PLACEMENT="cyclic"):
            self.assertIsNone(process_sector_device_groups(decomposition, self.devices, _LARGE, divided=True))
        with _environment():
            self.assertIsNone(process_sector_device_groups(decomposition, (2,), _LARGE, divided=True))
        self.assertEqual(
            process_placement_record(None, self.devices, 5), "cyclic: sector s on CUDA device s mod 4 (devices 0,1,2,3)"
        )
        self.assertEqual(process_placement_record({}, (1,), 3), "all 3 sectors on CUDA device 1")

    def test_five_divisible_sectors_take_the_device_pool(self) -> None:
        with _environment():
            groups = process_sector_device_groups(_decomposition(_ODD), self.devices, _LARGE, divided=True)
        self.assertEqual(groups, {sector: self.devices for sector in range(5)})
        self.assertEqual(
            process_placement_record(groups, self.devices, 5),
            "size: sectors 0 1 2 3 4 on CUDA devices 0,1,2,3 (device pool, solved in turn)",
        )
        # PARSEC_SECTOR_DEVICE_POOL=off keeps disjoint devices, as for the runner
        with _environment(PARSEC_SECTOR_DEVICE_POOL="off"):
            groups = process_sector_device_groups(_decomposition(_ODD), self.devices, _LARGE, divided=True)
        self.assertTrue(all(len(group) == 1 for group in groups.values()))

    def test_small_sectors_keep_disjoint_groups_below_the_least_basis_of_the_pool(self) -> None:
        # Ten sectors of 41,000 rows and 120 states: 9 MiB per GPU per sector (diamond 6^3 at Gamma: 9.4 MiB)
        decomposition = _decomposition(_EVEN, rows=41_000)
        plan = FilterPlan(states=1200, buffer=0, applications=200)
        costs = sector_placement_costs(decomposition, plan)
        share = process_pool_share_bytes(costs, len(self.devices))
        self.assertLess(share, _PROCESS_POOL_MIN_BYTES)
        with _environment(PARSEC_CUPY_DISTRIBUTED_STATE="on"):
            notes = []
            groups = process_sector_device_groups(decomposition, self.devices, plan, divided=True, notes=notes)
            self.assertTrue(all(len(group) == 1 for group in groups.values()), groups)
            self.assertEqual(len(notes), 1)
            self.assertTrue(notes[0].startswith("disjoint groups: the device pool would give"), notes)
            self.assertIn("below PARSEC_CUPY_SECTOR_POOL_MIN_BYTES = 24 MiB", notes[0])
            # the same disjoint groups as with the pool switched off
            with mock.patch.dict(os.environ, PARSEC_SECTOR_DEVICE_POOL="off"):
                self.assertEqual(groups, process_sector_device_groups(decomposition, self.devices, plan, divided=True))
            # a least basis of 0 takes the pool by the memory rule alone, as the runner does
            with mock.patch.dict(os.environ, PARSEC_CUPY_SECTOR_POOL_MIN_BYTES="0"):
                notes = []
                pooled = process_sector_device_groups(decomposition, self.devices, plan, divided=True, notes=notes)
                self.assertEqual(pooled, {sector: self.devices for sector in range(10)})
                self.assertTrue(notes[0].startswith("device pool:"), notes)
            # a least basis at the share takes the pool too
            with mock.patch.dict(os.environ, PARSEC_CUPY_SECTOR_POOL_MIN_BYTES=str(int(share))):
                self.assertEqual(process_sector_device_groups(decomposition, self.devices, plan, divided=True),
                                 {sector: self.devices for sector in range(10)})
            # without a filter plan the counts are not known and nothing is judged
            notes = []
            unplanned = process_sector_device_groups(decomposition, self.devices, None, divided=True, notes=notes)
            self.assertEqual(unplanned, {sector: self.devices for sector in range(10)})
            self.assertEqual(notes, [])
        # large sectors keep the pool, with its note
        with _environment():
            notes = []
            groups = process_sector_device_groups(_decomposition(_EVEN), self.devices, _LARGE, divided=True,
                                                  notes=notes)
        self.assertEqual(groups, {sector: self.devices for sector in range(10)})
        self.assertTrue(notes[0].startswith("device pool:") and "at least" in notes[0], notes)
        # no pool decision, no note: four sectors on four devices, and the one sector of the identity
        with _environment():
            for relative in ((1.0,) * 4, (1.0,)):
                notes = []
                process_sector_device_groups(_decomposition(relative, rows=1000), self.devices, _LARGE,
                                             divided=True, notes=notes)
                self.assertEqual(notes, [])

    def test_bases_that_are_not_divided_take_one_device_each_costliest_first(self) -> None:
        # CHEBDAV first solve: no pool (it would put every basis on the first device), the costliest sectors alone
        with _environment():
            groups = process_sector_device_groups(_decomposition(_ODD), self.devices, _LARGE, divided=False)
        self.assertTrue(all(len(group) == 1 for group in groups.values()))
        load = {}
        for sector, (device,) in groups.items():
            load[device] = load.get(device, 0.0) + _ODD[sector]
        # the two smallest sectors share a device; cyclic put sectors 0 and 4 (the two largest but one) on GPU 0
        self.assertEqual(sorted(len([s for s, g in groups.items() if g == (d,)]) for d in self.devices), [1, 1, 1, 2])
        self.assertLess(max(load.values()), _ODD[0] + _ODD[4])
        # nor where the sectors are below the threshold of the generalized Ritz step (bases on one device)
        with _environment():
            small = process_sector_device_groups(_decomposition(_ODD, rows=1000), self.devices,
                                                 FilterPlan(states=40, buffer=4, applications=200), divided=True)
        self.assertTrue(all(len(group) == 1 for group in small.values()))

    def test_the_one_sector_of_the_identity_gets_every_device(self) -> None:
        with _environment():
            for divided in (True, False):
                self.assertEqual(
                    process_sector_device_groups(_decomposition((1.0,)), self.devices, _LARGE, divided=divided),
                    {0: self.devices},
                )

    def test_equal_sectors_that_divide_the_devices_keep_the_former_deal(self) -> None:
        with _environment():
            groups = process_sector_device_groups(_decomposition((1.0,) * 8, rows=1000), self.devices, _LARGE,
                                                  divided=True)
        self.assertEqual(groups, {sector: (sector % 4,) for sector in range(8)})
        with _environment():
            groups = process_sector_device_groups(_decomposition((1.0,) * 4, rows=1000), self.devices, _LARGE,
                                                  divided=True)
        self.assertEqual(groups, {sector: (sector,) for sector in range(4)})

    def test_the_placement_of_a_runner_rank_that_owns_every_sector(self) -> None:
        # One process places as the runner's one rank with the GPUs of its node (PARSEC_SECTOR_PLACEMENT=size)
        # where it takes the pool by the memory rule alone, as the runner does (PARSEC_CUPY_SECTOR_POOL_MIN_BYTES=0).
        from parsec_python.acceleration.Eigensolvers.symmetry import sector_bases_divisible, sector_placement_costs

        for relative in (_ODD, _EVEN, (1.0,), (1.0, 0.5), (1.0, 0.98, 0.5)):
            for rows in (1000, 2_000_000):
                decomposition = _decomposition(relative, rows=rows)
                for divided in (True, False):
                    for devices in ((0, 1), (0, 1, 2), self.devices):
                        with _environment(PARSEC_CUPY_SECTOR_POOL_MIN_BYTES="0"):
                            mine = process_sector_device_groups(decomposition, devices, _LARGE, divided=divided)
                            runner = sector_device_groups(
                                len(relative), 1, 0, devices,
                                costs=sector_placement_costs(decomposition, _LARGE), divided=divided,
                                divisible=sector_bases_divisible(decomposition, _LARGE),
                            )
                        self.assertEqual(mine, runner, (relative, rows, divided, devices))


class DriverChoiceTests(unittest.TestCase):
    @staticmethod
    def _reference(*, method="chebff", hybrid=False, meta=False, hubbard=False):
        scf = SimpleNamespace(is_hybrid=hybrid, is_meta_gga=meta)
        species = {"C": SimpleNamespace(has_hubbard=hubbard)}
        return SimpleNamespace(input=SimpleNamespace(scf=scf, pseudopotentials=species,
                                                     eigensolver=SimpleNamespace(method=method)))

    def test_routes_that_keep_the_cyclic_deal_and_the_first_solve(self) -> None:
        from parsec_python.acceleration import driver

        decomposition = _decomposition(_ODD)
        devices = (0, 1, 2, 3)
        with _environment():
            groups, divided, note = driver._process_sector_placement(self._reference(), decomposition, devices,
                                                                     _LARGE)
            self.assertTrue(divided)
            self.assertEqual(groups, {sector: devices for sector in range(5)})
            # the record names the pool and why it was taken (PARSEC_CUPY_SECTOR_POOL_MIN_BYTES)
            self.assertTrue(note.startswith(
                "size: sectors 0 1 2 3 4 on CUDA devices 0,1,2,3 (device pool, solved in turn); device pool: "), note)
            self.assertIn("at least PARSEC_CUPY_SECTOR_POOL_MIN_BYTES = 24 MiB", note)
            # small sectors: disjoint groups, and the record says why
            groups, divided, note = driver._process_sector_placement(
                self._reference(), _decomposition(_EVEN, rows=41_000), devices,
                FilterPlan(states=1200, buffer=0, applications=200))
            self.assertTrue(all(len(group) == 1 for group in groups.values()))
            self.assertIn("; disjoint groups: the device pool would give", note)
            self.assertTrue(note.startswith("size: sectors "), note)
            groups, divided, note = driver._process_sector_placement(
                self._reference(method="chebdav"), decomposition, devices, _LARGE)
            self.assertEqual((divided, note), (False, None))
            self.assertTrue(all(len(group) == 1 for group in groups.values()))
            # below the threshold of the generalized Ritz step the bases stay whole: no pool
            groups, divided, note = driver._process_sector_placement(
                self._reference(), _decomposition(_ODD, rows=1000), devices,
                FilterPlan(states=40, buffer=4, applications=200))
            self.assertEqual((divided, note), (True, None))
            self.assertTrue(all(len(group) == 1 for group in groups.values()))
            for name, reason in (("hybrid", "a hybrid functional"), ("meta", "a meta-GGA functional"),
                                 ("hubbard", "LDA+U")):
                groups, divided, note = driver._process_sector_placement(
                    self._reference(**{name: True}), decomposition, devices, _LARGE)
                self.assertIsNone(groups)
                self.assertIn(reason, note)
            # one device, or cyclic named: the former deal, no note
            self.assertEqual(driver._process_sector_placement(self._reference(), decomposition, (0,), _LARGE),
                             (None, None, None))
        with _environment(PARSEC_CUPY_SECTOR_PLACEMENT="cyclic"):
            self.assertEqual(
                driver._process_sector_placement(self._reference(hybrid=True), decomposition, devices, _LARGE),
                (None, None, None))


class FirstSolveTests(unittest.TestCase):
    def test_the_memory_rules_read_the_first_solve_the_driver_names(self) -> None:
        # The sector solvers hold CHEBFF until their first call; a process placed by size says what the input names
        # (process_divided_bases), as the runner's context does (divided_bases), and nothing named reads the solver.
        from parsec_python.acceleration.Eigensolvers.symmetry import _first_solve_divides_bases

        sector = SimpleNamespace(settings=SimpleNamespace(initial_method="chebff"))
        for named, expected in ((None, True), (True, True), (False, False)):
            solver = SimpleNamespace(mpi_context=None, process_divided_bases=named, _solvers=[sector])
            self.assertEqual(_first_solve_divides_bases(solver, 0), expected)
        solver = SimpleNamespace(mpi_context=SimpleNamespace(divided_bases=False), process_divided_bases=True,
                                 _solvers=[sector])
        self.assertFalse(_first_solve_divides_bases(solver, 0))


def _gpus() -> int:
    if not (cupy_available() and native_available()):
        return 0
    from parsec_python.acceleration.backends.cupy import cupy_device_count

    return int(cupy_device_count())


def _ge5_problem(directory: Path):
    """Five Ge of Td (the cluster of the Td sector-force test): weighted sectors of D2d, CHEBFF."""

    from parsec_python.Input import parse_parsec_input
    from parsec_python.models import Atom
    from parsec_python.tests.force_reference import load_cases, write_case

    problem = parse_parsec_input(write_case(load_cases()["ge2h2_nlcc_tiny"], directory)).problem
    bond = 4.9 / np.sqrt(3.0)
    atoms = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
        Atom("Ge", (bond * x, bond * y, bond * z))
        for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    )
    return replace(
        problem,
        atoms=atoms,
        recenter_geometry=False,
        grid=replace(problem.grid, radius=9.0),
        scf=replace(problem.scf, number_of_states=16),
        eigensolver=replace(problem.eigensolver, method="chebff"),
    )


@unittest.skipUnless(_gpus() >= 2, "two CUDA devices and the native extension are needed")
class RunTests(unittest.TestCase):
    """Both placements on the GPUs of this process: the same SCF, forces and orbitals."""

    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.problem = _ge5_problem(Path(directory.name))

    def _run(self, problem, placement, *, symmetry="on", extra=None, forces=True):
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis

        environment = {"PARSEC_CUPY_SECTOR_PLACEMENT": placement, "PARSEC_CUPY_DEVICES": "auto",
                       # small sectors: share the bases although they are below the threshold of the generalized
                       # Ritz step, and pool them although they are below the least basis of the pool, as large
                       # ones would be
                       "PARSEC_CUPY_DISTRIBUTED_STATE": "on", "PARSEC_CUPY_SECTOR_POOL_MIN_BYTES": "0"}
        environment.update(extra or {})
        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, symmetry=symmetry)
            result = driver.run_scf(system)
            table = driver.compute_forces(system, result, symmetrize=False) if forces else None
            sectors = system.backend.symmetry_eigensolver
            shared = [isinstance(getattr(solver, "device_state", None) and solver.device_state.subspace.vectors,
                                 DistributedBasis) for solver in sectors._solvers]
        self.assertTrue(result.converged)
        return SimpleNamespace(system=system, result=result, forces=table, sectors=sectors, shared=shared,
                               details=dict(result.backend.details))

    def assert_same(self, first, second, *, energy=1.0e-8, force=1.0e-6) -> None:
        self.assertAlmostEqual(first.result.energies.total, second.result.energies.total, delta=energy)
        np.testing.assert_allclose(np.asarray(first.result.eigenvalues), np.asarray(second.result.eigenvalues),
                                   rtol=0, atol=10 * energy)
        np.testing.assert_allclose(np.asarray(first.result.density), np.asarray(second.result.density),
                                   rtol=0, atol=1.0e-6 * float(np.max(np.abs(second.result.density))))
        if first.forces is not None:
            np.testing.assert_allclose(first.forces.total, second.forces.total, rtol=0, atol=force)

    def test_weighted_sectors_in_the_device_pool(self) -> None:
        devices = _gpus()
        size = self._run(self.problem, "size")
        cyclic = self._run(self.problem, "cyclic")
        count = size.sectors.representation_count
        # The weighted sectors of the D2 subgroup of Td, which the rule spreads over all the GPUs here (the pool).
        self.assertGreater(count, 1, size.details.get("orbital_sector_placement"))
        self.assertTrue(size.details["orbital_sector_placement"].startswith("size: sectors"),
                        size.details["orbital_sector_placement"])
        self.assertIn("device pool", size.details["orbital_sector_placement"])
        self.assertTrue(cyclic.details["orbital_sector_placement"].startswith("cyclic"))
        self.assertTrue(all(size.shared), size.shared)
        self.assertFalse(any(cyclic.shared), cyclic.shared)
        self.assertEqual({len(group) for group in size.sectors.sector_device_groups.values()}, {devices})
        self.assertEqual(cyclic.sectors.sector_device_groups, {})
        self.assertGreater(float(np.abs(cyclic.forces.total).max()), 1.0e-3)
        self.assert_same(size, cyclic)
        # the exported orbitals of shared bases are those of the bases on one device (up to their signs)
        exported = [np.asarray(item.result.wavefunctions) for item in (size, cyclic)]
        for item in exported:
            self.assertEqual(item.dtype, np.float64)
        self.assertEqual(exported[0].shape, exported[1].shape)
        scale = float(np.max(np.abs(exported[1])))
        np.testing.assert_allclose(np.abs(exported[0]), np.abs(exported[1]), rtol=0, atol=1.0e-4 * scale)
        np.testing.assert_allclose(np.sum(exported[0] ** 2, axis=1), np.sum(exported[1] ** 2, axis=1),
                                   rtol=0, atol=1.0e-6 * scale**2 * exported[1].shape[1])

    def test_small_sectors_keep_disjoint_groups_by_default(self) -> None:
        # The sectors of Ge5 give a GPU a few KiB of basis in the pool, far below the least basis of the pool.
        size = self._run(self.problem, "size", forces=False, extra={"PARSEC_CUPY_SECTOR_POOL_MIN_BYTES": ""})
        record = size.details["orbital_sector_placement"]
        self.assertTrue(record.startswith("size: sectors"), record)
        self.assertIn("; disjoint groups: the device pool would give", record)
        self.assertNotIn("(device pool, solved in turn)", record)
        devices = tuple(size.sectors.device_ids)
        self.assertTrue(all(tuple(group) != devices for group in size.sectors.sector_device_groups.values()))
        cyclic = self._run(self.problem, "cyclic", forces=False)
        self.assert_same(size, cyclic)

    def test_cyclic_runs_take_the_former_deal_and_shared_bases_export_their_columns(self) -> None:
        # Two cyclic runs of this tree agree to the bit and take the constructor path of before (no device groups,
        # sector s on device s mod D, nothing shared).  That cyclic gives the bits of the tree before the switch
        # (121ef6235) is no claim of this test: it was checked on A100 by running both trees (2026-11-19: Ge5 on
        # four and three GPUs, with the stream scheduler, and spin-polarized O2: the same energy, density and
        # eigenvalue digests).
        from parsec_python.acceleration.Eigensolvers.distributed_state import DistributedBasis

        first = self._run(self.problem, "cyclic", forces=False)
        second = self._run(self.problem, "cyclic", forces=False)
        self.assertEqual(first.result.energies.total, second.result.energies.total)
        np.testing.assert_array_equal(np.asarray(first.result.density), np.asarray(second.result.density))
        devices = first.sectors.device_ids
        count = first.sectors.representation_count
        self.assertEqual(first.sectors.sector_device_groups, {})
        self.assertIsNone(first.sectors.process_divided_bases)
        self.assertEqual(tuple(first.sectors._sector_device_ids),
                         tuple(devices[index % len(devices)] for index in range(count)))
        self.assertFalse(any(first.shared), first.shared)
        self.assertFalse(any(hasattr(operator, "distributed_filter_devices") for operator in first.sectors._operators))
        size = self._run(self.problem, "size", forces=False)
        # to_full_device of the lazy orbitals gathers each shared basis on the current device
        from parsec_python.acceleration.backends.cupy import require_cupy

        cp, _ = require_cupy()
        sectors = size.sectors
        vectors = [solver.device_state.subspace.vectors for solver in sectors._solvers]
        self.assertTrue(all(isinstance(item, DistributedBasis) for item in vectors))
        basis = vectors[0]
        columns = np.array([3, 0, basis.shape[1] - 1, 1])
        host = basis.column_array(columns)
        whole = cp.asnumpy(basis.group.gather(basis))
        np.testing.assert_array_equal(host, whole[:, columns])
        with cp.cuda.Device(basis.group.devices[-1]):
            device = basis.column_array(columns, device=basis.group.devices[-1])
            self.assertEqual(int(device.device.id), basis.group.devices[-1])
            np.testing.assert_array_equal(cp.asnumpy(device), whole[:, columns])

    def test_the_one_sector_of_the_identity_is_shared_by_the_devices(self) -> None:
        extra = {"PARSEC_CUPY_SINGLE_SECTOR": "on"}
        size = self._run(self.problem, "size", symmetry="off", extra=extra)
        cyclic = self._run(self.problem, "cyclic", symmetry="off", extra=extra)
        self.assertEqual(size.sectors.representation_count, 1)
        self.assertEqual(size.shared, [True])
        self.assertEqual(cyclic.shared, [False])
        self.assertEqual(len(size.sectors.sector_device_groups[0]), _gpus())
        self.assert_same(size, cyclic)

    def test_a_moved_structure_continues_in_the_shared_bases(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.models import Atom

        found = {}
        for placement in ("size", "cyclic"):
            item = self._run(self.problem, placement)
            step = 0.04 * item.forces.total / np.abs(item.forces.total).max()
            atoms = tuple(Atom(atom.symbol, atom.position + delta)
                          for atom, delta in zip(item.system.atoms, step))
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_PLACEMENT": placement,
                                              "PARSEC_CUPY_DISTRIBUTED_STATE": "on",
                                              "PARSEC_CUPY_SECTOR_POOL_MIN_BYTES": "0"}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                moved = driver.update_geometry(item.system, atoms)
                warm = driver.run_scf(
                    moved, warm_start=driver.warm_start_for_moved_atoms(item.system, item.result, moved))
                table = driver.compute_forces(moved, warm, symmetrize=False)
            self.assertTrue(warm.converged)
            found[placement] = SimpleNamespace(result=warm, forces=table)
        self.assert_same(found["size"], found["cyclic"])


if __name__ == "__main__":
    unittest.main()
