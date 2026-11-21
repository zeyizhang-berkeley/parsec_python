"""The floor of the device pool for the GPUs of an MPI rank (since 2026-11-21).

One process (main.py) spreads several sectors over all its GPUs (the device pool) only where the mean sector gives
every GPU at least ``PARSEC_CUPY_SECTOR_POOL_MIN_BYTES`` (24 MiB) of basis.  The full-SCF runner sets
``MPISectorContext.pool_floor`` under ``PARSEC_CUPY_DEVICE_RANDOM=1`` or ``PARSEC_CUPY_DISTRIBUTED_FILTER=1``
(``runner_pool_floor_applies``), so that the GPUs of each rank follow the same rule there
(``experimental/mpi_scf._pool_floor_groups``); ``0`` takes the pool by the memory rule alone, as before, and so do the
code defaults of both switches.  Every rank's decision reaches
the root's ``parsec.out`` (``_rank_pool_lines``).  CPU only.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext, sector_device_groups
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World

# Rows times first states of the five sectors of an odd diamond cell, relative to the largest (test_sector_placement).
_ODD = (1.0, 0.9803, 0.9901, 0.9803, 0.9901)
_DEVICES = (0, 1, 2, 3)


def _scaled(share_mib):
    """``_ODD`` scaled so that the mean sector gives each of four GPUs ``share_mib`` MiB of FP64 basis."""
    total = share_mib * 2**20 * len(_ODD) * len(_DEVICES) / 8.0
    return tuple(cost * total / sum(_ODD) for cost in _ODD)


class PoolFloorTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        for name in ("PARSEC_CUPY_SECTOR_POOL_MIN_BYTES", "PARSEC_SECTOR_DEVICE_POOL", "PARSEC_SECTOR_PLACEMENT"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()

    def test_small_sectors_keep_the_disjoint_groups_and_large_ones_the_pool(self):
        pooled = {sector: _DEVICES for sector in range(5)}
        for share, expected_pool in ((9.4, False), (12.5, False), (23.9, False), (24.5, True), (51.3, True)):
            with self.subTest(share=share):
                costs = _scaled(share)
                disjoint = sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True,
                                                divisible=(True,) * 5, pool_floor=False, notes=None)
                # Without the floor: the pool, as before 2026-11-21.
                self.assertEqual(disjoint, pooled)
                notes = []
                groups = sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True, divisible=(True,) * 5,
                                              pool_floor=True, notes=notes)
                if expected_pool:
                    self.assertEqual(groups, pooled)
                    self.assertTrue(notes[0].startswith("device pool:"), notes)
                else:
                    with patch.dict(os.environ, {"PARSEC_SECTOR_DEVICE_POOL": "off"}):
                        expected = sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True)
                    self.assertEqual(groups, expected)
                    self.assertEqual({group[0] for group in groups.values()}, set(_DEVICES))
                    self.assertTrue(notes[0].startswith("disjoint groups:"), notes)
                    self.assertIn(f"{share:.1f} MiB of basis per GPU per sector", notes[0])
                    self.assertIn("PARSEC_CUPY_SECTOR_POOL_MIN_BYTES = 24 MiB", notes[0])
                    # 0: the memory rule alone, the layout of before.
                    with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_POOL_MIN_BYTES": "0"}):
                        self.assertEqual(sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True,
                                                              pool_floor=True), pooled)

    def test_the_floor_touches_no_layout_without_a_pool(self):
        costs = _scaled(1.0)
        # Bases that the first solve does not divide, a pool switched off, the cyclic deal, one GPU, one sector:
        # the groups of before, with or without the floor.
        cases = [dict(divided=False), dict(divided=True, divisible=(True,) * 4 + (False,))]
        for options in cases:
            with self.subTest(options=options):
                self.assertEqual(sector_device_groups(5, 1, 0, _DEVICES, costs=costs, pool_floor=True, **options),
                                 sector_device_groups(5, 1, 0, _DEVICES, costs=costs, **options))
        for environment in ({"PARSEC_SECTOR_DEVICE_POOL": "off"}, {"PARSEC_SECTOR_PLACEMENT": "cyclic"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                self.assertEqual(sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True, pool_floor=True),
                                 sector_device_groups(5, 1, 0, _DEVICES, costs=costs, divided=True))
        self.assertEqual(sector_device_groups(5, 1, 0, (0,), costs=costs, divided=True, pool_floor=True),
                         {sector: (0,) for sector in range(5)})
        self.assertEqual(sector_device_groups(1, 1, 0, _DEVICES, costs=costs[:1], divided=True, pool_floor=True),
                         {0: _DEVICES})
        # Without costs (no filter plan) the floor is not judged.
        self.assertEqual(sector_device_groups(5, 1, 0, _DEVICES, divided=True, pool_floor=True),
                         {sector: _DEVICES for sector in range(5)})

    def test_an_unknown_floor_is_refused(self):
        for value in ("-1", "24MiB", "1.5"):
            with self.subTest(value=value), patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_POOL_MIN_BYTES": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_SECTOR_POOL_MIN_BYTES"):
                    sector_device_groups(5, 1, 0, _DEVICES, costs=_scaled(1.0), divided=True, pool_floor=True)

    def test_the_context_plans_and_configures_alike_and_keeps_the_reason(self):
        from parsec_python.acceleration import driver

        for share, pooled in ((9.4, False), (51.3, True)):
            with self.subTest(share=share):
                costs = _scaled(share)
                context = MPISectorContext(_Comm(_World(1), 0))
                context.divided_bases = True
                context.pool_floor = True
                planned = context.planned_groups(5, _DEVICES, costs=costs)
                self.assertEqual(driver._planned_sector_device_groups(5, _DEVICES, context, costs=costs), planned)
                self.assertEqual(context.configure(5, _DEVICES, costs=costs), planned)
                self.assertEqual(set(planned.values()) == {_DEVICES}, pooled)
                self.assertTrue(context.pool_note.startswith("device pool:" if pooled else "disjoint groups:"))
        # A context of before (the default of the attribute): the pool by the memory rule alone, no reason.
        context = MPISectorContext(_Comm(_World(1), 0))
        context.divided_bases = True
        self.assertEqual(context.configure(5, _DEVICES, costs=_scaled(9.4)), {sector: _DEVICES for sector in range(5)})
        self.assertIsNone(context.pool_note)

    def test_the_runner_line(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _rank_devices_text

        context = SimpleNamespace(device_groups={0: (0,), 1: (3,), 2: (1,), 3: (3,), 4: (2,)})
        self.assertEqual(_rank_devices_text(context),
                         "sectors 0 on GPU 0; sectors 1 3 on GPU 3; sectors 2 on GPU 1; sectors 4 on GPU 2")
        context = SimpleNamespace(device_groups={sector: _DEVICES for sector in range(3)})
        self.assertEqual(_rank_devices_text(context), "sectors 0 1 2 on GPUs 0,1,2,3")


class _GatherComm(_Comm):
    """``_Comm`` with ``gather`` to a root (the list there, ``None`` on the other ranks)."""

    def gather(self, value, root=0):
        values = self.allgather(value)
        return values if self.rank == root else None


class RunnerFloorRuleTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        for name in ("PARSEC_CUPY_SECTOR_POOL_MIN_BYTES", "PARSEC_SECTOR_DEVICE_POOL", "PARSEC_SECTOR_PLACEMENT",
                     "PARSEC_CUPY_DISTRIBUTED_FILTER", "PARSEC_CUPY_DEVICE_RANDOM"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()

    def test_the_runner_judges_the_floor_under_the_launch_settings_only(self):
        from parsec_python.acceleration.experimental.mpi_scf import runner_pool_floor_applies

        # The code defaults of both switches (unset or 0): the pool by the memory rule alone, as before.
        os.environ.pop("PARSEC_CUPY_DEVICE_RANDOM", None)
        self.assertFalse(runner_pool_floor_applies())
        cases = (
            # the tests of Eigensolvers/chebff.py (start vectors on the GPU) and chebyshev.py (distributed filter)
            ({"PARSEC_CUPY_DEVICE_RANDOM": "0", "PARSEC_CUPY_DISTRIBUTED_FILTER": "0"}, False),
            ({"PARSEC_CUPY_DEVICE_RANDOM": "1"}, True),
            ({"PARSEC_CUPY_DEVICE_RANDOM": "on"}, True),
            ({"PARSEC_CUPY_DEVICE_RANDOM": "True"}, True),
            ({"PARSEC_CUPY_DISTRIBUTED_FILTER": "1"}, True),
            ({"PARSEC_CUPY_DISTRIBUTED_FILTER": "on"}, False),
            ({"PARSEC_CUPY_DEVICE_RANDOM": "1", "PARSEC_CUPY_DISTRIBUTED_FILTER": "0"}, True),
            ({"PARSEC_CUPY_DEVICE_RANDOM": "", "PARSEC_CUPY_DISTRIBUTED_FILTER": ""}, False),
        )
        for environment, applies in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment):
                self.assertIs(runner_pool_floor_applies(), applies)

    def test_ranks_of_three_sectors_out_of_ten_take_disjoint_groups(self):
        # Ten equal sectors of 9.4 MiB per GPU on four ranks of four GPUs (diamond 6^3, one rank per node): the
        # ranks that own three of them took the pool before and take disjoint groups under the floor; those of two
        # keep their groups.
        costs = (9.4 * 2**20 * 4 / 8.0,) * 10
        owners = {}
        for rank in range(4):
            before = sector_device_groups(10, 4, rank, _DEVICES, costs=costs, divided=True)
            notes = []
            after = sector_device_groups(10, 4, rank, _DEVICES, costs=costs, divided=True, pool_floor=True,
                                         notes=notes)
            owners[rank] = len(before)
            if len(before) == 3:
                self.assertEqual(set(before.values()), {_DEVICES})
                self.assertEqual(sorted(len(group) for group in after.values()), [1, 1, 2])
                self.assertTrue(notes and notes[0].startswith("disjoint groups:"), notes)
            else:
                self.assertEqual(after, before)
                self.assertEqual(notes, [])
        self.assertEqual(sorted(owners.values()), [2, 2, 3, 3])

    def test_every_rank_decision_reaches_the_root(self):
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _rank_pool_lines

        world = _World(4)
        notes = {1: "disjoint groups: the device pool would give 9.4 MiB", 2: "device pool: 30.0 MiB"}
        groups = {0: {0: (0, 1), 1: (2, 3)}, 1: {2: (0,), 3: (1,), 4: (2, 3)}, 2: {5: _DEVICES, 6: _DEVICES},
                  3: {7: (0, 1), 8: (2, 3)}}
        results = {}

        def run(rank):
            context = SimpleNamespace(pool_floor=True, pool_note=notes.get(rank), device_groups=groups[rank])
            results[rank] = _rank_pool_lines(_GatherComm(world, rank), context)

        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results[0], [
            "Sector devices of rank 1: sectors 2 on GPU 0; sectors 3 on GPU 1; sectors 4 on GPUs 2,3; "
            "disjoint groups: the device pool would give 9.4 MiB",
            "Sector devices of rank 2: sectors 5 6 on GPUs 0,1,2,3; device pool: 30.0 MiB",
        ])
        self.assertEqual([results[rank] for rank in (1, 2, 3)], [None, None, None])
        # Without the floor (the code defaults of the start vectors and the filter) nothing is gathered or written.
        self.assertIsNone(_rank_pool_lines(object(), SimpleNamespace(pool_floor=False, pool_note=None)))
        self.assertIsNone(_rank_pool_lines(object(), None))


if __name__ == "__main__":
    unittest.main()
