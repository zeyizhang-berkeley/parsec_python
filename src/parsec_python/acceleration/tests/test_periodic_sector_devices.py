"""One GPU for the small sector bases of a periodic cell in one process (``PARSEC_PERIODIC_SECTOR_DEVICES``).

CPU: the switch; the bases against the limit (a cluster, one device, both spin channels).  GPU (two devices or more):
the silicon cube keeps every sector on its first GPU under ``auto`` and spreads them under ``all``, with the same
energy to round-off.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from parsec_python.acceleration.tests.test_orbital_group_arrays import _diamond


def _decomposition():
    from parsec_python.acceleration.Symmetry import (
        ReflectionRepresentationDecomposition,
        SignedPermutationReduction,
        detect_point_group_reduction,
    )

    grid, atoms = _diamond(1)
    reduction = detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))
    return grid, ReflectionRepresentationDecomposition.build(grid, reduction)


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.driver import PERIODIC_ONE_DEVICE_BYTES, periodic_sector_devices_setting

        for value, expected in (("", PERIODIC_ONE_DEVICE_BYTES), ("auto", PERIODIC_ONE_DEVICE_BYTES),
                                ("ALL", None), ("0", 0), ("4096", 4096)):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": value}):
                self.assertEqual(periodic_sector_devices_setting(), expected)
        for value in ("on", "-1", "1e9", "off"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_SECTOR_DEVICES"):
                    periodic_sector_devices_setting()


class NoteTests(unittest.TestCase):
    def test_the_bases_against_the_limit(self) -> None:
        from parsec_python.acceleration.driver import _periodic_one_device_note
        from parsec_python.acceleration.Eigensolvers.symmetry import sector_placement_costs

        _, decomposition = _decomposition()
        plan = SimpleNamespace(states=20, buffer=4)
        cell = SimpleNamespace(input=SimpleNamespace(periodic_cell=object(), scf=SimpleNamespace(spin_polarized=False)))
        total = 8 * sum(sector_placement_costs(decomposition, plan))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": "auto"}):
            note = _periodic_one_device_note(cell, decomposition, (0, 1, 2, 3), plan)
            self.assertIsNotNone(note)
            self.assertIn("one GPU", note)
            self.assertIn("PARSEC_PERIODIC_SECTOR_DEVICES", note)
            # one device, a cluster, no filter plan: as before
            self.assertIsNone(_periodic_one_device_note(cell, decomposition, (0,), plan))
            cluster = SimpleNamespace(input=SimpleNamespace(periodic_cell=None, scf=None))
            self.assertIsNone(_periodic_one_device_note(cluster, decomposition, (0, 1), plan))
            self.assertIsNone(_periodic_one_device_note(cell, decomposition, (0, 1), None))
            # one sector (of the identity) is shared by the GPUs on purpose
            single = SimpleNamespace(representation_count=1)
            self.assertIsNone(_periodic_one_device_note(cell, single, (0, 1), plan))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": "all"}):
            self.assertIsNone(_periodic_one_device_note(cell, decomposition, (0, 1), plan))
        # the limit at the bases: spread; above them: one GPU; two spin channels count twice
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": str(int(total))}):
            self.assertIsNone(_periodic_one_device_note(cell, decomposition, (0, 1), plan))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": str(int(total) + 1)}):
            self.assertIsNotNone(_periodic_one_device_note(cell, decomposition, (0, 1), plan))
            spin = SimpleNamespace(input=SimpleNamespace(periodic_cell=object(), scf=SimpleNamespace(spin_polarized=True)))
            self.assertIsNone(_periodic_one_device_note(spin, decomposition, (0, 1), plan))


def _gpus() -> int:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available

        if not cupy_available():
            return 0
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:  # pragma: no cover
        return 0


@unittest.skipUnless(_gpus() >= 2, "needs two CUDA devices")
class PlacementTests(unittest.TestCase):
    def _run(self, setting: str):
        import tempfile
        import warnings

        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_periodic_sector_floor import _silicon

        with tempfile.TemporaryDirectory() as name:
            from pathlib import Path

            problem = _silicon(Path(name))
            with mock.patch.dict(
                os.environ, {"PARSEC_PERIODIC_SECTOR_DEVICES": setting, "PARSEC_PERIODIC_SECTOR_FLOOR": "off"}
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy", symmetry="auto")
                result = driver.run_scf(system)
        details = dict(system.backend_info.details)
        return details.get("orbital_sector_placement", ""), float(result.energies.total)

    def test_one_gpu_for_a_small_cell(self) -> None:
        one, one_energy = self._run("auto")
        spread, spread_energy = self._run("all")
        self.assertIn("one GPU", one)
        self.assertIn("on CUDA device 0", one)
        self.assertNotIn("one GPU", spread)
        self.assertAlmostEqual(one_energy, spread_energy, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
