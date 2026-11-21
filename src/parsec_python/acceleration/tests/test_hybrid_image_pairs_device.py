"""The mirror pairs of a wedge's image sources in the device exchange (``PARSEC_HYBRID_IMAGE_PAIRS``).

``apply_device_bloch_exchange`` with the mirrors (``on``) gives ``W`` of ``off`` to round-off with fewer pair solves,
on one device and on several, and agrees with the host loop.  Random orbitals (the relation is algebraic).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np


def _cupy_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no CUDA here
        return False


@unittest.skipUnless(_cupy_available(), "needs CuPy and a CUDA device")
class DeviceImagePairTests(unittest.TestCase):
    def run_case(self, mesh: str, shift_block: bool = False) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.bloch_exchange_device import (
            DevicePeriodicBox,
            DevicePeriodicPairSolvers,
            apply_device_bloch_exchange,
            device_bloch_channel,
        )
        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.SCF.hybrid_periodic import periodic_pair_potentials
        from parsec_python.tests.unit.test_hybrid_image_pairs import _cube_wedge, _random_set
        from parsec_python.V_xc.bloch_exchange import apply_bloch_exchange

        import parsec_python.acceleration.Hamiltonian.bloch_exchange_device as module

        system, k_points, weights, box, stars = _cube_wedge(mesh, shift_block)
        self.assertTrue(stars.mirrors)
        orbital_set = _random_set(k_points, weights, box, stars)
        potentials = periodic_pair_potentials(system, hybrid_definition(system.input.scf))
        host = apply_bloch_exchange(orbital_set, potentials, columns="all", image_pairs=False)
        count = cp.cuda.runtime.getDeviceCount()
        real_rows_on = module._rows_on
        for devices in ((0,), tuple(range(min(count, 3)))):
            found = {}
            # the devices whose tiles ran (each gathers its occupied rows once per build)
            used: list[int] = []

            def rows_on(cp_, vectors, positions, device):
                used.append(int(device))
                return real_rows_on(cp_, vectors, positions, device)

            for value in ("off", "on"):
                # these builds are far below SMALL_BUILD_POINTS, which would fold the devices into one: 0 spreads the
                # tiles over every device given (lead and follow blocks, and the partner rows of a self block, on
                # other devices than the tiles of their mirrors)
                with mock.patch.dict(os.environ, {"PARSEC_HYBRID_IMAGE_PAIRS": value}), mock.patch.object(
                    module, "SMALL_BUILD_POINTS", 0.0 if len(devices) > 1 else module.SMALL_BUILD_POINTS
                ), mock.patch.object(module, "_rows_on", side_effect=rows_on):
                    solvers = DevicePeriodicPairSolvers(
                        potentials.kernel, DevicePeriodicBox(box.shape, potentials.fft_shape, getattr(box, "flat", None)), 3
                    )
                    channel = device_bloch_channel(orbital_set, box.volume_element)
                    occupied = apply_device_bloch_exchange(channel, solvers, devices, columns="occupied")
                    completed = apply_device_bloch_exchange(channel, solvers, devices, known=occupied)
                    whole = apply_device_bloch_exchange(channel, solvers, devices, columns="all")
                    found[value] = (
                        [cp.asnumpy(block) for block in whole.applied],
                        [cp.asnumpy(block) for block in completed.applied],
                        whole.energy,
                        whole.pair_solves,
                        occupied.pair_solves,
                    )
            self.assertEqual(sorted(set(used)), sorted(devices), devices)
            scale = max(float(np.max(np.abs(block))) for block in host.applied)
            for value in ("off", "on"):
                applied, completed, energy, _solves, _occupied = found[value]
                for a, b, c in zip(applied, completed, host.applied):
                    np.testing.assert_allclose(a, c, rtol=0.0, atol=1.0e-12 * scale)
                    np.testing.assert_allclose(b, c, rtol=0.0, atol=1.0e-12 * scale)
                self.assertAlmostEqual(energy, host.energy, delta=1.0e-12 * abs(host.energy))
            # fewer solves with the mirrors, in the builds of every column and of the occupied ones
            self.assertLess(found["on"][3], found["off"][3], devices)
            self.assertLess(found["on"][4], found["off"][4], devices)

    def test_the_mesh_2x2x2(self) -> None:
        self.run_case("2 2 2")

    def test_the_mesh_4x4x4(self) -> None:
        self.run_case("4 4 4")

    def test_the_fortran_derived_routines(self) -> None:
        self.run_case("2 2 2", shift_block=True)


if __name__ == "__main__":
    unittest.main()
