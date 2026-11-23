"""PARSEC_LINEAR_TABLE_SLOPES on the accelerated cluster routes (real CUDA device).

Under ``interpolant`` the accelerated forces of a species without Cubic_Spline (the local terms summed by the
readable routine on the host, the projector gradients of every builder) are those of the readable routines from the
same result, in symmetry sectors as on the full grid; ``tabulated`` keeps the forces of before.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Forces import compute_forces as readable_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.tests.force_reference import load_cases, write_case

# PARSEC 1.5's numbers: its linear tables and tabulated projector tails (parsec_python.tests.parsec15_tables)
from parsec_python.tests.parsec15_tables import parsec15_tables_module as setUpModule  # noqa: F401, N812

_ENV = "PARSEC_LINEAR_TABLE_SLOPES"


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class AcceleratedLinearSlopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        problem = parse_parsec_input(write_case(load_cases()["ge2h2_nlcc_tiny"], Path(directory.name))).problem
        assert not any(spec.use_spline for spec in problem.pseudopotentials.values())
        cls.forces = {}
        for setting in ("tabulated", "interpolant"):
            for symmetry in ("auto", "off"):
                with mock.patch.dict(os.environ, {_ENV: setting}):
                    system = driver.prepare_single_point(problem, symmetry=symmetry)
                    result = driver.run_scf(system)
                    accelerated = driver.compute_forces(system, result, symmetrize=False)
                    readable = readable_forces(system, result, symmetrize=False) if symmetry == "off" else None
                cls.forces[(setting, symmetry)] = (accelerated, readable)

    def test_accelerated_equals_readable(self) -> None:
        for setting in ("tabulated", "interpolant"):
            accelerated, readable = self.forces[(setting, "off")]
            np.testing.assert_allclose(accelerated.total, readable.total, rtol=0, atol=1e-10, err_msg=setting)

    def test_sectors_equal_the_full_grid(self) -> None:
        for setting in ("tabulated", "interpolant"):
            np.testing.assert_allclose(
                self.forces[(setting, "auto")][0].total, self.forces[(setting, "off")][0].total, rtol=0, atol=2e-6,
                err_msg=setting,
            )

    def test_the_slopes_differ(self) -> None:
        tabulated = self.forces[("tabulated", "off")][0].total
        interpolant = self.forces[("interpolant", "off")][0].total
        self.assertGreater(float(np.max(np.abs(tabulated - interpolant))), 1e-6)


if __name__ == "__main__":
    unittest.main()
