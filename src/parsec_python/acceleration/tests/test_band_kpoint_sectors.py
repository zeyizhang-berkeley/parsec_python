"""Band k-points of a Gamma cell solved in the sectors of their little groups (``PARSEC_BAND_KPOINT_SECTORS``).

CPU: the operations a sector SCF hands to the bands (:class:`BandSymmetry`), the test that admits only them, and the
switch.  GPU: the silicon cube with a band path through Gamma, X, R and a general line, its SCF in the weighted
sectors and its bands in the sectors of the little groups (the floor of the k-point sectors lowered so that the small
cell splits) against the bands of every k-point on the whole grid.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.acceleration.tests.test_cli_periodic_sector_forces import displaced_positions, write_silicon


class BandSymmetryTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        from parsec_python.acceleration.SCF.band_sectors import band_kpoint_sectors_setting

        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_BAND_KPOINT_SECTORS": value}):
                self.assertEqual(band_kpoint_sectors_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_BAND_KPOINT_SECTORS": "on"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_BAND_KPOINT_SECTORS"):
                band_kpoint_sectors_setting()

    def test_only_the_operations_of_the_scf_are_admitted(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Input import parse_parsec_input
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.acceleration.SCF.band_sectors import BandSymmetry, _band_invariance_class
        from parsec_python.acceleration.SCF.kpoint_sectors import CellGroup

        with tempfile.TemporaryDirectory() as name:
            problem = parse_parsec_input(write_silicon(Path(name), positions=displaced_positions(0.0))).problem
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        cell = CellGroup(grid, problem.atoms)
        self.assertGreater(cell.order, 8)
        counts = grid.points_per_axis
        # the operations about the origin without translations: a symmetry of the potential that leaves the
        # face-centring translations out
        about_origin = [
            index for index in range(cell.order) if not np.any(np.asarray(cell.shifts[index]) % np.asarray(counts))
        ]
        symmetry = BandSymmetry.of_reduction(
            SimpleNamespace(
                group_order=len(about_origin),
                full_operations=np.asarray([cell.matrices[index] for index in about_origin]),
                full_translations=np.zeros((len(about_origin), 3), dtype=np.int64),
            ),
            grid,
        )
        self.assertEqual(symmetry.count, len(about_origin))
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
        k_points = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        invariance = _band_invariance_class()(cell, symmetry, k_points, np.ones(2), lattice)
        for index in range(cell.order):
            self.assertEqual(invariance(index), index in about_origin, index)
            self.assertEqual(invariance.in_density_group(index), index in about_origin, index)
        self.assertIsNone(BandSymmetry.of_reduction(SimpleNamespace(group_order=1), grid))


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available
        from parsec_python.acceleration.backends.native import native_available

        return bool(cupy_available() and native_available())
    except Exception:  # pragma: no cover
        return False


#: Gamma-X and Gamma-R of the simple cubic cell (little groups C4v and C3v with the face-centring translations), and
#: a general line (no operation but the identity: the k-points stay whole).
PATH = (
    "begin bandstruc\n"
    "1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\n"
    "1 0.0 0.0 0.0  0.5 0.5 0.5  G-R\n"
    "1 0.1 0.2 0.3  0.2 0.3 0.4  A-B\n"
    "end bandstruc\nbandstruc_points 2\n"
)
SETTINGS = {
    "Grid_Spacing": "0.57",
    "Expansion_Order": "8",
    "States_Num": "22",
    "Convergence_Criterion": "1.0e-9",
    "Diag_Tolerance": "1.0e-10",
    "Correlation_Type": "pbe",
}


def _bands(directory: Path, setting: str, transform=None):
    import parsec_python.acceleration.cli as cli_module
    from parsec_python.acceleration.SCF import band_sectors, kpoint_sectors

    path = write_silicon(directory, PATH, positions=displaced_positions(0.0), settings=SETTINGS)
    calls = []
    original = band_sectors.solve_band_sectors

    def recorder(*arguments, **keywords):
        found = original(*arguments, **keywords)
        calls.append(found is not None)
        return found if transform is None or found is None else transform(found)

    errors = StringIO()
    variables = {"PARSEC_BAND_KPOINT_SECTORS": setting, "PARSEC_PERIODIC_SECTOR_BANDS": "auto"}
    with mock.patch.dict(os.environ, variables), mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0), \
            mock.patch.object(band_sectors, "solve_band_sectors", side_effect=recorder), \
            redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", "--backend", "cupy"])
    bands = {}
    if status == 0:
        with np.load(directory / "parsec_python_bands.npz", allow_pickle=True) as data:
            bands = {key: data[key] for key in data.files}
    output = directory / "parsec.out"
    log = output.read_text(encoding="utf-8", errors="replace") if output.exists() else ""
    return status, bands, calls, errors.getvalue() + log


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class SectorBandPathTests(unittest.TestCase):
    def test_the_bands_of_the_sectors_equal_those_of_the_whole_grid(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            status, sectors, calls, errors = _bands(root / "sectors", "auto")
            self.assertEqual(status, 0, errors[-4000:])
            self.assertEqual(calls, [True])
            self.assertIn("Band k-points in the sectors of their little groups", errors)
            status, whole, calls, errors = _bands(root / "whole", "off")
            self.assertEqual(status, 0, errors[-4000:])
            self.assertEqual(calls, [])
            self.assertNotIn("Band k-points in the sectors of their little groups", errors)
        values, reference = np.asarray(sectors["eigenvalues_ry"]), np.asarray(whole["eigenvalues_ry"])
        self.assertEqual(values.shape, reference.shape)
        self.assertEqual(values.shape[-1], 22)
        difference = float(np.abs(values - reference).max())
        print(f"\n largest band difference, sectors of the little groups - whole k-points: {difference:.3e} Ry")
        np.testing.assert_allclose(values, reference, rtol=0, atol=1e-8)
        self.assertLess(float(np.nanmax(sectors["residual_norms"])), 1e-6)

    def test_a_chebdav_solve_re_solves_a_k_point_beyond_the_residual_tolerance_whole(self) -> None:
        # the first k-point reported with a residual of 1 Ry by the sectors: the CHEBDAV first solve of the input
        # re-solves it whole (SCF/bands.py, residual_tolerance 2e-2) and keeps its levels
        def spoiled(found):
            values, residuals = found
            residuals = np.array(residuals, copy=True)
            residuals[0] = 1.0
            return np.array(values, copy=True), residuals

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            status, spoilt, calls, errors = _bands(root / "spoilt", "auto", transform=spoiled)
            self.assertEqual(status, 0, errors[-4000:])
            self.assertEqual(calls, [True])
            self.assertIn("Chebyshev-Davidson", errors)
            status, whole, calls, errors = _bands(root / "whole", "off")
            self.assertEqual(status, 0, errors[-4000:])
        self.assertLess(float(spoilt["residual_norms"][0, 0]), 1e-6)
        self.assertGreaterEqual(int(spoilt["solver_attempts"][0, 0]), 2)
        self.assertEqual(int(np.max(spoilt["solver_attempts"][0, 1:])), 1)
        np.testing.assert_allclose(spoilt["eigenvalues_ry"], whole["eigenvalues_ry"], rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
