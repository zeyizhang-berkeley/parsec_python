"""Converged CHEBFF band k-points on the GPU (``PARSEC_BAND_CONVERGENCE``, ``SCF/bands.py``).

The silicon cube with a band path through Gamma, X, R and a general line, ``Eigensolver: chebff`` with one filter
cycle (``FF_MaxIter`` 1), its bands on the device in the sectors of the little groups (the floor of the k-point
sectors lowered so that the small cell splits) and on the whole grid: under ``auto`` the bands reach CHEBDAV's
residual test and those of ``Eigensolver: chebdav``; under ``off`` the one cycle leaves them unconverged.
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


def _cupy_ready() -> bool:
    try:
        from parsec_python.acceleration.backends.cupy import cupy_available
        from parsec_python.acceleration.backends.native import native_available

        return bool(cupy_available() and native_available())
    except Exception:  # pragma: no cover
        return False


PATH = (
    "begin bandstruc\n"
    "1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\n"
    "1 0.0 0.0 0.0  0.5 0.5 0.5  G-R\n"
    "1 0.1 0.2 0.3  0.2 0.3 0.4  A-B\n"
    "end bandstruc\nbandstruc_points 2\n"
)


def _settings(solver: str) -> dict:
    return {
        "Grid_Spacing": "0.57",
        "Expansion_Order": "8",
        "States_Num": "22",
        "Convergence_Criterion": "1.0e-9",
        "Diag_Tolerance": "1.0e-7",
        "Correlation_Type": "pbe",
        "Eigensolver": solver,
    }


def _bands(directory: Path, solver: str, convergence: str, sectors: str):
    import parsec_python.acceleration.cli as cli_module
    from parsec_python.acceleration.SCF import kpoint_sectors

    extra = PATH + ("FF_MaxIter: 1\nBands_FF_MaxIter: 1\n" if solver == "chebff" else "")
    path = write_silicon(directory, extra, positions=displaced_positions(0.0), settings=None)
    text = path.read_text(encoding="utf-8")
    lines = []
    wanted = _settings(solver)
    seen = set()
    for line in text.splitlines():
        keyword = line.split(":", 1)[0].strip()
        if ":" in line and keyword in wanted:
            lines.append(f" {keyword}: {wanted[keyword]}")
            seen.add(keyword)
        else:
            lines.append(line)
    lines.extend(f"{keyword}: {value}" for keyword, value in wanted.items() if keyword not in seen)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    errors = StringIO()
    variables = {
        "PARSEC_BAND_CONVERGENCE": convergence,
        "PARSEC_BAND_KPOINT_SECTORS": sectors,
        "PARSEC_PERIODIC_SECTOR_BANDS": "auto",
    }
    with mock.patch.dict(os.environ, variables), mock.patch.object(kpoint_sectors, "_MINIMUM_ENTRIES", 0), \
            redirect_stdout(StringIO()), redirect_stderr(errors):
        status = cli_module.main([str(path), "--quiet", "--backend", "cupy"])
    bands = {}
    if status == 0:
        with np.load(directory / "parsec_python_bands.npz", allow_pickle=True) as data:
            bands = {key: data[key] for key in data.files}
    output = directory / "parsec.out"
    log = output.read_text(encoding="utf-8", errors="replace") if output.exists() else ""
    return status, bands, errors.getvalue() + log


@unittest.skipUnless(_cupy_ready(), "CUDA runtime or native extension is unavailable")
class ConvergedBandTests(unittest.TestCase):
    def test_chebff_bands_reach_those_of_chebdav(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            status, reference, errors = _bands(root / "chebdav", "chebdav", "auto", "off")
            self.assertEqual(status, 0, errors[-4000:])
            for sectors in ("auto", "off"):
                with self.subTest(sectors=sectors):
                    status, converged, errors = _bands(root / f"auto_{sectors}", "chebff", "auto", sectors)
                    self.assertEqual(status, 0, errors[-4000:])
                    status, fixed, errors = _bands(root / f"off_{sectors}", "chebff", "off", sectors)
                    self.assertEqual(status, 0, errors[-4000:])
                    values = np.asarray(converged["eigenvalues_ry"])
                    expected = np.asarray(reference["eigenvalues_ry"])
                    worst = float(np.nanmax(converged["residual_norms"]))
                    loose = float(np.nanmax(fixed["residual_norms"]))
                    difference = float(np.abs(values - expected).max())
                    print(f"\n sectors {sectors}: residual auto {worst:.2e} off {loose:.2e} Ry, "
                          f"auto - chebdav {difference:.2e} Ry, off - chebdav "
                          f"{float(np.abs(np.asarray(fixed['eigenvalues_ry']) - expected).max()):.2e} Ry")
                    # (A100: residuals 3.1e-7 Ry in the sectors and 1.0e-7 whole after the extra cycles to the
                    # input's Diag_Tolerance 1e-7, against 1.2 Ry of one cycle, the bands within 1.2e-10 Ry of
                    # CHEBDAV's against 0.49 and 1.3 Ry; with CHEBDAV's default 1e-4 instead of Diag_Tolerance the
                    # residuals were 3.3e-4 and 1.0e-4 Ry)
                    self.assertLess(worst, 1.0e-5)
                    self.assertGreater(loose, 100.0 * worst)
                    np.testing.assert_allclose(values, expected, rtol=0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
