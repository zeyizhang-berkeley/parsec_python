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


def _bands(
    directory: Path, solver: str, convergence: str, sectors: str, reported: str = "auto", environment: dict = None
):
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
        "PARSEC_BAND_REPORTED_RESIDUALS": reported,
        "PARSEC_PERIODIC_SECTOR_BANDS": "auto",
        **(environment or {}),
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
                    # (A100: residuals of the reported levels 2.0e-9 Ry in the sectors and 1.0e-7 whole after the
                    # extra cycles to the input's Diag_Tolerance 1e-7, against 1.2 Ry of one cycle, the bands within
                    # 1.2e-10 Ry of
                    # CHEBDAV's against 0.49 and 1.3 Ry; with CHEBDAV's default 1e-4 instead of Diag_Tolerance the
                    # residuals were 3.3e-4 and 1.0e-4 Ry)
                    self.assertLess(worst, 1.0e-5)
                    self.assertGreater(loose, 100.0 * worst)
                    np.testing.assert_allclose(values, expected, rtol=0, atol=1.0e-6)

    def test_the_test_of_the_reported_levels_keeps_the_bands(self) -> None:
        # PARSEC_BAND_REPORTED_RESIDUALS: the margin states of the sectors are no longer converged, the reported levels
        # are those of the test of every state.  The cubic cell splits every k-point on its translations; the last two
        # k-points of the path (on its general line) are kept whole, as k-points without a little group are, so that
        # whole and split k-points share the chunks of the devices.
        from parsec_python.acceleration.SCF import kpoint_sectors

        original = kpoint_sectors.plan_sectors

        def plans(job, notes=None):
            found = original(job, notes)
            if found is not None and len(found) > 2:
                found[-2:] = [None, None]
            return found

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name, mock.patch.object(
            kpoint_sectors, "plan_sectors", plans
        ):
            root = Path(name)
            status, every, errors = _bands(root / "off", "chebff", "auto", "auto", "off")
            self.assertEqual(status, 0, errors[-4000:])
            status, reported, errors = _bands(root / "auto", "chebff", "auto", "auto", "auto")
            self.assertEqual(status, 0, errors[-4000:])
            difference = float(np.abs(np.asarray(reported["eigenvalues_ry"]) - np.asarray(every["eigenvalues_ry"])).max())
            worst = float(np.nanmax(reported["residual_norms"]))
            print(f"\n reported levels - every state's test {difference:.2e} Ry, residual {worst:.2e} Ry")
            self.assertLess(worst, 1.0e-5)
            np.testing.assert_allclose(reported["eigenvalues_ry"], every["eigenvalues_ry"], rtol=0, atol=1.0e-8)
            self.assertEqual(str(reported["residual_scope"]), "reported levels")
            self.assertNotIn("residual_scope", every)
            # the whole k-points: their first solve and test are those of before, whatever k-points share their chunk
            # (a test at 1 % of Diag_Tolerance would leave residuals far smaller)
            np.testing.assert_allclose(
                np.asarray(reported["residual_norms"])[:, -2:], np.asarray(every["residual_norms"])[:, -2:], rtol=1.0e-6
            )
            np.testing.assert_allclose(
                np.asarray(reported["eigenvalues_ry"])[:, -2:], np.asarray(every["eigenvalues_ry"])[:, -2:],
                rtol=0, atol=1.0e-12,
            )

    def test_the_band_buffer_on_the_whole_grid(self) -> None:
        # PARSEC_BAND_BUFFER: the whole-grid k-points take a buffer in proportion to their states; off keeps
        # Subspace_Buffer_Size.  With no extra CHEBFF cycles the one filter cycle cannot pass Diag_Tolerance 1e-7: the
        # WARNING names the residual.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            status, larger, errors = _bands(
                root / "auto", "chebff", "auto", "off", environment={"PARSEC_BAND_BUFFER": "auto"}
            )
            self.assertEqual(status, 0, errors[-4000:])
            self.assertNotIn("did not converge", errors)
            status, former, errors = _bands(
                root / "off", "chebff", "auto", "off", environment={"PARSEC_BAND_BUFFER": "off"}
            )
            self.assertEqual(status, 0, errors[-4000:])
            scale = float(np.max(np.abs(larger["eigenvalues_ry"])))
            self.assertLess(float(np.max(larger["residual_norms"])), 1.0e-7 * scale)
            np.testing.assert_allclose(larger["eigenvalues_ry"], former["eigenvalues_ry"], rtol=0, atol=1.0e-6)
            status, capped, errors = _bands(
                root / "capped", "chebff", "auto", "off", environment={"PARSEC_KPOINT_EXTRA_CYCLES": "0"}
            )
            self.assertEqual(status, 0, errors[-4000:])
            self.assertIn("did not converge: largest residual", errors)

    def test_unsplit_k_points_take_the_whole_route_with_the_buffer(self) -> None:
        # k-points that do not split (here the last two of the path, kept whole by hand) are solved on the whole grid
        # with the buffer of PARSEC_BAND_BUFFER; under off they stay in the solver of the sectors, as before
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter
        from parsec_python.acceleration.SCF import kpoint_sectors

        original = kpoint_sectors.plan_sectors
        real_eigensolve = DeviceBlochConverter.eigensolve

        def plans(job, notes=None):
            found = original(job, notes)
            if found is not None and len(found) > 2:
                found[-2:] = [None, None]
            return found

        for value in ("auto", "off"):
            buffers: list[int] = []

            def recording(converter, hamiltonian, number_of_states, settings, attempt=0):
                buffers.append(int(settings.safety_buffer))
                return real_eigensolve(converter, hamiltonian, number_of_states, settings, attempt)

            with self.subTest(buffer=value), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name, \
                    mock.patch.object(kpoint_sectors, "plan_sectors", plans), \
                    mock.patch.object(DeviceBlochConverter, "eigensolve", recording):
                status, bands, errors = _bands(
                    Path(name), "chebff", "auto", "auto", environment={"PARSEC_BAND_BUFFER": value}
                )
                self.assertEqual(status, 0, errors[-4000:])
                self.assertNotIn("did not converge", errors)
                if value == "auto":
                    self.assertEqual(len(buffers), 2)
                    self.assertEqual(len(set(buffers)), 1)
                    self.assertGreater(buffers[0], 6)
                else:
                    self.assertEqual(buffers, [])

    def test_a_short_device_keeps_the_subspaces_on_the_host_or_the_former_test(self) -> None:
        from parsec_python.acceleration.SCF import kpoint_parallel

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as name:
            root = Path(name)
            status, reference, errors = _bands(root / "plain", "chebff", "auto", "auto", "auto")
            self.assertEqual(status, 0, errors[-4000:])
            # the device is short and the host has room: the parts run with the subspaces on the host
            with mock.patch.object(kpoint_parallel, "_AUTO_FRACTION", 1.0e-12):
                status, spilled, errors = _bands(root / "spilled", "chebff", "auto", "auto", "auto")
            self.assertEqual(status, 0, errors[-4000:])
            self.assertEqual(str(spilled["residual_scope"]), "reported levels")
            np.testing.assert_allclose(spilled["eigenvalues_ry"], reference["eigenvalues_ry"], rtol=0, atol=1.0e-12)
            # neither has room: the former test of every state
            with mock.patch.object(kpoint_parallel, "_AUTO_FRACTION", 1.0e-12), mock.patch.object(
                kpoint_parallel, "_available_host_bytes", return_value=1
            ):
                status, former, errors = _bands(root / "former", "chebff", "auto", "auto", "auto")
            self.assertEqual(status, 0, errors[-4000:])
            self.assertNotIn("residual_scope", former)
            np.testing.assert_allclose(former["eigenvalues_ry"], reference["eigenvalues_ry"], rtol=0, atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
