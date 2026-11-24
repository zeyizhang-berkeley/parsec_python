"""Converged CHEBFF first solves of the band and nscf k-points (``PARSEC_BAND_CONVERGENCE``, ``SCF/bands.py``).

CPU: the switch, the settings it gives the band k-points, and the CHEBFF first solve of
:func:`parsec_python.Eigensolvers.solve_eigval` with :class:`ConvergedFirstSolveSettings`: more cycles to
CHEBDAV's relative residual test, the levels of the fixed cycles bit for bit where those already pass it.
The GPU routes (device first solve, band sectors) are in ``acceleration/tests/test_band_convergence.py``.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers import EigvalSettings, solve_eigval
from parsec_python.Eigensolvers.chebdav import ChebDavSettings
from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.eigval import (
    ConvergedFirstSolveSettings,
    first_solve_converges,
    first_solve_extra_cycles,
)
from parsec_python.SCF.bands import BAND_CONVERGENCE_SWITCH, band_convergence_setting, converged_band_settings


def _laplacian(points: int = 14) -> sp.csr_matrix:
    """A 3D finite-difference Laplacian with a weak random diagonal: clustered levels that 1 cycle leaves
    unconverged."""

    one = sp.diags([-np.ones(points - 1), 2.0 * np.ones(points), -np.ones(points - 1)], [-1, 0, 1])
    identity = sp.identity(points)
    operator = (
        sp.kron(sp.kron(one, identity), identity)
        + sp.kron(sp.kron(identity, one), identity)
        + sp.kron(sp.kron(identity, identity), one)
    )
    rng = np.random.default_rng(11)
    return (operator + sp.diags(0.05 * rng.random(points**3))).tocsr()


def _settings(cycles: int, tolerance: float = 1.0e-6) -> EigvalSettings:
    return EigvalSettings(
        safety_buffer=4,
        initial_method="chebff",
        chebff=ChebFFSettings(polynomial_degree=12, filter_cycles=cycles),
        chebdav=ChebDavSettings(convergence_tolerance=tolerance),
    )


def _largest_relative_residual(operator, result) -> float:
    values = np.asarray(result.eigenvalues)
    residual = operator @ result.vectors - result.vectors * values[None, :]
    return float(np.max(np.linalg.norm(residual, axis=0)) / max(float(np.max(np.abs(values))), 1.0e-8))


class SwitchTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: value}):
                self.assertEqual(band_convergence_setting(), expected)
        with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: "on"}):
            with self.assertRaisesRegex(ValueError, BAND_CONVERGENCE_SWITCH):
                band_convergence_setting()

    def test_reported_residuals_values(self) -> None:
        from parsec_python.SCF.bands import (
            BAND_REPORTED_RESIDUALS_SWITCH,
            band_reported_residuals_setting,
            validate_band_settings,
        )

        for value, expected in (("", "auto"), ("auto", "auto"), ("OFF", "off")):
            with mock.patch.dict(os.environ, {BAND_REPORTED_RESIDUALS_SWITCH: value}):
                self.assertEqual(band_reported_residuals_setting(), expected)
        with mock.patch.dict(os.environ, {BAND_REPORTED_RESIDUALS_SWITCH: "on"}):
            with self.assertRaisesRegex(ValueError, BAND_REPORTED_RESIDUALS_SWITCH):
                band_reported_residuals_setting()
            with self.assertRaisesRegex(ValueError, BAND_REPORTED_RESIDUALS_SWITCH):
                validate_band_settings()

    def test_the_scope_of_the_residual_is_named(self) -> None:
        import tempfile
        from pathlib import Path

        from parsec_python.Output.bands import load_band_structure_archive, save_band_structure_archive
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.models import BandStructureResult

        def result(scope):
            return BandStructureResult(
                mode="scalar", channels=("all",), segment_names=("G-X",), segment_counts=np.asarray([2]),
                k_points=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), eigenvalues=np.zeros((1, 2, 3)),
                residual_norms=np.full((1, 2), 1.0e-6), fermi_level=0.0, residual_scope=scope,
            )

        for scope, suffix in ((None, "[Ry]"), ("reported levels", "[Ry] (reported levels)")):
            written = []
            reporter = ParsecTextReporter.__new__(ParsecTextReporter)
            reporter.write = written.append
            reporter.band_structure(result(scope))
            line = next(row for row in written[0].splitlines() if "Largest eigenpair residual" in row)
            self.assertTrue(line.endswith(suffix), line)
            with tempfile.TemporaryDirectory() as folder:
                path = save_band_structure_archive(Path(folder) / "bands.npz", result(scope))
                with np.load(path) as archive:
                    self.assertEqual("residual_scope" in archive.files, scope is not None)
                self.assertEqual(load_band_structure_archive(path).residual_scope, scope)

    def test_the_band_buffer(self) -> None:
        from parsec_python.SCF.bands import (
            BAND_BUFFER_FRACTION,
            BAND_BUFFER_SWITCH,
            affordable_band_buffer,
            band_buffer_setting,
            band_working_buffer,
            validate_band_settings,
        )

        converged = converged_band_settings(_settings(3), True)
        fixed = _settings(3)
        self.assertEqual(int(converged.safety_buffer), int(fixed.safety_buffer))
        base = int(converged.safety_buffer)
        with mock.patch.dict(os.environ, {BAND_BUFFER_SWITCH: ""}):
            self.assertEqual(band_buffer_setting(), BAND_BUFFER_FRACTION)
            # in proportion to the states, never below Subspace_Buffer_Size; fixed cycles keep the buffer of PARSEC 1.5
            self.assertEqual(band_working_buffer(converged, 160), max(base, int(np.ceil(BAND_BUFFER_FRACTION * 160))))
            self.assertEqual(band_working_buffer(converged, 2), base)
            self.assertEqual(band_working_buffer(fixed, 160), base)
            self.assertEqual(band_working_buffer(converged, 160, host_chebdav=True), base)
        with mock.patch.dict(os.environ, {BAND_BUFFER_SWITCH: "off"}):
            self.assertIsNone(band_buffer_setting())
            self.assertEqual(band_working_buffer(converged, 160), base)
        with mock.patch.dict(os.environ, {BAND_BUFFER_SWITCH: "0.25"}):
            self.assertEqual(band_working_buffer(converged, 160), max(base, 40))
        for value in ("on", "0", "-1", "3", "nan"):
            with mock.patch.dict(os.environ, {BAND_BUFFER_SWITCH: value}):
                with self.assertRaisesRegex(ValueError, BAND_BUFFER_SWITCH):
                    band_buffer_setting()
                with self.assertRaisesRegex(ValueError, BAND_BUFFER_SWITCH):
                    validate_band_settings()
        # within half the memory for three complex bases, never below the buffer of before
        self.assertEqual(affordable_band_buffer(96, 6, 160, 110592, 80.0e9), 96)
        self.assertEqual(affordable_band_buffer(96, 6, 160, 110592, None), 96)
        rows = 729000
        fits = int(0.5 * 80.0e9 // (3 * 16 * rows)) - 2006
        self.assertEqual(affordable_band_buffer(1204, 6, 2006, rows, 80.0e9), max(6, fits))
        self.assertEqual(affordable_band_buffer(1204, 6, 2006, rows, 1.0e9), 6)
        self.assertEqual(affordable_band_buffer(6, 6, 2006, rows, 1.0), 6)

    def test_unconverged_band_levels_are_named_in_a_warning(self) -> None:
        from parsec_python.Output.parsec_output import ParsecTextReporter
        from parsec_python.SCF.bands import unconverged_band_points
        from parsec_python.models import BandStructureResult

        levels = np.asarray([[[-0.5, 0.2, 2.4], [-0.4, 0.3, 2.0]]])
        # the test of a converged first solve: residual <= relative * max|e| of the k-point
        residuals = np.asarray([[2.0e-10, 9.1e-3]])
        np.testing.assert_array_equal(unconverged_band_points(residuals, levels, 1.0e-10), [[False, True]])
        np.testing.assert_array_equal(unconverged_band_points(residuals, levels, None), [[False, False]])
        np.testing.assert_array_equal(unconverged_band_points([[np.nan, 1.0e-12]], levels, 1.0e-10), [[False, False]])
        # a k-point whose solve did not apply the test (the host CHEBDAV) is not counted
        np.testing.assert_array_equal(
            unconverged_band_points(residuals, levels, 1.0e-10, [[True, False]]), [[False, False]]
        )

        def report(norms, test):
            written = []
            reporter = ParsecTextReporter.__new__(ParsecTextReporter)
            reporter.write = written.append
            reporter.problem = mock.Mock()
            reporter.problem.eigensolver.method = "chebff"
            reporter.band_structure(
                BandStructureResult(
                    mode="scalar", channels=("all",), segment_names=("G-X",), segment_counts=np.asarray([2]),
                    k_points=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), eigenvalues=levels,
                    residual_norms=np.asarray(norms), fermi_level=0.0, residual_test=test,
                )
            )
            return [row for row in written[0].splitlines() if "WARNING" in row]

        # below the fixed 1e-2 Ry of before, and unconverged: named
        warning = report(residuals, 1.0e-10)
        self.assertEqual(len(warning), 1)
        self.assertIn("1 of 2 k-point(s) did not converge", warning[0])
        self.assertIn("9.100E-03 Ry", warning[0])
        # converged: no warning
        self.assertEqual(report([[2.0e-10, 1.0e-10]], 1.0e-10), [])
        # fixed cycles (no test): the warning of before, above 1e-2 Ry only
        self.assertEqual(report(residuals, None), [])
        self.assertEqual(len(report([[2.0e-10, 0.1]], None)), 1)

    def test_settings_of_the_band_k_points(self) -> None:
        chebff = _settings(3)
        with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: "auto"}):
            converged = converged_band_settings(chebff)
            self.assertIsInstance(converged, ConvergedFirstSolveSettings)
            self.assertTrue(first_solve_converges(converged))
            for item in dataclasses.fields(EigvalSettings):
                self.assertEqual(getattr(converged, item.name), getattr(chebff, item.name), item.name)
            # replace keeps the type (the CHEBDAV attempts of SCF/bands.py)
            self.assertIsInstance(dataclasses.replace(converged, safety_buffer=2), ConvergedFirstSolveSettings)
            chebdav = dataclasses.replace(chebff, initial_method="chebdav")
            self.assertIs(converged_band_settings(chebdav), chebdav)
            self.assertIs(converged_band_settings(chebff, False), chebff)
        with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: "off"}):
            self.assertIs(converged_band_settings(chebff), chebff)
            self.assertIsInstance(converged_band_settings(chebff, True), ConvergedFirstSolveSettings)
        self.assertFalse(first_solve_converges(chebff))
        self.assertTrue(first_solve_converges(dataclasses.replace(chebff, initial_method="chebdav")))
        # the text of the settings of an SCF is that of before
        self.assertNotIn("converge_first_solve", repr(chebff))

    def test_the_tolerance_of_the_input(self) -> None:
        # a CHEBFF SCF carries CHEBDAV's default tolerance; the band k-points take Diag_Tolerance
        chebff = dataclasses.replace(_settings(3), chebdav=ChebDavSettings())
        self.assertEqual(chebff.chebdav.convergence_tolerance, 1.0e-4)
        tight = converged_band_settings(chebff, True, tolerance=1.0e-7)
        self.assertEqual(tight.chebdav.convergence_tolerance, 1.0e-7)
        for item in dataclasses.fields(EigvalSettings):
            if item.name != "chebdav":
                self.assertEqual(getattr(tight, item.name), getattr(chebff, item.name), item.name)
        self.assertEqual(dataclasses.replace(tight.chebdav, convergence_tolerance=1.0e-4), chebff.chebdav)
        # the default 1e-4 (the input reader's value for CHEBFF without a positive Diag_Tolerance) changes nothing
        self.assertEqual(converged_band_settings(chebff, True, tolerance=1.0e-4).chebdav, chebff.chebdav)
        self.assertEqual(converged_band_settings(chebff, True).chebdav, chebff.chebdav)
        self.assertEqual(converged_band_settings(chebff, True, tolerance=-1.0).chebdav, chebff.chebdav)
        # off, or CHEBDAV, keeps the settings
        self.assertIs(converged_band_settings(chebff, False, tolerance=1.0e-7), chebff)

    def test_extra_cycles(self) -> None:
        from parsec_python.SCF.bands import validate_band_settings

        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": ""}):
            self.assertEqual(first_solve_extra_cycles(), 24)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": "7"}):
            self.assertEqual(first_solve_extra_cycles(), 7)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": "0"}):
            self.assertEqual(first_solve_extra_cycles(), 0)
        for value in ("many", "-3", "2.5"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": value}):
                with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_EXTRA_CYCLES"):
                    first_solve_extra_cycles()
                with self.assertRaisesRegex(ValueError, "PARSEC_KPOINT_EXTRA_CYCLES"):
                    validate_band_settings()
        with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: "on"}):
            with self.assertRaisesRegex(ValueError, BAND_CONVERGENCE_SWITCH):
                validate_band_settings()

    def test_the_input_reader_refuses_unknown_values(self) -> None:
        # a bandstruc block or nscf_kpoints: both switches are checked before the SCF
        import tempfile
        from pathlib import Path

        from parsec_python.Input.parsec_input import ParsecInputError, parse_parsec_input
        from parsec_python.tests.unit.test_kpoint_supercell_route import _silicon_input

        bands = "begin bandstruc\n1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\nend bandstruc\n"
        with tempfile.TemporaryDirectory() as folder:
            path = _silicon_input(Path(folder), spacing=1.2825, order=4)
            text = path.read_text(encoding="utf-8")
            path.write_text(text + "\n" + bands, encoding="utf-8")
            parse_parsec_input(path)
            for name, value in (("PARSEC_KPOINT_EXTRA_CYCLES", "x"), (BAND_CONVERGENCE_SWITCH, "yes")):
                with mock.patch.dict(os.environ, {name: value}):
                    with self.assertRaisesRegex(ParsecInputError, name):
                        parse_parsec_input(path)
            # without a band or nscf stage PARSEC_BAND_CONVERGENCE is not read; PARSEC_KPOINT_EXTRA_CYCLES is read for
            # every input since 2026-11-23 (the converged first solves of an SCF's k-points read it too)
            path.write_text(text, encoding="utf-8")
            with mock.patch.dict(os.environ, {BAND_CONVERGENCE_SWITCH: "yes"}):
                parse_parsec_input(path)
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": "x"}):
                with self.assertRaisesRegex(ParsecInputError, "PARSEC_KPOINT_EXTRA_CYCLES"):
                    parse_parsec_input(path)


class FirstSolveTests(unittest.TestCase):
    def test_more_cycles_reach_the_residual_test(self) -> None:
        operator = _laplacian()
        # ten states: the level above them stands apart (the eleventh, then a cluster of six that the buffer cuts)
        fixed = solve_eigval(operator, 10, settings=_settings(1))
        converged = solve_eigval(operator, 10, settings=converged_band_settings(_settings(1), True))
        self.assertGreater(_largest_relative_residual(operator, fixed), 1.0e-4)
        self.assertLessEqual(_largest_relative_residual(operator, converged), 1.0e-6)
        self.assertGreater(len(converged.chebff_result.cycles), 1)
        reference = np.linalg.eigvalsh(operator.toarray())[:10]
        np.testing.assert_allclose(converged.eigenvalues, reference, rtol=0, atol=1.0e-9)

    def test_a_solve_that_passes_after_the_fixed_cycles_keeps_its_levels(self) -> None:
        operator = _laplacian()
        loose = _settings(6, tolerance=1.0)
        fixed = solve_eigval(operator, 12, settings=loose)
        converged = solve_eigval(operator, 12, settings=converged_band_settings(loose, True))
        self.assertEqual(len(converged.chebff_result.cycles), 6)
        np.testing.assert_array_equal(converged.eigenvalues, fixed.eigenvalues)
        np.testing.assert_array_equal(converged.vectors, fixed.vectors)

    def test_the_cycles_stop_at_the_extra_cycles(self) -> None:
        operator = _laplacian()
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_EXTRA_CYCLES": "2"}):
            converged = solve_eigval(
                operator, 12, settings=converged_band_settings(_settings(1, tolerance=1.0e-14), True)
            )
        self.assertEqual(len(converged.chebff_result.cycles), 3)

    def test_the_scf_settings_keep_the_fixed_cycles(self) -> None:
        operator = _laplacian()
        result = solve_eigval(operator, 12, settings=_settings(2))
        self.assertEqual(len(result.chebff_result.cycles), 2)


if __name__ == "__main__":
    unittest.main()
