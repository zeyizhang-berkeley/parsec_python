"""Integration coverage for accelerated preparation, SCF, and CLI wiring."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

import parsec_python.acceleration.cli as cli_module
import parsec_python.acceleration.backends.selection as selection_module
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.backends.cupy import CuPyTimingStats
from parsec_python.acceleration.backends.cupy_runtime import CuPyHamiltonianBackend
from parsec_python.acceleration.backends.selection import (
    BackendSelection,
    resolve_backend,
)
from parsec_python.acceleration.driver import (
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
)
from parsec_python.acceleration.models import (
    BackendStatistics,
    BackendUnavailableError,
)
from parsec_python.Input import parse_parsec_input
from parsec_python.models import Atom


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT.parent
DATA = PACKAGE_ROOT / "tests" / "data"
SMOKE_INPUT = DATA / "H_cli_smoke.in"
NATIVE_AVAILABLE = native_available()


class DriverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = parse_parsec_input(SMOKE_INPUT).problem

    def test_explicit_scipy_prepare_and_component_profile(self) -> None:
        system = prepare_single_point(self.problem, backend="scipy")

        self.assertEqual(system.backend_info.requested, "scipy")
        self.assertEqual(system.backend_info.selected, "scipy")
        self.assertEqual(system.backend_info.fallback_reasons, ())
        self.assertIs(system.backend_info, system.backend.info)
        self.assertEqual(system.backend.shape, (system.grid.size, system.grid.size))
        self.assertGreater(system.negative_laplacian.nnz, system.grid.size)

        preparation = system.timings
        self.assertGreaterEqual(preparation.finite_difference_seconds, 0.0)
        self.assertGreaterEqual(preparation.local_ionic_seconds, 0.0)
        self.assertGreaterEqual(preparation.nonlocal_ionic_seconds, 0.0)
        self.assertGreaterEqual(preparation.total_seconds, 0.0)

        original_local = system.backend.local_potential.copy()
        updates_before = system.backend.statistics.local_updates
        profile = profile_hamiltonian_components(
            system,
            block_size=2,
            repeats=2,
            random_seed=81,
        )
        self.assertEqual(
            set(profile),
            {
                "finite_difference_apply",
                "local_potential_apply",
                "nonlocal_potential_apply",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in profile.values()))
        self.assertEqual(
            system.backend.statistics.component_profile_seconds,
            profile,
        )
        self.assertGreaterEqual(system.backend.statistics.warmup_seconds, 0.0)
        # Profiling temporarily binds V_ion,local and then restores the field
        # that was active before the diagnostic.
        np.testing.assert_array_equal(system.backend.local_potential, original_local)
        self.assertEqual(
            system.backend.statistics.local_updates - updates_before,
            2,
        )

    def test_symmetry_auto_detects_and_off_skips_reduction(self) -> None:
        automatic = prepare_single_point(
            self.problem, backend="scipy", symmetry="auto"
        )
        automatic_details = dict(automatic.backend_info.details)
        self.assertGreater(
            int(automatic_details["detected_symmetry_group_order"]), 1
        )
        self.assertIn("detected", automatic_details["symmetry_detection"])

        disabled = prepare_single_point(
            self.problem, backend="scipy", symmetry="off"
        )
        disabled_details = dict(disabled.backend_info.details)
        self.assertEqual(disabled_details["symmetry_mode"], "off")
        self.assertEqual(disabled_details["symmetry_detection"], "disabled by option")
        self.assertIn("disabled", disabled_details["orbital_symmetry"])
        with self.assertRaisesRegex(ValueError, "selected backend"):
            prepare_single_point(
                self.problem, backend="scipy", symmetry="on"
            )

    def test_asymmetric_system_falls_back_in_auto_and_errors_when_forced(self) -> None:
        asymmetric = replace(
            self.problem,
            atoms=(Atom("H", (0.13, 0.27, 0.41)),),
            recenter_geometry=False,
        )
        automatic = prepare_single_point(
            asymmetric, backend="scipy", symmetry="auto"
        )
        details = dict(automatic.backend_info.details)
        self.assertEqual(details["detected_symmetry_group_order"], "1")
        self.assertIn("identity only", details["symmetry_detection"])
        with self.assertRaisesRegex(ValueError, "only the identity"):
            prepare_single_point(
                asymmetric, backend="scipy", symmetry="on"
            )

    def test_one_iteration_result_carries_backend_statistics_and_timings(self) -> None:
        system = prepare_single_point(self.problem, backend="scipy")
        result = run_scf(system)

        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.converged)
        self.assertEqual(result.backend.selected, "scipy")
        statistics = result.backend_statistics
        self.assertGreater(statistics.applications, 0)
        self.assertGreaterEqual(statistics.vectors_applied, statistics.applications)
        self.assertGreater(statistics.apply_seconds, 0.0)
        self.assertGreater(statistics.local_updates, 0)
        self.assertGreaterEqual(statistics.local_update_seconds, 0.0)

        timings = result.timings
        self.assertIs(timings.preparation, system.timings)
        self.assertGreater(timings.total_seconds, 0.0)
        self.assertGreater(timings.diagonalization_seconds, 0.0)
        self.assertGreaterEqual(timings.hamiltonian_binding_seconds, 0.0)
        self.assertGreaterEqual(timings.initial_xc_seconds, 0.0)
        self.assertEqual(len(result.history), 1)
        self.assertGreater(result.history[0].diagonalization_seconds, 0.0)

        # AcceleratedSinglePointResult receives a snapshot rather than a live
        # alias that can be changed by later diagnostic applications.
        saved_applications = statistics.applications
        system.backend.apply(np.ones(system.grid.size))
        self.assertEqual(result.backend_statistics.applications, saved_applications)
        self.assertEqual(system.backend.statistics.applications, saved_applications + 1)

    @unittest.skipUnless(
        NATIVE_AVAILABLE,
        "parsec_accelerated_native has not been built",
    )
    def test_native_and_scipy_one_iteration_physical_parity(self) -> None:
        scipy_result = run_scf(
            prepare_single_point(self.problem, backend="scipy")
        )
        # Compare identical full-grid algorithms here.  The default native
        # path separately enables PARSEC-style symmetry-wedge Hartree, whose
        # reduced residual norm changes the last few solver digits.
        with patch.dict(os.environ, {"PARSEC_HARTREE_SYMMETRY": "0"}):
            native_result = run_scf(
                prepare_single_point(self.problem, backend="native")
            )

        self.assertEqual(native_result.backend.selected, "native")
        self.assertEqual(native_result.iterations, scipy_result.iterations)
        self.assertEqual(native_result.converged, scipy_result.converged)
        np.testing.assert_allclose(
            native_result.eigenvalues,
            scipy_result.eigenvalues,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            native_result.occupations,
            scipy_result.occupations,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            native_result.density,
            scipy_result.density,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        self.assertAlmostEqual(
            native_result.energies.total,
            scipy_result.energies.total,
            places=12,
        )
        self.assertAlmostEqual(
            native_result.history[0].weighted_residual,
            scipy_result.history[0].weighted_residual,
            places=12,
        )
        self.assertGreater(native_result.backend_statistics.applications, 0)


class BackendSelectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = parse_parsec_input(SMOKE_INPUT).problem

    def test_explicit_native_unavailable_is_not_replaced(self) -> None:
        with patch.object(
            selection_module,
            "_native_status",
            return_value=(False, "extension deliberately unavailable"),
        ):
            with self.assertRaisesRegex(
                BackendUnavailableError,
                "native backend was requested.*deliberately unavailable",
            ):
                resolve_backend("native", self.problem)

    def test_explicit_cupy_unavailable_is_not_replaced(self) -> None:
        with patch.object(
            selection_module,
            "_cupy_status",
            return_value=(False, "CUDA deliberately unavailable"),
        ):
            with self.assertRaisesRegex(
                BackendUnavailableError,
                "CuPy backend was requested.*deliberately unavailable",
            ):
                resolve_backend("cupy", self.problem)

    def test_explicit_backends_keep_controlled_component_paths(self) -> None:
        with (
            patch.object(selection_module, "_cupy_status") as cupy_status,
            patch.object(selection_module, "_native_status") as native_status,
        ):
            scipy = resolve_backend("scipy", self.problem)

        cupy_status.assert_not_called()
        native_status.assert_not_called()
        self.assertEqual(scipy.selected, "scipy")
        self.assertEqual(scipy.finite_difference_builder, "reference")
        self.assertEqual(scipy.hartree_backend, "scipy")

        with patch.object(
            selection_module,
            "_cupy_status",
            return_value=(True, None),
        ):
            cupy = resolve_backend("cupy", self.problem)
        self.assertEqual(cupy.selected, "cupy")
        self.assertEqual(cupy.finite_difference_builder, "reference")
        self.assertEqual(cupy.hartree_backend, "cupy")

        with patch.object(
            selection_module,
            "_native_status",
            return_value=(True, None),
        ):
            native = resolve_backend("native", self.problem)
        self.assertEqual(native.selected, "native")
        self.assertEqual(native.finite_difference_builder, "native")
        self.assertEqual(native.hartree_backend, "native")

    def test_auto_composes_cupy_execution_with_native_components(self) -> None:
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(True, None),
            ) as cupy_status,
            patch.object(
                selection_module,
                "_native_status",
                return_value=(True, None),
            ) as native_status,
        ):
            selected = resolve_backend("auto", self.problem)

        cupy_status.assert_called_once_with()
        native_status.assert_called_once_with()
        self.assertEqual(selected.selected, "cupy")
        self.assertEqual(selected.finite_difference_builder, "native")
        self.assertEqual(selected.hartree_backend, "native")
        self.assertEqual(selected.fallback_reasons, ())

    def test_auto_uses_pure_cupy_when_native_components_are_unavailable(self) -> None:
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(True, None),
            ),
            patch.object(
                selection_module,
                "_native_status",
                return_value=(False, "no test extension"),
            ),
        ):
            selected = resolve_backend("auto", self.problem)

        self.assertEqual(selected.selected, "cupy")
        self.assertEqual(selected.finite_difference_builder, "reference")
        self.assertEqual(selected.hartree_backend, "cupy")
        self.assertEqual(
            selected.fallback_reasons,
            ("native components skipped: no test extension",),
        )

    def test_auto_uses_native_when_cupy_is_unavailable(self) -> None:
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(False, "no test CUDA"),
            ),
            patch.object(
                selection_module,
                "_native_status",
                return_value=(True, None),
            ),
        ):
            selected = resolve_backend("auto", self.problem)

        self.assertEqual(selected.selected, "native")
        self.assertEqual(selected.finite_difference_builder, "native")
        self.assertEqual(selected.hartree_backend, "native")
        self.assertEqual(
            selected.fallback_reasons,
            ("CuPy skipped: no test CUDA",),
        )

    def test_auto_falls_back_to_scipy_when_both_runtimes_are_unavailable(
        self,
    ) -> None:
        with (
            patch.object(
                selection_module,
                "_cupy_status",
                return_value=(False, "no test CUDA"),
            ),
            patch.object(
                selection_module,
                "_native_status",
                return_value=(False, "no test extension"),
            ),
        ):
            selected = resolve_backend("auto", self.problem)

        self.assertEqual(selected.requested, "auto")
        self.assertEqual(selected.selected, "scipy")
        self.assertEqual(selected.finite_difference_builder, "reference")
        self.assertEqual(selected.hartree_backend, "scipy")
        self.assertEqual(len(selected.fallback_reasons), 2)
        self.assertIn("CuPy skipped: no test CUDA", selected.fallback_reasons[0])
        self.assertIn("native skipped: no test extension", selected.fallback_reasons[1])


class CuPyRuntimeStatisticsTests(unittest.TestCase):
    def test_synchronized_device_totals_map_to_common_statistics(self) -> None:
        # Construct only the host-side adapter state.  This exercises the
        # production reporting bridge without importing CuPy or requiring a
        # CUDA device in the test environment.
        backend = object.__new__(CuPyHamiltonianBackend)
        backend.statistics = BackendStatistics()
        backend.timing_stats = CuPyTimingStats(
            initialization_seconds=0.5,
            first_solve_seconds=1.25,
            subspace_solve_seconds=2.75,
            download_seconds=0.125,
            first_solve_calls=1,
            subspace_solve_calls=3,
            hamiltonian_applications=47,
            orbital_vectors_applied=286,
        )

        backend.synchronize_statistics()

        stats = backend.statistics
        self.assertEqual(stats.applications, 47)
        self.assertEqual(stats.vectors_applied, 286)
        self.assertEqual(stats.eigensolver_first_calls, 1)
        self.assertAlmostEqual(stats.eigensolver_first_seconds, 1.25)
        self.assertEqual(stats.eigensolver_subspace_calls, 3)
        self.assertAlmostEqual(stats.eigensolver_subspace_seconds, 2.75)
        self.assertAlmostEqual(stats.eigensolver_download_seconds, 0.125)
        self.assertAlmostEqual(stats.device_seconds, 4.0)
        self.assertAlmostEqual(stats.device_to_host_seconds, 0.125)


class AcceleratedCommandLineTests(unittest.TestCase):
    def test_canonical_package_module_dry_run(self) -> None:
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(SOURCE_ROOT)
            if not existing_pythonpath
            else str(SOURCE_ROOT) + os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "parsec_python",
                str(SMOKE_INPUT),
                "--backend",
                "scipy",
                "--dry-run",
            ],
            cwd=SOURCE_ROOT.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "Backend: requested=scipy, selected=scipy",
            completed.stdout,
        )
        self.assertIn("Dry run successful", completed.stdout)
        self.assertNotIn("Traceback", completed.stderr)

    def test_default_dry_run_reports_selection_and_fallback(self) -> None:
        selected = BackendSelection(
            requested="auto",
            selected="scipy",
            finite_difference_builder="reference",
            hartree_backend="scipy",
            fallback_reasons=("test accelerator unavailable",),
        )
        output = StringIO()
        error = StringIO()
        with (
            patch.object(cli_module, "resolve_backend", return_value=selected),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            return_code = cli_module.main([str(SMOKE_INPUT), "--dry-run"])

        self.assertEqual(return_code, 0, error.getvalue())
        self.assertEqual(error.getvalue(), "")
        report = output.getvalue()
        self.assertIn("Backend: requested=auto, selected=scipy", report)
        self.assertIn(
            "BACKEND FALLBACK: test accelerator unavailable",
            report,
        )
        self.assertIn("Symmetry: mode=auto", report)
        self.assertIn("Dry run successful", report)

    def test_ignore_symmetry_input_default_and_explicit_override(self) -> None:
        translation = replace(
            parse_parsec_input(SMOKE_INPUT), ignore_symmetry=True
        )
        for extra, expected in (((), "off"), (("--symmetry", "auto"), "auto")):
            with self.subTest(extra=extra):
                output = StringIO()
                error = StringIO()
                with (
                    patch.object(
                        cli_module,
                        "parse_parsec_input",
                        return_value=translation,
                    ),
                    redirect_stdout(output),
                    redirect_stderr(error),
                ):
                    return_code = cli_module.main(
                        [
                            str(SMOKE_INPUT),
                            "--backend",
                            "scipy",
                            "--dry-run",
                            *extra,
                        ]
                    )
                self.assertEqual(return_code, 0, error.getvalue())
                self.assertIn(
                    f"Symmetry: mode={expected}", output.getvalue()
                )

    def test_explicit_unavailable_dry_run_is_validation_error(self) -> None:
        output = StringIO()
        error = StringIO()
        with (
            patch.object(
                cli_module,
                "resolve_backend",
                side_effect=BackendUnavailableError("native intentionally missing"),
            ),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            return_code = cli_module.main(
                [str(SMOKE_INPUT), "--backend", "native", "--dry-run"]
            )

        self.assertEqual(return_code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn(
            "Validation error: native intentionally missing",
            error.getvalue(),
        )
        self.assertNotIn("selected=scipy", error.getvalue())

    def test_default_output_names_and_backend_reporting(self) -> None:
        messages: list[str] = []
        log_paths: list[Path] = []
        archive_paths: list[Path] = []

        class MemoryLog:
            def __init__(self, path: Path, quiet: bool = False) -> None:
                del quiet
                log_paths.append(path)

            def __enter__(self) -> "MemoryLog":
                return self

            def __exit__(self, _type, _value, _traceback) -> None:
                return None

            def write(self, message: str = "") -> None:
                messages.append(message)

        def fake_archive(path, _result, *, include_wavefunctions=False):
            del include_wavefunctions
            archive_paths.append(Path(path))
            return Path(path)

        with (
            patch.object(cli_module, "_RunLog", MemoryLog),
            patch.object(cli_module, "save_result_archive", side_effect=fake_archive),
        ):
            return_code = cli_module.main(
                [str(SMOKE_INPUT), "--backend", "scipy", "--quiet"]
            )

        # Max_Iter=1 intentionally stops before self-consistency.
        self.assertEqual(return_code, 3)
        self.assertEqual(
            log_paths,
            [SMOKE_INPUT.parent / "parsec.out"],
        )
        self.assertEqual(
            archive_paths,
            [SMOKE_INPUT.parent / "parsec_python_results.npz"],
        )
        report = "\n".join(messages)
        self.assertIn("Requested backend = scipy", report)
        self.assertIn("Selected backend  = scipy", report)
        self.assertIn("Accelerated Hamiltonian statistics", report)
        self.assertIn("H applications", report)
        self.assertIn("Text log:", report)


if __name__ == "__main__":
    unittest.main()
