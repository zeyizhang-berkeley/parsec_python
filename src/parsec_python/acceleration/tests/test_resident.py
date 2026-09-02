"""Standard-library tests for resident request isolation and capture."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from parsec_python.acceleration.resident import _worker_environment, execute_request


class ResidentRequestTests(unittest.TestCase):
    def test_worker_defaults_tiny_host_lapack_to_one_thread(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            environment = _worker_environment()
            self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
            self.assertTrue(
                environment["PARSEC_CUPY_TEMP_DIR"].endswith(
                    str(Path(".parsec_cache") / "cupy-temp")
                )
            )

    def test_worker_respects_explicit_openblas_thread_count(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENBLAS_NUM_THREADS": "3"},
            clear=True,
        ):
            self.assertEqual(_worker_environment()["OPENBLAS_NUM_THREADS"], "3")

    def test_worker_respects_explicit_cupy_temporary_directory(self) -> None:
        with patch.dict(
            os.environ,
            {"PARSEC_CUPY_TEMP_DIR": "D:/scratch/cupy"},
            clear=True,
        ):
            self.assertEqual(
                _worker_environment()["PARSEC_CUPY_TEMP_DIR"],
                "D:/scratch/cupy",
            )

    def test_request_uses_client_directory_and_captures_streams(self) -> None:
        original = Path.cwd()
        expected = (
            Path(__file__).resolve().parent
            / f".resident-request-test-{os.getpid()}"
        )
        expected.mkdir(parents=False, exist_ok=False)
        try:

            def runner(arguments):
                print(f"cwd={Path.cwd()}")
                self.assertEqual(arguments, ["parsec.in", "--quiet"])
                self.assertEqual(Path.cwd(), expected)
                return 7

            response = execute_request(
                {
                    "argv": ["parsec.in", "--quiet"],
                    "cwd": str(expected),
                },
                runner,
            )
        finally:
            expected.rmdir()

        self.assertEqual(response["exit_code"], 7)
        self.assertIn(f"cwd={expected}", response["stdout"])
        self.assertEqual(response["stderr"], "")
        self.assertEqual(Path.cwd(), original)

    def test_invalid_request_does_not_call_runner(self) -> None:
        called = False

        def runner(_arguments):
            nonlocal called
            called = True
            return 0

        response = execute_request({"argv": "not-a-list", "cwd": os.getcwd()}, runner)

        self.assertEqual(response["exit_code"], 2)
        self.assertIn("Invalid resident", response["stderr"])
        self.assertFalse(called)

    def test_argparse_style_system_exit_is_returned_without_traceback(self) -> None:
        def runner(_arguments):
            print("help text")
            raise SystemExit(0)

        response = execute_request(
            {"argv": ["--help"], "cwd": os.getcwd()}, runner
        )

        self.assertEqual(response["exit_code"], 0)
        self.assertEqual(response["stdout"], "help text\n")
        self.assertEqual(response["stderr"], "")


if __name__ == "__main__":
    unittest.main()
