"""Kernels compiled ahead of their first launch on background threads (``PARSEC_CUPY_COMPILE_AHEAD``).

CPU tests: the switch, the compilations planned for the inputs of the examples, nothing under ``off``, a failing
task dropped and the device a process compiles on.  On a GPU, in fresh processes with empty kernel caches: every
task of the examples compiles, and a run of the command line with compilations ahead prints the energies of a run
without.
"""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from parsec_python.acceleration.backends import compile_ahead
from parsec_python.acceleration.backends.cupy import cupy_available

_ROOT = Path(__file__).resolve().parents[4]


def _problem(folder: str, pp: str):
    from parsec_python.Input.parsec_input import parse_parsec_input

    return parse_parsec_input(_ROOT / folder / "parsec.in", pseudopotential_directory=_ROOT / pp).problem


class _FakeCuPy:
    """Enough of CuPy for a worker thread: devices that record which one a thread made its own."""

    def __init__(self) -> None:
        self.used = []
        self.cuda = SimpleNamespace(Device=lambda device_id: _Device(self, device_id))


class _Device:
    def __init__(self, cp, device_id) -> None:
        self.cp, self.id = cp, device_id

    def use(self) -> None:
        self.cp.used.append(self.id)

    def __enter__(self):
        raise AssertionError("the worker makes its device its own (use); leaving a block would set device 0")

    def __exit__(self, *exc):
        return False


class CompileAheadTests(unittest.TestCase):
    def setUp(self) -> None:
        compile_ahead.wait(60.0)
        compile_ahead._SUBMITTED.clear()
        compile_ahead._FAILURES.clear()
        compile_ahead._DONE.clear()

    def test_the_switch(self) -> None:
        for raw, value in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_AHEAD": raw}):
                self.assertEqual(compile_ahead.compile_ahead_setting(), value)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_AHEAD": "on"}):
            with self.assertRaises(ValueError):
                compile_ahead.compile_ahead_setting()
            with self.assertRaises(ValueError):
                compile_ahead.start(None)

    def test_the_plans_of_the_examples(self) -> None:
        hybrid = "examples/0d_hybrid_functionals"
        setup = [("orbital_density",), ("ionic_sum",), ("symmetric_boundary",), ("projectors",)]
        checks = ("device_checks",)
        cases = (
            (f"{hybrid}/CH4/pbe0", hybrid, [("hybrid_xc", "pbe0", False), checks, *setup, ("device_xc", "pbe", False)]),
            (f"{hybrid}/O2/b3lyp", hybrid, [("hybrid_xc", "b3lyp", True), checks, *setup, ("device_xc", "pbe", True)]),
            (f"{hybrid}/CH4/pbe", hybrid, [checks, *setup, ("device_xc", "pbe", False)]),
            ("examples/0d_meta_gga/CH4/scan", "examples/0d_meta_gga", [checks, *setup, ("stencil_xc",)]),
            # CA-LDA of a cluster is the native evaluator's; the fcc cell of silicon keeps its functional on the host
            ("examples/0d_benzene", "examples/0d_benzene", [checks, *setup]),
            ("examples/3d_meta_gga/Si_k222/pbe", "examples/3d_meta_gga", [checks]),
        )
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": "auto"}):
            for folder, pp, expected in cases:
                keys = [task.key[:3] for task in compile_ahead.planned_tasks(_problem(folder, pp))]
                self.assertEqual(keys, expected, folder)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": "off"}):
            keys = [task.key for task in compile_ahead.planned_tasks(_problem(f"{hybrid}/CH4/pbe0", hybrid))]
            self.assertEqual(keys, [checks, *setup])
        self.assertEqual([task.key for task in compile_ahead.planned_tasks(None)], [checks])

    def test_off_starts_nothing(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_AHEAD": "off"}):
            self.assertFalse(compile_ahead.start(None))
        self.assertEqual(compile_ahead._SUBMITTED, set())

    def test_tasks_run_once_and_a_failing_task_is_dropped(self) -> None:
        calls = []

        def good(cp, device_id):
            calls.append(("good", device_id))

        def bad(cp, device_id):
            raise RuntimeError("no such kernel")

        tasks = [compile_ahead.Task(("good",), good), compile_ahead.Task(("bad",), bad)]
        fake = _FakeCuPy()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_AHEAD": "auto", "PARSEC_CUPY_DEVICES": "1"}), \
                mock.patch.object(compile_ahead, "planned_tasks", return_value=tasks), \
                mock.patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(fake, None)), \
                mock.patch("parsec_python.acceleration.backends.cupy.cupy_device_count", return_value=2):
            self.assertTrue(compile_ahead.start(None, threads=2))
            self.assertTrue(compile_ahead.wait(30.0))
            # handed over once per process
            self.assertFalse(compile_ahead.start(None))
        self.assertEqual(calls, [("good", 1)])
        # each thread made device 1 its own, with no return to device 0 at its end
        self.assertEqual(set(fake.used), {1})
        report = compile_ahead.report()
        self.assertEqual(report["done"], [("good",)])
        self.assertEqual([key for key, _ in report["failures"]], [("bad",)])
        self.assertIn("no such kernel", report["failures"][0][1])

    def test_the_device_of_the_process(self) -> None:
        device = compile_ahead._device_of_process
        with mock.patch("parsec_python.acceleration.backends.cupy.cupy_device_count", return_value=4):
            for environment, expected in (
                ({"PARSEC_CUPY_DEVICES": "auto"}, 0),
                ({"PARSEC_CUPY_DEVICES": "2,3"}, 2),
                ({"PARSEC_CUPY_DEVICES": "current"}, None),
                ({"PARSEC_CUPY_DEVICES": "7"}, None),
                # an MPI rank that sees four devices and was given none of its own
                ({"PARSEC_CUPY_DEVICES": "auto", "PMI_SIZE": "4"}, None),
                ({"PARSEC_CUPY_DEVICES": "1", "PMI_SIZE": "4"}, 1),
            ):
                cleared = {name: "" for name in compile_ahead._WORLD_SIZE_VARIABLES}
                with mock.patch.dict(os.environ, {**cleared, **environment}):
                    self.assertEqual(device(None, None), expected, environment)
            self.assertEqual(device(None, 3), 3)
        with mock.patch("parsec_python.acceleration.backends.cupy.cupy_device_count", return_value=1), \
                mock.patch.dict(os.environ, {"PARSEC_CUPY_DEVICES": "auto", "PMI_SIZE": "4"}):
            # one device per rank (CUDA_VISIBLE_DEVICES of the launcher)
            self.assertEqual(device(None, None), 0)


_AHEAD = r"""
import os, sys
os.environ["CUPY_CACHE_DIR"] = sys.argv[1]
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.acceleration.backends import compile_ahead
for folder, pp in zip(sys.argv[2::2], sys.argv[3::2]):
    compile_ahead.start(parse_parsec_input(folder + "/parsec.in", pseudopotential_directory=pp).problem)
assert compile_ahead.wait(600.0)
report = compile_ahead.report()
print("DONE", sorted(key[0] for key in report["done"]))
print("FAILURES", report["failures"])
"""


_CLI = "import sys; from parsec_python.acceleration import cli; sys.exit(cli.main(sys.argv[1:]))"


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class CompileAheadDeviceTests(unittest.TestCase):
    def test_the_tasks_of_the_examples_compile(self) -> None:
        # every task of a hybrid, a spin hybrid, a GGA and a meta-GGA cluster and of a cell compiles on the device
        hybrid = _ROOT / "examples" / "0d_hybrid_functionals"
        folders = [
            (hybrid / "CH4" / "pbe0", hybrid), (hybrid / "O2" / "b3lyp", hybrid), (hybrid / "CH4" / "pbe", hybrid),
            (_ROOT / "examples" / "0d_meta_gga" / "CH4" / "scan", _ROOT / "examples" / "0d_meta_gga"),
        ]
        with tempfile.TemporaryDirectory(prefix="parsec_ahead_cache_") as cache:
            environment = dict(os.environ, PYTHONPATH=str(_ROOT / "src"), PARSEC_CUPY_COMPILE_AHEAD="auto")
            arguments = [str(value) for pair in folders for value in pair]
            result = subprocess.run([sys.executable, "-c", _AHEAD, cache, *arguments], env=environment,
                                    capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            lines = {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in result.stdout.splitlines()
                     if line.startswith(("DONE", "FAILURES"))}
            self.assertEqual(lines["FAILURES"], "[]")
            for name in ("hybrid_xc", "device_checks", "orbital_density", "ionic_sum", "symmetric_boundary",
                         "projectors", "device_xc", "stencil_xc"):
                self.assertIn(f"'{name}'", lines["DONE"])

    def test_a_run_with_compilations_ahead_prints_the_same_energies(self) -> None:
        folder = _ROOT / "examples" / "0d_hybrid_functionals"
        text = (folder / "CH4" / "pbe0" / "parsec.in").read_text()
        text = text.replace("Grid_Spacing: 0.160", "Grid_Spacing: 0.30").replace("Exx_Tolerance: 1e-8", "Exx_Tolerance: 1e-6")
        energies = {}
        with tempfile.TemporaryDirectory(prefix="parsec_ahead_run_") as work:
            for value in ("off", "auto"):
                run = Path(work) / value
                run.mkdir()
                (run / "parsec.in").write_text(text)
                environment = dict(os.environ, PYTHONPATH=str(_ROOT / "src"), PARSEC_CUPY_COMPILE_AHEAD=value,
                                   CUPY_CACHE_DIR=str(Path(work) / f"cache_{value}"))
                result = subprocess.run(
                    [sys.executable, "-c", _CLI, "parsec.in", "--pp-dir", str(folder), "--no-archive", "--gpus", "1"],
                    cwd=run, env=environment, capture_output=True, text=True, timeout=1200,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                output = (run / "parsec.out").read_text()
                energies[value] = re.findall(r"Total Energy[^=\n]*=\s*(\S+)", output)
        self.assertTrue(energies["off"])
        self.assertEqual(energies["auto"], energies["off"])


if __name__ == "__main__":
    unittest.main()
