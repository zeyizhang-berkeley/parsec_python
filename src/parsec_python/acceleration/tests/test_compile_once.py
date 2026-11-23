"""One compilation of a kernel that several threads ask CuPy for at once (``PARSEC_CUPY_COMPILE_ONCE``).

CPU tests with a stand-in for ``cupy.cuda.compiler``: the wrapper serializes the calls of one kernel and leaves the
others side by side, passes the in-memory compilations through, and is installed once.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from parsec_python.acceleration.backends import cupy_compile


def _compiler():
    """A compiler module whose entry records how many calls of each kernel run at the same time."""

    state = SimpleNamespace(active={}, most={}, calls=[], lock=threading.Lock())

    def _compile_with_cache_cuda(
        source, options, arch, extra_source=None, backend="nvrtc", enable_cooperative_groups=False,
        name_expressions=None, log_stream=None, cache_in_memory=False, jitify=False, to_ltoir=False,
    ):
        with state.lock:
            state.active[source] = state.active.get(source, 0) + 1
            state.most[source] = max(state.most.get(source, 0), state.active[source])
            state.calls.append(source)
        time.sleep(0.05)
        with state.lock:
            state.active[source] -= 1
        return f"module of {source}"

    return SimpleNamespace(_compile_with_cache_cuda=_compile_with_cache_cuda), state


class CompileOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        cupy_compile._COMPILE_LOCKS.clear()

    def test_the_switch(self) -> None:
        for raw, value in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": raw}):
                self.assertEqual(cupy_compile.compile_once_setting(), value)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "on"}):
            with self.assertRaises(ValueError):
                cupy_compile.compile_once_setting()

    def test_one_kernel_at_a_time_and_different_kernels_side_by_side(self) -> None:
        compiler, state = _compiler()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "auto"}):
            self.assertTrue(cupy_compile.install_compile_once(compiler))
        entry = compiler._compile_with_cache_cuda
        jobs = [("kernel a", ("-O3",), "80")] * 4 + [("kernel b", ("-O3",), "80")] * 4
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda job: entry(*job), jobs))
        self.assertEqual(results, [f"module of {job[0]}" for job in jobs])
        # every thread still calls the entry (the later ones read the kernel cache), one at a time per kernel
        self.assertEqual(len(state.calls), 8)
        self.assertEqual(state.most, {"kernel a": 1, "kernel b": 1})

    def test_other_options_are_another_kernel(self) -> None:
        compiler, state = _compiler()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "auto"}):
            cupy_compile.install_compile_once(compiler)
        entry = compiler._compile_with_cache_cuda
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda options: entry("kernel", options, "80"), [("-O3",), ("-O2",)]))
        self.assertEqual(state.most, {"kernel": 2})

    def test_compilations_kept_in_memory_do_not_wait(self) -> None:
        compiler, state = _compiler()
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "auto"}):
            cupy_compile.install_compile_once(compiler)
        entry = compiler._compile_with_cache_cuda
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: entry("kernel", (), "80", cache_in_memory=True), range(4)))
        self.assertGreater(state.most["kernel"], 1)

    def test_installed_once_and_not_under_off(self) -> None:
        compiler, _state = _compiler()
        original = compiler._compile_with_cache_cuda
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "off"}):
            self.assertFalse(cupy_compile.install_compile_once(compiler))
        self.assertIs(compiler._compile_with_cache_cuda, original)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_COMPILE_ONCE": "auto"}):
            self.assertTrue(cupy_compile.install_compile_once(compiler))
            wrapped = compiler._compile_with_cache_cuda
            self.assertTrue(cupy_compile.install_compile_once(compiler))
        self.assertIs(compiler._compile_with_cache_cuda, wrapped)
        self.assertIs(wrapped.__wrapped__, original)
        # a module without the entry is left alone
        self.assertFalse(cupy_compile.install_compile_once(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
