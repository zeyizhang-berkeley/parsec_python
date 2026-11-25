"""Tests of finiteness and of a set flag without CuPy's CUB reductions (``acceleration/backends/device_checks.py``).

``all_finite`` and ``any_set`` give the answers of ``isfinite(x).all()`` and ``mask.any()``: NumPy arrays on the
host, and on a GPU every layout and type of array, a value that is not finite (or set) at the first, a middle and
the last place, empty arrays, views of up to three axes with any strides (and one of four, which takes the CuPy
expression) and several arrays at once.  In a fresh process with an empty kernel cache neither compiles a CUB
kernel.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from parsec_python.acceleration.backends import device_checks
from parsec_python.acceleration.backends.cupy import cupy_available

_ROOT = Path(__file__).resolve().parents[4]


def _arrays(rng):
    for size in (0, 1, 7, 1000, 100_003):
        yield rng.standard_normal(size)
    yield rng.standard_normal((37, 53))
    yield np.asfortranarray(rng.standard_normal((37, 53)))
    yield rng.standard_normal((40, 30))[:, ::2]  # not one block
    yield rng.standard_normal((40, 30))[:, :1]
    yield (rng.standard_normal(5001) + 1j * rng.standard_normal(5001))
    yield np.asfortranarray(rng.standard_normal((21, 19)) + 1j * rng.standard_normal((21, 19)))
    yield rng.standard_normal(4099).astype(np.float32)
    yield (rng.standard_normal(999) + 1j * rng.standard_normal(999)).astype(np.complex64)
    yield np.arange(12, dtype=np.int64).reshape(3, 4)


def _spoiled(array, rng):
    """``array`` and copies with a NaN, an infinity or a negative infinity at the first, a middle and the last place."""

    yield array
    if array.size == 0 or array.dtype.kind not in "fc":
        return
    for value in (np.nan, np.inf, -np.inf):
        for place in (0, array.size // 2, array.size - 1):
            copy = array.copy()
            index = np.unravel_index(place, array.shape)
            copy[index] = value if array.dtype.kind == "f" or rng.random() < 0.5 else complex(0.0, value)
            yield copy


class HostTests(unittest.TestCase):
    def test_numpy_arrays(self) -> None:
        rng = np.random.default_rng(1)
        for array in _arrays(rng):
            for value in _spoiled(array, rng):
                self.assertEqual(device_checks.all_finite(value), bool(np.isfinite(value).all()))
        for mask in (np.zeros(0, bool), np.zeros(9, bool), np.eye(3, dtype=bool)):
            self.assertEqual(device_checks.any_set(mask), bool(mask.any()))
        self.assertTrue(device_checks.all_finite([1.0, 2.0]))
        self.assertFalse(device_checks.all_finite([1.0, np.nan]))
        # NumPy scalars (which carry device, data and flags in NumPy 2) and Python numbers are host values
        for value in (np.float64(np.nan), np.float64(-np.inf), np.float32(np.inf), np.complex128(complex(0.0, np.nan))):
            self.assertFalse(device_checks.all_finite(value), value)
        for value in (np.float64(1.5), np.float32(0.0), 2.0, 3, np.int64(4), np.True_):
            self.assertTrue(device_checks.all_finite(value), value)
        for value, expected in ((np.True_, True), (np.False_, False), (True, True), (False, False), (np.bool_(1), True)):
            self.assertIs(device_checks.any_set(value), expected)
        self.assertFalse(device_checks._is_device(np.float64(1.0)))


@unittest.skipUnless(cupy_available(), "CuPy is not available")
class DeviceTests(unittest.TestCase):
    def test_device_arrays(self) -> None:
        import cupy as cp

        rng = np.random.default_rng(2)
        count = 0
        for array in _arrays(rng):
            for value in _spoiled(array, rng):
                layout = "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C"
                device = cp.asarray(value, order=layout)
                self.assertEqual(device_checks.all_finite(device), bool(np.isfinite(value).all()),
                                 (value.shape, value.dtype, layout))
                count += 1
        for spoil in (None, np.nan, np.inf, -np.inf):
            # views that are not one block of memory, of up to three axes (complex: four with the parts), with
            # negative strides; and one of four axes (the CuPy expression)
            real = cp.asarray(rng.standard_normal((40, 30)))
            plural = cp.asarray(rng.standard_normal((6, 7, 8)) + 1j * rng.standard_normal((6, 7, 8)))
            fortran = cp.asfortranarray(cp.asarray(rng.standard_normal((33, 21)) + 1j * rng.standard_normal((33, 21))))
            four = cp.asarray(rng.standard_normal((3, 4, 5, 6)))
            for view in (real[:, ::2], real[:, :1], real.T[::3], real[::-2, 5:], plural[:, ::2, 1:],
                         plural[::-1, 3], fortran[:, 2:9], fortran[1:30:2, ::-3], four[:, ::2], real[3:4, ::4]):
                if spoil is not None:
                    index = tuple(min(1, n - 1) for n in view.shape)
                    view[index] = spoil if view.dtype.kind == "f" else complex(spoil, 0.0)
                expected = bool(np.isfinite(cp.asnumpy(view)).all())
                self.assertEqual(device_checks.all_finite(view), expected, (view.shape, view.strides, spoil))
                count += 1
        # several arrays at once, with a NumPy one among them: one flag
        arrays = [cp.asarray(rng.standard_normal(1000)), rng.standard_normal(10), cp.asarray(rng.standard_normal((9, 4)))[:, 1:]]
        self.assertTrue(device_checks.all_finite(*arrays))
        arrays[2][4, 1] = np.nan
        self.assertFalse(device_checks.all_finite(*arrays))
        arrays[1][3] = np.inf
        self.assertFalse(device_checks.all_finite(arrays[0], arrays[1]))
        self.assertTrue(device_checks.all_finite(cp.arange(5), cp.zeros(3, dtype=bool)))
        self.assertGreater(count, 100)
        for size in (0, 1, 1000, 300_001):
            for place in (None, 0, size // 2, size - 1):
                mask = np.zeros(size, dtype=bool)
                if place is not None and size:
                    mask[place] = True
                self.assertEqual(device_checks.any_set(cp.asarray(mask)), bool(mask.any()), (size, place))
        density = cp.asarray(rng.random(100_000))
        self.assertTrue(device_checks.any_set(density >= 0.5))
        self.assertFalse(device_checks.any_set(density >= 2.0))
        # another device than the current one
        if cp.cuda.runtime.getDeviceCount() > 1:
            with cp.cuda.Device(1):
                other = cp.asarray(np.array([1.0, np.inf]))
            with cp.cuda.Device(0):
                self.assertFalse(device_checks.all_finite(other))

    def test_no_cub_kernel_on_a_first_run(self) -> None:
        script = r"""
import os, re, sys
os.environ["CUPY_CACHE_DIR"] = sys.argv[1]
import numpy as np
import cupy as cp
import cupy.cuda.compiler as compiler
names = []
original = compiler._compile_with_cache_cuda
def record(source, *args, **kwargs):
    names.extend(re.findall(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)", source))
    return original(source, *args, **kwargs)
compiler._compile_with_cache_cuda = record
from parsec_python.acceleration.backends.device_checks import all_finite, any_set
x = cp.asarray(np.random.default_rng(0).random(1_000_003))
assert all_finite(x) and all_finite(x.reshape(1000003, 1)[::2]) and not any_set(x >= 2.0)
print("NAMES", sorted(set(names)))
"""
        with tempfile.TemporaryDirectory(prefix="parsec_checks_cache_") as cache:
            environment = dict(os.environ, PYTHONPATH=str(_ROOT / "src"))
            result = subprocess.run([sys.executable, "-c", script, cache], env=environment, capture_output=True,
                                    text=True, timeout=600)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            line = [text for text in result.stdout.splitlines() if text.startswith("NAMES")][-1]
            self.assertIn("parsec_check_f64", line)
            self.assertNotIn("cub", line)


if __name__ == "__main__":
    unittest.main()
