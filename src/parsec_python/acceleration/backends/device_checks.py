"""Exact tests of device arrays without CuPy's CUB reductions: whether every value is finite, whether any is set.

``bool(cupy.isfinite(x).all())`` and ``bool(mask.any())`` dispatch to CuPy's CUB block reduction, a kernel that
CuPy compiles for each operation, type and block size with the CUB headers: with an empty kernel cache (a first
run) 1.5 s a kernel, two kernels a test (3.0 to 3.2 s for ``all`` or ``any`` on an A100 node), and a compilation of
CUB holds back the other compilations of the process while it runs.  Every route of the accelerated solver tests
its densities, bases or overlaps so.  :func:`all_finite` and :func:`any_set` give the same answer, a boolean, from
a small kernel of their own (no CuPy headers, compiled in a twentieth of the time): a thread that meets a value
that is not finite (or that is set) writes a flag, which is read back as the reduction was.  The kernels walk an
array of up to three axes with any strides (a complex array as its real and imaginary parts); an array of more
axes that is not one block of memory, or of another type, takes the CuPy expression of before.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

_SOURCE = r"""
template <typename T>
__device__ void walk(const T* x, long long n0, long long n1, long long n2, long long s0, long long s1,
                     long long s2, int finite, int* flag) {
    long long count = n0 * n1 * n2;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += step) {
        long long c = i % n2;
        long long t = i / n2;
        long long b = t % n1;
        long long a = t / n1;
        T value = x[a * s0 + b * s1 + c * s2];
        if (finite ? !isfinite((double)value) : (value != (T)0)) { *flag = 1; return; }
    }
}
extern "C" __global__ void parsec_check_f64(const double* x, long long n0, long long n1, long long n2,
                                            long long s0, long long s1, long long s2, int* flag) {
    walk<double>(x, n0, n1, n2, s0, s1, s2, 1, flag);
}
extern "C" __global__ void parsec_check_f32(const float* x, long long n0, long long n1, long long n2,
                                            long long s0, long long s1, long long s2, int* flag) {
    walk<float>(x, n0, n1, n2, s0, s1, s2, 1, flag);
}
extern "C" __global__ void parsec_check_set(const unsigned char* x, long long n0, long long n1, long long n2,
                                            long long s0, long long s1, long long s2, int* flag) {
    walk<unsigned char>(x, n0, n1, n2, s0, s1, s2, 0, flag);
}
"""
_NAMES = ("parsec_check_f64", "parsec_check_f32", "parsec_check_set")
_THREADS = 256
_MAX_BLOCKS = 2048
_MODULES: dict[int, dict[str, Any]] = {}
_KERNEL_OF = {
    np.dtype(np.float64): ("parsec_check_f64", 8, False),
    np.dtype(np.complex128): ("parsec_check_f64", 8, True),
    np.dtype(np.float32): ("parsec_check_f32", 4, False),
    np.dtype(np.complex64): ("parsec_check_f32", 4, True),
    np.dtype(np.bool_): ("parsec_check_set", 1, False),
}


def kernels(cp) -> dict[str, Any]:
    """The kernels of this module for the current device, compiled once per process (also ahead of a run)."""

    device = int(cp.cuda.Device().id)
    found = _MODULES.get(device)
    if found is None:
        module = cp.RawModule(code=_SOURCE, options=("--std=c++11",))
        found = {name: module.get_function(name) for name in _NAMES}
        _MODULES[device] = found
    return found


def _layout(x):
    """``(kernel, axes, strides)``: ``x`` as at most three axes of its scalars (strides in scalars, the smallest last),
    or ``None`` where it has more axes that are not one block of memory or is of another type."""

    found = _KERNEL_OF.get(np.dtype(x.dtype))
    if found is None:
        return None
    name, scalar, complex_values = found
    parts = 2 if complex_values else 1
    if x.flags.c_contiguous or x.flags.f_contiguous:
        return name, (1, 1, int(x.size) * parts), (0, 0, 1)
    if any(int(stride) % scalar for stride in x.strides):
        return None
    pairs = [(int(n), int(stride) // scalar) for n, stride in zip(x.shape, x.strides) if int(n) != 1]
    if complex_values:
        pairs.append((2, 1))
    if len(pairs) > 3:
        return None
    pairs.sort(key=lambda pair: -abs(pair[1]))
    pairs = [(1, 0)] * (3 - len(pairs)) + pairs
    return name, tuple(n for n, _ in pairs), tuple(stride for _, stride in pairs)


def _flags(cp, arrays) -> bool:
    """Launch the kernel of every ``(device array, layout)`` onto one flag per device; ``True`` where some flag was
    written."""

    by_device: dict[int, Any] = {}
    for x, (name, axes, strides) in arrays:
        device = int(x.device.id)
        with x.device:
            flag = by_device.get(device)
            if flag is None:
                flag = by_device[device] = cp.zeros(1, dtype=cp.int32)
            count = axes[0] * axes[1] * axes[2]
            blocks = max(1, min(_MAX_BLOCKS, (count + _THREADS - 1) // _THREADS))
            kernels(cp)[name]((blocks,), (_THREADS,), (x, *(np.int64(v) for v in axes), *(np.int64(v) for v in strides),
                                                        flag))
    written = False
    for device, flag in by_device.items():
        with cp.cuda.Device(device):
            written = bool(int(flag.get()[0])) or written
    return written


def _is_device(x) -> bool:
    """Whether ``x`` is a CuPy array (a NumPy 2 scalar carries ``device``, ``data`` and ``flags`` too)."""

    cupy = sys.modules.get("cupy")
    return cupy is not None and isinstance(x, cupy.ndarray)


def all_finite(*arrays) -> bool:
    """``bool(xp.isfinite(x).all())`` for every NumPy or CuPy array ``x`` given, without a CUB reduction on the device.

    One read of a flag per device for all the device arrays together.
    """

    device = []
    for x in arrays:
        if not _is_device(x):
            if not bool(np.isfinite(x).all()):
                return False
            continue
        if x.size == 0 or np.dtype(x.dtype).kind in "biu":
            # (booleans and integers are finite)
            continue
        layout = _layout(x)
        if layout is None:
            import cupy as cp

            with x.device:
                if not bool(cp.isfinite(x).all()):
                    return False
            continue
        device.append((x, layout))
    if not device:
        return True
    import cupy as cp

    return not _flags(cp, device)


def any_set(mask) -> bool:
    """``bool(mask.any())`` for a boolean NumPy or CuPy array, without a CUB reduction on the device."""

    if not _is_device(mask):
        return bool(np.asarray(mask).any())
    import cupy as cp

    if mask.size == 0:
        return False
    layout = _layout(mask) if np.dtype(mask.dtype) == np.bool_ else None
    if layout is None:
        with mask.device:
            return bool(mask.any())
    return _flags(cp, [(mask, layout)])


__all__ = ["all_finite", "any_set", "kernels"]
