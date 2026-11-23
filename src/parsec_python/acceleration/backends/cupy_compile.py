"""Robust temporary workspace for CuPy RawKernel/RawModule compilation.

CuPy normally delegates NVRTC's transient ``.cu`` files to the operating
system temporary directory.  Restricted service accounts and sandboxes can
make that directory readable but not writable; silently falling back from a
specialized kernel in that situation is both surprising and much slower.

Resident workers set ``PARSEC_CUPY_TEMP_DIR`` to the repository's private
cache.  Standalone runs use the ordinary system location and retry once under
``.parsec_cache/cupy-temp`` only when that location raises ``PermissionError``.
This changes compilation placement only; generated device code and kernel
arguments are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import tempfile
from typing import Any


_TEMP_DIRECTORY_LOCK = Lock()


def _compile_in_directory(compilable: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # ``tempfile.tempdir`` is process-global.  Serialize the short cold-start
    # compile window and restore the caller's policy immediately afterward.
    with _TEMP_DIRECTORY_LOCK:
        previous = tempfile.tempdir
        try:
            tempfile.tempdir = str(directory.resolve())
            compilable.compile()
        finally:
            tempfile.tempdir = previous


def compile_cupy_raw(compilable: Any) -> None:
    """Compile one CuPy raw object with a writable deterministic fallback."""

    configured = os.environ.get("PARSEC_CUPY_TEMP_DIR", "").strip()
    if configured:
        _compile_in_directory(compilable, Path(configured))
        return
    try:
        compilable.compile()
    except PermissionError:
        _compile_in_directory(
            compilable,
            Path.cwd() / ".parsec_cache" / "cupy-temp",
        )



COMPILE_ONCE_SETTING = "PARSEC_CUPY_COMPILE_ONCE"
_COMPILE_LOCKS: dict = {}
_COMPILE_LOCKS_GUARD = Lock()


def compile_once_setting() -> str:
    """``PARSEC_CUPY_COMPILE_ONCE``: ``auto`` (the default) or ``off``.

    ``auto``: a kernel that several threads of one process ask CuPy for at the same time is compiled by one of them;
    the others wait for it and then load the binary that CuPy wrote to its kernel cache
    (:func:`install_compile_once`).  The GPUs of one process build their sector operators and solve their first
    sectors on one thread each, and on a first run each thread compiled every kernel itself, all at once, which made
    the first solve of four GPUs slower than that of one (a Td gold cluster of 56 atoms, four sectors on four A100,
    an empty kernel cache: the first eigensolve 13.7 s against 6.9 s with this, 1.6 s on one GPU).  The kernels and
    their binaries are the same; only the order of the compilations changes.  ``off`` lets every thread compile, as
    before.
    """

    value = os.environ.get(COMPILE_ONCE_SETTING, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{COMPILE_ONCE_SETTING} must be auto or off, not {value!r}")
    return value


def _kernel_key(signature: Any, args: tuple, kwargs: dict) -> tuple | None:
    """What makes two calls of CuPy's compile entry the same kernel, or ``None`` where they are not to wait.

    Every argument but the log stream (an object), with sequences as tuples.  A compilation kept in memory only
    (``cache_in_memory``) is not written to the kernel cache, so a waiting thread would compile it again after the
    first: such calls, those with a log stream and those whose arguments do not bind, keep their own compilation.
    """

    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    values = bound.arguments
    if values.get("cache_in_memory") or values.get("log_stream") is not None:
        return None
    key = []
    for name, value in values.items():
        if name == "log_stream":
            continue
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        elif isinstance(value, (set, frozenset)):
            value = tuple(sorted(value))
        try:
            hash(value)
        except TypeError:
            return None
        key.append((name, value))
    return tuple(key)


def install_compile_once(compiler: Any = None) -> bool:
    """Make the compilations of one kernel in several threads of this process wait for the first
    (``PARSEC_CUPY_COMPILE_ONCE=auto``); ``True`` where the wrapper is in place.

    The wrapper goes around ``cupy.cuda.compiler._compile_with_cache_cuda``, the entry through which CuPy compiles
    raw kernels, raw modules and its own elementwise and reduction kernels and reads them back from its kernel cache:
    one lock per kernel (its source, options, architecture and the other arguments), so that the threads of the
    other GPUs read the binary the first one wrote, while different kernels still compile side by side.  A CuPy
    without that entry, or whose entry is already wrapped, is left as it is.  ``compiler`` is the module (tests).
    """

    if compile_once_setting() == "off":
        return False
    if compiler is None:
        try:
            import cupy.cuda.compiler as compiler  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001 - no CuPy compiler module: nothing to wrap
            return False
    original = getattr(compiler, "_compile_with_cache_cuda", None)
    if not callable(original):
        return False
    if getattr(original, "_parsec_compile_once", False):
        return True
    import inspect

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False

    def compile_once(*args, **kwargs):
        key = _kernel_key(signature, args, kwargs)
        if key is None:
            return original(*args, **kwargs)
        with _COMPILE_LOCKS_GUARD:
            lock = _COMPILE_LOCKS.setdefault(key, Lock())
        with lock:
            return original(*args, **kwargs)

    compile_once._parsec_compile_once = True  # type: ignore[attr-defined]
    compile_once.__wrapped__ = original  # type: ignore[attr-defined]
    compiler._compile_with_cache_cuda = compile_once
    return True


__all__ = ["COMPILE_ONCE_SETTING", "compile_cupy_raw", "compile_once_setting", "install_compile_once"]
