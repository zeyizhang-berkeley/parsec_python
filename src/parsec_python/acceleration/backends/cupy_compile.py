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


__all__ = ["compile_cupy_raw"]
