"""Opt-in, scalar-only JSON diagnostics. Disabled by default.

PARSEC_DEBUG=summary (1) records existing solver decisions; audit (2) also
enables explicitly instrumented numerical checks. Neither level changes a
solver decision. PARSEC_DEBUG_DIR optionally keeps one JSONL file per MPI
rank; stdout always has the searchable ``PARSEC_DEBUG `` prefix.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time

_lock = threading.Lock()
_sequence = 0
_warned_paths = set()


def level() -> int:
    value = os.environ.get("PARSEC_DEBUG", "off").strip().lower()
    choices = {"": 0, "off": 0, "0": 0, "false": 0,
               "summary": 1, "1": 1, "on": 1, "true": 1,
               "audit": 2, "2": 2}
    if value not in choices:
        raise ValueError("PARSEC_DEBUG must be off, summary or audit (0, 1, 2)")
    return choices[value]


def enabled(minimum: int = 1) -> bool:
    return level() >= minimum


def world_rank() -> int:
    for key in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0


def wants(minimum: int = 1, *, leader: bool | None = None) -> bool:
    if not enabled(minimum):
        return False
    selection = os.environ.get("PARSEC_DEBUG_RANKS", "leaders").strip().lower()
    if selection not in {"leaders", "all"}:
        raise ValueError("PARSEC_DEBUG_RANKS must be leaders or all")
    return selection == "all" or (world_rank() == 0 if leader is None else leader)


def buffer_bytes() -> int:
    value = int(os.environ.get("PARSEC_DEBUG_BUFFER_BYTES", str(64 * 1024**2)))
    if value < 1024:
        raise ValueError("PARSEC_DEBUG_BUFFER_BYTES must be at least 1024")
    return value


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    # Never turn an array of orbitals into a log message.
    if getattr(value, "ndim", None) == 0:
        return _plain(value.item())
    raise TypeError("debug records accept scalars and small scalar containers only")


def emit(event: str, *, minimum: int = 1, leader: bool | None = None, **values) -> None:
    if not wants(minimum, leader=leader):
        return
    global _sequence
    with _lock:
        _sequence += 1
        record = dict(schema="parsec-debug-v1", event=event, time=time.time(),
                      rank=world_rank(), pid=os.getpid(), sequence=_sequence,
                      **_plain(values))
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        print("PARSEC_DEBUG " + line, flush=True)
        directory = os.environ.get("PARSEC_DEBUG_DIR", "").strip()
        if directory:
            path = Path(directory).expanduser() / f"rank-{world_rank():05d}.jsonl"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError as error:
                # A local logging error must not strand other ranks in MPI.
                # Stdout remains the primary durable campaign log.
                if str(path) not in _warned_paths:
                    _warned_paths.add(str(path))
                    print("PARSEC_DEBUG " + json.dumps(dict(
                        schema="parsec-debug-v1", event="debug_file_error", rank=world_rank(),
                        error=str(error), path=str(path))), flush=True)


def field_summary(values, *, weights=None, volume_element=1.0) -> dict:
    """Summarize a real host scalar field with bounded temporary storage.

    The caller gates this scan with ``wants(2)``. Orbit multiplicities, when
    supplied, make the integral that of the full physical grid.
    """
    import numpy as np

    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind != "f":
        raise ValueError("debug field_summary expects a real one-dimensional host field")
    if weights is not None and np.shape(weights) != values.shape:
        raise ValueError("debug field weights differ from the field shape")
    total = squares = 0.0
    finite = 0
    low, high = float("inf"), -float("inf")
    width = max(1, buffer_bytes() // (3 * values.dtype.itemsize))
    for first in range(0, values.size, width):
        piece = values[first:first + width]
        finite += int(np.count_nonzero(np.isfinite(piece)))
        low = min(low, float(np.min(piece)))
        high = max(high, float(np.max(piece)))
        weighted = piece if weights is None else piece * weights[first:first + width]
        total += float(np.sum(weighted, dtype=np.float64))
        squares += float(np.dot(piece, weighted))
    return dict(count=int(values.size), finite_count=finite, all_finite=finite == values.size,
                minimum=None if not values.size else low, maximum=None if not values.size else high,
                integral=total * volume_element, l2=math.sqrt(max(0., squares * volume_element)))
