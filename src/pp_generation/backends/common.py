from __future__ import annotations

import subprocess
from pathlib import Path

from ..errors import BackendError, GhostStateError
from ..models import LocalChannelResult


def run(command: list[str], *, cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command, cwd=cwd, input=stdin, text=True, capture_output=True, check=False
        )
    except OSError as exc:
        raise BackendError(f"cannot execute {command[0]}: {exc}") from exc
    if completed.returncode:
        raise BackendError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def choose_local_channel(
    candidates: list[LocalChannelResult], *, requested: int | None, reject_ghosts: bool
) -> LocalChannelResult:
    if not candidates:
        raise BackendError("the backend produced no local-channel validation results")
    if requested is not None:
        selected = next((x for x in candidates if x.local_channel == requested), None)
        if selected is None:
            raise BackendError(f"requested local channel l={requested} was not tested")
    else:
        safe = [x for x in candidates if x.passed]
        pool = safe or candidates
        # Maximize the worst spectral separation.  A purely local potential
        # has no projectors and therefore an infinite safe margin.
        selected = max(
            pool,
            key=lambda x: float("inf") if not x.channels else (
                x.minimum_margin_hartree
                if x.minimum_margin_hartree is not None
                else float("-inf")
            ),
        )
    if reject_ghosts and not selected.passed:
        summary = ", ".join(
            f"l={item.local_channel}:{'pass' if item.passed else 'fail'}"
            for item in candidates
        )
        raise GhostStateError(f"no acceptable requested KB representation ({summary})")
    return selected
