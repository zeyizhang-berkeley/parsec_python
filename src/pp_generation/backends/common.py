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


# A ghost-free verdict whose worst spectral margin is below this (Hartree) is on
# a knife edge: another functional or a core hole flips it.  Such a channel is
# chosen only when no other ghost-free channel exists.
MINIMUM_MARGIN_HARTREE = 2.0e-3


def comfortable(candidate: LocalChannelResult, minimum_margin: float = MINIMUM_MARGIN_HARTREE) -> bool:
    """Ghost-free with a margin of at least ``minimum_margin`` (or no projector at all)."""

    if not candidate.passed:
        return False
    margin = candidate.minimum_margin_hartree
    return not candidate.channels or margin is None or margin >= minimum_margin


def choose_local_channel(
    candidates: list[LocalChannelResult], *, requested: int | None, reject_ghosts: bool,
    minimum_margin: float = 0.0,
) -> LocalChannelResult:
    if not candidates:
        raise BackendError("the backend produced no local-channel validation results")
    if requested is not None:
        selected = next((x for x in candidates if x.local_channel == requested), None)
        if selected is None:
            raise BackendError(f"requested local channel l={requested} was not tested")
    else:
        safe = [x for x in candidates if x.passed]
        comfortable_ones = [x for x in safe if comfortable(x, minimum_margin)]
        pool = comfortable_ones or safe or candidates
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
