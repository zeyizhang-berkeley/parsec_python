"""Exceptions of the pure-Python FHI98PP port."""

from __future__ import annotations

from ..errors import BackendError


class FortranStop(BackendError):
    """A ``stop`` statement of the Fortran program was reached.

    ``str(exc)`` is the stop message of the original source, so the
    conditions can be mapped one to one.  ``messages`` holds the lines the
    program had written to the error unit before it stopped.

    Stops of the pseudopotential stage of ``fhipp`` also set ``state`` (the
    program state reached, with unit 23 as written so far) and, if the stop
    happened while one channel was being checked or constructed, ``channel``
    (its angular momentum); both are ``None`` otherwise.
    """

    messages: list[str] = []
    channel: int | None = None
    state: object | None = None
