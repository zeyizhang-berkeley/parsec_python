"""User-facing pseudopotential generation and validation tools.

The current numerical backends are the independently validated FHI98PP and
ATOM executables.  This package is not yet a pure-Python radial solver.
"""

from .models import (
    BackendName,
    ConstructionScheme,
    CoreHole,
    GenerationRequest,
    GenerationResult,
    GhostChannel,
    GhostStatus,
    LocalChannelResult,
    OutputFormat,
    PseudopotentialFamily,
)
from .generator import generate

__all__ = [
    "BackendName", "ConstructionScheme", "CoreHole", "GenerationRequest",
    "GenerationResult", "GhostChannel", "GhostStatus", "LocalChannelResult",
    "OutputFormat", "PseudopotentialFamily", "generate",
]
