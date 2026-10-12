"""User-facing pseudopotential generation and validation tools.

The default backend is ``pyfhi98pp``, a port of FHI98PP to Python shipped in
this package (numpy only).  The ``fhi98pp`` and ``atom`` backends drive the
external programs of those names and exist for cross-checks.
"""

from .models import (
    BackendName,
    ChannelSummary,
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
    "BackendName", "ChannelSummary", "ConstructionScheme", "CoreHole", "GenerationRequest",
    "GenerationResult", "GhostChannel", "GhostStatus", "LocalChannelResult",
    "OutputFormat", "PseudopotentialFamily", "generate",
]
