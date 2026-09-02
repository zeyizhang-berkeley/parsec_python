from __future__ import annotations

from typing import Protocol

from ..models import GenerationRequest, GenerationResult


class GeneratorBackend(Protocol):
    """Contract implemented by numerical pseudopotential generators."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
