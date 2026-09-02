"""Typed public data model for pseudopotential generation.

The model separates pseudopotential family, numerical backend, construction
scheme, and serialization format.  Future ONCV, PAW, or ultrasoft backends
can therefore use the API without being disguised as Troullier--Martins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BackendName(str, Enum):
    FHI98PP = "fhi98pp"
    ATOM = "atom"


class PseudopotentialFamily(str, Enum):
    NCPP = "ncpp"


class ConstructionScheme(str, Enum):
    TROULLIER_MARTINS = "tm"
    HAMANN = "hamann"


class OutputFormat(str, Enum):
    """Names describe actual formats rather than ambiguous extensions."""

    FHI = "fhi"
    PARSEC = "parsec"
    UPF = "upf"
    PSP8 = "psp8"
    SIESTA = "siesta"
    CPW2000 = "cpw2000"


class GhostStatus(str, Enum):
    PASS = "pass"
    GHOST = "ghost"
    INDETERMINATE = "indeterminate"
    ILL_DEFINED = "ill_defined"


@dataclass(frozen=True)
class CoreHole:
    shell: str
    removed_electrons: float = 1.0


@dataclass(frozen=True)
class GenerationRequest:
    element: str
    output_dir: Path
    backend: BackendName = BackendName.FHI98PP
    family: PseudopotentialFamily = PseudopotentialFamily.NCPP
    xc: str = "pbe"
    scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS
    core_hole: CoreHole | None = None
    cutoff_radius: float | None = None
    input_file: Path | None = None
    formats: tuple[OutputFormat, ...] = (OutputFormat.FHI,)
    prefix: str | None = None
    local_channel: int | None = None
    scan_local_channels: bool = True
    reject_ghosts: bool = True
    fhi_root: Path | None = None
    atom_executable: Path | None = None
    atom_kb_executable: Path | None = None
    qe_converter: Path | None = None
    potre_converter: Path | None = None


@dataclass(frozen=True)
class GhostChannel:
    angular_momentum: int
    label: str
    status: GhostStatus
    message: str
    margin_hartree: float | None = None
    reference_energy_hartree: float | None = None
    local_ground_energy_hartree: float | None = None
    local_first_excited_energy_hartree: float | None = None
    projector_sign: int | None = None

    @property
    def is_ghost(self) -> bool:
        return self.status is GhostStatus.GHOST

    @property
    def passed(self) -> bool:
        return self.status is GhostStatus.PASS


@dataclass(frozen=True)
class LocalChannelResult:
    local_channel: int
    channels: tuple[GhostChannel, ...]
    report_path: Path

    @property
    def passed(self) -> bool:
        # A one-channel local pseudopotential has no KB projectors and is
        # vacuously ghost-free.
        return all(channel.passed for channel in self.channels)

    @property
    def minimum_margin_hartree(self) -> float | None:
        margins = [x.margin_hartree for x in self.channels if x.margin_hartree is not None]
        return min(margins) if margins else None


@dataclass
class GenerationResult:
    request: GenerationRequest
    prefix: str
    input_path: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    local_channel_results: list[LocalChannelResult] = field(default_factory=list)
    selected_local_channel: int | None = None
    ae_energy_hartree: float | None = None
    pseudo_energy_hartree: float | None = None
    ionic_charge: float | None = None
    reference_electrons: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ghost_channels(self) -> list[GhostChannel]:
        for candidate in self.local_channel_results:
            if candidate.local_channel == self.selected_local_channel:
                return list(candidate.channels)
        return []

    @property
    def ghost_free(self) -> bool:
        for candidate in self.local_channel_results:
            if candidate.local_channel == self.selected_local_channel:
                return candidate.passed
        return False
