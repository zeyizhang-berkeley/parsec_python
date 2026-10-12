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
    PYFHI98PP = "pyfhi98pp"
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
    ABINIT = "abinit"
    PARSEC = "parsec"
    UPF = "upf"
    PSP8 = "psp8"
    SIESTA = "siesta"
    CPW2000 = "cpw2000"


# Formats written when a request names none.  The external FHI98PP backend needs
# external converters for anything but its own table.
DEFAULT_FORMATS = {
    BackendName.PYFHI98PP: (OutputFormat.FHI, OutputFormat.UPF, OutputFormat.PARSEC),
    BackendName.FHI98PP: (OutputFormat.FHI,),
    BackendName.ATOM: (OutputFormat.PARSEC,),
}


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
    backend: BackendName = BackendName.PYFHI98PP
    family: PseudopotentialFamily = PseudopotentialFamily.NCPP
    xc: str = "pbe"
    scheme: ConstructionScheme = ConstructionScheme.TROULLIER_MARTINS
    core_hole: CoreHole | None = None
    cutoff_radius: float | None = None
    input_file: Path | None = None
    # ``None``: the formats the backend writes by default (see DEFAULT_FORMATS).
    formats: tuple[OutputFormat, ...] | None = None
    prefix: str | None = None
    local_channel: int | None = None
    scan_local_channels: bool = True
    reject_ghosts: bool = True
    fhi_root: Path | None = None
    atom_executable: Path | None = None
    atom_kb_executable: Path | None = None
    qe_converter: Path | None = None
    potre_converter: Path | None = None
    # Template controls (pyfhi98pp backend).  ``None`` keeps the template value.
    variant: str | None = None
    relativistic: bool = True
    lmax: int | None = None
    channel_radii: dict[int, float] | None = None
    reference_energies_ev: dict[int, float] | None = None
    core_radius: float | None = None
    shift_reference_energies: bool = True
    # Core holes: take lmax, cutoff radii and local channel from the ground-state
    # potential of the same element, so that the two differ only by the hole.
    match_ground_state: bool = True
    # The ground-state result to match, if the caller already has it (the library
    # builder); otherwise the backend generates it.
    ground_state: "GenerationResult | None" = None
    # With ``strict_match`` a core hole that cannot be constructed in the matched
    # form raises instead of falling back to its own choice.
    strict_match: bool = False


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


@dataclass(frozen=True)
class ChannelSummary:
    """One generated angular channel, as tabulated in the generator log."""

    angular_momentum: int
    scheme: str
    cutoff_radius: float            # mesh point actually used, bohr
    match_radius: float
    ae_eigenvalue_ev: float
    pseudo_eigenvalue_ev: float
    norm_test: float
    slope_test: float


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
    lmax: int | None = None
    channels: list[ChannelSummary] = field(default_factory=list)
    core_radius: float | None = None
    reference_shift_ev: float = 0.0
    # Core holes: True if lmax, radii and local channel are those of the ground
    # state, False if that form was not ghost-free, None if matching was not asked.
    matched_to_ground_state: bool | None = None
    # Plane-wave cutoff (Ry) at which the kinetic energy of each channel is
    # converged to 1 meV, from the momentum-space test of the selected channel.
    kinetic_cutoff_ry: dict[int, float] = field(default_factory=dict)

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
