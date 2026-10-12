"""Backend built on the Python port of FHI98PP shipped in ``pyfhi98pp``.

Nothing outside the package is needed: the atom, the pseudopotential
construction, the Kleinman--Bylander ghost test and the format conversions
all run in-process.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from ..converters import ConversionError, convert_outputs
from ..errors import BackendError, ConfigurationError, GhostStateError
from ..fhi_input import (
    L_LABELS,
    REFERENCE_ENERGY_THRESHOLD_EV,
    XC_CODES,
    ParsedInput,
    customize,
    parse,
)
from ..models import (
    ChannelSummary,
    ConstructionScheme,
    GenerationRequest,
    GenerationResult,
    GhostChannel,
    GhostStatus,
    LocalChannelResult,
    OutputFormat,
    PseudopotentialFamily,
)
from ..templates import ELEMENTS_DIR, template_text
from .common import MINIMUM_MARGIN_HARTREE, choose_local_channel, comfortable
from .fhi98pp import _core_hole_species_label, _default_prefix

SUPPORTED_FORMATS = {OutputFormat.FHI, OutputFormat.ABINIT, OutputFormat.UPF, OutputFormat.PARSEC}
# A requested radius this far above a mesh point selects exactly that point.
MESH_SNAP = 1.0003
VERDICTS = {
    "no ghost": GhostStatus.PASS,
    "one or more ghosts": GhostStatus.GHOST,
    "undetermined": GhostStatus.INDETERMINATE,
    "illdefined": GhostStatus.ILL_DEFINED,
}


@dataclass(frozen=True)
class Reviewed:
    """Reviewed departures from an element template (``elements/reviewed.csv``)."""

    lmax: int | None
    radii: dict[int, float]
    reason: str


def parsed_charge(source: str) -> float:
    """Nuclear charge of an FHI98PP input text."""

    return parse(source).nuclear_charge


def reviewed_overrides(element: str, scheme: ConstructionScheme) -> Reviewed | None:
    path = ELEMENTS_DIR / "reviewed.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["element"] == element and row["scheme"] == scheme.value:
                radii = {
                    ell: float(row[f"radius_{L_LABELS[ell]}"])
                    for ell in range(4) if row.get(f"radius_{L_LABELS[ell]}")
                }
                return Reviewed(int(row["lmax"]) if row["lmax"] else None, radii, row["reason"])
    return None


class PyFhi98ppBackend:
    """Generate, ghost-test and convert with the built-in FHI98PP port."""

    def _validate(self, request: GenerationRequest) -> None:
        if request.family is not PseudopotentialFamily.NCPP:
            raise ConfigurationError("the FHI98PP port supports only the NCPP family")
        unknown = set(request.formats) - SUPPORTED_FORMATS
        if unknown:
            names = ", ".join(sorted(x.value for x in unknown))
            raise ConfigurationError(f"the FHI98PP port cannot write: {names}")
        if request.local_channel is not None and request.local_channel < 0:
            raise ConfigurationError("the local channel must be non-negative")

    # ------------------------------------------------------------------ input
    def _template(self, request: GenerationRequest) -> tuple[str, Reviewed | None]:
        text = template_text(request.element, request.scheme, request.variant)
        reviewed = None if request.variant else reviewed_overrides(request.element, request.scheme)
        return text, reviewed

    def _source(self, request: GenerationRequest, template: str, reviewed: Reviewed | None,
                lmax: int | None, shift: float) -> str:
        radii = dict(reviewed.radii) if reviewed and request.cutoff_radius is None else {}
        radii.update(request.channel_radii or {})
        return customize(
            template, xc=request.xc, scheme=request.scheme, core_hole=request.core_hole,
            cutoff_radius=request.cutoff_radius, lmax=lmax, channel_radii=radii or None,
            reference_energies_ev=request.reference_energies_ev,
            core_radius=request.core_radius, reference_shift_ev=shift,
        )

    def _reference_shift(self, request: GenerationRequest, template: str,
                         reviewed: Reviewed | None, warnings: list[str]) -> float:
        """Shift of the valence levels caused by the core hole (eV).

        Explicit reference energies of a template were chosen relative to the
        valence levels of the neutral atom; with a core hole those levels
        move down by several eV and the reference energies have to follow.
        """

        if request.core_hole is None or not request.shift_reference_energies:
            return 0.0
        explicit = [
            ell for ell, _, energy, _ in parse(template).channel_lines
            if abs(energy) >= REFERENCE_ENERGY_THRESHOLD_EV
            and ell not in (request.reference_energies_ev or {})
        ]
        if not explicit or request.cutoff_radius is not None:
            return 0.0
        from ..pyfhi98pp.psgen import psgen

        def emax(core_hole) -> float:
            text = customize(template, xc=request.xc, scheme=request.scheme, core_hole=core_hole,
                             cutoff_radius=None)
            atom = psgen(text, only_atom=True, write=False,
                         nonrelativistic=not request.relativistic)
            return atom.emax_ev

        shift = emax(request.core_hole) - emax(None)
        warnings.append(
            f"Reference energies of the unbound channels ({', '.join(L_LABELS[x] for x in explicit)}) "
            f"shifted by {shift:+.2f} eV, the shift of the highest valence level caused by the core hole."
        )
        return shift

    # ------------------------------------------------------------ ghost tests
    @staticmethod
    def _ghost_channels(result) -> tuple[GhostChannel, ...]:
        return tuple(
            GhostChannel(
                angular_momentum=ch.l, label=L_LABELS[ch.l], status=VERDICTS[ch.verdict],
                message=ch.message.strip(), margin_hartree=ch.margin,
                reference_energy_hartree=ch.eref, local_ground_energy_hartree=ch.eloc0,
                local_first_excited_energy_hartree=ch.eloc1,
                projector_sign=1 if ch.ekb > 0 else -1,
            )
            for ch in result.channels
        )

    def _candidates(self, request: GenerationRequest, source: str, cpi_text: str,
                    lmax: int, out: Path, prefix: str) -> list[LocalChannelResult]:
        from ..pyfhi98pp.pswatch import ghost_scan

        if request.local_channel is not None:
            if request.local_channel > lmax:
                raise ConfigurationError(
                    f"local channel l={request.local_channel} exceeds generated lmax={lmax}"
                )
            channels = [request.local_channel]
        elif request.scan_local_channels:
            channels = list(range(lmax + 1))
        else:
            channels = [lmax]
        for stale in out.glob(f"{prefix}.local-*.test"):  # from a rejected higher lmax
            stale.unlink()
        candidates = []
        for local, result in ghost_scan(source, cpi_text, channels).items():
            report = out / f"{prefix}.local-{local}.test"
            report.write_text("".join(line + "\n" for line in result.lines), encoding="utf-8",
                              newline="\n")
            candidates.append(LocalChannelResult(local, self._ghost_channels(result), report))
        return candidates

    # ------------------------------------------------- core hole: ground match
    def _matched_to_ground_state(self, request: GenerationRequest) -> GenerationResult | None:
        """Generate a core hole with the ground state's lmax, radii and local channel.

        Returns ``None`` when the ground state cannot be generated or the core
        hole is not ghost-free in that form; the caller then chooses freely.
        """

        ground = request.ground_state
        if ground is None:
            with tempfile.TemporaryDirectory(prefix="pp-ground-") as scratch:
                try:
                    ground = self.generate(replace(
                        request, core_hole=None, formats=(), output_dir=Path(scratch),
                        prefix=None, reject_ghosts=True,
                    ))
                except (BackendError, GhostStateError):
                    return None
        # The same input records as the ground state.  Default Troullier-Martins radii
        # are tabulated over (Z, n, l), so they come out equal by themselves; explicit
        # records are avoided because a record also resets the channel's reference energy.
        matched = replace(
            request, lmax=ground.lmax, local_channel=ground.selected_local_channel,
            reject_ghosts=True, ground_state=None,
        )
        try:
            result = self.generate(matched)
            wanted = {x.angular_momentum: x.cutoff_radius for x in ground.channels}
            differing = [x for x in result.channels
                         if abs(x.cutoff_radius - wanted.get(x.angular_momentum, x.cutoff_radius)) > 1.0e-9]
            if differing:
                # A default that depends on the configuration: impose the ground-state
                # radius, and keep the channel's own reference energy explicitly.
                radii = dict(request.channel_radii or {})
                energies = dict(request.reference_energies_ev or {})
                for channel in differing:
                    radii[channel.angular_momentum] = wanted[channel.angular_momentum] * MESH_SNAP
                    energies.setdefault(channel.angular_momentum, channel.ae_eigenvalue_ev)
                result = self.generate(replace(matched, channel_radii=radii,
                                               reference_energies_ev=energies))
        except GhostStateError:
            return None
        except BackendError:
            if request.strict_match:
                raise
            return None
        result.request = request
        result.matched_to_ground_state = True
        result.warnings.insert(
            0,
            f"Matched to the ground-state potential: lmax {ground.lmax}, local channel "
            f"{L_LABELS[ground.selected_local_channel]}, same cutoff radii.",
        )
        return result

    # --------------------------------------------------------------- generate
    def generate(self, request: GenerationRequest) -> GenerationResult:
        from ..pyfhi98pp.errors import FortranStop
        from ..pyfhi98pp.psgen import psgen
        from ..pyfhi98pp.pswatch import FortranCrash, pswatch

        self._validate(request)
        out = request.output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        prefix = request.prefix or _default_prefix(request)
        warnings: list[str] = []
        shift = 0.0
        unmatched = False

        if (request.core_hole is not None and request.match_ground_state
                and request.input_file is None and request.lmax is None
                and request.local_channel is None and request.scan_local_channels):
            matched = self._matched_to_ground_state(request)
            if matched is not None:
                return matched
            unmatched = True
            warnings.append(
                "Not matched to the ground-state potential: with its lmax, radii and local "
                "channel this core hole cannot be generated ghost-free; own choice used."
            )

        if request.input_file:
            if (request.core_hole is not None or request.cutoff_radius is not None
                    or request.channel_radii or request.lmax is not None):
                raise ConfigurationError(
                    "--input-file is authoritative; do not combine it with core-hole, radius or lmax options"
                )
            sources = [request.input_file.expanduser().read_text(encoding="utf-8")]
            parsed = parse(sources[0])
            expected = XC_CODES.get(request.xc)
            token = "t" if request.scheme is ConstructionScheme.TROULLIER_MARTINS else "h"
            if (expected is not None and parsed.xc_code != expected) or parsed.default_scheme != token:
                raise ConfigurationError("--xc/--scheme disagree with the authoritative --input-file")
        else:
            template, reviewed = self._template(request)
            if reviewed is not None:
                warnings.append(f"Reviewed setting for {request.element}: {reviewed.reason}.")
            base = parse(template)
            shift = self._reference_shift(request, template, reviewed, warnings)
            if request.lmax is not None:
                tries = [request.lmax]
            elif reviewed is not None and reviewed.lmax is not None:
                tries = [reviewed.lmax]
            elif request.local_channel is not None or not request.scan_local_channels:
                tries = [base.lmax]
            else:
                # Never drop a channel that holds valence electrons.
                lowest = max([ell for _, ell, occ in base.valence if occ > 0.0] or [0])
                tries = list(range(base.lmax, min(lowest, base.lmax) - 1, -1))
            sources = [self._source(request, template, reviewed, lmax, shift) for lmax in tries]

        chosen = None       # (source, generated, candidates, selected)
        fallback = None     # ghost-free but with a knife-edge margin
        first = None        # highest lmax, kept for diagnosis when nothing passes
        for source in sources:
            try:
                generated = psgen(source, out=prefix, write=False,
                                  nonrelativistic=not request.relativistic)
            except FortranStop as exc:
                error = BackendError(f"FHI98PP stopped: {exc}")
                error.channel = getattr(exc, "channel", None)  # type: ignore[attr-defined]
                raise error from exc
            lmax = generated.lmax
            candidates = self._candidates(request, source, generated.cpi_text, lmax, out, prefix)
            attempt = (source, generated, candidates)
            first = first or attempt
            if request.local_channel is not None:
                chosen = attempt + (candidates[0],)
                break
            good = [x for x in candidates if comfortable(x)]
            if good:
                chosen = attempt + (choose_local_channel(
                    candidates, requested=None, reject_ghosts=False,
                    minimum_margin=MINIMUM_MARGIN_HARTREE),)
                break
            if fallback is None and any(x.passed for x in candidates):
                fallback = attempt + (choose_local_channel(
                    candidates, requested=None, reject_ghosts=False),)
        if chosen is None:
            chosen = fallback
        if chosen is None:
            source, generated, candidates = first
            selected = choose_local_channel(candidates, requested=None, reject_ghosts=False)
            chosen = (source, generated, candidates, selected)
        source, generated, candidates, selected = chosen
        if request.reject_ghosts and not selected.passed:
            summary = ", ".join(
                f"l={x.local_channel}:{'pass' if x.passed else 'fail'}" for x in candidates
            )
            raise GhostStateError(
                f"no ghost-free Kleinman-Bylander form at lmax={generated.lmax} ({summary})"
            )
        if selected.passed and not comfortable(selected):
            warnings.append(
                f"The ghost-free margin of the selected local channel is only "
                f"{selected.minimum_margin_hartree:.4f} Ha."
            )

        parsed = parse(source)
        result = GenerationResult(
            request=request, prefix=prefix, input_path=out / f"{prefix}.ini",
            ionic_charge=parsed.ionic_charge, reference_electrons=parsed.reference_electrons,
            warnings=warnings,
        )
        result.input_path.write_text(source, encoding="utf-8", newline="\n")
        for extension, text in (("dat", generated.dat_text), ("cpi", generated.cpi_text),
                                ("aep", generated.aep_text), ("fc", generated.fc_text)):
            if text is None:
                continue
            path = out / f"{prefix}.{extension}"
            path.write_text(text, encoding="utf-8", newline="\n")
            result.artifacts[extension] = path
        result.ae_energy_hartree = generated.etot_ae
        result.pseudo_energy_hartree = generated.etot_ps
        result.lmax = generated.lmax
        result.core_radius = generated.rcore
        result.reference_shift_ev = shift
        if unmatched:
            result.matched_to_ground_state = False
        result.channels = [
            ChannelSummary(ch.l, ch.scheme, ch.rc, ch.rmatch, ch.e_ae_ev, ch.e_ps_ev, ch.norm, ch.slope)
            for ch in generated.channels
        ]
        for ch in result.channels:
            if not (abs(ch.norm_test - 1.0) < 1.0e-3 and abs(ch.slope_test - 1.0) < 1.0e-2):
                warnings.append(
                    f"Channel {L_LABELS[ch.angular_momentum]}: norm test {ch.norm_test:.6f}, "
                    f"slope test {ch.slope_test:.6f}; inspect this potential before use."
                )
        result.local_channel_results = candidates
        result.selected_local_channel = selected.local_channel
        # A local channel that is not ghost-free was listed above; keep only the
        # reports of this lmax (an earlier, rejected lmax wrote the same names).
        reference_charge = parsed.ionic_charge - parsed.reference_electrons
        if abs(reference_charge) > 1.0e-10:
            warnings.append(
                f"Generating atomic reference charge {reference_charge:+g}; choose the electron "
                "count (full or neutralized core hole) explicitly in the many-electron calculation."
            )

        # Full report (logarithmic derivatives, kinetic-energy convergence) for the selection.
        # The covalent-radius table of the test program ends at Z = 100; beyond it the
        # diagnostic radius of the logarithmic derivatives has to be given.
        options = dict(prefix=prefix, lloc=selected.local_channel,
                       nonrelativistic=not request.relativistic,
                       rdiag=3.2 if parsed_charge(source) > 100.0 else 0.0)
        report = None
        try:
            report = pswatch(source, generated.cpi_text, generated.aep_text, **options)
        except FortranCrash:
            # The Fortran program dies in its kinetic-energy test when an occupied state of
            # the pseudo-atom has no bound solution; the rest of the report is still valid.
            try:
                report = pswatch(source, generated.cpi_text, generated.aep_text,
                                 kinetic=False, **options)
                warnings.append(
                    "Kinetic-energy convergence table omitted: a state of the pseudo-atom "
                    "has no bound solution (the Fortran test program crashes on this input)."
                )
            except FortranStop as exc:
                warnings.append(f"The full test report stopped: {exc}")
        except FortranStop as exc:
            warnings.append(f"The full test report stopped: {exc}")
        if report is not None:
            path = out / f"{prefix}.test"
            path.write_text(report.report, encoding="utf-8", newline="\n")
            result.artifacts["test"] = path
            for state in report.kinetic:
                # the tightest bracket of the table is 1 meV (printed 1.0E-03)
                row = next((r for r in state.rows if abs(r.bracket_eV - 1.0e-3) < 1.0e-6), None)
                if state.valid and row is not None and row.cutoff_Ry is not None and row.met:
                    result.kinetic_cutoff_ry[state.l] = float(row.cutoff_Ry)

        if request.formats:
            if request.xc not in XC_CODES and set(request.formats) - {OutputFormat.FHI}:
                raise ConfigurationError(
                    "conversion of an expert input with another functional code is limited to the FHI format"
                )
            try:
                written = convert_outputs(
                    result.artifacts["cpi"], out, element=request.element,
                    xc=request.xc if request.xc in XC_CODES else "pbe",
                    local_channel=selected.local_channel, valence=parsed.valence, prefix=prefix,
                    formats=tuple(request.formats), ae_energy_hartree=generated.etot_ae,
                    pseudo_energy_hartree=generated.etot_ps,
                    ionized_reference=abs(reference_charge) > 1.0e-8,
                    species_label=(_core_hole_species_label(request) if request.core_hole else None),
                    lmax=generated.lmax,
                    relativistic="scalar" if request.relativistic else "no",
                )
            except ConversionError as exc:
                raise BackendError(f"format conversion failed: {exc}") from exc
            if request.core_hole is not None:
                # A file named C_POTRE.DAT is always the ground-state carbon: for a core
                # hole only the files named after the species label (C-1s) are kept.
                for name in (OutputFormat.FHI.value, OutputFormat.PARSEC.value):
                    if name in written and f"{name}_species" in written:
                        Path(written.pop(name)).unlink(missing_ok=True)
            result.artifacts.update(written)
        return result
