from __future__ import annotations

import re
from dataclasses import dataclass
import math
from pathlib import Path

from .errors import ConfigurationError
from .models import ConstructionScheme, CoreHole

# FHI98PP iexc codes.  "ca" is Ceperley--Alder in the Perdew--Zunger 1981
# parametrization, the functional behind ``Correlation_Type ca`` of the solver;
# "pw92" is the Perdew--Wang 1992 parametrization (FHI98PP's own LDA default).
XC_CODES = {"pbe": 6, "ca": 3, "pw92": 8}
L_LABELS = "spdfgh"
ORBITAL_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+([-+0-9.eEdD]+)")
# FHI98PP reads a reference energy below this magnitude as "use the default".
REFERENCE_ENERGY_THRESHOLD_EV = 0.5e-2


def _real(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class ParsedInput:
    nuclear_charge: float
    ncore: int
    nvalence: int
    xc_code: int
    orbitals: tuple[tuple[int, int, float], ...]
    lmax: int
    default_scheme: str
    core_radius: float = 0.0
    # Optional per-channel records: (l, cutoff radius, reference energy in eV, scheme).
    channel_lines: tuple[tuple[int, float, float, str], ...] = ()

    @property
    def ionic_charge(self) -> float:
        return self.nuclear_charge - sum(x[2] for x in self.orbitals[: self.ncore])

    @property
    def valence(self) -> tuple[tuple[int, int, float], ...]:
        return self.orbitals[self.ncore :]

    @property
    def reference_electrons(self) -> float:
        return sum(x[2] for x in self.valence)


def parse(text: str) -> ParsedInput:
    records = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not records:
        raise ConfigurationError("empty FHI98PP input")
    head = records[0].split(":", 1)[0].split()
    if len(head) < 5:
        raise ConfigurationError("FHI98PP header must contain Z, nc, nv, iexc, and rnlc")
    z, nc, nv, xc_code = float(head[0]), int(head[1]), int(head[2]), int(head[3])
    if z <= 0 or nc < 0 or nv <= 0:
        raise ConfigurationError("invalid nuclear charge or orbital counts in FHI98PP input")
    if len(records) < 2 + nc + nv:
        raise ConfigurationError("FHI98PP input ends before all orbital/configuration records")
    orbitals = []
    for line in records[1 : 1 + nc + nv]:
        match = ORBITAL_RE.match(line)
        if not match:
            raise ConfigurationError(f"cannot parse FHI98PP orbital record: {line!r}")
        n, ell = int(match.group(1)), int(match.group(2))
        occupation = float(match.group(3).replace("D", "E").replace("d", "e"))
        if n <= 0 or not 0 <= ell < n or not math.isfinite(occupation) or occupation < 0:
            raise ConfigurationError(f"invalid orbital record: {line!r}")
        if occupation > 2 * (2 * ell + 1) + 1e-10:
            raise ConfigurationError(f"occupation exceeds shell capacity: {line!r}")
        orbitals.append((n, ell, occupation))
    config = records[1 + nc + nv].split(":", 1)[0].split()
    if len(config) < 2:
        raise ConfigurationError("missing FHI98PP lmax/construction record")
    lmax, default_scheme = int(config[0]), config[1].lower()
    if lmax < 0 or lmax >= len(L_LABELS):
        raise ConfigurationError(f"unsupported generated angular momentum lmax={lmax}")
    channel_lines = []
    for line in records[2 + nc + nv :]:
        fields = line.split(":", 1)[0].split()
        try:
            channel_lines.append((int(fields[0]), _real(fields[1]), _real(fields[2]), fields[3].lower()))
        except (IndexError, ValueError):
            break  # FHI98PP stops reading channel records at the first one it cannot read
    return ParsedInput(
        z, nc, nv, xc_code, tuple(orbitals), lmax, default_scheme,
        core_radius=_real(head[4]), channel_lines=tuple(channel_lines),
    )


def find_template(root: Path, element: str, scheme: ConstructionScheme) -> Path:
    is_tm = scheme is ConstructionScheme.TROULLIER_MARTINS
    directory = root / "bin" / "Elements" / ("TM" if is_tm else "Ham")
    suffix = ":tm.ini" if is_tm else ":h.ini"
    matches = sorted(directory.glob(f"*-{element}{suffix}"))
    if not matches:
        raise ConfigurationError(
            f"no FHI98PP {scheme.value} default template for {element}; "
            "provide --input-file or select the ATOM backend"
        )
    return matches[0]


def customize(text: str, *, xc: str, scheme: ConstructionScheme,
              core_hole: CoreHole | None, cutoff_radius: float | None,
              lmax: int | None = None,
              channel_radii: dict[int, float] | None = None,
              reference_energies_ev: dict[int, float] | None = None,
              core_radius: float | None = None,
              reference_shift_ev: float = 0.0) -> str:
    """Apply the request to a template and return the FHI98PP input text.

    ``reference_shift_ev`` is added to every explicit reference energy of the
    template (used for core holes, where all valence levels move down);
    ``reference_energies_ev`` sets a channel's reference energy outright.
    Apart from the functional code, the template text is left as it is when
    none of the optional controls is used.
    """

    lines = text.splitlines()
    head_data, sep, comment = lines[0].partition(":")
    fields = head_data.split()
    if xc not in XC_CODES:
        raise ConfigurationError(f"unsupported XC {xc!r}; choose one of {', '.join(XC_CODES)}")
    fields[3] = str(XC_CODES[xc])
    if core_radius is not None:
        if not math.isfinite(core_radius) or core_radius < 0:
            raise ConfigurationError("core radius must be finite and non-negative")
        fields[4] = f"{core_radius:.4f}"
    lines[0] = "  ".join(fields) + ((" :" + comment) if sep else "")

    parsed = parse("\n".join(lines))
    if core_hole:
        match = re.fullmatch(r"(\d+)([spdfgh])", core_hole.shell.lower())
        if not match:
            raise ConfigurationError("core hole must look like '1s' or '2p'")
        if not math.isfinite(core_hole.removed_electrons) or core_hole.removed_electrons <= 0:
            raise ConfigurationError("removed core-hole charge must be finite and positive")
        target = (int(match.group(1)), L_LABELS.index(match.group(2)))
        found = False
        for index in range(parsed.ncore):
            line_index = 1 + index
            orbital = ORBITAL_RE.match(lines[line_index])
            assert orbital is not None
            if (int(orbital.group(1)), int(orbital.group(2))) == target:
                occupation = float(orbital.group(3).replace("D", "E")) - core_hole.removed_electrons
                if occupation < -1e-12:
                    raise ConfigurationError(f"hole charge exceeds occupation of {core_hole.shell}")
                occupation = max(0.0, occupation)
                tail = lines[line_index][orbital.end():]
                lines[line_index] = f"    {target[0]}  {target[1]}  {occupation:.8f}" + tail
                found = True
                break
        if not found:
            raise ConfigurationError(f"{core_hole.shell} is not a core orbital in the selected template")

    scheme_token = "t" if scheme is ConstructionScheme.TROULLIER_MARTINS else "h"
    config_index = 1 + parsed.ncore + parsed.nvalence
    config = lines[config_index].split(":", 1)[0].split()
    config[1] = scheme_token
    lines[config_index] = f"{config[0]}  {config[1]} : lmax  s_pp_def"

    if lmax is not None:
        if not 0 <= lmax <= 4:
            raise ConfigurationError("lmax must lie between 0 and 4")
        config[0] = str(lmax)
        lines[config_index] = f"{config[0]}  {config[1]} : lmax  s_pp_def"
    final_lmax = int(config[0])

    rebuild = (
        lmax is not None or bool(channel_radii) or bool(reference_energies_ev)
        or abs(reference_shift_ev) > 0.0
    )
    if cutoff_radius is not None:
        if not math.isfinite(cutoff_radius) or cutoff_radius <= 0:
            raise ConfigurationError("cutoff radius must be finite and positive")
        channels = {ell: (cutoff_radius, 0.0) for ell in range(final_lmax + 1)}
    elif rebuild:
        channels = {ell: (radius, energy) for ell, radius, energy, _ in parsed.channel_lines}
    else:
        return "\n".join(lines) + "\n"

    for ell, radius in (channel_radii or {}).items():
        if not math.isfinite(radius) or radius < 0:
            raise ConfigurationError("channel cutoff radii must be finite and non-negative")
        channels[ell] = (radius, channels.get(ell, (0.0, 0.0))[1])
    if abs(reference_shift_ev) > 0.0:
        channels = {
            ell: (radius, energy + reference_shift_ev
                  if abs(energy) >= REFERENCE_ENERGY_THRESHOLD_EV else energy)
            for ell, (radius, energy) in channels.items()
        }
    for ell, energy in (reference_energies_ev or {}).items():
        channels[ell] = (channels.get(ell, (0.0, 0.0))[0], energy)
    lines = lines[: config_index + 1]
    lines.extend(
        f"{ell}  {radius:.8f}  {energy:.4f}  {scheme_token}" if rebuild
        else f"{ell}  {radius:.8f}  0.00  {scheme_token}"
        for ell, (radius, energy) in sorted(channels.items()) if ell <= final_lmax
    )
    return "\n".join(lines) + "\n"
