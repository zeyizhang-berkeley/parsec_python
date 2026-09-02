from __future__ import annotations

import re
from pathlib import Path

from .errors import BackendError
from .models import GhostChannel, GhostStatus


ENERGY_RE = re.compile(r"^\s*total energy\s+([-+0-9.EDed]+)\s*$", re.MULTILINE)
ATOM_ENERGY_RE = re.compile(
    r"^\s*total energy\s*=\s*([-+0-9.EDed]+)\s*$", re.MULTILINE
)
CHANNEL_RE = re.compile(r"analysis of kb potentials:\s*([spdfgh]) waves", re.I)
RESULT_RE = re.compile(
    r"\*\s*(no ghost|one or more ghosts|undetermined|illdefined)[^\n]*", re.I
)
VALUE_RE = re.compile(
    r"^\s*(kb energy|local potential groundstate|dto\. 1st excited state|reference energy)"
    r"\s+([-+0-9.EDed]+)",
    re.I | re.MULTILINE,
)
ATOM_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(?:(\d+)/2\s+)?"
    r"([-+0-9.EDed]+)\s+([-+0-9.EDed]+)\s+([-+0-9.EDed]+)\s+(-?1)\s*$",
    re.MULTILINE,
)
EV_PER_HARTREE = 27.211386245988


def energies(path: Path) -> tuple[float, float]:
    values = [float(x.replace("D", "E").replace("d", "e")) for x in ENERGY_RE.findall(path.read_text(errors="replace"))]
    if len(values) < 2:
        raise BackendError(f"{path} does not contain all-electron and pseudoatom energies")
    return values[0], values[1]


def atom_reference_energies(path: Path) -> tuple[float, float]:
    """Return ATOM's first AE and pseudo energies in Hartree (ATOM prints Ry)."""

    values = [
        float(x.replace("D", "E").replace("d", "e")) / 2.0
        for x in ATOM_ENERGY_RE.findall(path.read_text(errors="replace"))
    ]
    if len(values) < 2:
        raise BackendError(f"{path} does not contain ATOM AE/pseudo total energies")
    return values[0], values[1]


def fhi_ghosts(path: Path) -> list[GhostChannel]:
    """Parse FHI98PP's Gonze analysis, preserving non-pass outcomes."""

    text = path.read_text(errors="replace")
    channels = list(CHANNEL_RE.finditer(text))
    results: list[GhostChannel] = []
    for index, channel in enumerate(channels):
        end = channels[index + 1].start() if index + 1 < len(channels) else len(text)
        match = RESULT_RE.search(text, channel.end(), end)
        if match:
            label = channel.group(1).lower()
            message = " ".join(match.group(0).split())
            normalized = match.group(1).lower()
            status = {
                "no ghost": GhostStatus.PASS,
                "one or more ghosts": GhostStatus.GHOST,
                "undetermined": GhostStatus.INDETERMINATE,
                "illdefined": GhostStatus.ILL_DEFINED,
            }[normalized]
            values = {
                name.lower(): float(value.replace("D", "E").replace("d", "e")) / EV_PER_HARTREE
                for name, value in VALUE_RE.findall(text[channel.end() : end])
            }
            ekb = values.get("kb energy")
            eloc0 = values.get("local potential groundstate")
            eloc1 = values.get("dto. 1st excited state")
            eref = values.get("reference energy")
            margin = None
            if status is GhostStatus.PASS and None not in (ekb, eloc0, eref):
                if ekb is not None and ekb > 0 and eloc1 is not None:
                    margin = min(eref - eloc0, eloc1 - eref)
                else:
                    margin = eloc0 - eref
            results.append(
                GhostChannel(
                    "spdfgh".index(label), label, status, message, margin,
                    eref, eloc0, eloc1, None,
                )
            )
    if channels and len(results) != len(channels):
        raise BackendError(f"{path} has an incomplete FHI98PP ghost-state analysis")
    return results


def atom_ghosts(path: Path) -> list[GhostChannel]:
    """Re-evaluate ATOM's printed Gonze criterion for each nonlocal channel."""

    text = path.read_text(errors="replace")
    marker = text.rfind("Ghost State Test (X. Gonze et al)")
    if marker < 0:
        raise BackendError(f"{path} contains no ATOM ghost-state table")
    section = text[marker:]
    rows = ATOM_ROW_RE.findall(section)
    results: list[GhostChannel] = []
    for l_text, j_text, ev0_text, ev1_text, true_text, sign_text in rows:
        ell = int(l_text)
        ev0 = float(ev0_text.replace("D", "E").replace("d", "e"))
        ev1 = float(ev1_text.replace("D", "E").replace("d", "e"))
        eref = float(true_text.replace("D", "E").replace("d", "e"))
        sign = int(sign_text)
        tolerance = 1.0e-6 if eref > 0 else 0.0
        threshold = ev0 + tolerance if sign < 0 else ev1 + tolerance
        margin = threshold - eref
        status = GhostStatus.PASS if margin >= 0 else GhostStatus.GHOST
        j_label = f", j={j_text}/2" if j_text else ""
        results.append(
            GhostChannel(
                ell, "spdfgh"[ell], status,
                f"ATOM Gonze criterion for l={ell}{j_label}: margin={margin:.6g} Ha",
                margin, eref, ev0, ev1, sign,
            )
        )
    if "WARNING:     GHOST STATE" in section and not any(x.is_ghost for x in results):
        raise BackendError(f"{path} reports a ghost but its diagnostic table could not be reconciled")
    return results


# Backward-compatible import for early callers.
ghosts = fhi_ghosts
