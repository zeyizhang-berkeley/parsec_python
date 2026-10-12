from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .backends import AtomBackend, Fhi98ppBackend, GeneratorBackend
from .errors import ConfigurationError
from .models import BackendName, GenerationRequest, GenerationResult


ELEMENTS = {
    symbol
    for symbol in (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
        "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
        "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
        "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
        "Lv Ts Og"
    ).split()
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(result: GenerationResult) -> Path:
    path = result.request.output_dir.expanduser().resolve() / f"{result.prefix}.report.json"
    request = result.request
    output_root = request.output_dir.expanduser().resolve()

    def portable_path(value: Path) -> str:
        """Use a package-relative path when an artifact lives in its output."""

        resolved = value.expanduser().resolve()
        try:
            return str(resolved.relative_to(output_root))
        except ValueError:
            return str(resolved)

    payload = {
        "schema_version": 1,
        "element": request.element,
        "prefix": result.prefix,
        "backend": request.backend.value,
        "family": request.family.value,
        "xc": request.xc,
        "scheme": request.scheme.value,
        "core_hole": asdict(request.core_hole) if request.core_hole else None,
        "ionic_charge": result.ionic_charge,
        "reference_electrons": result.reference_electrons,
        "reference_charge": (
            result.ionic_charge - result.reference_electrons
            if result.ionic_charge is not None and result.reference_electrons is not None
            else None
        ),
        "ae_energy_hartree": result.ae_energy_hartree,
        "pseudo_energy_hartree": result.pseudo_energy_hartree,
        "ghost_free": result.ghost_free,
        "selected_local_channel": result.selected_local_channel,
        "local_channel_results": [
            {
                "local_channel": candidate.local_channel,
                "passed": candidate.passed,
                "minimum_margin_hartree": candidate.minimum_margin_hartree,
                "report_path": portable_path(candidate.report_path),
                "channels": [
                    {**asdict(channel), "status": channel.status.value}
                    for channel in candidate.channels
                ],
            }
            for candidate in result.local_channel_results
        ],
        "warnings": result.warnings,
        "artifacts": {
            key: {"path": portable_path(value), "sha256": _hash(value)}
            for key, value in result.artifacts.items()
            if value.is_file()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def generate(request: GenerationRequest) -> GenerationResult:
    element = request.element.strip()
    if not element or len(element) > 2 or not element.isalpha():
        raise ConfigurationError(f"invalid element symbol {request.element!r}")
    if element != element[0].upper() + element[1:].lower():
        raise ConfigurationError(
            f"element symbol must use canonical capitalization: {element[0].upper() + element[1:].lower()}"
        )
    if element not in ELEMENTS:
        raise ConfigurationError(f"unknown chemical element {element!r}")
    backends: dict[BackendName, GeneratorBackend] = {
        BackendName.FHI98PP: Fhi98ppBackend(),
        BackendName.ATOM: AtomBackend(),
    }
    result = backends[request.backend].generate(request)
    result.artifacts["report"] = _write_report(result)
    return result
