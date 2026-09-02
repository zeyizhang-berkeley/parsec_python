from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import PseudopotentialError
from .generator import generate
from .models import (
    BackendName,
    ConstructionScheme,
    CoreHole,
    GenerationRequest,
    OutputFormat,
    PseudopotentialFamily,
)


FORMAT_ALIASES = {
    "fhi": OutputFormat.FHI,
    "fhipp": OutputFormat.FHI,
    "parsec": OutputFormat.PARSEC,
    "potre": OutputFormat.PARSEC,
    "upf": OutputFormat.UPF,
    "psp8": OutputFormat.PSP8,
    "siesta": OutputFormat.SIESTA,
    "psf": OutputFormat.SIESTA,
    "cpw2000": OutputFormat.CPW2000,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="pp-generate",
        description="Generate and ghost-check norm-conserving pseudopotentials",
    )
    result.add_argument("element", help="chemical symbol, for example Si")
    result.add_argument("-o", "--output-dir", type=Path, default=Path("pp-output"))
    result.add_argument("--backend", choices=tuple(x.value for x in BackendName), default="fhi98pp")
    result.add_argument("--family", choices=("ncpp",), default="ncpp")
    result.add_argument("--xc", choices=("pbe", "ca"), default="pbe")
    result.add_argument("--scheme", choices=("tm", "hamann"), default="tm")
    result.add_argument("--core-hole", metavar="SHELL", help="core shell such as 1s or 2p")
    result.add_argument("--hole-charge", type=float, default=1.0, help="electrons removed from the core shell")
    result.add_argument("--cutoff-radius", type=float, help="common channel cutoff in bohr")
    result.add_argument("--input-file", type=Path, help="expert-mode backend-native input")
    result.add_argument(
        "--format", dest="formats", action="append", choices=tuple(FORMAT_ALIASES),
        help="repeat for multiple formats; DAT aliases are 'fhi' and 'parsec'",
    )
    result.add_argument("--prefix")
    result.add_argument("--local-channel", type=int, help="force KB local angular channel")
    result.add_argument("--no-local-scan", action="store_true", help="test only the default/highest channel")
    result.add_argument("--allow-ghosts", action="store_true", help="retain failed output for diagnosis")
    result.add_argument("--fhi-root", type=Path)
    result.add_argument("--atom-executable", type=Path, help="path to atom_all*.exe")
    result.add_argument("--atom-kb-executable", type=Path, help="path to kb_conv*.exe")
    result.add_argument("--qe-converter", type=Path)
    result.add_argument("--potre-converter", type=Path)
    result.add_argument("--debug", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    element = args.element[0].upper() + args.element[1:].lower() if args.element else ""
    backend = BackendName(args.backend)
    if args.formats:
        formats = tuple(dict.fromkeys(FORMAT_ALIASES[item] for item in args.formats))
    else:
        formats = (OutputFormat.FHI,) if backend is BackendName.FHI98PP else (OutputFormat.PARSEC,)
    request = GenerationRequest(
        element=element, output_dir=args.output_dir, backend=backend,
        family=PseudopotentialFamily(args.family), xc=args.xc,
        scheme=ConstructionScheme(args.scheme),
        core_hole=CoreHole(args.core_hole, args.hole_charge) if args.core_hole else None,
        cutoff_radius=args.cutoff_radius, input_file=args.input_file, formats=formats,
        prefix=args.prefix, fhi_root=args.fhi_root, qe_converter=args.qe_converter,
        potre_converter=args.potre_converter, local_channel=args.local_channel,
        scan_local_channels=not args.no_local_scan, reject_ghosts=not args.allow_ghosts,
        atom_executable=args.atom_executable, atom_kb_executable=args.atom_kb_executable,
    )
    try:
        result = generate(request)
    except (PseudopotentialError, ValueError, OSError) as exc:
        print(f"pp-generate: error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 2
    print(f"generated {result.prefix}: ghost-free={result.ghost_free}")
    print(f"  selected local channel: l={result.selected_local_channel}")
    for candidate in result.local_channel_results:
        margin = candidate.minimum_margin_hartree
        rendered_margin = "n/a" if margin is None else f"{margin:.6g} Ha"
        print(
            f"  local l={candidate.local_channel}: "
            f"{'PASS' if candidate.passed else 'FAIL'} (minimum margin {rendered_margin})"
        )
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
