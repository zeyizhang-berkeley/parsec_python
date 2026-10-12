from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import PseudopotentialError
from .fhi_input import L_LABELS, XC_CODES
from .models import (
    DEFAULT_FORMATS,
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
    "abinit": OutputFormat.ABINIT,
    "parsec": OutputFormat.PARSEC,
    "potre": OutputFormat.PARSEC,
    "upf": OutputFormat.UPF,
    "psp8": OutputFormat.PSP8,
    "siesta": OutputFormat.SIESTA,
    "psf": OutputFormat.SIESTA,
    "cpw2000": OutputFormat.CPW2000,
}


def _channel_values(items: list[str] | None, what: str) -> dict[int, float] | None:
    """Parse repeated ``s=2.5`` style options into ``{l: value}``."""

    if not items:
        return None
    values: dict[int, float] = {}
    for item in items:
        label, sep, text = item.partition("=")
        label = label.strip().lower()
        if not sep or len(label) != 1 or label not in L_LABELS[:4]:
            raise ValueError(f"{what} must look like s=2.5 (channels s, p, d, f), got {item!r}")
        try:
            values[L_LABELS.index(label)] = float(text)
        except ValueError:
            raise ValueError(f"{what}: {text!r} is not a number") from None
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="pp-generate",
        description="Generate and ghost-check norm-conserving pseudopotentials",
        epilog="Other commands:  pp-generate gui   (graphical front end);  "
               "pp-generate library --help   (build the libraries NCPPs and CoreHole_NCPPs)",
    )
    result.add_argument("element", help="chemical symbol, for example Si")
    result.add_argument("-o", "--output-dir", type=Path, default=Path("pp-output"))
    result.add_argument(
        "--backend", choices=tuple(x.value for x in BackendName), default=BackendName.PYFHI98PP.value,
        help="pyfhi98pp = built-in Python port of FHI98PP (default); fhi98pp, atom = external programs",
    )
    result.add_argument("--family", choices=("ncpp",), default="ncpp")
    result.add_argument(
        "--xc", choices=tuple(XC_CODES), default="pbe",
        help="pbe; ca = Ceperley-Alder/Perdew-Zunger LDA (Correlation_Type ca); pw92 = Perdew-Wang LDA",
    )
    result.add_argument("--scheme", choices=("tm", "hamann"), default="tm")
    result.add_argument("--core-hole", metavar="SHELL", help="core shell such as 1s or 2p")
    result.add_argument("--hole-charge", type=float, default=1.0, help="electrons removed from the core shell")
    result.add_argument("--cutoff-radius", type=float, help="common channel cutoff in bohr")
    result.add_argument("--radius", action="append", metavar="L=BOHR",
                        help="cutoff radius of one channel, e.g. --radius s=2.5; repeatable")
    result.add_argument("--reference-energy", action="append", metavar="L=EV",
                        help="reference energy of an unoccupied channel in eV; repeatable")
    result.add_argument("--core-radius", type=float,
                        help="radius of the nonlinear core correction in bohr; 0 switches it off")
    result.add_argument("--lmax", type=int, help="highest generated angular momentum")
    result.add_argument("--variant", help="template variant, e.g. 3d for the Ga template with 3d valence")
    result.add_argument("--non-relativistic", action="store_true",
                        help="non-relativistic atom instead of scalar-relativistic")
    result.add_argument("--no-reference-shift", action="store_true",
                        help="core holes: keep the template reference energies unshifted")
    result.add_argument("--no-ground-state-match", action="store_true",
                        help="core holes: choose lmax and local channel independently of the "
                             "ground-state potential")
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
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "gui":
        from .gui import run_gui

        return run_gui()
    if argv and argv[0] == "library":
        from .library import main as library_main

        return library_main(argv[1:])
    args = parser().parse_args(argv)
    element = args.element[0].upper() + args.element[1:].lower() if args.element else ""
    backend = BackendName(args.backend)
    if args.formats:
        formats = tuple(dict.fromkeys(FORMAT_ALIASES[item] for item in args.formats))
    else:
        formats = DEFAULT_FORMATS[backend]
    try:
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
            variant=args.variant, relativistic=not args.non_relativistic, lmax=args.lmax,
            channel_radii=_channel_values(args.radius, "--radius"),
            reference_energies_ev=_channel_values(args.reference_energy, "--reference-energy"),
            core_radius=args.core_radius,
            shift_reference_energies=not args.no_reference_shift,
            match_ground_state=not args.no_ground_state_match,
        )
        from .generator import generate

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
    if result.ae_energy_hartree is not None and result.pseudo_energy_hartree is not None:
        print(f"  all-electron energy {result.ae_energy_hartree:.6f} Ha, "
              f"pseudo-atom energy {result.pseudo_energy_hartree:.6f} Ha")
        print("  Atomic_Energy_Correction "
              f"{2.0 * (result.ae_energy_hartree - result.pseudo_energy_hartree):.6f} Ry")
    for text in result.warnings:
        print(f"  note: {text}")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
