"""Build the pseudopotential libraries ``NCPPs`` and ``CoreHole_NCPPs``.

One *family* is an element and a functional: the ground-state potential and
one potential per core shell with an electron removed.  The rules are:

1. Ground state: the element template (with the reviewed overrides), the
   highest ``lmax`` not above the template's that has a ghost-free
   Kleinman--Bylander local channel, then the channel with the largest worst
   margin.
2. Core holes: the same ``lmax``, the same cutoff radii and the same local
   channel as the ground state, so that initial- and final-state potentials
   of a binding-energy calculation differ only by the hole.  Explicit
   reference energies of unbound channels move with the valence levels.
3. If a member of the family cannot be constructed at these radii, one
   channel's radius is moved by whole mesh points for the whole family.
4. A core hole that is not ghost-free with the ground-state channel takes its
   own best channel and is marked ``unmatched``.

``python -m pp_generation library --help`` runs it from the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import BackendError, GhostStateError, PseudopotentialError
from .fhi_input import L_LABELS, parse
from .models import (
    BackendName,
    ConstructionScheme,
    CoreHole,
    GenerationRequest,
    GenerationResult,
    OutputFormat,
)
from .templates import atomic_number, available_elements, core_shells, template_text

PACKAGE_DIR = Path(__file__).resolve().parent
XC_DIRECTORIES = {"pbe": "PBE", "ca": "CA", "pw92": "PW92"}
PARSEC_XC = {"pbe": "pbe", "ca": "ca", "pw92": "pl"}
DEFAULT_FORMATS = (OutputFormat.FHI, OutputFormat.UPF, OutputFormat.PARSEC)
PRODUCT_FOLDERS = {
    OutputFormat.FHI.value: "FHI", "fhi_species": "FHI",
    OutputFormat.UPF.value: "UPF",
    OutputFormat.PARSEC.value: "PARSEC", "parsec_species": "PARSEC",
    OutputFormat.ABINIT.value: "ABINIT",
}
MESH_RATIO = 1.0247          # FHI98PP radial mesh, r(i+1)/r(i)
SNAP = 1.0003                # a request this far above a mesh point selects that point
NUDGES = (1, -1, 2, -2)
MANIFEST_COLUMNS = (
    "Z", "element", "xc", "state", "status", "lmax", "local_channel", "minimum_margin_Ha",
    "ionic_charge", "reference_electrons", "net_charge_for_full_core_hole",
    "ae_energy_Ha", "pseudo_energy_Ha", "atomic_energy_correction_Ry",
    "radius_s", "radius_p", "radius_d", "radius_f", "core_radius", "reference_shift_eV",
    "ecut_1meV_Ry", "matched_to_ground_state", "directory", "note",
)


@dataclass
class Entry:
    """One row of a manifest."""

    Z: int
    element: str
    xc: str
    state: str                       # "ground" or the core shell, e.g. "2p"
    status: str = "failed"           # ok | unmatched | failed
    lmax: int | None = None
    local_channel: str = ""
    minimum_margin_Ha: float | None = None
    ionic_charge: float | None = None
    reference_electrons: float | None = None
    net_charge_for_full_core_hole: float | None = None
    ae_energy_Ha: float | None = None
    pseudo_energy_Ha: float | None = None
    atomic_energy_correction_Ry: float | None = None
    radii: dict[int, float] = field(default_factory=dict)
    core_radius: float | None = None
    reference_shift_eV: float = 0.0
    ecut_1meV_Ry: float | None = None
    matched_to_ground_state: bool | None = None
    directory: str = ""
    note: str = ""

    def row(self) -> dict[str, object]:
        data = asdict(self)
        radii = data.pop("radii")
        for ell in range(4):
            data[f"radius_{L_LABELS[ell]}"] = radii.get(ell)
        return {
            key: ("" if data.get(key) is None else
                  (f"{data[key]:.10g}" if isinstance(data[key], float) else data[key]))
            for key in MANIFEST_COLUMNS
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrange(result: GenerationResult, target: Path, *, core_hole: bool) -> None:
    """Move the products out of ``diagnostics`` into one folder per format."""

    diagnostics = result.request.output_dir
    moved: dict[str, Path] = {}
    for name, folder in PRODUCT_FOLDERS.items():
        path = result.artifacts.get(name)
        if path is None or not Path(path).is_file():
            continue
        # A core-hole directory keeps only the files named after the species
        # label (C-1s_POTRE.DAT), never a second "C_POTRE.DAT".
        if core_hole and name in (OutputFormat.FHI.value, OutputFormat.PARSEC.value) \
                and f"{name}_species" in result.artifacts:
            Path(path).unlink()
            continue
        destination = target / folder / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)
        moved[name] = destination
    # The table in FHI/ is the generator's .cpi; the frozen-core density is only an
    # input for frozen-core atom runs.  Neither is kept twice in a library.
    if "fhi" in moved or "fhi_species" in moved:
        Path(result.artifacts.get("cpi", diagnostics / "-")).unlink(missing_ok=True)
    Path(result.artifacts.get("fc", diagnostics / "-")).unlink(missing_ok=True)
    report = result.artifacts.get("report")
    if report is not None and Path(report).is_file():
        payload = json.loads(Path(report).read_text(encoding="utf-8"))
        artifacts = {
            key: value for key, value in payload.get("artifacts", {}).items()
            if (diagnostics / value["path"]).is_file()
        }
        for name, destination in moved.items():
            artifacts[name] = {
                "path": f"../{destination.parent.name}/{destination.name}",
                "sha256": _sha256(destination),
            }
        payload["artifacts"] = artifacts
        Path(report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                                newline="\n")
    snippet = _parsec_snippet(result)
    if snippet and (target / "PARSEC").is_dir():
        (target / "PARSEC" / "parsec_atom_type.in").write_text(
            snippet, encoding="utf-8", newline="\n")


def _species_label(request: GenerationRequest) -> str:
    if request.core_hole is None:
        return request.element
    charge = request.core_hole.removed_electrons
    suffix = "" if abs(charge - 1.0) <= 1.0e-12 else f"-{charge:g}"
    return f"{request.element}-{request.core_hole.shell.lower()}{suffix}"


def _parsec_snippet(result: GenerationResult) -> str:
    """The ``Atom_Type`` block that uses this potential in ``parsec.in``."""

    request = result.request
    if result.selected_local_channel is None or result.ae_energy_hartree is None:
        return ""
    correction = 2.0 * (result.ae_energy_hartree - result.pseudo_energy_hartree)
    lines = [
        f"Atom_Type: {_species_label(request)}",
        f"Element_Symbol: {request.element}",
        "Pseudopotential_Format: martins_new",
        f"Local_Component: {L_LABELS[result.selected_local_channel]}",
        f"Atomic_Energy_Correction: {correction:.6f} Ry",
        "",
        f"# Correlation_Type: {PARSEC_XC.get(request.xc, request.xc)}",
    ]
    if request.core_hole is not None:
        charge = request.core_hole.removed_electrons
        lines.append(
            f"# One atom of this type with the photoelectron removed: add {charge:g} e to "
            "Net_Charges (full core hole);"
        )
        lines.append("# leave Net_Charges unchanged for a neutralized (excited-electron) core hole.")
    return "\n".join(lines) + "\n"


def _entry(result: GenerationResult, state: str, directory: Path, root: Path) -> Entry:
    request = result.request
    selected = next(
        (x for x in result.local_channel_results if x.local_channel == result.selected_local_channel),
        None,
    )
    entry = Entry(Z=atomic_number(request.element), element=request.element, xc=request.xc, state=state)
    entry.status = "ok" if result.ghost_free else "failed"
    entry.lmax = result.lmax
    entry.local_channel = (
        "" if result.selected_local_channel is None else L_LABELS[result.selected_local_channel]
    )
    entry.minimum_margin_Ha = None if selected is None else selected.minimum_margin_hartree
    entry.ionic_charge = result.ionic_charge
    entry.reference_electrons = result.reference_electrons
    if request.core_hole is not None:
        entry.net_charge_for_full_core_hole = request.core_hole.removed_electrons
    entry.ae_energy_Ha = result.ae_energy_hartree
    entry.pseudo_energy_Ha = result.pseudo_energy_hartree
    if result.ae_energy_hartree is not None and result.pseudo_energy_hartree is not None:
        entry.atomic_energy_correction_Ry = 2.0 * (result.ae_energy_hartree - result.pseudo_energy_hartree)
    entry.radii = {x.angular_momentum: x.cutoff_radius for x in result.channels}
    entry.core_radius = result.core_radius
    entry.reference_shift_eV = result.reference_shift_ev
    if result.kinetic_cutoff_ry:
        entry.ecut_1meV_Ry = max(result.kinetic_cutoff_ry.values())
    entry.directory = directory.relative_to(root).as_posix()
    return entry


def _request(element: str, xc: str, out: Path, formats, backend: BackendName, **controls) -> GenerationRequest:
    return GenerationRequest(
        element=element, output_dir=out / "diagnostics", backend=backend, xc=xc,
        scheme=ConstructionScheme.TROULLIER_MARTINS, formats=tuple(formats), **controls,
    )


def _clean(directory: Path) -> None:
    if directory.is_dir():
        shutil.rmtree(directory)


def build_family(
    element: str, xc: str, root: Path, *, core_holes: bool = True,
    formats=DEFAULT_FORMATS, backend: BackendName = BackendName.PYFHI98PP,
    shells: tuple[str, ...] | None = None,
) -> list[Entry]:
    """Generate the ground state and the core holes of one element and functional."""

    from .generator import generate  # late import: worker processes

    z = atomic_number(element)
    ground_dir = root / "NCPPs" / XC_DIRECTORIES[xc] / element
    hole_root = root / "CoreHole_NCPPs" / XC_DIRECTORIES[xc] / element
    wanted = core_shells(parse(template_text(element))) if core_holes else []
    if shells is not None:
        wanted = [x for x in wanted if x in shells]

    class GroundFailure(Exception):
        """The ground state itself could not be generated (cause attached)."""

    troublesome: list[str] = []     # core shells that could not be constructed so far

    def first_line(exc: BaseException) -> str:
        return (str(exc).splitlines() or [type(exc).__name__])[0]

    def attempt(overrides: dict[int, float], strict: bool) -> list[Entry]:
        """Generate the family with the given radius overrides.

        ``strict``: a core hole that cannot be constructed raises (the caller then
        tries other radii); otherwise it becomes a failed row and the rest is kept.
        """

        _clean(ground_dir)
        _clean(hole_root)
        try:
            ground = generate(_request(element, xc, ground_dir, formats, backend,
                                       channel_radii=dict(overrides) or None))
        except PseudopotentialError as exc:
            _clean(ground_dir)
            raise GroundFailure() from exc
        results = [("ground", ground, ground_dir)]
        failed: list[tuple[str, str]] = []
        # Shells that broke an earlier attempt come first: a trial radius that does
        # not cure them is then abandoned after one potential instead of a dozen.
        for shell in sorted(wanted, key=lambda name: name not in troublesome):
            target = hole_root / f"{element}-{shell}"
            # Same input records as the ground state; the backend takes lmax and local
            # channel from ``ground`` and falls back to the hole's own choice only when
            # that form is not ghost-free.
            try:
                hole = generate(_request(
                    element, xc, target, formats, backend, core_hole=CoreHole(shell, 1.0),
                    channel_radii=dict(overrides) or None, ground_state=ground,
                    strict_match=strict,
                ))
            except GhostStateError as exc:
                _clean(target)
                failed.append((shell, f"no ghost-free form: {first_line(exc)}"))
                continue
            except BackendError as exc:
                _clean(target)
                if strict:
                    if shell not in troublesome:
                        troublesome.append(shell)
                    raise
                failed.append((shell, f"cannot be constructed: {first_line(exc)}"))
                continue
            results.append((shell, hole, target))
        entries: list[Entry] = []
        wanted_radii = {x.angular_momentum: x.cutoff_radius for x in ground.channels}
        for state, result, target in results:
            _arrange(result, target, core_hole=state != "ground")
            entry = _entry(result, state, target, root)
            if state != "ground":
                same_radii = all(abs(radius - wanted_radii.get(ell, radius)) < 1.0e-9
                                 for ell, radius in entry.radii.items())
                entry.matched_to_ground_state = bool(result.matched_to_ground_state) and same_radii
                if not entry.matched_to_ground_state:
                    entry.status = "unmatched" if entry.status == "ok" else entry.status
                    entry.note = ("own lmax and local channel: the ground-state form is not "
                                  "ghost-free for this hole")
            entries.append(entry)
        entries.extend(Entry(Z=z, element=element, xc=xc, state=shell, note=note)
                       for shell, note in failed)
        return entries

    def ground_failed(exc: BaseException | None) -> list[Entry]:
        kind = "no ghost-free form" if isinstance(exc, GhostStateError) else "cannot be constructed"
        return [Entry(Z=z, element=element, xc=xc, state="ground",
                      note=f"{kind}: {first_line(exc) if exc else ''}")]

    # Rule 1 and 2: template radii.
    try:
        return attempt({}, strict=True)
    except GroundFailure as exc:
        failure = exc.__cause__
        if isinstance(failure, GhostStateError):
            return ground_failed(failure)
    except BackendError as exc:
        failure = exc

    # Rule 3: a member cannot be constructed.  Move one channel's radius by whole mesh
    # points for the whole family; the channel that failed first, then the others.
    template = parse(template_text(element))
    base = {ell: radius for ell, radius, _, _ in template.channel_lines if radius > 0.0}
    try:
        probe = generate(_request(element, xc, ground_dir, (), backend))
        base.update({x.angular_momentum: x.cutoff_radius for x in probe.channels})
    except PseudopotentialError:
        pass
    _clean(ground_dir)
    channel = getattr(failure, "channel", None)
    order = ([channel] if channel in base else []) + [ell for ell in sorted(base) if ell != channel]
    for ell in order:
        for step in NUDGES:
            overrides = {ell: base[ell] * MESH_RATIO ** step * SNAP}
            try:
                entries = attempt(overrides, strict=True)
            except (GroundFailure, PseudopotentialError):
                continue
            note = (f"{L_LABELS[ell]} radius moved {step:+d} mesh point(s) from the default "
                    f"{base[ell]:.4f} bohr: a family member cannot be constructed there")
            for entry in entries:
                entry.note = f"{entry.note}; {note}".lstrip("; ")
            return entries

    # No radius works for everything: keep whatever can be generated at the template radii.
    try:
        return attempt({}, strict=False)
    except GroundFailure as exc:
        return ground_failed(exc.__cause__)


def _family_task(args) -> list[Entry]:
    element, xc, root, core_holes, formats, backend = args
    try:
        return build_family(element, xc, Path(root), core_holes=core_holes,
                            formats=formats, backend=BackendName(backend))
    except Exception:  # a crash in one family must not stop the build
        return [Entry(Z=atomic_number(element), element=element, xc=xc, state="ground",
                      note="internal error: " + traceback.format_exc().strip().splitlines()[-1])]


def write_manifest(path: Path, entries: list[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for entry in sorted(entries, key=lambda x: (x.xc, x.Z, x.state != "ground", x.state)):
            writer.writerow(entry.row())


def build_library(
    root: Path, elements: list[str], functionals: list[str], *, core_holes: bool = True,
    formats=DEFAULT_FORMATS, jobs: int = 1, backend: BackendName = BackendName.PYFHI98PP,
    progress=print,
) -> list[Entry]:
    """Generate every family and write the two manifests."""

    if jobs > 1:
        # One thread per worker: the linear algebra here is tiny, and a threaded BLAS in
        # every process only makes the workers fight for the processors.
        for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
    # Heavy atoms have the most core shells: start them first so the pool stays busy.
    tasks = [(element, xc, str(root), core_holes, tuple(formats), backend.value)
             for element in sorted(elements, key=atomic_number, reverse=True)
             for xc in functionals]
    entries: list[Entry] = []
    started = time.time()
    if jobs <= 1:
        for index, task in enumerate(tasks, 1):
            family = _family_task(task)
            entries.extend(family)
            progress(_progress_line(index, len(tasks), task, family, started))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_family_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                family = future.result()
                entries.extend(family)
                progress(_progress_line(index, len(tasks), futures[future], family, started))
    write_manifest(root / "NCPPs" / "manifest.csv", [x for x in entries if x.state == "ground"])
    if core_holes:
        write_manifest(root / "CoreHole_NCPPs" / "manifest.csv",
                       [x for x in entries if x.state != "ground"])
    return entries


def _progress_line(index, total, task, family, started) -> str:
    good = sum(1 for x in family if x.status == "ok")
    return (f"[{index}/{total}] {task[0]:2s} {task[1]:4s} {good}/{len(family)} ok"
            f"  ({time.time() - started:.0f} s)"
            + "".join(f"\n      {x.state}: {x.status} {x.note}" for x in family if x.status != "ok"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pp-generate library",
        description="Build the ground-state and core-hole pseudopotential libraries",
    )
    parser.add_argument("--root", type=Path, default=PACKAGE_DIR,
                        help="folder that receives NCPPs/ and CoreHole_NCPPs/ (default: the package)")
    parser.add_argument("--elements", nargs="+", default=["all"],
                        help="chemical symbols, or 'all' for every element with a template")
    parser.add_argument("--xc", nargs="+", default=["pbe", "ca"], choices=tuple(XC_DIRECTORIES))
    parser.add_argument("--no-core-holes", action="store_true")
    parser.add_argument("--format", dest="formats", action="append",
                        choices=("fhi", "upf", "parsec", "abinit"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--backend", choices=tuple(x.value for x in BackendName),
                        default=BackendName.PYFHI98PP.value)
    args = parser.parse_args(argv)
    elements = available_elements() if args.elements == ["all"] else args.elements
    formats = tuple(OutputFormat(x) for x in args.formats) if args.formats else DEFAULT_FORMATS
    entries = build_library(
        args.root, elements, args.xc, core_holes=not args.no_core_holes, formats=formats,
        jobs=args.jobs, backend=BackendName(args.backend),
    )
    failed = [x for x in entries if x.status == "failed"]
    print(f"{len(entries) - len(failed)} potentials written, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
