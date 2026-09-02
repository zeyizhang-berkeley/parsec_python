from __future__ import annotations

import re
import shutil
from pathlib import Path

from ..errors import BackendError, ConfigurationError
from ..models import (
    ConstructionScheme,
    GenerationRequest,
    GenerationResult,
    LocalChannelResult,
    OutputFormat,
    PseudopotentialFamily,
)
from ..parsers import atom_ghosts, atom_reference_energies
from .common import choose_local_channel, run


SUPPORTED_FORMATS = {
    OutputFormat.PARSEC,
    OutputFormat.UPF,
    OutputFormat.PSP8,
    OutputFormat.SIESTA,
    OutputFormat.CPW2000,
}


class AtomBackend:
    """Adapter for the modern ATOM 6.x all-in-one generator."""

    def _validate(self, request: GenerationRequest) -> tuple[Path, Path]:
        if request.family is not PseudopotentialFamily.NCPP:
            raise ConfigurationError("ATOM adapter currently supports only NCPP")
        if request.scheme is not ConstructionScheme.TROULLIER_MARTINS:
            raise ConfigurationError("ATOM 6.x implements Troullier--Martins, not Hamann")
        if request.core_hole is not None:
            raise ConfigurationError(
                "automatic core-hole construction is supported only by the FHI98PP backend; "
                "ATOM core-hole semantics have no checked regression and are intentionally not guessed"
            )
        if request.input_file is None and request.xc != "ca":
            raise ConfigurationError(
                "ATOM's built-in default writer emits CA-LDA input only; for PBE use a reviewed "
                "--input-file (the online portal maintains a separate PBE input library)"
            )
        unknown = set(request.formats) - SUPPORTED_FORMATS
        if unknown:
            raise ConfigurationError(
                "ATOM backend cannot write: " + ", ".join(sorted(x.value for x in unknown))
            )
        if request.atom_executable is None:
            raise ConfigurationError("ATOM backend requires --atom-executable (atom_all*.exe)")
        executable = request.atom_executable.expanduser().resolve()
        if not executable.is_file():
            raise ConfigurationError(f"ATOM executable does not exist: {executable}")
        kb = request.atom_kb_executable
        if kb is None:
            kb_name = executable.name.replace("atom_all", "kb_conv", 1)
            kb = executable.with_name(kb_name)
        kb = kb.expanduser().resolve()
        if not kb.is_file():
            raise ConfigurationError(f"ATOM KB executable does not exist: {kb}")
        return executable, kb

    @staticmethod
    def _customize_default(text: str, request: GenerationRequest) -> str:
        lines = text.splitlines()
        xc_line = next((i for i, line in enumerate(lines) if re.search(r"\bc=\w+", line)), None)
        if xc_line is None:
            raise BackendError("cannot find exchange-correlation token in generated atom.dat")
        old = re.search(r"\bc=(\w+)", lines[xc_line])
        assert old is not None
        relativistic = old.group(1).lower().endswith("r")
        token = ("pb" if request.xc == "pbe" else "ca") + ("r" if relativistic else "")
        lines[xc_line] = re.sub(r"\bc=\w+", f"c={token}", lines[xc_line], count=1)

        if request.cutoff_radius is not None:
            if request.cutoff_radius <= 0:
                raise ConfigurationError("cutoff radius must be positive")
            count_index = xc_line + 2
            try:
                _, nvalence = (int(x) for x in lines[count_index].split()[:2])
            except (ValueError, IndexError) as exc:
                raise BackendError("cannot locate orbital counts in generated atom.dat") from exc
            radii_index = count_index + 1 + nvalence
            count = len(lines[radii_index].split())
            if count == 0:
                raise BackendError("cannot locate cutoff-radius record in generated atom.dat")
            lines[radii_index] = " ".join(
                f"{request.cutoff_radius:.8f}" for _ in range(count)
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _semilocal_metadata(psd_path: Path) -> tuple[int, float]:
        lines = psd_path.read_text(errors="replace").splitlines()
        if len(lines) < 4:
            raise BackendError(f"invalid ATOM semilocal file {psd_path}")
        fields = lines[3].split()
        try:
            number_down, number_up = (int(x) for x in fields[:2])
            ionic_charge = float(fields[5].replace("D", "E").replace("d", "e"))
        except (ValueError, IndexError) as exc:
            raise BackendError(f"cannot read angular channels from {psd_path}") from exc
        count = max(number_down, number_up)
        if count <= 0:
            raise BackendError(f"{psd_path} contains no angular channels")
        return count - 1, ionic_charge

    @staticmethod
    def _kb_input(local: int) -> str:
        # Change parameters -> change local -> l -> keep Fourier grid, then
        # retain every basis default.  Extra answers are harmless after EOF.
        return "\n".join(["y", "y", str(local), "n"] + ["n"] * 32) + "\n"

    def _run_kb(self, executable: Path, work: Path, prefix: str, local: int) -> LocalChannelResult:
        completed = run([str(executable)], cwd=work, stdin=self._kb_input(local))
        report = work / f"{prefix}.local-{local}.atom-kb.out"
        report.write_text(completed.stdout, encoding="utf-8")
        return LocalChannelResult(local, tuple(atom_ghosts(report)), report)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        executable, kb_executable = self._validate(request)
        out = request.output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        prefix = request.prefix or f"{request.element.lower()}_{request.xc}_tm_atom"
        input_path = out / "atom.dat"

        if request.input_file:
            source = request.input_file.expanduser().read_text(encoding="utf-8")
            input_path.write_text(source, encoding="utf-8")
        else:
            # Let ATOM's own periodic table construct the baseline, then make
            # explicit, reviewable changes and rerun from atom.dat.
            run([str(executable), request.element], cwd=out)
            source = self._customize_default(input_path.read_text(encoding="utf-8"), request)
            input_path.write_text(source, encoding="utf-8")

        completed = run([str(executable)], cwd=out)
        stdout_path = out / f"{prefix}.atom-all.stdout"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        atom_out = out / "atom.out"
        psd = out / "psd.pot"
        if not atom_out.is_file() or not psd.is_file():
            raise BackendError("ATOM did not create atom.out and psd.pot")
        atom_text = atom_out.read_text(errors="replace")
        if "potential not converged" in atom_text.lower():
            raise BackendError(
                "ATOM returned success but atom.out reports 'potential not converged'; "
                "adjust the expert atom.dat input instead of using stale artifacts"
            )
        ae_energy, pseudo_energy = atom_reference_energies(atom_out)
        lmax, ionic_charge = self._semilocal_metadata(psd)
        if request.local_channel is not None and not 0 <= request.local_channel <= lmax:
            raise ConfigurationError(
                f"local channel l={request.local_channel} is outside 0..{lmax}"
            )
        local_channels = (
            [request.local_channel]
            if request.local_channel is not None
            else (list(range(lmax + 1)) if request.scan_local_channels else [lmax])
        )
        candidates = [self._run_kb(kb_executable, out, prefix, local) for local in local_channels]
        selected = choose_local_channel(
            candidates, requested=request.local_channel, reject_ghosts=request.reject_ghosts
        )
        if selected.local_channel != local_channels[-1]:
            self._run_kb(kb_executable, out, prefix, selected.local_channel)

        result = GenerationResult(
            request=request, prefix=prefix, input_path=input_path,
            local_channel_results=candidates, selected_local_channel=selected.local_channel,
            ionic_charge=ionic_charge, reference_electrons=ionic_charge,
        )
        result.warnings.append(
            "ATOM built-in parameters are starting points; portal inputs and production potentials "
            "require independent transferability and cutoff validation."
        )
        result.ae_energy_hartree, result.pseudo_energy_hartree = ae_energy, pseudo_energy
        result.artifacts.update({"atom_out": atom_out, "semilocal": psd, "stdout": stdout_path})
        result.artifacts["test"] = selected.report_path

        patterns = {
            OutputFormat.PARSEC: f"{request.element}_POTRE.DAT",
            OutputFormat.UPF: f"{request.element}_*_TM.UPF",
            OutputFormat.PSP8: f"{request.element}.psp8",
            OutputFormat.SIESTA: f"{request.element}.psf",
            OutputFormat.CPW2000: f"{request.element}_POTKB_F.DAT",
        }
        for output_format in request.formats:
            matches = sorted(out.glob(patterns[output_format]))
            if not matches:
                raise BackendError(
                    f"ATOM did not create requested {output_format.value} output "
                    f"({patterns[output_format]})"
                )
            # A clean output directory has one match.  Prefer a GGA-named UPF
            # for PBE if a bootstrap LDA file is also present.
            if output_format is OutputFormat.UPF and request.xc == "pbe":
                preferred = [path for path in matches if "GGA" in path.name]
                if preferred:
                    matches = preferred
            result.artifacts[output_format.value] = matches[-1]
        return result
