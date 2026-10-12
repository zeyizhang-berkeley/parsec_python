"""Write one generated FHI98PP potential in the formats the solvers read."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..errors import ConfigurationError
from .cpi import ANGULAR_LABELS, CpiTable, atomic_number, format_cpi, format_fhi, read_cpi
from .potre import (
    convert_upf_data,
    infer_parsec_xc,
    parse_upf_text,
    write_validated_parsec,
)
from .upf import DEFAULT_GENERATED, cpi_to_upf, format_upf


@dataclass(frozen=True)
class XcLabels:
    """How one exchange-correlation functional is named in each format."""

    name: str
    description: str
    fhi98pp_iexc: int
    upf_functional: str
    parsec_code: str
    abinit_pspxc: int


#: Functionals the conversion chain carries through every format.  ``ca`` is
#: Ceperley-Alder in the Perdew-Zunger 1981 parametrization, which is what
#: ``Correlation_Type ca`` means in PARSEC; ``pw92`` is the Perdew-Wang 1992
#: parametrization of the same data.  The UPF labels are Quantum ESPRESSO's
#: short names, the POTRE codes PARSEC's, ``abinit_pspxc`` abinit's ``ixc``.
XC_LABELS = {
    labels.name: labels
    for labels in (
        XcLabels("pbe", "GGA Perdew/Burke/Ernzerhof (1996)", 6, "PBE", "pb", 11),
        XcLabels("ca", "LDA Ceperley/Alder Perdew/Zunger (1980)", 3, "PZ", "ca", 2),
        XcLabels("pw92", "LDA Ceperley/Alder Perdew/Wang (1991)", 8, "PW", "pl", 7),
    )
}
OUTPUT_KINDS = ("fhi", "abinit", "upf", "parsec")


def xc_labels(xc: str) -> XcLabels:
    """Format labels of ``"pbe"``, ``"ca"`` or ``"pw92"``; nothing is guessed."""

    try:
        return XC_LABELS[xc]
    except (KeyError, TypeError):
        raise ConfigurationError(
            f"unsupported XC {xc!r}; choose one of " + ", ".join(XC_LABELS)
        ) from None


def valence_channels(
    valence: Sequence[tuple[int, int, float]], lmax: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Reference label and occupation of each channel l = 0..``lmax``.

    ``valence`` lists the valence states as ``(n, l, occupation)``, as
    :attr:`pp_generation.fhi_input.ParsedInput.valence` does.  A channel
    without a valence state is unoccupied and takes the largest valence
    ``n``; of two states with the same ``l`` the later one labels the channel.
    This is the rule by which the FHI98PP backend answers ``fhi2upf``.
    """

    if not valence:
        raise ConfigurationError("the valence configuration is empty")
    if not 0 <= lmax < len(ANGULAR_LABELS):
        raise ConfigurationError(f"unsupported lmax={lmax}")
    by_l = {int(ell): (int(n), float(occupation)) for n, ell, occupation in valence}
    default_n = max(int(n) for n, _, _ in valence)
    channels = [by_l.get(ell, (default_n, 0.0)) for ell in range(lmax + 1)]
    return (
        tuple(f"{n}{ANGULAR_LABELS[ell]}" for ell, (n, _) in enumerate(channels)),
        tuple(occupation for _, occupation in channels),
    )


def _output_kinds(formats: Iterable[object]) -> tuple[str, ...]:
    kinds = []
    for item in formats:
        # OutputFormat members are accepted through their string value.
        kind = str(getattr(item, "value", item)).lower()
        if kind not in OUTPUT_KINDS:
            raise ConfigurationError(
                f"cannot convert to {kind!r}; choose from " + ", ".join(OUTPUT_KINDS)
            )
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def convert_outputs(
    potential: CpiTable | str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    element: str,
    xc: str,
    local_channel: int,
    valence: Sequence[tuple[int, int, float]],
    prefix: str,
    formats: Iterable[object] = ("fhi", "upf", "parsec"),
    ae_energy_hartree: float | None = None,
    pseudo_energy_hartree: float | None = None,
    ionized_reference: bool = False,
    species_label: str | None = None,
    lmax: int | None = None,
    relativistic: str = "scalar",
    generated: str = DEFAULT_GENERATED,
    abinit_title: str | None = None,
    abinit_date: int = 0,
) -> dict[str, Path]:
    """Convert one generated potential and write the requested files.

    ``potential`` is a :class:`CpiTable` or the path of a ``.cpi``/``.fhi``
    file.  ``element`` is the chemical symbol, ``xc`` one of ``"pbe"``,
    ``"ca"`` (Perdew-Zunger LDA) and ``"pw92"``, ``local_channel`` the local
    angular momentum and ``valence`` the valence states ``(n, l, occupation)``
    of the generating configuration.  The energies of the all-electron atom
    and of the pseudo atom are FHI98PP's, in Hartree; the UPF stores them in
    Rydberg.  ``ionized_reference`` must be true when the valence occupations
    intentionally do not add up to the valence charge (a core-hole or ionic
    reference).  ``lmax`` defaults to the table's highest channel.

    ``formats`` selects any of ``"fhi"``, ``"abinit"``, ``"upf"`` and
    ``"parsec"``, as strings or as the
    :class:`pp_generation.models.OutputFormat` members with these values.  The
    files are named as the FHI98PP backend names them and are returned under
    these keys:

    ``fhi``
        ``<element>_FHIPP.DAT``, the ``.cpi`` table.
    ``abinit``
        ``<prefix>.fhi``, the table behind an abinit ``pspcod`` 6 header.
    ``upf``
        ``<prefix>.UPF``, semilocal UPF v2 as ``fhi2upf_qe.x`` writes it.
    ``parsec``
        ``<element>_POTRE.DAT``, converted from that UPF whether or not the
        UPF itself is written.
    ``fhi_species``, ``parsec_species``
        copies named ``<species_label>_FHIPP.DAT`` and
        ``<species_label>_POTRE.DAT`` when ``species_label`` (such as
        ``"C-1s"``) is given.

    Every file derives from the table as its ``.cpi`` text holds it (14
    significant digits), so the UPF and POTRE files are the ones that
    converting the written ``<element>_FHIPP.DAT`` gives.  Everything is
    converted before the first file is written, so a refused conversion
    leaves no partial output.
    """

    labels = xc_labels(xc)
    kinds = _output_kinds(formats)
    table = potential if isinstance(potential, CpiTable) else read_cpi(potential)
    lmax = table.lmax if lmax is None else lmax
    channel_labels, occupations = valence_channels(valence, lmax)
    directory = Path(output_dir)

    contents: dict[str, tuple[Path, str]] = {}
    if "fhi" in kinds:
        text = format_cpi(table)
        contents["fhi"] = (directory / f"{element}_FHIPP.DAT", text)
        if species_label is not None:
            contents["fhi_species"] = (directory / f"{species_label}_FHIPP.DAT", text)
    if "abinit" in kinds:
        title = abinit_title or (
            f"{element}, fhi98PP : {labels.description}, l= {local_channel} local"
        )
        contents["abinit"] = (
            directory / f"{prefix}.fhi",
            format_fhi(
                table, zatom=float(atomic_number(element)), pspxc=labels.abinit_pspxc,
                lloc=local_channel, title=title, pspdat=abinit_date,
            ),
        )

    parsec_data = None
    if "upf" in kinds or "parsec" in kinds:
        upf_name = f"{prefix}.UPF"
        upf_text = format_upf(
            cpi_to_upf(
                table,
                element=element,
                functional=labels.upf_functional,
                local_channel=local_channel,
                labels=channel_labels,
                occupations=occupations,
                lmax=lmax,
                total_psenergy_ry=(
                    None if pseudo_energy_hartree is None else 2.0 * pseudo_energy_hartree
                ),
                total_aeenergy_ry=(
                    None if ae_energy_hartree is None else 2.0 * ae_energy_hartree
                ),
                relativistic=relativistic,
                generated=generated,
            )
        )
        if "upf" in kinds:
            contents["upf"] = (directory / upf_name, upf_text)
        if "parsec" in kinds:
            if infer_parsec_xc(labels.upf_functional) != labels.parsec_code:
                raise ConfigurationError(
                    f"UPF label {labels.upf_functional!r} and PARSEC code "
                    f"{labels.parsec_code!r} of XC {xc!r} disagree"
                )
            parsec_data = convert_upf_data(
                parse_upf_text(upf_text, upf_name, ionized_reference),
                upf_name,
                labels.parsec_code,
            )

    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for kind, (path, text) in contents.items():
        path.write_text(text, encoding="latin-1", newline="\n")
        written[kind] = path
    if parsec_data is not None:
        written["parsec"] = directory / f"{element}_POTRE.DAT"
        write_validated_parsec(written["parsec"], parsec_data)
        if species_label is not None:
            written["parsec_species"] = directory / f"{species_label}_POTRE.DAT"
            write_validated_parsec(written["parsec_species"], parsec_data)
    return written
