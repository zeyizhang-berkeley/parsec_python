"""FHI98PP table to semilocal UPF v2, as ``fhi2upf_qe.x`` writes it.

This is an in-process port of the conversion done by Quantum ESPRESSO 7.5
(``upflib/read_fhi.f90`` and ``write_upf_new.f90``, driven by
``Dfhipp/tools/fhi2upf_qe.f90``).  What that program asks interactively
(element, functional, ``lmax lloc``, one ``nl occupation`` per channel) is
passed as arguments.  The arithmetic follows the Fortran statement by
statement and the numbers are printed with the same edit descriptors, so the
output has the same digits; with :data:`FHI2UPF_GENERATED` as the provenance
line it is the same file.

UPF conventions: radii in bohr, potentials in Rydberg (twice the table's
Hartree values), ``PP_NLCC`` the core density ``rho_core`` (the table's
``4*pi*rho_core`` divided by ``4*pi``), ``PP_CHI`` the reduced radial functions
and ``PP_RHOATOM`` their occupation-weighted squares.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ._fortran import correctly_rounded_log, format_es, format_list_directed, format_r2c
from .cpi import ANGULAR_LABELS, CpiTable, atomic_number
from .potre import ConversionError


UPF_VERSION = "2.0.1"
#: Provenance line of ``fhi2upf_qe.x``; pass it as ``generated`` to obtain that
#: program's file byte for byte.
FHI2UPF_GENERATED = "Generated using FHI98PP, converted with fhi2upf.x v.5.0.2"
#: Provenance line of this converter.
DEFAULT_GENERATED = "Generated using FHI98PP, converted with pp_generation.converters"
FHI2UPF_COMMENT = "Info: automatically converted from FHI format"
RELATIVISTIC_LABELS = {"scalar": "Scalar-Relativistic", "no": "Non-Relativistic"}
_FUNCTIONAL_RE = re.compile(r"[A-Za-z0-9+_ -]+")
# ``1.0E-6`` in read_fhi.f90 is a single-precision constant.
_PROJECTOR_TOLERANCE = 9.99999997475242707810e-07
_FOUR_PI = 4.0 * 3.14159265358979323846


@dataclass(frozen=True)
class UpfWavefunction:
    """One ``PP_CHI`` reference function."""

    label: str
    principal_n: int
    angular_momentum: int
    occupation: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class UpfProjector:
    """One ``PP_BETA`` projector with its ``PP_DIJ`` diagonal element."""

    label: str
    angular_momentum: int
    cutoff_radius_index: int
    cutoff_radius: float
    values: tuple[float, ...]
    dion: float
    # V_l in Rydberg, written as PP_VNL.l
    semilocal_potential: tuple[float, ...]


@dataclass(frozen=True)
class SemilocalUpf:
    """Everything ``fhi2upf_qe.x`` writes into a semilocal UPF v2 file."""

    element: str
    functional: str
    relativistic: str
    z_valence: float
    l_max: int
    l_local: int
    zmesh: float
    dx: float
    xmin: float
    rmax: float
    radius: tuple[float, ...]
    rab: tuple[float, ...]
    core_density: tuple[float, ...] | None
    local_potential: tuple[float, ...]
    projectors: tuple[UpfProjector, ...]
    wavefunctions: tuple[UpfWavefunction, ...]
    valence_density: tuple[float, ...]
    total_psenergy_ry: float | None = None
    total_aeenergy_ry: float | None = None
    generated: str = DEFAULT_GENERATED
    author: str = "unknown"
    date: str = "unknown"
    comment: str = FHI2UPF_COMMENT

    @property
    def mesh_size(self) -> int:
        return len(self.radius)


def xml_protect(text: str) -> str:
    """Quantum ESPRESSO's escaping of free text placed in an attribute."""

    table = str.maketrans({"&": "@", "<": "*", ">": "*", '"': "'"})
    return text.strip().translate(table)


def _simpson(count: int, function: Sequence[float], rab: Sequence[float]) -> float:
    """``simpson`` of upflib/simpsn.f90 over the first ``count`` mesh points."""

    total = 0.0
    for index in range(1, count - 1):
        # 4 for even Fortran indices, 2 for odd ones.
        factor = 4.0 if index % 2 == 1 else 2.0
        total = total + factor * function[index] * rab[index]
    third = 1.0 / 3.0
    last = count - 1
    if count % 2 == 1:
        return (total + function[0] * rab[0] + function[last] * rab[last]) * third
    return (
        total
        + function[0] * rab[0]
        - function[last - 2] * rab[last - 2] * 0.25
        + function[last - 1] * rab[last - 1]
        + function[last] * rab[last] * 1.25
    ) * third


def _projector_extent(
    potential: Sequence[float], local_potential: Sequence[float]
) -> int:
    """``kbeta``: number of mesh points that carry a projector."""

    mesh = len(potential)
    extent = mesh
    for index in range(mesh - 1, -1, -1):
        if abs(potential[index] - local_potential[index]) > _PROJECTOR_TOLERANCE:
            # One point beyond the last one with a potential difference.
            extent = index + 2
            break
    # Simpson integration wants an odd number of points.
    if extent % 2 == 0:
        extent += 1
    return min(mesh, extent)


def cpi_to_upf(
    table: CpiTable,
    *,
    element: str,
    functional: str,
    local_channel: int,
    labels: Sequence[str],
    occupations: Sequence[float],
    lmax: int | None = None,
    total_psenergy_ry: float | None = None,
    total_aeenergy_ry: float | None = None,
    relativistic: str = "scalar",
    generated: str = DEFAULT_GENERATED,
    author: str = "unknown",
    date: str = "unknown",
    comment: str = FHI2UPF_COMMENT,
) -> SemilocalUpf:
    """Convert a table to semilocal UPF data.

    ``element`` is the chemical symbol, ``functional`` the label written
    verbatim into the file (Quantum ESPRESSO reads ``PBE``, ``PZ`` and ``PW``),
    ``local_channel`` the local angular momentum.  ``labels`` and
    ``occupations`` give, for each channel l = 0..``lmax``, the reference
    state as ``nl`` (``"3s"``) and its occupation.  ``lmax`` defaults to the
    table's and may be lower to drop its highest channels.

    The table is converted as its ``.cpi`` file holds it, with 14 significant
    digits, so a table of full-precision arrays and the file written from it
    give the same UPF.

    The energies are optional and in Rydberg.  ``relativistic`` is
    ``"scalar"`` or ``"no"``; ``fhi2upf_qe.x`` writes ``"scalar"`` for every
    bare ``.cpi`` table.  ``generated``, ``author``, ``date`` and ``comment``
    are free provenance text.
    """

    lmax = table.lmax if lmax is None else lmax
    if not 0 <= lmax <= table.lmax:
        raise ConversionError(f"lmax={lmax} is outside the table's 0..{table.lmax}")
    if not 0 <= local_channel <= lmax:
        raise ConversionError(f"local channel l={local_channel} is outside 0..{lmax}")
    if len(labels) != lmax + 1 or len(occupations) != lmax + 1:
        raise ConversionError(
            f"need one label and one occupation for each of the {lmax + 1} channels"
        )
    z_nuclear = atomic_number(element)
    functional = functional.strip()
    if not _FUNCTIONAL_RE.fullmatch(functional):
        raise ConversionError(f"invalid UPF functional label {functional!r}")
    if relativistic not in RELATIVISTIC_LABELS:
        raise ConversionError(
            f"relativistic must be 'scalar' or 'no', not {relativistic!r}"
        )
    principal: list[int] = []
    for angular_momentum, label in enumerate(labels):
        expected = ANGULAR_LABELS[angular_momentum]
        # As fhi2upf, accept the unphysical label of an unoccupied channel
        # ("2d" when the valence shell is n=2).
        if len(label) != 2 or not label[0].isdigit() or label[1].lower() != expected:
            raise ConversionError(
                f"label {label!r} of channel l={angular_momentum} must look like "
                f"'{angular_momentum + 1}{expected}'"
            )
        principal.append(int(label[0]))
    occupations = tuple(float(value) for value in occupations)
    if any(not math.isfinite(value) or value < 0.0 for value in occupations):
        raise ConversionError("occupations must be finite and non-negative")

    table = table.as_written()
    mesh = table.mesh_size
    if mesh < 4:
        raise ConversionError(f"a radial mesh of {mesh} points cannot be integrated")
    radius = table.radius
    # read_fhi sets rab from log(amesh); fhi2upf_qe then replaces only the
    # header's dx by the value fitted to the two ends of the mesh.
    log_increment = correctly_rounded_log(table.mesh_increment)
    rab = tuple(value * log_increment for value in radius)
    rmax = radius[-1]
    zmesh = float(z_nuclear)
    dx = correctly_rounded_log(rmax / radius[0]) / float(mesh - 1)
    xmin = correctly_rounded_log(radius[0] * zmesh)

    local = table.channels[local_channel]
    local_potential = tuple(2.0 * value for value in local.potential)

    projectors: list[UpfProjector] = []
    for angular_momentum in range(lmax + 1):
        if angular_momentum == local_channel:
            continue
        channel = table.channels[angular_momentum]
        potential = tuple(2.0 * value for value in channel.potential)
        extent = _projector_extent(potential, local_potential)
        beta = [0.0] * mesh
        integrand = [0.0] * extent
        for index in range(extent):
            beta[index] = 2.0 * channel.wavefunction[index] * (
                channel.potential[index] - local.potential[index]
            )
            integrand[index] = channel.wavefunction[index] * beta[index]
        denominator = _simpson(extent, integrand, rab)
        if denominator == 0.0:
            raise ConversionError(
                f"the l={angular_momentum} potential equals the local potential; "
                "its Kleinman-Bylander projector vanishes"
            )
        projectors.append(
            UpfProjector(
                label=labels[angular_momentum],
                angular_momentum=angular_momentum,
                cutoff_radius_index=extent,
                cutoff_radius=radius[extent - 1],
                values=tuple(beta),
                dion=1.0 / denominator,
                semilocal_potential=potential,
            )
        )

    wavefunctions = tuple(
        UpfWavefunction(
            label=labels[angular_momentum],
            principal_n=principal[angular_momentum],
            angular_momentum=angular_momentum,
            occupation=occupations[angular_momentum],
            values=table.channels[angular_momentum].wavefunction,
        )
        for angular_momentum in range(lmax + 1)
    )
    density = [0.0] * mesh
    for wavefunction in wavefunctions:
        for index, value in enumerate(wavefunction.values):
            density[index] = density[index] + wavefunction.occupation * (value * value)

    return SemilocalUpf(
        element=element,
        functional=functional,
        relativistic=relativistic,
        z_valence=table.valence_charge,
        l_max=lmax - 1 if local_channel == lmax else lmax,
        l_local=local_channel,
        zmesh=zmesh,
        dx=dx,
        xmin=xmin,
        rmax=rmax,
        radius=radius,
        rab=rab,
        core_density=(
            None if table.core is None
            else tuple(value / _FOUR_PI for value in table.core.density)
        ),
        local_potential=local_potential,
        projectors=tuple(projectors),
        wavefunctions=wavefunctions,
        valence_density=tuple(density),
        total_psenergy_ry=total_psenergy_ry,
        total_aeenergy_ry=total_aeenergy_ry,
        generated=generated,
        author=author,
        date=date,
        comment=comment,
    )


def _values(parts: list[str], values: Sequence[float]) -> None:
    """Append an array as ``(1p3es24.15)`` records."""

    for start in range(0, len(values), 3):
        parts.append(
            "".join(format_es(value, 24, 15) for value in values[start : start + 3])
            + "\n"
        )


def format_upf(upf: SemilocalUpf) -> str:
    """The UPF v2 text, laid out as ``write_upf(schema='v2')`` writes it."""

    mesh = upf.mesh_size
    relativity = RELATIVISTIC_LABELS[upf.relativistic]
    # QE escapes free text only in the header; escaping it in PP_INFO as well
    # changes nothing for text without markup characters.
    generated = xml_protect(upf.generated)
    author = xml_protect(upf.author)
    date = xml_protect(upf.date)
    comment = xml_protect(upf.comment)
    parts = [
        f'<UPF version="{UPF_VERSION}">\n',
        "  <PP_INFO>\n",
        f"    {generated}\n",
        f"    Author: {author}\n",
        f"    Generation date: {date}\n",
        "    Pseudopotential type: SL\n",
        f"    Element: {upf.element}\n",
        f"    Functional: {upf.functional}\n",
        "    Suggested minimum cutoff for wavefunctions:   0. Ry\n",
        "    Suggested minimum cutoff for charge density:   0. Ry\n",
        f"    The Pseudo was generated with a {relativity} Calculation\n",
        "    L component and cutoff radius for Local Potential:"
        f"{upf.l_local:3d}{0.0:9.4f}\n",
        "    Valence configuration: \n",
        "    nl pn  l   occ       Rcut    Rcut US       E pseu\n",
    ]
    parts.extend(
        f"    {item.label:2s}{item.principal_n:3d}{item.angular_momentum:3d}"
        f"{item.occupation:6.2f}{0.0:11.3f}{0.0:11.3f}{0.0:13.6f}\n"
        for item in upf.wavefunctions
    )
    parts.append("    Generation configuration: not available.\n")
    if comment:
        parts.append(f"    Comment:  {comment}\n")
    parts.append("  </PP_INFO>\n")
    parts.append("    <!-- END OF HUMAN READABLE SECTION -->\n")

    # formats.annotate_upf_energies stores the energies with ten decimals and
    # puts the nonstandard all-electron attribute on a line of its own.
    psenergy = (
        format_r2c(0.0) if upf.total_psenergy_ry is None
        else f"{upf.total_psenergy_ry:.10f}"
    )
    aeenergy = (
        "" if upf.total_aeenergy_ry is None
        else f'\ntotal_aeenergy="{upf.total_aeenergy_ry:.10f}"'
    )
    zero = format_r2c(0.0)
    parts.append(
        f'  <PP_HEADER generated="{generated}" author="{author}" date="{date}"'
        f' comment="{comment}" element="{upf.element}"'
        f' pseudo_type="SL" relativistic="{upf.relativistic}"'
        ' is_ultrasoft="false" is_paw="false" is_coulomb="false" has_so="false"'
        ' has_wfc="false" has_gipaw="false" paw_as_gipaw="false"'
        f' core_correction="{"true" if upf.core_density is not None else "false"}"'
        f' with_metagga_info="false" functional="{upf.functional}"'
        f' z_valence="{format_r2c(upf.z_valence)}" total_psenergy="{psenergy}"'
        f' wfc_cutoff="{zero}" rho_cutoff="{zero}" l_max="{upf.l_max}"'
        f' l_max_rho="0" l_local="{upf.l_local}" mesh_size="{mesh}"'
        f' number_of_wfc="{len(upf.wavefunctions)}"'
        f' number_of_proj="{len(upf.projectors)}"{aeenergy}/>\n'
    )

    parts.append(
        f'  <PP_MESH mesh="{mesh}" dx="{format_r2c(upf.dx)}"'
        f' xmin="{format_r2c(upf.xmin)}" rmax="{format_r2c(upf.rmax)}"'
        f' zmesh="{format_r2c(upf.zmesh)}">\n'
    )
    parts.append("    <PP_R>\n")
    _values(parts, upf.radius)
    parts.append("    </PP_R>\n    <PP_RAB>\n")
    _values(parts, upf.rab)
    parts.append("    </PP_RAB>\n  </PP_MESH>\n")
    if upf.core_density is not None:
        parts.append(f'  <PP_NLCC size="{mesh}">\n')
        _values(parts, upf.core_density)
        parts.append("  </PP_NLCC>\n")
    parts.append(f'  <PP_LOCAL size="{mesh}">\n')
    _values(parts, upf.local_potential)
    parts.append("  </PP_LOCAL>\n  <PP_SEMILOCAL>\n")
    for projector in upf.projectors:
        tag = f"PP_VNL.{projector.angular_momentum}"
        parts.append(f'    <{tag} l="{projector.angular_momentum}">\n')
        _values(parts, projector.semilocal_potential)
        parts.append(f"    </{tag}>\n")
    parts.append("  </PP_SEMILOCAL>\n  <PP_NONLOCAL>\n")
    for index, projector in enumerate(upf.projectors, start=1):
        parts.append(
            f'    <PP_BETA.{index} index="{index}" label="{projector.label}"'
            f' angular_momentum="{projector.angular_momentum}"'
            f' cutoff_radius_index="{projector.cutoff_radius_index}"'
            f' cutoff_radius="{format_r2c(projector.cutoff_radius)}"'
            f' ultrasoft_cutoff_radius="{zero}">\n'
        )
        _values(parts, projector.values)
        parts.append(f"    </PP_BETA.{index}>\n")
    count = len(upf.projectors)
    parts.append(f'    <PP_DIJ columns="{count}" rows="{count}">\n')
    for row in range(count):
        parts.append(
            "".join(
                " " + format_list_directed(
                    upf.projectors[row].dion if column == row else 0.0
                )
                for column in range(count)
            )
            + "\n"
        )
    parts.append("    </PP_DIJ>\n  </PP_NONLOCAL>\n  <PP_PSWFC>\n")
    for index, item in enumerate(upf.wavefunctions, start=1):
        parts.append(
            f'    <PP_CHI.{index} size="{mesh}" index="{index}" label="{item.label}"'
            f' l="{item.angular_momentum}"'
            f' occupation="{format_r2c(item.occupation)}"'
            # QE writes n only when it exceeds l, so "2d" carries none.
            + (
                f' n="{item.principal_n}"'
                if item.principal_n > item.angular_momentum else ""
            )
            + ">\n"
        )
        _values(parts, item.values)
        parts.append(f"    </PP_CHI.{index}>\n")
    parts.append(f'  </PP_PSWFC>\n  <PP_RHOATOM size="{mesh}">\n')
    _values(parts, upf.valence_density)
    parts.append("  </PP_RHOATOM>\n</UPF>\n")
    return "".join(parts)


def write_upf(path: str | os.PathLike[str], upf: SemilocalUpf) -> None:
    """Write semilocal UPF data to a UPF v2 file."""

    Path(path).write_text(format_upf(upf), encoding="utf-8", newline="\n")
