"""FHI98PP potential tables: the ``.cpi`` file and its abinit ``.fhi`` form.

A ``.cpi`` file is what FHI98PP's ``psgen`` assembles for one generated
norm-conserving pseudopotential::

    Zval  number_of_channels
    (ten records of unused fit parameters)
    mmax  amesh                      one block per channel l = 0, 1, ...
    i  r(i)  u_l(r(i))  V_l(r(i))    mmax records
    r(i)  4*pi*rho_core  d/dr  d2/dr2    mmax records, only with a nonlinear
                                         core correction

Radii are in bohr on the logarithmic mesh ``r(i+1) = amesh*r(i)``,
``u_l = r*R_l`` is the reduced radial pseudo wavefunction and ``V_l`` the ionic
semilocal potential in Hartree.  The abinit ``.fhi`` form (``pspcod`` 6) is
the same body after a seven-line header.

Everything is held as tuples of Python floats; no numpy is needed, and numpy
arrays are accepted wherever a sequence of numbers is.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from ._fortran import format_e, parse_real
from .potre import ConversionError


ANGULAR_LABELS = "spdfgh"
# FHI98PP writes the mesh index with I4.
MAX_MESH_SIZE = 9999
# QE's reader stops at seven channels.
MAX_CHANNELS = 7
UNUSED_FIT_PARAMETERS = (
    "  0.0000    0.0000    0.0000   0.0000\n"
    + "  0.0000    .00e+00   .00e+00\n" * 9
)
ABINIT_HEADER_LINES = 7
ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og"
).split()


def atomic_number(element: str) -> int:
    """Nuclear charge of a chemical symbol such as ``"Si"``."""

    try:
        return ELEMENT_SYMBOLS.index(element) + 1
    except ValueError:
        raise ConversionError(f"unknown chemical symbol {element!r}") from None


def _floats(values: Iterable[float], label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ConversionError(f"{label} contains a non-finite value")
    return result


@dataclass(frozen=True)
class CpiChannel:
    """One angular channel: ``u_l(r)`` and the ionic ``V_l(r)`` in Hartree."""

    angular_momentum: int
    wavefunction: tuple[float, ...]
    potential: tuple[float, ...]

    def __post_init__(self) -> None:
        label = f"l={self.angular_momentum}"
        object.__setattr__(
            self, "wavefunction", _floats(self.wavefunction, f"{label} wavefunction")
        )
        object.__setattr__(
            self, "potential", _floats(self.potential, f"{label} potential")
        )


@dataclass(frozen=True)
class CoreDensity:
    """Model core density of a nonlinear core correction.

    ``density`` is ``4*pi*rho_core(r)`` in electrons/bohr**3, the other two
    its first and second radial derivatives, as FHI98PP tabulates them.
    """

    density: tuple[float, ...]
    first_derivative: tuple[float, ...]
    second_derivative: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("density", "first_derivative", "second_derivative"):
            object.__setattr__(self, name, _floats(getattr(self, name), f"core {name}"))


@dataclass(frozen=True)
class AbinitHeader:
    """The seven-line abinit header of a ``.fhi`` file (``pspcod`` 6)."""

    title: str
    zatom: float
    zion: float
    pspdat: int
    pspcod: int
    pspxc: int
    lmax: int
    lloc: int
    mmax: int
    r2well: float
    rchrg: float
    fchrg: float
    qchrg: float


@dataclass(frozen=True)
class CpiTable:
    """One FHI98PP pseudopotential: valence charge, mesh, channels, core."""

    valence_charge: float
    mesh_increment: float
    radius: tuple[float, ...]
    channels: tuple[CpiChannel, ...]
    core: CoreDensity | None = None
    abinit_header: AbinitHeader | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence_charge", float(self.valence_charge))
        object.__setattr__(self, "mesh_increment", float(self.mesh_increment))
        object.__setattr__(self, "radius", _floats(self.radius, "radial mesh"))
        object.__setattr__(self, "channels", tuple(self.channels))
        if not (math.isfinite(self.valence_charge) and self.valence_charge > 0.0):
            raise ConversionError(f"invalid valence charge {self.valence_charge!r}")
        if not (math.isfinite(self.mesh_increment) and self.mesh_increment > 1.0):
            raise ConversionError(f"invalid mesh increment {self.mesh_increment!r}")
        mesh = len(self.radius)
        if not 2 <= mesh <= MAX_MESH_SIZE:
            raise ConversionError(f"radial mesh has {mesh} points; need 2 to {MAX_MESH_SIZE}")
        if self.radius[0] <= 0.0 or any(
            right <= left for left, right in zip(self.radius, self.radius[1:])
        ):
            raise ConversionError("radial mesh must be positive and strictly increasing")
        if not 1 <= len(self.channels) <= MAX_CHANNELS:
            raise ConversionError(
                f"{len(self.channels)} angular channels; need 1 to {MAX_CHANNELS}"
            )
        for expected, channel in enumerate(self.channels):
            if channel.angular_momentum != expected:
                raise ConversionError(
                    "channels must be consecutive from l=0; found "
                    f"l={channel.angular_momentum} at position {expected}"
                )
            if len(channel.wavefunction) != mesh or len(channel.potential) != mesh:
                raise ConversionError(f"l={expected} does not have {mesh} mesh values")
        if self.core is not None and any(
            len(values) != mesh
            for values in (
                self.core.density, self.core.first_derivative, self.core.second_derivative,
            )
        ):
            raise ConversionError(f"core density does not have {mesh} mesh values")

    @classmethod
    def from_arrays(
        cls,
        valence_charge: float,
        radius: Sequence[float],
        wavefunctions: Sequence[Sequence[float]],
        potentials: Sequence[Sequence[float]],
        *,
        core_density: Sequence[float] | None = None,
        core_first_derivative: Sequence[float] | None = None,
        core_second_derivative: Sequence[float] | None = None,
        mesh_increment: float | None = None,
    ) -> CpiTable:
        """Build a table from per-channel arrays ordered l = 0, 1, ...

        ``potentials`` are in Hartree and ``core_density`` is
        ``4*pi*rho_core``; all three core arrays are required together.
        ``mesh_increment`` defaults to ``radius[1]/radius[0]``, which is what
        FHI98PP prints.
        """

        if len(wavefunctions) != len(potentials):
            raise ConversionError("need one wavefunction per potential channel")
        radius = _floats(radius, "radial mesh")
        if len(radius) < 2:
            raise ConversionError("radial mesh needs at least two points")
        core_arrays = (core_density, core_first_derivative, core_second_derivative)
        if any(values is None for values in core_arrays) != all(
            values is None for values in core_arrays
        ):
            raise ConversionError(
                "a core correction needs the density and both of its derivatives"
            )
        return cls(
            valence_charge=valence_charge,
            mesh_increment=(
                radius[1] / radius[0] if mesh_increment is None else mesh_increment
            ),
            radius=radius,
            channels=tuple(
                CpiChannel(angular_momentum, wavefunction, potential)
                for angular_momentum, (wavefunction, potential) in enumerate(
                    zip(wavefunctions, potentials)
                )
            ),
            core=None if core_density is None else CoreDensity(*core_arrays),
        )

    def as_written(self) -> CpiTable:
        """This table with every number as its ``.cpi`` file holds it.

        The file keeps 14 significant digits, and those are what ``fhi2upf``
        and every later conversion start from.
        """
        return replace(parse_cpi(format_cpi(self)), abinit_header=self.abinit_header)

    @property
    def lmax(self) -> int:
        """Highest tabulated angular momentum."""
        return len(self.channels) - 1

    @property
    def mesh_size(self) -> int:
        return len(self.radius)

    @property
    def has_core_correction(self) -> bool:
        return self.core is not None


def _tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def _reals(line: str, count: int) -> list[float] | None:
    tokens = _tokens(line)
    if len(tokens) < count:
        return None
    try:
        return [parse_real(token) for token in tokens[:count]]
    except ValueError:
        return None


def _integer(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


def _first_record(line: str) -> tuple[float, int] | None:
    """``Zval, number_of_channels`` of a bare table, as QE's reader tests it."""

    tokens = _tokens(line)
    if len(tokens) < 2:
        return None
    try:
        valence_charge = parse_real(tokens[0])
    except ValueError:
        return None
    count = _integer(tokens[1])
    if count is None or not 0.0 < valence_charge <= 100.0:
        return None
    return valence_charge, count


def parse_cpi(text: str, source: str = "FHI98PP table") -> CpiTable:
    """Parse the text of a ``.cpi`` or abinit ``.fhi`` file.

    The two forms are told apart as Quantum ESPRESSO's ``read_fhi`` does: a
    first line that is not ``Zval number_of_channels`` starts an abinit
    header.  A file whose tail is not a complete core-density block is
    refused, where QE would silently assume no core correction.
    """

    lines = text.splitlines()

    def fail(message: str) -> ConversionError:
        return ConversionError(f"{source}: {message}")

    def line(index: int, label: str) -> str:
        if index >= len(lines):
            raise fail(f"file ends before {label}")
        return lines[index]

    position = 0
    header: AbinitHeader | None = None
    first = _first_record(line(0, "the first record"))
    if first is None:
        charges = _reals(line(1, "zatom, zion, pspdat"), 3)
        codes = _reals(line(2, "pspcod ... r2well"), 6)
        core = _reals(line(3, "rchrg, fchrg, qchrg"), 3)
        if charges is None or codes is None or core is None:
            raise fail("neither a cpi table nor an abinit pspcod-6 header")
        header = AbinitHeader(
            title=lines[0].strip(),
            zatom=charges[0], zion=charges[1], pspdat=int(charges[2]),
            pspcod=int(codes[0]), pspxc=int(codes[1]), lmax=int(codes[2]),
            lloc=int(codes[3]), mmax=int(codes[4]), r2well=codes[5],
            rchrg=core[0], fchrg=core[1], qchrg=core[2],
        )
        if header.pspcod != 6:
            raise fail(f"abinit pseudopotential type {header.pspcod} is not FHI (6)")
        position = ABINIT_HEADER_LINES
        first = _first_record(line(position, "the cpi table"))
        if first is None:
            raise fail("no cpi table after the abinit header")
        if abs(header.zion - first[0]) > 1.0e-8:
            raise fail(f"zion {header.zion} and Zval {first[0]} disagree")
        if first[1] - 1 != header.lmax:
            raise fail(f"lmax {header.lmax} disagrees with {first[1]} channels")
    valence_charge, channel_count = first
    if not 1 <= channel_count <= MAX_CHANNELS:
        raise fail(f"{channel_count} angular channels; need 1 to {MAX_CHANNELS}")
    position += 11

    radius: tuple[float, ...] = ()
    mesh_increment = 0.0
    channels: list[CpiChannel] = []
    for angular_momentum in range(channel_count):
        label = f"the l={angular_momentum} channel"
        tokens = _tokens(line(position, label))
        mesh = _integer(tokens[0]) if tokens else None
        increment = _reals(" ".join(tokens[1:]), 1)
        if mesh is None or mesh < 2 or increment is None:
            raise fail(f"cannot read 'mmax amesh' of {label}")
        if header is not None and header.mmax > 0 and mesh != header.mmax:
            raise fail(f"{label} has {mesh} mesh points, the header says {header.mmax}")
        position += 1
        rows = []
        for row in range(mesh):
            values = _reals(line(position + row, label), 4)
            if values is None:
                raise fail(f"cannot read mesh record {row + 1} of {label}")
            rows.append(values)
        position += mesh
        channel_radius = tuple(row[1] for row in rows)
        if angular_momentum == 0:
            radius, mesh_increment = channel_radius, increment[0]
        elif channel_radius != radius or increment[0] != mesh_increment:
            raise fail("channels on different radial meshes are not supported")
        channels.append(
            CpiChannel(
                angular_momentum,
                tuple(row[2] for row in rows),
                tuple(row[3] for row in rows),
            )
        )

    core: CoreDensity | None = None
    if any(entry.strip() for entry in lines[position:]):
        rows = []
        for row in range(len(radius)):
            values = _reals(line(position + row, "the core density"), 4)
            # QE's tolerance for the core mesh.
            if values is None or abs(values[0] - radius[row]) > 1.0e-6:
                raise fail(
                    f"record {row + 1} after the last channel is not a core-density "
                    "record on the channel mesh"
                )
            rows.append(values)
        core = CoreDensity(
            tuple(row[1] for row in rows),
            tuple(row[2] for row in rows),
            tuple(row[3] for row in rows),
        )
    try:
        return CpiTable(
            valence_charge, mesh_increment, radius, tuple(channels), core, header
        )
    except ConversionError as exc:
        raise fail(str(exc)) from None


def read_cpi(path: str | os.PathLike[str]) -> CpiTable:
    """Read a ``.cpi`` file or its abinit ``.fhi`` form.

    For a ``.fhi`` file the header is returned in ``table.abinit_header``.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="latin-1")
    except OSError as exc:
        raise ConversionError(f"cannot read {source}: {exc}") from exc
    return parse_cpi(text, str(source))


def format_cpi(table: CpiTable) -> str:
    """The ``.cpi`` text of a table, laid out as ``psgen`` writes it."""

    def real(value: float) -> str:
        return format_e(value, 20, 14)

    parts = [
        f"{real(table.valence_charge)}   {len(table.channels)}\n",
        UNUSED_FIT_PARAMETERS,
    ]
    for channel in table.channels:
        parts.append(f"{table.mesh_size}  {real(table.mesh_increment)}\n")
        parts.extend(
            f"{index:4d} {real(radius)} {real(wavefunction)} {real(potential)}\n"
            for index, (radius, wavefunction, potential) in enumerate(
                zip(table.radius, channel.wavefunction, channel.potential), start=1
            )
        )
    if table.core is not None:
        parts.extend(
            f"{real(radius)} {real(density)} {real(first)} {real(second)}\n"
            for radius, density, first, second in zip(
                table.radius, table.core.density,
                table.core.first_derivative, table.core.second_derivative,
            )
        )
    return "".join(parts)


def write_cpi(path: str | os.PathLike[str], table: CpiTable) -> None:
    """Write a ``.cpi`` file (also the content of ``Element_FHIPP.DAT``).

    A table read from a ``psgen`` file is written back byte for byte.
    """

    Path(path).write_text(format_cpi(table), encoding="ascii", newline="\n")


def core_charge_radius(table: CpiTable, threshold: float = 1.0e-8) -> float:
    """Mesh radius from which ``4*pi*rho_core`` stays below ``threshold``."""

    if table.core is None:
        return 0.0
    for index in range(table.mesh_size - 1, -1, -1):
        if abs(table.core.density[index]) >= threshold:
            return table.radius[min(index + 1, table.mesh_size - 1)]
    return table.radius[0]


def format_fhi(
    table: CpiTable,
    *,
    zatom: float,
    pspxc: int,
    lloc: int,
    title: str,
    pspdat: int = 0,
    rchrg: float | None = None,
    qchrg: float = 0.0,
) -> str:
    """The text of an abinit ``.fhi`` file: header and cpi body."""

    if "\n" in title or "\r" in title:
        raise ConversionError("the abinit title must be a single line")
    if _first_record(title) is not None:
        # Readers recognise the abinit form by a first line that is not
        # "Zval number_of_channels".
        raise ConversionError("the abinit title must not start with two numbers")
    if not 0 <= lloc <= table.lmax:
        raise ConversionError(f"local channel l={lloc} is not tabulated")
    if not 0 <= pspdat <= 999999:
        raise ConversionError("pspdat is a date written as yymmdd")
    fchrg = 1.0 if table.core is not None else 0.0
    if rchrg is None:
        rchrg = core_charge_radius(table)
    # A blank and F6.3, which widens for zatom >= 100 as the library files do.
    header = (
        f"{title}\n"
        f" {zatom:6.3f} {table.valence_charge:6.3f}    {pspdat:06d}"
        "              zatom,zion,pspdat\n"
        f"{6:4d}{pspxc:10d}{table.lmax:5d}{lloc:4d}{table.mesh_size:7d}{0:6d}"
        "       pspcod,pspxc,lmax,lloc,mmax,r2well\n"
        f" {rchrg:6.3f} {fchrg:6.3f} {qchrg:6.3f}"
        "                 rchrg, fchrg, qchrg\n"
        "5--- These two lines are available for giving more information, later\n"
        "6\n"
        "7-Here follows the cpi file from the fhi98pp code-\n"
    )
    return header + format_cpi(table)


def write_fhi(
    path: str | os.PathLike[str],
    table: CpiTable,
    *,
    zatom: float,
    pspxc: int,
    lloc: int,
    title: str,
    pspdat: int = 0,
    rchrg: float | None = None,
    qchrg: float = 0.0,
) -> None:
    """Write the abinit ``.fhi`` form: a ``pspcod`` 6 header, then the table.

    The header follows the FHI98PP library files distributed for abinit
    (``NN-El.GGA.fhi``): line 1 a free title; line 2 ``zatom, zion, pspdat``
    (nuclear charge, valence charge, a date ``yymmdd``); line 3 ``pspcod`` 6,
    ``pspxc`` (abinit's ``ixc``: 2 Perdew-Zunger, 7 Perdew-Wang 1992, 11 PBE),
    ``lmax`` (channels - 1), ``lloc``, ``mmax`` (mesh points), ``r2well`` 0;
    line 4 ``rchrg, fchrg, qchrg``; three fixed lines; then the cpi body.

    Verified: the layout of the seven header lines, which reproduces those of
    all 103 library files from their own values; that :func:`read_cpi` reads
    the result back identically; and that Quantum ESPRESSO's ``read_fhi``
    accepts it and converts it to the same UPF numbers as the bare table (it
    requires ``pspcod`` 6, ``zion`` equal to the table's ``Zval``, ``lmax``
    equal to channels - 1 and ``mmax`` equal to the mesh size, and it reads
    ``pspxc`` 7 as ``SLA-PW`` and 11 as ``PBE``).

    Not verified: nothing here was read by abinit itself.  ``pspxc`` 2 and 7
    for the two LDA parametrizations come from abinit's ``ixc`` convention;
    the library has only ``pspxc`` 11 files.  ``rchrg`` is the radius beyond
    which abinit takes the model core charge as zero; the library's values
    sit on mesh radii where ``4*pi*rho_core`` is between 1e-8 and 2e-7 and
    follow no rule that could be recovered, so the default here is the first
    mesh radius from which it stays below 1e-8, which is larger.  ``fchrg``
    is written as 1 with a core correction and 0 without, ``qchrg`` as 0, as
    in every library file.  The library files tabulate four channels
    (``lmax`` 3); tables with fewer are written with their own ``lmax``.
    ``pspdat`` defaults to 0 so that the output does not depend on the day it
    is written.
    """

    text = format_fhi(
        table, zatom=zatom, pspxc=pspxc, lloc=lloc, title=title,
        pspdat=pspdat, rchrg=rchrg, qchrg=qchrg,
    )
    Path(path).write_text(text, encoding="latin-1", newline="\n")
