"""Reader of FHI98PP pseudopotentials in PARSEC's ``*_FHIPP.DAT`` layout.

``Pseudopotential_Format: fhipp`` selects the ``FHIPP`` branches of
``pseudo.f90`` (lines 204-220 and 611-655).  The file is the ``.cpi`` output
of the FHI98PP generator, read with list-directed statements::

    zion  lmax+1                          ionic charge, number of channels
    ten records                           not read (the abinit header block)
    mesh  amesh                           for each channel l = 0, 1, ...:
    i  r  u  V                            mesh records
    r  rho_core  ...                      mesh records, read only with NLCC

``u`` is the reduced radial function ``r*R_l`` and ``V`` the potential in
Hartree, which becomes Rydberg.  The logarithmic grid is
``r(i) = a*exp(b*(i-1))`` with Fortran index ``i`` (the inserted origin is
``i = 1``), so ``a = r_1/amesh``, ``b = log(amesh)`` and ``c = 0``.  The
core density column is ``4*pi`` times the volume density.

The input of the species adds what the file does not hold: the number of
channels (``Potential_Num``, which must equal the file's), the occupations
(``Electron_Per_Orbital``), the local channel (``Local_Component``), the
radius of the nonlocal projectors (``Core_Cutoff_Radius``) and whether the
core is corrected (``Nonlinear_Core_Correction``).  The ionic charge is the
one of the file.

``pseudo.f90:638`` compares the ``amesh`` of each channel with
``exp(log(amesh))`` of the same channel, which tests nothing; the check
meant, that every channel has the ``amesh`` of the first, is made here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .potr import _ListDirectedRecords
from .potre import ParsecPseudopotential

#: Records between the first one and the first channel (``pseudo.f90:217``).
SKIPPED_RECORDS = 10

FHI_ORIGIN_ENV = "PARSEC_FHI_ORIGIN"
FHI_ORIGIN_DEFAULT = "auto"
#: ``auto`` continues the mesh of a table below its first sample down to this radius (bohr) or just below it.
FHI_ORIGIN_RADIUS = 1.0e-5
#: The first samples of the file the even continuation of the core density is fitted to.
FHI_ORIGIN_FIT_SAMPLES = 12
#: The file samples of a potential and of a pseudo-wavefunction within this many first radii are replaced by the even
#: fit of the samples out to :data:`FHI_ORIGIN_REFIT_OUTER` first radii (FHI98PP's start, see :func:`fhi_origin_setting`).
FHI_ORIGIN_REFIT_INNER = 4.0
FHI_ORIGIN_REFIT_OUTER = 10.0
#: The powers ``r^2 .. r^(2 n)`` of the even fit of a potential beyond its constant.
FHI_ORIGIN_REFIT_DEGREE = 3
#: A table keeps its first samples where :func:`near_origin_ratio` exceeds this for a channel and the even fit would move a
#: potential at the first sample by more than :data:`FHI_ORIGIN_KEEP_CHANGE` (Ry): see :func:`keeps_file_samples`.
FHI_ORIGIN_KEEP_RATIO = 10.0
FHI_ORIGIN_KEEP_CHANGE = 2.7e-6


def fhi_origin_setting() -> str:
    """``PARSEC_FHI_ORIGIN``: what an FHIPP table holds between the atom and its first radial sample.

    FHI98PP starts its mesh at ``r(1) = 0.00625/Z`` bohr (``rmin`` of its ``default.h``): 1.04e-3 bohr for C, 7.8e-4
    for O, 1.25e-3 for B, 6.25e-3 for H, where a POTRE table starts at 4e-7 to 8e-5 bohr.  ``pseudo.f90`` fills the
    interval from the atom to the first sample as for a POTRE table, and with an FHI table that reaches every grid
    point within about 1.5e-3 bohr of an atom:

    - the cubic spline of a radial function runs through padding knots at ``r = -M-1, ..., 0`` that repeat the first
      sample, so between the atom and ``r(1)`` it is a crater with a cusp at the atom instead of the even
      continuation of the table: the C local potential lies 4.4e-4 Ry below it at the atom and starts with a slope of
      +0.42 Ry/bohr (O 5.7e-4 Ry and +0.73 Ry/bohr);
    - the projectors take the value of the 17th sample at the first eight (``vw(j) = vw(2*mor)``), so a p projector
      is 1.48 to 1.25 times the ``c r`` of the table out to 1.24e-3 bohr (C), and its spline between the atom and
      ``r(1)`` 3 to 8 times;
    - within ``r(1)`` the harmonics divide by ``r(1)`` (``nonloc.F90``'s ``rnorm``).

    ``auto`` (the default): the reader continues the logarithmic mesh of the file (its ratio ``amesh``) below the
    first sample down to :data:`FHI_ORIGIN_RADIUS` (1e-5 bohr; 191 samples for C), and where the near-origin
    structure of the file is FHI98PP's start it replaces the first samples as well.  The unscreening of FHI98PP starts
    at its first mesh point, and the potentials of a table carry that start out to a few first radii: for the light
    tables measured (H, B, C, N, O, Si with PBE, C and O with CA) the same input generated on a mesh that starts 284
    samples further in differs from the file there by 5.8e-6 to 3.0e-5 Ry beyond a constant, and ``V'(r)/r`` of the
    PBE carbon table at ``r(1)`` is -827.6 Ry/bohr^2 in the file against -793.5 in the finer generation (in the CA
    carbon table the start flips its sign).  Every potential within :data:`FHI_ORIGIN_REFIT_INNER` first radii and below
    the first then takes the even polynomial ``V(r_j) + c2 (r^2 - r_j^2) + ...`` up to ``r^6`` fitted to the samples out
    to :data:`FHI_ORIGIN_REFIT_OUTER` first radii through the junction sample ``r_j`` (:func:`refit_window`), every
    pseudo-wavefunction ``A r^(l+1) (1 + beta r^2)`` fitted alike, and the core density (a closed form at every sample
    of the generator, kept) ``c0 + c2 r^2 + c4 r^4`` through the first sample; below the junction those tables then
    follow the finer generation to 9e-7 to 4.6e-6 Ry beyond the constant (``V'(r)/r`` of carbon at ``r(1)`` -793.9).
    A table whose near-origin structure is a ``C/r``-like term that a refit would move by more than 2.7e-6 Ry at the
    first sample keeps its samples (:func:`keeps_file_samples`; the 4f lanthanides, Cu, Lu, Cr, and tables whose start
    exceeds that limit): below the first sample it takes the even fit of its first samples through the first one,
    and the power law of the generator's start for its pseudo-wavefunctions.  Such a term is a property of the table
    (the finer generation continues it inside ``r(1)``; Yb reaches about 6e-3 bohr), which central differences over
    1e-3 bohr next to a grid point resolve with the finer generation as well.  The three conventions above act within
    1.5e-5 bohr of an atom.  A table that starts at or below 1e-5 bohr is read as before (same bits).  ``off``: the
    table as the file gives it, as ``pseudo.f90`` reads it (PARSEC 1.5; the bits of before; the value under
    ``PARSEC_FORTRAN_NUMERICS=on`` where the switch is not set).  An unknown value is refused when the
    pseudopotentials are read, before the SCF.
    """

    from ..fortran_numerics import switch_value

    value = switch_value(FHI_ORIGIN_ENV, FHI_ORIGIN_DEFAULT)
    if value not in ("auto", "off"):
        raise ValueError(f"{FHI_ORIGIN_ENV} must be auto or off, not {value!r}")
    return value


def origin_sample_count(first_radius: float, amesh: float) -> int:
    """The number of samples ``PARSEC_FHI_ORIGIN=auto`` adds below a first radial sample on a mesh of ratio ``amesh``:
    the fewest that reach :data:`FHI_ORIGIN_RADIUS` or below it, none for a table that starts there already."""

    first = float(first_radius)
    if not first > FHI_ORIGIN_RADIUS * (1.0 + 1.0e-9):
        return 0
    return int(math.ceil(math.log(first / FHI_ORIGIN_RADIUS) / math.log(float(amesh)) - 1.0e-9))


def _anchored_even_fit(radii: np.ndarray, values: np.ndarray, anchor: int, stop: int, degree: int):
    """The even polynomial ``P(r) = values[anchor] + sum_k c_k t^k`` (``k = 1..degree``, ``t = (r/r_a)^2 - 1`` with
    ``r_a = radii[anchor]``) fitted by least squares to the samples ``anchor .. stop - 1``: it passes through the
    anchor sample, so a table continued with it below the anchor has no step there.  Returns ``P`` as a function of
    the radii.

    The normal equations are summed and solved in Python floats in a fixed order, so that every process and machine
    forms the same bits (no LAPACK)."""

    origin = float(radii[anchor])
    base = float(values[anchor])
    t = [(float(r) / origin) ** 2 - 1.0 for r in radii[anchor:stop]]
    y = [float(v) - base for v in values[anchor:stop]]
    basis = [[s ** (k + 1) for k in range(degree)] for s in t]
    matrix = [[math.fsum(row[i] * row[j] for row in basis) for j in range(degree)] for i in range(degree)]
    right = [math.fsum(row[i] * value for row, value in zip(basis, y)) for i in range(degree)]
    # Gaussian elimination with partial pivoting
    for column in range(degree):
        pivot = max(range(column, degree), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        right[column], right[pivot] = right[pivot], right[column]
        for row in range(column + 1, degree):
            factor = matrix[row][column] / matrix[column][column]
            for j in range(column, degree):
                matrix[row][j] -= factor * matrix[column][j]
            right[row] -= factor * right[column]
    coefficients = [0.0] * degree
    for row in range(degree - 1, -1, -1):
        coefficients[row] = (
            right[row] - sum(matrix[row][j] * coefficients[j] for j in range(row + 1, degree))
        ) / matrix[row][row]

    def polynomial(points: np.ndarray) -> np.ndarray:
        s = (np.asarray(points, dtype=float) / origin) ** 2 - 1.0
        total = np.zeros_like(s)
        for coefficient in reversed(coefficients):
            total = (total + coefficient) * s
        return base + total

    return polynomial


def refit_window(radii: np.ndarray) -> tuple[int, int]:
    """The samples ``anchor .. stop - 1`` of a table whose first ``anchor`` samples ``PARSEC_FHI_ORIGIN=auto`` replaces:
    from the first at or beyond :data:`FHI_ORIGIN_REFIT_INNER` times the first radius to the last within
    :data:`FHI_ORIGIN_REFIT_OUTER` times it; ``(0, FHI_ORIGIN_FIT_SAMPLES)`` (no sample replaced) for a table too short
    for that window."""

    first = float(radii[0])
    anchor = int(np.searchsorted(radii, FHI_ORIGIN_REFIT_INNER * first * (1.0 - 1.0e-12), side="left"))
    stop = int(np.searchsorted(radii, FHI_ORIGIN_REFIT_OUTER * first * (1.0 + 1.0e-12), side="right"))
    if stop - anchor < FHI_ORIGIN_REFIT_DEGREE + 4 or stop > radii.size - 2:
        return 0, min(FHI_ORIGIN_FIT_SAMPLES, radii.size)
    return anchor, stop


def near_origin_ratio(radii: np.ndarray, values: np.ndarray) -> float:
    """``|(V'/r)(r(3)) / (V'/r)(r(95)) - 1|`` of a potential on the samples of its file (central differences): about
    0.1 to 3 where the even behaviour ``V(0) + a r^2`` dominates next to the atom, about 840 for a pure ``C/r`` term
    (``(r(95)/r(3))^3`` on a mesh of ratio 1.0247)."""

    def slope_over_radius(index: int) -> float:
        return float((values[index + 1] - values[index - 1]) / (radii[index + 1] - radii[index - 1]) / radii[index])

    denominator = slope_over_radius(94)
    if denominator == 0.0:
        return math.inf
    return abs(slope_over_radius(2) / denominator - 1.0)


def keeps_file_samples(radii: np.ndarray, potentials: dict, fits: dict) -> bool:
    """Whether ``PARSEC_FHI_ORIGIN=auto`` keeps the first samples of a table instead of replacing them by ``fits`` (the
    even fits of :func:`refit_window`): where the near-origin structure of the file is a ``C/r``-like term that
    dominates the even behaviour (:func:`near_origin_ratio` above :data:`FHI_ORIGIN_KEEP_RATIO` for a channel) and
    whose replacement would move a potential at the first sample by more than :data:`FHI_ORIGIN_KEEP_CHANGE`.  One
    decision for the table: the structure is common to its channels, and the differences ``V_l - V_local`` of the
    projectors keep one form.

    FHI98PP's start is ``C/r``-like too, so the shape alone does not tell it from a ``C/r`` term of the table itself,
    and both occur with ratios from 10 to several thousand.  Against the same inputs generated on a mesh that starts
    284 samples further in (where a start moves with the mesh and a term of the table does not): of the tables with a
    ratio above 10, Mg, Na, Ca, W, Au and Hf carry a start whose replacement moves the first sample by 6e-7 to 2.5e-6
    Ry and brings them to 1e-7 to 4e-7 Ry of the finer generation (the file 7.6e-7 to 2.9e-6); the tables with a term
    of their own (the file within 1.7e-6 Ry of the finer generation) would move by 3.0e-6 (Tb), 5.3e-6 (Ho), 7.5e-6
    (CA Cr), 2.1e-5 (Cr), 2.7e-5 (Lu), 9.4e-5 (Cu) and 5.3e-3 to 1.2e-2 Ry (Pr, Yb).  Tables with a start above the
    limit (Si, Ge, P, S, Cl, Ar, Br, Kr with CA, Be: 3.2e-6 to 1.25e-5 Ry in the file) keep it, as the file has it."""

    if radii.size < 96:
        return True
    ratio = max(near_origin_ratio(radii, values) for values in potentials.values())
    if not ratio > FHI_ORIGIN_KEEP_RATIO:
        return False
    first = radii[:1]
    change = max(abs(float(fits[key](first)[0] - values[0])) for key, values in potentials.items())
    return change > FHI_ORIGIN_KEEP_CHANGE


def read_fhipp_pseudopotential(
    path: str | Path,
    *,
    symbol: str,
    number_of_channels: int,
    electrons_per_orbital: Sequence[float],
    core_cutoff_radius: float,
    nonlinear_core_correction: bool = False,
) -> ParsecPseudopotential:
    """Read a PARSEC ``FHIPP``/``*_FHIPP.DAT`` pseudopotential.

    ``number_of_channels`` is ``Potential_Num``, ``electrons_per_orbital``
    the line of the ``Electron_Per_Orbital`` block (s, p, d, f; its first
    ``number_of_channels`` values are used), ``core_cutoff_radius`` the
    ``Core_Cutoff_Radius`` in bohr and ``nonlinear_core_correction`` the
    ``Nonlinear_Core_Correction`` flag of the species, which makes PARSEC
    read the core density after the channels.  ``symbol`` names the element.

    The tables are aligned with the radial samples of the file: potentials
    in Rydberg, reduced radial functions, the valence density of the
    occupied pseudo-orbitals and the core density in electrons/bohr**3.
    ``radial_parameter_c`` is zero.
    """

    source = Path(path)
    channels = int(number_of_channels)
    if channels != number_of_channels or not 1 <= channels <= 4:
        raise ValueError(
            f"{source}: impossible number of potentials {number_of_channels!r}; "
            "Potential_Num must be 1 to 4"
        )
    occupations = [float(value) for value in electrons_per_orbital]
    if len(occupations) < channels or not np.all(np.isfinite(occupations)):
        raise ValueError(
            f"{source}: Electron_Per_Orbital must give a finite occupation for "
            f"each of the {channels} channels"
        )
    occupations = occupations[:channels]
    cutoff = float(core_cutoff_radius)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"{source}: Core_Cutoff_Radius must be positive")

    records = _ListDirectedRecords(
        source.read_text(encoding="utf-8").splitlines(), source
    )
    ionic_charge, file_channels = records.reals(2, "the first record 'zion, lmax+1'")
    if int(file_channels) != file_channels or int(file_channels) != channels:
        # pseudo.f90 error 203.
        raise ValueError(
            f"{source} holds {file_channels:g} pseudopotential components, but "
            f"Potential_Num is {channels}"
        )
    for _ in range(SKIPPED_RECORDS):
        records.skip()

    count = None
    amesh = None
    radii = None
    channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    for angular_momentum in range(channels):
        mesh_value, mesh_ratio = records.reals(2, f"the mesh record of l={angular_momentum}")
        mesh = int(mesh_value)
        if mesh != mesh_value or mesh < 2:
            raise ValueError(f"{source}: invalid radial mesh size {mesh_value!r}")
        if count is None:
            count, amesh = mesh, mesh_ratio
        elif mesh != count:
            # pseudo.f90 error 211.
            raise ValueError(
                f"{source}: radial grid size for angular component "
                f"l = {angular_momentum} is {mesh}, not the {count} of the "
                "other components"
            )
        elif mesh_ratio != amesh:
            # The check pseudo.f90 error 212 means to make.
            raise ValueError(
                f"{source}: radial increment {mesh_ratio!r} of angular "
                f"component l = {angular_momentum} differs from the "
                f"increment {amesh!r} of the other components"
            )
        table = np.empty((mesh, 4))
        for row in range(mesh):
            table[row] = records.reals(4, f"the table of l={angular_momentum}")
        channel_radii = table[:, 1].copy()
        if radii is not None and not np.allclose(channel_radii, radii, rtol=1.0e-10, atol=0.0):
            raise ValueError(
                f"{source}: the radial grid of l={angular_momentum} differs "
                "from that of l=0"
            )
        radii = channel_radii
        radial_wavefunctions[angular_momentum] = table[:, 2].copy()
        channel_potentials[angular_momentum] = table[:, 3] * 2.0
    assert radii is not None and count is not None and amesh is not None
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")
    if not amesh > 1.0:
        raise ValueError(f"{source}: the radial increment amesh must exceed one")

    core_density = np.zeros(count)
    if nonlinear_core_correction:
        core_table = np.empty((count, 2))
        for row in range(count):
            values = records.reals(
                2, "the core charge density (Nonlinear_Core_Correction)", end_ok=True
            )
            if values is None:
                raise ValueError(
                    f"{source} ends before its core charge density: "
                    "Nonlinear_Core_Correction needs one record 'r, rho_core' "
                    "per radial sample after the channels"
                )
            core_table[row] = values
        if not np.allclose(core_table[:, 0], radii, rtol=1.0e-10, atol=0.0):
            raise ValueError(
                f"{source}: the radial grid of the core charge differs from "
                "that of the channels"
            )
        # PARSEC keeps the radii read last.
        radii = core_table[:, 0].copy()
        core_density = core_table[:, 1] / (4.0 * np.pi)

    # PARSEC_FHI_ORIGIN=auto: the mesh continued below the first sample with the even continuation of every column
    inner_count = origin_sample_count(radii[0], amesh) if fhi_origin_setting() == "auto" else 0
    anchor = 0
    if inner_count:
        inner = float(radii[0]) * np.exp(-math.log(float(amesh)) * np.arange(inner_count, 0, -1, dtype=float))
        # the samples within FHI_ORIGIN_REFIT_INNER first radii (FHI98PP's start) and the new ones below the first take
        # the even fit of the samples out to FHI_ORIGIN_REFIT_OUTER first radii, through the sample at the junction,
        # unless the table's own near-origin structure is a C/r-like term (keeps_file_samples)
        anchor, stop = refit_window(radii)
        fits = {
            angular_momentum: _anchored_even_fit(
                radii, channel_potentials[angular_momentum], anchor, stop, FHI_ORIGIN_REFIT_DEGREE
            )
            for angular_momentum in range(channels)
        }
        if anchor and keeps_file_samples(radii, channel_potentials, fits):
            anchor = 0
        if anchor:
            replaced = np.concatenate((inner, radii[:anchor]))
            for angular_momentum in range(channels):
                power = angular_momentum + 1
                wavefunction = radial_wavefunctions[angular_momentum]
                # u_l = A r^(l+1) (1 + beta r^2)
                reduced = _anchored_even_fit(radii, wavefunction / radii**power, anchor, stop, 1)
                radial_wavefunctions[angular_momentum] = np.concatenate(
                    (replaced**power * reduced(replaced), wavefunction[anchor:])
                )
                potential = channel_potentials[angular_momentum]
                channel_potentials[angular_momentum] = np.concatenate(
                    (fits[angular_momentum](replaced), potential[anchor:])
                )
        else:
            # the samples of the file kept: below the first one the even fit of the first samples through it, and the
            # power law of the generator's own start for the pseudo-wavefunctions
            count_fit = min(FHI_ORIGIN_FIT_SAMPLES, count)
            for angular_momentum in range(channels):
                wavefunction = radial_wavefunctions[angular_momentum]
                radial_wavefunctions[angular_momentum] = np.concatenate(
                    (wavefunction[0] * (inner / radii[0]) ** (angular_momentum + 1), wavefunction)
                )
                potential = channel_potentials[angular_momentum]
                continuation = _anchored_even_fit(radii, potential, 0, count_fit, 2)
                channel_potentials[angular_momentum] = np.concatenate((continuation(inner), potential))
        if nonlinear_core_correction:
            # the model core of the generator is a closed form at every sample: its first samples are kept
            core_fit = _anchored_even_fit(radii, core_density, 0, min(FHI_ORIGIN_FIT_SAMPLES, count), 2)
            core_density = np.concatenate((core_fit(inner), core_density))
        else:
            core_density = np.zeros(inner_count + count)
        radii = np.concatenate((inner, radii))

    channel_occupations = {
        angular_momentum: occupations[angular_momentum]
        for angular_momentum in range(channels)
    }
    valence_density = np.zeros(radii.size)
    for angular_momentum in range(channels):
        wavefunction = radial_wavefunctions[angular_momentum]
        valence_density += (
            channel_occupations[angular_momentum]
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * radii * radii)
        )

    return ParsecPseudopotential(
        symbol=symbol,
        correlation="",
        relativity="",
        core_marker="pc" if nonlinear_core_correction else "nc",
        comment="",
        configuration="",
        number_of_channels=channels,
        number_of_spin_orbit_channels=0,
        radial_parameter_a=float(radii[0] / amesh),
        radial_parameter_b=float(np.log(amesh)),
        ionic_charge=float(ionic_charge),
        radii=radii,
        channel_potentials=channel_potentials,
        valence_density=valence_density,
        core_density=core_density,
        radial_wavefunctions=radial_wavefunctions,
        channel_occupations=channel_occupations,
        channel_cutoffs={
            angular_momentum: cutoff for angular_momentum in range(channels)
        },
        source=source,
        core_radius=cutoff,
        file_format="fhipp",
        radial_parameter_c=0.0,
        origin_samples=inner_count,
        origin_refit=anchor,
    )


__all__ = ["SKIPPED_RECORDS", "read_fhipp_pseudopotential"]
