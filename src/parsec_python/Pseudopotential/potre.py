"""Reader and radial operations for PARSEC ``*_POTRE.DAT`` files.

The supported layout is the ``MARTINS_NEW``/``potre`` format read by
``pseudo.f90``.  The file stores ``r*V_l(r)`` and radial charge distributions;
this module converts them to the volume-density and potential conventions used
by the rest of the Python solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import re

import numpy as np

from .radial_quadrature import (
    LOG_MESH_TOLERANCE,
    kb_quadrature_setting,
    log_mesh_deviation,
    log_mesh_integral,
    parsec_radial_integral,
)
from .radial_spline import ParsecRadialSpline


# Fortran drops the exponent letter when the exponent needs three digits:
# ``.699054148690-131`` is 0.699e-131.  The second alternative of the exponent
# accepts that form.
_FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+|[-+]\d{3}(?!\d))?"
)
_BARE_EXPONENT_RE = re.compile(r"(?<=[\d.])([-+]\d{3})$")

PROJECTOR_TAIL_ENV = "PARSEC_PROJECTOR_TAIL"
PROJECTOR_TAIL_DEFAULT = "auto"


def projector_tail_setting() -> str:
    """``PARSEC_PROJECTOR_TAIL``: what a ``MARTINS_NEW`` table's projectors hold beyond its channel cutoffs.

    ``off``: the tabulated ``V_l - V_local`` at every radius, as ``pseudo.f90`` forms it (PARSEC 1.5, and the
    default before 2026-11-22).  ``nonloc.F90``
    keeps the grid points up to ``rcore``, the first radial sample beyond the largest cutoff ``rc`` of the file's
    pseudo-wavefunction headers, and drops the rest.  A table whose projector is not zero at ``rcore`` makes the
    nonlocal energy jump whenever a grid point crosses that sphere: the energy is then not a continuous function of
    the positions, and a force cannot follow its slope.  The tables converted from UPF files carry such a value (the
    interpolation of ``V_l`` and ``V_local`` onto the denser PARSEC grid overshoots at ``rc``, where the UPF holds
    ``V_l - V_local = 0`` exactly from ``rc`` on): 943 of the 1921 distinct tables of the package with projectors,
    up to 5.7e-4 of the projector's maximum, among them the carbon table of the diamond examples (projector 1.6e-4 at
    ``rcore``, 2e-5 of its maximum) and the PBE Cr table (2.1e-4).  At steps of 1e-3 bohr the energy slope then misses the force by 7.4e-5 Ry/bohr (diamond cell of eight
    atoms, h 0.37) and 6.7e-4 Ry/bohr (Cr2, h 0.3), both with ``Cubic_Spline``.

    ``auto`` (the default): ``V_l - V_local = 0`` beyond the largest channel cutoff of the file, as the
    norm-conserving construction (and the UPF the table came from) has it, so that every projector is zero at
    ``rcore`` and the energy is continuous (8.9e-6 and 3.2e-6 Ry/bohr in the two cases above; the energies move by
    1e-6 to 2e-5 Ry).  Tables without channel cutoffs (FHI, ``MARTINS_OLD``) and tables whose difference is zero
    there already are not changed (same bits).  The rule is a property of the file, so every process of a run
    decides the same.
    """

    from ..fortran_numerics import switch_value

    value = switch_value(PROJECTOR_TAIL_ENV, PROJECTOR_TAIL_DEFAULT)
    if value not in ("auto", "off"):
        raise ValueError(f"{PROJECTOR_TAIL_ENV} must be auto or off, not {value!r}")
    return value


def _numbers(text: str) -> list[float]:
    values = []
    for token in _FLOAT_RE.findall(text):
        token = token.replace("D", "E").replace("d", "e")
        if "e" not in token.lower():
            token = _BARE_EXPONENT_RE.sub(r"E\g<1>", token)
        values.append(float(token))
    return values


def _read_values(lines: list[str], start: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    index = start
    while len(values) < count and index < len(lines):
        values.extend(_numbers(lines[index]))
        index += 1
    if len(values) != count:
        raise ValueError(f"expected {count} radial values, found {len(values)}")
    return np.asarray(values, dtype=float), index


@dataclass(frozen=True)
class ParsecPseudopotential:
    """One norm-conserving PARSEC pseudopotential.

    Attributes use PARSEC units: bohr for radii and Rydberg for potentials.
    ``radial_wavefunctions`` contain the file's reduced radial functions
    ``u_l(r)``.  Densities are converted to electrons/bohr**3.
    """

    symbol: str
    correlation: str
    relativity: str
    core_marker: str
    comment: str
    configuration: str
    number_of_channels: int
    number_of_spin_orbit_channels: int
    radial_parameter_a: float
    radial_parameter_b: float
    ionic_charge: float
    radii: np.ndarray
    channel_potentials: dict[int, np.ndarray]
    valence_density: np.ndarray
    core_density: np.ndarray
    radial_wavefunctions: dict[int, np.ndarray]
    channel_occupations: dict[int, float]
    channel_cutoffs: dict[int, float]
    source: Path
    spin_orbit_channel_potentials: dict[int, np.ndarray] = field(default_factory=dict)
    # ``p_pot%rcore`` of a species whose file holds no cutoff radius: the
    # ``Core_Cutoff_Radius`` of a ``MARTINS_OLD`` species.  ``None`` for a
    # ``MARTINS_NEW`` file, whose radius follows from ``channel_cutoffs``.
    core_radius: float | None = None
    # ``Pseudopotential_Format`` the tables were read with.
    file_format: str = "martins_new"
    # ``par_c`` of ``pseudo.f90``: the radial grid is
    # ``r(i) = a*exp(b*(i-1)) - c``.  ``None`` is ``c = a``, the grid of every
    # Martins file; an FHIPP file has ``c = 0`` (see :attr:`radial_offset`).
    radial_parameter_c: float | None = None
    # Tables replaced by the Fourier filter of ``fourier_f.f90``
    # (:mod:`parsec_python.Pseudopotential.fourier_filter`): the normalized
    # KB radial projector of each filtered channel and the atomic valence
    # density.  Empty and ``None`` for an unfiltered potential.
    filtered_projectors: dict[int, np.ndarray] = field(default_factory=dict)
    filtered_valence_density: np.ndarray | None = None
    # The local angular momentum a ``MARTINS_WANG`` file names (``loc``);
    # ``None`` where the input chooses it (``Local_Component``).
    local_channel: int | None = None
    # ``dvw`` of a filtered channel: the radial derivative of its filtered
    # projector, ``dfint`` of PARSEC 1.5 (``math/fourier_f.f90:267``; the
    # previous tree stored the projector itself there).  Same keys as
    # ``filtered_projectors``; empty for an unfiltered potential.
    filtered_projector_slopes: dict[int, np.ndarray] = field(default_factory=dict)
    # The samples the reader added below the first sample of the file (``PARSEC_FHI_ORIGIN=auto`` for an FHIPP
    # table, :func:`parsec_python.Pseudopotential.fhipp.fhi_origin_setting`); 0 for a table as its file gives it.
    origin_samples: int = 0
    # The samples of the file the reader replaced by their even fit (``PARSEC_FHI_ORIGIN=auto``); 0 otherwise.
    origin_refit: int = 0

    @property
    def radial_offset(self) -> float:
        """``c`` of the radial grid ``r(i) = a*exp(b*(i-1)) - c`` in bohr.

        ``pseudo.f90`` differentiates radial tables with ``d/dr =
        1/(b*(r + c)) d/di``.
        """
        if self.radial_parameter_c is None:
            return self.radial_parameter_a
        return self.radial_parameter_c

    @property
    def has_nonlinear_core_correction(self) -> bool:
        return self.core_marker.strip().lower() not in {"nc", "none", "no"}

    @property
    def projector_support_radius(self) -> float:
        """``rc`` of ``nonloc.F90``: radius of the Kleinman-Bylander sphere.

        Grid points at distances up to this radius, inclusive, carry the
        nonlocal projectors of an atom.  It is ``p_pot%rcore`` capped at the
        penultimate radial sample.  ``pseudo.f90`` sets ``rcore`` for a
        ``MARTINS_NEW`` file to the first tabulated radius beyond the largest
        channel cutoff of the file; for a ``MARTINS_OLD`` file it is the
        ``Core_Cutoff_Radius`` of the input as given, which is in general
        not a tabulated radius.
        """
        if self.core_radius is not None:
            return float(min(self.core_radius, self.interpolation_cutoff))
        requested = max(self.channel_cutoffs.values(), default=self.radii[0])
        next_index = int(np.searchsorted(self.radii, requested, side="right"))
        next_index = min(next_index, self.radii.size - 2)
        return float(self.radii[next_index])

    @property
    def interpolation_cutoff(self) -> float:
        """Largest radius used by PARSEC before analytic/zero continuation."""
        if self.radii.size < 2:
            return float(self.radii[-1])
        return float(self.radii[-2])

    @property
    def projector_tail_radius(self) -> float | None:
        """The largest channel cutoff ``rc`` of the pseudo-wavefunction headers of a ``MARTINS_NEW`` table, beyond
        which ``PARSEC_PROJECTOR_TAIL=auto`` sets ``V_l - V_local`` to zero; ``None`` for a table without one cutoff
        per channel or with ``Core_Cutoff_Radius`` (``MARTINS_OLD``)."""

        if self.core_radius is not None or not self.channel_cutoffs:
            return None
        if not set(self.channel_potentials) <= set(self.channel_cutoffs):
            return None
        return float(max(self.channel_cutoffs.values()))

    def potential_difference(self, potential: np.ndarray, local_l: int) -> np.ndarray:
        """``potential - V_local``, the radial factor of a Kleinman-Bylander projector, with the tail rule of
        :func:`projector_tail_setting`: under ``auto`` zero at every sample beyond :attr:`projector_tail_radius`."""

        difference = np.asarray(potential, dtype=float) - self.channel_potentials[local_l]
        radius = self.projector_tail_radius
        if radius is None or projector_tail_setting() == "off":
            return difference
        beyond = self.radii > radius * (1.0 + 1.0e-9)
        if not np.any(difference[beyond] != 0.0):
            return difference
        return np.where(beyond, 0.0, difference)

    @cached_property
    def on_log_mesh(self) -> bool:
        """Whether the radii follow the logarithmic mesh of the table's parameters to
        :data:`~.radial_quadrature.LOG_MESH_TOLERANCE` (:func:`~.radial_quadrature.log_mesh_deviation`)."""

        deviation = log_mesh_deviation(self.radii, self.radial_parameter_b, self.radial_offset)
        return bool(deviation <= LOG_MESH_TOLERANCE)

    def kb_quadrature(self) -> str:
        """``"logmesh"`` or ``"trapezoid"``: the quadrature of this table's Kleinman-Bylander denominators under
        ``PARSEC_KB_QUADRATURE`` (:func:`~.radial_quadrature.kb_quadrature_setting`); ``auto`` takes ``logmesh`` where
        the radii follow the logarithmic mesh (:attr:`on_log_mesh`), the trapezoid of ``pseudo.f90`` elsewhere.  A
        property of the file, so every process of a run decides the same."""

        setting = kb_quadrature_setting()
        if setting == "auto":
            return "logmesh" if self.on_log_mesh else "trapezoid"
        return setting

    def kb_denominator(self, integrand: np.ndarray) -> float:
        """``integral integrand dr`` of a Kleinman-Bylander denominator (``u_l^2 (V_l - V_local)`` on the radii) by the
        quadrature of :meth:`kb_quadrature` (``PARSEC_KB_QUADRATURE``)."""

        if self.kb_quadrature() == "trapezoid":
            return parsec_radial_integral(self.radii, integrand)
        return log_mesh_integral(self.radii, integrand, self.radial_parameter_b, self.radial_offset)

    def validate_local_channel(self, local_l: int) -> None:
        if local_l not in self.channel_potentials:
            available = ", ".join(str(value) for value in sorted(self.channel_potentials))
            raise ValueError(
                f"local channel l={local_l} is absent from {self.source}; "
                f"available channels: {available}"
            )

    def local_potential_spline(self, local_l: int, spline_padding_width: int) -> ParsecRadialSpline:
        """The cubic spline :meth:`local_potential` evaluates with ``use_spline`` (PARSEC's origin padding).

        Built from the table each time; a caller that evaluates many batches passes it as ``spline``.
        """
        self.validate_local_channel(local_l)
        return ParsecRadialSpline.from_positive_grid(
            self.radii, self.channel_potentials[local_l], spline_padding_width
        )

    def local_potential(
        self,
        distances: np.ndarray,
        local_l: int,
        *,
        use_spline: bool = False,
        spline_padding_width: int | None = None,
        direct: bool = False,
        spline: ParsecRadialSpline | None = None,
    ) -> np.ndarray:
        """Evaluate the selected local potential with PARSEC's Coulomb tail.

        When ``use_spline`` is true, ``spline_padding_width`` must equal the
        real-space stencil half-width ``Expansion_Order/2``.  PARSEC uses it
        to extend the radial spline through the atomic origin.  ``spline``,
        if given, is that spline already built (:meth:`local_potential_spline`
        of the same channel and width): the same values without rebuilding it.

        Between the radial samples PARSEC interpolates ``r*V(r)`` linearly
        and divides by ``r``.  ``direct`` interpolates ``V(r)`` itself, the
        ``Old_Interpolation_Format`` branch of ``ionpot.f90:107-112``; the
        spline takes precedence over both.
        """
        self.validate_local_channel(local_l)
        radius = np.asarray(distances, dtype=float)
        if np.any(radius < 0):
            raise ValueError("radial distances cannot be negative")
        values = self.channel_potentials[local_l]
        result = np.empty_like(radius)
        # ``ionpot.f90`` switches at ``dist >= rs(ns-1)``.
        inside = radius < self.interpolation_cutoff
        if use_spline:
            if spline_padding_width is None:
                raise ValueError(
                    "spline_padding_width is required when use_spline=True"
                )
            if np.any(inside):
                if spline is None:
                    spline = ParsecRadialSpline.from_positive_grid(
                        self.radii,
                        values,
                        spline_padding_width,
                    )
                result[inside] = spline(radius[inside])
        elif direct:
            # ``V_j + delr*(V_(j+1) - V_j)``; below the first sample the
            # inserted origin carries the first value.
            if np.any(inside):
                result[inside] = np.interp(
                    radius[inside], self.radii, values, left=values[0]
                )
        else:
            near_zero = inside & (radius <= self.radii[0])
            result[near_zero] = values[0]
            interpolate = inside & ~near_zero
            if np.any(interpolate):
                sample_r = radius[interpolate]
                radial_potential = self.radii * values
                interpolated = np.interp(sample_r, self.radii, radial_potential)
                result[interpolate] = interpolated / sample_r

        outside = ~inside
        if np.any(outside):
            result[outside] = -2.0 * self.ionic_charge / radius[outside]
        return result

    def density_spline(self, spline_padding_width: int, *, core: bool = False) -> ParsecRadialSpline:
        """The cubic spline :meth:`interpolate_density` evaluates with ``use_spline`` (built from the table each time;
        a caller that evaluates many batches passes it as ``spline``)."""
        values = self.core_density if core else self.valence_density
        return ParsecRadialSpline.from_positive_grid(self.radii, values, spline_padding_width)

    def interpolate_density(
        self,
        distances: np.ndarray,
        *,
        core: bool = False,
        use_spline: bool = False,
        spline_padding_width: int | None = None,
        spline: ParsecRadialSpline | None = None,
    ) -> np.ndarray:
        """Interpolate a stored atomic volume density and set its tail to zero.

        The optional spline uses PARSEC's origin padding, whose width is the
        finite-difference stencil half-width ``Expansion_Order/2``.  ``spline``,
        if given, is that spline already built (:meth:`density_spline` of the
        same table and width): the same values without rebuilding it.
        """
        radius = np.asarray(distances, dtype=float)
        values = self.core_density if core else self.valence_density
        result = np.zeros_like(radius)
        # ``initchrg.f90``/``corecd.f90`` drop the density at and beyond
        # ``rs(ns-1)``.
        inside = radius < self.interpolation_cutoff
        if use_spline and spline_padding_width is None:
            raise ValueError(
                "spline_padding_width is required when use_spline=True"
            )
        if not np.any(inside):
            return result
        sample = radius[inside]
        if use_spline:
            if spline is None:
                spline = ParsecRadialSpline.from_positive_grid(
                    self.radii,
                    values,
                    spline_padding_width,
                )
            result[inside] = spline(sample)
        else:
            result[inside] = np.interp(sample, self.radii, values, left=values[0])
        return result

    def radial_projector(self, angular_momentum: int, local_l: int) -> tuple[np.ndarray, float]:
        """Return PARSEC's normalized KB radial projector and denominator sign.

        The unnormalized projector is
        ``(V_l - V_local) * u_l / r``.  PARSEC divides it by
        ``sqrt(abs(integral u_l**2 * (V_l - V_local) dr))`` and stores the
        denominator sign separately.
        """
        self.validate_local_channel(local_l)
        if angular_momentum == local_l:
            raise ValueError("the selected local channel has no KB projector")
        if angular_momentum not in self.radial_wavefunctions:
            raise ValueError(f"no pseudo-wavefunction for l={angular_momentum}")
        if angular_momentum not in self.channel_potentials:
            raise ValueError(f"no potential channel for l={angular_momentum}")

        delta_v = self.potential_difference(
            self.channel_potentials[angular_momentum], local_l
        )
        radial_wave = self.radial_wavefunctions[angular_momentum]
        denominator = self.kb_denominator(radial_wave * radial_wave * delta_v)
        if abs(denominator) < 1.0e-18:
            raise ValueError(f"nearly zero Kleinman-Bylander denominator for l={angular_momentum}")
        if angular_momentum in self.filtered_projectors:
            # ``fourier_f.f90`` replaces the normalized projector at every
            # sample; the denominator and its sign stay those of the file.
            return (
                np.array(self.filtered_projectors[angular_momentum], dtype=float),
                float(np.sign(denominator)),
            )
        projector = delta_v * radial_wave / self.radii
        projector /= np.sqrt(abs(denominator))
        # PARSEC's 18th-order radial derivative setup (mor=9) replaces
        # internal indices through mor with the value at 2*mor.  POTRE file
        # index zero is internal index two, so this is Python 0:8 <- 16.
        if projector.size > 16:
            projector[:8] = projector[16]
        return projector, float(np.sign(denominator))

    @property
    def has_spin_orbit_channels(self) -> bool:
        return bool(self.spin_orbit_channel_potentials)

    def spin_orbit_projectors(
        self, angular_momentum: int, local_l: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return PARSEC's KB-normalized average and spin-orbit projectors.

        A relativistic (``irel='rel'``) POTRE file stores, for a spin-orbit
        channel ``l`` (PARSEC supports only ``l=1`` and ``l=2``), the ordinary
        channel potential ``V_l`` together with a raw spin-orbit correction
        ``V_so_raw`` (the ``"Minor comp. of Pseudop. follows"`` block).
        ``pseudo.f90`` reconstructs the two total-angular-momentum channels as

        ``V_(j=l+1/2) = V_l + (l/2)*V_so_raw``
        ``V_(j=l-1/2) = V_l - ((l+1)/2)*V_so_raw``

        Each j-channel is then Kleinman-Bylander-normalized independently
        against the *same* reduced pseudo-wavefunction ``u_l(r)`` (PARSEC
        stores one wavefunction per ``l``, not per ``j``), exactly as
        :meth:`radial_projector` normalizes a single channel against the
        local potential.  The two normalized projectors are finally
        recombined into PARSEC's "average" (``vionr``) and "spin-orbit"
        (``vsor``) nonlocal projector pair:

        ``V_ion,l = [(l+1)*proj_(j+) + l*proj_(j-)] / (2l+1)``
        ``V_so,l  = 2*(proj_(j+) - proj_(j-)) / (2l+1)``

        The returned ``sign`` reconciles the two j-channels' KB-denominator
        signs the way ``pseudo.f90:797-819`` does: when they agree, that
        shared sign is used; when they disagree, the larger-magnitude
        channel's sign wins.  ``V_ion``/``V_so`` are themselves separable
        nonlocal projector radial functions, not local potentials -- they
        feed the spin-orbit nonlocal operator, not
        :meth:`local_potential`.
        """
        if angular_momentum not in (1, 2):
            raise ValueError(
                "spin-orbit projectors are only defined for l=1 (p) and l=2 (d)"
            )
        if angular_momentum not in self.spin_orbit_channel_potentials:
            raise ValueError(f"no spin-orbit channel for l={angular_momentum}")
        self.validate_local_channel(local_l)
        if angular_momentum not in self.channel_potentials:
            raise ValueError(f"no potential channel for l={angular_momentum}")
        if angular_momentum not in self.radial_wavefunctions:
            raise ValueError(f"no pseudo-wavefunction for l={angular_momentum}")

        channel_v = self.channel_potentials[angular_momentum]
        so_raw = self.spin_orbit_channel_potentials[angular_momentum]
        radial_wave = self.radial_wavefunctions[angular_momentum]

        v_plus = channel_v + 0.5 * angular_momentum * so_raw
        v_minus = channel_v - 0.5 * (angular_momentum + 1) * so_raw

        def _normalize(v_j: np.ndarray) -> tuple[np.ndarray, float]:
            delta_v = self.potential_difference(v_j, local_l)
            denominator = self.kb_denominator(radial_wave * radial_wave * delta_v)
            if abs(denominator) < 1.0e-18:
                raise ValueError(
                    "nearly zero spin-orbit Kleinman-Bylander denominator for "
                    f"l={angular_momentum}"
                )
            projector = delta_v * radial_wave / self.radii
            projector /= np.sqrt(abs(denominator))
            if projector.size > 16:
                projector[:8] = projector[16]
            return projector, denominator

        proj_plus, denom_plus = _normalize(v_plus)
        proj_minus, denom_minus = _normalize(v_minus)

        sign_plus = float(np.sign(denom_plus))
        sign_minus = float(np.sign(denom_minus))
        if sign_plus == sign_minus:
            sign = sign_plus
        elif abs(denom_plus) > abs(denom_minus):
            sign = sign_plus
        else:
            sign = sign_minus

        weight = 1.0 / (2 * angular_momentum + 1)
        v_ion = weight * (
            (angular_momentum + 1) * proj_plus + angular_momentum * proj_minus
        )
        v_so = 2.0 * weight * (proj_plus - proj_minus)
        return v_ion, v_so, sign


def read_parsec_pseudopotential(path: str | Path) -> ParsecPseudopotential:
    """Read a PARSEC ``MARTINS_NEW``/``*_POTRE.DAT`` pseudopotential."""
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{source} is too short to be a PARSEC potential")

    identity = lines[0].split()
    if len(identity) < 4:
        raise ValueError(f"invalid PARSEC pseudopotential identity line in {source}")
    symbol, correlation, relativity, core_marker = identity[:4]

    header = _numbers(lines[3])
    if len(header) < 6:
        raise ValueError(f"invalid numeric pseudopotential header in {source}")
    number_of_channels = int(header[0])
    number_of_spin_orbit_channels = int(header[1])
    number_of_grid_points = int(header[2])
    radial_parameter_a, radial_parameter_b, ionic_charge = header[3:6]
    # POTRE stores ``number_of_grid_points`` positive-radius samples.  PARSEC
    # inserts an additional origin entry internally and then reads indices
    # 2..ns, hence the easy-to-misread Fortran bounds.
    radial_count = number_of_grid_points

    marker = next(
        (index for index, line in enumerate(lines) if "radial grid follows" in line.lower()),
        None,
    )
    if marker is None:
        raise ValueError(f"missing radial grid marker in {source}")
    radii, _ = _read_values(lines, marker + 1, radial_count)
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")

    channel_potentials: dict[int, np.ndarray] = {}
    spin_orbit_channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    channel_occupations: dict[int, float] = {}
    channel_cutoffs: dict[int, float] = {}
    valence_radial = np.zeros(radial_count)
    core_radial = np.zeros(radial_count)

    for index, line in enumerate(lines):
        lower = line.lower()
        compact_marker = re.sub(r"[^a-z]", "", lower)
        if "minorcompofpseudopfollows" in compact_marker:
            # A relativistic (``irel='rel'``) file's raw spin-orbit
            # correction block, e.g. "Minor comp. of Pseudop. follows (l on
            # next line)".  Checked before the plain "pseudopotential
            # follows" channels below since this marker does not contain
            # that substring, but keeping the SO check first documents the
            # dependency explicitly.
            l_values = _numbers(lines[index + 1])
            if not l_values:
                raise ValueError(f"missing angular momentum after line {index + 1} in {source}")
            angular_momentum = int(l_values[0])
            radial_so, _ = _read_values(lines, index + 2, radial_count)
            spin_orbit_channel_potentials[angular_momentum] = radial_so / radii
        elif "pseudopotential follows" in lower:
            l_values = _numbers(lines[index + 1])
            if not l_values:
                raise ValueError(f"missing angular momentum after line {index + 1} in {source}")
            angular_momentum = int(l_values[0])
            radial_v, _ = _read_values(lines, index + 2, radial_count)
            channel_potentials[angular_momentum] = radial_v / radii
        elif "core charge follows" in lower:
            core_radial, _ = _read_values(lines, index + 1, radial_count)
        elif "valence charge follows" in lower:
            valence_radial, _ = _read_values(lines, index + 1, radial_count)
        elif "pseudowavefunctionfollows" in compact_marker:
            wave_header = _numbers(lines[index + 1])
            if len(wave_header) < 3:
                raise ValueError(f"invalid pseudo-wavefunction header in {source}")
            angular_momentum = int(wave_header[0])
            channel_occupations[angular_momentum] = float(wave_header[1])
            channel_cutoffs[angular_momentum] = float(wave_header[2])
            wave, _ = _read_values(lines, index + 2, radial_count)
            radial_wavefunctions[angular_momentum] = wave

    if len(channel_potentials) != number_of_channels:
        raise ValueError(
            f"{source} declares {number_of_channels} channels but contains "
            f"{len(channel_potentials)}"
        )
    missing_waves = sorted(
        set(channel_potentials).difference(radial_wavefunctions)
    )
    if missing_waves:
        channels = ", ".join(f"l={value}" for value in missing_waves)
        raise ValueError(
            f"{source} is missing pseudo-wavefunctions for potential "
            f"channels {channels}"
        )
    if len(spin_orbit_channel_potentials) != number_of_spin_orbit_channels:
        raise ValueError(
            f"{source} declares {number_of_spin_orbit_channels} spin-orbit "
            f"channels but contains {len(spin_orbit_channel_potentials)}"
        )
    invalid_so_channels = sorted(set(spin_orbit_channel_potentials) - {1, 2})
    if invalid_so_channels:
        raise ValueError(
            f"{source} has spin-orbit channels for unsupported l="
            f"{invalid_so_channels}; PARSEC only supports l=1 (p) and l=2 (d)"
        )

    radial_to_volume = 1.0 / (4.0 * np.pi * radii * radii)
    return ParsecPseudopotential(
        symbol=symbol,
        correlation=correlation,
        relativity=relativity,
        core_marker=core_marker,
        comment=lines[1].strip(),
        configuration=lines[2].strip(),
        number_of_channels=number_of_channels,
        number_of_spin_orbit_channels=number_of_spin_orbit_channels,
        radial_parameter_a=float(radial_parameter_a),
        radial_parameter_b=float(radial_parameter_b),
        ionic_charge=float(ionic_charge),
        radii=radii,
        channel_potentials=channel_potentials,
        valence_density=valence_radial * radial_to_volume,
        core_density=core_radial * radial_to_volume,
        radial_wavefunctions=radial_wavefunctions,
        channel_occupations=channel_occupations,
        channel_cutoffs=channel_cutoffs,
        source=source,
        spin_orbit_channel_potentials=spin_orbit_channel_potentials,
    )
