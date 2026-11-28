"""Spin-polarized (collinear) PARSEC-style SCF iteration of an isolated system.

``Spin_Polarization: true`` gives the calculation two spin channels.  The
loop of ``parsec.F90`` is that of the unpolarized case with these changes:

* each channel ``s`` has its own local potential
  ``V_s = V_ion + V_H[rho] + V_xc,s[rho_up, rho_down]`` and its own
  eigenproblem, solved with its own saved subspace (``eigval.F90``);
* one Fermi level fills both channels, a state holding one electron
  (``flevel.f90``);
* the density of a channel is ``rho_s = h**-3 sum_i f_s,i |psi_s,i|**2`` and
  the Hartree potential is that of ``rho = rho_up + rho_down``
  (``newrho.F90``, ``rho_hart.F90``);
* the exchange-correlation functional is the spin-polarized one
  (``exc_spn.f90``);
* the residual norms are evaluated per channel and the larger one is tested
  (``getsre.f90``);
* the mixer sees the two potentials as one vector, the up channel first
  (``spinmix.f90``, ``spinunmix.f90``);
* the band energy and the double-counting integrals are sums over the
  channels (``totnrg.f90``).

The first potentials come from the starting densities of
:func:`parsec_python.SCF.single_point.initial_spin_densities`.

The module also holds :func:`run_scf_spin_polarized`, the collinear
spin-polarized loop for periodic Gamma-point cells and for the spin-orbit
drivers (see the section of that name below); an isolated calculation without
spin-orbit coupling runs :func:`run_spin_scf`.
"""

from __future__ import annotations

from dataclasses import replace
import sys
import time
from typing import Callable, Union

import numpy as np

from ..Eigensolvers import (
    EigvalSettings,
    EigvalState,
    dynamic_diagonalization_tolerance,
    solve_eigval,
)
from ..Energy import spin_total_energy, total_energy_spin_polarized
from ..Grid import symmetrize_density
from ..Mixer import (
    make_mixer,
    potential_residual_metrics,
    preconditioned_mixer,
    spin_potential_residual_metrics,
)
from ..Occupations import (
    density_from_orbitals,
    fermi_occupations,
    spin_density_from_orbitals,
    spin_fermi_occupations,
    spin_fixed_occupations,
)
from ..models import (
    RunTimings,
    SCFIteration,
    SinglePointResult,
    SpinChannelResult,
    SpinPolarizedSCFIteration,
    SpinPolarizedSinglePointResult,
)
from ..Hamiltonian.hubbard import hubbard_operator, hubbard_terms_from_orbitals
from ..Hartree.variational import electrostatic_hartree
from .pbc import PeriodicPreparedSinglePointSystem
from .single_point import (
    PreparedSinglePointSystem,
    SCFWarmStart,
    _number_of_states,
    _public_states,
    dispersion_after_convergence,
    dispersion_wanted,
    eigval_settings_for,
    first_diagonalization_tolerance,
    hubbard_iteration_record,
    hubbard_terms_of_iteration,
    iteration_eigensolver_fields,
    loop_criterion,
    meta_gga_loop,
    preconditioner_neighbors,
    refine_levels,
    refuse_exchange_hooks,
    require_reference_hooks,
    warm_start_mixer,
)
from .state_extension import MultipletCloser, working_levels
from .filter_degree import LaterFilterDegree


SPIN_NAMES = ("up", "down")
#: Levels up to this far (Ry) above the Fermi level enter the convergence of a
#: start spectrum that a reduced adapter asks to solve again (test 1 of
#: acceleration/SCF/spin_sectors.py: the spread bound of its multiplets).
START_LEVEL_WINDOW = 3.0e-2
#: ``min(f, 1 - f)`` of a level of the up channel from which the converged loop of ``PARSEC_SPIN_FLIP_SYMMETRY``
#: is not kept (Fermi-Dirac occupations): the level and its image in the down channel hold the symmetric ensemble of
#: a partly filled pair, which two solved channels may fill unequally (test 3 of the spin sectors, for the pair of
#: channels that the operation exchanges; ``acceleration.SCF.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE``).
SPIN_FLIP_OCCUPATION_TOLERANCE = 0.05

# PARSEC_SPIN_FLIP_STABILITY=auto (_flip_stability): the test of the converged state of a loop of
# PARSEC_SPIN_FLIP_SYMMETRY=auto against perturbations that break the operation.
#: Largest change (Ry) of the down potential of one probe of the linear response.
FLIP_STABILITY_KICK = 5.0e-3
#: Probes (Arnoldi steps) of the response, at most.
FLIP_STABILITY_STEPS = 5
#: Largest eigenvalue of the response from which the state is unstable: an SCF map ``V -> V_out`` whose Jacobian
#: has an eigenvalue above 1 on the perturbations that break the operation has a direction of negative curvature
#: of the energy there (the second variation is ``<d rho, ((-chi_0)^-1 + K) d rho>``).
FLIP_STABILITY_BOUND = 1.0
#: Largest change (Ry) of a level up to the window above the Fermi level between two passes of the down channel's
#: eigensolver after which a probe (or the first solve) is taken as converged.
FLIP_STABILITY_LEVEL_TOLERANCE = 1.0e-8
#: Eigensolver passes of a probe at most, and of the first solve of the down channel.
FLIP_STABILITY_PASSES = 4
FLIP_STABILITY_FIRST_PASSES = 16
#: Seed of the random part of the first probe direction (the same on every process).
FLIP_STABILITY_SEED = 20261122


def _spin_occupations(eigenvalues, system, settings, weights=None, partners=None):
    """One Fermi level for both channels (or the occupations of ``occup.in``).

    ``weights``: the state weights of both channels where the symmetry
    sectors of a subgroup with weights solve them (one partner of some
    degenerate levels, acceleration/Symmetry/orbital_groups.py), ``None``
    otherwise.  The occupations returned are then per partner.
    ``partners``: with ``weights``, the partner masks of the states of both
    channels (``state_partners`` of the sector eigensolver, or ``None``),
    which ``occup.in`` must fill alike where they share a level.
    """

    keywords = {} if weights is None else {"weights": weights}
    if settings.fixed_occupations is None:
        return spin_fermi_occupations(
            eigenvalues[0],
            eigenvalues[1],
            system.electron_count,
            settings.fermi_temperature_kelvin,
            **keywords,
        )
    if weights is not None and partners is not None and any(item is not None for item in partners):
        keywords["partners"] = tuple(partners)
    return spin_fixed_occupations(
        eigenvalues[0],
        eigenvalues[1],
        system.electron_count,
        settings.fixed_occupations,
        settings.fixed_occupations_down,
        **keywords,
    )


def _channel_state_weights(weights):
    """Both channels' state weights, or ``None`` where the channels have none (the full grid, abelian sectors)."""

    if weights[0] is None and weights[1] is None:
        return None
    if weights[0] is None or weights[1] is None:
        raise RuntimeError("the two spin channels were solved in sectors of different kinds")
    return (np.asarray(weights[0], dtype=np.float64), np.asarray(weights[1], dtype=np.float64))


def _weighted_occupations(occupation_result, weights):
    """The occupations of the loop: per partner times the weights (electrons per solved state)."""

    occupations = list(occupation_result.occupations)
    if weights is None:
        return occupations
    return [np.asarray(filled, dtype=np.float64) * weight for filled, weight in zip(occupations, weights)]


def _channel_solvers(eigenproblem_solver) -> tuple[Callable[..., object], ...]:
    """Return one eigenproblem solver per spin channel.

    The readable :func:`solve_eigval` carries its saved subspace in the
    state it is handed, so one function serves both channels.  A substituted
    solver that keeps a subspace of its own must say how it serves a second
    channel, through ``for_spin_channel(index)``; one that does not is
    refused rather than run with the subspace of the wrong channel.
    """

    if eigenproblem_solver is None:
        return (solve_eigval, solve_eigval)
    for_channel = getattr(eigenproblem_solver, "for_spin_channel", None)
    if not callable(for_channel):
        raise NotImplementedError(
            "the eigensolver of this execution backend keeps one saved "
            "subspace and cannot serve the two channels of a spin-polarized "
            f"calculation ({type(eigenproblem_solver).__name__})"
        )
    return (for_channel(0), for_channel(1))


class SpinOrbitalBlocks:
    """Orbitals of two spin channels that an execution backend keeps apart.

    A backend may hand the SCF loop orbitals that live on a device.  Joining
    them there would hold every orbital twice, so the result keeps the two
    blocks as they are and joins them where they are wanted:
    :meth:`to_full_host` returns one host array with the columns of the up
    channel followed by those of the down channel.
    """

    def __init__(self, up, down) -> None:
        self.blocks = (up, down)

    @property
    def shape(self) -> tuple[int, int]:
        up, down = self.blocks
        return (int(up.shape[0]), int(up.shape[1]) + int(down.shape[1]))

    @property
    def ndim(self) -> int:
        return 2

    def to_full_host(self) -> np.ndarray:
        """Download both blocks into one host array, up first.

        A block that exports itself (``to_full_host``: the orbitals of the
        symmetry sectors of an accelerated loop) is asked to.
        """

        host = np.empty(self.shape, dtype=np.float64)
        column = 0
        for index, block in enumerate(self.blocks):
            fill = getattr(block, "fill_from", None)
            if index == 1 and callable(fill) and getattr(block, "source", None) is self.blocks[0]:
                # The down channel as the image of the up channel
                # (PARSEC_SPIN_FLIP_SYMMETRY): from its columns just written.
                count = int(self.blocks[0].shape[1])
                fill(host[:, :count], host[:, column : column + count])
                column += count
                continue
            export = getattr(block, "to_full_host", None)
            download = getattr(block, "get", None)
            values = (
                export()
                if callable(export)
                else download()
                if callable(download)
                else np.asarray(block)
            )
            host[:, column : column + values.shape[1]] = values
            column += values.shape[1]
        return host

    def to_full_device(self):
        """Join the two blocks with the array library they belong to."""

        up, down = self.blocks
        if callable(getattr(up, "to_full_device", None)):
            # The orbitals of the symmetry sectors: each channel expands itself.
            up, down = up.to_full_device(), down.to_full_device()
        elif callable(getattr(down, "to_full_device", None)):
            # The down channel as the image of the up channel's device matrix
            # (PARSEC_SPIN_FLIP_SYMMETRY).
            down = down.to_full_device()
        library = sys.modules.get(type(up).__module__.split(".")[0])
        stack = getattr(library, "hstack", None)
        if stack is None:
            raise TypeError(f"cannot join orbital blocks of type {type(up).__name__}")
        return stack((up, down))


def _join_orbitals(up, down):
    """Return the orbitals of both channels and a view of each channel.

    Host arrays are joined column-wise, up first, and the channels are views
    of the joined array.  Other arrays stay apart in
    :class:`SpinOrbitalBlocks` and are the channels themselves.
    """

    if isinstance(up, np.ndarray) and isinstance(down, np.ndarray):
        joined = np.hstack((up, down))
        count = up.shape[1]
        return joined, (joined[:, :count], joined[:, count:])
    return SpinOrbitalBlocks(up, down), (up, down)


def _public_channels(eigenvalues, occupations, representations, wavefunctions, state_weights) -> list[tuple]:
    """:func:`_public_states` of both channels.

    A down channel that is the image of the up channel (PARSEC_SPIN_FLIP_SYMMETRY) is rebound to the up orbitals of
    the result, which carry the weights: its columns are then the images of every partner, one per listed level.
    """

    public = [
        _public_states(
            eigenvalues[spin], occupations[spin], representations[spin], wavefunctions[spin], state_weights[spin]
        )
        for spin in range(2)
    ]
    rebind = getattr(wavefunctions[1], "with_source", None)
    if (
        callable(rebind)
        and getattr(wavefunctions[1], "source", None) is wavefunctions[0]
        and public[0][3] is not wavefunctions[0]
    ):
        public[1] = (*public[1][:3], rebind(public[0][3]))
    return public


def _spin_flip_of(system, *, exchange_term, meta, hubbard_projectors, ionic_potential):
    """``(operation, refusal)``: the spin-flip operation the loop takes the down channel from, or why not.

    ``system.spin_flip`` is attached by the accelerated driver with
    ``PARSEC_SPIN_FLIP_SYMMETRY=auto`` or ``late``
    (:mod:`parsec_python.acceleration.SCF.spin_flip`).  It is not taken with
    an exchange operator, a meta-GGA, LDA+U or ``Symmetrize_Charge_Density``,
    whose terms read the orbitals or fields of each channel, nor where it no
    longer maps the atoms of this structure onto their partners or the local
    ionic potential of this loop (``ionic_potential``, as the loop keeps it:
    point charges, the field of a ``Polarizability`` loop) onto itself.
    ``refusal`` is ``None`` where the system has no operation or the loop
    takes it, else ``(reason, measure, value, bound)`` for the note of
    :func:`_spin_flip_record`.
    """

    flip = getattr(system, "spin_flip", None)
    if flip is None or getattr(system, SPIN_FLIP_LEFT, None):
        # (a system whose run left the route, leave_spin_flip: its note was given then)
        return None, None
    for condition, reason in (
        (exchange_term is not None, "an exchange operator (a hybrid functional)"),
        (meta is not None, "a meta-GGA"),
        (hubbard_projectors is not None, "LDA+U"),
        (getattr(system, "density_symmetry", None) is not None, "Symmetrize_Charge_Density"),
    ):
        if condition:
            return None, (reason, "", float("nan"), float("nan"))
    deviation_of = getattr(flip, "atom_deviation", None)
    if callable(deviation_of):
        deviation = deviation_of(system.atoms)
        if not deviation <= flip.tolerance:
            return None, (
                f"the operation {flip.description} no longer carries the atoms of this structure onto their partners",
                "largest distance [bohr]",
                deviation,
                float(flip.tolerance),
            )
    check = getattr(flip, "ionic_potential_check", None)
    if callable(check):
        deviation, bound = check(ionic_potential)
        if not deviation <= bound:
            return None, (
                f"the operation {flip.description} does not map the local ionic potential of this loop onto itself "
                "(point charges, the field of a Polarizability loop or another external term)",
                "largest deviation [Ry]",
                deviation,
                bound,
            )
    return flip, None


#: The attribute of a prepared system whose run left ``PARSEC_SPIN_FLIP_SYMMETRY`` (:func:`leave_spin_flip`): why.
SPIN_FLIP_LEFT = "spin_flip_left"


def leave_spin_flip(system, reason: str) -> None:
    """Solve both channels in every later loop of ``system``: the run left the spin-flip route for ``reason``.

    The mark stays on the system object, through its later structures and the objects that the spin guard of the
    accelerated driver prepares again in its place (``driver._SpinSectorFallback``, which calls this again to correct
    the backend detail ``spin_flip_symmetry`` of what it prepared); the detail says so from here on.
    """

    try:
        setattr(system, SPIN_FLIP_LEFT, str(reason))
    except (AttributeError, TypeError):
        return
    info = getattr(system, "backend_info", None)
    details = getattr(info, "details", None)
    if details is None or not any(item[0] == "spin_flip_symmetry" for item in details):
        return
    mode = getattr(getattr(system, "spin_flip", None), "mode", "auto")
    text = f"{mode}: both channels solved (the run left the route: {reason})"
    info = replace(
        info, details=tuple(("spin_flip_symmetry", text) if item[0] == "spin_flip_symmetry" else item for item in details)
    )
    try:
        system.backend_info = info
        backend = getattr(system, "backend", None)
        if backend is not None:
            backend.info = info
    except (AttributeError, TypeError):
        pass


def _sector_guard(error) -> bool:
    """Whether ``error`` is a failed test of the spin sectors of the accelerated driver
    (``acceleration.SCF.spin_sectors.SpinSectorsAbandoned``)."""

    try:
        from parsec_python.acceleration.SCF.spin_sectors import SpinSectorsAbandoned
    except ImportError:
        return False
    return isinstance(error, SpinSectorsAbandoned)


class _SpinFlipFractional(Exception):
    """The converged loop of ``PARSEC_SPIN_FLIP_SYMMETRY`` holds a fractional occupation in its up channel
    (:data:`SPIN_FLIP_OCCUPATION_TOLERANCE`); :func:`run_spin_scf` solves the loop again with both channels."""

    def __init__(self, largest: float, lines: tuple[str, ...]) -> None:
        super().__init__(f"largest min(f, 1-f) of the up channel {largest:.3e}")
        self.largest = float(largest)
        self.lines = tuple(lines)


def _flip_fractional_occupations(eigenvalues, occupations) -> tuple[float, tuple[str, ...]]:
    """The largest ``min(f, 1 - f)`` of the up channel of a flip loop and the levels that reach the tolerance."""

    values = np.asarray(occupations, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, ()
    measure = np.minimum(values, 1.0 - values)
    levels = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    lines = tuple(
        f"up level {int(index) + 1}: {levels[index]:.6f} Ry, occupation {values[index]:.4f}"
        for index in np.flatnonzero(measure >= SPIN_FLIP_OCCUPATION_TOLERANCE)[:8]
        if index < levels.size
    )
    return float(np.max(measure)), lines


def _spin_flip_record(refusal):
    """The note (:class:`parsec_python.models.SpinSectorFallbackRecord`, ``route`` ``spin_flip``) of a loop that
    solves both channels although the prepared system has a spin-flip operation."""

    from ..models import SpinSectorFallbackRecord

    reason, measure, value, bound = refusal
    return SpinSectorFallbackRecord(
        stage="the start of the SCF loop",
        largest_fractional_occupation=float(value),
        tolerance=float(bound),
        reason=reason,
        measure=measure,
        route="spin_flip",
    )


def _spin_flip_fractional_record(broken: _SpinFlipFractional):
    """The note of a flip loop whose up channel ended with a fractional occupation: the loop starts again from its
    start and solves both channels."""

    from ..models import SpinSectorFallbackRecord

    return SpinSectorFallbackRecord(
        stage="the end of the SCF loop",
        largest_fractional_occupation=broken.largest,
        tolerance=SPIN_FLIP_OCCUPATION_TOLERANCE,
        multiplets=broken.lines,
        reason=(
            "the up channel of the converged loop holds a fractional occupation, a level that the operation shares "
            "with the down channel and that two solved channels may fill unequally; the loop starts again from its "
            "start"
        ),
        measure="largest min(f, 1-f) of the up channel",
        route="spin_flip",
    )


def _spin_flip_guard_record(error):
    """The note of a flip loop in which a test of the spin sectors failed (``error``, a ``SpinSectorsAbandoned``):
    the loop starts again from its start with both channels, in the sectors it started in."""

    from ..models import SpinSectorFallbackRecord

    report = error.report
    return SpinSectorFallbackRecord(
        stage=str(error.stage),
        largest_fractional_occupation=float(report.largest),
        tolerance=float(report.tolerance),
        multiplets=tuple(report.lines),
        reason=(
            f"a test of the spin sectors failed in a loop of the route ({report.reason} after {error.stage}); the "
            "two channels held the same levels by the operation, not as two solved channels hold them, so the loop "
            "starts again from its start in the same sectors before the guard may leave them"
        ),
        measure=str(report.measure),
        route="spin_flip",
    )


class FlipStabilityReport:
    """What the stability test of a converged flip loop found (:func:`_flip_stability`).

    ``largest`` is the largest real part of the Ritz values of the response after the last probe, ``history`` that
    after every probe, ``ritz`` the real parts of the last Ritz values (largest first); ``probes`` the probes made,
    ``passes`` the eigensolver passes of the down channel in all, ``first_deviation`` the largest difference of its
    levels at the converged potential from those of the up channel (Ry, up to the window above the Fermi level),
    ``seconds`` the time of the test, ``first_passes`` and ``first_seconds`` those of the first solve of the down
    channel at the converged potential.
    """

    def __init__(
        self, *, largest, history, ritz, probes, passes, first_deviation, seconds, bound, first_passes=0,
        first_seconds=0.0,
    ):
        self.largest = float(largest)
        self.history = tuple(float(value) for value in history)
        self.ritz = tuple(float(value) for value in ritz)
        self.probes = int(probes)
        self.passes = int(passes)
        self.first_deviation = float(first_deviation)
        self.seconds = float(seconds)
        self.bound = float(bound)
        self.first_passes = int(first_passes)
        self.first_seconds = float(first_seconds)

    @property
    def unstable(self) -> bool:
        return not self.largest < self.bound

    def text(self) -> str:
        return (
            f"largest eigenvalue {self.largest:.4f} of the SCF response to perturbations that break the operation "
            f"(unstable from {self.bound:g}; {self.probes} probes, Ritz values "
            + " ".join(f"{value:.3f}" for value in self.ritz[:4])
            + f"; {self.seconds:.2f} s, of which {self.first_seconds:.2f} s for the {self.first_passes} passes of the "
            f"first solve of the down channel, its levels {self.first_deviation:.1e} Ry from the up channel's; "
            f"{self.passes} passes in all)"
        )


class _SpinFlipUnstable(Exception):
    """The converged loop of ``PARSEC_SPIN_FLIP_SYMMETRY`` is unstable against perturbations that break the
    operation (``PARSEC_SPIN_FLIP_STABILITY=auto``); :func:`run_spin_scf` solves the loop again with both channels."""

    def __init__(self, report: FlipStabilityReport) -> None:
        super().__init__(report.text())
        self.report = report


def _spin_flip_unstable_record(report: FlipStabilityReport):
    """The note of a flip loop whose state the stability test found unstable: the loop starts again from its start
    and solves both channels."""

    from ..models import SpinSectorFallbackRecord

    return SpinSectorFallbackRecord(
        stage="the stability test after the SCF loop",
        largest_fractional_occupation=report.largest,
        tolerance=report.bound,
        multiplets=(report.text(),),
        reason=(
            "the state of the converged loop is unstable against perturbations that break the operation (an "
            "eigenvalue of their SCF response above 1: two solved channels move away from it); the loop starts again "
            "from its start"
        ),
        measure="largest eigenvalue of the response",
        route="spin_flip",
    )


def _spin_flip_stable_record(report: FlipStabilityReport):
    """The note of a flip loop whose state the stability test kept (``route`` ``spin_flip_stable``)."""

    from ..models import SpinSectorFallbackRecord

    return SpinSectorFallbackRecord(
        stage="the stability test after the SCF loop",
        largest_fractional_occupation=report.largest,
        tolerance=report.bound,
        multiplets=(report.text(),),
        reason="the state of the converged loop is stable against perturbations that break the operation",
        measure="largest eigenvalue of the response",
        route="spin_flip_stable",
    )


class _SpinFlipContinue(Exception):
    """A start lies below the converged state of a loop of ``PARSEC_SPIN_FLIP_SYMMETRY``
    (``PARSEC_SPIN_FLIP_STABILITY=auto`` with ``PARSEC_SPIN_FLIP_CONTINUATION=auto``,
    :func:`parsec_python.acceleration.SCF.spin_flip_stability.flip_state_safety`): :func:`run_spin_scf` continues
    without the flip from it.  ``start`` and ``centre`` are the full-grid input potentials ``(up, down)`` of the start
    and of the state of the route, ``hartree`` the last Hartree potential, ``states`` the eigensolver states of the
    two channels (the up channel at the state of the route, the down channel at the start), ``reference`` the free
    energy of the state of the route."""

    def __init__(self, report, *, start, centre, hartree, states, reference) -> None:
        super().__init__(report.candidates_text())
        self.report = report
        self.start = start
        self.centre = centre
        self.hartree = hartree
        self.states = states
        self.reference = float(reference)


def _spin_flip_safety_record(report, *, route: str, reason: str, summary: bool = True):
    """A note of the test and the starts of :mod:`parsec_python.acceleration.SCF.spin_flip_stability` (``route``
    ``spin_flip_stable``: the state of the route is kept; ``spin_flip_continued``: the loop continues without the flip
    from a start below it, or, without ``summary``, what that continuation ended in)."""

    from ..models import SpinSectorFallbackRecord

    return SpinSectorFallbackRecord(
        stage="the stability test after the SCF loop",
        largest_fractional_occupation=float(report.largest),
        tolerance=float(report.bound),
        multiplets=(report.test_text(), report.candidates_text()) if summary else (),
        reason=reason,
        measure="largest eigenvalue of the response",
        route=route,
    )


def _result_free_energy(result, temperature: float) -> float:
    """``E - TS`` of a spin result without its dispersion term (the footing of the loop's iterations)."""

    from parsec_python.acceleration.SCF.spin_flip_stability import entropy_term

    dispersion = getattr(result, "dispersion", None)
    energy = float(result.energies.total) - float(getattr(dispersion, "energy", 0.0) or 0.0)
    return energy + entropy_term([channel.occupations for channel in result.spin], None, temperature)


def _continue_without_flip(system, keywords: dict, found: _SpinFlipContinue, callback):
    """The loop without the flip from the start below the state of the route (``found``), stopped by a
    :class:`parsec_python.acceleration.SCF.noncollinear_stability.ContinuationWatch`; the lower converged state is the
    result.  Where it does not end :data:`parsec_python.acceleration.SCF.spin_flip_stability.MARGIN` below the state of
    the route, that state is converged again from its potential with the flip (the continuation moved the subspace of
    the up channel on) and kept."""

    from parsec_python.acceleration.SCF.noncollinear_stability import ContinuationWatch
    from parsec_python.acceleration.SCF.spin_flip_stability import MARGIN

    report = found.report
    best = report.best
    settings = system.input.scf
    temperature = float(settings.fermi_temperature_kelvin)
    info = getattr(system, "backend_info", None)
    why = f"a start {report.gain:.3e} Ry below the state of the route ({best.describe()})"
    leave_spin_flip(system, why)
    if callback is not None:
        callback(
            _spin_flip_safety_record(
                report,
                route="spin_flip_continued",
                reason=(
                    f"the state of the converged loop lies {report.gain:.3e} Ry above the free energy of one SCF step "
                    f"at a start that breaks the operation ({best.describe()}); the loop continues from it without "
                    "the flip, and the lower converged state is kept"
                ),
            )
        )
    watch = ContinuationWatch(found.reference, criterion=float(settings.convergence_criterion))
    original = keywords.get("warm_start")
    warm = SCFWarmStart(input_potential=found.start, hartree_potential=found.hartree, eigensolver_state=found.states)
    if original is not None:
        warm = replace(warm, level_tolerance=getattr(original, "level_tolerance", None))
    started = time.perf_counter()
    continued = _run_spin_loop(system, **{**keywords, "warm_start": warm}, give_up=watch)
    seconds = time.perf_counter() - started
    free = _result_free_energy(continued, temperature)
    below = found.reference - free
    detail = {
        "spin_flip_stability_largest": float(report.largest),
        "spin_flip_stability_probes": float(report.probes),
        "spin_flip_stability_seconds": float(report.seconds),
        "spin_flip_candidates_seconds": float(report.candidate_seconds),
        "spin_flip_candidate_gain": float(report.gain),
        "spin_flip_down_passes": float(report.passes),
        "spin_flip_continuation_iterations": float(continued.iterations),
        "spin_flip_continuation_seconds": float(seconds),
        "spin_flip_continuation_lower": float(below),
    }
    if watch.stopped_at is not None:
        detail["spin_flip_continuation_stopped"] = float(watch.stopped_at)
    if continued.converged and below > MARGIN:
        continued.timings.detail.update(detail)
        if callback is not None:
            callback(
                _spin_flip_safety_record(
                    report,
                    route="spin_flip_continued",
                    reason=(
                        f"the loop without the flip converged in {continued.iterations} iterations to a state "
                        f"{below:.6e} Ry below the state of the route (free energy), which is kept"
                    ),
                    summary=False,
                )
            )
        return continued
    if not continued.converged:
        outcome = (
            f"stopped after {watch.stopped_at} iterations ({watch.reason})"
            if watch.stopped_at is not None
            else f"did not converge in {continued.iterations} iterations"
        )
    else:
        outcome = f"converged {-below:.3e} Ry above it"
    del continued
    # The state of the route again: its potential with the flip, from the subspace the up channel holds now.
    try:
        delattr(system, SPIN_FLIP_LEFT)
    except AttributeError:
        pass
    if info is not None:
        try:
            system.backend_info = info
            backend = getattr(system, "backend", None)
            if backend is not None:
                backend.info = info
        except (AttributeError, TypeError):
            pass
    solvers = _channel_solvers(keywords.get("eigenproblem_solver"))
    restart = getattr(solvers[0], "start_new_scf_loop", None)
    up_state = restart() if callable(restart) else None
    restore = SCFWarmStart(input_potential=found.centre, hartree_potential=found.hartree, eigensolver_state=(up_state, None))
    if original is not None:
        restore = replace(restore, level_tolerance=getattr(original, "level_tolerance", None))
    started = time.perf_counter()
    restored_taken: list[str] = []
    try:
        restored = _run_spin_loop(
            system, **{**keywords, "warm_start": restore}, flip_test=False, flip_taken=restored_taken
        )
    except Exception as error:
        # (a test of the spin sectors that fails in the loop of the route: the route of off from the start, as in
        # run_spin_scf)
        if not restored_taken or not _sector_guard(error):
            raise
        return _rerun_without_flip(
            system,
            keywords,
            _spin_flip_guard_record(error),
            f"a test of the spin sectors failed in a flip loop after {error.stage} ({error.report.reason})",
        )
    detail["spin_flip_restored_iterations"] = float(restored.iterations)
    detail["spin_flip_restored_seconds"] = float(time.perf_counter() - started)
    restored.timings.detail.update(detail)
    if callback is not None:
        callback(
            _spin_flip_safety_record(
                report,
                route="spin_flip_continued",
                reason=(
                    f"the loop without the flip {outcome}: the state of the route is kept, converged again from its "
                    f"potential in {restored.iterations} iterations"
                ),
                summary=False,
            )
        )
    return restored


#: PARSEC_SPIN_FLIP_SYMMETRY=late: the residual (Ry, that the loop tests) from which a loop that solves both channels
#: takes the flip, where its channels are images of each other within the residual and no level is partly filled
#: (:func:`_late_flip_switch`).
LATE_FLIP_RESIDUAL = 1.0e-3
#: PARSEC_SPIN_FLIP_SYMMETRY=late: how far (a multiple of the residual, in its norm) the input potentials of the two
#: channels may be from images of each other under the operation of the route at the switch.  Measured on A100 at the
#: iterations where the levels allowed the switch: 0.4-2.1 times the residual for H2, Cr2, Cr4, Mn2, Fe2, MnO, the B2
#: Cr cell and the H4 square, 2.8-4.7 for NiO AFM-II, whose part that breaks the operation decays more slowly than the
#: residual; 100-1000 for Mn2 compressed by 0.25 bohr, whose loop of off ends in a state of another operation.
LATE_FLIP_POTENTIAL_FACTOR = 10.0
#: PARSEC_SPIN_FLIP_SYMMETRY=late: what the history of the mixer holds after the switch: ``reset`` (none: the mixer
#: starts again with the inputs of the route), ``project`` (every input and residual it keeps made a pair of the route,
#: :func:`_project_mixer_history`) or ``keep`` (as the iterations before left it, which cost 8-13 more iterations on
#: Cr2, Cr4 and H2).  Iterations on A100 against off, reset / project: 28 / 35 (37) Cr2, 31 / 40 (28) Cr2 stretched
#: by 0.2 bohr, 25 / 31 (28) H2 stretched by 1.5 bohr, 14 / 20 (20) the H4 square, 20 / 23 (26) the B2 Cr cell, 49 / 46
#: (65) Mn2, 42 / 39 (40) Fe2, on the full grid 32 / 27 (26) H2, 33 / 34 (31) Cr2, 34 / 34 (34) Cr4.
LATE_FLIP_MIXER = "reset"


def _late_flip_switch(eigenvalues, occupations, fermi_level, residual, settings, history=None, *, deviation) -> bool:
    """Whether a loop of ``PARSEC_SPIN_FLIP_SYMMETRY=late`` that solves both channels takes the flip after this
    iteration.  Its residual must be at most :data:`LATE_FLIP_RESIDUAL`; the levels of the two channels up to
    ``START_LEVEL_WINDOW`` above the Fermi level the same within the residual (sorted, as the operation maps the
    levels of one channel onto those of the other) and no further apart than two iterations before (``history``, the
    list of that difference by iteration, which this call extends: an asymmetry that grows from the noise of the
    eigensolvers, as along an unstable direction, is not taken for noise); no level partly filled (``min(f, 1 - f)``
    below :data:`SPIN_FLIP_OCCUPATION_TOLERANCE`, which the route needs at its end); and the input potentials images of
    each other under the operation of the route within :data:`LATE_FLIP_POTENTIAL_FACTOR` times the residual, in the
    norm of the residual (``deviation()``, of ``V_down - g V_up``: :func:`_flip_potential_deviation`).  Equal spectra alone do not tell which operation relates
    the channels: Mn2 compressed by 0.25 bohr on the full grid ends, in the loop of off, in a state whose down density
    is the image of its up density under an S4 rotoreflection, 1.41 electrons from its image under the inversion of
    the route, although the levels of its channels agree to 4e-8 Ry.  The state the loop of off then converges to is,
    within its residual, one of the operation: from there the route converges to it with half the eigensolves.  Where
    the channels stay apart (Ni2 and Co2 from antiparallel starts, Mn2 compressed, whose loops of off end in states
    that break the operation), the loop stays that of off."""

    up = np.sort(np.asarray(eigenvalues[0], dtype=np.float64))
    down = np.sort(np.asarray(eigenvalues[1], dtype=np.float64))
    count = min(up.size, down.size)
    if count == 0:
        return False
    window = float(fermi_level) + START_LEVEL_WINDOW
    shown = (up[:count] <= window) | (down[:count] <= window)
    asymmetry = float(np.max(np.abs(up[:count] - down[:count])[shown], initial=0.0))
    if history is not None:
        history.append(asymmetry)
        if len(history) >= 3 and not asymmetry <= history[-3]:
            return False
    if not float(residual) <= LATE_FLIP_RESIDUAL:
        return False
    if not asymmetry <= float(residual):
        return False
    if settings.fixed_occupations is None:
        for filled in occupations:
            values = np.asarray(filled, dtype=np.float64).reshape(-1)
            if values.size and float(np.max(np.minimum(values, 1.0 - values))) >= SPIN_FLIP_OCCUPATION_TOLERANCE:
                return False
    return bool(float(deviation()) <= LATE_FLIP_POTENTIAL_FACTOR * float(residual))


def _flip_potential_deviation(flip, potentials, densities, electrons, metrics_of, volume_element, plain: bool) -> float:
    """The norm of ``V_down - g V_up`` (``potentials``, the inputs of an iteration) in the norm of the residual of the
    loop (``metrics_of``, its residual evaluator; weighted by the down density and electrons, or plain): how far the
    two channels are from a pair of the route of ``flip``."""

    image = flip.down_field(potentials[0])
    metrics = metrics_of(
        (image, image), (potentials[1], potentials[1]), (densities[1], densities[1]), volume_element,
        (electrons[1], electrons[1]),
    )
    return float(metrics.plain if plain else metrics.weighted)


def _flip_density_deviation(flip, densities, adapter, volume_element) -> float:
    """``integral |rho_down - g rho_up|`` (electrons): how far the densities of the two channels are from images of each
    other under the operation of ``flip`` (orbit values weighted by their multiplicities in the sectors)."""

    image = np.asarray(getattr(flip.down_field(densities[0]), "values", flip.down_field(densities[0])), dtype=np.float64)
    down = np.asarray(getattr(densities[1], "values", densities[1]), dtype=np.float64)
    difference = np.abs(image - down)
    if adapter is not None:
        difference = difference * np.asarray(adapter.weights, dtype=np.float64)
    return float(volume_element * np.sum(difference))


def _flip_symmetric(flip, up, down, adapter):
    """``(u, g u)`` with ``u = (up + g^-1 down) / 2``: the pair of potentials of the route nearest to ``(up, down)``
    (orbit fields of ``adapter`` in the sectors)."""

    if adapter is not None:
        values = 0.5 * (
            np.asarray(getattr(up, "values", up), dtype=np.float64)
            + np.asarray(getattr(flip.up_field(down), "values", None), dtype=np.float64)
        )
        symmetric = adapter.field(values)
    else:
        symmetric = 0.5 * (np.asarray(up, dtype=np.float64) + np.asarray(flip.up_field(down), dtype=np.float64))
    return symmetric, flip.down_field(symmetric)


def _project_mixer_history(mixer, flip) -> bool:
    """Every input and residual that the mixer keeps (both channels end to end, up first) made a pair of the route of
    ``flip``, ``(u, g u)`` with ``u = (up + g^-1 down) / 2`` (the Anderson mixers of the full grid and of the sectors,
    also inside the Kerker preconditioner); false where the mixer keeps no such history."""

    target = mixer
    for _ in range(4):
        if target is None:
            return False
        inputs = getattr(target, "_inputs", None)
        residuals = getattr(target, "_residuals", None)
        if isinstance(inputs, list) and isinstance(residuals, list):
            break
        target = getattr(target, "inner", None)
    else:
        return False

    def projected(vector):
        values = np.asarray(vector, dtype=np.float64)
        half = values.size // 2
        up = values[:half]
        symmetric = 0.5 * (up + np.asarray(flip.up_field(values[half:]), dtype=np.float64))
        return np.concatenate((symmetric, np.asarray(flip.down_field(symmetric), dtype=np.float64)))

    for index, vector in enumerate(inputs):
        inputs[index] = projected(vector)
    for index, vector in enumerate(residuals):
        residuals[index] = projected(vector)
    return True


def _late_flip_record(flip, *, switched: bool, iteration: int, lines):
    """The note (:class:`parsec_python.models.SpinSectorFallbackRecord`, ``route`` ``spin_flip_late``) of a loop of
    ``PARSEC_SPIN_FLIP_SYMMETRY=late``: from which iteration on it takes the route, or that its channels stayed apart
    and it is the loop of off."""

    from ..models import SpinSectorFallbackRecord

    if switched:
        reason = (
            f"the two solved channels are images of each other under {flip.description} within the residual after "
            f"iteration {iteration}: from the next iteration on the up channel alone, the down channel its image"
        )
        stage = f"iteration {iteration} of the SCF loop"
    else:
        reason = (
            f"the two solved channels did not become images of each other under {flip.description} within the "
            f"residual: both channels solved in all {iteration} iterations (the loop of off)"
        )
        stage = "the end of the SCF loop"
    return SpinSectorFallbackRecord(
        stage=stage,
        largest_fractional_occupation=float("nan"),
        tolerance=float(LATE_FLIP_RESIDUAL),
        multiplets=tuple(lines),
        reason=reason,
        measure="",
        route="spin_flip_late",
    )


def _mirrored_levels(base, base_labels, levels, labels) -> np.ndarray | None:
    """The levels of the up channel under the opposite perturbation, to first order: ``2 e0 - e`` sector by sector
    (``base`` and ``levels`` the down channel's levels at the converged and at the perturbed potential, ``labels``
    their sectors); ``None`` where a sector holds different numbers of levels in the two."""

    base = np.asarray(base, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    base_labels = np.asarray(base_labels)
    labels = np.asarray(labels)
    if base.shape != levels.shape or base_labels.shape != base.shape or labels.shape != levels.shape:
        return None
    mirrored = np.empty_like(levels)
    for label in np.unique(labels):
        before = np.flatnonzero(base_labels == label)
        after = np.flatnonzero(labels == label)
        if before.size != after.size:
            return None
        before = before[np.argsort(base[before], kind="stable")]
        after = after[np.argsort(levels[after], kind="stable")]
        mirrored[after] = 2.0 * base[before] - levels[after]
    return mirrored


def _flip_stability(
    system,
    flip,
    *,
    solve,
    number_of_states: int,
    eigensolver_settings,
    neighbors,
    settings,
    up_potential,
    up_eigenvalues,
    fermi_level: float,
    ionic_potential,
    hartree_potential,
    reduced: bool,
    adapter,
    orbital_density_builder,
    volume_element: float,
) -> FlipStabilityReport:
    """The stability of the converged state of a flip loop against perturbations that break the operation
    (``PARSEC_SPIN_FLIP_STABILITY=auto``).

    At a state of the route the down potential is the image of the up potential, ``d = g u``.  A perturbation
    ``(du, dd)`` breaks the operation where ``w = dd - g du`` is not zero; the SCF map ``(u, d) -> (U, D)`` commutes
    with the operation, so ``w_out = dD - g dU`` is a linear map ``J`` of ``w`` to first order, whose eigenvalues
    decide: the second variation of the energy along such perturbations is negative in some direction where ``J``
    has an eigenvalue above 1 (:data:`FLIP_STABILITY_BOUND`), and two solved channels move away from the state.

    The response is probed with the down channel alone, which the route does not solve otherwise: the up channel
    keeps its subspace and the orbitals of the result.  A probe ``dd = s v`` (``max |dd|`` =
    :data:`FLIP_STABILITY_KICK`) with ``du = -g^-1 dd`` has ``w = 2 s v``; the down channel is solved at ``g u + dd``
    (the levels of the up channel then follow from the operation, ``2 e0 - e`` to first order, for the common Fermi
    level), its density change ``d rho_down`` gives ``d rho_up = -g^-1 d rho_down`` by the operation, and the
    Hartree and exchange-correlation potentials of both densities give ``J v = w_out / (2 s)``.  The first solve
    of the down channel is at the converged potential (its levels are those of the up channel); the Ritz values of
    :data:`FLIP_STABILITY_STEPS` Arnoldi steps on ``J`` (inner product of the grid or of the orbits) give the
    largest eigenvalue.  The first probe direction mixes the moment density, its magnitude (a field on the moments)
    and a seeded random field times the density.
    """

    started = time.perf_counter()
    if reduced:
        field_of = adapter.field
        weights = np.asarray(adapter.weights, dtype=np.float64)

        def values_of(field):
            return np.asarray(getattr(field, "values", field), dtype=np.float64)

    else:
        weights = None

        def field_of(values):
            return values

        def values_of(field):
            return np.asarray(field, dtype=np.float64)

    def dot(left, right) -> float:
        return float(np.dot(left * weights, right)) if weights is not None else float(np.dot(left, right))

    def operator(potential):
        hamiltonian = system.hamiltonian(potential)
        factory = getattr(hamiltonian, "as_eigensolver_operator", hamiltonian.as_linear_operator)
        return factory()

    def density(vectors, occupations):
        if orbital_density_builder is None:
            return spin_density_from_orbitals(vectors, occupations, volume_element)
        built = orbital_density_builder(vectors, 0.5 * occupations, volume_element)
        return built if reduced else np.asarray(built, dtype=np.float64)

    probe_settings = eigval_settings_for(eigensolver_settings, eigensolver_settings.filter_degree, neighbors=neighbors)
    window = float(fermi_level) + START_LEVEL_WINDOW
    passes_total = 0
    vectors = None

    def solve_down(potential, state, previous, passes):
        """Passes of the down channel at ``potential`` until no level up to the window moves by more than the
        tolerance; returns the last solution."""

        nonlocal vectors, passes_total
        hamiltonian_operator = operator(potential)
        solution = None
        for _ in range(passes):
            release = getattr(vectors, "release_intermediate_storage", None)
            if reduced and callable(release):
                release()
            solution = solve(hamiltonian_operator, number_of_states, settings=probe_settings, state=state)
            passes_total += 1
            state, vectors = solution.state, solution.vectors
            values = np.asarray(solution.eigenvalues, dtype=np.float64)
            if previous is not None and previous.shape == values.shape:
                shown = (values <= window) | (previous <= window)
                if float(np.max(np.abs(values - previous)[shown], initial=0.0)) < FLIP_STABILITY_LEVEL_TOLERANCE:
                    break
            previous = values
        return solution

    def occupations_of(solution, mirrored):
        weights_down = getattr(solution, "state_weights", None)
        partners_down = getattr(solution, "state_partners", None)
        levels = np.asarray(solution.eigenvalues, dtype=np.float64)
        # The levels of the up channel in ascending order (the occupations of both channels sort them), with the
        # weights and partners of their states: only their Fermi level is read.
        order = np.arange(levels.size) if mirrored is None else np.argsort(mirrored, kind="stable")
        up_levels = levels if mirrored is None else np.asarray(mirrored, dtype=np.float64)[order]
        pair_weights = (
            None
            if weights_down is None
            else (
                np.asarray(weights_down, dtype=np.float64)[order],
                np.asarray(weights_down, dtype=np.float64),
            )
        )
        pair_partners = (
            None
            if partners_down is None
            else (tuple(np.asarray(partners_down)[order].tolist()), partners_down)
        )
        result = _spin_occupations([up_levels, levels], system, settings, pair_weights, pair_partners)
        return _weighted_occupations(result, pair_weights)[1]

    def labels_of(solution):
        labels = getattr(solution, "representations", None)
        levels = np.asarray(solution.eigenvalues)
        return np.zeros(levels.size, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)

    def outputs(up_density, down_density):
        hartree = system.solve_hartree(up_density + down_density, initial_potential=hartree_potential)
        xc = system.evaluate_spin_xc(up_density, down_density)
        return (
            values_of(ionic_potential + hartree.potential + xc.potentials[0]),
            values_of(ionic_potential + hartree.potential + xc.potentials[1]),
        )

    # The down channel at the converged potential of the route: its levels are those of the up channel.
    base_down = flip.down_field(up_potential)
    restart = getattr(solve, "start_new_scf_loop", None)
    state = restart() if callable(restart) else None
    first_started = time.perf_counter()
    base = solve_down(base_down, state, None, FLIP_STABILITY_FIRST_PASSES)
    first_passes, first_seconds = passes_total, time.perf_counter() - first_started
    state = base.state
    base_levels = np.asarray(base.eigenvalues, dtype=np.float64)
    base_labels = labels_of(base)
    up_levels = np.asarray(up_eigenvalues, dtype=np.float64)
    count = min(base_levels.size, up_levels.size)
    shown = np.sort(base_levels)[:count] <= window
    first_deviation = float(
        np.max(np.abs(np.sort(base_levels)[:count] - np.sort(up_levels)[:count])[shown], initial=0.0)
    )
    down_density = density(base.vectors, occupations_of(base, None))
    down_values = values_of(down_density)
    up_values = values_of(flip.up_field(down_values))
    base_up, base_down_output = outputs(field_of(up_values), field_of(down_values))
    base_potential = values_of(base_down)

    rng = np.random.default_rng(FLIP_STABILITY_SEED)
    moment = down_values - up_values
    total = down_values + up_values
    scale = max(float(np.max(np.abs(moment), initial=0.0)), 1.0e-30)
    start = (
        np.abs(moment) / scale
        + 0.5 * moment / scale
        + 0.25 * rng.standard_normal(moment.shape) * total / max(float(np.max(np.abs(total), initial=0.0)), 1.0e-30)
    )
    basis = [start / np.sqrt(dot(start, start))]
    hessenberg = np.zeros((FLIP_STABILITY_STEPS + 1, FLIP_STABILITY_STEPS), dtype=np.float64)
    history: list[float] = []
    ritz = np.zeros(0)
    probes = 0
    for step in range(FLIP_STABILITY_STEPS):
        direction = basis[step]
        size = FLIP_STABILITY_KICK / max(float(np.max(np.abs(direction), initial=0.0)), 1.0e-30)
        solution = solve_down(
            field_of(base_potential + size * direction), state, np.asarray(base.eigenvalues), FLIP_STABILITY_PASSES
        )
        state = solution.state
        probes += 1
        mirrored = _mirrored_levels(base_levels, base_labels, solution.eigenvalues, labels_of(solution))
        changed = values_of(density(solution.vectors, occupations_of(solution, mirrored))) - down_values
        up_new = up_values - values_of(flip.up_field(changed))
        up_output, down_output = outputs(field_of(up_new), field_of(down_values + changed))
        response = (
            (down_output - base_down_output) - values_of(flip.down_field(up_output - base_up))
        ) / (2.0 * size)
        for _ in range(2):
            # (Gram-Schmidt twice)
            for row in range(step + 1):
                projection = dot(basis[row], response)
                hessenberg[row, step] += projection
                response = response - projection * basis[row]
        norm = float(np.sqrt(max(dot(response, response), 0.0)))
        hessenberg[step + 1, step] = norm
        ritz = np.sort(np.linalg.eigvals(hessenberg[: step + 1, : step + 1]).real)[::-1]
        history.append(float(ritz[0]))
        if norm <= 1.0e-12 * max(1.0, float(np.max(np.abs(hessenberg[: step + 1, : step + 1])))):
            break
        basis.append(response / norm)
    release = getattr(vectors, "release_intermediate_storage", None)
    if reduced and callable(release):
        release()
    # The device Hamiltonian holds the potential of the up channel again, as the loop left it.
    system.hamiltonian(up_potential)
    return FlipStabilityReport(
        largest=history[-1] if history else float("nan"),
        history=history,
        ritz=ritz,
        probes=probes,
        passes=passes_total,
        first_deviation=first_deviation,
        first_passes=first_passes,
        first_seconds=first_seconds,
        seconds=time.perf_counter() - started,
        bound=FLIP_STABILITY_BOUND,
    )


def _take_flipped_channel(flip, eigenvalues, wavefunctions, representations, states, weights=None,
                          partners=None) -> None:
    """The down channel of a step as the image of the up channel (PARSEC_SPIN_FLIP_SYMMETRY).

    ``H_down = U_g H_up U_g^-1``: the same levels, the orbitals
    ``psi_up(g^-1 r)`` (formed where they are read) in the images of the up
    channel's sectors; no subspace of its own.  (with the weighted
    sectors of spin channels, PARSEC_ORBITAL_GROUP_SPIN) The state weights and
    partner masks of the down channel (``weights``, ``partners``: the lists of
    both channels, where the loop keeps them) are those of the up channel: the
    image of a level of the up channel is a level of the down channel with the
    same partners, in the image sector, whose weight ``d / m`` the operation
    keeps; before, the down channel had none and the loop stopped ("solved in
    sectors of different kinds").
    """

    eigenvalues[1] = np.array(eigenvalues[0], dtype=np.float64, copy=True)
    wavefunctions[1] = flip.down_orbitals(wavefunctions[0])
    representations[1] = flip.down_representations(representations[0])
    states[1] = flip.channel_state()
    if weights is not None:
        weights[1] = None if weights[0] is None else np.array(weights[0], copy=True)
    if partners is not None:
        partners[1] = partners[0]


def run_spin_scf(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    eigenproblem_solver: Callable[..., object] | None = None,
    orbital_density_builder: Callable[..., np.ndarray] | None = None,
    mixer_factory: Callable[..., object] | None = None,
    residual_metrics_evaluator: Callable[..., object] | None = None,
    total_energy_evaluator: Callable[..., object] | None = None,
    scalar_field_adapter: object | None = None,
    warm_start: SCFWarmStart | None = None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    exchange_term: object | None = None,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing with two spin channels (:func:`_run_spin_loop`).

    A loop of ``PARSEC_SPIN_FLIP_SYMMETRY`` (the down channel taken as the image of the up channel) is not kept where
    its up channel ends with a fractional Fermi-Dirac occupation (``min(f, 1 - f)`` from
    :data:`SPIN_FLIP_OCCUPATION_TOLERANCE`), where its converged state is unstable against perturbations that break
    the operation (``PARSEC_SPIN_FLIP_STABILITY=auto``, :func:`_flip_stability`), or where a test of the spin sectors
    of the accelerated driver fails in it (test 1 of the start Hamiltonian, tests 2 and 3 of the result: they would
    leave the sectors for a state of the operation that two solved channels need not reach).  A note goes to
    ``callback``, ``system`` leaves the route for
    the rest of the run (:func:`leave_spin_flip`), and the loop runs again from its start with both channels, in the
    sectors the flip loop started in: the loop of ``off``, whose result the guard of the spin sectors then tests as
    any other.  A warm start (an ionic step) is taken again without its saved subspaces, which the flip loop moved on.
    Under ``PARSEC_SPIN_FLIP_STABILITY=auto`` with ``PARSEC_SPIN_FLIP_CONTINUATION=auto`` a converged loop of the route
    with a start below it continues from that start without the flip (:func:`_continue_without_flip`).  A loop of
    ``PARSEC_SPIN_FLIP_SYMMETRY=late`` solves both channels until they are images of each other within its residual
    and takes the route from there (:func:`_late_flip_switch`).
    """

    keywords = dict(
        callback=callback,
        eigenproblem_solver=eigenproblem_solver,
        orbital_density_builder=orbital_density_builder,
        mixer_factory=mixer_factory,
        residual_metrics_evaluator=residual_metrics_evaluator,
        total_energy_evaluator=total_energy_evaluator,
        scalar_field_adapter=scalar_field_adapter,
        warm_start=warm_start,
        convergence_criterion=convergence_criterion,
        iteration_state_callback=iteration_state_callback,
        exchange_term=exchange_term,
    )
    record = None
    taken: list[str] = []
    continuation = None
    try:
        return _run_spin_loop(system, flip_taken=taken, **keywords)
    except _SpinFlipContinue as found:
        # PARSEC_SPIN_FLIP_CONTINUATION=auto: a start lies below the state of the route (only what the exception holds
        # leaves this block, not the frames of the flip loop)
        continuation = found.with_traceback(None)
    except _SpinFlipFractional as broken:
        # Only the note leaves this block: the frames of the flip loop, and the orbitals they hold, go with the
        # exception.
        record = _spin_flip_fractional_record(broken)
        why = f"a fractional occupation of the up channel, min(f, 1-f) = {broken.largest:.3e}"
    except _SpinFlipUnstable as unstable:
        # PARSEC_SPIN_FLIP_STABILITY=auto: the state the route converged to is unstable.
        record = _spin_flip_unstable_record(unstable.report)
        why = (
            "a state unstable against perturbations that break the operation (largest eigenvalue "
            f"{unstable.report.largest:.3f} of their SCF response)"
        )
    except Exception as error:
        if not taken or not _sector_guard(error):
            raise
        record = _spin_flip_guard_record(error)
        why = f"a test of the spin sectors failed in a flip loop after {error.stage} ({error.report.reason})"
    if continuation is not None:
        return _continue_without_flip(system, keywords, continuation, callback)
    return _rerun_without_flip(system, keywords, record, why)


def _rerun_without_flip(system, keywords: dict, record, why: str):
    """The loop again from its start with both channels (the route of ``off``) after ``record``, for the rest of the
    run (:func:`leave_spin_flip`); a warm start without its saved subspaces, which the flip loop moved on."""

    leave_spin_flip(system, why)
    callback = keywords.get("callback")
    if callback is not None:
        callback(record)
    warm_start = keywords.get("warm_start")
    if warm_start is not None:
        keywords = {**keywords, "warm_start": replace(warm_start, eigensolver_state=None, first_iteration_degree=None)}
    return _run_spin_loop(system, **keywords)


def _run_spin_loop(
    system: PreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
    eigenproblem_solver: Callable[..., object] | None = None,
    orbital_density_builder: Callable[..., np.ndarray] | None = None,
    mixer_factory: Callable[..., object] | None = None,
    residual_metrics_evaluator: Callable[..., object] | None = None,
    total_energy_evaluator: Callable[..., object] | None = None,
    scalar_field_adapter: object | None = None,
    warm_start: SCFWarmStart | None = None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    exchange_term: object | None = None,
    flip_taken: list | None = None,
    give_up: Callable[[int, float, float], bool] | None = None,
    flip_test: bool = True,
) -> SinglePointResult:
    """Run PARSEC-style potential mixing with two spin channels.

    The arguments are those of :func:`parsec_python.SCF.run_scf`, which
    dispatches here.  ``eigenproblem_solver`` and ``orbital_density_builder``
    are the execution hooks of the accelerated package.  The hooks of a
    symmetry-reduced backend, which keep scalar fields on one value per
    orbit, are accepted from an adapter made for two channels
    (``scalar_field_adapter.spin_hooks()``, the symmetry sectors of the
    accelerated driver with ``Spin_Symmetry_Sectors``): its mixer takes the
    two channels' orbit values joined, up first, and its residual norms and
    energy take the per-channel arguments of
    :func:`parsec_python.Mixer.spin_potential_residual_metrics` and
    :func:`parsec_python.Energy.spin_total_energy`.  Other reduced hooks are
    refused.  Such an adapter may also check the occupations of the result
    of a loop without an exchange operator (``check_spin_result``).

    The returned result holds the two channels in ``spin`` and describes
    them together in its other fields, as
    :class:`parsec_python.models.SinglePointResult` documents; its fields are
    full-grid arrays also where the loop kept them on the orbits.

    ``flip_taken`` (a list, from :func:`run_spin_scf`) receives the description of the spin-flip operation where
    this loop takes the down channel from the up channel.  ``give_up(iteration, free energy, residual)`` (the
    continuation of :func:`_continue_without_flip`) is asked after every iteration that did not converge and stops the
    loop, unconverged, where it returns true; ``flip_test`` false skips the test of the converged state of the route
    (the loop that converges that state again after a continuation that did not end below it).
    """

    reduced_hooks = {
        "mixer_factory": mixer_factory,
        "residual_metrics_evaluator": residual_metrics_evaluator,
        "total_energy_evaluator": total_energy_evaluator,
        "scalar_field_adapter": scalar_field_adapter,
    }
    given = [name for name, hook in reduced_hooks.items() if hook is not None]
    if given:
        spin_hooks = getattr(scalar_field_adapter, "spin_hooks", None)
        expected = spin_hooks() if callable(spin_hooks) else None
        if expected is None or any(reduced_hooks[name] != hook for name, hook in expected.items()):
            raise NotImplementedError(
                "a spin-polarized calculation keeps its scalar fields on the full "
                "grid; the symmetry-reduced SCF hooks are not available for it "
                "unless an adapter of two channels gives them "
                f"({', '.join(given)})"
            )
    reduced = scalar_field_adapter is not None
    to_internal = (lambda values: values) if not reduced else scalar_field_adapter.from_full
    to_public = (lambda values: values) if not reduced else scalar_field_adapter.to_full
    evaluate_residual_metrics = (
        spin_potential_residual_metrics if residual_metrics_evaluator is None else residual_metrics_evaluator
    )
    evaluate_total_energy = spin_total_energy if total_energy_evaluator is None else total_energy_evaluator
    build_mixer = make_mixer if mixer_factory is None else mixer_factory
    if system.initial_spin_densities is None and warm_start is None:
        raise ValueError(
            "the system was not prepared for a spin-polarized calculation: "
            "it has no starting spin densities"
        )
    hubbard_projectors = require_reference_hooks(
        system,
        eigenproblem_solver=eigenproblem_solver,
        orbital_density_builder=orbital_density_builder,
    )
    if exchange_term is not None and getattr(system, "exchange_device", None) is None:
        # A hybrid functional: the exchange operator of each channel joins
        # its Hamiltonian (parsec_python.SCF.hybrid); the reference path, or
        # an accelerated system whose device Hamiltonians hold it.
        refuse_exchange_hooks(
            eigenproblem_solver=eigenproblem_solver,
            orbital_density_builder=orbital_density_builder,
        )
    # LDA+U potential matrices of both channels, zero in the first iteration.
    hubbard_state = None
    # A meta-GGA: tau and vtau of each channel (SCF.meta_gga).
    meta = meta_gga_loop(
        system,
        channels=2,
        adapter=scalar_field_adapter,
        warm_start=warm_start,
        exchange_term=exchange_term,
    )

    scf_start = time.perf_counter()
    level_detail = None
    solvers = _channel_solvers(eigenproblem_solver)
    ionic_potential = to_internal(system.ionic_potential)
    # PARSEC_SPIN_FLIP_SYMMETRY (acceleration/SCF/spin_flip.py): the down
    # channel is the image of the up channel under an operation of the atoms
    # that exchanges the partner labels, and only the up channel is solved.
    # (none once the run left the route, leave_spin_flip: the second loop of run_spin_scf and every later one)
    flip, flip_refusal = _spin_flip_of(
        system,
        exchange_term=exchange_term,
        meta=meta,
        hubbard_projectors=hubbard_projectors,
        ionic_potential=ionic_potential,
    )
    # (PARSEC_SPIN_FLIP_SYMMETRY=late: the operation the loop takes once its channels are images of each other)
    pending_flip = None
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    settings = system.input.scf
    if convergence_criterion is not None:
        settings = replace(settings, convergence_criterion=float(convergence_criterion))
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    neighbors = preconditioner_neighbors(system, eigensolver_settings)
    diagonalization_tolerance = first_diagonalization_tolerance(eigensolver_settings)
    # PARSEC_STATES_CLOSE_MULTIPLET: more states where States_Num cuts an occupied multiplet (full-grid loops).
    closer = MultipletCloser.for_loop(
        system,
        adapter=scalar_field_adapter,
        exchange=exchange_term is not None,
        criterion=settings.convergence_criterion,
    )

    # 1. First potentials.  The Hartree potential is that of the normalized
    # total starting density; the exchange-correlation potentials are those
    # of the two starting channels, which initchrg.f90 builds so that they
    # add up to twice that density (see initial_spin_densities).
    eigval_states: list[object | None] = [None, None]
    # PARSEC_STATES_CLOSE_MULTIPLET: the Ritz values of the working subspace of each channel's last solve.
    channel_levels: list[np.ndarray | None] = [None, None]
    if warm_start is None:
        stage_start = time.perf_counter()
        initial_hartree = system.solve_hartree(
            to_internal(system.initial_density.copy()), initial_potential=-ionic_potential
        )
        initial_hartree_seconds = time.perf_counter() - stage_start
        hartree_potential = initial_hartree.potential
        stage_start = time.perf_counter()
        start_densities = tuple(to_internal(density) for density in system.initial_spin_densities)
        xc = (
            system.evaluate_spin_xc(*start_densities)
            if meta is None
            else meta.start_spin_xc(*start_densities)
        )
        initial_xc_seconds = time.perf_counter() - stage_start
        input_potentials = [
            ionic_potential + hartree_potential + xc_potential
            for xc_potential in xc.potentials
        ]
    else:
        # After an ionic move: the mixed potentials of both channels with
        # the new local ionic potential, the last Hartree potential as the
        # start of the first Poisson solve, and the saved subspaces.
        initial_hartree_seconds = 0.0
        initial_xc_seconds = 0.0
        hartree_potential = to_internal(
            np.array(warm_start.hartree_potential, dtype=np.float64)
        )
        if len(warm_start.input_potential) != 2:
            raise ValueError(
                "a spin-polarized warm start needs the potentials of both channels"
            )
        input_potentials = [
            to_internal(np.array(potential, dtype=np.float64))
            for potential in warm_start.input_potential
        ]
        states = warm_start.eigensolver_state
        eigval_states = [None, None] if states is None else list(states)
        # The down channel of a loop that took it from the up channel has no
        # subspace of its own (PARSEC_SPIN_FLIP_SYMMETRY).
        eigval_states = [
            None if getattr(state, "spin_flip_channel", False) else state for state in eigval_states
        ]
        if flip is not None:
            # A warm start keeps the route only where its down potential is
            # the image of its up potential (that of a loop that took it, to
            # the rounding), not where it continues a state that breaks the
            # operation (Restart_Run of such a state): that one is solved in
            # both channels from its own potentials.
            deviation, bound = flip.warm_start_check(input_potentials[0], input_potentials[1])
            if deviation <= bound:
                input_potentials[1] = flip.down_field(input_potentials[0])
            elif getattr(flip, "mode", "auto") == "late":
                # (PARSEC_SPIN_FLIP_SYMMETRY=late: both channels until they are images of each other)
                pending_flip, flip = flip, None
            else:
                flip_refusal = (
                    f"the down potential of the warm start (Restart_Run, the structure before) is not the image of "
                    f"its up potential under the operation {flip.description}",
                    "largest deviation [Ry]",
                    deviation,
                    bound,
                )
                flip = None
    # PARSEC_SPIN_FLIP_SYMMETRY=late: the loop solves both channels, as off does, until they are images of each other
    # within its residual (_late_flip_switch), and takes the flip from there; a warm start whose potentials are images
    # of each other (the structure before, in the route) takes it at once.
    if flip is not None and warm_start is None and getattr(flip, "mode", "auto") == "late":
        pending_flip, flip = flip, None
    late_switched = None
    late_asymmetries: list[float] = []
    # (the ratios of the potential deviation to the residual where the other conditions of the switch held)
    late_ratios: list[float] = []
    if closer is not None:
        # (a loop that continues one whose States_Num was raised keeps that count)
        number_of_states = closer.carried(number_of_states, eigval_states)
    if flip_refusal is not None and callback is not None:
        callback(_spin_flip_record(flip_refusal))
    if flip is not None and flip_taken is not None:
        flip_taken.append(str(getattr(flip, "description", "")))

    # The mixer a warm start carries (the Fock loop's Exx_Mixer_History), else a new one.
    # Mixing_Preconditioner: kerker filters the charge part of the joined
    # channels' residual (Mixer.preconditioner); "none" keeps this mixer.
    mixer = warm_start_mixer(
        warm_start,
        (
            lambda: preconditioned_mixer(
                build_mixer(system.input.mixing), system, adapter=scalar_field_adapter, channels=2
            )
        )
        if meta is None
        else (lambda: meta.mixer(build_mixer, system.input.mixing)),
    )
    if reduced:
        # The orbit values of both channels, up first, and back.
        join_channels = scalar_field_adapter.join_channels
        split_channels = scalar_field_adapter.split_channels
    else:
        join_channels = np.concatenate

        def split_channels(mixed):
            return [mixed[:grid_size].copy(), mixed[grid_size:].copy()]

    history: list[SCFIteration] = []
    filter_degree = eigensolver_settings.filter_degree
    minimum_filter_degree = max(10, eigensolver_settings.filter_degree_delta + 1)
    degree_schedule = LaterFilterDegree.for_loop(filter_degree, minimum_filter_degree)
    converged = False
    energies = None
    grid_size = system.grid.size
    eigenvalues: list[np.ndarray] = [np.empty(0), np.empty(0)]
    occupations: list[np.ndarray] = [np.empty(0), np.empty(0)]
    representations: list[np.ndarray] = [
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]
    # The state weights of the weighted sectors of a subgroup (one partner of some degenerate levels), per channel.
    state_weights: list[np.ndarray | None] = [None, None]
    # and the partner masks of their states (CuPySymmetryEigvalResult.state_partners), which occup.in fills alike
    state_partners: list[tuple | None] = [None, None]
    wavefunctions: list[object] = [np.empty((grid_size, 0)), np.empty((grid_size, 0))]
    densities: list[np.ndarray] = [np.empty(0), np.empty(0)]
    electrons = (0.0, 0.0)
    fermi_level = np.nan
    last_input_potentials = [potential.copy() for potential in input_potentials]
    mixed_potentials = [potential.copy() for potential in input_potentials]
    xc_potentials = list(xc.potentials) if warm_start is None else [None, None]
    hamiltonian_binding_total = 0.0
    diagonalization_total = 0.0
    occupations_density_total = 0.0
    hartree_total = 0.0
    xc_total = 0.0
    mixing_energy_total = 0.0

    # SCFWarmStart.level_tolerance (run_scf): one more iteration on the same V_in.
    level_tolerance = None if warm_start is None else getattr(warm_start, "level_tolerance", None)
    refining = False
    # Test 2 of a reduced loop (acceleration/SCF/spin_sectors.py): the
    # criterion is tightened while the asymmetry of a converged state looks
    # like SCF noise; the loop counts as converged once the given criterion
    # was met.
    criterion_scale = 1.0
    symmetry_tightenings = 0
    last_asymmetry = None
    met_criterion = False
    for iteration in range(1, settings.max_iterations + 1 + int(level_tolerance is not None)):
        if iteration > settings.max_iterations and not refining:
            break
        iteration_start = time.perf_counter()
        last_input_potentials = [potential.copy() for potential in input_potentials]
        eigval_settings = eigval_settings_for(
            eigensolver_settings,
            filter_degree,
            tolerance=diagonalization_tolerance,
            neighbors=neighbors,
        )

        # 2. One eigenproblem per channel, each with the potential and the
        # saved subspace of that channel.  A backend binds the potential to
        # one Hamiltonian allocation, so the channels are solved in turn.
        hamiltonian_binding_seconds = 0.0
        diagonalization_seconds = 0.0
        residual_maxima: list[float] = []
        if hubbard_projectors is not None and iteration > 1:
            # calcden.f90z: N and D of both channels from the orbitals and
            # occupations of the iteration before.
            hubbard_state = hubbard_terms_of_iteration(
                system, hubbard_projectors, wavefunctions, occupations
            )
        solver_counts: list[tuple[int, int] | None] = []
        for spin in range(2):
            if spin == 1 and flip is not None:
                _take_flipped_channel(flip, eigenvalues, wavefunctions, representations, eigval_states,
                                      state_weights, state_partners)
                solver_counts.append(solver_counts[0])
                if residual_maxima:
                    residual_maxima.append(residual_maxima[0])
                continue
            binding_start = time.perf_counter()
            if meta is not None and meta.kinetic_potential(spin) is not None:
                # A meta-GGA hybrid: the orbital term and the fixed exchange operator.
                hamiltonian = system.hamiltonian(
                    input_potentials[spin],
                    None,
                    None if exchange_term is None else exchange_term.operator(spin),
                    kinetic_potential=meta.kinetic_potential(spin),
                    **meta.flux_keywords(spin),
                )
            elif hubbard_projectors is None and exchange_term is None:
                hamiltonian = system.hamiltonian(input_potentials[spin])
            elif hubbard_projectors is None:
                hamiltonian = system.hamiltonian(
                    input_potentials[spin], None, exchange_term.operator(spin)
                )
            else:
                hamiltonian = system.hamiltonian(
                    input_potentials[spin],
                    hubbard_operator(hubbard_projectors, hubbard_state, spin),
                )
            operator_factory = getattr(
                hamiltonian,
                "as_eigensolver_operator",
                hamiltonian.as_linear_operator,
            )
            hamiltonian_operator = operator_factory()
            hamiltonian_binding_seconds += time.perf_counter() - binding_start
            diagonalization_start = time.perf_counter()
            if reduced:
                # The sector views of the channel's last orbitals are no state
                # once its next solve begins (as in run_scf).
                release = getattr(wavefunctions[spin], "release_intermediate_storage", None)
                if callable(release):
                    release()
            if refining:
                # The loop converged at the iteration before (run_scf).
                eigensolution, passes, seconds, change = refine_levels(
                    solvers[spin],
                    hamiltonian_operator,
                    number_of_states,
                    eigval_settings,
                    eigval_states[spin],
                    eigenvalues[spin],
                    None,
                    float(level_tolerance),
                    release=reduced,
                )
                level_detail = {
                    "level_passes": float(passes) + (level_detail or {}).get("level_passes", 0.0),
                    "level_seconds": float(seconds) + (level_detail or {}).get("level_seconds", 0.0),
                    "level_change": max(float(change), (level_detail or {}).get("level_change", 0.0)),
                }
            else:
                eigensolution = solvers[spin](
                    hamiltonian_operator,
                    number_of_states,
                    settings=eigval_settings,
                    state=eigval_states[spin],
                )
            eigval_states[spin] = eigensolution.state
            channel_levels[spin] = working_levels(eigensolution)
            solver_counts.append(getattr(eigensolution, "iterative_counts", None))
            diagonalization_seconds += time.perf_counter() - diagonalization_start
            eigenvalues[spin] = np.asarray(eigensolution.eigenvalues, dtype=np.float64)
            wavefunctions[spin] = eigensolution.vectors
            state_weights[spin] = getattr(eigensolution, "state_weights", None)
            state_partners[spin] = getattr(eigensolution, "state_partners", None)
            sector_labels = getattr(eigensolution, "representations", None)
            representations[spin] = (
                np.ones(eigenvalues[spin].size, dtype=np.int32)
                if sector_labels is None
                else np.asarray(sector_labels, dtype=np.int32) + 1
            )
            if eigensolution.residual_norms is not None:
                residual_maxima.append(
                    float(np.max(eigensolution.residual_norms, initial=0.0))
                )
        hamiltonian_binding_total += hamiltonian_binding_seconds
        diagonalization_total += diagonalization_seconds
        eigen_residual_max = (
            max(residual_maxima) if len(residual_maxima) == 2 else float("nan")
        )

        # 3. One Fermi level for both channels, then the density of each.
        occupations_density_start = time.perf_counter()
        # Weighted sectors: each state counts for its partners (its weight) in the Fermi level, the
        # occupations, the density and the band energy, as in run_scf.
        weights = _channel_state_weights(state_weights)
        occupation_result = _spin_occupations(eigenvalues, system, settings, weights, state_partners)
        occupations = _weighted_occupations(occupation_result, weights)
        electrons = occupation_result.electrons
        fermi_level = occupation_result.fermi_level
        start_report = getattr(scalar_field_adapter, "start_spectrum_report", None)
        if reduced and warm_start is None and iteration == 1 and callable(start_report):
            # The levels of the start Hamiltonian, which has the point group
            # of the atoms: a reduced adapter refuses a partly filled
            # multiplet here (acceleration/SCF/spin_sectors.py), once the
            # levels are converged; until then the start Hamiltonian is
            # solved again from its subspaces.
            passes = 0
            change = np.inf
            while True:
                report = start_report(eigenvalues, occupations, representations)
                if report is None:
                    break
                if passes and change <= scalar_field_adapter.start_level_change:
                    scalar_field_adapter.abandon_start(report, passes)
                    # (it returns only where the adapter puts the report aside, PARSEC_SPIN_SECTOR_STABILITY=auto:
                    # the loop goes on in the sectors and the state is tested at its end)
                    break
                if passes >= scalar_field_adapter.start_spectrum_passes or hubbard_projectors is not None:
                    # Still moving: tests 2 and 3 decide at the end.
                    break
                previous = [values.copy() for values in eigenvalues]
                window = fermi_level + START_LEVEL_WINDOW
                diagonalization_start = time.perf_counter()
                for spin in range(2):
                    if spin == 1 and flip is not None:
                        _take_flipped_channel(flip, eigenvalues, wavefunctions, representations, eigval_states,
                                              state_weights, state_partners)
                        continue
                    if meta is not None and meta.kinetic_potential(spin) is not None:
                        hamiltonian = system.hamiltonian(
                            input_potentials[spin],
                            None,
                            None if exchange_term is None else exchange_term.operator(spin),
                            kinetic_potential=meta.kinetic_potential(spin),
                            **meta.flux_keywords(spin),
                        )
                    elif exchange_term is None:
                        hamiltonian = system.hamiltonian(input_potentials[spin])
                    else:
                        hamiltonian = system.hamiltonian(
                            input_potentials[spin], None, exchange_term.operator(spin)
                        )
                    operator_factory = getattr(
                        hamiltonian, "as_eigensolver_operator", hamiltonian.as_linear_operator
                    )
                    release = getattr(wavefunctions[spin], "release_intermediate_storage", None)
                    if callable(release):
                        release()
                    eigensolution = solvers[spin](
                        operator_factory(),
                        number_of_states,
                        settings=eigval_settings,
                        state=eigval_states[spin],
                    )
                    eigval_states[spin] = eigensolution.state
                    channel_levels[spin] = working_levels(eigensolution)
                    eigenvalues[spin] = np.asarray(eigensolution.eigenvalues, dtype=np.float64)
                    wavefunctions[spin] = eigensolution.vectors
                    state_weights[spin] = getattr(eigensolution, "state_weights", None)
                    state_partners[spin] = getattr(eigensolution, "state_partners", None)
                    sector_labels = getattr(eigensolution, "representations", None)
                    representations[spin] = (
                        np.ones(eigenvalues[spin].size, dtype=np.int32)
                        if sector_labels is None
                        else np.asarray(sector_labels, dtype=np.int32) + 1
                    )
                diagonalization_total += time.perf_counter() - diagonalization_start
                passes += 1
                change = max(
                    float(np.max(np.abs(new - old)[(old <= window) | (new <= window)], initial=0.0))
                    if new.shape == old.shape
                    else np.inf
                    for new, old in zip(eigenvalues, previous)
                )
                weights = _channel_state_weights(state_weights)
                occupation_result = _spin_occupations(eigenvalues, system, settings, weights, state_partners)
                occupations = _weighted_occupations(occupation_result, weights)
                electrons = occupation_result.electrons
                fermi_level = occupation_result.fermi_level
        state_count_change = None
        if closer is not None and weights is None and not refining:
            # PARSEC_STATES_CLOSE_MULTIPLET (SCF/state_extension.py), with the levels of both channels.
            state_count_change = closer.check(
                iteration,
                number_of_states,
                tuple(
                    (
                        SPIN_NAMES[spin],
                        eigenvalues[spin],
                        occupations[spin],
                        channel_levels[0 if spin == 1 and flip is not None else spin],
                    )
                    for spin in range(2)
                ),
                [step.plain_residual if settings.use_plain_residual else step.weighted_residual for step in history],
                fermi_level,
            )
            if state_count_change is not None:
                number_of_states = state_count_change.states
                reset = getattr(mixer, "reset", None)
                if callable(reset):
                    reset()
        for spin in range(2):
            if spin == 1 and flip is not None:
                # rho_down(r) = rho_up(g^-1 r): one gather of the up channel's values.
                densities[1] = flip.down_field(densities[0])
                continue
            if orbital_density_builder is None:
                densities[spin] = spin_density_from_orbitals(
                    wavefunctions[spin], occupations[spin], volume_element
                )
            else:
                # A backend builder forms 2/dV sum f |psi|^2, the density of
                # doubly occupied orbitals.  Half the occupation gives the
                # one electron per state of a spin channel, bit for bit.
                built = orbital_density_builder(
                    wavefunctions[spin], 0.5 * occupations[spin], volume_element
                )
                # (A reduced builder returns the field of the orbits.)
                densities[spin] = built if reduced else np.asarray(built, dtype=np.float64)
        # A meta-GGA: tau of each channel, one electron per state.
        taus = (
            None
            if meta is None
            else [meta.tau(wavefunctions[spin], occupations[spin], 1.0, spin) for spin in range(2)]
        )
        symmetrization = ()
        density_symmetry = getattr(system, "density_symmetry", None)
        # A group that brings its own method averages the channels of an accelerated loop that keeps one value per
        # symmetry orbit (acceleration/SCF/density_symmetry.py, PARSEC_SYMMETRIZED_DENSITY_SECTORS=auto).
        symmetrize_own = getattr(density_symmetry, "symmetrize_density", None)
        if density_symmetry is not None and reduced and not callable(symmetrize_own):
            raise NotImplementedError(
                "Symmetrize_Charge_Density is not implemented for spin channels in the symmetry sectors"
            )
        if density_symmetry is not None:
            # Symmetrize_Charge_Density: symm_scalar of newrho.F90 for the
            # total density and for each channel (isp = 1, 2, 3).
            def symmetrized(field):
                if callable(symmetrize_own):
                    return symmetrize_own(field, volume_element)
                return symmetrize_density(field, density_symmetry, volume_element)

            _, total_report = symmetrized(densities[0] + densities[1])
            reports = [total_report]
            for spin in range(2):
                densities[spin], report = symmetrized(densities[spin])
                reports.append(report)
                if taus is not None:
                    taus[spin], _ = symmetrized(taus[spin])
                    meta.symmetrize_gradient(spin, density_symmetry)
            symmetrization = tuple(reports)
        density = densities[0] + densities[1]
        occupations_density_seconds = time.perf_counter() - occupations_density_start
        occupations_density_total += occupations_density_seconds

        # 4. Hartree potential of the total density and the two
        # exchange-correlation potentials.
        hartree_start = time.perf_counter()
        hartree = system.solve_hartree(density, initial_potential=hartree_potential)
        hartree_seconds = time.perf_counter() - hartree_start
        hartree_total += hartree_seconds
        hartree_potential = hartree.potential
        xc_start = time.perf_counter()
        if meta is None:
            xc = system.evaluate_spin_xc(densities[0], densities[1])
        else:
            xc = system.evaluate_spin_xc(
                densities[0], densities[1], tau_up=taus[0], tau_down=taus[1], **meta.spin_gradient_keywords()
            )
        xc_seconds = time.perf_counter() - xc_start
        xc_total += xc_seconds
        xc_potentials = list(xc.potentials)
        output_potentials = [
            ionic_potential + hartree_potential + xc_potential
            for xc_potential in xc_potentials
        ]

        # 5. Residuals and energies.  The eigenpairs belong to the input
        # potentials and the Hartree and XC terms to the new densities, as
        # in the unpolarized loop.
        mixing_energy_start = time.perf_counter()
        metrics = evaluate_residual_metrics(
            input_potentials,
            output_potentials,
            densities,
            volume_element,
            electrons,
        )
        energies = evaluate_total_energy(
            eigenvalues,
            occupations,
            densities,
            input_potentials,
            ionic_potential,
            hartree_potential,
            xc_potentials,
            xc.total_energy,
            system.ion_ion_energy,
            volume_element,
            **(
                {}
                if hubbard_projectors is None
                else {
                    "hubbard_energy": 0.0
                    if hubbard_state is None
                    else hubbard_state.energy
                }
            ),
            # The alpha term of a periodic cell (forces, relaxation and
            # molecular dynamics of a spin-polarized bulk cell run here).
            **(
                {"alpha_z_energy": system.alpha_z_energy}
                if getattr(system, "alpha_z_energy", 0.0)
                else {}
            ),
        )
        if meta is not None:
            energies = meta.energies(energies, taus, getattr(xc, "fermi_hole_clamped", 0), xc=xc)
        if exchange_term is not None:
            # (After the meta-GGA's double counting: a meta-GGA hybrid keeps the
            # hybrid breakdown of its inner loops.)
            energies = exchange_term.iteration_energies(
                energies, tuple(wavefunctions), tuple(occupations)
            )

        # 6. Mix the two potentials as one vector, up first, and test the
        # larger residual of the two channels.
        if meta is None:
            mixed = mixer.mix(
                join_channels(input_potentials),
                join_channels(output_potentials),
                iteration=iteration,
            )
            mixed_potentials = split_channels(mixed)
            if late_switched is not None and flip is not None:
                # (PARSEC_SPIN_FLIP_SYMMETRY=late: the history of the mixer holds the iterations before the switch, in
                # which the channels were solved apart; the next input is one of the route)
                mixed_potentials = list(
                    _flip_symmetric(flip, mixed_potentials[0], mixed_potentials[1], scalar_field_adapter)
                )
        else:
            # The vtau of both channels joins the vector (SCF.meta_gga).
            mixed_potentials = meta.mix(
                mixer,
                input_potentials,
                output_potentials,
                xc.kinetic_potentials,
                iteration=iteration,
                join=join_channels,
                split=split_channels,
                densities=densities,
            )
        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        # PARSEC_FILTER_DEGREE_STALL (SCF/filter_degree.py): the degree lowered near convergence, and the
        # starting degree back where the loop stalls after that
        filter_degree = degree_schedule.after_iteration(
            iteration, metrics.weighted, settings.approach_criterion, settings.degree_decrease_after
        )
        if iteration == 1:
            first_residual = selected_residual
        iteration_converged = selected_residual < criterion_scale * loop_criterion(
            warm_start, settings.convergence_criterion, first_residual
        )
        if eigensolver_settings.dynamic_tolerance:
            # The smaller weighted SRE of the two channels (minval(sre)).
            diagonalization_tolerance = dynamic_diagonalization_tolerance(
                diagonalization_tolerance,
                metrics.weighted_by_spin,
                eigensolver_settings.tolerance,
            )
        mixing_energy_seconds = time.perf_counter() - mixing_energy_start
        mixing_energy_total += mixing_energy_seconds

        # Weighted sectors: every partner of a weighted state, as the full grid lists them (run_scf).
        public = _public_channels(eigenvalues, occupations, representations, wavefunctions, state_weights)
        # The record holds the channels end to end in two halves: as many levels of each (at least States_Num).
        listed = min(len(public[0][0]), len(public[1][0]))
        diagnostic = SCFIteration(
            iteration=iteration,
            # PARSEC_FILTER_DEGREE_STALL: the NOTE of a restoration (once)
            filter_degree_note=degree_schedule.take_note(),
            weighted_residual=metrics.weighted,
            plain_residual=metrics.plain,
            eigen_residual_max=eigen_residual_max,
            hartree_residual=hartree.residual_norm,
            energies=energies,
            eigenvalues=tuple(
                float(value) for values in public for value in values[0][:listed]
            ),
            occupations=tuple(
                float(value) for values in public for value in values[1][:listed]
            ),
            representations=tuple(
                int(value) for values in public for value in values[2][:listed]
            ),
            fermi_level=float(fermi_level),
            density_minimum=(
                scalar_field_adapter.minimum(density) if reduced else float(np.min(density))
            ),
            density_maximum=(
                scalar_field_adapter.maximum(density) if reduced else float(np.max(density))
            ),
            diagonalization_seconds=diagonalization_seconds,
            hartree_seconds=hartree_seconds,
            hamiltonian_binding_seconds=hamiltonian_binding_seconds,
            occupations_density_seconds=occupations_density_seconds,
            xc_seconds=xc_seconds,
            mixing_energy_seconds=mixing_energy_seconds,
            total_seconds=time.perf_counter() - iteration_start,
            **({"level_refinement": True} if refining else {}),
            spin_plain_residuals=metrics.plain_by_spin,
            spin_weighted_residuals=metrics.weighted_by_spin,
            spin_electrons=(float(electrons[0]), float(electrons[1])),
            spin_density_minimum=tuple(
                scalar_field_adapter.minimum(values) if reduced else float(np.min(values))
                for values in densities
            ),
            spin_density_maximum=tuple(
                scalar_field_adapter.maximum(values) if reduced else float(np.max(values))
                for values in densities
            ),
            density_symmetrization=symmetrization,
            **({} if state_count_change is None else {"state_count_note": state_count_change.lines()}),
            **hubbard_iteration_record(hubbard_state),
            **iteration_eigensolver_fields(
                tuple(solver_counts),
                diagonalization_tolerance
                if eigensolver_settings.dynamic_tolerance
                else None,
            ),
        )
        history.append(diagnostic)
        if callback is not None:
            callback(diagnostic)
        if iteration_state_callback is not None:
            from .single_point import SCFIterationFields

            iteration_channels = tuple(
                SpinChannelResult(
                    eigenvalues=public[spin][0],
                    occupations=public[spin][1],
                    wavefunctions=public[spin][3],
                    representations=public[spin][2],
                    density=to_public(densities[spin]),
                    xc_potential=to_public(xc_potentials[spin]),
                    input_effective_potential=to_public(last_input_potentials[spin]),
                    output_effective_potential=to_public(output_potentials[spin]),
                    next_effective_potential=to_public(mixed_potentials[spin]),
                    electron_count=float(electrons[spin]),
                )
                for spin in range(2)
            )
            iteration_state_callback(
                SCFIterationFields(
                    iteration=iteration,
                    eigenvalues=np.concatenate([values[0] for values in public]),
                    occupations=np.concatenate([values[1] for values in public]),
                    representations=np.concatenate([values[2] for values in public]),
                    wavefunctions=None,
                    density=to_public(density),
                    next_effective_potential=0.5
                    * (
                        to_public(mixed_potentials[0]) + to_public(mixed_potentials[1])
                    ),
                    spin=iteration_channels,
                )
            )

        measure_asymmetry = getattr(scalar_field_adapter, "density_symmetry_measure", None)
        if iteration_converged and not refining and reduced and exchange_term is None and callable(measure_asymmetry):
            met_criterion = True
            asymmetry = measure_asymmetry([to_public(values) for values in densities])
            log = getattr(scalar_field_adapter, "symmetry_measures", None)
            if isinstance(log, list):
                log.append((iteration, asymmetry, criterion_scale * settings.convergence_criterion))
            if (
                asymmetry is not None
                and scalar_field_adapter.density_symmetry_tolerance
                <= asymmetry
                < scalar_field_adapter.density_symmetry_noise_bound
                and symmetry_tightenings < scalar_field_adapter.density_symmetry_tightenings
                and (last_asymmetry is None or asymmetry <= 0.5 * last_asymmetry)
                and iteration < settings.max_iterations
            ):
                # Noise shrinks with the residual, a broken state keeps its
                # asymmetry: go on to a criterion ten times tighter.
                symmetry_tightenings += 1
                last_asymmetry = asymmetry
                criterion_scale *= 0.1
                input_potentials = mixed_potentials
                if meta is not None:
                    meta.advance()
                continue
        if iteration_converged or refining:
            if level_tolerance is not None and not refining:
                # V_in stays: the next iteration solves the same Hamiltonians.
                refining = True
                continue
            converged = True
            break
        late_deviation: list[float] = []

        def deviation_of_inputs():
            # (the input potentials of this iteration against a pair of the route, in the norm of its residual)
            late_deviation.append(
                _flip_potential_deviation(
                    pending_flip, input_potentials, densities, electrons, evaluate_residual_metrics, volume_element,
                    bool(settings.use_plain_residual),
                )
            )
            late_ratios.append(late_deviation[-1] / max(float(selected_residual), 1.0e-300))
            return late_deviation[-1]

        if pending_flip is not None and _late_flip_switch(
            eigenvalues, occupation_result.occupations, fermi_level, selected_residual, settings, late_asymmetries,
            deviation=deviation_of_inputs,
        ):
            # PARSEC_SPIN_FLIP_SYMMETRY=late: the two solved channels are images of each other within the residual;
            # the next input is that of the route, which takes the down channel from the up channel from here on.
            mixed_potentials = list(
                _flip_symmetric(pending_flip, mixed_potentials[0], mixed_potentials[1], scalar_field_adapter)
            )
            flip, pending_flip = pending_flip, None
            late_switched = iteration
            if LATE_FLIP_MIXER == "project":
                _project_mixer_history(mixer, flip)
            elif LATE_FLIP_MIXER == "reset":
                reset = getattr(mixer, "reset", None)
                if callable(reset):
                    reset()
            if flip_taken is not None:
                flip_taken.append(str(getattr(flip, "description", "")))
            if callback is not None:
                callback(
                    _late_flip_record(
                        flip,
                        switched=True,
                        iteration=iteration,
                        lines=(
                            f"residual {float(selected_residual):.3e} Ry; input potentials {late_deviation[-1]:.3e} Ry "
                            f"apart in its norm; levels {late_asymmetries[-1]:.3e} Ry; densities "
                            f"{_flip_density_deviation(flip, densities, scalar_field_adapter, volume_element):.3e} "
                            "electrons",
                        ),
                    )
                )
        if give_up is not None:
            from parsec_python.acceleration.SCF.spin_flip_stability import entropy_term

            free = float(energies.total) + entropy_term(
                occupation_result.occupations, weights, settings.fermi_temperature_kelvin
            )
            if give_up(iteration, free, float(selected_residual)):
                break
        input_potentials = mixed_potentials
        if meta is not None:
            meta.advance()

    if energies is None:
        raise RuntimeError("SCF loop did not execute")
    # A loop that met its criterion and then ran out of iterations while
    # tightening it (test 2 of a reduced loop) has converged.
    converged = converged or met_criterion
    if flip is not None and settings.fixed_occupations is None:
        # PARSEC_SPIN_FLIP_SYMMETRY: a partly filled level of the up channel and its image in the down channel are
        # the symmetric ensemble of a pair that two solved channels may fill unequally (a moment the operation
        # forbids): run_spin_scf solves the loop again with both channels.
        # (per partner: the occupations of the weighted sectors of a subgroup before their weights)
        largest, lines = _flip_fractional_occupations(eigenvalues[0], occupation_result.occupations[0])
        if largest >= SPIN_FLIP_OCCUPATION_TOLERANCE:
            raise _SpinFlipFractional(largest, lines)
    stability = None
    safety = None
    # (a loop of PARSEC_SPIN_FLIP_SYMMETRY=late took the flip only where off's channels were images of each other)
    tested = flip is not None and converged and flip_test and getattr(flip, "mode", "auto") != "late"
    if (
        tested
        and getattr(flip, "stability", "off") == "auto"
        and getattr(flip, "continuation", "off") == "auto"
    ):
        # PARSEC_SPIN_FLIP_CONTINUATION=auto: the test of the converged state and the starts below it, from the down
        # channel alone (acceleration/SCF/spin_flip_stability.py); where one lies GAIN below the state of the route,
        # run_spin_scf continues from it without the flip.
        from parsec_python.acceleration.SCF.spin_flip_stability import FlipLoopState, entropy_term, flip_state_safety

        temperature = float(settings.fermi_temperature_kelvin)
        reference = float(energies.total) + entropy_term(occupation_result.occupations, weights, temperature)
        safety = flip_state_safety(
            FlipLoopState(
                system=system,
                flip=flip,
                solve=solvers[1],
                number_of_states=number_of_states,
                eigensolver_settings=eigensolver_settings,
                neighbors=neighbors,
                settings=settings,
                up_potential=last_input_potentials[0],
                up_eigenvalues=eigenvalues[0],
                fermi_level=float(fermi_level),
                ionic_potential=ionic_potential,
                hartree_potential=hartree_potential,
                reduced=reduced,
                adapter=scalar_field_adapter,
                orbital_density_builder=orbital_density_builder,
                volume_element=volume_element,
                total_energy=evaluate_total_energy,
                reference=reference,
                temperature=temperature,
                alpha_z_energy=float(getattr(system, "alpha_z_energy", 0.0) or 0.0),
            )
        )
        if safety.pays:
            best = safety.best
            restart = getattr(solvers[0], "start_new_scf_loop", None)
            up_state = restart() if callable(restart) else eigval_states[0]
            restart = getattr(solvers[1], "start_new_scf_loop", None)
            down_state = restart() if callable(restart) else safety.down_state
            field_of = scalar_field_adapter.field if reduced else (lambda values: values)
            raise _SpinFlipContinue(
                safety,
                start=(
                    np.asarray(to_public(field_of(best.up)), dtype=np.float64),
                    np.asarray(to_public(field_of(best.down)), dtype=np.float64),
                ),
                centre=(
                    np.asarray(to_public(last_input_potentials[0]), dtype=np.float64),
                    np.asarray(to_public(flip.down_field(last_input_potentials[0])), dtype=np.float64),
                ),
                hartree=np.asarray(to_public(hartree_potential), dtype=np.float64),
                states=(up_state, down_state),
                reference=reference,
            )
    elif tested and getattr(flip, "stability", "off") == "auto":
        # PARSEC_SPIN_FLIP_STABILITY=auto: the converged state against perturbations that break the operation, from
        # the response of the down channel (the up channel and its orbitals are not touched); run_spin_scf solves the
        # loop again with both channels where it is unstable.
        stability = _flip_stability(
            system,
            flip,
            solve=solvers[1],
            number_of_states=number_of_states,
            eigensolver_settings=eigensolver_settings,
            neighbors=neighbors,
            settings=settings,
            up_potential=last_input_potentials[0],
            up_eigenvalues=eigenvalues[0],
            fermi_level=fermi_level,
            ionic_potential=ionic_potential,
            hartree_potential=hartree_potential,
            reduced=reduced,
            adapter=scalar_field_adapter,
            orbital_density_builder=orbital_density_builder,
            volume_element=volume_element,
        )
        if stability.unstable:
            raise _SpinFlipUnstable(stability)
    output_potentials = [
        ionic_potential + hartree_potential + xc_potential
        for xc_potential in xc_potentials
    ]
    if reduced:
        # The result holds full-grid fields, as that of the full-grid loop.
        densities = [to_public(values) for values in densities]
        xc_potentials = [to_public(values) for values in xc_potentials]
        last_input_potentials = [to_public(values) for values in last_input_potentials]
        output_potentials = [to_public(values) for values in output_potentials]
        mixed_potentials = [to_public(values) for values in mixed_potentials]
        hartree_potential = to_public(hartree_potential)
    dispersion = None
    if dispersion_wanted(settings, converged):
        dispersion, energies = dispersion_after_convergence(
            system, densities[0] + densities[1], energies
        )

    run_timings = RunTimings(
        preparation=system.timings,
        initial_hartree_seconds=initial_hartree_seconds,
        initial_xc_seconds=initial_xc_seconds,
        hamiltonian_binding_seconds=hamiltonian_binding_total,
        diagonalization_seconds=diagonalization_total,
        occupations_density_seconds=occupations_density_total,
        hartree_seconds=hartree_total,
        xc_seconds=xc_total,
        mixing_energy_seconds=mixing_energy_total,
        total_seconds=time.perf_counter() - scf_start,
        **({} if level_detail is None else {"detail": level_detail}),
    )
    # Weighted sectors: the lists of every partner of a weighted state (run_scf), and the states the
    # sectors solved with their occupations times their weights (the nonlocal forces of the sectors).
    public = _public_channels(eigenvalues, occupations, representations, wavefunctions, state_weights)
    # The orbitals are stored once: host channels are views of the joined
    # array, device channels the blocks the backend returned.
    stacked, channel_orbitals = _join_orbitals(public[0][3], public[1][3])
    channels = tuple(
        SpinChannelResult(
            eigenvalues=public[spin][0],
            occupations=public[spin][1],
            wavefunctions=channel_orbitals[spin],
            representations=public[spin][2],
            sector_states=(
                None
                if state_weights[spin] is None
                else (
                    np.asarray(representations[spin]),
                    np.asarray(occupations[spin]),
                    np.asarray(state_weights[spin], dtype=np.int64),
                )
            ),
            density=densities[spin],
            xc_potential=xc_potentials[spin],
            input_effective_potential=last_input_potentials[spin],
            output_effective_potential=output_potentials[spin],
            next_effective_potential=mixed_potentials[spin],
            electron_count=float(electrons[spin]),
            eigensolver_state=eigval_states[spin],
        )
        for spin in range(2)
    )
    result = SinglePointResult(
        converged=converged,
        iterations=len(history),
        atoms=system.atoms,
        electron_count=system.electron_count,
        grid=system.grid,
        pseudopotentials=system.pseudopotentials,
        density=densities[0] + densities[1],
        core_density=system.core_density,
        ionic_potential=system.ionic_potential,
        # V_H of the Poisson solve, also where the loop ran with V' (PARSEC_HARTREE_VARIATIONAL)
        hartree_potential=electrostatic_hartree(system, hartree_potential),
        # Averages of the two channels, as forcloc.f90 forms them.
        xc_potential=0.5 * (xc_potentials[0] + xc_potentials[1]),
        input_effective_potential=0.5
        * (last_input_potentials[0] + last_input_potentials[1]),
        output_effective_potential=0.5 * (output_potentials[0] + output_potentials[1]),
        next_effective_potential=0.5 * (mixed_potentials[0] + mixed_potentials[1]),
        nonlocal_operator=system.nonlocal_operator,
        eigenvalues=np.concatenate([values[0] for values in public]),
        occupations=np.concatenate([values[1] for values in public]),
        wavefunctions=stacked,
        fermi_level=fermi_level,
        energies=energies,
        history=history,
        timings=run_timings,
        representations=np.concatenate([values[2] for values in public]),
        atomic_reference_correction=system.atomic_reference_correction,
        eigensolver_state=tuple(eigval_states),
        spin=channels,
        hubbard=hubbard_state,
        dispersion=dispersion,
        **({} if meta is None else {"meta_gga": meta.record()}),
    )
    check = getattr(scalar_field_adapter, "check_spin_result", None)
    if exchange_term is None and callable(check):
        # The symmetry guard of a reduced loop (the symmetry sectors of the
        # accelerated driver): a result that breaks the symmetry of the atoms
        # or has fractional occupations is not kept.
        check(result)
    if stability is not None and callback is not None:
        # (PARSEC_SPIN_FLIP_STABILITY: the note of a kept state)
        callback(_spin_flip_stable_record(stability))
    if late_switched is not None:
        result.timings.detail["spin_flip_switch_iteration"] = float(late_switched)
    if pending_flip is not None and callback is not None:
        # (PARSEC_SPIN_FLIP_SYMMETRY=late: a loop that never switched says so)
        callback(
            _late_flip_record(
                pending_flip,
                switched=False,
                iteration=len(history),
                lines=(
                    # (the fields of the result are on the full grid here)
                    f"at the end: densities {_flip_density_deviation(pending_flip, densities, None, volume_element):.3e} "
                    f"electrons apart from images of each other, levels "
                    f"{(late_asymmetries[-1] if late_asymmetries else float('nan')):.3e} Ry"
                    + (
                        f"; where the levels allowed the switch, the input potentials at least {min(late_ratios):.3g} "
                        "times the residual apart"
                        if late_ratios
                        else ""
                    ),
                ),
            )
        )
    if safety is not None:
        # (PARSEC_SPIN_FLIP_CONTINUATION=auto: no start lies GAIN below the state of the route, which is kept)
        result.timings.detail.update(
            {
                "spin_flip_stability_largest": float(safety.largest),
                "spin_flip_stability_probes": float(safety.probes),
                "spin_flip_stability_seconds": float(safety.seconds),
                "spin_flip_candidates_seconds": float(safety.candidate_seconds),
                "spin_flip_candidate_gain": float(safety.gain),
                "spin_flip_down_passes": float(safety.passes),
            }
        )
        if callback is not None:
            callback(
                _spin_flip_safety_record(
                    safety,
                    route="spin_flip_stable",
                    reason=(
                        "the state of the converged loop is kept: "
                        + (
                            "it is unstable against perturbations that break the operation, "
                            if safety.unstable
                            else "it is stable against perturbations that break the operation, "
                        )
                        + f"and no start tried lies {safety.tolerance:g} Ry below it"
                    ),
                )
            )
    return result


# ----------------------------------------------------------------------------
# run_scf_spin_polarized: collinear spin polarization for isolated and periodic
# (Gamma-point) single points, the base of the perturbative and self-consistent
# spin-orbit drivers.
#
# Collinear spin-polarized PARSEC-style SCF for isolated and periodic
# (Gamma-point) single points.
#
# This is the spin-polarized counterpart to :mod:`single_point` (and, for
# periodic inputs, :mod:`pbc`), following the same overall potential-mixing
# algorithm but iterating two independent Kohn--Sham eigenproblems (spin up,
# spin down) that share the grid, kinetic operator, KB nonlocal projectors,
# local ionic potential, and Hartree potential, and differ only in their
# exchange-correlation field:
#
# ``H_sigma[V_eff,sigma] = -nabla_FD^2 + diag(V_eff,sigma) + V_NL``,
# ``V_eff,sigma = V_ion,local + V_H[rho] + V_xc,sigma[rho_up,rho_down]``.
#
# Matching ``flevel.f90``, both channels' eigenvalues are pooled into *one*
# Fermi-level bisection with a shared chemical potential (electron count is
# fixed; the magnetic moment ``N_up-N_down`` is free), not two independently
# constrained channels.
#
# ``run_scf_spin_polarized`` accepts either
# :class:`~parsec_python.SCF.single_point.PreparedSinglePointSystem` (isolated)
# or :class:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem`
# (periodic, Gamma-point only) unchanged: every attribute/method this module
# uses (``ionic_potential``, ``initial_density``, ``core_density``,
# ``solve_hartree``, ``hamiltonian``, ``grid.volume_element``, ``grid.size``,
# ``electron_count``, ``ion_ion_energy``, ``alpha_z_energy``, ``atoms``,
# ``input``) already has the same shape and meaning on both.  Neither
# :mod:`parsec_python.Eigensolvers.perturbative_soc` nor real periodic k-point
# sampling is included: only the Gamma-point real-arithmetic SCF loop is
# shared, matching :mod:`pbc`'s own scope.
#
# Unlike :func:`~parsec_python.SCF.single_point.run_scf`, this function does
# not yet carry PARSEC's full per-iteration timing/history instrumentation --
# see :class:`~parsec_python.models.SpinPolarizedSinglePointResult`.  It is
# wired into the CLI/Output path (:mod:`parsec_python.cli`,
# :class:`~parsec_python.Output.ParsecTextReporter`), and produces a converged
# collinear ground state (eigenvalues, occupations, and orbitals for both
# channels) that, for isolated inputs,
# :mod:`parsec_python.Eigensolvers.perturbative_soc` then uses for PARSEC's
# default perturbative spin-orbit correction.
# ----------------------------------------------------------------------------


_PreparedSystem = Union[PreparedSinglePointSystem, PeriodicPreparedSinglePointSystem]


SPIN_START_ENV = "PARSEC_SPIN_START"


def spin_start_setting() -> str:
    """``PARSEC_SPIN_START``: the start of the two reference spin loops that began from one polarization fraction.

    Those are the collinear k-point loop (:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`: spin
    cells with k-points, the 1x1x1 run of a spin cell at the Gamma point on the reference command line, its forces,
    relaxations and dynamics there), :func:`run_scf_spin_polarized` (a periodic spin single point without a device,
    an isolated spin run with spin-orbit species) and the two self-consistent spin-orbit loops with collinear spin
    (:func:`parsec_python.SCF.self_consistent_soc_spin_polarized.run_self_consistent_soc_spin_polarized`, isolated,
    and :func:`parsec_python.SCF.kpoints_soc.run_self_consistent_soc_kpoints_spin_polarized`, periodic).  Before, they
    started from ``(1 +/- p)/2`` of the starting density with ``p`` the ionic-charge weighted mean of the species'
    ``Initial_Spin_Polarization`` (:func:`_weighted_initial_polarization`), which is zero for an antiferromagnetic or
    broken-symmetry start of two labels with opposite values: such a run started non-magnetic and stayed there (a
    Cr2 pair in a 14.4 bohr cube ended 0.68 Ry above its antiferromagnetic state, without converging in 300
    iterations; MnO in 64 atoms 174 Ry off).

    * ``auto`` (the default since 2026-11-21): PARSEC 1.5's start where the species of the atoms start with values of
      both signs (:func:`opposite_spin_starts`), the former start elsewhere (one value, or values of one sign, whose
      single fraction keeps the sign of the pattern; bit for bit as before);
    * ``initchrg``: PARSEC 1.5's start for every input (``initchrg.f90``: ``p_fac = 1 +/- spol`` of each atom's
      species, the channels scaled by the factor of the total density, so that they add up to twice it;
      :func:`initchrg_spin_start`), as :func:`run_spin_scf` and the accelerated driver start; with one value it changes
      only the path (the same energies; one more iteration for V- in diamond, 72 against 33 for the SCAN silicon cell
      of the meta-GGA tests);
    * ``weighted``: the former start for every input, bit for bit.

    PARSEC 1.5's start takes the first Hartree potential of the normalized total starting density (``rho(:,1)``); the
    k-point loop takes that one also for the channel pair a hybrid's Fock loop hands in under ``initchrg`` (their
    sum, twice the density, before and under ``auto``).  The command lines read the value before every
    spin-polarized run.
    """

    import os

    value = os.environ.get(SPIN_START_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "initchrg", "weighted"):
        raise ValueError(f"{SPIN_START_ENV} must be auto, initchrg or weighted, got {value!r}")
    return value


def initchrg_start_needed(system: _PreparedSystem) -> bool:
    """Whether the loop starts as ``initchrg.f90`` (:func:`spin_start_setting`): always under ``initchrg``, never
    under ``weighted``, and under ``auto`` where the species of the atoms start with values of both signs
    (:func:`opposite_spin_starts`: the one fraction of the former start has a single sign and loses their pattern)."""

    setting = spin_start_setting()
    if setting != "auto":
        return setting == "initchrg"
    return opposite_spin_starts(system)


def opposite_spin_starts(system: _PreparedSystem) -> bool:
    """Whether the species of the atoms start with ``Initial_Spin_Polarization`` values of both signs.

    That is where the former start, one polarization fraction of the whole cell (:func:`_weighted_initial_polarization`),
    loses the pattern of the input: an antiferromagnetic or broken-symmetry start of two labels (zero there for
    ``+p`` and ``-p``).  Values of one sign (a species at ``0.3`` beside another at the default ``0.1``, or at ``0``)
    keep a moment of that sign everywhere in the former start, whose loop finds the magnetic state on its own; PARSEC
    1.5's doubled first density only lengthens their path there (SCAN silicon with labels at 0.3 and 0.1: 86 against
    45 iterations, and a locally magnetized state 1.4e-5 Ry from the non-magnetic one; A100, 2026-11-21).
    """

    specifications = system.input.pseudopotentials
    values = [float(specifications[atom.symbol].initial_spin_polarization) for atom in system.atoms]
    return bool(values) and min(values) < 0.0 < max(values)


def initchrg_spin_start(system: _PreparedSystem) -> tuple[np.ndarray, np.ndarray]:
    """The starting densities of the two channels of ``initchrg.f90`` (:func:`spin_start_setting`).

    ``system.initial_spin_densities`` where the preparation built them; ``(1 + p) rho0`` and ``(1 - p) rho0`` where
    every species starts with the same ``Initial_Spin_Polarization`` ``p`` (``rho0`` the normalized starting density,
    :func:`parsec_python.SCF.initial_spin_densities` of one species); otherwise the per-species superposition of a
    periodic cell (:func:`parsec_python.SCF.pbc.periodic_spin_start`, what ``run_spin_scf`` of the cell starts from).
    """

    given = getattr(system, "initial_spin_densities", None)
    if given is not None:
        return tuple(np.asarray(value, dtype=np.float64) for value in given)
    specifications = system.input.pseudopotentials
    polarizations = {float(specifications[atom.symbol].initial_spin_polarization) for atom in system.atoms}
    if len(polarizations) <= 1:
        value = polarizations.pop() if polarizations else 0.0
        total = np.asarray(system.initial_density, dtype=np.float64)
        return (1.0 + value) * total, (1.0 - value) * total
    if getattr(system.input, "periodic_cell", None) is None:
        raise ValueError(
            "the system was not prepared for a spin-polarized calculation: it has no starting spin densities"
        )
    from .pbc import periodic_spin_start

    return tuple(np.asarray(value, dtype=np.float64) for value in periodic_spin_start(system))


def _weighted_initial_polarization(system: _PreparedSystem) -> float:
    """Ionic-charge-weighted average of each species' ``Initial_Spin_Polarization``.

    PARSEC seeds the spin-polarized initial density per atom, via
    ``initchrg.f90``'s ``p_fac = 1 +/- p_pot%spol(itype)`` applied to each
    atom's own contribution before summing.  Splitting the *already summed*
    total initial density (as :func:`prepare_single_point` builds it, shared
    with the unpolarized path) can only apply one shared fraction, not a
    per-atom one; this weighted average is a deliberate simplification for
    the initial guess only.  It has no effect on the converged
    self-consistent solution beyond which local minimum SCF may find, and
    any nonzero value serves its sole purpose: breaking the up/down symmetry
    so SCF does not stay trapped at zero moment.  It is zero for an
    antiferromagnetic start of two labels with opposite values, which then
    starts and stays non-magnetic; the collinear k-point loop
    (:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`),
    :func:`run_scf_spin_polarized` and the two spin-orbit loops with collinear
    spin take the per-species start of ``initchrg.f90`` there instead since
    2026-11-21 (:func:`spin_start_setting`, :func:`opposite_spin_starts`;
    ``PARSEC_SPIN_START=weighted`` keeps this one everywhere).
    """
    specifications = system.input.pseudopotentials
    total_charge = 0.0
    weighted = 0.0
    for atom in system.atoms:
        charge = system.pseudopotentials[atom.symbol].ionic_charge
        weighted += charge * specifications[atom.symbol].initial_spin_polarization
        total_charge += charge
    if total_charge <= 0:
        return 0.0
    return weighted / total_charge


def _number_of_states_per_channel(system: _PreparedSystem) -> int:
    """States requested from *each* spin channel's eigensolver.

    Each spin channel's orbitals hold at most one electron (no spin
    degeneracy), so in the fully polarized limit one channel alone could
    need close to the full electron count.  Unlike the unpolarized
    ``_number_of_states`` (sized for ``N_e/2`` two-electron orbitals plus a
    buffer), an explicit ``States_Num`` here is taken as a per-channel count
    directly, matching Fortran's ``nstate = nstate*elec_st%mxwd`` scaling
    convention for extra spinor/spin components.
    """
    requested = system.input.scf.number_of_states
    if requested is None:
        requested = int(np.ceil(system.electron_count)) + 6
    requested = int(requested)
    if requested + system.input.eigensolver.subspace_buffer >= system.grid.size:
        raise ValueError("the grid is too small for the requested states and eigensolver buffer")
    return requested


def _eigval_settings(
    system: _PreparedSystem,
    filter_degree: int,
    *,
    tolerance: float | None = None,
    neighbors: np.ndarray | None = None,
) -> EigvalSettings:
    """The policy of :func:`eigval_settings_for` (the same for every solver)."""

    return eigval_settings_for(
        system.input.eigensolver,
        filter_degree,
        tolerance=tolerance,
        neighbors=neighbors,
    )


def run_scf_spin_polarized(
    system: _PreparedSystem,
    *,
    callback: Callable[[SpinPolarizedSCFIteration], None] | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
) -> SpinPolarizedSinglePointResult:
    """Run collinear spin-polarized PARSEC-style potential mixing.

    The XC functional is ``system.input.scf.xc_functional``: ``'ca'`` is
    LSDA (``exc_spn.f90``); ``'pbe'`` is collinear spin-polarized PBE
    (:func:`~parsec_python.V_xc.pbe_spin_polarized`).

    LDA+U (``system.hubbard``) enters as in :func:`run_spin_scf`: from the
    second iteration on, ``D`` of each channel from the orbitals of the
    iteration before (``calcden.f90z``) joins that channel's Hamiltonian and
    ``E_U`` the energy; the TS correction (``Van_Der_Waals``) is evaluated on
    the converged density.
    ``iteration_state_callback`` receives the fields of every finished
    iteration (``Save_Intermediate_Charge_Density`` /
    ``Save_Intermediate_Eigenvalues``, parsec.F90:1381-1402: PARSEC writes
    ``parsec.dat`` and ``eigen.dat`` after every SCF iteration); it observes
    only.
    """
    settings = system.input.scf
    if not settings.spin_polarized:
        raise ValueError(
            "run_scf_spin_polarized requires SCFSettings.spin_polarized=True"
        )
    if getattr(settings, "is_meta_gga", False):
        raise NotImplementedError(
            f"run_scf_spin_polarized has no meta-GGA term ({settings.xc_functional!r}): a meta-GGA runs on "
            "run_spin_scf, or for a periodic cell on the k-point spin loop at the Gamma point"
        )
    hubbard_projectors = getattr(system, "hubbard", None)
    hubbard_state = None

    number_of_states = _number_of_states_per_channel(system)
    ionic_potential = system.ionic_potential

    if initchrg_start_needed(system):
        # PARSEC 1.5's per-species start (initchrg.f90), the Hartree potential of the total starting density
        density_up, density_down = initchrg_spin_start(system)
        hartree_source = np.asarray(system.initial_density, dtype=np.float64)
    else:
        polarization = _weighted_initial_polarization(system)
        density_up = 0.5 * (1.0 + polarization) * system.initial_density
        density_down = 0.5 * (1.0 - polarization) * system.initial_density
        hartree_source = density_up + density_down

    initial_hartree = system.solve_hartree(
        hartree_source, initial_potential=-ionic_potential
    )
    hartree_potential = initial_hartree.potential
    xc = system.evaluate_xc_spin_polarized(density_up, density_down)
    input_potential_up = ionic_potential + hartree_potential + xc.potential_up
    input_potential_down = ionic_potential + hartree_potential + xc.potential_down

    eigval_state_up: EigvalState | None = None
    eigval_state_down: EigvalState | None = None
    # One mixer history over the concatenated (V_up, V_down) vector, the
    # spin-polarized analog of mixing a single scalar field -- PARSEC treats
    # vnew(:,1:2) as one combined potential for mixing purposes.
    mixer = preconditioned_mixer(make_mixer(system.input.mixing), system, channels=2)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    degree_schedule = LaterFilterDegree.for_loop(filter_degree, minimum_filter_degree)
    eigensolver_settings = system.input.eigensolver
    if eigensolver_settings.method == "none":
        raise ValueError(
            "Eigensolver: none is PARSEC's TEST mode: setup only, no SCF"
        )
    neighbors = preconditioner_neighbors(system, eigensolver_settings)
    diagonalization_tolerance = first_diagonalization_tolerance(eigensolver_settings)

    eigenvalues_up = eigenvalues_down = np.empty(0)
    occupations_up = occupations_down = np.empty(0)
    wavefunctions_up = wavefunctions_down = np.empty((system.grid.size, 0))
    fermi_level = float("nan")
    energies = None
    converged = False

    for iteration in range(1, settings.max_iterations + 1):
        eigval_settings = _eigval_settings(
            system,
            filter_degree,
            tolerance=diagonalization_tolerance,
            neighbors=neighbors,
        )

        hubbard_terms_now = {}
        if hubbard_projectors is not None:
            if iteration > 1:
                # calcden.f90z: N and D of both channels from the orbitals and
                # occupations of the iteration before.
                hubbard_state = hubbard_terms_from_orbitals(
                    hubbard_projectors,
                    [wavefunctions_up, wavefunctions_down],
                    [occupations_up, occupations_down],
                )
            hubbard_terms_now = {
                "hubbard_energy": 0.0 if hubbard_state is None else hubbard_state.energy
            }
            hamiltonian_up = system.hamiltonian(
                input_potential_up, hubbard_operator(hubbard_projectors, hubbard_state, 0)
            )
        else:
            hamiltonian_up = system.hamiltonian(input_potential_up)
        solution_up = solve_eigval(
            hamiltonian_up.as_linear_operator(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state_up,
        )
        eigval_state_up = solution_up.state
        eigenvalues_up = solution_up.eigenvalues
        wavefunctions_up = solution_up.vectors

        if hubbard_projectors is not None:
            hamiltonian_down = system.hamiltonian(
                input_potential_down,
                hubbard_operator(hubbard_projectors, hubbard_state, 1),
            )
        else:
            hamiltonian_down = system.hamiltonian(input_potential_down)
        solution_down = solve_eigval(
            hamiltonian_down.as_linear_operator(),
            number_of_states,
            settings=eigval_settings,
            state=eigval_state_down,
        )
        eigval_state_down = solution_down.state
        eigenvalues_down = solution_down.eigenvalues
        wavefunctions_down = solution_down.vectors

        # Pool both channels for PARSEC's single shared Fermi level
        # (flevel.f90): each pooled level is one non-degenerate orbital, so
        # degeneracy=1 here (contrast the unpolarized path's degeneracy=2).
        pooled = np.concatenate([eigenvalues_up, eigenvalues_down])
        if settings.fixed_occupations is not None:
            # occup.in (a negative Fermi_Temp): each channel takes its own
            # section in order of energy, as run_spin_scf does; no Fermi level.
            fixed = _spin_occupations((eigenvalues_up, eigenvalues_down), system, settings)
            occupations_up, occupations_down = (
                np.asarray(values, dtype=np.float64) for values in fixed.occupations
            )
            fermi_level = fixed.fermi_level
        else:
            order = np.argsort(pooled, kind="stable")
            occupation_result = fermi_occupations(
                pooled[order],
                system.electron_count,
                settings.fermi_temperature_kelvin,
                degeneracy=1.0,
            )
            pooled_occupations = np.empty_like(pooled)
            pooled_occupations[order] = occupation_result.occupations
            occupations_up = pooled_occupations[: eigenvalues_up.size]
            occupations_down = pooled_occupations[eigenvalues_up.size :]
            fermi_level = occupation_result.fermi_level

        density_up = density_from_orbitals(
            wavefunctions_up, occupations_up, system.grid.volume_element, degeneracy=1.0
        )
        density_down = density_from_orbitals(
            wavefunctions_down,
            occupations_down,
            system.grid.volume_element,
            degeneracy=1.0,
        )
        symmetrization = ()
        density_symmetry = getattr(system, "density_symmetry", None)
        if density_symmetry is not None:
            # Symmetrize_Charge_Density of a periodic cell: isp = 1 (total,
            # reported), 2 and 3 (parsec.F90 lines 1223-1232).
            volume_element = system.grid.volume_element
            _, total_report = symmetrize_density(
                density_up + density_down, density_symmetry, volume_element
            )
            density_up, up_report = symmetrize_density(
                density_up, density_symmetry, volume_element
            )
            density_down, down_report = symmetrize_density(
                density_down, density_symmetry, volume_element
            )
            symmetrization = (total_report, up_report, down_report)

        hartree = system.solve_hartree(
            density_up + density_down, initial_potential=hartree_potential
        )
        hartree_potential = hartree.potential
        xc = system.evaluate_xc_spin_polarized(density_up, density_down)
        output_potential_up = ionic_potential + hartree_potential + xc.potential_up
        output_potential_down = ionic_potential + hartree_potential + xc.potential_down

        input_combined = np.concatenate([input_potential_up, input_potential_down])
        output_combined = np.concatenate([output_potential_up, output_potential_down])
        density_combined = np.concatenate([density_up, density_down])
        metrics = potential_residual_metrics(
            input_combined,
            output_combined,
            density_combined,
            system.grid.volume_element,
            system.electron_count,
        )

        energies = total_energy_spin_polarized(
            eigenvalues_up,
            occupations_up,
            eigenvalues_down,
            occupations_down,
            density_up,
            density_down,
            input_potential_up,
            input_potential_down,
            ionic_potential,
            hartree_potential,
            xc.potential_up,
            xc.potential_down,
            xc.total_energy,
            system.ion_ion_energy,
            system.grid.volume_element,
            alpha_z_energy=system.alpha_z_energy,
            # totnrg.f90 adds etot_plusu, zero before the first calcden.
            **hubbard_terms_now,
        )

        mixed_combined = mixer.mix(input_combined, output_combined, iteration=iteration)
        input_potential_up = mixed_combined[: system.grid.size]
        input_potential_down = mixed_combined[system.grid.size :]

        selected_residual = (
            metrics.plain if settings.use_plain_residual else metrics.weighted
        )
        if eigensolver_settings.dynamic_tolerance:
            # This loop measures one weighted SRE over both channels.
            diagonalization_tolerance = dynamic_diagonalization_tolerance(
                diagonalization_tolerance,
                (metrics.weighted,),
                eigensolver_settings.tolerance,
            )
        if callback is not None:
            callback(
                SpinPolarizedSCFIteration(
                    iteration=iteration,
                    # PARSEC_FILTER_DEGREE_STALL: the NOTE of a restoration (once)
                    filter_degree_note=degree_schedule.take_note(),
                    weighted_residual=metrics.weighted,
                    plain_residual=metrics.plain,
                    energies=energies,
                    eigenvalues_up=tuple(float(value) for value in eigenvalues_up),
                    eigenvalues_down=tuple(float(value) for value in eigenvalues_down),
                    occupations_up=tuple(float(value) for value in occupations_up),
                    occupations_down=tuple(float(value) for value in occupations_down),
                    fermi_level=float(fermi_level),
                    density_symmetrization=symmetrization,
                    **hubbard_iteration_record(hubbard_state),
                    # Dynamic_Diag_Tol: the tolerance of the next iteration.
                    **(
                        {"next_eigensolver_tolerance": diagonalization_tolerance}
                        if eigensolver_settings.dynamic_tolerance
                        else {}
                    ),
                )
            )
        if iteration_state_callback is not None:
            from .single_point import IterationChannelFields, SCFIterationFields

            iteration_state_callback(
                SCFIterationFields(
                    iteration=iteration,
                    eigenvalues=np.concatenate([eigenvalues_up, eigenvalues_down]),
                    occupations=np.concatenate([occupations_up, occupations_down]),
                    representations=np.empty(0, dtype=np.int32),
                    wavefunctions=None,
                    density=density_up + density_down,
                    next_effective_potential=0.5
                    * (input_potential_up + input_potential_down),
                    spin=(
                        IterationChannelFields(
                            eigenvalues_up,
                            occupations_up,
                            density_up,
                            input_potential_up,
                            wavefunctions=wavefunctions_up,
                        ),
                        IterationChannelFields(
                            eigenvalues_down,
                            occupations_down,
                            density_down,
                            input_potential_down,
                            wavefunctions=wavefunctions_down,
                        ),
                    ),
                )
            )
        # PARSEC_FILTER_DEGREE_STALL (SCF/filter_degree.py): the degree lowered near convergence, and the
        # starting degree back where the loop stalls after that
        filter_degree = degree_schedule.after_iteration(
            iteration, metrics.weighted, settings.approach_criterion, settings.degree_decrease_after
        )
        if selected_residual < settings.convergence_criterion:
            converged = True
            break

    dispersion = None
    if dispersion_wanted(settings, converged):
        dispersion, energies = dispersion_after_convergence(
            system, density_up + density_down, energies
        )
    return SpinPolarizedSinglePointResult(
        converged=converged,
        iterations=iteration,
        atoms=system.atoms,
        electron_count=system.electron_count,
        eigenvalues_up=eigenvalues_up,
        eigenvalues_down=eigenvalues_down,
        occupations_up=occupations_up,
        occupations_down=occupations_down,
        wavefunctions_up=wavefunctions_up,
        wavefunctions_down=wavefunctions_down,
        fermi_level=fermi_level,
        density_up=density_up,
        density_down=density_down,
        energies=energies,
        scf_potentials={
            "potential_up": output_potential_up,
            "potential_down": output_potential_down,
        },
        hubbard=hubbard_state,
        dispersion=dispersion,
    )


__all__ = [
    "SPIN_NAMES",
    "SpinOrbitalBlocks",
    "run_scf_spin_polarized",
    "run_spin_scf",
]
