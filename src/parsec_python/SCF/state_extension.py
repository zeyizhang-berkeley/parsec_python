"""More states where an SCF loop stalls on a truncated occupied spectrum (``PARSEC_STATES_CLOSE_MULTIPLET``).

``States_Num`` fixes how many levels the eigensolver returns to the occupations.  Where the highest of them holds a
non-negligible occupation (a small gap or a high ``Fermi_Temp``) and belongs to a degenerate multiplet whose other
partners lie beyond ``States_Num``, every iteration returns another part of the multiplet: the density of a part of a
degenerate subspace depends on which part it is, so the residual of the loop stops falling at a floor of about 0.06
times the occupation of the cut states (:mod:`parsec_python.Output.state_truncation`, whose NOTE reports such a loop
under every value of the switch).  An 8-atom silicon cube at the Gamma point with ``States_Num`` 20 or 21 and
``Fermi_Temp`` 300 K (the six-fold lowest conduction level cut after four or five of its states) keeps its residual
near 5e-6 Ry on the reference solver; with all six it converges in 15 iterations.

:class:`MultipletCloser` (``auto``) acts only on a loop that stalls:

* the residual of the last iteration is above the SCF criterion and has fallen by less than a factor
  1 / :data:`STALL_RATIO` over the :data:`STALL_WINDOW` iterations before it, counted from the third iteration after
  the start of the loop or after its last extension (the levels of a first filter are not converged), and
* the highest returned state of a channel holds more than :data:`CRITERION_FACTOR` times the criterion.

The loop then takes the levels of the eigensolver's buffer (``Subspace_Buffer_Size``, whose Ritz values the solver
keeps) up to the first whose Fermi occupation at the present Fermi level is below that threshold, so that the highest
returned state holds less than it (at most the whole buffer per step), from the next iteration on, and its mixer
begins a new history; it may do so again while the loop keeps stalling, at most :data:`MAXIMUM_EXTENSIONS` times.  A
loop that continues one (an ionic step, a warm start) keeps the raised count; the inner loops of a hybrid keep the
count of the loop before them and take no state of their own.

The loops are those on the full grid (:func:`parsec_python.SCF.run_scf` and the spin loop of
:func:`parsec_python.SCF.spin_polarized.run_spin_scf`: the reference solver, the accelerated full grid) and the loops
in one sector of the identity (the full grid of the MPI runner), whose solver grows its count; not the symmetry
sectors of a point group, the k-point loops, ``occup.in`` or ``Fermi_Temp`` 0.

``PARSEC_STATES_CLOSE_MULTIPLET``:

* ``off`` (the default): ``States_Num`` as given, bit for bit as before;
* ``auto``: as above (opt-in: a raised count moves the energy of a structure by the truncation it removes, so the
  energy of a relaxation or of a finite-difference step may jump where one structure stalls and the next does not).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..Occupations.fermi_dirac import BOLTZMANN_RYDBERG_PER_KELVIN

SWITCH = "PARSEC_STATES_CLOSE_MULTIPLET"
#: The highest returned state counts as held above this many times the SCF criterion (the residual floor of a cut
#: multiplet is about 0.06 times the occupation of its states: silicon cube 8e-5 -> 5e-6 Ry, diamond cube at 3000 K
#: 1.4e-4 -> 1e-5 Ry), and the extension stops at the first level below it.
CRITERION_FACTOR = 3.0
#: A loop stalls where its residual fell by less than a factor 1 / STALL_RATIO over STALL_WINDOW iterations (eight
#: iterations instead let the transient plateaus of slightly displaced cells, about seven long, decide from one
#: structure to the next whether the count is raised; three raise it on every such plateau) ...
STALL_RATIO = 0.5
STALL_WINDOW = 3
#: ... counted from this many iterations after the start of the loop or its last extension.
FIRST_ITERATION = 3
#: Extensions of one loop at most.
MAXIMUM_EXTENSIONS = 8


def close_multiplet_setting() -> str:
    """The value of ``PARSEC_STATES_CLOSE_MULTIPLET`` (``off`` or ``auto``); any other value is an error."""

    value = os.environ.get(SWITCH, "off").strip().lower() or "off"
    if value not in ("auto", "off"):
        raise ValueError(f"{SWITCH} must be off or auto, got {value!r}")
    return value


def occupied_threshold(criterion: float | None) -> float:
    """The occupation of the highest returned state above which a loop with SCF criterion ``criterion`` (Ry) counts
    as truncated."""

    if criterion is None or not np.isfinite(criterion) or criterion <= 0:
        raise ValueError("the SCF criterion must be positive")
    return CRITERION_FACTOR * float(criterion)


def _host(values) -> np.ndarray:
    get = getattr(values, "get", None)
    if callable(get) and not isinstance(values, np.ndarray):
        values = get()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def working_levels(eigensolution) -> np.ndarray | None:
    """The Ritz values of the whole working subspace of a solve (the returned states and the buffer), sorted;
    ``None`` where the solver keeps none.  The sector solver lists them for all its sectors
    (``working_eigenvalues``), the plain solvers in their saved state."""

    listed = getattr(eigensolution, "working_eigenvalues", None)
    if listed is not None:
        try:
            return np.sort(_host(listed))
        except (TypeError, ValueError):
            return None
    return state_levels(getattr(eigensolution, "state", None))


def state_levels(state) -> np.ndarray | None:
    """:func:`working_levels` of the saved state of a plain eigensolver."""

    subspace = getattr(state, "subspace", None)
    values = getattr(subspace, "eigenvalues", None)
    if values is None:
        return None
    try:
        return np.sort(_host(values))
    except (TypeError, ValueError):
        return None


def identity_reduction(adapter) -> bool:
    """Whether a scalar-field adapter keeps one value per grid point (one sector of the identity)."""

    for owner in (adapter, getattr(adapter, "reducer", None)):
        reduction = getattr(owner, "reduction", None)
        if reduction is None:
            continue
        wedge, full = getattr(reduction, "wedge_size", None), getattr(reduction, "full_size", None)
        if wedge is not None and full is not None:
            return int(wedge) == int(full)
    return False


@dataclass(frozen=True)
class StateCountChange:
    """One extension: from ``previous`` to ``states`` states after iteration ``iteration``."""

    iteration: int
    previous: int
    states: int
    occupation: float
    residual: float
    channel: str
    threshold: float

    def lines(self) -> tuple[str, ...]:
        where = f" ({self.channel})" if self.channel else ""
        return (
            f" NOTE: the SCF residual stalls at {self.residual:.2e} Ry and the highest computed state{where}, state "
            f"{self.previous}, holds an occupation of {self.occupation:.1e} (> {self.threshold:.1e}):",
            f"   States_Num {self.previous} -> {self.states} from the next iteration on, the levels of the "
            "eigensolver's buffer down to that occupation",
            f"   (the mixer begins a new history; {SWITCH}=off keeps States_Num).",
        )


class MultipletCloser:
    """The test of :mod:`this module <parsec_python.SCF.state_extension>` for one SCF loop.

    ``extend=False``: a loop that only keeps the count a loop before it raised (the inner loops of a hybrid)."""

    def __init__(self, maximum_states: int, criterion: float, temperature_kelvin: float, *, extend: bool = True):
        self.maximum_states = int(maximum_states)
        self.criterion = float(criterion)
        self.threshold = occupied_threshold(criterion)
        self.kbt = BOLTZMANN_RYDBERG_PER_KELVIN * float(temperature_kelvin)
        self.extend = bool(extend)
        self.extensions = 0
        self.restart_iteration = 0
        self.changes: list[StateCountChange] = []

    @classmethod
    def for_loop(cls, system, *, adapter=None, exchange: bool = False,
                 criterion: float | None = None) -> "MultipletCloser | None":
        """The test of a loop on ``system``: ``None`` under ``off``, in symmetry sectors (an ``adapter`` other than
        one sector of the identity: the sectors share out the states themselves), with ``occup.in`` occupations or at
        ``Fermi_Temp`` 0; one that only keeps a raised count in the inner loops of a hybrid (``exchange``: its
        operator holds the orbitals of the loop before).  ``criterion``: the SCF criterion of the loop (Ry)."""

        if close_multiplet_setting() == "off":
            return None
        if adapter is not None and not identity_reduction(adapter):
            return None
        scf = system.input.scf
        if scf.fixed_occupations is not None or scf.fermi_temperature_kelvin <= 0:
            return None
        buffer = int(getattr(system.input.eigensolver, "subspace_buffer", 0))
        return cls(
            int(system.grid.size) - buffer - 1,
            scf.convergence_criterion if criterion is None else criterion,
            scf.fermi_temperature_kelvin,
            extend=not exchange,
        )

    def carried(self, states: int, saved) -> int:
        """The state count of a loop that continues another (an ionic step, a warm start, a hybrid's inner loop): that
        of the saved eigensolver states ``saved`` where an extension before raised it, so that it is not found again
        at the cost of a new start of the eigensolver; ``states`` otherwise."""

        counts = [int(getattr(state, "requested_states", 0) or 0) for state in saved if state is not None]
        largest = max(counts, default=0)
        return largest if states < largest <= self.maximum_states else states

    def stalled(self, iteration: int, residuals) -> float | None:
        """The residual of the iteration before ``iteration`` where the loop stalls (see the module), else ``None``.
        ``residuals``: the residuals of the iterations of this loop before ``iteration``, the first first."""

        values = [float(value) for value in residuals]
        last = iteration - 1
        earlier = last - STALL_WINDOW
        if earlier < self.restart_iteration + FIRST_ITERATION or len(values) < last:
            return None
        recent, before = values[last - 1], values[earlier - 1]
        if not (np.isfinite(recent) and np.isfinite(before)) or recent <= self.criterion:
            return None
        return recent if recent > STALL_RATIO * before else None

    def check(self, iteration: int, states: int, channels, residuals, fermi_level: float) -> StateCountChange | None:
        """``channels``: ``(label, eigenvalues, occupations per state at most one, working levels or None)`` of every
        eigenproblem of the iteration; ``residuals`` as :meth:`stalled` takes them; ``fermi_level`` (Ry) that of the
        occupations.  Returns the change where the loop takes more states from the next iteration on, else ``None``."""

        if not self.extend or self.extensions >= MAXIMUM_EXTENSIONS or states >= self.maximum_states:
            return None
        residual = self.stalled(iteration, residuals)
        if residual is None or not np.isfinite(fermi_level) or self.kbt <= 0:
            return None
        best = None
        for label, eigenvalues, occupations, working in channels:
            values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
            filled = np.asarray(occupations, dtype=np.float64).reshape(-1)
            if working is None or values.size != states or filled.size != states:
                continue
            top = int(np.argmax(values))
            occupation = float(filled[top])
            if not occupation > self.threshold:
                continue
            beyond = np.sort(np.asarray(working, dtype=np.float64))[states:]
            if beyond.size == 0:
                continue
            exponent = np.clip((beyond - float(fermi_level)) / self.kbt, -700.0, 700.0)
            held = 1.0 / (np.exp(exponent) + 1.0)
            below = np.flatnonzero(held < self.threshold)
            # the first buffer level below the threshold becomes the highest returned state (the whole buffer where none)
            added = int(below[0]) + 1 if below.size else int(beyond.size)
            if best is None or added > best.states - states:
                best = StateCountChange(iteration, states, states + added, occupation, residual, label, self.threshold)
        if best is None:
            return None
        target = min(best.states, self.maximum_states)
        if target <= states:
            return None
        if target != best.states:
            best = StateCountChange(best.iteration, states, target, best.occupation, best.residual, best.channel,
                                    best.threshold)
        self.extensions += 1
        self.restart_iteration = iteration
        self.changes.append(best)
        return best


__all__ = [
    "CRITERION_FACTOR",
    "FIRST_ITERATION",
    "MAXIMUM_EXTENSIONS",
    "MultipletCloser",
    "STALL_RATIO",
    "STALL_WINDOW",
    "SWITCH",
    "StateCountChange",
    "close_multiplet_setting",
    "identity_reduction",
    "occupied_threshold",
    "state_levels",
    "working_levels",
]
