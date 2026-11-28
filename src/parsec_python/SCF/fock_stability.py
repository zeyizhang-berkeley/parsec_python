"""Two safeguards of the stop of a Fock loop (:func:`parsec_python.SCF.hybrid.run_hybrid_scf`).

The bias of ``dexx`` (``PARSEC_FOCK_DEXX_FLOOR``).  The loop stops when
``dexx = a_x [2 E_x(P, P0) - E_x(P0) - E_x(P)]`` falls below ``Exx_Tolerance``,
with ``E_x(P, P0)`` evaluated with the ACE ``V_ACE = -W M_s^-1 W^T`` of the
orbitals ``Phi`` of ``P0`` (``W = V_x Phi``, ``M = -Phi^T W``).  Where the pair
kernel of the discretization is not symmetric (the multipole boundary of a
finite-difference Poisson solve: ``max|M - M^T|/max|M|`` in the ACE line of
``parsec.out``), ``M`` is not symmetric either, the operator is built from its
symmetric part ``M_s``, and on the orbitals of ``P0`` it gives
``Phi^T V_ACE Phi = -(M_s + A^T M_s^-1 A)`` with ``A = (M - M^T)/2``, while the
pair sums of ``E_x`` see ``M_s`` only.  At ``P = P0`` the estimate is then not
0 but ``-a_x w sum_j f_j (A^T M_s^-1 A)_jj <= 0``: dexx has a negative floor,
second order in the asymmetry, which passes any tolerance below it.  Measured
(the host loop, a spin H3 chain with Hartree-Fock in a sphere of 6 bohr, h 0.5
bohr, multipole order 4, asymmetry 3.4e-5): the floor of the last operator
-6.7e-9 Ry, the last dexx -4.2e-9 Ry, which ended a loop asked for 1e-10; on
the A100 a loop asked for 1e-12 stopped at -1.2e-9 Ry on its spin-symmetric
saddle.  With ``auto`` (the default) every plain operator carries its own
estimate at its orbitals, ``bias = a_x [2 E_x^ACE(P0, P0) - 2 E_x(P0)]``
(:func:`operator_bias`, one more expectation value of the occupied orbitals),
and the estimate of the next step is ``dexx - bias`` (exactly 0 at the fixed
point; terms of order ``|A| |P - P0|`` remain, of either sign), tested as
before (below the tolerance).  In the inputs of ``examples`` the asymmetry is
at most 4e-10 and the bias at the level of round-off, so nothing there
changes.  ``off``: the uncorrected estimate, as before.

Spin-symmetric states (``PARSEC_FOCK_SPIN_STABILITY``).  In a spin-polarized
loop the state whose two channels are equal is stationary for every
functional that treats the channels alike, and the loop keeps it exactly:
from such a start (a semilocal start that lost its moment, an
``Initial_Spin_Polarization`` of 0) nothing but round-off breaks the
symmetry.  With a partly filled level the state can be a saddle of exact
exchange (which penalizes fractional occupations): an H3 chain with
Hartree-Fock at 6000 K, whose PBE start ends without a moment, converged on
the host at the default ``Exx_Tolerance`` to the spin-symmetric state 0.206 Ry
above the doublet (moment 1) that a loop reaches when round-off grows long
enough.  With ``auto`` (the default) a converged spin-polarized loop whose
channels hold the same occupations (sorted, within ``SYMMETRIC_SPREAD``) and a
partly filled level (``min(f, 1 - f)`` above ``FRACTIONAL_LEVEL``) is tested:
the next operator is that of the converged orbitals with ``PROBE_SHARE`` of
the electrons of each partly filled level of the down channel moved to the
same level of the up channel (a small spin-breaking displacement of the
density matrix, the direction in which exact exchange lowers the energy of a
fractional pair), and the loop goes on with plain steps (no outer DIIS, which
can be drawn to a saddle).  The moments of the outputs decide.  The ratios of
the displaced moment (less that of the symmetric state) from step to step
tend to the factor of the slowest mode, which a stable state holds below 1;
the first steps also carry faster modes, so the ratios rise towards it.
Where, from the second step on, the last ratio is at most ``STEP_RATIO`` and
its linear continuation (twice it less the one before) at most
``PREDICTED_RATIO``, the state is stable at once and the next operator is that
of copies of its occupied orbitals (kept for that), which brings the loop back
to the symmetric state and, where the copies hold every level the exact
exchange counts, ends it there (a plain operator of that state); a step that
raises the moment, and ratios near 1, which a slowly unstable saddle shows as
well as a marginally stable state (an H3 chain at 33000 K: 0.87, 1.00, 1.02,
and it lies 8.2e-4 Ry above the state its displacement grows to), leave the
verdict to the end of the continuation: the state it converges to is kept
where it is lower in ``E - TS`` (:func:`free_energy`), else the copies bring
the loop back.  A structure whose loop starts from the orbitals of one found
stable with a decay ratio of at most ``CARRY_RATIO`` (the steps of a
relaxation or of dynamics) is not tested again; one found stable only
marginally is.  Not tested: loops with the occupations of ``occup.in``,
without a Fermi temperature, of meshes of more than one k-point (a cell with a
single k-point, such as the Gamma point of a spin-polarized cell, which runs
as a mesh of one point, is tested), and states with integer occupations (an
instability of a closed shell, such as a stretched bond, needs a rotation of
the orbitals, which the sectors of the symmetry may not hold).  ``off``: no
test, as before.

Units Ry.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

#: ``PARSEC_FOCK_DEXX_FLOOR``: ``auto`` (default) or ``off`` (see the module).
FOCK_DEXX_FLOOR_ENV = "PARSEC_FOCK_DEXX_FLOOR"
#: ``PARSEC_FOCK_SPIN_STABILITY``: ``auto`` (default) or ``off`` (see the module).
FOCK_SPIN_STABILITY_ENV = "PARSEC_FOCK_SPIN_STABILITY"

#: Largest difference of the sorted occupations of the two channels of a spin-symmetric state.  At the stop of a
#: loop that approached a saddle the channels differ by about sqrt(dexx / K) (1e-3 at 1e-6 Ry; the H3 chain on the
#: host 2.4e-3); a magnetic state differs by order 1.
SYMMETRIC_SPREAD = 2.0e-2
#: ``min(f, 1 - f)`` above which a level counts as partly filled (the fractional report of the Fock loop).
FRACTIONAL_LEVEL = 1.0e-2
#: Share of the electrons of a partly filled down level moved to the up level by the displacement of the test
#: (H3 at 6000 K: 0.05 of an electron from 0.5 / 0.5; small enough to stay near the state, large enough to grow
#: out of round-off in two or three steps where the state is unstable).
PROBE_SHARE = 0.1
#: Free energy (Ry) by which the continued state must lie below the symmetric one to replace it, and within which
#: the two count as one state (far below chemical accuracy, far above the noise of converged totals).
MARGIN = 1.0e-6
#: Outer iterations the loop is given to go back to the symmetric state from the copies of its orbitals.
RETURN_ITERATIONS = 20
#: The largest ratio of the displaced moment from one step of the test to the next (from the second step on) and the
#: largest linear continuation of the ratios (twice the last less the one before) with which a state is stable at once.
#: Chosen on a ladder of states near the boundary of stability (H3 chains with two tables from 24000 to 60000 K, O2
#: with PBE0 from no moment at 15000 and 30000 K): every state that the continuation took to a lower one shows a
#: ratio above 0.9 at its second step (0.93 to 1.38), every one stable with these thresholds is stable at the end of
#: the continuation as well.
STEP_RATIO = 0.9
PREDICTED_RATIO = 0.95
#: A structure whose Fock loop starts from the orbitals of a structure found stable with a decay ratio at most this
#: (a relaxation or dynamics step), or taken over from such a structure, is not tested again; a state found stable
#: with a larger ratio is tested again.
CARRY_RATIO = 0.5
#: Boltzmann's constant in Ry/K.
KELVIN_RY = 6.33327186e-06


def _setting(name: str) -> str:
    value = os.environ.get(name, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{name} must be auto or off, got {value!r}")
    return value


def dexx_floor_setting() -> str:
    """``auto`` (default) or ``off``, from ``PARSEC_FOCK_DEXX_FLOOR``; another value is an error."""

    return _setting(FOCK_DEXX_FLOOR_ENV)


def spin_stability_setting() -> str:
    """``auto`` (default) or ``off``, from ``PARSEC_FOCK_SPIN_STABILITY``; another value is an error."""

    return _setting(FOCK_SPIN_STABILITY_ENV)


def operator_bias(engine, operator, sets, fraction: float) -> float:
    """``a_x [2 E_x^ACE(P0, P0) - 2 E_x(P0)]`` (Ry): the estimate ``dexx`` of the operator at its own orbitals.

    ``sets`` are the orbitals the operator was built from (before the engine drops them).
    """

    cross = float(engine.cross_energy(operator, sets))
    return float(fraction) * (2.0 * cross - 2.0 * float(operator.reference_energy))


def _partner(orbital_set) -> np.ndarray:
    partner = getattr(orbital_set, "partner_occupations", None)
    values = partner() if callable(partner) else orbital_set.occupations
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _levels(orbital_set) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(per partner, counted, weight)`` of every level of a channel, flat.

    The Bloch factors of a periodic cell (:class:`parsec_python.V_xc.bloch_exchange.BlochOrbitalSet`) hold one
    occupation array per k-point and the weights of the k-points; the orbitals of a cluster or a Gamma-point cell one
    array, the occupation of each partner of a state of a weighted sector beside it.
    """

    occupations = orbital_set.occupations
    if isinstance(occupations, (tuple, list)):
        weights = np.asarray(getattr(orbital_set, "weights", np.ones(len(occupations))), dtype=np.float64)
        flat = [np.ravel(np.asarray(values, dtype=np.float64)) for values in occupations]
        values = np.concatenate(flat) if flat else np.zeros(0)
        weight = np.concatenate([np.full(item.size, weights[t]) for t, item in enumerate(flat)]) if flat else values
        return values, values, weight
    counted = np.asarray(occupations, dtype=np.float64).reshape(-1)
    return _partner(orbital_set), counted, np.ones_like(counted)


def channel_electrons(sets) -> list[float]:
    """The electrons of each channel (a state of a weighted sector counts its partners, a k-point its weight)."""

    electrons = []
    for orbital_set in sets:
        _partners, counted, weight = _levels(orbital_set)
        electrons.append(float(np.dot(weight, counted)))
    return electrons


def entropy(sets) -> float:
    """``-sum (f ln f + (1 - f) ln (1 - f))`` over the partners of every state of every channel (k-points by weight)."""

    total = 0.0
    for orbital_set in sets:
        partner, counted, weight = _levels(orbital_set)
        inside = (partner > 0.0) & (partner < 1.0)
        if not np.any(inside):
            continue
        f = partner[inside]
        # the partners of a state of a weighted sector: its counted occupation over that of one partner
        partners = counted[inside] / f if counted.size == partner.size else np.ones_like(f)
        total -= float(np.dot(weight[inside] * partners, f * np.log(f) + (1.0 - f) * np.log1p(-f)))
    return total


def free_energy(total_energy: float, sets, temperature_kelvin: float) -> float:
    """``E - TS`` (Ry) of a converged loop: its total energy and the occupations of its orbitals ``sets``."""

    if not temperature_kelvin > 0.0:
        return float(total_energy)
    return float(total_energy) - KELVIN_RY * float(temperature_kelvin) * entropy(sets)


def single_point(result) -> bool:
    """Whether ``result`` is a cluster or a cell with a single k-point, whichever it is (the Gamma point of a
    spin-polarized cell runs as a mesh of one point): the results the test takes.  A mesh of several k-points is not
    tested (the cost on metals, whose levels at the Fermi level are partly filled, is not measured)."""

    points = getattr(result, "k_points", None)
    return points is None or len(np.atleast_2d(np.asarray(points, dtype=np.float64))) == 1


@dataclass(frozen=True)
class SymmetricState:
    """What made a converged state spin-symmetric with a partly filled level."""

    spread: float
    fractional: float
    moment: float


def symmetric_state(sets) -> SymmetricState | None:
    """The spin-symmetric state of two channels ``sets`` with a partly filled level, or ``None``."""

    if len(sets) != 2:
        return None
    up, down = (_levels(orbital_set)[0] for orbital_set in sets)
    if up.size != down.size or not up.size:
        return None
    spread = float(np.max(np.abs(np.sort(up) - np.sort(down))))
    fractional = float(max(np.max(np.minimum(up, 1.0 - up)), np.max(np.minimum(down, 1.0 - down))))
    if spread > SYMMETRIC_SPREAD or fractional <= FRACTIONAL_LEVEL:
        return None
    electrons = channel_electrons(sets)
    return SymmetricState(spread=spread, fractional=fractional, moment=electrons[0] - electrons[1])


def _copy_natural(blocks, occupations=None) -> dict:
    """``{key: (Phi, n)}`` of the copied occupied blocks of one channel (``occupations`` by key replaces ``n``)."""

    natural = {}
    for key, block in blocks.items():
        size = int(np.asarray(block.occupations).size)
        phi = type(block).combine([(block, np.eye(size))])
        values = np.asarray(block.occupations if occupations is None else occupations[key], dtype=np.float64)
        natural[key] = (phi, np.array(values, dtype=np.float64))
    return natural


def displaced_occupations(blocks) -> tuple[list, float] | None:
    """The occupations of the displacement of the test, per channel and key, and the electrons moved.

    ``blocks`` are the copied occupied blocks of the two channels (per channel ``{key: block}``, occupations per
    partner).  The levels of a key are paired in their order (the sector bases of the two channels of a
    spin-symmetric state); a key with blocks of other sizes in the two channels moves nothing.  ``None`` where
    nothing moves.
    """

    up, down = blocks
    moved_up, moved_down = {}, {}
    moved = 0.0
    for key in up:
        first = np.asarray(up[key].occupations, dtype=np.float64)
        moved_up[key] = first.copy()
    for key in down:
        moved_down[key] = np.asarray(down[key].occupations, dtype=np.float64).copy()
    for key in set(up) & set(down):
        first, second = moved_up[key], moved_down[key]
        if first.size != second.size:
            continue
        partly = (np.minimum(first, 1.0 - first) > FRACTIONAL_LEVEL) & (np.minimum(second, 1.0 - second) > FRACTIONAL_LEVEL)
        shift = np.where(partly, PROBE_SHARE * np.minimum(second, 1.0 - first), 0.0)
        partners = float(getattr(up[key], "partners", 1.0))
        moved += partners * float(shift.sum())
        moved_up[key] = first + shift
        moved_down[key] = second - shift
    if moved <= 0.0:
        return None
    return [moved_up, moved_down], moved


class SpinStabilityTest:
    """The test of a converged spin-symmetric state of a spin-polarized Fock loop (see the module).

    :func:`parsec_python.SCF.hybrid.run_hybrid_scf` asks :meth:`start` when its loop converged; where it returns
    the orbital sets of the displaced density matrix the loop goes on from their operator and reports the output of
    every step to :meth:`observe`.  Where that says ``"stable"`` (the displacement decayed), or where :meth:`decide`
    says ``"return"`` at the end of a growing displacement that found no lower state, the loop goes on from the
    operator of the copies of the symmetric state's orbitals (:meth:`return_sets`) and ends there.  ``text`` is the
    note of the summary, ``ratio`` the decay ratio of a stable verdict (``None`` otherwise), which a structure that
    starts from these orbitals reads (:meth:`skip`).
    """

    def __init__(self, scf, *, enabled: bool) -> None:
        temperature = float(getattr(scf, "fermi_temperature_kelvin", 0.0) or 0.0)
        self.temperature = temperature
        self.armed = bool(
            enabled
            and getattr(scf, "spin_polarized", False)
            and getattr(scf, "fixed_occupations", None) is None
            and temperature > 0.0
        )
        self.phase = "armed" if self.armed else "off"
        settings = getattr(scf, "exact_exchange", None)
        self.tolerance = float(getattr(settings, "tolerance", 0.0) or 0.0)
        self.text = ""
        self.note = ""
        self.kept_free = None
        self.kept_total = None
        self.kept_moment = None
        self.kept_blocks = None
        self.kept_iteration = None
        self.moments: list[float] = []
        self.start_iteration = None
        self.returned = False
        self.growing = False
        self.tail = 0.0
        self.ratio = None
        self.carried_ratio = None

    @property
    def probing(self) -> bool:
        return self.phase == "probing"

    @property
    def waiting(self) -> bool:
        """Whether a converged state is still to be looked at (tested, or noted as not tested again)."""

        return self.phase in ("armed", "skipped")

    @property
    def return_plain(self) -> bool:
        """Whether the operator of the copies stands for the plain ACE of the symmetric state's orbitals and may end
        the loop: the copies (those of the window of the outer DIIS) leave out the levels at or below
        :func:`~parsec_python.SCF.fock_diis.window_occupation_floor`, the thermal tail of a gap, and the operator may
        end the loop where none of them holds more than the limit at which the window of the outer DIIS leaves such a
        level out (:func:`~parsec_python.SCF.fock_diis.tail_limit` of ``Exx_Tolerance``); else one plain step more."""

        from .fock_diis import tail_limit

        return self.tail <= tail_limit(self.tolerance)

    def skip(self, ratio: float) -> None:
        """The orbitals that start this loop come from a structure found stable with decay ratio ``ratio`` (at most
        ``CARRY_RATIO``): its spin-symmetric state is not tested again."""

        if self.phase == "armed":
            self.phase = "skipped"
            self.carried_ratio = float(ratio)

    def start(self, engine, sets, total_energy: float, outer: int):
        """The orbital sets of the displaced density matrix of a converged spin-symmetric state, or ``None``.

        ``None`` also where the state is not one (no test), where it is not tested again (:meth:`skip`) or where the
        copies of its orbitals do not fit (``text`` says so).
        """

        if not self.waiting:
            return None
        state = symmetric_state(sets)
        if self.phase == "skipped":
            self.phase = "done"
            if state is not None:
                self.ratio = self.carried_ratio
                self.text = (
                    f"the spin-symmetric state of outer iteration {outer} was not tested again: the orbitals of this "
                    "Fock loop come from the structure before, whose spin-symmetric state was found stable or, not "
                    f"tested again, was taken over from such a structure (decay ratio {self.carried_ratio:.2f} at the "
                    "test)"
                )
            return None
        if state is None:
            return None
        self.phase = "done"
        try:
            blocks = engine.occupied_blocks(sets)
        except MemoryError:  # cupy.cuda.memory.OutOfMemoryError is one
            blocks = None
        if blocks is None:
            self.text = (
                f"the converged state of outer iteration {outer} is spin-symmetric with a partly filled level "
                f"(largest min(f, 1 - f) = {state.fractional:.3f}); its stability against a spin transfer was not "
                "tested: the copies of its orbitals do not fit the device memory"
            )
            return None
        displaced = displaced_occupations(blocks)
        if displaced is None:
            return None
        occupations, moved = displaced
        try:
            natural = [_copy_natural(channel, values) for channel, values in zip(blocks, occupations)]
            displaced_sets = engine.natural_sets(sets, natural)
        except MemoryError:
            displaced_sets = None
        if displaced_sets is None:
            self.text = (
                f"the converged state of outer iteration {outer} is spin-symmetric with a partly filled level "
                f"(largest min(f, 1 - f) = {state.fractional:.3f}); its stability against a spin transfer was not "
                "tested: the displaced orbitals do not fit"
            )
            return None
        from ..V_xc.exact_exchange import exchange_occupation_threshold
        from .fock_diis import window_occupation_floor

        cut, floor = exchange_occupation_threshold(), window_occupation_floor()
        self.tail = max(
            (float(np.max(values[(values > cut) & (values <= floor)], initial=0.0)) for values in
             (_levels(item)[0] for item in sets)),
            default=0.0,
        )
        self.phase = "probing"
        self.kept_blocks = blocks
        self.kept_free = free_energy(total_energy, sets, self.temperature)
        self.kept_total = float(total_energy)
        self.kept_moment = state.moment
        self.kept_iteration = int(outer)
        self.start_iteration = int(outer)
        self.moments = [state.moment + 2.0 * moved]
        self.note = (
            f"spin-symmetric state with a partly filled level (largest min(f, 1 - f) = {state.fractional:.3f}): "
            f"stability test, the next operator moves {moved:.3f} electrons of those levels from the down to the up "
            "channel; plain steps from here"
        )
        return displaced_sets

    def _moments_text(self) -> str:
        shown = self.moments[:4]
        return " -> ".join(f"{value:.3f}" for value in shown) + (" -> ..." if len(self.moments) > 4 else "")

    def observe(self, sets, outer: int) -> str | None:
        """The moment of the output of a step of the test; ``"stable"`` once the displacement decays clearly.

        The displacement is the moment less that of the symmetric state, the ratios those of successive
        displacements.  Stable, from the second step on: the last ratio at most ``STEP_RATIO`` and its linear
        continuation ``2 r_k - r_(k-1)`` at most ``PREDICTED_RATIO``.  A step that raises the displacement marks it as
        growing; growing or not, without that verdict the test goes on to the end of the loop (:meth:`decide`).  After
        ``"stable"`` the loop goes back to the symmetric state (``note``, :meth:`return_sets`).
        """

        electrons = channel_electrons(sets)
        self.moments.append(electrons[0] - electrons[1])
        if self.returned:
            return None
        amplitudes = [abs(value - self.kept_moment) for value in self.moments]
        if amplitudes[-1] > amplitudes[-2]:
            self.growing = True
        if self.growing or len(amplitudes) < 3 or not all(value > 0.0 for value in amplitudes[-3:-1]):
            return None
        ratio = amplitudes[-1] / amplitudes[-2]
        previous = amplitudes[-2] / amplitudes[-3]
        if ratio > STEP_RATIO or 2.0 * ratio - previous > PREDICTED_RATIO:
            return None
        self.returned = True
        self.ratio = float(ratio)
        self.text = (
            f"the spin-symmetric state of outer iteration {self.kept_iteration} is stable against a spin transfer "
            f"(moments of the steps {self._moments_text()}, decay ratios {previous:.2f} and {ratio:.2f} at outer "
            f"iteration {outer}): the loop went back to it from the copies of its orbitals"
        )
        self.note = "the operator of the copied orbitals of the spin-symmetric state (stable: the end of its test)"
        return "stable"

    def decide(self, sets, total_energy: float, converged: bool, outer: int):
        """``"keep"`` (the loop ends with this state) or ``"return"`` (back to the symmetric state).

        Asked at the end of the loop.  After a return: ``"keep"`` (``text`` says whether the loop converged there
        again).  Otherwise the displacement grew: a converged state lower in ``E - TS`` by more than ``MARGIN`` is
        kept; else the copies of the symmetric state's orbitals build the next operator (:meth:`return_sets`) and the
        loop is given ``RETURN_ITERATIONS``.
        """

        if self.returned:
            self.phase = "done"
            self.kept_blocks = None
            if converged:
                self.text += f"; it ends there after outer iteration {outer}"
            else:
                self.text += f"; the loop did not converge on it again within {RETURN_ITERATIONS} outer iterations"
            return "keep"
        free = free_energy(total_energy, sets, self.temperature)
        below = self.kept_free - free
        electrons = channel_electrons(sets)
        moment = electrons[0] - electrons[1]
        steps = f"moments of the steps {self._moments_text()}"
        if converged and below > MARGIN:
            self.phase = "done"
            self.kept_blocks = None
            self.text = (
                f"the spin-symmetric state of outer iteration {self.kept_iteration} was unstable ({steps}): the loop "
                f"went on to a state {below:.6f} Ry lower in E - TS (total {total_energy - self.kept_total:+.6f} Ry), "
                f"moment {moment:.3f}"
            )
            return "keep"
        self.returned = True
        self.text = (
            f"the stability test of the spin-symmetric state of outer iteration {self.kept_iteration} "
            + (
                f"found no lower state ({steps}; E - TS {-below:+.1E} Ry from it)"
                if converged
                else f"did not converge ({steps})"
            )
            + ": the loop went back to the symmetric state from the copies of its orbitals"
        )
        self.note = "the operator of the copied orbitals of the spin-symmetric state (the end of its stability test)"
        return "return"

    def abandon(self, reason: str) -> None:
        """End the test where a build does not fit: ``reason`` joins ``text``."""

        self.phase = "done"
        self.kept_blocks = None
        self.text = (self.text + "; " if self.text else "the stability test of a spin-symmetric state: ") + reason

    def return_sets(self, engine, sets):
        """The orbital sets of the symmetric state from its copies (the empty states of ``sets``), or ``None``."""

        blocks, self.kept_blocks = self.kept_blocks, None
        try:
            natural = [_copy_natural(channel) for channel in blocks]
            return engine.natural_sets(sets, natural)
        except MemoryError:
            return None


__all__ = [
    "CARRY_RATIO",
    "PREDICTED_RATIO",
    "STEP_RATIO",
    "FOCK_DEXX_FLOOR_ENV",
    "FOCK_SPIN_STABILITY_ENV",
    "SpinStabilityTest",
    "SymmetricState",
    "dexx_floor_setting",
    "displaced_occupations",
    "entropy",
    "free_energy",
    "operator_bias",
    "spin_stability_setting",
    "single_point",
    "symmetric_state",
]
