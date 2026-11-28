"""The test of a converged loop of the spin-flip route and the starts that may lie below it
(``PARSEC_SPIN_FLIP_STABILITY=auto`` with ``PARSEC_SPIN_FLIP_CONTINUATION=auto``).

A loop of ``PARSEC_SPIN_FLIP_SYMMETRY=auto`` (:mod:`.spin_flip`) holds the down channel as the image of the up channel
under an operation ``g`` that exchanges the partner labels, so it converges to a self-consistent state of that magnetic
symmetry.  Two solved channels from the same start (``off``) need not end there:

1. the state may be unstable against perturbations that break the operation (a direction of negative curvature of
   the free energy): two solved channels leave it through the noise of their eigensolvers;
2. the state may be a local minimum while the loop without the flip, on its own path from the start, reaches another
   state below it: Ni2 at 4.06 bohr from ``+0.6 / -0.6`` (PBE, 80 K, A100): the flip loop converges to one filling of
   the d levels at -138.76440 Ry (largest eigenvalue of the response 0.69; kicks that keep or break the operation
   return to it), the loop without the flip, whose first eigensolves of the two channels differ by 0.08 Ry, to a
   state that breaks the operation at -138.78083 Ry, and a ferromagnetic state lies lower still (-138.81535 Ry).

No test of the converged state alone sees the second, and the starts below do not either where the one-step free energy
of the ferromagnetic start lies above the state of the route (Ni2 20.7 mRy above, its converged state 51 mRy below):
``PARSEC_SPIN_FLIP_SYMMETRY=late`` (the default) avoids it instead, by following the loop without the flip until its
channels are images of each other.  Here a converged loop of the route of ``auto`` is followed by
(:func:`flip_state_safety`; every eigensolve is one of the down channel, which the route does not solve otherwise, so
that the up channel keeps its subspace and the orbitals of the result):

**The test.**  ``J`` maps a perturbation ``w = dd - g du`` of the input potentials ``(u, d = g u)`` that breaks the
operation onto that of the output of one SCF step (the SCF map commutes with the operation that exchanges the
channels).  A probe ``dd = s v`` with ``du = -g^-1 dd`` solves the down channel at ``g u + s v``; the up channel then
follows from the operation to first order (its levels ``2 e0 - e`` for the common Fermi level, its density change
``-g^-1`` that of the down channel), and the Hartree and exchange-correlation potentials of both densities give
``J v``.  Arnoldi steps on ``J`` (inner product of the grid, or of the orbits with their multiplicities) give its
largest eigenvalue; they stop as those of :mod:`.noncollinear_stability` do (a Ritz value from its ``CLEAR``, or after
its ``MINIMUM_STEPS`` once the largest Ritz value has settled), and the state is unstable from its ``BOUND``.  The free
energy is a minimum along these perturbations where every eigenvalue of ``J`` is below 1 (Herbst and Levitt 2022).

**The starts.**  The free energy ``F[P(x)]`` of the density matrix of one SCF step at an input ``x`` (both channels,
one Fermi level, the Mermin free energy) is an upper bound of the lowest free energy, so a start below the state of the
route shows a lower state.  Tried: along the top Ritz vector of an unstable state (``F(s) = F(-s)``: the state at
``-s`` is the image of that at ``s``), from :data:`LINE_FIRST` doubling while it goes down; the partner pattern (the
local fields of the atoms whose labels start at a negative polarization turned over, ``V_up`` and ``V_down`` exchanged
on the grid points nearest to them: the ferromagnetic start of an antiferromagnetic pair); and the patterns of the top
Ritz vectors (the local fields of the atoms whose local moment the vector lowers turned over).  A pattern and its
complement are one start (the free energy of collinear spins does not change when every spin turns over).

**The continuation** (:func:`parsec_python.SCF.spin_polarized.run_spin_scf`): where the lowest start lies
:data:`GAIN` below the state of the route, the loop without the flip continues from it (both channels solved, from the
subspaces of the up channel and of that start), stopped as :class:`.noncollinear_stability.ContinuationWatch` stops it;
the lower converged state is the result.  Otherwise the state of the route is kept.  Units Ry, bohr.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .noncollinear_stability import (
    BOUND,
    CLEAR,
    CONVERGENCE,
    GAIN,
    LINE_FIRST,
    LINE_LARGEST,
    MINIMUM_STEPS,
    NEAR,
    PATTERN_MOMENT,
    PATTERN_VECTORS,
    STEPS,
    WINDOW,
    nearest_atoms,
)

#: Largest change (Ry) of the down potential of one probe of the response.
KICK = 5.0e-3
#: Eigensolver passes of the down channel per probe at most (fewer once no level up to the window above the Fermi
#: level moves by more than :data:`LEVEL_TOLERANCE`), and of its first solve at the state of the route.
PROBE_PASSES = 2
FIRST_PASSES = 16
LEVEL_TOLERANCE = 1.0e-8
#: Eigensolver passes of each channel at a point of the line along the top Ritz vector and at a pattern.
LINE_PASSES = 2
PATTERN_PASSES = 4
#: How far (Ry) the free energy of the converged continuation must lie below the state of the route to replace it.
MARGIN = 1.0e-6
#: Seed of the random part of the first probe direction (the same on every process).
SEED = 20261127
#: Boltzmann's constant in Ry/K.
KELVIN_RY = 6.33327186e-06


def entropy_term(occupations, weights, temperature: float) -> float:
    """``-T S`` (Ry) of the occupations of both channels, each state counted ``weight`` times (one electron a state)."""

    if not temperature > 0.0:
        return 0.0
    total = 0.0
    for spin, filled in enumerate(occupations):
        values = np.clip(np.asarray(filled, dtype=np.float64).reshape(-1), 0.0, 1.0)
        inside = (values > 0.0) & (values < 1.0)
        terms = np.zeros_like(values)
        part = values[inside]
        terms[inside] = part * np.log(part) + (1.0 - part) * np.log(1.0 - part)
        if weights is not None and weights[spin] is not None:
            terms = terms * np.asarray(weights[spin], dtype=np.float64).reshape(-1)
        total += float(np.sum(terms))
    return KELVIN_RY * float(temperature) * total


@dataclass
class FlipCandidate:
    """A start below which the state of the route may lie: ``kind`` ``line``, ``partner`` or ``pattern``, the input
    potentials ``(up, down)`` (the loop's representation: orbit values in sectors), the free energy of one SCF step
    there; ``shift`` the largest change of the down potential along the top Ritz vector (``line``), ``flipped`` the
    atoms whose local field is turned over (patterns)."""

    kind: str
    up: np.ndarray
    down: np.ndarray
    free_energy: float = float("inf")
    shift: float = 0.0
    flipped: int = 0
    mask: tuple = ()

    def describe(self) -> str:
        if self.kind == "line":
            return f"along the top Ritz vector at a largest change of {self.shift:.4g} Ry"
        what = "partner pattern" if self.kind == "partner" else "pattern of a Ritz vector"
        return f"{what} (the local field of {self.flipped} atom(s) turned over)"


@dataclass
class FlipSafetyReport:
    """What :func:`flip_state_safety` found: the test (``largest`` the largest real part of the Ritz values after the
    last probe, ``history`` after every probe, ``ritz`` the last ones, ``probes``, ``converged``), the starts tried
    (``candidates``, each with its free energy) against ``reference``, the free energy of the state of the route; the
    eigensolver ``passes`` of the down channel in all (``first_passes`` of its first solve), ``seconds`` and
    ``candidate_seconds`` the time of the test and of the starts."""

    largest: float
    history: tuple
    ritz: tuple
    probes: int
    converged: bool
    reference: float
    candidates: tuple = ()
    passes: int = 0
    first_passes: int = 0
    first_seconds: float = 0.0
    first_deviation: float = 0.0
    seconds: float = 0.0
    candidate_seconds: float = 0.0
    bound: float = BOUND
    tolerance: float = GAIN
    #: the eigensolver state of the down channel after the last solve (at the input of the best start, where it pays)
    down_state: object = None
    states_at: str = ""

    @property
    def unstable(self) -> bool:
        return not self.largest < self.bound

    @property
    def best(self) -> FlipCandidate | None:
        found = [item for item in self.candidates if np.isfinite(item.free_energy)]
        return min(found, key=lambda item: item.free_energy) if found else None

    @property
    def gain(self) -> float:
        best = self.best
        return float("-inf") if best is None else float(self.reference - best.free_energy)

    @property
    def pays(self) -> bool:
        return self.best is not None and not self.gain < self.tolerance

    def test_text(self) -> str:
        return (
            f"largest eigenvalue {self.largest:.4f} of the SCF response to perturbations that break the operation "
            f"(unstable from {self.bound:g}; {self.probes} probes, Ritz values "
            + " ".join(f"{value:.3f}" for value in self.ritz[:4])
            + ("" if self.converged else "; not converged within the probes")
            + f"; {self.seconds:.2f} s, of which {self.first_seconds:.2f} s for the {self.first_passes} passes of the "
            f"first solve of the down channel, its levels {self.first_deviation:.1e} Ry from the up channel's)"
        )

    def candidates_text(self) -> str:
        if not self.candidates:
            return "no start tried"
        parts = [
            f"{item.describe()}: {self.reference - item.free_energy:+.3e} Ry below"
            for item in sorted(self.candidates, key=lambda item: item.free_energy)
        ]
        return (
            f"free energy of one SCF step at {len(self.candidates)} start(s) against the state of the route: "
            + "; ".join(parts[:6])
            + f" (a continuation from {self.tolerance:g} Ry; {self.candidate_seconds:.2f} s)"
        )


@dataclass
class FlipLoopState:
    """What :func:`flip_state_safety` reads of a converged loop of the route (``SCF.spin_polarized``)."""

    system: object
    flip: object
    solve: object
    number_of_states: int
    eigensolver_settings: object
    neighbors: object
    settings: object
    up_potential: object
    up_eigenvalues: np.ndarray
    fermi_level: float
    ionic_potential: object
    hartree_potential: object
    reduced: bool
    adapter: object
    orbital_density_builder: object
    volume_element: float
    total_energy: object
    reference: float
    temperature: float
    alpha_z_energy: float = 0.0
    seconds: dict = field(default_factory=dict)


class _DownChannel:
    """The eigensolves of the down channel at given potentials, and the fields of one SCF step."""

    def __init__(self, state: FlipLoopState) -> None:
        from parsec_python.SCF.single_point import eigval_settings_for

        self.loop = state
        self.system = state.system
        self.flip = state.flip
        if state.reduced:
            self.field_of = state.adapter.field
            self.weights = np.asarray(state.adapter.weights, dtype=np.float64)
        else:
            self.field_of = lambda values: values
            self.weights = None
        self.settings = eigval_settings_for(
            state.eigensolver_settings, state.eigensolver_settings.filter_degree, neighbors=state.neighbors
        )
        restart = getattr(state.solve, "start_new_scf_loop", None)
        self.state = restart() if callable(restart) else None
        self.vectors = None
        self.passes = 0

    @staticmethod
    def values_of(field_values) -> np.ndarray:
        return np.asarray(getattr(field_values, "values", field_values), dtype=np.float64)

    def dot(self, left, right) -> float:
        if self.weights is not None:
            return float(np.dot(left * self.weights, right))
        return float(np.dot(left, right))

    def _release(self) -> None:
        release = getattr(self.vectors, "release_intermediate_storage", None)
        if self.loop.reduced and callable(release):
            release()

    def solve(self, potential: np.ndarray, passes: int, previous=None, window: float | None = None):
        """Passes of the down channel at ``potential`` (values) until no level up to ``window`` moves by more than
        :data:`LEVEL_TOLERANCE` (at most ``passes``); the last solution."""

        hamiltonian = self.system.hamiltonian(self.field_of(np.asarray(potential, dtype=np.float64)))
        operator = getattr(hamiltonian, "as_eigensolver_operator", hamiltonian.as_linear_operator)()
        window = self.loop.fermi_level + 3.0e-2 if window is None else window
        solution = None
        for _ in range(max(1, int(passes))):
            self._release()
            solution = self.loop.solve(operator, self.loop.number_of_states, settings=self.settings, state=self.state)
            self.passes += 1
            self.state, self.vectors = solution.state, solution.vectors
            values = np.asarray(solution.eigenvalues, dtype=np.float64)
            if previous is not None and np.shape(previous) == values.shape:
                shown = (values <= window) | (np.asarray(previous) <= window)
                if float(np.max(np.abs(values - previous)[shown], initial=0.0)) < LEVEL_TOLERANCE:
                    break
            previous = values
        return solution

    def density(self, solution, occupations) -> np.ndarray:
        from parsec_python.Occupations import spin_density_from_orbitals

        builder = self.loop.orbital_density_builder
        if builder is None:
            return self.values_of(spin_density_from_orbitals(solution.vectors, occupations, self.loop.volume_element))
        return self.values_of(builder(solution.vectors, 0.5 * np.asarray(occupations), self.loop.volume_element))

    def outputs(self, up_density, down_density):
        """``(V_out up, V_out down, hartree, xc)`` of one SCF step from the densities (values)."""

        hartree = self.system.solve_hartree(
            self.field_of(up_density + down_density), initial_potential=self.loop.hartree_potential
        )
        xc = self.system.evaluate_spin_xc(self.field_of(up_density), self.field_of(down_density))
        ionic = self.values_of(self.loop.ionic_potential)
        common = ionic + self.values_of(hartree.potential)
        return (common + self.values_of(xc.potentials[0]), common + self.values_of(xc.potentials[1]), hartree, xc)

    def occupations(self, levels, solutions):
        """One Fermi level for the levels of both channels (``solutions`` give their state weights and partners):
        ``(per-state occupations times the weights, the occupation result, the weights)``."""

        from parsec_python.SCF.spin_polarized import _channel_state_weights, _spin_occupations, _weighted_occupations

        weights = _channel_state_weights([getattr(item, "state_weights", None) for item in solutions])
        partners = [getattr(item, "state_partners", None) for item in solutions]
        result = _spin_occupations(levels, self.system, self.loop.settings, weights, partners)
        return _weighted_occupations(result, weights), result, weights

    def free_energy(self, up_input: np.ndarray, down_input: np.ndarray, passes: int) -> float:
        """The Mermin free energy of the density matrix of one SCF step at the input potentials ``(up_input,
        down_input)`` (values).  The solver holds one set of orbitals: the down channel at ``down_input`` first (its
        levels), then the up channel as the image of the down channel at ``g up_input`` (its levels, one Fermi level
        for both, its density), then the down channel again for its density, with the occupations of its first solve
        carried over sector by sector (a density matrix of the electron count).  It ends with the subspace of the
        down channel at ``down_input``."""

        flip = self.flip
        first = self.solve(down_input, passes)
        first_levels = np.asarray(first.eigenvalues, dtype=np.float64).copy()
        first_labels = _labels(first)
        first_weights = getattr(first, "state_weights", None)
        image = self.solve(self.values_of(flip.down_field(up_input)), passes)
        image_levels = np.asarray(image.eigenvalues, dtype=np.float64).copy()
        filled, result, weights = self.occupations([image_levels, first_levels], [image, first])
        up_density = self.values_of(flip.up_field(self.density(image, filled[0])))
        image_weights = None if weights is None else weights[0]
        third = self.solve(down_input, passes)
        third_levels = np.asarray(third.eigenvalues, dtype=np.float64)
        carried = _carried(first_levels, first_labels, np.asarray(result.occupations[1]), third_levels, _labels(third))
        third_weights = getattr(third, "state_weights", None)
        if carried is None or (first_weights is None) != (third_weights is None):
            # (the sectors hold other numbers of states than in the first solve: its electrons filled into the levels
            # of this one at the temperature of the run, still a density matrix of the electron count)
            electrons = float(
                np.dot(
                    np.asarray(result.occupations[1], dtype=np.float64),
                    np.ones(first_levels.size) if first_weights is None else np.asarray(first_weights, dtype=np.float64),
                )
            )
            carried = _fill(third_levels, third_weights, electrons, self.loop.temperature)
        down_filled = carried if third_weights is None else carried * np.asarray(third_weights, dtype=np.float64)
        down_density = self.density(third, down_filled)
        _, _, hartree, xc = self.outputs(up_density, down_density)
        loop = self.loop
        extra = {"alpha_z_energy": loop.alpha_z_energy} if loop.alpha_z_energy else {}
        energies = loop.total_energy(
            [image_levels, third_levels],
            [filled[0], down_filled],
            [self.field_of(up_density), self.field_of(down_density)],
            [self.field_of(np.asarray(up_input, dtype=np.float64)), self.field_of(np.asarray(down_input, dtype=np.float64))],
            loop.ionic_potential,
            hartree.potential,
            list(xc.potentials),
            xc.total_energy,
            self.system.ion_ion_energy,
            loop.volume_element,
            **extra,
        )
        entropy_weights = None if weights is None else (image_weights, third_weights)
        return float(energies.total) + entropy_term(
            [result.occupations[0], carried], entropy_weights, loop.temperature
        )


def _mirrored_levels(base, base_labels, levels, labels):
    from parsec_python.SCF.spin_polarized import _mirrored_levels as mirrored

    return mirrored(base, base_labels, levels, labels)


def _labels(solution) -> np.ndarray:
    labels = getattr(solution, "representations", None)
    levels = np.asarray(solution.eigenvalues)
    return np.zeros(levels.size, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)


def _fill(levels, weights, electrons: float, temperature: float) -> np.ndarray:
    """Occupations (per state, each state counted ``weight`` times) of one channel that hold ``electrons`` in
    ``levels``: Fermi-Dirac at ``temperature`` with its own Fermi level, or filled from the bottom at zero."""

    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    weights = np.ones(levels.size) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    electrons = min(max(float(electrons), 0.0), float(np.sum(weights)))
    if not temperature > 0.0:
        filled = np.zeros(levels.size)
        left = electrons
        for index in np.argsort(levels, kind="stable"):
            take = min(1.0, left / weights[index]) if weights[index] > 0.0 else 0.0
            filled[index] = max(take, 0.0)
            left -= filled[index] * weights[index]
            if left <= 0.0:
                break
        return filled
    width = KELVIN_RY * float(temperature)

    def occupied(level):
        return 0.5 * (1.0 - np.tanh(0.5 * (levels - level) / width))

    low, high = float(np.min(levels)) - 1.0, float(np.max(levels)) + 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if float(np.dot(weights, occupied(middle))) < electrons:
            low = middle
        else:
            high = middle
    return occupied(0.5 * (low + high))


def _carried(levels, labels, occupations, new_levels, new_labels) -> np.ndarray | None:
    """The occupations of the states of a first solve (``levels`` in sectors ``labels``) carried to those of a later
    solve of the same channel, the k-th level of a sector to the k-th of that sector; ``None`` where a sector holds
    different numbers of states in the two."""

    levels = np.asarray(levels, dtype=np.float64)
    new_levels = np.asarray(new_levels, dtype=np.float64)
    labels = np.asarray(labels)
    new_labels = np.asarray(new_labels)
    occupations = np.asarray(occupations, dtype=np.float64)
    if levels.shape != new_levels.shape or occupations.shape != levels.shape:
        return None
    carried = np.empty_like(occupations)
    for label in np.unique(np.concatenate([labels, new_labels])):
        before = np.flatnonzero(labels == label)
        after = np.flatnonzero(new_labels == label)
        if before.size != after.size:
            return None
        carried[after[np.argsort(new_levels[after], kind="stable")]] = occupations[
            before[np.argsort(levels[before], kind="stable")]
        ]
    return carried


def _arnoldi(apply, start, dot, steps: int):
    """Arnoldi steps on ``apply`` from ``start``, stopped as :func:`.noncollinear_stability.sector_state_stability`
    stops them; ``(history, ritz, vectors, converged)`` with ``vectors`` the Ritz vectors of the largest Ritz values
    (largest first, largest component 1, up to :data:`PATTERN_VECTORS`)."""

    basis = [start / np.sqrt(max(dot(start, start), 1.0e-300))]
    hessenberg = np.zeros((steps + 1, steps), dtype=np.float64)
    history: list[float] = []
    ritz = np.zeros(0)
    coefficients: list[np.ndarray] = []
    converged = False
    for step in range(steps):
        response = apply(basis[step])
        for _ in range(2):
            # (Gram-Schmidt twice)
            for row in range(step + 1):
                projection = dot(basis[row], response)
                hessenberg[row, step] += projection
                response = response - projection * basis[row]
        length = float(np.sqrt(max(dot(response, response), 0.0)))
        hessenberg[step + 1, step] = length
        values, vectors = np.linalg.eig(hessenberg[: step + 1, : step + 1])
        order = np.argsort(values.real)[::-1]
        ritz = values.real[order]
        coefficients = [vectors[:, index].real for index in order[:PATTERN_VECTORS]]
        history.append(float(ritz[0]))
        if length <= 1.0e-12 * max(1.0, float(np.max(np.abs(hessenberg[: step + 1, : step + 1])))):
            converged = True
            break
        if ritz[0] >= CLEAR:
            converged = True
            break
        settled = CONVERGENCE if ritz[0] < NEAR * BOUND else 0.1 * CONVERGENCE
        if len(history) >= max(MINIMUM_STEPS, WINDOW + 1) and max(history[-WINDOW - 1:]) - min(
            history[-WINDOW - 1:]
        ) < settled:
            converged = True
            break
        basis.append(response / length)
    found = []
    for weights in coefficients:
        vector = np.zeros_like(basis[0])
        for weight, column in zip(weights, basis):
            vector += float(weight) * column
        largest = float(np.max(np.abs(vector), initial=0.0))
        if largest > 0.0:
            found.append(vector / largest)
    return tuple(history), tuple(float(value) for value in ritz), found, converged


def _atom_partition(loop: FlipLoopState) -> np.ndarray:
    system = loop.system
    cell = getattr(getattr(system, "input", None), "periodic_cell", None)
    return nearest_atoms(
        system.grid.coordinates,
        [atom.position for atom in system.atoms],
        None if cell is None else cell.lattice_vectors,
    )


def _turned(loop: FlipLoopState, channel: _DownChannel, mask_atoms: np.ndarray, partition: np.ndarray, up, down):
    """The input with ``V_up`` and ``V_down`` exchanged on the grid points nearest to the atoms of ``mask_atoms``
    (orbit-averaged in the sectors)."""

    weight = mask_atoms.astype(np.float64)[partition]
    if loop.reduced:
        weight = channel.values_of(loop.adapter.from_full(weight))
    return up + weight * (down - up), down + weight * (up - down)


def _canonical(mask: np.ndarray) -> tuple:
    """A pattern and its complement are one start (every spin turned over leaves the free energy)."""

    flags = tuple(bool(item) for item in mask)
    other = tuple(not item for item in flags)
    return min(flags, other)


def flip_state_safety(loop: FlipLoopState, *, kick: float | None = None, steps: int | None = None) -> FlipSafetyReport:
    """The test of the converged state of a loop of the route and the starts below it (see the module).

    The up channel and its eigensolver are not touched; the Hamiltonian holds the up potential again at the end.
    """

    started = time.perf_counter()
    kick = KICK if kick is None else float(kick)
    steps = STEPS if steps is None else int(steps)
    flip = loop.flip
    channel = _DownChannel(loop)
    values_of = channel.values_of
    up = values_of(loop.up_potential)
    down = values_of(flip.down_field(loop.up_potential))
    window = float(loop.fermi_level) + 3.0e-2

    # The down channel at the state of the route: the levels of the up channel.
    first_started = time.perf_counter()
    base = channel.solve(down, FIRST_PASSES, window=window)
    first_passes, first_seconds = channel.passes, time.perf_counter() - first_started
    base_levels = np.asarray(base.eigenvalues, dtype=np.float64).copy()
    base_labels = _labels(base)
    up_levels = np.asarray(loop.up_eigenvalues, dtype=np.float64)
    count = min(base_levels.size, up_levels.size)
    shown = np.sort(base_levels)[:count] <= window
    first_deviation = float(
        np.max(np.abs(np.sort(base_levels)[:count] - np.sort(up_levels)[:count])[shown], initial=0.0)
    )

    def mirrored_occupations(solution, mirrored):
        levels = np.asarray(solution.eigenvalues, dtype=np.float64)
        order = np.arange(levels.size) if mirrored is None else np.argsort(mirrored, kind="stable")
        up_side = levels if mirrored is None else np.asarray(mirrored, dtype=np.float64)[order]
        weights = getattr(solution, "state_weights", None)
        partners = getattr(solution, "state_partners", None)
        twin = SimpleNamespace(
            state_weights=None if weights is None else np.asarray(weights, dtype=np.float64)[order],
            state_partners=None if partners is None else tuple(np.asarray(partners)[order].tolist()),
        )
        filled, _, _ = channel.occupations([up_side, levels], [twin, solution])
        return filled[1]

    down_values = channel.density(base, mirrored_occupations(base, None))
    up_values = values_of(flip.up_field(down_values))
    base_up_output, base_down_output, _, _ = channel.outputs(up_values, down_values)

    def apply(direction):
        size = kick / max(float(np.max(np.abs(direction), initial=0.0)), 1.0e-30)
        solution = channel.solve(down + size * direction, PROBE_PASSES, previous=base_levels, window=window)
        mirrored = _mirrored_levels(base_levels, base_labels, solution.eigenvalues, _labels(solution))
        changed = channel.density(solution, mirrored_occupations(solution, mirrored)) - down_values
        up_new = up_values - values_of(flip.up_field(changed))
        up_output, down_output, _, _ = channel.outputs(up_new, down_values + changed)
        return ((down_output - base_down_output) - values_of(flip.down_field(up_output - base_up_output))) / (
            2.0 * size
        )

    rng = np.random.default_rng(SEED)
    moment = down_values - up_values
    total = down_values + up_values
    scale = max(float(np.max(np.abs(moment), initial=0.0)), 1.0e-30)
    start = (
        np.abs(moment) / scale
        + 0.5 * moment / scale
        + 0.25 * rng.standard_normal(moment.shape) * total / max(float(np.max(np.abs(total), initial=0.0)), 1.0e-30)
    )
    history, ritz, vectors, converged = _arnoldi(apply, start, channel.dot, steps)
    tested = time.perf_counter()
    report = FlipSafetyReport(
        largest=history[-1] if history else float("nan"),
        history=history,
        ritz=ritz,
        probes=len(history),
        converged=converged,
        reference=float(loop.reference),
        first_passes=first_passes,
        first_seconds=first_seconds,
        first_deviation=first_deviation,
        seconds=tested - started,
        bound=BOUND,
        tolerance=GAIN,
    )

    # The starts: the line along the top Ritz vector of an unstable state, the partner pattern, the patterns of the
    # top Ritz vectors; each by the free energy of one SCF step.
    candidates: list[FlipCandidate] = []
    if report.unstable and vectors:
        direction = vectors[0]
        shift = LINE_FIRST
        previous = float(loop.reference)
        while shift <= LINE_LARGEST * (1.0 + 1.0e-12):
            point = FlipCandidate(
                "line", up - shift * values_of(flip.up_field(direction)), down + shift * direction, shift=shift
            )
            point.free_energy = channel.free_energy(point.up, point.down, LINE_PASSES)
            candidates.append(point)
            if not point.free_energy < previous:
                break
            previous = point.free_energy
            shift *= 2.0
    system = loop.system
    species = getattr(getattr(system, "input", None), "pseudopotentials", {}) or {}
    polarization = np.asarray(
        [float(getattr(species.get(atom.symbol), "initial_spin_polarization", 0.0) or 0.0) for atom in system.atoms]
    )
    partition = None
    seen = set()
    masks = []
    if np.any(polarization < 0.0) and np.any(polarization > 0.0):
        masks.append(("partner", polarization < 0.0))
    if vectors:
        partition = _atom_partition(loop)
        atoms = len(system.atoms)
        field_full = 0.5 * (up - down)
        if loop.reduced:
            field_full = np.asarray(loop.adapter.to_full(loop.adapter.field(field_full)), dtype=np.float64)
        local = np.bincount(partition, weights=field_full, minlength=atoms)
        for vector in vectors:
            change = -0.5 * (values_of(flip.up_field(vector)) + vector)
            if loop.reduced:
                change = np.asarray(loop.adapter.to_full(loop.adapter.field(change)), dtype=np.float64)
            along = np.bincount(partition, weights=change, minlength=atoms)
            qualified = (np.abs(local) >= PATTERN_MOMENT * float(np.max(np.abs(local), initial=0.0))) & (
                np.abs(along) >= PATTERN_MOMENT * float(np.max(np.abs(along), initial=0.0))
            )
            for sign in (1.0, -1.0):
                lowered = qualified & (sign * along * local < 0.0)
                if np.any(lowered) and not np.all(lowered):
                    masks.append(("pattern", lowered))
    for kind, mask in masks:
        key = _canonical(mask)
        if key in seen or not any(key) or all(key):
            continue
        seen.add(key)
        if partition is None:
            partition = _atom_partition(loop)
        turned_up, turned_down = _turned(loop, channel, np.asarray(mask), partition, up, down)
        point = FlipCandidate(kind, turned_up, turned_down, flipped=int(np.count_nonzero(mask)), mask=key)
        point.free_energy = channel.free_energy(point.up, point.down, PATTERN_PASSES)
        candidates.append(point)
    report.candidates = tuple(candidates)
    best = report.best
    if report.pays and best is not None and candidates[-1] is not best:
        # (the subspace of the down channel at the input of the start the loop without the flip continues from)
        channel.solve(best.down, PATTERN_PASSES)
    report.down_state = channel.state
    report.passes = channel.passes
    report.candidate_seconds = time.perf_counter() - tested
    channel._release()
    # The Hamiltonian holds the potential of the up channel again, as the loop left it.
    system.hamiltonian(loop.up_potential)
    return report


__all__ = [
    "FIRST_PASSES",
    "FlipCandidate",
    "FlipLoopState",
    "FlipSafetyReport",
    "KICK",
    "LEVEL_TOLERANCE",
    "LINE_PASSES",
    "MARGIN",
    "PATTERN_PASSES",
    "PROBE_PASSES",
    "SEED",
    "entropy_term",
    "flip_state_safety",
]
