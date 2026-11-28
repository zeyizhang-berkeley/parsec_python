"""The stability test of the state of a collinear spin run in the symmetry sectors (``PARSEC_SPIN_SECTOR_STABILITY``).

The sectors of a spin-polarized cluster or Gamma cell (``Spin_Symmetry_Sectors``, :mod:`.spin_sectors`) hold the
channel densities and potentials in the symmetric form of the point group ``G`` of the atoms on the grid.  A loop in
them converges to a self-consistent state of that symmetry, a stationary point of the free energy that need not be a
minimum: the symmetric ensemble of a partly filled degenerate level (a half-filled e_g pair of bcc Mn2 with
antiparallel start moments, the doublet of charged fcc Cr4) lies above a state that splits it, and an
antiferromagnetic dimer can keep a symmetric moment pattern above one that breaks it.  The guard of the sectors
leaves them for the full grid where a level is partly filled (tests 1 and 3 of :mod:`.spin_sectors`), and the full
grid then starts again from the atomic densities; where the partners lie in one sector the guard keeps the state with
a note (``PARSEC_SPIN_GUARD_DOUBLETS``).

Under ``auto`` the loop in the sectors goes on where tests 1 and 3 would stop it, and where one of them would have
stopped it, where the doublet note would be printed, or where the species start with polarizations of opposite sign
(antiferromagnetic labels), its converged state is tested on the full grid, as the stability test of the
non-collinear k-point sectors does (:mod:`.noncollinear_stability`, ``PARSEC_NONCOLLINEAR_SECTOR_STABILITY``):

* ``x = (V_up, V_down)`` is the input of the last step of the loop, averaged over ``G`` (:class:`OrbitSymmetry`: the
  average over the orbits of the grid points under the operations of test 2 of the guard, ``atom_group_images``),
  and ``F`` one SCF step of both channels on the full grid (:class:`FullGridSpinStep`: the levels and orbitals of each
  channel from :data:`PASSES` SUBSPACE steps of its eigensolver started from the orbitals of the sectors on the full
  grid, one Fermi level, the densities, the Hartree and exchange-correlation potentials).  ``F`` commutes with ``G``,
  so its Jacobian ``J`` maps the perturbations that break ``G`` onto themselves; the state is a minimum along them
  where every eigenvalue of ``J`` on them is below 1 (Herbst and Levitt 2022: the Hessian of the free energy at a
  self-consistent state has the inertia of ``1 - K chi_0``), and the free energy goes down along a direction that
  breaks ``G`` where one is above 1.  :func:`response_stability` takes Arnoldi steps on ``J`` with the adaptive stop
  of the non-collinear test (``noncollinear_stability.MINIMUM_STEPS``, ``WINDOW``, ``CONVERGENCE``, ``CLEAR``,
  ``STEPS``) and its bound :data:`BOUND`; the first direction is the part that breaks ``G`` of a seeded random field
  times the density of each channel (:func:`first_direction`).
* An unstable state: the free energy of one SCF step (the Mermin free energy of its density matrix, an upper bound of
  the lowest free energy) along the top Ritz vector (``noncollinear_stability.line_search``) and at the flip patterns
  of the top Ritz vectors (:func:`flip_pattern`: the moments of one of the two groups of atoms that a soft mode moves
  against each other turned over); where no point lies ``GAIN`` below the state of the sectors that state is kept,
  else the loop on the full grid continues from the lowest point with the orbitals of the step there, stopped by
  ``noncollinear_stability.ContinuationWatch`` where its free energy has not been ``GAIN`` below the sectors' for
  ``CONTINUATION_WINDOW`` iterations, and a converged state below that of the sectors is the result (an unconverged
  one never replaces anything).
* A stable state is kept.
* Where a guard test was put aside, the result of ``off`` is that of the guard's route (the full grid from the atomic
  densities, or the abelian sectors): that route runs after the test as under ``off``, and its result is kept unless
  the continuation converged to a state with a lower free energy (by ``CONTINUATION_MARGIN``), or that route did not
  converge where the continuation did.  So the result is never above that of ``off`` and never unconverged where
  that of ``off`` converged; such a run costs ``off`` plus the loop in the sectors and the test.
* A loop in the sectors that does not converge, a test that cannot be made, and a test or loop on the full grid that
  does not fit beside the sectors in device memory go the guard's route of ``off``.
* The NOTE of a continued state gives the moments of the atoms by label before and after, and says where they no
  longer follow the signs of the start polarizations of the labels (:func:`labelled_moments`).

``off`` (the default): tests 1 and 3 send the run to the full grid and the note keeps a doublet, bit for bit as
before.  The driver side is :meth:`parsec_python.acceleration.driver._SpinSectorFallback.run`.

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .noncollinear_stability import (
    CLEAR,
    CONTINUATION_WINDOW,
    CONVERGENCE,
    GAIN,
    KICK,
    LINE_FIRST,
    LINE_LARGEST,
    LINE_PASSES,
    MINIMUM_STEPS,
    NEAR,
    PATTERN_PASSES,
    PATTERN_VECTORS,
    STEPS,
    WINDOW,
    ContinuationWatch,
    LineReport,
    flip_pattern as _vector_flip_pattern,
    line_search,
    nearest_atoms,
)

SWITCH = "PARSEC_SPIN_SECTOR_STABILITY"
_SETTINGS = ("auto", "off")
#: The value of :data:`SWITCH` where none is named.
DEFAULT = "off"
#: SUBSPACE steps of the eigensolver of each channel per probe, from the states at the centre.
PASSES = 1
#: SUBSPACE steps at the centre after the orbitals of the sectors were taken over (the columns the sectors did not list
#: start random), and the most by which a level of the full grid there may differ from that of the sectors (Ry).
CENTRE_PASSES = 3
CENTRE_LEVEL_TOLERANCE = 1.0e-4
#: The largest eigenvalue of the response from which the state is unstable (1 and the accuracy of one step per probe).
BOUND = 1.02
#: The free energy (Ry) by which the converged state of the loop on the full grid must lie below that of the sectors to
#: replace it.
CONTINUATION_MARGIN = 1.0e-6
#: Whether test 1 of the guard (a partly filled multiplet of the start Hamiltonian) puts its report aside too, so
#: that the loop converges in the sectors and its state is tested (else it sends the run to the full grid as before).
DEFER_START = True
#: The seed of the random part of the first direction.
SEED = 20261127
#: The moment (bohr magnetons) of an atom from which its sign counts in :func:`labelled_moments`.
MOMENT_THRESHOLD = 0.1
#: Boltzmann's constant in Ry/K.
KELVIN_RY = 6.33327186e-06


def spin_sector_stability_setting() -> str:
    """``PARSEC_SPIN_SECTOR_STABILITY``: ``auto`` tests the converged state of a collinear spin run in the sectors on
    the full grid where tests 1 or 3 of the guard would leave them, where the doublet note would be printed, or where
    the species start with polarizations of opposite sign, and continues on the full grid from an unstable state (this
    module); ``off`` (the default) keeps the guard as before, bit for bit.  Another value is an error before the
    SCF."""

    text = os.environ.get(SWITCH, DEFAULT).strip().lower() or DEFAULT
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be auto or off, not {text!r}")
    return text


class OrbitSymmetry:
    """The average of fields of ``channels`` channels (one array, channel after channel) over a group of grid
    operations: the average over the orbits of the grid points, which the row maps of generators of the group give
    (``rows[i]`` the grid row of the image of point ``i``; :func:`.spin_sectors.atom_group_images`).  For a group that
    acts on the points by permutations the average over the group is that over the orbits."""

    def __init__(self, size: int, generators, channels: int = 2) -> None:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        size = int(size)
        rows = [np.asarray(item, dtype=np.int64).reshape(-1) for item in generators]
        for item in rows:
            if item.size != size:
                raise ValueError("a row map of the group does not match the grid")
        self.size = size
        self.channels = int(channels)
        self.generators = len(rows)
        if rows:
            start = np.concatenate([np.arange(size, dtype=np.int64)] * len(rows))
            end = np.concatenate(rows)
            graph = coo_matrix((np.ones(start.size, dtype=np.int8), (start, end)), shape=(size, size)).tocsr()
            count, labels = connected_components(graph, directed=True, connection="weak")
        else:
            count, labels = size, np.arange(size)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.orbits = int(count)
        self.counts = np.bincount(self.labels, minlength=self.orbits).astype(np.float64)

    def _one(self, values: np.ndarray) -> np.ndarray:
        sums = np.bincount(self.labels, weights=values, minlength=self.orbits)
        return (sums / self.counts)[self.labels]

    def symmetric(self, combined: np.ndarray) -> np.ndarray:
        """The average over the group of every channel of ``combined``."""

        combined = np.asarray(combined, dtype=np.float64).reshape(self.channels, self.size)
        return np.concatenate([self._one(row) for row in combined])

    def breaking(self, combined: np.ndarray) -> np.ndarray:
        """The part of ``combined`` that breaks the group."""

        combined = np.asarray(combined, dtype=np.float64).reshape(-1)
        return combined - self.symmetric(combined)


@dataclass
class SpinStabilityReport:
    """What :func:`response_stability` found: ``largest`` the largest real part of the Ritz values of the response
    after the last probe, ``history`` that after every probe, ``ritz`` the real parts of the last Ritz values (largest
    first), ``probes`` the probes made, ``seconds`` the time, ``vector`` the Ritz vector of the largest Ritz value
    (largest entry 1) and ``others`` those of the next ones (up to ``PATTERN_VECTORS`` in all), ``centre`` the
    symmetric input they belong to; ``converged`` whether the largest Ritz value settled within the probes."""

    largest: float
    history: tuple[float, ...]
    ritz: tuple[float, ...]
    probes: int
    seconds: float
    orbits: int = 0
    generators: int = 0
    bound: float = BOUND
    passes: int = PASSES
    vector: np.ndarray | None = None
    centre: np.ndarray | None = None
    others: tuple = ()
    converged: bool = True
    # the largest difference (Ry) of the levels of the full grid at the centre from those of the sectors, up to the
    # Fermi level plus a window
    centre_deviation: float = 0.0

    @property
    def unstable(self) -> bool:
        return not self.largest < self.bound

    def text(self) -> str:
        return (
            f"largest eigenvalue {self.largest:.4f} of the SCF response to perturbations of the channel potentials "
            f"that break the point group of the sectors (unstable from {self.bound:g}; {self.probes} probes of "
            f"{self.passes} eigensolver step(s) of both channels on the full grid, Ritz values "
            + " ".join(f"{value:.3f}" for value in self.ritz[:4])
            + ("" if self.converged else "; not converged within the probes")
            + f"; the levels of the full grid at the state of the sectors within {self.centre_deviation:.1e} Ry of"
            f" theirs; {self.seconds:.2f} s)"
        )


def first_direction(symmetry: OrbitSymmetry, densities, seed: int = SEED) -> np.ndarray:
    """The part that breaks the group of a seeded random field times the density of each channel, and of the same
    field on the moment density (one array of both channels; zero where nothing breaks the group)."""

    densities = [np.asarray(item, dtype=np.float64).reshape(-1) for item in densities]
    scale = max(max(float(np.max(np.abs(item), initial=0.0)) for item in densities), 1.0e-30)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(densities), densities[0].size))
    field = np.concatenate([noise[index] * densities[index] / scale for index in range(len(densities))])
    if len(densities) == 2:
        moment = (densities[0] - densities[1]) / scale
        common = rng.standard_normal(moment.size) * np.abs(moment)
        field = field + np.concatenate([common, -common])
    return symmetry.breaking(field)


def response_stability(output, base: np.ndarray, symmetry: OrbitSymmetry, start: np.ndarray, *,
                       kick: float | None = None, steps: int | None = None,
                       reference: np.ndarray | None = None) -> SpinStabilityReport:
    """The largest eigenvalue of the response of one SCF step to perturbations that break the group of ``symmetry``
    at the input ``base`` (averaged over the group here) by Arnoldi steps from ``start`` (the plain inner product of
    the grid): ``output(x)`` is the output of one SCF step at the input ``x`` (the layout of ``base``), ``reference``
    that at the centre (its part that breaks the group, the noise of the eigensolver there, is taken from every
    probe; ``None``: none).  The steps stop at the first Ritz value from ``CLEAR`` (a Ritz value of the response is a
    lower bound of its largest eigenvalue), after at least ``MINIMUM_STEPS`` probes once the largest Ritz value moved
    by less than ``CONVERGENCE`` over the last ``WINDOW`` (a tenth of it from ``NEAR`` of the bound), or after
    ``steps`` (the report says it did not converge): the rules of
    :func:`.noncollinear_stability.sector_state_stability`."""

    started = time.perf_counter()
    kick = KICK if kick is None else float(kick)
    steps = STEPS if steps is None else int(steps)
    centre = symmetry.symmetric(base)
    noise = 0.0 if reference is None else symmetry.breaking(reference)
    start = symmetry.breaking(start)
    norm = float(np.linalg.norm(start))
    if not norm > 0.0:
        return SpinStabilityReport(float("-inf"), (), (), 0, time.perf_counter() - started, symmetry.orbits,
                                   symmetry.generators, centre=centre)
    basis = [start / norm]
    hessenberg = np.zeros((steps + 1, steps), dtype=np.float64)
    history: list[float] = []
    ritz = np.zeros(0)
    top = None
    following: list = []
    probes = 0
    converged = False
    for step in range(steps):
        direction = basis[step]
        size = float(kick) / max(float(np.max(np.abs(direction), initial=0.0)), 1.0e-30)
        result = np.asarray(output(centre + size * direction), dtype=np.float64).reshape(-1)
        probes += 1
        response = (symmetry.breaking(result) - noise) / size
        for _ in range(2):
            # (Gram-Schmidt twice)
            for row in range(step + 1):
                projection = float(np.dot(basis[row], response))
                hessenberg[row, step] += projection
                response = response - projection * basis[row]
        length = float(np.linalg.norm(response))
        hessenberg[step + 1, step] = length
        values, vectors = np.linalg.eig(hessenberg[: step + 1, : step + 1])
        order = np.argsort(values.real)[::-1]
        ritz = values.real[order]
        top = vectors[:, order[0]].real
        following = [vectors[:, index].real for index in order[1:PATTERN_VECTORS]]
        history.append(float(ritz[0]))
        if length <= 1.0e-12 * max(1.0, float(np.max(np.abs(hessenberg[: step + 1, : step + 1])))):
            converged = True
            break
        if ritz[0] >= CLEAR:
            converged = True
            break
        settled = CONVERGENCE if ritz[0] < NEAR * BOUND else 0.1 * CONVERGENCE
        if (
            len(history) >= max(MINIMUM_STEPS, WINDOW + 1)
            and max(history[-WINDOW - 1:]) - min(history[-WINDOW - 1:]) < settled
        ):
            converged = True
            break
        basis.append(response / length)
    vector = None
    others = []
    if top is not None:
        vector = np.zeros_like(centre)
        for weight, column in zip(top, basis):
            vector += float(weight) * column
        vector /= max(float(np.max(np.abs(vector))), 1.0e-300)
        for coefficients in following:
            other = np.zeros_like(centre)
            for weight, column in zip(coefficients, basis):
                other += float(weight) * column
            largest_entry = float(np.max(np.abs(other)))
            if largest_entry > 0.0:
                others.append(other / largest_entry)
    return SpinStabilityReport(
        largest=history[-1] if history else float("-inf"),
        history=tuple(history),
        ritz=tuple(float(value) for value in ritz),
        probes=probes,
        seconds=time.perf_counter() - started,
        orbits=symmetry.orbits,
        generators=symmetry.generators,
        vector=vector,
        centre=centre,
        others=tuple(others),
        converged=converged,
    )


def flip_pattern(centre: np.ndarray, vector: np.ndarray, labels: np.ndarray, atoms: int):
    """The collinear form of :func:`.noncollinear_stability.flip_pattern`: the moments of one of the two groups of
    atoms into which the change of the exchange field ``(dV_up - dV_down) / 2`` of the Ritz vector ``vector`` splits
    them turned over (``V_up`` and ``V_down`` exchanged on the grid points nearest to them), the smaller group (else
    the one the field change lowers); ``(input, turned atoms)`` or ``None``."""

    centre = np.asarray(centre, dtype=np.float64).reshape(2, -1)
    vector = np.asarray(vector, dtype=np.float64).reshape(2, -1)
    points = centre.shape[1]

    def as_field(pair):
        # (V, 0, 0, B) with V the mean of the channels and B half their difference
        return np.concatenate([0.5 * (pair[0] + pair[1]), np.zeros(2 * points), 0.5 * (pair[0] - pair[1])])

    found = _vector_flip_pattern(as_field(centre), as_field(vector), labels, atoms)
    if found is None:
        return None
    start, flipped = found
    potential = start[:points]
    field = start[3 * points:]
    return np.concatenate([potential + field, potential - field]), flipped


def labelled_moments(coordinates, positions, labels, starts, densities, volume_element: float, lattice_vectors=None,
                     threshold: float = MOMENT_THRESHOLD) -> tuple[str, bool]:
    """The moments of the atoms (bohr magnetons: ``up - down`` summed over the grid points nearest to each atom,
    :func:`.noncollinear_stability.nearest_atoms`) by label, as text, and whether they keep the order of the labels:
    every atom whose label starts with a nonzero ``Initial_Spin_Polarization`` (``starts``, one per atom) has a moment
    of at least ``threshold`` whose sign is that of its start, or the opposite for every such atom (an order turned as
    a whole is the same order).  A label of at most four atoms lists them, a larger one its range and mean."""

    partition = nearest_atoms(coordinates, positions, lattice_vectors)
    up, down = (np.asarray(item, dtype=np.float64).reshape(-1) for item in densities)
    moments = np.bincount(partition, weights=up - down, minlength=len(positions)) * float(volume_element)
    starts = np.asarray(starts, dtype=np.float64)
    parts = []
    for label in dict.fromkeys(labels):
        chosen = np.array([index for index, name in enumerate(labels) if name == label], dtype=np.int64)
        values = moments[chosen]
        start = float(starts[chosen[0]])
        if values.size <= 4:
            shown = " ".join(f"{value:+.2f}" for value in values)
        else:
            shown = (f"{values.size} atoms from {float(np.min(values)):+.2f} to {float(np.max(values)):+.2f}, "
                     f"mean {float(np.mean(values)):+.2f}")
        parts.append(f"{label} (start {start:+g}) {shown}")
    expected = np.sign(starts)
    read = expected != 0.0
    found = np.where(np.abs(moments) >= threshold, np.sign(moments), 0.0)
    kept = (not np.any(read)) or any(bool(np.all(found[read] == sign * expected[read])) for sign in (1.0, -1.0))
    return "; ".join(parts), bool(kept)


def spin_free_energy(total: float, occupations, temperature: float) -> float:
    """``E - TS`` of the two channels' occupations (one electron per state at most) at the Fermi temperature."""

    energy = float(total)
    if not temperature > 0.0:
        return energy
    entropy = 0.0
    for values in occupations:
        f = np.asarray(values, dtype=np.float64).reshape(-1)
        clipped = np.clip(f, 1.0e-300, 1.0 - 1.0e-16)
        terms = clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        entropy += float(np.sum(np.where((f <= 0.0) | (f >= 1.0), 0.0, terms)))
    return energy + KELVIN_RY * float(temperature) * entropy


def host_states(states):
    """Host copies of the device subspace states of both channels of a loop (``CuPyEigvalDeviceState``, the eigensolver
    states of its result), ``None`` where one is missing: what a result keeps of its eigensolvers when its system goes
    (:func:`adopt_states`)."""

    from dataclasses import replace

    if states is None or any(state is None or getattr(state, "subspace", None) is None for state in states):
        return None
    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    return tuple(
        replace(state, subspace=replace(
            state.subspace, vectors=np.asfortranarray(cp.asnumpy(state.subspace.vectors)),
            eigenvalues=np.asarray(cp.asnumpy(state.subspace.eigenvalues)), ritz_workspace=None,
        ))
        for state in states
    )


def adopt_states(system, states):
    """The persistent eigensolvers of both channels of the full-grid ``system`` (``backends.cupy_runtime.
    CuPySCFEigensolver``, made where they have not solved yet) hold device copies of ``states`` (of
    :func:`host_states`) in place of theirs; returns the states they hold, ``None`` where the system has no such
    eigensolvers.  Every later reader of the device state (the nonlocal forces, a warm start of the next structure)
    then reads the orbitals of ``states``."""

    from dataclasses import replace

    from parsec_python.SCF.single_point import eigval_settings_for, preconditioner_neighbors

    from ..backends.cupy import require_cupy
    from ..Eigensolvers.eigval import CuPyEigvalSolver

    for_channel = getattr(getattr(system, "eigenproblem_solver", None), "for_spin_channel", None)
    if not callable(for_channel) or states is None:
        return None
    channels = [for_channel(spin) for spin in range(len(states))]
    if not all(hasattr(channel, "operator") and hasattr(channel, "solver") for channel in channels):
        return None
    cp, _ = require_cupy()
    eigensolver = system.input.eigensolver
    held = []
    for channel, state in zip(channels, states):
        if channel.solver is None:
            channel.solver = CuPyEigvalSolver(
                channel.operator,
                settings=eigval_settings_for(
                    eigensolver, eigensolver.filter_degree, neighbors=preconditioner_neighbors(system, eigensolver)
                ),
                retain_vectors_on_device=True,
                compute_subspace_residuals=False,
            )
        channel.solver._state = None
        channel.solver._state = replace(state, subspace=replace(
            state.subspace, vectors=cp.asarray(state.subspace.vectors, dtype=cp.float64, order="F"),
            eigenvalues=cp.asarray(state.subspace.eigenvalues, dtype=cp.float64),
        ))
        held.append(channel.solver.device_state)
    return tuple(held)


class FullGridSpinStep:
    """One SCF step of both spin channels on a prepared full-grid accelerated system (a semilocal functional,
    Fermi-Dirac occupations): the levels of each channel from SUBSPACE steps of its eigensolver from the states it
    holds, one Fermi level, the densities, the Hartree potential of their sum and the exchange-correlation potentials.
    :meth:`take_orbitals` gives each channel's eigensolver the orbitals of the sectors on the full grid (completed by
    random columns to its working states, orthonormalized and rotated to Ritz vectors), :meth:`keep_centre` saves the
    states and :meth:`restore` puts them back before a probe."""

    def __init__(self, system, states: int, hartree_potential) -> None:
        from parsec_python.SCF.single_point import eigval_settings_for, preconditioner_neighbors
        from parsec_python.SCF.spin_polarized import _channel_solvers

        self.system = system
        self.states = int(states)
        self.settings = system.input.scf
        eigensolver = system.input.eigensolver
        self.eigval_settings = eigval_settings_for(
            eigensolver, eigensolver.filter_degree, neighbors=preconditioner_neighbors(system, eigensolver)
        )
        self.solvers = _channel_solvers(system.eigenproblem_solver)
        for channel in self.solvers:
            if not (hasattr(channel, "operator") and hasattr(channel, "solver")):
                # (the persistent device eigensolver of one GPU, backends.cupy_runtime.CuPySCFEigensolver)
                raise NotImplementedError(
                    f"the eigensolver of the full grid ({type(channel).__name__}) takes no orbitals from outside"
                )
        self.ionic_potential = np.asarray(system.ionic_potential, dtype=np.float64)
        self.hartree_potential = np.asarray(hartree_potential, dtype=np.float64)
        self.volume_element = float(system.grid.volume_element)
        self.temperature = float(self.settings.fermi_temperature_kelvin)
        self._centre = None
        self.last = None

    def _channel_solver(self, spin: int):
        """The persistent eigensolver of channel ``spin`` (made here before its first solve)."""

        from ..Eigensolvers.eigval import CuPyEigvalSolver

        channel = self.solvers[spin]
        if channel.solver is None:
            channel.solver = CuPyEigvalSolver(
                channel.operator,
                settings=self.eigval_settings,
                retain_vectors_on_device=True,
                compute_subspace_residuals=False,
            )
        return channel.solver

    def _operator(self, potential):
        bound = self.system.hamiltonian(np.asarray(potential, dtype=np.float64))
        factory = getattr(bound, "as_eigensolver_operator", bound.as_linear_operator)
        return factory()

    def take_orbitals(self, orbitals, potentials, seed: int = SEED) -> None:
        """The orbitals of the sectors on the full grid (``orbitals``: one host or device ``(n, states)`` array per
        channel) as the states of each channel's eigensolver at the potentials of the sectors (``potentials``)."""

        from ..backends.cupy import require_cupy
        from ..Eigensolvers.eigval import CuPyEigvalDeviceState
        from ..Eigensolvers.subspace import DeviceSubspaceState

        cp, _ = require_cupy()
        rng = np.random.default_rng(seed)
        for spin in range(2):
            solver = self._channel_solver(spin)
            solver.update_settings(self.eigval_settings)
            working = solver._working_state_count(self.states)
            given = cp.asarray(orbitals[spin], dtype=cp.float64)
            count = min(int(given.shape[1]), working)
            block = cp.empty((given.shape[0], working), dtype=cp.float64, order="F")
            block[:, :count] = given[:, :count]
            if working > count:
                block[:, count:] = cp.asarray(rng.standard_normal((given.shape[0], working - count)))
            del given
            block, _ = cp.linalg.qr(block)
            operator = self._operator(potentials[spin])
            projected = block.T @ operator.apply(block)
            values, rotation = cp.linalg.eigh(0.5 * (projected + projected.T))
            block = cp.asfortranarray(block @ rotation)
            subspace = DeviceSubspaceState(
                operator_dimension=int(block.shape[0]), working_states=working, eigenvalues=values, vectors=block,
                filter_lower_bound=None, first_filter=True, filters_completed=0,
            )
            solver._state = CuPyEigvalDeviceState(
                operator_dimension=int(block.shape[0]), requested_states=self.states, working_states=working,
                initial_method=solver.settings.initial_method, subspace=subspace, solves_completed=1,
            )

    def keep_centre(self) -> None:
        """Save copies of the present states of both channels (the states at the centre), on the host: the device
        holds the states of the probe only (beside the sectors, whose state the run may keep)."""

        from dataclasses import replace

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        kept = []
        for spin in range(2):
            state = self._channel_solver(spin).device_state
            kept.append(replace(state, subspace=replace(
                state.subspace, vectors=np.asfortranarray(cp.asnumpy(state.subspace.vectors)),
                eigenvalues=np.asarray(cp.asnumpy(state.subspace.eigenvalues)), ritz_workspace=None, first_filter=True,
            )))
        self._centre = kept

    def restore(self) -> None:
        """Put device copies of the saved states back into both channels' eigensolvers."""

        from dataclasses import replace

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        for spin in range(2):
            state = self._centre[spin]
            solver = self._channel_solver(spin)
            # (the state of the probe before goes first)
            solver._state = None
            solver._state = replace(state, subspace=replace(
                state.subspace, vectors=cp.asarray(state.subspace.vectors, dtype=cp.float64, order="F"),
                eigenvalues=cp.asarray(state.subspace.eigenvalues, dtype=cp.float64),
            ))

    def states_now(self):
        """Host copies of the present states of both channels (for a loop that continues from them,
        :meth:`adopt`)."""

        from dataclasses import replace

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        copies = []
        for spin in range(2):
            state = self._channel_solver(spin).device_state
            copies.append(replace(state, subspace=replace(
                state.subspace, vectors=np.asfortranarray(cp.asnumpy(state.subspace.vectors)),
                eigenvalues=np.asarray(cp.asnumpy(state.subspace.eigenvalues)), ritz_workspace=None,
            )))
        return tuple(copies)

    def adopt(self, states):
        """Both channels' eigensolvers hold ``states`` (of :meth:`states_now`) on the device; returns the states they
        hold, those a loop that continues from them is handed."""

        from dataclasses import replace

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        held = []
        for spin, state in enumerate(states):
            solver = self._channel_solver(spin)
            solver._state = None
            solver._state = replace(state, subspace=replace(
                state.subspace, vectors=cp.asarray(state.subspace.vectors, dtype=cp.float64, order="F"),
                eigenvalues=cp.asarray(state.subspace.eigenvalues, dtype=cp.float64),
            ))
            held.append(solver.device_state)
        return tuple(held)

    def __call__(self, combined, passes: int, energy: bool = False):
        """``(outputs, densities, free energy or None)`` of one step at the input ``combined`` (``V_up`` then
        ``V_down``), each pair as one array; ``passes`` SUBSPACE steps per channel from the states held."""

        from parsec_python.Energy import spin_total_energy
        from parsec_python.SCF.spin_polarized import _spin_occupations

        combined = np.asarray(combined, dtype=np.float64).reshape(2, -1)
        levels = []
        vectors = []
        for spin in range(2):
            solver = self._channel_solver(spin)
            operator = self._operator(combined[spin])
            solution = None
            for _ in range(int(passes)):
                solution = self.solvers[spin](operator, self.states, settings=self.eigval_settings,
                                              state=solver.device_state)
            levels.append(np.asarray(solution.eigenvalues, dtype=np.float64))
            vectors.append(solution.vectors)
        # (one Fermi level of the levels in ascending order, the occupations back in the order of the orbitals: a
        # SUBSPACE step need not return its levels sorted)
        orders = [np.argsort(values, kind="stable") for values in levels]
        filled = _spin_occupations([values[order] for values, order in zip(levels, orders)], self.system,
                                   self.settings)
        occupations = []
        for order, values in zip(orders, filled.occupations):
            placed = np.empty(order.size, dtype=np.float64)
            placed[order] = np.asarray(values, dtype=np.float64)
            occupations.append(placed)
        builder = self.system.orbital_density_builder
        densities = [
            np.asarray(builder(vectors[spin], 0.5 * occupations[spin], self.volume_element), dtype=np.float64)
            for spin in range(2)
        ]
        hartree = self.system.solve_hartree(densities[0] + densities[1], initial_potential=self.hartree_potential)
        xc = self.system.evaluate_spin_xc(densities[0], densities[1])
        outputs = [self.ionic_potential + hartree.potential + xc.potentials[spin] for spin in range(2)]
        free = None
        if energy:
            energies = spin_total_energy(
                levels, occupations, densities, [combined[0], combined[1]], self.ionic_potential, hartree.potential,
                list(xc.potentials), xc.total_energy, self.system.ion_ion_energy, self.volume_element,
                **({"alpha_z_energy": self.system.alpha_z_energy}
                   if getattr(self.system, "alpha_z_energy", 0.0) else {}),
            )
            free = spin_free_energy(energies.total, occupations, self.temperature)
        self.last = (levels, occupations, float(filled.fermi_level))
        return np.concatenate(outputs), np.concatenate(densities), free


@dataclass
class StabilityOutcome:
    """What :func:`test_sector_state` decided: ``report`` of the test, ``line`` of the line search (``None`` for a
    stable state), and for a continuation the input ``start`` (``(V_up, V_down)``) and the states of both channels
    there (``states``); ``seconds`` the whole time, ``setup`` that of the orbitals taken over and the centre passes."""

    report: SpinStabilityReport
    line: LineReport | None = None
    start: tuple | None = None
    states: tuple | None = None
    seconds: float = 0.0
    setup: float = 0.0

    @property
    def continues(self) -> bool:
        return self.start is not None


def test_sector_state(step: FullGridSpinStep, symmetry: OrbitSymmetry, centre, densities, orbitals, levels,
                      fermi_level: float, labels=None) -> StabilityOutcome:
    """The stability test of the converged state of the sectors (the module text) with ``step`` on the full grid:
    ``centre`` the input ``(V_up, V_down)`` of the last step of the sectors, ``densities`` their channel densities,
    ``orbitals`` and ``levels`` their orbitals (full grid) and levels per channel, ``labels()`` the nearest atom of
    every grid point and the number of atoms (the partition of the flip patterns; ``None``: no patterns)."""

    started = time.perf_counter()
    base = symmetry.symmetric(np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in centre]))
    pair = base.reshape(2, -1)
    step.take_orbitals(orbitals, pair)
    del orbitals
    reference = step(base, CENTRE_PASSES)[0]
    found, _, _ = step.last
    window = float(fermi_level) + 3.0e-2
    deviation = 0.0
    for given, now in zip(levels, found):
        given = np.sort(np.asarray(given, dtype=np.float64))
        now = np.sort(np.asarray(now, dtype=np.float64))
        count = min(given.size, now.size)
        shown = given[:count] <= window
        deviation = max(deviation, float(np.max(np.abs(given[:count] - now[:count])[shown], initial=0.0)))
    step.keep_centre()
    setup = time.perf_counter() - started

    def output(probe):
        step.restore()
        return step(probe, PASSES)[0]

    report = response_stability(output, base, symmetry, first_direction(symmetry, densities), reference=reference)
    report.centre_deviation = deviation
    outcome = StabilityOutcome(report=report, setup=setup)
    if report.unstable and report.vector is not None:
        direction = np.asarray(report.vector, dtype=np.float64)

        def free_energy(shift):
            return step(report.centre + float(shift) * direction, LINE_PASSES, energy=True)[2]

        step.restore()
        line = line_search(free_energy, restart=step.restore)
        best_states = None
        if line.pays:
            # the states of the lowest point of the line, from the centre's
            step.restore()
            step(report.centre + line.shift * direction, LINE_PASSES)
            best_states = step.states_now()
        if labels is not None:
            partition, atoms = labels()
            patterns = []
            for candidate in (direction, *report.others):
                found_pattern = flip_pattern(report.centre, candidate, partition, atoms)
                if found_pattern is not None and not any(
                    np.array_equal(found_pattern[1], other[1]) for other in patterns
                ):
                    patterns.append(found_pattern)
            from dataclasses import replace

            started_pattern = time.perf_counter()
            for start_point, flipped in patterns:
                step.restore()
                value = step(start_point, PATTERN_PASSES, energy=True)[2]
                if line.pattern is None or value < line.pattern:
                    line = replace(line, pattern=float(value), flipped=int(np.count_nonzero(flipped)),
                                   start=start_point)
                    if line.from_pattern and line.pays:
                        best_states = step.states_now()
            if patterns:
                line = replace(line, seconds=line.seconds + time.perf_counter() - started_pattern,
                               patterns=len(patterns))
        outcome.line = line
        if line.pays:
            start = line.start if line.from_pattern else report.centre + line.shift * direction
            start = np.asarray(start, dtype=np.float64).reshape(2, -1)
            outcome.start = (start[0].copy(), start[1].copy())
            outcome.states = best_states
    outcome.seconds = time.perf_counter() - started
    return outcome


__all__ = [
    "BOUND",
    "CENTRE_LEVEL_TOLERANCE",
    "CENTRE_PASSES",
    "CONTINUATION_MARGIN",
    "CONTINUATION_WINDOW",
    "ContinuationWatch",
    "DEFAULT",
    "FullGridSpinStep",
    "GAIN",
    "LINE_FIRST",
    "LINE_LARGEST",
    "LineReport",
    "MOMENT_THRESHOLD",
    "OrbitSymmetry",
    "PASSES",
    "SWITCH",
    "SpinStabilityReport",
    "StabilityOutcome",
    "adopt_states",
    "first_direction",
    "host_states",
    "flip_pattern",
    "labelled_moments",
    "line_search",
    "nearest_atoms",
    "response_stability",
    "spin_free_energy",
    "spin_sector_stability_setting",
    "test_sector_state",
]
