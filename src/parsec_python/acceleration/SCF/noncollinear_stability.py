"""The stability test of the state of a non-collinear loop in the sectors of its k-points
(``PARSEC_NONCOLLINEAR_SECTOR_STABILITY``).

The sectors of the k-points of a non-collinear loop (:mod:`.kpoint_spinor_sectors`,
``PARSEC_NONCOLLINEAR_KPOINT_SECTORS``) hold every state in the symmetric form of the operations they take: ``V(g r) = V(r)`` and ``B(g r) = R' B(r)`` for the
operations ``g`` of the sectors of its k-points, ``B(h r) = -R' B(r)`` for the operation ``h`` of their Kramers pairs
(``h`` times time reversal).  The loop converges to a self-consistent state of that symmetry, a stationary point of the
free energy, which need not be a minimum: the symmetric ensemble of a partly filled multiplet (the four pi* levels of
O2 from antiparallel start moments, two electrons in a sector and its Kramers partner) lies above a state that breaks
the symmetry (the triplet), and the loop without sectors moves to it.

**The test.**  Let ``x`` be the input potential ``(V, B)`` of the last step of the loop, averaged over the group ``G``
that those operations generate (:class:`SectorSymmetry`), and ``F`` the map of one SCF step on the whole grid (the
spinors of every k-point at ``x``, their occupations, the density and magnetization, the Hartree and
exchange-correlation potentials).  ``F`` commutes with ``G``, so its Jacobian ``J`` at ``x`` maps the perturbations
that break ``G`` (``w`` with ``S w = 0``, ``S`` the average over ``G``) onto themselves.  ``J = K chi_0`` (``K`` the
Hartree and exchange-correlation kernel, ``chi_0`` the static response of the Kohn-Sham system, negative
semidefinite), and the Hessian of the free energy at a self-consistent state has the inertia of the dielectric
operator ``1 - K chi_0`` (Herbst and Levitt 2022, eqs. 18 and 22; the stability analysis of Kohn-Sham solutions,
Bauernschmitt and Ahlrichs 1996): the state is a minimum along the perturbations that break ``G`` where every eigenvalue
of ``J`` on them is below 1, and the free energy goes down along a direction that breaks ``G`` where one is above 1.
A loop on the whole grid then leaves the symmetric state (a damped SCF step multiplies such a component by
``1 + alpha (lambda - 1)``).

``J`` is probed by Arnoldi steps in the subspace that breaks ``G``: ``J w = (1 - S) F(x + s w) / s`` (``(1 - S) F(x) =
0``) with ``max |s w|`` = :data:`KICK`, each ``F`` from a solver with one task per k-point whose :data:`PASSES` SUBSPACE
steps start from the Ritz states of the sectors unfolded to the whole grid (:meth:`.kpoint_spinor_sectors.
SpinorSectorKPointSolver.whole_states`).  The first direction is the part that breaks ``G`` of a field along a generic
direction on the density and of a seeded random field on the density (no potential: the charge response, whose large
negative eigenvalues slow the Arnoldi steps towards the largest one, enters through the coupling of a magnetic state).
Without spin-orbit coupling the rotations of the field, along which the energy does not change and ``J`` has the
eigenvalue 1 in the local frame of the exchange-correlation field, are taken out (:func:`spin_rotation_modes`; in the
collinear frame of a gradient functional, ``PARSEC_NONCOLLINEAR_GGA_FRAME``, the field stays along the axis of the
initial moments, a turned field gives no turned response and these directions have an eigenvalue near 0, so taking
them out changes nothing).  The largest Ritz value of a metal climbs slowly and in steps
(Au8 at Gamma: 0.286 after six probes, 0.425 after about eighteen; with antiparallel moments a plateau of 0.182-0.186
over probes 2-7 before it climbs), so the steps stop at the first Ritz value from :data:`CLEAR` (it is a lower bound of
the largest eigenvalue of the symmetric response), or after at least :data:`MINIMUM_STEPS` probes once the largest Ritz
value moved by less than :data:`CONVERGENCE` over the last :data:`WINDOW` (a tenth of it near the bound), or after
:data:`STEPS` (the report says it did not converge).  The bound is 1 and the accuracy of one eigensolver step per probe.
On request (``energy_metric``) the probes also give the response of the density and magnetization, ``-chi_0 b_j``: in
the inner product of the energy ``<a, -chi_0 b>`` the response is symmetric, and the report then holds its Ritz values
in that inner product (:func:`energy_metric_ritz`; bounded by the largest eigenvalue, where those of the plain steps
need not be).

**An unstable state** (:func:`parsec_python.SCF.noncollinear.run_self_consistent_noncollinear_kpoints`): the loop
without sectors continues from the input of the sectors displaced along the Ritz vector of the largest Ritz value, the
direction in which the free energy goes down, and the lower of its state and that of the sectors is the result (a loop
that starts again from the start reaches the same symmetric state in the tests, where nothing but the noise of its
eigensolver breaks the symmetry: O2 at 3000 to 6000 K, Au8 lattices of 10.5 to 12 bohr).  Under
``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto`` (the default) a line search of the free energy along that vector comes
first (:func:`line_search`), and the free energy at the flip patterns of the top Ritz vectors (:func:`flip_pattern`: the
local fields of one of the two groups of atoms that a soft mode moves against each other turned over, a finite-amplitude
start such as the antiparallel start of a parallel state): the free energy of the density matrix of one SCF step at any
input is an upper bound of the lowest free energy, so a point below the state of the sectors shows a lower state.  Where
no point lies :data:`GAIN` below it the state of the sectors is kept: a threshold on what these few points show, not a
bound on what the loop without sectors could reach (the gain of a soft mode goes as ``(lambda - 1)^2`` along the line,
and a lower state need not lie along any of them).  Otherwise the loop continues from the lowest of them (with the Ritz
states of the sectors, or those of the pattern) and stops, keeping the state of the sectors, once its free energy has
not been :data:`GAIN` below theirs for :data:`CONTINUATION_WINDOW` iterations (:class:`ContinuationWatch`; an Anderson
loop can converge back to the unstable state).  ``off``: the displacement :data:`DISPLACEMENT`, a fresh start of the eigensolver
and up to ``Max_Iter`` iterations, as before.
A state with a largest eigenvalue from 1 to the bound is kept, its note says that a lower state may exist.  The same test
of the flip route of the spin loops is ``PARSEC_SPIN_FLIP_STABILITY`` (:func:`parsec_python.SCF.spin_polarized.
_flip_stability`).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

SWITCH = "PARSEC_NONCOLLINEAR_SECTOR_STABILITY"
_SETTINGS = ("auto", "off")
#: The value of :data:`SWITCH` where none is named.
DEFAULT = "auto"
#: The largest change (Ry) of the potential or of a component of the field of one probe of the response.
KICK = 5.0e-3
#: Probes (Arnoldi steps) of the response at most, and at least: the largest Ritz value of a metal climbs slowly and
#: in steps (Au8 at Gamma: 0.286 after 6 probes, 0.425 after about 18; with antiparallel moments a plateau 0.182-0.186
#: over probes 2-7, then 0.425), so a test stops only after :data:`MINIMUM_STEPS` probes and once the largest Ritz value
#: moved by less than :data:`CONVERGENCE` over the last :data:`WINDOW` probes (a tenth of it from :data:`NEAR` of the
#: bound on: an Au8 lattice of 12 bohr settled at 1.015 within 1e-2 and converges to 1.040), or as soon as it reaches
#: :data:`CLEAR` (a Ritz value of the symmetric response is a lower bound of its largest eigenvalue; below that the
#: steps go on, so that the Ritz vector the loop without sectors continues along is converged as well).
STEPS = 30
MINIMUM_STEPS = 12
WINDOW = 3
CONVERGENCE = 1.0e-2
NEAR = 0.8
CLEAR = 1.5
#: SUBSPACE steps of the eigensolver of every k-point per probe, from the Ritz states of the sectors (one gives the
#: eigenvalue 1 of the rotations of the moments of O2 without spin-orbit coupling to 1.6e-2, two to 1.1e-2).
PASSES = 1
#: The largest eigenvalue of the response from which the state is unstable: 1 and the accuracy of one eigensolver step
#: per probe (the eigenvalue 1 of the rotations of the moments of O2 without spin-orbit coupling comes out 0.984).
BOUND = 1.02
#: The largest change (Ry) of the potential or a component of the field along the top Ritz vector from which the loop
#: without sectors continues after an unstable state (``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=off``).
DISPLACEMENT = 5.0e-2
#: The seed of the random part of the first direction.
SEED = 20261125

CONTINUATION_SWITCH = "PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION"
#: The value of :data:`CONTINUATION_SWITCH` where none is named.
CONTINUATION_DEFAULT = "auto"
#: The free energy (Ry) below the state of the sectors from which an unstable state is continued
#: (``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto``): where neither the line search along the top Ritz vector nor a flip
#: pattern goes down by this much the state of the sectors is kept, and the loop without sectors stops once its free
#: energy has not been that far below the state of the sectors for :data:`CONTINUATION_WINDOW` iterations.  A threshold
#: on the points tried, not a bound on the gain missed.
GAIN = 1.0e-4
#: The first displacement (Ry, the largest change of the potential or a component of the field) of the line search,
#: on both sides; from the lower side the displacement doubles while the free energy goes down, up to
#: :data:`LINE_LARGEST`.
LINE_FIRST = 1.25e-2
LINE_LARGEST = 0.4
#: SUBSPACE steps of the eigensolver of every k-point at each point of the line search, from the states of the centre
#: or of the point before (one step put O2 at 5850 K a third short at the first point and missed its gain).
LINE_PASSES = 2
#: Iterations of the loop without sectors after the last one whose free energy was :data:`GAIN` below that of the
#: sectors before it stops (``auto``).
CONTINUATION_WINDOW = 40
#: Iterations in which the residual of the loop without sectors must halve while it is more than :data:`STALL_FACTOR`
#: times the convergence criterion (``auto``; else it stops and its state is not taken).
STALL_WINDOW = 100
STALL_FACTOR = 10.0
#: The flip pattern (:func:`flip_pattern`): the share of the largest local field below which an atom keeps its field,
#: the share of the change of a local field along it from which the atom turns over, and the eigensolver steps of every
#: k-point at its input (from the states of the sectors).
PATTERN_MOMENT = 0.2
PATTERN_ALIGNMENT = 0.5
PATTERN_PASSES = 4
#: The Ritz vectors (of the largest Ritz values) whose flip patterns are tried.
PATTERN_VECTORS = 3


def noncollinear_sector_stability_setting() -> str:
    """``PARSEC_NONCOLLINEAR_SECTOR_STABILITY``: ``auto`` (the default) tests the state of a converged non-collinear
    loop in the sectors of its k-points against perturbations that break their operations (this module) in place of
    tests 1 and 3 of the guard (``PARSEC_KPOINT_SECTOR_GUARD``: partly filled multiplets across the sectors at the start
    and at the end; a loop in sectors that did not converge is kept untested, with a note); ``off`` keeps tests 1 and
    3 and no stability test, bit for bit as before.  Another value is an error before the SCF (this also checks
    :data:`CONTINUATION_SWITCH`)."""

    text = os.environ.get(SWITCH, DEFAULT).strip().lower() or DEFAULT
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be auto or off, not {text!r}")
    noncollinear_sector_continuation_setting()
    return text


def noncollinear_sector_continuation_setting() -> str:
    """``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION``: what follows a state of the sectors that the stability test finds
    unstable.  ``auto`` (the default): a line search of the free energy along the top Ritz vector (:func:`line_search`);
    where it goes down by less than :data:`GAIN` the state of the sectors is kept, else the loop without sectors
    continues from the lowest point of the line with the Ritz states of the sectors and stops, keeping the state of the
    sectors, once its free energy has not been :data:`GAIN` below theirs for :data:`CONTINUATION_WINDOW` iterations
    (:class:`ContinuationWatch`).  ``off``: the loop without sectors continues from the displacement
    :data:`DISPLACEMENT` with a fresh start of its eigensolver for up to ``Max_Iter`` iterations, bit for bit as
    before.  Another value is an error before the SCF."""

    text = os.environ.get(CONTINUATION_SWITCH, CONTINUATION_DEFAULT).strip().lower() or CONTINUATION_DEFAULT
    if text not in _SETTINGS:
        raise ValueError(f"{CONTINUATION_SWITCH} must be auto or off, not {text!r}")
    return text


def _axial(matrix) -> np.ndarray:
    """``R' = det(R) R``: how an operation turns an axial vector (a moment, a field)."""

    rotation = np.asarray(matrix, dtype=np.float64)
    return float(round(float(np.linalg.det(rotation)))) * rotation


class SectorSymmetry:
    """The group ``G`` that the operations of the sectors generate, as ``(operation, sign)`` of the cell group:
    ``sign`` 1 for an operation ``g`` of the sectors (``V(g r) = V(r)``, ``B(g r) = R' B(r)``), -1 for one times time
    reversal (``B(g r) = -R' B(r)``).  :meth:`symmetric` is the average of a potential and field ``(V, B)`` (one array:
    ``V`` then the three components of ``B``) over ``G``, :meth:`breaking` the rest."""

    def __init__(self, cell, operations) -> None:
        table = cell.table
        found = {(0, 1)} | {(int(index), int(sign)) for index, sign in operations}
        frontier = sorted(found)
        while frontier:
            fresh = []
            for left, left_sign in frontier:
                for right, right_sign in sorted(found):
                    for product_ in ((int(table[left, right]), left_sign * right_sign),
                                     (int(table[right, left]), left_sign * right_sign)):
                        if product_ not in found:
                            found.add(product_)
                            fresh.append(product_)
            frontier = fresh
        self.elements = sorted(found)
        self._rows: dict[int, np.ndarray] = {}
        self._turns: dict[int, np.ndarray] = {}
        for index, _ in self.elements:
            if index in self._rows:
                continue
            rows = cell.row_map(index)
            if rows is None:
                raise RuntimeError(f"operation {index} of the sectors does not map the grid onto itself")
            self._rows[index] = np.asarray(rows, dtype=np.int64)
            self._turns[index] = _axial(cell.matrices[index])

    @property
    def order(self) -> int:
        return len(self.elements)

    def symmetric(self, combined: np.ndarray) -> np.ndarray:
        """The average over ``G`` of ``(V, B)``: ``V(g r)`` and ``s R'^T B(g r)`` of every element."""

        combined = np.asarray(combined, dtype=np.float64)
        points = combined.size // 4
        potential = combined[:points]
        field = combined[points:].reshape(3, points)
        out_potential = np.zeros(points)
        out_field = np.zeros((3, points))
        for index, sign in self.elements:
            rows = self._rows[index]
            out_potential += potential[rows]
            out_field += float(sign) * (self._turns[index].T @ field[:, rows])
        scale = 1.0 / len(self.elements)
        return np.concatenate([scale * out_potential, scale * out_field.reshape(-1)])

    def breaking(self, combined: np.ndarray) -> np.ndarray:
        """The part of ``(V, B)`` that breaks ``G``."""

        combined = np.asarray(combined, dtype=np.float64)
        return combined - self.symmetric(combined)


@dataclass
class StabilityReport:
    """What :func:`sector_state_stability` found: ``largest`` the largest real part of the Ritz values of the response
    after the last probe, ``history`` that after every probe, ``ritz`` the real parts of the last Ritz values (largest
    first), ``probes`` the probes made, ``order`` the order of ``G``, ``seconds`` the time of the test."""

    largest: float
    history: tuple[float, ...]
    ritz: tuple[float, ...]
    probes: int
    order: int
    seconds: float
    passes: int = PASSES
    bound: float = BOUND
    rotations: int = 0
    # the Ritz vector of the largest Ritz value as (V, B) (None where no probe was made), and the input it belongs to
    vector: np.ndarray | None = None
    centre: np.ndarray | None = None
    converged: bool = True
    # where the probes also give the response of the density and magnetization (:func:`energy_metric_ritz`): the
    # largest Ritz value of the response in the inner product of the energy, and the Rayleigh quotient and the
    # curvature ``<v, -chi_0 v>`` (per grid point) of the top Ritz vector ``vector``
    metric_largest: float | None = None
    metric_quotient: float | None = None
    metric_curvature: float | None = None
    metric_history: tuple[float, ...] = ()
    # the Ritz vectors of the next largest Ritz values (largest component 1), after ``vector``: up to
    # :data:`PATTERN_VECTORS` in all, the moment patterns of the soft modes (:func:`flip_pattern`)
    others: tuple = ()

    @property
    def unstable(self) -> bool:
        return not self.largest < self.bound

    def text(self) -> str:
        return (
            f"largest eigenvalue {self.largest:.4f} of the SCF response to perturbations that break the {self.order} "
            f"operations of the sectors (unstable from {self.bound:g}; {self.probes} probes of {self.passes} "
            "eigensolver steps on the whole grid, Ritz values "
            + " ".join(f"{value:.3f}" for value in self.ritz[:4])
            + ("" if self.metric_largest is None else f"; in the energy metric {self.metric_largest:.4f}")
            + (f"; {self.rotations} rotations of the moments left out" if self.rotations else "")
            + ("" if self.converged else "; not converged within the probes")
            + f"; {self.seconds:.2f} s)"
        )


def first_direction(symmetry: SectorSymmetry, density: np.ndarray, seed: int = SEED) -> np.ndarray:
    """The part that breaks ``G`` of a field along ``(1, 0.8, 0.6)`` on the density and of a seeded random field on the
    density, as ``(V, B)`` (no potential: the charge response, whose large negative eigenvalues slow the Arnoldi steps
    towards the largest, enters through the coupling of a magnetic state alone); zero where nothing breaks ``G``."""

    density = np.asarray(density, dtype=np.float64).reshape(-1)
    points = density.size
    weight = density / max(float(np.max(np.abs(density), initial=0.0)), 1.0e-30)
    rng = np.random.default_rng(seed)
    direction = np.array([1.0, 0.8, 0.6]) / np.sqrt(2.0)
    field = direction[:, None] * weight[None, :] + 0.5 * rng.standard_normal((3, points)) * weight[None, :]
    return symmetry.breaking(np.concatenate([np.zeros(points), field.reshape(-1)]))


def sector_state_stability(
    output: Callable[[np.ndarray, np.ndarray], np.ndarray],
    base: np.ndarray,
    symmetry: SectorSymmetry,
    density: np.ndarray,
    *,
    kick: float | None = None,
    steps: int | None = None,
    seed: int = SEED,
    rotations: bool = False,
    weights: np.ndarray | None = None,
    energy_metric: bool = False,
) -> StabilityReport:
    """The largest eigenvalue of the response of one SCF step to perturbations that break ``G`` at the state of the
    input ``base`` (``(V, B)`` as one array, averaged over ``G`` here), by ``steps`` Arnoldi steps (module text).
    ``output(V, B)`` is the ``(V_out, B_out)`` of one SCF step on the whole grid at ``(V, B)`` (``B`` of shape ``(3,
    n)``), ``density`` the density of the state (the weight of the first direction).  ``rotations`` (a loop without
    spin-orbit coupling, whose energy does not change under a rotation of every moment): the rotations of the field
    (:func:`spin_rotation_modes`, eigenvectors of the response with the eigenvalue 1 and no change of the energy in the
    local frame; near 0 in the collinear frame) are taken out of the probes and their responses, so that the Ritz
    values are those of the other eigenvectors.
    ``weights`` (one per grid point): the inner product of the Arnoldi steps, ``sum_r w(r) a(r) b(r)`` over the four
    components (``None``: the plain one).
    With ``energy_metric`` (on request, a diagnostic: it keeps the response of the density of every probe) and an
    ``output`` that returns ``((V_out, B_out), (n, m))`` (the density and magnetization of the step as a second array of
    the same layout), the report also holds the Ritz values in the inner product of the energy
    (:func:`energy_metric_ritz`); the steps and the verdict are the same."""

    started = time.perf_counter()
    kick = KICK if kick is None else float(kick)
    steps = STEPS if steps is None else int(steps)
    points = np.size(density)
    centre = symmetry.symmetric(base)
    metric = None if weights is None else np.tile(np.asarray(weights, dtype=np.float64).reshape(-1), 4)

    def dot(left, right) -> float:
        return float(np.dot(left, right)) if metric is None else float(np.dot(left * metric, right))

    modes = spin_rotation_modes(symmetry, centre, dot) if rotations else []

    def deflated(vector):
        for _ in range(2):
            for mode in modes:
                vector = vector - dot(mode, vector) * mode
        return vector

    start = deflated(first_direction(symmetry, density, seed))
    norm = float(np.sqrt(max(dot(start, start), 0.0)))
    if not norm > 0.0:
        return StabilityReport(float("-inf"), (), (), 0, symmetry.order, time.perf_counter() - started)
    basis = [start / norm]
    hessenberg = np.zeros((steps + 1, steps), dtype=np.float64)
    history: list[float] = []
    ritz = np.zeros(0)
    top = None
    following: list = []
    probes = 0
    converged = False
    # (the energy metric: <b_i, c_j> and <c_i, r_j> of the probes, c_j = -chi_0 b_j the response of the density and
    # magnetization, r_j = J b_j that of the potential and field before the rotations are taken out; A is symmetric, so
    # its upper triangle <c_i, r_j>, i <= j, is formed and mirrored)
    densities: list[np.ndarray] = []
    gram = np.zeros((steps, steps))
    operator = np.zeros((steps, steps))
    energy_history: list[float] = []
    for step in range(steps):
        direction = basis[step]
        size = float(kick) / max(float(np.max(np.abs(direction), initial=0.0)), 1.0e-30)
        probe = centre + size * direction
        result = output(probe[:points], probe[points:].reshape(3, points))
        density_out = None
        if isinstance(result, tuple):
            result, density_out = result
        result = np.asarray(result, dtype=np.float64)
        probes += 1
        response = symmetry.breaking(result) / size
        if energy_metric and density_out is not None and len(densities) == step:
            densities.append(-symmetry.breaking(np.asarray(density_out, dtype=np.float64)) / size)
            for row in range(step + 1):
                gram[step, row] = float(np.dot(basis[step], densities[row]))
                gram[row, step] = float(np.dot(basis[row], densities[step]))
                operator[row, step] = operator[step, row] = float(np.dot(densities[row], response))
            found = energy_metric_ritz(gram[: step + 1, : step + 1], operator[: step + 1, : step + 1])[0]
            energy_history.append(float("nan") if found is None else found)
        response = deflated(response)
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
        top = vectors[:, order[0]].real
        following = [vectors[:, index].real for index in order[1:PATTERN_VECTORS]]
        history.append(float(ritz[0]))
        if length <= 1.0e-12 * max(1.0, float(np.max(np.abs(hessenberg[: step + 1, : step + 1])))):
            converged = True
            break
        if ritz[0] >= CLEAR:
            # (a lower bound of the largest eigenvalue far above the bound: unstable)
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
    scale = 1.0
    if top is not None:
        vector = np.zeros_like(centre)
        for weight, column in zip(top, basis):
            vector += float(weight) * column
        scale = max(float(np.max(np.abs(vector))), 1.0e-300)
        vector /= scale
        for coefficients in following:
            other = np.zeros_like(centre)
            for weight, column in zip(coefficients, basis):
                other += float(weight) * column
            largest_entry = float(np.max(np.abs(other)))
            if largest_entry > 0.0:
                others.append(other / largest_entry)
    energy = (None, None, None)
    if top is not None and densities and len(densities) == probes:
        size = len(top)
        energy = energy_metric_ritz(gram[:size, :size], operator[:size, :size], np.asarray(top) / scale)
    return StabilityReport(
        largest=history[-1] if history else float("-inf"),
        history=tuple(history),
        ritz=tuple(float(value) for value in ritz),
        probes=probes,
        order=symmetry.order,
        seconds=time.perf_counter() - started,
        rotations=len(modes),
        vector=vector,
        centre=centre,
        converged=converged,
        metric_largest=energy[0],
        metric_quotient=energy[1],
        metric_curvature=energy[2],
        metric_history=tuple(energy_history),
        others=tuple(others),
    )


def energy_metric_ritz(gram, operator, coefficients=None):
    """The Rayleigh-Ritz values of the response ``J = K chi_0`` on the span of the probed directions ``b_j`` in the
    inner product of the energy, ``<a, b>_E = <a, -chi_0 b>``, in which ``J`` is symmetric (``K`` and ``chi_0`` are;
    Herbst and Levitt 2022: the Hessian of the free energy of a potential at a self-consistent state is ``-chi_0 (1 -
    J)``): ``A y = theta M y`` with ``gram`` ``M_ij = <b_i, c_j>`` and ``operator`` ``A_ij = <c_i, r_j>``, ``c_j =
    -chi_0 b_j`` (the response of the density and magnetization to ``b_j``, its sign turned) and ``r_j = J b_j``, both
    symmetrized.  These Ritz values are bounded by the largest eigenvalue of ``J`` (those of the plain Arnoldi steps, of
    a ``J`` that is not normal in the plain inner product, need not be).  Returns ``(largest Ritz value, Rayleigh
    quotient of the vector v = sum_j coefficients_j b_j, <v, -chi_0 v>)`` (``None`` for the last two without
    ``coefficients``; all ``None`` where ``M`` has no positive part).  The free energy along ``s v`` changes by ``s^2
    (1 - quotient) curvature / 2`` (times the volume element) to second order."""

    gram = np.asarray(gram, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    operator = np.asarray(operator, dtype=np.float64)
    operator = 0.5 * (operator + operator.T)
    values, vectors = np.linalg.eigh(gram)
    keep = values > 1.0e-10 * max(float(np.max(values, initial=0.0)), 0.0)
    if not np.any(keep):
        return None, None, None
    reduce = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    largest = float(np.max(np.linalg.eigvalsh(reduce.T @ operator @ reduce)))
    if coefficients is None:
        return largest, None, None
    coefficients = np.asarray(coefficients, dtype=np.float64)
    curvature = float(coefficients @ gram @ coefficients)
    quotient = float(coefficients @ operator @ coefficients) / curvature if curvature > 0.0 else None
    return largest, quotient, curvature


@dataclass
class LineReport:
    """What :func:`line_search` found along the top Ritz vector of an unstable state: ``reference`` the free energy at
    the state of the sectors, ``points`` ``(displacement, free energy)`` of every evaluation (the displacement in Ry,
    the largest change of the potential or a component of the field, its sign the side), ``shift`` and ``lowest`` the
    lowest of them, ``seconds`` the time; ``pattern`` the free energy at the input of the flip pattern
    (:func:`flip_pattern`; ``None`` where it turns no atom over), ``flipped`` the atoms it turns over and ``start`` its
    input."""

    reference: float
    points: tuple[tuple[float, float], ...]
    shift: float
    lowest: float
    seconds: float = 0.0
    tolerance: float = GAIN
    pattern: float | None = None
    flipped: int = 0
    start: np.ndarray | None = None
    patterns: int = 0

    @property
    def best(self) -> float:
        """The lowest free energy of the line and of the flip pattern."""

        return float(self.lowest if self.pattern is None else min(self.lowest, self.pattern))

    @property
    def from_pattern(self) -> bool:
        """Whether the flip pattern lies below every point of the line (the continuation then starts from it)."""

        return self.pattern is not None and self.pattern < self.lowest

    @property
    def gain(self) -> float:
        """How far the lowest point of the line or the flip pattern lies below the state of the sectors (Ry)."""

        return float(self.reference - self.best)

    @property
    def pays(self) -> bool:
        """Whether a continuation is worth trying: a gain of :data:`GAIN` or more."""

        return not self.gain < self.tolerance

    def text(self) -> str:
        return (
            f"free energy along the top Ritz vector at most {self.reference - self.lowest:.3e} Ry below the state of "
            f"the sectors (at a largest change of {self.shift:+.4g} Ry; {len(self.points)} points of {LINE_PASSES} "
            "eigensolver steps)"
            + ("" if self.pattern is None else
               f"; of {self.patterns} flip pattern(s) of the soft modes the lowest, the local field of {self.flipped} "
               f"atom(s) turned over, {self.reference - self.pattern:.3e} Ry below ({PATTERN_PASSES} eigensolver steps)")
            + f"; a continuation from {self.tolerance:g} Ry; {self.seconds:.2f} s"
        )


def line_search(free_energy: Callable[[float], float], *, first: float | None = None,
                largest: float | None = None, restart: Callable[[], None] | None = None) -> LineReport:
    """The free energy along the top Ritz vector of an unstable state: ``free_energy(s)`` is that of one SCF step at the
    input ``centre + s v`` (``v`` the Ritz vector with largest component 1: the Mermin free energy of the density
    matrix of that step, an upper bound of the lowest free energy at any ``s``).  It is taken at 0, at ``-first`` and
    ``+first``, and on the lower side at twice the displacement while it goes down, up to ``largest`` (a line search by
    bracketing, as the energy line searches of SCF steps: Herbst and Levitt 2022).  ``restart()`` puts the eigensolver
    back to the states of the centre: before each of the two first displacements and before the doubling on the minus
    side, so that every point starts from the states of the centre or of the point before on its side.  Where the free
    energy goes down by less than :data:`GAIN` along the line, the instability does not pay a continuation (to second
    order the gain of a soft mode goes as ``(lambda - 1)^2``)."""

    started = time.perf_counter()
    first = LINE_FIRST if first is None else float(first)
    largest = LINE_LARGEST if largest is None else float(largest)
    restart = restart if restart is not None else (lambda: None)
    reference = float(free_energy(0.0))
    points = [(0.0, reference)]
    restart()
    minus = float(free_energy(-first))
    points.append((-first, minus))
    restart()
    plus = float(free_energy(first))
    points.append((first, plus))
    sign = 1.0 if plus <= minus else -1.0
    if sign < 0.0:
        restart()
    previous = min(plus, minus)
    shift = first
    while 2.0 * shift <= largest * (1.0 + 1.0e-12):
        shift *= 2.0
        value = float(free_energy(sign * shift))
        points.append((sign * shift, value))
        if not value < previous:
            break
        previous = value
    lowest_shift, lowest = min(points, key=lambda point: point[1])
    return LineReport(reference, tuple(points), float(lowest_shift), float(lowest), time.perf_counter() - started)


class ContinuationWatch:
    """Stops the loop without sectors that continues from an unstable state where it does not pay
    (``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto``): called with every iteration and the free energy of its density
    matrix (an upper bound of the lowest free energy; without a dispersion correction, as ``kept``, the free energy of
    the state of the sectors on that footing), it returns true once :data:`CONTINUATION_WINDOW` iterations have passed
    since the last one :data:`GAIN` below ``kept`` (a loop that goes back to the state of the sectors, or stays near
    it, ends there with that state kept), or, given the residual of every iteration and the ``criterion`` of the loop,
    once the residual has not halved for :data:`STALL_WINDOW` iterations while it is more than :data:`STALL_FACTOR`
    times the criterion (a loop that will not converge, whose state is not taken: the antiparallel state of an Au8
    lattice of 11 bohr with PBE at 5e-6 to 2e-5 Ry for 360 iterations against 1e-8; loops that hover just above the
    criterion, at the noise of a metal, go on: Au8 lattices of 11.5 and 11.7 bohr at 1.2e-8 to 2.6e-8 converged after
    240 and 380 iterations).  ``reason`` says which."""

    def __init__(self, kept: float, tolerance: float | None = None, window: int | None = None,
                 criterion: float | None = None, stall: int | None = None) -> None:
        self.kept = float(kept)
        self.tolerance = GAIN if tolerance is None else float(tolerance)
        self.window = CONTINUATION_WINDOW if window is None else int(window)
        self.criterion = None if criterion is None else float(criterion)
        self.stall = STALL_WINDOW if stall is None else int(stall)
        self.last_below = 0
        self.lowest = float("inf")
        self.mark = float("inf")
        self.halved_at = 0
        self.stopped_at: int | None = None
        self.reason = ""

    def __call__(self, iteration: int, free: float, residual: float | None = None) -> bool:
        self.lowest = min(self.lowest, float(free))
        if float(free) < self.kept - self.tolerance:
            self.last_below = int(iteration)
        if residual is not None and float(residual) < 0.5 * self.mark:
            self.mark = float(residual)
            self.halved_at = int(iteration)
        reason = ""
        if int(iteration) - self.last_below >= self.window:
            reason = (f"in its last {self.window} iterations its free energy was not {self.tolerance:g} Ry below that "
                      "of the sectors")
        elif (
            residual is not None and self.criterion is not None
            and int(iteration) - self.halved_at >= self.stall and self.mark > STALL_FACTOR * self.criterion
        ):
            reason = (f"its residual did not halve in its last {self.stall} iterations, at {self.mark:.2e} more than "
                      f"{STALL_FACTOR:g} times the criterion")
        if reason:
            if self.stopped_at is None:
                self.stopped_at = int(iteration)
                self.reason = reason
            return True
        return False


def nearest_atoms(coordinates, positions, lattice_vectors=None) -> np.ndarray:
    """The index of the nearest atom (over the images in the 26 neighbouring cells of a periodic system) of every grid
    point: the partition of the grid of :func:`flip_pattern`; a tie goes to the first atom.  In an orthorhombic cell
    (every cell the k-point sectors take) the nearest image of each atom comes from rounding along each axis, one
    distance per point and atom; another cell compares the 27 images."""

    from itertools import product

    coordinates = np.asarray(coordinates, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    best = np.full(coordinates.shape[0], np.inf)
    labels = np.zeros(coordinates.shape[0], dtype=np.int64)

    def take(index, distance):
        closer = distance < best
        best[closer] = distance[closer]
        labels[closer] = index

    vectors = None if lattice_vectors is None else np.asarray(lattice_vectors, dtype=np.float64)
    if vectors is not None and np.array_equal(vectors, np.diag(np.diag(vectors))):
        lengths = np.diag(vectors)
        for index, position in enumerate(positions):
            # (the image shift -round((r - R) / L) L of each axis, as the 27 shifts write it)
            shift = -np.round((coordinates - position) / lengths) * lengths
            take(index, np.sum((coordinates + shift - position) ** 2, axis=1))
        return labels
    shifts = [np.zeros(3)]
    if vectors is not None:
        shifts = [np.asarray(cell, dtype=np.float64) @ vectors for cell in product((-1, 0, 1), repeat=3)]
    for index, position in enumerate(positions):
        for shift in shifts:
            take(index, np.sum((coordinates + shift - position) ** 2, axis=1))
    return labels


def flip_pattern(centre: np.ndarray, vector: np.ndarray, labels: np.ndarray, atoms: int):
    """A finite-amplitude start from an unstable state (``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto``): the input
    ``centre`` (``(V, B)``) with the field turned over, ``B -> -B``, on the grid points (``labels``: the nearest atom of
    each point) of one of the two groups of atoms into which a Ritz vector ``vector`` splits them.  The change of the
    local field of atom ``a`` is ``b_a`` (the field part of the vector summed over its points); the groups are the
    signs of ``b_a . e`` along the principal axis ``e`` of the ``b_a``, taken where that component is at least
    :data:`PATTERN_ALIGNMENT` of ``|b_a|`` and ``|b_a|`` and the local field at least :data:`PATTERN_MOMENT` of their
    largest (the other atoms keep their field), and the smaller group (else the one against ``e``) turns over.  A soft
    mode that cants the moments of two sublattices against each other (Au8 lattices with moments along z: staggered
    changes in the plane normal to them) or changes their sizes against each other thus becomes the antiparallel start
    along the moments of the sectors, at their sizes (the stability analysis of quantum chemistry follows an
    instability to a new start in the same way).  ``(input, turned atoms)``, or ``None`` where the changes do not split
    the atoms with a field into two groups (a global rotation, a single atom)."""

    centre = np.asarray(centre, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    points = centre.size // 4
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    field = centre[points:].reshape(3, points)
    change = vector[points:].reshape(3, points)
    local = np.stack([np.bincount(labels, weights=field[j], minlength=atoms) for j in range(3)], axis=1)
    along = np.stack([np.bincount(labels, weights=change[j], minlength=atoms) for j in range(3)], axis=1)
    size = np.linalg.norm(local, axis=1)
    reach = np.linalg.norm(along, axis=1)
    if not float(np.max(size, initial=0.0)) > 0.0 or not float(np.max(reach, initial=0.0)) > 0.0:
        return None
    axis = np.linalg.eigh(along.T @ along)[1][:, -1]
    component = along @ axis
    counted = (
        (size >= PATTERN_MOMENT * float(np.max(size)))
        & (reach >= PATTERN_MOMENT * float(np.max(reach)))
        & (np.abs(component) >= PATTERN_ALIGNMENT * reach)
    )
    plus = counted & (component > 0.0)
    minus = counted & (component < 0.0)
    if not np.any(plus) or not np.any(minus):
        return None
    flipped = plus if np.count_nonzero(plus) < np.count_nonzero(minus) else minus
    signs = np.where(flipped, -1.0, 1.0)[labels]
    start = centre.copy()
    start[points:] = (field * signs[None, :]).reshape(-1)
    return start, flipped


def spin_rotation_modes(symmetry: SectorSymmetry, centre: np.ndarray, dot=None) -> list[np.ndarray]:
    """The parts that break ``G`` of the infinitesimal rotations of the field of ``centre`` (``(V, B)``), ``e_j x
    B(r)`` with no change of ``V`` for the three axes ``j``, orthonormal (those that vanish left out).  Without
    spin-orbit coupling and in the local frame of the exchange-correlation field one SCF step turns with the field,
    ``F(R B) = R F(B)``, so these are eigenvectors of the response with the eigenvalue 1 along which the energy does
    not change (the moments of a magnetic state turn freely).  In the collinear frame of a gradient functional
    (``PARSEC_NONCOLLINEAR_GGA_FRAME``) the output field stays along the axis of the initial moments, so ``F(R B)
    != R F(B)`` and a turned field has a response near 0: taking these directions out removes stable ones only."""

    if dot is None:
        def dot(left, right):
            return float(np.dot(left, right))
    centre = np.asarray(centre, dtype=np.float64)
    points = centre.size // 4
    field = centre[points:].reshape(3, points)
    scale = float(np.sqrt(max(dot(centre, centre), 0.0)))
    modes: list[np.ndarray] = []
    if not scale > 0.0:
        return modes
    for axis in range(3):
        turned = np.cross(np.eye(3)[axis][:, None], field, axis=0)
        mode = symmetry.breaking(np.concatenate([np.zeros(points), turned.reshape(-1)]))
        for _ in range(2):
            for other in modes:
                mode = mode - dot(other, mode) * other
        length = float(np.sqrt(max(dot(mode, mode), 0.0)))
        if length > 1.0e-8 * scale:
            modes.append(mode / length)
    return modes


__all__ = [
    "BOUND",
    "CLEAR",
    "CONTINUATION_DEFAULT",
    "CONTINUATION_SWITCH",
    "CONTINUATION_WINDOW",
    "CONVERGENCE",
    "ContinuationWatch",
    "DEFAULT",
    "DISPLACEMENT",
    "GAIN",
    "KICK",
    "LINE_FIRST",
    "LINE_LARGEST",
    "LINE_PASSES",
    "LineReport",
    "MINIMUM_STEPS",
    "NEAR",
    "PASSES",
    "PATTERN_ALIGNMENT",
    "PATTERN_MOMENT",
    "PATTERN_PASSES",
    "PATTERN_VECTORS",
    "STALL_FACTOR",
    "STALL_WINDOW",
    "STEPS",
    "SWITCH",
    "SectorSymmetry",
    "WINDOW",
    "StabilityReport",
    "energy_metric_ritz",
    "first_direction",
    "flip_pattern",
    "line_search",
    "nearest_atoms",
    "noncollinear_sector_continuation_setting",
    "noncollinear_sector_stability_setting",
    "sector_state_stability",
    "spin_rotation_modes",
]
