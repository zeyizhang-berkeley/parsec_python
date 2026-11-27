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
:data:`STEPS` (the report says it did not converge).  The bound is 1 and the accuracy of one eigensolver step per probe.  An unstable state
(:func:`parsec_python.SCF.noncollinear.run_self_consistent_noncollinear_kpoints`): the loop without sectors continues from
the input of the sectors displaced along the Ritz vector of the largest Ritz value (largest change
:data:`DISPLACEMENT`), the direction in which the free energy goes down, and the lower of its state and that of the
sectors is the result (a loop that starts again from the start reaches the same symmetric state in the tests, where
nothing but the noise of its eigensolver breaks the symmetry: O2 at 3000 to 6000 K, Au8 lattices of 10.5 to 12 bohr).
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
#: without sectors continues after an unstable state.
DISPLACEMENT = 5.0e-2
#: The seed of the random part of the first direction.
SEED = 20261125


def noncollinear_sector_stability_setting() -> str:
    """``PARSEC_NONCOLLINEAR_SECTOR_STABILITY``: ``auto`` (the default) tests the state of a converged non-collinear
    loop in the sectors of its k-points against perturbations that break their operations (this module) in place of
    tests 1 and 3 of the guard (``PARSEC_KPOINT_SECTOR_GUARD``: partly filled multiplets across the sectors at the start
    and at the end; a loop in sectors that did not converge is kept untested, with a note); ``off`` keeps tests 1 and
    3 and no stability test, bit for bit as before.  Another value is an error before the SCF."""

    text = os.environ.get(SWITCH, DEFAULT).strip().lower() or DEFAULT
    if text not in _SETTINGS:
        raise ValueError(f"{SWITCH} must be auto or off, not {text!r}")
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

    @property
    def unstable(self) -> bool:
        return not self.largest < self.bound

    def text(self) -> str:
        return (
            f"largest eigenvalue {self.largest:.4f} of the SCF response to perturbations that break the {self.order} "
            f"operations of the sectors (unstable from {self.bound:g}; {self.probes} probes of {self.passes} "
            "eigensolver steps on the whole grid, Ritz values "
            + " ".join(f"{value:.3f}" for value in self.ritz[:4])
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
    components (``None``: the plain one)."""

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
    probes = 0
    converged = False
    for step in range(steps):
        direction = basis[step]
        size = float(kick) / max(float(np.max(np.abs(direction), initial=0.0)), 1.0e-30)
        probe = centre + size * direction
        result = np.asarray(output(probe[:points], probe[points:].reshape(3, points)), dtype=np.float64)
        probes += 1
        response = deflated(symmetry.breaking(result) / size)
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
    if top is not None:
        vector = np.zeros_like(centre)
        for weight, column in zip(top, basis):
            vector += float(weight) * column
        vector /= max(float(np.max(np.abs(vector))), 1.0e-300)
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
    )


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
    "CONVERGENCE",
    "DEFAULT",
    "DISPLACEMENT",
    "KICK",
    "MINIMUM_STEPS",
    "NEAR",
    "PASSES",
    "STEPS",
    "SWITCH",
    "SectorSymmetry",
    "WINDOW",
    "StabilityReport",
    "first_direction",
    "noncollinear_sector_stability_setting",
    "sector_state_stability",
    "spin_rotation_modes",
]
