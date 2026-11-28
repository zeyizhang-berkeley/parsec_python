"""Self-consistent non-collinear magnetism (optionally with spin-orbit coupling).

The spinor wavefunctions ``psi = (psi_up, psi_down)`` give a charge density
``n = |psi_up|^2 + |psi_down|^2`` and a magnetization vector

``m = psi^dagger sigma psi = (2 Re(psi_up^* psi_down),
                              2 Im(psi_up^* psi_down),
                              |psi_up|^2 - |psi_down|^2)``

(summed over occupied states).  The exchange-correlation potential is
``V_avg * 1 + B_xc . sigma`` with the local-rotation form of
:mod:`parsec_python.V_xc.noncollinear`, so the Hamiltonian acting on each
spinor is

``H = [T + V_ion + V_H + V_avg + V_NL] * 1 + B_xc . sigma (+ L.S terms)``.

``V_avg`` and the three components of ``B_xc`` are mixed together as one
vector.  The direction of ``m`` is free to vary in space and to rotate during
the SCF; the initial moments are seeded per atom from
``Atom.initial_moment`` (Fortran's ``Initial_NCL_Moment``) with magnitude set
by the species' ``initial_spin_polarization``.  Without spin-orbit coupling
the energy is invariant under a global rotation of all moments, which the
tests use as an independent check.

:func:`run_self_consistent_noncollinear` handles an isolated system;
:func:`run_self_consistent_noncollinear_kpoints` a periodic cell sampled on a
k-point set (magnetism breaks time-reversal symmetry, so ``k`` and ``-k``
must both appear: Fortran's reduced grid uses no time reversal here,
``irrbz.f90``); unequal weights weight every level by its k-point
(:mod:`~parsec_python.SCF.kpoint_weights`).  The magnetization is a vector
field, which ``Symmetrize_Charge_Density`` averages by PARSEC's rule only
(:func:`symmetrize_noncollinear`: the density and ``|m|`` as scalars, the
direction kept).  With the accelerated package the k-points of a periodic
cell can split into the sectors of the magnetic group of the initial
moments (``PARSEC_NONCOLLINEAR_KPOINT_SECTORS``, opt-in,
:mod:`parsec_python.acceleration.SCF.kpoint_spinor_sectors`), with the
symmetry guard of :func:`run_self_consistent_noncollinear_kpoints`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Callable

import numpy as np

from ..Eigensolvers import EigvalState, solve_eigval
from ..Energy import total_energy_noncollinear
from ..Hamiltonian.spinor_operator import SpinorKohnShamHamiltonian
from ..Mixer import make_mixer, potential_residual_metrics
from ..V_ion import superpose_atomic_density
from ..models import SelfConsistentSOCIteration, SelfConsistentSOCResult
from .kpoint_solver import KPointJob, kept_vectors, make_kpoint_solver, split_pooled
from .kpoint_weights import KPointWeights
from .orbital_density import spinor_sums
from .single_point import first_diagonalization_tolerance, next_diagonalization_tolerance
from .stage_timer import StageTimer, solver_residual_max, solver_timings
from .filter_degree import LaterFilterDegree
from ..Hartree.variational import electrostatic_hartree
from .kpoints_soc import (
    SpinorForceExtras,
    _eigval_settings,
    _k_point_operators,
    _number_of_states,
    _result_weights,
    _sector_geometry,
    _spinor_hamiltonian,
    _state_sz,
    _validated_k_points,
    export_subspaces,
    import_subspaces,
)


def spinor_density_and_magnetization(
    spinors: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    weight: float = 1.0,
    xp=np,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, m)`` from stacked spinors ``(2*n_grid, n_states)``.

    ``n`` has shape ``(n_grid,)`` and ``m`` has shape ``(3, n_grid)``;
    ``weight`` is the k-point weight of these states.  ``xp`` is the array
    namespace the spinors live in (a device block gives device arrays); a block that holds only some
    rows of the grid gives those rows.
    """

    occupations = np.asarray(occupations, dtype=float)
    if spinors.ndim != 2 or spinors.shape[0] % 2:
        raise ValueError("spinors must have shape (2*n_grid, n_states)")
    if occupations.shape != (spinors.shape[1],):
        raise ValueError("occupation count does not match the spinor columns")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    scale = weight / volume_element
    up_sum, down_sum, cross_sum = spinor_sums(spinors, occupations, xp)
    rows = up_sum.shape[0]
    density = scale * (up_sum + down_sum)
    magnetization = xp.empty((3, rows))
    magnetization[0] = scale * (2.0 * cross_sum.real)
    magnetization[1] = scale * (2.0 * cross_sum.imag)
    magnetization[2] = scale * (up_sum - down_sum)
    return density, magnetization


@dataclass(frozen=True)
class NoncollinearMoments:
    """The magnetization diagnostics of ``bxc.F90`` (``bxc`` and ``nclmom``).

    ``absolute`` is ``Integral{|m|} = h^3 sum_i |m_i|``, ``atomic`` the
    ``(atoms, 3)`` sums ``h^3 sum m_i`` over the grid points within the
    ``Atomic_Radius`` of each atom (the Wigner-Seitz lists ``wsindw`` of
    ``nonloc.F90``, with the periodic images of a point in the neighbouring
    cells) and ``total`` the ``h^3 sum_i m_i`` of all points, all in Bohr
    magnetons.
    """

    absolute: float
    atomic: np.ndarray
    total: np.ndarray


def atomic_sphere_masks(
    coordinates: np.ndarray,
    atoms,
    radii,
    lattice_vectors: np.ndarray | None = None,
) -> np.ndarray:
    """``(atoms, points)``: the grid points within ``radii[a]`` of atom ``a``.

    A point is inside where it or one of its images in the 26 neighbouring
    cells (rows of ``lattice_vectors``, for a periodic system) is at most
    the radius from the nucleus (``nonloc.F90``: ``rr <= p_pot%rws``).
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    shifts = [np.zeros(3)]
    if lattice_vectors is not None:
        vectors = np.asarray(lattice_vectors, dtype=np.float64)
        shifts = [
            np.asarray(cell, dtype=np.float64) @ vectors
            for cell in product((-1, 0, 1), repeat=3)
        ]
    masks = np.zeros((len(atoms), coordinates.shape[0]), dtype=bool)
    for index, (atom, radius) in enumerate(zip(atoms, radii)):
        for shift in shifts:
            distance = np.linalg.norm(
                coordinates + shift - np.asarray(atom.position), axis=1
            )
            masks[index] |= distance <= float(radius)
    return masks


def noncollinear_moments(
    magnetization: np.ndarray, masks: np.ndarray, volume_element: float
) -> NoncollinearMoments:
    """The moments of a magnetization density ``(3, points)`` in mu_B/bohr^3."""

    magnetization = np.asarray(magnetization, dtype=np.float64)
    weight = float(volume_element)
    return NoncollinearMoments(
        absolute=weight * float(np.sum(np.linalg.norm(magnetization, axis=0))),
        atomic=weight * (np.asarray(masks, dtype=np.float64) @ magnetization.T),
        total=weight * np.sum(magnetization, axis=1),
    )


def _initial_magnetization(system) -> np.ndarray:
    """Per-atom seed ``m(r) = sum_a p_a * d_a * rho_a(r)`` (shape ``(3, n_grid)``).

    ``rho_a`` is atom ``a``'s own superposed atomic density, ``d_a`` its unit
    initial-moment direction and ``p_a`` the species' ``initial_spin_polarization``;
    the sum is scaled by the same factor that normalized the total initial
    density, so ``|m| <= n`` holds wherever ``p_a <= 1``.
    """

    periodic_cell = getattr(system.input, "periodic_cell", None)
    lattice_vectors = None if periodic_cell is None else periodic_cell.lattice_vectors
    specifications = system.input.pseudopotentials
    per_atom = []
    for atom in system.atoms:
        density_a = superpose_atomic_density(
            system.grid,
            [atom],
            system.pseudopotentials,
            specifications,
            lattice_vectors,
        )
        per_atom.append(density_a)
    unnormalized_charge = float(sum(np.sum(rho) for rho in per_atom))
    scale = (
        float(np.sum(system.initial_density)) / unnormalized_charge
        if unnormalized_charge > 0.0
        else 0.0
    )
    magnetization = np.zeros((3, system.grid.size))
    for atom, density_a in zip(system.atoms, per_atom):
        if atom.initial_moment is None:
            continue
        polarization = specifications[atom.symbol].initial_spin_polarization
        magnetization += (
            scale * polarization * np.asarray(atom.initial_moment)[:, None] * density_a
        )
    return magnetization


def symmetrize_noncollinear(density, magnetization, group, volume_element):
    """``Symmetrize_Charge_Density`` of a non-collinear density (PARSEC's rule).

    ``newrho.F90`` hands ``symm_scalar`` the total density and the densities
    of the local frame ``(rho +- |m|)/2`` (``isp = 1, 2, 3``); the direction
    of the magnetization (``spin3d`` of ``bxc.F90``) is not averaged.  This is
    the same: ``rho`` and ``|m|`` are averaged as scalars and ``m`` keeps its
    direction, ``m <- <|m|> m/|m|`` (zero where ``|m| = 0``).  Without
    spin-orbit coupling a spatial operation does not act on the spin, so the
    scalar average is the symmetry of the problem; with it the operations act
    on ``m`` as on an axial vector and the symmetry of a magnetic state is a
    subgroup, which neither this nor a full-group axial average respects
    (PARSEC's choice is kept).  Returns the density, the magnetization and
    the three reports.
    """

    from ..Grid.point_group import symmetrize_density

    magnetization = np.asarray(magnetization, dtype=np.float64)
    length = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    new_density, total_report = symmetrize_density(density, group, volume_element)
    _, up_report = symmetrize_density(0.5 * (density + length), group, volume_element)
    _, down_report = symmetrize_density(0.5 * (density - length), group, volume_element)
    new_length, _ = symmetrize_density(length, group, volume_element)
    scale = np.divide(new_length, length, out=np.zeros_like(length), where=length > 0.0)
    return new_density, magnetization * scale[None, :], (total_report, up_report, down_report)


class NoncollinearSectorsAbandoned(RuntimeError):
    """The non-collinear loop in the sectors of its k-points failed a test of their guard (``PARSEC_KPOINT_SECTOR_GUARD
    =auto``): a density or magnetization that breaks an operation they take (test 2), a state that is unstable against
    perturbations that break them (``PARSEC_NONCOLLINEAR_SECTOR_STABILITY=auto``, ``stage`` "the stability test after
    the SCF loop": ``continuation``, the input from which the loop without sectors continues, and ``kept``, the result
    of the sectors, of which the lower state is kept), or under its ``off`` a partly filled
    multiplet across the sectors or Kramers partners of a k-point in the start Hamiltonian (test 1, ``stage`` "the start
    Hamiltonian") or a fractional occupation of such a multiplet at the end (test 3; :meth:`parsec_python.acceleration.
    SCF.kpoint_spinor_sectors.SpinorSectorKPointSolver.noncollinear_multiplets`, ``noncollinear_asymmetries``):
    :func:`run_self_consistent_noncollinear_kpoints` runs it again with one task per k-point.  ``report`` is a
    ``FractionalOccupations`` of :mod:`parsec_python.acceleration.SCF.spin_sectors`, ``measured`` the values of the
    timings that the loop without sectors then carries."""

    def __init__(self, report, measured: dict, stage: str = "the SCF loop") -> None:
        self.report = report
        self.measured = dict(measured)
        self.stage = stage
        super().__init__(
            f"non-collinear k-point sectors: {report.reason} after {stage} ({report.measure} = {report.largest:.3e} "
            f">= {report.tolerance:.0e}): " + "; ".join(report.lines)
        )


def _multiplet_guard(solver, levels, occupations, weights, *, end: bool, measured: dict) -> dict:
    """Test 1 (``end`` false: the levels of the start Hamiltonian) or test 3 (``end``: Fermi-Dirac occupations at the
    end of the loop) of the guard of a non-collinear loop in sectors: the largest value of
    ``SpinorSectorKPointSolver.noncollinear_multiplets`` as ``noncollinear_guard_start_share`` or
    ``noncollinear_guard_fractional_share`` (nothing where no k-point is in sectors); above
    :data:`..acceleration.SCF.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE` it raises
    :class:`NoncollinearSectorsAbandoned` under ``PARSEC_KPOINT_SECTOR_GUARD=auto`` (``off`` measures only)."""

    measure = getattr(solver, "noncollinear_multiplets", None)
    if not callable(measure):
        return {}
    found = measure(levels, occupations, weights, end=end)
    from ..acceleration.SCF.kpoint_sectors import sector_guard_setting
    from ..acceleration.SCF.spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, FractionalOccupations

    largest = max((value for value, _ in found), default=0.0)
    key = "noncollinear_guard_fractional_share" if end else "noncollinear_guard_start_share"
    values = {key: float(largest)}
    trips = [line for value, line in found if (value >= SPIN_SECTOR_OCCUPATION_TOLERANCE if end
                                               else value > SPIN_SECTOR_OCCUPATION_TOLERANCE)]
    if trips and sector_guard_setting() == "auto":
        report = FractionalOccupations(
            largest=float(largest),
            tolerance=SPIN_SECTOR_OCCUPATION_TOLERANCE,
            lines=tuple(trips),
            reason="a fractional occupation of a multiplet across the non-collinear k-point sectors" if end
            else "a partly filled multiplet across the non-collinear k-point sectors",
            measure="largest w_k min(f, 1-f)" if end else "largest w_k min(n, d - n) of d levels holding n electrons",
        )
        raise NoncollinearSectorsAbandoned(
            report, {**measured, **values}, "the SCF loop" if end else "the start Hamiltonian"
        )
    return values


def _sector_guard(solver, density, magnetization, volume_element: float, measured: dict | None = None) -> dict:
    """Test 2 of the guard of a non-collinear loop in the sectors of its k-points: the largest asymmetry of its final
    density and magnetization under the operations of the sectors (``noncollinear_guard_asymmetry`` of the timings,
    empty where no k-point is in sectors); at :data:`..acceleration.SCF.spin_sectors.DENSITY_SYMMETRY_TOLERANCE` or
    above it raises :class:`NoncollinearSectorsAbandoned` (``PARSEC_KPOINT_SECTOR_GUARD=auto``; ``off`` keeps the
    result and only measures)."""

    measure = getattr(solver, "noncollinear_asymmetries", None)
    if not callable(measure):
        return dict(measured or {})
    found = measure(density, magnetization, volume_element)
    if not found:
        return dict(measured or {})
    from ..acceleration.SCF.kpoint_sectors import sector_guard_setting
    from ..acceleration.SCF.spin_sectors import DENSITY_SYMMETRY_TOLERANCE, FractionalOccupations

    largest = max(value for _, value in found)
    values = {**(measured or {}), "noncollinear_guard_asymmetry": float(largest)}
    if sector_guard_setting() == "auto" and not largest < DENSITY_SYMMETRY_TOLERANCE:
        report = FractionalOccupations(
            largest=float(largest),
            tolerance=DENSITY_SYMMETRY_TOLERANCE,
            lines=tuple(
                f"{label}: {value:.4f} from its image" for label, value in found
                if not value < DENSITY_SYMMETRY_TOLERANCE
            ),
            reason="a density or magnetization that breaks the symmetry the non-collinear k-point sectors assume",
            measure="largest int |n(g r) - n(r)| [e] or int sum_j |m_j(g r) - (R' m(r))_j| [mu_B]",
        )
        raise NoncollinearSectorsAbandoned(report, values)
    return values


def _run_noncollinear(
    system,
    hamiltonian_for_k: Callable[[int, np.ndarray, np.ndarray], object],
    n_kpoints: int,
    weighting: KPointWeights,
    callback: Callable[[SelfConsistentSOCIteration], None] | None,
    k_points: np.ndarray | None = None,
    keep_wavefunctions: bool = False,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    extras: SpinorForceExtras | None = None,
    warm_start=None,
    allow_sectors: bool = False,
    start_states: dict | None = None,
    give_up: Callable[[int, float], bool] | None = None,
) -> SelfConsistentSOCResult:
    # ``start_states``: Ritz states by task that the solver starts from (the continuation of an unstable state of the
    # sectors); ``give_up(iteration, free energy, residual)``: true stops the loop unconverged (PARSEC_NONCOLLINEAR_
    # SECTOR_CONTINUATION=auto, acceleration.SCF.noncollinear_stability.ContinuationWatch)
    settings = system.input.scf
    n_grid = system.grid.size
    volume_element = system.grid.volume_element
    number_of_states = _number_of_states(system)
    ionic_potential = system.ionic_potential
    density_symmetry = getattr(system, "density_symmetry", None)

    timer = StageTimer()
    density = system.initial_density
    if warm_start is None:
        with timer.stage("initial_magnetization"):
            magnetization = _initial_magnetization(system)
    if initial_potential is None and warm_start is None:
        with timer.stage("setup_potentials"):
            hartree = system.solve_hartree(density, initial_potential=-ionic_potential)
            hartree_potential = hartree.potential
            xc = system.evaluate_xc_noncollinear(density, magnetization)
        input_potential = ionic_potential + hartree_potential + xc.potential
        input_field = xc.field
    else:
        if warm_start is not None:
            # Andrew Grieder's ionic loops: the structure before
            # the exchange-correlation field of the previous structure carries the moments to the moved atoms
            hartree_potential = np.array(warm_start.hartree_potential, dtype=np.float64)
            input_potential, input_field = (np.array(part, dtype=np.float64) for part in warm_start.input_potential)
        else:
            # The warm start of an ionic step (parsec.F90 after domove: vnew with
            # the ionic part replaced, the magnetization direction kept): the
            # scalar potential and the exchange-correlation field of the first
            # eigenproblem; -V_ion starts the first Poisson solve, as a restart.
            hartree_potential = -np.asarray(ionic_potential, dtype=np.float64)
            input_potential = np.array(initial_potential[0], dtype=np.float64)
            input_field = np.array(initial_potential[1], dtype=np.float64).reshape(3, n_grid)
            if input_potential.shape != np.shape(ionic_potential):
                raise ValueError("initial_potential must hold one value per grid point")

    # one step's fields from its density and magnetization, shared by the loop and the stability test of its sectors
    def symmetrized(new_density, new_magnetization):
        """``Symmetrize_Charge_Density`` of the new density and magnetization (PARSEC's rule), with its reports."""

        if density_symmetry is None:
            return new_density, new_magnetization, ()
        return symmetrize_noncollinear(new_density, new_magnetization, density_symmetry, volume_element)

    def output_fields(new_density, new_magnetization, new_hartree):
        """``(xc, V_out, B_out)`` of the density, magnetization and Hartree potential of a step."""

        new_xc = system.evaluate_xc_noncollinear(new_density, new_magnetization)
        return new_xc, ionic_potential + new_hartree + new_xc.potential, new_xc.field

    def accumulate(task, vectors, occupations, xp=np):
        density_k, magnetization_k = spinor_density_and_magnetization(
            vectors, occupations, volume_element, weighting.of_k(task), xp
        )
        return {"density": density_k, "magnetization": magnetization_k}

    # the sectors of the magnetic group of the initial moments (PARSEC_NONCOLLINEAR_KPOINT_SECTORS; None under off or
    # for the loop the guard sends back to one task per k-point: the job of before)
    geometry = None
    if allow_sectors and k_points is not None:
        geometry = _sector_geometry(
            system,
            k_points,
            weighting,
            bool(extras is not None and extras.compute_forces),
            noncollinear=True,
            field=input_field,
        )
    job = KPointJob(
        n_kpoints,
        number_of_states,
        lambda degree, tolerance=None: _eigval_settings(system, degree, tolerance),
        lambda task, potentials: hamiltonian_for_k(
            task, potentials["V"], potentials["field"]
        ),
        accumulate,
        extras=extras if extras is not None else (lambda task, vectors, xp=np: _state_sz([vectors], xp)),
        geometry=geometry,
    )
    solver = make_kpoint_solver(job, problem_size=system.grid.size)
    # PARSEC_NONCOLLINEAR_SECTOR_STABILITY=auto (a loop in sectors under PARSEC_KPOINT_SECTOR_GUARD=auto): the stability
    # test of the converged state in place of tests 1 and 3 of the guard
    stability = _stability_wanted(solver)
    if extras is not None and extras.compute_forces and type(solver).__name__ == "MPIKPointSolver":
        raise NotImplementedError(
            "the nonlocal force of the row-distributed MPI k-point solver is not implemented: its extras are "
            "sums over row blocks, which does not apply to a product of overlaps"
        )
    mixer = make_mixer(system.input.mixing)
    filter_degree = system.input.eigensolver.filter_degree
    minimum_filter_degree = max(10, system.input.eigensolver.filter_degree_delta + 1)
    degree_schedule = LaterFilterDegree.for_loop(filter_degree, minimum_filter_degree)
    # Dynamic_Diag_Tol: 0.1 for the first iteration (parsec.F90:1106-1111).
    dynamic_tolerance = system.input.eigensolver.dynamic_tolerance
    diagonalization_tolerance = first_diagonalization_tolerance(system.input.eigensolver)

    pooled_eigenvalues = np.empty(0)
    pooled_occupations = np.empty(0)
    fermi_level = float("nan")
    energies = None
    converged = False
    specifications = system.input.pseudopotentials
    # the guard of the sectors (PARSEC_KPOINT_SECTOR_GUARD): test 1 on the levels of a start of the loop's own (not a
    # warm start or given potentials), the measured values in the timings
    guard_start = warm_start is None and initial_potential is None
    guard_values: dict = {}
    k_weights = [weighting.of_k(task) for task in range(n_kpoints)]
    periodic_cell = getattr(system.input, "periodic_cell", None)
    sphere_masks = atomic_sphere_masks(
        system.grid.coordinates,
        system.atoms,
        [getattr(specifications[atom.symbol], "atomic_radius", 0.0) for atom in system.atoms],
        None if periodic_cell is None else periodic_cell.lattice_vectors,
    )
    stopped = 0
    with solver:
        import_subspaces(solver, warm_start)
        if start_states and getattr(solver, "import_states", None) is not None:
            solver.import_states(start_states)
            # (the solver holds them now; the caller's dict lets them go with the solver's first solve)
            start_states.clear()
        start_states = None
        timer.lap()  # what came before the loop (potentials, solver start-up) is not part of iteration 1
        for iteration in range(1, settings.max_iterations + 1):
            used_potential = input_potential  # what enters this eigenproblem (the mixed one replaces it below)
            used_field = input_field

            with timer.stage("diagonalization"):
                eigenvalues_by_k = solver.solve(
                    {"V": input_potential, "field": input_field},
                    filter_degree,
                    diagonalization_tolerance if dynamic_tolerance else None,
                )
            residual_max = solver_residual_max(solver)

            with timer.stage("density"):
                stacked = np.concatenate(eigenvalues_by_k)
                state_weights = (
                    None
                    if weighting.uniform
                    else weighting.state_weights(eigenvalues_by_k, range(n_kpoints))
                )
                stacked_occupations, fermi_level = weighting.occupy(
                    stacked,
                    system.electron_count,
                    settings.fermi_temperature_kelvin,
                    1.0,
                    state_weights,
                )
                pooled_eigenvalues = stacked
                pooled_occupations = stacked_occupations
                if iteration == 1 and guard_start and not stability:
                    # test 1 of the guard of the sectors: no partly filled multiplet across the sectors or Kramers
                    # partners of a k-point in the start Hamiltonian (before the loop spends its iterations on it)
                    guard_values.update(
                        _multiplet_guard(
                            solver, eigenvalues_by_k, split_pooled(stacked_occupations, eigenvalues_by_k),
                            k_weights, end=False, measured=guard_values,
                        )
                    )

                parts = solver.accumulate(split_pooled(stacked_occupations, eigenvalues_by_k))
                density, magnetization, symmetrization = symmetrized(parts["density"], parts["magnetization"])

            with timer.stage("hartree"):
                hartree = system.solve_hartree(density, initial_potential=hartree_potential)
                hartree_potential = hartree.potential
            with timer.stage("xc"):
                xc, output_potential, output_field = output_fields(density, magnetization, hartree_potential)

            input_combined = np.concatenate([input_potential, input_field.reshape(-1)])
            output_combined = np.concatenate([output_potential, output_field.reshape(-1)])
            with timer.stage("mixing_energy"):
                metrics = potential_residual_metrics(
                    input_combined,
                    output_combined,
                    np.tile(density, 4),
                    volume_element,
                    system.electron_count,
                )
                energy_occupations, band_energy_weight = weighting.band_energy_terms(
                    pooled_occupations, state_weights
                )
                energies = total_energy_noncollinear(
                    pooled_eigenvalues,
                    energy_occupations,
                    density,
                    magnetization,
                    input_potential,
                    input_field,
                    ionic_potential,
                    hartree_potential,
                    xc.potential,
                    xc.field,
                    xc.total_energy,
                    system.ion_ion_energy,
                    volume_element,
                    alpha_z_energy=system.alpha_z_energy,
                    band_energy_weight=band_energy_weight,
                )

                mixed = mixer.mix(input_combined, output_combined, iteration=iteration)
            lap = timer.lap()
            selected_residual = (
                metrics.plain if settings.use_plain_residual else metrics.weighted
            )
            if dynamic_tolerance:
                # parsec.F90:1302-1311 with this loop's SRE.
                diagonalization_tolerance = next_diagonalization_tolerance(
                    system.input.eigensolver, diagonalization_tolerance, (metrics.weighted,)
                )
            if callback is not None:
                first_k = eigenvalues_by_k[0].size
                callback(
                    SelfConsistentSOCIteration(
                        iteration=iteration,
                        # PARSEC_FILTER_DEGREE_STALL: the NOTE of a restoration (once)
                        filter_degree_note=degree_schedule.take_note(),
                        weighted_residual=metrics.weighted,
                        plain_residual=metrics.plain,
                        energies=energies,
                        eigenvalues=tuple(float(v) for v in eigenvalues_by_k[0]),
                        occupations=tuple(float(v) for v in stacked_occupations[:first_k]),
                        fermi_level=float(fermi_level),
                        eigen_residual_max=residual_max,
                        diagonalization_seconds=lap.get("diagonalization", 0.0),
                        total_seconds=sum(lap.values()),
                        magnetic_moments=noncollinear_moments(
                            magnetization, sphere_masks, volume_element
                        ),
                        density_symmetrization=symmetrization,
                        **(
                            {"next_eigensolver_tolerance": diagonalization_tolerance}
                            if dynamic_tolerance
                            else {}
                        ),
                    )
                )
            if iteration_state_callback is not None:
                # Save_Intermediate_*: the local-frame fields of the final
                # parsec.dat after mixing; the spinors of an isolated loop.
                from .single_point import SpinorIterationFields

                iteration_state_callback(
                    SpinorIterationFields(
                        iteration=iteration,
                        eigenvalues=pooled_eigenvalues,
                        occupations=pooled_occupations,
                        # (the MPI solver gathers no spinors: an empty block, which
                        # parsec.dat then writes without wavefunction records)
                        spinors=None if k_points is not None else solver.vectors(0),
                        parsec_dat_fields=_noncollinear_parsec_dat_fields(
                            system,
                            mixed[:n_grid],
                            mixed[n_grid:].reshape(3, n_grid),
                            density,
                            magnetization,
                        ),
                        k_points=k_points,
                        k_weights=(
                            None if k_points is None else _result_weights(weighting, n_kpoints)
                        ),
                    )
                )
            last_input_potential = input_potential
            input_potential = mixed[:n_grid]
            input_field = mixed[n_grid:].reshape(3, n_grid)
            # PARSEC_FILTER_DEGREE_STALL (SCF/filter_degree.py): the degree lowered near convergence, and the
            # starting degree back where the loop stalls after that
            filter_degree = degree_schedule.after_iteration(
                iteration, metrics.weighted, settings.approach_criterion, settings.degree_decrease_after
            )
            if selected_residual < settings.convergence_criterion:
                converged = True
                break
            if give_up is not None and give_up(iteration, _free_energy(SimpleNamespace(
                energies=energies, occupations=pooled_occupations, electron_count=system.electron_count,
                k_weights=None if k_points is None else _result_weights(weighting, n_kpoints),
            ), settings.fermi_temperature_kelvin), selected_residual):
                stopped = iteration
                break

        if settings.fermi_temperature_kelvin > 0.0 and not stability:
            # test 3: no fractional occupation of such a multiplet at the end (Fermi-Dirac occupations)
            guard_values.update(
                _multiplet_guard(
                    solver, eigenvalues_by_k, split_pooled(pooled_occupations, eigenvalues_by_k), k_weights,
                    end=True, measured=guard_values,
                )
            )
        guard_values = _sector_guard(solver, density, magnetization, volume_element, guard_values)
        stability_start = None
        if stability and converged:
            import time

            working = number_of_states + int(_eigval_settings(system, filter_degree).safety_buffer)
            unfolded = time.perf_counter()
            stability_start = (
                solver.whole_states(working),
                solver.stability_operations(),
                np.concatenate([np.asarray(used_potential, dtype=np.float64),
                                np.asarray(used_field, dtype=np.float64).reshape(-1)]),
                time.perf_counter() - unfolded,
            )
        moment_vector = volume_element * np.sum(magnetization, axis=1)
        from .single_point import dispersion_after_convergence, dispersion_wanted
        dispersion = None
        if dispersion_wanted(settings, converged):
            # efvdw.f90 after a converged loop: the TS correction of the density.
            dispersion, energies = dispersion_after_convergence(system, density, energies)
        subspaces = export_subspaces(solver)
        nonlocal_forces = None
        state_moments = None
        if k_points is not None:
            if extras is not None and extras.compute_forces:
                # the eigenvectors and occupations of the last step belong together
                extras.set_occupations(split_pooled(pooled_occupations, eigenvalues_by_k))
            if extras is not None:
                state_moments, nonlocal_forces = extras.collect(solver)
            else:
                state_moments = np.concatenate(solver.extras())
        timings = solver_timings(timer, solver)
        timings.update(guard_values)
        if stopped:
            timings["noncollinear_continuation_stopped"] = float(stopped)
        frame = getattr(system, "noncollinear_frame", None)
        if getattr(frame, "left_axis_at", None) is not None:
            # PARSEC_NONCOLLINEAR_GGA_FRAME: the guard of the axis frame took the local frame at that evaluation
            timings["noncollinear_frame_left_axis"] = float(frame.left_axis_at)
        result = SelfConsistentSOCResult(
            converged=converged,
            iterations=iteration,
            atoms=system.atoms,
            electron_count=system.electron_count,
            eigenvalues=pooled_eigenvalues,
            occupations=pooled_occupations,
            spinors=solver.vectors(0),
            fermi_level=fermi_level,
            density=density,
            energies=energies,
            magnetic_moment=float(np.linalg.norm(moment_vector)),
            magnetic_moment_vector=moment_vector,
            magnetization=magnetization,
            k_points=k_points,
            k_weights=None if k_points is None else _result_weights(weighting, n_kpoints),
            state_moments=state_moments,
            timings=timings,
            scf_potentials={
                "effective_potential": output_potential,
                "xc_field": output_field,
            },
            # The scalar parts the forces read (parsec_python.Forces.spin_orbit).
            scf_input_potential=last_input_potential,
            scf_output_potential=output_potential,
            xc_potential=xc.potential,
            k_spinors=(
                tuple(
                    kept_vectors(solver, task, "the non-collinear forces")
                    for task in range(n_kpoints)
                )
                if keep_wavefunctions and k_points is not None
                else None
            ),
            dispersion=dispersion,
            parsec_dat_fields=_noncollinear_parsec_dat_fields(
                system, input_potential, input_field, density, magnetization
            ),
            # pot%vnew and the field: where the next structure of an ionic
            # loop starts.
            next_potential=(input_potential, input_field),
            input_effective_potential=used_potential,
            output_effective_potential=output_potential,
            nonlocal_forces=nonlocal_forces,
            next_effective_potential=(input_potential, input_field),
            hartree_potential=electrostatic_hartree(system, hartree_potential),
            eigensolver_state=subspaces,
        )
    if stability_start is None:
        if stability and callback is not None:
            # (a state that is not self-consistent cannot be tested: kept as the loop left it, with test 2)
            from ..models import SpinSectorFallbackRecord

            callback(
                SpinSectorFallbackRecord(
                    stage="the SCF loop",
                    largest_fractional_occupation=float(iteration),
                    tolerance=float(settings.max_iterations),
                    multiplets=("the stability test takes a self-consistent state; a lower state that breaks the "
                                "operations of the sectors may exist",),
                    reason="the loop in the sectors did not converge; its state is kept untested",
                    measure="iterations",
                    route="kpoint_stable",
                )
            )
        if stability:
            result.timings["noncollinear_stability_untested"] = 1.0
        return result

    def step_output(step_solver, potential, field, passes: int, energy: bool = False):
        """``((V_out, B_out), (n, m), free energy)`` of one step of the loop at ``(potential, field)`` with
        ``step_solver`` (the stability test; ``passes`` eigensolver steps from the states it holds), each pair as one
        array; the free energy (that of the density matrix of the step, :func:`_free_energy`) only with ``energy``,
        else ``None``."""

        for _ in range(passes):
            levels = step_solver.solve(
                {"V": potential, "field": field}, filter_degree,
                diagonalization_tolerance if dynamic_tolerance else None,
            )
        level_weights = None if weighting.uniform else weighting.state_weights(levels, range(n_kpoints))
        filled, _ = weighting.occupy(
            np.concatenate(levels), system.electron_count, settings.fermi_temperature_kelvin, 1.0, level_weights
        )
        step_parts = step_solver.accumulate(split_pooled(filled, levels))
        step_density, step_magnetization, _ = symmetrized(step_parts["density"], step_parts["magnetization"])
        step_hartree = system.solve_hartree(step_density, initial_potential=hartree_potential)
        step_xc, step_potential, step_field = output_fields(step_density, step_magnetization, step_hartree.potential)
        free = None
        if energy:
            energy_occupations, band_energy_weight = weighting.band_energy_terms(filled, level_weights)
            step_energies = total_energy_noncollinear(
                np.concatenate(levels), energy_occupations, step_density, step_magnetization, potential,
                np.asarray(field, dtype=np.float64).reshape(3, -1), ionic_potential, step_hartree.potential,
                step_xc.potential, step_xc.field, step_xc.total_energy, system.ion_ion_energy, volume_element,
                alpha_z_energy=system.alpha_z_energy, band_energy_weight=band_energy_weight,
            )
            free = _free_energy(SimpleNamespace(
                energies=step_energies, occupations=filled, electron_count=system.electron_count,
                k_weights=None if k_points is None else _result_weights(weighting, n_kpoints),
            ), settings.fermi_temperature_kelvin)
        return (
            np.concatenate([step_potential, np.asarray(step_field, dtype=np.float64).reshape(-1)]),
            np.concatenate([np.asarray(step_density, dtype=np.float64).reshape(-1),
                            np.asarray(step_magnetization, dtype=np.float64).reshape(-1)]),
            free,
        )

    # (without spin-orbit coupling the moments turn freely: those rotations are no instability)
    spin_orbit = any(
        getattr(specification, "spin_orbit", False) for specification in system.input.pseudopotentials.values()
    )
    from ..acceleration.SCF.noncollinear_stability import noncollinear_sector_continuation_setting

    continuation = noncollinear_sector_continuation_setting() == "auto"
    labels = None
    if continuation:
        from ..acceleration.SCF.noncollinear_stability import nearest_atoms

        # (the nearest atom of every grid point, made only for an unstable state: the partition of its flip pattern)
        def labels():
            return nearest_atoms(
                system.grid.coordinates, [atom.position for atom in system.atoms],
                None if periodic_cell is None else periodic_cell.lattice_vectors,
            )
    values, report, line, kept_states = _state_stability(
        job, stability_start, step_output, density, system.grid.size, guard_values, not spin_orbit, continuation,
        labels,
    )
    result.timings.update(values)
    if report.unstable and line is not None and not line.pays:
        # PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto: no point of the line search along the top Ritz vector lies
        # GAIN below the state of the sectors: a continuation would not pay, the state of the sectors is kept
        del kept_states
        result.timings["noncollinear_sector_abandoned"] = 0.0
        result.timings["noncollinear_continuation_skipped"] = 1.0
        if callback is not None:
            from ..models import SpinSectorFallbackRecord

            callback(
                SpinSectorFallbackRecord(
                    stage="the line search along the direction of the instability",
                    largest_fractional_occupation=float(line.gain),
                    tolerance=float(line.tolerance),
                    multiplets=(report.text(), line.text()),
                    reason=(
                        "the state of the sectors is kept: its free energy goes down by less than "
                        f"{line.tolerance:g} Ry along the direction of the instability, which does not pay a loop "
                        "without sectors"
                    ),
                    measure="free energy below that of the sectors along the top Ritz vector [Ry]",
                    route="kpoint_continued",
                )
            )
        return result
    if report.unstable:
        raise _unstable(report, values, guard_values, result, line, kept_states)
    del kept_states
    if callback is not None:
        from ..models import SpinSectorFallbackRecord

        lines = [report.text()]
        if report.largest >= 1.0:
            lines.append(
                f"(an eigenvalue of {report.largest:.4f}, from 1 to the bound of the accuracy of the test: a lower state "
                "that breaks the operations of the sectors may exist, as the loop without sectors would find it only "
                "through the noise of its eigensolver)"
            )
        if not report.converged:
            lines.append("(the largest Ritz value had not converged within the probes: a lower bound)")
        callback(
            SpinSectorFallbackRecord(
                stage="the stability test after the SCF loop",
                largest_fractional_occupation=float(report.largest),
                tolerance=float(report.bound),
                multiplets=tuple(lines),
                reason="the state of the sectors is stable against perturbations that break their operations",
                measure="largest eigenvalue of the response",
                route="kpoint_stable",
            )
        )
    return result


def _stability_wanted(solver) -> bool:
    """Whether the loop of ``solver`` takes the stability test of its state (``PARSEC_NONCOLLINEAR_SECTOR_STABILITY=
    auto`` and ``PARSEC_KPOINT_SECTOR_GUARD=auto`` for a loop with k-points in the sectors of a non-collinear loop) in
    place of tests 1 and 3 of the guard."""

    operations = getattr(solver, "stability_operations", None)
    if not callable(operations) or operations() is None:
        return False
    from ..acceleration.SCF.kpoint_sectors import sector_guard_setting
    from ..acceleration.SCF.noncollinear_stability import noncollinear_sector_stability_setting

    return sector_guard_setting() == "auto" and noncollinear_sector_stability_setting() == "auto"


def _state_stability(job, start, step_output, density, problem_size: int, measured: dict, rotations: bool,
                     continuation: bool = False, labels=None):
    """The stability test of the converged state of a non-collinear loop in sectors (:mod:`parsec_python.acceleration.
    SCF.noncollinear_stability`) with a solver of one task per k-point that starts from the Ritz states of the sectors
    (``start``: those states, the operations of the sectors and the input ``(V, B)`` of the last step); the timings of
    the test, its report, the line search and the states for a continuation (``None`` both but where ``continuation``,
    ``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto``, and the state is unstable: the line search of the free energy along
    the top Ritz vector, :func:`..acceleration.SCF.noncollinear_stability.line_search`, with the same solver, and the
    Ritz states of the sectors on the whole grid, by task).  ``rotations``: a loop without spin-orbit coupling, whose
    rotations of the moments the test leaves out.  The solver of the test is that of the loop without sectors (one task
    per k-point), which keeps its subspaces on the host where the device is short (``PARSEC_KPOINT_STATE_STORAGE``)."""

    import time
    from dataclasses import replace

    from ..acceleration.SCF import kpoint_spinor_sectors as sectors
    from ..acceleration.SCF.noncollinear_stability import (
        LINE_PASSES,
        PASSES,
        PATTERN_PASSES,
        SectorSymmetry,
        flip_pattern,
        line_search,
        sector_state_stability,
    )
    from ..acceleration.SCF.spin_sectors import FractionalOccupations
    from .kpoint_solver import start_kpoint_setup_notes, stop_kpoint_setup_notes

    started = time.perf_counter()
    states, (cell, operations), base, unfolded = start
    # the device memory of the loop in the sectors back to the devices before the solver of the whole grid
    sectors.release_device_pools()
    token = start_kpoint_setup_notes(lambda _text: None)  # (the line of the solver of the test is not the run's)
    try:
        probe_solver = make_kpoint_solver(replace(job, geometry=None), problem_size=problem_size)
    finally:
        stop_kpoint_setup_notes(token)
    probes: list[float] = []
    with probe_solver:
        if len(states) < job.n_tasks:
            # a k-point without Ritz states of the sectors: its subspace solved at the state first, then kept
            probe_solver.import_states({task: replace(state) for task, state in states.items()})
            points = np.size(density)
            step_output(probe_solver, base[:points], base[points:].reshape(3, points), 4)
            states = dict(probe_solver.export_states())

        set_up = time.perf_counter() - started

        def output(potential, field):
            probed = time.perf_counter()
            probe_solver.import_states({task: replace(state) for task, state in states.items()})
            found, found_density, _ = step_output(probe_solver, potential, field, PASSES)
            probes.append(time.perf_counter() - probed)
            return found, found_density

        report = sector_state_stability(output, base, SectorSymmetry(cell, operations), density, rotations=rotations)
        report.seconds = time.perf_counter() - started + unfolded
        line = None
        if continuation and report.unstable and report.vector is not None and report.centre is not None:
            # the free energy along the top Ritz vector, each point from the states of the centre or of the point
            # before on its side
            def restart():
                probe_solver.import_states({task: replace(state) for task, state in states.items()})

            restart()
            centre = np.asarray(report.centre, dtype=np.float64)
            direction = np.asarray(report.vector, dtype=np.float64)
            points = np.size(density)

            def free_energy(shift):
                displaced = centre + float(shift) * direction
                return step_output(
                    probe_solver, displaced[:points], displaced[points:].reshape(3, points), LINE_PASSES, energy=True
                )[2]

            line = line_search(free_energy, restart=restart)
            patterns = []
            if labels is not None:
                partition = labels()
                for candidate in (direction, *report.others):
                    found_pattern = flip_pattern(centre, candidate, partition, int(np.max(partition)) + 1)
                    if found_pattern is not None and not any(
                        np.array_equal(found_pattern[1], other[1]) for other in patterns
                    ):
                        patterns.append(found_pattern)
            pattern_states = None
            started_pattern = time.perf_counter()
            for start_point, flipped in patterns:
                # the finite-amplitude starts: the local fields of the atoms that a soft mode lowers turned over (a new
                # moment pattern), each from the states of the centre with more eigensolver steps; the lowest is kept
                restart()
                found = step_output(
                    probe_solver, start_point[:points], start_point[points:].reshape(3, points), PATTERN_PASSES,
                    energy=True,
                )[2]
                if line.pattern is None or found < line.pattern:
                    line = replace(line, pattern=float(found), flipped=int(np.count_nonzero(flipped)),
                                   start=start_point)
                    if line.from_pattern and line.pays:
                        # (the continuation starts from the states of that pattern)
                        pattern_states = dict(probe_solver.export_states())
            if patterns:
                line = replace(line, seconds=line.seconds + time.perf_counter() - started_pattern,
                               patterns=len(patterns))
    kept_states = None
    if line is not None and line.pays:
        kept_states = pattern_states if pattern_states is not None else {
            task: replace(state) for task, state in states.items()
        }
    del states
    sectors.release_device_pools()
    report.passes = PASSES
    values = {
        "noncollinear_stability_largest": float(report.largest),
        "noncollinear_stability_probes": float(report.probes),
        "noncollinear_stability_seconds": float(report.seconds),
        # (where the time goes: the states of the sectors unfolded, the solver made, the first probe, the others)
        "noncollinear_stability_unfold_seconds": float(unfolded),
        "noncollinear_stability_setup_seconds": float(set_up),
        "noncollinear_stability_first_probe_seconds": float(probes[0]) if probes else 0.0,
        "noncollinear_stability_later_probe_seconds": float(sum(probes[1:])),
    }
    if report.metric_largest is not None:
        # (the Ritz values of the response in the inner product of the energy: bounded by its largest eigenvalue)
        values["noncollinear_stability_metric_largest"] = float(report.metric_largest)
    if line is not None:
        values.update({
            "noncollinear_line_gain": float(line.reference - line.lowest),
            "noncollinear_line_shift": float(line.shift),
            "noncollinear_line_points": float(len(line.points)),
            "noncollinear_line_seconds": float(line.seconds),
        })
        if line.pattern is not None:
            values["noncollinear_pattern_gain"] = float(line.reference - line.pattern)
            values["noncollinear_pattern_flipped"] = float(line.flipped)
    return values, report, line, kept_states


def _unstable(report, values: dict, measured: dict, kept, line=None, states=None) -> "NoncollinearSectorsAbandoned":
    """What a loop in sectors whose state the stability test found unstable raises: the loop without sectors continues
    from the input of its last step displaced along the Ritz vector of the largest eigenvalue, and the lower of its state
    and ``kept`` (the result of the sectors) is the result.  The displacement is the lowest point of the line search
    ``line`` (``PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto``; the loop then starts from ``states``, the Ritz states of
    the sectors on the whole grid) or :data:`..acceleration.SCF.noncollinear_stability.DISPLACEMENT` (``off``, or no
    line search)."""

    from ..acceleration.SCF.noncollinear_stability import DISPLACEMENT
    from ..acceleration.SCF.spin_sectors import FractionalOccupations

    shift = DISPLACEMENT if line is None else float(line.shift)
    continuation = None
    if line is not None and line.from_pattern:
        displaced = np.asarray(line.start, dtype=np.float64)
        points = displaced.size // 4
        continuation = (displaced[:points].copy(), displaced[points:].reshape(3, points).copy())
    elif report.vector is not None and report.centre is not None:
        displaced = np.asarray(report.centre, dtype=np.float64) + shift * np.asarray(report.vector)
        points = displaced.size // 4
        continuation = (displaced[:points].copy(), displaced[points:].reshape(3, points).copy())
    if line is None:
        how = [
            "the loop without sectors continues from the state of the sectors displaced along the Ritz vector of "
            f"that eigenvalue (largest change {DISPLACEMENT:g} Ry); the lower of the two states is kept",
        ]
    elif line.from_pattern:
        how = [
            line.text(),
            f"the loop without sectors starts again from the state of the sectors with the local field of {line.flipped} "
            "atom(s) turned over (the moment pattern of the instability) and the states of that pattern; the lower of "
            "the two states is kept",
        ]
    else:
        how = [
            line.text(),
            "the loop without sectors continues from the lowest point of that line with the Ritz states of the sectors "
            f"(largest change {shift:+.4g} Ry); the lower of the two states is kept",
        ]
    error = NoncollinearSectorsAbandoned(
        FractionalOccupations(
            largest=float(report.largest),
            tolerance=float(report.bound),
            lines=(report.text(), *how),
            reason=(
                "a symmetric state of the non-collinear k-point sectors that is unstable against perturbations that "
                "break their operations (an eigenvalue of the SCF response above 1: the free energy goes down along it)"
            ),
            measure="largest eigenvalue of the response",
        ),
        {**measured, **values},
        "the stability test after the SCF loop",
    )
    error.continuation = continuation
    error.kept = kept
    error.line = line
    error.states = states
    return error


def _dispersion_energy(result) -> float:
    """The dispersion correction (Van_Der_Waals) in the total energy of a result, 0 where it has none (a loop that did
    not converge)."""

    return float(getattr(result.energies, "dispersion", 0.0) or 0.0)


def _free_energy(result, temperature: float) -> float:
    """``E - TS`` of a non-collinear result (the functional the loop makes stationary at a Fermi temperature)."""

    energy = float(result.energies.total)
    if not temperature > 0.0:
        return energy
    occupations = np.asarray(result.occupations, dtype=np.float64).reshape(-1)
    weights = np.asarray(result.k_weights if result.k_weights is not None else np.ones(1), dtype=np.float64)
    rows = occupations.reshape(weights.size, -1) if occupations.size % weights.size == 0 else occupations[None, :]
    if rows.shape[0] != weights.size:
        weights = np.ones(1)
    clipped = np.clip(rows, 1.0e-300, 1.0 - 1.0e-16)
    entropy = clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    entropy = np.where((rows <= 0.0) | (rows >= 1.0), 0.0, entropy)
    filled = float(np.dot(weights, rows.sum(axis=1)))
    factor = float(result.electron_count) / filled if filled > 0.0 else 0.0
    return energy + factor * _KELVIN_RY * float(temperature) * float(np.dot(weights, entropy.sum(axis=1)))


#: Boltzmann's constant in Ry/K.
_KELVIN_RY = 6.33327186e-06
#: The free energy (Ry) by which the state the loop without sectors reaches from the displaced state must lie below
#: that of the sectors to replace it (the noise of converged energies is far smaller; chemical accuracy far larger).
CONTINUATION_MARGIN = 1.0e-6


def _noncollinear_parsec_dat_fields(system, potential, field, density, magnetization) -> dict:
    """``vnew(:,1:2)`` and ``rho(:,2:3)`` of a non-collinear run (``wfnsave.F90``).

    PARSEC stores the local-frame quantities: the potentials of the spin
    along and against the local magnetization, ``V +- |B|``, and the
    densities ``(rho +- |m|)/2`` (``newrho.F90``); the direction of ``m``
    (``spin3d``) is not written.  ``stype(2)`` is 4 with spin-orbit species,
    3 without.
    """

    potential = np.asarray(potential, dtype=np.float64)
    strength = np.sqrt(np.sum(np.asarray(field, dtype=np.float64) ** 2, axis=0))
    length = np.sqrt(np.sum(np.asarray(magnetization, dtype=np.float64) ** 2, axis=0))
    spin_orbit = any(
        getattr(specification, "spin_orbit", False)
        for specification in system.input.pseudopotentials.values()
    )
    return {
        "kind": 4 if spin_orbit else 3,
        "potentials": (potential + strength, potential - strength),
        "densities": (0.5 * (density + length), 0.5 * (density - length)),
    }


def run_self_consistent_noncollinear(
    system,
    soc_projectors: tuple = (),
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for an isolated system.

    ``warm_start`` (:class:`~parsec_python.SCF.single_point.SCFWarmStart`, whose ``input_potential`` is the pair
    ``(V, B_xc)``) continues from the loop of the previous structure of a relaxation or dynamics run.

    ``soc_projectors`` is
    :func:`~parsec_python.Eigensolvers.perturbative_soc.build_spin_orbit_projectors`'s
    output; leave it empty for non-collinear magnetism without spin-orbit.
    ``initial_potential`` is ``(V, B)``, the scalar potential and the
    exchange-correlation field (shape ``(3, n_grid)``) of the first
    eigenproblem in place of those of the initial density and moments (the
    warm start of an ionic step, ``result.next_potential`` of the structure
    before with its ionic part replaced).
    ``iteration_state_callback`` receives a
    :class:`parsec_python.SCF.single_point.SpinorIterationFields` after every
    iteration (``Save_Intermediate_*``).
    """
    from .single_point import refuse_hubbard_and_dispersion

    refuse_hubbard_and_dispersion(
        system, "run_self_consistent_noncollinear", dispersion=False
    )

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear requires SCFSettings.noncollinear=True"
        )
    soc_projectors = tuple(soc_projectors)

    def hamiltonian_for_k(_k_index, potential, field):
        scalar = system.hamiltonian(potential)
        return SpinorKohnShamHamiltonian(
            scalar.negative_laplacian,
            scalar.effective_potential,
            scalar.nonlocal_operator,
            soc_projectors,
            xc_field=field,
        )

    return _run_noncollinear(
        system,
        hamiltonian_for_k,
        1,
        KPointWeights(weights=np.ones(1), uniform=True, weight=1.0),
        callback,
        iteration_state_callback=iteration_state_callback,
        initial_potential=initial_potential,
        warm_start=warm_start,
    )


def run_self_consistent_noncollinear_kpoints(
    system,
    k_points: np.ndarray,
    weights: np.ndarray,
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    keep_wavefunctions: bool = False,
    iteration_state_callback: Callable[[object], None] | None = None,
    initial_potential: tuple[np.ndarray, np.ndarray] | None = None,
    compute_forces: bool = False,
    warm_start=None,
) -> SelfConsistentSOCResult:
    """Non-collinear SCF for a periodic cell on an unreduced uniform k grid.

    ``warm_start`` continues from the loop of the previous structure, as for the isolated driver.

    Spin-orbit projectors are built for every k-point from the species with
    ``SO_PSP`` (none gives plain non-collinear magnetism).
    ``keep_wavefunctions`` keeps the spinors of every k-point
    (``k_spinors``, for the forces); ``initial_potential`` is ``(V, B)`` as
    in :func:`run_self_consistent_noncollinear`; ``iteration_state_callback``
    receives a :class:`parsec_python.SCF.single_point.SpinorIterationFields`
    after every iteration (``Save_Intermediate_*``).
    ``compute_forces``
    also returns the nonlocal force (``result.nonlocal_forces``), accumulated from the orbitals of every k-point by
    a last pass after the SCF loop (see :class:`~parsec_python.SCF.kpoints_soc.SpinorForceExtras`); ``warm_start``
    continues from the structure before (Andrew Grieder's ionic loops).

    With the accelerated package and ``PARSEC_NONCOLLINEAR_KPOINT_SECTORS=auto`` or ``on`` (opt-in) the k-points split
    into the sectors of the magnetic group of the initial moments (:mod:`parsec_python.acceleration.SCF.
    kpoint_spinor_sectors`; the operations the start field breaks are left out).  Where such a loop fails a test of
    the guard (``PARSEC_KPOINT_SECTOR_GUARD=auto``: a final density or magnetization that breaks an operation of the
    sectors; under its ``off`` a partly filled multiplet across the sectors or Kramers partners of a k-point in the
    start Hamiltonian, after its first solve, or a fractional occupation of such a multiplet at the end), the loop runs
    again from the same start with one task per k-point; under ``PARSEC_NONCOLLINEAR_SECTOR_STABILITY=auto`` a final
    state that is unstable against perturbations that break them (:mod:`parsec_python.acceleration.SCF.
    noncollinear_stability`) makes the loop without sectors continue from that state displaced along the direction the
    free energy goes down, and the lower of the two states is the result (its free energy at least
    :data:`CONTINUATION_MARGIN` below; a NOTE says which).  ``callback`` first receives a
    :class:`parsec_python.models.SpinSectorFallbackRecord` (the note of ``parsec.out``) and the result's ``timings``
    hold ``noncollinear_sector_abandoned`` = 1; ``noncollinear_guard_asymmetry`` is what test 2 measured,
    ``noncollinear_stability_largest`` the largest eigenvalue of the response that the stability test found.
    """
    from .single_point import refuse_hubbard_and_dispersion

    refuse_hubbard_and_dispersion(
        system, "run_self_consistent_noncollinear_kpoints", dispersion=False
    )

    if not system.input.scf.noncollinear:
        raise ValueError(
            "run_self_consistent_noncollinear_kpoints requires "
            "SCFSettings.noncollinear=True"
        )
    k_points, weighting = _validated_k_points(
        k_points, weights, "run_self_consistent_noncollinear_kpoints"
    )
    operator_cache: dict[int, tuple] = {}

    def operators_for(k_index):
        if k_index not in operator_cache:
            operator_cache[k_index] = _k_point_operators(
                system, k_points[k_index : k_index + 1]
            )[0]
        return operator_cache[k_index]

    def hamiltonian_for_k(k_index, potential, field):
        nonlocal_operator_k, soc_projectors_k = operators_for(k_index)
        return _spinor_hamiltonian(
            system,
            k_points[k_index],
            nonlocal_operator_k,
            soc_projectors_k,
            potential,
            xc_field=field,
        )

    if compute_forces and not weighting.uniform:
        raise NotImplementedError(
            "the nonlocal force of a non-collinear k-point run accumulated in the SCF step (compute_forces, "
            "Andrew Grieder's ionic loops) needs equal k-point weights"
        )

    def run(allow_sectors: bool, start=None, start_states=None, give_up=None) -> SelfConsistentSOCResult:
        extras = SpinorForceExtras(
            system, k_points, weighting.weight if weighting.uniform else 1.0, operators_for, compute_forces
        )
        return _run_noncollinear(
            system,
            hamiltonian_for_k,
            k_points.shape[0],
            weighting,
            callback,
            k_points,
            keep_wavefunctions=keep_wavefunctions,
            iteration_state_callback=iteration_state_callback,
            initial_potential=initial_potential if start is None else start,
            extras=extras,
            warm_start=warm_start if start is None else None,
            allow_sectors=allow_sectors,
            start_states=start_states,
            give_up=give_up,
        )

    try:
        return run(True)
    except NoncollinearSectorsAbandoned as error:
        # only the report, and the result of an unstable state, leave this block: the frames of the loop in the
        # sectors, and their orbitals, go
        abandoned = (error.report, error.measured, error.stage, getattr(error, "continuation", None),
                     getattr(error, "kept", None), getattr(error, "line", None), getattr(error, "states", None))
        error.states = None
    report, measured, stage, continuation, kept, line, start_states = abandoned
    abandoned = None
    try:
        # the device memory of the loop in the sectors, which its arrays returned to the CuPy pool, back to the devices
        # before the loop with one task per k-point
        from ..acceleration.SCF.kpoint_spinor_sectors import release_device_pools
    except ImportError:
        pass
    else:
        release_device_pools()
    if callback is not None:
        from ..models import SpinSectorFallbackRecord

        callback(
            SpinSectorFallbackRecord(
                stage=stage,
                largest_fractional_occupation=report.largest,
                tolerance=report.tolerance,
                multiplets=tuple(report.lines),
                reason=report.reason,
                measure=report.measure,
                route="kpoint",
            )
        )
    if continuation is None:
        result = run(False)
        result.timings["noncollinear_sector_abandoned"] = 1.0
        result.timings.update(measured)
        return result
    # an unstable state of the sectors: the loop without sectors continues from it, displaced along the direction the
    # free energy goes down; the lower of the two states is the result
    temperature = float(system.input.scf.fermi_temperature_kelvin)
    watch = None
    if line is not None:
        # PARSEC_NONCOLLINEAR_SECTOR_CONTINUATION=auto: from the flip pattern or the lowest point of the line search,
        # stopped where its free energy stays within GAIN of the state of the sectors or above it, or where its residual
        # stalls far above the criterion; its iterations carry no dispersion correction, so the state of the sectors is compared without it
        from ..acceleration.SCF.noncollinear_stability import ContinuationWatch

        watch = ContinuationWatch(_free_energy(kept, temperature) - _dispersion_energy(kept),
                                  criterion=float(system.input.scf.convergence_criterion))
    continued = run(False, start=continuation, start_states=start_states, give_up=watch)
    start_states = None
    below = _free_energy(kept, temperature) - _free_energy(continued, temperature)
    chosen = continued if continued.converged and below > CONTINUATION_MARGIN else kept
    if (_dispersion_energy(kept) == 0.0) != (_dispersion_energy(continued) == 0.0):
        # (an unconverged loop carries no dispersion correction: both without it)
        below = (_free_energy(kept, temperature) - _dispersion_energy(kept)) - (
            _free_energy(continued, temperature) - _dispersion_energy(continued)
        )
    chosen.timings["noncollinear_sector_abandoned"] = 1.0 if chosen is continued else 0.0
    chosen.timings.update(measured)
    chosen.timings["noncollinear_continuation_iterations"] = float(continued.iterations)
    chosen.timings["noncollinear_continuation_lower"] = float(below)
    stopped = watch is not None and watch.stopped_at is not None
    if stopped:
        chosen.timings["noncollinear_continuation_stopped"] = float(watch.stopped_at)
    if callback is not None:
        from ..models import SpinSectorFallbackRecord

        lines = [
            f"its free energy {below:+.6e} Ry below that of the sectors after {continued.iterations} iterations"
            + ("" if continued.converged else " (not converged)")
        ]
        if stopped:
            lines.append(f"(stopped: {watch.reason}; its lowest {watch.kept - watch.lowest:+.3e} Ry below)")
        callback(
            SpinSectorFallbackRecord(
                stage="the loop without sectors from the displaced state",
                largest_fractional_occupation=float(below),
                tolerance=float(CONTINUATION_MARGIN),
                multiplets=tuple(lines),
                reason=(
                    "the state of the loop without sectors is kept: it lies below the unstable state of the sectors"
                    if chosen is continued
                    else "the state of the sectors is kept: the loop without sectors from the displaced state did not "
                    "converge"
                    if watch is not None and below > CONTINUATION_MARGIN
                    else "the state of the sectors is kept: the loop without sectors from the displaced state did not "
                    "end below it"
                ),
                measure="free energy below that of the sectors [Ry]",
                route="kpoint_continued",
            )
        )
    return chosen


__all__ = [
    "NoncollinearMoments",
    "NoncollinearSectorsAbandoned",
    "atomic_sphere_masks",
    "noncollinear_moments",
    "run_self_consistent_noncollinear",
    "run_self_consistent_noncollinear_kpoints",
    "spinor_density_and_magnetization",
    "symmetrize_noncollinear",
]
