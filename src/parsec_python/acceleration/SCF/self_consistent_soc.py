"""Self-consistent spin-orbit coupling of an isolated system on the GPU (Gamma point).

The reference loop of :mod:`parsec_python.SCF.self_consistent_soc` with three
device pieces plugged in:

* the spinor Hamiltonian is the device operator of
  :mod:`parsec_python.acceleration.Hamiltonian.device_bloch` (real sparse
  products on the real view of the complex block, the spin-orbit term of all
  atoms as two sparse products), with its static matrices uploaded once and only
  the local potential uploaded per SCF iteration;
* the eigensolver is :func:`~parsec_python.acceleration.Eigensolvers.chebff_spinor.solve_spinor_eigval`:
  CHEBFF in the first iteration and one SUBSPACE step in every later one, as the
  reference ``solve_eigval`` does, with the orbitals resident on the device;
* the density is formed on the device from those orbitals.

Hartree, exchange-correlation, mixing and the energies stay on the host exactly
as in the reference, as in :mod:`parsec_python.acceleration.SCF.single_point`.
The orbitals come to the host once, at the end.

``backend`` selects the array backend: ``None`` is the current CUDA device through
CuPy; the tests pass a :class:`NumpyArrayBackend` to run this code on the host.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from parsec_python.Eigensolvers.perturbative_soc import AtomSpinOrbitProjectors
from parsec_python.SCF.self_consistent_soc import (
    run_self_consistent_soc as run_reference_self_consistent_soc,
)
from parsec_python.models import SelfConsistentSOCIteration, SelfConsistentSOCResult

from ..Eigensolvers.chebff_spinor import solve_spinor_eigval, spinor_density_on_device
from ..Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
from .single_point import AcceleratedPreparedSinglePointSystem


class _HostScalar:
    """Host-side description of the scalar Gamma-point Hamiltonian, as the converter reads it."""

    _k_gradient = None

    def __init__(self, negative_laplacian, nonlocal_operator, potential) -> None:
        self.negative_laplacian = negative_laplacian
        self.nonlocal_operator = nonlocal_operator
        self._diagonal = potential


class _HostSpinor:
    xc_delta = None
    xc_field = None

    def __init__(self, scalar: _HostScalar, soc_projectors) -> None:
        self.scalar_hamiltonian = scalar
        self.soc_projectors = soc_projectors


def _host_parts(scalar_hamiltonian):
    """Kinetic matrix, nonlocal operator and local potential of a bound scalar Hamiltonian."""
    backend = getattr(scalar_hamiltonian, "backend", None)
    if backend is not None and hasattr(backend, "host_negative_laplacian"):
        return (
            backend.host_negative_laplacian,
            backend.host_nonlocal_operator,
            np.asarray(backend.local_potential, dtype=np.float64),
        )
    return (
        scalar_hamiltonian.negative_laplacian,
        scalar_hamiltonian.nonlocal_operator,
        np.asarray(scalar_hamiltonian.effective_potential, dtype=np.float64),
    )


class DeviceSpinorHamiltonianBuilder:
    """``spinor_hamiltonian_builder`` hook: the device operator of one SCF iteration.

    The converter caches the uploaded Laplacian, projectors and spin-orbit
    matrices by the identity of the host objects, which persist across the
    iterations, so an iteration uploads only its local potential.
    """

    def __init__(self, backend) -> None:
        self.converter = DeviceBlochConverter(backend)

    def __call__(self, scalar_hamiltonian, soc_projectors):
        laplacian, nonlocal_operator, potential = _host_parts(scalar_hamiltonian)
        return self.converter.operator(
            _HostSpinor(_HostScalar(laplacian, nonlocal_operator, potential), soc_projectors)
        )


class _DeviceKeeper:
    """What a loop of an ionic step keeps on its device for the forces and the next structure.

    It wraps the eigensolver and ``to_host`` hooks of the reference loop: the last eigensolver state (the subspace
    the next structure filters again, :func:`parsec_python.SCF.single_point.state_for_new_potential`) and the final
    device spinors (a view of that subspace, or the :class:`.soc_sectors.SectorSpinors` of the sectors), which the
    device forces read (:mod:`..Forces.spinor_device`).
    """

    def __init__(self, device: int, accepts) -> None:
        self.device = int(device)
        self.accepts = accepts
        self.state = None
        self.spinors = None

    def solver(self, solve):
        def wrapped(operator, requested_states, *, settings, state=None):
            if state is not None and not self.accepts(operator, state):
                # the subspace of a structure solved on another route (whole grid or other sectors): a first solve
                state = None
            solution = solve(operator, requested_states, settings=settings, state=state)
            self.state = solution.state
            return solution

        return wrapped

    def to_host(self, convert):
        def wrapped(spinors):
            # the loop converts its final spinors last
            self.spinors = spinors
            return convert(spinors)

        return wrapped

    def attach(self, result, soc_projectors) -> None:
        result.eigensolver_state = self.state
        result.device_spinors = self.spinors
        result.device_spinors_device = self.device
        result.soc_projectors = soc_projectors


def _whole_grid_state(operator, state) -> bool:
    from ..Eigensolvers.chebff_spinor import SpinorEigvalState

    return isinstance(state, SpinorEigvalState) and int(state.operator_dimension) == int(operator.shape[0])


def run_self_consistent_soc(
    system: AcceleratedPreparedSinglePointSystem,
    soc_projectors: tuple[AtomSpinOrbitProjectors, ...],
    *,
    callback: Callable[[SelfConsistentSOCIteration], None] | None = None,
    backend=None,
    note: Callable[[str], None] | None = None,
    symmetry: bool = True,
    forced: bool = False,
    warm_start=None,
    keep_device: bool = False,
    plan_cache: dict | None = None,
) -> SelfConsistentSOCResult:
    """Run self-consistent SOC with the device spinor Hamiltonian and eigensolver.

    Without ``backend`` the system must be bound to CuPy
    (``system.backend_info.selected == "cupy"``); periodic systems are not
    covered by this Gamma-point tier.

    ``PARSEC_SOC_SECTORS=kramers`` solves the ``+i`` sector of a two-fold
    operation of the double group, one state of every Kramers pair, and ``on``
    splits it by inversion parity where the cluster has a centre of inversion
    (:mod:`.soc_sectors`); ``auto`` (the default since 2026-11-21) is
    ``kramers`` where the whole-grid spinor basis reaches the floor of
    :func:`.soc_sectors.resolve_soc_sectors`, or where ``--symmetry on`` asks
    for a reduction (``forced``), and the whole grid below it; under the
    opt-in ``PARSEC_SOC_AUTO_FOURFOLD=auto`` it is ``fourfold`` (``kramers``
    split by a four-fold operation about z) from a larger floor on.  ``note``
    receives a line that says which route the run takes and why.
    ``symmetry=False`` (``Ignore_Symmetry``, ``--symmetry off``) keeps the
    whole grid whatever the switch.

    The loop of a structure of a relaxation or molecular dynamics run
    (``PARSEC_SOC_DEVICE_FORCES=auto``) passes ``warm_start``
    (:func:`parsec_python.SCF.single_point.warm_start_for_moved_atoms` of the
    structure before: its mixed potential with the ionic part exchanged, its
    Hartree potential and its device subspace, which a structure on the same
    route filters again) and ``keep_device``: the result then keeps its last
    eigensolver state (``eigensolver_state``), its final device spinors
    (``device_spinors`` on the device ``device_spinors_device``, the whole-grid
    subspace or the sector spinors) and ``soc_projectors``, which the device
    forces read (:mod:`..Forces.spinor_device`).  ``plan_cache`` (a dict the ionic loop keeps) lets a structure
    whose atoms keep the operations of the structure before take its sector bases and the folds that do not depend on
    the atoms (the same numbers as made again).
    """
    from .soc_sectors import SWITCH, resolve_soc_sectors, soc_sectors_setting

    sectors = soc_sectors_setting()
    requested = sectors
    if sectors == "auto":
        if not symmetry:
            # (as the former default off: no note for an input without symmetry)
            sectors = "off"
        else:
            sectors, reason = resolve_soc_sectors(sectors, system, forced=forced)
            if reason and note is not None:
                note(f"NOTE: {SWITCH}=auto: {reason}; the spinors stay on the whole grid")
    if backend is None:
        if system.backend_info.selected != "cupy":
            raise ValueError(
                "accelerated run_self_consistent_soc requires a CuPy-bound "
                f"system, got backend={system.backend_info.selected!r}"
            )
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        backend = CupyArrayBackend(int(cp.cuda.Device().id))
    backend.use()
    xp = backend.xp

    if sectors != "off" and not symmetry:
        if note is not None:
            note(
                f"NOTE: PARSEC_SOC_SECTORS={sectors}: the input uses no symmetry (Ignore_Symmetry or "
                "--symmetry off); the spinors stay on the whole grid"
            )
    elif sectors != "off":
        route = _sector_route(
            system, soc_projectors, backend, note, parity=sectors == "on",
            label="auto" if requested == "auto" else None, fourfold=sectors == "fourfold", cache=plan_cache,
        )
        if route is not None:
            keeper = _DeviceKeeper(backend.device_id, route.accepts) if keep_device else None
            result = run_reference_self_consistent_soc(
                system,
                soc_projectors,
                callback=callback,
                eigenproblem_solver=route.eigensolver if keeper is None else keeper.solver(route.eigensolver),
                spinor_hamiltonian_builder=route.hamiltonian,
                orbital_density_builder=route.density,
                synchronize=backend.synchronize,
                to_host=route.to_host if keeper is None else keeper.to_host(route.to_host),
                **({} if warm_start is None else {"warm_start": warm_start}),
            )
            result.timings["soc_sector_setup"] = route.setup_seconds
            if route.regrown:
                result.timings["soc_sector_regrown"] = float(route.regrown)
            if keeper is not None:
                keeper.attach(result, soc_projectors)
            return result

    def solver(operator, requested_states, *, settings, state=None):
        return solve_spinor_eigval(
            operator,
            requested_states,
            settings=settings,
            state=state,
            xp=xp,
            sync=backend.synchronize,
        )

    keeper = _DeviceKeeper(backend.device_id, _whole_grid_state) if keep_device else None
    result = run_reference_self_consistent_soc(
        system,
        soc_projectors,
        callback=callback,
        eigenproblem_solver=solver if keeper is None else keeper.solver(solver),
        spinor_hamiltonian_builder=DeviceSpinorHamiltonianBuilder(backend),
        orbital_density_builder=lambda spinors, occupations, volume_element: spinor_density_on_device(
            xp, spinors, occupations, volume_element
        ),
        synchronize=backend.synchronize,
        to_host=backend.asnumpy if keeper is None else keeper.to_host(backend.asnumpy),
        **({} if warm_start is None else {"warm_start": warm_start}),
    )
    if keeper is not None:
        keeper.attach(result, soc_projectors)
    return result


def _sector_route(
    system, soc_projectors, backend, note, *, parity: bool, label: str | None = None, fourfold: bool = False,
    cache: dict | None = None,
):
    """The hooks of the double-group sectors (:class:`.soc_sectors.SectorRoute`), split by inversion parity with
    ``parity`` or by a four-fold operation about z with ``fourfold``, or ``None`` (the whole grid) for a cluster
    without a two-fold rotation or mirror; ``note`` hears which.  ``cache``: the plans of the structures before
    (:func:`.soc_sectors.detect_sector_plan`)."""
    import time

    from .soc_sectors import SWITCH, SectorRoute, detect_sector_plan

    setting = label or ("on" if parity else "fourfold" if fourfold else "kramers")
    start = time.perf_counter()
    plan, reason = detect_sector_plan(system.grid, system.atoms, parity=parity, fourfold=fourfold, cache=cache)
    seconds = time.perf_counter() - start
    if plan is None:
        if note is not None:
            note(f"NOTE: {SWITCH}={setting}: {reason}; the spinors stay on the whole grid")
        return None
    route = SectorRoute(plan, backend, soc_projectors)
    route.setup_seconds = seconds
    if note is not None:
        note(f"Spin-orbit sectors ({SWITCH}={setting}): {plan.description()}")
    return route


def soc_ionic_backend(
    *,
    symmetry: bool = True,
    forced: bool = False,
    note: Callable[[str], None] | None = None,
    potential_extrapolation: str = "none",
    fortran: bool = False,
):
    """The ionic loop of a self-consistent spin-orbit cluster on the device (``PARSEC_SOC_DEVICE_FORCES=auto``).

    The counterpart of :func:`parsec_python.Relaxation.backend.spinor_backend` (the reference solver's) for a system
    prepared on CuPy: every structure runs :func:`run_self_consistent_soc` (the double-group sectors of
    ``PARSEC_SOC_SECTORS`` where ``symmetry`` allows them, ``forced`` under ``--symmetry on``) with the spin-orbit
    projectors of its atoms, keeping its device spinors and subspace; the forces are those of
    :func:`..Forces.spinor_device.compute_device_spinor_forces`; the moved atoms are installed by the accelerated
    :func:`..driver.update_geometry`; a moved structure starts from the one before as on the reference path
    (:func:`parsec_python.SCF.single_point.warm_start_for_moved_atoms` with ``potential_extrapolation``: the mixed
    potential with its ionic part exchanged, the Hartree potential and the subspace, filtered again where the route
    of the sectors is the same).  ``note`` hears the route of the first structure and of every structure whose route
    line differs from that of the one before (another sector plan, or the whole grid).  ``fortran``: the forces of the
    Fortran-derived routines (:func:`parsec_python.Forces.spin_orbit_fortran.compute_spin_orbit_forces` on the
    device) and the moved atoms as new atoms of symbol and position, as
    :func:`parsec_python.Relaxation.backend.fortran_spinor_backend` moves them.
    """

    from dataclasses import replace
    from functools import partial

    from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
    from parsec_python.Relaxation.backend import IonicBackend
    from parsec_python.SCF.single_point import extrapolation_keywords, warm_start_for_moved_atoms

    from ..driver import update_geometry
    from ..Forces.spinor_device import compute_device_spinor_forces, release_device_spinors

    # the route lines of the structure before: a structure on the same route says nothing, one whose plan or route
    # differs (its atoms left operations of the one before, or another route was taken) says which it takes
    previous: list[str] = []
    # the sector plans of the structures (the same grid; a structure that keeps the operations reuses its bases)
    plans: dict = {}

    def with_criterion(system, criterion):
        # the loop reads its criterion from the settings of the system: a copy (the same fields) with the loosened one
        loosened = replace(system)
        scf = replace(system.input.scf, convergence_criterion=float(criterion))
        loosened.input = replace(system.input, scf=scf)
        return loosened

    def solve(system, *, callback=None, warm_start=None, convergence_criterion=None):
        if convergence_criterion is not None:
            system = with_criterion(system, convergence_criterion)
        projectors = tuple(
            build_spin_orbit_projectors(system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials)
        )
        lines: list[str] = []

        def heard(line: str) -> None:
            index = len(lines)
            lines.append(line)
            if note is not None and (index >= len(previous) or previous[index] != line):
                note(line)

        result = run_self_consistent_soc(
            system,
            projectors,
            callback=callback,
            note=heard,
            symmetry=symmetry,
            forced=forced,
            warm_start=warm_start,
            keep_device=True,
            plan_cache=plans,
        )
        previous[:] = lines
        return result

    def forces(system, result, *, symmetrize=False, operations=None):
        return compute_device_spinor_forces(
            system, result, symmetrize=symmetrize, operations=operations, fortran=fortran
        )

    def carried(previous_system, previous_result, system, **keywords):
        start = warm_start_for_moved_atoms(previous_system, previous_result, system, **keywords)
        # the subspace moved into the start; the spinors of the forces are not needed again
        previous_result.eigensolver_state = None
        release_device_spinors(previous_result)
        return start

    return IonicBackend(
        run_scf=solve,
        compute_forces=forces,
        update_geometry=update_geometry,
        warm_start=partial(carried, **extrapolation_keywords(potential_extrapolation)),
        restart_scf=lambda system, result: carried(system, result, system),
        keep_atom_fields=not fortran,
    )


__all__ = ["DeviceSpinorHamiltonianBuilder", "run_self_consistent_soc", "soc_ionic_backend"]
