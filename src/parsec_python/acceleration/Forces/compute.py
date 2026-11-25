"""Atomic forces of a calculation that ran on an accelerated backend.

The assembly is :func:`parsec_python.Forces.assemble_forces`; what differs
from the readable path is where the two expensive sums are formed.  The
local-pseudopotential terms couple every atom with every grid point and run
on a GPU when the calculation did.  The nonlocal term needs every occupied
orbital: it is contracted on the devices and MPI ranks that hold the
orbitals, in their symmetry sectors, and only one small array per rank comes
back.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Sequence

import numpy as np

from parsec_python.Grid.pbc import uses_fortran_routines
from parsec_python.Forces import (
    AtomSymmetryOperation,
    ForceResult,
    assemble_forces,
    atom_symmetry_operations,
    ewald_ion_ion_forces,
    ion_ion_forces,
    local_forces,
    nonlocal_forces,
    point_charge_forces,
)
from parsec_python.Forces.symmetry import symmetrize_forces
from parsec_python.Forces.total import embedding_force_fields


def sector_columns(representations: np.ndarray) -> np.ndarray:
    """Column of every selected state inside its own symmetry sector.

    The states are listed in global eigenvalue order and each sector keeps
    its own in ascending order, so a state is the ``k``-th of its sector when
    ``k`` states of that sector come before it.
    """

    representations = np.asarray(representations)
    columns = np.zeros(representations.size, dtype=np.int32)
    for representation in np.unique(representations):
        members = np.flatnonzero(representations == representation)
        columns[members] = np.arange(members.size, dtype=np.int32)
    return columns


def _atom_sums(terms: np.ndarray, labels, atom_count: int) -> np.ndarray:
    owners = np.fromiter((label[0] for label in labels), dtype=np.int64, count=len(labels))
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    for component in range(3):
        forces[:, component] = np.bincount(
            owners, weights=terms[:, component], minlength=atom_count
        )
    return forces


def _nonlocal_forces(system, result, clock: dict) -> np.ndarray:
    atom_count = len(system.atoms)
    operator = system.nonlocal_operator
    if operator.projectors.shape[1] == 0:
        return np.zeros((atom_count, 3), dtype=np.float64)
    occupations = np.asarray(result.occupations, dtype=np.float64)
    channels = getattr(result, "spin", None)
    if channels is not None:
        return _spin_nonlocal_forces(system, channels, operator, atom_count)

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    if sectors is not None:
        # Sector labels of the reference result are PARSEC's, from one.
        representations = np.asarray(result.representations, dtype=np.int32) - 1
        solved = getattr(result, "sector_states", None)
        if solved is not None:
            # Weighted sectors: the states they solved, each with its occupation times its weight.
            representations = np.asarray(solved[0], dtype=np.int32) - 1
            occupations = np.asarray(solved[1], dtype=np.float64)
        terms = sectors.projector_force_terms(
            representations, sector_columns(representations), occupations
        )
        # What this process spent inside its own part of the command.
        for name, seconds in (getattr(sectors, "force_phase_seconds", None) or {}).items():
            clock[f"nonlocal_{name}"] = seconds
        return _sector_atom_forces(system, sectors, terms, operator, atom_count)

    gradients = system.projector_gradients()
    device_solver = getattr(system.eigenproblem_solver, "solver", None)
    device_operator = getattr(system.backend, "device_operator", None)
    state = None if device_solver is None else device_solver.device_state
    if state is not None and device_operator is not None:
        from .nonlocal_device import gradient_transposes, projector_force_terms

        occupied = np.flatnonzero(occupations != 0.0)
        if not occupied.size:
            return np.zeros((atom_count, 3), dtype=np.float64)
        weights = occupations[: int(occupied[-1]) + 1]
        terms = projector_force_terms(
            device_operator,
            state.subspace.vectors,
            weights,
            gradient_transposes(gradients.components),
        )
        return _atom_sums(terms, operator.labels, atom_count)
    return nonlocal_forces(
        operator, gradients, result.wavefunctions, occupations, atom_count
    )


def _sector_atom_forces(system, sectors, terms: np.ndarray, operator, atom_count: int) -> np.ndarray:
    """The nonlocal force terms of the projectors of the sector operators, summed per atom.

    The sector operators may hold the projectors of one atom per orbit of the group of the sectors
    (:func:`..Symmetry.axis_reflection.compress_projectors_for_sectors`): the terms then belong to the labels of their
    gradients (``projector_force_labels``), each ``n_a`` times the term of one atom of its orbit, and the average over
    the full group of a :class:`..Symmetry.axis_reflection.PointGroupReduction` gives every atom its term and the
    partners of the weighted levels theirs (:mod:`.sector_symmetry`), whatever the symmetrization asked for; the
    average over the group of Abelian sectors with such projectors gives every atom of an orbit its term
    (:func:`.sector_symmetry.abelian_orbit_operations`).
    """

    from .sector_symmetry import abelian_orbit_operations, sector_force_operations

    labels = getattr(sectors, "projector_force_labels", None)
    forces = _atom_sums(terms, operator.labels if not labels else labels, atom_count)
    operations = sector_force_operations(system)
    if operations is None:
        operations = abelian_orbit_operations(system)
    if operations is not None:
        forces = symmetrize_forces(forces, operations)
    return forces


def _spin_nonlocal_forces(system, channels, operator, atom_count: int) -> np.ndarray:
    """Nonlocal forces of the two channels of a spin-polarized result.

    ``forcnloc.f90`` weights a state by four over the number of spin
    channels: each channel enters the unpolarized contraction with half its
    occupations.  The orbitals are read where the channel keeps them, on
    its device or in the result.
    """

    # PARSEC_SPIN_FLIP_SYMMETRY: a down channel taken from the up channel has no
    # subspace; its forces are those of the up channel carried by the operation.
    down_state = getattr(channels[1], "eigensolver_state", None) if len(channels) == 2 else None
    if getattr(down_state, "spin_flip_channel", False):
        up = _spin_nonlocal_forces(system, channels[:1], operator, atom_count)
        return up + down_state.operation.down_forces(up)
    sectors = getattr(system.backend, "symmetry_eigensolver", None)
    if sectors is not None:
        # Both channels in the symmetry sectors (Spin_Symmetry_Sectors): each
        # channel's sector subspaces, half its occupations.
        selections = []
        for channel in channels:
            representations = np.asarray(channel.representations, dtype=np.int32) - 1
            occupations = np.asarray(channel.occupations, dtype=np.float64)
            solved = getattr(channel, "sector_states", None)
            if solved is not None:
                # Weighted sectors: the states they solved, each with its occupation times its weight.
                representations = np.asarray(solved[0], dtype=np.int32) - 1
                occupations = np.asarray(solved[1], dtype=np.float64)
            selections.append(
                (
                    representations,
                    sector_columns(representations),
                    0.5 * occupations,
                )
            )
        return _sector_atom_forces(
            system, sectors, sectors.spin_projector_force_terms(selections), operator, atom_count
        )
    gradients = system.projector_gradients()
    for_channel = getattr(system.eigenproblem_solver, "for_spin_channel", None)
    device_operator = getattr(system.backend, "device_operator", None)
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    transposes = None
    for index, channel in enumerate(channels):
        occupations = 0.5 * np.asarray(channel.occupations, dtype=np.float64)
        occupied = np.flatnonzero(occupations != 0.0)
        if not occupied.size:
            continue
        device_solver = (
            getattr(for_channel(index), "solver", None)
            if callable(for_channel)
            else None
        )
        state = None if device_solver is None else device_solver.device_state
        if state is not None and device_operator is not None:
            from .nonlocal_device import gradient_transposes, projector_force_terms

            if transposes is None:
                transposes = gradient_transposes(gradients.components)
            terms = projector_force_terms(
                device_operator,
                state.subspace.vectors,
                occupations[: int(occupied[-1]) + 1],
                transposes,
            )
            forces += _atom_sums(terms, operator.labels, atom_count)
        else:
            forces += nonlocal_forces(
                operator, gradients, channel.wavefunctions, occupations, atom_count
            )
    return forces


def correction_forces(system, result) -> dict[str, np.ndarray]:
    """The LDA+U, TS and nuclear-gradient force terms of an accelerated result, by their field names.

    Empty without LDA+U and TS and under ``PARSEC_GGA_NUCLEAR_GRADIENT=off``, so that :func:`assemble_forces` is
    called as before.  The LDA+U term is formed where the orbitals are
    (``system.hubbard_device``, Hamiltonian.hubbard_device) or, on the CPU
    backends, by :func:`parsec_python.Forces.total.correction_forces` from
    the host orbitals; the TS forces are those of the correction of the
    converged density.
    """

    from parsec_python.V_xc.nuclear_gradient import correction_forces as nuclear_gradient_forces

    device = getattr(system, "hubbard_device", None)
    if device is None:
        if getattr(system, "hubbard", None) is None and getattr(result, "dispersion", None) is None:
            # PARSEC_GGA_NUCLEAR_GRADIENT: the term of the correction of the density gradient near the nuclei
            nuclear = nuclear_gradient_forces(system, result)
            return {} if nuclear is None else {"nuclear_gradient": nuclear}
        from parsec_python.Forces.total import correction_forces as host_correction_forces

        # (the host terms hold the nuclear-gradient term too)
        return host_correction_forces(system, result)
    nuclear = nuclear_gradient_forces(system, result)
    terms = {"hubbard": device.forces(system, result)}
    dispersion = getattr(result, "dispersion", None)
    if dispersion is not None:
        terms["dispersion"] = np.asarray(dispersion.forces, dtype=np.float64)
    if nuclear is not None:
        terms["nuclear_gradient"] = nuclear
    return terms


def _array_module(system):
    """``(xp, erf, device)`` of the sums of a periodic cell: CuPy on the current device when the calculation ran on a GPU."""

    if system.backend_info.selected != "cupy":
        return np, None, None
    from cupyx.scipy.special import erf

    from ..backends.cupy import require_cupy
    from .local_device import local_force_device_ids

    cp, _ = require_cupy()
    return cp, erf, local_force_device_ids(system)[0]


def _abelian_local_operations(system, abelian=None):
    """``H`` of the Abelian sectors of a periodic cell for its local forces (:func:`.sector_symmetry.
    abelian_cell_operations`, or ``abelian`` where the caller has them), or ``None``: no such sectors, a group whose
    operations do not map the atoms (the sums of every atom then), or ``PARSEC_PERIODIC_LOCAL_ORBITS=off``."""

    from .periodic_local import periodic_local_orbits_setting
    from .sector_symmetry import abelian_cell_operations

    if periodic_local_orbits_setting() == "off":
        return None
    if abelian is None:
        try:
            abelian = abelian_cell_operations(system)
        except ValueError:
            return None
    return abelian if abelian is not None and len(abelian) > 1 else None


def _left_spin_sectors(system) -> bool:
    """Whether ``system`` is the full grid that a spin-polarized run in the sectors went back to (its guard fired).

    ``driver._SpinSectorFallback`` names that in the ``spin_symmetry_sectors`` detail of the replaced system
    (``auto: left the sectors after ...``), which moved structures carry on.
    """

    details = dict(getattr(getattr(system, "backend_info", None), "details", ()) or ())
    return str(details.get("spin_symmetry_sectors", "")).startswith("auto: left the sectors")


def compute_periodic_forces(
    system,
    result,
    *,
    symmetrize: bool = False,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished accelerated SCF calculation of a periodic cell at the Gamma point.

    The terms are those of :func:`parsec_python.Forces.compute_periodic_forces`.  The Ewald ion-ion force is formed on
    the host.  The local ionic and core-correction forces are the ball sums and the one FFT of
    :func:`.periodic_local.fast_periodic_local_forces`, on the GPU when the calculation ran on one.  The nonlocal term is
    contracted where the orbitals are, as for an isolated structure, with the gradients of the periodic projectors.
    A crystal has no torque, and its net force is removed.  A cell solved in the sectors of a point group with
    weighted representations (:class:`..Symmetry.axis_reflection.PointGroupReduction`, one partner of each degenerate
    level solved) has its nonlocal term completed by the average over the full group ``G`` of the reduction, its
    translations included (:mod:`.sector_symmetry`), and the total averaged over ``G`` (or over ``operations``);
    otherwise a crystal has no symmetrization here.  The density and exchange-correlation potential of such a run are
    averaged over ``G``, so its local terms are those of one atom per orbit of ``G`` turned onto the others
    (:func:`.periodic_local.orbit_periodic_local_forces`, ``PARSEC_PERIODIC_SECTOR_LOCAL_FORCES``), and
    ``phase_seconds["local_ewald_asymmetry"]`` (Ry/bohr) reports how much the average over the group changed the
    Ewald and local terms.  A cell in the sectors of an Abelian group ``H`` (every level solved, the density and
    exchange-correlation potential those of the orbits of ``H``) takes the local terms of one atom per orbit of ``H``
    (``PARSEC_PERIODIC_LOCAL_ORBITS``, :func:`.sector_symmetry.abelian_cell_operations`), with
    ``phase_seconds["local_representatives"]`` and ``["local_site_asymmetry"]`` (Ry/bohr).  The ball sums of the
    local terms run batched on the GPU (``PARSEC_PERIODIC_LOCAL_BALLS``).
    """

    from .sector_symmetry import abelian_cell_operations, sector_force_operations

    completion = sector_force_operations(system)
    # A cell in the sectors of an Abelian group solves every level: its forces need no completion, but a relaxation
    # averages them (and its moved positions) over that group, so that atoms the group fixes stay on it.
    abelian = abelian_cell_operations(system) if completion is None and (symmetrize or operations) else None
    # A spin-polarized cell that left its sectors for the full grid (its guard fired, driver._SpinSectorFallback):
    # a relaxation keeps averaging over the operations of its first structure, which its moved positions keep; a
    # single point gives the forces of the full grid as they are (its state may break that group).
    kept = None
    if (symmetrize or operations) and completion is None and abelian is None and _left_spin_sectors(system):
        kept = tuple(operations) if operations else ()
    if (symmetrize or operations) and completion is None and abelian is None and kept is None:
        from ..SCF.gamma_hybrid_route import full_grid_forces_averaged

        if full_grid_forces_averaged(system):
            # A Gamma hybrid on the full grid below the floor of its sectors (PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto):
            # its forces averaged over the operations of the cell, those the ionic loop found for the first structure
            # (which its moved positions keep) or, for a single point, those the Fortran-derived routines take on the
            # full grid (periodic_force_operations).
            from parsec_python.Forces.periodic import periodic_force_operations

            kept = tuple(operations) if operations else tuple(periodic_force_operations(system, None))
    if (symmetrize or operations) and completion is None and abelian is None and kept is None:
        raise ValueError(
            "a periodic cell has no symmetrized forces: its structure is not reduced by space-group operations"
        )
    from parsec_python.Forces.periodic import core_force_xc

    from .periodic_local import (
        fast_periodic_local_forces,
        orbit_periodic_local_forces,
        periodic_sector_local_forces_setting,
    )

    started = perf_counter()
    clock: dict[str, float] = {}
    reference = system.reference
    atoms = reference.cell_atoms
    positions = np.array([atom.position for atom in reference.atoms], dtype=np.float64)
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    repulsion = ewald_ion_ion_forces(atoms, system.pseudopotentials, lattice)
    clock["ion_ion"] = perf_counter() - started
    mark = perf_counter()
    xp, erf, device = _array_module(system)
    arguments = (
        system.grid,
        atoms,
        system.pseudopotentials,
        system.input.pseudopotentials,
        lattice,
        result.density,
        # the potential of the core term with the average potshift removed (PARSEC_CORE_FORCE_XC)
        core_force_xc(system, result, result.xc_potential),
    )
    # In sectors the density and the exchange-correlation potential are averaged over G (PARSEC_PERIODIC_SECTOR_
    # LOCAL_FORCES=orbits): the local terms of one atom per orbit of G, turned onto the others.
    orbits = completion is not None and periodic_sector_local_forces_setting() == "orbits"
    # Abelian sectors: the density and the exchange-correlation potential are those of the orbits of H
    # (PARSEC_PERIODIC_LOCAL_ORBITS=auto): the local terms of one atom per orbit of H.
    abelian_orbits = None if completion is not None else _abelian_local_operations(system, abelian)
    site_asymmetry = 0.0

    def local_terms():
        if orbits:
            return orbit_periodic_local_forces(*arguments, completion, xp=xp, erf=erf)
        if abelian_orbits is not None:
            return orbit_periodic_local_forces(*arguments, abelian_orbits, xp=xp, erf=erf)
        if device is None:
            return fast_periodic_local_forces(*arguments), None
        return fast_periodic_local_forces(*arguments, xp=xp, erf=erf), None

    if device is None:
        local, details = local_terms()
    else:
        with xp.cuda.Device(device):
            local, details = local_terms()
    if details is not None:
        site_asymmetry = details["site_asymmetry"]
        if abelian_orbits is not None:
            # Ry/bohr, not seconds: the checked atoms against their turned values (round-off)
            clock["local_representatives"] = float(details["representatives"])
            clock["local_site_asymmetry"] = float(site_asymmetry)
    clock["local"] = perf_counter() - mark
    mark = perf_counter()
    separable = _nonlocal_forces(system, result, clock)
    averaged_over = None
    if completion is not None:
        # The nonlocal term is complete over G already (_sector_atom_forces); the total is averaged over G too.
        averaged_over = tuple(operations) if operations else completion
    elif abelian is not None:
        averaged_over = tuple(operations) if operations else abelian
    elif kept:
        averaged_over = kept
    clock["nonlocal"] = perf_counter() - mark
    mark = perf_counter()
    # the LDA+U and TS terms, as the Fortran-derived routines (periodic_device) and Forces.compute_periodic_forces add them
    corrections = correction_forces(system, result)
    if corrections:
        clock["corrections"] = perf_counter() - mark
    forces = assemble_forces(
        positions, repulsion, local, separable, averaged_over, phase_seconds=clock, periodic=True, **corrections
    )
    if completion is not None:
        # What the average over the group changed (Ry/bohr, not seconds): the asymmetry of the Ewald and local terms
        # (the nonlocal term is averaged already), with that of the local forces of the orbits (the representatives in
        # their site groups, a few other atoms against their turned values); round-off where the density is invariant,
        # as it is in sectors.
        clock["local_ewald_asymmetry"] = max(
            site_asymmetry, float(np.max(np.abs(forces.unsymmetrized - forces.symmetrized), initial=0.0))
        )
    return replace(forces, seconds=perf_counter() - started)


def compute_forces(
    system,
    result,
    *,
    symmetrize: bool = True,
    operations: Sequence[AtomSymmetryOperation] | None = None,
) -> ForceResult:
    """Forces of a finished accelerated SCF calculation.

    ``system`` is the prepared accelerated system and ``result`` what its SCF
    loop returned; call this before the next eigensolve of the system, whose
    eigensolver still holds the orbitals.  The other arguments are those of
    :func:`parsec_python.Forces.compute_forces`.  In an MPI run this is called
    on the root rank.
    """

    if getattr(system.input, "periodic_cell", None) is not None:
        sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
        decomposition = getattr(sectors, "decomposition", None)
        if (
            decomposition is not None
            and int(decomposition.reduction.group_order) > 1
            and uses_fortran_routines(system.grid)
        ):
            from .sector_symmetry import fortran_sector_forces_setting

            if fortran_sector_forces_setting() == "off":
                # The Fortran-derived routines read the orbitals and density of a cell on the full grid.
                raise NotImplementedError(
                    "the forces of a periodic cell of the Fortran-derived routines solved in the sectors of its point "
                    "group are not computed under PARSEC_FORTRAN_SECTOR_FORCES=off: prepare it with symmetry='off' "
                    "or PARSEC_PERIODIC_SYMMETRY=off"
                )
        if uses_fortran_routines(system.grid):
            # The Fortran-derived routines (Use_Fortran_Routines): the device builders of
            # their Ewald terms and projectors with their lattice images; in the sectors of
            # the point group of the cell (PARSEC_FORTRAN_SECTOR_FORCES=auto) the local terms
            # read the fields the sectors expand to the full grid and the nonlocal term is
            # contracted in the sectors.
            from .periodic_device import compute_device_periodic_forces

            return compute_device_periodic_forces(
                system, result, symmetrize=symmetrize, operations=operations
            )
        return compute_periodic_forces(system, result, symmetrize=symmetrize, operations=operations)
    started = perf_counter()
    clock: dict[str, float] = {}
    atoms = system.atoms
    specifications = system.input.pseudopotentials
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    repulsion = ion_ion_forces(atoms, system.pseudopotentials)
    point_charges = getattr(system, "point_charges", ())
    charged = None
    if point_charges:
        # forcept.f90 adds the forces of the point charges on the nuclei to
        # the ion-ion forces (Forces.compute_forces does the same).
        charged = point_charge_forces(atoms, system.pseudopotentials, point_charges)
        repulsion = repulsion + charged
    clock["ion_ion"] = perf_counter() - started
    mark = perf_counter()
    residual = np.asarray(result.output_effective_potential) - np.asarray(
        result.input_effective_potential
    )
    # FDET: rho_B + rho_A and V_xc + v_nad (Forces.total.embedding_force_fields).
    density, xc_potential, embedded = embedding_force_fields(system, result)
    fields = (
        system.grid,
        atoms,
        system.pseudopotentials,
        specifications,
        density,
        residual,
        xc_potential,
    )
    from parsec_python.Pseudopotential.radial_derivatives import linear_slope_setting

    # PARSEC_LINEAR_TABLE_SLOPES=interpolant: the device kernels interpolate the slope tables of pseudo.f90; the
    # slopes of the linear interpolants of a species with linear tables are summed by the readable routine
    host_slopes = linear_slope_setting() == "interpolant" and any(
        not specifications[atom.symbol].use_spline for atom in atoms
    )
    if system.backend_info.selected == "cupy" and not host_slopes:
        from .local_device import (
            cupy_local_forces,
            cupy_wedge_local_forces,
            group_wedge,
            local_force_device_ids,
            local_force_sums_setting,
        )

        devices = local_force_device_ids(system)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
        sums = local_force_sums_setting()
        if (
            reduction is not None
            and reduction.group_order > 1
            and sums in ("orbits", "wedge")
        ):
            # PARSEC_LOCAL_FORCE_SUMS=orbits: one point per orbit of the full group of weighted sectors
            grouped = group_wedge(reduction) if sums == "orbits" else None
            local = cupy_wedge_local_forces(
                *fields, reduction, device_ids=devices, full_group=grouped is not None
            )
            clock["local_points"] = float(reduction.wedge_size if grouped is None else grouped[0].size)
        else:
            local = cupy_local_forces(*fields, device_ids=devices)
            clock["local_points"] = float(system.grid.size)
    else:
        local = local_forces(*fields)
    clock["local"] = perf_counter() - mark
    mark = perf_counter()
    separable = _nonlocal_forces(system, result, clock)
    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    if symmetrize and hasattr(getattr(getattr(sectors, "decomposition", None), "reduction", None), "coset_images"):
        # The one-dimensional sectors of a subgroup of the point group solve one partner of some degenerate
        # levels (weighted in the occupations): the nonlocal term of all partners is its average over the
        # full point group, which the total gets below in any case and its report gets here.
        if operations is None:
            operations = atom_symmetry_operations(atoms)
        separable = symmetrize_forces(separable, operations)
    clock["nonlocal"] = perf_counter() - mark
    mark = perf_counter()
    corrections = correction_forces(system, result)
    if corrections:
        clock["corrections"] = perf_counter() - mark
    mark = perf_counter()
    from parsec_python.Hartree.variational import hartree_tail_forces

    # The atomic tail of the Hartree boundary moves with the atoms (PARSEC_HARTREE_VARIATIONAL).
    tail_term = hartree_tail_forces(system, result)
    if tail_term is not None:
        clock["hartree_boundary"] = perf_counter() - mark
    mark = perf_counter()
    if not symmetrize:
        operations = None
    elif operations is None:
        operations = atom_symmetry_operations(atoms)
    forces = assemble_forces(
        positions,
        repulsion,
        local,
        separable,
        operations,
        phase_seconds=clock,
        **corrections,
        **({} if charged is None else {"point_charge": charged}),
        **({"keep_net_force": True} if embedded else {}),
        **({} if tail_term is None else {"hartree_boundary": tail_term}),
    )
    clock["symmetrization"] = perf_counter() - mark
    return replace(forces, seconds=perf_counter() - started)


__all__ = ["compute_forces", "compute_periodic_forces", "correction_forces", "sector_columns"]
