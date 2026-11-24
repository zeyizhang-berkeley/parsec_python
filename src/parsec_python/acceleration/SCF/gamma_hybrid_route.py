"""Unpolarized hybrid functionals of bulk cells at the Gamma point in symmetry sectors (``PARSEC_GAMMA_HYBRID_SECTORS``).

Until 2026-11-19 the device Fock loop of the accelerated driver (:mod:`.hybrid`) solved an unpolarized hybrid of a
periodic cell at the Gamma point on the full grid of one GPU (``driver.periodic_sector_refusal``: "a hybrid
functional"), while a spin-polarized one runs as the 1x1x1 k-point mesh of the reference solver
(``cli._gamma_spin_on_kpoint_driver``) in the k-point sectors of the little group of Gamma.  The switch has three
values:

* ``auto`` (default): the cell takes the route of the sectors of the Gamma point in one process
  (``driver._gamma_hybrid_in_sectors``): its orbitals real, in the sectors of the abelian group of the cell (the
  commuting involutions with the half-cell translations of a supercell, ``periodic_abelian_reduction``; a hybrid
  keeps no weighted sectors, ``driver._orbital_group_refusal``), the device exchange from their orbitals gathered
  into the coset form of the half-cell translations the sectors carry, whose pair densities the cuFFT pair solver of
  the cell solves class by class on the sublattice of each class (``Hamiltonian/exchange_translations.py``,
  ``PARSEC_CUPY_EXCHANGE_TRANSLATIONS``; a cell without such translations: unfolded to the full grid and solved
  there, ``PARSEC_CUPY_EXCHANGE_POOL``), folded back to every sector, whose compressed operator
  ``-a_x xi_G xi_G^T`` joins its Hamiltonian (``Hamiltonian/exchange_device.py``).  Its energies are those of the
  full grid to the SCF convergence.  A cell whose
  basis (grid points times states) is below ``2^24`` entries (:func:`sector_basis_reached`) keeps the full grid;
  with forces or a relaxation by an optimizer its forces and moved positions are averaged there over the operations
  of the cell (``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto``; ``sectors`` takes the sectors instead:
  :func:`gamma_hybrid_sector_forces_setting`).
  The MPI full-SCF runner takes the SCF of such a cell since 2026-11-22 (``driver._mpi_periodic_hybrid``): the same
  sectors above the same floor, every sector basis on one GPU of its rank, and the one sector of the identity on the
  first GPU of the root below it and under ``off``; the exchange with the cuFFT pair solver of the cell
  (``SCF/hybrid_mpi.py``): since 2026-11-23 the coset blocks or (from 32 sectors) the pooled block of one process
  gathered on every rank, a small build whole on each rank and a larger one divided among the GPUs of all ranks, and
  a small build of the sector of the identity on its GPU alone (``PARSEC_MPI_EXCHANGE_POOL=auto``; ``off``: the
  pair tiles of the sectors on every GPU of every rank, as before).  Its forces and moves, ``kpoint`` and a
  spin-polarized hybrid stay refused there.
* ``kpoint``: a cell this module finds large enough (:func:`gamma_hybrid_sector_route`) runs as the 1x1x1 k-point
  mesh of the reference solver, whose k-point sectors split the Gamma point into the sectors of the abelian subgroup
  ``H_Gamma`` of its little group (:mod:`.kpoint_sectors`), the exchange built from their orbitals on the whole grid
  and reduced to every sector (``-a_x (U^H xi)(U^H xi)^H``): a bulk orthorhombic cell whose basis (grid points times
  states) reaches the floor of the k-point sectors (``kpoint_sectors._MINIMUM_ENTRIES``, ``2^24``) and whose
  ``H_Gamma`` would at least halve the work of the eigensolver (the rule of :func:`.kpoint_sectors.plan_sectors`, with
  sectors of equal size).  Complex Bloch factors at ``k = 0``, the same energies; measured on one A100 node
  (2026-11-19) it is 2-2.5x slower than the full grid for Si64 and diamond 3^3 (small sector tasks, and the level
  passes of the Fock loop leave the k-point sectors), so it is an opt-in.
* ``off``: the full grid of one device, bit for bit as before.

The ``kpoint`` route is a single point of one process: forces, relaxations, molecular dynamics, an energy
refinement, bands, ``Restart_Run``, ``occup.in``, LDA+U, meta-GGA, the dispersion correction, FDET, ``nscf`` and
``--mpi`` keep the device Fock loop, as do the cells for which ``PARSEC_KPOINT_SECTORS=off`` or the GPU k-point
workers are not available (the reference solver would then solve the 1x1x1 mesh on the full grid in complex
arithmetic).

Nothing here prints or opens files except the pseudopotential tables of an input without ``States_Num``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

_SWITCH = "PARSEC_GAMMA_HYBRID_SECTORS"


def gamma_hybrid_sectors_setting() -> str:
    """``PARSEC_GAMMA_HYBRID_SECTORS``: ``auto`` (default) solves an unpolarized Gamma-point hybrid of a bulk cell in
    the real sectors of the Gamma route; ``kpoint`` runs a large one as the 1x1x1 k-point mesh in the k-point sectors
    of its little group; ``off`` keeps the device Fock loop of the full grid, bit for bit as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in {"auto", "kpoint", "off"}:
        raise ValueError(f"{_SWITCH} must be auto, kpoint or off, not {value!r}")
    return value


_FORCES_SWITCH = "PARSEC_GAMMA_HYBRID_SECTOR_FORCES"


def gamma_hybrid_sector_forces_setting() -> str:
    """``PARSEC_GAMMA_HYBRID_SECTOR_FORCES``: ``auto`` (default, since 2026-11-21), ``sectors`` or ``off``.

    What an unpolarized Gamma-point hybrid below the floor of :func:`sector_basis_reached` does when its caller counts
    its forces or moved atoms as forces of the sectors (``sector_forces`` of ``driver.prepare_single_point``: the
    command line under ``PARSEC_PERIODIC_SECTOR_FORCES=auto``).  Such a cell keeps the full grid, and on the default
    (Grieder) routines nothing averaged its forces and moved positions over a group (until 2026-11-21); the
    Fortran-derived routines average them over their own operations on that full grid
    (:func:`parsec_python.Forces.periodic.periodic_force_operations`).

    * ``auto``: the full grid, as before; on the default routines its forces and the moved positions of a relaxation
      by an optimizer are averaged over the operations of the cell that the Fortran-derived routines take there (the
      space group the problem carries, else the point operations about the origin that keep the lattice, the grid and
      the atoms), the SCF unchanged (:func:`full_grid_forces_averaged`).  The Fortran-derived routines run as before.
    * ``sectors``: the cell takes the sectors of ``PARSEC_GAMMA_HYBRID_SECTORS=auto`` whatever the size of its basis,
      as a semilocal cell keeps them below ``PARSEC_PERIODIC_SECTOR_FLOOR`` (the same forces within the SCF noise,
      averaged over the abelian group of the sectors; a first run of an 8-atom cell is 1.1-1.6 times slower on one
      node of 4 A100, a 64-atom cell about as fast).
    * ``off``: the full grid without the average, bit for bit as before.

    Read by the preparation of every run.
    """

    value = os.environ.get(_FORCES_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "sectors", "off"}:
        raise ValueError(f"{_FORCES_SWITCH} must be auto, sectors or off, not {value!r}")
    return value


#: The detail of the backend information that marks a Gamma-point hybrid kept on the full grid below the floor whose
#: forces and moves are averaged over the operations of the cell (``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto``).
FULL_GRID_FORCES_DETAIL = "gamma_hybrid_sector_forces"
#: How its value starts.
FULL_GRID_FORCES_VALUE = "auto: the full grid below the floor of the sectors"


def full_grid_forces_averaged(system) -> bool:
    """Whether ``system`` is a Gamma-point hybrid on the full grid below the floor whose forces and moved positions are
    averaged over the operations of the cell (the detail the driver records under
    ``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=auto``; moved structures keep it)."""

    details = dict(getattr(getattr(system, "backend_info", None), "details", ()) or ())
    return str(details.get(FULL_GRID_FORCES_DETAIL, "")).startswith(FULL_GRID_FORCES_VALUE)


@dataclass(frozen=True)
class GammaHybridSectorRoute:
    """What the route of a Gamma-point hybrid in the k-point sectors found (for the NOTE of the run)."""

    points_per_axis: tuple[int, int, int]
    states: int
    sectors: int
    #: the states of a sector of ``1/sectors`` of the rows (the first solve; the refit cuts them after it)
    sector_states: int
    #: ``sum n_s M_s^2 / (N M^2)``, the work of the Rayleigh-Ritz steps against the full grid (equal sectors)
    work_fraction: float

    def note(self) -> str:
        counts = "x".join(str(int(value)) for value in self.points_per_axis)
        return (
            f"NOTE: the hybrid functional of this Gamma-point cell ({counts} points, {self.states} states) runs as "
            f"the 1x1x1 k-point mesh of the reference solver, whose k-point sectors split the Gamma point into "
            f"{self.sectors} sectors of its little group (about {self.sector_states} states each) on the GPUs "
            f"({_SWITCH}=kpoint; auto solves the real sectors of the Gamma route, off the full grid)"
        )


def _gamma_point_only(problem) -> bool:
    if not getattr(problem, "has_kpoint_sampling", False):
        return True
    from parsec_python.Grid import kpoint_sampling

    sampling = kpoint_sampling(problem)
    return sampling is not None and len(sampling[0]) == 1 and not np.any(sampling[0])


def _translation_refusal(translation) -> str | None:
    """Why the inputs of ``translation`` keep the device Fock loop of the full grid (``None`` where they need not)."""

    problem = translation.problem
    scf = problem.scf
    if problem.periodic_cell is None or int(getattr(problem.grid, "periodicity", 3)) != 3:
        return "not a bulk cell"
    if int(getattr(problem.periodic_cell, "dimensions", 3)) != 3:
        return "not a bulk cell"
    if not getattr(scf, "is_hybrid", False):
        return "no hybrid functional"
    if scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit:
        return "spin (the spin-polarized cell has its own route)"
    if not _gamma_point_only(problem):
        return "k-points beyond the Gamma point"
    if getattr(scf, "is_meta_gga", False):
        return "a meta-GGA"
    if getattr(scf, "fixed_occupations", None) is not None:
        return "occup.in"
    if getattr(scf, "van_der_waals", False):
        return "the dispersion correction"
    if any(getattr(item, "has_hubbard", False) for item in getattr(problem, "pseudopotentials", {}).values()):
        return "LDA+U"
    if any(getattr(item, "spin_orbit", False) for item in getattr(problem, "pseudopotentials", {}).values()):
        return "spin-orbit species"
    if (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    ):
        return "forces, a relaxation or molecular dynamics"
    for name in ("energy_refinement", "polarizability_field", "nscf", "gw"):
        if getattr(translation, name, None) is not None:
            return name
    if getattr(problem, "bands", None) is not None:
        return "bands"
    if getattr(problem, "embedding", None) is not None:
        return "FDET"
    if getattr(translation, "restart_run", False):
        return "Restart_Run"
    dos = getattr(translation, "dos", None)
    if dos is not None and getattr(dos, "ylm_l", -1) >= 0:
        return "the Ylm-projected density of states"
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    if not np.allclose(lattice, np.diag(np.diag(lattice)), atol=1.0e-10):
        return "an oblique cell"
    return None


def _states(problem) -> int:
    requested = problem.scf.number_of_states
    if requested is not None:
        return int(requested)
    from parsec_python.V_ion.ionic_potential import ionic_charge, load_pseudopotentials

    tables = load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
    electrons = ionic_charge(tuple(problem.atoms), tables) - problem.scf.net_charge
    return int(np.ceil(0.5 * electrons)) + 6


def periodic_grid_points(problem) -> int:
    """The points of the grid of the cell of ``problem`` (``build_periodic_grid(...).size``), counted without
    building the grid (its coordinates and lookup, about 90 bytes per point) on the default routines of a bulk cell
    (``Grid.pbc.default_periodic_grid_counts``); the grids of the Fortran-derived routines and of a slab are built.
    The checks of the builder (stencil width, one common spacing) are left to the preparation, which builds it."""

    from parsec_python.Grid.pbc import build_periodic_grid, default_periodic_grid_counts

    cell, settings = problem.periodic_cell, problem.grid
    if int(getattr(settings, "periodicity", 3)) in (1, 2):
        from ..Symmetry.axis_reflection import slab_wire_sectors_setting

        if slab_wire_sectors_setting() == "auto":
            # a wire or slab of the Fortran-derived routines in the sectors (PARSEC_SLAB_WIRE_SECTORS=auto): the points
            # of its cylinder or slab (off: the bulk grid of its cell, as before, which its stencil refuses)
            from parsec_python.Grid.confined import build_confined_periodic_grid

            return int(build_confined_periodic_grid(cell, settings).size)
    if getattr(settings, "fortran_routines", False) or int(getattr(cell, "dimensions", 3)) == 2:
        return int(build_periodic_grid(cell, settings).size)
    return int(np.prod(default_periodic_grid_counts(cell, settings)))


def sector_basis_reached(problem) -> bool:
    """Whether the basis of ``problem`` (grid points of its cell times its states) reaches the floor of the k-point
    sectors (``kpoint_sectors._MINIMUM_ENTRIES``, ``2^24``): below it the ``auto`` route keeps the full grid (where
    forces and moves are averaged over the operations of the cell, or the sectors take them under
    ``PARSEC_GAMMA_HYBRID_SECTOR_FORCES=sectors``: :func:`gamma_hybrid_sector_forces_setting`).

    ``False`` where the grid or the states cannot be found here (the driver then keeps the full grid as well).
    """

    from .kpoint_sectors import _MINIMUM_ENTRIES

    try:
        rows = periodic_grid_points(problem)
        states = _states(problem)
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return rows * states >= _MINIMUM_ENTRIES


def gamma_hybrid_sector_route(translation) -> GammaHybridSectorRoute | None:
    """The ``kpoint`` route of :mod:`this module <.gamma_hybrid_route>` for ``translation``, or ``None`` where the
    accelerated driver keeps the input (``auto``, ``off``, an input the route does not take, a small or symmetry-poor
    cell).

    The decision depends on the input and the switches only, so that the accelerated command line and the
    reference solver it hands the input to take the same one; whether the k-point workers run on a GPU is the
    callers' to ask (the accelerated command line before the hand-over: its backend is CuPy in one process; the
    reference solver: ``parallel_backend_uses_device``).
    """

    if gamma_hybrid_sectors_setting() != "kpoint":
        return None
    from .kpoint_sectors import (
        _EXTRA_STATES,
        _MINIMUM_ENTRIES,
        _WORK_FRACTION,
        CellGroup,
        generated_subgroup,
        kpoint_sector_atom_tolerance,
        kpoint_sectors_setting,
        largest_commuting_involutions,
        sector_margin,
    )

    if kpoint_sectors_setting() == "off":
        return None
    if _translation_refusal(translation) is not None:
        return None
    problem = translation.problem
    from parsec_python.Grid.pbc import build_periodic_grid
    from parsec_python.SCF.pbc import fold_atoms_into_cell

    from ..Symmetry.axis_reflection import _lattice_period, _periodic_row_major

    # the grid of the preparation (prepare_periodic_single_point: either set of routines), cheap beside the run
    try:
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        if _lattice_period(grid) is None or not _periodic_row_major(grid):
            return None
    except ValueError:
        return None
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    rows = int(grid.size)
    try:
        states = _states(problem)
    except (OSError, ValueError):
        return None
    if rows * states < _MINIMUM_ENTRIES:
        return None
    atoms = fold_atoms_into_cell(tuple(problem.atoms), problem.periodic_cell.lattice_vectors)
    # the tolerance of plan_sectors for this route (a single point: PARSEC_SYMMETRY_ATOM_TOLERANCE as it is, 1e-7 under
    # PARSEC_KPOINT_SECTOR_TOLERANCE=fixed)
    cell = CellGroup(grid, atoms, atom_tolerance=kpoint_sector_atom_tolerance())
    if cell.table is None or cell.order < 2:
        return None
    # At the Gamma point alone every operation of the cell fixes k and maps the mesh {0} onto itself
    # (kpoint_sectors.PotentialInvariance): the little group is the whole group of the cell.
    members = generated_subgroup(cell.table, cell.little_group(np.zeros(3), lambda index: True))
    elements, _ = largest_commuting_involutions(cell.table, members)
    order = len(elements)
    if order < 2:
        return None
    size = rows / order
    count = min(size, math.ceil(sector_margin() * states * size / rows) + _EXTRA_STATES)
    fraction = order * size * count * count / (rows * float(states) * states)
    if fraction > _WORK_FRACTION:
        return None
    return GammaHybridSectorRoute(
        points_per_axis=tuple(int(value) for value in counts),
        states=int(states),
        sectors=int(order),
        sector_states=int(count),
        work_fraction=float(fraction),
    )


__all__ = [
    "FULL_GRID_FORCES_DETAIL",
    "FULL_GRID_FORCES_VALUE",
    "GammaHybridSectorRoute",
    "full_grid_forces_averaged",
    "gamma_hybrid_sector_forces_setting",
    "gamma_hybrid_sector_route",
    "gamma_hybrid_sectors_setting",
    "sector_basis_reached",
]
