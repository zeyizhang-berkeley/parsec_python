"""Spin-polarized bulk cells at the Gamma point in the symmetry sectors (``PARSEC_GAMMA_SPIN_SECTORS``).

A collinear spin-polarized bulk cell sampled at the Gamma point alone (a defect supercell: a vacancy, an NV-type
centre, an open-shell molecule in a box) has real orbitals, as its unpolarized twin.  With
``PARSEC_GAMMA_SPIN_SECTORS=auto`` (the default) the accelerated driver solves it on the route of the Gamma sectors
(:func:`parsec_python.acceleration.driver.periodic_sector_refusal`): the weighted sectors of the cell's point group
with its translations, as clusters take them (``PARSEC_ORBITAL_GROUP_SPIN=auto`` since 2026-11-19, in one process and,
since 2026-11-21, on several MPI ranks, ``PARSEC_ORBITAL_GROUP_MPI``;
:class:`~parsec_python.acceleration.Symmetry.PointGroupReduction`; ``driver._weighted_spin_refusal``:
``PARSEC_ORBITAL_GROUP_SPIN=off`` and ``PARSEC_ORBITAL_GROUP_MPI=off`` on several ranks keep the abelian group), else
the commuting involutions of the cell group with
their half-cell translations (:func:`~parsec_python.acceleration.Symmetry.axis_reflection.periodic_abelian_reduction`),
both channels in those sectors with the symmetry guard of
:mod:`.spin_sectors` (``Spin_Symmetry_Sectors``; its density test takes the operations ``r -> R r + t`` of the sector
group and of the cell, about a defect off the origin as well: ``PARSEC_SPIN_GUARD_CELL_GROUP``), the fields of both
channels on the orbits, the Hartree potential of
the total density solved on the full grid of the cell (``SCF.pbc.solve_periodic_hartree``) and reduced again, and the
spin functional on the orbits.  A run whose guard fires in the weighted sectors is first prepared again in the
abelian ones (``driver._SpinSectorFallback``); one whose guard fires there is prepared again with
``Spin_Symmetry_Sectors: off``: both
channels on the full grid of one device in the same driver (real orbitals, the periodic Hartree potential, the spin
functional of the cell), or, where they would not fit there, as one sector of the identity that the GPUs of the
process or the ranks of the runner share (``PARSEC_SPIN_FALLBACK_SHARED``, ``driver.SpinFallbackPlan``).  ``off`` keeps the route of before: the reference CLI's 1x1x1 k-point driver (complex Bloch
factors, the abelian little-group sectors of ``acceleration/SCF/kpoint_sectors.py``), bit for bit.

What stays on that route with ``auto`` too (:func:`gamma_spin_sector_refusal`): spin-orbit species and spinors, a
wire or slab of the Fortran-derived routines, k-points beyond Gamma, an oblique cell, a hybrid run, an LDA+U, a
meta-GGA, libxc or TS run that its switch keeps off the sectors (LDA+U also under ``PARSEC_LDAU_SPIN_SECTORS=off``;
the refusals of the unpolarized route:
:mod:`.gamma_features`; under ``auto`` a cell below 2^27 (a meta-GGA), 2^28 (libxc) or 2^26 (LDA+U) grid points x
states, and TS of every size, whose spin sectors were slower than this route), ``Spin_Symmetry_Sectors: off``, ``Symmetrize_Charge_Density``, the
occupations of ``occup.in`` and a ``Restart_Run``.  The command line (:func:`gamma_spin_on_sector_route`) also keeps
``--mpi`` (the k-point pools of ``main.py``) and a backend that does not resolve to CuPy on that route.  Andrew
Grieder's slab (``Boundary_Conditions: slab`` on the default routines, a 3-D grid with
``PeriodicCell.dimensions = 2``) passes these refusals as its unpolarized twin does, and since this route exists its
spin single point runs on the accelerated driver, on the full grid of one device (its symmetry detection refuses a
grid that is not periodic along three axes): three H atoms in such a slab, CA, gave the energy of the 1x1x1 k-point
driver of ``off`` to 1e-8 Ry on A100 (2026-11-19).

Forces and relaxations (``PARSEC_SPIN_SECTOR_FORCES``, :func:`spin_sector_forces_setting`): under ``auto`` (the
default since 2026-11-19) the single-point forces of such a cell and its relaxations by an optimizer (``bfgs``,
``simple``) take the sectors as the unpolarized cell does wherever its rule takes them
(:func:`parsec_python.acceleration.cli.periodic_sector_forces_refusal`: every atom movable, the default routines, no
point charges, field or supercell of a k-point mesh): the nonlocal force terms of both channels from their sector
states (``CuPySymmetrySCFEigensolver.spin_projector_force_terms``), the forces and the moved positions averaged over
the sector group, the guard of every structure and its full-grid fallback (a structure after the first starts that
full grid from the potentials of the one before).  Molecular dynamics, manual movements and ``Move_Flag`` fixing
atoms (and forces with ``Ignore_Symmetry`` or ``--symmetry off``) take the accelerated driver too, as for an
unpolarized cell: both channels on the full grid (no point group, no guard), planned from the input
(:func:`spin_full_grid_moves_plan`, ``driver.spin_full_grid_plan``): of one device where 4 times both bases fit it,
one sector of the identity whose basis the GPUs of the process share where they do not
(``PARSEC_SPIN_FALLBACK_SHARED``), the reference solver with a NOTE where neither holds them.  The
Fortran-derived routines and ``PARSEC_PERIODIC_SECTOR_FORCES=off`` keep the reference solver, as ``off`` keeps
every forces, relaxation and dynamics run of a spin cell there: its ionic loop is the 1x1x1 k-point run of the
device k-point solver on the Fortran-derived routines (``PARSEC_SPIN_IONIC_KPOINT``, since 2026-11-21) and the
Gamma-point loop of the host on the default routines.  The full-SCF runner
takes the forces and relaxations of spin-polarized runs on several ranks under ``auto`` where they keep their
sectors (clusters and these cells alike), and refuses the others (molecular dynamics among them), and every one
under ``off``, as before.  Units Ry, bohr.
"""

from __future__ import annotations

import os

__all__ = [
    "gamma_spin_sector_refusal",
    "gamma_spin_sector_setting",
    "gamma_spin_on_sector_route",
    "spin_sector_forces_setting",
]


def spin_sector_forces_setting() -> str:
    """``PARSEC_SPIN_SECTOR_FORCES``: ``auto`` (the default) or ``off`` (also ``0``, ``false``, ``no``).

    ``auto`` lets the forces and the relaxations (``bfgs``, ``simple``) of a collinear spin-polarized calculation
    keep the symmetry sectors of both channels: on the command line those of a bulk cell at the Gamma point
    (:func:`gamma_spin_on_sector_route`, instead of the 1x1x1 k-point driver), in the full-SCF runner on several
    ranks those of a cluster or such a cell.  ``off`` keeps the routes of before: the k-point driver for the forces,
    relaxations and dynamics of a spin cell on the command line, and the runner's refusal of the forces, relaxations
    and dynamics of every spin-polarized run on several ranks.
    """

    value = os.environ.get("PARSEC_SPIN_SECTOR_FORCES", "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "off"):
        raise ValueError(f"PARSEC_SPIN_SECTOR_FORCES must be auto or off, not {value!r}")
    return value


def gamma_spin_sector_setting() -> str:
    """``PARSEC_GAMMA_SPIN_SECTORS``: ``auto`` (the default) or ``off``.

    ``auto`` puts a collinear spin-polarized bulk cell at the Gamma point on the route of the Gamma sectors of the
    accelerated driver where :func:`gamma_spin_sector_refusal` allows it; ``off`` keeps it on the 1x1x1 k-point
    driver of the reference CLI (and refused by the route of the sectors and the MPI runner), as before.
    """

    value = os.environ.get("PARSEC_GAMMA_SPIN_SECTORS", "auto").strip().lower() or "auto"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in ("auto", "off"):
        raise ValueError("PARSEC_GAMMA_SPIN_SECTORS must be auto or off")
    return value


def gamma_spin_sector_refusal(problem) -> str | None:
    """Why the spin channels of this periodic input do not take the route of the Gamma sectors, or ``None``.

    Called for a spin-polarized input by :func:`parsec_python.acceleration.driver.periodic_sector_refusal`, which
    refuses the shapes, k-points and functionals of the unpolarized route after it.  ``None`` with
    ``PARSEC_GAMMA_SPIN_SECTORS=auto`` for collinear spin without spin-orbit species, ``Symmetrize_Charge_Density``,
    the occupations of ``occup.in`` and ``Spin_Symmetry_Sectors: off``; that last one names the full grid of the
    driver (the run that a firing guard prepares again), which this route does not take either.
    """

    scf = problem.scf
    if gamma_spin_sector_setting() == "off":
        return "spin polarization, spin-orbit coupling or non-collinear magnetism"
    if scf.noncollinear or scf.self_consistent_spin_orbit:
        return "spin-orbit coupling or non-collinear magnetism"
    if any(getattr(item, "spin_orbit", False) for item in getattr(problem, "pseudopotentials", {}).values()):
        return "spin polarization with spin-orbit species"
    if scf.is_hybrid:
        # the spin Fock loop of a cell keeps the 1x1x1 Bloch loop of the extended drivers (SCF/hybrid_periodic.py),
        # whatever the route of an unpolarized hybrid at the Gamma point
        return "spin polarization with a hybrid functional"
    if scf.spin_symmetry_sector_mode == "off":
        return "spin polarization with Spin_Symmetry_Sectors: off"
    if getattr(scf, "symmetrize_density", False):
        return "spin polarization with Symmetrize_Charge_Density"
    if getattr(scf, "fixed_occupations", None) is not None:
        return "spin polarization with the occupations of occup.in"
    return None


def _full_grid_moves(translation) -> bool:
    """Molecular dynamics, manual movements and a relaxation with fixed atoms (``Move_Flag``): moves that the
    accelerated driver solves on the full grid of a cell (no point group), spin-polarized as unpolarized."""

    if getattr(translation, "molecular_dynamics", None) is not None:
        return True
    relaxation = getattr(translation, "relaxation", None)
    if relaxation is None:
        return False
    if relaxation.method not in ("bfgs", "simple"):
        return True
    movable = getattr(translation, "movable", None)
    return movable is not None and not all(bool(flag) for flag in movable)


def _moves_or_forces(translation) -> bool:
    return bool(
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    )


def spin_full_grid_moves_plan(translation, *, symmetry_off: bool = False):
    """The memory plan of a spin cell whose forces or moves the accelerated driver solves on the full grid, or ``None``.

    Molecular dynamics, manual movements and ``Move_Flag`` fixing atoms (:func:`_full_grid_moves`), point charges,
    and forces or relaxations with ``Ignore_Symmetry`` or ``--symmetry off`` (``symmetry_off``) take no point group:
    both channels on the full grid.  :func:`parsec_python.acceleration.driver.spin_full_grid_plan` estimates it from
    the input: one device where it fits (``layout`` ``device``), one sector of the identity whose basis the GPUs of the
    process share where it does not (``shared``, ``PARSEC_SPIN_FALLBACK_SHARED``), and ``fits`` false where neither
    holds it, which keeps the 1x1x1 k-point driver (:func:`gamma_spin_on_sector_route`).  ``None`` for a single
    point, for forces in the sectors, and where the input does not tell.
    """

    if not _moves_or_forces(translation):
        return None
    if not (
        _full_grid_moves(translation)
        or symmetry_off
        or bool(getattr(translation.problem, "point_charges", ()))
    ):
        return None
    from ..driver import spin_full_grid_plan

    return spin_full_grid_plan(translation.problem)


def gamma_spin_on_sector_route(
    translation,
    *,
    mpi: bool,
    backend_is_cupy,
    supercell: bool = False,
    symmetry_off: bool = False,
    notes: list | None = None,
) -> bool:
    """Whether the command line runs this spin-polarized periodic input on the accelerated Gamma sectors.

    A single point of a collinear spin-polarized bulk cell at the Gamma point that the route of the sectors takes
    (:func:`parsec_python.acceleration.driver.periodic_sector_refusal` returns ``None``), without ``Restart_Run``, a
    band structure or a polarizability loop, in one process (``mpi`` false), on a backend that resolves to CuPy
    (``backend_is_cupy()``, called last).  Under ``PARSEC_SPIN_SECTOR_FORCES=auto`` also its forces and its
    relaxation by an optimizer where the forces of an unpolarized cell keep the sectors
    (:func:`parsec_python.acceleration.cli.periodic_sector_forces_refusal` returns ``None``), and its molecular
    dynamics, manual movements and relaxations with fixed atoms (the full grid of the driver, :func:`_full_grid_moves`),
    on the default routines, not under ``PARSEC_PERIODIC_SECTOR_FORCES=off``, never for the supercell
    of a k-point mesh (``supercell``).  ``False`` for every other input, which keeps its route of before (the
    reference CLI's 1x1x1 k-point driver for a spin-polarized cell).  A forces or moves run that the driver solves
    on the full grid (:func:`spin_full_grid_moves_plan`; ``symmetry_off``: ``Ignore_Symmetry`` or ``--symmetry off``)
    keeps that k-point driver too where the full grid of both channels fits neither one device nor the GPUs of the
    process together; the NOTE that says so goes to ``notes``.
    """

    problem = translation.problem
    if problem.periodic_cell is None or not getattr(problem.scf, "spin_polarized", False):
        return False
    if mpi or gamma_spin_sector_setting() == "off":
        return False
    # (read in every spin run, so that a value it does not know is refused)
    forces_setting = spin_sector_forces_setting()
    if (
        getattr(translation, "restart_run", False)
        or getattr(translation, "polarizability_field", None) is not None
        or getattr(problem, "bands", None) is not None
        or getattr(translation, "energy_refinement", None) is not None
    ):
        return False
    if (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    ):
        if forces_setting == "off" or supercell:
            return False
        if getattr(getattr(translation, "routines", None), "fortran", False) or getattr(
            problem.grid, "fortran_routines", False
        ):
            # The Fortran-derived routines (the default of the forces of a spin-polarized PBE cell, and of every
            # Cubic_Spline input): their sector forces have not been validated for two spin channels.
            return False
        from ..cli import periodic_sector_forces_refusal, periodic_sector_forces_setting

        if periodic_sector_forces_setting() == "off":
            # (every forces run of a cell keeps its route of before)
            return False
        if periodic_sector_forces_refusal(translation) is not None and not _full_grid_moves(translation):
            return False
    from ..driver import periodic_sector_refusal

    # one process: TS, a meta-GGA and libxc take the sectors under their switches (SCF/gamma_features.py)
    if periodic_sector_refusal(problem, features=True) is not None:
        return False
    if not backend_is_cupy():
        return False
    # The full grid of both channels from the start (no point group): where it fits nowhere on this process, the
    # k-point driver of before (its abelian little-group sectors hold it).
    plan = spin_full_grid_moves_plan(translation, symmetry_off=symmetry_off)
    if plan is not None and not plan.fits:
        if notes is not None:
            notes.append(plan.moves_note())
        return False
    return True
