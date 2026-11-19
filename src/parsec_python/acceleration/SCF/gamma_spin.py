"""Spin-polarized bulk cells at the Gamma point in the symmetry sectors (``PARSEC_GAMMA_SPIN_SECTORS``).

A collinear spin-polarized bulk cell sampled at the Gamma point alone (a defect supercell: a vacancy, an NV-type
centre, an open-shell molecule in a box) has real orbitals, as its unpolarized twin.  With
``PARSEC_GAMMA_SPIN_SECTORS=auto`` (the default) the accelerated driver solves it on the route of the Gamma sectors
(:func:`parsec_python.acceleration.driver.periodic_sector_refusal`): the commuting involutions of the cell group with
their half-cell translations (:func:`~parsec_python.acceleration.Symmetry.axis_reflection.periodic_abelian_reduction`;
the weighted sectors of a non-abelian group are refused for two channels, as on clusters:
``driver._orbital_group_refusal``), both channels in those sectors with the symmetry guard of
:mod:`.spin_sectors` (``Spin_Symmetry_Sectors``; its density test takes the operations ``r -> R r + t`` of the sector
group and of the cell, about a defect off the origin as well: ``PARSEC_SPIN_GUARD_CELL_GROUP``), the fields of both
channels on the orbits, the Hartree potential of
the total density solved on the full grid of the cell (``SCF.pbc.solve_periodic_hartree``) and reduced again, and the
spin functional on the orbits.  A run whose guard fires is prepared again with ``Spin_Symmetry_Sectors: off``: both
channels on the full grid of one device in the same driver (real orbitals, the periodic Hartree potential, the spin
functional of the cell).  ``off`` keeps the route of before: the reference CLI's 1x1x1 k-point driver (complex Bloch
factors, the abelian little-group sectors of ``acceleration/SCF/kpoint_sectors.py``), bit for bit.

What stays on that route with ``auto`` too (:func:`gamma_spin_sector_refusal`): spin-orbit species and spinors, a
wire or slab, k-points beyond Gamma, an oblique cell, a hybrid, meta-GGA, libxc, LDA+U or TS run (the refusals of
the unpolarized route), ``Spin_Symmetry_Sectors: off``, ``Symmetrize_Charge_Density``, the occupations of
``occup.in`` and a ``Restart_Run``.  The command line (:func:`gamma_spin_on_sector_route`) also keeps forces,
relaxations and molecular dynamics, ``--mpi`` (the k-point pools of ``main.py``) and a backend that does not resolve
to CuPy on that route.  Units Ry, bohr.
"""

from __future__ import annotations

import os

__all__ = [
    "gamma_spin_sector_refusal",
    "gamma_spin_sector_setting",
    "gamma_spin_on_sector_route",
]


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


def gamma_spin_on_sector_route(translation, *, mpi: bool, backend_is_cupy) -> bool:
    """Whether the command line runs this spin-polarized periodic input on the accelerated Gamma sectors.

    A single point (no forces, relaxation or molecular dynamics) of a collinear spin-polarized bulk cell at the
    Gamma point that the route of the sectors takes (:func:`parsec_python.acceleration.driver.periodic_sector_refusal`
    returns ``None``), without ``Restart_Run``, a band structure or a polarizability loop, in one process (``mpi``
    false), on a backend that resolves to CuPy (``backend_is_cupy()``, called last).  ``False`` for every other input,
    which keeps its route of before (the reference CLI's 1x1x1 k-point driver for a spin-polarized cell).
    """

    problem = translation.problem
    if problem.periodic_cell is None or not getattr(problem.scf, "spin_polarized", False):
        return False
    if mpi or gamma_spin_sector_setting() == "off":
        return False
    if (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
        or getattr(translation, "restart_run", False)
        or getattr(translation, "polarizability_field", None) is not None
        or getattr(problem, "bands", None) is not None
        or getattr(translation, "energy_refinement", None) is not None
    ):
        return False
    from ..driver import periodic_sector_refusal

    if periodic_sector_refusal(problem) is not None:
        return False
    return bool(backend_is_cupy())
