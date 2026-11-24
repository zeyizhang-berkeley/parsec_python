"""Which periodic inputs of the round's features the accelerated driver runs, and which it routes.

The accelerated driver runs a Gamma-point SCF of a periodic cell on the
reference preparation (:func:`parsec_python.SCF.pbc.prepare_periodic_single_point`,
which hands a wire or slab to :mod:`parsec_python.SCF.wire_slab`) with its own
device eigensolver, and the Hartree solve of the prepared system
(:mod:`parsec_python.acceleration.Hartree.periodic_device` on the device).
That covers, at the Gamma point and without spin or forces (routed by their
own entries of :func:`parsec_python.acceleration.cli._reference_only_features`):

* ``Boundary_Conditions: wire`` / ``slab``, with charged sheets;
* ``Calc_Hartree_G`` (the spectral Hartree solve on the device);
* ``Periodic_Local_Potential: g_space`` and ``Periodic_Alpha_Term`` (in the
  reference preparation);
* ``Zero_Average_Vxc`` (the readable XC shifts its potential; the device
  evaluators are wrapped by :class:`parsec_python.acceleration.V_xc.device_xc.ZeroAverageXC`).

This module names what stays with the reference solver.
"""

from __future__ import annotations

import os

GENERAL_GRID_SWITCH = "PARSEC_GENERAL_GRID_ACCELERATED"
#: What the reference solver was handed for every bulk cell on Fortran's general grid before
#: ``PARSEC_GENERAL_GRID_ACCELERATED`` (and still is, in these words, under ``off``; an MPI run, a CPU backend and the
#: inputs :func:`general_grid_reference_only` names give their own reasons).
GENERAL_GRID_FEATURE = (
    "a periodic cell on Fortran's general grid (non-orthogonal lattice vectors, or sides without one common grid "
    "spacing)"
)


def _mpi_run() -> bool:
    from ..Parallel import mpi

    return mpi.requested()


def general_grid_accelerated_setting() -> str:
    """``PARSEC_GENERAL_GRID_ACCELERATED``: ``auto`` (the default) runs a bulk cell on Fortran's general grid (the
    Fortran-derived routines with oblique lattice vectors, or sides of different grid steps) at the Gamma point on
    the accelerated driver where :func:`general_grid_reference_only` names nothing: one process, unpolarized, a
    native LDA or GGA, single points, forces, relaxations and molecular dynamics.  ``off`` hands every such cell to
    the reference solver, as before."""

    value = os.environ.get(GENERAL_GRID_SWITCH, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{GENERAL_GRID_SWITCH} must be auto or off, not {value!r}")
    return value


def general_grid_reference_only(translation, *, on_device: bool = False) -> list[str]:
    """What keeps a bulk cell on Fortran's general grid on the reference solver (empty: the accelerated driver).

    Under ``PARSEC_GENERAL_GRID_ACCELERATED=off``, in an MPI run (whose sectors and shared full grid do not take that
    grid, ``driver.prepare_single_point``) and where the backend does not resolve to CuPy (``on_device`` false: the
    CPU backends, on which that route was not validated) the cell itself, as before.  Otherwise the accelerated driver
    runs the Gamma-point SCF of such a cell, its forces (``Forces/periodic_device.compute_device_periodic_forces``:
    the local terms of the general grid's potential, the nonlocal term of the projectors with their lattice
    images), relaxations and molecular dynamics, on the full grid or, under ``PARSEC_OBLIQUE_SECTORS``, in the
    sectors of the frame of its grid indices; these inputs, not validated there, stay with the reference solver:
    spin polarization and spinors, a hybrid functional, a meta-GGA or a libxc functional, LDA+U, the TS correction,
    ``Calc_Hartree_G`` and ``Periodic_Local_Potential: g_space``, ``Symmetrize_Charge_Density`` and ``Restart_Run``.
    A band structure (``bandstruc``) follows the accelerated SCF, as for the other Gamma-point cells
    (:mod:`..Output.post_scf`: the reference band solver on the converged potential of that SCF).  (k-points, the
    DOS, nscf, Output_GW and the like are named by their own entries.)
    """

    problem = translation.problem
    if general_grid_accelerated_setting() == "off":
        # (the words of before, which named every such cell)
        return [GENERAL_GRID_FEATURE]
    if _mpi_run():
        return ["a periodic cell on Fortran's general grid in an MPI run (--mpi)"]
    if not on_device:
        return [
            "a periodic cell on Fortran's general grid on a CPU backend (its accelerated route takes the CuPy "
            "backend)"
        ]
    from parsec_python.V_xc.libxc import is_libxc_functional

    scf = problem.scf
    names = []
    if scf.spin_polarized or scf.noncollinear or scf.self_consistent_spin_orbit or any(
        getattr(item, "spin_orbit", False) for item in problem.pseudopotentials.values()
    ):
        names.append("spin polarization or spin-orbit coupling")
    if getattr(scf, "is_hybrid", False):
        names.append("a hybrid functional")
    elif getattr(scf, "is_meta_gga", False) or is_libxc_functional(getattr(scf, "xc_functional", "ca")):
        names.append("a meta-GGA or libxc functional")
    if any(getattr(item, "has_hubbard", False) for item in problem.pseudopotentials.values()):
        names.append("LDA+U")
    if getattr(scf, "van_der_waals", False):
        names.append("the Tkatchenko-Scheffler correction")
    if getattr(problem.hartree, "periodic_g_space", False):
        names.append("Calc_Hartree_G")
    if getattr(problem.grid, "local_potential", "real_space") != "real_space":
        names.append("Periodic_Local_Potential: g_space")
    if getattr(scf, "symmetrize_density", False):
        names.append("Symmetrize_Charge_Density")
    if getattr(translation, "restart_run", False):
        names.append("Restart_Run")
    return [f"{name} of a periodic cell on Fortran's general grid" for name in names]


def wire_slab_reference_only(translation) -> list[str]:
    """The features of a wire or slab input that the accelerated driver does not run.

    A restart from ``parsec.dat`` (``Restart_Run``) of a wire or slab reads the
    stored fields through the reference CLI only, and an MPI run (``--mpi``)
    keeps the reference route it had before the accelerated one existed.
    """

    features = []
    if getattr(translation, "restart_run", False):
        features.append(
            "Restart_Run of a wire or slab (Boundary_Conditions: wire or slab)"
        )
    if _mpi_run():
        features.append(
            "a wire or slab in an MPI run (Boundary_Conditions: wire or slab with --mpi)"
        )
    return features


def cell_options_reference_only(translation) -> list[str]:
    """``Calc_Hartree_G`` and ``g_space`` of a 3-D cell that stay routed.

    Only an MPI run (``--mpi``), which keeps the reference route it had before.  ``Zero_Average_Vxc`` (PARSEC 1.5's
    potshift) is the default of a 3-D cell since 2026-11-08 and takes the route of the cell: every XC evaluation
    of both routes shifts its potential (the readable ``evaluate_xc`` and
    :class:`parsec_python.acceleration.V_xc.device_xc.ZeroAverageXC`).
    """

    problem = translation.problem
    if problem.periodic_cell is None or not _mpi_run():
        return []
    names = []
    if getattr(problem.hartree, "periodic_g_space", False):
        names.append("the reciprocal-space Hartree potential (Calc_Hartree_G)")
    if getattr(problem.grid, "local_potential", "real_space") != "real_space":
        names.append("Fortran's G-space local potential (Periodic_Local_Potential: g_space)")
    return [f"{name} in an MPI run (--mpi)" for name in names]


def moving_cell_options_reference_only(translation) -> list[str]:
    """Forces, relaxation and dynamics of a cell with the options above: on the reference solver.

    The accelerated forces of a periodic cell (``Forces/periodic_device.py``)
    are those of a bulk cell at the Gamma point on the readable or device
    fields of the real-space local potential, validated without these
    options (and with LDA+U and TS); the reference solver keeps the wire and
    slab forces and the forces of a cell with ``Calc_Hartree_G`` or ``g_space``.
    ``Zero_Average_Vxc`` (the default of a 3-D cell since 2026-11-08) moves on the
    accelerated path: its forces read the shifted potentials of the result through
    the reference formula (:func:`parsec_python.Forces.periodic.periodic_force_fields`),
    and a constant in V_xc adds nothing to them but the grid's egg-box term of
    the core density.
    """

    problem = translation.problem
    if problem.periodic_cell is None or not (
        getattr(translation, "compute_forces", False)
        or getattr(translation, "relaxation", None) is not None
        or getattr(translation, "molecular_dynamics", None) is not None
    ):
        return []
    names = []
    if getattr(problem.grid, "periodicity", 3) in (1, 2):
        names.append("a wire or slab (Boundary_Conditions: wire or slab)")
    if getattr(problem.hartree, "periodic_g_space", False):
        names.append("Calc_Hartree_G")
    if getattr(problem.grid, "local_potential", "real_space") != "real_space":
        names.append("Periodic_Local_Potential: g_space")
    # LDA+U (J = 0) and TS of a bulk cell at Gamma move on the accelerated
    # path: the LDA+U term from the orbitals on the device with the gradients
    # of the cell images (Hamiltonian/hubbard_device.py), the TS forces of the
    # device sums (Energy/dispersion_device.py), validated against the
    # reference forces, relaxations and dynamics.
    return [
        f"forces, relaxation or molecular dynamics of a periodic cell with {name}"
        for name in names
    ]


__all__ = [
    "GENERAL_GRID_FEATURE",
    "GENERAL_GRID_SWITCH",
    "cell_options_reference_only",
    "general_grid_accelerated_setting",
    "general_grid_reference_only",
    "moving_cell_options_reference_only",
    "wire_slab_reference_only",
]
