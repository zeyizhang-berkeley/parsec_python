"""``PARSEC_FORTRAN_NUMERICS``: the numerics of PARSEC 1.5 in one switch, for comparisons with Fortran PARSEC.

Several defaults of this package depart from PARSEC 1.5 on purpose, because they make energies or forces more
accurate: the cubic spline of a species without ``Cubic_Spline``, projectors that end at the channel cutoffs, the exact
quadrature of the Kleinman-Bylander denominators, a continuous Perdew-Zunger correlation, the Hartree potential of a
slab that makes its energy stationary, the controlled Hartree boundary values of a cluster, and force rules next to an
atom and for the Gaussian charges of wires and slabs; the converged band solves, the Fermi level of a gapped system
at the root of its electron count, FHI tables continued below their first radial sample, the Coulomb tail of a
splined local potential beyond its Coulomb onset, the local potential and projectors Fourier-filtered to the grid,
smooth LDA+U projectors, grid points exactly on a cluster's sphere decided alike for all images under the operations
of the grid, and (opt-in) a later-SCF filter degree that returns where a loop stalls and the exact gradient of the
superposed atomic densities in the density gradient beside a nucleus.  ``on`` gives every one of them the value of
PARSEC 1.5 at once (:data:`FORTRAN_NUMERICS_VALUES`, and for a cluster or slab whose input gives its
domain the defaults ``Hartree_Boundary_Tolerance: off`` and ``Hartree_Atomic_Tail: off`` of
:func:`fortran_boundary_defaults`).  A switch set in the environment, or a keyword in the input, wins over the
umbrella.  ``off`` (the default) leaves every default as it is.

The switches of a setting read their value through :func:`switch_value`, so the umbrella reaches every route that
reads them (both command lines, the MPI runners, every rank alike).  An unknown value is refused when the input is
read, before the SCF.
"""

from __future__ import annotations

import os

FORTRAN_NUMERICS_ENV = "PARSEC_FORTRAN_NUMERICS"
FORTRAN_NUMERICS_DEFAULT = "off"

#: The value of PARSEC 1.5 of every default that departs from it for accuracy (the README switch table, one row each).
FORTRAN_NUMERICS_VALUES = {
    # linear radial tables for a species without Cubic_Spline (usrinputfile.F90's .false.)
    "PARSEC_CUBIC_SPLINE_DEFAULT": "off",
    # the tabulated V_l - V_local beyond the channel cutoffs (pseudo.f90)
    "PARSEC_PROJECTOR_TAIL": "off",
    # the weights 0.5 (r(i+1) - r(i-1)) of the Kleinman-Bylander denominators (pseudo.f90)
    "PARSEC_KB_QUADRATURE": "trapezoid",
    # the four-digit high-density coefficients of the Perdew-Zunger correlation (exc.f90)
    "PARSEC_CA_CONTINUITY": "off",
    # the potential of the Poisson solve in a slab with forces (and in a cluster)
    "PARSEC_HARTREE_VARIATIONAL": "off",
    # the projector gradients of nonloc.F90 within the first radial sample
    "PARSEC_PROJECTOR_CENTER": "tip",
    # the local-force skip of forcloc.f90 within the first radial sample
    "PARSEC_SPLINE_FORCE_CENTER": "tip",
    # forcloc.f90's skip of every local term of a grid point within rs(2) of an atom of a wire or slab
    "PARSEC_WIRE_SLAB_GAUSSIAN": "off",
    # the potential of the Gaussian charges of a wire or slab solved at the Hartree tolerance of the run (hpotcg)
    "PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE": "off",
    # the finite-difference gradient of V_H in the force of the Gaussian charges (forcloc_slab.f90)
    "PARSEC_WIRE_SLAB_GAUSSIAN_FORCE": "gradient",
    # the fixed Bands_FF_MaxIter cycles of the band and nscf k-points
    "PARSEC_BAND_CONVERGENCE": "off",
    # the working buffer nstate + winsize of calc_bands
    "PARSEC_BAND_BUFFER": "off",
    # the Fermi level of a gapped system where the bisection of flevel.f90 meets the electron count
    # (Occupations/fermi_dirac.py, fermi_gap_root_setting, which reads it through switch_value)
    "PARSEC_FERMI_GAP_ROOT": "off",
    # an FHI table between the atom and its first radial sample as pseudo.f90 reads it
    # (Pseudopotential/fhipp.py, fhi_origin_setting, which reads it through switch_value)
    "PARSEC_FHI_ORIGIN": "off",
    # the later-SCF Chebyshev degree of PARSEC 1.5, never raised in a loop (its default as well; listed so that the
    # umbrella keeps PARSEC's rule where another default changes; SCF/filter_degree.py, filter_degree_stall_setting,
    # which reads it through switch_value)
    "PARSEC_FILTER_DEGREE_STALL": "off",
    # the cubic spline of a local potential out to rs(ns-1), as ionpot.f90 evaluates a species with Cubic_Spline
    # (Pseudopotential/potre.py, spline_coulomb_tail_setting, which reads it through switch_value); a species without
    # Cubic_Spline takes linear tables under the umbrella anyway (PARSEC_CUBIC_SPLINE_DEFAULT above)
    "PARSEC_SPLINE_COULOMB_TAIL": "off",
    # the local potential and the projectors as the tables give them, unfiltered (Pseudopotential/local_filter.py,
    # local_filter_setting, which reads it through switch_value)
    "PARSEC_LOCAL_FILTER": "off",
    # the LDA+U projectors of upot.F90, interpolated linearly and cut at r_c (Hamiltonian/hubbard.py,
    # ldau_smooth_projectors_fraction, which reads it through switch_value)
    "PARSEC_LDAU_SMOOTH_PROJECTORS": "off",
    # the grid points exactly on a cluster's sphere as outside_domain (grid_partition.f90) decides them,
    # (x^2 + y^2) + z^2 against R*R (Grid/cluster.py, sphere_ties_setting, which reads it through switch_value)
    "PARSEC_SPHERE_TIES": "fortran",
    # the finite-difference density gradient of exc.f90 at every point, also beside a nucleus (V_xc/nuclear_gradient.py,
    # nuclear_gradient_setting, which reads it through switch_value; off is its default as well)
    "PARSEC_GGA_NUCLEAR_GRADIENT": "off",
}


def fortran_numerics_setting() -> str:
    """``PARSEC_FORTRAN_NUMERICS``: ``off`` (default) or ``on`` (the module docstring)."""

    value = os.environ.get(FORTRAN_NUMERICS_ENV, FORTRAN_NUMERICS_DEFAULT).strip().lower() or FORTRAN_NUMERICS_DEFAULT
    if value not in ("off", "on"):
        raise ValueError(f"{FORTRAN_NUMERICS_ENV} must be off or on, not {value!r}")
    return value


def switch_value(name: str, default: str) -> str:
    """The value of the switch ``name`` (stripped, lower case): its own value where the environment sets one, else
    PARSEC 1.5's under ``PARSEC_FORTRAN_NUMERICS=on`` where :data:`FORTRAN_NUMERICS_VALUES` lists it, else
    ``default``."""

    value = os.environ.get(name, "").strip().lower()
    if value:
        return value
    if name in FORTRAN_NUMERICS_VALUES and fortran_numerics_setting() == "on":
        return FORTRAN_NUMERICS_VALUES[name]
    return default


def fortran_boundary_defaults() -> bool:
    """Whether the Hartree boundary keywords default to PARSEC's boundary values (``Hartree_Boundary_Tolerance: off``,
    ``Hartree_Atomic_Tail: off``): under ``PARSEC_FORTRAN_NUMERICS=on``, for an input that gives its domain."""

    return fortran_numerics_setting() == "on"


__all__ = [
    "FORTRAN_NUMERICS_ENV",
    "FORTRAN_NUMERICS_VALUES",
    "fortran_boundary_defaults",
    "fortran_numerics_setting",
    "switch_value",
]
