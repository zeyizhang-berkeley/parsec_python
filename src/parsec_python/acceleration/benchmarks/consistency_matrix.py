"""Consistency matrix: physical invariants of the accelerated command line across its routes, on GPUs.

Every case runs the command line (``parsec_python.acceleration.cli``) in fresh processes, one GPU each, and prints
one line

    CONSIST <group> <case> PASS|FAIL|ERROR <numbers>

followed by a summary that repeats every FAIL and ERROR.  The inputs are written by this file from tables of the
repository, so that the matrix can be run unchanged on any later tree.  Groups:

``force``
    The force on an atom against the slope of the SCF free energy ``E - TS`` (central differences at steps of 1e-3
    and 2e-3 bohr, SCF to 1e-10 Ry; the Richardson value of the two is printed too).  Tolerance 1e-5 Ry/bohr at both
    steps.  Atoms on a grid point, 5e-4 bohr from one and 0.05 bohr off a site; C (POTRE and FHI), Si, Ge (core
    correction), Ag (d channel), Mg (core correction), O, H, B, N; both routine sets; clusters, Gamma cells, a 2x2x2
    mesh, two slabs (one under PARSEC_HARTREE_VARIATIONAL=off) and a wire; spin, spin-orbit (a Gamma cell, and since
    308edad4e AuH and Au2 clusters on the device, on the whole grid and in their double-group sectors) and LDA+U;
    SCAN in the weighted sectors of a Gamma-point cell (added after 308edad4e); Au with PBE on a grid point at steps
    of 2e-5 and 4e-5 bohr (``FINE_STEPS``: the GGA energy varies on a scale of 1e-3 bohr there, see ``au2_gga``).
    The force
    compared is the total before its net part was removed (``forces_total + forces_net / N``), the derivative of the
    energy.  ``dgrid`` is the distance of the atom from the nearest grid point where the archive holds the grid.
``mpi``
    One case per launch under ``srun`` or ``mpirun`` (2 or 4 ranks, one GPU each): the MPI full-SCF runner for a
    cluster, a Gamma cell and (added 2026-11-23) the SCF of a PBE0 cell below and above the floor of the sectors of
    a Gamma-point hybrid, ``--mpi`` of the command line for a k-point mesh.  Rank 0 then runs the same input in
    one process and compares: energies within 1e-8 Ry, forces within 1e-7 Ry/bohr.
``symmetry``
    The default route against the same input without symmetry: ``--symmetry off`` with
    ``PARSEC_PERIODIC_SYMMETRY=off`` and ``PARSEC_KPOINT_SECTORS=off``; a Monkhorst-Pack mesh against the whole
    mesh written as a ``Kpoint_List`` with every shortest image of each point (equal shares of its weight), or
    against the same mesh without sectors.  SCF to 1e-9 Ry; energies within 1e-7 Ry, forces within 1e-5 Ry/bohr.
``switch``
    The off values of the switches published after 3b5d134c0 (``ALL_OFF`` below, all at once) against the result of
    3b5d134c0 itself pinned in ``tests/data/consistency_pins.json`` (``--write-pins`` records it): ``bits`` where equal,
    otherwise within 1e-9 Ry and 1e-8 Ry/bohr (GPU noise).  A later tree that publishes new switches adds their off
    values to ``ALL_OFF``; the pins stay.  Where a later tree changes a table a case reads, the pin is recomputed by
    3b5d134c0 on the new table (``w_c7n_spin_*``: the cubic ``spin_ca_*_POTRE.DAT`` of the test data).
``repeat``
    The same input twice in fresh processes: energies within 1e-10 Ry and the same number of SCF steps.  The spread
    is printed.
``traj``
    Three steps of a BFGS relaxation and of molecular dynamics (seeded velocities) of a cluster and a cell: the
    default route against ``--symmetry off`` (the dynamics in the sectors of ``PARSEC_MOVING_SECTORS=on``, which the
    default takes from 2^24 entries on) and against itself in a fresh process; every structure's energy within 1e-7 Ry
    and its positions within 1e-6 bohr.
``band``
    A short band path after the SCF (Si8, a spin cell, and a 64-atom cube above the floor of the k-point sectors):
    the default route against no symmetry, every level within 1e-7 Ry (with the SCF energy and forces as above).
``kpoint``
    A mesh the guard of PARSEC_KPOINT_MESH_GUARD treats (primitive bcc Na and spin-orbit bcc Au on the grids that
    choose the Fortran-derived routines) against the whole mesh written in [0, 1); the real solves of Gamma and of
    the time-reversal invariant points (PARSEC_KPOINT_REAL_GAMMA, PARSEC_KPOINT_TRIM_REAL, with
    PARSEC_KPOINT_KINETIC=trim) and the k-point sectors against their off values, on a basis above their floors.
``soc``
    Self-consistent spin-orbit clusters in the sectors that PARSEC_SOC_SECTORS forces at any size (``kramers``,
    ``on``) against the whole grid (``off``); a spin-orbit cell with a 2 x 2 x 2 mesh and forces in the double-group
    sectors of its k-points that ``PARSEC_SOC_KPOINT_SECTORS=on`` forces at any size, against ``off``; and (added
    after 308edad4e) the energy and forces of AuH in the Kramers sector on the device against those of the
    reference solver (``PARSEC_SOC_DEVICE_FORCES=off``) on both routine sets, forces within 1e-7 Ry/bohr.
The groups ``traj``, ``band``, ``kpoint`` and ``soc`` and the MPI band case were added after the first run of the
matrix (on ed4144f13); ``--group all`` takes them.  The MPI cases also run on several nodes (the line names them).

Usage (from the root of the repository, on a node whose GPUs the process may use)::

    python src/parsec_python/acceleration/benchmarks/consistency_matrix.py --group force [--cases 'f_c8*']
    python src/parsec_python/acceleration/benchmarks/consistency_matrix.py --list
    srun -n 4 python src/parsec_python/acceleration/benchmarks/consistency_matrix.py --group mpi --cases m_h2o

The run folders go to ``--out`` (default ``$CONSISTENCY_OUT`` or ``consistency_out``); each keeps its ``parsec.in``,
``parsec.out`` where a run failed or was a base run, and ``results.json`` the numbers of every run.  ``--prepare``
writes the inputs only.  Environment of every run: ``OMP_NUM_THREADS`` of the parent split over the GPUs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import fnmatch
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PINS = ROOT / "src" / "parsec_python" / "tests" / "data" / "consistency_pins.json"
#: ``tempkry`` of PARSEC's ``const.f90``: one kelvin in Ry.
KELVIN_RY = 6.33327186e-06
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
CLI_CODE = "import sys; from parsec_python.acceleration import cli; sys.exit(cli.main(sys.argv[1:]))"
#: The variables of an MPI launch that a child process must not inherit (it would join the parent's world).
MPI_VARIABLES = (
    "PARSEC_MPI", "PMI_RANK", "PMI_SIZE", "PMI_FD", "PMI_JOBID", "PMIX_RANK", "PMIX_NAMESPACE", "PMIX_SERVER_URI2",
    "PMIX_SERVER_URI3", "PMIX_SERVER_URI4", "PMIX_SERVER_URI21", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS",
    "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
)

# ---------------------------------------------------------------------------------------------------------------
# Tolerances
FORCE_TOLERANCE = 1.0e-5
FORCE_STEPS = (1.0e-3, 2.0e-3)
#: The steps of the force cases whose energy varies on a scale below FORCE_STEPS (a GGA at a grid point on a nucleus
#: of low pseudo-valence density: au2_gga)
FINE_STEPS = (2.0e-5, 4.0e-5)
MPI_ENERGY_TOLERANCE = 1.0e-8
MPI_FORCE_TOLERANCE = 1.0e-7
SYMMETRY_ENERGY_TOLERANCE = 1.0e-7
SYMMETRY_FORCE_TOLERANCE = 1.0e-5
SWITCH_ENERGY_TOLERANCE = 1.0e-9
SWITCH_FORCE_TOLERANCE = 1.0e-8
REPEAT_ENERGY_TOLERANCE = 1.0e-10
TRAJECTORY_ENERGY_TOLERANCE = 1.0e-7
TRAJECTORY_POSITION_TOLERANCE = 1.0e-6
BAND_TOLERANCE = 1.0e-7
OFFSET_DIRECTION = np.array([1.0, 0.7, 0.4]) / np.linalg.norm([1.0, 0.7, 0.4])

#: The off value of every switch published after 3b5d134c0, and the former default of every switch whose default
#: changed after it.  Each restores the bits of 3b5d134c0 on the routes it covers.
ALL_OFF = {
    "PARSEC_BAND_CONVERGENCE": "off",
    "PARSEC_BAND_REPORTED_RESIDUALS": "off",
    "PARSEC_BFGS_FIRST_STEP": "off",
    "PARSEC_CORE_FORCE_XC": "shifted",
    "PARSEC_CUPY_EXCHANGE_COLUMN_UPDATE": "off",
    "PARSEC_CUPY_SECTOR_WARM_GROWTH": "off",
    "PARSEC_FIXED_STATES_BUFFER": "off",
    "PARSEC_GAMMA_HYBRID_SECTOR_FORCES": "off",
    "PARSEC_HYBRID_IMAGE_PAIRS": "off",
    "PARSEC_HYBRID_KPOINT_FORCE_WEDGE": "off",
    "PARSEC_HYBRID_WEDGE_SECTOR_STATES": "off",
    "PARSEC_KPOINT_FOLD": "centred",
    "PARSEC_KPOINT_FORCE_GROUP": "wedge",
    "PARSEC_KPOINT_LIST_FOLD": "literal",
    "PARSEC_KPOINT_MESH_GUARD": "off",
    "PARSEC_KPOINT_REAL_GAMMA": "off",
    "PARSEC_KPOINT_SUPERCELL_LEVELS": "off",
    "PARSEC_KPOINT_TRIM_REAL": "off",
    "PARSEC_LDAU_PROJECTOR_ORBITS": "off",
    "PARSEC_LDAU_SECTOR_FORCES": "off",
    "PARSEC_LDAU_SMOOTH_PROJECTORS": "off",
    "PARSEC_LDAU_SPIN_FULL_GRID": "off",
    "PARSEC_LDAU_SPIN_SECTORS": "off",
    "PARSEC_LDAU_WEIGHTED_SECTORS": "off",
    "PARSEC_MOVE_FLAG_SPACE_GROUP": "off",
    "PARSEC_MOVING_SECTORS": "off",
    "PARSEC_ORBITAL_GROUP_META_GGA": "off",
    "PARSEC_ORBITAL_GROUP_MPI": "off",
    "PARSEC_ORBITAL_GROUP_SPIN_FLOOR": "off",
    "PARSEC_PERIODIC_IMAGE_ROWS": "off",
    "PARSEC_PERIODIC_SECTOR_DEVICES": "all",
    "PARSEC_PERIODIC_SPIN_SECTOR_FLOOR": "off",
    "PARSEC_PROJECTOR_CENTER": "off",
    "PARSEC_SCALAR_KPOINT_WEDGE": "grid",
    "PARSEC_SECTOR_ORBITS_EXACT": "off",
    "PARSEC_SECTOR_OVERHEAD": "off",
    "PARSEC_SPIN_GUARD_DOUBLETS": "off",
    "PARSEC_SPIN_IONIC_KPOINT": "off",
    "PARSEC_SPIN_START": "weighted",
    "PARSEC_SPLINE_FORCE_CENTER": "off",
    "PARSEC_SYMMETRY_KEYWORD_ROUTINES": "fortran",
    "PARSEC_WEIGHTED_GUARD_SHORTCUT": "off",
    "PARSEC_WIRE_SLAB_GAUSSIAN": "off",
    # defaults changed after 3b5d134c0
    "PARSEC_WEIGHTED_FIRST_SHARE": "even",
    "PARSEC_GAMMA_TS_SECTORS": "off",
    "PARSEC_GAMMA_LDAU_SECTORS": "off",
    "PARSEC_SOC_SECTORS": "off",
    # published after ed4144f13 (PARSEC_SPLINE_FORCE_CENTER, PARSEC_CORE_FORCE_XC, PARSEC_SPIN_GUARD_DOUBLETS and
    # PARSEC_WIRE_SLAB_GAUSSIAN keep the values above, which restore 3b5d134c0; their values that restore ed4144f13
    # are tip, shifted, note and auto)
    "PARSEC_CUPY_COMPILE_ONCE": "off",
    "PARSEC_CUPY_SECTOR_WARM_GROWTH_HOST": "process",
    "PARSEC_MPI_SECTOR_POOL_MIN_GAIN": "0.05",
    "PARSEC_MOVING_SPIN_SECTORS": "off",
    "PARSEC_WEIGHTED_GUARD_SHORTCUT_TRANSLATIONS": "off",
    "PARSEC_ORBITAL_IRREP_ROWS": "sector",
    "PARSEC_WIRE_SLAB_GAUSSIAN_SOLVE": "off",
    "PARSEC_WIRE_SLAB_GAUSSIAN_FORCE": "gradient",
    "PARSEC_KPOINT_LIST_FORCE_GROUP": "off",
    "PARSEC_KPOINT_WAVEFUNCTIONS": "off",
    "PARSEC_GAMMA_PARSEC_DAT": "off",
    "PARSEC_KPOINT_SECTOR_CLASSES": "off",
    "PARSEC_ORDERED_SUMS": "off",  # its own sums are not bit-repeatable (cuSPARSE): GPU noise, not bits
    "PARSEC_LONG_ROW_SPLIT": "off",
    "PARSEC_MPI_ACCELERATED_BANDS": "off",
    "PARSEC_CONFINED_SETUP": "readable",
    "PARSEC_PERIODIC_OBLIQUE_SPLIT": "auto",
    "PARSEC_KPOINT_GUARD_WEDGE": "crystal",
    "PARSEC_KPOINT_ORTHOGONAL_LIST_FOLD": "off",
    "PARSEC_WEIGHTED_STATE_COUNT": "off",
    "PARSEC_BAND_BUFFER": "off",
    "PARSEC_SPLINE_SETUP": "readable",
    "PARSEC_SOC_PROJECTOR_SPLINE": "off",
    "PARSEC_SPINOR_MOVE_FLAG_SUBGROUP": "off",
    "PARSEC_RUNNER_FORTRAN_FIXED_SECTORS": "off",
    # defaults changed after 30edaacf1: the cubic spline of a species without Cubic_Spline, the projector tail
    "PARSEC_CUBIC_SPLINE_DEFAULT": "off",
    "PARSEC_PROJECTOR_TAIL": "off",
    "PARSEC_EXCHANGE_OCCUPATION_CUT": "off",
    "PARSEC_FOCK_DIIS_TAIL": "off",
    # defaults changed after b72375f60: the continuous Perdew-Zunger correlation, the exact quadrature of the KB
    # denominators, the Hartree potential of a slab with forces
    "PARSEC_CA_CONTINUITY": "off",
    "PARSEC_KB_QUADRATURE": "trapezoid",
    "PARSEC_HARTREE_VARIATIONAL": "off",
    # after b72375f60: the Fermi level of a gap at the root of the electron count, FHI tables continued below their
    # first radial sample
    "PARSEC_FERMI_GAP_ROOT": "off",
    "PARSEC_FHI_ORIGIN": "off",
    # after a6401263f: one sector of the identity keeps the states of the full grid
    "PARSEC_IDENTITY_SECTOR_STATES": "off",
    # after a6401263f: the keywords of the files a run reads or writes choose no routines
    "PARSEC_FILE_KEYWORD_ROUTINES": "fortran",
    # after a6401263f: the exchange of periodic hybrids on the ranks of the MPI runner takes the routes of one process
    "PARSEC_MPI_EXCHANGE_POOL": "off",
    # defaults changed after a6401263f: the complex characters of the abelian little groups of the k-points
    "PARSEC_KPOINT_SECTOR_CHARACTERS": "real",
    # defaults changed after a6401263f: the sectors of a wire or slab above the size floor
    "PARSEC_SLAB_WIRE_SECTORS": "off",
    # after a6401263f: a Gamma-point cell on Fortran's general grid on the accelerated driver of the command line (off:
    # the reference solver), and the sectors of oblique cells by default (off: the full grid, the default before)
    "PARSEC_GENERAL_GRID_ACCELERATED": "off",
    "PARSEC_OBLIQUE_SECTORS": "off",
    # after a6401263f: the starting later-SCF degree back where a loop stalls after lowering it
    "PARSEC_FILTER_DEGREE_STALL": "off",
    # after a6401263f: the local potential and the projectors Fourier-filtered to the grid
    "PARSEC_LOCAL_FILTER": "off",
    # after a6401263f: the k-point sectors of meta-GGA loops (off: one task per k-point)
    "PARSEC_KPOINT_SECTOR_META_GGA": "off",
    # defaults changed after 308edad4e: the wedge of a hybrid's Monkhorst-Pack mesh from the semilocal start, metals
    # too (auto: the start on the time-reversal pairs, the wedge after it only where no state is partly filled)
    "PARSEC_HYBRID_KPOINT_WEDGE": "auto",
    # after a6401263f: the Coulomb tail of a splined local potential beyond its Coulomb onset
    "PARSEC_SPLINE_COULOMB_TAIL": "off",
    # after 308edad4e: the full hexagonal and cubic groups of oblique cells without a density gradient
    "PARSEC_OBLIQUE_FULL_GROUP": "off",
    # after 308edad4e: the weighted sectors of the meta-GGA of a Gamma-point cell, and its floor of 2^24 entries
    "PARSEC_ORBITAL_GROUP_META_GGA_CELL": "off",
    "PARSEC_GAMMA_META_GGA_FLOOR": "former",
    # after 308edad4e: the forces, relaxations and dynamics of self-consistent spin-orbit clusters on the device
    # (off: the reference solver)
    "PARSEC_SOC_DEVICE_FORCES": "off",
    # after 308edad4e: the order of the squares in the sphere test of a cluster grid (dot, the default, as before)
    "PARSEC_SPHERE_TIES": "dot",
    # after 308edad4e: the double-group sectors of the k-points of a spin-orbit loop
    "PARSEC_SOC_KPOINT_SECTORS": "off",
    # after 308edad4e: the double-group sectors of the k-points of a spin-polarized spin-orbit loop (opt-in on)
    "PARSEC_SOC_SPIN_KPOINT_SECTORS": "off",
    # after 308edad4e: the weighted sectors of a cluster's hybrid (off: the abelian group)
    "PARSEC_ORBITAL_GROUP_HYBRID": "off",
    # after 308edad4e: the k-point sectors of one process with several GPUs in worker processes (off: the threads, the
    # floor of a meta-GGA on several GPUs), and the k-point loops of LDA+U in the sectors (off: one task per k-point)
    "PARSEC_KPOINT_SECTOR_PROCESSES": "off",
    "PARSEC_KPOINT_SECTOR_LDAU": "off",
    # after 308edad4e: the kernels of a run compiled ahead on background threads (the same binaries)
    "PARSEC_CUPY_COMPILE_AHEAD": "off",
}
NO_SYMMETRY_ENV = {"PARSEC_PERIODIC_SYMMETRY": "off", "PARSEC_KPOINT_SECTORS": "off"}
NO_SYMMETRY_ARGS = ("--symmetry", "off")

# ---------------------------------------------------------------------------------------------------------------
# Species: (file in the repository, file name in the run folder, block of the input)

PP_FILES = {
    "C_POTRE": "examples/3d_diamond_benchmark/C_POTRE.DAT",
    "C_FHI": "examples/3d_hybrid_functionals/C_FHIPP.DAT",
    "Si_POTRE": "src/parsec_python/tests/data/Si_POTRE.DAT",
    "Si_FHI": "examples/3d_hybrid_functionals/Si_FHIPP.DAT",
    "Ge_POTRE": "src/parsec_python/tests/data/Ge_POTRE.DAT",
    "Gep_POTRE": "src/parsec_python/tests/data/Ge_POTRE.DAT",
    "Ag_POTRE": "src/parsec_python/tests/data/Ag_POTRE.DAT",
    "Agn_POTRE": "src/parsec_python/tests/data/Ag_POTRE.DAT",
    "H_POTRE": "examples/0d_benzene/H_POTRE.DAT",
    "Hw_POTRE": "examples/0d_water_geoopt/H_POTRE.DAT",
    "Ow_POTRE": "examples/0d_water_geoopt/O_POTRE.DAT",
    "O_FHI": "examples/3d_hybrid_functionals/O_FHIPP.DAT",
    "Mg_FHI": "examples/3d_hybrid_functionals/Mg_FHIPP.DAT",
    "B_FHI": "examples/3d_hybrid_functionals/B_FHIPP.DAT",
    "Na_POTRE": "src/parsec_python/tests/data/Na_POTRE.DAT",
    "Nap_POTRE": "src/parsec_python/tests/data/Na_POTRE.DAT",
    "Au_POTRE": "examples/0d_AuH/Au_POTRE.DAT",
    "Auso_POTRE": "examples/0d_AuH/Au_POTRE.DAT",
    "Hso_POTRE": "examples/0d_AuH/H_POTRE.DAT",
    "Oa_FHI": "examples/3d_hybrid_functionals/O_FHIPP.DAT",
    "Ob_FHI": "examples/3d_hybrid_functionals/O_FHIPP.DAT",
    "Bx_POTRE": "examples/0d_xps_core_levels/pseudopotentials/B_POTRE.DAT",
    "Nx_POTRE": "examples/0d_xps_core_levels/pseudopotentials/N_POTRE.DAT",
    "N_FHI": "examples/3d_hybrid_functionals/N_FHIPP.DAT",
    "Cs_POTRE": "src/parsec_python/tests/data/spin_ca_C_POTRE.DAT",
    "Ns_POTRE": "src/parsec_python/tests/data/spin_ca_N_POTRE.DAT",
}


def _fhi(symbol: str, radius: float, electrons: str, core: bool = False) -> str:
    return (
        f"Atom_Type: {symbol}\nPseudopotential_Format: fhipp\nPotential_Num: 3\nLocal_Component: d\n"
        f"Core_Cutoff_Radius: {radius}\nbegin Electron_Per_Orbital\n{electrons}\nend Electron_Per_Orbital\n"
        "Cubic_Spline: .true.\n" + ("Nonlinear_Core_Correction: .true.\n" if core else "")
    )


def _potre(symbol: str, local: str, element: bool = False) -> str:
    return (
        f"Atom_Type: {symbol}\n" + (f"Element_Symbol: {symbol}\n" if element else "")
        + f"Pseudopotential_Format: martins_new\nLocal_Component: {local}\nCubic_Spline: true\n"
    )


SPECIES = {
    "C_POTRE": ("C_POTRE.DAT", _potre("C", "d")),
    "C_FHI": ("C_FHIPP.DAT", _fhi("C", 1.6, "2 2 0")),
    "Si_POTRE": ("Si_POTRE.DAT", _potre("Si", "p")),
    "Si_FHI": ("Si_FHIPP.DAT", _fhi("Si", 2.1, "2 2 0")),
    "Ge_POTRE": ("Ge_POTRE.DAT", _potre("Ge", "d")),
    "Gep_POTRE": ("Ge_POTRE.DAT", _potre("Ge", "p")),
    "Ag_POTRE": ("Ag_POTRE.DAT", _potre("Ag", "s", element=True)),
    # (without Element_Symbol, which the reader takes for every species: beside Au)
    "Agn_POTRE": ("Ag_POTRE.DAT", _potre("Ag", "s")),
    "H_POTRE": ("H_POTRE.DAT", _potre("H", "s")),
    "Hw_POTRE": ("H_POTRE.DAT", _potre("H", "s")),
    "Ow_POTRE": ("O_POTRE.DAT", _potre("O", "p")),
    "O_FHI": ("O_FHIPP.DAT", _fhi("O", 1.6, "2 4 0")),
    "Mg_FHI": ("Mg_FHIPP.DAT", _fhi("Mg", 2.6, "2 0 0", core=True)),
    "B_FHI": ("B_FHIPP.DAT", _fhi("B", 1.8, "2 1 0", core=True)),
    # POTRE tables, which the default routines of oblique cells and slabs compute (FHI tables and core corrections
    # take the Fortran-derived routines there)
    "Na_POTRE": ("Na_POTRE.DAT", _potre("Na", "d")),
    "Nap_POTRE": ("Na_POTRE.DAT", _potre("Na", "p")),
    "Au_POTRE": ("Au_POTRE.DAT", _potre("Au", "s")),
    "Auso_POTRE": ("Au_POTRE.DAT", _potre("Au", "s") + "SO_PSP: .true.\n"),
    "Hso_POTRE": ("H_POTRE.DAT", _potre("H", "s") + "SO_PSP: .false.\n"),
    # two labels of O (a broken-symmetry start)
    "Oa_FHI": ("O_a_FHIPP.DAT", _fhi("O_a", 1.6, "2 4 0").replace("Atom_Type: O_a\n", "Atom_Type: O_a\nElement_Symbol: O\n")),
    "Ob_FHI": ("O_b_FHIPP.DAT", _fhi("O_b", 1.6, "2 4 0").replace("Atom_Type: O_b\n", "Atom_Type: O_b\nElement_Symbol: O\n")),
    "Bx_POTRE": ("B_POTRE.DAT", _potre("B", "p")),
    "Nx_POTRE": ("N_POTRE.DAT", _potre("N", "p")),
    "N_FHI": ("N_FHIPP.DAT", _fhi("N", 1.6, "2 3 0")),
    "Cs_POTRE": ("C_POTRE.DAT", _potre("C", "d")),
    "Ns_POTRE": ("N_POTRE.DAT", _potre("N", "d")),
}

# ---------------------------------------------------------------------------------------------------------------
# Inputs


def coordinates(positions) -> str:
    return "".join(f"  {x:.15f} {y:.15f} {z:.15f}\n" for x, y, z in np.asarray(positions, dtype=float))


def scf_block(xc, states, *, temperature=500.0, criterion=1.0e-10, diag=1.0e-11, mixing=0.3, max_iter=400,
              forces=True, order=8, extra="") -> str:
    return (
        f"Coordinate_Unit: Cartesian_Bohr\nExpansion_Order: {order}\nCorrelation_Type: {xc}\nStates_Num: {states}\n"
        f"Fermi_Temp: {temperature:g} K\nMax_Iter: {max_iter}\nConvergence_Criterion: {criterion:g} Ry\n"
        f"Eigensolver: chebff\nDiag_Tolerance: {diag:g}\nMixing_Method: Anderson\nMixing_Param: {mixing:g}\n"
        "Memory_Param: 4\nOutput_Level: 1\n" + ("Calculate_Force: .true.\n" if forces else "") + extra
    )


def cell_head(vectors, spacing, *, boundary="bulk", fortran=None, radius=None, sad=True) -> str:
    text = f"Periodic_System: .true.\nBoundary_Conditions: {boundary}\nbegin Cell_Shape\n"
    for vector in vectors:
        text += "  " + " ".join(f"{value:.10f}" for value in np.atleast_1d(vector)) + "\n"
    text += "end Cell_Shape\nLattice_Vector_Scale: 1.0 bohr\n"
    if radius is not None:
        text += f"Boundary_Sphere_Radius: {radius:.2f}\n"
    if fortran is not None:
        text += f"Use_Fortran_Routines: {'.true.' if fortran else '.false.'}\n"
    # a hair below the spacing, so that PARSEC's adjust_step takes the intended number of points
    text += f"Grid_Spacing: {spacing * (1.0 - 1.0e-9):.14f}\n"
    if sad:
        text += "Initial_Density: sad\n"
    return text


def cluster_head(radius, spacing) -> str:
    # Grid_Shift 0: the origin is a grid point
    return (
        f"Boundary_Sphere_Radius: {radius}\nGrid_Spacing: {spacing}\nbegin Grid_Shift\n0.0 0.0 0.0\nend Grid_Shift\n"
    )


def atom_blocks(groups) -> str:
    """``groups``: (species key, positions, extra lines of the block)."""

    text = f"Atom_Types_Num: {len(groups)}\n\n"
    for key, positions, extra in groups:
        text += SPECIES[key][1] + extra + "begin Atom_Coord\n" + coordinates(positions) + "end Atom_Coord\n\n"
    return text


def mesh_text(dims) -> str:
    """A Gamma-centred Monkhorst-Pack mesh (no shift block, which would choose the Fortran-derived routines)."""

    return "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n" + " ".join(str(n) for n in dims) + "\nend Monkhorst_Pack_Grid\n"


def full_mesh_list(lattice, dims, periodic=3, tie_tolerance=1.0e-9, images="shortest") -> str:
    """The whole Gamma-centred mesh as a ``Kpoint_List`` (reciprocal lattice units).

    ``images="shortest"``: every point at each of its shortest images modulo the reciprocal lattice, the images
    sharing its weight equally, as the default routines compute a mesh (``PARSEC_KPOINT_FOLD``).  ``"unit"``: one
    image per point in [0, 1).  ``"centred"``: one
    image per point, each fraction in (-1/2, 1/2], as the Fortran-derived routines compute a mesh (they keep the
    written vectors of a list, and the energy of a point of the zone boundary depends on the image on an oblique
    grid)."""

    lattice = np.asarray(lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    rows = []
    axes = [np.arange(n) / n for n in dims]
    shifts = [(-1, 0, 1) if axis < periodic else (0,) for axis in range(3)]
    for fraction in itertools.product(*axes):
        fraction = np.array(fraction)
        fraction = fraction - np.rint(fraction)  # into [-1/2, 1/2]
        if images == "centred":
            fraction = np.where(fraction <= -0.5 + 1.0e-12, fraction + 1.0, fraction)
            rows.append((fraction, 1.0 / np.prod(dims)))
            continue
        if images == "unit":  # each fraction in [0, 1), as written by hand and by PARSEC 1.5's irrbz
            rows.append((np.where(fraction < -1.0e-12, fraction + 1.0, np.abs(fraction)), 1.0 / np.prod(dims)))
            continue
        images_of = []
        for g in itertools.product(*shifts):
            candidate = fraction - np.array(g, dtype=float)
            images_of.append((float(np.sum((candidate @ reciprocal) ** 2)), candidate))
        shortest = min(length for length, _ in images_of)
        chosen = [c for length, c in images_of if length <= shortest * (1.0 + tie_tolerance) + 1.0e-14]
        unique = []
        for c in chosen:
            if not any(np.allclose(c, u, atol=1.0e-12) for u in unique):
                unique.append(c)
        weight = 1.0 / (np.prod(dims) * len(unique))
        rows.extend((c, weight) for c in unique)
    text = "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
    for c, weight in rows:
        text += f"  {c[0]:.16f} {c[1]:.16f} {c[2]:.16f} {weight:.16e}\n"
    return text + "end Kpoint_List\n"


@dataclass
class Structure:
    """An input as a function of the positions of its atoms."""

    name: str
    text: Callable[[np.ndarray], str]
    positions: np.ndarray
    species: tuple
    lattice: np.ndarray | None = None  # rows: periodic vectors (for distances to grid points)
    periodic: int = 0
    env: dict = field(default_factory=dict)
    args: tuple = ()
    gpus: int = 1
    snap: int | None = None  # atom moved onto its nearest grid point before the case starts
    extra_files: Callable | None = None  # positions -> {file name: text} written beside the input

    def files(self) -> dict:
        return {SPECIES[key][0]: ROOT / PP_FILES[key] for key in self.species}

    def with_text(self, name, transform) -> "Structure":
        base = self.text
        return Structure(
            name, lambda p: transform(base(p)), self.positions.copy(), self.species, self.lattice, self.periodic,
            dict(self.env), self.args, self.gpus, self.snap, self.extra_files,
        )


FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
A_C = 3.567 * BOHR_PER_ANGSTROM
A_SI = 10.26
A_GE = 10.69
A_AG = 7.729
A_MGO = 7.96


def diamond_cube(a):
    """Eight atoms of the conventional diamond cube: 0-3 on the fcc sites, 4-7 a quarter along the diagonal.  On the
    grid of ``a / 18`` (points half a step off the origin) atoms 4-7 sit on grid points."""

    return np.array([(b + shift) * a for shift in (0.0, 0.25) for b in FCC])


def diamond(name, key, a, xc, states, *, fortran=False, move=(0.05, 0.05, 0.05), mesh=None, spin=False,
            soc=False, extra_scf="", extra_block="", criterion=1.0e-10, forces=True, temperature=500.0):
    """A cubic cell of eight atoms (C, Si, Ge), atom 0 moved by ``move`` (C3v about [111] by default), h = a / 18."""

    positions = diamond_cube(a)
    positions[0] += np.asarray(move, dtype=float)
    vectors = np.diag([a, a, a])
    extra = extra_scf
    block = extra_block
    if spin:
        extra += "Spin_Polarization: .true.\nZero_Average_Vxc: .true.\n"
    if soc:
        extra += "SO_from_scratch: .true.\n"
        block += "SO_PSP: .true.\n"
        states *= 2
    kpoints = mesh_text(mesh) if isinstance(mesh, tuple) else (mesh or "")

    def text(p):
        return (
            cell_head(vectors, a / 18, fortran=fortran)
            + scf_block(xc, states, criterion=criterion, diag=criterion / 10, forces=forces, temperature=temperature,
                        extra=extra)
            + kpoints + atom_blocks([(key, p, block)])
        )

    return Structure(name, text, positions, (key,), vectors, 3)


#: The four bonds of the diamond cube's atom at the origin to atoms 4-7 (unit vectors, the order of ``diamond_cube``).
TETRAHEDRON = np.array([(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)]) / np.sqrt(3.0)


def si8_td_meta(name, *, xc="scan", fortran=False, steps=1.0, criterion=1.0e-10, forces=True):
    """Si8 with atoms 4-7 moved out along their bonds to atom 0 by ``steps`` grid steps along each axis (Td about the
    origin, no face-centring translation; on the half-shifted grid of h = a/18 they stay on grid points for whole
    steps), a meta-GGA in the Gamma sectors at any size (``PARSEC_GAMMA_META_GGA_SECTORS=on``): the weighted sectors of
    ``PARSEC_ORBITAL_GROUP_META_GGA_CELL`` (added after 308edad4e)."""

    positions = diamond_cube(A_SI)
    positions[4:] = np.mod((0.25 * A_SI * np.sqrt(3.0) + steps * np.sqrt(3.0) * A_SI / 18) * TETRAHEDRON, A_SI)
    vectors = np.diag([A_SI] * 3)

    def text(p):
        return (
            cell_head(vectors, A_SI / 18, fortran=fortran)
            + scf_block(xc, 24, temperature=100.0, criterion=criterion, diag=criterion / 10, forces=forces)
            + atom_blocks([("Si_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("Si_POTRE",), vectors, 3, env={"PARSEC_GAMMA_META_GGA_SECTORS": "on"})


def c7n(name, *, fortran=False, mesh=None, criterion=1.0e-10, forces=True, move=0.03):
    """C7N in the diamond cube (spin doublet, CA), N at the origin moved by ``move`` along [111] (C3v)."""

    positions = diamond_cube(A_C)
    positions[0] += move
    vectors = np.diag([A_C] * 3)
    kpoints = mesh_text(mesh) if isinstance(mesh, tuple) else (mesh or "")

    def text(p):
        return (
            cell_head(vectors, A_C / 18, fortran=fortran)
            + scf_block("ca", 24, criterion=criterion, diag=criterion / 10, forces=forces,
                        extra="Spin_Polarization: .true.\n")
            + kpoints
            + atom_blocks([("Ns_POTRE", p[:1], "Initial_Spin_Polarization: 0.2\n"), ("Cs_POTRE", p[1:], "")])
        )

    return Structure(name, text, positions, ("Ns_POTRE", "Cs_POTRE"), vectors, 3)


def ag4(name, *, fortran=False, u=False, move=(0.05, 0.05, 0.05), criterion=1.0e-10, forces=True):
    """fcc Ag (four atoms, local s: p and d projectors), every site on a grid point of h = a/16; atom 0 moved."""

    positions = FCC * A_AG + 0.5 * A_AG / 16
    positions[0] += np.asarray(move, dtype=float)
    vectors = np.diag([A_AG] * 3)
    block = "ldaplusu_u 4.0 eV\n" if u else ""

    def text(p):
        return (
            cell_head(vectors, A_AG / 16, fortran=fortran)
            + scf_block("ca", 36, criterion=criterion, diag=criterion / 10, forces=forces)
            + atom_blocks([("Ag_POTRE", p, block)])
        )

    return Structure(name, text, positions, ("Ag_POTRE",), vectors, 3)


def mgo8(name, *, fortran=False):
    """Rock salt MgO (Mg FHI with a core correction, O FHI), h = a/40, O atom 4 moved by 0.05 along [111]."""

    positions = np.vstack([FCC * A_MGO, (FCC + [0.5, 0.0, 0.0]) * A_MGO])
    positions[4] += 0.05
    vectors = np.diag([A_MGO] * 3)

    def text(p):
        return (
            cell_head(vectors, A_MGO / 40, fortran=fortran)
            + scf_block("pbe", 24)
            + atom_blocks([("Mg_FHI", p[:4], ""), ("O_FHI", p[4:], "")])
        )

    return Structure(name, text, positions, ("Mg_FHI", "O_FHI"), vectors, 3)


def primitive_c2(name, mesh, *, move=0.03, criterion=1.0e-9):
    """Diamond in its primitive fcc cell (C POTRE, PBE), atom 1 moved along [111] (C3v), h about 0.25."""

    half = 0.5 * A_C
    vectors = np.array([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.25 * A_C] * 3])
    positions[1] += move
    kpoints = mesh if isinstance(mesh, str) else mesh_text(mesh)

    def text(p):
        return (
            cell_head(vectors, 0.25, fortran=False)
            + scf_block("pbe", 8, criterion=criterion, diag=criterion / 10)
            + kpoints + atom_blocks([("C_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("C_POTRE",), vectors, 3)


def hcp_mg2(name, mesh, *, move=0.05, criterion=1.0e-9):
    """hcp Mg (two atoms, Mg FHI with its core correction: the Fortran-derived routines), atom 0 moved along c (C3v),
    h about 0.40."""

    key = "Mg_FHI"
    a, c = 3.209 * BOHR_PER_ANGSTROM, 5.211 * BOHR_PER_ANGSTROM
    vectors = np.array([[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]])
    fractions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]])
    positions = fractions @ vectors
    positions[0, 2] += move
    kpoints = mesh if isinstance(mesh, str) else mesh_text(mesh)

    def text(p):
        return (
            cell_head(vectors, 0.40, fortran=False)
            + scf_block("pbe", 8, criterion=criterion, diag=criterion / 10)
            + kpoints + atom_blocks([(key, p, "")])
        )

    return Structure(name, text, positions, (key,), vectors, 3)


def hcp_na2(name, mesh, *, move=0.05, criterion=1.0e-9, forces=True, xc="ca", extra=""):
    """hcp Na (two atoms, TM Na without a core correction, CA: the default routines of an oblique cell), atom 0 moved
    along c (C3v), h about 0.50; ``xc`` another functional (``pbe0``: a metal's hybrid on the wedge of its mesh),
    ``extra`` lines of the SCF block."""

    a, c = 3.767 * BOHR_PER_ANGSTROM, 6.154 * BOHR_PER_ANGSTROM
    vectors = np.array([[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]])
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]) @ vectors
    positions[0, 2] += move
    kpoints = mesh if isinstance(mesh, str) else mesh_text(mesh)

    def text(p):
        return (
            cell_head(vectors, 0.50, fortran=False)
            + scf_block(xc, 8, criterion=criterion, diag=criterion / 10, forces=forces, extra=extra)
            + kpoints + atom_blocks([("Na_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("Na_POTRE",), vectors, 3)


def primitive_cubic(name, kind, *, move=0.03, criterion=1.0e-9, forces=True, fortran=False, extra=""):
    """A 2 x 2 x 2 supercell of a primitive cubic cell under ``PARSEC_OBLIQUE_FULL_GROUP=auto`` (CA; the floor of the
    sectors and their fixed cost off): ``fcc`` diamond Si (POTRE, 16 atoms, h about 0.6), atom 1 moved along z (C2v about z, whose two-fold
    axis is no signed permutation of the grid axes); ``bcc`` Na (POTRE, 8 atoms, h about 0.6), atom 1 moved along z
    (C4v).  Both keep an operation of the full cubic group that the signed permutations of the grid axes do not hold."""

    if kind == "fcc":
        key, a, basis, states = "Si_POTRE", A_SI, [np.zeros(3), np.full(3, 0.25 * A_SI)], 40
        vectors = np.array([[0.0, 0.5 * a, 0.5 * a], [0.5 * a, 0.0, 0.5 * a], [0.5 * a, 0.5 * a, 0.0]])
    else:
        key, a, basis, states = "Na_POTRE", 7.98, [np.zeros(3)], 12
        vectors = np.array([[-0.5 * a, 0.5 * a, 0.5 * a], [0.5 * a, -0.5 * a, 0.5 * a], [0.5 * a, 0.5 * a, -0.5 * a]])
    positions = np.array([
        position + i * vectors[0] + j * vectors[1] + k * vectors[2]
        for i in range(2) for j in range(2) for k in range(2) for position in basis
    ])
    positions[1, 2] += move
    supercell = 2.0 * vectors

    def text(p):
        return (
            cell_head(supercell, 0.60, fortran=fortran)
            + scf_block("ca", states, criterion=criterion, diag=criterion / 10, forces=forces, extra=extra)
            + atom_blocks([(key, p, "")])
        )

    # (the fixed cost of a sector off: these small cells take the plan of the least rows x states, the full cubic
    # group, which the default cost of a sector gives larger cells)
    return Structure(name, text, positions, (key,), supercell, 3,
                     env={"PARSEC_OBLIQUE_FULL_GROUP": "auto", "PARSEC_PERIODIC_SECTOR_FLOOR": "off",
                          "PARSEC_SECTOR_OVERHEAD": "off"})


MD_RESTART_LINES = (
    "Molecular_Dynamics: .true.\nStep_num: 3\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\nTime_Step: 40.0\n"
    "Restart_mode: .true.\n"
)


def md_restart(symbols, atom, direction, amount=0.002):
    """``mdinit.dat`` of a symmetric start: the previous structure is the current one with ``atom`` moved back by
    ``amount`` bohr along ``direction`` (an axis of the group the structure keeps), no other velocity, so that the
    trajectory keeps the group (the dynamics in sectors, PARSEC_MOVING_SECTORS)."""

    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)

    def files(positions):
        from parsec_python.MolecularDynamics.atomic_masses import atomic_masses
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState

        count = len(symbols)
        previous = np.array(positions, dtype=float)
        previous[atom] -= amount * direction
        state = MolecularDynamicsState(
            masses=np.asarray(atomic_masses(list(symbols)), dtype=float),
            movable=np.ones(count, dtype=np.int64),
            current_positions=np.array(positions, dtype=float),
            previous_positions=previous,
            velocities=np.zeros((count, 3)),
            accelerations=np.zeros((count, 3)),
            temperature=300.0,
            frame=4,
            move_count=0,
            restarted=True,
        )
        return {"mdinit.dat": format_md_restart(state)}

    return files


RELAX_LINES = "Minimization: bfgs\nMovement_Num: 3\nForce_Min: 1e-5 Ry/bohr\n"
MD_LINES = (
    "Molecular_Dynamics: .true.\nStep_num: 3\nTinit: 300.0\nT_Final: 300.0\nT_Step: 0.0\nTime_Step: 40.0\n"
    "Rng_Seed: 7\n"
)
BAND_LINES = "begin bandstruc\n1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\n1 0.5 0.0 0.0  0.5 0.5 0.0  X-M\nend bandstruc\n"


def ch4_c3v(name, extra, *, criterion=1.0e-9):
    """CH4 (C FHI at the origin, H POTRE) with the first H moved out along its bond by 0.1 bohr (C3v), ``extra``
    lines (a relaxation or dynamics)."""

    d = 1.1859538349
    hydrogens = np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
    hydrogens[0] *= 1.0 + 0.1 / np.linalg.norm(hydrogens[0])
    positions = np.vstack([[[0.0, 0.0, 0.0]], hydrogens])

    def text(p):
        return cluster_head(9.0, 0.25) + scf_block("pbe", 8, criterion=criterion, diag=criterion / 10,
                                                   extra=extra) + atom_blocks(
            [("C_FHI", p[:1], ""), ("H_POTRE", p[1:], "")]
        )

    return Structure(name, text, positions, ("C_FHI", "H_POTRE"), None, 0)


def c64_cube(name, mesh=None, *, move=0.05, extra="", criterion=1.0e-9, forces=True):
    """64 carbon atoms (2x2x2 diamond cubes, C POTRE, PBE) on h = a/24 with 160 states: 110,592 points x 160 states,
    above the 2^24 entries of the floor of the k-point sectors and of PARSEC_KPOINT_REAL_GAMMA; atom 0 moved along
    [111] by ``move``."""

    positions = np.array([cell + site for cell in itertools.product((0.0, A_C), repeat=3)
                          for site in diamond_cube(A_C)])
    positions[0] += move
    side = 2.0 * A_C
    vectors = np.diag([side] * 3)
    kpoints = mesh_text(mesh) if isinstance(mesh, tuple) else (mesh or "")

    def text(p):
        return (
            cell_head(vectors, A_C / 24, fortran=False)
            + scf_block("pbe", 160, criterion=criterion, diag=criterion / 10, forces=forces, extra=extra)
            + kpoints + atom_blocks([("C_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("C_POTRE",), vectors, 3)


def bcc_guarded(name, metal, kpoints, *, criterion=1.0e-9):
    """A primitive bcc cell on a grid coarse enough to choose the Fortran-derived routines (12 points or fewer an axis)
    where a mesh point has a free level below zero (PARSEC_KPOINT_MESH_GUARD): Na (a = 7.99, h 0.6, 6x6x6) or Au with
    spin-orbit coupling (a = 6.2, h 0.5, 4x4x4); CA, order 12, 1000 K.  ``kpoints``: a mesh or a list."""

    a, spacing, dims, states, key, extra = {
        "na": (7.99, 0.6, (6, 6, 6), 8, "Nap_POTRE", ""),
        "au_soc": (6.2, 0.5, (4, 4, 4), 16, "Auso_POTRE", "SO_from_scratch: .true.\n"),
    }[metal]
    half = 0.5 * a
    vectors = np.array([[-half, half, half], [half, -half, half], [half, half, -half]])
    positions = np.zeros((1, 3))
    block = mesh_text(dims) if kpoints == "mesh" else full_mesh_list(vectors, dims, images="unit")

    def text(p):
        return (
            cell_head(vectors, spacing, sad=False)
            + scf_block("ca", states, temperature=1000.0, criterion=criterion, diag=criterion / 10, order=12,
                        forces=False, extra=extra)
            + block + atom_blocks([(key, p, "")])
        )

    return Structure(name, text, positions, (key,), vectors, 3)


def au2_soc(name, *, hydride=False, criterion=1.0e-9, forces=False, positions=None, env=None, fortran=False):
    """Au2 (or AuH) with self-consistent spin-orbit coupling (TM Au with its spin-orbit channels, CA), along z.  The
    symmetry checks have no forces (``forces``: their forces, which ran on the reference solver without spin-orbit
    sectors before PARSEC_SOC_DEVICE_FORCES; the force cases added after 308edad4e take them on the device, in the
    sectors that ``env`` asks for).  ``positions`` replaces the atoms along z of the default; ``fortran`` adds
    ``Use_Fortran_Routines``."""

    if hydride:
        default = np.array([[0.0, 0.0, -1.445], [0.0, 0.0, 1.445]])
        groups = lambda p: [("Auso_POTRE", p[:1], ""), ("Hso_POTRE", p[1:], "")]
        species, states = ("Auso_POTRE", "Hso_POTRE"), 20
    else:
        default = np.array([[0.0, 0.0, -2.36], [0.0, 0.0, 2.36]])
        groups = lambda p: [("Auso_POTRE", p, "")]
        species, states = ("Auso_POTRE",), 30
    positions = default if positions is None else np.asarray(positions, dtype=float)

    def text(p):
        return (
            cluster_head(9.0, 0.35)
            + scf_block("ca", states, temperature=300.0, criterion=criterion, diag=criterion / 10, mixing=0.25,
                        forces=forces,
                        extra="SO_from_scratch: .true.\n" + ("Use_Fortran_Routines: .true.\n" if fortran else ""))
            + atom_blocks(groups(p))
        )

    return Structure(name, text, positions, species, None, 0, env=dict(env or {}))


def ge5_soc(name, *, criterion=1.0e-10, env=None):
    """A Ge5 square pyramid (C4v about z) with self-consistent spin-orbit coupling (Ge with its spin-orbit channels and
    core correction, CA), forces; the apex on a grid point (z = 3h), the base atoms off the grid.  Added after
    308edad4e for the device forces (PARSEC_SOC_DEVICE_FORCES)."""

    positions = np.array([[0.0, 0.0, 1.2], [2.3, 2.3, -1.2], [-2.3, 2.3, -1.2], [2.3, -2.3, -1.2], [-2.3, -2.3, -1.2]])

    def text(p):
        return (
            cluster_head(10.5, 0.4)
            + scf_block("ca", 32, temperature=300.0, criterion=criterion, diag=criterion / 10, mixing=0.25,
                        extra="SO_from_scratch: .true.\n")
            + atom_blocks([("Ge_POTRE", p, "SO_PSP: .true.\n")])
        )

    return Structure(name, text, positions, ("Ge_POTRE",), None, 0, env=dict(env or {}))


def au20_soc(name, *, criterion=1.0e-9):
    """The Au20 tetrahedron (Td, the group D2 of its axes on the grid) with self-consistent spin-orbit coupling, forces
    (h = 0.4 bohr, 16 bohr sphere, 236 states).  Added after 308edad4e: the base structure in the Kramers sector of
    PARSEC_SOC_SECTORS=auto, the moved atom leaves no two-fold operation (the whole grid)."""

    unit = 2.88 / 2 ** 0.5 / 0.529177 * 0.98
    found = []
    for x, y, z in itertools.product((-1.5, -0.5, 0.5, 1.5), repeat=3):
        if int(round(x + y + z + 4.5)) % 2:
            continue
        if max(x + y + z, x - y - z, -x + y - z, -x - y + z) <= 1.5 + 1e-9:
            found.append((x * unit, y * unit, z * unit))
    positions = np.array(found)

    def text(p):
        return (
            cluster_head(16.0, 0.4)
            + scf_block("ca", 236, temperature=300.0, criterion=criterion, diag=criterion / 10, mixing=0.25,
                        extra="SO_from_scratch: .true.\n")
            + atom_blocks([("Auso_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("Auso_POTRE",), None, 0)


def _hybrid_tolerance(xc) -> str:
    """The Fock-loop stop of a hybrid ``xc`` in a case (1e-10 Ry: totals to about 1e-10 Ry), none for another."""

    return "Exx_Tolerance: 1e-10 Ry\n" if xc in ("pbe0", "hse06", "b3lyp", "hf") else ""


def ch4(name, *, perturb=True, offset_h=None, criterion=1.0e-10, xc="pbe", scale=1.0, extra=""):
    """CH4 (C FHI at the origin, a grid point; H POTRE), the H atoms moved off the Td sites unless ``perturb`` is
    false; ``offset_h`` moves the first H by that many bohr along x; ``xc`` a functional (a hybrid with its Fock loop
    to 1e-10 Ry); ``scale`` the C-H bonds (Td kept); ``extra`` lines of the SCF block (a relaxation)."""

    d = 1.1859538349 * scale
    hydrogens = np.array([[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]])
    if perturb:
        hydrogens = hydrogens + np.array(
            [[0.04, -0.02, 0.01], [-0.01, 0.03, -0.02], [0.02, 0.01, 0.03], [-0.03, -0.01, 0.02]]
        )
    if offset_h is not None:
        hydrogens[0, 0] += offset_h
    positions = np.vstack([[[0.0, 0.0, 0.0]], hydrogens])

    def text(p):
        return cluster_head(9.0, 0.25) + scf_block(
            xc, 8, criterion=criterion, diag=criterion / 10, extra=_hybrid_tolerance(xc) + extra
        ) + atom_blocks([("C_FHI", p[:1], ""), ("H_POTRE", p[1:], "")])

    return Structure(name, text, positions, ("C_FHI", "H_POTRE"), None, 0)


def h2o(name, *, criterion=1.0e-10, ignore_symmetry=False):
    """H2O (TM O and H, CA), C2v, O on a grid point (0, -0.3, 0); the box of the atoms is centred on the origin, so
    that PARSEC does not re-centre the cluster (``center_cluster_geometry``) at any step.  ``ignore_symmetry``: the
    group is not used (one process solves the full grid, the MPI runner one sector of the identity)."""

    positions = np.array([[0.0, -0.3, 0.0], [1.6, 0.3, 0.0], [-1.6, 0.3, 0.0]])
    extra = "Ignore_Symmetry: .true.\n" if ignore_symmetry else ""

    def text(p):
        return cluster_head(10.0, 0.30) + scf_block(
            "ca", 8, criterion=criterion, diag=criterion / 10, extra=extra
        ) + atom_blocks([("Ow_POTRE", p[:1], ""), ("Hw_POTRE", p[1:], "")])

    return Structure(name, text, positions, ("Ow_POTRE", "Hw_POTRE"), None, 0)


def ag3(name):
    """Ag3 (TM Ag, local s, CA), atom 0 at the origin (a grid point)."""

    positions = np.array([[0.0, 0.0, 0.0], [-4.8, 0.05, 0.03], [4.75, -0.03, -0.05]])

    def text(p):
        return cluster_head(10.0, 0.45) + scf_block("ca", 26) + atom_blocks([("Ag_POTRE", p, "")])

    return Structure(name, text, positions, ("Ag_POTRE",), None, 0)


def au2_gga(name, *, xc="pbe", cluster=False):
    """Au2 (TM Au without a core correction, local s) with a GGA, atom 0 on a grid point (h = 0.31), atom 1 off the
    grid points.  The cell: a 6.2 bohr cube with both atoms on its (1, 1, 1) line (atom 0 at 1.5 h on every axis,
    the points of the default routines lie at (i + 1/2) h), a 4x4x4 mesh, 1000 K.  The cluster: atom 0 at
    (5h, 5h, 5h), the box of the atoms centred within 0.1 bohr of the origin (no re-centring at any step).

    The pseudo-valence density of Au is low at the nucleus (9.6e-4 e/bohr^3, 3.5e-2 one step away: the 5d shell);
    at a grid point on the nucleus the finite-difference gradient of the density grows with the offset of the atom,
    and the reduced gradient there reaches the scale of the PBE enhancement (s = 1) within about 1e-3 bohr, so that
    the discrete GGA energy varies on that scale.  The force is its derivative: the cases with these structures take
    steps of 2e-5 and 4e-5 bohr, which resolve that scale."""

    if cluster:
        a = 5 * 0.31
        positions = np.array([[a, a, a], [-a + 0.05, -a + 0.1, -a - 0.05]])

        def text(p):
            return (cluster_head(10.0, 0.31) + scf_block(xc, 16, criterion=1.0e-10, diag=1.0e-11)
                    + atom_blocks([("Au_POTRE", p, "")]))

        return Structure(name, text, positions, ("Au_POTRE",), None, 0)

    vectors = np.diag([6.2, 6.2, 6.2])
    positions = np.array([[0.0, 0.0, 0.0], [3.35, 3.35, 3.35]]) + 1.5 * 0.31

    def text(p):
        return (
            cell_head(vectors, 0.31, fortran=False, sad=False)
            + scf_block(xc, 30, temperature=1000.0, criterion=1.0e-10, diag=1.0e-11, order=6)
            + mesh_text((4, 4, 4)) + atom_blocks([("Au_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("Au_POTRE",), vectors, 3)


def o2_broken(name, *, criterion=1.0e-10):
    """O2 (O FHI, PBE, spin) from opposite starting polarizations of its two atoms (labels O_a and O_b)."""

    positions = np.array([[0.0, 0.0, -1.15], [0.02, -0.01, 1.17]])

    def text(p):
        return (
            cluster_head(9.0, 0.23)
            + scf_block("pbe", 10, criterion=criterion, diag=criterion / 10, extra="Spin_Polarization: .true.\n")
            + atom_blocks([("Oa_FHI", p[:1], "Initial_Spin_Polarization: 0.5\n"),
                           ("Ob_FHI", p[1:], "Initial_Spin_Polarization: -0.5\n")])
        )

    return Structure(name, text, positions, ("Oa_FHI", "Ob_FHI"), None, 0)


def o2(name, *, symmetric=False, criterion=1.0e-10, xc="pbe"):
    """O2 triplet (O FHI, PBE or ``xc``, spin), O 0 on a grid point (0, 0, -1.15) of h = 0.23; D-infinity-h if
    ``symmetric``.  The box of the atoms is centred on the origin within 0.1 bohr: PARSEC does not re-centre the
    cluster."""

    positions = np.array([[0.0, 0.0, -1.15], [0.0, 0.0, 1.15]]) if symmetric else np.array(
        [[0.0, 0.0, -1.15], [0.02, -0.01, 1.17]]
    )

    def text(p):
        return (
            cluster_head(9.0, 0.23)
            + scf_block(xc, 10, criterion=criterion, diag=criterion / 10,
                        extra="Spin_Polarization: .true.\n" + _hybrid_tolerance(xc))
            + atom_blocks([("O_FHI", p, "Initial_Spin_Polarization: 0.3\n")])
        )

    return Structure(name, text, positions, ("O_FHI",), None, 0)


def bn_slab(name, *, fortran=None, mesh=None, snap=True, criterion=1.0e-10, potre=False, forces=True):
    """A monolayer of h-BN (B FHI with a core correction, N FHI) as a slab, B moved off its site; with ``snap`` the B
    atom is put on its nearest grid point first."""

    vectors = np.array([[4.73, 0.0, 0.0], [-2.365, 4.0963001599, 0.0]])
    positions = np.array([[0.0, 0.0, 0.03], [0.02, 2.7308667733, -0.02]])
    kpoints = mesh if isinstance(mesh, str) else (mesh_text(mesh) if mesh else "")

    def text(p):
        return (
            cell_head(vectors, 0.22, boundary="slab", fortran=fortran, radius=10.0, sad=False)
            + scf_block("pbe", 8, criterion=criterion, diag=criterion / 10, forces=forces) + kpoints
            + atom_blocks([(boron, p[:1], ""), (nitrogen, p[1:], "")])
        )

    # POTRE tables: the default routines of a slab (the FHI tables and the core correction of B take the Fortran-derived)
    boron, nitrogen = ("Bx_POTRE", "Nx_POTRE") if potre else ("B_FHI", "N_FHI")
    lattice = np.array([[4.73, 0.0, 0.0], [-2.365, 4.0963001599, 0.0], [0.0, 0.0, 1.0]])
    return Structure(name, text, positions, (boron, nitrogen), lattice, 2, snap=0 if snap else None)


def bn_rect_slab(name, *, nx=4, ny=2, criterion=1.0e-9):
    """Rectangular h-BN (four atoms a cell, B FHI with a core correction, N FHI, PBE) repeated nx x ny as a slab of
    half height 8 bohr at h 0.25, the origin on a B atom, no forces (which take a slab to the reference solver).  The
    4 x 2 cell (2.4e7 grid points x states) lies below the floor of the sectors of a wire or slab
    (``PARSEC_SLAB_WIRE_SECTORS``, ``driver.WIRE_SLAB_FLOOR_ENTRIES``): its runs take
    ``PARSEC_PERIODIC_SECTOR_FLOOR=off``, so that the route of the sectors is checked at the cost of a small cell."""

    a = 4.73
    b = a * np.sqrt(3.0)
    lx, ly = nx * a, ny * b
    borons, nitrogens = [], []
    for i in range(nx):
        for j in range(ny):
            borons += [(i * a, j * b), (i * a + a / 2, j * b + b / 2)]
            nitrogens += [(i * a, j * b + b / 3), (i * a + a / 2, j * b + 5 * b / 6)]

    def folded(points):
        return [((x + lx / 2) % lx - lx / 2, (y + ly / 2) % ly - ly / 2, 0.0) for x, y in points]

    positions = np.array(folded(borons) + folded(nitrogens))
    count = len(borons)
    states = int(round(4 * count * 1.15)) + 4

    def text(p):
        return (
            cell_head([[lx, 0.0], [0.0, ly]], 0.25, boundary="slab", radius=8.0, sad=False)
            + scf_block("pbe", states, criterion=criterion, diag=criterion / 10, forces=False, order=12)
            + atom_blocks([("B_FHI", p[:count], ""), ("N_FHI", p[count:], "")])
        )

    lattice = np.array([[lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, 1.0]])
    return Structure(name, text, positions, ("B_FHI", "N_FHI"), lattice, 2, env={"PARSEC_PERIODIC_SECTOR_FLOOR": "off"})


def ribbon_wire(name, *, half=12.0, criterion=1.0e-9):
    """An H-passivated armchair graphene ribbon along x as a wire (C FHI, H POTRE, CA; period 3 d = 8.049 bohr, the
    carbons within ``half`` bohr of the axis, the origin at a hexagon centre: the eight operations of D2h), radius
    ``half`` + 9 bohr at h 0.3, no forces: 26 atoms for 12 bohr (2.5e7 grid points x states, below the floor of the
    sectors of a wire or slab; its runs take ``PARSEC_PERIODIC_SECTOR_FLOOR=off``, as :func:`bn_rect_slab`)."""

    d, ch = 2.683, 2.06
    period = 3.0 * d
    a1 = np.array([1.5 * d, 0.5 * np.sqrt(3.0) * d])
    a2 = np.array([1.5 * d, -0.5 * np.sqrt(3.0) * d])
    basis = (np.array([d, 0.0]), np.array([0.5 * d, 0.5 * np.sqrt(3.0) * d]))
    reach = int(np.ceil(half / d)) + 6
    carbons = []
    for n in range(-reach, reach + 1):
        for m in range(-reach, reach + 1):
            for b in basis:
                p = b + n * a1 + m * a2
                if -period / 2 - 1e-6 <= p[0] < period / 2 - 1e-6 and abs(p[1]) <= half + 1e-9:
                    carbons.append(p)
    hydrogens = []
    for c in carbons:
        bonds = [other + np.array([shift, 0.0]) - c for shift in (-period, 0.0, period) for other in carbons]
        bonds = [v / np.linalg.norm(v) for v in bonds if 1e-6 < np.linalg.norm(v) < 1.1 * d]
        if len(bonds) == 2:
            direction = -(bonds[0] + bonds[1])
            hydrogens.append(c + ch * direction / np.linalg.norm(direction))
    positions = np.array([(p[0], p[1], 0.0) for p in carbons] + [(p[0], p[1], 0.0) for p in hydrogens])
    count = len(carbons)
    states = int(round((4 * count + len(hydrogens)) / 2 * 1.15)) + 4
    radius = float(np.ceil(half + 2.1 + 7.0))

    def text(p):
        return (
            cell_head([period], 0.3, boundary="wire", radius=radius, sad=False)
            + scf_block("ca", states, criterion=criterion, diag=criterion / 10, forces=False, order=12)
            + atom_blocks([("C_FHI", p[:count], ""), ("H_POTRE", p[count:], "")])
        )

    lattice = np.array([[period, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return Structure(name, text, positions, ("C_FHI", "H_POTRE"), lattice, 1, env={"PARSEC_PERIODIC_SECTOR_FLOOR": "off"})


def ge2_slab(name, *, height=6.0, env=None):
    """Two Ge atoms (POTRE with a core correction, local p, CA) in a square slab cell of 5.6 bohr at h 0.45, the domain
    |z| <= ``height``; at 6 bohr the density at the edge of the slab is 1.1e-4 e/bohr^3, where the Hartree operator of
    the slab is not symmetric (``PARSEC_HARTREE_VARIATIONAL``)."""

    positions = np.array([[0.1, -0.1, 1.2], [-2.7, 2.7, -1.1]])
    vectors = np.array([[5.6, 0.0, 0.0], [0.0, 5.6, 0.0]])

    def text(p):
        return (
            cell_head(vectors, 0.45, boundary="slab", radius=height, sad=False)
            + scf_block("ca", 8, temperature=300.0) + atom_blocks([("Gep_POTRE", p, "")])
        )

    lattice = np.array([[5.6, 0.0, 0.0], [0.0, 5.6, 0.0], [0.0, 0.0, 1.0]])
    return Structure(name, text, positions, ("Gep_POTRE",), lattice, 2, env=dict(env or {}))


def c_wire(name, *, snap=True):
    """A chain of C (C FHI, PBE) as a wire along x, two atoms per cell; with ``snap`` atom 0 sits on a grid point."""

    positions = np.array([[0.0, 0.0, 0.0], [2.4, 0.03, -0.02]])

    def text(p):
        return (
            cell_head([4.8], 0.24, boundary="wire", radius=8.0, sad=False)
            + scf_block("pbe", 8, mixing=0.2) + atom_blocks([("C_FHI", p, "")])
        )

    lattice = np.array([[4.8, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return Structure(name, text, positions, ("C_FHI",), lattice, 1, snap=0 if snap else None)


def si_pbe0(name, *, mesh=True):
    """Si in its primitive fcc cell (Si FHI), PBE0, atom 1 moved by 0.05 along x, a 2x2x2 mesh (as
    examples/3d_hybrid_functionals/Si_disp_k222/pbe0 with a Monkhorst_Pack_Grid)."""

    half = 5.13
    vectors = np.array([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0], [2.615, 2.565, 2.565]])
    kpoints = mesh_text((2, 2, 2)) if mesh else full_mesh_list(vectors, (2, 2, 2))

    def text(p):
        return (
            cell_head(vectors, 0.25, fortran=False, sad=False)
            + scf_block("pbe0", 8, temperature=1.0, criterion=1.0e-8, diag=1.0e-9, order=12,
                        extra="Exx_Tolerance: 1e-9 Ry\n")
            + kpoints + atom_blocks([("Si_FHI", p, "")])
        )

    return Structure(name, text, positions, ("Si_FHI",), vectors, 3)


#: The Fock-loop stop of the metal hybrids of the force group (the slope needs the free energy to 1e-10 Ry)
#: Why the atom on a grid point of the Au cases is exempt (force_checks)
_GGA_ON_GRID = (
    "Au on a grid point with a GGA: the energy varies on a scale of 1e-3 bohr there (the finite-difference density "
    "gradient at a grid point on a nucleus of low pseudo-valence density); the force is its derivative (steps of "
    "2e-5 and 4e-5 bohr: f_au2_pbe_fine, f_au2c_pbe_fine)"
)
#: Why a displacement that removes the group of a metal hybrid is exempt (force_checks)
_METAL_PAIRS = (
    "the displaced structures keep no group and run on the time-reversal pairs, whose Fock loop may fill equivalent "
    "k-points unequally (a lower, broken state, an artefact of the finite mesh): their slope is that of the broken "
    "branch, not of the wedge state at the base"
)
METAL_HYBRID_EXX = "Exx_Tolerance: 1e-10 Ry\nExx_Max_Iter: 200\n"


def au2_pbe0(name, *, fortran=False, mesh=(4, 4, 4), second="Au_POTRE"):
    """bcc Au2 in its cube (a = 6.2 bohr, TM Au, PBE0, 1000 K), the second atom moved from the body centre along
    (1, 1, 1) (C3v): a metal whose levels at the Fermi level are partly filled, on the irreducible wedge of its mesh
    (PARSEC_HYBRID_KPOINT_WEDGE=on, the default since 2026-11-23).  Both atoms moved by 0.465 bohr along (1, 1, 1),
    inside the cell at either step (an atom at a corner leaves the cell at one sign and takes the input to the
    Fortran-derived routines) and the first on a point of the default routines' grid (points at (i + 1/2) h).
    ``second="Agn_POTRE"``: AuAg, whose group has no fractional translation (that of Au2, which swaps the atoms,
    keeps no grid of the Fortran-derived routines near the spacing, so they would run the time-reversal pairs)."""

    vectors = np.diag([6.2, 6.2, 6.2])
    positions = np.array([[0.0, 0.0, 0.0], [3.35, 3.35, 3.35]]) + 0.465
    if second == "Au_POTRE":
        groups, species = (lambda p: [("Au_POTRE", p, "")]), ("Au_POTRE",)
    else:
        groups, species = (lambda p: [("Au_POTRE", p[:1], ""), (second, p[1:], "")]), ("Au_POTRE", second)

    def text(p):
        return (
            cell_head(vectors, 0.31, fortran=fortran, sad=False)
            + scf_block("pbe0", 30, temperature=1000.0, criterion=1.0e-10, diag=1.0e-11, order=6,
                        extra=METAL_HYBRID_EXX)
            + mesh_text(mesh) + atom_blocks(groups(p))
        )

    return Structure(name, text, positions, species, vectors, 3)


def au2b_pbe0(name):
    """bcc Au2 in its cube as :func:`au2_pbe0`, the second atom moved from the body centre by (0.25, 0.25, 0) bohr and
    both by 0.40 bohr along (1, 1, 1): the group {E, x <-> y}, neither atom near a grid point.  A displacement along z
    keeps the group (the wedge of its mesh at every step); one along x or y removes it, so the displaced structures
    run on the time-reversal pairs."""

    vectors = np.diag([6.2, 6.2, 6.2])
    positions = np.array([[0.0, 0.0, 0.0], [3.35 + 0.25, 3.35 + 0.25, 3.35]]) + 0.40

    def text(p):
        return (
            cell_head(vectors, 0.31, fortran=False, sad=False)
            + scf_block("pbe0", 30, temperature=1000.0, criterion=1.0e-10, diag=1.0e-11, order=6,
                        extra=METAL_HYBRID_EXX)
            + mesh_text((4, 4, 4)) + atom_blocks([("Au_POTRE", p, "")])
        )

    return Structure(name, text, positions, ("Au_POTRE",), vectors, 3)


def c8_pbe0(name, *, forces=True):
    """C8 (C POTRE) at the Gamma point with PBE0, atom 0 moved along [111]."""

    return diamond(name, "C_POTRE", A_C, "pbe0", 20, criterion=1.0e-8,
                   extra_scf="Exx_Tolerance: 1e-9 Ry\n", forces=forces)


def si64_pbe0(name):
    """Si64 (2x2x2 cubes, Si FHI, h = a / 34) at the Gamma point with PBE0, no forces: a basis above the floor of the
    sectors of a Gamma-point hybrid, whose 32 abelian sectors carry the seven half-cell translations (the exchange in
    their translation classes)."""

    cells = np.array(list(itertools.product(range(2), repeat=3)), dtype=float)
    positions = (cells[:, None, :] * A_SI + diamond_cube(A_SI)[None, :, :]).reshape(-1, 3)
    vectors = np.diag([2.0 * A_SI] * 3)

    def text(p):
        return (
            cell_head(vectors, A_SI / 34, fortran=False, sad=False)
            + scf_block("pbe0", 140, temperature=1.0, criterion=1.0e-8, diag=1.0e-9, order=12, forces=False,
                        extra="Exx_Tolerance: 1e-9 Ry\n")
            + atom_blocks([("Si_FHI", p, "")])
        )

    return Structure(name, text, positions, ("Si_FHI",), vectors, 3)


# ---------------------------------------------------------------------------------------------------------------
# Checks


@dataclass
class Job:
    folder: str
    structure: Structure
    positions: np.ndarray
    env: dict = field(default_factory=dict)
    args: tuple = ()
    keep_output: bool = False


@dataclass
class Check:
    group: str
    name: str
    note: str = ""

    def structures(self) -> list:
        return []

    def jobs(self) -> dict:
        raise NotImplementedError

    def judge(self, results: dict) -> list:
        raise NotImplementedError


def _within(value, tolerance) -> bool:
    """Whether a difference is finite and within ``tolerance``: a NaN or infinite force, slope, energy or level (whose
    comparisons with a number are all false) is a failure, never a pass."""

    return value is not None and bool(np.isfinite(value)) and value <= tolerance


def compare_extras(a, b):
    """Text and verdict (None: nothing to compare) of the trajectories and band levels of two results."""

    text, ok = "", None
    if a.trajectory_energies is not None or b.trajectory_energies is not None:
        if a.trajectory_energies is None or b.trajectory_energies is None:
            return " trajectory missing in one run", False
        if a.trajectory_energies.shape != b.trajectory_energies.shape or (
            a.trajectory_positions.shape != b.trajectory_positions.shape
        ):
            return (f" trajectories of {a.trajectory_energies.size} and {b.trajectory_energies.size} structures",
                    False)
        de = float(np.max(np.abs(a.trajectory_energies - b.trajectory_energies)))
        dr = float(np.max(np.abs(a.trajectory_positions - b.trajectory_positions)))
        text += (f" trajectory {a.trajectory_energies.size} structures max|dE| {de:.2e} max|dR| {dr:.2e}"
                 f" (tol {TRAJECTORY_ENERGY_TOLERANCE:.0e}/{TRAJECTORY_POSITION_TOLERANCE:.0e})")
        ok = _within(de, TRAJECTORY_ENERGY_TOLERANCE) and _within(dr, TRAJECTORY_POSITION_TOLERANCE)
    if a.bands is not None or b.bands is not None:
        if a.bands is None or b.bands is None or a.bands.shape != b.bands.shape:
            return text + " band levels missing or of other shapes in one run", False
        # (levels missing in both runs, NaN at the same places, are left out; a level finite in one run only, or no
        # finite level at all, is a failure)
        finite_a, finite_b = np.isfinite(a.bands), np.isfinite(b.bands)
        both = finite_a & finite_b
        db = float(np.max(np.abs(a.bands[both] - b.bands[both]))) if both.any() else float("nan")
        if np.any(finite_a != finite_b):
            db = float("nan")
        text += f" bands {a.bands.shape} max|dlevel| {db:.2e} (tol {BAND_TOLERANCE:.0e})"
        ok = (ok is not False) and _within(db, BAND_TOLERANCE)
    return text, ok


def _fmt(value, spec="+.9e"):
    return "n/a" if value is None else format(value, spec)


@dataclass
class ForceCheck(Check):
    structure: Structure = None
    targets: tuple = ()
    offset: tuple | None = None  # (atom, bohr): the atom moved that far along OFFSET_DIRECTION first
    tolerance: float = FORCE_TOLERANCE
    #: documented exceptions: atom -> (steps whose slope is not judged, the reason printed[, the largest difference of
    #: force and slope that is exempt, Ry/bohr: beyond it the step fails])
    exempt: dict = field(default_factory=dict)
    #: the two steps of the central differences (bohr), the second twice the first for the Richardson value
    steps: tuple = FORCE_STEPS

    def structures(self):
        return [self.structure]

    def start(self):
        positions = self.structure.positions.copy()
        if self.offset is not None:
            atom, amount = self.offset
            positions[atom] += amount * OFFSET_DIRECTION
        return positions

    def jobs(self):
        start = self.start()
        jobs = {"base": Job(f"{self.name}/base", self.structure, start, keep_output=True)}
        for (atom, axis), step, sign in itertools.product(self.targets, self.steps, (1.0, -1.0)):
            moved = start.copy()
            moved[atom, axis] += sign * step
            jobs[(atom, axis, step, sign)] = Job(
                f"{self.name}/a{atom}x{axis}_s{step:g}_{'p' if sign > 0 else 'm'}", self.structure, moved
            )
        return jobs

    def judge(self, results):
        base = results["base"]
        lines = []
        if not base.ok:
            return [("ERROR", f"base run {base.why()}")]
        for atom, axis in self.targets:
            force = float(base.forces[atom, axis])
            slopes, diffs, missing = [], [], []
            for step in self.steps:
                plus, minus = results[(atom, axis, step, 1.0)], results[(atom, axis, step, -1.0)]
                if not (plus.ok and minus.ok):
                    missing.append(f"step {step:g}: {(plus if not plus.ok else minus).why()}")
                    slopes.append(None)
                    diffs.append(None)
                    continue
                slope = -(plus.free - minus.free) / (2.0 * step)
                slopes.append(slope)
                diffs.append(abs(force - slope))
            richardson = None
            if None not in slopes:
                richardson = (4.0 * slopes[0] - slopes[1]) / 3.0
            dgrid = base.grid_distance(atom)
            text = (
                f"atom {atom} axis {axis} F {force:+.9e} "
                + " ".join(
                    f"slope({step:g}) {_fmt(s)} diff {_fmt(d, '.2e')}" for step, s, d in zip(self.steps, slopes, diffs)
                )
                + f" rich {_fmt(richardson)} diff {_fmt(None if richardson is None else abs(force - richardson), '.2e')}"
                + f" dgrid {_fmt(dgrid, '.2e')} tol {self.tolerance:.0e} E {base.free:.10f}"
            )
            if base.moved is not None and base.moved > 1.0e-8:
                text += f" (PARSEC moved the atoms by {base.moved:.2e} bohr: the slope is not that of fixed grid)"
            factor = base.levels_factor
            if factor is not None and abs(factor - round(factor)) > 1.0e-6:
                text += f" (electrons per listed level {factor:.6f}: entropy term uncertain)"
            if missing:
                lines.append(("ERROR", text + " | " + "; ".join(missing)))
                continue
            if not np.isfinite(force) or not all(np.isfinite(slope) for slope in slopes):
                # (no exemption: a NaN or infinite force or slope is never a step that may miss)
                lines.append(("FAIL", text + " | non-finite force or slope"))
                continue
            failing = [step for step, d in zip(self.steps, diffs) if not _within(d, self.tolerance)]
            exemption = self.exempt.get(atom, ((), ""))
            exempt_steps, reason = exemption[0], exemption[1]
            bound = exemption[2] if len(exemption) > 2 else None
            first, second = (s - force for s in slopes)
            if not failing:
                lines.append(("PASS", text))
            elif all(step in exempt_steps for step in failing) and (
                bound is None or all(_within(d, bound) for step, d in zip(self.steps, diffs) if step in failing)
            ):
                within = "" if bound is None else f" (exempt up to {bound:.0e})"
                lines.append(
                    ("EXEMPT", text + f" | exempt step(s) {','.join(f'{s:g}' for s in failing)}{within}: {reason}")
                )
            elif (
                failing == [self.steps[1]]
                and _within(abs(force - richardson), self.tolerance)
                and first * second > 0.0
                and abs(second) >= 2.0 * abs(first)
            ):
                # the larger step alone misses, by the h^2 term of the central difference: the two slopes approach
                # the force from one side, about four times as far at twice the step, and their Richardson value
                # meets the tolerance (a large force with a large third derivative of the energy)
                lines.append(("EXEMPT", text + f" | step {self.steps[1]:g}: h^2 truncation of the central "
                              "difference (one-sided, Richardson within the tolerance)"))
            elif (
                _within(abs(force - richardson), self.tolerance)
                and first * second > 0.0
                and 3.0 <= abs(second) / abs(first) <= 5.0
            ):
                # both steps miss, by the h^2 term alone: one side, four times as far at twice the step, and the
                # Richardson value within the tolerance
                lines.append(("EXEMPT", text + " | both steps: h^2 truncation of the central difference (ratio "
                              f"{abs(second) / abs(first):.2f}, Richardson within the tolerance)"))
            else:
                lines.append(("FAIL", text))
        return lines


@dataclass
class SymmetryCheck(Check):
    structure: Structure = None
    reference: Structure = None  # the input without symmetry (the same input if None)
    reference_env: dict = field(default_factory=lambda: dict(NO_SYMMETRY_ENV))
    reference_args: tuple = NO_SYMMETRY_ARGS
    energy_tolerance: float = SYMMETRY_ENERGY_TOLERANCE
    force_tolerance: float = SYMMETRY_FORCE_TOLERANCE

    def structures(self):
        return [self.structure] + ([self.reference] if self.reference is not None else [])

    def jobs(self):
        reference = self.reference or self.structure
        return {
            "sym": Job(f"{self.name}/symmetry", self.structure, self.structure.positions.copy(), keep_output=True),
            "off": Job(f"{self.name}/nosymmetry", reference, reference.positions.copy(), env=self.reference_env,
                       args=self.reference_args, keep_output=True),
        }

    def judge(self, results):
        sym, off = results["sym"], results["off"]
        if not (sym.ok and off.ok):
            return [("ERROR", f"symmetry {sym.why()} | nosymmetry {off.why()}")]
        de = abs(sym.free - off.free)
        text = f"E {sym.free:.10f} nosym {off.free:.10f} dE {de:.2e} it {sym.iterations}/{off.iterations}"
        df = None
        if sym.forces is not None and off.forces is not None:
            # the forces as reported (net force removed): a symmetrized route averages the net force of the grid away
            sym_f = sym.reported if sym.reported is not None else sym.forces
            off_f = off.reported if off.reported is not None else off.forces
            df = float(np.max(np.abs(sym_f - off_f)))
            text += (f" max|dF| {df:.2e} max|F| {float(np.max(np.abs(sym_f))):.2e}"
                     f" (before the net force was removed: {float(np.max(np.abs(sym.forces - off.forces))):.2e})")
        if sym.moment is not None or off.moment is not None:
            text += f" moment {_fmt(sym.moment, '.6f')}/{_fmt(off.moment, '.6f')}"
        text += f" tol {self.energy_tolerance:.0e}/{self.force_tolerance:.0e}"
        ok = _within(de, self.energy_tolerance) and (df is None or _within(df, self.force_tolerance))
        extra, extra_ok = compare_extras(sym, off)
        text += extra
        if extra_ok is False:
            ok = False
        return [("PASS" if ok else "FAIL", text)]


@dataclass
class SwitchCheck(Check):
    structure: Structure = None
    env: dict = field(default_factory=dict)
    pinning: bool = False  # only the run of the off values (recorded as a pin)

    def structures(self):
        return [self.structure]

    def jobs(self):
        env = dict(ALL_OFF)
        env.update(self.env)
        jobs = {"off": Job(f"{self.name}/all_off", self.structure, self.structure.positions.copy(), env=env,
                           keep_output=True)}
        if not self.pinning:
            # the default values beside: how far the switches move this case (0: it exercises none of them)
            jobs["default"] = Job(f"{self.name}/default", self.structure, self.structure.positions.copy(),
                                  env=dict(self.env), keep_output=True)
        return jobs

    def judge(self, results, pins=None):
        run = results["off"]
        pin = (pins or {}).get(self.name)
        if pin is None:
            return [("ERROR", f"no pinned base result | head {run.summary()}")]
        if not pin.get("ok"):
            same = not run.ok
            return [("PASS" if same else "FAIL", f"base failed ({pin.get('why')}) | head {run.summary()}")]
        if not run.ok:
            return [("ERROR", f"head {run.why()} | base E {pin['free']!r}")]
        de = abs(run.free - pin["free"])
        text = f"E {run.free:.12f} base {pin['free']:.12f} dE {de:.2e} it {run.iterations}/{pin.get('iterations')}"
        df = None
        if run.forces is not None and pin.get("forces") is not None:
            df = float(np.max(np.abs(run.forces - np.asarray(pin["forces"]))))
            text += f" max|dF| {df:.2e}"
        bits = de == 0.0 and (df is None or df == 0.0)
        ok = _within(de, SWITCH_ENERGY_TOLERANCE) and (df is None or _within(df, SWITCH_FORCE_TOLERANCE))
        default = results.get("default")
        if default is not None and default.ok:
            text += f" | default dE {abs(default.free - run.free):.2e}"
            if default.forces is not None and run.forces is not None:
                text += f" dF {float(np.max(np.abs(default.forces - run.forces))):.2e}"
            text += f" it {default.iterations}"
        elif default is not None:
            text += f" | default {default.why()}"
        return [("PASS" if ok else "FAIL", ("bits " if bits else "") + text + f" base {pin.get('commit', '?')}")]


@dataclass
class RepeatCheck(Check):
    structure: Structure = None

    def structures(self):
        return [self.structure]

    def jobs(self):
        return {
            index: Job(f"{self.name}/run{index}", self.structure, self.structure.positions.copy(), keep_output=True)
            for index in (0, 1)
        }

    def judge(self, results):
        first, second = results[0], results[1]
        if not (first.ok and second.ok):
            return [("ERROR", f"run0 {first.why()} | run1 {second.why()}")]
        de = abs(first.free - second.free)
        text = f"E {first.free:.12f} {second.free:.12f} spread {de:.2e} it {first.iterations}/{second.iterations}"
        if first.forces is not None and second.forces is not None:
            text += f" max|dF| {float(np.max(np.abs(first.forces - second.forces))):.2e}"
        ok = _within(de, REPEAT_ENERGY_TOLERANCE) and first.iterations == second.iterations
        extra, extra_ok = compare_extras(first, second)
        text += extra
        if extra_ok is False:
            ok = False
        return [("PASS" if ok else "FAIL", text)]


@dataclass
class MpiCheck(Check):
    structure: Structure = None
    launcher: str = "runner"  # runner: benchmarks/mpi_full_scf.py; cli: the command line with --mpi


# ---------------------------------------------------------------------------------------------------------------
# Runs


class Result:
    def __init__(self, status=None, error=None):
        self.status = status
        self.error = error
        self.energy = None
        self.free = None
        self.entropy_term = 0.0
        self.levels_factor = None
        self.forces = None
        self.reported = None
        self.trajectory_energies = None
        self.trajectory_positions = None
        self.bands = None
        self.iterations = None
        self.converged = None
        self.moment = None
        self.grid = None
        self.atoms = None
        self.moved = None
        self.dgrids = None
        self.cell = None
        self.periodic = 0
        self.seconds = None
        self.tail = ""

    @property
    def ok(self):
        # 4: a relaxation that used up Movement_Num (its structures are the result)
        return self.status in (0, 4) and self.free is not None

    def why(self):
        if self.ok:
            return "ok"
        if self.error:
            return self.error
        return f"status {self.status} converged {self.converged} {self.tail[-240:]!r}"

    def summary(self):
        return f"E {self.free!r} it {self.iterations} status {self.status}" if self.ok else self.why()

    def distance(self, difference):
        """Lengths of difference vectors, modulo the lattice along the periodic directions."""

        difference = np.atleast_2d(np.asarray(difference, dtype=float))
        if self.cell is not None and self.periodic:
            fractions = difference @ np.linalg.inv(self.cell)
            fractions[:, : self.periodic] -= np.rint(fractions[:, : self.periodic])
            difference = fractions @ self.cell
        return np.linalg.norm(difference, axis=1)

    def grid_distance(self, atom):
        if self.dgrids is not None:
            return self.dgrids[atom]
        if self.grid is None or self.atoms is None:
            return None
        return float(np.min(self.distance(self.grid - self.atoms[atom])))

    def record(self):
        return {
            "status": self.status, "error": self.error, "energy": self.energy, "free": self.free,
            "entropy_term": self.entropy_term, "levels_factor": self.levels_factor,
            "forces": None if self.forces is None else self.forces.tolist(), "iterations": self.iterations,
            "reported": None if self.reported is None else self.reported.tolist(),
            "converged": self.converged, "moment": self.moment, "seconds": self.seconds, "ok": self.ok,
            "dgrids": None if self.dgrids is None else [float(d) for d in self.dgrids],
            "moved": None if self.moved is None else float(self.moved),
            "why": None if self.ok else self.why(),
            "trajectory_energies": None if self.trajectory_energies is None else self.trajectory_energies.tolist(),
            "trajectory_positions": None if self.trajectory_positions is None else self.trajectory_positions.tolist(),
            "bands": None if self.bands is None else self.bands.tolist(),
        }


def result_of_record(record) -> Result:
    """A :class:`Result` from :meth:`Result.record` (a pinned run)."""

    result = Result(record.get("status"), record.get("error"))
    result.energy, result.free = record.get("energy"), record.get("free")
    result.iterations, result.converged = record.get("iterations"), record.get("converged")
    result.forces = None if record.get("forces") is None else np.asarray(record["forces"], dtype=float)
    result.reported = None if record.get("reported") is None else np.asarray(record["reported"], dtype=float)
    result.moment, result.seconds = record.get("moment"), record.get("seconds")
    for key in ("trajectory_energies", "trajectory_positions", "bands"):
        if record.get(key) is not None:
            setattr(result, key, np.asarray(record[key], dtype=float))
    result.levels_factor, result.moved, result.dgrids = (
        record.get("levels_factor"), record.get("moved"), record.get("dgrids")
    )
    if not record.get("ok") and record.get("why"):
        result.error = result.error or record["why"]
    return result


def temperature_of(text):
    match = re.search(r"^Fermi_Temp:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else 80.0


def free_energy(data, temperature):
    """``E - TS`` of an archive, the functional whose derivative the forces are, and the entropy term and the
    electrons per listed level it used (from the electron count over the weighted occupations)."""

    energy = float(data["energy_total_ry"])
    if temperature <= 0.0:
        return energy, 0.0, None
    if "occupations_up" in data.files:
        channels = [np.asarray(data["occupations_up"], float), np.asarray(data["occupations_down"], float)]
    elif "spin_occupations" in data.files:
        channels = [np.asarray(row, float) for row in np.asarray(data["spin_occupations"], float)]
    else:
        channels = [np.asarray(data["occupations"], float)]
    weights = np.asarray(data["k_weights"], float) if "k_weights" in data.files else np.ones(1)
    filled = entropy = 0.0
    scale = 2.0 if max(float(np.nanmax(c)) if c.size else 0.0 for c in channels) > 1.0 + 1.0e-9 else 1.0
    for occupations in channels:
        f = occupations.ravel() / scale
        f = np.where(np.isfinite(f), f, 0.0)
        if f.size % weights.size:
            rows, w = f.reshape(1, -1), np.ones(1)
        else:
            rows, w = f.reshape(weights.size, -1), weights
        clipped = np.clip(rows, 1.0e-300, 1.0 - 1.0e-16)
        s = clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
        s = np.where((rows <= 0.0) | (rows >= 1.0), 0.0, s)
        filled += float(np.dot(w, rows.sum(axis=1)))
        entropy += float(np.dot(w, s.sum(axis=1)))
    electrons = float(data["electron_count"])
    factor = electrons / filled if filled > 0.0 else None
    term = (factor or 0.0) * KELVIN_RY * temperature * entropy
    return energy + term, term, factor


def read_archive(path, temperature, result):
    with np.load(path) as data:
        result.energy = float(data["energy_total_ry"])
        result.free, result.entropy_term, result.levels_factor = free_energy(data, temperature)
        result.iterations = int(data["iterations"]) if "iterations" in data.files else None
        result.converged = bool(data["converged"]) if "converged" in data.files else None
        if "forces_total" in data.files:
            total = np.asarray(data["forces_total"], float)
            result.forces = total + np.asarray(data["forces_net"], float) / total.shape[0]
            result.reported = total
        for key in ("net_magnetic_moment", "magnetic_moment"):
            if key in data.files and np.asarray(data[key]).size == 1:
                result.moment = float(data[key])
                break
        result.atoms = np.asarray(data["atom_coordinates_bohr"], float)
        result.grid = np.asarray(data["coordinates_bohr"], float) if "coordinates_bohr" in data.files else None
        for prefix in ("relaxation", "dynamics"):
            if f"{prefix}_energies" in data.files:
                result.trajectory_energies = np.asarray(data[f"{prefix}_energies"], float)
                result.trajectory_positions = np.asarray(data[f"{prefix}_positions"], float)
    bands = Path(path).with_name("parsec_python_bands.npz")
    if bands.exists():
        with np.load(bands) as data:
            result.bands = np.asarray(data["eigenvalues_ry"], float)


class Runner:
    """Runs jobs as fresh processes of the command line, each on its own GPU(s)."""

    def __init__(self, out: Path, gpus: list, timeout: float, threads: int, keep: bool, extra_args=()):
        self.out = out
        self.extra_args = tuple(extra_args)
        self.free_gpus = list(gpus)
        self.lock = threading.Condition()
        self.timeout = timeout
        self.threads = threads
        self.keep = keep
        self.records = {}

    def _take(self, count):
        with self.lock:
            while len(self.free_gpus) < count:
                self.lock.wait()
            taken, self.free_gpus = self.free_gpus[:count], self.free_gpus[count:]
            return taken

    def _give(self, gpus):
        with self.lock:
            self.free_gpus.extend(gpus)
            self.lock.notify_all()

    def write(self, job: Job) -> Path:
        folder = self.out / job.folder
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / "parsec.in").write_text(job.structure.text(job.positions))
        for target, source in job.structure.files().items():
            shutil.copyfile(source, folder / target)
        if job.structure.extra_files is not None:
            for target, text in job.structure.extra_files(job.positions).items():
                (folder / target).write_text(text)
        return folder

    def environment(self, gpus, extra):
        environment = dict(os.environ)
        for name in MPI_VARIABLES:
            environment.pop(name, None)
        source = str(ROOT / "src")
        environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        environment["OMP_NUM_THREADS"] = str(self.threads * len(gpus))
        environment["OPENBLAS_NUM_THREADS"] = str(self.threads * len(gpus))
        environment.update(extra)
        return environment

    def run(self, job: Job) -> Result:
        folder = self.write(job)
        text = (folder / "parsec.in").read_text()
        gpus = self._take(job.structure.gpus)
        started = time.perf_counter()
        result = Result()
        try:
            env = dict(job.structure.env)
            env.update(job.env)
            args = tuple(job.structure.args) + tuple(job.args) + self.extra_args
            with open(folder / "stdout.txt", "w") as out, open(folder / "stderr.txt", "w") as err:
                try:
                    result.status = subprocess.call(
                        [sys.executable, "-c", CLI_CODE, "parsec.in", *args], cwd=folder,
                        env=self.environment(gpus, env), stdout=out, stderr=err, timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired:
                    result.status, result.error = -9, f"timeout {self.timeout:.0f} s"
        finally:
            self._give(gpus)
        result.seconds = time.perf_counter() - started
        archive = folder / "parsec_python_results.npz"
        if archive.exists():
            try:
                read_archive(archive, temperature_of(text), result)
                result.cell, result.periodic = job.structure.lattice, job.structure.periodic
                result.moved = result.distance(result.atoms - job.positions).max()
                if result.grid is not None:
                    result.dgrids = [result.grid_distance(atom) for atom in range(len(result.atoms))]
            except Exception as error:  # noqa: BLE001 - reported as the run's error
                result.error = f"archive: {error}"
                result.free = None
        elif result.error is None:
            result.error = f"status {result.status}, no archive"
        if result.status not in (0, None):
            result.tail = (folder / "stderr.txt").read_text(errors="replace")[-400:]
        self.tidy(folder, keep_output=job.keep_output or not result.ok)
        self.records[job.folder] = result.record()
        return result

    def tidy(self, folder: Path, keep_output: bool):
        if self.keep:
            return
        for path in folder.iterdir():
            if path.name in ("parsec.in", "stderr.txt") or (keep_output and path.name in ("parsec.out", "stdout.txt")):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        stderr = folder / "stderr.txt"
        if stderr.exists() and stderr.stat().st_size == 0:
            stderr.unlink()


def snap_structures(runner: Runner, structures: list) -> None:
    """Move the ``snap`` atom of each structure onto its nearest grid point (the grid of a one-iteration run)."""

    pending = [s for s in structures if s.snap is not None and not getattr(s, "_snapped", False)]
    if not pending:
        return

    def probe(structure):
        transform = lambda text: re.sub(r"^Max_Iter:.*$", "Max_Iter: 1", text, flags=re.MULTILINE).replace(
            "Calculate_Force: .true.\n", ""
        )
        probe_structure = structure.with_text(structure.name + "_probe", transform)
        probe_structure.snap = None
        job = Job(f"_snap/{structure.name}", probe_structure, structure.positions.copy())
        folder = runner.write(job)
        gpus = runner._take(1)
        try:
            subprocess.call(
                [sys.executable, "-c", CLI_CODE, "parsec.in", *runner.extra_args], cwd=folder,
                env=runner.environment(gpus, structure.env),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=runner.timeout,
            )
        finally:
            runner._give(gpus)
        archive = folder / "parsec_python_results.npz"
        if not archive.exists():
            print(f"NOTE snap of {structure.name}: no archive, the atom stays", flush=True)
            return
        with np.load(archive) as data:
            grid = np.asarray(data["coordinates_bohr"], float)
        atom = structure.snap
        difference = grid - structure.positions[atom]
        inverse = np.linalg.inv(structure.lattice)
        fractions = difference @ inverse
        fractions[:, : structure.periodic] -= np.rint(fractions[:, : structure.periodic])
        difference = fractions @ structure.lattice
        nearest = int(np.argmin(np.linalg.norm(difference, axis=1)))
        structure.positions[atom] = structure.positions[atom] + difference[nearest]
        structure._snapped = True
        print(f"NOTE snapped atom {atom} of {structure.name} by {np.linalg.norm(difference[nearest]):.6f} bohr to "
              f"{structure.positions[atom].tolist()}", flush=True)
        runner.tidy(folder, keep_output=False)

    with ThreadPoolExecutor(max_workers=max(1, len(runner.free_gpus))) as pool:
        list(pool.map(probe, pending))


# ---------------------------------------------------------------------------------------------------------------
# The matrix


def linear(structure):
    """``structure`` without its ``Cubic_Spline`` lines: the interpolation of ``PARSEC_CUBIC_SPLINE_DEFAULT`` (the
    cubic spline by default, the linear radial tables of PARSEC 1.5 under ``off``)."""

    def strip(text):
        return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("Cubic_Spline"))

    return structure.with_text(structure.name, strip)


def force_checks():
    c8p = lambda n, f: diamond(n, "C_POTRE", A_C, "pbe", 20, fortran=f)
    c8f = lambda n, f: diamond(n, "C_FHI", A_C, "pbe", 20, fortran=f)
    ge8 = lambda n, f, **k: diamond(n, "Ge_POTRE", A_GE, "ca", 24, fortran=f, **k)
    si8 = lambda n, f: diamond(n, "Si_POTRE", A_SI, "ca", 24, fortran=f)
    on, near = None, (6, 5.0e-4)
    table = [
        # name, structure, targets, offset
        ("f_c8p_d_grid", c8p("c8p_d", False), ((6, 0), (0, 0)), on),
        ("f_c8p_f_grid", c8p("c8p_f", True), ((6, 0), (0, 0)), on),
        ("f_c8p_d_5e-4", c8p("c8p_d", False), ((6, 0), (6, 2)), near),
        ("f_c8p_f_5e-4", c8p("c8p_f", True), ((6, 0),), near),
        ("f_c8f_d_grid", c8f("c8f_d", False), ((6, 0), (0, 0)), on),
        ("f_c8f_f_grid", c8f("c8f_f", True), ((6, 0),), on),
        ("f_c8f_d_5e-4", c8f("c8f_d", False), ((6, 0),), near),
        ("f_c8f_f_5e-4", c8f("c8f_f", True), ((6, 0),), near),
        ("f_ge8_d_grid", ge8("ge8_d", False), ((6, 0), (0, 0)), on),
        ("f_ge8_f_5e-4", ge8("ge8_f", True), ((6, 0),), near),
        ("f_si8_d_grid", si8("si8_d", False), ((6, 0), (0, 0)), on),
        ("f_si8_f_5e-4", si8("si8_f", True), ((6, 0),), near),
        ("f_ge8k_d_grid", ge8("ge8k_d", False, mesh=(2, 2, 2)), ((6, 0), (0, 0)), on),
        ("f_ge8k_f_grid", ge8("ge8k_f", True, mesh=(2, 2, 2)), ((6, 0),), on),
        ("f_ge8s_d_grid", ge8("ge8s_d", False, spin=True), ((6, 0), (0, 0)), on),
        ("f_ge8o_d_grid", ge8("ge8o_d", False, soc=True), ((6, 0), (0, 0)), on),
        ("f_mgo8_d", mgo8("mgo8_d", fortran=False), ((0, 0), (4, 0)), on),
        ("f_mgo8_f", mgo8("mgo8_f", fortran=True), ((0, 0),), on),
        ("f_ag4_d_grid", ag4("ag4_d"), ((1, 0), (0, 0)), on),
        ("f_ag4u_d_grid", ag4("ag4u_d", u=True), ((1, 0), (0, 0)), on),
        ("f_ag4u_f_5e-4", ag4("ag4u_f", u=True, fortran=True), ((1, 0),), (1, 5.0e-4)),
        ("f_ch4_grid", ch4("ch4"), ((0, 0), (1, 0)), on),
        ("f_ch4_5e-4", ch4("ch4"), ((0, 0),), (0, 5.0e-4)),
        ("f_h2o_grid", h2o("h2o"), ((0, 1), (1, 0)), on),
        ("f_h2o_5e-4", h2o("h2o"), ((0, 1),), (0, 5.0e-4)),
        ("f_ag3_grid", ag3("ag3"), ((0, 0), (1, 0)), on),
        ("f_o2s_grid", o2("o2s"), ((0, 2), (1, 0)), on),
        ("f_bnslab_f_grid", bn_slab("bnslab_f"), ((0, 0), (0, 2), (1, 0)), on),
        ("f_bnslab_d_grid", bn_slab("bnslab_d", fortran=False), ((0, 2), (1, 0)), on),
        ("f_cwire_grid", c_wire("cwire"), ((0, 0), (0, 2), (1, 0)), on),
        # a slab whose density reaches the edge of its domain, with the Poisson potential and with the derivative of
        # the Hartree energy (added after 30edaacf1; the derivative is the default of a slab since b72375f60)
        ("f_ge2slab", ge2_slab("ge2slab", env={"PARSEC_HARTREE_VARIATIONAL": "off"}), ((0, 2), (0, 0)), on),
        ("f_ge2slab_var", ge2_slab("ge2slab_var"), ((0, 2), (1, 2)), on),
        # oblique cells at the Gamma point (added after a6401263f): hcp Mg on Fortran's general grid, which the command
        # line ran on the reference solver before PARSEC_GENERAL_GRID_ACCELERATED, and diamond in its primitive fcc
        # cell on the default routines.  Moves along c only for Mg: a move in the plane gives the structure the near
        # fractional translations of its space group, from which adjust_step takes 15 instead of 14 points along a1
        # (the slope would compare two grids); atom 1 only for C: atom 0 at the corner of the cell leaves it at either
        # step, which takes the input to the Fortran-derived routines and their grid
        ("f_mg2g_f", hcp_mg2("mg2g_f", "", criterion=1.0e-10), ((0, 2), (1, 2)), on),
        ("f_c2g_d", primitive_c2("c2g_d", "", criterion=1.0e-10), ((1, 0), (1, 2)), on),
        # primitive cubic supercells in the sectors of their full cubic group (PARSEC_OBLIQUE_FULL_GROUP=auto, added
        # after 308edad4e): the moved atom along z and an atom on its site
        ("f_si16_fcc_fg", primitive_cubic("si16_fcc_fg", "fcc", criterion=1.0e-10), ((1, 2), (2, 0)), on),
        ("f_na8_bcc_fg", primitive_cubic("na8_bcc_fg", "bcc", criterion=1.0e-10), ((1, 2), (2, 0)), on),
        # Au on a grid point with PBE (added after 308edad4e), at the steps of FINE_STEPS: the energy varies on a scale
        # of 1e-3 bohr there (au2_gga), so that the slopes over 1e-3 and 2e-3 bohr miss the force (cell 1.1e-4 and
        # 2.5e-5, cluster 1.7e-4 and 3.2e-4 Ry/bohr) while those over 2e-5 and 4e-5 bohr approach it as the square of
        # the step (cell 4.9e-7 and 1.3e-6, cluster 8.1e-8 and 4.4e-7 Ry/bohr on A100)
        ("f_au2_pbe_fine", au2_gga("au2_pbe_fine"), ((0, 0),), on),
        ("f_au2c_pbe_fine", au2_gga("au2c_pbe_fine", cluster=True), ((0, 0),), on),
        # no Cubic_Spline in the input (added after the first run, when such an input took PARSEC 1.5's linear tables
        # and these cases were exempt; after 30edaacf1 the default of PARSEC_CUBIC_SPLINE_DEFAULT is the cubic spline)
        ("f_si8k_lin_d", linear(diamond("si8k_lin_d", "Si_POTRE", A_SI, "ca", 24, mesh=(2, 2, 2))),
         ((6, 0), (0, 0)), on),
        ("f_ch4_lin", linear(ch4("ch4_lin")), ((0, 0), (1, 0)), on),
        ("f_h2o_lin", linear(h2o("h2o_lin")), ((0, 1), (1, 0)), on),
        # a meta-GGA of a Gamma-point cell in the weighted sectors of Td (added after 308edad4e): SCAN with atoms
        # 4-7 on grid points (default routines) and off them (Fortran-derived routines)
        ("f_si8td_scan_d_grid", si8_td_meta("si8td_scan_d"), ((4, 0), (5, 1)), on),
        ("f_si8td_scan_f_off", si8_td_meta("si8td_scan_f", fortran=True, steps=0.37), ((4, 0), (5, 1)), on),
        # self-consistent spin-orbit clusters on the device (added after 308edad4e, PARSEC_SOC_DEVICE_FORCES=auto;
        # the reference solver took them before): AuH with Au on a grid point on the whole grid (below the floor of
        # the sectors), AuH with Au 5e-4 bohr from one in the Kramers sector, Au2 off the grid in its parity sectors
        ("f_auh_soc_grid", au2_soc("auh_soc_grid", hydride=True, criterion=1.0e-10, forces=True,
                                   positions=[[0.0, 0.0, -1.4], [0.0, 0.0, 1.3]]), ((0, 2), (1, 2)), on),
        ("f_auh_soc_kramers", au2_soc("auh_soc_kramers", hydride=True, criterion=1.0e-10, forces=True,
                                      positions=[[0.0, 0.0, -1.3995], [0.0, 0.0, 1.35]],
                                      env={"PARSEC_SOC_SECTORS": "kramers"}), ((0, 2), (1, 2)), on),
        ("f_au2_soc_on", au2_soc("au2_soc_on", criterion=1.0e-10, forces=True,
                                 positions=[[0.0, 0.0, -2.31], [0.0, 0.0, 2.31]],
                                 env={"PARSEC_SOC_SECTORS": "on"}), ((0, 2),), on),
        # the same AuH on the Fortran-derived routines (their spin-orbit gradients), in the Kramers sector
        ("f_auh_soc_f", au2_soc("auh_soc_f", hydride=True, criterion=1.0e-10, forces=True, fortran=True,
                                positions=[[0.0, 0.0, -1.3995], [0.0, 0.0, 1.35]],
                                env={"PARSEC_SOC_SECTORS": "kramers"}), ((0, 2), (1, 2)), on),
        # germanium (its spin-orbit channels and core correction): the apex of a Ge5 pyramid on a grid point, a base
        # atom off the grid, in the Kramers sector; and the Au20 tetrahedron (its sector, the moved atom on the grid)
        ("f_ge5_soc_kramers", ge5_soc("ge5_soc_kramers", env={"PARSEC_SOC_SECTORS": "kramers"}), ((0, 2), (1, 0)),
         on),
        ("f_au20_soc", au20_soc("au20_soc"), ((0, 0),), on),
        # metals with a hybrid on the wedge of their mesh (added after 308edad4e, PARSEC_HYBRID_KPOINT_WEDGE=on by
        # default): bcc Au2 and AuAg on the default routines, AuAg on the Fortran-derived ones (Au2's group keeps no
        # grid of theirs), hcp Na2 on the oblique grid of the default routines
        ("f_au2_pbe0_d", au2_pbe0("au2_pbe0_d"), ((1, 0), (1, 2), (0, 0)), on),
        ("f_auag_pbe0_f", au2_pbe0("auag_pbe0_f", fortran=True, second="Agn_POTRE"), ((1, 0), (0, 0)), on),
        ("f_auag_pbe0_d", au2_pbe0("auag_pbe0_d", second="Agn_POTRE"), ((1, 0), (0, 0)), on),
        ("f_na2_pbe0_k333", hcp_na2("na2_pbe0", (3, 3, 3), criterion=1.0e-10, xc="pbe0", extra=METAL_HYBRID_EXX),
         ((0, 2), (1, 2)), on),
        # bcc Au2 of the group {E, x <-> y}: atom 1 along z keeps it (the wedge at every step), atom 1 along y and atom 0
        # along x remove it (the displaced structures on the time-reversal pairs)
        ("f_au2b_pbe0_d", au2b_pbe0("au2b_pbe0_d"), ((1, 2), (1, 1), (0, 0)), on),
    ]
    notes = {
        "f_ag4u_d_grid": "atom 0: LDA+U energy kink reported before",
    }
    # Documented exceptions.  (Up to b72375f60 the FHI cases of an atom on a grid point were exempt at 1e-3 bohr: an FHI
    # table starts at 0.00625/Z bohr, and the padded spline below its first sample and the projector values of its
    # first eight samples gave the energy features within about 1.5e-3 bohr of the atom.  Under PARSEC_FHI_ORIGIN=auto
    # the reader continues the table to 1e-5 bohr, replaces its first samples (FHI98PP's start) by their even fit and
    # the projector gradients are exact from 2 r(17) on: these cases, and those 5e-4 bohr from a grid point, are
    # ordinary checks (at converged steps of 2e-5 / 5e-5 bohr within 1.4e-8 Ry/bohr); under off they fail at 1e-3.)
    exempt = {
        # the Hartree operator of the slab is not symmetric where the density reaches the edge of its domain: the
        # potential of the Poisson solve is not the derivative of the Hartree energy, the SCF does not make the energy
        # stationary, and the force misses the term of the density response at every step (1.05e-5 Ry/bohr on
        # 30edaacf1); f_ge2slab_var takes the derivative (the default PARSEC_HARTREE_VARIATIONAL=slab)
        "f_ge2slab": {0: (FORCE_STEPS, "slab Hartree operator not symmetric where the density reaches the edge of the "
                          "slab under PARSEC_HARTREE_VARIATIONAL=off (the default slab: f_ge2slab_var)")},
        # an Au atom on a grid point with a GGA: the finite-difference gradient of the density at a grid point on a
        # nucleus of low pseudo-valence density (the 5d shell) makes the GGA energy vary on a scale of 1e-3 bohr there,
        # so the slopes over steps of 1e-3 and 2e-3 bohr miss the force by 1.1e-4 (PBE) to 2.2e-4 Ry/bohr (PBE0), on
        # the time-reversal pairs of PBE0 as on its wedge, while the force is the derivative of that energy (steps of
        # 2e-5 and 4e-5 bohr; the CA functional agrees within 2e-8 at the default steps).  Bounded at 5e-4 Ry/bohr.
        "f_au2_pbe0_d": {0: (FORCE_STEPS, _GGA_ON_GRID, 5.0e-4)},
        "f_auag_pbe0_d": {0: (FORCE_STEPS, _GGA_ON_GRID, 5.0e-4)},
        "f_auag_pbe0_f": {0: (FORCE_STEPS, _GGA_ON_GRID, 5.0e-4)},
        # a displacement that removes the group of a metal hybrid: the displaced structures run on the time-reversal
        # pairs (no wedge group left), whose Fock loop may break into a lower state (bcc Au2 at 1000 K: 1.3e-5 Ry below
        # the wedge state of the base), so their slope misses the force of the wedge by about 1e-5 Ry/bohr; a
        # displacement that keeps a subgroup of the wedge passes (atom 1 along z).  Bounded at 5e-5 Ry/bohr.
        "f_au2b_pbe0_d": {0: (FORCE_STEPS, _METAL_PAIRS, 5.0e-5), 1: (FORCE_STEPS, _METAL_PAIRS, 5.0e-5)},
    }
    # The linear radial tables of PARSEC 1.5 (PARSEC_CUBIC_SPLINE_DEFAULT=off for an input without Cubic_Spline) have
    # a kink at every radial sample a grid point crosses, so the slope of the energy over a step is not the force at
    # the point; with the default cubic spline the three cases above are ordinary checks.
    # Cases whose energy varies on a scale below the steps of FORCE_STEPS: steps that resolve it
    fine = {"f_au2_pbe_fine": FINE_STEPS, "f_au2c_pbe_fine": FINE_STEPS}
    return [ForceCheck("force", name, notes.get(name, ""), structure, targets, offset, exempt=exempt.get(name, {}),
                       steps=fine.get(name, FORCE_STEPS))
            for name, structure, targets, offset in table]


def symmetry_checks():
    si8 = lambda name, mesh: diamond(name, "Si_POTRE", A_SI, "ca", 24, mesh=mesh, criterion=1.0e-9)
    half = 0.5 * A_C
    c2_lattice = np.array([[0.0, half, half], [half, 0.0, half], [half, half, 0.0]])
    a, c = 3.209 * BOHR_PER_ANGSTROM, 5.211 * BOHR_PER_ANGSTROM
    mg_lattice = np.array([[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]])
    bn_lattice = np.array([[4.73, 0.0, 0.0], [-2.365, 4.0963001599, 0.0], [0.0, 0.0, 1.0]])
    na_lattice = hcp_na2("lattice", (1, 1, 1)).lattice
    k_off = {"PARSEC_KPOINT_SECTORS": "off", "PARSEC_PERIODIC_SYMMETRY": "off"}
    checks = [
        ("s_c8p_c3v_gamma", diamond("c8p_c3v", "C_POTRE", A_C, "pbe", 20, criterion=1.0e-9), None),
        ("s_c8p_cs_gamma", diamond("c8p_cs", "C_POTRE", A_C, "pbe", 20, move=(0.04, 0.04, 0.0), criterion=1.0e-9),
         None),
        ("s_c8p_td_f_gamma", diamond("c8p_td_f", "C_POTRE", A_C, "pbe", 20, fortran=True, move=(0.0, 0.0, 0.0),
                                     criterion=1.0e-9), None),
        ("s_ag4_oh_gamma", ag4("ag4_oh", move=(0.0, 0.0, 0.0), criterion=1.0e-9), None),
        ("s_ag4u_c3v_gamma", ag4("ag4u_c3v", u=True, criterion=1.0e-9), None),
        ("s_si8_c3v_k333_list", si8("si8_k333", (3, 3, 3)), si8("si8_k333_list", full_mesh_list(np.diag([A_SI] * 3), (3, 3, 3)))),
        ("s_si8_c3v_k333_sectors", si8("si8_k333", (3, 3, 3)), "sectors"),
        ("s_c2_fcc_k333_list", primitive_c2("c2_k333", (3, 3, 3)), primitive_c2("c2_k333_list", full_mesh_list(c2_lattice, (3, 3, 3)))),
        # default routines (POTRE, no core correction): the whole mesh written with equal weights, one image a
        # point (a list of unequal weights would take the Fortran-derived routines); the default routines run each
        # point of the zone boundary at all its shortest images (PARSEC_KPOINT_LIST_FOLD), as the mesh does
        ("s_na2_hcp_k333_list", hcp_na2("na2_k333", (3, 3, 3)),
         hcp_na2("na2_k333_list", full_mesh_list(na_lattice, (3, 3, 3), images="centred"))),
        # Fortran-derived routines (core correction): the whole mesh at the images the mesh computes
        ("s_mg2_hcp_k333_f_list", hcp_mg2("mg2_k333_f", (3, 3, 3)),
         hcp_mg2("mg2_k333_f_list", full_mesh_list(mg_lattice, (3, 3, 3), images="centred"))),
        # oblique cells at the Gamma point in the sectors of the frame of their grid indices (PARSEC_OBLIQUE_SECTORS,
        # the default after a6401263f; these cells are below the floor of the sectors, which is off here); hcp Mg on
        # Fortran's general grid
        ("s_mg2_hcp_gamma_f", replace(hcp_mg2("mg2g_sym_f", "", criterion=1.0e-9),
                                      env={"PARSEC_PERIODIC_SECTOR_FLOOR": "off"}), None),
        ("s_c2_fcc_gamma", replace(primitive_c2("c2g_sym", "", criterion=1.0e-9),
                                   env={"PARSEC_PERIODIC_SECTOR_FLOOR": "off"}), None),
        # primitive cubic supercells in the sectors of their full cubic group (PARSEC_OBLIQUE_FULL_GROUP=auto, added
        # after 308edad4e), both routine sets for Si
        ("s_si16_fcc_fg", primitive_cubic("si16_fcc_fg_sym", "fcc"), None),
        ("s_si16_fcc_fg_f", primitive_cubic("si16_fcc_fg_sym_f", "fcc", fortran=True), None),
        ("s_si16_fcc_fg_perfect", primitive_cubic("si16_fcc_fg_perfect", "fcc", move=0.0, forces=False), None),
        ("s_na8_bcc_fg", primitive_cubic("na8_bcc_fg_sym", "bcc"), None),
        ("s_h2o_c2v", h2o("h2o_c2v", criterion=1.0e-9), None),
        ("s_ch4_td", ch4("ch4_td", perturb=False, criterion=1.0e-9), None),
        ("s_o2_spin", o2("o2_sym", symmetric=True, criterion=1.0e-9), None),
        # hybrids of clusters (added after 308edad4e): the weighted sectors of Td and D4h (PARSEC_ORBITAL_GROUP_HYBRID)
        # against the full grid, unpolarized and spin
        ("s_ch4_td_pbe0", ch4("ch4_td_pbe0", perturb=False, criterion=1.0e-9, xc="pbe0"), None),
        ("s_o2_spin_pbe0", o2("o2_sym_pbe0", symmetric=True, criterion=1.0e-9, xc="pbe0"), None),
        ("s_c7n_spin_gamma", c7n("c7n", criterion=1.0e-9), None),
        ("s_c7n_spin_k333_list", c7n("c7n_k333", mesh=(3, 3, 3), criterion=1.0e-9),
         c7n("c7n_k333_list", mesh=full_mesh_list(np.diag([A_C] * 3), (3, 3, 3)), criterion=1.0e-9)),
        # the default routines of a slab (POTRE; no forces, which take a slab to the Fortran-derived routines)
        ("s_bnslab_k331_list", bn_slab("bn_k331", fortran=False, mesh=(3, 3, 1), snap=False, criterion=1.0e-9,
                                       potre=True, forces=False),
         bn_slab("bn_k331_list", fortran=False, mesh=full_mesh_list(bn_lattice, (3, 3, 1), periodic=2,
                                                                    images="centred"),
                 snap=False, criterion=1.0e-9, potre=True, forces=False)),
        ("s_bnslab_k331_f_list", bn_slab("bn_k331_f", mesh=(3, 3, 1), snap=False, criterion=1.0e-9),
         bn_slab("bn_k331_f_list", mesh=full_mesh_list(bn_lattice, (3, 3, 1), periodic=2, images="centred"),
                 snap=False, criterion=1.0e-9)),
        # Gamma-point sectors of a slab and a wire (PARSEC_SLAB_WIRE_SECTORS, added after a6401263f; the floor off)
        # against the full grid
        ("s_bnslab_rect_gamma", bn_rect_slab("bn_rect"), None),
        ("s_ribbon_wire_gamma", ribbon_wire("ribbon"), None),
        # the weighted sectors of a meta-GGA of a Gamma-point cell (added after 308edad4e)
        ("s_si8td_scan_gamma", si8_td_meta("si8td_sym", steps=0.37, criterion=1.0e-9), None),
    ]
    out = []
    for name, structure, reference in checks:
        if reference == "sectors":
            out.append(SymmetryCheck("symmetry", name, "", structure, None, dict(k_off), NO_SYMMETRY_ARGS))
        else:
            out.append(SymmetryCheck("symmetry", name, "", structure, reference))
    return out


def switch_checks():
    c8p = lambda n, f, **k: diamond(n, "C_POTRE", A_C, "pbe", 20, fortran=f, **k)
    table = [
        ("w_c8p_d_grid", c8p("w_c8p_d", False)),
        ("w_c8p_perfect_d", c8p("w_c8p_perfect", False, move=(0.0, 0.0, 0.0), forces=False)),
        ("w_c8p_perfect_d_4gpu", c8p("w_c8p_perfect4", False, move=(0.0, 0.0, 0.0), forces=False)),
        ("w_c8f_f_5e-4", diamond("w_c8f_f", "C_FHI", A_C, "pbe", 20, fortran=True)),
        ("w_ge8_f", diamond("w_ge8_f", "Ge_POTRE", A_GE, "ca", 24, fortran=True)),
        ("w_ge8_d", diamond("w_ge8_d", "Ge_POTRE", A_GE, "ca", 24, fortran=False)),
        ("w_ge8o_d", diamond("w_ge8o", "Ge_POTRE", A_GE, "ca", 24, soc=True)),
        ("w_bnslab_f", bn_slab("w_bnslab", snap=False)),
        ("w_cwire", c_wire("w_cwire", snap=False)),
        ("w_si8_k333_d", diamond("w_si8_k333", "Si_POTRE", A_SI, "ca", 24, mesh=(3, 3, 3))),
        ("w_si8_k222_d", diamond("w_si8_k222", "Si_POTRE", A_SI, "ca", 24, mesh=(2, 2, 2))),
        ("w_mg2_hcp_k333", hcp_mg2("w_mg2_k333", (3, 3, 3), criterion=1.0e-10)),
        ("w_mg2_hcp_list", hcp_mg2("w_mg2_list", full_mesh_list(
            np.array([[3.209 * BOHR_PER_ANGSTROM, 0, 0], [-0.5 * 3.209 * BOHR_PER_ANGSTROM,
                      0.5 * np.sqrt(3.0) * 3.209 * BOHR_PER_ANGSTROM, 0], [0, 0, 5.211 * BOHR_PER_ANGSTROM]]),
            (3, 3, 2)), criterion=1.0e-10)),
        ("w_ch4_orbits", ch4("w_ch4_orbits", perturb=False, offset_h=3.0e-6)),
        ("w_ag4u_d", ag4("w_ag4u_d", u=True)),
        ("w_ag4u_f", ag4("w_ag4u_f", u=True, fortran=True)),
        ("w_c7n_spin_d", c7n("w_c7n_d")),
        ("w_c7n_spin_f", c7n("w_c7n_f", fortran=True)),
        ("w_o2_spin", o2("w_o2")),
        ("w_si_pbe0_k222", si_pbe0("w_si_pbe0")),
        ("w_c8p_pbe0_gamma", c8_pbe0("w_c8p_pbe0")),
        # atoms on grid points (the Gaussian term of a wire or slab next to its atom)
        ("w_bnslab_f_grid", bn_slab("w_bnslab_grid")),
        ("w_cwire_grid", c_wire("w_cwire_grid")),
        # the default routines of an oblique cell: the wedge, the fold and the force group of a mesh, the fold of a list
        ("w_na2_hcp_k333", hcp_na2("w_na2_k333", (3, 3, 3), criterion=1.0e-10)),
        ("w_na2_hcp_list", hcp_na2("w_na2_list", full_mesh_list(hcp_na2("lattice", (1, 1, 1)).lattice, (3, 3, 3),
                                                                images="centred"), criterion=1.0e-10)),
        # starting polarizations of both signs (PARSEC_SPIN_START); 1e-6 Ry: from the former start (no polarization)
        # the loop stalls near 3e-7 Ry in 400 steps
        ("w_o2_broken", o2_broken("w_o2_broken", criterion=1.0e-6)),
    ]
    checks = []
    for name, structure in table:
        if name.endswith("_4gpu"):
            structure.gpus = 4
        checks.append(SwitchCheck("switch", name, "", structure))
    return checks


def repeat_checks():
    table = [
        ("r_ch4_td", ch4("r_ch4", perturb=False)),
        ("r_c8p_perfect", diamond("r_c8p", "C_POTRE", A_C, "pbe", 20, move=(0.0, 0.0, 0.0), forces=False)),
        ("r_si8_k333", diamond("r_si8_k333", "Si_POTRE", A_SI, "ca", 24, mesh=(3, 3, 3))),
        ("r_o2_spin", o2("r_o2")),
        ("r_ge8o_soc", diamond("r_ge8o", "Ge_POTRE", A_GE, "ca", 24, soc=True)),
        ("r_ag4u", ag4("r_ag4u", u=True)),
        ("r_ag4", ag4("r_ag4")),
        ("r_c7n_spin_k222", c7n("r_c7n_k222", mesh=(2, 2, 2))),
        ("r_bnslab", bn_slab("r_bnslab", snap=False)),
        ("r_mg2_hcp_k333", hcp_mg2("r_mg2", (3, 3, 3), criterion=1.0e-10)),
    ]
    return [RepeatCheck("repeat", name, "", structure) for name, structure in table]


def trajectory_checks():
    c8 = lambda name, extra: diamond(name, "C_POTRE", A_C, "pbe", 20, criterion=1.0e-9, extra_scf=extra)
    # symmetric starts of the dynamics: the velocity of one atom along the axis of C3v
    restarts = {"ch4": md_restart(["C", "H", "H", "H", "H"], 1, (1.0, 1.0, 1.0)),
                "c8p": md_restart(["C"] * 8, 0, (1.0, 1.0, 1.0))}
    checks = []
    for kind in ("bfgs", "md"):
        for label, make in (("ch4", lambda n, x: ch4_c3v(n, x)), ("c8p", c8)):
            if kind == "bfgs":
                structure = make(f"t_{label}_bfgs", RELAX_LINES)
                symmetric = make(f"t_{label}_bfgs_sym", RELAX_LINES)
                reference = structure
            else:
                # the sectors of the dynamics (PARSEC_MOVING_SECTORS=on: at every size; the default takes them from
                # 2^24 entries on) from a start that keeps C3v, against the full grid; the repeat from seeded random
                # velocities
                symmetric = make(f"t_{label}_md_sym", MD_RESTART_LINES)
                symmetric.extra_files = restarts[label]
                symmetric.env = {"PARSEC_MOVING_SECTORS": "on"}
                reference = make(f"t_{label}_md_full", MD_RESTART_LINES)
                reference.extra_files = restarts[label]
                structure = make(f"t_{label}_md", MD_LINES)
            checks.append(SymmetryCheck("traj", f"t_{label}_{kind}_sym", "", symmetric, reference))
            checks.append(RepeatCheck("traj", f"t_{label}_{kind}_rep", "", structure))
    # an oblique cell on Fortran's general grid (added after a6401263f): its BFGS steps in the sectors of the frame of
    # its grid indices (the floor of the sectors off) against the full grid of those routines (PARSEC_OBLIQUE_SECTORS=
    # off), whose forces and moves are averaged over the operations of those routines too.  (Against --symmetry off,
    # whose forces are not averaged, the paths part by 2.9e-4 bohr in three steps: the operations of these routines
    # include the three-fold axis, which their finite-difference stencil on this grid does not keep exactly.)
    mg2 = replace(hcp_mg2("t_mg2_bfgs", "", criterion=1.0e-9), env={"PARSEC_PERIODIC_SECTOR_FLOOR": "off"})
    mg2 = mg2.with_text("t_mg2_bfgs", lambda text: text + RELAX_LINES)
    checks.append(SymmetryCheck("traj", "t_mg2_bfgs_sym", "", mg2, mg2, reference_env={"PARSEC_OBLIQUE_SECTORS": "off"},
                                reference_args=()))
    # a primitive cubic supercell in the sectors of its full cubic group (PARSEC_OBLIQUE_FULL_GROUP=auto, added after
    # 308edad4e): its BFGS steps against the full grid
    si16 = primitive_cubic("t_si16_bfgs_fg", "fcc", extra=RELAX_LINES)
    checks.append(SymmetryCheck("traj", "t_si16_bfgs_fg", "", si16, si16,
                                reference_env={"PARSEC_OBLIQUE_SECTORS": "off", "PARSEC_OBLIQUE_FULL_GROUP": "off"},
                                reference_args=()))
    # and its dynamics from a start that keeps C2v about z (the moved atom's previous position along z), in the
    # sectors of PARSEC_MOVING_SECTORS=on against the full grid
    si16_md = primitive_cubic("t_si16_md_fg", "fcc", extra=MD_RESTART_LINES)
    si16_md.extra_files = md_restart(["Si"] * 16, 1, (0.0, 0.0, 1.0))
    si16_md.env = dict(si16_md.env, PARSEC_MOVING_SECTORS="on")
    checks.append(SymmetryCheck("traj", "t_si16_md_fg", "", si16_md, si16_md,
                                reference_env={"PARSEC_OBLIQUE_SECTORS": "off", "PARSEC_OBLIQUE_FULL_GROUP": "off",
                                               "PARSEC_MOVING_SECTORS": "off"},
                                reference_args=()))
    # a hybrid of Td (added after 308edad4e): the BFGS steps of CH4 with its bonds 3 % long, PBE0 in the weighted sectors
    # of D2d (PARSEC_ORBITAL_GROUP_HYBRID), the orbitals of a structure building the first operator of the next,
    # against the full grid
    ch4h = ch4("t_ch4_pbe0_bfgs", perturb=False, criterion=1.0e-9, xc="pbe0", scale=1.03, extra=RELAX_LINES)
    checks.append(SymmetryCheck("traj", "t_ch4_pbe0_bfgs_sym", "", ch4h, ch4h))
    return checks


def band_checks():
    no_symmetry = dict(NO_SYMMETRY_ENV, PARSEC_PERIODIC_SECTOR_BANDS="off")
    table = [
        ("b_si8_sym", diamond("b_si8", "Si_POTRE", A_SI, "ca", 24, move=(0.0, 0.0, 0.0), criterion=1.0e-9,
                              forces=False, extra_scf="Bandstruc_Points: 4\n" + BAND_LINES)),
        ("b_c7n_spin_sym", c7n("b_c7n", criterion=1.0e-9, forces=False).with_text(
            "b_c7n", lambda t: t + "Bandstruc_Points: 4\n" + BAND_LINES)),
        ("b_c64_sym", c64_cube("b_c64", move=0.0, forces=False,
                               extra="Bandstruc_Points: 3\nbegin bandstruc\n1 0.0 0.0 0.0  0.5 0.0 0.0  G-X\n"
                                     "end bandstruc\n")),
    ]
    return [SymmetryCheck("band", name, "", structure, None, dict(no_symmetry), NO_SYMMETRY_ARGS)
            for name, structure in table]


def kpoint_checks():
    real_off = {"PARSEC_KPOINT_KINETIC": "trim", "PARSEC_KPOINT_REAL_GAMMA": "off", "PARSEC_KPOINT_TRIM_REAL": "off"}
    real_on = c64_cube("k_c64_real", (2, 2, 2))
    real_on.env = {"PARSEC_KPOINT_KINETIC": "trim"}
    return [
        SymmetryCheck("kpoint", "k_na_guard_list", "", bcc_guarded("k_na_guard", "na", "mesh"),
                      bcc_guarded("k_na_list", "na", "list")),
        SymmetryCheck("kpoint", "k_au_soc_guard_list", "", bcc_guarded("k_au_soc_guard", "au_soc", "mesh"),
                      bcc_guarded("k_au_soc_list", "au_soc", "list")),
        SymmetryCheck("kpoint", "k_c64_k222_real", "", real_on, None, real_off, ()),
        SymmetryCheck("kpoint", "k_c64_k222_sectors", "", c64_cube("k_c64_sectors", (2, 2, 2)), None,
                      {"PARSEC_KPOINT_SECTORS": "off"}, ()),
    ]


def soc_checks():
    checks = []
    for name, make, value in (
        ("o_au2_kramers", lambda n: au2_soc(n), "kramers"),
        ("o_au2_on", lambda n: au2_soc(n), "on"),
        ("o_auh_kramers", lambda n: au2_soc(n, hydride=True), "kramers"),
    ):
        structure = make(name)
        structure.env = {"PARSEC_SOC_SECTORS": value}
        checks.append(SymmetryCheck("soc", name, "", structure, make(name + "_off"), {"PARSEC_SOC_SECTORS": "off"}, ()))
    # (added after 308edad4e) the forces of the device route in the Kramers sector against those of the reference
    # solver (PARSEC_SOC_DEVICE_FORCES=off: its CPU spinor loop on the whole grid and the readable forces)
    positions = [[0.0, 0.0, -1.3995], [0.0, 0.0, 1.35]]
    device = au2_soc("o_auh_forces", hydride=True, criterion=1.0e-10, forces=True, positions=positions,
                     env={"PARSEC_SOC_SECTORS": "kramers"})
    reference = au2_soc("o_auh_forces_reference", hydride=True, criterion=1.0e-10, forces=True, positions=positions)
    checks.append(SymmetryCheck("soc", "o_auh_forces_reference", "", device, reference,
                                {"PARSEC_SOC_DEVICE_FORCES": "off"}, (), force_tolerance=1.0e-7))
    device = au2_soc("o_auh_forces_f", hydride=True, criterion=1.0e-10, forces=True, positions=positions,
                     env={"PARSEC_SOC_SECTORS": "kramers"}, fortran=True)
    reference = au2_soc("o_auh_forces_reference_f", hydride=True, criterion=1.0e-10, forces=True, positions=positions,
                        fortran=True)
    checks.append(SymmetryCheck("soc", "o_auh_forces_reference_f", "", device, reference,
                                {"PARSEC_SOC_DEVICE_FORCES": "off"}, (), force_tolerance=1.0e-7))
    # the double-group sectors of the k-points (PARSEC_SOC_KPOINT_SECTORS): Ge8 with atom 0 moved along z (C2z and the
    # mirrors about z kept), 2 x 2 x 2, forces
    make = lambda n: diamond(n, "Ge_POTRE", A_GE, "ca", 24, soc=True, mesh=(2, 2, 2), move=(0.0, 0.0, 0.05),
                             criterion=1.0e-10)
    structure = make("o_ge8k_soc_sectors")
    structure.env = {"PARSEC_SOC_KPOINT_SECTORS": "on"}
    checks.append(SymmetryCheck("soc", "o_ge8k_soc_sectors", "", structure, make("o_ge8k_soc_sectors_off"),
                                {"PARSEC_SOC_KPOINT_SECTORS": "off"}, ()))
    return checks


def mpi_checks():
    table = [
        ("m_h2o", h2o("m_h2o"), "runner"),
        # no group: the runner's one sector of the identity against the full grid of one process, at a loose
        # criterion, where the two take the same SCF steps only if they solve the same states (added after
        # a6401263f, PARSEC_IDENTITY_SECTOR_STATES)
        ("m_h2o_nosym", h2o("m_h2o_nosym", criterion=1.0e-5, ignore_symmetry=True), "runner"),
        ("m_c8p_gamma", diamond("m_c8p", "C_POTRE", A_C, "pbe", 20), "runner"),
        # D-infinity-h: the runner divides the sectors of both spin channels over the ranks (a spin run without a
        # group is refused there by name)
        ("m_o2_spin", o2("m_o2", symmetric=True), "runner"),
        ("m_c8p_k222", diamond("m_c8p_k222", "C_POTRE", A_C, "pbe", 20, mesh=(2, 2, 2)), "cli"),
        # a band path after a k-point SCF (added after the first run)
        ("m_si8_bands", diamond("m_si8_bands", "Si_POTRE", A_SI, "ca", 24, move=(0.0, 0.0, 0.0), mesh=(2, 2, 2),
                                criterion=1.0e-9, forces=False, extra_scf="Bandstruc_Points: 4\n" + BAND_LINES), "cli"),
        # periodic hybrids on the runner (added 2026-11-23): the sector of the identity below the floor of the
        # sectors, whose small builds the root forms on its GPU, and 32 sectors whose exchange every rank forms in
        # the translation classes of one process (PARSEC_MPI_EXCHANGE_POOL)
        ("m_c8p_pbe0_gamma", c8_pbe0("m_c8p_pbe0", forces=False), "runner"),
        ("m_si64_pbe0_gamma", si64_pbe0("m_si64_pbe0"), "runner"),
    ]
    return [MpiCheck("mpi", name, "", structure, launcher) for name, structure, launcher in table]


GROUPS = {
    "force": force_checks,
    "symmetry": symmetry_checks,
    "switch": switch_checks,
    "repeat": repeat_checks,
    "mpi": mpi_checks,
    "traj": trajectory_checks,
    "band": band_checks,
    "kpoint": kpoint_checks,
    "soc": soc_checks,
}


def select(groups, patterns):
    checks = []
    for group in groups:
        checks.extend(GROUPS[group]())
    if patterns:
        checks = [c for c in checks if any(fnmatch.fnmatch(c.name, p) for p in patterns)]
    return checks


def report(lines_out, group, name, verdict, text, note=""):
    line = f"CONSIST {group} {name} {verdict} {text}" + (f" | note: {note}" if note and verdict != "PASS" else "")
    print(line, flush=True)
    lines_out.append((verdict, line))


def run_checks(checks, runner: Runner, pins: dict, write_pins: Path | None, prepare: bool):
    structures = [s for check in checks for s in check.structures()]
    if prepare:
        for check in checks:
            for job in check.jobs().values():
                runner.write(job)
        print(f"wrote the inputs of {len(checks)} cases to {runner.out}", flush=True)
        return []
    snap_structures(runner, structures)
    lines = []
    width = max(1, len(runner.free_gpus))
    for check in checks:
        if isinstance(check, SwitchCheck):
            check.pinning = write_pins is not None
    with ThreadPoolExecutor(max_workers=width) as pool:
        submitted = [(check, {key: pool.submit(runner.run, job) for key, job in check.jobs().items()})
                     for check in checks]
        for check, futures in submitted:
            results = {key: future.result() for key, future in futures.items()}
            if isinstance(check, SwitchCheck) and write_pins is not None:
                result = results["off"]
                pins[check.name] = dict(result.record(), commit=os.environ.get("CONSIST_COMMIT", "unknown"))
                report(lines, check.group, check.name, "PINNED", result.summary())
                continue
            verdicts = check.judge(results, pins) if isinstance(check, SwitchCheck) else check.judge(results)
            for verdict, text in verdicts:
                report(lines, check.group, check.name, verdict, text, check.note)
    if write_pins is not None:
        write_pins.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
        print(f"pins written to {write_pins}", flush=True)
    return lines


# ---------------------------------------------------------------------------------------------------------------
# MPI


def _rank_size():
    for rank_key, size_key in (("SLURM_PROCID", "SLURM_NTASKS"), ("PMI_RANK", "PMI_SIZE"),
                               ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")):
        if rank_key in os.environ and size_key in os.environ:
            return int(os.environ[rank_key]), int(os.environ[size_key])
    return 0, 1


def run_mpi(check: MpiCheck, out: Path, timeout: float, threads: int) -> list:
    """One MPI case: every rank runs it; rank 0 then runs the same input in one process and compares."""

    rank, size = _rank_size()
    folder = out / check.name / f"ranks{size}"
    inputs = folder / "mpi"
    ready = folder / "ready"
    runner = Runner(folder, [0], timeout, threads, keep=False)
    structure = check.structure
    if rank == 0:
        if folder.exists():
            shutil.rmtree(folder)
        runner.write(Job("mpi", structure, structure.positions.copy()))
        ready.write_text("ok")
    else:
        for _ in range(6000):
            if ready.exists():
                break
            time.sleep(0.1)
    started = time.perf_counter()
    os.environ.update(structure.env)
    if check.launcher == "runner":
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        code = mpi_full_scf.main(["--input", str(inputs / "parsec.in"), "--output-dir", str(inputs), "--devices", "0"])
    else:
        from parsec_python.acceleration import cli

        code = cli.main([str(inputs / "parsec.in"), "--mpi"])
    seconds = time.perf_counter() - started
    if rank != 0:
        return []
    lines = []
    mpi = Result(status=code)
    archive = inputs / "parsec_python_results.npz"
    if archive.exists():
        read_archive(archive, temperature_of((inputs / "parsec.in").read_text()), mpi)
    nodes = int(os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES", "1")) or 1)
    one = runner.run(Job("one", structure, structure.positions.copy(), keep_output=True))
    if not (mpi.ok and one.ok):
        report(lines, "mpi", check.name, "ERROR", f"ranks {size} mpi {mpi.why()} | one {one.why()}")
        return lines
    de = abs(mpi.free - one.free)
    text = (f"ranks {size} nodes {nodes} launcher {check.launcher} E {mpi.free:.12f} one {one.free:.12f} dE {de:.2e} "
            f"it {mpi.iterations}/{one.iterations} seconds {seconds:.1f}/{one.seconds:.1f}")
    df = None
    if mpi.forces is not None and one.forces is not None:
        df = float(np.max(np.abs(mpi.forces - one.forces)))
        text += f" max|dF| {df:.2e}"
    ok = _within(de, MPI_ENERGY_TOLERANCE) and (df is None or _within(df, MPI_FORCE_TOLERANCE))
    extra, extra_ok = compare_extras(mpi, one)
    text += extra
    if extra_ok is False:
        ok = False
    report(lines, "mpi", check.name, "PASS" if ok else "FAIL", text + f" tol {MPI_ENERGY_TOLERANCE:.0e}/{MPI_FORCE_TOLERANCE:.0e}")
    return lines


# ---------------------------------------------------------------------------------------------------------------


def visible_gpus(requested):
    if requested:
        return list(range(int(requested)))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [v for v in visible.split(",") if v.strip()]
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
        count = sum(1 for line in listing.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        count = 0
    return list(range(max(count, 1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", default="all",
                        help="force, symmetry, switch, repeat, traj, band, kpoint, soc, mpi, or all (every one but mpi)")
    parser.add_argument("--cases", default="", help="comma-separated names or patterns (fnmatch)")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("CONSISTENCY_OUT", "consistency_out")))
    parser.add_argument("--gpus", default=None, help="number of GPUs (default: those the process sees)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    parser.add_argument("--pins", type=Path, default=PINS)
    parser.add_argument("--write-pins", type=Path, default=None,
                        help="record the switch cases in this file (relative: below --out) instead of judging them")
    parser.add_argument("--keep", action="store_true", help="keep every file of every run")
    parser.add_argument("--prepare", action="store_true", help="write the inputs only")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--cli-args", default="", help="arguments added to every run of the command line")
    parser.add_argument("--from-results", nargs="+", type=Path, default=None, metavar="JSON",
                        help="judge the selected cases from results_*.json files of earlier runs (no runs)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "HEAD"), default=None,
                        help="judge the switch cases from two files written by --write-pins (no runs)")
    arguments = parser.parse_args(argv)
    groups = (["force", "symmetry", "switch", "repeat", "traj", "band", "kpoint", "soc"] if arguments.group == "all"
              else arguments.group.split(","))
    patterns = [p for p in arguments.cases.split(",") if p]
    checks = select(groups, patterns)
    if arguments.from_results is not None:
        runs = {}
        for path in arguments.from_results:
            runs.update(json.loads(path.read_text())["runs"])
        lines = []
        for check in checks:
            if isinstance(check, (MpiCheck, SwitchCheck)):
                continue
            jobs = check.jobs()
            if not all(job.folder in runs for job in jobs.values()):
                continue
            results = {key: result_of_record(runs[job.folder]) for key, job in jobs.items()}
            for verdict, text in check.judge(results):
                report(lines, check.group, check.name, verdict, text, check.note)
        summarize(lines)
        return 0
    if arguments.compare is not None:
        base = json.loads(arguments.compare[0].read_text())
        head = json.loads(arguments.compare[1].read_text())
        lines = []
        for check in select(["switch"], patterns):
            if check.name in head:
                for verdict, text in check.judge({"off": result_of_record(head[check.name])}, base):
                    report(lines, check.group, check.name, verdict, text + f" head {head[check.name].get('commit', '?')}")
        failed = [line for verdict, line in lines if verdict != "PASS"]
        print(f"CONSIST SUMMARY PASS {len(lines) - len(failed)} FAIL+ERROR {len(failed)}", flush=True)
        return 0
    if arguments.list:
        for check in checks:
            print(check.group, check.name, check.note)
        return 0
    if not checks:
        print("no case selected", flush=True)
        return 2
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if arguments.write_pins is not None and not arguments.write_pins.is_absolute():
        arguments.write_pins = out / arguments.write_pins  # beside the run folders
    gpus = visible_gpus(arguments.gpus)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    threads = max(1, min(16, cpus // (2 * len(gpus))))
    print(f"consistency matrix: {len(checks)} cases, groups {','.join(groups)}, GPUs {gpus}, {threads} threads a "
          f"GPU, tree {ROOT}", flush=True)
    started = time.perf_counter()
    if groups == ["mpi"]:
        lines = []
        for check in checks:
            lines += run_mpi(check, out, arguments.timeout, threads)
        records = {}
    else:
        pins = {}
        if arguments.write_pins is None and arguments.pins.exists():
            pins = json.loads(arguments.pins.read_text())
        runner = Runner(out, gpus, arguments.timeout, threads, arguments.keep, arguments.cli_args.split())
        lines = run_checks(checks, runner, pins, arguments.write_pins, arguments.prepare)
        records = runner.records
    if _rank_size()[0] == 0:
        (out / f"results_{groups[0]}_{int(time.time())}.json").write_text(
            json.dumps({"lines": [line for _, line in lines], "runs": records}, indent=1) + "\n"
        )
        summarize(lines, time.perf_counter() - started)
    return 0


def summarize(lines, seconds=None):
    counts = {v: sum(1 for verdict, _ in lines if verdict == v) for v in ("PASS", "EXEMPT", "FAIL", "ERROR", "PINNED")}
    print(f"CONSIST SUMMARY {' '.join(f'{k} {n}' for k, n in counts.items())}"
          + ("" if seconds is None else f" seconds {seconds:.0f}"), flush=True)
    for verdict, line in lines:
        if verdict in ("FAIL", "ERROR", "EXEMPT"):
            print("  " + line, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
