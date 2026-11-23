"""The k-points of a periodic SCF in the sectors of their little groups (``PARSEC_KPOINT_SECTORS``).

The Bloch Hamiltonian of a k-point acts on the periodic factor ``u_k`` (the u gauge),
``H_k = e^{-ik.r} H e^{ik.r}``.  An operation ``g: r -> R r + t`` of a bulk orthorhombic cell (a Cartesian
signed permutation ``R`` with a translation ``t`` of whole grid steps,
:func:`..Symmetry.axis_reflection._space_group_operations`) permutes the grid points, ``(P_g f)(r) = f(g^-1 r)``,
and commutes with ``H``.  It moves the phase, ``P_g e^{ik.r} P_g^-1 = e^{i (R k).(r - t)}``, so where ``R k = k``
it commutes with ``H_k`` as well (the constant ``e^{-ik.t}`` drops out).  ``R k = k`` exactly, not modulo a
reciprocal lattice vector: where ``R k = k + G``, ``P_g`` carries ``H_k`` into ``H_{k+G} = e^{-iG.r} H_k e^{iG.r}``,
so the operator that commutes with ``H_k`` is ``e^{iG.r} P_g``, a complex operator whose characters are not the
real ``+-1`` of the sector bases below.  (The Bloch form of ``PARSEC_KPOINT_KINETIC=bloch``, ``e^{-ik.r} L e^{ik.r}``
of the finite-difference Laplacian ``L``, obeys that identity exactly, since ``e^{iG.r}`` is periodic on the grid;
the default ``-nabla^2 - 2ik.grad + |k|^2`` of the u gauge only to its discretization, about 1e-8 Ry on the even
meshes of the tests.)  These operations form the little group ``G_k``.

On the grid each term commutes with ``P_g``: the Laplacian and the first derivatives of a uniform periodic grid
are invariant under its translations and turn as a scalar and a vector under ``R`` (the coefficients along the
axes that ``R`` exchanges agree, as their grid steps do where ``R`` maps the grid), so ``k.grad`` is invariant
where ``R^T k = k``.  The Bloch projectors of an atom go over into those of its image times one constant phase
(``e^{ik.(t - T)}`` for ``R tau_a + t = tau_b + T``, the same for every ``m``), which drops out of ``B D B^H``.
The local potential is invariant where the density is (below).

The largest subgroup ``H_k`` of ``G_k`` whose elements commute and are of order two (the half-cell translations,
the twofold axes and the mirrors that fix ``k``) has the real characters ``+-1``: the sector basis ``U`` of
:class:`..Symmetry.representations.ReflectionRepresentationDecomposition` is real, and the complex Hermitian
``H_k`` reduces as a real symmetric operator does, ``U^T H_k U`` from the representative rows
(:meth:`~..Symmetry.representations.ReflectionRepresentationDecomposition.reduce_hermitian_operators`; the kinetic
term, an axial stencil of the grid, on the device as the gathered stencil of :mod:`..Hamiltonian.sector_bloch_stencil`
straight from the grid, ``PARSEC_KPOINT_SECTOR_STENCIL``).  The
spectrum of ``H_k`` is the union of the spectra of the sectors: each k-point becomes ``|H_k|`` eigenproblems of
about ``1/|H_k|`` of its rows and states, and the work of the eigensolver (rows times states squared) falls by
about ``|H_k|^2`` per k-point.  The projectors of one atom per orbit of ``H_k`` serve every sector
(:func:`..Symmetry.axis_reflection.compress_projectors_for_sectors`, which holds for the Bloch projectors as
well: their phases drop out of ``B D B^H``).

The potential is invariant under ``g`` where the density is: the density symmetrized over the space group of
the input (``Symmetrize_Charge_Density``, the wedge of a reduced mesh) under its operations; a density summed
over a set of k-points that ``R`` maps onto itself (with time reversal, modulo the reciprocal lattice, at equal
weights) under ``g`` at self-consistency; every ``rho_k`` under the half-cell translations.  An operation that
meets none of these stays out of ``G_k``.  The sector Hamiltonian takes the potential averaged over each orbit:
its invariant part, the potential itself up to the error of the eigensolver in the iterations before.

A sector of ``n_s`` of the ``N`` rows solves ``M_s = ceil(c M n_s / N) + 6`` states (``M`` the states of a
k-point, ``c`` = ``PARSEC_KPOINT_SECTOR_MARGIN``, 1.3 by default), at most ``n_s``; the k-point keeps the lowest
``M`` levels of the union.  After every solve, each sector with fewer states than rows must reach the ``M``-th
level of the union (``max e_s >= e_M``): every level it did not compute then lies above the levels kept.  A
sector that falls short gets 1.5 times its states and the step is solved again, from new starts.  The margin is
needed only until the levels are known: after the first solve of a loop each sector keeps the ``h_s`` levels it
holds among the ``M`` kept plus a tenth of them, at least six more (``PARSEC_KPOINT_SECTOR_REFIT=auto``, the
default; :meth:`SectorKPointSolver._refit`), its Ritz states cut to their lowest pairs at its next solve, and a sector
so cut that later falls short grows by its estimated deficit (the rule of the Gamma sectors).  A k-point
whose sectors would not halve its work (``sum n_s M_s^2 <= N M^2 / 2``) stays one task, and so do the k-points
of a basis of fewer than ``2^24`` entries (``N M``, 268 MB of complex vectors): their eigenproblems do not fill a
device, and more of them would add the cost of the tasks (the 4631 k-points of an 8-atom cube with a 21^3 mesh).

The density of a sector, ``|U v|^2``, is ``q_s / m_w`` on the rows of the orbit ``w`` of its column ``s``
(``q = sum_n f_n |v_n|^2``, ``m_w`` the orbit size): expanded on the host and summed by the job of the driver.

At a time-reversal invariant point ``k = G/2`` in the Bloch form of the kinetic term the antiunitary ``e^{-iG.r} conj``
commutes with ``H_k`` and carries the sector of a character ``chi`` into that of ``chi lambda``, ``lambda(h) = e^{-iG.t}``
(``-1`` on the half-cell translations along the axes where ``k`` sits at ``pi/L``): the two have one spectrum and the
same densities.  With ``PARSEC_KPOINT_TRIM_REAL=auto`` (the default, :mod:`.kpoint_real`) an unpolarized loop without the exchange of a hybrid solves the lower sector of each pair only
(:class:`SectorKPointSolver`: the partner takes its levels, its occupations go onto the columns of the sector solved,
its orbitals are ``e^{-iG.r} conj(u)``, its nonlocal forces those of ``u``).

The operations of ``G_k`` beyond ``H_k`` that normalize it carry the sector of a character ``chi`` into that of
``chi(g^-1 h g)``, and so does the antiunitary ``P_g K`` of an operation with ``R k = -k`` (``K H_k K = H_{-k}``):
the two sectors have one spectrum.  With ``PARSEC_KPOINT_SECTOR_CLASSES=auto`` (:func:`sector_classes`; operations the
density is symmetrized over, so that the sector Hamiltonians are images of each other at every step) one sector of
each class is solved, the others take its levels and the images ``P_g u`` of its orbitals (:class:`ClassMember`, a
signed permutation of the sector rows), its occupations go onto the columns of the sector solved and the density of
the class is the mean of the images of its density (the weighted sectors of a cluster, :mod:`..Symmetry.orbital_groups`,
on the real characters of ``H_k``).  A sector that such an antiunitary maps onto itself with ``A^2 = 1`` has a real
structure, and with ``PARSEC_KPOINT_SECTOR_REAL=auto`` it is solved on real blocks (:mod:`.kpoint_real_forms`).

The k-point SCF of a bulk orthorhombic cell takes it, scalar or spin-polarized, with the exchange of a hybrid, forces
and kept orbitals, and LDA+U of unpolarized orbitals under ``PARSEC_KPOINT_SECTOR_LDAU=on`` (opt-in), without
meta-GGA or the fixed occupations of ``occup.in`` (the driver then gives the job its
``geometry``); ``PARSEC_KPOINT_SECTORS=off`` keeps one task per k-point, bit for bit as before, and
``PARSEC_HYBRID_KPOINT_SECTORS=on`` (opt-in) splits the k-points of the loops of a hybrid's Fock loop at any size
(:func:`hybrid_kpoint_sectors_setting`, :data:`ANY_SIZE_KEY`).  The loop with refined
levels (the last inner loop of the Fock loop of a hybrid with the adaptive inner criterion or the outer DIIS) takes it
too since 2026-11-19 (``PARSEC_KPOINT_SECTOR_LEVELS=auto``: its level passes solve the sectors, started from the Ritz
states of the loop before; ``off``: one task per k-point from fresh subspaces, as before).

The sectors hold the potential invariant under the operations of ``H_k``, also where only the self-consistent
solution is (a single Gamma point, a full mesh, the translations of a supercell: the operations the density is not
symmetrized over, ``KPointSectors.assumed``).  A spin-polarized state that breaks that symmetry (an open shell, a
Jahn-Teller or magnetic order) cannot be held, as on the sectors of a cluster (``Spin_Symmetry_Sectors``,
:mod:`.spin_sectors`), and the same three tests guard a spin-polarized loop (``PARSEC_KPOINT_SECTOR_GUARD=auto``):
no partly filled multiplet across the sectors of such a k-point in the start Hamiltonian, channel densities
invariant under the operations assumed (:class:`SectorGuardGroup`; with those of the atoms found from their
environments, about a defect off the origin as well, ``PARSEC_SPIN_GUARD_CELL_GROUP``) at the end, and with
Fermi-Dirac occupations no fractional occupation of such a multiplet at the end, each weighted by its k-point (the
electrons it adds to the density).  A loop that fails one runs again with one task per k-point
(:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`).  An unpolarized loop is not tested, as the
unpolarized sectors of a cluster are not.

The Fock loop of a spin-polarized hybrid: its semilocal start is such a loop; since 2026-11-19 its inner loops run
tests 2 and 3 at their end as well (``PARSEC_KPOINT_SECTOR_GUARD_FOCK=auto``; before, they were not tested).  An inner
loop at the final criterion that fails one makes the Fock loop start again from its semilocal start with one task per
k-point (:func:`parsec_python.SCF.hybrid_periodic.run_periodic_hybrid_scf`; its carried mixer and the states of the
sectors cannot be taken back), a looser inner loop is measured only.  What no test can see: a state that breaks an
operation of ``H_k`` itself (the sector orbitals and their densities are invariant under ``H_k`` by construction, so a
hybrid ground state that localizes, a polaron, on one of the sites ``H_k`` exchanges is held in its symmetric form);
the tests see a multiplet left partly filled across the sectors and a channel density that breaks the operations
beyond ``H_k`` (those of the cell group and of the atoms the sectors assume).
"""

from __future__ import annotations

import dataclasses
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from ...Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from ...SCF.kpoint_solver import KPointJob, job_eigval_settings
from ...SCF.orbital_density import weighted_square_sum
from ..Symmetry.axis_reflection import (
    SignedPermutationReduction,
    _RowOrderTable,
    _affine_matrix,
    _atom_orbits,
    _fast_maps_requested,
    _grid_row_mapping_operation,
    _lattice_period,
    _orbit_labels,
    _periodic_row_major,
    _space_group_operations,
    compress_projectors_for_sectors,
    sector_orbits_exact,
)
from ..Symmetry.orbital_groups import multiplication_table
from ..Symmetry.representations import ReflectionRepresentationDecomposition

_SWITCH = "PARSEC_KPOINT_SECTORS"
# the key of the states of the sectors in what export_states hands to the next loop
_STATES_KEY = "kpoint_sectors"
# the key of the plans of the sectors in a caller's cache of the structure (the projector cache of the k-point loops)
_CACHE_KEY = "kpoint_sector_plans"
# states of a sector expanded to the whole grid at once for its extras (the forces)
_FORCE_COLUMNS = 32
_MARGIN_SWITCH = "PARSEC_KPOINT_SECTOR_MARGIN"
_REFIT_SWITCH = "PARSEC_KPOINT_SECTOR_REFIT"
_LEVELS_SWITCH = "PARSEC_KPOINT_SECTOR_LEVELS"
_FOCK_GUARD_SWITCH = "PARSEC_KPOINT_SECTOR_GUARD_FOCK"
_HYBRID_SWITCH = "PARSEC_HYBRID_KPOINT_SECTORS"
_CLASSES_SWITCH = "PARSEC_KPOINT_SECTOR_CLASSES"
# the key of a structure's cache (the projector cache the SCF loops of one Fock loop share) under which every k-point
# with sectors is split, whatever the size of its basis and the work the sectors save (PARSEC_HYBRID_KPOINT_SECTORS=on)
ANY_SIZE_KEY = "kpoint_sectors_any_size"
# the buffer of a refitted sector over the levels it holds among those kept (at least _EXTRA_STATES states)
_REFIT_BUFFER = 0.1
_EXTRA_STATES = 6
_GROWTH = 1.5
_ATTEMPTS = 4
_WORK_FRACTION = 0.5
# the smallest basis of a k-point (rows times states) that is split
_MINIMUM_ENTRIES = 1 << 24
# guards the cache the k-points of one structure share (KPointSectors._shared)
_SHARED_LOCK = threading.Lock()


def kpoint_sectors_setting() -> str:
    """``PARSEC_KPOINT_SECTORS``: ``auto`` (default) splits the k-points of a scalar or collinear-spin periodic SCF
    into the sectors of their little groups where that halves the work; ``off`` keeps one task per k-point, bit for
    bit as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


def sector_margin() -> float:
    """``PARSEC_KPOINT_SECTOR_MARGIN``: the states of a sector over its share of the states of the k-point (1.3)."""

    raw = os.environ.get(_MARGIN_SWITCH, "").strip()
    if not raw:
        return 1.3
    value = float(raw)
    if not value >= 1.0:
        raise ValueError(f"{_MARGIN_SWITCH} must be at least 1, not {raw!r}")
    return value


def reported_level_tolerance(tolerance: float) -> float:
    """The relative residual :meth:`SectorKPointSolver.solve_reported` asks of the reported levels: 1 % of
    ``tolerance`` (``Diag_Tolerance``), but not below ``1e-10`` unless ``tolerance`` itself is.

    The test of every state of a sector left the reported levels far below ``tolerance``, since the margin states
    above them converge last.  A test of the reported levels at ``tolerance`` itself moved them by up to 1.8e-7 Ry from
    those of that test, at 10 % of it by 1.7e-9 Ry and at 1 % by 1.9e-11 Ry (diamond 216 to 1000 atoms, silicon 216)."""

    return max(0.01 * float(tolerance), min(float(tolerance), 1.0e-10))


def sector_guard_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_GUARD``: ``auto`` (default) tests a spin-polarized loop in the sectors of its k-points
    for a state that breaks the symmetry the sectors assume and runs it again with one task per k-point where it
    does (:func:`parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`); ``off`` keeps the sectors without the
    tests, as before."""

    value = os.environ.get("PARSEC_KPOINT_SECTOR_GUARD", "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"PARSEC_KPOINT_SECTOR_GUARD must be auto or off, not {value!r}")
    return value


def sector_guard_fock_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_GUARD_FOCK``: ``auto`` (default) also runs tests 2 and 3 of the guard
    (``PARSEC_KPOINT_SECTOR_GUARD``) at the end of the inner loops of the Fock loop of a spin-polarized hybrid in the
    sectors of its k-points: an inner loop at the final criterion that fails one makes the Fock loop start again from
    its semilocal start with one task per k-point (:func:`parsec_python.SCF.hybrid_periodic.run_periodic_hybrid_scf`),
    a looser one is measured only; ``off`` tests the semilocal start alone, bit for bit as before.  A loop that
    passes gives the same numbers, except where the density asymmetry lies between the tolerance and the noise bound
    at the final criterion (the criterion is then tightened, as for a semilocal loop)."""

    value = os.environ.get(_FOCK_GUARD_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_FOCK_GUARD_SWITCH} must be auto or off, not {value!r}")
    return value


def hybrid_kpoint_sectors_setting() -> str:
    """``PARSEC_HYBRID_KPOINT_SECTORS``: ``off`` (default) splits the k-points of the SCF loops of a hybrid's Fock
    loop as every k-point loop (``PARSEC_KPOINT_SECTORS``: bases of at least 2^24 entries whose sectors halve the
    work); ``on`` splits every k-point whose little group has commuting involutions into their sectors, whatever its
    size (:data:`ANY_SIZE_KEY` in the cache of the loops, :func:`plan_sectors`), so that the orbitals of the Fock loop
    keep the operations of ``H_k``, the translations of a cell that repeats a smaller one among them: a metal's
    exchange can no longer break them (the 4-atom cube of fcc Au, PBE0, 2x2x2, 1000 K: the plain loop breaks the
    face-centring translations and creeps for hundreds of outer iterations; in the sectors both routes of
    ``PARSEC_HYBRID_KPOINT_WEDGE`` converge, to dexx 1e-10 Ry in 26 outer iterations on the pairs and 12 on the
    wedge).  ``on`` with ``PARSEC_KPOINT_SECTORS=off`` is refused: the two ask for opposite things."""

    value = os.environ.get(_HYBRID_SWITCH, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{_HYBRID_SWITCH} must be off or on, not {value!r}")
    if value == "on" and kpoint_sectors_setting() == "off":
        raise ValueError(
            f"{_HYBRID_SWITCH}=on splits the k-points of a Fock loop into their sectors, which {_SWITCH}=off keeps "
            "whole: unset one of them"
        )
    return value


def sector_levels_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_LEVELS``: ``auto`` (default) keeps a k-point loop with refined levels (the last inner
    loop at the final criterion of the Fock loop of a hybrid with the adaptive inner criterion or the outer DIIS,
    ``SCFWarmStart.level_tolerance``) in the sectors of its k-points, started from the Ritz states of the loop before,
    and its level passes solve the sectors; ``off`` solves that loop with one task per k-point from fresh subspaces,
    bit for bit as before."""

    value = os.environ.get(_LEVELS_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_LEVELS_SWITCH} must be auto or off, not {value!r}")
    return value


def sector_refit_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_REFIT``: ``auto`` (default) cuts the states of every sector after the first solve of a
    loop to the levels it holds among those its k-point keeps plus a buffer (:meth:`SectorKPointSolver._refit`);
    ``off`` keeps the first counts (and their growth), bit for bit as before."""

    value = os.environ.get(_REFIT_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_REFIT_SWITCH} must be auto or off, not {value!r}")
    return value


def sector_classes_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_CLASSES``: ``auto`` (default, since 2026-11-22) solves one sector of each class of
    sectors that the operations of the little group beyond ``H_k`` carry into each other (those that normalize
    ``H_k``, and the antiunitary ``P_g K`` of the operations with ``R k = -k``), among the operations the density is
    symmetrized over, and takes the orbitals of the others as images of those solved (:func:`sector_classes`);
    ``off`` solves every sector of a k-point, bit for bit as before."""

    value = os.environ.get(_CLASSES_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"off", "auto"}:
        raise ValueError(f"{_CLASSES_SWITCH} must be off or auto, not {value!r}")
    return value


_LDAU_SWITCH = "PARSEC_KPOINT_SECTOR_LDAU"


def kpoint_sector_ldau_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_LDAU``: the k-point loops with LDA+U in the sectors of their little groups.

    * ``off`` (default): one task per k-point, as before.
    * ``on`` (opt-in): the loops of unpolarized orbitals take the sectors.  The term ``P_k D P_k^H`` of a k-point
      reduces to a sector as ``(U^H P_k) D (U^H P_k)^H`` (:meth:`KPointSectors.hamiltonian`; the blocks between two
      sectors are dropped, the average of the term over the operations of the sectors, as the local potential takes
      its orbit means), and the occupation matrices of a sector are formed from its states with the same reduced
      projectors, ``<phi|U v> = (U^H phi)^H v`` (:meth:`SectorKPointSolver._accumulate`).
    * ``spin`` (opt-in): the loops of collinear spin as well, with the symmetry guard of the spin loops
      (``PARSEC_KPOINT_SECTOR_GUARD``); a magnetic order that the term holds on some atoms only is averaged over the
      sectors' operations (the Gamma route's ``PARSEC_LDAU_SPIN_SECTORS`` stays off for that reason).

    ``on`` gives the energies and occupation matrices of one task per k-point (A100: Ag 32, 108, Cu 32 and SrTiO3 40
    atoms, 2 x 2 x 2 meshes, within the SCF criterion) in fewer steps and on one GPU in about half the time, but not
    by default: on four GPUs of one process the 32- and 40-atom cells took 1.3-1.4 times as long as one task per
    k-point.  The sector loop of such cells does not scale with the GPUs of a process (SrTiO3 40 in sectors without
    U 43.2 s on one GPU and 43.7 s on four, with U 105.9 and 115.0 s; the summed solve time of the workers grows about
    four times), while one task per k-point divides its four k-points over them; the complex arithmetic that the term
    asks of the real sectors costs about 8 %.  Any other value is refused before the first solve."""

    value = os.environ.get(_LDAU_SWITCH, "off").strip().lower() or "off"
    if value not in {"off", "on", "spin"}:
        raise ValueError(f"{_LDAU_SWITCH} must be off, on or spin, not {value!r}")
    return value


_WARM_GROWTH_SWITCH = "PARSEC_KPOINT_SECTOR_WARM_GROWTH"


def kpoint_sector_warm_growth_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_WARM_GROWTH``: the start of a k-point sector that grows during the SCF
    (:meth:`SectorKPointSolver._grow`, ``resize_tasks`` of the device solvers of one process and of the MPI pools).

    * ``off`` (default): every growth from random vectors, as before.
    * ``chebff`` (opt-in): a CHEBFF first solve starts from the Ritz vectors the sector had, where a SUBSPACE step has
      refined them, and as many columns of the usual random start as it gains (the rule of the Gamma sectors,
      ``PARSEC_CUPY_SECTOR_WARM_GROWTH``).  The vectors wait on the host between the growth and the solve.
    * ``all`` (opt-in): also the first solve of ``Eigensolver: chebdav`` on the device (CHEBFF cycles to CHEBDAV's
      residual test, ``PARSEC_KPOINT_FIRST_SOLVE=refine``).

    Not the default: the first solve of a grown sector makes all its cycles from either start, and with a sector of
    occupied states forced to grow the SCF took as many steps or one more (A100: Al 108, Cu 32, NaCl 64 and Si 216
    atoms, 2 x 2 x 2 meshes); no sector grew in their SCF without forcing.  Any other value is refused before the first
    solve."""

    value = os.environ.get(_WARM_GROWTH_SWITCH, "off").strip().lower() or "off"
    if value not in {"off", "chebff", "all"}:
        raise ValueError(f"{_WARM_GROWTH_SWITCH} must be off, chebff or all, not {value!r}")
    return value


def warm_growth_takes(settings) -> bool:
    """Whether the first solve of ``settings`` (an ``EigvalSettings``) of a grown sector starts from its saved vectors
    (:func:`kpoint_sector_warm_growth_setting`): CHEBFF under ``chebff`` and ``all``, the device CHEBDAV under ``all``."""

    value = kpoint_sector_warm_growth_setting()
    method = str(getattr(settings, "initial_method", ""))
    if value == "off":
        return False
    return method == "chebff" or (value == "all" and method == "chebdav")


def refined_state(state) -> bool:
    """Whether a device Ritz state (one device or the columns of a pool) has had a SUBSPACE step after its first
    solve, so that its vectors are worth keeping when its sector grows (the vectors of a first solve alone are not:
    the rule of the Gamma sectors)."""

    return (
        state is not None
        and getattr(state, "vectors", None) is not None
        and int(getattr(state, "filters_completed", 0) or 0) >= 1
        and getattr(state, "cursor", None) is None
    )


def kpoint_force_orbits_setting() -> str:
    """``PARSEC_KPOINT_FORCE_ORBITS``: ``auto`` (default) forms the nonlocal forces of the states of a k-point in
    sectors for one atom per orbit of its little group (their projector gradients and overlaps only) and turns them
    onto the other atoms of the orbit (:meth:`KPointSectors.complete_orbit_forces`); ``off`` forms them for every atom,
    bit for bit as before."""

    value = os.environ.get("PARSEC_KPOINT_FORCE_ORBITS", "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"PARSEC_KPOINT_FORCE_ORBITS must be auto or off, not {value!r}")
    return value


def refitted_count(held: int, size: int, count: int) -> int:
    """The states of a sector that holds ``held`` of the levels its k-point keeps: ``held`` plus
    :data:`_REFIT_BUFFER` of them, at least :data:`_EXTRA_STATES` more, at most its rows and its ``count`` so far."""

    held = int(held)
    return min(int(size), int(count), held + max(_EXTRA_STATES, math.ceil(_REFIT_BUFFER * held)))


def _leading_columns(vectors, working: int):
    """A copy of the first ``working`` columns of ``vectors`` (host or device) in its memory order."""

    order = "F" if vectors.flags.f_contiguous and not vectors.flags.c_contiguous else "C"
    return vectors[:, : int(working)].copy(order=order)


def trimmed_state(state, requested: int, working: int, *, group=None, block_size: int | None = None):
    """The Ritz state ``state`` of an eigensolver cut to its lowest ``working`` Ritz pairs, for a solve of
    ``requested`` states that keeps it (``None`` where it cannot be cut: a state of ARPACK or DIAGLA).

    The Ritz vectors of a state are in the order of their values, so the first ``working`` columns are the lowest
    pairs.  The three kinds of states of the k-point solvers: that of the host eigensolver
    (:class:`parsec_python.Eigensolvers.eigval.EigvalState`), that of one device
    (:class:`..Eigensolvers.chebff_spinor.SpinorEigvalState`) and that of a basis the ranks of an MPI pool share by
    columns (:class:`..Eigensolvers.chebff_columns.ColumnEigvalState`, ``group`` its ranks, collective: the columns
    are laid out again over them, ``block_size`` that of the solves).  The lower bound of the next filter is the
    largest value kept (the filter of a larger subspace began above it).  A state with no more than ``working``
    columns comes back as it is.
    """

    from dataclasses import replace as _replace

    if state is None or int(state.working_states) <= int(working):
        return state
    working, requested = int(working), int(requested)
    subspace = getattr(state, "subspace", None)
    if subspace is not None:  # the host eigensolver
        if getattr(state, "window_vectors", None) is not None or getattr(state, "start_seed", None) is not None:
            return None
        values = np.asarray(subspace.eigenvalues, dtype=np.float64)[:working].copy()
        cut = _replace(
            subspace,
            working_states=working,
            eigenvalues=values,
            vectors=_leading_columns(np.asarray(subspace.vectors), working),
            filter_lower_bound=None if subspace.filter_lower_bound is None else float(np.max(values)),
        )
        return _replace(state, requested_states=requested, working_states=working, subspace=cut)
    values = np.asarray(_host(state.eigenvalues), dtype=np.float64)[:working].copy()
    bound = None if state.filter_lower_bound is None else float(np.max(values))
    layout = getattr(state, "layout", None)
    if layout is None:  # one device
        return _replace(
            state, vectors=_leading_columns(state.vectors, working), eigenvalues=values, working_states=working,
            filter_lower_bound=bound,
        )
    from ..Eigensolvers.chebff_columns import ColumnLayout

    fresh = ColumnLayout(working, layout.block_size if block_size is None else int(block_size), layout.size)
    if layout.size == 1:
        vectors = _leading_columns(state.vectors, working)
    else:
        dimension = int(state.operator_dimension)
        rows = group.to_rows(state.vectors, layout, dimension)
        vectors = group.to_columns(rows[:, :working], fresh, dimension)
        del rows
    return _replace(
        state, vectors=vectors, eigenvalues=values, working_states=working, filter_lower_bound=bound, layout=fresh
    )


def carried_state(state, requested: int, settings, dimension: int, *, group=None):
    """The Ritz state a k-point solver hands its next solve of ``requested`` states: ``state`` cut to the working
    states of that solve (:func:`trimmed_state`) where it holds more (a task whose count the refit of the sectors
    cut, ``trim_tasks``) and ``PARSEC_KPOINT_SECTOR_REFIT`` is ``auto``; else ``state`` as it is (a state the solve
    cannot take starts afresh, as before)."""

    if state is None or getattr(state, "operator_dimension", None) != int(dimension):
        return state
    working = min(int(requested) + int(settings.safety_buffer), int(dimension))
    if int(state.working_states) <= working or sector_refit_setting() == "off":
        return state
    cut = trimmed_state(state, requested, working, group=group, block_size=int(settings.subspace.block_size))
    return state if cut is None else cut


def _host(value) -> np.ndarray:
    get = getattr(value, "get", None)
    return np.asarray(get() if get is not None else value)


@dataclass(frozen=True)
class LittleGroupReduction(SignedPermutationReduction):
    """The orbits of the commuting involutions ``H_k`` of a little group.

    ``operations`` and ``translations`` (grid steps) are the operations ``r -> R r + t`` of ``H_k``; operation
    ``p`` is the product of the generators ``j`` with bit ``j`` of ``p`` set (``generator_bits``).  The atom orbits
    under them give the projectors of one atom per orbit (:func:`..Symmetry.axis_reflection._atom_orbits`).
    """

    translations: np.ndarray = None
    atom_representatives: np.ndarray = None
    atom_orbit_sizes: np.ndarray = None


def _cell_group_key(grid, atoms, atom_tolerance: float, lattice_tolerance: float) -> bytes:
    """What the operations of :class:`CellGroup` depend on, hashed: the atoms, the cell and its grid, the tolerances
    and the switches its search reads."""

    import hashlib

    digest = hashlib.blake2b(digest_size=32)
    digest.update(repr(tuple(atom.symbol for atom in atoms)).encode())
    digest.update(np.ascontiguousarray([atom.position for atom in atoms], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.points_per_axis, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(grid.cell.lattice_vectors, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.settings.shift, dtype=np.float64).tobytes())
    digest.update(
        repr(
            (
                type(grid).__name__,
                tuple(getattr(grid, "boundary_modes", ())),
                bool(getattr(grid, "is_orthorhombic", False)),
                int(grid.size),
                float(atom_tolerance),
                float(lattice_tolerance),
                tuple(
                    os.environ.get(name, "")
                    for name in (
                        "PARSEC_PERIODIC_TRANSLATIONS",
                        "PARSEC_PERIODIC_SYMMETRY_CENTRE",
                        "PARSEC_PERIODIC_UNEQUAL_STEPS",
                    )
                ),
            )
        ).encode()
    )
    return digest.digest()


# The operations of the last cell group built: (key, matrices, shifts) (recent_cell_group_operations).
_RECENT_CELL_GROUP: tuple | None = None
_RECENT_CELL_GROUP_LOCK = threading.Lock()


def recent_cell_group_operations(grid, atoms, *, atom_tolerance: float = 1.0e-7, lattice_tolerance: float = 1.0e-10):
    """``(matrices, shifts)`` of the :class:`CellGroup` of these very inputs where it is the last one built (the
    detection of the sectors of a cell builds it: the symmetry guard of the spin sectors reads it here instead of
    searching again, :mod:`.spin_sectors`), else ``None``.  The identity first, as :class:`CellGroup` holds them."""

    try:
        key = _cell_group_key(grid, atoms, atom_tolerance, lattice_tolerance)
    except (AttributeError, TypeError, ValueError):
        return None
    with _RECENT_CELL_GROUP_LOCK:
        recent = _RECENT_CELL_GROUP
    if recent is None or recent[0] != key:
        return None
    return list(recent[1]), list(recent[2])


class CellGroup:
    """The operations ``r -> R r + t`` of a bulk orthorhombic cell (the identity first) and their group table.

    Those of the sectors of the Gamma point (:func:`..Symmetry.axis_reflection._space_group_operations`): with the
    operations whose translation is a fraction of the cell where they close into a group, else the half-cell
    translations alone.  ``table`` is ``None`` where the operations form no group.  The operations of the last group
    built are kept for :func:`recent_cell_group_operations` (a few hundred small arrays, no reference to the grid).
    """

    def __init__(self, grid, atoms, *, atom_tolerance: float = 1.0e-7, lattice_tolerance: float = 1.0e-10) -> None:
        global _RECENT_CELL_GROUP

        self.grid = grid
        self.atoms = tuple(atoms)
        self.atom_tolerance = float(atom_tolerance)
        self.lattice_tolerance = float(lattice_tolerance)
        self.counts = np.asarray(grid.points_per_axis, dtype=np.int64)
        self.matrices: list[np.ndarray] = []
        self.shifts: list[np.ndarray] = []
        self.table: np.ndarray | None = None
        for fractional in (True, False):
            matrices, shifts = _space_group_operations(
                grid, self.atoms, atom_tolerance, lattice_tolerance, fractional=fractional
            )
            shifts = [np.mod(np.asarray(shift, dtype=np.int64), self.counts) for shift in shifts]
            try:
                table = multiplication_table(
                    [_affine_matrix(matrix, shift) for matrix, shift in zip(matrices, shifts)], modulus=self.counts
                )
            except ValueError:
                continue
            self.matrices = [np.asarray(matrix, dtype=np.int8) for matrix in matrices]
            self.shifts = shifts
            self.table = table
            break
        if self.table is not None:
            try:
                key = _cell_group_key(grid, self.atoms, self.atom_tolerance, self.lattice_tolerance)
            except (AttributeError, TypeError, ValueError):
                key = None
            if key is not None:
                with _RECENT_CELL_GROUP_LOCK:
                    _RECENT_CELL_GROUP = (
                        key,
                        tuple(matrix.copy() for matrix in self.matrices),
                        tuple(shift.copy() for shift in self.shifts),
                    )

    @property
    def order(self) -> int:
        return len(self.matrices)

    def little_group(self, k_point, admissible: Callable[[int], bool]) -> list[int]:
        """The operations with ``R k = k`` (to round-off of ``k``) that ``admissible`` accepts, the identity first."""

        k = np.asarray(k_point, dtype=np.float64).reshape(3)
        tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(k))))
        members = []
        for index, matrix in enumerate(self.matrices):
            if float(np.max(np.abs(matrix.astype(np.float64) @ k - k))) > tolerance:
                continue
            if index and not admissible(index):
                continue
            members.append(index)
        return members

    def row_map(self, index: int, rows: np.ndarray | None = None, table=None) -> np.ndarray | None:
        """The grid rows of ``R r + t`` for the rows ``rows`` (all where ``None``), ``None`` where they miss."""

        return _grid_row_mapping_operation(
            self.grid,
            self.matrices[index],
            self.lattice_tolerance,
            rows,
            table=table,
            translation=self.shifts[index],
        )


def generated_subgroup(table: np.ndarray, members) -> list[int]:
    """The subgroup that ``members`` generate (ascending, the identity 0 first): the potential is invariant under
    the products of the operations it is invariant under (the space group of the density and the translations of
    the cell, say), and the products of operations that fix ``k`` fix it."""

    found = {0} | {int(index) for index in members}
    frontier = sorted(found)
    while frontier:
        fresh = set()
        for left in frontier:
            for right in sorted(found):
                for product_ in (int(table[left, right]), int(table[right, left])):
                    if product_ not in found:
                        fresh.add(product_)
        found |= fresh
        frontier = sorted(fresh)
    return sorted(found)


def largest_commuting_involutions(table: np.ndarray, members) -> tuple[list[int], list[int]]:
    """``(elements, generators)`` of the largest subgroup of ``members`` whose elements commute and square to the
    identity (index 0); element ``p`` is the product of the generators ``j`` with bit ``j`` of ``p`` set.

    The group a set of commuting involutions generates consists of commuting involutions, so the search adds one
    generator at a time among the involutions that commute with the group so far and lie outside it; a group of
    order ``2^r`` that grows past the best one so far needs ``2^(r+1)`` elements, all among those candidates, which
    bounds the search.  Products outside ``members`` (the operations the potential is invariant under) refuse a
    generator.  The first largest group found is kept (deterministic).
    """

    allowed = {int(index) for index in members}
    involutions = [int(g) for g in members if int(g) != 0 and int(table[g, g]) == 0]
    best: list[list[int]] = [[0], []]

    def search(group: list[int], generators: list[int], candidates: list[int]) -> None:
        if len(group) > len(best[0]):
            best[0], best[1] = group, generators
        for position, candidate in enumerate(candidates):
            if len(group) + len(candidates) - position < 2 * len(best[0]):
                return
            images = [int(table[candidate, element]) for element in group]
            if any(image not in allowed for image in images):
                continue
            grown = group + images
            inside = set(grown)
            rest = [
                other
                for other in candidates[position + 1 :]
                if int(table[candidate, other]) == int(table[other, candidate]) and other not in inside
            ]
            search(grown, generators + [candidate], rest)

    search([0], [], involutions)
    return best[0], best[1]


def little_group_reduction(cell: CellGroup, elements: list[int], generators: list[int]) -> LittleGroupReduction | None:
    """The orbits of the grid rows under ``elements`` (``None`` where one of them misses the grid)."""

    row_table = _RowOrderTable(cell.grid) if _fast_maps_requested() else None
    # The smallest image of every row under the group, map by map, as detect_point_group_reduction keeps it: the
    # maps are not held together (32 elements of a 20^3 diamond cell, 46.7 M points, are 12 GB); _orbit_labels of
    # that one array is _orbit_labels of all the maps.
    smallest = None
    for index in elements:
        mapping = cell.row_map(index, table=row_table)
        if mapping is None:
            return None
        if smallest is None:
            smallest = np.array(mapping, dtype=np.int64, copy=True)
        else:
            np.minimum(smallest, mapping, out=smallest)
        del mapping
    representative_rows, full_to_wedge, multiplicities = _orbit_labels([smallest])
    del smallest
    order, rank = len(elements), len(generators)
    if order != 1 << rank:
        raise RuntimeError("the commuting involutions do not form a group of order 2^rank")
    bits = np.array([[(p >> j) & 1 for j in range(rank)] for p in range(order)], dtype=np.int8).reshape(order, rank)
    operations = np.ascontiguousarray([cell.matrices[index] for index in elements], dtype=np.int8)
    translations = np.ascontiguousarray([cell.shifts[index] for index in elements], dtype=np.int64)
    orbits = _atom_orbits(cell.atoms, list(operations), list(translations), cell.grid, cell.atom_tolerance)
    return LittleGroupReduction(
        signs=np.ones((order, 3), dtype=np.int8),
        representative_rows=np.ascontiguousarray(representative_rows, dtype=np.int64),
        full_to_wedge=np.ascontiguousarray(full_to_wedge, dtype=np.int64),
        multiplicities=np.ascontiguousarray(multiplicities, dtype=np.int64),
        operations=operations,
        generator_bits=np.ascontiguousarray(bits),
        translations=translations,
        atom_representatives=None if orbits is None else orbits[0],
        atom_orbit_sizes=None if orbits is None else orbits[1],
    )


@dataclass(frozen=True)
class ClassMember:
    """A sector of a k-point that is not solved (``PARSEC_KPOINT_SECTOR_CLASSES``, :func:`sector_classes`): its
    orbitals are ``P_g u`` of the orbitals ``u`` of the sector ``source`` (``P_g conj(u)`` where ``antiunitary``),
    ``g`` the operation ``operation`` of the cell group, with the levels of ``source``.  In the coefficients of the
    sectors, row ``j`` of ``source`` goes to row ``rows[j]`` of this sector with the sign ``signs[j]``."""

    source: int
    operation: int
    antiunitary: bool
    rows: np.ndarray
    signs: np.ndarray


def _inverse_elements(table: np.ndarray) -> np.ndarray:
    """``inverse[g]``: the index of the inverse of operation ``g`` of a group table (the identity is 0)."""

    rows, columns = np.nonzero(np.asarray(table) == 0)
    inverse = np.empty(table.shape[0], dtype=np.int64)
    inverse[rows] = columns
    return inverse


def exact_operation_test(cell) -> Callable[[int], bool]:
    """Whether an operation of ``cell`` carries every atom within
    :data:`..Symmetry.axis_reflection.EXACT_ORBIT_TOLERANCE` (1e-6 bohr) onto an atom of its species, the rule of the
    projectors of one atom per orbit (:func:`..Symmetry.axis_reflection.sector_orbits_exact`) for the operations whose
    images the classes and the real forms of the sectors take: a group built at a looser
    ``PARSEC_SYMMETRY_ATOM_TOLERANCE`` would copy the offset of the atoms into the images.  Every operation passes under
    ``PARSEC_SECTOR_ORBITS_EXACT=off`` or at a tolerance of at most 1e-6 bohr (none is then checked)."""

    from ..Symmetry.axis_reflection import (
        EXACT_ORBIT_TOLERANCE,
        atom_image_table,
        sector_orbits_exact_setting,
        symmetry_atom_tolerance,
    )

    if (
        sector_orbits_exact_setting() == "off"
        or float(symmetry_atom_tolerance()) <= EXACT_ORBIT_TOLERANCE
        or not cell.atoms
    ):
        return lambda index: True
    found: dict[int, bool] = {}

    def holds(index: int) -> bool:
        index = int(index)
        if index not in found:
            found[index] = atom_image_table(
                tuple(cell.atoms), [cell.matrices[index]], [cell.shifts[index]], cell.grid, EXACT_ORBIT_TOLERANCE,
                within_tolerance=True,
            ) is not None
        return found[index]

    return holds


def sector_classes(cell: CellGroup, members, elements, decomposition, k_point, invariance) -> dict[int, ClassMember] | None:
    """The sectors of a k-point that need not be solved: ``{sector: ClassMember}``, ``None`` where every sector is.

    An operation ``g`` of the little group (``R k = k``) that normalizes ``H_k`` (``elements``, the commuting
    involutions of the sectors) carries an orbital ``u`` of the sector of character ``chi`` (``P_h u = chi(h) u``) into
    one of the sector of ``chi^g(h) = chi(g^-1 h g)``: ``P_h P_g u = P_g P_{g^-1 h g} u``.  ``P_g`` commutes with
    ``H_k`` where the potential is invariant under ``g`` (module text), so the two sectors have one spectrum, and the
    orbitals of the second are ``P_g u``.  The antiunitary ``P_g K`` (``K`` complex conjugation) of an operation with
    ``R k = -k`` commutes with ``H_k`` as well (``K H_k K = H_{-k}`` of the real potential and the real stencils and
    projectors of the u gauge) and carries ``chi`` into ``chi^g`` in the same way (the characters are real).  The
    operations taken are those the density is symmetrized over (:meth:`PotentialInvariance.in_density_group`): the
    potential of every iteration is invariant under them, as is its average over the orbits of ``H_k``, so the
    sector Hamiltonians are images of each other at every step, not only at self-consistency.  ``members`` are the
    operations of the little group the sectors admit (indices of ``cell``).

    The sectors fall into the orbits of their characters under these operations (with ``H_k`` they form a group); the
    sector of the lowest index of each orbit is solved, the others are its images.  In the coefficients of the sectors
    ``P_g U_s e_w = chi_t(h_0) U_t e_w'`` for the orbit ``w'`` of ``g r_w`` (``r_w`` the representative of orbit ``w``,
    ``g r_w = h_0 r_w'``), with ``chi_t(h_0)`` the phase of the row ``g r_w`` in sector ``t``: a signed permutation of
    the rows (:class:`ClassMember`).
    """

    table = cell.table
    # the characters as rows of signs, or of exponents modulo ``modulus`` for complex characters
    # (PARSEC_KPOINT_SECTOR_CHARACTERS=complex, ..Symmetry.complex_sectors: P_g K takes chi to conj(chi^g))
    complex_characters = bool(getattr(decomposition, "is_complex", False))
    modulus = int(getattr(decomposition, "modulus", 2))
    if complex_characters:
        characters = np.ascontiguousarray(decomposition.exponents, dtype=np.int64)
    else:
        characters = np.ascontiguousarray(decomposition.characters, dtype=np.int8)
    order = int(characters.shape[0])
    if table is None or order < 2:
        return None
    inverse = _inverse_elements(table)
    position = {int(element): index for index, element in enumerate(elements)}
    by_character = {characters[sector].tobytes(): sector for sector in range(order)}
    k = np.asarray(k_point, dtype=np.float64).reshape(3)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(k))))
    # operations that hold the atoms only to a looser PARSEC_SYMMETRY_ATOM_TOLERANCE than 1e-6 bohr give no images
    keeps_atoms = exact_operation_test(cell)

    def permutation(g: int) -> np.ndarray | None:
        # the element of H_k at the place of g^-1 h g for every h of H_k (None: g does not normalize H_k)
        found = np.empty(len(elements), dtype=np.int64)
        for index, h in enumerate(elements):
            image = position.get(int(table[table[inverse[g], int(h)], g]))
            if image is None:
                return None
            found[index] = image
        return found

    candidates: list[tuple[int, bool, np.ndarray]] = []
    for g in members:
        g = int(g)
        if g in position or not invariance.in_density_group(g) or not keeps_atoms(g):
            continue
        found = permutation(g)
        if found is not None:
            candidates.append((g, False, found))
    if np.any(k != 0.0):  # at the Gamma point the antiunitary operations are the unitary ones with K
        for g in range(cell.order):
            matrix = np.asarray(cell.matrices[g], dtype=np.float64)
            if float(np.max(np.abs(matrix @ k + k))) > tolerance or not invariance.in_density_group(g):
                continue
            if not keeps_atoms(g):
                continue
            found = permutation(g)
            if found is not None:
                candidates.append((g, True, found))
    images: list[tuple[int, bool, np.ndarray]] = []
    for g, antiunitary, found in candidates:
        target = np.empty(order, dtype=np.int64)
        for sector in range(order):
            label = characters[sector, found]
            if complex_characters and antiunitary:
                label = np.mod(-label, modulus)
            image = by_character.get(np.ascontiguousarray(label).tobytes())
            if image is None:
                break
            target[sector] = image
        else:
            images.append((g, antiunitary, target))
    if not images:
        return None
    parent = list(range(order))

    def root(sector: int) -> int:
        while parent[sector] != sector:
            parent[sector] = parent[parent[sector]]
            sector = parent[sector]
        return sector

    for _, _, target in images:
        for sector in range(order):
            a, b = root(sector), root(int(target[sector]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    orbits: dict[int, list[int]] = {}
    for sector in range(order):
        orbits.setdefault(root(sector), []).append(sector)
    sizes = decomposition.sector_sizes
    to_wedge = np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64)
    representatives = np.asarray(decomposition.reduction.representative_rows, dtype=np.int64)
    found_members: dict[int, ClassMember] = {}
    for orbit in orbits.values():
        if len(orbit) < 2:
            continue
        source = min(orbit)
        rows_of_source = representatives[decomposition.sector_orbit_indices(source)]
        for sector in orbit:
            if sector == source or sizes[sector] != sizes[source]:
                continue
            # a unitary operation where one carries the source there, else an antiunitary one
            choices = [entry for entry in images if int(entry[2][source]) == sector]
            choices.sort(key=lambda entry: (entry[1], entry[0]))
            for g, antiunitary, _ in choices:
                image = cell.row_map(g, rows=rows_of_source)
                if image is None:
                    continue
                image = np.asarray(image, dtype=np.int64)
                rows = np.asarray(decomposition.orbit_to_sector[sector][to_wedge[image]], dtype=np.int64)
                if complex_characters:
                    signs = np.conj(np.asarray(decomposition.phases[sector][image], dtype=np.complex128))
                else:
                    signs = np.asarray(decomposition.phases[sector][image], dtype=np.int8)
                if (
                    np.any(rows < 0)
                    or np.any(signs == 0)
                    or np.unique(rows).size != rows.size
                    or rows.size != sizes[sector]
                ):
                    continue
                rows.setflags(write=False)
                signs.setflags(write=False)
                found_members[sector] = ClassMember(source, int(g), bool(antiunitary), rows, signs)
                break
    return found_members or None


def _complex_candidate(cell: CellGroup, elements):
    """The decomposition of the complex characters of the abelian group ``elements`` of ``cell``
    (:func:`..Symmetry.complex_sectors.complex_decomposition`), its atom orbits dropped where the atoms keep the group
    only to more than 1e-6 bohr (the rule of :func:`..Symmetry.axis_reflection.sector_orbits_exact`); ``None`` where
    it cannot be built."""

    from ..Symmetry.axis_reflection import (
        EXACT_ORBIT_TOLERANCE,
        atom_image_table,
        sector_orbits_exact_setting,
        symmetry_atom_tolerance,
    )
    from ..Symmetry.complex_sectors import complex_decomposition

    decomposition = complex_decomposition(cell, list(elements))
    if decomposition is None:
        return None
    reduction = decomposition.reduction
    if (
        reduction.atom_representatives is not None
        and sector_orbits_exact_setting() == "auto"
        and float(symmetry_atom_tolerance()) > EXACT_ORBIT_TOLERANCE
        and atom_image_table(
            tuple(cell.atoms), list(reduction.operations), list(reduction.translations), cell.grid,
            EXACT_ORBIT_TOLERANCE, within_tolerance=True,
        ) is None
    ):
        decomposition.reduction = dataclasses.replace(reduction, atom_representatives=None, atom_orbit_sizes=None)
    return decomposition


def _image_row(symmetry, operation: int) -> np.ndarray:
    """``images[operation]`` of a density group: through ``image_row`` where it has one (a
    :class:`parsec_python.Grid.space_group.PeriodicGridSymmetry`, which may hold no table), else of its table."""

    reader = getattr(symmetry, "image_row", None)
    return np.asarray(reader(operation) if reader is not None else np.asarray(symmetry.images)[operation])


def _image_rows(symmetry, operation: int, rows) -> np.ndarray:
    """``images[operation, rows]`` of a density group, as :func:`_image_row` reads a whole row."""

    reader = getattr(symmetry, "image_rows", None)
    if reader is not None:
        return np.asarray(reader(operation, rows))
    return np.asarray(symmetry.images)[operation][np.asarray(rows)]


class PotentialInvariance:
    """Whether the potential of the loop is invariant under an operation of the cell (see the module text)."""

    _SAMPLES = 61

    def __init__(self, cell: CellGroup, density_symmetry, k_points, weights, lattice) -> None:
        self.cell = cell
        self.symmetry = density_symmetry
        self.k_points = np.asarray(k_points, dtype=np.float64).reshape(-1, 3)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        self.lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
        self.fractional = self.k_points @ self.lattice.T / (2.0 * np.pi)
        self._decided: dict[int, bool] = {}
        self._in_group: dict[int, bool] = {}
        self._row_table = None
        self._sample = None
        self._sampled = None
        images = getattr(density_symmetry, "images", None)
        shape = getattr(images, "shape", None) if images is not None else None
        if shape is None and images is not None:
            shape = np.shape(images)
        if shape is not None and len(shape) == 2 and shape[1] == cell.grid.size:
            size = int(cell.grid.size)
            self._sample = np.unique(np.linspace(0, size - 1, self._SAMPLES).astype(np.int64))
            # the rows of the group at the sample only (Grid.space_group.PeriodicGridSymmetry.image_rows: from the
            # affine maps of a group that holds no table, PARSEC_PERIODIC_IMAGE_ROWS), the same integers
            self._sampled = np.stack(
                [_image_rows(density_symmetry, operation, self._sample) for operation in range(int(shape[0]))]
            )

    def __call__(self, index: int) -> bool:
        decided = self._decided.get(index)
        if decided is None:
            decided = self._decided[index] = self._decide(index)
        return decided

    def _decide(self, index: int) -> bool:
        matrix = self.cell.matrices[index]
        if np.array_equal(matrix, np.eye(3, dtype=np.int8)):
            return True  # a translation of the cell: every rho_k is invariant under it
        return self.in_density_group(index) or self._maps_kpoints(matrix)

    def in_density_group(self, index: int) -> bool:
        """Whether the density is symmetrized over operation ``index`` (``Symmetrize_Charge_Density``, the wedge of a
        reduced mesh): then the loop without sectors makes the potential invariant under it as well."""

        if index == 0:
            return True
        found = self._in_group.get(index)
        if found is None:
            found = self._in_group[index] = self._in_density_group(index)
        return found

    def _in_density_group(self, index: int) -> bool:
        return self.symmetrized_over(self.cell.matrices[index], self.cell.shifts[index])

    def symmetrized_over(self, matrix, shift) -> bool:
        """Whether the density is symmetrized over ``r -> R r + t`` (``t`` in grid steps; any such operation of the
        grid, also one outside the cell group: :meth:`admits`)."""

        if self._sample is None:
            return False
        grid, tolerance = self.cell.grid, self.cell.lattice_tolerance
        sampled = _grid_row_mapping_operation(grid, matrix, tolerance, self._sample, translation=shift)
        if sampled is None:
            return False
        hits = np.flatnonzero(np.all(self._sampled == sampled[None, :], axis=1))
        if hits.size == 0:
            return False
        if self._row_table is None and _fast_maps_requested():
            self._row_table = _RowOrderTable(grid)
        whole = _grid_row_mapping_operation(grid, matrix, tolerance, table=self._row_table, translation=shift)
        if whole is None:
            return False
        # the maps of a group are the same set in either direction (it holds the inverses); one row at a time
        return any(np.array_equal(_image_row(self.symmetry, int(hit)), whole) for hit in hits)

    def admits(self, matrix, shift) -> bool:
        """:meth:`__call__` for an operation ``r -> R r + t`` of the atoms given by itself (one the cell group may
        not hold: those found from the environments of the atoms, :class:`SectorGuardGroup`)."""

        matrix = np.asarray(matrix)
        if np.array_equal(matrix, np.eye(3, dtype=matrix.dtype)):
            return True
        return self.symmetrized_over(matrix, shift) or self._maps_kpoints(matrix)

    def _maps_kpoints(self, matrix: np.ndarray) -> bool:
        rotated = (self.k_points @ matrix.astype(np.float64).T) @ self.lattice.T / (2.0 * np.pi)
        scale = max(float(np.max(np.abs(self.weights), initial=0.0)), 1.0e-300)
        for row, weight in zip(rotated, self.weights):
            same = np.abs(self.weights - weight) <= 1.0e-10 * scale
            direct = row[None, :] - self.fractional
            reversed_ = row[None, :] + self.fractional
            match = (
                np.all(np.abs(direct - np.rint(direct)) <= 1.0e-8, axis=1)
                | np.all(np.abs(reversed_ - np.rint(reversed_)) <= 1.0e-8, axis=1)
            ) & same
            if not np.any(match):
                return False
        return True


class SectorGuardGroup:
    """The operations of a cell against which the symmetry guard measures the densities of a loop in the sectors
    (test 2 of :meth:`SectorKPointSolver.density_symmetry_report`).

    Those the potential of the loop is taken to be invariant under (:class:`PotentialInvariance`), beyond those the
    density is symmetrized over (whose invariance holds by construction): a generating set of the subgroup they
    generate together, chosen in the order of the operations (the closure grows by right products, through the
    group table).  ``rows`` of an operation are the grid rows of its images ``R r + t``.

    With ``environments`` (``PARSEC_SPIN_GUARD_CELL_GROUP=auto``, the default) the operations of the atoms found from
    their environments follow (:func:`.spin_sectors._environment_operations`, at the atom tolerance of the cell group
    and at most 1e-6 bohr), those the potential is taken to be invariant under and the subgroup so far does not hold
    (:meth:`PotentialInvariance.admits`; at a single Gamma point all of them).  Under
    ``PARSEC_PERIODIC_DEFECT_CENTRE=off`` the cell group misses most of the operations about a defect off the origin
    of a cell larger than 2 x 2 x 2 diamond cubes (its search of the translation of each operation gives up after 32
    candidates; a vacancy at ``(a/2, a/2, 0)`` of 3 x 3 x 3 cubes keeps the swap of x and y), and so do its sectors: a
    broken state that fills two partners of the t2 level of one sector of that swap would pass a test of the swap
    alone, and the test would fire on the same cell with the vacancy at the origin (under ``auto``, the default since
    2026-11-19, the cell group holds them, and these add none).  These operations are given as their affine
    ``[[R, t], [0, 1]]`` (``t`` in grid steps) in place of an index of the cell group.  With a cell group of a looser
    atom tolerance than 1e-6 bohr (``PARSEC_SYMMETRY_ATOM_TOLERANCE``) its operations beyond the subgroup of those the
    density is symmetrized over are tested only where they hold the atoms to 1e-6 bohr (:meth:`_strict_test`), as on
    the Gamma sectors: one that holds them only to the looser tolerance would measure the structure.
    """

    def __init__(self, cell: CellGroup, invariance: PotentialInvariance, *, environments: bool = False) -> None:
        self.cell = cell
        self.invariance = invariance
        self.environments = bool(environments)
        self._images: list[tuple[int | np.ndarray, np.ndarray]] | None = None

    def _close(self, closure: set[int], generators: list[int]) -> None:
        table = self.cell.table
        frontier = list(closure)
        while frontier:
            fresh = []
            for element in frontier:
                for generator in generators:
                    product_ = int(table[element, generator])
                    if product_ not in closure:
                        closure.add(product_)
                        fresh.append(product_)
            frontier = fresh

    def images(self) -> list[tuple[int | np.ndarray, np.ndarray]]:
        """``(operation, rows)`` of the operations to test (built once): ``operation`` the index of one of the cell
        group, or the affine matrix of one found from the environments."""

        if self._images is not None:
            return self._images
        admitted = [index for index in range(1, self.cell.order) if self.invariance(index)]
        free = [index for index in admitted if self.invariance.in_density_group(index)]
        closure = {0}
        generators: list[int] = []
        for index in free:
            if index not in closure:
                generators.append(index)
                self._close(closure, generators)
        tested: list[tuple[int | np.ndarray, np.ndarray]] = []
        table = _RowOrderTable(self.cell.grid) if _fast_maps_requested() else None
        holds = self._strict_test()
        for index in admitted:
            if index in closure:
                continue
            if holds is not None and not holds(index):
                # an operation the atoms keep only to the looser tolerance of the cell group: the densities break
                # it by about as much as the atoms are off (as the operations beyond the sectors of the Gamma guard)
                continue
            rows = self.cell.row_map(index, table=table)
            if rows is None:
                continue
            generators.append(index)
            self._close(closure, generators)
            tested.append((index, np.asarray(rows, dtype=np.int64)))
        if self.environments:
            tested += self._environment_images(generators, table)
        self._images = tested
        return tested

    def _strict_test(self) -> Callable[[int], bool] | None:
        """Whether an operation of the cell group carries every atom within 1e-6 bohr
        (:data:`parsec_python.Forces.symmetry.POSITION_TOLERANCE`) onto an atom of its species, as a permutation;
        ``None`` (every one does) for a cell group of that tolerance or a stricter one
        (``PARSEC_SYMMETRY_ATOM_TOLERANCE`` at most 1e-6, the default 1e-7 among them)."""

        from parsec_python.Forces.symmetry import POSITION_TOLERANCE

        from ..Symmetry.axis_reflection import _atoms_are_invariant_operation, _positions_by_species

        cell = self.cell
        if cell.atom_tolerance <= POSITION_TOLERANCE:
            return None
        species = _positions_by_species(cell.atoms)
        period = _lattice_period(cell.grid)
        steps = period / cell.counts

        def holds(index: int) -> bool:
            return _atoms_are_invariant_operation(
                species, cell.matrices[index], POSITION_TOLERANCE, period, cell.shifts[index] * steps
            )

        return holds

    def _environment_images(self, generators: list[int], table) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(affine, rows)`` of the operations found from the environments of the atoms that the subgroup of
        ``generators`` (of the cell group) does not hold and the potential is taken to be invariant under: those the
        density is symmetrized over join that group untested, the others are tested (see the class)."""

        from parsec_python.Forces.symmetry import POSITION_TOLERANCE

        from .spin_sectors import _GUARD_GROUP_LIMIT, _AffineClosure, _affine, _environment_operations, _is_diagonal

        cell = self.cell
        try:
            found = _environment_operations(cell.grid, cell.atoms, min(cell.atom_tolerance, POSITION_TOLERANCE))
        except ValueError:
            found = None
        if not found:
            return []
        group = _AffineClosure(cell.counts)
        for index in generators:
            group.add(_affine(cell.matrices[index], cell.shifts[index]))

        def kind(affine) -> int:
            # the translations first, then the sign changes, then the others
            matrix = affine[:3, :3]
            return 0 if np.array_equal(matrix, np.eye(3)) else 1 if _is_diagonal(matrix) else 2

        admitted, free = [], []
        for affine in sorted(found, key=kind):
            affine[:3, 3] = np.mod(affine[:3, 3], cell.counts)
            if affine in group:
                continue
            if self.invariance.symmetrized_over(affine[:3, :3], affine[:3, 3]):
                free.append(affine)
            elif self.invariance.admits(affine[:3, :3], affine[:3, 3]):
                admitted.append(affine)
        for affine in free:
            if affine not in group:
                group.add(affine, limit=_GUARD_GROUP_LIMIT)
        tested: list[tuple[np.ndarray, np.ndarray]] = []
        for affine in admitted:
            if affine in group:
                continue
            rows = _grid_row_mapping_operation(
                cell.grid, affine[:3, :3], cell.lattice_tolerance, table=table, translation=affine[:3, 3]
            )
            if rows is None or not group.add(affine, limit=_GUARD_GROUP_LIMIT):
                continue
            tested.append((affine, np.asarray(rows, dtype=np.int64)))
        return tested

    def operation_text(self, operation) -> str:
        if isinstance(operation, (int, np.integer)):
            matrix = np.asarray(self.cell.matrices[operation], dtype=int).tolist()
            shift = np.asarray(self.cell.shifts[operation], dtype=int).tolist()
        else:
            operation = np.asarray(operation, dtype=int)
            matrix, shift = operation[:3, :3].tolist(), operation[:3, 3].tolist()
        return f"{matrix} + {shift} grid steps"


class SectorBlochHamiltonian(KPointKohnShamHamiltonian):
    """The Bloch Hamiltonian of one sector of a k-point: its kinetic term in the slot of the Bloch Laplacian (the
    gathered stencil of :mod:`..Hamiltonian.sector_bloch_stencil`, ``PARSEC_KPOINT_SECTOR_STENCIL=auto``, or the
    reduced complex CSR) and its reduced projectors; the complex matrices take the complex CSR kernel of the device
    (:mod:`..Hamiltonian.complex_csr_device`, ``device_csr``)."""

    device_csr = True


def _materialized(laplacian):
    """``laplacian`` as a matrix: the deferred stencil of a Gamma cell (the band k-points after its SCF in sectors,
    ``SCF/band_sectors.py``) materialized, any other operator as it is."""

    if not sp.issparse(laplacian) and callable(getattr(laplacian, "materialize", None)):
        return laplacian.materialize()
    return laplacian


class KPointSectors:
    """The sectors of one k-point: their bases, states and static operators (built at the first Hamiltonian)."""

    def __init__(self, k_index: int, k_point, decomposition: ReflectionRepresentationDecomposition, counts) -> None:
        self.k_index = int(k_index)
        self.k_point = np.asarray(k_point, dtype=np.float64).reshape(3)
        self.decomposition = decomposition
        self.sizes = tuple(int(size) for size in decomposition.sector_sizes)
        self.counts = [int(count) for count in counts]
        self._multiplicities = np.asarray(decomposition.reduction.multiplicities, dtype=np.float64)
        self._lock = threading.Lock()
        # by sector: (kinetic, projectors, empty stand-in of the real stencils), built for the sectors asked for
        self._static: dict[int, tuple] = {}
        self._potential: tuple[Any, np.ndarray | None] = (None, None)
        self._exchange: dict[int, tuple] = {}  # by sector: (exchange term of the whole grid, its reduction)
        # by sector: (LDA+U projectors of the whole grid, U^H P), shared by the channels of the k-point (twin)
        self._hubbard: dict[int, tuple] = {}
        self._gathers: dict[int, sp.csr_matrix] = {}  # by sector: U^H (shared by the tasks of the k-point)
        self._expanders: dict[tuple, tuple] = {}  # by (sector, device): the rows and factors of U on a device
        # whether the sectors assume an invariance the density is not symmetrized over, and the operations of the
        # symmetry guard (set by plan_sectors)
        self.assumed = False
        self.guard_group: SectorGuardGroup | None = None
        # the atom orbits of the nonlocal forces (force_orbits): None not yet made, False none
        self._orbit_forces = None
        # what the k-points of one structure share (plan_sectors): the check of the kinetic stencils of the grid
        self._shared: dict = {}
        # whether the operators of the sectors go to the device code (sector_solver sets it from the k-point
        # backend): only there does the kinetic term take its gathered stencil (_stencil_parts)
        self.device_operators = True
        # the partner of every sector at a time-reversal invariant k-point in the Bloch form, None elsewhere
        # (PARSEC_KPOINT_TRIM_REAL, .kpoint_real.trim_partners; set by plan_sectors): the sector of the lower index of
        # each pair is solved, its partner's orbitals are ``e^{-iG.r} conj(u)`` (``G = 2k``)
        self.partners: np.ndarray | None = None
        # the sectors not solved because an operation of the little group carries a sector solved into them
        # (PARSEC_KPOINT_SECTOR_CLASSES, sector_classes; set by plan_sectors), None where every sector is solved
        self.classes: dict[int, ClassMember] | None = None
        self._member_maps: dict[tuple, tuple] = {}  # by (sector, device): the rows and signs of its ClassMember there
        # the sectors solved on real blocks through an antiunitary P_g K (PARSEC_KPOINT_SECTOR_REAL,
        # .kpoint_real_forms; set by plan_sectors), None where every sector is complex; their static real parts
        self.real_forms: dict | None = None
        self._real_static: dict[int, tuple] = {}

    @property
    def order(self) -> int:
        return self.decomposition.representation_count

    def _phase_row(self, sector: int) -> np.ndarray:
        """The phase of every grid row in ``sector``: the signs of the real characters as float64, the complex phases
        of a decomposition of complex characters (:mod:`..Symmetry.complex_sectors`)."""

        phases = self.decomposition.phases[sector]
        return phases.astype(np.float64) if not np.iscomplexobj(phases) else np.asarray(phases)

    def class_of(self, sector: int) -> list[int]:
        """The sectors of the class of ``sector`` (:attr:`classes`): the sector solved first, then its images."""

        if not self.classes:
            return [int(sector)]
        source = self.classes[sector].source if sector in self.classes else int(sector)
        return [source] + sorted(member for member, entry in self.classes.items() if entry.source == source)

    def coefficients(self, sector: int, block, xp=np):
        """The coefficients of ``sector`` of the vectors of its solve ``block``: ``T x`` of the real coordinates of a
        sector in its real form (:attr:`real_forms`), ``block`` itself otherwise."""

        form = None if not self.real_forms else self.real_forms.get(int(sector))
        return block if form is None else form.to_complex(block, xp)

    def _real_parts(self, sector: int, kinetic, projectors):
        """The static parts of the real form of ``sector`` (made once): the paired stencil table or the real kinetic
        CSR, and the real projector parts (``None`` without projectors)."""

        found = self._real_static.get(sector)
        if found is None:
            from ..Hamiltonian.sector_pair_stencil import PairStencilTable
            from .kpoint_real_forms import real_kinetic_matrix, real_projector_parts

            form = self.real_forms[sector]
            table = PairStencilTable(kinetic, form) if getattr(kinetic, "sector_stencil", False) else None
            matrix = None if table is not None else real_kinetic_matrix(kinetic, form)
            parts = None
            if projectors is not None and int(projectors.projectors.shape[1]):
                parts = real_projector_parts(projectors.projectors, projectors.signs, form)
            found = self._real_static[sector] = (table, matrix, parts)
        return found

    def member_block(self, sector: int, block, xp=np):
        """The coefficients in ``sector`` (a :class:`ClassMember`) of the images ``P_g u`` (``P_g conj(u)``) of the
        orbitals whose coefficients in its source are the columns of ``block`` (host or device)."""

        member = self.classes[sector]
        values = member.signs if np.iscomplexobj(member.signs) else member.signs.astype(np.float64)
        if xp is np or isinstance(block, np.ndarray):
            rows, signs = member.rows, values
        else:
            device = int(block.device.id)
            found = self._member_maps.get((sector, device))
            if found is None:
                with xp.cuda.Device(device):
                    found = (xp.asarray(member.rows), xp.asarray(values))
                self._member_maps[(sector, device)] = found
            rows, signs = found
        source = xp.conj(block) if member.antiunitary else block
        out = xp.empty_like(source)
        out[rows] = signs[:, None] * source
        return out

    def class_density(self, source: int, members, sums: np.ndarray) -> np.ndarray:
        """The density of the class of ``source`` from ``q = sum_n f_n |v_n|^2`` on its rows, ``f`` the occupations
        of all the sectors of the class summed onto those of ``source``: the mean over the class of ``|U_t (S v)|^2``
        (``S`` the signed permutation of :class:`ClassMember`) times its size, the sum of the densities of the class
        where its sectors hold equal occupations (their levels are equal; the mean over the class otherwise)."""

        decomposition = self.decomposition
        sums = np.asarray(sums, dtype=np.float64)
        wedge = np.zeros(decomposition.wedge_size, dtype=np.float64)
        for sector in [int(source)] + [int(member) for member in members]:
            orbits = decomposition.sector_orbit_indices(sector)
            if sector != source:
                orbits = orbits[self.classes[sector].rows]
            wedge[orbits] += sums / self._multiplicities[orbits]
        wedge /= float(1 + len(members))
        return wedge[decomposition.reduction.full_to_wedge]

    def force_orbits(self, geometry):
        """``(representatives, images, matrices)`` for the nonlocal forces of one atom per orbit of the little group,
        or ``None`` (the job's force takes no atoms, every atom is its own orbit, or an image misses).

        ``images[p, a]`` is the atom that operation ``p`` of the group (``matrices[p]``, with its translation) carries
        atom ``a`` onto (:func:`..Symmetry.axis_reflection.atom_image_table`).  The states of a sector are
        eigenfunctions of every operation of the group, so the nonlocal force of their density on atom ``R a + t`` is
        ``R`` times that on atom ``a`` (:meth:`complete_orbit_forces`)."""

        if not geometry.get("force_atoms"):
            return None
        with self._lock:
            if self._orbit_forces is None:
                self._orbit_forces = self._make_force_orbits(geometry) or False
            return self._orbit_forces or None

    def _make_force_orbits(self, geometry):
        from ..Symmetry.axis_reflection import atom_image_table

        reduction = self.decomposition.reduction
        representatives = getattr(reduction, "atom_representatives", None)
        operations = getattr(reduction, "operations", None)
        if representatives is None or operations is None:
            return None
        members = np.unique(np.asarray(representatives, dtype=np.int64))
        if members.size >= np.size(representatives):
            return None
        translations = getattr(reduction, "translations", None)
        images = atom_image_table(
            list(geometry["atoms"]), list(operations), None if translations is None else list(translations),
            geometry["grid"], 1.0e-6,
        )
        if images is None:
            return None
        return members, images, np.asarray(operations, dtype=np.float64).reshape(-1, 3, 3)

    @staticmethod
    def complete_orbit_forces(forces: np.ndarray, orbits) -> np.ndarray:
        """The forces of every atom from those of the representatives of their orbits (:meth:`force_orbits`)."""

        members, images, matrices = orbits
        forces = np.asarray(forces, dtype=np.float64)
        out = np.full_like(forces, np.nan)
        for operation in range(images.shape[0]):
            out[images[operation, members]] = forces[members] @ matrices[operation].T
        if np.isnan(out).any():
            raise RuntimeError("the orbits of the little group miss an atom of the nonlocal forces")
        return out

    def twin(self, k_index: int) -> "KPointSectors":
        """The sectors of the same k-point for another task (the other channel of a spin-polarized run): the bases
        and the static operators shared, the numbers of states and the potential of its own."""

        other = object.__new__(KPointSectors)
        other.__dict__.update(self.__dict__)
        other.k_index = int(k_index)
        other.counts = list(self.counts)
        other._potential = (None, None)
        other._exchange = {}
        return other

    def _gather(self, sector: int) -> sp.csr_matrix:
        """``U^H`` of ``sector`` as a sparse matrix (its rows by the whole grid's), made once."""

        found = self._gathers.get(sector)
        if found is None:
            decomposition = self.decomposition
            rows = decomposition.reduction.full_to_wedge.shape[0]
            # (the conjugate phases of a sector of complex characters, PARSEC_KPOINT_SECTOR_CHARACTERS=complex)
            phases = np.conj(self._phase_row(sector))
            gather = sp.csr_matrix(
                (phases, (np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64), np.arange(rows))),
                shape=(decomposition.wedge_size, rows),
            )
            orbits = decomposition.sector_orbit_indices(sector)
            scale = sp.diags(1.0 / np.sqrt(self._multiplicities[orbits]))
            found = self._gathers[sector] = (scale @ gather[orbits]).tocsr().astype(np.complex128)
        return found

    def reduce(self, sector: int, block):
        """``U^H x`` on the rows of ``sector`` for the columns of ``block`` (rows of the whole grid): the adjoint
        of :meth:`orbitals`.  A block on a device (the exchange operator of a hybrid, built where its k-point is
        solved) is reduced there, and the result stays there."""

        gather = self._gather(sector)
        get = getattr(block, "get", None)
        if get is not None and not isinstance(block, np.ndarray):
            import cupy as cp
            import cupyx.scipy.sparse as csp

            from ..Hamiltonian.complex_csr_device import ordered_device_matrix

            with cp.cuda.Device(int(block.device.id)):
                # (PARSEC_ORDERED_SUMS=auto: in a fixed order of summation)
                device = ordered_device_matrix(csp.csr_matrix(gather))
                source = block[:, None] if block.ndim == 1 else block
                return cp.asfortranarray(device @ cp.ascontiguousarray(source, dtype=cp.complex128))
        block = np.asarray(block, dtype=np.complex128)
        block = block[:, None] if block.ndim == 1 else block
        return np.asarray(gather @ block)

    def _reduced_exchange(self, sector: int, exchange):
        """The compressed exchange ``-a_x xi xi^H`` of the whole grid on the rows of ``sector``: ``U^H xi`` (made
        once per term and sector)."""

        from parsec_python.V_xc.bloch_exchange import BlochCompressedExchange

        cached = self._exchange.get(sector)
        if cached is not None and cached[0] is exchange:
            return cached[1]
        projectors = getattr(exchange, "projectors", None)
        if projectors is None or getattr(exchange, "released", False):
            raise RuntimeError("the exchange operator of this Hamiltonian was released")
        reduced = BlochCompressedExchange(
            self.reduce(sector, projectors),
            fraction=float(exchange.fraction),
            asymmetry=float(getattr(exchange, "asymmetry", 0.0)),
        )
        self._exchange[sector] = (exchange, reduced)
        return reduced

    def _reduced_hubbard(self, sector: int, projectors):
        """The LDA+U projectors of the whole grid on the rows of ``sector``: ``U^H P`` (made once per set of projectors
        and sector; the caller holds the lock)."""

        cached = self._hubbard.get(sector)
        if cached is not None and cached[0] is projectors:
            return cached[1]
        matrix = (self._gather(sector) @ sp.csc_matrix(projectors.projectors, dtype=np.complex128)).tocsc()
        reduced = dataclasses.replace(projectors, projectors=matrix)
        self._hubbard[sector] = (projectors, reduced)
        return reduced

    def reduced_hubbard(self, sector: int, projectors):
        """The LDA+U projectors ``projectors`` of the whole grid reduced to the rows of ``sector``
        (:meth:`_reduced_hubbard`, shared with its Hamiltonians)."""

        with self._lock:
            return self._reduced_hubbard(sector, projectors)

    def _stencil_parts(self, full):
        """``(shape, half width, centre, weights)`` of the kinetic term of this k-point as an axial stencil
        (:mod:`..Hamiltonian.sector_bloch_stencil`), ``None`` where the matrices of the whole grid are not one,
        ``PARSEC_KPOINT_SECTOR_STENCIL`` is ``off`` or the operators stay on the CPU (:attr:`device_operators`
        false: the host applies the reduced CSR in one scipy product, 3-4 times faster than a loop over the slots of
        the table).  The check of the matrices of the whole grid is made once for the k-points of a structure (they
        share them: the cache of the plans)."""

        from ..Hamiltonian.sector_bloch_stencil import checked_axial_stencils, kinetic_weights, sector_stencil_setting

        if sector_stencil_setting() == "off" or not self.device_operators:
            return None
        laplacian, gradient = full.negative_laplacian, tuple(full.gradient)
        anchors = (laplacian, *gradient)
        with _SHARED_LOCK:
            entry = self._shared.get("axial_stencils")
            if entry is None or len(entry[0]) != len(anchors) or any(a is not b for a, b in zip(entry[0], anchors)):
                entry = (anchors, checked_axial_stencils(_materialized(laplacian), gradient))
                self._shared["axial_stencils"] = entry
        stencils = entry[1]
        if stencils is None:
            return None
        found = kinetic_weights(stencils, self.k_point, getattr(full, "bloch_laplacian", None))
        if found is None:
            return None
        centre, weights = found
        return stencils.shape, stencils.half_width, centre, weights

    def _build_static(self, full, sectors) -> None:
        """The kinetic terms (with ``|k|^2``), projectors and empty stand-ins of the real stencils of ``sectors``
        (one pass over the operator of the whole grid for all of them).

        The kinetic term of a sector is its gathered stencil from the grid
        (:class:`..Hamiltonian.sector_bloch_stencil.SectorStencilTable`, ``PARSEC_KPOINT_SECTOR_STENCIL=auto``) where
        the kinetic matrices are an axial stencil of the grid, else the reduced complex CSR of the matrix of the whole
        grid (as before)."""

        sectors = sorted({int(sector) for sector in sectors} - set(self._static))
        if not sectors:
            return
        from ..Hamiltonian.sector_bloch_stencil import build_sector_table

        kinetic: dict[int, Any] = {}
        parts = self._stencil_parts(full)
        if parts is not None:
            for sector in sectors:
                table = build_sector_table(self.decomposition, sector, *parts)
                if table is not None:
                    kinetic[sector] = table
        rest = [sector for sector in sectors if sector not in kinetic]
        if rest:
            k = self.k_point
            bloch = getattr(full, "bloch_laplacian", None)
            if bloch is not None:
                matrix = sp.csr_matrix(bloch, dtype=np.complex128)
            else:
                laplacian = _materialized(full.negative_laplacian)
                matrix = sp.csr_matrix(laplacian, dtype=np.complex128)
                # sum_c k_c G_c, here only (the driver's cache of it would keep a matrix of the whole grid per k-point)
                k_gradient = None
                for axis, gradient in enumerate(full.gradient):
                    if k[axis] != 0.0:
                        term = float(k[axis]) * sp.csr_matrix(gradient, dtype=np.float64)
                        k_gradient = term if k_gradient is None else k_gradient + term
                if k_gradient is not None:
                    matrix = matrix - 2.0j * k_gradient
                matrix = matrix + float(k @ k) * sp.identity(matrix.shape[0], dtype=np.complex128, format="csr")
            matrices = self.decomposition.reduce_hermitian_operators(matrix, rest)
            del matrix
            for sector in rest:
                kinetic[sector] = matrices[sector]
        projectors = compress_projectors_for_sectors(full.nonlocal_operator, self.decomposition.reduction)
        reduced = self.decomposition.reduce_nonlocal_operators(projectors, sectors)
        for sector in sectors:
            empty = sp.csr_matrix((self.sizes[sector], self.sizes[sector]), dtype=np.float64)
            self._static[sector] = (kinetic[sector], reduced[sector], empty)

    def _orbit_means(self, potential: np.ndarray) -> np.ndarray:
        cached, means = self._potential
        if cached is not potential:
            means = self.decomposition.invariant_wedge_values(potential)
            # the array is held, so that its identity is not reused by another one while cached
            self._potential = (potential, means)
        return means

    def hamiltonian(self, sector: int, full, wanted=None) -> KPointKohnShamHamiltonian:
        """The Hamiltonian of ``sector`` from the Bloch Hamiltonian ``full`` of the whole grid.

        The static operators of the sectors in ``wanted`` (those this process solves; all where ``None``) are built
        with those of ``sector`` the first time.  The exact exchange of a hybrid functional
        (``KPointExchangeHamiltonian``: ``base`` and ``exchange``) joins as ``-a_x (U^H xi)(U^H xi)^H``.
        """

        exchange = getattr(full, "exchange", None)
        if exchange is not None and getattr(full, "base", None) is not None:
            from parsec_python.Hamiltonian.kpoint_operator import KPointExchangeHamiltonian

            if self.real_forms and sector in self.real_forms:
                raise RuntimeError("the real form of a sector does not take the exchange of a hybrid")
            base = self.hamiltonian(sector, full.base, wanted)
            with self._lock:
                reduced = self._reduced_exchange(sector, exchange)
            return KPointExchangeHamiltonian(base, reduced)
        term = getattr(full, "hubbard", None)
        if term is not None and getattr(full, "base", None) is not None:
            # LDA+U (PARSEC_KPOINT_SECTOR_LDAU): (U^H P) D (U^H P)^H on the rows of the sector
            from parsec_python.Hamiltonian.hubbard import HubbardOperator
            from parsec_python.SCF.kpoint_hubbard import BlochHubbardHamiltonian

            base = self.hamiltonian(sector, full.base, wanted)
            with self._lock:
                reduced = self._reduced_hubbard(sector, term.projectors)
            return BlochHubbardHamiltonian(base, HubbardOperator(reduced, term.potentials))
        with self._lock:
            if sector not in self._static:
                self._build_static(full, range(self.order) if wanted is None else set(wanted) | {sector})
            kinetic, projectors, empty = self._static[sector]
            means = self._orbit_means(full.effective_potential)
            real = None
            if self.real_forms and sector in self.real_forms:
                real = self._real_parts(sector, kinetic, projectors)
        orbits = self.decomposition.sector_orbit_indices(sector)
        hamiltonian = SectorBlochHamiltonian(
            empty,
            (empty,) * 3,
            np.ascontiguousarray(means[orbits]),
            projectors,
            self.k_point,
            bloch_laplacian=kinetic,
        )
        if real is None:
            return hamiltonian
        from .kpoint_real_forms import RealFormHamiltonian

        return RealFormHamiltonian(hamiltonian, self.real_forms[sector], real)

    def density(self, sector: int, sums: np.ndarray) -> np.ndarray:
        """``|U v|^2`` on the whole grid from ``q = sum_n f_n |v_n|^2`` on the rows of ``sector``."""

        decomposition = self.decomposition
        orbits = decomposition.sector_orbit_indices(sector)
        wedge = np.zeros(decomposition.wedge_size, dtype=np.float64)
        wedge[orbits] = np.asarray(sums, dtype=np.float64) / self._multiplicities[orbits]
        return wedge[decomposition.reduction.full_to_wedge]

    def expand(self, sector: int, block, xp=np):
        """:meth:`orbitals` where ``block`` is: a device block becomes ``U v`` on its device (one gather with the
        phase and orbit factor of every row, made once per sector and device)."""

        if xp is np or isinstance(block, np.ndarray):
            return self.orbitals(sector, np.asarray(block))
        device = int(block.device.id)
        found = self._expanders.get((sector, device))
        if found is None:
            decomposition = self.decomposition
            orbits = decomposition.sector_orbit_indices(sector)
            place = np.full(decomposition.wedge_size, -1, dtype=np.int64)
            place[orbits] = np.arange(orbits.size, dtype=np.int64)
            to_wedge = np.asarray(decomposition.reduction.full_to_wedge, dtype=np.int64)
            rows = place[to_wedge]
            factor = self._phase_row(sector) / np.sqrt(self._multiplicities[to_wedge])
            factor[rows < 0] = 0.0
            rows[rows < 0] = 0
            with xp.cuda.Device(device):
                found = (xp.asarray(rows), xp.asarray(factor))
            self._expanders[(sector, device)] = found
        rows, factor = found
        return xp.take(block, rows, axis=0) * factor[:, None]

    def partner_orbitals(self, grid, whole: np.ndarray) -> np.ndarray:
        """The orbitals of the partner sector from those of the sector solved (whole grid, :attr:`partners`):
        ``A u = e^{-iG.r} conj(u)``, ``G = 2k``."""

        phases = np.exp(-1j * (np.asarray(grid.coordinates, dtype=np.float64) @ (2.0 * self.k_point)))
        return phases[:, None] * np.conj(np.asarray(whole, dtype=np.complex128))

    def orbitals(self, sector: int, block: np.ndarray) -> np.ndarray:
        """``U v`` on the whole grid for the columns of ``block`` (rows of ``sector``)."""

        decomposition = self.decomposition
        block = np.asarray(block, dtype=np.complex128).reshape(-1, np.shape(block)[-1] if np.ndim(block) > 1 else 1)
        orbits = decomposition.sector_orbit_indices(sector)
        wedge = np.zeros((decomposition.wedge_size, block.shape[1]), dtype=np.complex128)
        wedge[orbits] = block / np.sqrt(self._multiplicities[orbits])[:, None]
        phases = self._phase_row(sector)
        return phases[:, None] * wedge[decomposition.reduction.full_to_wedge]


_TOLERANCE_SWITCH = "PARSEC_KPOINT_SECTOR_TOLERANCE"


def kpoint_sector_tolerance_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_TOLERANCE``: ``follow`` (default, since 2026-11-19) builds the cell group of the k-point
    sectors and of the ``kpoint`` route of ``PARSEC_GAMMA_HYBRID_SECTORS`` at ``PARSEC_SYMMETRY_ATOM_TOLERANCE``, as
    the Gamma detection (:func:`kpoint_sector_atom_tolerance`); ``fixed`` keeps the 1e-7 bohr of before whatever that
    says, bit for bit.  Unset, the two agree."""

    value = os.environ.get(_TOLERANCE_SWITCH, "follow").strip().lower() or "follow"
    if value not in {"follow", "fixed"}:
        raise ValueError(f"{_TOLERANCE_SWITCH} must be follow or fixed, not {value!r}")
    return value


def kpoint_sector_atom_tolerance(*, forces: bool = False) -> float:
    """The atom tolerance (bohr) of the cell group of the k-point sectors: ``PARSEC_SYMMETRY_ATOM_TOLERANCE``, as for
    the detection of the Gamma sectors, at most :data:`parsec_python.Forces.symmetry.POSITION_TOLERANCE` (1e-6) for
    a loop that computes forces (the extras of the sectors take the projectors of one atom per orbit, so a looser
    group would give forces of the Hamiltonian averaged over it; the Gamma route refuses forces off by more as
    well).  ``1e-7`` by default, the tolerance of before, and always under ``PARSEC_KPOINT_SECTOR_TOLERANCE=fixed``
    (:func:`kpoint_sector_tolerance_setting`)."""

    from parsec_python.Forces.symmetry import POSITION_TOLERANCE

    from ..Symmetry.axis_reflection import DEFAULT_ATOM_TOLERANCE, symmetry_atom_tolerance

    tolerance = symmetry_atom_tolerance()
    if kpoint_sector_tolerance_setting() == "fixed":
        return DEFAULT_ATOM_TOLERANCE
    return min(tolerance, POSITION_TOLERANCE) if forces else tolerance


def check_kpoint_sector_switches() -> None:
    """Read the switches of the k-point sectors and of the detection of their cell group, so that a value they do not
    know is refused at the preparation of a run (``ValueError``) and not first at the setup of its k-point solver:
    ``PARSEC_KPOINT_SECTORS``, ``PARSEC_KPOINT_SECTOR_TOLERANCE``, ``PARSEC_SYMMETRY_ATOM_TOLERANCE``,
    ``PARSEC_PERIODIC_TRANSLATIONS``, ``PARSEC_PERIODIC_TRIVIAL_ORIGIN``, ``PARSEC_PERIODIC_SYMMETRY_CENTRE``,
    ``PARSEC_PERIODIC_DEFECT_CENTRE``, ``PARSEC_HYBRID_KPOINT_SECTORS``, ``PARSEC_PERIODIC_IMAGE_ROWS`` (the
    density group of the k-point wedge), ``PARSEC_KPOINT_SECTOR_CLASSES``, ``PARSEC_KPOINT_SECTOR_REAL``, the real
    arithmetic of the Gamma point (``PARSEC_KPOINT_REAL_GAMMA``, :mod:`.kpoint_real`),
    ``PARSEC_KPOINT_SECTOR_WARM_GROWTH`` and ``PARSEC_KPOINT_SECTOR_LDAU``.  Reading them changes nothing else."""

    from parsec_python.Grid.space_group import periodic_image_rows_setting

    from .kpoint_real import check_switches as check_real_switches
    from ..Symmetry.axis_reflection import (
        periodic_centre_setting,
        periodic_defect_centre_setting,
        periodic_translation_setting,
        periodic_trivial_origin_setting,
        symmetry_atom_tolerance,
    )

    periodic_image_rows_setting()
    kpoint_sectors_setting()
    kpoint_sector_tolerance_setting()
    symmetry_atom_tolerance()
    periodic_translation_setting()
    periodic_trivial_origin_setting()
    periodic_centre_setting()
    periodic_defect_centre_setting()
    hybrid_kpoint_sectors_setting()
    sector_classes_setting()
    from .kpoint_real_forms import sector_real_setting
    from ..Symmetry.complex_sectors import sector_characters_setting

    sector_real_setting()
    sector_characters_setting()
    check_real_switches()
    kpoint_sector_warm_growth_setting()
    kpoint_sector_ldau_setting()


def _cell_group_text(cell: "CellGroup", tolerance: float) -> str:
    """What the detection of the k-point sectors names of their cell group: its operations, those with a
    translation, and the atom tolerance it was built at (with the switches that set it)."""

    translated = sum(1 for shift in cell.shifts if np.any(np.asarray(shift) != 0))
    return (
        f"cell group of {cell.order} operations r -> R r + t of the grid and the atoms detected (signed permutations, "
        f"{translated} of them with a translation; PARSEC_PERIODIC_TRANSLATIONS, PARSEC_PERIODIC_SYMMETRY_CENTRE) at an "
        f"atom tolerance of {tolerance:.1e} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE, "
        f"{_TOLERANCE_SWITCH}={kpoint_sector_tolerance_setting()})"
    )


def plan_sectors(job: KPointJob, notes: list | None = None) -> list[KPointSectors | None] | None:
    """The sectors of every k-point of ``job`` (``None`` for a k-point that stays one task), ``None`` for none.

    The cell group is built at :func:`kpoint_sector_atom_tolerance` (``PARSEC_SYMMETRY_ATOM_TOLERANCE``, at most 1e-6
    bohr for a job with extras, the forces; 1e-7 under ``PARSEC_KPOINT_SECTOR_TOLERANCE=fixed``); the plans of a loop
    before on the same structure (the cache of ``geometry``, same signature) are taken before any search of the group.
    ``notes`` (a list): the line of the report that says which cell group was found and at which tolerance, or why
    none was searched, is appended to it (:func:`sector_solver`; ``kpoint_symmetry_detection`` of ``parsec.out``).
    """

    def said(text: str, plans=None):
        if notes is not None:
            notes.append(text)
        return plans

    geometry = job.geometry
    grid = geometry["grid"]
    try:
        if _lattice_period(grid) is None or not _periodic_row_major(grid):
            return said("skipped: the k-point sectors take a bulk orthorhombic box of row-major points, not this grid")
    except ValueError:  # not a bulk orthorhombic cell
        return said("skipped: the k-point sectors take a bulk orthorhombic box of row-major points, not this grid")
    k_points = np.asarray(geometry["k_points"], dtype=np.float64).reshape(-1, 3)
    # the k-point of every task (the channels of a spin-polarized run share the k-points: tasks spin * n_k + k)
    k_of_task = np.asarray(geometry.get("k_of_task", np.arange(k_points.shape[0])), dtype=np.int64).reshape(-1)
    if k_of_task.shape[0] != job.n_tasks or k_of_task.size == 0 or int(k_of_task.max()) >= k_points.shape[0]:
        return said("skipped: the tasks of this job are not one per k-point and spin channel")
    states, rows = int(job.number_of_states), int(grid.size)
    cache = geometry.get("cache")
    # PARSEC_HYBRID_KPOINT_SECTORS=on (the loops of a Fock loop): every k-point with sectors, whatever its size and the
    # work they save
    any_size = isinstance(cache, dict) and bool(cache.get(ANY_SIZE_KEY))
    if not any_size and rows * states < _MINIMUM_ENTRIES:  # before any search of the group: a small cell stays as it was
        return said(
            f"skipped: {rows} grid points x {states} states are below the floor of the k-point sectors, "
            f"{_MINIMUM_ENTRIES} entries (no cell group searched; one task per k-point)"
        )
    tolerance = kpoint_sector_atom_tolerance(forces=job.extras is not None)
    margin = sector_margin()
    # the conjugate pairs of the sectors of the time-reversal invariant points in the Bloch form
    # (PARSEC_KPOINT_TRIM_REAL, .kpoint_real): an unpolarized loop without the exchange of a hybrid, not the band
    # k-points of a Gamma cell (their own invariance)
    from parsec_python.Laplacian.finite_difference import bloch_form_at

    from .kpoint_real import trim_partners, trim_real_setting

    pairs = (
        trim_real_setting() == "auto"
        and len(set(int(value) for value in k_of_task)) == int(k_of_task.size)
        and geometry.get("exchange_cross_scale") is None
        and geometry.get("invariance") is None
    )
    bloch = tuple(bool(pairs and bloch_form_at(grid, k_point)) for k_point in k_points)
    # the classes of sectors carried into each other by the operations of the little group beyond H_k
    # (PARSEC_KPOINT_SECTOR_CLASSES, sector_classes): loops without the exchange of a hybrid (whose operator is made
    # from the orbitals of every k-point), both spin channels, the band k-points of a Gamma cell (their potential is
    # invariant under its group), at the points without the pairs above
    # (the start loop of a Fock loop too: its states go to the loops with the exchange, which keep every sector)
    from parsec_python.SCF.hybrid_periodic import FOCK_LOOP_KEY

    fock_loop = isinstance(cache, dict) and bool(cache.get(FOCK_LOOP_KEY))
    # (and not with the Hubbard term of LDA+U in the sectors, a geometry with "hubbard_part": the occupation matrices
    # are summed from the orbitals of the sectors solved, and the images of a class would add theirs only where the
    # occupation matrices are symmetrized over a group that holds the operations of the classes)
    hubbard = geometry.get("hubbard_part") is not None or geometry.get("hubbard_projectors") is not None
    classes_wanted = (
        sector_classes_setting() == "auto"
        and geometry.get("exchange_cross_scale") is None
        and not fock_loop
        and not hubbard
    )
    # the real forms of the sectors kept by an antiunitary P_g K (PARSEC_KPOINT_SECTOR_REAL, .kpoint_real_forms): loops
    # without the exchange of a hybrid, not the band k-points, at the points without the pairs above
    from .kpoint_real_forms import sector_real_forms, sector_real_setting

    from ..Symmetry.complex_sectors import largest_abelian_subgroup, sector_characters_setting

    complex_wanted = sector_characters_setting() == "complex" and not hubbard
    real_wanted = (
        sector_real_setting() == "auto"
        and geometry.get("exchange_cross_scale") is None
        and geometry.get("invariance") is None
        and not fock_loop
        and not hubbard
    )
    signature = (
        int(job.n_tasks), states, rows, margin, _EXTRA_STATES, tuple(int(value) for value in k_of_task),
        tuple(np.round(k_points, 12).ravel().tolist()), tolerance, any_size, bloch,
    ) + ((("classes",) if classes_wanted else ())) + ((("real",) if real_wanted else ())) + (
        (("complex",) if complex_wanted else ())
    )
    # The plans the loops before kept for this structure (the cache of its projectors), before the group is searched
    # again: they were made from the same atoms and grid.
    if isinstance(cache, dict):
        found = cache.get(_CACHE_KEY)
        if found is not None and found[0] == signature:
            if len(found) > 2 and found[2] is not None:
                said(found[2])
            return found[1]
    cell = CellGroup(grid, geometry["atoms"], atom_tolerance=tolerance)
    if cell.table is None or cell.order < 2:
        return said(
            "identity only: no cell group of more than one operation at an atom tolerance of "
            f"{tolerance:.1e} bohr (PARSEC_SYMMETRY_ATOM_TOLERANCE, {_TOLERANCE_SWITCH}="
            f"{kpoint_sector_tolerance_setting()}); one task per k-point"
        )
    group_text = _cell_group_text(cell, tolerance)
    # a caller whose potential is invariant under a known group (the band k-points of a Gamma cell solved in sectors,
    # .band_sectors) gives its own test
    invariance_of = geometry.get("invariance")
    invariance = (
        PotentialInvariance(cell, geometry.get("density_symmetry"), k_points, geometry["weights"], geometry["lattice"])
        if invariance_of is None
        else invariance_of(cell, k_points, geometry["weights"], geometry["lattice"])
    )
    decompositions: dict[tuple[int, ...], ReflectionRepresentationDecomposition | None] = {}
    plans: list[KPointSectors | None] = []
    for index, k_point in enumerate(k_points):
        members = generated_subgroup(cell.table, cell.little_group(k_point, invariance))
        elements, generators = largest_commuting_involutions(cell.table, members)
        # the groups whose sectors may serve the k-point: the commuting involutions, and under
        # PARSEC_KPOINT_SECTOR_CHARACTERS=complex a largest abelian subgroup with complex characters (not at Gamma, not
        # with the pairs of a time-reversal invariant point in the Bloch form); the one of the least work is taken
        candidates: list[tuple] = []
        if len(elements) >= 2:
            key = tuple(elements)
            if key not in decompositions:
                decomposition = None
                reduction = little_group_reduction(cell, elements, generators)
                if reduction is not None and not sector_orbits_exact(reduction, cell.atoms, grid):
                    # atoms off the group by more than 1e-6 bohr (a loose PARSEC_SYMMETRY_ATOM_TOLERANCE): the
                    # projectors of every atom, since one atom per orbit would copy the offset of its representative
                    # (PARSEC_SECTOR_ORBITS_EXACT)
                    reduction = dataclasses.replace(reduction, atom_representatives=None, atom_orbit_sizes=None)
                if reduction is not None:
                    try:
                        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                    except RuntimeError:  # a character that no orbit admits (a grid too small for the group)
                        decomposition = None
                decompositions[key] = decomposition
            if decompositions[key] is not None:
                candidates.append((elements, generators, decompositions[key]))
        if complex_wanted and not bloch[index] and np.any(np.asarray(k_point) != 0.0):
            abelian = largest_abelian_subgroup(cell.table, members)
            if len(abelian) >= 2 and sorted(abelian) != sorted(int(e) for e in elements):
                key = ("complex",) + tuple(abelian)
                if key not in decompositions:
                    decompositions[key] = _complex_candidate(cell, abelian)
                if decompositions[key] is not None:
                    candidates.append((abelian, None, decompositions[key]))
        best = None
        for elements_of, generators_of, decomposition in candidates:
            sizes = decomposition.sector_sizes
            counts = [min(size, math.ceil(margin * states * size / rows) + _EXTRA_STATES) for size in sizes]
            partners = trim_partners(cell, elements_of, generators_of, k_point) if bloch[index] else None
            classes = (
                sector_classes(cell, members, elements_of, decomposition, k_point, invariance)
                if classes_wanted and partners is None
                else None
            )
            solved = range(len(sizes)) if classes is None else [s for s in range(len(sizes)) if s not in classes]
            work = sum(sizes[s] * counts[s] * counts[s] for s in solved)
            if best is None or work < best[0]:
                best = (work, elements_of, generators_of, decomposition, sizes, counts, partners, classes, solved)
        if best is None:
            plans.append(None)
            continue
        work, elements, generators, decomposition, sizes, counts, partners, classes, solved = best
        if not any_size and work > _WORK_FRACTION * rows * states * states:
            plans.append(None)
            continue
        plan = KPointSectors(index, k_point, decomposition, counts)
        # the operations of the sectors that the density is not symmetrized over: the sectors assume the invariance
        # of the potential under them (the symmetry guard, SectorKPointSolver.start_spectrum_report)
        plan.assumed = any(not invariance.in_density_group(element) for element in elements[1:])
        if bloch[index]:
            plan.partners = partners
        plan.classes = classes
        if real_wanted and partners is None:
            plan.real_forms = sector_real_forms(cell, elements, decomposition, k_point, invariance, solved)
        plans.append(plan)
    if not any(plan is not None for plan in plans):
        return said(
            f"{group_text}; no k-point of the {k_points.shape[0]} splits (no commuting involutions in a little group "
            "the potential keeps, or sectors that would not halve the work)"
        )
    split = [plan for plan in plans if plan is not None]
    detection = (
        f"{group_text}; {len(split)} of {k_points.shape[0]} k-point(s) split into the sectors of commuting "
        f"involutions of their little groups (the largest of {max(plan.order for plan in split)} operations)"
    )
    complex_points = sum(1 for plan in split if getattr(plan.decomposition, "is_complex", False))
    if complex_points:
        detection += (
            f"; {complex_points} of them in the complex characters of an abelian subgroup "
            "(PARSEC_KPOINT_SECTOR_CHARACTERS=complex)"
        )
    images = sum(len(plan.classes) for plan in split if plan.classes)
    if images:
        detection += (
            f"; {images} sector(s) not solved, images P_g u of sectors solved under operations of the little groups "
            f"beyond them ({_CLASSES_SWITCH}=auto)"
        )
    said(detection)
    from .spin_sectors import spin_guard_cell_group_setting

    # (read here, before the first solve: a bad value stops the run before its first loop)
    guard = SectorGuardGroup(cell, invariance, environments=spin_guard_cell_group_setting() == "auto")
    shared: dict = {}
    for plan in plans:
        if plan is not None:
            plan.guard_group = guard
            plan._shared = shared
    # the tasks of one k-point share its bases and static operators (KPointSectors.twin)
    by_task: list[KPointSectors | None] = []
    taken: set[int] = set()
    for task, k_index in enumerate(int(value) for value in k_of_task):
        plan = plans[k_index]
        if plan is None:
            by_task.append(None)
        elif k_index in taken:
            by_task.append(plan.twin(task))
        else:
            taken.add(k_index)
            plan.k_index = task
            by_task.append(plan)
    if isinstance(cache, dict):
        # the next loops of this structure (the Fock loop of a hybrid) take these plans and their static operators
        # (and the line of the report that names their group)
        cache[_CACHE_KEY] = (signature, by_task, detection)
    return by_task


class SectorKPointSolver:
    """The solver of a k-point job whose k-points are split into the sectors of their little groups.

    The tasks of the inner solver (made by ``make_solver``, the solver the job would have had) are the pairs
    ``(k-point, sector)``, ``(k-point, None)`` for a k-point that stays whole, in the order of their work (rows
    times states squared, the largest first: the MPI pools take them in this order, by their sizes).  The driver sees one
    task per k-point as before: the lowest levels of the union of its sectors, their occupations sent back to
    the sectors that hold them.
    """

    backend_name = "kpoint-sectors"

    def __init__(self, job: KPointJob, plans: list[KPointSectors | None], make_solver: Callable[..., Any], options) -> None:
        self.job = job
        self._plans = plans
        self._make_solver = make_solver
        self._options = dict(options)
        self._grid_size = int(job.geometry["grid"].size)
        # the Gamma point in real arithmetic (PARSEC_KPOINT_REAL_GAMMA), decided on the k-points of this job (the
        # size of a whole k-point; the band paths not) and handed to every inner solver
        if self._options.get("real_gamma") is None:
            from .kpoint_real import real_gamma_applies

            self._options["real_gamma"] = real_gamma_applies(job, self._grid_size)
        items: list[tuple[int, int | None]] = []
        for k_index in range(job.n_tasks):
            plan = plans[k_index]
            if plan is None:
                items.append((k_index, None))
            else:
                items.extend((k_index, sector) for sector in range(plan.order))
        self._items = sorted(items, key=lambda item: -self._rows(item) * self._states(item) ** 2)
        self._position = {item: position for position, item in enumerate(self._items)}
        # The conjugate pairs of the sectors of a time-reversal invariant point in the Bloch form (KPointSectors.partners,
        # PARSEC_KPOINT_TRIM_REAL): the item of the higher sector of a pair is not solved, it takes the levels of its
        # partner (``_source``: its position -> the partner's).  ``_solved``: the positions the inner solver solves, in
        # order (all of them without pairs: the inner tasks are the items, as before).
        # The sectors of a class (KPointSectors.classes, PARSEC_KPOINT_SECTOR_CLASSES) that are not solved join
        # ``_source`` as well: they take the levels of the sector solved of their class, their orbitals are its images
        # (``_images``: the positions of such items; ``_class_members``: the position of each sector solved -> the
        # sectors of its class that are not).
        self._source: dict[int, int] = {}
        self._images: set[int] = set()
        self._class_members: dict[int, list[int]] = {}
        for position, (k_index, sector) in enumerate(self._items):
            plan = plans[k_index]
            if plan is None or sector is None:
                continue
            if plan.classes and sector in plan.classes:
                source = self._position[(k_index, plan.classes[sector].source)]
                self._source[position] = source
                self._images.add(position)
                self._class_members.setdefault(source, []).append(sector)
                continue
            partners = plan.partners
            if partners is None:
                continue
            other = int(partners[sector])
            if other < sector and plan.sizes[other] == plan.sizes[sector]:
                self._source[position] = self._position[(k_index, other)]
        for members in self._class_members.values():
            members.sort()
        self._solved = [position for position in range(len(self._items)) if position not in self._source]
        self._task_of = {position: task for task, position in enumerate(self._solved)}
        self._kept: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._returned: dict[int, int] = {}
        self._regrowths = 0
        self._regrown_tasks = 0  # sectors solved again alone (their solver kept the others)
        self._rebuilds = 0  # growths that made a new inner solver
        # PARSEC_KPOINT_SECTOR_REFIT: the counts cut after the first solve (_refit), once per solver
        self._refit_wanted = sector_refit_setting() == "auto"
        self._refitted = False
        self._trimmed: set[tuple[int, int]] = set()  # the sectors whose counts the refit cut
        self._refit_states = 0  # the states the refit took off all sectors
        # PARSEC_KPOINT_FORCE_ORBITS (read here, before the first solve: a bad value stops the run before its loop)
        self._force_orbits = kpoint_force_orbits_setting() == "auto"
        # PARSEC_KPOINT_SECTOR_WARM_GROWTH (read here for the same reason; the inner solvers apply it in resize_tasks)
        kpoint_sector_warm_growth_setting()
        self._inner = None
        self._inner = self._new_inner()

    # ------------------------------------------------------------------ tasks
    def _rows(self, item) -> int:
        k_index, sector = item
        return self._grid_size if sector is None else self._plans[k_index].sizes[sector]

    def _states(self, item) -> int:
        k_index, sector = item
        return int(self.job.number_of_states) if sector is None else self._plans[k_index].counts[sector]

    def _new_inner(self):
        solved = [self._items[position] for position in self._solved]
        counts = tuple(self._states(item) for item in solved)
        inner_job = KPointJob(
            len(solved),
            max(counts),
            self.job.eigval_settings,
            self._make_hamiltonian,
            self._accumulate,
            states_per_task=counts,
            rows_per_task=tuple(self._rows(item) for item in solved),
            grid_points=self._grid_size,
            extras=None if self.job.extras is None else self._inner_extras,
            extras_by_columns=self.job.extras is not None,
        )
        options = dict(self._options)
        options["problem_size"] = max(self._rows(item) for item in solved)
        return self._make_solver(inner_job, **options)

    def _inner_task(self, position: int) -> int:
        """The task of the inner solver that solves the item at ``position`` (that of its partner for an item of a
        conjugate pair that is not solved)."""

        return self._task_of[self._source.get(position, position)]

    def _expanded(self, values) -> list:
        """The levels of the inner solver's tasks as levels of every item (an item of a pair not solved takes those
        of its partner)."""

        if not self._source:
            return list(values)
        return [values[self._inner_task(position)] for position in range(len(self._items))]

    def _make_hamiltonian(self, task: int, potentials: dict):
        k_index, sector = self._items[self._solved[task]]
        full = self.job.make_hamiltonian(k_index, potentials)
        if sector is None:
            return full
        return self._plans[k_index].hamiltonian(sector, full, self._owned_sectors(k_index))

    def _owned_sectors(self, k_index: int):
        """The sectors of ``k_index`` that this process solves: those of the tasks of its pool (MPI pools), all
        sectors otherwise (``None``)."""

        tasks = getattr(self._inner, "tasks", None)
        if not isinstance(tasks, (list, tuple)) or not tasks:
            return None
        return [
            sector for position, (k, sector) in enumerate(self._items)
            if k == k_index and sector is not None and position in self._task_of and self._task_of[position] in tasks
        ]

    def _accumulate(self, task: int, vectors, occupations, xp=np) -> dict:
        k_index, sector = self._items[self._solved[task]]
        if sector is None:
            return self.job.accumulate(k_index, vectors, occupations, xp=xp)
        # (the coefficients of a sector in its real form, PARSEC_KPOINT_SECTOR_REAL: T x of its real coordinates)
        vectors = self._plans[k_index].coefficients(sector, vectors, xp)
        sums = _host(weighted_square_sum(vectors, occupations, xp))
        members = self._class_members.get(self._solved[task])
        if members:
            # the occupations of the class are summed onto this sector (accumulate): the density of the class
            whole = self._plans[k_index].class_density(sector, members, sums)
        else:
            whole = self._plans[k_index].density(sector, sums)
        # the job's own density of the k-point (its weight and normalization) from one column |U v|; a job that sums
        # more (the exchange energy of a hybrid) gives its density part apart
        geometry = self.job.geometry
        parts = geometry.get("density_part", self.job.accumulate)(k_index, np.sqrt(whole)[:, None], np.ones(1))
        hubbard_part = geometry.get("hubbard_part")
        if hubbard_part is not None:
            # LDA+U (PARSEC_KPOINT_SECTOR_LDAU): the occupation matrices of the sector's states, <phi|U v> with the
            # projectors of the k-point reduced to its rows (those its Hamiltonians take)
            reduced = self._plans[k_index].reduced_hubbard(sector, geometry["hubbard_projectors"](k_index))
            parts = dict(parts)
            parts.update(hubbard_part(k_index, reduced, vectors, occupations, xp))
        scale = geometry.get("exchange_cross_scale")
        if scale is not None:
            parts = dict(parts)
            parts["exchange_cross"] = self._exchange_cross(k_index, sector, vectors, occupations, scale)
        return parts

    def _exchange_cross(self, k_index: int, sector: int, vectors, occupations, scale) -> np.ndarray:
        """``(s/2) w_k sum_n f_n <v_n|U^H V_ACE U|v_n>`` of a sector (unscaled): the exchange term reduced to it
        (made with its Hamiltonian, :meth:`KPointSectors.hamiltonian`), on the host."""

        from parsec_python.V_xc.exact_exchange import exchange_occupation_threshold

        cached = self._plans[k_index]._exchange.get(sector)
        values = np.asarray(_host(occupations), dtype=np.float64).reshape(-1)
        occupied = np.flatnonzero(values > exchange_occupation_threshold())
        if cached is None or not occupied.size:
            return np.zeros(1)
        block = np.asarray(_host(vectors))[:, occupied]
        reduced = cached[1]
        projectors = reduced.projectors
        if isinstance(projectors, np.ndarray):
            expectation = reduced.expectation_values(block, scaled=False)
        else:  # the reduction of a device exchange stays on its device: the overlaps are formed there
            import cupy as cp

            with cp.cuda.Device(int(projectors.device.id)):
                overlaps = projectors.conj().T @ cp.asarray(block)
                expectation = cp.asnumpy(cp.sum(cp.abs(overlaps) ** 2, axis=0))
            expectation = -expectation
        return np.asarray([float(scale(k_index)) * float(np.dot(values[occupied], expectation))])

    # ------------------------------------------------------------------- steps
    def _short_sectors(self, values) -> list[tuple[int, int]]:
        states = int(self.job.number_of_states)
        short = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                continue
            levels = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            union = np.sort(np.concatenate(levels))
            if union.size < states:
                short.extend((k_index, sector) for sector in range(plan.order) if plan.counts[sector] < plan.sizes[sector])
                continue
            cut = union[states - 1]
            for sector, part in enumerate(levels):
                if plan.counts[sector] < plan.sizes[sector] and (part.size == 0 or float(np.max(part)) < cut):
                    short.append((k_index, sector))
        return short

    def _deficits(self, short, values) -> dict[tuple[int, int], int]:
        """The estimated levels that each sector of ``short`` the refit cut lacks below the cut of its k-point
        (:func:`..Eigensolvers.symmetry.bracket_deficits`, every sector counted once)."""

        cut = [item for item in short if item in self._trimmed]
        if not cut:
            return {}
        from ..Eigensolvers.symmetry import bracket_deficits

        found: dict[tuple[int, int], int] = {}
        for k_index in sorted({k for k, _ in cut}):
            plan = self._plans[k_index]
            levels = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            if any(part.size == 0 for part in levels):
                continue
            failing = [sector for k, sector in cut if k == k_index]
            try:
                estimate = bracket_deficits(levels, [1] * plan.order, int(self.job.number_of_states), failing)
            except (ValueError, IndexError):
                continue
            found.update({(k_index, sector): int(value) for sector, value in estimate.items()})
        return found

    def _refit(self, values) -> None:
        """Cut the states of every sector to the levels it holds among those its k-point keeps, plus a buffer.

        The first counts carry the margin of :func:`sector_margin` over each sector's share of the rows, which a
        sector needs before its levels are known; after the first solve of the loop they are (the bracket of
        :meth:`_short_sectors` holds).  A sector that holds ``h`` of the ``M`` levels of its k-point keeps
        :func:`refitted_count` states.  The inner solver cuts their Ritz states at their next solves
        (``trim_tasks``: the lowest Ritz pairs are kept, nothing is solved again); a cut sector that later misses
        levels grows by its estimated deficit (:meth:`_deficits`).  Once per solver, where the inner solver can.
        """

        self._refitted = True
        trim = getattr(self._inner, "trim_tasks", None)
        if trim is None:
            return
        states = int(self.job.number_of_states)
        counts: dict[int, int] = {}
        cut: list[tuple[int, int, int]] = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                continue
            parts = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            levels = np.concatenate(parts)
            if levels.size < states:
                continue
            owners = np.concatenate([np.full(part.size, sector, dtype=np.int64) for sector, part in enumerate(parts)])
            held = np.bincount(owners[np.argsort(levels, kind="stable")[:states]], minlength=plan.order)
            if plan.partners is not None and self._source:
                # a conjugate pair holds as many as the member that holds more (the cut may split a degenerate pair)
                held = np.maximum(held, held[np.asarray(plan.partners, dtype=np.int64)])
            if plan.classes:
                # so does a class (PARSEC_KPOINT_SECTOR_CLASSES): its sectors hold copies of the same levels
                for member, entry in plan.classes.items():
                    if member < len(held):
                        members = plan.class_of(entry.source)
                        held[members] = int(np.max(held[members]))
            for sector in range(plan.order):
                count = refitted_count(int(held[sector]), plan.sizes[sector], plan.counts[sector])
                if count < plan.counts[sector]:
                    cut.append((k_index, sector, count))
                    position = self._position[(k_index, sector)]
                    if position not in self._source:
                        counts[self._task_of[position]] = count
        if not counts or not trim(counts):
            return
        for k_index, sector, count in cut:
            if self._position[(k_index, sector)] not in self._source:
                self._refit_states += self._plans[k_index].counts[sector] - count
            self._plans[k_index].counts[sector] = count
            self._trimmed.add((k_index, sector))

    def _grow(self, short, potentials: dict, filter_degree: int, tolerance, values=None) -> list[np.ndarray]:
        """More states for the sectors ``short``, and the step solved again.

        Only those sectors are solved again where the inner solver can change the states of some tasks
        (``resize_tasks``: the others keep their Ritz states and levels); else a new inner solver solves every
        sector from the start.  ``values``: the levels of the solve that fell short (the deficits of the sectors the
        refit cut).
        """
        counts = {}
        deficits = self._deficits(short, values) if values is not None else {}
        for k_index, sector in short:
            plan = self._plans[k_index]
            count = plan.counts[sector]
            if (k_index, sector) in deficits:
                # a sector the refit cut grows by its estimated deficit (the Gamma route's rule, bracket_deficits)
                grown = count + _EXTRA_STATES + deficits[(k_index, sector)]
            else:
                grown = max(math.ceil(_GROWTH * count), count + _EXTRA_STATES)
            position = self._position[(k_index, sector)]
            solved = self._items[self._source.get(position, position)][1]  # the sector of a pair that is solved
            task = self._inner_task(position)
            plan.counts[solved] = min(plan.sizes[sector], max(grown, counts.get(task, 0)))
            counts[task] = plan.counts[solved]
        self._pair_counts()
        self._regrowths += 1
        resize = getattr(self._inner, "resize_tasks", None)
        if resize is not None and resize(counts):
            self._regrown_tasks += len(counts)
            return self._expanded(self._inner.solve(potentials, filter_degree, tolerance, tasks=sorted(counts)))
        self._rebuilds += 1
        self._inner.close()
        self._inner = None
        self._inner = self._new_inner()
        return self._expanded(self._inner.solve(potentials, filter_degree, tolerance))

    def _pair_counts(self) -> None:
        """Give each item of a conjugate pair that is not solved the states of its partner, which solves both."""

        for position, source in self._source.items():
            k_index, sector = self._items[position]
            plan = self._plans[k_index]
            plan.counts[sector] = plan.counts[self._items[source][1]]

    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None) -> list[np.ndarray]:
        """The lowest ``number_of_states`` levels of every k-point (task order)."""

        values = self._expanded(self._inner.solve(potentials, filter_degree, tolerance))
        for _ in range(_ATTEMPTS):
            short = self._short_sectors(values)
            if not short:
                union = self._union(values)
                if self._refit_wanted and not self._refitted:
                    self._refit(values)
                return union
            values = self._grow(short, potentials, filter_degree, tolerance, values)
        raise RuntimeError(
            f"the sectors of the k-points did not hold the lowest levels after {_ATTEMPTS} attempts; "
            f"{_SWITCH}=off solves each k-point whole"
        )

    def reported_test_available(self, filter_degree: int) -> bool:
        """Whether the inner solver can make the first solves of :meth:`solve_reported` (the MPI pools cannot) and the
        memory holds their subspaces; decides where these wait between the parts."""

        prepare = getattr(self._inner, "prepare_reported", None)
        if prepare is None:
            return False
        return bool(prepare(int(job_eigval_settings(self.job, filter_degree).safety_buffer)))

    def _reported_cuts(self, values) -> dict[int, float]:
        """The highest level each k-point reports (no limit where it stays whole or its sectors hold too few)."""

        states = int(self.job.number_of_states)
        cuts: dict[int, float] = {}
        for k_index, plan in enumerate(self._plans):
            cuts[k_index] = float("inf")
            if plan is None:
                continue
            union = np.sort(
                np.concatenate([np.asarray(values[self._position[(k_index, s)]], float) for s in range(plan.order)])
            )
            if union.size >= states:
                cuts[k_index] = float(union[states - 1])
        return cuts

    def solve_reported(self, potentials: dict, filter_degree: int) -> list[np.ndarray]:
        """The levels of :meth:`solve` for a single converged first solve whose test covers the reported states.

        The band k-points (:mod:`.band_sectors`) report the lowest ``number_of_states`` levels of the union of their
        sectors, so the margin of a sector above them (:func:`sector_margin`) need not converge; which levels those are
        is known only from all the sectors.  Every sector makes its fixed cycles; then each goes on from where it stood
        until the levels it holds at or below the highest level of its k-point pass the relative test
        (:func:`reported_level_tolerance`).  With the levels of all the sectors, now close to their limits, the test is
        made again on every pair with ``e - r <= cut`` (the highest level of the k-point), and the sectors that miss it
        go on, up to the extra cycles of
        :func:`~parsec_python.Eigensolvers.eigval.first_solve_extra_cycles`.  That condition is a heuristic: it does not
        prove that no level is missing from a sector.  Sectors that then hold too few grow as in :meth:`solve`.  The order
        of the decisions depends on the levels alone, not on the timing of the workers.

        A k-point that stays whole has no margin: its task makes the one first solve of before, with its test at
        ``Diag_Tolerance``, whatever k-points share its chunk.  It runs in the first part, in its place among the
        tasks (the largest first), so that its whole basis is not filtered next to the subspaces of all the sectors.
        """

        from parsec_python.Eigensolvers.eigval import first_solve_extra_cycles

        settings = job_eigval_settings(self.job, filter_degree)
        relative = reported_level_tolerance(settings.chebdav.convergence_tolerance)
        cap = int(settings.chebff.filter_cycles) + first_solve_extra_cycles()
        whole = frozenset(self._task_of[p] for p in self._solved if self._items[p][1] is None)
        self._inner.set_reported_cuts({}, relative, whole)
        values = self._expanded(self._inner.solve(potentials, filter_degree))
        growths = 0
        fixed = set(self._task_of.values()) - whole  # the tasks that have made only their fixed cycles
        while True:
            progress = self._inner.reported_progress()
            cuts = self._reported_cuts(values)
            pending: dict[int, tuple[int, float]] = {}
            for position in self._solved:
                k_index, sector = self._items[position]
                if sector is None:
                    continue
                task = self._task_of[position]
                norms, cycles = progress[task]
                if cycles >= cap:
                    continue
                levels = np.asarray(values[position], dtype=float)[: norms.size]
                cut = cuts[k_index]
                held = int(np.count_nonzero(levels <= cut))
                within = levels - norms <= cut
                within[:held] = True
                scale = max(float(np.max(np.abs(levels))), 1.0e-8)
                if np.any(norms[within] > relative * scale):
                    # from the fixed cycles: the levels the sector holds; later also the pairs that can reach the cut
                    pending[task] = (held, float("-inf") if task in fixed else cut)
            fixed.clear()
            if pending:
                self._inner.set_reported_cuts(pending, relative, whole)
                values = self._expanded(self._inner.solve(potentials, filter_degree, None, tasks=sorted(pending)))
                continue
            short = self._short_sectors(values)
            if not short:
                break
            growths += 1
            if growths > _ATTEMPTS:
                raise RuntimeError(
                    f"the sectors of the k-points did not hold the lowest levels after {_ATTEMPTS} attempts; "
                    f"{_SWITCH}=off solves each k-point whole"
                )
            self._inner.set_reported_cuts({}, relative, whole)
            values = self._grow(short, potentials, filter_degree, None, values)
            fixed.update(self._inner_task(self._position[item]) for item in short)
        self._reported_norms = {task: found[0] for task, found in self._inner.reported_progress().items()}
        self._whole_residuals = self._inner.residuals()
        self._inner.set_reported_cuts(None)
        return self._union(values)

    def reported_residuals(self) -> list[float]:
        """Largest residual of the levels each k-point reports, after :meth:`solve_reported`."""

        result = []
        for k_index, plan in enumerate(self._plans):
            if plan is None:
                result.append(float(self._whole_residuals[self._inner_task(self._position[(k_index, None)])]))
                continue
            owners, indices = self._kept[k_index]
            worst = 0.0
            for sector in range(plan.order):
                mask = owners == sector
                if np.any(mask):
                    norms = self._reported_norms[self._inner_task(self._position[(k_index, sector)])]
                    worst = max(worst, float(np.max(norms[indices[mask]])))
            result.append(worst)
        return result

    def _union(self, values) -> list[np.ndarray]:
        states = int(self.job.number_of_states)
        result = []
        self._returned = {position: int(np.size(values[position])) for position in range(len(self._items))}
        for k_index in range(self.job.n_tasks):
            plan = self._plans[k_index]
            if plan is None:
                self._kept[k_index] = None
                result.append(np.asarray(values[self._position[(k_index, None)]], dtype=float))
                continue
            parts = [np.asarray(values[self._position[(k_index, sector)]], dtype=float) for sector in range(plan.order)]
            levels = np.concatenate(parts)
            owners = np.concatenate([np.full(part.size, sector, dtype=np.int64) for sector, part in enumerate(parts)])
            indices = np.concatenate([np.arange(part.size, dtype=np.int64) for part in parts])
            order = np.argsort(levels, kind="stable")[:states]
            self._kept[k_index] = (owners[order], indices[order])
            result.append(levels[order])
        return result

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """The density of the occupations of the levels of every k-point (those of :meth:`solve`)."""

        inner: list[np.ndarray] = []
        for position, (k_index, sector) in enumerate(self._items):
            if sector is None:
                inner.append(np.asarray(occupations[k_index], dtype=np.float64))
            else:
                inner.append(np.zeros(self._returned[position], dtype=np.float64))
        for k_index, kept in self._kept.items():
            if kept is None:
                continue
            owners, indices = kept
            values = np.asarray(occupations[k_index], dtype=np.float64)
            for sector in range(self._plans[k_index].order):
                mask = owners == sector
                if np.any(mask):
                    inner[self._position[(k_index, sector)]][indices[mask]] = values[mask]
        if self._source:
            # an item of a conjugate pair that is not solved: its orbitals have the density of its partner's
            # (A u = e^{-iG.r} conj(u)), so its occupations go onto the partner's columns
            for position, source in self._source.items():
                inner[source] = inner[source] + inner[position]
            inner = [inner[position] for position in self._solved]
        return self._inner.accumulate(inner)

    def residuals(self) -> list[float]:
        inner = self._expanded(self._inner.residuals())
        result = []
        for k_index in range(self.job.n_tasks):
            plan = self._plans[k_index]
            items = [(k_index, None)] if plan is None else [(k_index, sector) for sector in range(plan.order)]
            norms = np.asarray([inner[self._position[item]] for item in items], dtype=float)
            finite = norms[np.isfinite(norms)]
            result.append(float(np.max(finite)) if finite.size else float("nan"))
        return result

    def vectors(self, task: int, columns: int | None = None) -> np.ndarray:
        """The kept orbitals of k-point ``task`` on the whole grid (an empty block where a rank holds none); its
        lowest ``columns`` ones where given."""

        plan = self._plans[task]
        if plan is None:
            inner_task = self._inner_task(self._position[(task, None)])
            if columns is None:
                return self._inner.vectors(inner_task)
            return self._inner.vectors(inner_task, columns=int(columns))
        owners, indices = self._kept[task]
        limited = columns is not None
        if limited:
            owners, indices = owners[: int(columns)], indices[: int(columns)]
        columns = np.zeros((self._grid_size, owners.size), dtype=np.complex128)
        for sector in range(plan.order):
            mask = owners == sector
            if not np.any(mask):
                continue
            position = self._position[(task, sector)]
            block = _host(
                self._inner.vectors(self._inner_task(position), columns=int(np.max(indices[mask])) + 1)
                if limited
                else self._inner.vectors(self._inner_task(position))
            )
            if block.ndim != 2 or block.shape[1] == 0:
                return np.zeros((0, 0), dtype=np.complex128)
            # the coefficients of the sector solved (T x of a sector in its real form, PARSEC_KPOINT_SECTOR_REAL)
            block = plan.coefficients(self._items[self._source.get(position, position)][1], block)
            if position in self._images:
                # a sector of a class not solved: the images P_g u (P_g conj(u)) of the orbitals of its source
                columns[:, mask] = plan.orbitals(sector, plan.member_block(sector, block[:, indices[mask]]))
                continue
            if position in self._source:
                # the partner of a conjugate pair: A u = e^{-iG.r} conj(u) of the orbitals of the sector solved
                solved = self._items[self._source[position]][1]
                whole = plan.orbitals(solved, block[:, indices[mask]])
                columns[:, mask] = plan.partner_orbitals(self.job.geometry["grid"], whole)
                continue
            columns[:, mask] = plan.orbitals(sector, block[:, indices[mask]])
        return columns

    def extras(self) -> list[Any]:
        """The job's extras (the nonlocal forces) of every k-point: the sums of its sectors (:meth:`_inner_extras`,
        computed by the inner solver where the orbitals are: a process, or the ranks of the MPI pools)."""

        if self.job.extras is None:
            raise NotImplementedError("the job computes no per-state extras")
        inner = self._inner.extras()
        found: list[Any] = [None] * self.job.n_tasks
        # the items solved (an item of a conjugate pair that is not solved is in its partner's extras)
        for task, position in enumerate(self._solved):
            k_index = self._items[position][0]
            value = np.asarray(_host(inner[task]), dtype=np.float64)
            found[k_index] = value if found[k_index] is None else found[k_index] + value
        return found

    def _inner_extras(self, position: int, vectors, xp=np, columns=None):
        """The extras of the states ``columns`` of task ``position`` (a sector): its kept states, ``U v`` on the
        whole grid a block of :data:`_FORCE_COLUMNS` at a time, with their places among the levels of the k-point
        (the occupations the job holds).  The sector solved for a conjugate pair (``PARSEC_KPOINT_TRIM_REAL``) stands
        for its partner as well: the partner's orbitals ``A u`` give the forces of ``u``, so its columns are taken
        once more at the partner's places."""

        position = self._solved[position]  # the inner task -> its item
        k_index, sector = self._items[position]
        count = int(vectors.shape[1])
        columns = np.arange(count, dtype=np.int64) if columns is None else np.asarray(columns, dtype=np.int64)
        if sector is None:
            return self.job.extras(k_index, vectors, xp=xp, columns=columns)
        if position in self._class_members:
            return self._class_extras(position, vectors, xp, columns)
        owners, indices = self._kept[k_index]
        keep_parts, place_parts = [], []
        held = [sector] + [self._items[other][1] for other, source in self._source.items() if source == position]
        for member in held:
            mine = np.flatnonzero(owners == member)
            size = int(max(int(indices[mine].max()) + 1 if mine.size else 0, int(columns.max()) + 1 if columns.size else 0))
            place = np.full(max(size, 1), -1, dtype=np.int64)
            place[indices[mine]] = mine
            wanted = place[columns] if columns.size else np.zeros(0, dtype=np.int64)
            keep = np.flatnonzero(wanted >= 0)
            keep_parts.append(keep)
            place_parts.append(wanted[keep])
        keep = np.concatenate(keep_parts) if len(keep_parts) > 1 else keep_parts[0]
        places = np.concatenate(place_parts) if len(place_parts) > 1 else place_parts[0]
        plan = self._plans[k_index]
        # PARSEC_KPOINT_FORCE_ORBITS: the forces of one atom per orbit of the little group, turned onto the others
        orbits = plan.force_orbits(self.job.geometry) if self._force_orbits else None
        options = {} if orbits is None else {"atoms": orbits[0]}
        total = None
        for c0 in range(0, int(keep.size), _FORCE_COLUMNS):
            chosen = keep[c0 : c0 + _FORCE_COLUMNS]
            block = plan.coefficients(sector, vectors[:, chosen if xp is np else xp.asarray(chosen)], xp)
            piece = np.asarray(
                _host(
                    self.job.extras(
                        k_index, plan.expand(sector, block, xp), xp=xp, columns=places[c0 : c0 + _FORCE_COLUMNS],
                        **options,
                    )
                ),
                dtype=np.float64,
            )
            total = piece if total is None else total + piece
        if total is None:
            total = np.zeros((len(self.job.geometry["atoms"]), 3), dtype=np.float64)
        elif orbits is not None:
            total = plan.complete_orbit_forces(total, orbits)
        return total

    def _class_extras(self, position: int, vectors, xp, columns):
        """:meth:`_inner_extras` of a sector solved for the sectors of its class (``PARSEC_KPOINT_SECTOR_CLASSES``):
        the states of each sector of the class at their places among the levels of the k-point, those of a sector not
        solved as the images of the states solved (:meth:`KPointSectors.member_block`), expanded in that sector."""

        k_index, sector = self._items[position]
        owners, indices = self._kept[k_index]
        plan = self._plans[k_index]
        orbits = plan.force_orbits(self.job.geometry) if self._force_orbits else None
        options = {} if orbits is None else {"atoms": orbits[0]}
        total = None
        for member in [sector] + self._class_members[position]:
            mine = np.flatnonzero(owners == member)
            size = int(max(int(indices[mine].max()) + 1 if mine.size else 0, int(columns.max()) + 1 if columns.size else 0))
            place = np.full(max(size, 1), -1, dtype=np.int64)
            place[indices[mine]] = mine
            wanted = place[columns] if columns.size else np.zeros(0, dtype=np.int64)
            keep = np.flatnonzero(wanted >= 0)
            places = wanted[keep]
            for c0 in range(0, int(keep.size), _FORCE_COLUMNS):
                chosen = keep[c0 : c0 + _FORCE_COLUMNS]
                block = plan.coefficients(sector, vectors[:, chosen if xp is np else xp.asarray(chosen)], xp)
                if member != sector:
                    block = plan.member_block(member, block, xp)
                piece = np.asarray(
                    _host(
                        self.job.extras(
                            k_index, plan.expand(member, block, xp), xp=xp, columns=places[c0 : c0 + _FORCE_COLUMNS],
                            **options,
                        )
                    ),
                    dtype=np.float64,
                )
                total = piece if total is None else total + piece
        if total is None:
            total = np.zeros((len(self.job.geometry["atoms"]), 3), dtype=np.float64)
        elif orbits is not None:
            total = plan.complete_orbit_forces(total, orbits)
        return total

    # ----------------------------------------------------------- the symmetry guard
    def guarded_tasks(self) -> list[int]:
        """The tasks whose sectors assume an invariance of the potential that the density is not symmetrized over
        (a single Gamma point, a full mesh, the translations of a supercell): those the symmetry guard tests."""

        return [task for task, plan in enumerate(self._plans) if plan is not None and getattr(plan, "assumed", False)]

    def _task_weight(self, task: int) -> float:
        geometry = self.job.geometry
        k_of_task = np.asarray(geometry.get("k_of_task", np.arange(len(geometry["weights"]))), dtype=np.int64)
        return float(np.asarray(geometry["weights"], dtype=np.float64)[int(k_of_task[task])])

    def _guard_levels(self, task: int, levels, occupations, capacity: float):
        owners, _ = self._kept[task]
        values = np.asarray(levels[task], dtype=np.float64)
        filled = np.asarray(occupations[task], dtype=np.float64) / float(capacity)
        return values, filled, np.asarray(owners, dtype=np.int64)

    def start_spectrum_report(self, levels, occupations, *, capacity: float = 1.0):
        """Test 1 of the guard: the partly filled multiplets of the levels of the start Hamiltonian, ``None`` where
        there are none.

        ``levels`` and ``occupations`` are those of every task (the levels of :meth:`solve`), ``capacity`` the
        occupation of a full level (1 for a spin channel).  As for the sectors of a cluster
        (:func:`.spin_sectors.partly_filled_multiplets`): a multiplet of :func:`.spin_sectors.sector_multiplets`
        (levels in two or more sectors of a guarded task) of ``d`` levels holding ``n`` electrons is partly filled
        when ``w_k min(n, d - n)`` exceeds :data:`.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE`, ``w_k`` the
        weight of its k-point (the electrons it adds to the density; 1 at a single Gamma point).
        """

        from .spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, FractionalOccupations

        largest = 0.0
        lines: list[str] = []
        for part, line in self._start_multiplet_parts(levels, occupations, capacity):
            if part <= SPIN_SECTOR_OCCUPATION_TOLERANCE:
                continue
            largest = max(largest, part)
            lines.append(line())
        if not lines:
            return None
        return FractionalOccupations(
            largest=largest,
            tolerance=float(SPIN_SECTOR_OCCUPATION_TOLERANCE),
            lines=tuple(lines),
            reason="a partly filled multiplet",
            measure="largest w_k min(n, d - n) of d levels holding n electrons",
        )

    def _start_multiplet_parts(self, levels, occupations, capacity: float):
        """``(w_k min(n, d - n), line)`` of every multiplet of :meth:`start_spectrum_report`, below its tolerance as
        well; ``line()`` describes it."""

        from .spin_sectors import START_MULTIPLET_WIDTH, sector_multiplets

        for task in self.guarded_tasks():
            values, filled, sectors = self._guard_levels(task, levels, occupations, capacity)
            weight = self._task_weight(task)
            for group in sector_multiplets(values, sectors):
                electrons = float(np.sum(filled[group]))
                part = weight * min(electrons, len(group) - electrons)

                def line(task=task, weight=weight, group=group, electrons=electrons, values=values, sectors=sectors):
                    text = " ".join(f"{values[member] * 13.605693122994:.4f}" for member in group)
                    labels = " ".join(str(int(sectors[member])) for member in group)
                    return (
                        f"task {task} (k-point weight {weight:.4g}): {len(group)} levels within "
                        f"{START_MULTIPLET_WIDTH:g} Ry at {text} eV hold {electrons:.4f} electrons (sectors {labels})"
                    )

                yield part, line

    def start_spectrum_measure(self, levels, occupations, *, capacity: float = 1.0) -> float:
        """The value test 1 compares with its tolerance: the largest ``w_k min(n, d - n)`` of the multiplets of
        :meth:`start_spectrum_report`, 0 where there are none (for the report of the guard that passed)."""

        return max((part for part, _ in self._start_multiplet_parts(levels, occupations, capacity)), default=0.0)

    def density_symmetry_measure(self, densities, volume_element: float) -> float:
        """Test 2 of the guard: the largest ``int |rho(g r) - rho(r)|`` (electrons) of the channel ``densities``
        (whole grid) over the operations of :class:`SectorGuardGroup` (0 where there are none)."""

        return max((value for _, value in self._asymmetries(densities, volume_element)), default=0.0)

    def _asymmetries(self, densities, volume_element: float) -> list[tuple[str, float]]:
        guarded = self.guarded_tasks()
        if not guarded:
            return []
        group = self._plans[guarded[0]].guard_group
        if group is None:
            return []
        found = []
        for index, rows in group.images():
            for channel, density in enumerate(densities):
                values = np.asarray(density, dtype=np.float64)
                deviation = float(volume_element * np.sum(np.abs(values[rows] - values)))
                found.append((f"channel {channel + 1}, operation {group.operation_text(index)}", deviation))
        return found

    def density_symmetry_report(self, densities, volume_element: float, tolerance: float | None = None):
        """Test 2 as a report (:class:`.spin_sectors.FractionalOccupations`), ``None`` below ``tolerance``
        (:data:`.spin_sectors.DENSITY_SYMMETRY_TOLERANCE`)."""

        from .spin_sectors import DENSITY_SYMMETRY_TOLERANCE, FractionalOccupations

        tolerance = DENSITY_SYMMETRY_TOLERANCE if tolerance is None else float(tolerance)
        found = self._asymmetries(densities, volume_element)
        largest = max((value for _, value in found), default=0.0)
        if largest < tolerance:
            return None
        return FractionalOccupations(
            largest=largest,
            tolerance=tolerance,
            lines=tuple(
                f"{label}: the density differs by {value:.4f} electrons from its image" for label, value in found
                if value >= tolerance
            ),
            reason="a channel density that breaks the symmetry the k-point sectors assume",
            measure="largest int |rho(g r) - rho(r)| (electrons)",
        )

    def fractional_report(self, levels, occupations, *, capacity: float = 1.0):
        """Test 3 of the guard (Fermi-Dirac occupations, at the end of a loop): a level of a multiplet whose levels
        lie in two or more sectors of a guarded task holds ``w_k min(f, 1 - f)`` of at least
        :data:`.spin_sectors.SPIN_SECTOR_OCCUPATION_TOLERANCE` (:func:`.spin_sectors.fractional_occupations`), else
        ``None``."""

        from .spin_sectors import SPIN_SECTOR_OCCUPATION_TOLERANCE, FractionalOccupations

        largest = 0.0
        lines: list[str] = []
        for part, line in self._fractional_parts(levels, occupations, capacity):
            if part < SPIN_SECTOR_OCCUPATION_TOLERANCE:
                continue
            largest = max(largest, part)
            lines.append(line())
        if not lines:
            return None
        return FractionalOccupations(
            largest=largest, tolerance=float(SPIN_SECTOR_OCCUPATION_TOLERANCE), lines=tuple(lines),
            measure="largest w_k min(f, 1-f)",
        )

    def _fractional_parts(self, levels, occupations, capacity: float):
        """``(w_k max min(f, 1 - f), line)`` of every multiplet of :meth:`fractional_report`, below its tolerance as
        well; ``line()`` describes it."""

        from .spin_sectors import MULTIPLET_WIDTH, _multiplets, sector_multiplets

        for task in self.guarded_tasks():
            values, filled, sectors = self._guard_levels(task, levels, occupations, capacity)
            weight = self._task_weight(task)
            fractional = weight * np.minimum(filled, 1.0 - filled)
            groups = [group for group in _multiplets(values, MULTIPLET_WIDTH) if np.unique(sectors[group]).size >= 2]
            exact = {tuple(group) for group in groups}
            groups += [group for group in sector_multiplets(values, sectors) if tuple(group) not in exact]
            for group in groups:
                part = float(np.max(fractional[group]))

                def line(task=task, weight=weight, group=group, values=values, filled=filled, sectors=sectors):
                    text = " ".join(f"{values[member] * 13.605693122994:.4f}" for member in group)
                    shares = " ".join(f"{filled[member]:.4f}" for member in group)
                    labels = " ".join(str(int(sectors[member])) for member in group)
                    return (
                        f"task {task} (k-point weight {weight:.4g}): {len(group)} level(s) at {text} eV "
                        f"(occupations {shares}; sectors {labels})"
                    )

                yield part, line

    def fractional_measure(self, levels, occupations, *, capacity: float = 1.0) -> float:
        """The value test 3 compares with its tolerance: the largest ``w_k min(f, 1 - f)`` of the multiplets of
        :meth:`fractional_report`, 0 where there are none (for the report of the guard that passed)."""

        return max((part for part, _ in self._fractional_parts(levels, occupations, capacity)), default=0.0)

    def _layout(self) -> list[tuple]:
        layout = [(int(k), -1 if sector is None else int(sector), self._rows((k, sector))) for k, sector in self._items]
        if self._source:  # the conjugate pairs (PARSEC_KPOINT_TRIM_REAL): which items their partners solve
            layout = [entry + (self._source.get(position, position),) for position, entry in enumerate(layout)]
        if any(plan is not None and plan.real_forms for plan in self._plans):
            # the sectors in their real forms (PARSEC_KPOINT_SECTOR_REAL): their states are real coordinates
            layout = [
                entry + (int(sector is not None and bool(self._plans[k].real_forms) and sector in self._plans[k].real_forms),)
                for entry, (k, sector) in zip(layout, self._items)
            ]
        return layout

    def export_states(self) -> dict:
        """The Ritz states of the sectors with their layout and numbers of states (for the next loop of the same
        structure or of a moved one: :meth:`import_states`)."""

        counts = [self._states(item) for item in self._items]
        return {_STATES_KEY: {"layout": self._layout(), "counts": counts, "states": self._inner.export_states()}}

    def import_states(self, states: dict) -> None:
        """Start from the states of :meth:`export_states`: those of sectors of the same layout (k-points, sectors,
        rows), with their numbers of states; anything else (the states of a solver without sectors, or of other
        sectors) is not taken, and the sectors start afresh."""

        carried = states.get(_STATES_KEY) if isinstance(states, dict) else None
        if not isinstance(carried, dict) or [tuple(item) for item in carried.get("layout", ())] != self._layout():
            return
        counts = [int(count) for count in carried["counts"]]
        if counts != [self._states(item) for item in self._items]:
            for (k_index, sector), count in zip(self._items, counts):
                if sector is not None:
                    self._plans[k_index].counts[sector] = count
            self._inner.close()
            self._inner = None
            self._inner = self._new_inner()
        self._inner.import_states(carried["states"])

    def device_vectors(self, task: int) -> np.ndarray:
        """The kept orbitals of k-point ``task`` on the whole grid (host), for a driver that asks for them where the
        solver keeps them (the Fock loop of a hybrid)."""

        return self.vectors(task)

    def stats(self) -> dict[str, float]:
        inner = getattr(self._inner, "stats", None)
        stats = dict(inner()) if inner is not None else {}
        rows, states = self._grid_size, int(self.job.number_of_states)
        split = [plan for plan in self._plans if plan is not None]
        whole_work = float(rows) * states * states * self.job.n_tasks
        work = sum(self._rows(self._items[position]) * self._states(self._items[position]) ** 2 for position in self._solved)
        stats.update(
            {
                "kpoint_sector_tasks": float(len(self._items)),
                "kpoint_sector_kpoints": float(len(split)),
                "kpoint_sector_largest_group": float(max(plan.order for plan in split)),
                "kpoint_sector_work_fraction": float(work) / whole_work,
                "kpoint_sector_regrowths": float(self._regrowths),
                "kpoint_sector_regrown_tasks": float(self._regrown_tasks),
                "kpoint_sector_rebuilds": float(self._rebuilds),
                "kpoint_sector_refit_states": float(self._refit_states),
                "kpoint_sector_refit_sectors": float(len(self._trimmed)),
            }
        )
        pairs = len(self._source) - len(self._images)
        if pairs:  # the sectors of conjugate pairs not solved (PARSEC_KPOINT_TRIM_REAL)
            stats["kpoint_sector_pairs"] = float(pairs)
        if self._images:  # the sectors of classes not solved (PARSEC_KPOINT_SECTOR_CLASSES)
            stats["kpoint_sector_images"] = float(len(self._images))
        real = sum(
            1 for position in self._solved
            if self._items[position][1] is not None and self._plans[self._items[position][0]].real_forms
            and self._items[position][1] in self._plans[self._items[position][0]].real_forms
        )
        if real:  # the sectors solved in their real forms (PARSEC_KPOINT_SECTOR_REAL)
            stats["kpoint_sector_real_forms"] = float(real)
        return stats

    def describe_sectors(self) -> str:
        orders = [1 if plan is None else plan.order for plan in self._plans]
        return f"k-point sectors: {len(self._items)} tasks for {self.job.n_tasks} k-points (sectors per k-point {orders})"

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()

    def __enter__(self) -> "SectorKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __getattr__(self, name: str):
        # what the drivers read of the solver they made (the projector builder and density symmetrizer of the MPI
        # pools, its backend, ...) is the inner solver's
        if name.startswith("__") or name in {"_inner", "job", "_plans", "_items"}:
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


def sector_solver(job: KPointJob, make_solver: Callable[..., Any], *, notes: list | None = None, **options):
    """A :class:`SectorKPointSolver` for ``job`` where its k-points split, else ``None``.

    A job with extras takes them where they sum over the states (the forces of the scalar SCF,
    ``extras_by_columns``): each sector's part from its kept states, in one process or on the ranks of the MPI pools.
    ``notes`` (a list) receives the line of the report on the cell group of the sectors, or why there are none
    (:func:`plan_sectors`).
    """

    from ..Hamiltonian.sector_bloch_stencil import sector_stencil_setting

    sector_stencil_setting()  # read here: a bad value stops the run before its first solve
    # (in the order of before: a reason found first leaves the switches after it unread)
    if getattr(job, "geometry", None) is None:
        refusal = (
            "skipped: this k-point loop takes no sectors (the spinors of a spin-orbit or non-collinear run, LDA+U "
            "without PARSEC_KPOINT_SECTOR_LDAU=on or spin, a meta-GGA, fixed "
            "occupations, refined levels under PARSEC_KPOINT_SECTOR_LEVELS=off, or the loop after the symmetry guard "
            "left the sectors)"
        )
    elif job.extras is not None and not job.extras_by_columns:
        refusal = "skipped: the extras of this k-point job (spinor or non-collinear forces) need whole k-points"
    elif job.states_per_task is not None:
        refusal = "skipped: the k-point job holds its own count of states per task"
    elif kpoint_sectors_setting() == "off":
        refusal = f"disabled by option ({_SWITCH}=off; one task per k-point)"
    else:
        refusal = None
    if refusal is not None:
        if notes is not None:
            notes.append(refusal)
        return None
    plans = plan_sectors(job, notes)
    if plans is None:
        return None
    # the gathered stencils of the kinetic terms only where the operators go to the device code
    device = sector_stencil_setting() == "auto" and sector_operators_on_device(options.get("backend"))
    for plan in plans:
        if plan is not None:
            plan.device_operators = device
            if not device:
                # the real forms are solved by the device code (PARSEC_KPOINT_SECTOR_REAL): the CPU route keeps the
                # complex sectors
                plan.real_forms = None
    return SectorKPointSolver(job, plans, make_solver, options)


def sector_operators_on_device(backend: str | None = None) -> bool:
    """Whether the operators of the sectors of a k-point go to the device code: the k-point backend of the solver
    (``backend``, else ``PARSEC_KPOINT_BACKEND`` / ``PARSEC_DEVICE``) resolves to a device, CuPy or ``numpy-device``
    (the device code on the host, the tests' double of CuPy).  On the CPU route (the worker threads of this process
    or the MPI pools on CPU ranks) the Hamiltonians of the sectors are applied on the host.  Every rank takes the same
    answer: the backend follows the environment of the job."""

    from .kpoint_parallel import resolve_backend

    return resolve_backend(backend) is not None


__all__ = [
    "CellGroup",
    "KPointSectors",
    "LittleGroupReduction",
    "PotentialInvariance",
    "SectorBlochHamiltonian",
    "SectorGuardGroup",
    "SectorKPointSolver",
    "check_kpoint_sector_switches",
    "kpoint_sector_warm_growth_setting",
    "kpoint_sector_ldau_setting",
    "kpoint_sector_atom_tolerance",
    "kpoint_sector_tolerance_setting",
    "kpoint_sectors_setting",
    "hybrid_kpoint_sectors_setting",
    "ANY_SIZE_KEY",
    "kpoint_force_orbits_setting",
    "generated_subgroup",
    "largest_commuting_involutions",
    "little_group_reduction",
    "plan_sectors",
    "carried_state",
    "refitted_count",
    "sector_guard_fock_setting",
    "sector_guard_setting",
    "sector_levels_setting",
    "sector_margin",
    "sector_operators_on_device",
    "sector_refit_setting",
    "sector_solver",
    "trimmed_state",
]
