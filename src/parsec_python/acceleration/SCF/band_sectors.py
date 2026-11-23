"""The band k-points of a Gamma cell solved in the sectors of their little groups (``PARSEC_BAND_KPOINT_SECTORS``).

The band structure after the SCF of a Gamma bulk cell (:mod:`..Output.post_scf`) solves every k-point of the path
from scratch with the converged potential (:func:`parsec_python.SCF.bands.solve_kpoint_bands`).  Each k-point is a
complex eigenproblem on the whole grid: ``rows x States_Num`` complex numbers per basis, several times over in the
filter, which is beyond one device from about 6^3 diamond (1.26 million rows, 3,462 states: 70 GB a basis) even when
the SCF itself ran in the sectors of the point group of the cell.

The converged potential of such an SCF is invariant under the operations ``r -> R r + t`` its density was averaged
over: the full group ``G`` of weighted sectors, the group ``H`` of abelian ones (:class:`BandSymmetry`).  The Bloch
Hamiltonian of a band k-point then commutes with the operations of ``G`` that fix ``k`` (``R k = k``), its little
group, and the machinery of the k-point SCF splits it into the sectors of the commuting involutions of that group
(:mod:`.kpoint_sectors`: :func:`~.kpoint_sectors.plan_sectors`, :class:`~.kpoint_sectors.SectorKPointSolver`) with a
fraction of the rows and of the states each.  Only the operations of ``G`` are admitted (:class:`_BandInvariance`):
the k-point SCF also admits operations that map its mesh onto itself, which says nothing about a band path.

The k-points are solved a few at a time (as many as the devices of the k-point workers) by one sector solver each,
which is closed before the next, so the memory is that of a few k-points in their sectors.  A k-point whose basis
stays below the floor of the k-point sectors (``2^24`` rows times states) or whose sectors would not halve the work
is solved whole by the same solver.  ``PARSEC_BAND_KPOINT_SECTORS=off`` (or ``PARSEC_KPOINT_SECTORS=off``) keeps the
solve of every k-point on the whole grid, as before.

A converged first solve tests the levels its k-point reports, not the margin states of the sectors above them
(``PARSEC_BAND_REPORTED_RESIDUALS``, :meth:`.kpoint_sectors.SectorKPointSolver.solve_reported`).

Under ``--mpi`` (``PARSEC_MPI_ACCELERATED_BANDS``) every rank holds the converged potential of the accelerated SCF;
the chunks of k-points are dealt to the ranks in turn, each solved by the solver of one process on the GPUs of its
rank, and the levels and residuals are gathered on every rank: the bands of one process, in a share of its time.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

_SWITCH = "PARSEC_BAND_KPOINT_SECTORS"
# what the last call of solve_band_sectors did (last_band_sector_record), for the log of the band stage
_LAST_RECORD: dict = {}


def last_band_sector_record() -> dict:
    """What the last :func:`solve_band_sectors` did: ``kpoints``, ``split`` (those solved in sectors), ``largest``
    (the most sectors of one), ``group`` (the operations of the potential), ``residuals_of_reported`` (whether every
    residual is that of the levels its k-point reports) and ``images`` (the sectors not solved, images of others under
    ``PARSEC_KPOINT_SECTOR_CLASSES=auto``, where there are any); empty where it kept the path whole."""

    return dict(_LAST_RECORD)


def band_kpoint_sectors_setting() -> str:
    """``PARSEC_BAND_KPOINT_SECTORS``: ``auto`` (default) solves the band k-points of a Gamma cell whose SCF ran in
    the sectors of its point group in the sectors of their little groups; ``off`` solves each on the whole grid, bit
    for bit as before."""

    value = os.environ.get(_SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{_SWITCH} must be auto or off, not {value!r}")
    return value


@dataclass(frozen=True)
class BandSymmetry:
    """The operations ``r -> R r + t`` (``t`` in grid steps) the converged potential of an SCF is invariant under.

    ``keys`` holds ``(R, t mod points per axis)`` of each as bytes; ``count`` their number.
    """

    keys: frozenset
    count: int
    points_per_axis: tuple[int, int, int]

    @staticmethod
    def key(matrix, shift, counts) -> tuple[bytes, bytes]:
        matrix = np.asarray(matrix, dtype=np.int8).reshape(3, 3)
        shift = np.mod(np.asarray(shift, dtype=np.int64).reshape(3), np.asarray(counts, dtype=np.int64))
        return matrix.tobytes(), shift.tobytes()

    @classmethod
    def of_reduction(cls, reduction, grid) -> "BandSymmetry | None":
        """The operations of the density of an SCF in the sectors of ``reduction``: the full group of a weighted
        reduction (its density is averaged over it), the group of the sectors otherwise (``None``: no group)."""

        if reduction is None or int(getattr(reduction, "group_order", 1)) < 2:
            return None
        operations = getattr(reduction, "full_operations", None)
        shifts = getattr(reduction, "full_translations", None)
        if operations is None:
            operations = getattr(reduction, "operations", None)
            shifts = getattr(reduction, "translations", None)
        if operations is None:
            signs = getattr(reduction, "signs", None)
            if signs is None:
                return None
            operations = [np.diag(row) for row in np.asarray(signs)]
            shifts = None
        operations = np.asarray(operations)
        if shifts is None:
            shifts = np.zeros((operations.shape[0], 3), dtype=np.int64)
        counts = tuple(int(value) for value in grid.points_per_axis)
        keys = frozenset(cls.key(matrix, shift, counts) for matrix, shift in zip(operations, np.asarray(shifts)))
        return cls(keys=keys, count=len(keys), points_per_axis=counts)

    def holds(self, matrix, shift) -> bool:
        return self.key(matrix, shift, self.points_per_axis) in self.keys


def band_symmetry_of(system) -> BandSymmetry | None:
    """The :class:`BandSymmetry` of an accelerated Gamma-cell system whose SCF ran in sectors (``None`` otherwise)."""

    sectors = getattr(getattr(system, "backend", None), "symmetry_eigensolver", None)
    reduction = getattr(getattr(sectors, "decomposition", None), "reduction", None)
    grid = getattr(getattr(system, "reference", system), "grid", None)
    if reduction is None or grid is None:
        return None
    if getattr(reduction, "index_frame", None) is not None:
        # sectors of an oblique cell detected in the frame of its grid indices (PARSEC_OBLIQUE_SECTORS): their
        # operations act on the indices, which the k-point sectors of the bands (Cartesian, CellGroup) do not take
        return None
    try:
        return BandSymmetry.of_reduction(reduction, grid)
    except (AttributeError, TypeError, ValueError):
        return None


def _band_invariance_class():
    from .kpoint_sectors import PotentialInvariance

    class _BandInvariance(PotentialInvariance):
        """:class:`~.kpoint_sectors.PotentialInvariance` that admits the operations of a :class:`BandSymmetry` only."""

        def __init__(self, cell, symmetry: BandSymmetry, k_points, weights, lattice) -> None:
            super().__init__(cell, None, k_points, weights, lattice)
            self.band_symmetry = symmetry

        def _decide(self, index: int) -> bool:
            return self.in_density_group(index)

        def _in_density_group(self, index: int) -> bool:
            return self.band_symmetry.holds(self.cell.matrices[index], self.cell.shifts[index])

        def symmetrized_over(self, matrix, shift) -> bool:
            return self.band_symmetry.holds(matrix, shift)

        def admits(self, matrix, shift) -> bool:
            return self.band_symmetry.holds(matrix, shift)

    return _BandInvariance


def solve_band_sectors(
    system,
    k_points: np.ndarray,
    *,
    potential: np.ndarray,
    number_of_states: int,
    eigval_settings,
    symmetry: BandSymmetry,
    make_hamiltonian: Callable[[np.ndarray, np.ndarray], object],
    callback=None,
    workers=None,
    backend=None,
    keep_whole: bool = True,
):
    """Eigenvalues ``(n_k, number_of_states)``, largest residuals ``(n_k,)`` of the band k-points in sectors (of the
    reported levels under ``PARSEC_BAND_REPORTED_RESIDUALS=auto``, else of every state of the sectors) and the list of
    the k-points left to the caller.

    ``make_hamiltonian(k_point, potential)`` is the scalar Bloch Hamiltonian of a k-point
    (:func:`parsec_python.SCF.bands.solve_kpoint_bands`), ``eigval_settings`` the settings of its first solve (with
    ``Bands_FF_MaxIter``).  A k-point that does not split is solved whole by the same solvers with
    ``eigval_settings`` where ``keep_whole``, else left to the caller (its whole-grid route, with the buffer of
    ``PARSEC_BAND_BUFFER``): its levels stay zero and its residual NaN.  ``None`` where no k-point of the path splits
    (the caller keeps its route).
    """

    from parsec_python.Eigensolvers.eigval import first_solve_converges
    from parsec_python.SCF.bands import band_reported_residuals_setting
    from parsec_python.SCF.kpoint_solver import KPointJob

    from .kpoint_parallel import PooledKPointSolver, device_worker_count, resolve_backend
    from .kpoint_sectors import (
        _MINIMUM_ENTRIES,
        KPointSectors,
        SectorKPointSolver,
        kpoint_sectors_setting,
        plan_sectors,
    )

    _LAST_RECORD.clear()
    if band_kpoint_sectors_setting() == "off" or kpoint_sectors_setting() == "off" or symmetry is None:
        return None
    grid = system.grid
    if int(grid.size) * int(number_of_states) < _MINIMUM_ENTRIES:
        return None
    k_points = np.asarray(k_points, dtype=np.float64).reshape(-1, 3)
    n_k = int(k_points.shape[0])
    lattice = np.asarray(system.input.periodic_cell.lattice_vectors, dtype=np.float64)
    atoms = tuple(getattr(system, "cell_atoms", None) or system.atoms)
    invariance_class = _band_invariance_class()
    potential = np.asarray(potential, dtype=np.float64)

    def settings(degree, tolerance=None):
        # every band k-point is one first solve with the settings of the band stage (Bands_FF_MaxIter)
        return eigval_settings

    def job_of(points):
        # the Hamiltonian of a k-point is made once for its sectors (its Bloch projectors), by the first worker that
        # asks for it; the others wait for it
        made: dict[int, dict] = {}
        lock = threading.Lock()

        def hamiltonian(task, potentials):
            with lock:
                entry = made.setdefault(int(task), {"lock": threading.Lock(), "value": None})
            with entry["lock"]:
                if entry["value"] is None:
                    entry["value"] = make_hamiltonian(points[int(task)], potentials["V"])
                return entry["value"]

        return KPointJob(
            int(points.shape[0]),
            int(number_of_states),
            settings,
            hamiltonian,
            lambda task, vectors, occupations, xp=np: {},
            geometry={
                "grid": grid,
                "atoms": atoms,
                "k_points": points,
                "weights": np.ones(points.shape[0], dtype=np.float64),
                "lattice": lattice,
                "density_symmetry": None,
                "invariance": lambda cell, k, w, lat: invariance_class(cell, symmetry, k, w, lat),
                "cache": {},
                "density_part": None,
                "exchange_cross_scale": None,
            },
        )

    # The plans of the whole path, once: the group of the cell, the little groups and the decompositions of the
    # distinct ones (no operator).  Each part of the path takes fresh sectors of them, whose static operators go with
    # its solver.
    plans = plan_sectors(job_of(k_points))
    if plans is None:
        return None
    split = [found for found in plans if found is not None]
    _LAST_RECORD.update(
        kpoints=n_k, split=len(split), largest=max(found.order for found in split), group=symmetry.count
    )
    images = sum(len(found.classes) for found in split if found.classes)
    if images:  # the sectors not solved, images of others (PARSEC_KPOINT_SECTOR_CLASSES)
        _LAST_RECORD["images"] = images
    plan = resolve_backend(backend)
    # the k-points this route solves (an unsplit one goes to the caller's whole-grid route unless keep_whole)
    todo = [k for k in range(n_k) if keep_whole or plans[k] is not None]
    unsplit = [k for k in range(n_k) if k not in set(todo)]
    chunk = len(todo) if plan is None else max(1, device_worker_count(plan, len(todo)))
    eigenvalues = np.zeros((n_k, int(number_of_states)))
    residuals = np.full(n_k, np.nan)
    degree = int(system.input.eigensolver.filter_degree)
    options = {"problem_size": int(grid.size), "backend": backend, "workers": workers}
    # PARSEC_BAND_REPORTED_RESIDUALS: the residual test of a converged first solve covers the levels the k-point
    # reports, not the margin of its sectors above them
    reported_test = band_reported_residuals_setting() == "auto" and first_solve_converges(eigval_settings)
    of_reported = True  # no k-point of the path has a residual over the margin states of its sectors
    # Several MPI ranks (the accelerated SCF of a Gamma cell under --mpi, PARSEC_MPI_ACCELERATED_BANDS): every rank
    # holds the converged potential, so the chunks of k-points are dealt to the ranks, each solved by the solver of one
    # process on the devices of its rank (not by the pools of ranks, whose first solves cannot test the reported
    # levels; nor with --mpi on one rank), and every rank gets every level.  The ranks agree on one chunk length (the
    # smallest of theirs: ranks may see different numbers of GPUs), so that they deal the same partition.
    context = _mpi_context()
    ranks = 1 if context is None else int(context.size)
    rank = 0 if context is None else int(context.rank)
    if ranks > 1:
        chunk = int(min(int(value) for value in context.comm.allgather(int(chunk))))
    mine: list[int] = []
    for number, start in enumerate(range(0, len(todo), chunk)):
        if number % ranks != rank:
            continue
        indices = todo[start : start + chunk]
        points = k_points[indices]
        job = job_of(points)
        parts = []
        for offset, k_index in enumerate(indices):
            found = plans[k_index]
            if found is None:
                parts.append(None)
                continue
            fresh = KPointSectors(offset, points[offset], found.decomposition, list(found.counts))
            fresh.assumed = False
            fresh.guard_group = found.guard_group
            # the sectors that are images of others (PARSEC_KPOINT_SECTOR_CLASSES; None by default)
            fresh.classes = found.classes
            parts.append(fresh)
        if any(part is not None for part in parts):
            solver = SectorKPointSolver(job, parts, PooledKPointSolver, options)
        else:
            solver = PooledKPointSolver(job, **options)
        with solver:
            in_sectors = isinstance(solver, SectorKPointSolver)
            if reported_test and in_sectors and solver.reported_test_available(degree):
                values = solver.solve_reported({"V": potential}, degree)
                norms = solver.reported_residuals()
            else:
                of_reported = of_reported and not in_sectors
                values = solver.solve({"V": potential}, degree)
                norms = solver.residuals()
        for offset, k_index in enumerate(indices):
            levels = np.asarray(values[offset], dtype=np.float64)[: int(number_of_states)]
            eigenvalues[k_index, : levels.size] = levels
            residuals[k_index] = float(norms[offset]) if math.isfinite(float(norms[offset])) else float("nan")
            mine.append(k_index)
            if callback is not None and ranks == 1:
                callback(k_index, n_k, eigenvalues[k_index][None, :], residuals[k_index], 1)
    if ranks > 1:
        found = {k_index: (eigenvalues[k_index].copy(), float(residuals[k_index])) for k_index in mine}
        dealt: list[int] = []
        for part, flag in context.comm.allgather((found, of_reported)):
            for k_index, (levels, norm) in part.items():
                eigenvalues[k_index] = levels
                residuals[k_index] = norm
                dealt.append(int(k_index))
            of_reported = of_reported and bool(flag)
        if sorted(dealt) != sorted(todo):
            missing = sorted(set(todo) - set(dealt))
            twice = sorted({k_index for k_index in dealt if dealt.count(k_index) > 1})
            raise RuntimeError(
                f"the MPI ranks did not solve every band k-point exactly once (missing {missing}, solved more than "
                f"once {twice})"
            )
        if callback is not None:
            for k_index in todo:
                callback(k_index, n_k, eigenvalues[k_index][None, :], residuals[k_index], 1)
    _LAST_RECORD["residuals_of_reported"] = of_reported
    return eigenvalues, residuals, unsplit


def _mpi_context():
    """The MPI context of the run (``None`` in one process)."""

    from ..Parallel import mpi

    context = mpi.context()
    return None if context is None or int(context.size) < 2 else context


__all__ = [
    "BandSymmetry",
    "band_kpoint_sectors_setting",
    "band_symmetry_of",
    "last_band_sector_record",
    "solve_band_sectors",
]
