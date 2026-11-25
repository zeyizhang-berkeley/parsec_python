"""Parallel solution of the per-k-point eigenproblems of a periodic SCF.

The performance counterpart of
:class:`parsec_python.SCF.kpoint_solver.SerialKPointSolver`, which is the
readable definition.  The diagonalizations of the k-points are independent, so
they are spread over workers (:mod:`parsec_python.acceleration.Parallel`):
threads on the CPU, one thread per CUDA device on a GPU.  A worker keeps what
belongs to its tasks: the Bloch-phase projectors, the eigenvectors and the Ritz
subspace the next SCF step starts from.  The driver sends the effective
potentials, receives the eigenvalues, sends back the occupations and receives
the summed partial densities.  The eigenvectors never leave the worker.

On a device the worker keeps the Ritz subspace of every k-point *on the device* and runs the
SUBSPACE steps of the later SCF iterations there (:func:`..Eigensolvers.chebff_spinor.solve_spinor_eigval`:
filter, QR and Rayleigh-Ritz without leaving the GPU).  The first solve is CHEBFF on the device; for ``Eigensolver: chebdav`` it is CHEBFF followed by
more CHEBFF cycles until CHEBDAV's residual test is met (the device counterpart of
CHEBDAV; ``PARSEC_KPOINT_FIRST_SOLVE=chebdav`` runs the host CHEBDAV with the device filter instead and
moves its result to the device).  The eigenvectors come to the host only to form the density.  When the subspaces of all
the worker's k-points would not fit in device memory they are kept on the host between solves
(``PARSEC_KPOINT_STATE_STORAGE``).  ``PARSEC_KPOINT_SOLVER=host`` restores the former route: the host
eigensolver with only the Chebyshev filter on the device.

With one worker the inline backend runs this in the calling process.  Parallel
results equal the serial ones to round-off (the eigenvalues exactly: each task
is the same computation; the densities differ only by the order of the sum over
tasks).  Under ``PARSEC_ORDERED_SUMS=auto`` the density parts of the workers are
added in the order of the workers, not in the order they finish, and the sparse
products of the device operators sum in a fixed order
(:class:`..Hamiltonian.complex_csr_device.OrderedCSR`): an input on the same
devices gives the same bits in every run.  Importing
:mod:`parsec_python.acceleration` installs this solver and :func:`map_tasks` as
the parallel backend of the reference drivers
(:func:`parsec_python.SCF.kpoint_solver.set_parallel_backend`).
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers import solve_eigval
from parsec_python.Eigensolvers.eigval import first_solve_converges
from parsec_python.SCF.kpoint_solver import (
    KPointJob,
    device_solver_supports,
    job_eigval_settings,
    residual_max,
    resized_job,
    set_parallel_backend,
    task_rows,
    task_states,
)

from ..Eigensolvers.chebff_spinor import ReportedFirstSolve, SpinorEigvalState, solve_spinor_eigval
from .kpoint_real import as_complex, matched_state, psi_gauge_for, real_gamma_applies
from .kpoint_sectors import carried_state, refined_state, warm_growth_takes
from ..Parallel import WORKERS_ENV, make_pool, mpi

SOLVER_ENV = "PARSEC_KPOINT_SOLVER"  # device (default) | host
STORAGE_ENV = "PARSEC_KPOINT_STATE_STORAGE"  # auto (default) | device | host
FIRST_SOLVE_ENV = "PARSEC_KPOINT_FIRST_SOLVE"  # refine (default) | chebdav
_AUTO_FRACTION = 0.75  # of the device memory the subspaces (and one transient copy set) may take
_REPORTED_HOST_FRACTION = 0.75  # of the available host memory the subspaces of a first solve in parts may take


def _ordered_sums() -> bool:
    """``PARSEC_ORDERED_SUMS`` (:func:`..Hamiltonian.complex_csr_device.ordered_sums_setting`) is ``auto``."""
    from ..Hamiltonian.complex_csr_device import ordered_sums_setting

    return ordered_sums_setting() == "auto"


def _device_solver_requested() -> bool:
    value = os.environ.get(SOLVER_ENV, "device").strip().lower() or "device"
    if value not in ("device", "host"):
        raise ValueError(f"{SOLVER_ENV} must be device or host, not {value!r}")
    return value == "device"


def _first_solve_on_host() -> bool:
    """Whether ``Eigensolver: chebdav`` runs as the host CHEBDAV (``chebdav``) or is replaced on the
    device by CHEBFF with extra cycles until CHEBDAV's residual test is met (``refine``)."""
    value = os.environ.get(FIRST_SOLVE_ENV, "refine").strip().lower() or "refine"
    if value not in ("refine", "chebdav"):
        raise ValueError(f"{FIRST_SOLVE_ENV} must be refine or chebdav, not {value!r}")
    return value == "chebdav"


def _available_host_bytes() -> int | None:
    """Bytes of host memory available now (``None`` where the system does not say)."""
    from ..Eigensolvers.spill_buffers import host_available_bytes

    return host_available_bytes()


def _storage_mode() -> str:
    value = os.environ.get(STORAGE_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "device", "host"):
        raise ValueError(f"{STORAGE_ENV} must be auto, device or host, not {value!r}")
    return value


def resolve_backend(requested: str | None = None):
    """The device backend plan, or ``None`` for the CPU.

    The device follows ``PARSEC_DEVICE`` (what ``--device`` sets, default
    ``auto``): ``gpu`` and ``auto`` use the GPU path (one worker thread per
    CUDA device, ``PARSEC_CUPY_DEVICES`` selects them) when CuPy and a device
    work, ``auto`` falling back to the CPU and ``gpu`` raising without one;
    ``cpu`` never uses it.  ``requested`` or ``PARSEC_KPOINT_BACKEND``
    (``cpu``, ``cupy``, ``auto`` or ``numpy-device``, the device code on the
    host, for tests) overrides that.  The acceleration package is only
    imported when a device could be used.
    """
    name = (requested or os.environ.get("PARSEC_KPOINT_BACKEND", "")).strip().lower()
    if not name:
        device = os.environ.get("PARSEC_DEVICE", "auto").strip().lower() or "auto"
        name = {"cpu": "cpu", "gpu": "cupy", "auto": "auto"}.get(device)
        if name is None:
            raise ValueError(f"PARSEC_DEVICE must be auto, gpu or cpu, not {device!r}")
    if name == "cpu":
        return None
    from ..Hamiltonian.device_bloch import resolve_kpoint_backend

    plan = resolve_kpoint_backend(name)
    return plan if plan.uses_device else None


def band_stage_route(backend: str | None = None, workers=None, n_tasks: int = 1, problem_size: int | None = None):
    """How the band k-points of this run are solved and what memory one solve may take: ``(device, host chebdav in
    sectors, bytes)``.

    ``device``: the device-resident first solve (a device plan with ``PARSEC_KPOINT_SOLVER=device``), whose
    converged first solve is CHEBFF with more cycles to the relative residual test; else the host eigensolvers.
    ``host chebdav in sectors``: the workers of the band sectors make the first solve of ``Eigensolver: chebdav`` as
    the host CHEBDAV (``PARSEC_KPOINT_FIRST_SOLVE=chebdav``).  ``bytes``: the memory of a device (the total of each
    device that solves band k-points, divided by the processes that share it) or the available host memory divided
    by the worker threads and the processes of a host; the least of them, the same on every MPI rank (``None`` where
    the host does not say).  Read once per band stage, so that every k-point takes the same decision.
    """
    import socket
    from collections import Counter

    from ..Parallel.pool import resolve_workers

    plan = resolve_backend(backend)
    device = plan is not None and _device_solver_requested()
    context = mpi.context()
    host = socket.gethostname()
    entries: list[tuple[str, str, float]] = []
    if device and plan.name == "cupy":
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        ids = list(plan.device_ids)
        if context is not None and context.size > 1:
            ids = [ids[context.local_rank % len(ids)]]
        else:
            ids = ids[: device_worker_count(plan, n_tasks)]
        for device_id in ids:
            with cp.cuda.Device(device_id):
                total = float(cp.cuda.runtime.memGetInfo()[1])
                bus = str(cp.cuda.Device(device_id).pci_bus_id)
            entries.append((host, bus, total))
    else:
        from ..Eigensolvers.spill_buffers import host_available_bytes

        available = host_available_bytes()
        if available is not None:
            threads = device_worker_count(plan, n_tasks) if plan is not None else resolve_workers(
                workers, n_tasks, problem_size=problem_size
            )
            entries.append((host, "host", float(available) / max(1, int(threads))))
    if context is not None and context.size > 1:
        entries = [entry for part in context.comm.allgather(entries) for entry in part]
        if not all(context.comm.allgather(bool(entries))):
            entries = []
    if not entries:
        return device, device and _first_solve_on_host(), None
    users = Counter((name, bus) for name, bus, _ in entries)
    budget = min(total / users[(name, bus)] for name, bus, total in entries)
    return device, device and _first_solve_on_host(), int(budget)


def device_worker_count(plan, n_tasks: int) -> int:
    """Worker threads of a device plan: one per device (host double: ``PARSEC_KPOINT_WORKERS``)."""
    if plan.name == "numpy-device":
        requested = os.environ.get(WORKERS_ENV, "1")
        count = int(requested) if requested.strip().isdigit() else 1
    else:
        count = len(plan.device_ids)
    return max(1, min(count, n_tasks))


class _SolveWorker:
    """Owns the k-point tasks ``tasks``: their Ritz states and eigenvectors."""

    def __init__(
        self, job: KPointJob, tasks: list[int], index: int, plan=None, real_gamma: bool = False
    ) -> None:
        self.job = job
        self.tasks = tasks
        self.index = index
        self.states: dict[int, Any] = {}
        self.vectors: dict[int, Any] = {}
        self.stats: dict[str, float] = {}
        self.residual: dict[int, float] = {}
        # Built here, in the worker (thread), so the device is current for it.
        self.converter = plan.converter(index) if plan is not None else None
        self.device_solver = self.converter is not None and _device_solver_requested()
        if self.device_solver:
            # the Gamma point in real arithmetic on the device solver (``real_gamma``: the solver's decision,
            # .kpoint_real.real_gamma_applies, PARSEC_KPOINT_REAL_GAMMA); not where a host CHEBDAV makes the first
            # solve (PARSEC_KPOINT_FIRST_SOLVE=chebdav: its subspace is complex)
            self.converter.real_gamma = bool(real_gamma) and not _first_solve_on_host()
        # the psi gauge of the time-reversal invariant points in the Bloch form, solved in real arithmetic
        # (PARSEC_KPOINT_TRIM_REAL, .kpoint_real.PsiGauge): by task, None where the task keeps the u gauge
        self.gauges: dict[int, Any] = {}
        self._psi = self.device_solver and not _first_solve_on_host()
        self._spill: bool | None = None
        # export_states hands the device subspaces as they are (the SCF loops of a hybrid's Fock loop)
        self.device_states = False
        # A first solve in parts whose test covers the reported pairs (the band k-points in sectors,
        # SectorKPointSolver.solve_reported): None, or the (count, cut) of the pairs each task's test covers
        # (ReportedFirstSolve; a task missing from it makes its fixed cycles); per task the residuals of its pairs and
        # the cycles made.
        self.reported_cuts: dict[int, tuple[int, float]] | None = None
        self.reported_tolerance: float | None = None
        self.reported_progress: dict[int, tuple[np.ndarray, int]] = {}
        # the tasks of whole k-points among them: one first solve with the test of all their pairs, as before
        self.reported_whole: frozenset[int] = frozenset()
        # PARSEC_KPOINT_SECTOR_WARM_GROWTH: by task that grew (resize_tasks), the saved Ritz vectors of its former basis
        # on the host and their gauge, for the first solve of its new count
        self.growth_starts: dict[int, tuple[Any, str | None]] = {}

    def _tick(self, name: str, start: float) -> None:
        """Add the seconds since ``start`` to the worker's ``worker_<name>_seconds`` counter."""
        if self.converter is not None:
            self.converter.backend.synchronize()
        key = f"worker_{name}_seconds"
        self.stats[key] = self.stats.get(key, 0.0) + time.perf_counter() - start

    # ------------------------------------------------------------- host route
    def operator(self, hamiltonian):
        if self.converter is not None:
            # the host eigensolvers (ARPACK, DIAGLA) take complex blocks: the complex operator, as before
            return self.converter.operator(hamiltonian, real=False).as_linear_operator()
        return hamiltonian.as_linear_operator()

    # ----------------------------------------------------------- device route
    def _should_spill(self, dimension: int, working: int) -> bool:
        """Whether the subspaces of this worker's tasks stay on the host between solves."""
        mode = _storage_mode()
        if mode != "auto":
            return mode == "host"
        backend = self.converter.backend
        if backend.name != "cupy":
            return False
        backend.use()
        _, total = backend.cp.cuda.runtime.memGetInfo()
        needed = (len(self.tasks) + 6) * dimension * working * 16
        return needed > _AUTO_FRACTION * total

    def _to_host(self, array) -> np.ndarray:
        return array if isinstance(array, np.ndarray) else self.converter.backend.asnumpy(array)

    def _gauge(self, task: int, hamiltonian):
        """The psi gauge of ``task`` (:func:`.kpoint_real.psi_gauge_for`), ``None`` for the u gauge."""
        if not self._psi:
            return None
        return psi_gauge_for(self.job, task, hamiltonian, self.gauges)

    def _solve_on_device(self, task: int, hamiltonian, settings):
        backend = self.converter.backend
        backend.use()
        start = time.perf_counter()
        gauge = self._gauge(task, hamiltonian)
        if gauge is not None:
            hamiltonian = gauge.hamiltonian(hamiltonian)
        operator = self.converter.operator(hamiltonian)
        self._tick("upload", start)
        requested = task_states(self.job, task)
        solve_start = time.perf_counter()
        reported = None
        # a real subspace into a complex solve, a complex one (or one of another gauge) dropped before a real solve
        # (.kpoint_real)
        state = matched_state(
            self.states.pop(task, None), bool(getattr(operator, "real_arithmetic", False)),
            None if gauge is None else "psi",
        )
        # a task that grew starts its first solve from the vectors of its former basis (PARSEC_KPOINT_SECTOR_WARM_GROWTH)
        start = self._growth_start(task, state, settings, operator, None if gauge is None else "psi")
        self.vectors.pop(task, None)  # release the previous basis before the next one is made
        # a task whose states the refit of the k-point sectors cut keeps its lowest Ritz pairs (kpoint_sectors)
        state = carried_state(state, requested, settings, int(operator.shape[0]))
        if state is not None and isinstance(state.vectors, np.ndarray) and backend.xp is not np:
            state.vectors = backend.asarray(state.vectors)  # a spilled subspace comes back
        elif (
            state is not None
            and backend.xp is not np
            and isinstance(state.vectors, backend.xp.ndarray)
            and int(state.vectors.device.id) != int(backend.device_id)
        ):
            # a subspace kept on another device (a pool of another layout): copied here
            source = backend.xp.ascontiguousarray(state.vectors)
            copy = backend.xp.empty(source.shape, dtype=source.dtype)
            copy.data.copy_from_device(source.data, source.nbytes)
            state.vectors = copy
        if state is None and settings.initial_method == "chebdav" and _first_solve_on_host():
            # CHEBDAV (the first solve of Eigensolver: chebdav) runs on the host with the device
            # filter; its subspace then continues on the device.
            host = solve_eigval(operator.as_linear_operator(), requested, settings=settings, state=None)
            saved = host.state.subspace
            new_state = SpinorEigvalState(
                vectors=backend.asarray(np.ascontiguousarray(saved.vectors)),
                eigenvalues=np.asarray(saved.eigenvalues, dtype=float),
                working_states=saved.working_states,
                operator_dimension=saved.operator_dimension,
                first_filter=True,
                filter_lower_bound=None,
                filters_completed=0,
            )
            eigenvalues = np.asarray(host.eigenvalues, dtype=float)
            self.residual[task] = residual_max(host.residual_norms)
        else:
            if (
                self.reported_cuts is not None
                and task not in self.reported_whole
                and first_solve_converges(settings)
            ):
                part = self.reported_cuts.get(task)
                reported = ReportedFirstSolve() if part is None else ReportedFirstSolve(count=part[0], cut=part[1])
            result = solve_spinor_eigval(
                operator,
                requested,
                settings=settings,
                state=state,
                xp=backend.xp,
                sync=backend.synchronize,
                # Eigensolver: chebdav converges its first solve; on the device that is CHEBFF plus
                # more cycles to the same residual test
                # and the CHEBFF band k-points under PARSEC_BAND_CONVERGENCE=auto
                converge_first=first_solve_converges(settings),
                tolerance=None if reported is None else self.reported_tolerance,
                reported=reported,
                start=start,
            )
            del start
            new_state = result.state
            eigenvalues = result.eigenvalues
            self.residual[task] = residual_max(result.residual_norms)
            if reported is not None:
                self.reported_progress[task] = (np.asarray(result.residual_norms), int(new_state.cursor.cycles))
            for name, value in result.stats.items():
                self.stats[name] = self.stats.get(name, 0.0) + value
        self._tick("solve", solve_start)
        new_state.gauge = None if gauge is None else "psi"
        if self._spill is None:
            self._spill = self._should_spill(new_state.operator_dimension, new_state.working_states)
        if self._spill:
            spill_start = time.perf_counter()
            host_vectors = self._to_host(new_state.vectors)
            new_state.vectors = host_vectors
            self.vectors[task] = host_vectors[:, :requested]
            self._tick("spill", spill_start)
        else:
            self.vectors[task] = new_state.vectors[:, :requested]
        self.states[task] = new_state
        return eigenvalues

    def _host_vectors(self, task: int) -> np.ndarray:
        return np.asarray(self._to_host(self.vectors[task]))

    # ------------------------------------------------------------- SCF steps
    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None, only=None):
        # ``tolerance``: the eigensolver tolerance of this SCF iteration
        # (Dynamic_Diag_Tol); None keeps the job's Diag_Tolerance.  ``only``: the tasks to solve (all of
        # this worker's where None).
        settings = job_eigval_settings(self.job, filter_degree, tolerance)
        # ARPACK and DIAGLA are host eigensolvers: the host route below, with the
        # device Hamiltonian as their operator.
        on_device = self.device_solver and device_solver_supports(settings)
        for task in self.tasks if only is None else [task for task in self.tasks if task in only]:
            start = time.perf_counter()
            hamiltonian = self.job.make_hamiltonian(task, potentials)
            self._tick("hamiltonian", start)  # builds the Bloch projectors of a k-point on first use
            if on_device:
                yield task, self._solve_on_device(task, hamiltonian, settings)
                continue
            start = time.perf_counter()
            operator = self.operator(hamiltonian)
            self._tick("upload", start)
            start = time.perf_counter()
            solution = solve_eigval(
                operator,
                task_states(self.job, task),
                settings=settings,
                state=carried_state(self.states.get(task), task_states(self.job, task), settings, operator.shape[0]),
            )
            self._tick("solve", start)
            self.states[task] = solution.state
            self.vectors[task] = np.asarray(solution.vectors)
            self.residual[task] = residual_max(solution.residual_norms)
            yield task, np.asarray(solution.eigenvalues, dtype=float)

    def accumulate(self, occupations: dict[int, np.ndarray]):
        total: dict[str, np.ndarray] = {}
        for task in self.tasks:
            vectors = self.vectors[task]
            if self.converter is not None and not isinstance(vectors, np.ndarray):
                # form the density where the orbitals are: only the grid-length fields come back
                start = time.perf_counter()
                backend = self.converter.backend
                backend.use()
                parts = {
                    name: np.asarray(backend.asnumpy(value))
                    for name, value in self.job.accumulate(
                        task, vectors, occupations[task], xp=backend.xp
                    ).items()
                }
                self._tick("density", start)
            else:
                start = time.perf_counter()
                vectors = self._host_vectors(task)
                self._tick("download", start)
                start = time.perf_counter()
                parts = self.job.accumulate(task, vectors, occupations[task])
                self._tick("density", start)
            for name, value in parts.items():
                if name in total:
                    total[name] += value
                else:
                    total[name] = np.array(value, copy=True)
        # keyed by the worker's first task: the driver adds the parts of the workers in that order (PARSEC_ORDERED_SUMS)
        yield (min(self.tasks) if self.tasks else -1), total

    def resize_tasks(self, counts: dict):
        """New numbers of states of some tasks (:meth:`PooledKPointSolver.resize_tasks`): their Ritz states and
        orbitals are dropped here.  Under ``PARSEC_KPOINT_SECTOR_WARM_GROWTH`` the Ritz vectors of a device state that a
        SUBSPACE step has refined wait on the host for the first solve of the new count (:meth:`_growth_start`), where
        the host has room for them (:func:`..Eigensolvers.sector_growth.host_room`)."""
        from .kpoint_sectors import kpoint_sector_warm_growth_setting

        self.job = resized_job(self.job, counts)
        warm = self.device_solver and kpoint_sector_warm_growth_setting() != "off"
        for task in counts:
            if task in self.tasks:
                state = self.states.pop(task, None)
                self.vectors.pop(task, None)
                self.growth_starts.pop(task, None)
                if warm and refined_state(state):
                    saved = self._stage_growth(state)
                    if saved is not None:
                        self.growth_starts[task] = (saved, getattr(state, "gauge", None))
                del state
        yield None, None

    def _stage_growth(self, state):
        """The Ritz vectors of ``state`` on the host (a spilled subspace as it is), or ``None`` where the host has no
        room for them or the copy fails."""
        from ..Eigensolvers.sector_growth import host_room

        vectors = state.vectors
        if isinstance(vectors, np.ndarray):
            return vectors
        record = self.stats
        with host_room(int(vectors.nbytes)) as fits:
            if not fits:
                record["kpoint_warm_growth_no_room"] = record.get("kpoint_warm_growth_no_room", 0.0) + 1.0
                return None
            try:
                self.converter.backend.use()
                return np.asarray(self.converter.backend.asnumpy(vectors))
            except Exception:  # noqa: BLE001 - any failure of the copy falls back to random vectors
                record["kpoint_warm_growth_failed"] = record.get("kpoint_warm_growth_failed", 0.0) + 1.0
                return None

    def _growth_start(self, task: int, state, settings, operator, gauge):
        """The saved columns that the first solve of ``task`` starts from (:meth:`resize_tasks`), or ``None``: none
        saved, a state the solve continues, a first solve that does not take them (:func:`.kpoint_sectors
        .warm_growth_takes`; the host CHEBDAV of ``PARSEC_KPOINT_FIRST_SOLVE=chebdav``), or columns of another gauge,
        arithmetic or number of rows."""
        saved = self.growth_starts.pop(task, None)
        if saved is None:
            return None
        columns, saved_gauge = saved
        if state is not None or not warm_growth_takes(settings) or _first_solve_on_host():
            return None
        rows = int(operator.shape[0])
        real = bool(getattr(operator, "real_arithmetic", False))
        if saved_gauge != gauge or int(columns.shape[0]) != rows or (real and np.iscomplexobj(columns)):
            return None
        self.stats["kpoint_warm_growths"] = self.stats.get("kpoint_warm_growths", 0.0) + 1.0
        return columns

    def trim_tasks(self, counts: dict):
        """Fewer states for some tasks (:meth:`PooledKPointSolver.trim_tasks`): their Ritz states are kept, and
        their next solves take the lowest of them (:func:`.kpoint_sectors.carried_state`)."""
        self.job = resized_job(self.job, counts)
        yield None, None

    def prepare_reported(self, buffer: int):
        """``(can, on the host, bytes)`` of a first solve in the parts of :class:`ReportedFirstSolve`: whether this
        worker can make one, whether its subspaces wait on the host between the parts (:meth:`_should_spill` for its
        largest task: every subspace of the worker stays resident during the parts), and their size."""
        if not self.device_solver or _first_solve_on_host():
            yield None, (False, False, 0)
            return
        sizes = []
        for task in self.tasks:
            rows = task_rows(self.job, task, int(self.job.grid_points or 0))
            sizes.append((rows, min(task_states(self.job, task) + int(buffer), rows)))
        if sizes:
            self._spill = self._should_spill(*max(sizes, key=lambda size: size[0] * size[1]))
        yield None, (True, bool(self._spill), sum(rows * working * 16 for rows, working in sizes))

    def set_reported_cuts(self, cuts, tolerance=None, whole=()):
        """The pairs of the next first-solve part by task (:attr:`reported_cuts`), the relative residual its test asks
        for and the tasks of whole k-points, which keep the first solve of before; ``cuts=None`` ends the parts."""
        self.reported_cuts = None if cuts is None else dict(cuts)
        self.reported_tolerance = tolerance
        self.reported_whole = frozenset(whole)
        yield None, None

    def reported_progress_of(self):
        yield None, dict(self.reported_progress)

    def keep_device_states(self):
        """Hand the device subspaces as they are from now on (no host copy in :meth:`export_states`)."""
        self.device_states = True
        yield None, None

    def device_vectors_of(self, task: int):
        """The final eigenvectors of ``task`` where they are (a device array, or a spilled host one); complex, as
        Bloch orbitals are (a real Gamma point's are cast, .kpoint_real)."""
        if task in self.vectors:
            vectors = self.vectors[task]
            xp = np
            if self.converter is not None and not isinstance(vectors, np.ndarray):
                self.converter.backend.use()
                xp = self.converter.backend.xp
            yield task, as_complex(self._u_gauge(task, vectors, xp))

    def device_of(self):
        """``(task, device id)`` of every task of a device worker (none on the CPU)."""
        if self.converter is not None and self.converter.backend.name == "cupy":
            for task in self.tasks:
                yield task, int(self.converter.backend.device_id)

    def export_states(self):
        """``(task, state)`` of every task with a state, its subspace on the host (or kept on the device)."""
        for task in self.tasks:
            state = self.states.get(task)
            if state is None:
                continue
            vectors = getattr(state, "vectors", None)
            if self.device_states:
                yield task, replace(state)
                continue
            if vectors is not None and not isinstance(vectors, np.ndarray):  # the subspace of a device
                self.converter.backend.use()
                state = replace(state, vectors=self._to_host(vectors))
            yield task, state

    def import_states(self, states: dict[int, Any]):
        for task in self.tasks:
            if task in states:
                self.states[task] = states[task]  # a host subspace goes to the device when it is solved
        yield None, None

    def _u_gauge(self, task: int, vectors, xp=np):
        """``vectors`` of ``task`` in the u gauge of the Bloch factor (those of the psi gauge converted)."""
        gauge = self.gauges.get(task)
        return vectors if gauge is None else gauge.to_u(vectors, xp)

    def extras(self):
        for task in self.tasks:
            vectors = self.vectors[task]
            if self.converter is not None and not isinstance(vectors, np.ndarray):
                self.converter.backend.use()
                xp = self.converter.backend.xp
                yield task, self.job.extras(task, self._u_gauge(task, vectors, xp), xp=xp)
            else:
                yield task, self.job.extras(task, self._u_gauge(task, self._host_vectors(task)))

    def vectors_of(self, task: int, columns: int | None = None):
        if task in self.vectors:
            if columns is None:
                yield task, as_complex(self._u_gauge(task, self._host_vectors(task)))
            else:
                # the first ``columns`` eigenvectors only: copied to the host and gauged without the rest
                yield task, as_complex(
                    self._u_gauge(task, np.asarray(self._to_host(self.vectors[task][:, : int(columns)])))
                )

    def stats_of(self):
        yield None, dict(self.stats)

    def residuals_of(self):
        yield None, dict(self.residual)


class PooledKPointSolver:
    """Drives a pool of workers through the SCF steps of a k-point job."""

    def __init__(
        self,
        job: KPointJob,
        *,
        workers: int | str | None = None,
        problem_size: int | None = None,
        backend: str | None = None,
        real_gamma: bool | None = None,
    ) -> None:
        self.job = job
        plan = resolve_backend(backend)
        if plan is not None:
            workers = device_worker_count(plan, job.n_tasks)
        self.backend = plan.name if plan is not None else "cpu"
        self._values: dict[int, np.ndarray] = {}
        # the Gamma point in real arithmetic (PARSEC_KPOINT_REAL_GAMMA): given by a solver of sectors (decided on the
        # k-points it splits), else decided on this job (.kpoint_real.real_gamma_applies)
        real = real_gamma_applies(job, problem_size) if real_gamma is None else bool(real_gamma)
        self.pool = make_pool(
            range(job.n_tasks),
            lambda tasks, index: _SolveWorker(job, tasks, index, plan, real),
            workers=workers,
            problem_size=problem_size,
        )

    @property
    def n_workers(self) -> int:
        return self.pool.n_workers

    def solve(
        self, potentials: dict, filter_degree: int, tolerance: float | None = None, tasks=None
    ) -> list[np.ndarray]:
        """One eigenvalue array per task (task order), vectors kept in the workers.

        ``tolerance`` replaces ``Diag_Tolerance`` for this call
        (``Dynamic_Diag_Tol``); the job's ``eigval_settings`` then takes it
        as a second argument.  ``tasks`` solves those only, the others keep the
        eigenvalues of their last solve.
        """
        if tasks is not None:
            by_task = self.pool.by_task("solve", potentials, filter_degree, tolerance, frozenset(int(t) for t in tasks))
        elif tolerance is None:
            by_task = self.pool.by_task("solve", potentials, filter_degree)
        else:
            by_task = self.pool.by_task("solve", potentials, filter_degree, tolerance)
        self._values.update(by_task)
        return [self._values[task] for task in range(self.job.n_tasks)]

    def resize_tasks(self, counts: dict) -> bool:
        """New numbers of states for the tasks of ``counts`` (task -> states); the workers drop their Ritz states and
        keep those of the other tasks.  ``True``: the next :meth:`solve` of those tasks takes them."""
        counts = {int(task): int(count) for task, count in counts.items()}
        self.job = resized_job(self.job, counts)
        self.pool.results("resize_tasks", counts)
        return True

    def trim_tasks(self, counts: dict) -> bool:
        """Fewer states for the tasks of ``counts`` (task -> states, each at most its present count): the workers
        keep their Ritz states, and the next solve of each such task starts from its lowest Ritz pairs (the refit of
        the k-point sectors).  ``True``: done."""
        counts = {int(task): int(count) for task, count in counts.items()}
        self.job = resized_job(self.job, counts)
        self.pool.results("trim_tasks", counts)
        return True

    def prepare_reported(self, buffer: int) -> bool:
        """Whether a first solve in parts can run: every worker has the device solver and, where the device is short
        (``PARSEC_KPOINT_STATE_STORAGE=auto``) so that the subspaces wait on the host, the host has room for them."""
        found = [part for _, part in self.pool.results("prepare_reported", int(buffer))]
        if not all(part[0] for part in found):
            return False
        on_host = sum(part[2] for part in found if part[1])
        if not on_host or _storage_mode() != "auto":
            return True
        available = _available_host_bytes()
        return available is None or on_host <= _REPORTED_HOST_FRACTION * available

    def set_reported_cuts(self, cuts, tolerance=None, whole=()) -> None:
        """The next :meth:`solve` of a task continues its CHEBFF cycles until the pairs of ``cuts[task]``, ``(count,
        cut)`` of :class:`ReportedFirstSolve`, reach the relative residual ``tolerance``; a task not in ``cuts`` makes
        its fixed cycles, and a task in ``whole`` (a whole k-point) the first solve of before; ``None`` goes back to
        whole first solves."""
        self.pool.results("set_reported_cuts", cuts, tolerance, whole)

    def reported_progress(self) -> dict[int, tuple[np.ndarray, int]]:
        """``task -> (residuals of its pairs, CHEBFF cycles made)`` of the solves in parts."""
        merged: dict[int, tuple[np.ndarray, int]] = {}
        for _, part in self.pool.results("reported_progress_of"):
            merged.update(part)
        return merged

    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Sum of every task's density contribution for the given occupations."""
        payload = {task: occupations[task] for task in range(self.job.n_tasks)}
        total: dict[str, np.ndarray] = {}
        parts = self.pool.results("accumulate", payload)
        if _ordered_sums():
            # the parts in the order of the workers, not in the order they finish: the same sum in every run
            parts = sorted(parts, key=lambda item: item[0])
        for _, part in parts:
            for name, value in part.items():
                if name in total:
                    total[name] = total[name] + value
                else:
                    total[name] = value
        return total

    def extras(self) -> list[Any]:
        by_task = self.pool.by_task("extras")
        return [by_task[task] for task in range(self.job.n_tasks)]

    def export_states(self) -> dict[int, Any]:
        """The Ritz state of every task, with the subspace on the host (see the serial solver)."""
        return self.pool.by_task("export_states")

    def import_states(self, states: dict[int, Any]) -> None:
        """Start the tasks from the Ritz states ``states`` (by task) of an earlier loop."""
        self.pool.results("import_states", states)

    def vectors(self, task: int, columns: int | None = None) -> np.ndarray:
        """The final eigenvectors of one task (fetched from its worker; its first ``columns`` ones where given)."""
        arguments = {} if columns is None else {"columns": int(columns)}
        for _, vectors in self.pool.results("vectors_of", task, **arguments):
            return vectors
        raise KeyError(task)

    def device_vectors(self, task: int):
        """The final eigenvectors of one task where its worker keeps them: a device array (a view of
        the task's Ritz subspace, which its next solve overwrites), or a host array where spilled."""
        for _, vectors in self.pool.results("device_vectors_of", task):
            return vectors
        raise KeyError(task)

    def keep_states_on_devices(self) -> None:
        """Let :meth:`export_states` hand the device subspaces as they are, without host copies (the SCF
        loops of a hybrid's Fock loop, whose next loop runs on the same devices)."""
        if not getattr(self, "_device_states", False):
            self.pool.results("keep_device_states")
            self._device_states = True

    def task_devices(self) -> dict[int, int]:
        """The CUDA device of every task (empty on the CPU)."""
        return dict(self.pool.results("device_of"))

    def stats(self) -> dict[str, float]:
        """Eigensolver counters and seconds summed over the workers (device route)."""
        total: dict[str, float] = {}
        for _, part in self.pool.results("stats_of"):
            for name, value in part.items():
                total[name] = total.get(name, 0.0) + value
        return total

    def residuals(self) -> list[float]:
        """Largest eigenpair residual of each task's last solve (NaN where none was measured)."""
        merged: dict[int, float] = {}
        for _, part in self.pool.results("residuals_of"):
            merged.update(part)
        return [merged.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> "PooledKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _TaskWorker:
    """Runs ``solve_task(task, converter)`` for its tasks (band paths and the like)."""

    def __init__(self, solve_task, tasks, index, plan=None) -> None:
        self.solve_task = solve_task
        self.tasks = tasks
        # Built here, in the worker (thread), so the device is current for it.
        self.converter = plan.converter(index) if plan is not None else None

    def run(self):
        for task in self.tasks:
            yield task, self.solve_task(task, self.converter)


def _mpi_map_tasks(context, solve_task, n_tasks: int, backend=None):
    """The independent tasks dealt out to the MPI ranks, each solved whole on the device of its rank;
    every rank gets all the results (in task order) and yields them."""
    plan = resolve_backend(backend)
    converter = plan.converter(context.local_rank) if plan is not None else None
    mine = [(task, solve_task(task, converter)) for task in range(context.rank, n_tasks, context.size)]
    gathered = {}
    for part in context.comm.allgather(mine):
        gathered.update(part)
    for task in range(n_tasks):
        yield task, gathered[task]


def map_tasks(solve_task, n_tasks: int, *, problem_size=None, workers=None, backend=None):
    """Yield ``(task, result)`` as the workers finish the independent tasks."""
    context = mpi.context()
    if context is not None and context.size > 1:
        yield from _mpi_map_tasks(context, solve_task, n_tasks, backend)
        return
    plan = resolve_backend(backend)
    if plan is not None:
        workers = device_worker_count(plan, n_tasks)
    with make_pool(
        range(n_tasks),
        lambda tasks, index: _TaskWorker(solve_task, tasks, index, plan),
        workers=workers,
        problem_size=problem_size,
    ) as pool:
        yield from pool.stream("run")


def make_solver(job: KPointJob, **options):
    """The solver of a k-point job: in an MPI run (``PARSEC_MPI=1``) the pools of ranks
    (:class:`.kpoint_pools.MPIPoolKPointSolver`), or for a job with per-task extras (the spin-orbit and
    non-collinear drivers, forces) the basis split by rows over the ranks (:class:`.kpoint_mpi.MPIKPointSolver`);
    else the pool of threads and devices of this process.  A job whose k-points split into the sectors of their
    little groups (``PARSEC_KPOINT_SECTORS``, :mod:`.kpoint_sectors`) gets that solver of its sectors; a spin-orbit
    job whose geometry says so (``spinor``) the solver of its double-group sectors (``PARSEC_SOC_KPOINT_SECTORS``,
    :mod:`.kpoint_spinor_sectors`)."""
    from .kpoint_sectors import sector_solver

    notes: list[str] = []
    geometry = getattr(job, "geometry", None)
    if isinstance(geometry, dict) and geometry.get("spinor"):
        from .kpoint_spinor_sectors import spinor_sector_solver

        solver = spinor_sector_solver(job, _plain_solver, notes=notes, **options)
    else:
        solver = sector_solver(job, _plain_solver, notes=notes, **options)
    chosen = _plain_solver(job, **options) if solver is None else solver
    if notes:
        # the cell group of the sectors and its tolerance, or why there are none: the kpoint_symmetry_detection line
        # of parsec.out (SCF/kpoints.py, _run_timings)
        try:
            chosen.kpoint_symmetry_detection = notes[-1]
        except AttributeError:  # a solver that takes no attributes
            pass
    return chosen


def _plain_solver(job: KPointJob, **options):
    """The solver of :func:`make_solver` with one task per k-point (or per task of the job)."""
    if mpi.context() is not None:
        # the pools also take extras that sum over any subset of the states (the forces of the scalar SCF)
        if (job.extras is None or job.extras_by_columns) and options.get("problem_size") is not None:
            from .kpoint_pools import MPIPoolKPointSolver

            return MPIPoolKPointSolver(job, **options)
        from .kpoint_mpi import MPIKPointSolver

        return MPIKPointSolver(job, **options)
    return PooledKPointSolver(job, **options)


def describe() -> list[str]:
    """The lines of ``parsec.out`` that say what the k-point eigenproblems run on."""
    lines = [" Execution resources (k-point eigensolver):", " -------------------------------------------"]
    context = mpi.context()
    try:
        plan = resolve_backend()
    except Exception as error:  # no GPU where one was requested, for instance
        return lines + [f" Device            = unavailable ({error})"]
    if plan is None:
        lines += [
            " Device            = CPU",
            " Eigensolver       = host, k-points and spin channels spread over worker threads",
        ]
    else:
        names = []
        if plan.name == "cupy":
            try:
                from ..backends.cupy import require_cupy

                cp, _ = require_cupy()
                for device_id in plan.device_ids[:1]:
                    names.append(cp.cuda.runtime.getDeviceProperties(device_id)["name"].decode())
            except Exception:
                pass
        kind = {"cupy": "GPU", "numpy-device": "CPU (device code on the host, a test double)"}[plan.name]
        ids = ",".join(str(i) for i in plan.device_ids)
        lines.append(f" Device            = {kind}" + (f" ({names[0]})" if names else ""))
        if plan.name == "cupy":
            if context is not None:
                lines.append(f" GPUs/node         = {len(plan.device_ids)} visible, index {ids}")
            else:
                lines.append(f" GPUs              = {len(plan.device_ids)} active, index {ids}")
        route = (
            "device-resident Ritz subspaces (filter, QR and Rayleigh-Ritz on the device)"
            if _device_solver_requested()
            else "host eigensolver, only the Chebyshev filter on the device"
        )
        lines.append(f" Eigensolver       = {route}")
    if context is not None:
        import socket

        device = None
        if plan is not None and plan.name == "cupy":
            device = plan.device_ids[context.local_rank % len(plan.device_ids)]
        # every rank takes part: the report says where all of them run, not only rank 0
        placement = context.comm.allgather((socket.gethostname(), device))
        nodes: dict[str, list] = {}
        for host, gpu in placement:
            nodes.setdefault(host, []).append(gpu)
        from .kpoint_pools import requested_pools

        try:
            pools = requested_pools()
        except ValueError:
            pools = None
        lines.append(
            f" MPI               = {context.size} ranks on {len(nodes)} node(s), {context.ranks_per_node} per node; "
            "k-points and spin channels are solved by pools of ranks"
            + (f" ({pools} pools, --kpoint-pools)" if pools is not None else " (pools by balance and memory)")
            + ", each basis shared by columns within its pool; spin-orbit and non-collinear runs split the "
            "basis by blocks of grid rows over all ranks"
        )
        if device is not None:
            from ..Eigensolvers.mpi_shared_basis import shared_transport

            try:
                shared = shared_transport()
            except ValueError:
                shared = "host"
            lines.append(
                " GPU-aware MPI     = "
                + (
                    "on: the pools move their bases between columns and rows as device arrays"
                    if shared == "cuda"
                    else "off: the pools move their bases through the host (MPICH_GPU_SUPPORT_ENABLED=1 for "
                    "device arrays)"
                )
                + "; rows of spinors: "
                + (
                    "halo rows between the GPUs, sums "
                    + ("between the GPUs" if context.cuda_aware_reduce else "through the host")
                    if context.cuda_aware
                    else "halo rows and sums through the host (PARSEC_MPI_CUDA_AWARE=1 for GPU buffers)"
                )
            )
            lines.append(
                f" GPUs              = {sum(len(g) for g in nodes.values())} active, one per rank, "
                f"on {len(nodes)} node(s):"
            )
            for host, gpus in nodes.items():
                lines.append(f"   node {host}: GPU index {','.join(str(g) for g in gpus)}")
    elif plan is not None and plan.name == "cupy":
        line = " Workers           = one thread per GPU; the k-points and spin channels are dealt out to them"
        from .kpoint_processes import processes_expected

        try:
            if processes_expected():
                # PARSEC_KPOINT_SECTOR_PROCESSES: the tasks of the k-point sectors (.kpoint_processes), where a loop
                # takes them; the K-point sectors block of the timings says which did
                line += "; the small sectors of the k-points to one worker process per GPU where a loop takes them"
        except ValueError:  # (an unknown value of the switch is refused at the preparation of the run)
            pass
        lines.append(line)
    return lines


def install() -> None:
    """Make this the parallel backend of the reference k-point drivers and band paths."""
    set_parallel_backend(
        make_solver,
        map_tasks,
        uses_device=lambda: mpi.requested() or resolve_backend() is not None,
        describe=describe,
    )


__all__ = ["PooledKPointSolver", "device_worker_count", "install", "map_tasks", "resolve_backend"]
