"""Worker processes for the k-point sectors on the devices of one process (``PARSEC_KPOINT_SECTOR_PROCESSES``).

The device solver of one process (:mod:`.kpoint_parallel`) gives every device a thread.  The tasks of the k-point
sectors (:mod:`.kpoint_sectors`) are small: a sector of a cell of a few hundred thousand grid points has a few thousand
rows and a few states, and its solve is some hundred short kernels whose launches, the Python of CuPy, take the time,
not the device.  The threads of one process take turns at the interpreter lock, so four devices solved the sectors no
faster than one (SrTiO3 40 atoms on a 2 x 2 x 2 mesh: 12.3 s of solves on one A100, 11.3 s on each of four, whose
threads waited two thirds of that time).  Here every device gets a process of its own, so the launches of the devices
run side by side.

The parent keeps the job: it builds the Hamiltonians of the sectors (they need the closures of the driver) and sends
them; their static parts, the objects the device converters keep by identity (the kinetic tables of the sectors, the
reduced projectors, sparse matrices, the geometry of a meta-GGA's orbital term), go once per solver and stay in the
worker process, so a step sends only the orbit means of the potential (and the matrices of an LDA+U term, the orbit
values of the fields of a meta-GGA).  A worker process runs the solver of a device thread (``_SolveWorker``) on the tasks
that thread would hold, on its device; the parts of the density that need the orbitals, ``sum_n f_n |v_n|^2``, the
LDA+U occupation matrices and the orbital sums of a meta-GGA, are formed there, the rest by the parent (a thread per
worker) in the order of the threads.  So this route gives the numbers of the threads, bit for bit: the same partition
of the tasks, the same operations on the same devices, the same order of the sums.  The forces are formed in the
workers as well, from the parts of the driver's nonlocal force (the projectors and their gradients of the atoms of the
orbits, the occupations) and the expansions of the sectors, which they receive with the recipes of their tasks.

Taken by the solver of the sectors where every k-point is split, on CuPy with two devices or more in the default
compute mode, without MPI, for the local, LDA+U and meta-GGA terms and the nonlocal forces, where the median task of the
sectors holds at most :func:`size_bound` rows times states, a loop with forces up to larger ones (larger tasks keep
their devices busy from the threads, and a process per device would only add memory); not the exchange of a hybrid or the real forms of
``PARSEC_KPOINT_SECTOR_REAL`` (those keep the threads), nor the band paths, whose solver of sectors has an inner solver
of its own.  The worker processes
start with the first such solver, beside the preparation of its first step, and serve the later ones of the run; where
they cannot start, the threads solve the sectors, with a warning.  A meta-GGA on several devices then takes the floor of
the sectors of one device (``.kpoint_sectors.sector_solver``).  ``off`` keeps the threads and the floor of a
meta-GGA on several devices; ``on`` also takes the host double of the device code (``PARSEC_KPOINT_BACKEND=numpy-device``
with ``PARSEC_KPOINT_WORKERS``), for the tests.
"""

from __future__ import annotations

import atexit
import functools
import gc
import io
import os
import pickle
import secrets
import subprocess
import sys
import threading
import time
import traceback
import warnings
from typing import Any

import numpy as np

SWITCH = "PARSEC_KPOINT_SECTOR_PROCESSES"
#: Under ``auto``, the largest median of rows times states of the sector tasks that takes the worker processes: above
#: it a solve is set by its device, the threads of one process keep the devices busy, and a process per device would
#: only add memory (four A100: Cu 108 with LDA+U at 3.8e5 38.5 against 61.1 s, diamond 4^3 cubes at 1.7e6 13.1
#: against 12.2 s, 5^3 at 2.5e7 36.3 against 32.8 s).
_TASK_ENTRIES = 1 << 20
#: The same bound for a loop with extras (the nonlocal forces, formed in the workers): their solves and forces gain up to
#: larger sectors (four A100, with forces, ``on`` against ``off``: diamond 4^3 cubes at 1.7e6 14.3 against 15.2 s, 6^3
#: at 1.9e7 46.7 against 52.0 s, 5^3 at 2.5e7 37.4 against 39.1 s), and stop below the sizes whose subspaces wait in
#: host memory, where the workers' own copies of the static parts add memory near the limit of the node for a small
#: gain (2 x 2 x 2 mesh: 7^3 cubes at 1.9e8 1079 against 1149 s with 186 against 170 GB of host memory; 8^3 cubes at
#: 1.1e8 1639 s with 161 GB under ``on``).
_FORCE_TASK_ENTRIES = 1 << 26


def size_bound(extras: bool) -> int:
    """The largest median task (rows times states) the worker processes take under ``auto``: :data:`_TASK_ENTRIES`,
    :data:`_FORCE_TASK_ENTRIES` for a loop with extras."""

    return _FORCE_TASK_ENTRIES if extras else _TASK_ENTRIES
#: Whether the extras of a job (the nonlocal forces) are formed in the workers (ProcessKPointSolver.extras; the job
#: gives the parts of its force, ``force_parts`` of its geometry)
_WORKER_EXTRAS = True
_START_SECONDS = 180.0  # the most the worker processes of a solver may take to connect
_STOP_SECONDS = 10.0


def sector_processes_setting() -> str:
    """``PARSEC_KPOINT_SECTOR_PROCESSES``: ``auto`` (default) solves the small k-point sectors of one process with
    several devices in a worker process per device; ``off`` keeps one thread per device, as before; ``on`` takes the
    worker processes whatever the size of the sectors, also on the host double of the device code (tests)."""

    value = os.environ.get(SWITCH, "auto").strip().lower() or "auto"
    if value not in {"auto", "off", "on"}:
        raise ValueError(f"{SWITCH} must be auto, off or on, not {value!r}")
    return value


def _plan_route(backend, tasks: int) -> tuple[Any, int] | str:
    """``(device plan, workers)`` of the solver of ``tasks`` tasks of this process where its devices may take worker
    processes (the switch, no MPI, CuPy with two devices or more for the tasks, the device solver), else why not."""

    value = sector_processes_setting()
    if value == "off":
        return "off"
    from ..Parallel import mpi
    from .kpoint_parallel import _device_solver_requested, device_worker_count, resolve_backend

    if mpi.context() is not None:
        return "MPI"
    plan = resolve_backend(backend)
    if plan is None:
        return "CPU"
    if plan.name != "cupy" and not (value == "on" and plan.name == "numpy-device"):
        return "no CUDA devices"
    workers = device_worker_count(plan, tasks)
    if workers < 2:
        return "one device"
    if not _device_solver_requested():
        return "host solver"
    if plan.name == "cupy" and not _default_compute_modes(tuple(plan.device_ids)):
        return "a device not in the default compute mode"
    return plan, int(workers)


def _default_compute_modes(devices: tuple) -> bool:
    """Whether every device takes contexts of several processes (compute mode Default): a process per device next to
    this process's own context needs it (an exclusive device refuses the worker's context)."""

    found = _COMPUTE_MODES.get(devices)
    if found is None:
        try:
            from ..backends.cupy import require_cupy

            cp, _ = require_cupy()
            found = all(int(cp.cuda.runtime.getDeviceProperties(int(d))["computeMode"]) == 0 for d in devices)
        except Exception:  # noqa: BLE001 - a device that does not say keeps the threads
            found = False
        _COMPUTE_MODES[devices] = found
    return found


_COMPUTE_MODES: dict[tuple, bool] = {}


def task_entries(rows, states) -> int:
    """The median of rows times states over the tasks of a solver of sectors: the size that decides whether a task's
    solve is set by its launches (small) or by its device (large) (:data:`_TASK_ENTRIES`)."""

    sizes = sorted(int(r) * int(s) for r, s in zip(rows, states))
    return sizes[len(sizes) // 2] if sizes else 0


def processes_expected(backend=None) -> bool:
    """Whether the k-point sectors of this process would be solved by worker processes on its devices, as far as
    that is known before the sectors are planned (:func:`_plan_route`): the floor of the sectors of a meta-GGA on
    several devices, which the threads do not scale over, is then that of one device
    (``.kpoint_sectors.sector_solver``)."""

    return not isinstance(_plan_route(backend, 1 << 30), str)


def plans_take_processes(job, plans) -> bool:
    """Whether the sectors ``plans`` of ``job`` (``.kpoint_sectors.plan_sectors``) would take the worker processes, as
    far as the plans tell (:func:`process_route` decides): every k-point split, no real forms, no extras the workers
    cannot form, and under ``auto`` sectors no larger than :func:`size_bound`."""

    if plans is None or any(plan is None for plan in plans):
        return False
    if any(plan.real_forms for plan in plans):
        return False
    if job.extras is not None and (not _WORKER_EXTRAS or (job.geometry or {}).get("force_parts") is None):
        return False
    if sector_processes_setting() == "auto":
        rows = [size for plan in plans for size in plan.sizes]
        states = [count for plan in plans for count in plan.counts]
        if task_entries(rows, states) > size_bound(job.extras is not None):
            return False
    return True


def process_route(solver, inner_job, options) -> tuple[Any, int] | str:
    """``(device plan, workers)`` where the inner solver of the sectors ``solver`` (a ``SectorKPointSolver``) takes the
    worker processes, else why not (a short text)."""

    from .kpoint_parallel import _plain_solver

    found = _plan_route(options.get("backend"), inner_job.n_tasks)
    if isinstance(found, str):
        return found
    plan, workers = found
    if getattr(solver, "_make_solver", None) is not _plain_solver:
        return "another inner solver"
    geometry = solver.job.geometry or {}
    if geometry.get("exchange_cross_scale") is not None:
        return "hybrid"
    part = geometry.get("hubbard_part")
    if part is not None and not hasattr(part, "recipe"):
        return "LDA+U part without recipe"
    if any(sector is None for _, sector in solver._items):
        return "whole k-points"
    if any(plan_ is not None and plan_.real_forms for plan_ in solver._plans):
        return "real forms"
    if inner_job.extras is not None and (not _WORKER_EXTRAS or geometry.get("force_parts") is None):
        # extras (the nonlocal forces) whose parts the job does not give: they need the orbitals, which live in the
        # workers
        return "extras"
    if sector_processes_setting() == "auto" and inner_job.rows_per_task is not None:
        entries = task_entries(inner_job.rows_per_task, inner_job.states_per_task or ())
        if entries > size_bound(inner_job.extras is not None):
            return "large sectors"
    return plan, int(workers)


def make_process_solver(solver, inner_job, options):
    """The inner solver of the sectors ``solver`` on worker processes, or ``None`` (the threads: the route does not
    apply, or its processes could not start)."""

    found = process_route(solver, inner_job, options)
    if isinstance(found, str):
        return None
    plan, workers = found
    try:
        return ProcessKPointSolver(inner_job, solver, plan, workers, options)
    except _StartError as error:
        warnings.warn(f"{SWITCH}: the worker processes did not start ({error}); the threads solve the sectors")
        return None


# ------------------------------------------------------------------ shipping the static objects
def _static_types() -> tuple:
    from parsec_python.Hamiltonian.hubbard import HubbardProjectors
    from parsec_python.V_ion.ionic_potential import NonlocalProjectorOperator

    from ..Hamiltonian.sector_bloch_meta_gga import SectorMetaGeometry
    from ..Hamiltonian.sector_bloch_stencil import SectorStencilTable
    from .kpoint_sectors import ClassMember, SectorExpansion

    return (
        NonlocalProjectorOperator, SectorStencilTable, HubbardProjectors, SectorMetaGeometry, ClassMember,
        SectorExpansion,
    )


class _Shipping:
    """The static objects one worker process holds, by ``id``; held here as long (so no ``id`` is reused)."""

    def __init__(self) -> None:
        self.held: dict[int, Any] = {}
        self._types = _static_types()

    def static(self, value) -> bool:
        import scipy.sparse as sp

        return isinstance(value, self._types) or sp.issparse(value)

    def dumps(self, message) -> bytes:
        buffer = io.BytesIO()
        _StaticPickler(buffer, self).dump(message)
        return buffer.getvalue()

    def clear(self) -> None:
        self.held.clear()


class _StaticPickler(pickle.Pickler):
    """Sends a static object (:meth:`_Shipping.static`) once: ``("new", id, object)`` the first time, ``("ref", id)``
    after."""

    def __init__(self, file, shipping: _Shipping) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._shipping = shipping
        self._sending: set[int] = set()

    def persistent_id(self, value):  # noqa: D102 - pickle protocol
        if not self._shipping.static(value):
            return None
        key = id(value)
        if key in self._sending:
            return None  # the object itself, inside its "new" record (and any later reference of this message)
        held = self._shipping.held
        if held.get(key) is value:
            return ("ref", key)
        self._sending.add(key)
        held[key] = value
        return ("new", key, value)


class _CacheUnpickler(pickle.Unpickler):
    def __init__(self, file, cache: dict) -> None:
        super().__init__(file)
        self._cache = cache

    def persistent_load(self, pid):  # noqa: D102 - pickle protocol
        if pid[0] == "new":
            self._cache[pid[1]] = pid[2]
            return pid[2]
        return self._cache[pid[1]]


# ------------------------------------------------------------------ the worker processes
class _StartError(RuntimeError):
    pass


class WorkerProcessError(RuntimeError):
    """An exception raised in a worker process (with its traceback text), or a worker process that died."""


class _WorkerProcesses:
    """One worker process per device, started here and connected at the first :meth:`ensure_ready` (they import the
    k-point modules and open their CUDA contexts while this process prepares the first step)."""

    def __init__(self, workers: int, devices) -> None:
        from multiprocessing.connection import Listener

        import parsec_python

        self.workers = int(workers)
        self.devices = tuple(devices)
        self.broken = False
        self._ready = False
        authkey = secrets.token_bytes(32)
        self._listener = Listener(authkey=authkey)
        source = os.path.dirname(os.path.dirname(os.path.abspath(parsec_python.__file__)))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH", "")]))
        environment["PARSEC_KPOINT_PROCESS_KEY"] = authkey.hex()
        address = self._listener.address
        address = address if isinstance(address, str) else repr(address)
        self.processes = []
        self.connections: list[Any] = [None] * self.workers
        self._accepted: list[Any] = []
        self._failure: list[BaseException] = []
        try:
            for index in range(self.workers):
                device = "-" if not self.devices else str(self.devices[index % len(self.devices)])
                self.processes.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "parsec_python.acceleration.SCF.kpoint_processes", address, str(index),
                         device],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                    )
                )
        except BaseException:
            self.stop()
            raise
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        try:
            for _ in range(self.workers):
                self._accepted.append(self._listener.accept())
        except BaseException as error:  # noqa: BLE001 - reported by ensure_ready
            self._failure.append(error)

    def ensure_ready(self) -> None:
        """Wait until every worker process has connected (``_StartError`` where one exits first or none in time)."""

        if self._ready:
            return
        try:
            deadline = time.monotonic() + _START_SECONDS
            while self._thread.is_alive():
                self._thread.join(0.05)
                dead = [process.returncode for process in self.processes if process.poll() is not None]
                if dead:
                    raise _StartError(f"a worker process exited with code {dead[0]} before it connected")
                if time.monotonic() > deadline:
                    raise _StartError(f"the worker processes did not connect within {_START_SECONDS:.0f} s")
            if self._failure:
                raise _StartError(repr(self._failure[0]))
            for connection in self._accepted:
                # the greeting comes after the worker opened the context of its device
                while not connection.poll(0.05):
                    dead = [process.returncode for process in self.processes if process.poll() is not None]
                    if dead:
                        raise _StartError(f"a worker process exited with code {dead[0]} before it was ready")
                    if time.monotonic() > deadline:
                        raise _StartError(f"the worker processes were not ready within {_START_SECONDS:.0f} s")
                kind, index, _, failure = pickle.loads(connection.recv_bytes())
                if kind != "ready":
                    raise _StartError(f"unexpected greeting {kind!r}")
                if failure:
                    raise _StartError(f"worker process {index}: {failure}")
                self.connections[int(index)] = connection
            self._ready = True
        except BaseException:
            self.stop()
            raise
        finally:
            self._close_listener()

    def _close_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def alive(self) -> bool:
        return not self.broken and all(process.poll() is None for process in self.processes)

    def send(self, index: int, data: bytes) -> None:
        try:
            self.connections[index].send_bytes(data)
        except (OSError, EOFError) as error:
            self.broken = True
            raise WorkerProcessError(f"worker process {index} is gone ({error})") from error

    def receive(self, index: int):
        try:
            kind, payload = pickle.loads(self.connections[index].recv_bytes())
        except (OSError, EOFError) as error:
            self.broken = True
            code = self.processes[index].poll()
            raise WorkerProcessError(f"worker process {index} is gone (exit code {code})") from error
        if kind == "error":
            raise WorkerProcessError(f"worker process {index} failed:\n{payload}")
        return payload

    def stop(self) -> None:
        self.broken = True
        for connection in self.connections:
            if connection is None:
                continue
            try:
                connection.send_bytes(pickle.dumps(None))
                connection.close()
            except (OSError, EOFError):
                pass
        self._close_listener()
        deadline = time.monotonic() + _STOP_SECONDS
        for process in self.processes:
            try:
                process.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
        self.connections = [None] * self.workers


_FREE: dict[tuple, list[_WorkerProcesses]] = {}
_ALL: list[_WorkerProcesses] = []
_LOCK = threading.Lock()
_FAILED: list[str] = []


def _acquire(workers: int, devices) -> _WorkerProcesses:
    """Worker processes for one solver: a free set of the run, else new ones (started, not yet connected)."""

    if _FAILED:
        raise _StartError(_FAILED[0])
    key = (int(workers), tuple(devices))
    with _LOCK:
        free = _FREE.setdefault(key, [])
        while free:
            found = free.pop()
            if found.alive():
                return found
            found.stop()
    try:
        found = _WorkerProcesses(workers, devices)
    except Exception as error:  # noqa: BLE001 - any failure to start leaves the threads in place
        _FAILED.append(repr(error))
        raise _StartError(repr(error)) from error
    with _LOCK:
        _ALL.append(found)
    return found


def _release(found: _WorkerProcesses) -> None:
    with _LOCK:
        if found.alive():
            _FREE.setdefault((found.workers, found.devices), []).append(found)
        else:
            found.stop()


@atexit.register
def _stop_all() -> None:
    with _LOCK:
        everything = list(_ALL)
        _ALL.clear()
        _FREE.clear()
    for found in everything:
        found.stop()


# ------------------------------------------------------------------ the solver of the parent
def _partition(task_ids, workers: int) -> list[list[int]]:
    from ..Parallel.pool import _partition as partition

    return partition(list(task_ids), workers)


def _routed(method):
    """A method of :class:`ProcessKPointSolver` that starts the worker processes on first use, or hands the call to the
    solver of threads where they could not start."""

    name = method.__name__

    @functools.wraps(method)
    def wrapper(self, *arguments, **keywords):
        if not self._started:
            self._start()
        if self._fallback is not None:
            return getattr(self._fallback, name)(*arguments, **keywords)
        return method(self, *arguments, **keywords)

    return wrapper


class ProcessKPointSolver:
    """The interface of :class:`.kpoint_parallel.PooledKPointSolver` for the inner job of a ``SectorKPointSolver``, its
    tasks solved in worker processes, one per device (see the module text).  The processes are started here and
    connected at the first call; where they cannot connect, the solver of threads takes over (with a warning)."""

    def __init__(self, job, sectors, plan, workers: int, options: dict) -> None:
        self.job = job
        self._sectors = sectors
        self._plan = plan
        self._options = dict(options)
        self.backend = plan.name
        self._workers = int(workers)
        devices = tuple(plan.device_ids) if plan.name == "cupy" else ()
        self._processes = _acquire(workers, devices)
        self._chunks = _partition(range(job.n_tasks), workers)
        self._shipping = [_Shipping() for _ in range(workers)]
        self._values: dict[int, np.ndarray] = {}
        self._seconds: dict[str, float] = {}
        self._closed = False
        self._started = False
        self._fallback = None
        self._host_pool = None  # the threads of the host parts of the density (accumulate)
        # the BLAS of this process as the device threads leave it (Parallel.pool._ThreadPool): one thread while the
        # solver lives, the count of before after
        from ..Parallel.blas import blas_threads, set_blas_threads
        from ..Parallel.pool import available_cpus

        previous = blas_threads()
        self._blas_restore = (available_cpus(), previous) if set_blas_threads(1) else None

    def _start(self) -> None:
        """Connect the worker processes and give them their tasks; where one cannot start (it exits, does not connect,
        its device fails, or its solver cannot be made), the solver of threads takes over."""

        from .kpoint_real import real_gamma_applies

        self._started = True
        started = time.perf_counter()
        try:
            self._processes.ensure_ready()
            real_gamma = self._options.get("real_gamma")
            real = real_gamma_applies(self.job, None) if real_gamma is None else bool(real_gamma)
            environment = {name: value for name, value in os.environ.items() if name.startswith("PARSEC_")}
            plan, job = self._plan, self.job
            layout = {
                "n_tasks": int(job.n_tasks),
                "number_of_states": int(job.number_of_states),
                "states_per_task": job.states_per_task,
                "rows_per_task": job.rows_per_task,
                "grid_points": job.grid_points,
            }
            self._all(
                "init",
                lambda index: (environment, plan.name, tuple(plan.device_ids), index, self._chunks[index], real, layout),
            )
        except (_StartError, WorkerProcessError) as error:
            text = str(error).strip().splitlines()
            _FAILED.append(text[-1] if text else repr(error))
            self._processes.stop()
            self._restore_blas()
            warnings.warn(f"{SWITCH}: the worker processes did not start ({error}); the threads solve the sectors")
            self._fallback = self._sectors._make_solver(self.job, **self._options)
            return
        self._add("process_start_seconds", time.perf_counter() - started)

    def _restore_blas(self) -> None:
        if self._blas_restore is not None:
            from ..Parallel.blas import set_blas_threads

            count, previous = self._blas_restore
            set_blas_threads(count, previous=previous)
            self._blas_restore = None

    @property
    def n_workers(self) -> int:
        return self._workers

    # ------------------------------------------------------------- messages
    def _send(self, index: int, method: str, arguments: tuple) -> None:
        self._processes.send(index, self._shipping[index].dumps((method, arguments)))

    def _all(self, method: str, arguments, indices=None) -> list:
        """Send ``method`` to the worker processes ``indices`` (all), with ``arguments(index)``, and their replies."""

        indices = list(range(self.n_workers)) if indices is None else list(indices)
        try:
            for index in indices:
                self._send(index, method, arguments(index))
            replies, failure = [], None
            for index in indices:
                try:
                    replies.append(self._processes.receive(index))
                except WorkerProcessError as error:
                    failure = failure or error
                    replies.append(None)
        except BaseException:
            self._processes.broken = True
            raise
        if failure is not None:
            raise failure
        return replies

    def _one(self, index: int, method: str, *arguments):
        return self._all(method, lambda _: arguments, [index])[0]

    def _owner(self, task: int) -> int:
        return int(task) % self.n_workers  # the round-robin partition of the pools (Parallel.pool._partition)

    def _add(self, name: str, seconds: float) -> None:
        self._seconds[name] = self._seconds.get(name, 0.0) + seconds

    # ------------------------------------------------------------- SCF steps
    @_routed
    def solve(self, potentials: dict, filter_degree: int, tolerance: float | None = None, tasks=None) -> list[np.ndarray]:
        """One eigenvalue array per task.  Every worker gets the order to solve its tasks, then their Hamiltonians one
        at a time, round-robin over the workers in the order they solve them: a worker starts on its first task while
        this process builds the next ones (a worker process reads its messages as they come)."""

        from parsec_python.SCF.kpoint_solver import job_eigval_settings

        settings = job_eigval_settings(self.job, filter_degree, tolerance)
        wanted = None if tasks is None else frozenset(int(task) for task in tasks)
        mine = {}
        for index, chunk in enumerate(self._chunks):
            found = list(chunk) if wanted is None else [task for task in chunk if task in wanted]
            if found:
                mine[index] = found
        try:
            for index, found in mine.items():
                self._send(index, "solve", (settings, filter_degree, tolerance, None if wanted is None else frozenset(found)))
            for position in range(max((len(found) for found in mine.values()), default=0)):
                for index, found in mine.items():
                    if position >= len(found):
                        continue
                    task = found[position]
                    started = time.perf_counter()
                    hamiltonian = self.job.make_hamiltonian(task, potentials)
                    self._add("worker_hamiltonian_seconds", time.perf_counter() - started)
                    started = time.perf_counter()
                    self._send(index, "hamiltonian", (task, hamiltonian))
                    self._add("process_send_seconds", time.perf_counter() - started)
            failure = None
            for index in mine:
                try:
                    for task, values in self._processes.receive(index):
                        self._values[int(task)] = values
                except WorkerProcessError as error:
                    failure = failure or error
        except BaseException:
            self._processes.broken = True
            raise
        if failure is not None:
            raise failure
        return [self._values[task] for task in range(self.job.n_tasks)]

    @_routed
    def accumulate(self, occupations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """The density parts of every task (the order of the sums of the threads under ``PARSEC_ORDERED_SUMS=auto``).
        The parts of each worker's tasks that need no orbitals (the density on the whole grid from the sums of a sector)
        are formed here by a thread per worker as its payloads come (NumPy leaves the interpreter lock in them)."""

        def arguments(index):
            chunk = self._chunks[index]
            recipes = {}
            for task in chunk:
                recipe = self._sectors.device_recipe(task)
                if recipe is not None:
                    recipes[task] = recipe
            return ({task: occupations[task] for task in chunk}, recipes)

        def chunk_total(found) -> dict[str, np.ndarray]:
            # the sum of _SolveWorker.accumulate: the tasks of the worker in their order
            total: dict[str, np.ndarray] = {}
            for task, sums, made in found:
                for name, value in self._sectors.sector_parts(task, sums, made=made).items():
                    if name in total:
                        total[name] += value
                    else:
                        total[name] = np.array(value, copy=True)
            return total

        if self._host_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._host_pool = ThreadPoolExecutor(max_workers=self.n_workers)
        indices = list(range(self.n_workers))
        futures, failure = {}, None
        try:
            for index in indices:
                self._send(index, "payloads", arguments(index))
            for index in indices:
                try:
                    found = self._processes.receive(index)
                except WorkerProcessError as error:
                    failure = failure or error
                    continue
                futures[index] = self._host_pool.submit(chunk_total, found)
        except BaseException:
            self._processes.broken = True
            raise
        if failure is not None:
            raise failure
        parts = []
        for index in indices:
            chunk = self._chunks[index]
            parts.append(((min(chunk) if chunk else -1), futures[index].result()))
        parts.sort(key=lambda item: item[0])
        result: dict[str, np.ndarray] = {}
        for _, part in parts:
            for name, value in part.items():
                if name in result:
                    result[name] = result[name] + value
                else:
                    result[name] = value
        return result

    @_routed
    def extras(self) -> list[Any]:
        """The extras (the nonlocal forces) of every task, formed in the workers where the orbitals are: each gets the
        recipes of its tasks (``SectorKPointSolver.extras_recipe``: the columns, places and expansions, the parts of
        the nonlocal force of each k-point) and gives back their sums; the forces of the atoms of each orbit are
        completed here.  The same operations on the same devices as the threads."""

        def arguments(index):
            return ({task: self._sectors.extras_recipe(task) for task in self._chunks[index]},)

        found: dict[int, Any] = {}
        for part in self._all("extras", arguments):
            for task, total in part:
                found[int(task)] = self._sectors.complete_extras(int(task), total)
        return [found[task] for task in range(self.job.n_tasks)]

    @_routed
    def vectors(self, task: int, columns: int | None = None) -> np.ndarray:
        return self._one(self._owner(task), "vectors_of", int(task), columns)

    @_routed
    def device_vectors(self, task: int):
        """The final eigenvectors of one task (on the host: they live in another process)."""

        return self.vectors(task)

    @_routed
    def resize_tasks(self, counts: dict) -> bool:
        from parsec_python.SCF.kpoint_solver import resized_job

        counts = {int(task): int(count) for task, count in counts.items()}
        self.job = resized_job(self.job, counts)
        self._all("resize_tasks", lambda _: (counts,))
        return True

    @_routed
    def trim_tasks(self, counts: dict) -> bool:
        from parsec_python.SCF.kpoint_solver import resized_job

        counts = {int(task): int(count) for task, count in counts.items()}
        self.job = resized_job(self.job, counts)
        self._all("trim_tasks", lambda _: (counts,))
        return True

    @_routed
    def prepare_reported(self, buffer: int) -> bool:
        from .kpoint_parallel import _REPORTED_HOST_FRACTION, _available_host_bytes, _storage_mode

        found = self._all("prepare_reported", lambda _: (int(buffer),))
        if not all(part[0] for part in found):
            return False
        on_host = sum(part[2] for part in found if part[1])
        if not on_host or _storage_mode() != "auto":
            return True
        available = _available_host_bytes()
        return available is None or on_host <= _REPORTED_HOST_FRACTION * available

    @_routed
    def set_reported_cuts(self, cuts, tolerance=None, whole=()) -> None:
        self._all("set_reported_cuts", lambda _: (cuts, tolerance, tuple(whole)))

    @_routed
    def reported_progress(self) -> dict[int, tuple[np.ndarray, int]]:
        merged: dict[int, tuple[np.ndarray, int]] = {}
        for part in self._all("reported_progress", lambda _: ()):
            merged.update(part)
        return merged

    @_routed
    def export_states(self) -> dict[int, Any]:
        merged: dict[int, Any] = {}
        for part in self._all("export_states", lambda _: ()):
            merged.update(part)
        return merged

    @_routed
    def import_states(self, states: dict[int, Any]) -> None:
        self._all("import_states", lambda index: ({task: states[task] for task in self._chunks[index] if task in states},))

    @_routed
    def keep_states_on_devices(self) -> None:
        """(The device subspaces live in the worker processes: :meth:`export_states` hands host copies.)"""

    @_routed
    def task_devices(self) -> dict[int, int]:
        if self._plan.name != "cupy":
            return {}
        ids = tuple(self._plan.device_ids)
        return {task: int(ids[index % len(ids)]) for index, chunk in enumerate(self._chunks) for task in chunk}

    @_routed
    def stats(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for part in self._all("stats", lambda _: ()):
            for name, value in part.items():
                total[name] = total.get(name, 0.0) + value
        for name, value in self._seconds.items():
            total[name] = total.get(name, 0.0) + value
        total["kpoint_sector_processes"] = float(self.n_workers)
        return total

    @_routed
    def residuals(self) -> list[float]:
        merged: dict[int, float] = {}
        for part in self._all("residuals", lambda _: ()):
            merged.update(part)
        return [merged.get(task, float("nan")) for task in range(self.job.n_tasks)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._host_pool is not None:
            self._host_pool.shutdown(wait=True)
            self._host_pool = None
        if self._fallback is not None:
            self._fallback.close()
            return
        try:
            if self._started and self._processes.alive():
                self._all("reset", lambda _: ())
        except BaseException:  # noqa: BLE001 - a set that fails to reset is not reused
            self._processes.broken = True
        finally:
            for shipping in self._shipping:
                shipping.clear()
            _release(self._processes)
            self._restore_blas()

    def __enter__(self) -> "ProcessKPointSolver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ------------------------------------------------------------------ the worker process
class _HamiltonianStream:
    """The ``potentials`` a worker process hands its solver: the Hamiltonian of each task is the next message of the
    parent (:meth:`ProcessKPointSolver.solve`), read when the solver asks for it."""

    def __init__(self, state: "_ChildState") -> None:
        self._state = state

    def take(self, task: int):
        data = self._state.inbox.get()
        if data is None:
            raise SystemExit(0)  # the parent is gone or stops the process
        message = self._state.load(data)
        if message is None:
            raise SystemExit(0)
        kind, (found, hamiltonian) = message
        if kind != "hamiltonian" or int(found) != int(task):
            raise RuntimeError(f"expected the Hamiltonian of task {task}, received {kind!r} of {found!r}")
        return hamiltonian


def _shipped_hamiltonian(task: int, stream: _HamiltonianStream):
    return stream.take(task)


def _no_accumulate(*_arguments, **_keywords):  # pragma: no cover - the payloads replace it
    raise RuntimeError("a worker process forms the parts of the density in its payloads")


class _ChildState:
    """What a worker process holds: the solver of its device (``_SolveWorker``) and the static objects of the parent."""

    def __init__(self, inbox=None) -> None:
        self.inbox = inbox  # the messages of the parent as they come (child_loop)
        self.cache: dict[int, Any] = {}
        self.worker = None
        self.settings = None
        self.occupation = None
        self.member_maps: dict = {}

    def load(self, data: bytes):
        return _CacheUnpickler(io.BytesIO(data), self.cache).load()

    def _settings(self, _degree, _tolerance=None):
        return self.settings

    def do_init(self, environment, backend_name, device_ids, index, tasks, real_gamma, layout):
        from parsec_python.SCF.kpoint_hubbard import SectorOccupation
        from parsec_python.SCF.kpoint_solver import KPointJob

        from ..Hamiltonian.device_bloch import KPointBackend
        from .kpoint_parallel import _SolveWorker

        for name in [name for name in os.environ if name.startswith("PARSEC_") and name not in environment]:
            del os.environ[name]
        os.environ.update(environment)
        self.do_reset()
        job = KPointJob(
            layout["n_tasks"],
            layout["number_of_states"],
            self._settings,
            _shipped_hamiltonian,
            _no_accumulate,
            states_per_task=layout["states_per_task"],
            rows_per_task=layout["rows_per_task"],
            grid_points=layout["grid_points"],
        )
        self.occupation = SectorOccupation()
        self.worker = _SolveWorker(job, list(tasks), int(index), KPointBackend(backend_name, tuple(device_ids)), real_gamma)
        return None

    def do_solve(self, settings, filter_degree, tolerance, only):
        self.settings = settings
        return list(self.worker.solve(_HamiltonianStream(self), filter_degree, tolerance, only))

    def do_payloads(self, occupations, recipes):
        """Per task ``(task, sum_n f_n |v_n|^2 on its rows, the other parts made from the orbitals or None)``: what
        ``SectorKPointSolver._accumulate`` forms from the orbitals (the device or host branch of
        ``_SolveWorker.accumulate``; ``SectorKPointSolver.device_recipe``: the LDA+U occupation matrices, the sums of
        the orbital term of a meta-GGA)."""

        from parsec_python.SCF.orbital_density import weighted_square_sum

        from ..Hamiltonian.sector_bloch_meta_gga import sector_orbital_sums
        from .kpoint_sectors import _host, class_member_block

        worker = self.worker
        found = []
        for task in worker.tasks:
            vectors = worker.vectors[task]
            if worker.converter is not None and not isinstance(vectors, np.ndarray):
                start = time.perf_counter()
                backend = worker.converter.backend
                backend.use()
                xp = backend.xp
            else:
                start = time.perf_counter()
                vectors = worker._host_vectors(task)
                worker._tick("download", start)
                start = time.perf_counter()
                xp = np
            sums = _host(weighted_square_sum(vectors, occupations[task], xp))
            made = None
            recipe = recipes.get(task)
            if recipe is not None:
                made = {}
                meta = recipe.get("meta")
                if meta is not None:
                    gradient, entries = meta
                    parts = []
                    for geometry, shifts, member in entries:
                        block = vectors if member is None else class_member_block(
                            member, vectors, xp, self.member_maps, id(member)
                        )
                        parts.append(sector_orbital_sums(geometry, shifts, block, occupations[task], xp, gradient=gradient))
                    made["meta"] = parts
                hubbard = recipe.get("hubbard")
                if hubbard is not None:
                    projectors, name, weight = hubbard
                    made["hubbard"] = {name: self.occupation(projectors, vectors, occupations[task], weight, xp=xp)}
            worker._tick("density", start)
            found.append((task, sums, made))
        return found

    def do_extras(self, recipes):
        """Per task ``(task, sum of its extras or None)``: ``kpoint_nonlocal_force`` of the driver (``SCF/kpoints.py``)
        on the blocks of :func:`.kpoint_sectors.sector_extras`, from the parts of the force its recipe carries (the
        device or host branch of ``_SolveWorker.extras``)."""

        from parsec_python.Forces.nonlocal_force import nonlocal_forces

        from ..Forces.kpoint_nonlocal_device import device_nonlocal_forces, prepare_device_nonlocal
        from .kpoint_sectors import sector_extras

        worker = self.worker
        prepared_of: dict[tuple, Any] = {}  # (operator, device) -> the device matrices of the contraction
        found = []
        for task in worker.tasks:
            recipe = recipes[task]
            vectors = worker.vectors[task]
            if worker.converter is not None and not isinstance(vectors, np.ndarray):
                backend = worker.converter.backend
                backend.use()
                xp = backend.xp
                vectors = worker._u_gauge(task, vectors, xp)
            else:
                xp = np
                vectors = worker._u_gauge(task, worker._host_vectors(task))
            if int(vectors.shape[1]) != int(recipe["count"]):
                raise RuntimeError(f"task {task} holds {vectors.shape[1]} states, its recipe {recipe['count']}")
            operator, gradients, weight, occupations, atom_count = recipe["force"]

            def force(expanded, places, operator=operator, gradients=gradients, weight=weight,
                      occupations=occupations, atom_count=atom_count, xp=xp):
                chosen = occupations[np.asarray(places)]
                if xp is not np and not isinstance(expanded, np.ndarray):
                    key = (id(operator), int(expanded.device.id))
                    prepared = prepared_of.get(key)
                    if prepared is None:
                        prepared = prepared_of[key] = prepare_device_nonlocal(xp, operator, gradients)
                    return device_nonlocal_forces(
                        xp, operator, gradients, expanded, weight * chosen, atom_count, prepared=prepared
                    )
                return nonlocal_forces(operator, gradients, np.asarray(expanded), weight * chosen, atom_count)

            found.append((task, sector_extras(recipe, vectors, xp, force, None, self.member_maps)))
        return found

    def do_vectors_of(self, task: int, columns):
        arguments = {} if columns is None else {"columns": int(columns)}
        for _, vectors in self.worker.vectors_of(task, **arguments):
            return vectors
        raise KeyError(task)

    def do_resize_tasks(self, counts):
        list(self.worker.resize_tasks(counts))

    def do_trim_tasks(self, counts):
        list(self.worker.trim_tasks(counts))

    def do_prepare_reported(self, buffer):
        return list(self.worker.prepare_reported(buffer))[0][1]

    def do_set_reported_cuts(self, cuts, tolerance, whole):
        list(self.worker.set_reported_cuts(cuts, tolerance, whole))

    def do_reported_progress(self):
        return dict(self.worker.reported_progress)

    def do_export_states(self):
        return dict(self.worker.export_states())

    def do_import_states(self, states):
        list(self.worker.import_states(states))

    def do_stats(self):
        return dict(self.worker.stats)

    def do_residuals(self):
        return dict(self.worker.residual)

    def do_reset(self):
        self.worker = None
        self.settings = None
        self.occupation = None
        self.member_maps = {}
        self.cache.clear()
        gc.collect()
        cupy = sys.modules.get("cupy")
        if cupy is not None:
            try:
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:  # noqa: BLE001 - a pool that cannot be emptied keeps its blocks
                pass
        return None


def child_loop(address: str, index: int, device: int | None = None) -> None:
    """The loop of a worker process: connect to the parent at ``address``, open the CUDA context of ``device`` (while
    the parent prepares its first message), then answer the messages."""

    from multiprocessing.connection import Client

    from ..Parallel.blas import set_blas_threads

    # the BLAS of the device threads (Parallel.pool._ThreadPool): one thread
    set_blas_threads(1)
    authkey = bytes.fromhex(os.environ.pop("PARSEC_KPOINT_PROCESS_KEY"))
    connection = Client(address, authkey=authkey)
    failure = None
    try:
        from . import kpoint_parallel  # noqa: F401  (the solver of the device threads)

        if device is not None:
            import cupy

            # the device of this worker as its current device (a context manager would make the first device current
            # again on exit, which opens a context there as well)
            cupy.cuda.Device(int(device)).use()
            float(cupy.zeros(1).sum())
    except Exception as error:  # noqa: BLE001 - reported in the greeting: the parent takes the threads
        failure = f"{type(error).__name__}: {error}"
    connection.send_bytes(pickle.dumps(("ready", int(index), os.getpid(), failure)))
    if failure is not None:
        return
    # a thread takes the messages of the parent as they come, so that its sends never wait for this process to finish
    # a solve (the Hamiltonians of the tasks of a step arrive while the first ones are solved)
    import queue

    inbox: queue.Queue = queue.Queue()

    def read() -> None:
        while True:
            try:
                inbox.put(connection.recv_bytes())
            except (EOFError, OSError):
                inbox.put(None)
                return

    threading.Thread(target=read, daemon=True).start()
    state = _ChildState(inbox)
    while True:
        data = inbox.get()
        if data is None:
            return
        try:
            message = state.load(data)
            if message is None:
                return
            method, arguments = message
            reply = ("ok", getattr(state, "do_" + method)(*arguments))
            data = pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL)
        except SystemExit:
            return
        except BaseException:  # noqa: BLE001 - sent to the parent
            data = pickle.dumps(("error", traceback.format_exc()), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            connection.send_bytes(data)
        except (EOFError, OSError):
            return


def _main(arguments: list[str]) -> None:
    address, index = arguments[0], int(arguments[1])
    device = None if len(arguments) < 3 or arguments[2] == "-" else int(arguments[2])
    if address.startswith("("):  # an address of a socket family (the repr of a tuple)
        import ast

        address = ast.literal_eval(address)
    from parsec_python.acceleration.SCF import kpoint_processes

    kpoint_processes.child_loop(address, index, device)


__all__ = [
    "ProcessKPointSolver",
    "plans_take_processes",
    "processes_expected",
    "SWITCH",
    "WorkerProcessError",
    "make_process_solver",
    "process_route",
    "sector_processes_setting",
]


if __name__ == "__main__":
    _main(sys.argv[1:])
