"""Kernels of a calculation compiled ahead of their first launch, beside its preparation (``PARSEC_CUPY_COMPILE_AHEAD``).

With an empty CuPy kernel cache (the first run after an installation or an upgrade of CuPy, or on a node whose
``CUPY_CACHE_DIR`` is new) a calculation compiles every kernel where it first launches it, one after the other on
the thread that needs it.  NVRTC lets go of the interpreter lock while it compiles, and the compilations of raw
modules of one process run side by side (four device XC modules on four threads in 0.59 s against 1.74 s one
after the other on an A100 node).  The CUB reductions of CuPy do not: two of them on two threads took as long as
one after the other, and a compilation of CUB held back the other compilations of the process while it ran.  Their
tests of finiteness take a kernel of their own instead (:mod:`.device_checks`), and nothing here compiles CUB.

:func:`start` hands the kernels an input will need to a few background threads as soon as the input is read.
They compile the same source with the same options through the same CuPy entry as the code that launches the
kernel later, which then reads the binary from CuPy's kernel cache or waits for a compilation still running
(:mod:`.cupy_compile`, ``PARSEC_CUPY_COMPILE_ONCE``).  The binaries, and every number of the calculation, are the
ones the calculation would have compiled itself; only the time of the compilation moves.  A kernel compiled ahead
that the calculation then does not launch costs a background thread its seconds and nothing else, and a task that
fails is dropped: the calculation compiles and reports what it needs itself.

The threads load the modules on a device of the process (the first of ``PARSEC_CUPY_DEVICES``) and launch nothing.
An MPI rank that sees several devices and has not been given its own does not compile ahead (it could pick a
device another rank uses).
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from queue import Empty, SimpleQueue
import threading
from typing import Any, Callable

SETTING = "PARSEC_CUPY_COMPILE_AHEAD"
#: Background threads at most (each compiles one kernel at a time; NVRTC is single-threaded).
MAX_THREADS = 4
_WORLD_SIZE_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_STEP_NUM_TASKS",
    "SLURM_NTASKS",
)


def compile_ahead_setting() -> str:
    """``PARSEC_CUPY_COMPILE_AHEAD``: ``auto`` (the default) or ``off``.

    ``auto``: :func:`start` compiles the kernels an input will need on background threads beside its preparation
    (the same binaries, the same numbers); ``off``: every kernel is compiled where it is first launched, as before.
    """

    value = os.environ.get(SETTING, "auto").strip().lower() or "auto"
    if value not in {"auto", "off"}:
        raise ValueError(f"{SETTING} must be auto or off, not {value!r}")
    return value


@dataclass(frozen=True)
class Task:
    """One compilation ahead: ``key`` names it once per process, ``run(cp, device_id)`` compiles it."""

    key: tuple
    run: Callable[[Any, int], Any]


# ---------------------------------------------------------------------------
# The tasks
# ---------------------------------------------------------------------------


def _device_checks(cp, device_id: int) -> None:
    # the tests of finiteness of every route (backends/device_checks.py)
    from . import device_checks

    device_checks.kernels(cp)


def _orbital_density(cp, device_id: int) -> None:
    from ..Occupations.device_density import density_kernel

    density_kernel(cp)


def _ionic_sum(cp, device_id: int) -> None:
    from ..V_ion import cupy_ionic
    from .cupy_compile import compile_cupy_raw

    compile_cupy_raw(cupy_ionic._kernel())


def _symmetric_boundary(cp, device_id: int) -> None:
    from ..Hartree.cupy_boundary import symmetry_kernels

    symmetry_kernels(cp)


def _projectors(cp, device_id: int) -> None:
    from . import cupy_projectors

    cupy_projectors._kernels(cp)


#: The kernels a cluster builds first, in its preparation: the orbital densities, the ionic potential, the
#: multipole boundary of the Hartree potential in the symmetry wedge and the factors of the projectors.
_CLUSTER_SETUP = (
    Task(("orbital_density",), _orbital_density),
    Task(("ionic_sum",), _ionic_sum),
    Task(("symmetric_boundary",), _symmetric_boundary),
    Task(("projectors",), _projectors),
)


def _device_xc(functional: str, spin_polarized: bool) -> Callable[[Any, int], Any]:
    def run(cp, device_id: int) -> None:
        from ..V_xc import device_xc

        device_xc._kernels(cp, device_id, device_xc.kernel_names(functional, spin_polarized=spin_polarized))

    return run


def _stencil_xc(cp, device_id: int) -> None:
    # the stencils of the meta-GGA evaluator (V_xc/meta_gga_xc.py)
    from ..V_xc import device_xc

    device_xc._kernels(cp, device_id, ("xc_adjoint", "xc_gradient"))


def _hybrid_xc(scf, spin_polarized: bool) -> Callable[[Any, int], Any]:
    def run(cp, device_id: int) -> None:
        from parsec_python.SCF.hybrid import hybrid_functional_of

        from ..V_xc import device_hybrid_xc

        functional = hybrid_functional_of(scf)
        device_hybrid_xc.evaluator_kernels(
            cp, device_id, functional.semilocal, spin_polarized=spin_polarized
        )

    return run


def _device_cell(cell) -> bool:
    """A bulk orthorhombic cell, the periodic grid whose functionals the device evaluates (``device_grid_supported``)."""

    import numpy as np

    if int(getattr(cell, "dimensions", 3)) != 3:
        return False
    vectors = np.asarray(getattr(cell, "lattice_vectors", np.eye(3)), dtype=np.float64)
    return bool(np.allclose(vectors, np.diag(np.diag(vectors))))


def planned_tasks(problem) -> list[Task]:
    """The compilations ahead for ``problem`` (a ``SinglePointInput``), the largest and earliest needed first.

    Every device run tests its arrays for finite values (:mod:`.device_checks`), and every cluster builds the
    kernels of its preparation (:data:`_CLUSTER_SETUP`).  Then the semilocal kernels of the functional where the
    device evaluates it: a hybrid's weighted components (and the functional of its start), a GGA or LDA of a
    cluster or cell, the stencils of a meta-GGA.  Only what the input says is read: a route the run then does not
    take leaves its kernels unused.
    """

    checks = Task(("device_checks",), _device_checks)
    scf = getattr(problem, "scf", None)
    if scf is None:
        return [checks]
    from ..V_xc.device_xc import device_xc_mode, device_xc_selected, periodic_device_xc_selected

    spin = bool(getattr(scf, "spin_polarized", False))
    cell = getattr(problem, "periodic_cell", None)
    periodic = cell is not None
    device_grid = not periodic or _device_cell(cell)
    xc_on_device = device_xc_mode() != "off"
    hybrid = bool(getattr(scf, "is_hybrid", False))
    libxc = bool(getattr(scf, "is_libxc", False))
    meta = bool(getattr(scf, "is_meta_gga", False))
    tasks: list[Task] = []
    if hybrid and not libxc and xc_on_device and device_grid:
        from parsec_python.SCF.hybrid import hybrid_definition, hybrid_functional_of
        from parsec_python.V_xc.semilocal_components import ComponentSemilocal

        try:
            semilocal = hybrid_functional_of(scf).semilocal
            start_functional = hybrid_definition(scf).start_functional
        except (ValueError, KeyError):
            semilocal = start_functional = None
        if isinstance(semilocal, ComponentSemilocal):
            tasks.append(Task(("hybrid_xc", scf.xc_functional, spin, repr(scf.exact_exchange)), _hybrid_xc(scf, spin)))
        tasks.append(checks)
        if not periodic:
            tasks.extend(_CLUSTER_SETUP)
        if start_functional is not None and device_xc_selected(start_functional, spin_polarized=spin):
            tasks.append(Task(("device_xc", start_functional, spin), _device_xc(start_functional, spin)))
        return tasks
    tasks.append(checks)
    if not periodic:
        tasks.extend(_CLUSTER_SETUP)
    if meta and not periodic and xc_on_device:
        tasks.append(Task(("stencil_xc",), _stencil_xc))
    elif not hybrid and not libxc and not meta and device_grid:
        functional = str(scf.xc_functional)
        selected = periodic_device_xc_selected if periodic else device_xc_selected
        try:
            chosen = selected(functional, spin_polarized=spin)
        except (TypeError, ValueError):
            chosen = False
        if chosen:
            tasks.append(Task(("device_xc", functional, spin), _device_xc(functional, spin)))
    return tasks


# ---------------------------------------------------------------------------
# The threads
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_SUBMITTED: set = set()
_THREADS: list[threading.Thread] = []
_FAILURES: list[tuple[tuple, str]] = []
_DONE: list[tuple] = []
_STOP = threading.Event()
_ATEXIT = False


def _world_size() -> int:
    for name in _WORLD_SIZE_VARIABLES:
        try:
            value = int(os.environ.get(name, "") or 0)
        except ValueError:
            continue
        if value > 0:
            return value
    return 1


def _thread_count() -> int:
    """A quarter of the CPUs the process may use, from one to :data:`MAX_THREADS`.

    The OpenMP threads of the native kernels leave four of them free by default (``resources.RESERVED_THREADS``).
    """

    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_THREADS, cpus // 4))


def _device_of_process(cp, requested: int | None) -> int | None:
    """The device the threads compile on, or ``None`` where this process has none it surely uses."""

    from .cupy import cupy_device_count

    count = cupy_device_count()
    if requested is not None:
        return int(requested) if 0 <= int(requested) < count else None
    setting = os.environ.get("PARSEC_CUPY_DEVICES", "auto").strip().lower()
    if setting in {"", "auto"}:
        # the ranks of an MPI run that see several devices choose theirs later
        return 0 if count == 1 or _world_size() == 1 else None
    if setting in {"current", "off"}:
        return None
    try:
        first = int(setting.split(",")[0].strip())
    except ValueError:
        return None
    return first if 0 <= first < count else None


def _worker(queue: SimpleQueue, requested: int | None) -> None:
    try:
        from .cupy import require_cupy

        cp, _ = require_cupy()
        device_id = _device_of_process(cp, requested)
    except Exception:  # noqa: BLE001 - no usable CuPy or device: the calculation reports it itself
        return
    if device_id is None:
        return
    try:
        # The device of the thread for good (``use``): leaving a ``with`` block would set the thread back to
        # device 0, whose primary context CUDA then creates (some 400 MiB on a device the process may not use).
        cp.cuda.Device(device_id).use()
        while not _STOP.is_set():
            try:
                task = queue.get_nowait()
            except Empty:
                break
            try:
                task.run(cp, device_id)
            except Exception as error:  # noqa: BLE001 - the calculation compiles it itself
                with _LOCK:
                    _FAILURES.append((task.key, f"{type(error).__name__}: {error}"))
            else:
                with _LOCK:
                    _DONE.append(task.key)
    except Exception as error:  # noqa: BLE001
        with _LOCK:
            _FAILURES.append((("device",), f"{type(error).__name__}: {error}"))


def start(problem, *, device_id: int | None = None, threads: int | None = None) -> bool:
    """Compile the kernels ``problem`` will need on background threads; ``True`` where threads were started.

    Nothing under ``PARSEC_CUPY_COMPILE_AHEAD=off`` (an unknown value raises ``ValueError``), and nothing for a
    kernel already handed over by an earlier call of this process.  ``device_id`` is the device to compile on
    (default: the first of ``PARSEC_CUPY_DEVICES``).  The call returns at once; the threads end when their tasks
    are done (:func:`wait`).
    """

    global _ATEXIT
    if compile_ahead_setting() == "off":
        return False
    tasks = planned_tasks(problem)
    with _LOCK:
        fresh = [task for task in tasks if task.key not in _SUBMITTED]
        _SUBMITTED.update(task.key for task in fresh)
        if not fresh:
            return False
        if not _ATEXIT:
            atexit.register(_drain)
            _ATEXIT = True
    queue: SimpleQueue = SimpleQueue()
    for task in fresh:
        queue.put(task)
    count = max(1, min(int(_thread_count() if threads is None else threads), len(fresh)))
    started = []
    for index in range(count):
        thread = threading.Thread(
            target=_worker, args=(queue, device_id), name=f"parsec-compile-ahead-{index}", daemon=True
        )
        thread.start()
        started.append(thread)
    with _LOCK:
        _THREADS.extend(started)
    return True


def wait(timeout: float | None = None) -> bool:
    """Wait for the threads of :func:`start`; ``True`` where none runs any more."""

    import time

    end = None if timeout is None else time.monotonic() + timeout
    with _LOCK:
        threads = list(_THREADS)
    for thread in threads:
        left = None if end is None else max(0.0, end - time.monotonic())
        thread.join(left)
    with _LOCK:
        _THREADS[:] = [thread for thread in _THREADS if thread.is_alive()]
        return not _THREADS


def _drain() -> None:
    # At exit: start no further task and let a compilation that runs end before CuPy is torn down.
    _STOP.set()
    wait(120.0)


def report() -> dict:
    """What was compiled ahead in this process: the keys done and the failures (with their reason)."""

    with _LOCK:
        return {"done": list(_DONE), "failures": list(_FAILURES), "running": sum(t.is_alive() for t in _THREADS)}


__all__ = ["SETTING", "Task", "compile_ahead_setting", "planned_tasks", "report", "start", "wait"]
